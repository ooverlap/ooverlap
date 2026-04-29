import argparse
import importlib.util
from pathlib import Path

import torch
import torch.multiprocessing as mp


def load_ooverlap_ext():
    root = Path(__file__).resolve().parents[1]
    so = root / "build" / "lib" / "ooverlap_ext.so"
    if not so.exists():
        raise FileNotFoundError(f"Could not find {so}. Build first.")
    spec = importlib.util.spec_from_file_location("ooverlap_ext", str(so))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def make_column_major_reorder(M, N, tile_m=128, tile_n=128, device="cuda"):
    """
    Return RA where:

      RA[logical_tile] = packed_tile

    Logical tile id is row-major over the original GEMM tile grid:

      logical_tile = tile_row * tile_cols + tile_col

    Packed order here is deliberately column-major, just to prove that the
    epilogue is physically moving tiles.

    Example for a 2x2 tile grid:

      logical layout:

        tile 0 | tile 1
        tile 2 | tile 3

      packed execution/order:

        tile 0, tile 2, tile 1, tile 3

      RA:
        RA[0] = 0
        RA[2] = 1
        RA[1] = 2
        RA[3] = 3
    """
    assert M % tile_m == 0
    assert N % tile_n == 0

    tile_rows = M // tile_m
    tile_cols = N // tile_n
    num_tiles = tile_rows * tile_cols

    ra = torch.empty((num_tiles,), dtype=torch.int32, device=device)

    packed_pos = 0
    for tile_col in range(tile_cols):
        for tile_row in range(tile_rows):
            logical_tile = tile_row * tile_cols + tile_col
            ra[logical_tile] = packed_pos
            packed_pos += 1

    return ra


def make_identity_reorder(M, N, tile_m=128, tile_n=128, device="cuda"):
    assert M % tile_m == 0
    assert N % tile_n == 0
    num_tiles = (M // tile_m) * (N // tile_n)
    return torch.arange(num_tiles, device=device, dtype=torch.int32)


def unpack_packed_to_normal(D_packed, RA, M, N, reldn, tile_m=128, tile_n=128):
    """
    Convert packed D back to normal [M, N] layout for correctness checking.

    D_packed physical layout is:

      packed tile id p is located at:

        packed_tile_row = p // reldn
        packed_tile_col = p %  reldn

      and occupies:

        rows = packed_tile_row * tile_m : ...
        cols = packed_tile_col * tile_n : ...

    RA maps logical_tile -> packed_tile.
    """
    assert D_packed.is_cuda
    assert RA.is_cuda
    assert M % tile_m == 0
    assert N % tile_n == 0

    tile_rows = M // tile_m
    tile_cols = N // tile_n
    num_tiles = tile_rows * tile_cols

    assert RA.numel() == num_tiles

    D_normal = torch.empty((M, N), device=D_packed.device, dtype=D_packed.dtype)

    # Small debug/test helper. This does per-tile GPU slice copies.
    # For production, fuse this inverse mapping into the next consumer kernel.
    ra_cpu = RA.detach().cpu().tolist()

    for logical_tile in range(num_tiles):
        packed_tile = int(ra_cpu[logical_tile])

        logical_tile_row = logical_tile // tile_cols
        logical_tile_col = logical_tile % tile_cols

        packed_tile_row = packed_tile // reldn
        packed_tile_col = packed_tile % reldn

        src_r0 = packed_tile_row * tile_m
        src_c0 = packed_tile_col * tile_n

        dst_r0 = logical_tile_row * tile_m
        dst_c0 = logical_tile_col * tile_n

        D_normal[
            dst_r0 : dst_r0 + tile_m,
            dst_c0 : dst_c0 + tile_n,
        ].copy_(
            D_packed[
                src_r0 : src_r0 + tile_m,
                src_c0 : src_c0 + tile_n,
            ]
        )

    return D_normal


def worker(rank, world, nccl_id, M, N, K, reldn, cseg, reorder_kind):
    torch.cuda.set_device(rank)
    torch.manual_seed(1234 + rank)

    ext = load_ooverlap_ext()

    if not hasattr(ext, "OverlapImpl"):
        raise RuntimeError(
            "ooverlap_ext does not expose OverlapImpl yet. "
            "Rebuild after adding the pybind class."
        )
    if not hasattr(ext, "generate_nccl_id"):
        raise RuntimeError(
            "ooverlap_ext does not expose generate_nccl_id yet. "
            "Rebuild after adding the pybind function."
        )

    tile_m, tile_n = 128, 128
    assert M % tile_m == 0 and N % tile_n == 0, "M/N must be multiples of 128"

    tile_rows = M // tile_m
    tile_cols = N // tile_n
    num_tiles = tile_rows * tile_cols

    assert reldn > 0
    assert sum(cseg) == num_tiles, f"sum(cseg) must equal num_tiles={num_tiles}"

    packed_tile_cols = reldn
    packed_tile_rows = (num_tiles + packed_tile_cols - 1) // packed_tile_cols

    packed_M = packed_tile_rows * tile_m
    packed_N = packed_tile_cols * tile_n

    ov = ext.OverlapImpl()
    ov.nccl_init(rank, world, nccl_id)
    ov.cutlass_init()
    ov.overlap_init()

    # A is (M,K) row-major.
    A = torch.randn((M, K), device="cuda", dtype=torch.float16)

    # Logical B_ref is (K,N); kernel expects B_packed as (N,K).
    B_ref = torch.randn((K, N), device="cuda", dtype=torch.float16)
    B_packed = B_ref.t().contiguous()

    if reorder_kind == "column_major":
        RA = make_column_major_reorder(M, N, tile_m, tile_n, device="cuda")
    elif reorder_kind == "identity":
        RA = make_identity_reorder(M, N, tile_m, tile_n, device="cuda")
    else:
        raise ValueError(f"Unknown reorder_kind={reorder_kind}")

    # Segment metadata, in packed tile order.
    cseg_cpu = torch.tensor(cseg, dtype=torch.int32)
    cseg_gpu = cseg_cpu.cuda(rank)

    # MM layout expected by current store_tail():
    #
    #   MM[0 : num_segments]                  -> segment counters
    #   MM[num_segments : num_segments+tiles] -> per-packed-tile done counters
    MM = torch.zeros((len(cseg) + num_tiles,), device="cuda", dtype=torch.int32)

    # This is the important part:
    #
    #   C_packed is NOT [M, N].
    #
    # For reldn=1:
    #
    #   C_packed shape = [num_tiles * 128, 128]
    #
    # Every GEMM output tile is physically contiguous.
    C_packed = torch.empty((packed_M, packed_N), device="cuda", dtype=torch.float16)

    algo = 0
    monitor = False

    ov.gemm_allreduce_overlap(
        A,
        B_packed,
        C_packed,
        MM,
        RA,
        int(reldn),
        cseg_cpu,
        cseg_gpu,
        algo,
        monitor,
    )
    torch.cuda.synchronize()

    # Reference: normal GEMM output, then normal allreduce.
    Cref = (A.float() @ B_ref.float()).half()
    ov.nccl_allreduce(Cref)
    torch.cuda.synchronize()

    # Unpack packed communication buffer back to normal matrix for comparison.
    C_unpacked = unpack_packed_to_normal(
        C_packed,
        RA,
        M,
        N,
        reldn,
        tile_m,
        tile_n,
    )
    torch.cuda.synchronize()

    max_err = (C_unpacked - Cref).abs().max().item()

    mm_host = MM.cpu().tolist()
    seg_counters = mm_host[:len(cseg)]
    tile_done = mm_host[len(cseg):]

    if rank == 0:
        print(f"[packed allreduce] M={M} N={N} K={K}")
        print(f"[packed allreduce] tile_rows={tile_rows} tile_cols={tile_cols} num_tiles={num_tiles}")
        print(f"[packed allreduce] reldn={reldn} packed_shape=({packed_M}, {packed_N})")
        print(f"[packed allreduce] reorder_kind={reorder_kind}")
        print(f"[packed allreduce] max_abs_err={max_err}")
        print(f"[packed allreduce] MM(seg counters)={seg_counters}")
        print(f"[packed allreduce] MM(tile_done)={tile_done}")

    assert max_err < 0.75, f"packed allreduce max_err too large: {max_err}"

    # tile_done is indexed by packed tile id.
    assert tile_done == [1] * num_tiles, (
        f"tile_done mismatch: got {tile_done}, expected {[1] * num_tiles}"
    )

    # If world > 1, communication wait kernels consume/reset segment counters.
    # If world == 1, overlap_impl returns after GEMM, so counters remain cseg.
    if world > 1:
        assert seg_counters == [0] * len(cseg), (
            f"segment counters should be zero after wait-kernel consumption: got {seg_counters}"
        )
    else:
        assert seg_counters == cseg, (
            f"with world=1, segment counters should remain cseg: got {seg_counters}, expected {cseg}"
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", type=int, default=2)
    ap.add_argument("--m", type=int, default=256)
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--k", type=int, default=128)

    ap.add_argument(
        "--reldn",
        type=int,
        default=1,
        help=(
            "Packed tile columns. Use --reldn 1 for fully contiguous per-tile packing. "
            "If reldn == tile_cols and RA is identity, this degenerates to normal layout."
        ),
    )

    ap.add_argument(
        "--reorder",
        choices=["column_major", "identity"],
        default="column_major",
        help="RA pattern. column_major physically proves the tile reorder.",
    )

    ap.add_argument(
        "--cseg",
        type=str,
        default=None,
        help=(
            'Comma-separated segment sizes in units of packed tiles, e.g. "2,2" for 4 tiles total. '
            "Default: one segment per tile."
        ),
    )

    args = ap.parse_args()

    assert torch.cuda.is_available(), "torch.cuda.is_available() is False"
    assert args.gpus >= 1, "--gpus must be >= 1"
    assert torch.cuda.device_count() >= args.gpus, "Not enough visible GPUs"

    tile_m, tile_n = 128, 128
    assert args.m % tile_m == 0 and args.n % tile_n == 0, "M/N must be multiples of 128"

    tile_rows = args.m // tile_m
    tile_cols = args.n // tile_n
    num_tiles = tile_rows * tile_cols

    assert args.reldn > 0, "--reldn must be > 0"
    assert args.reldn <= num_tiles, "--reldn cannot exceed number of tiles"

    if args.cseg is None:
        # Default: one segment per packed tile.
        cseg = [1] * num_tiles
    else:
        cseg = [int(x) for x in args.cseg.split(",") if x.strip()]
        assert sum(cseg) == num_tiles, f"sum(cseg) must equal num_tiles={num_tiles}"

    ext = load_ooverlap_ext()
    if not hasattr(ext, "generate_nccl_id"):
        raise RuntimeError(
            "ooverlap_ext does not expose generate_nccl_id yet. "
            "Rebuild after adding the pybind function."
        )

    nccl_id = ext.generate_nccl_id()

    mp.spawn(
        worker,
        args=(
            args.gpus,
            nccl_id,
            args.m,
            args.n,
            args.k,
            args.reldn,
            cseg,
            args.reorder,
        ),
        nprocs=args.gpus,
        join=True,
    )


if __name__ == "__main__":
    main()
