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


def make_identity_reorder(M, N, tile_m=128, tile_n=128, device="cuda"):
    assert M % tile_m == 0
    assert N % tile_n == 0
    num_tiles = (M // tile_m) * (N // tile_n)
    return torch.arange(num_tiles, device=device, dtype=torch.int32)


def make_identity_row_map(M, N, tile_n=128, device="cuda"):
    # RE length matches the FlashOverlap-style micro-row domain: M * N / tile_n
    num_micro_rows = (M * N) // tile_n
    return torch.arange(num_micro_rows, device=device, dtype=torch.int32)


def worker(rank, world, op, nccl_id, M, N, K, reldn, cseg):
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

    assert sum(cseg) == num_tiles, f"sum(cseg) must equal num_tiles={num_tiles}"

    ov = ext.OverlapImpl()
    ov.nccl_init(rank, world, nccl_id)
    ov.cutlass_init()
    ov.overlap_init()

    # A is (M,K) row-major
    A = torch.randn((M, K), device="cuda", dtype=torch.float16)

    # Logical B_ref is (K,N); kernel expects B_packed as (N,K)
    B_ref = torch.randn((K, N), device="cuda", dtype=torch.float16)
    B_packed = B_ref.t().contiguous()

    # For signal-only bring-up, keep these as identity
    RA = make_identity_reorder(M, N, tile_m, tile_n, device="cuda")
    RE = make_identity_row_map(M, N, tile_n, device="cuda")

    # Segment metadata
    cseg_cpu = torch.tensor(cseg, dtype=torch.int32)
    cseg_gpu = cseg_cpu.cuda(rank)

    # IMPORTANT:
    # MM layout expected by current store_tail():
    #   MM[0 : num_segments]                  -> segment counters
    #   MM[num_segments : num_segments+tiles] -> per-tile done counters
    MM = torch.zeros((len(cseg) + num_tiles,), device="cuda", dtype=torch.int32)

    algo = 0
    monitor = False

    if op == "allreduce":
        # For signal-only bring-up, use ordinary MxN layout
        C = torch.empty((M, N), device="cuda", dtype=torch.float16)

        ov.gemm_allreduce_overlap(
            A, B_packed, C, MM, RA, int(reldn), cseg_cpu, cseg_gpu, algo, monitor
        )
        torch.cuda.synchronize()

        Cref = (A.float() @ B_ref.float()).half()
        ov.nccl_allreduce(Cref)
        torch.cuda.synchronize()

        max_err = (C - Cref).abs().max().item()

        mm_host = MM.cpu().tolist()
        seg_counters = mm_host[:len(cseg)]
        tile_done = mm_host[len(cseg):]

        if rank == 0:
            print(f"[overlap allreduce] max_abs_err={max_err}")
            print(f"[overlap allreduce] MM(seg)={seg_counters}, expected={cseg}")
            print(f"[overlap allreduce] MM(tile_done)={tile_done}")

        assert max_err < 0.75, f"allreduce overlap max_err too large: {max_err}"
        assert seg_counters == cseg, (
            f"segment counters mismatch: got {seg_counters}, expected {cseg}"
        )

    elif op == "reducescatter":
        # For signal-only bring-up, keep layout ordinary and RE = identity
        Ctmp = torch.empty((M, N), device="cuda", dtype=torch.float16)
        Dout = torch.empty((M // world, N), device="cuda", dtype=torch.float16)

        ov.gemm_reducescatter_overlap(
            A, B_packed, Ctmp, Dout, MM, RA, RE, int(reldn), cseg_cpu, cseg_gpu, algo, monitor
        )
        torch.cuda.synchronize()

        Cref = (A.float() @ B_ref.float()).half()
        Dref = torch.empty_like(Dout)
        ov.nccl_reducescatter(Cref, Dref)
        torch.cuda.synchronize()

        max_err = (Dout - Dref).abs().max().item()

        mm_host = MM.cpu().tolist()
        seg_counters = mm_host[:len(cseg)]
        tile_done = mm_host[len(cseg):]

        if rank == 0:
            print(f"[overlap reducescatter] max_abs_err={max_err}")
            print(f"[overlap reducescatter] MM(seg)={seg_counters}, expected={cseg}")
            print(f"[overlap reducescatter] MM(tile_done)={tile_done}")

        assert max_err < 0.75, f"reducescatter overlap max_err too large: {max_err}"
        assert seg_counters == cseg, (
            f"segment counters mismatch: got {seg_counters}, expected {cseg}"
        )

    else:
        raise ValueError(f"Unsupported op: {op}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--op", choices=["allreduce", "reducescatter"], default="allreduce")
    ap.add_argument("--gpus", type=int, default=2)
    ap.add_argument("--m", type=int, default=256)
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--k", type=int, default=128)
    ap.add_argument(
        "--reldn",
        type=int,
        default=None,
        help="For signal-only bring-up, keep this equal to tile_cols (= N/128).",
    )
    ap.add_argument(
        "--cseg",
        type=str,
        default=None,
        help='Comma-separated segment sizes in units of tiles, e.g. "2,2" for 4 tiles total. '
             "Default: one segment per tile.",
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

    # For signal-only bring-up, keep reldn equal to the original tile-grid width
    reldn = args.reldn if args.reldn is not None else tile_cols

    if args.cseg is None:
        # Default: one segment per tile to exercise signaling/overlap maximally
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
        args=(args.gpus, args.op, nccl_id, args.m, args.n, args.k, reldn, cseg),
        nprocs=args.gpus,
        join=True,
    )


if __name__ == "__main__":
    main()
