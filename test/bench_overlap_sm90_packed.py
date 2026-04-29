import argparse
import importlib.util
import time
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


def make_segments(num_tiles, group_tiles):
    segs = []
    remaining = num_tiles
    while remaining > 0:
        x = min(group_tiles, remaining)
        segs.append(x)
        remaining -= x
    return segs


def worker(rank, world, nccl_id, M, N, K, reldn, group_tiles, warmup, iters):
    torch.cuda.set_device(rank)
    torch.manual_seed(1234 + rank)

    ext = load_ooverlap_ext()

    tile_m = 128
    tile_n = 128

    assert M % tile_m == 0
    assert N % tile_n == 0

    tile_rows = M // tile_m
    tile_cols = N // tile_n
    num_tiles = tile_rows * tile_cols

    packed_tile_cols = reldn
    packed_tile_rows = (num_tiles + packed_tile_cols - 1) // packed_tile_cols
    packed_M = packed_tile_rows * tile_m
    packed_N = packed_tile_cols * tile_n

    cseg = make_segments(num_tiles, group_tiles)

    ov = ext.OverlapImpl()
    ov.nccl_init(rank, world, nccl_id)
    ov.cutlass_init()
    ov.overlap_init()

    A = torch.randn((M, K), device="cuda", dtype=torch.float16)
    B_ref = torch.randn((K, N), device="cuda", dtype=torch.float16)
    B_packed = B_ref.t().contiguous()

    C_packed = torch.empty((packed_M, packed_N), device="cuda", dtype=torch.float16)

    RA = make_column_major_reorder(M, N, tile_m, tile_n, device="cuda")

    cseg_cpu = torch.tensor(cseg, dtype=torch.int32)
    cseg_gpu = cseg_cpu.cuda(rank)

    MM = torch.empty((len(cseg) + num_tiles,), device="cuda", dtype=torch.int32)

    algo = 0
    monitor = False

    # Warmup.
    for _ in range(warmup):
        MM.zero_()
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

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(iters):
        MM.zero_()
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
    end.record()

    torch.cuda.synchronize()

    ms = start.elapsed_time(end) / iters

    if rank == 0:
        print("========================================")
        print(f"M={M} N={N} K={K}")
        print(f"tile_rows={tile_rows} tile_cols={tile_cols} num_tiles={num_tiles}")
        print(f"reldn={reldn} packed_shape=({packed_M}, {packed_N})")
        print(f"group_tiles={group_tiles}")
        print(f"num_segments={len(cseg)}")
        print(f"segments={cseg[:16]}{' ...' if len(cseg) > 16 else ''}")
        print(f"overlap avg latency: {ms:.4f} ms")
        print("========================================")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", type=int, default=2)
    ap.add_argument("--m", type=int, default=1024)
    ap.add_argument("--n", type=int, default=1024)
    ap.add_argument("--k", type=int, default=4096)
    ap.add_argument("--reldn", type=int, default=1)
    ap.add_argument("--group-tiles", type=int, default=8)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iters", type=int, default=100)
    args = ap.parse_args()

    ext = load_ooverlap_ext()
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
            args.group_tiles,
            args.warmup,
            args.iters,
        ),
        nprocs=args.gpus,
        join=True,
    )


if __name__ == "__main__":
    main()
