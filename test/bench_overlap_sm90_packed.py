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
    tile_rows = M // tile_m
    tile_cols = N // tile_n
    num_tiles = tile_rows * tile_cols
    return torch.arange(num_tiles, device=device, dtype=torch.int32)


def make_segments(num_tiles, group_tiles):
    segs = []
    remaining = num_tiles
    while remaining > 0:
        x = min(group_tiles, remaining)
        segs.append(x)
        remaining -= x
    return segs


def time_cuda(fn, warmup, iters):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(iters):
        fn()
    end.record()

    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def worker(rank, world, nccl_id, M, N, K, reldn, group_tiles, warmup, iters, reorder, algo, skip_extra):
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

    overlap_cseg = make_segments(num_tiles, group_tiles)
    full_cseg = [num_tiles]

    ov = ext.OverlapImpl()
    ov.nccl_init(rank, world, nccl_id)
    ov.cutlass_init()
    ov.overlap_init()

    A = torch.randn((M, K), device="cuda", dtype=torch.float16)
    B_ref = torch.randn((K, N), device="cuda", dtype=torch.float16)
    B_packed = B_ref.t().contiguous()

    C_packed = torch.empty((packed_M, packed_N), device="cuda", dtype=torch.float16)
    C_packed_full = torch.empty((packed_M, packed_N), device="cuda", dtype=torch.float16)
    C_torch = torch.empty((M, N), device="cuda", dtype=torch.float16)
    C_nccl_only = torch.randn((packed_M, packed_N), device="cuda", dtype=torch.float16)

    if reorder == "column_major":
        RA = make_column_major_reorder(M, N, tile_m, tile_n, device="cuda")
    elif reorder == "identity":
        RA = make_identity_reorder(M, N, tile_m, tile_n, device="cuda")
    else:
        raise ValueError(f"unknown reorder={reorder}")

    overlap_cseg_cpu = torch.tensor(overlap_cseg, dtype=torch.int32)
    overlap_cseg_gpu = overlap_cseg_cpu.cuda(rank)

    full_cseg_cpu = torch.tensor(full_cseg, dtype=torch.int32)
    full_cseg_gpu = full_cseg_cpu.cuda(rank)

    MM_overlap = torch.empty((len(overlap_cseg) + num_tiles,), device="cuda", dtype=torch.int32)
    MM_full = torch.empty((len(full_cseg) + num_tiles,), device="cuda", dtype=torch.int32)

    monitor = False

    def run_packed_overlap():
        MM_overlap.zero_()
        ov.gemm_allreduce_overlap(
            A,
            B_packed,
            C_packed,
            MM_overlap,
            RA,
            int(reldn),
            overlap_cseg_cpu,
            overlap_cseg_gpu,
            int(algo),
            monitor,
        )

    def run_packed_full_segment():
        MM_full.zero_()
        ov.gemm_allreduce_overlap(
            A,
            B_packed,
            C_packed_full,
            MM_full,
            RA,
            int(reldn),
            full_cseg_cpu,
            full_cseg_gpu,
            int(algo),
            monitor,
        )

    def run_torch_matmul_plus_nccl():
        torch.matmul(A, B_ref, out=C_torch)
        ov.nccl_allreduce(C_torch)

    def run_nccl_only_full():
        ov.nccl_allreduce(C_nccl_only)

    overlap_ms = time_cuda(run_packed_overlap, warmup, iters)
    full_segment_ms = time_cuda(run_packed_full_segment, warmup, iters)

    torch_baseline_ms = None
    nccl_only_ms = None

    if not skip_extra:
        torch_baseline_ms = time_cuda(run_torch_matmul_plus_nccl, warmup, iters)
        nccl_only_ms = time_cuda(run_nccl_only_full, warmup, iters)

    packed_gemm_only_ms = None
    if world == 1:
        packed_gemm_only_ms = full_segment_ms

    if rank == 0:
        speedup_vs_full = full_segment_ms / overlap_ms if overlap_ms > 0 else float("nan")

        print("========================================")
        print(f"M={M} N={N} K={K}")
        print(f"tile_rows={tile_rows} tile_cols={tile_cols} num_tiles={num_tiles}")
        print(f"reldn={reldn} packed_shape=({packed_M}, {packed_N})")
        print(f"reorder={reorder}")
        print(f"algo={algo}")
        print("")
        print(f"overlap group_tiles={group_tiles}")
        print(f"overlap num_segments={len(overlap_cseg)}")
        print(f"overlap segments={overlap_cseg[:16]}{' ...' if len(overlap_cseg) > 16 else ''}")
        print("")
        print(f"packed overlap latency:        {overlap_ms:.4f} ms")
        print(f"packed full-segment latency:   {full_segment_ms:.4f} ms")

        if torch_baseline_ms is not None:
            print(f"torch matmul + NCCL latency:   {torch_baseline_ms:.4f} ms")
        if nccl_only_ms is not None:
            print(f"NCCL-only full-buffer latency: {nccl_only_ms:.4f} ms")
        if packed_gemm_only_ms is not None:
            print(f"packed GEMM-only approx:       {packed_gemm_only_ms:.4f} ms")

        print("")
        print(f"speedup vs packed full-seg:    {speedup_vs_full:.4f}x")
        print("========================================")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", type=int, default=2)
    ap.add_argument("--m", type=int, default=1024)
    ap.add_argument("--n", type=int, default=1024)
    ap.add_argument("--k", type=int, default=4096)
    ap.add_argument("--reldn", type=int, default=1)
    ap.add_argument("--group-tiles", type=int, default=8)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--reorder", choices=["column_major", "identity"], default="column_major")
    ap.add_argument("--algo", type=int, default=1)
    ap.add_argument("--skip-extra", action="store_true")
    args = ap.parse_args()

    assert torch.cuda.is_available()
    assert args.gpus >= 1
    assert torch.cuda.device_count() >= args.gpus

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
            args.reorder,
            args.algo,
            args.skip_extra,
        ),
        nprocs=args.gpus,
        join=True,
    )


if __name__ == "__main__":
    main()
