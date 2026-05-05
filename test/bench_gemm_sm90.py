#!/usr/bin/env python3

import argparse
import sys
from pathlib import Path

import torch


def import_ooverlap_ext(ext_dir=None):
    if ext_dir is not None:
        sys.path.insert(0, str(Path(ext_dir).resolve()))

    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "build" / "lib"))

    import ooverlap_ext
    return ooverlap_ext


def make_segments(tile_num, num_segments, device):
    assert num_segments > 0
    assert tile_num >= num_segments

    base = tile_num // num_segments
    rem = tile_num % num_segments

    seg = [base + (1 if i < rem else 0) for i in range(num_segments)]
    return torch.tensor(seg, device=device, dtype=torch.int32)


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


def tflops(m, n, k, ms):
    return (2.0 * m * n * k) / (ms * 1.0e9)


def max_mean_abs(x, y):
    diff = (x.float() - y.float()).abs()
    return diff.max().item(), diff.mean().item()


def unpack_signal_output(packed, m, n, tile_m, tile_n, rldn):
    tile_rows = m // tile_m
    tile_cols = n // tile_n

    out = torch.empty((m, n), device=packed.device, dtype=packed.dtype)

    for tr in range(tile_rows):
        for tc in range(tile_cols):
            tile_id = tr * tile_cols + tc
            pr = tile_id // rldn
            pc = tile_id % rldn

            src = packed[
                pr * tile_m : (pr + 1) * tile_m,
                pc * tile_n : (pc + 1) * tile_n,
            ]

            out[
                tr * tile_m : (tr + 1) * tile_m,
                tc * tile_n : (tc + 1) * tile_n,
            ].copy_(src)

    return out


def check_shape(m, n, k, tile_m, tile_n):
    if m % tile_m != 0:
        raise ValueError(f"M={m} must be divisible by tile_m={tile_m}")
    if n % tile_n != 0:
        raise ValueError(f"N={n} must be divisible by tile_n={tile_n}")
    if k <= 0:
        raise ValueError("K must be > 0")


def bench_baseline(ext, a, b, ref, args):
    d = torch.empty((args.n, args.m), device=a.device, dtype=torch.float16)

    def run():
        ext.baseline_gemm_col(a, b, d)

    ms = time_cuda(run, args.warmup, args.iters)
    got = d.t().contiguous()

    if args.check:
        mx, mean = max_mean_abs(got, ref)
        print(f"baseline_gemm_col: {ms:.4f} ms  {tflops(args.m, args.n, args.k, ms):.2f} TFLOP/s  max={mx:.5f} mean={mean:.5f}")
    else:
        print(f"baseline_gemm_col: {ms:.4f} ms  {tflops(args.m, args.n, args.k, ms):.2f} TFLOP/s")


def bench_plain(ext, a, b, ref, args):
    d = torch.empty((args.n, args.m), device=a.device, dtype=torch.float16)

    def run():
        ext.gemm_plain_sm90(a, b, d, args.plain_algo)

    ms = time_cuda(run, args.warmup, args.iters)
    got = d.t().contiguous()

    if args.check:
        mx, mean = max_mean_abs(got, ref)
        print(f"plain_sm90 algo={args.plain_algo}: {ms:.4f} ms  {tflops(args.m, args.n, args.k, ms):.2f} TFLOP/s  max={mx:.5f} mean={mean:.5f}")
    else:
        print(f"plain_sm90 algo={args.plain_algo}: {ms:.4f} ms  {tflops(args.m, args.n, args.k, ms):.2f} TFLOP/s")


def bench_signal(ext, a, b, ref, args):
    tile_rows = args.m // args.tile_m
    tile_cols = args.n // args.tile_n
    tile_num = tile_rows * tile_cols

    rldn = args.rldn if args.rldn > 0 else tile_cols
    packed_rows = (tile_num + rldn - 1) // rldn

    d = torch.empty(
        (packed_rows * args.tile_m, rldn * args.tile_n),
        device=a.device,
        dtype=torch.float16,
    )

    comm_thr = make_segments(tile_num, args.segments, a.device)
    ra = torch.arange(tile_num, device=a.device, dtype=torch.int32)

    mm_extra = tile_num + 1 if args.monitor else 0
    mm = torch.zeros(args.segments + tile_num + mm_extra, device=a.device, dtype=torch.int32)

    def run():
        mm.zero_()
        ext.gemm_signal_sm90(
            a,
            b,
            d,
            mm,
            ra,
            comm_thr,
            rldn,
            args.signal_algo,
            args.monitor,
        )

    ms = time_cuda(run, args.warmup, args.iters)

    if args.check:
        got = unpack_signal_output(d, args.m, args.n, args.tile_m, args.tile_n, rldn)
        mx, mean = max_mean_abs(got, ref)
        print(f"signal_sm90 algo={args.signal_algo}: {ms:.4f} ms  {tflops(args.m, args.n, args.k, ms):.2f} TFLOP/s  max={mx:.5f} mean={mean:.5f}")
    else:
        print(f"signal_sm90 algo={args.signal_algo}: {ms:.4f} ms  {tflops(args.m, args.n, args.k, ms):.2f} TFLOP/s")


def main():
    parser = argparse.ArgumentParser(
        description="Eager-mode SM90 GEMM benchmark for plain and signal kernels."
    )

    parser.add_argument("--mode", choices=["plain", "signal", "both"], default="both")
    parser.add_argument("--m", type=int, default=4096)
    parser.add_argument("--n", type=int, default=4096)
    parser.add_argument("--k", type=int, default=4096)

    parser.add_argument("--plain-algo", type=int, default=0)
    parser.add_argument("--signal-algo", type=int, default=0)

    parser.add_argument("--tile-m", type=int, default=128)
    parser.add_argument("--tile-n", type=int, default=128)
    parser.add_argument("--rldn", type=int, default=0)
    parser.add_argument("--segments", type=int, default=1)

    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--monitor", action="store_true")
    parser.add_argument("--no-check", dest="check", action="store_false")
    parser.add_argument("--with-baseline", action="store_true")
    parser.add_argument("--ext-dir", type=str, default=None)

    parser.set_defaults(check=True)

    args = parser.parse_args()

    check_shape(args.m, args.n, args.k, args.tile_m, args.tile_n)

    tile_num = (args.m // args.tile_m) * (args.n // args.tile_n)
    if args.segments <= 0 or args.segments > tile_num:
        raise ValueError(f"segments must be in [1, {tile_num}]")

    torch.cuda.set_device(args.device)
    torch.manual_seed(args.seed)

    ext = import_ooverlap_ext(args.ext_dir)

    a = torch.randn((args.m, args.k), device="cuda", dtype=torch.float16)
    b = torch.randn((args.n, args.k), device="cuda", dtype=torch.float16)

    ref = None
    if args.check:
        ref = torch.matmul(a.float(), b.t().float()).half()
        torch.cuda.synchronize()

    print(
        f"M={args.m} N={args.n} K={args.k} "
        f"tile=({args.tile_m},{args.tile_n}) "
        f"warmup={args.warmup} iters={args.iters}"
    )

    if args.with_baseline:
        bench_baseline(ext, a, b, ref, args)

    if args.mode in ("plain", "both"):
        bench_plain(ext, a, b, ref, args)

    if args.mode in ("signal", "both"):
        bench_signal(ext, a, b, ref, args)


if __name__ == "__main__":
    main()
