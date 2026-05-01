#!/usr/bin/env python3
"""
Plain SM90 CUTLASS GEMM benchmark for ooverlap.

This intentionally avoids RA/MM/CommThr/ReLDN and the reorder/signal epilogue.
It benchmarks only the new plain SM90 GEMM entry point:

    ext.gemm_plain_sm90(A, B_col, D_col, algo)

Tensor convention used here:

    A      physical row-major [M, K]
    B_col  physical row-major [N, K], interpreted as logical column-major B [K, N]
    D_col  physical row-major [N, M], interpreted as logical column-major D [M, N]

So the normal logical output is D_col.t(), shape [M, N].

Example:

  python test/bench_gemm_plain_sm90.py \
    --device 0 \
    --m 16384 --n 8192 --k 8192 \
    --algo 0 \
    --warmup 20 --iters 100 \
    --timing-mode eager \
    --check
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import statistics
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch


# ----------------------------- repo / extension -----------------------------


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_ooverlap_ext(root: Optional[Path] = None):
    if root is None:
        root = repo_root()

    so = root / "build" / "lib" / "ooverlap_ext.so"
    if not so.exists():
        raise FileNotFoundError(f"Could not find {so}. Build first.")

    module_name = "ooverlap_ext"
    if module_name in sys.modules:
        return sys.modules[module_name]

    spec = importlib.util.spec_from_file_location(module_name, str(so))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load Python extension spec for {so}")

    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


# ----------------------------- timing helpers -----------------------------


def cuda_sync() -> None:
    torch.cuda.synchronize()


def time_cuda_eager(fn: Callable[[], None], warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    cuda_sync()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(iters):
        fn()
    end.record()

    cuda_sync()
    return float(start.elapsed_time(end)) / float(iters)


def time_cuda_graph_unrolled(
    fn: Callable[[], None],
    warmup: int,
    iters: int,
    graph_repeats: int,
) -> float:
    if graph_repeats <= 0:
        raise ValueError("graph_repeats must be > 0")

    # Make sure any lazy init/cache allocation happens before capture.
    for _ in range(max(1, warmup)):
        fn()
    cuda_sync()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(graph_repeats):
            fn()

    for _ in range(max(1, warmup)):
        graph.replay()
    cuda_sync()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(iters):
        graph.replay()
    end.record()

    cuda_sync()
    return float(start.elapsed_time(end)) / float(iters * graph_repeats)


def percentile(vals: Sequence[float], q: float) -> float:
    xs = sorted(float(x) for x in vals)
    if not xs:
        return float("nan")
    if len(xs) == 1:
        return xs[0]
    pos = (len(xs) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return xs[lo]
    frac = pos - lo
    return xs[lo] * (1.0 - frac) + xs[hi] * frac


# ----------------------------- correctness helpers -----------------------------


def error_summary(x: torch.Tensor, y: torch.Tensor) -> Dict[str, float]:
    diff = (x - y).abs().float().reshape(-1)
    return {
        "max_abs": float(diff.max().item()),
        "mean_abs": float(diff.mean().item()),
        "p99_abs": float(torch.quantile(diff, 0.99).item()),
        "p999_abs": float(torch.quantile(diff, 0.999).item()),
    }


def assert_reasonable_close(
    ours: torch.Tensor,
    ref: torch.Tensor,
    atol: float,
    rtol: float,
) -> None:
    try:
        torch.testing.assert_close(ours, ref, atol=atol, rtol=rtol)
    except AssertionError as e:
        # Keep the original useful torch error, but make the default tolerance intent clear.
        raise AssertionError(
            f"plain SM90 output failed assert_close(atol={atol}, rtol={rtol}).\n{e}"
        ) from e


# ----------------------------- extension call helpers -----------------------------


def get_plain_func(ext):
    for name in ("gemm_plain_sm90", "plain_gemm_sm90", "gemm_sm90_plain"):
        if hasattr(ext, name):
            return getattr(ext, name), name
    raise AttributeError(
        "Could not find plain GEMM entry point on ooverlap_ext. "
        "Expected ext.gemm_plain_sm90(A, B_col, D_col, algo)."
    )


def maybe_get_cublas_col_func(ext):
    # Optional helper if you later expose a true column-output cublas baseline.
    for name in (
        "baseline_gemm_col",
        "gemm_baseline_col",
        "cublas_gemm_col",
        "gemm_cublas_col",
    ):
        if hasattr(ext, name):
            return getattr(ext, name), name
    return None, None


def call_plain(fn, A: torch.Tensor, B_col: torch.Tensor, D_col: torch.Tensor, algo: int) -> None:
    fn(A, B_col, D_col, int(algo))


def call_cublas_col(fn, A: torch.Tensor, B_col: torch.Tensor, D_col: torch.Tensor) -> None:
    fn(A, B_col, D_col)


# ----------------------------- main benchmark -----------------------------


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--m", type=int, required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--k", type=int, required=True)
    ap.add_argument("--algo", type=int, required=True)

    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--timing-mode", choices=["eager", "graph", "both"], default="eager")
    ap.add_argument("--graph-repeats", type=int, default=100)

    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--atol", type=float, default=8.0)
    ap.add_argument("--rtol", type=float, default=2.0e-2)
    ap.add_argument("--no-torch-baseline", action="store_true")
    ap.add_argument("--try-cublas-col-baseline", action="store_true")

    args = ap.parse_args()

    if args.warmup < 0:
        raise ValueError("--warmup must be >= 0")
    if args.iters <= 0:
        raise ValueError("--iters must be > 0")
    if args.graph_repeats <= 0:
        raise ValueError("--graph-repeats must be > 0")

    torch.cuda.set_device(args.device)
    device = torch.device("cuda", args.device)
    torch.manual_seed(args.seed)

    ext = load_ooverlap_ext()
    plain_fn, plain_name = get_plain_func(ext)
    cublas_col_fn, cublas_col_name = maybe_get_cublas_col_func(ext)

    M, N, K = args.m, args.n, args.k

    A = torch.randn((M, K), device=device, dtype=torch.float16)
    B_col = torch.randn((N, K), device=device, dtype=torch.float16)

    # Plain CUTLASS output is physical [N, M]. Logical output is D_col.t() = [M, N].
    D_col = torch.empty((N, M), device=device, dtype=torch.float16)

    # Row-major reference output.
    C_ref = torch.empty((M, N), device=device, dtype=torch.float16)

    # Optional true column-output baseline. Only used if your extension exposes it.
    D_cublas_col = torch.empty((N, M), device=device, dtype=torch.float16)

    B_view = B_col.t()

    def torch_row_fn() -> None:
        torch.matmul(A, B_view, out=C_ref)

    def plain_fn_bound() -> None:
        call_plain(plain_fn, A, B_col, D_col, args.algo)

    def cublas_col_fn_bound() -> None:
        assert cublas_col_fn is not None
        call_cublas_col(cublas_col_fn, A, B_col, D_cublas_col)

    # First touch and extension lazy init outside timing.
    torch_row_fn()
    plain_fn_bound()
    cuda_sync()

    flops = 2.0 * M * N * K

    print("========================================")
    print("bench_gemm_plain_sm90")
    print(f"shape:        M={M} N={N} K={K}")
    print(f"device:       {args.device} ({torch.cuda.get_device_name(args.device)})")
    print(f"entry:        {plain_name}")
    print(f"algo:         {args.algo}")
    print("layout:       A=[M,K] row, B_col=[N,K], D_col=[N,M], logical D=D_col.t()")
    print(f"warmup:       {args.warmup}")
    print(f"iters:        {args.iters}")
    print(f"timing_mode:  {args.timing_mode}")
    print(f"graph_repeat: {args.graph_repeats}")
    print("========================================")

    if args.check:
        torch_row_fn()
        plain_fn_bound()
        cuda_sync()

        ours_normal = D_col.t()
        err = error_summary(ours_normal, C_ref)
        print("")
        print("Correctness vs torch.matmul(A, B_col.t()):")
        print(f"  max_abs:  {err['max_abs']:.6f}")
        print(f"  mean_abs: {err['mean_abs']:.6f}")
        print(f"  p99_abs:  {err['p99_abs']:.6f}")
        print(f"  p999_abs: {err['p999_abs']:.6f}")
        assert_reasonable_close(ours_normal, C_ref, atol=args.atol, rtol=args.rtol)

        if args.try_cublas_col_baseline and cublas_col_fn is not None:
            cublas_col_fn_bound()
            cuda_sync()
            err2 = error_summary(ours_normal, D_cublas_col.t())
            print("")
            print(f"Correctness vs {cublas_col_name}(A, B_col, D_col):")
            print(f"  max_abs:  {err2['max_abs']:.6f}")
            print(f"  mean_abs: {err2['mean_abs']:.6f}")
            print(f"  p99_abs:  {err2['p99_abs']:.6f}")
            print(f"  p999_abs: {err2['p999_abs']:.6f}")

    results: List[Tuple[str, float]] = []

    if not args.no_torch_baseline:
        torch_ms = time_cuda_eager(torch_row_fn, args.warmup, args.iters)
        results.append(("torch_row_eager", torch_ms))

    if args.try_cublas_col_baseline:
        if cublas_col_fn is None:
            print("")
            print("column-output cuBLAS baseline skipped: no baseline_gemm_col-like function found")
        else:
            cublas_ms = time_cuda_eager(cublas_col_fn_bound, args.warmup, args.iters)
            results.append((f"{cublas_col_name}_eager", cublas_ms))

    if args.timing_mode in ("eager", "both"):
        ms = time_cuda_eager(plain_fn_bound, args.warmup, args.iters)
        results.append(("plain_eager", ms))

    if args.timing_mode in ("graph", "both"):
        try:
            ms = time_cuda_graph_unrolled(
                plain_fn_bound,
                warmup=args.warmup,
                iters=args.iters,
                graph_repeats=args.graph_repeats,
            )
            results.append((f"plain_graph_x{args.graph_repeats}", ms))
        except Exception as e:
            print("")
            print(f"plain graph timing failed: {repr(e)}")

    print("")
    print("Timing:")
    for name, ms in results:
        tflops = flops / (ms * 1.0e-3) / 1.0e12
        print(f"  {name:28s} {ms:10.6f} ms  {tflops:10.2f} TFLOP/s")

    print("========================================")


if __name__ == "__main__":
    main()
