#!/usr/bin/env python3
import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Any, Optional, Tuple

import torch


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_ooverlap_ext():
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


def make_baseline_impl(ext: Any):
    if not hasattr(ext, "BaselineImpl"):
        return None
    baseline = ext.BaselineImpl()
    baseline.cublas_init()
    return baseline


def get_baseline_col(ext: Any):
    for name in ("baseline_gemm_col", "cublas_gemm_col", "gemm_col_baseline"):
        if hasattr(ext, name):
            return getattr(ext, name), name
    return None, None


def load_algo_dict(path=None):
    root = repo_root()
    path = root / "configs" / "AlgoDictSm90.json" if path is None else Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Could not find AlgoDictSm90 json: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    by_algo = {}
    for item in data["algorithms"]:
        algo = int(item["algo"])
        by_algo[algo] = {
            "algo": algo,
            "tile_m": int(item["tile_m"]),
            "tile_n": int(item["tile_n"]),
            "tile_k": int(item["tile_k"]),
            "cluster": [int(x) for x in item.get("cluster", [1, 1, 1])],
            "stages": item.get("stages", "auto"),
            "mainloop": str(item.get("mainloop", "unknown")),
            "epilogue": str(item.get("epilogue", "auto")),
            "scheduler": str(item.get("scheduler", "normal")),
        }
    return by_algo, path


def algo_info(algo, algo_dict):
    if algo not in algo_dict:
        known = sorted(algo_dict.keys())
        raise ValueError(
            f"Unsupported algo={algo}. Known algo range: {known[0]}..{known[-1]}, count={len(known)}"
        )
    return algo_dict[algo]


def make_identity_ra(M, N, tile_m, tile_n, device="cuda"):
    tile_rows = M // tile_m
    tile_cols = N // tile_n
    return torch.arange(tile_rows * tile_cols, device=device, dtype=torch.int32)


def make_column_major_ra(M, N, tile_m, tile_n, device="cuda"):
    tile_rows = M // tile_m
    tile_cols = N // tile_n
    num_tiles = tile_rows * tile_cols
    host = [0] * num_tiles
    packed = 0
    for tc in range(tile_cols):
        for tr in range(tile_rows):
            logical = tr * tile_cols + tc
            host[logical] = packed
            packed += 1
    return torch.tensor(host, device=device, dtype=torch.int32)


def make_segments(num_tiles, group_tiles, device="cuda"):
    if group_tiles <= 0:
        return torch.tensor([num_tiles], device=device, dtype=torch.int32)
    segs = []
    left = num_tiles
    while left > 0:
        x = min(group_tiles, left)
        segs.append(x)
        left -= x
    return torch.tensor(segs, device=device, dtype=torch.int32)


def unpack_packed_to_normal(D_packed_logical, RA, M, N, tile_m, tile_n, reldn):
    tile_rows = M // tile_m
    tile_cols = N // tile_n
    out = torch.empty((M, N), device=D_packed_logical.device, dtype=D_packed_logical.dtype)
    ra_cpu = RA.detach().cpu().tolist()

    for tr in range(tile_rows):
        for tc in range(tile_cols):
            logical_tile = tr * tile_cols + tc
            packed_tile = int(ra_cpu[logical_tile])
            pm = packed_tile // reldn
            pn = packed_tile % reldn
            src = D_packed_logical[
                pm * tile_m:(pm + 1) * tile_m,
                pn * tile_n:(pn + 1) * tile_n,
            ]
            out[
                tr * tile_m:(tr + 1) * tile_m,
                tc * tile_n:(tc + 1) * tile_n,
            ].copy_(src)
    return out


def safe_quantile_1d(x: torch.Tensor, q: float, max_samples: int = 1_000_000) -> float:
    x = x.reshape(-1)
    n = int(x.numel())
    if n == 0:
        return float("nan")
    if n > max_samples:
        step = (n + max_samples - 1) // max_samples
        x = x[::step][:max_samples]
    return float(torch.quantile(x, q).item())


def error_summary(x: torch.Tensor, y: torch.Tensor):
    diff = (x - y).abs().float().reshape(-1)
    return {
        "max_abs": float(diff.max().item()),
        "mean_abs": float(diff.mean().item()),
        "p99_abs": safe_quantile_1d(diff, 0.99),
        "p999_abs": safe_quantile_1d(diff, 0.999),
    }


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
    return float(start.elapsed_time(end)) / float(iters)


def fmt_ms(x):
    if x is None:
        return "SKIPPED"
    return f"{x:.6f} ms"


def fmt_tflops(x, flops):
    if x is None:
        return "SKIPPED"
    return f"{flops / (x * 1.0e-3) / 1.0e12:.2f}"


def print_algo_list(algo_dict):
    print("Available SM90 GEMM signal algos:")
    for algo in sorted(algo_dict):
        x = algo_dict[algo]
        print(
            f"algo={algo:4d} tile={x['tile_m']}x{x['tile_n']}x{x['tile_k']} "
            f"cluster={x['cluster']} stages={x.get('stages')} "
            f"mainloop={x['mainloop']} epilogue={x['epilogue']} scheduler={x.get('scheduler', 'normal')}"
        )


def resolve_check_ref(mode: str, baseline_col_fn, baseline_impl) -> str:
    if mode == "auto":
        if baseline_col_fn is not None:
            return "baseline_col"
        if baseline_impl is not None:
            return "baseline_impl"
        return "torch"
    if mode == "baseline_col" and baseline_col_fn is None:
        raise RuntimeError("--check-ref baseline_col requested, but ext.baseline_gemm_col was not found")
    if mode == "baseline_impl" and baseline_impl is None:
        raise RuntimeError("--check-ref baseline_impl requested, but ext.BaselineImpl was not found")
    return mode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--m", type=int, default=1024)
    ap.add_argument("--n", type=int, default=1024)
    ap.add_argument("--k", type=int, default=4096)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--iters", type=int, default=1000)
    ap.add_argument("--algo", type=int, default=0)
    ap.add_argument("--algo-dict", type=str, default=None)

    ap.add_argument("--check", action="store_true")
    ap.add_argument(
        "--check-ref",
        choices=["auto", "baseline_col", "baseline_impl", "torch"],
        default="baseline_impl",
        help="Correctness reference. auto prefers baseline_gemm_col, then BaselineImpl, then torch.",
    )
    ap.add_argument(
        "--check-atol",
        type=float,
        default=8.0,
        help="Default is loose enough for f16-accumulate/tile-order differences.",
    )
    ap.add_argument("--also-check-torch", action="store_true")
    ap.add_argument("--skip-eager", action="store_true")

    ap.add_argument("--layout", choices=["normal", "packed"], default="normal")
    ap.add_argument("--reorder", choices=["identity", "column_major"], default=None)
    ap.add_argument("--reldn", type=int, default=None)
    ap.add_argument("--group-tiles", type=int, default=0)
    ap.add_argument("--reset-mm", action="store_true")
    ap.add_argument("--list-algos", action="store_true")
    args = ap.parse_args()

    torch.cuda.set_device(args.device)
    torch.manual_seed(1234)

    algo_dict, algo_dict_path = load_algo_dict(args.algo_dict)
    if args.list_algos:
        print_algo_list(algo_dict)
        return

    ext = load_ooverlap_ext()
    baseline_impl = make_baseline_impl(ext)
    baseline_col_fn, baseline_col_name = get_baseline_col(ext)

    M, N, K = args.m, args.n, args.k
    info = algo_info(args.algo, algo_dict)
    tile_m = info["tile_m"]
    tile_n = info["tile_n"]
    tile_k = info["tile_k"]

    assert M % tile_m == 0, f"M={M} must be multiple of tile_m={tile_m}"
    assert N % tile_n == 0, f"N={N} must be multiple of tile_n={tile_n}"

    tile_rows = M // tile_m
    tile_cols = N // tile_n
    num_tiles = tile_rows * tile_cols

    if args.reldn is None:
        reldn = tile_cols if args.layout == "normal" else 1
    else:
        reldn = args.reldn
    if reldn <= 0:
        raise ValueError("reldn must be > 0")

    if args.reorder is None:
        reorder = "identity" if args.layout == "normal" else "column_major"
    else:
        reorder = args.reorder

    if reorder == "identity":
        RA = make_identity_ra(M, N, tile_m, tile_n, device="cuda")
    elif reorder == "column_major":
        RA = make_column_major_ra(M, N, tile_m, tile_n, device="cuda")
    else:
        raise ValueError(f"unknown reorder={reorder}")

    if args.layout == "normal":
        if reldn != tile_cols:
            raise ValueError(f"normal layout requires reldn=tile_cols={tile_cols}, got {reldn}")
        D_shape = (M, N)
    else:
        packed_tile_rows = math.ceil(num_tiles / reldn)
        D_shape = (packed_tile_rows * tile_m, reldn * tile_n)

    CommThr = make_segments(num_tiles, args.group_tiles, device="cuda")
    num_segments = int(CommThr.numel())
    MM = torch.empty((num_segments + num_tiles,), device="cuda", dtype=torch.int32)

    A = torch.randn((M, K), device="cuda", dtype=torch.float16)
    B_nk = torch.randn((N, K), device="cuda", dtype=torch.float16)
    C_baseline_impl = torch.empty((M, N), device="cuda", dtype=torch.float16)
    D_ref_col = torch.empty((N, M), device="cuda", dtype=torch.float16)
    C_torch = torch.empty((M, N), device="cuda", dtype=torch.float16)
    C_ours = torch.empty(D_shape, device="cuda", dtype=torch.float16)
    monitor = False

    def run_baseline_impl():
        if baseline_impl is None:
            raise RuntimeError("BaselineImpl not available")
        baseline_impl.gemm(A, B_nk, C_baseline_impl)

    def run_baseline_col():
        if baseline_col_fn is None:
            raise RuntimeError("baseline_gemm_col not available")
        baseline_col_fn(A, B_nk, D_ref_col)

    def run_torch():
        torch.matmul(A, B_nk.t(), out=C_torch)

    def run_ours():
        if args.reset_mm:
            MM.zero_()
        ext.gemm_signal_sm90(
            A, B_nk, C_ours, MM, RA, CommThr,
            int(reldn), int(args.algo), monitor,
        )

    def ours_normal_view():
        if args.layout == "normal":
            return C_ours[:M, :N]
        return unpack_packed_to_normal(C_ours, RA, M, N, tile_m, tile_n, reldn)

    check_ref = resolve_check_ref(args.check_ref, baseline_col_fn, baseline_impl)

    # Correctness pre-run.
    MM.zero_()
    if check_ref == "baseline_col":
        run_baseline_col()
        C_ref = D_ref_col.t()
    elif check_ref == "baseline_impl":
        run_baseline_impl()
        C_ref = C_baseline_impl
    else:
        run_torch()
        C_ref = C_torch

    if args.also_check_torch and check_ref != "torch":
        run_torch()

    run_ours()
    torch.cuda.synchronize()
    C_ours_normal = ours_normal_view()
    err = error_summary(C_ours_normal, C_ref)
    torch_err = error_summary(C_ours_normal, C_torch) if args.also_check_torch else None

    baseline_eager_ms = None
    ours_eager_ms = None
    if not args.skip_eager:
        # Timing baseline: prefer the same reference used for correctness, but avoid torch unless needed.
        if check_ref == "baseline_col":
            baseline_eager_ms = time_cuda(run_baseline_col, args.warmup, args.iters)
        elif check_ref == "baseline_impl":
            baseline_eager_ms = time_cuda(run_baseline_impl, args.warmup, args.iters)
        else:
            baseline_eager_ms = time_cuda(run_torch, args.warmup, args.iters)

        if not args.reset_mm:
            MM.zero_()
            torch.cuda.synchronize()
        ours_eager_ms = time_cuda(run_ours, args.warmup, args.iters)

    # Graph timing: BaselineImpl is the safest cuBLAS graph path in this repo.
    baseline_graph_ms = None
    if baseline_impl is not None:
        torch.cuda.synchronize()
        g_baseline = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g_baseline):
            baseline_impl.gemm(A, B_nk, C_baseline_impl)
        baseline_graph_ms = time_cuda(g_baseline.replay, args.warmup, args.iters)

    is_stream_k = info.get("scheduler", "normal") == "stream_k"
    our_graph_capture_ok = True
    our_graph_ms = None
    our_graph_error = None
    if is_stream_k:
        our_graph_capture_ok = False
        our_graph_error = "skipped: Stream-K workspace/state reinit is not graph-safe in this benchmark yet"
    else:
        try:
            if not args.reset_mm:
                MM.zero_()
                torch.cuda.synchronize()
            g_ours = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g_ours):
                if args.reset_mm:
                    MM.zero_()
                ext.gemm_signal_sm90(
                    A, B_nk, C_ours, MM, RA, CommThr,
                    int(reldn), int(args.algo), monitor,
                )
            our_graph_ms = time_cuda(g_ours.replay, args.warmup, args.iters)
        except Exception as e:
            our_graph_capture_ok = False
            our_graph_error = repr(e)

    flops = 2.0 * M * N * K
    print("========================================")
    print("GEMM-only baseline vs SM90 signal GEMM benchmark")
    print(f"algo={args.algo}")
    print(f"algo_dict={algo_dict_path}")
    print(
        f"tile={tile_m}x{tile_n}x{tile_k} cluster={info['cluster']} stages={info.get('stages')} "
        f"mainloop={info['mainloop']} epilogue={info['epilogue']} scheduler={info.get('scheduler', 'normal')}"
    )
    print(f"M={M} N={N} K={K}")
    print(f"tile_rows={tile_rows} tile_cols={tile_cols} num_tiles={num_tiles}")
    print(f"layout={args.layout} reorder={reorder}")
    print(f"reldn={reldn} output_shape={tuple(C_ours.shape)}")
    print(f"group_tiles={args.group_tiles} num_segments={num_segments}")
    print(f"reset_mm_each_iter={args.reset_mm}")
    print(f"check_ref={check_ref} check_atol={args.check_atol}")
    print("")
    print(f"max_abs_err:            {err['max_abs']}")
    print(f"mean_abs_err:           {err['mean_abs']}")
    print(f"p99_abs_err:            {err['p99_abs']}")
    print(f"p999_abs_err:           {err['p999_abs']}")
    if torch_err is not None:
        print(f"torch max_abs_err:      {torch_err['max_abs']}")
        print(f"torch mean_abs_err:     {torch_err['mean_abs']}")
    print("")
    print(f"baseline eager latency: {fmt_ms(baseline_eager_ms)}")
    print(f"baseline graph latency: {fmt_ms(baseline_graph_ms)}")
    print(f"our eager latency:      {fmt_ms(ours_eager_ms)}")
    if our_graph_capture_ok:
        print(f"our graph latency:      {fmt_ms(our_graph_ms)}")
    else:
        print("our graph latency:      CAPTURE FAILED")
        print(f"our graph error:        {our_graph_error}")
    print("")
    print(f"baseline eager TFLOP/s: {fmt_tflops(baseline_eager_ms, flops)}")
    print(f"baseline graph TFLOP/s: {fmt_tflops(baseline_graph_ms, flops)}")
    print(f"our eager TFLOP/s:      {fmt_tflops(ours_eager_ms, flops)}")
    if our_graph_capture_ok:
        print(f"our graph TFLOP/s:      {fmt_tflops(our_graph_ms, flops)}")
        print("")
        if ours_eager_ms is not None and our_graph_ms is not None:
            print(f"our eager / graph slow: {ours_eager_ms / our_graph_ms:.4f}x")
        if baseline_graph_ms is not None and our_graph_ms is not None:
            print(f"baseline graph / our graph speed: {baseline_graph_ms / our_graph_ms:.4f}x")
    print("========================================")

    if args.check:
        assert err["max_abs"] <= args.check_atol, f"max_abs_err too large: {err['max_abs']} > {args.check_atol}"


if __name__ == "__main__":
    main()
