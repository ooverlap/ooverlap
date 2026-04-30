import argparse
import importlib.util
import json
import math
from pathlib import Path

import torch


def repo_root():
    return Path(__file__).resolve().parents[1]


def load_ooverlap_ext():
    root = repo_root()
    so = root / "build" / "lib" / "ooverlap_ext.so"
    if not so.exists():
        raise FileNotFoundError(f"Could not find {so}. Build first.")
    spec = importlib.util.spec_from_file_location("ooverlap_ext", str(so))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_algo_dict(path=None):
    root = repo_root()
    if path is None:
        path = root / "configs" / "AlgoDictSm90.json"
    else:
        path = Path(path)

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
            "cluster": [int(x) for x in item["cluster"]],
            "mainloop": str(item["mainloop"]),
            "epilogue": str(item["epilogue"]),
        }

    return by_algo, path


def algo_info(algo, algo_dict):
    if algo not in algo_dict:
        known = sorted(algo_dict.keys())
        raise ValueError(
            f"Unsupported algo={algo}. "
            f"Known algo range: {known[0]}..{known[-1]}, count={len(known)}"
        )
    return algo_dict[algo]


def make_identity_ra(M, N, tile_m, tile_n, device="cuda"):
    assert M % tile_m == 0
    assert N % tile_n == 0
    tile_rows = M // tile_m
    tile_cols = N // tile_n
    num_tiles = tile_rows * tile_cols
    return torch.arange(num_tiles, device=device, dtype=torch.int32)


def make_column_major_ra(M, N, tile_m, tile_n, device="cuda"):
    assert M % tile_m == 0
    assert N % tile_n == 0

    tile_rows = M // tile_m
    tile_cols = N // tile_n
    num_tiles = tile_rows * tile_cols

    ra = torch.empty((num_tiles,), device=device, dtype=torch.int32)

    packed = 0
    for tc in range(tile_cols):
        for tr in range(tile_rows):
            logical = tr * tile_cols + tc
            ra[logical] = packed
            packed += 1

    return ra


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


def unpack_packed_to_normal(D_packed, RA, M, N, tile_m, tile_n, reldn):
    tile_rows = M // tile_m
    tile_cols = N // tile_n

    out = torch.empty((M, N), device=D_packed.device, dtype=D_packed.dtype)
    ra_cpu = RA.detach().cpu().tolist()

    for tr in range(tile_rows):
        for tc in range(tile_cols):
            logical_tile = tr * tile_cols + tc
            packed_tile = int(ra_cpu[logical_tile])

            pm = packed_tile // reldn
            pn = packed_tile % reldn

            src = D_packed[
                pm * tile_m:(pm + 1) * tile_m,
                pn * tile_n:(pn + 1) * tile_n,
            ]

            out[
                tr * tile_m:(tr + 1) * tile_m,
                tc * tile_n:(tc + 1) * tile_n,
            ].copy_(src)

    return out


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
            f"algo={algo:4d} "
            f"tile={x['tile_m']}x{x['tile_n']}x{x['tile_k']} "
            f"cluster={x['cluster']} "
            f"mainloop={x['mainloop']} "
            f"epilogue={x['epilogue']}"
        )


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
    ap.add_argument("--skip-eager", action="store_true")

    ap.add_argument(
        "--layout",
        choices=["normal", "packed"],
        default="normal",
        help=(
            "normal: identity RA and D is normal [M,N]. "
            "packed: packed RA and D is packed tile buffer."
        ),
    )

    ap.add_argument(
        "--reorder",
        choices=["identity", "column_major"],
        default=None,
        help=(
            "RA order. Default: identity for normal layout, "
            "column_major for packed layout."
        ),
    )

    ap.add_argument(
        "--reldn",
        type=int,
        default=None,
        help=(
            "Packed tile columns. Default: tile_cols for normal layout, "
            "1 for packed layout."
        ),
    )

    ap.add_argument(
        "--group-tiles",
        type=int,
        default=0,
        help="0 means one full segment. Otherwise segment by this many tiles.",
    )

    ap.add_argument(
        "--reset-mm",
        action="store_true",
        help=(
            "Zero MM before every measured kernel call. "
            "This is correct for signal timing but includes MM reset overhead."
        ),
    )

    ap.add_argument(
        "--list-algos",
        action="store_true",
        help="Print generated algos from AlgoDictSm90.json and exit.",
    )

    args = ap.parse_args()

    torch.cuda.set_device(args.device)
    torch.manual_seed(1234)

    algo_dict, algo_dict_path = load_algo_dict(args.algo_dict)

    if args.list_algos:
        print_algo_list(algo_dict)
        return

    ext = load_ooverlap_ext()

    M = args.m
    N = args.n
    K = args.k

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
        if args.layout == "normal":
            reldn = tile_cols
        else:
            reldn = 1
    else:
        reldn = args.reldn

    if reldn <= 0:
        raise ValueError("reldn must be > 0")

    if args.reorder is None:
        if args.layout == "normal":
            reorder = "identity"
        else:
            reorder = "column_major"
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
            raise ValueError(
                f"normal layout requires reldn=tile_cols={tile_cols}, got {reldn}"
            )
        D_shape = (M, N)
    else:
        packed_tile_rows = math.ceil(num_tiles / reldn)
        D_shape = (packed_tile_rows * tile_m, reldn * tile_n)

    CommThr = make_segments(num_tiles, args.group_tiles, device="cuda")
    num_segments = int(CommThr.numel())
    MM = torch.empty((num_segments + num_tiles,), device="cuda", dtype=torch.int32)

    A = torch.randn((M, K), device="cuda", dtype=torch.float16)

    # BaselineImpl and our SM90 wrapper both expect B as [N, K].
    B_packed = torch.randn((N, K), device="cuda", dtype=torch.float16)

    C_baseline = torch.empty((M, N), device="cuda", dtype=torch.float16)

    baseline = ext.BaselineImpl()
    baseline.cublas_init()
    
    C_ours = torch.empty(D_shape, device="cuda", dtype=torch.float16)

    monitor = False

    def run_torch_eager():
        baseline.gemm(A, B_packed, C_baseline)

    def run_ours_eager():
        if args.reset_mm:
            MM.zero_()
        ext.gemm_signal_sm90(
            A,
            B_packed,
            C_ours,
            MM,
            RA,
            CommThr,
            int(reldn),
            int(args.algo),
            monitor,
        )

    # Correctness check before timing. Always reset MM here.
    MM.zero_()
    run_torch_eager()
    ext.gemm_signal_sm90(
        A,
        B_packed,
        C_ours,
        MM,
        RA,
        CommThr,
        int(reldn),
        int(args.algo),
        monitor,
    )
    torch.cuda.synchronize()

    physical_rows = C_ours.shape[0]
    C_ours_logical = torch.as_strided(
        C_ours,
        size=tuple(C_ours.shape),
        stride=(1, physical_rows),
    )
    
    if args.layout == "normal":
        C_ours_normal = C_ours_logical
    else:
        C_ours_normal = unpack_packed_to_normal(
            C_ours_logical,
            RA,
            M,
            N,
            tile_m,
            tile_n,
            reldn,
        )

    max_err = (C_ours_normal - C_baseline).abs().max().item()

    baseline_eager_ms = None
    ours_eager_ms = None

    if not args.skip_eager:
        baseline_eager_ms = time_cuda(run_torch_eager, args.warmup, args.iters)

        # Make no-reset mode deterministic enough before timing.
        if not args.reset_mm:
            MM.zero_()
            torch.cuda.synchronize()

        ours_eager_ms = time_cuda(run_ours_eager, args.warmup, args.iters)

    # Capture cuBLAS baseline GEMM.
    torch.cuda.synchronize()
    g_baseline = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g_baseline):
        baseline.gemm(A, B_packed, C_baseline)

    baseline_graph_ms = time_cuda(g_baseline.replay, args.warmup, args.iters) 

    # Capture our GEMM.
    our_graph_capture_ok = True
    our_graph_ms = None
    our_graph_error = None

    try:
        if not args.reset_mm:
            MM.zero_()
            torch.cuda.synchronize()

        torch.cuda.synchronize()
        g_ours = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g_ours):
            if args.reset_mm:
                MM.zero_()
            ext.gemm_signal_sm90(
                A,
                B_packed,
                C_ours,
                MM,
                RA,
                CommThr,
                int(reldn),
                int(args.algo),
                monitor,
            )

        our_graph_ms = time_cuda(g_ours.replay, args.warmup, args.iters)

    except Exception as e:
        our_graph_capture_ok = False
        our_graph_error = repr(e)

    flops = 2.0 * M * N * K

    print("========================================")
    print("GEMM-only cuBLAS baseline vs SM90 signal GEMM benchmark")
    print(f"algo={args.algo}")
    print(f"algo_dict={algo_dict_path}")
    print(
        f"tile={tile_m}x{tile_n}x{tile_k} "
        f"cluster={info['cluster']} "
        f"mainloop={info['mainloop']} "
        f"epilogue={info['epilogue']}"
    )
    print(f"M={M} N={N} K={K}")
    print(f"tile_rows={tile_rows} tile_cols={tile_cols} num_tiles={num_tiles}")
    print(f"layout={args.layout} reorder={reorder}")
    print(f"reldn={reldn} output_shape={tuple(C_ours.shape)}")
    print(f"group_tiles={args.group_tiles} num_segments={num_segments}")
    print(f"reset_mm_each_iter={args.reset_mm}")
    print("")
    print(f"max_abs_err:            {max_err}")
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
        if ours_eager_ms is not None:
            print(f"our eager / graph slow: {ours_eager_ms / our_graph_ms:.4f}x")
        print(f"baseline graph / our graph speed: {baseline_graph_ms / our_graph_ms:.4f}x")

    print("========================================")

    if args.check:
        assert max_err < 0.75, f"max_abs_err too large: {max_err}"


if __name__ == "__main__":
    main()
