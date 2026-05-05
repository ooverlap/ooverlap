#!/usr/bin/env python3
"""
Custom SM90 GEMM-with-signal profiler for ooverlap.

This intentionally does NOT use CUTLASS profiler CSV files. It profiles the
algorithms that are actually compiled into ooverlap_ext, using the same Python
call path that the rest of ooverlap uses.

Important layout convention:
  A    is [M, K]
  B_nk is [N, K]
  C    is [M, N]

So the GEMM is:
  C = A @ B_nk.T

This matches FlashOverlap and BaselineImpl.

Recommended first run:

  python tool/profile_gemm_signal_sm90_custom.py \
    --m 16384 --n 4096 --k 2048 \
    --layout normal \
    --rounds 5 --warmup 30 --iters 300 \
    --timing-mode eager \
    --rank-metric median \
    --check top1 \
    --include-baseline

For packed reorder+signal:

  python tool/profile_gemm_signal_sm90_custom.py \
    --m 4096 --n 4096 --k 8192 \
    --layout packed --reorder column_major --reldn 1 \
    --group-tiles 128 \
    --rounds 5 --warmup 20 --iters 200 \
    --timing-mode eager \
    --mm-mode each \
    --check top1
"""

import argparse
import csv
import importlib.util
import json
import math
import random
import statistics
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch


# -------------------------------
# Repo / extension loading
# -------------------------------

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


def make_baseline_impl(ext: Any):
    """
    Create BaselineImpl from either pybind11 or TORCH_LIBRARY registration.

    Preferred:
      ext.BaselineImpl()

    Fallback:
      torch.classes.ooverlap_class.BaselineImpl()

    The fallback requires OOVERLAP_ENABLE_TORCH_LIBRARY=1 at build time.
    """
    if hasattr(ext, "BaselineImpl"):
        baseline = ext.BaselineImpl()
        baseline.cublas_init()
        return baseline

    try:
        baseline = torch.classes.ooverlap_class.BaselineImpl()
        baseline.cublas_init()
        return baseline
    except Exception as e:
        raise RuntimeError(
            "Could not construct BaselineImpl. Expose it in pybind.cpp with:\n"
            "  py::class_<BaselineImpl>(m, \"BaselineImpl\")\n"
            "    .def(py::init<>())\n"
            "    .def(\"cublas_init\", &BaselineImpl::CublasInit)\n"
            "    .def(\"gemm\", &BaselineImpl::Gemm);\n"
            "or build with OOVERLAP_ENABLE_TORCH_LIBRARY=1."
        ) from e


def load_algo_dict(path: Optional[str] = None) -> Tuple[Dict[int, Dict[str, Any]], Path]:
    root = repo_root()
    algo_path = Path(path) if path is not None else root / "configs" / "AlgoDictSm90.json"

    if not algo_path.exists():
        raise FileNotFoundError(f"Could not find algo dict: {algo_path}")

    with open(algo_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    out: Dict[int, Dict[str, Any]] = {}
    for item in data.get("algorithms", []):
        algo = int(item["algo"])
        stages_raw = item.get("stages", "auto")
        stages: Any
        if isinstance(stages_raw, str):
            stages = stages_raw
        else:
            stages = int(stages_raw)

        out[algo] = {
            "algo": algo,
            "tile_m": int(item["tile_m"]),
            "tile_n": int(item["tile_n"]),
            "tile_k": int(item["tile_k"]),
            "cluster": [int(x) for x in item.get("cluster", [1, 1, 1])],
            "stages": stages,
            "mainloop": str(item.get("mainloop", "unknown")),
            "epilogue": str(item.get("epilogue", "auto")),
        }

    if not out:
        raise RuntimeError(f"No algorithms found in {algo_path}")

    return out, algo_path


# -------------------------------
# Layout helpers
# -------------------------------

def make_identity_ra(M: int, N: int, tile_m: int, tile_n: int, device: str) -> torch.Tensor:
    tile_rows = M // tile_m
    tile_cols = N // tile_n
    return torch.arange(tile_rows * tile_cols, device=device, dtype=torch.int32)


def make_column_major_ra(M: int, N: int, tile_m: int, tile_n: int, device: str) -> torch.Tensor:
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


def make_segments(num_tiles: int, group_tiles: int, device: str) -> torch.Tensor:
    if group_tiles <= 0:
        return torch.tensor([num_tiles], device=device, dtype=torch.int32)

    segs: List[int] = []
    left = num_tiles
    while left > 0:
        x = min(group_tiles, left)
        segs.append(x)
        left -= x

    return torch.tensor(segs, device=device, dtype=torch.int32)


def unpack_packed_to_normal(
    D_packed: torch.Tensor,
    RA: torch.Tensor,
    M: int,
    N: int,
    tile_m: int,
    tile_n: int,
    reldn: int,
) -> torch.Tensor:
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


def parse_int_list(s: Optional[str]) -> Optional[List[int]]:
    if s is None or s == "":
        return None
    out = []
    for x in s.split(","):
        x = x.strip()
        if not x:
            continue
        out.append(int(x))
    return out


def parse_str_list(s: Optional[str]) -> Optional[List[str]]:
    if s is None or s == "":
        return None
    return [x.strip() for x in s.split(",") if x.strip()]


# -------------------------------
# Timing helpers
# -------------------------------

def cuda_sync() -> None:
    torch.cuda.synchronize()


def event_time_loop(fn: Callable[[], None], warmup: int, iters: int) -> float:
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


def event_time_graph(
    capture_fn: Callable[[], None],
    warmup: int,
    iters: int,
    graph_repeats: int,
) -> float:
    if graph_repeats <= 0:
        raise ValueError("graph_repeats must be > 0")

    cuda_sync()
    graph = torch.cuda.CUDAGraph()

    with torch.cuda.graph(graph):
        for _ in range(graph_repeats):
            capture_fn()

    for _ in range(warmup):
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


def percentile(xs: Sequence[float], q: float) -> float:
    if not xs:
        return float("nan")
    if len(xs) == 1:
        return float(xs[0])

    vals = sorted(float(x) for x in xs)
    pos = (len(vals) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return vals[lo]
    frac = pos - lo
    return vals[lo] * (1.0 - frac) + vals[hi] * frac


def summarize_samples(samples: Sequence[float]) -> Dict[str, Any]:
    vals = [float(x) for x in samples]
    if not vals:
        return {
            "samples": [],
            "num_samples": 0,
            "min_ms": None,
            "p20_ms": None,
            "median_ms": None,
            "mean_ms": None,
            "p80_ms": None,
            "max_ms": None,
            "std_ms": None,
        }

    return {
        "samples": vals,
        "num_samples": len(vals),
        "min_ms": min(vals),
        "p20_ms": percentile(vals, 0.20),
        "median_ms": statistics.median(vals),
        "mean_ms": statistics.mean(vals),
        "p80_ms": percentile(vals, 0.80),
        "max_ms": max(vals),
        "std_ms": statistics.pstdev(vals) if len(vals) > 1 else 0.0,
    }


def metric_value(summary: Dict[str, Any], metric: str) -> float:
    key = {
        "min": "min_ms",
        "p20": "p20_ms",
        "median": "median_ms",
        "mean": "mean_ms",
        "p80": "p80_ms",
        "max": "max_ms",
    }[metric]
    val = summary[key]
    if val is None:
        return float("inf")
    return float(val)


def fmt_ms(x: Optional[float]) -> str:
    if x is None:
        return "NA"
    return f"{x:.6f}"


# -------------------------------
# Candidate preparation
# -------------------------------

def candidate_key(meta: Dict[str, Any]) -> Tuple[Any, ...]:
    return (
        int(meta["tile_m"]),
        int(meta["tile_n"]),
        int(meta["tile_k"]),
        int(meta["cluster"][0]),
        int(meta["cluster"][1]),
        int(meta["cluster"][2]),
        meta["stages"],
        str(meta["mainloop"]),
        str(meta["epilogue"]),
    )


def candidate_name(meta: Dict[str, Any]) -> str:
    return (
        f"algo={meta['algo']} "
        f"tile={meta['tile_m']}x{meta['tile_n']}x{meta['tile_k']} "
        f"cluster={meta['cluster']} "
        f"stages={meta['stages']} "
        f"mainloop={meta['mainloop']} "
        f"epilogue={meta['epilogue']}"
    )


def filter_candidates(
    algo_dict: Dict[int, Dict[str, Any]],
    M: int,
    N: int,
    K: int,
    include_algos: Optional[List[int]],
    exclude_algos: Optional[List[int]],
    include_mainloops: Optional[List[str]],
    include_tiles: Optional[List[str]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    include_algo_set = set(include_algos) if include_algos is not None else None
    exclude_algo_set = set(exclude_algos) if exclude_algos is not None else set()
    include_mainloop_set = set(include_mainloops) if include_mainloops is not None else None
    include_tile_set = set(include_tiles) if include_tiles is not None else None

    candidates: List[Dict[str, Any]] = []
    skipped: List[Dict[str, Any]] = []

    for algo in sorted(algo_dict):
        meta = dict(algo_dict[algo])
        meta["algo"] = int(algo)

        reason = None

        if include_algo_set is not None and algo not in include_algo_set:
            reason = "not_in_include_algos"
        elif algo in exclude_algo_set:
            reason = "in_exclude_algos"
        elif include_mainloop_set is not None and str(meta["mainloop"]) not in include_mainloop_set:
            reason = "mainloop_filtered"
        elif include_tile_set is not None:
            tile_s = f"{meta['tile_m']}x{meta['tile_n']}x{meta['tile_k']}"
            if tile_s not in include_tile_set:
                reason = "tile_filtered"
        elif M % int(meta["tile_m"]) != 0:
            reason = "M_not_multiple_of_tile_m"
        elif N % int(meta["tile_n"]) != 0:
            reason = "N_not_multiple_of_tile_n"
        elif K % int(meta["tile_k"]) != 0:
            reason = "K_not_multiple_of_tile_k"

        if reason is None:
            candidates.append(meta)
        else:
            s = dict(meta)
            s["skip_reason"] = reason
            skipped.append(s)

    return candidates, skipped


# -------------------------------
# Benchmark object
# -------------------------------

class GemmSignalBench:
    def __init__(
        self,
        ext: Any,
        M: int,
        N: int,
        K: int,
        layout: str,
        reorder: str,
        reldn_arg: Optional[int],
        group_tiles: int,
        device: str,
        seed: int,
        fill_output: bool,
        monitor: bool,
    ) -> None:
        self.ext = ext
        self.baseline = make_baseline_impl(ext)

        self.M = M
        self.N = N
        self.K = K
        self.layout = layout
        self.reorder = reorder
        self.reldn_arg = reldn_arg
        self.group_tiles = group_tiles
        self.device = device
        self.seed = seed
        self.fill_output = fill_output
        self.monitor = monitor

        torch.manual_seed(seed)

        self.A = torch.randn((M, K), device=device, dtype=torch.float16)

        # FlashOverlap / ooverlap convention:
        # B is [N, K], and GEMM computes A @ B.T.
        self.B_nk = torch.randn((N, K), device=device, dtype=torch.float16)

        self.C_baseline = torch.empty((M, N), device=device, dtype=torch.float16)

        self.C_ours: Optional[torch.Tensor] = None
        self.MM: Optional[torch.Tensor] = None
        self.RA: Optional[torch.Tensor] = None
        self.CommThr: Optional[torch.Tensor] = None
        self.reldn: int = 0
        self.num_tiles: int = 0
        self.num_segments: int = 0
        self.tile_rows: int = 0
        self.tile_cols: int = 0
        self.output_shape: Tuple[int, int] = (0, 0)

    def prepare_candidate(self, meta: Dict[str, Any]) -> None:
        tile_m = int(meta["tile_m"])
        tile_n = int(meta["tile_n"])

        self.tile_rows = self.M // tile_m
        self.tile_cols = self.N // tile_n
        self.num_tiles = self.tile_rows * self.tile_cols

        if self.reldn_arg is None:
            if self.layout == "normal":
                self.reldn = self.tile_cols
            else:
                self.reldn = 1
        else:
            self.reldn = int(self.reldn_arg)

        if self.reldn <= 0:
            raise ValueError("reldn must be > 0")

        if self.layout == "normal" and self.reldn != self.tile_cols:
            raise ValueError(
                f"normal layout requires reldn=tile_cols={self.tile_cols}, got {self.reldn}"
            )

        if self.reorder == "identity":
            self.RA = make_identity_ra(self.M, self.N, tile_m, tile_n, self.device)
        elif self.reorder == "column_major":
            self.RA = make_column_major_ra(self.M, self.N, tile_m, tile_n, self.device)
        else:
            raise ValueError(f"unknown reorder={self.reorder}")

        if self.layout == "normal":
            self.output_shape = (self.M, self.N)
        else:
            packed_tile_rows = math.ceil(self.num_tiles / self.reldn)
            self.output_shape = (packed_tile_rows * tile_m, self.reldn * tile_n)

        self.C_ours = torch.empty(self.output_shape, device=self.device, dtype=torch.float16)
        self.CommThr = make_segments(self.num_tiles, self.group_tiles, self.device)
        self.num_segments = int(self.CommThr.numel())
        self.MM = torch.empty((self.num_segments + self.num_tiles,), device=self.device, dtype=torch.int32)

        self.C_ours.zero_()
        self.MM.zero_()
        cuda_sync()

    def baseline_fn(self) -> None:
        self.baseline.gemm(self.A, self.B_nk, self.C_baseline)

    def ours_fn(self, algo: int, mm_mode: str) -> None:
        assert self.C_ours is not None
        assert self.MM is not None
        assert self.RA is not None
        assert self.CommThr is not None

        if mm_mode == "each":
            self.MM.zero_()

        if self.fill_output:
            self.C_ours.zero_()

        self.ext.gemm_signal_sm90(
            self.A,
            self.B_nk,
            self.C_ours,
            self.MM,
            self.RA,
            self.CommThr,
            int(self.reldn),
            int(algo),
            bool(self.monitor),
        )

    def reset_mm_once(self) -> None:
        assert self.MM is not None
        self.MM.zero_()
        cuda_sync()

    def check_correctness(self, meta: Dict[str, Any], atol: float) -> float:
        assert self.C_ours is not None
        assert self.RA is not None

        algo = int(meta["algo"])
        tile_m = int(meta["tile_m"])
        tile_n = int(meta["tile_n"])

        self.reset_mm_once()
        self.baseline_fn()
        self.ours_fn(algo, mm_mode="never")
        cuda_sync()

        if self.layout == "normal":
            C_ours_normal = self.C_ours
        else:
            C_ours_normal = unpack_packed_to_normal(
                self.C_ours,
                self.RA,
                self.M,
                self.N,
                tile_m,
                tile_n,
                self.reldn,
            )

        err = float((C_ours_normal - self.C_baseline).abs().max().item())
        if err > atol:
            raise RuntimeError(f"max_abs_err too large: {err} > {atol}")
        return err


def benchmark_one(
    bench: GemmSignalBench,
    meta: Dict[str, Any],
    timing_mode: str,
    warmup: int,
    iters: int,
    graph_repeats: int,
    mm_mode: str,
) -> float:
    algo = int(meta["algo"])

    if mm_mode == "once":
        bench.reset_mm_once()

    def fn() -> None:
        bench.ours_fn(algo, mm_mode=mm_mode)

    if timing_mode == "eager":
        return event_time_loop(fn, warmup=warmup, iters=iters)

    if timing_mode == "graph":
        return event_time_graph(fn, warmup=warmup, iters=iters, graph_repeats=graph_repeats)

    raise ValueError(f"unknown timing_mode={timing_mode}")


def benchmark_baseline(
    bench: GemmSignalBench,
    timing_mode: str,
    warmup: int,
    iters: int,
    graph_repeats: int,
) -> float:
    if timing_mode == "eager":
        return event_time_loop(bench.baseline_fn, warmup=warmup, iters=iters)

    if timing_mode == "graph":
        return event_time_graph(bench.baseline_fn, warmup=warmup, iters=iters, graph_repeats=graph_repeats)

    raise ValueError(f"unknown timing_mode={timing_mode}")


# -------------------------------
# Output
# -------------------------------

def make_output_paths(
    root: Path,
    M: int,
    N: int,
    K: int,
    gpu_name: str,
    layout: str,
    suffix: str,
) -> Tuple[Path, Path]:
    safe_gpu = gpu_name.lower().replace(" ", "_").replace("/", "_")
    stem = f"m{M}n{N}k{K}_{safe_gpu}_{layout}_custom_sm90"
    if suffix:
        stem += f"_{suffix}"
    return root / "configs" / f"{stem}.json", root / "configs" / f"{stem}.csv"


def write_csv(path: Path, rows: List[Dict[str, Any]], rank_metric: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "rank",
        "algo",
        "rank_metric",
        "rank_metric_ms",
        "min_ms",
        "p20_ms",
        "median_ms",
        "mean_ms",
        "p80_ms",
        "max_ms",
        "std_ms",
        "num_samples",
        "tile_m",
        "tile_n",
        "tile_k",
        "cluster",
        "stages",
        "mainloop",
        "epilogue",
        "samples",
    ]

    with open(path, "w", encoding="utf-8", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=columns)
        wr.writeheader()
        for r in rows:
            wr.writerow({k: r.get(k, "") for k in columns})


def print_top(rows: List[Dict[str, Any]], top_save: int, rank_metric: str, flops: float) -> None:
    print("")
    print("Top selected configs:")
    for idx, r in enumerate(rows[:top_save], start=1):
        ms = float(r["rank_metric_ms"])
        tflops = flops / (ms * 1e-3) / 1e12
        print(
            f"  #{idx}: algo={r['algo']} "
            f"tile={r['tile_m']}x{r['tile_n']}x{r['tile_k']} "
            f"cluster={r['cluster']} "
            f"stages={r['stages']} "
            f"mainloop={r['mainloop']} "
            f"{rank_metric}_ms={ms:.6f} "
            f"min={fmt_ms(r['min_ms'])} "
            f"median={fmt_ms(r['median_ms'])} "
            f"max={fmt_ms(r['max_ms'])} "
            f"TFLOP/s={tflops:.2f}"
        )


# -------------------------------
# Main
# -------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--m", type=int, required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--k", type=int, required=True)

    ap.add_argument("--algo-dict", type=str, default=None)
    ap.add_argument("--algos", type=str, default=None, help="Comma-separated algo ids to include.")
    ap.add_argument("--exclude-algos", type=str, default=None)
    ap.add_argument("--mainloops", type=str, default=None, help="Comma-separated mainloops, e.g. pingpong,cooperative,ws.")
    ap.add_argument("--tiles", type=str, default=None, help="Comma-separated tile shapes, e.g. 128x128x64,64x256x64.")

    ap.add_argument("--layout", choices=["normal", "packed"], default="normal")
    ap.add_argument("--reorder", choices=["identity", "column_major"], default=None)
    ap.add_argument("--reldn", type=int, default=None)
    ap.add_argument("--group-tiles", type=int, default=0)
    ap.add_argument("--monitor", action="store_true")

    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=30)
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--timing-mode", choices=["eager", "graph"], default="eager")
    ap.add_argument("--graph-repeats", type=int, default=1)
    ap.add_argument("--rank-metric", choices=["min", "p20", "median", "mean", "p80", "max"], default="median")
    ap.add_argument("--top-save", type=int, default=10)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--randomize", action=argparse.BooleanOptionalAction, default=True)

    ap.add_argument(
        "--mm-mode",
        choices=["never", "once", "each"],
        default="once",
        help=(
            "never: never zero MM during timing; once: zero before each timed sample; "
            "each: zero before every GEMM call and include reset cost."
        ),
    )
    ap.add_argument(
        "--fill-output",
        action="store_true",
        help="Zero output before each GEMM call. Debug only; not for normal tuning.",
    )

    ap.add_argument(
        "--check",
        choices=["none", "top1", "all"],
        default="top1",
        help="Correctness check policy.",
    )
    ap.add_argument("--check-atol", type=float, default=0.75)

    ap.add_argument("--include-baseline", action="store_true")
    ap.add_argument(
        "--include-torch",
        action="store_true",
        help="Deprecated alias for --include-baseline. This script now uses BaselineImpl, not torch.matmul.",
    )

    ap.add_argument("--suffix", type=str, default="")
    ap.add_argument("--dry-run", action="store_true")

    args = ap.parse_args()

    if args.include_torch:
        args.include_baseline = True

    if args.rounds <= 0:
        raise ValueError("--rounds must be > 0")
    if args.iters <= 0:
        raise ValueError("--iters must be > 0")
    if args.warmup < 0:
        raise ValueError("--warmup must be >= 0")

    torch.cuda.set_device(args.device)
    device = "cuda"

    root = repo_root()
    algo_dict, algo_dict_path = load_algo_dict(args.algo_dict)

    include_algos = parse_int_list(args.algos)
    exclude_algos = parse_int_list(args.exclude_algos)
    include_mainloops = parse_str_list(args.mainloops)
    include_tiles = parse_str_list(args.tiles)

    candidates, skipped = filter_candidates(
        algo_dict=algo_dict,
        M=args.m,
        N=args.n,
        K=args.k,
        include_algos=include_algos,
        exclude_algos=exclude_algos,
        include_mainloops=include_mainloops,
        include_tiles=include_tiles,
    )

    if args.reorder is None:
        reorder = "identity" if args.layout == "normal" else "column_major"
    else:
        reorder = args.reorder

    gpu_name = torch.cuda.get_device_name(args.device)
    flops = 2.0 * args.m * args.n * args.k

    print("========================================")
    print("profile_gemm_signal_sm90_custom")
    print(f"shape:          M={args.m} N={args.n} K={args.k}")
    print(f"gpu:            {gpu_name}")
    print(f"algo_dict:      {algo_dict_path}")
    print(f"candidates:     {len(candidates)}")
    print(f"skipped:        {len(skipped)}")
    print(f"layout:         {args.layout}")
    print(f"reorder:        {reorder}")
    print(f"reldn:          {'auto' if args.reldn is None else args.reldn}")
    print(f"group_tiles:    {args.group_tiles}")
    print(f"timing_mode:    {args.timing_mode}")
    print(f"graph_repeats:  {args.graph_repeats}")
    print(f"rounds:         {args.rounds}")
    print(f"warmup:         {args.warmup}")
    print(f"iters:          {args.iters}")
    print(f"mm_mode:        {args.mm_mode}")
    print(f"rank_metric:    {args.rank_metric}")
    print(f"baseline:       {'enabled' if args.include_baseline else 'disabled'}")
    print("========================================")

    if not candidates:
        raise RuntimeError("No candidates left after filters and divisibility checks.")

    print("")
    print("Candidate list:")
    for c in candidates:
        print("  " + candidate_name(c))

    if args.dry_run:
        return

    ext = load_ooverlap_ext(root)

    bench = GemmSignalBench(
        ext=ext,
        M=args.m,
        N=args.n,
        K=args.k,
        layout=args.layout,
        reorder=reorder,
        reldn_arg=args.reldn,
        group_tiles=args.group_tiles,
        device=device,
        seed=args.seed,
        fill_output=args.fill_output,
        monitor=args.monitor,
    )

    baseline_ms: Optional[float] = None
    if args.include_baseline:
        print("")
        print("Profiling BaselineImpl GEMM...")
        baseline_ms = benchmark_baseline(
            bench=bench,
            timing_mode=args.timing_mode,
            warmup=args.warmup,
            iters=args.iters,
            graph_repeats=args.graph_repeats,
        )
        baseline_tflops = flops / (baseline_ms * 1e-3) / 1e12
        print(f"  baseline_{args.timing_mode}_ms={baseline_ms:.6f} TFLOP/s={baseline_tflops:.2f}")

    sample_map: Dict[int, List[float]] = {int(c["algo"]): [] for c in candidates}
    fail_map: Dict[int, str] = {}
    err_map: Dict[int, Optional[float]] = {int(c["algo"]): None for c in candidates}

    rng = random.Random(args.seed)

    print("")
    print("Profiling candidates...")
    for round_idx in range(args.rounds):
        order = list(candidates)
        if args.randomize:
            rng.shuffle(order)

        print(f"round {round_idx + 1}/{args.rounds}")

        # Guard warmup before each round. Keep it outside per-candidate timing.
        try:
            event_time_loop(bench.baseline_fn, warmup=3, iters=3)
        except Exception:
            pass

        for meta in order:
            algo = int(meta["algo"])
            if algo in fail_map:
                continue

            try:
                bench.prepare_candidate(meta)

                if args.check == "all" and err_map[algo] is None:
                    err_map[algo] = bench.check_correctness(meta, args.check_atol)

                ms = benchmark_one(
                    bench=bench,
                    meta=meta,
                    timing_mode=args.timing_mode,
                    warmup=args.warmup,
                    iters=args.iters,
                    graph_repeats=args.graph_repeats,
                    mm_mode=args.mm_mode,
                )

                sample_map[algo].append(ms)
                print(
                    f"  algo={algo:4d} "
                    f"sample={ms:.6f} ms "
                    f"n={len(sample_map[algo])}"
                )

            except Exception as e:
                fail_map[algo] = repr(e)
                print(f"  algo={algo:4d} FAILED: {repr(e)}")

    rows: List[Dict[str, Any]] = []
    for meta in candidates:
        algo = int(meta["algo"])
        if algo in fail_map:
            continue

        summary = summarize_samples(sample_map[algo])
        if summary["num_samples"] == 0:
            continue

        rank_ms = metric_value(summary, args.rank_metric)

        row: Dict[str, Any] = {
            "algo": algo,
            "rank_metric": args.rank_metric,
            "rank_metric_ms": rank_ms,
            "tile_m": int(meta["tile_m"]),
            "tile_n": int(meta["tile_n"]),
            "tile_k": int(meta["tile_k"]),
            "cluster": [int(x) for x in meta["cluster"]],
            "stages": meta["stages"],
            "mainloop": str(meta["mainloop"]),
            "epilogue": str(meta["epilogue"]),
            "key": list(candidate_key(meta)),
            "max_abs_err": err_map.get(algo),
        }
        row.update(summary)
        rows.append(row)

    rows.sort(key=lambda r: (float(r["rank_metric_ms"]), int(r["algo"])))

    if args.check == "top1" and rows:
        best_algo = int(rows[0]["algo"])
        best_meta = next(c for c in candidates if int(c["algo"]) == best_algo)
        bench.prepare_candidate(best_meta)
        err = bench.check_correctness(best_meta, args.check_atol)
        rows[0]["max_abs_err"] = err
        print("")
        print(f"Correctness top1: algo={best_algo} max_abs_err={err}")

    for idx, row in enumerate(rows, start=1):
        row["rank"] = idx

    out_json, out_csv = make_output_paths(
        root=root,
        M=args.m,
        N=args.n,
        K=args.k,
        gpu_name=gpu_name,
        layout=args.layout,
        suffix=args.suffix,
    )

    top_rows = rows[: args.top_save]

    config = {
        "description": "ooverlap custom SM90 GEMM profiler result; no CUTLASS CSV used",
        "M": args.m,
        "N": args.n,
        "K": args.k,
        "gpu": gpu_name,
        "algo_dict": str(algo_dict_path),
        "layout": args.layout,
        "reorder": reorder,
        "reldn": args.reldn if args.reldn is not None else "auto",
        "group_tiles": args.group_tiles,
        "timing_mode": args.timing_mode,
        "graph_repeats": args.graph_repeats,
        "rounds": args.rounds,
        "warmup": args.warmup,
        "iters": args.iters,
        "mm_mode": args.mm_mode,
        "rank_metric": args.rank_metric,
        "baseline_ms": baseline_ms,
        "BM": [int(r["tile_m"]) for r in top_rows],
        "BN": [int(r["tile_n"]) for r in top_rows],
        "BK": [int(r["tile_k"]) for r in top_rows],
        "Algo": [int(r["algo"]) for r in top_rows],
        "dur": [float(r["rank_metric_ms"]) for r in top_rows],
        "top": top_rows,
        "all_profiled": rows,
        "failed": [
            {
                "algo": int(algo),
                "error": err,
                **{k: v for k, v in algo_dict.get(int(algo), {}).items() if k != "algo"},
            }
            for algo, err in sorted(fail_map.items())
        ],
        "skipped": skipped,
    }

    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    write_csv(out_csv, rows, args.rank_metric)

    print("")
    print("========================================")
    print("DONE")
    print(f"profiled ok:    {len(rows)}")
    print(f"failed:         {len(fail_map)}")
    print(f"skipped:        {len(skipped)}")
    print(f"wrote json:     {out_json}")
    print(f"wrote csv:      {out_csv}")

    if rows:
        print_top(rows, args.top_save, args.rank_metric, flops)
    else:
        print("No successful configs.")

    print("========================================")


if __name__ == "__main__":
    main()
