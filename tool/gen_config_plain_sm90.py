#!/usr/bin/env python3
"""
Generate/profile plain SM90 GEMM configs against a CUTLASS profiler CSV.

This script is intentionally separate from gen_config_sm90.py. It does not use
RA/MM/CommThr/ReLDN and does not call the reorder/signal epilogue path. It only
benchmarks the new plain GEMM entry point:

    ext.gemm_plain_sm90(A, B_col, D_col, algo)

Expected tensor convention:

    A      physical [M, K], row-major
    B_col  physical [N, K], interpreted as logical column-major B [K, N]
    D_col  physical [N, M], interpreted as logical column-major D [M, N]

Default CSV filter is apples-to-apples with this convention:

    A=f16:row, B=f16:column, C=f16:column, D=f16:column, accum=f32

Example:

  python tool/gen_config_plain_sm90.py \
    --m 16384 --n 8192 --k 8192 \
    --csv ~/csv_out_h100/m16384n8192k8192.gemm.csv \
    --top-csv 50 --top-save 10 \
    --warmup 20 --iters 100 \
    --include-stream-k
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import torch


# ----------------------------- default plain algo map -----------------------------


# Adjust these IDs if your plain dispatch table uses different numbering.
DEFAULT_PLAIN_ALGOS: Dict[int, Dict[str, Any]] = {
    0:  dict(tile_m=128, tile_n=256, tile_k=64, stages=4, cluster=[2, 1, 1], mainloop="cooperative", epilogue="auto", scheduler="normal"),
    1:  dict(tile_m=256, tile_n=128, tile_k=64, stages=4, cluster=[2, 1, 1], mainloop="cooperative", epilogue="auto", scheduler="normal"),
    2:  dict(tile_m=128, tile_n=256, tile_k=64, stages=4, cluster=[1, 2, 1], mainloop="cooperative", epilogue="auto", scheduler="normal"),
    3:  dict(tile_m=256, tile_n=128, tile_k=64, stages=4, cluster=[1, 2, 1], mainloop="cooperative", epilogue="auto", scheduler="normal"),
    10: dict(tile_m=128, tile_n=256, tile_k=64, stages=4, cluster=[2, 1, 1], mainloop="cooperative", epilogue="auto", scheduler="stream_k"),
    11: dict(tile_m=256, tile_n=128, tile_k=64, stages=4, cluster=[2, 1, 1], mainloop="cooperative", epilogue="auto", scheduler="stream_k"),
    12: dict(tile_m=128, tile_n=256, tile_k=64, stages=4, cluster=[1, 2, 1], mainloop="cooperative", epilogue="auto", scheduler="stream_k"),
    13: dict(tile_m=256, tile_n=128, tile_k=64, stages=4, cluster=[1, 2, 1], mainloop="cooperative", epilogue="auto", scheduler="stream_k"),
}


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


def get_plain_func(ext):
    for name in ("gemm_plain_sm90", "plain_gemm_sm90", "gemm_sm90_plain"):
        if hasattr(ext, name):
            return getattr(ext, name), name
    raise AttributeError(
        "Could not find plain GEMM entry point on ooverlap_ext. "
        "Expected ext.gemm_plain_sm90(A, B_col, D_col, algo)."
    )


# ----------------------------- parsing helpers -----------------------------


RUNTIME_NAMES = ["runtime", "Runtime", "runtime_ms", "time", "duration"]
OP_NAMES = ["operation", "Operation", "kernel", "Kernel", "name", "procedural_name"]
A_NAMES = ["a", "A"]
B_NAMES = ["b", "B"]
C_NAMES = ["c", "C"]
D_NAMES = ["d", "D"]
ACCUM_NAMES = ["accum", "Accum", "accumulator", "Accumulator"]
SPLIT_K_NAMES = ["split_k_slices", "split_k", "SplitK"]
CTA_M_NAMES = ["cta_m", "threadblock_m", "tile_m"]
CTA_N_NAMES = ["cta_n", "threadblock_n", "tile_n"]
CTA_K_NAMES = ["cta_k", "threadblock_k", "tile_k"]
CLUSTER_M_NAMES = ["cluster_m", "cluster_shape_m"]
CLUSTER_N_NAMES = ["cluster_n", "cluster_shape_n"]
CLUSTER_K_NAMES = ["cluster_k", "cluster_shape_k"]
STAGES_NAMES = ["stages", "Stages"]


def norm_col(x: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(x).strip().lower()).strip("_")


def find_col(fieldnames: Sequence[str], names: Sequence[str]) -> Optional[str]:
    cmap = {norm_col(c): c for c in fieldnames}
    for n in names:
        k = norm_col(n)
        if k in cmap:
            return cmap[k]
    return None


def as_int(x: Any, default: Optional[int] = None) -> Optional[int]:
    try:
        if x is None:
            return default
        s = str(x).strip()
        if not s:
            return default
        return int(float(s))
    except Exception:
        return default


def as_float(x: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        if x is None:
            return default
        s = str(x).strip()
        if not s:
            return default
        v = float(s)
        if math.isnan(v):
            return default
        return v
    except Exception:
        return default


def norm_layout(x: Any) -> Optional[str]:
    if x is None:
        return None
    s = str(x).strip().lower()
    if ":" in s:
        s = s.split(":")[-1]
    if s in ("row", "row_major", "rowmajor", "n"):
        return "row"
    if s in ("column", "col", "column_major", "columnmajor", "t"):
        return "column"
    if "row" in s:
        return "row"
    if "col" in s:
        return "column"
    return None


def norm_dtype(x: Any) -> Optional[str]:
    if x is None:
        return None
    s = str(x).strip().lower()
    if ":" in s:
        s = s.split(":")[0]
    if s in ("half", "fp16", "float16"):
        return "f16"
    if s in ("float", "fp32", "float32"):
        return "f32"
    if s in ("void", "none", "null"):
        return "void"
    return s or None


def parse_op_dtypes(op: str) -> Dict[str, Optional[str]]:
    s = str(op)
    m = re.search(r"gemm_([^_]+)_([^_]+)_([^_]+)_([^_]+)_([^_]+)_", s)
    if not m:
        return {"a": None, "b": None, "accum": None, "c": None, "d": None}
    return {
        "a": norm_dtype(m.group(1)),
        "b": norm_dtype(m.group(2)),
        "accum": norm_dtype(m.group(3)),
        "c": norm_dtype(m.group(4)),
        "d": norm_dtype(m.group(5)),
    }


def map_mainloop(op: str) -> str:
    s = str(op).lower().replace("-", "_")
    if "pingpong" in s or "ping_pong" in s:
        return "pingpong"
    if "cooperative" in s or "coop" in s:
        return "cooperative"
    if "warpspecialized" in s or "warp_specialized" in s or "tma" in s:
        return "ws"
    return "ws"


def map_scheduler(op: str) -> str:
    s = str(op).lower().replace("-", "_")
    if "stream_k" in s or "streamk" in s:
        return "stream_k"
    return "normal"


def filename_shape(path: Path) -> Optional[Tuple[int, int, int]]:
    m = re.search(r"m(\d+)n(\d+)k(\d+)", path.name, re.IGNORECASE)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2)), int(m.group(3))


def gpu_name_slug() -> Tuple[str, str]:
    name = torch.cuda.get_device_properties(torch.cuda.current_device()).name
    slug = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    return slug, name


def make_key(meta: Dict[str, Any]) -> Tuple[Any, ...]:
    c = meta["cluster"]
    return (
        int(meta["tile_m"]),
        int(meta["tile_n"]),
        int(meta["tile_k"]),
        int(c[0]),
        int(c[1]),
        int(c[2]),
        int(meta["stages"]),
        str(meta["mainloop"]),
        str(meta.get("epilogue", "auto")),
        str(meta["scheduler"]),
    )

def normalize_stage_for_key(x: Any) -> int:
    if x is None:
        return -1
    s = str(x).strip().lower()
    if s in ("auto", "-1"):
        return -1
    return int(float(s))


def load_plain_algo_map(path: Optional[str]) -> Dict[int, Dict[str, Any]]:
    # Default to the generated shared AlgoDict. Fall back to the old hardcoded
    # tiny map only if the generated file does not exist.
    if path is None or path == "auto":
        p = repo_root() / "configs" / "AlgoDictSm90.json"
        if not p.exists():
            print("[WARN] configs/AlgoDictSm90.json not found; using built-in DEFAULT_PLAIN_ALGOS")
            return {k: dict(v, algo=k) for k, v in DEFAULT_PLAIN_ALGOS.items()}
    else:
        p = Path(path).expanduser().resolve()

    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)

    items = data.get("algorithms", data if isinstance(data, list) else [])
    out: Dict[int, Dict[str, Any]] = {}

    for item in items:
        algo = int(item["algo"])
        cluster = item.get(
            "cluster",
            [item.get("cluster_m", 1), item.get("cluster_n", 1), item.get("cluster_k", 1)],
        )

        out[algo] = {
            "algo": algo,
            "tile_m": int(item["tile_m"]),
            "tile_n": int(item["tile_n"]),
            "tile_k": int(item["tile_k"]),
            "stages": normalize_stage_for_key(item.get("stages", -1)),
            "cluster": [int(x) for x in cluster],
            "mainloop": str(item.get("mainloop", "cooperative")),
            "epilogue": str(item.get("epilogue", "auto")),
            "scheduler": str(item.get("scheduler", "normal")),
        }

    print(f"loaded plain algo map: {p} ({len(out)} algos)")
    return out


# ----------------------------- CSV parsing -----------------------------


def parse_csv_rows(args: argparse.Namespace, csv_path: Path) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []

        c_runtime = find_col(fieldnames, RUNTIME_NAMES)
        c_op = find_col(fieldnames, OP_NAMES)
        c_a = find_col(fieldnames, A_NAMES)
        c_b = find_col(fieldnames, B_NAMES)
        c_c = find_col(fieldnames, C_NAMES)
        c_d = find_col(fieldnames, D_NAMES)
        c_accum = find_col(fieldnames, ACCUM_NAMES)
        c_split = find_col(fieldnames, SPLIT_K_NAMES)
        c_tm = find_col(fieldnames, CTA_M_NAMES)
        c_tn = find_col(fieldnames, CTA_N_NAMES)
        c_tk = find_col(fieldnames, CTA_K_NAMES)
        c_cm = find_col(fieldnames, CLUSTER_M_NAMES)
        c_cn = find_col(fieldnames, CLUSTER_N_NAMES)
        c_ck = find_col(fieldnames, CLUSTER_K_NAMES)
        c_stages = find_col(fieldnames, STAGES_NAMES)

        if c_runtime is None:
            raise RuntimeError(f"No runtime column found. CSV columns: {fieldnames}")

        shape_from_name = filename_shape(csv_path)

        counts = {
            "raw_rows": 0,
            "bad_runtime": 0,
            "split_k": 0,
            "shape": 0,
            "layout_or_dtype": 0,
            "tile_parse": 0,
            "stream_k_rows_seen": 0,
            "kept": 0,
        }

        records: List[Dict[str, Any]] = []

        for idx, row in enumerate(reader):
            counts["raw_rows"] += 1

            runtime = as_float(row.get(c_runtime))
            if runtime is None:
                counts["bad_runtime"] += 1
                continue

            op = str(row.get(c_op, "")) if c_op is not None else ""
            scheduler = map_scheduler(op)
            if scheduler == "stream_k":
                counts["stream_k_rows_seen"] += 1

            split_k = as_int(row.get(c_split), 1) if c_split is not None else 1
            if split_k != 1 and not args.allow_split_k:
                counts["split_k"] += 1
                continue

            if shape_from_name is not None:
                m0, n0, k0 = shape_from_name
                if (m0, n0, k0) != (args.m, args.n, args.k):
                    counts["shape"] += 1
                    continue

            tm = as_int(row.get(c_tm)) if c_tm is not None else None
            tn = as_int(row.get(c_tn)) if c_tn is not None else None
            tk = as_int(row.get(c_tk)) if c_tk is not None else None
            stages = as_int(row.get(c_stages)) if c_stages is not None else None
            if tm is None or tn is None or tk is None or stages is None:
                counts["tile_parse"] += 1
                continue

            cm = as_int(row.get(c_cm), 1) if c_cm is not None else 1
            cn = as_int(row.get(c_cn), 1) if c_cn is not None else 1
            ck = as_int(row.get(c_ck), 1) if c_ck is not None else 1

            op_dt = parse_op_dtypes(op)

            a_dtype = norm_dtype(row.get(c_a)) if c_a is not None else op_dt["a"]
            b_dtype = norm_dtype(row.get(c_b)) if c_b is not None else op_dt["b"]
            c_dtype = norm_dtype(row.get(c_c)) if c_c is not None else op_dt["c"]
            d_dtype = norm_dtype(row.get(c_d)) if c_d is not None else op_dt["d"]
            accum = norm_dtype(row.get(c_accum)) if c_accum is not None else op_dt["accum"]

            a_layout = norm_layout(row.get(c_a)) if c_a is not None else None
            b_layout = norm_layout(row.get(c_b)) if c_b is not None else None
            c_layout = norm_layout(row.get(c_c)) if c_c is not None else None
            d_layout = norm_layout(row.get(c_d)) if c_d is not None else None

            ok = True
            if a_dtype is not None and a_dtype != args.csv_a_dtype:
                ok = False
            if b_dtype is not None and b_dtype != args.csv_b_dtype:
                ok = False
            if accum is not None and accum != args.csv_accum_dtype:
                ok = False
            if args.csv_c_dtype != "any" and (c_dtype is None or c_dtype != args.csv_c_dtype):
                ok = False
            if args.csv_d_dtype != "any" and (d_dtype is None or d_dtype != args.csv_d_dtype):
                ok = False

            if args.csv_a_layout != "any" and a_layout is not None and a_layout != args.csv_a_layout:
                ok = False
            if args.csv_b_layout != "any" and b_layout is not None and b_layout != args.csv_b_layout:
                ok = False
            if args.csv_c_layout != "any" and c_layout is not None and c_layout != args.csv_c_layout:
                ok = False
            if args.csv_d_layout != "any" and d_layout is not None and d_layout != args.csv_d_layout:
                ok = False

            if not ok:
                counts["layout_or_dtype"] += 1
                continue

            rec = {
                "source_row": idx,
                "cutlass_runtime": float(runtime),
                "tile_m": int(tm),
                "tile_n": int(tn),
                "tile_k": int(tk),
                "cluster": [int(cm), int(cn), int(ck)],
                "stages": int(stages),
                "mainloop": map_mainloop(op),
                "epilogue": "auto",
                "scheduler": scheduler,
                "a": row.get(c_a) if c_a is not None else None,
                "b": row.get(c_b) if c_b is not None else None,
                "c": row.get(c_c) if c_c is not None else op_dt["c"],
                "d": row.get(c_d) if c_d is not None else op_dt["d"],
                "a_dtype": a_dtype,
                "b_dtype": b_dtype,
                "c_dtype": c_dtype,
                "d_dtype": d_dtype,
                "a_layout": a_layout,
                "b_layout": b_layout,
                "c_layout": c_layout,
                "d_layout": d_layout,
                "accum_dtype": accum,
                "operation": op,
            }
            rec["key"] = list(make_key(rec))
            records.append(rec)
            counts["kept"] += 1

    records.sort(key=lambda r: float(r["cutlass_runtime"]))
    return records, counts


def match_records(
    records: Sequence[Dict[str, Any]],
    algo_map: Dict[int, Dict[str, Any]],
    include_stream_k: bool,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    by_key: Dict[Tuple[Any, ...], int] = {}
    for algo, meta in algo_map.items():
        mm = dict(meta)
        mm["algo"] = int(algo)
        by_key[make_key(mm)] = int(algo)

    matched_by_algo: Dict[int, Dict[str, Any]] = {}
    missing: List[Dict[str, Any]] = []
    skipped_stream_k: List[Dict[str, Any]] = []

    for r in records:
        if r["scheduler"] == "stream_k" and not include_stream_k:
            rr = dict(r)
            rr["error"] = "stream_k_not_included"
            skipped_stream_k.append(rr)
            continue

        k = tuple(r["key"])
        algo = by_key.get(k)
        if algo is None:
            rr = dict(r)
            rr["error"] = "no_plain_algo_for_key"
            missing.append(rr)
            continue

        rr = dict(r)
        rr["algo"] = int(algo)
        rr.update({f"algo_{kk}": vv for kk, vv in algo_map[algo].items() if kk != "algo"})

        if algo not in matched_by_algo or rr["cutlass_runtime"] < matched_by_algo[algo]["cutlass_runtime"]:
            matched_by_algo[algo] = rr

    matched = list(matched_by_algo.values())
    matched.sort(key=lambda r: float(r["cutlass_runtime"]))
    return matched, missing, skipped_stream_k


# ----------------------------- timing / benchmark -----------------------------


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


def time_cuda_graph_unrolled(fn: Callable[[], None], warmup: int, iters: int, graph_repeats: int) -> float:
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


def safe_quantile_1d(x: torch.Tensor, q: float, max_samples: int = 1_000_000) -> float:
    """Compute a quantile without feeding huge tensors to torch.quantile.

    torch.quantile can error on very large tensors. For profiler diagnostics we
    only need a stable summary, so use an evenly-strided sample when the tensor
    is huge. max/mean are still computed exactly in error_summary.
    """
    x = x.reshape(-1)
    n = int(x.numel())
    if n == 0:
        return float("nan")

    if n > max_samples:
        step = (n + max_samples - 1) // max_samples
        x = x[::step][:max_samples]

    return float(torch.quantile(x, q).item())


def error_summary(x: torch.Tensor, y: torch.Tensor) -> Dict[str, float]:
    diff = (x - y).abs().float().reshape(-1)
    return {
        "max_abs": float(diff.max().item()),
        "mean_abs": float(diff.mean().item()),
        "p99_abs": safe_quantile_1d(diff, 0.99),
        "p999_abs": safe_quantile_1d(diff, 0.999),
    }


def benchmark_algo(
    ext,
    plain_fn,
    args: argparse.Namespace,
    algo: int,
    A: torch.Tensor,
    B_col: torch.Tensor,
    D_col: torch.Tensor,
) -> Tuple[float, str, Optional[str]]:
    def run() -> None:
        plain_fn(A, B_col, D_col, int(algo))

    # Prime lazy init outside timing/capture.
    run()
    cuda_sync()

    if args.timing_mode == "eager":
        return time_cuda_eager(run, args.warmup, args.iters), "eager", None

    if args.timing_mode == "graph":
        return (
            time_cuda_graph_unrolled(run, args.warmup, args.iters, args.graph_repeats),
            f"graph_unrolled_{args.graph_repeats}",
            None,
        )

    try:
        return (
            time_cuda_graph_unrolled(run, args.warmup, args.iters, args.graph_repeats),
            f"graph_unrolled_{args.graph_repeats}",
            None,
        )
    except Exception as e:
        return time_cuda_eager(run, args.warmup, args.iters), "eager_fallback", repr(e)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n")
        return

    cols = [
        "rank", "algo", "plain_gemm_ms", "cutlass_runtime", "timing_used",
        "tile_m", "tile_n", "tile_k", "cluster", "stages", "mainloop", "scheduler",
        "max_abs", "mean_abs", "p99_abs", "p999_abs", "source_row", "operation",
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=cols)
        wr.writeheader()
        for r in rows:
            wr.writerow({k: r.get(k, "") for k in cols})


# ----------------------------- main -----------------------------


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", type=int, required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--k", type=int, required=True)
    ap.add_argument("--csv", type=str, required=True)
    ap.add_argument("--plain-algo-json", type=str, default="auto")
    ap.add_argument("--device", type=int, default=0)

    ap.add_argument("--top-csv", type=int, default=50)
    ap.add_argument("--top-save", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--timing-mode", choices=["auto", "graph", "eager"], default="eager")
    ap.add_argument("--graph-repeats", type=int, default=100)
    ap.add_argument("--include-stream-k", action="store_true")
    ap.add_argument("--allow-split-k", action="store_true")

    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--check", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--atol", type=float, default=8.0)
    ap.add_argument("--rtol", type=float, default=2.0e-2)
    ap.add_argument("--dry-run", action="store_true")

    # Defaults match the plain column-output kernel convention.
    ap.add_argument("--csv-a-dtype", choices=["f16"], default="f16")
    ap.add_argument("--csv-b-dtype", choices=["f16"], default="f16")
    ap.add_argument("--csv-accum-dtype", choices=["f16", "f32"], default="f16")
    ap.add_argument("--csv-c-dtype", choices=["f16", "void", "any"], default="f16")
    ap.add_argument("--csv-d-dtype", choices=["f16", "f32", "any"], default="f16")
    ap.add_argument("--csv-a-layout", choices=["row", "column", "any"], default="row")
    ap.add_argument("--csv-b-layout", choices=["row", "column", "any"], default="column")
    ap.add_argument("--csv-c-layout", choices=["row", "column", "any"], default="column")
    ap.add_argument("--csv-d-layout", choices=["row", "column", "any"], default="column")

    args = ap.parse_args()

    if args.top_csv <= 0:
        raise ValueError("--top-csv must be > 0")
    if args.top_save <= 0:
        raise ValueError("--top-save must be > 0")
    if args.iters <= 0:
        raise ValueError("--iters must be > 0")
    if args.warmup < 0:
        raise ValueError("--warmup must be >= 0")
    if args.graph_repeats <= 0:
        raise ValueError("--graph-repeats must be > 0")

    torch.cuda.set_device(args.device)
    device = torch.device("cuda", args.device)
    torch.manual_seed(args.seed)

    root = repo_root()
    csv_path = Path(args.csv).expanduser().resolve()
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    algo_map = load_plain_algo_map(args.plain_algo_json)
    records, counts = parse_csv_rows(args, csv_path)
    considered = records[: args.top_csv]
    matched, missing, skipped_stream_k = match_records(considered, algo_map, args.include_stream_k)

    slug, gpu_name = gpu_name_slug()
    base = root / "configs" / f"m{args.m}n{args.n}k{args.k}_{slug}_plain_sm90"
    out_json = base.with_suffix(".json")
    out_csv = base.with_suffix(".csv")
    out_missing = Path(str(base) + "_missing.json")
    out_failed = Path(str(base) + "_failed.json")

    print("========================================")
    print("gen_config_plain_sm90")
    print(f"shape:            M={args.m} N={args.n} K={args.k}")
    print(f"gpu:              {gpu_name}")
    print(f"csv:              {csv_path}")
    print(f"csv raw rows:     {counts['raw_rows']}")
    print(f"csv rows kept:    {counts['kept']}")
    print(f"top_csv:          {args.top_csv}")
    print(f"matched unique:   {len(matched)}")
    print(f"missing:          {len(missing)}")
    print(f"stream_k skipped: {len(skipped_stream_k)}")
    print(f"include_stream_k: {args.include_stream_k}")
    print(f"timing_mode:      {args.timing_mode}")
    print(f"graph_repeats:    {args.graph_repeats}")
    print(f"csv A/B/C/D:      {args.csv_a_dtype}:{args.csv_a_layout} / {args.csv_b_dtype}:{args.csv_b_layout} / {args.csv_c_dtype}:{args.csv_c_layout} / {args.csv_d_dtype}:{args.csv_d_layout}")
    print(f"csv accum:        {args.csv_accum_dtype}")
    print("")
    print("CSV filter counts:")
    for k, v in counts.items():
        print(f"  {k:20s}: {v}")
    print("========================================")

    if args.dry_run:
        write_json(out_missing, {"missing": missing, "stream_k_skipped": skipped_stream_k, "counts": counts})
        print(f"dry run wrote: {out_missing}")
        return

    ext = load_ooverlap_ext(root)
    plain_fn, plain_name = get_plain_func(ext)

    M, N, K = args.m, args.n, args.k
    A = torch.randn((M, K), device=device, dtype=torch.float16)
    B_col = torch.randn((N, K), device=device, dtype=torch.float16)
    D_col = torch.empty((N, M), device=device, dtype=torch.float16)
    C_ref = torch.empty((M, N), device=device, dtype=torch.float16)
    B_view = B_col.t()

    def torch_ref_fn() -> None:
        torch.matmul(A, B_view, out=C_ref)

    torch_ref_ms = time_cuda_eager(torch_ref_fn, args.warmup, args.iters)

    ok: List[Dict[str, Any]] = []
    failed: List[Dict[str, Any]] = []

    for i, r in enumerate(matched, 1):
        algo = int(r["algo"])
        print(
            f"[{i}/{len(matched)}] "
            f"algo={algo} tile={r['tile_m']}x{r['tile_n']}x{r['tile_k']} "
            f"cluster={r['cluster']} stages={r['stages']} "
            f"mainloop={r['mainloop']} scheduler={r['scheduler']} "
            f"cutlass={r['cutlass_runtime']:.6f}"
        )

        try:
            ms, timing_used, graph_error = benchmark_algo(ext, plain_fn, args, algo, A, B_col, D_col)
            rr = dict(r)
            rr["plain_gemm_ms"] = float(ms)
            rr["timing_used"] = timing_used
            rr["graph_capture_error"] = graph_error

            if args.check:
                plain_fn(A, B_col, D_col, algo)
                torch_ref_fn()
                cuda_sync()
                err = error_summary(D_col.t(), C_ref)
                rr.update(err)
                try:
                    torch.testing.assert_close(D_col.t(), C_ref, atol=args.atol, rtol=args.rtol)
                    rr["check_ok"] = True
                except AssertionError as e:
                    rr["check_ok"] = False
                    rr["check_error"] = str(e).splitlines()[0]

            ok.append(rr)
            print(f"  plain_gemm_ms={ms:.6f} timing={timing_used}")
            if args.check:
                print(
                    f"  err max={rr.get('max_abs', float('nan')):.6f} "
                    f"mean={rr.get('mean_abs', float('nan')):.6f} "
                    f"p99={rr.get('p99_abs', float('nan')):.6f} "
                    f"check_ok={rr.get('check_ok')}"
                )
            if graph_error:
                print(f"  graph_capture_error={graph_error}")

        except Exception as e:
            rr = dict(r)
            rr["error"] = repr(e)
            failed.append(rr)
            print(f"  FAILED: {repr(e)}")

    ok.sort(key=lambda x: float(x["plain_gemm_ms"]))
    selected = ok[: args.top_save]

    flops = 2.0 * M * N * K
    result = {
        "description": "plain SM90 GEMM configs matched against CUTLASS CSV; no RA/MM/reorder/signal epilogue",
        "M": M,
        "N": N,
        "K": K,
        "gpu": gpu_name,
        "csv": str(csv_path),
        "entry": plain_name,
        "warmup": args.warmup,
        "iters": args.iters,
        "timing_mode": args.timing_mode,
        "graph_repeats": args.graph_repeats,
        "include_stream_k": args.include_stream_k,
        "csv_filter": {
            "a_dtype": args.csv_a_dtype,
            "b_dtype": args.csv_b_dtype,
            "accum_dtype": args.csv_accum_dtype,
            "c_dtype": args.csv_c_dtype,
            "d_dtype": args.csv_d_dtype,
            "a_layout": args.csv_a_layout,
            "b_layout": args.csv_b_layout,
            "c_layout": args.csv_c_layout,
            "d_layout": args.csv_d_layout,
        },
        "torch_row_ms": torch_ref_ms,
        "torch_row_tflops": flops / (torch_ref_ms * 1.0e-3) / 1.0e12,
        "BM": [int(x["tile_m"]) for x in selected],
        "BN": [int(x["tile_n"]) for x in selected],
        "BK": [int(x["tile_k"]) for x in selected],
        "Stages": [int(x["stages"]) for x in selected],
        "Algo": [int(x["algo"]) for x in selected],
        "Scheduler": [str(x["scheduler"]) for x in selected],
        "cutlass_runtime": [float(x["cutlass_runtime"]) for x in selected],
        "dur": [float(x["plain_gemm_ms"]) for x in selected],
        "top": selected,
        "all_profiled": ok,
        "csv_filter_counts": counts,
    }

    failed_obj = {
        "description": "plain SM90 CSV rows that matched/parsed but were not benchmarked successfully, plus skipped stream-k rows",
        "failed": failed,
        "stream_k_skipped": skipped_stream_k,
    }
    missing_obj = {
        "description": "plain SM90 CSV rows that passed filters but did not match the plain algo map",
        "missing": missing,
        "csv_filter_counts": counts,
    }

    write_json(out_json, result)
    write_csv(out_csv, ok)
    write_json(out_missing, missing_obj)
    write_json(out_failed, failed_obj)

    print("")
    print("========================================")
    print("DONE")
    print(f"profiled ok:       {len(ok)}")
    print(f"failed:            {len(failed)}")
    print(f"missing:           {len(missing)}")
    print(f"stream_k skipped:  {len(skipped_stream_k)}")
    print(f"torch_row_ms:      {torch_ref_ms:.6f}")
    print("")
    print(f"wrote json:        {out_json}")
    print(f"wrote csv:         {out_csv}")
    print(f"wrote missing:     {out_missing}")
    print(f"wrote failed:      {out_failed}")

    if selected:
        print("")
        print("Top selected configs:")
        for idx, r in enumerate(selected, 1):
            tflops = flops / (float(r["plain_gemm_ms"]) * 1.0e-3) / 1.0e12
            print(
                f"  #{idx}: algo={r['algo']} "
                f"tile={r['tile_m']}x{r['tile_n']}x{r['tile_k']} "
                f"cluster={r['cluster']} stages={r['stages']} "
                f"mainloop={r['mainloop']} scheduler={r['scheduler']} "
                f"plain={r['plain_gemm_ms']:.6f} ms "
                f"cutlass={r['cutlass_runtime']:.6f} ms "
                f"TFLOP/s={tflops:.2f} "
                f"check_ok={r.get('check_ok', 'NA')}"
            )
    else:
        print("")
        print("No successful configs.")

    print("========================================")

    if not matched:
        raise RuntimeError(f"No CSV candidates matched the plain algo map. See {out_missing}")


if __name__ == "__main__":
    main()
