#!/usr/bin/env python3
"""
Generate/profile plain SM90 GEMM configs against a CUTLASS profiler CSV.

Fix in this version: correctness uses baseline_gemm_col by default when that
binding exists. That avoids judging f16-accumulate CUTLASS kernels against
Torch matmul's different accumulation path.

Tensor convention:
  A      physical [M, K], row-major
  B_col  physical [N, K], logical column-major B [K, N]
  D_col  physical [N, M], logical column-major D [M, N]

Typical run:
  python tool/gen_config_plain_sm90.py \
    --m 16384 --n 8192 --k 8192 \
    --csv ~/csv_out_h100/m16384n8192k8192.gemm.csv \
    --csv-accum-dtype f16 \
    --top-csv 100 --top-save 20 \
    --warmup 50 --iters 500 \
    --timing-mode eager
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
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch


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


DEFAULT_PLAIN_ALGOS: Dict[int, Dict[str, Any]] = {
    0: dict(tile_m=128, tile_n=256, tile_k=64, stages=4, cluster=[2, 1, 1], mainloop="cooperative", epilogue="auto", scheduler="normal"),
    1: dict(tile_m=256, tile_n=128, tile_k=64, stages=4, cluster=[2, 1, 1], mainloop="cooperative", epilogue="auto", scheduler="normal"),
    2: dict(tile_m=128, tile_n=256, tile_k=64, stages=4, cluster=[1, 2, 1], mainloop="cooperative", epilogue="auto", scheduler="normal"),
    3: dict(tile_m=256, tile_n=128, tile_k=64, stages=4, cluster=[1, 2, 1], mainloop="cooperative", epilogue="auto", scheduler="normal"),
}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_ooverlap_ext(root: Optional[Path] = None):
    root = root or repo_root()
    so = root / "build" / "lib" / "ooverlap_ext.so"
    if not so.exists():
        raise FileNotFoundError(f"Could not find {so}. Build first.")
    module_name = "ooverlap_ext"
    if module_name in sys.modules:
        return sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, str(so))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load extension spec for {so}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


def get_plain_func(ext):
    for name in ("gemm_plain_sm90", "plain_gemm_sm90", "gemm_sm90_plain"):
        if hasattr(ext, name):
            return getattr(ext, name), name
    raise AttributeError("Could not find ext.gemm_plain_sm90(A, B_col, D_col, algo)")


def get_baseline_col_func(ext):
    for name in ("baseline_gemm_col", "cublas_gemm_col", "gemm_col_baseline"):
        if hasattr(ext, name):
            return getattr(ext, name), name
    return None, None


def norm_col(x: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(x).strip().lower()).strip("_")


def find_col(fieldnames: Sequence[str], names: Sequence[str]) -> Optional[str]:
    cmap = {norm_col(c): c for c in fieldnames}
    for n in names:
        if norm_col(n) in cmap:
            return cmap[norm_col(n)]
    return None


def as_int(x: Any, default: Optional[int] = None) -> Optional[int]:
    try:
        if x is None:
            return default
        s = str(x).strip()
        return int(float(s)) if s else default
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
        return default if math.isnan(v) else v
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
    m = re.search(r"gemm_([^_]+)_([^_]+)_([^_]+)_([^_]+)_([^_]+)_", str(op))
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
    return "ws"


def map_scheduler(op: str) -> str:
    s = str(op).lower().replace("-", "_")
    return "stream_k" if ("stream_k" in s or "streamk" in s) else "normal"


def filename_shape(path: Path) -> Optional[Tuple[int, int, int]]:
    m = re.search(r"m(\d+)n(\d+)k(\d+)", path.name, re.IGNORECASE)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2)), int(m.group(3))


def gpu_name_slug() -> Tuple[str, str]:
    name = torch.cuda.get_device_properties(torch.cuda.current_device()).name
    slug = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    return slug, name


def normalize_stage(x: Any) -> int:
    if x is None:
        return -1
    s = str(x).strip().lower()
    if s in ("auto", "-1"):
        return -1
    return int(float(s))


def make_key(meta: Dict[str, Any]) -> Tuple[Any, ...]:
    c = meta["cluster"]
    return (
        int(meta["tile_m"]), int(meta["tile_n"]), int(meta["tile_k"]),
        int(c[0]), int(c[1]), int(c[2]),
        normalize_stage(meta.get("stages", -1)),
        str(meta["mainloop"]),
        str(meta.get("epilogue", "auto")),
        str(meta.get("scheduler", "normal")),
    )


def load_plain_algo_map(path: Optional[str]) -> Dict[int, Dict[str, Any]]:
    if path is None or path == "auto":
        p = repo_root() / "configs" / "AlgoDictSm90.json"
        if not p.exists():
            print("[WARN] configs/AlgoDictSm90.json not found; using built-in fallback map")
            return {k: dict(v, algo=k) for k, v in DEFAULT_PLAIN_ALGOS.items()}
    else:
        p = Path(path).expanduser().resolve()

    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)

    items = data.get("algorithms", data if isinstance(data, list) else [])
    out: Dict[int, Dict[str, Any]] = {}
    for item in items:
        algo = int(item["algo"])
        cluster = item.get("cluster", [item.get("cluster_m", 1), item.get("cluster_n", 1), item.get("cluster_k", 1)])
        out[algo] = {
            "algo": algo,
            "tile_m": int(item["tile_m"]),
            "tile_n": int(item["tile_n"]),
            "tile_k": int(item["tile_k"]),
            "stages": normalize_stage(item.get("stages", -1)),
            "cluster": [int(x) for x in cluster],
            "mainloop": str(item.get("mainloop", "cooperative")),
            "epilogue": str(item.get("epilogue", "auto")),
            "scheduler": str(item.get("scheduler", "normal")),
        }
    print(f"loaded plain algo map: {p} ({len(out)} algos)")
    return out


def parse_csv_rows(args: argparse.Namespace, csv_path: Path) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames or []
        c_runtime = find_col(fields, RUNTIME_NAMES)
        c_op = find_col(fields, OP_NAMES)
        c_a = find_col(fields, A_NAMES)
        c_b = find_col(fields, B_NAMES)
        c_c = find_col(fields, C_NAMES)
        c_d = find_col(fields, D_NAMES)
        c_accum = find_col(fields, ACCUM_NAMES)
        c_split = find_col(fields, SPLIT_K_NAMES)
        c_tm = find_col(fields, CTA_M_NAMES)
        c_tn = find_col(fields, CTA_N_NAMES)
        c_tk = find_col(fields, CTA_K_NAMES)
        c_cm = find_col(fields, CLUSTER_M_NAMES)
        c_cn = find_col(fields, CLUSTER_N_NAMES)
        c_ck = find_col(fields, CLUSTER_K_NAMES)
        c_stages = find_col(fields, STAGES_NAMES)
        if c_runtime is None:
            raise RuntimeError(f"No runtime column found. CSV columns: {fields}")

        shape_from_name = filename_shape(csv_path)
        counts = {
            "raw_rows": 0, "bad_runtime": 0, "split_k": 0, "shape": 0,
            "layout_or_dtype": 0, "tile_parse": 0, "stream_k_rows_seen": 0, "kept": 0,
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
                if shape_from_name != (args.m, args.n, args.k):
                    counts["shape"] += 1
                    continue

            tm = as_int(row.get(c_tm)) if c_tm else None
            tn = as_int(row.get(c_tn)) if c_tn else None
            tk = as_int(row.get(c_tk)) if c_tk else None
            stages = as_int(row.get(c_stages)) if c_stages else None
            if tm is None or tn is None or tk is None or stages is None:
                counts["tile_parse"] += 1
                continue

            cm = as_int(row.get(c_cm), 1) if c_cm else 1
            cn = as_int(row.get(c_cn), 1) if c_cn else 1
            ck = as_int(row.get(c_ck), 1) if c_ck else 1
            op_dt = parse_op_dtypes(op)

            a_dtype = norm_dtype(row.get(c_a)) if c_a else op_dt["a"]
            b_dtype = norm_dtype(row.get(c_b)) if c_b else op_dt["b"]
            c_dtype = norm_dtype(row.get(c_c)) if c_c else op_dt["c"]
            d_dtype = norm_dtype(row.get(c_d)) if c_d else op_dt["d"]
            accum = norm_dtype(row.get(c_accum)) if c_accum else op_dt["accum"]
            a_layout = norm_layout(row.get(c_a)) if c_a else None
            b_layout = norm_layout(row.get(c_b)) if c_b else None
            c_layout = norm_layout(row.get(c_c)) if c_c else None
            d_layout = norm_layout(row.get(c_d)) if c_d else None

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
                "tile_m": int(tm), "tile_n": int(tn), "tile_k": int(tk),
                "cluster": [int(cm), int(cn), int(ck)],
                "stages": int(stages),
                "mainloop": map_mainloop(op),
                "epilogue": "auto",
                "scheduler": scheduler,
                "a": row.get(c_a) if c_a else None,
                "b": row.get(c_b) if c_b else None,
                "c": row.get(c_c) if c_c else op_dt["c"],
                "d": row.get(c_d) if c_d else op_dt["d"],
                "a_dtype": a_dtype, "b_dtype": b_dtype, "c_dtype": c_dtype, "d_dtype": d_dtype,
                "a_layout": a_layout, "b_layout": b_layout, "c_layout": c_layout, "d_layout": d_layout,
                "accum_dtype": accum,
                "operation": op,
            }
            rec["key"] = list(make_key(rec))
            records.append(rec)
            counts["kept"] += 1

    records.sort(key=lambda r: float(r["cutlass_runtime"]))
    return records, counts


def match_records(records: Sequence[Dict[str, Any]], algo_map: Dict[int, Dict[str, Any]], include_stream_k: bool):
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
        algo = by_key.get(tuple(r["key"]))
        if algo is None:
            rr = dict(r)
            rr["error"] = "no_plain_algo_for_key"
            missing.append(rr)
            continue
        rr = dict(r)
        rr["algo"] = int(algo)
        rr.update({f"algo_{k}": v for k, v in algo_map[algo].items() if k != "algo"})
        if algo not in matched_by_algo or rr["cutlass_runtime"] < matched_by_algo[algo]["cutlass_runtime"]:
            matched_by_algo[algo] = rr
    matched = list(matched_by_algo.values())
    matched.sort(key=lambda r: float(r["cutlass_runtime"]))
    return matched, missing, skipped_stream_k


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


def sample_for_quantile(x: torch.Tensor, max_samples: int = 1_000_000) -> torch.Tensor:
    x = x.reshape(-1)
    n = int(x.numel())
    if n > max_samples:
        step = (n + max_samples - 1) // max_samples
        x = x[::step][:max_samples]
    return x


def error_summary(x: torch.Tensor, y: torch.Tensor) -> Dict[str, float]:
    diff = (x - y).abs().float().reshape(-1)
    qdiff = sample_for_quantile(diff)
    return {
        "max_abs": float(diff.max().item()),
        "mean_abs": float(diff.mean().item()),
        "p99_abs": float(torch.quantile(qdiff, 0.99).item()),
        "p999_abs": float(torch.quantile(qdiff, 0.999).item()),
    }


def resolve_check_ref(args: argparse.Namespace, baseline_col_fn) -> str:
    if args.check_ref == "none":
        return "none"
    if args.check_ref == "torch":
        return "torch"
    if args.check_ref == "cublas":
        if baseline_col_fn is None:
            raise RuntimeError("--check-ref cublas requested, but ext.baseline_gemm_col was not found")
        return "cublas_col"
    return "cublas_col" if baseline_col_fn is not None else "torch"


def default_tolerance(args: argparse.Namespace, check_ref: str) -> Tuple[float, float]:
    if args.atol is not None:
        return float(args.atol), float(args.rtol)
    if check_ref == "cublas_col":
        return 0.75, float(args.rtol)
    if args.csv_accum_dtype == "f16":
        return 8.0, float(args.rtol)
    return 0.75, float(args.rtol)


def make_reference(check_ref: str, baseline_col_fn, A: torch.Tensor, B_col: torch.Tensor, d_shape: Tuple[int, int]) -> Optional[torch.Tensor]:
    if check_ref == "none":
        return None
    if check_ref == "cublas_col":
        out = torch.empty(d_shape, device=A.device, dtype=torch.float16)
        baseline_col_fn(A, B_col, out)
        cuda_sync()
        return out
    if check_ref == "torch":
        out = torch.matmul(A, B_col.t()).contiguous()
        cuda_sync()
        return out
    raise ValueError(f"unknown check_ref={check_ref}")


def compare_output(check_ref: str, D_col: torch.Tensor, ref: Optional[torch.Tensor]) -> Dict[str, Any]:
    if check_ref == "none" or ref is None:
        return {"check_ref": "none", "check_ok": None, "max_abs": None, "mean_abs": None, "p99_abs": None, "p999_abs": None}
    if check_ref == "cublas_col":
        err = error_summary(D_col, ref)
    elif check_ref == "torch":
        err = error_summary(D_col.t().contiguous(), ref)
    else:
        raise ValueError(f"unknown check_ref={check_ref}")
    return {"check_ref": check_ref, **err}


def benchmark_algo(plain_fn, args: argparse.Namespace, algo: int, A: torch.Tensor, B_col: torch.Tensor, D_col: torch.Tensor):
    def run() -> None:
        plain_fn(A, B_col, D_col, int(algo))
    run()
    cuda_sync()
    if args.timing_mode == "eager":
        return time_cuda_eager(run, args.warmup, args.iters), "eager", None
    if args.timing_mode == "graph":
        return time_cuda_graph_unrolled(run, args.warmup, args.iters, args.graph_repeats), f"graph_unrolled_{args.graph_repeats}", None
    try:
        return time_cuda_graph_unrolled(run, args.warmup, args.iters, args.graph_repeats), f"graph_unrolled_{args.graph_repeats}", None
    except Exception as e:
        return time_cuda_eager(run, args.warmup, args.iters), "eager_fallback", repr(e)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = [
        "rank", "algo", "plain_gemm_ms", "cutlass_runtime", "timing_used",
        "tile_m", "tile_n", "tile_k", "cluster", "stages", "mainloop", "scheduler",
        "check_ref", "check_ok", "max_abs", "mean_abs", "p99_abs", "p999_abs",
        "source_row", "operation",
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=cols)
        wr.writeheader()
        for r in rows:
            wr.writerow({k: r.get(k, "") for k in cols})


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
    ap.add_argument("--check-ref", choices=["auto", "cublas", "torch", "none"], default="auto")
    ap.add_argument("--also-check-torch", action="store_true")
    ap.add_argument("--atol", type=float, default=None)
    ap.add_argument("--rtol", type=float, default=2.0e-2)
    ap.add_argument("--reject-failed-check", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
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

    if args.top_csv <= 0 or args.top_save <= 0 or args.iters <= 0 or args.graph_repeats <= 0:
        raise ValueError("top-csv, top-save, iters, and graph-repeats must be > 0")
    if args.warmup < 0:
        raise ValueError("warmup must be >= 0")

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
    baseline_col_fn, baseline_col_name = get_baseline_col_func(ext)

    check_ref = "none" if not args.check else resolve_check_ref(args, baseline_col_fn)
    atol, rtol = default_tolerance(args, check_ref)

    if args.check:
        print("")
        print("Correctness reference:")
        print(f"  check_ref:       {check_ref}")
        print(f"  baseline_col_fn: {baseline_col_name if baseline_col_name else 'not found'}")
        print(f"  atol:            {atol}")
        print(f"  rtol:            {rtol}")
        if check_ref == "torch" and args.csv_accum_dtype == "f16":
            print("  note:            f16-accum kernels can differ from torch.matmul; cublas_col is preferred.")
        print("========================================")

    A = torch.randn((args.m, args.k), device=device, dtype=torch.float16)
    B_col = torch.randn((args.n, args.k), device=device, dtype=torch.float16)
    D_col = torch.empty((args.n, args.m), device=device, dtype=torch.float16)
    ref = make_reference(check_ref, baseline_col_fn, A, B_col, (args.n, args.m)) if args.check else None
    torch_ref = make_reference("torch", baseline_col_fn, A, B_col, (args.n, args.m)) if args.also_check_torch else None

    flops = 2.0 * args.m * args.n * args.k
    ok_rows: List[Dict[str, Any]] = []
    failed: List[Dict[str, Any]] = []

    for i, r in enumerate(matched, start=1):
        algo = int(r["algo"])
        print(f"[{i}/{len(matched)}] algo={algo} tile={r['tile_m']}x{r['tile_n']}x{r['tile_k']} cluster={r['cluster']} stages={r['stages']} mainloop={r['mainloop']} scheduler={r['scheduler']} cutlass={float(r['cutlass_runtime']):.6f}")
        try:
            ms, timing_used, graph_error = benchmark_algo(plain_fn, args, algo, A, B_col, D_col)
            rr = dict(r)
            rr["plain_gemm_ms"] = float(ms)
            rr["measured_ms"] = float(ms)
            rr["timing_used"] = timing_used
            rr["graph_capture_error"] = graph_error
            rr["plain_entry"] = plain_name

            if args.check:
                check_obj = compare_output(check_ref, D_col, ref)
                check_ok = True if check_obj.get("max_abs") is None else (float(check_obj["max_abs"]) <= atol)
                check_obj["check_ok"] = bool(check_ok)
                check_obj["check_atol"] = float(atol)
                check_obj["check_rtol"] = float(rtol)
                rr.update(check_obj)
                if args.also_check_torch and torch_ref is not None:
                    torch_obj = compare_output("torch", D_col, torch_ref)
                    rr["torch_max_abs"] = torch_obj["max_abs"]
                    rr["torch_mean_abs"] = torch_obj["mean_abs"]
                    rr["torch_p99_abs"] = torch_obj["p99_abs"]

                print(f"  plain_gemm_ms={ms:.6f} timing={timing_used}")
                print(f"  err({check_ref}) max={rr.get('max_abs'):.6f} mean={rr.get('mean_abs'):.6f} p99={rr.get('p99_abs'):.6f} check_ok={rr.get('check_ok')}")
                if args.also_check_torch and torch_ref is not None:
                    print(f"  err(torch)     max={rr.get('torch_max_abs'):.6f} mean={rr.get('torch_mean_abs'):.6f} p99={rr.get('torch_p99_abs'):.6f}")
                if args.reject_failed_check and not rr["check_ok"]:
                    raise RuntimeError(f"correctness check failed: max_abs={rr.get('max_abs')} > atol={atol}")
            else:
                rr.update(compare_output("none", D_col, None))
                print(f"  plain_gemm_ms={ms:.6f} timing={timing_used}")

            if graph_error:
                print(f"  graph_capture_error={graph_error}")
            ok_rows.append(rr)
        except KeyboardInterrupt:
            raise
        except Exception as e:
            rr = dict(r)
            rr["error"] = repr(e)
            failed.append(rr)
            print(f"  FAILED: {repr(e)}")

    ok_rows.sort(key=lambda x: float(x["plain_gemm_ms"]))
    for rank, r in enumerate(ok_rows, start=1):
        r["rank"] = rank
        r["tflops"] = flops / (float(r["plain_gemm_ms"]) * 1.0e-3) / 1.0e12
    selected = ok_rows[: args.top_save]

    result = {
        "description": "ooverlap plain SM90 GEMM configs selected from CUTLASS CSV then benchmarked",
        "M": args.m, "N": args.n, "K": args.k,
        "gpu": gpu_name,
        "csv": str(csv_path),
        "plain_algo_json": args.plain_algo_json,
        "plain_entry": plain_name,
        "baseline_col_entry": baseline_col_name,
        "warmup": args.warmup,
        "iters": args.iters,
        "timing_mode": args.timing_mode,
        "graph_repeats": args.graph_repeats,
        "include_stream_k": args.include_stream_k,
        "check": args.check,
        "check_ref": check_ref,
        "check_atol": atol,
        "check_rtol": rtol,
        "csv_filter": {
            "a_dtype": args.csv_a_dtype, "b_dtype": args.csv_b_dtype, "accum_dtype": args.csv_accum_dtype,
            "c_dtype": args.csv_c_dtype, "d_dtype": args.csv_d_dtype,
            "a_layout": args.csv_a_layout, "b_layout": args.csv_b_layout,
            "c_layout": args.csv_c_layout, "d_layout": args.csv_d_layout,
        },
        "BM": [int(x["tile_m"]) for x in selected],
        "BN": [int(x["tile_n"]) for x in selected],
        "BK": [int(x["tile_k"]) for x in selected],
        "Stages": [int(x["stages"]) for x in selected],
        "Algo": [int(x["algo"]) for x in selected],
        "dur": [float(x["plain_gemm_ms"]) for x in selected],
        "top": selected,
        "all_profiled": ok_rows,
    }
    missing_obj = {
        "description": "CUTLASS CSV rows considered but not matched to generated plain algos",
        "csv_filter_counts": counts,
        "top_csv": args.top_csv,
        "missing": missing,
        "stream_k_skipped": skipped_stream_k,
    }
    failed_obj = {"description": "Rows that matched but failed during benchmarking", "failed": failed}

    write_json(out_json, result)
    write_csv(out_csv, ok_rows)
    write_json(out_missing, missing_obj)
    write_json(out_failed, failed_obj)

    print("")
    print("========================================")
    print("DONE")
    print(f"profiled ok:       {len(ok_rows)}")
    print(f"failed:            {len(failed)}")
    print(f"missing:           {len(missing)}")
    print(f"stream_k skipped:  {len(skipped_stream_k)}")
    print("")
    print(f"wrote json:        {out_json}")
    print(f"wrote csv:         {out_csv}")
    print(f"wrote missing:     {out_missing}")
    print(f"wrote failed:      {out_failed}")

    if selected:
        print("")
        print("Top selected configs:")
        for i, r in enumerate(selected, start=1):
            print(f"  #{i}: algo={r['algo']} tile={r['tile_m']}x{r['tile_n']}x{r['tile_k']} cluster={r['cluster']} stages={r['stages']} mainloop={r['mainloop']} scheduler={r['scheduler']} plain={float(r['plain_gemm_ms']):.6f} ms cutlass={float(r['cutlass_runtime']):.6f} ms TFLOP/s={float(r['tflops']):.2f} check_ref={r.get('check_ref')} check_ok={r.get('check_ok')}")
    else:
        print("")
        print("No successful configs.")
    print("========================================")

    if not matched:
        raise RuntimeError(f"No CSV candidates matched generated plain algos. See {out_missing}")


if __name__ == "__main__":
    main()
