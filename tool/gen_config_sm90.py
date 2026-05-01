#!/usr/bin/env python3
"""
Profile ooverlap SM90 signal GEMM candidates against a CUTLASS profiler CSV.

This replacement uses the same f16/B=[N,K]/column-output convention as the
plain GEMM bring-up. Correctness checks use baseline_gemm_col when available
and default to a tolerance suitable for f16-accumulate / different reduction
orders.
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


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


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


def map_mainloop(op: str, default: str = "ws") -> str:
    s = str(op).lower().replace("-", "_")
    if "pingpong" in s or "ping_pong" in s:
        return "pingpong"
    if "cooperative" in s or "coop" in s:
        return "cooperative"
    if "warpspecialized" in s or "warp_specialized" in s or "tma" in s:
        return default
    return default


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
        str(meta.get("mainloop", "ws")),
        str(meta.get("epilogue", "auto")),
        str(meta.get("scheduler", "normal")),
    )


def gpu_name_slug() -> Tuple[str, str]:
    name = torch.cuda.get_device_properties(torch.cuda.current_device()).name
    slug = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    return slug, name


def load_ext(root: Path):
    so = root / "build" / "lib" / "ooverlap_ext.so"
    if not so.exists():
        raise FileNotFoundError(f"Could not find {so}. Build first.")
    module_name = "ooverlap_ext"
    if module_name in sys.modules:
        return sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, str(so))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load Python extension spec from {so}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


def get_baseline_col(ext):
    for name in ("baseline_gemm_col", "cublas_gemm_col", "gemm_col_baseline"):
        if hasattr(ext, name):
            return getattr(ext, name), name
    return None, None


def load_algo_map(path: Path) -> Dict[int, Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    out: Dict[int, Dict[str, Any]] = {}
    for item in data.get("algorithms", []):
        algo = int(item["algo"])
        cluster = item.get("cluster", [item.get("cluster_m", 1), item.get("cluster_n", 1), item.get("cluster_k", 1)])
        out[algo] = {
            "algo": algo,
            "tile_m": int(item["tile_m"]),
            "tile_n": int(item["tile_n"]),
            "tile_k": int(item["tile_k"]),
            "cluster": [int(x) for x in cluster],
            "stages": normalize_stage(item.get("stages", -1)),
            "mainloop": str(item.get("mainloop", "ws")),
            "epilogue": str(item.get("epilogue", "auto")),
            "scheduler": str(item.get("scheduler", "normal")),
        }
    return out


def parse_csv_rows(args, csv_path: Path) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
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
                "tile_m": int(tm), "tile_n": int(tn), "tile_k": int(tk),
                "cluster": [int(cm), int(cn), int(ck)],
                "stages": int(stages),
                "mainloop": map_mainloop(op, args.default_mainloop),
                "epilogue": "auto",
                "scheduler": scheduler,
                "a": row.get(c_a) if c_a is not None else None,
                "b": row.get(c_b) if c_b is not None else None,
                "c": row.get(c_c) if c_c is not None else op_dt["c"],
                "d": row.get(c_d) if c_d is not None else op_dt["d"],
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


def match_records(records, algo_map, include_stream_k):
    by_key: Dict[Tuple[Any, ...], int] = {}
    for algo, meta in algo_map.items():
        by_key[make_key(meta)] = int(algo)

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
            rr["error"] = "no_signal_algo_for_key"
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


def make_ra(tile_rows: int, tile_cols: int, reorder: str, device: torch.device) -> torch.Tensor:
    num_tiles = tile_rows * tile_cols
    if reorder == "identity":
        return torch.arange(num_tiles, device=device, dtype=torch.int32)
    if reorder == "column_major":
        host = [0] * num_tiles
        packed = 0
        for tc in range(tile_cols):
            for tr in range(tile_rows):
                host[tr * tile_cols + tc] = packed
                packed += 1
        return torch.tensor(host, device=device, dtype=torch.int32)
    raise ValueError(f"unknown reorder={reorder}")


def logical_col_major_view(buf: torch.Tensor, rows: int, cols: int) -> torch.Tensor:
    return torch.as_strided(buf, size=(rows, cols), stride=(1, rows))


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
            out[tr * tile_m:(tr + 1) * tile_m, tc * tile_n:(tc + 1) * tile_n].copy_(
                D_packed_logical[pm * tile_m:(pm + 1) * tile_m, pn * tile_n:(pn + 1) * tile_n]
            )
    return out


def make_output_normal(D, layout, RA, M, N, tile_m, tile_n, reldn):
    D_logical = logical_col_major_view(D, D.shape[0], D.shape[1])
    if layout == "normal":
        return D_logical
    return unpack_packed_to_normal(D_logical, RA, M, N, tile_m, tile_n, reldn)


def make_segments(num_tiles: int, group_tiles: int, device: torch.device):
    if group_tiles <= 0:
        return torch.tensor([num_tiles], device=device, dtype=torch.int32)
    vals = []
    left = num_tiles
    while left > 0:
        x = min(group_tiles, left)
        vals.append(x)
        left -= x
    return torch.tensor(vals, device=device, dtype=torch.int32)


def time_cuda_eager(fn: Callable[[], None], warmup: int, iters: int) -> float:
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


def time_cuda_graph_unrolled(fn: Callable[[], None], warmup: int, iters: int, graph_repeats: int) -> float:
    for _ in range(max(1, warmup)):
        fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(graph_repeats):
            fn()
    for _ in range(max(1, warmup)):
        graph.replay()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        graph.replay()
    end.record()
    torch.cuda.synchronize()
    return float(start.elapsed_time(end)) / float(iters * graph_repeats)


def safe_quantile_1d(x: torch.Tensor, q: float, max_samples: int = 1_000_000) -> float:
    x = x.reshape(-1)
    if int(x.numel()) > max_samples:
        step = (int(x.numel()) + max_samples - 1) // max_samples
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


def benchmark_algo(ext, args, meta, A, B_nk, D, RA, CommThr, MM):
    M, N, K = args.m, args.n, args.k
    tm, tn = int(meta["tile_m"]), int(meta["tile_n"])
    tile_rows, tile_cols = M // tm, N // tn
    reldn = tile_cols if args.layout == "normal" and args.reldn == 0 else args.reldn
    if args.layout == "packed" and reldn == 0:
        reldn = 1

    def run():
        ext.gemm_signal_sm90(A, B_nk, D, MM, RA, CommThr, int(reldn), int(meta["algo"]), False)

    run()
    torch.cuda.synchronize()
    if args.timing_mode == "eager":
        return time_cuda_eager(run, args.warmup, args.iters), "eager", None
    if args.timing_mode == "graph":
        return time_cuda_graph_unrolled(run, args.warmup, args.iters, args.graph_repeats), f"graph_unrolled_{args.graph_repeats}", None
    try:
        return time_cuda_graph_unrolled(run, args.warmup, args.iters, args.graph_repeats), f"graph_unrolled_{args.graph_repeats}", None
    except Exception as e:
        return time_cuda_eager(run, args.warmup, args.iters), "eager_fallback", repr(e)


def write_json(path: Path, data: Any):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = [
        "rank", "algo", "signal_gemm_ms", "cutlass_runtime", "timing_used",
        "tile_m", "tile_n", "tile_k", "cluster", "stages", "mainloop", "scheduler",
        "check_ref", "check_ok", "max_abs", "mean_abs", "p99_abs", "p999_abs",
        "source_row", "operation",
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=cols)
        wr.writeheader()
        for r in rows:
            wr.writerow({k: r.get(k, "") for k in cols})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", type=int, required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--k", type=int, required=True)
    ap.add_argument("--csv", type=str, default=None)
    ap.add_argument("--csv-dir", type=str, default=None)
    ap.add_argument("--algo-dict", type=str, default=None)
    ap.add_argument("--layout", choices=["normal", "packed"], default="packed")
    ap.add_argument("--reorder", choices=["auto", "identity", "column_major"], default="auto")
    ap.add_argument("--reldn", type=int, default=0, help="0 means auto")
    ap.add_argument("--group-tiles", type=int, default=0)
    ap.add_argument("--top-csv", type=int, default=40)
    ap.add_argument("--top-save", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--default-mainloop", choices=["ws", "pingpong", "cooperative"], default="ws")
    ap.add_argument("--timing-mode", choices=["auto", "graph", "eager"], default="eager")
    ap.add_argument("--graph-repeats", type=int, default=100)
    ap.add_argument("--include-stream-k", action="store_true")
    ap.add_argument("--allow-split-k", action="store_true")
    ap.add_argument("--check", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--check-ref", choices=["auto", "baseline_col", "torch", "none"], default="auto")
    ap.add_argument("--check-atol", type=float, default=8.0)
    ap.add_argument("--reject-failed-check", action="store_true")
    ap.add_argument("--seed", type=int, default=1234)
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

    root = repo_root()
    csv_path = Path(args.csv).expanduser().resolve() if args.csv else None
    if csv_path is None:
        if args.csv_dir is None:
            raise ValueError("Pass --csv or --csv-dir")
        csv_path = Path(args.csv_dir).expanduser().resolve() / f"m{args.m}n{args.n}k{args.k}.gemm.csv"
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)
    algo_path = Path(args.algo_dict).expanduser().resolve() if args.algo_dict else root / "configs" / "AlgoDictSm90.json"
    if not algo_path.exists():
        raise FileNotFoundError(algo_path)

    torch.cuda.set_device(args.device)
    torch.manual_seed(args.seed)
    device = torch.device("cuda", args.device)

    algo_map = load_algo_map(algo_path)
    records, counts = parse_csv_rows(args, csv_path)
    considered = records[:args.top_csv]
    matched, missing, skipped_stream_k = match_records(considered, algo_map, args.include_stream_k)

    slug, gpu_name = gpu_name_slug()
    base = root / "configs" / f"m{args.m}n{args.n}k{args.k}_{slug}_{args.layout}_sm90"
    out_json = base.with_suffix(".json")
    out_csv = base.with_suffix(".csv")
    out_missing = Path(str(base) + "_missing.json")
    out_failed = Path(str(base) + "_failed.json")

    print("========================================")
    print("gen_config_sm90")
    print(f"shape:            M={args.m} N={args.n} K={args.k}")
    print(f"gpu:              {gpu_name}")
    print(f"layout:           {args.layout}")
    print(f"csv:              {csv_path}")
    print(f"algo_dict:        {algo_path}")
    print(f"csv raw rows:     {counts['raw_rows']}")
    print(f"csv rows kept:    {counts['kept']}")
    print(f"top_csv:          {args.top_csv}")
    print(f"matched unique:   {len(matched)}")
    print(f"missing:          {len(missing)}")
    print(f"stream_k skipped: {len(skipped_stream_k)}")
    print(f"include_stream_k: {args.include_stream_k}")
    print(f"timing_mode:      {args.timing_mode}")
    print(f"csv A/B/C/D:      {args.csv_a_dtype}:{args.csv_a_layout} / {args.csv_b_dtype}:{args.csv_b_layout} / {args.csv_c_dtype}:{args.csv_c_layout} / {args.csv_d_dtype}:{args.csv_d_layout}")
    print(f"csv accum:        {args.csv_accum_dtype}")
    print(f"check:            {args.check} atol={args.check_atol}")
    print("CSV filter counts:")
    for k, v in counts.items():
        print(f"  {k:20s}: {v}")
    print("========================================")

    if args.dry_run:
        write_json(out_missing, {"missing": missing, "stream_k_skipped": skipped_stream_k, "counts": counts})
        return

    ext = load_ext(root)
    baseline_col_fn, baseline_col_name = get_baseline_col(ext)
    check_ref = args.check_ref
    if check_ref == "auto":
        check_ref = "baseline_col" if baseline_col_fn is not None else "torch"
    if check_ref == "baseline_col" and baseline_col_fn is None:
        raise RuntimeError("baseline_gemm_col not found; use --check-ref torch or expose baseline_gemm_col")

    M, N, K = args.m, args.n, args.k
    A = torch.randn((M, K), device=device, dtype=torch.float16)
    B_nk = torch.randn((N, K), device=device, dtype=torch.float16)
    ref = None
    if args.check and check_ref == "baseline_col":
        D_ref_col = torch.empty((N, M), device=device, dtype=torch.float16)
        baseline_col_fn(A, B_nk, D_ref_col)
        torch.cuda.synchronize()
        ref = D_ref_col.t()
    elif args.check and check_ref == "torch":
        ref = torch.matmul(A, B_nk.t()).contiguous()
        torch.cuda.synchronize()

    ok_rows: List[Dict[str, Any]] = []
    failed: List[Dict[str, Any]] = []
    flops = 2.0 * M * N * K

    for i, r in enumerate(matched, 1):
        algo = int(r["algo"])
        tm, tn = int(r["tile_m"]), int(r["tile_n"])
        tile_rows, tile_cols = M // tm, N // tn
        if args.layout == "normal":
            reorder = "identity" if args.reorder == "auto" else args.reorder
            reldn = tile_cols if args.reldn == 0 else args.reldn
            out_m, out_n = M, N
        else:
            reorder = "column_major" if args.reorder == "auto" else args.reorder
            reldn = 1 if args.reldn == 0 else args.reldn
            packed_rows = (tile_rows * tile_cols + reldn - 1) // reldn
            out_m, out_n = packed_rows * tm, reldn * tn
        RA = make_ra(tile_rows, tile_cols, reorder, device)
        CommThr = make_segments(tile_rows * tile_cols, args.group_tiles, device)
        MM = torch.zeros((int(CommThr.numel()) + tile_rows * tile_cols,), device=device, dtype=torch.int32)
        D = torch.empty((out_m, out_n), device=device, dtype=torch.float16)

        print(
            f"[{i}/{len(matched)}] algo={algo} tile={r['tile_m']}x{r['tile_n']}x{r['tile_k']} "
            f"cluster={r['cluster']} stages={r['stages']} mainloop={r['mainloop']} "
            f"scheduler={r['scheduler']} cutlass={float(r['cutlass_runtime']):.6f}"
        )
        try:
            meta = {"algo": algo, "tile_m": tm, "tile_n": tn}
            ms, timing_used, graph_error = benchmark_algo(ext, args, meta, A, B_nk, D, RA, CommThr, MM)
            rr = dict(r)
            rr["signal_gemm_ms"] = float(ms)
            rr["measured_ms"] = float(ms)
            rr["timing_used"] = timing_used
            rr["graph_capture_error"] = graph_error
            rr["check_ref"] = check_ref if args.check else "none"
            if args.check and ref is not None:
                out_normal = make_output_normal(D, args.layout, RA, M, N, tm, tn, reldn)
                err = error_summary(out_normal, ref)
                rr.update(err)
                rr["check_ok"] = bool(err["max_abs"] <= args.check_atol)
                print(f"  signal_gemm_ms={ms:.6f} timing={timing_used}")
                print(f"  err({check_ref}) max={err['max_abs']:.6f} mean={err['mean_abs']:.6f} p99={err['p99_abs']:.6f} check_ok={rr['check_ok']}")
                if args.reject_failed_check and not rr["check_ok"]:
                    raise RuntimeError(f"correctness check failed: max_abs={err['max_abs']} > {args.check_atol}")
            else:
                rr["check_ok"] = None
                print(f"  signal_gemm_ms={ms:.6f} timing={timing_used}")
            ok_rows.append(rr)
        except KeyboardInterrupt:
            raise
        except Exception as e:
            rr = dict(r)
            rr["error"] = repr(e)
            failed.append(rr)
            print(f"  FAILED: {repr(e)}")

    ok_rows.sort(key=lambda x: float(x["signal_gemm_ms"]))
    for rank, r in enumerate(ok_rows, 1):
        r["rank"] = rank
        r["tflops"] = flops / (float(r["signal_gemm_ms"]) * 1.0e-3) / 1.0e12
    selected = ok_rows[:args.top_save]

    result = {
        "description": "ooverlap signal SM90 configs selected from CUTLASS CSV then benchmarked",
        "M": M, "N": N, "K": K, "gpu": gpu_name,
        "csv": str(csv_path), "algo_dict": str(algo_path), "layout": args.layout,
        "warmup": args.warmup, "iters": args.iters, "timing_mode": args.timing_mode,
        "graph_repeats": args.graph_repeats,
        "include_stream_k": args.include_stream_k,
        "check": args.check, "check_ref": check_ref, "check_atol": args.check_atol,
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
        "dur": [float(x["signal_gemm_ms"]) for x in selected],
        "top": selected,
        "all_profiled": ok_rows,
    }
    write_json(out_json, result)
    write_csv(out_csv, ok_rows)
    write_json(out_missing, {"csv_filter_counts": counts, "missing": missing, "stream_k_skipped": skipped_stream_k})
    write_json(out_failed, {"failed": failed})

    print("\n========================================")
    print("DONE")
    print(f"profiled ok:       {len(ok_rows)}")
    print(f"failed:            {len(failed)}")
    print(f"missing:           {len(missing)}")
    print(f"stream_k skipped:  {len(skipped_stream_k)}")
    print(f"wrote json:        {out_json}")
    print(f"wrote csv:         {out_csv}")
    print(f"wrote missing:     {out_missing}")
    print(f"wrote failed:      {out_failed}")
    if selected:
        print("\nTop selected configs:")
        for i, r in enumerate(selected, 1):
            print(
                f"  #{i}: algo={r['algo']} tile={r['tile_m']}x{r['tile_n']}x{r['tile_k']} "
                f"cluster={r['cluster']} stages={r['stages']} mainloop={r['mainloop']} "
                f"scheduler={r['scheduler']} signal={float(r['signal_gemm_ms']):.6f} ms "
                f"cutlass={float(r['cutlass_runtime']):.6f} ms TFLOP/s={float(r['tflops']):.2f} "
                f"check_ref={r.get('check_ref')} check_ok={r.get('check_ok')}"
            )
    else:
        print("\nNo successful configs.")
    print("========================================")
    if not matched:
        raise RuntimeError(f"No CSV candidates matched generated SM90 signal algos. See {out_missing}")


if __name__ == "__main__":
    main()
