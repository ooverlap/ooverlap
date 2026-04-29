#!/usr/bin/env python3
"""
SM90/H100 gen_config script for ooverlap.

This is the ooverlap SM90 analogue of FlashOverlap's gen_config.py:

  CUTLASS profiler CSV
    -> choose top CUTLASS rows by runtime
    -> map each row to our generated AlgoDictSm90
    -> benchmark the actual ooverlap packed reorder + signal GEMM
    -> save selected configs to JSON

This does NOT benchmark plain torch GEMM.
This does NOT benchmark NCCL.
This benchmarks:

  ext.gemm_signal_sm90(
      A,
      B_packed,
      D_packed,
      MM,
      RA,
      CommThr,
      ReLDN,
      algo,
      monitor=False,
  )

with packed output layout.

Expected generated files:
  configs/AlgoDictSm90.json
  build/lib/ooverlap_ext.so

Typical usage:

  python tool/gen_config_sm90.py \
    --m 4096 --n 4096 --k 8192 \
    --csv /path/to/m4096n4096k8192.gemm.csv \
    --top-csv 40 \
    --top-save 10 \
    --warmup 20 \
    --iters 100

or:

  python tool/gen_config_sm90.py \
    --m 4096 --n 4096 --k 8192 \
    --csv-dir /path/to/csvs \
    --top-csv 40 \
    --top-save 10
"""

import argparse
import importlib.util
import json
import math
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import torch


# -----------------------------
# Column alias helpers
# -----------------------------

RUNTIME_ALIASES = [
    "runtime",
    "runtime_ms",
    "time",
    "time_ms",
    "duration",
    "duration_ms",
    "Runtime",
]

DISPOSITION_ALIASES = [
    "disposition",
    "status",
    "verification",
    "verification_status",
]

M_ALIASES = ["m", "M", "problem_m", "gemm_m"]
N_ALIASES = ["n", "N", "problem_n", "gemm_n"]
K_ALIASES = ["k", "K", "problem_k", "gemm_k"]

TILE_M_ALIASES = [
    "cta_m",
    "threadblock_m",
    "threadblock_shape_m",
    "tile_m",
    "tb_m",
    "m_per_cta",
    "cta_shape_m",
]

TILE_N_ALIASES = [
    "cta_n",
    "threadblock_n",
    "threadblock_shape_n",
    "tile_n",
    "tb_n",
    "n_per_cta",
    "cta_shape_n",
]

TILE_K_ALIASES = [
    "cta_k",
    "threadblock_k",
    "threadblock_shape_k",
    "tile_k",
    "tb_k",
    "k_per_cta",
    "cta_shape_k",
]

CLUSTER_M_ALIASES = [
    "cluster_m",
    "cluster_shape_m",
    "cluster_shape_0",
    "cga_m",
    "cta_cluster_m",
]

CLUSTER_N_ALIASES = [
    "cluster_n",
    "cluster_shape_n",
    "cluster_shape_1",
    "cga_n",
    "cta_cluster_n",
]

CLUSTER_K_ALIASES = [
    "cluster_k",
    "cluster_shape_k",
    "cluster_shape_2",
    "cga_k",
    "cta_cluster_k",
]

SPLIT_K_ALIASES = [
    "split_k_slices",
    "split_k",
    "splitk",
    "split_k_serial",
]

SCHEDULE_ALIASES = [
    "kernel_schedule",
    "schedule",
    "mainloop_schedule",
    "gemm_schedule",
]

EPILOGUE_ALIASES = [
    "epilogue_schedule",
    "epilogue",
]

OPERATION_ALIASES = [
    "operation",
    "kernel",
    "kernel_name",
    "name",
    "procedural_name",
]


def normalize_col(name: str) -> str:
    x = str(name).strip().lower()
    x = re.sub(r"[^a-z0-9]+", "_", x)
    return x.strip("_")


def build_colmap(columns) -> Dict[str, str]:
    out = {}
    for c in columns:
        n = normalize_col(c)
        if n not in out:
            out[n] = c
    return out


def find_col(colmap: Dict[str, str], aliases: List[str]) -> Optional[str]:
    for a in aliases:
        n = normalize_col(a)
        if n in colmap:
            return colmap[n]
    return None


def get_cell(row, col: Optional[str]) -> Any:
    if col is None:
        return None
    return row[col]


def to_float(x) -> Optional[float]:
    if x is None:
        return None
    try:
        if isinstance(x, str):
            x = x.strip()
            if not x:
                return None
        v = float(x)
        if math.isnan(v):
            return None
        return v
    except Exception:
        return None


def to_int(x) -> Optional[int]:
    if x is None:
        return None
    try:
        if isinstance(x, str):
            x = x.strip()
            if not x:
                return None
            return int(float(x))
        return int(x)
    except Exception:
        return None


def passed_disposition(x) -> bool:
    if x is None:
        return True
    s = str(x).strip().lower()
    if not s:
        return True

    bad_words = [
        "failed",
        "fail",
        "incorrect",
        "error",
        "invalid",
        "not supported",
        "not_supported",
    ]

    return not any(w in s for w in bad_words)


def filename_shape(path: Path) -> Optional[Tuple[int, int, int]]:
    m = re.search(r"m(\d+)n(\d+)k(\d+)", path.name, re.IGNORECASE)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2)), int(m.group(3))


def infer_shape_from_row(row, cols, path: Path) -> Optional[Tuple[int, int, int]]:
    m = to_int(get_cell(row, cols["m"]))
    n = to_int(get_cell(row, cols["n"]))
    k = to_int(get_cell(row, cols["k"]))
    if m is not None and n is not None and k is not None:
        return m, n, k
    return filename_shape(path)


def map_mainloop(schedule_value, operation_value, default_mainloop: str) -> str:
    s = ""
    if schedule_value is not None:
        s += " " + str(schedule_value).lower()
    if operation_value is not None:
        s += " " + str(operation_value).lower()

    s = s.replace("-", "_")

    if "pingpong" in s or "ping_pong" in s:
        return "pingpong"
    if "cooperative" in s or "coop" in s:
        return "cooperative"

    if (
        "warpspecialized" in s
        or "warp_specialized" in s
        or "tma" in s
        or "wgmma" in s
        or not s.strip()
    ):
        return default_mainloop

    return default_mainloop


def map_epilogue(epilogue_value, default_epilogue: str) -> str:
    if epilogue_value is None:
        return default_epilogue

    s = str(epilogue_value).lower().replace("-", "_")
    if "auto" in s:
        return "auto"

    return default_epilogue


def extract_columns(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    colmap = build_colmap(df.columns)

    return {
        "runtime": find_col(colmap, RUNTIME_ALIASES),
        "disposition": find_col(colmap, DISPOSITION_ALIASES),
        "m": find_col(colmap, M_ALIASES),
        "n": find_col(colmap, N_ALIASES),
        "k": find_col(colmap, K_ALIASES),
        "tile_m": find_col(colmap, TILE_M_ALIASES),
        "tile_n": find_col(colmap, TILE_N_ALIASES),
        "tile_k": find_col(colmap, TILE_K_ALIASES),
        "cluster_m": find_col(colmap, CLUSTER_M_ALIASES),
        "cluster_n": find_col(colmap, CLUSTER_N_ALIASES),
        "cluster_k": find_col(colmap, CLUSTER_K_ALIASES),
        "split_k": find_col(colmap, SPLIT_K_ALIASES),
        "schedule": find_col(colmap, SCHEDULE_ALIASES),
        "epilogue": find_col(colmap, EPILOGUE_ALIASES),
        "operation": find_col(colmap, OPERATION_ALIASES),
    }


# -----------------------------
# Repo / extension loading
# -----------------------------

def repo_root_from_script() -> Path:
    return Path(__file__).resolve().parents[1]


def load_ooverlap_ext(root: Path):
    so = root / "build" / "lib" / "ooverlap_ext.so"
    if not so.exists():
        raise FileNotFoundError(f"Could not find extension: {so}. Build first.")

    spec = importlib.util.spec_from_file_location("ooverlap_ext", str(so))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_algo_dict(path: Path) -> Dict[Tuple[Any, ...], int]:
    if not path.exists():
        raise FileNotFoundError(f"AlgoDictSm90 not found: {path}")

    if path.suffix == ".json":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        out = {}
        for item in data.get("algorithms", []):
            key = (
                int(item["tile_m"]),
                int(item["tile_n"]),
                int(item["tile_k"]),
                int(item["cluster"][0]),
                int(item["cluster"][1]),
                int(item["cluster"][2]),
                str(item["mainloop"]),
                str(item["epilogue"]),
            )
            out[key] = int(item["algo"])
        return out

    if path.suffix == ".pt":
        raw = torch.load(path, map_location="cpu", weights_only=False)
        return {tuple(k): int(v) for k, v in raw.items()}

    raise ValueError(f"Unsupported AlgoDictSm90 format: {path}")


def algo_dict_inverse(algo_dict: Dict[Tuple[Any, ...], int]) -> Dict[int, Tuple[Any, ...]]:
    out = {}
    for k, v in algo_dict.items():
        out[int(v)] = tuple(k)
    return out


def key_to_record(key: Tuple[Any, ...]) -> Dict[str, Any]:
    return {
        "tile_m": int(key[0]),
        "tile_n": int(key[1]),
        "tile_k": int(key[2]),
        "cluster": [int(key[3]), int(key[4]), int(key[5])],
        "mainloop": str(key[6]),
        "epilogue": str(key[7]),
    }


# -----------------------------
# CSV parsing
# -----------------------------

def find_csv(args, root: Path) -> Path:
    if args.csv is not None:
        p = Path(args.csv).resolve()
        if not p.exists():
            raise FileNotFoundError(f"CSV not found: {p}")
        return p

    if args.csv_dir is None:
        raise ValueError("Pass either --csv or --csv-dir")

    csv_dir = Path(args.csv_dir).resolve()
    candidates = [
        csv_dir / f"m{args.m}n{args.n}k{args.k}.gemm.csv",
        csv_dir / f"m{args.m}n{args.n}k{args.k}.csv",
    ]

    for p in candidates:
        if p.exists():
            return p

    matches = sorted(csv_dir.rglob(f"*m{args.m}n{args.n}k{args.k}*.csv"))
    if matches:
        return matches[0]

    raise FileNotFoundError(
        f"Could not find CSV for m{args.m}n{args.n}k{args.k} under {csv_dir}"
    )


def parse_cutlass_csv(csv_path: Path, args) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    df = pd.read_csv(csv_path)
    cols = extract_columns(df)

    if cols["runtime"] is None:
        raise RuntimeError(
            f"No runtime column found in {csv_path}. Columns were: {list(df.columns)}"
        )

    records = []
    skipped = []

    for idx, row in df.iterrows():
        runtime = to_float(get_cell(row, cols["runtime"]))
        if runtime is None:
            skipped.append({"row": int(idx), "reason": "bad runtime"})
            continue

        if not passed_disposition(get_cell(row, cols["disposition"])):
            continue

        split_k = to_int(get_cell(row, cols["split_k"]))
        if split_k is not None and split_k != 1 and not args.allow_split_k:
            continue

        shape = infer_shape_from_row(row, cols, csv_path)
        if shape is None:
            skipped.append({"row": int(idx), "reason": "could not infer shape"})
            continue

        M, N, K = shape
        if M != args.m or N != args.n or K != args.k:
            continue

        tile_m = to_int(get_cell(row, cols["tile_m"]))
        tile_n = to_int(get_cell(row, cols["tile_n"]))
        tile_k = to_int(get_cell(row, cols["tile_k"]))

        if tile_m is None or tile_n is None or tile_k is None:
            skipped.append({
                "row": int(idx),
                "reason": "could not parse tile_m/tile_n/tile_k",
            })
            continue

        cluster_m = to_int(get_cell(row, cols["cluster_m"])) or 1
        cluster_n = to_int(get_cell(row, cols["cluster_n"])) or 1
        cluster_k = to_int(get_cell(row, cols["cluster_k"])) or 1

        mainloop = map_mainloop(
            get_cell(row, cols["schedule"]),
            get_cell(row, cols["operation"]),
            args.default_mainloop,
        )

        epilogue = map_epilogue(
            get_cell(row, cols["epilogue"]),
            args.default_epilogue,
        )

        key = (
            int(tile_m),
            int(tile_n),
            int(tile_k),
            int(cluster_m),
            int(cluster_n),
            int(cluster_k),
            str(mainloop),
            str(epilogue),
        )

        records.append({
            "source_row": int(idx),
            "cutlass_runtime": float(runtime),
            "key": key,
            "tile_m": int(tile_m),
            "tile_n": int(tile_n),
            "tile_k": int(tile_k),
            "cluster": [int(cluster_m), int(cluster_n), int(cluster_k)],
            "mainloop": str(mainloop),
            "epilogue": str(epilogue),
        })

    records.sort(key=lambda x: x["cutlass_runtime"])
    return records, skipped


def dedupe_by_key_keep_best(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    best = {}
    for r in records:
        key = r["key"]
        if key not in best or r["cutlass_runtime"] < best[key]["cutlass_runtime"]:
            best[key] = r
    out = list(best.values())
    out.sort(key=lambda x: x["cutlass_runtime"])
    return out


# -----------------------------
# Packed reorder helpers
# -----------------------------

def make_column_major_reorder(M: int, N: int, tile_m: int, tile_n: int, device) -> torch.Tensor:
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


def make_identity_reorder(M: int, N: int, tile_m: int, tile_n: int, device) -> torch.Tensor:
    tile_rows = M // tile_m
    tile_cols = N // tile_n
    num_tiles = tile_rows * tile_cols
    return torch.arange(num_tiles, device=device, dtype=torch.int32)


def make_segments(num_tiles: int, group_tiles: int) -> List[int]:
    if group_tiles <= 0:
        return [num_tiles]

    segs = []
    left = num_tiles
    while left > 0:
        x = min(group_tiles, left)
        segs.append(x)
        left -= x
    return segs


# -----------------------------
# Benchmarking
# -----------------------------

def time_cuda(fn, warmup: int, iters: int) -> float:
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

    # This is debug/check only. CPU loop with GPU slicing is fine for optional correctness.
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


def benchmark_algo(
    ext,
    algo: int,
    meta: Dict[str, Any],
    args,
    device,
) -> Tuple[Optional[float], Optional[float], Optional[str]]:
    M, N, K = args.m, args.n, args.k

    tile_m = int(meta["tile_m"])
    tile_n = int(meta["tile_n"])

    if M % tile_m != 0 or N % tile_n != 0:
        return None, None, f"M/N not divisible by tile {tile_m}x{tile_n}"

    tile_rows = M // tile_m
    tile_cols = N // tile_n
    num_tiles = tile_rows * tile_cols

    reldn = int(args.reldn)
    if reldn <= 0:
        return None, None, "ReLDN must be > 0"

    packed_tile_cols = reldn
    packed_tile_rows = (num_tiles + packed_tile_cols - 1) // packed_tile_cols

    packed_M = packed_tile_rows * tile_m
    packed_N = packed_tile_cols * tile_n

    if args.reorder == "column_major":
        RA = make_column_major_reorder(M, N, tile_m, tile_n, device)
    elif args.reorder == "identity":
        RA = make_identity_reorder(M, N, tile_m, tile_n, device)
    else:
        return None, None, f"unknown reorder={args.reorder}"

    segments = make_segments(num_tiles, args.group_tiles)
    CommThr = torch.tensor(segments, device=device, dtype=torch.int32)
    MM = torch.empty((len(segments) + num_tiles,), device=device, dtype=torch.int32)

    A = torch.randn((M, K), device=device, dtype=torch.float16)
    B_ref = torch.randn((K, N), device=device, dtype=torch.float16)
    B_packed = B_ref.t().contiguous()
    D = torch.empty((packed_M, packed_N), device=device, dtype=torch.float16)

    monitor = False

    def run():
        MM.zero_()
        ext.gemm_signal_sm90(
            A,
            B_packed,
            D,
            MM,
            RA,
            CommThr,
            int(reldn),
            int(algo),
            monitor,
        )

    try:
        ms = time_cuda(run, args.warmup, args.iters)
    except Exception as e:
        torch.cuda.synchronize()
        return None, None, f"benchmark failed: {type(e).__name__}: {e}"

    max_abs_err = None
    if args.check:
        try:
            run()
            torch.cuda.synchronize()
            D_normal = unpack_packed_to_normal(D, RA, M, N, tile_m, tile_n, reldn)
            ref = A @ B_ref
            max_abs_err = float((D_normal - ref).abs().max().item())
        except Exception as e:
            return ms, None, f"check failed: {type(e).__name__}: {e}"

    return ms, max_abs_err, None


# -----------------------------
# Main
# -----------------------------

def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--m", type=int, required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--k", type=int, required=True)

    ap.add_argument("--csv", type=str, default=None)
    ap.add_argument("--csv-dir", type=str, default=None)

    ap.add_argument("--root", type=str, default=None)
    ap.add_argument("--algo-dict", type=str, default=None)

    ap.add_argument("--device", type=int, default=0)

    ap.add_argument("--top-csv", type=int, default=40)
    ap.add_argument("--top-save", type=int, default=10)

    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--iters", type=int, default=100)

    ap.add_argument("--reldn", type=int, default=1)
    ap.add_argument("--group-tiles", type=int, default=0,
                    help="0 means one full segment. Otherwise segment by this many tiles.")

    ap.add_argument("--reorder", choices=["column_major", "identity"], default="column_major")

    ap.add_argument("--default-mainloop", choices=["ws", "pingpong", "cooperative"], default="ws")
    ap.add_argument("--default-epilogue", choices=["auto"], default="auto")

    ap.add_argument("--allow-split-k", action="store_true")
    ap.add_argument("--check", action="store_true")

    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--missing-out", type=str, default=None)
    ap.add_argument("--failed-out", type=str, default=None)

    args = ap.parse_args()

    root = Path(args.root).resolve() if args.root else repo_root_from_script()

    algo_dict_path = Path(args.algo_dict).resolve() if args.algo_dict else root / "configs" / "AlgoDictSm90.json"
    algo_dict = load_algo_dict(algo_dict_path)
    inv_algo = algo_dict_inverse(algo_dict)

    csv_path = find_csv(args, root)
    records, skipped = parse_cutlass_csv(csv_path, args)
    records = dedupe_by_key_keep_best(records)

    selected_csv = records[: args.top_csv]

    matched = []
    missing = []

    for r in selected_csv:
        key = tuple(r["key"])
        algo = algo_dict.get(key)

        item = dict(r)
        item["key"] = list(key)

        if algo is None:
            item["matched"] = False
            item["algo"] = None
            missing.append(item)
        else:
            item["matched"] = True
            item["algo"] = int(algo)
            matched.append(item)

    # Avoid profiling the same algo more than once.
    by_algo = {}
    for r in matched:
        algo = int(r["algo"])
        if algo not in by_algo or r["cutlass_runtime"] < by_algo[algo]["cutlass_runtime"]:
            by_algo[algo] = r

    candidate_rows = sorted(by_algo.values(), key=lambda x: x["cutlass_runtime"])

    if not candidate_rows:
        raise RuntimeError(
            "No CSV candidates matched AlgoDictSm90. "
            "Check configs/missing_instances_sm90.json or regenerate instances."
        )

    torch.cuda.set_device(args.device)
    device = torch.device(f"cuda:{args.device}")

    ext = load_ooverlap_ext(root)

    results = []
    failed = []

    print("========================================")
    print("gen_config_sm90")
    print(f"shape:          M={args.m} N={args.n} K={args.k}")
    print(f"csv:            {csv_path}")
    print(f"algo_dict:      {algo_dict_path}")
    print(f"csv rows parsed:{len(records)}")
    print(f"top_csv:        {args.top_csv}")
    print(f"matched unique: {len(candidate_rows)}")
    print(f"missing rows:   {len(missing)}")
    print(f"reorder:        {args.reorder}")
    print(f"reldn:          {args.reldn}")
    print(f"group_tiles:    {args.group_tiles} (0 means full segment)")
    print("========================================")

    for i, row in enumerate(candidate_rows):
        algo = int(row["algo"])
        key = inv_algo[algo]
        meta = key_to_record(key)

        print(
            f"[{i+1}/{len(candidate_rows)}] "
            f"algo={algo} "
            f"tile={meta['tile_m']}x{meta['tile_n']}x{meta['tile_k']} "
            f"cluster={meta['cluster']} "
            f"mainloop={meta['mainloop']} "
            f"cutlass={row['cutlass_runtime']:.6f}"
        )

        ms, max_abs_err, err = benchmark_algo(ext, algo, meta, args, device)

        out_item = {
            "algo": algo,
            "cutlass_runtime": row["cutlass_runtime"],
            "source_row": row["source_row"],
            **meta,
        }

        if err is not None:
            out_item["error"] = err
            failed.append(out_item)
            print(f"  FAILED: {err}")
            continue

        out_item["packed_signal_ms"] = float(ms)
        out_item["max_abs_err"] = max_abs_err
        results.append(out_item)

        if max_abs_err is None:
            print(f"  packed_signal_ms={ms:.6f}")
        else:
            print(f"  packed_signal_ms={ms:.6f} max_abs_err={max_abs_err}")

    results.sort(key=lambda x: x["packed_signal_ms"])
    top = results[: args.top_save]

    gpu_name = torch.cuda.get_device_properties(args.device).name
    gpu_slug = re.sub(r"[^a-zA-Z0-9]+", "_", gpu_name).strip("_").lower()

    out_path = Path(args.out).resolve() if args.out else (
        root / "configs" / f"m{args.m}n{args.n}k{args.k}_{gpu_slug}_sm90.json"
    )

    missing_path = Path(args.missing_out).resolve() if args.missing_out else (
        root / "configs" / f"m{args.m}n{args.n}k{args.k}_{gpu_slug}_sm90_missing.json"
    )

    failed_path = Path(args.failed_out).resolve() if args.failed_out else (
        root / "configs" / f"m{args.m}n{args.n}k{args.k}_{gpu_slug}_sm90_failed.json"
    )

    # FlashOverlap-compatible fields:
    # BM/BN/Algo/dur. Here BM=TileM and BN=TileN for our SM90 table.
    save_data = {
        "description": "ooverlap SM90 configs selected from CUTLASS CSV then benchmarked with packed reorder + signal GEMM",
        "M": args.m,
        "N": args.n,
        "K": args.k,
        "gpu": gpu_name,
        "csv": str(csv_path),
        "algo_dict": str(algo_dict_path),
        "reorder": args.reorder,
        "reldn": args.reldn,
        "group_tiles": args.group_tiles,
        "warmup": args.warmup,
        "iters": args.iters,

        "BM": [int(x["tile_m"]) for x in top],
        "BN": [int(x["tile_n"]) for x in top],
        "BK": [int(x["tile_k"]) for x in top],
        "Algo": [int(x["algo"]) for x in top],
        "dur": [float(x["packed_signal_ms"]) for x in top],

        "top": top,
        "all_profiled": results,
    }

    missing_data = {
        "description": "CUTLASS CSV candidates that did not exist in AlgoDictSm90",
        "M": args.m,
        "N": args.n,
        "K": args.k,
        "csv": str(csv_path),
        "algo_dict": str(algo_dict_path),
        "missing": missing,
        "skipped_rows_sample": skipped[:50],
        "skipped_rows_count": len(skipped),
    }

    failed_data = {
        "description": "Matched AlgoDictSm90 candidates that failed during ooverlap packed signal profiling",
        "M": args.m,
        "N": args.n,
        "K": args.k,
        "csv": str(csv_path),
        "algo_dict": str(algo_dict_path),
        "failed": failed,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(save_data, f, indent=2)

    with open(missing_path, "w", encoding="utf-8") as f:
        json.dump(missing_data, f, indent=2)

    with open(failed_path, "w", encoding="utf-8") as f:
        json.dump(failed_data, f, indent=2)

    print("")
    print("========================================")
    print("DONE")
    print(f"profiled ok: {len(results)}")
    print(f"failed:      {len(failed)}")
    print(f"missing:     {len(missing)}")
    print("")
    print(f"wrote:       {out_path}")
    print(f"wrote:       {missing_path}")
    print(f"wrote:       {failed_path}")
    print("")

    if top:
        print("Top selected configs:")
        for i, x in enumerate(top):
            print(
                f"  #{i+1}: "
                f"algo={x['algo']} "
                f"tile={x['tile_m']}x{x['tile_n']}x{x['tile_k']} "
                f"cluster={x['cluster']} "
                f"mainloop={x['mainloop']} "
                f"packed_signal_ms={x['packed_signal_ms']:.6f} "
                f"cutlass_runtime={x['cutlass_runtime']:.6f}"
            )
    else:
        print("No successful configs.")
    print("========================================")


if __name__ == "__main__":
    main()
