#!/usr/bin/env python3
"""
SM90/H100 gen_config script for ooverlap.

Pipeline:

  CUTLASS profiler CSV
    -> filter rows to match ooverlap's actual GEMM assumptions
    -> choose top CUTLASS rows by runtime
    -> map each row to AlgoDictSm90
    -> benchmark the actual ooverlap GEMM path
    -> save selected configs to JSON

Important modes:

  --layout normal
      Normal GEMM output layout.
      Use this when testing OOVERLAP_USE_BASE_EPILOGUE_ONLY.
      Defaults:
        reorder = identity
        reldn   = tile_cols
        D shape = [M, N]
        reset_mm = False

  --layout packed
      FlashOverlap-style packed tile output.
      Use this when testing reorder/signal epilogue.
      Defaults:
        reorder = column_major
        reldn   = 1
        D shape = [ceil(num_tiles / reldn) * TileM, reldn * TileN]
        reset_mm = True

CSV filters default to our current kernel assumptions:

  A layout: row
  B layout: column
  accumulator: f32
  split_k_slices: 1

You can relax those with:

  --no-filter-csv-layouts
  --accum any
"""

import argparse
import importlib.util
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import torch


# -----------------------------------------------------------------------------
# CSV aliases
# -----------------------------------------------------------------------------

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

STAGES_ALIASES = [
    "stages",
    "stage_count",
    "pipeline_stages",
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

A_ALIASES = [
    "a",
    "operand_a",
    "element_a",
    "layout_a",
]

B_ALIASES = [
    "b",
    "operand_b",
    "element_b",
    "layout_b",
]

C_ALIASES = [
    "c",
    "operand_c",
    "element_c",
    "layout_c",
]

D_ALIASES = [
    "d",
    "operand_d",
    "element_d",
    "layout_d",
]

ACCUM_ALIASES = [
    "accum",
    "accumulator",
    "accumulator_type",
    "element_accumulator",
]

SWIZZLE_ALIASES = [
    "swizzle",
    "swizzle_size",
    "threadblock_swizzle",
]

WARPS_M_ALIASES = [
    "warps_m",
    "warp_m",
    "warp_count_m",
]

WARPS_N_ALIASES = [
    "warps_n",
    "warp_n",
    "warp_count_n",
]

WARPS_K_ALIASES = [
    "warps_k",
    "warp_k",
    "warp_count_k",
]

INST_M_ALIASES = [
    "inst_m",
    "instruction_m",
    "mma_m",
    "math_inst_m",
]

INST_N_ALIASES = [
    "inst_n",
    "instruction_n",
    "mma_n",
    "math_inst_n",
]

INST_K_ALIASES = [
    "inst_k",
    "instruction_k",
    "mma_k",
    "math_inst_k",
]


# -----------------------------------------------------------------------------
# General helpers
# -----------------------------------------------------------------------------

def normalize_col(name: str) -> str:
    x = str(name).strip().lower()
    x = re.sub(r"[^a-z0-9]+", "_", x)
    return x.strip("_")


def build_colmap(columns) -> Dict[str, str]:
    out: Dict[str, str] = {}
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


def normalize_layout_value(x) -> Optional[str]:
    """
    Parse values like:
      f16:row
      f16:column
      row
      column
    """
    if x is None:
        return None

    s = str(x).strip().lower()
    if not s:
        return None

    if ":" in s:
        s = s.split(":")[-1].strip()

    if s in ("row", "row_major", "rowmajor", "n"):
        return "row"

    if s in ("column", "col", "column_major", "col_major", "columnmajor", "t"):
        return "column"

    if "row" in s:
        return "row"

    if "column" in s or "col" in s:
        return "column"

    return None


def normalize_dtype_value(x) -> Optional[str]:
    """
    Parse values like:
      f16
      f32
      f16:row
      void:column
    """
    if x is None:
        return None

    s = str(x).strip().lower()
    if not s:
        return None

    if ":" in s:
        s = s.split(":")[0].strip()

    if s in ("half", "fp16", "float16"):
        return "f16"

    if s in ("float", "fp32", "float32"):
        return "f32"

    if s in ("void", "none", "null"):
        return "void"

    return s


def normalize_stage_value(x) -> Optional[Any]:
    if x is None:
        return None

    if isinstance(x, str):
        s = x.strip().lower()
        if not s:
            return None
        if s in ("auto", "stagecountauto", "stage_count_auto"):
            return "auto"
        try:
            return int(float(s))
        except Exception:
            return s

    try:
        return int(x)
    except Exception:
        return x


def stage_to_key_value(x) -> Any:
    y = normalize_stage_value(x)
    if y is None:
        return "auto"
    return y


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

        "stages": find_col(colmap, STAGES_ALIASES),
        "split_k": find_col(colmap, SPLIT_K_ALIASES),
        "schedule": find_col(colmap, SCHEDULE_ALIASES),
        "epilogue": find_col(colmap, EPILOGUE_ALIASES),
        "operation": find_col(colmap, OPERATION_ALIASES),

        "a": find_col(colmap, A_ALIASES),
        "b": find_col(colmap, B_ALIASES),
        "c": find_col(colmap, C_ALIASES),
        "d": find_col(colmap, D_ALIASES),
        "accum": find_col(colmap, ACCUM_ALIASES),

        "swizzle": find_col(colmap, SWIZZLE_ALIASES),
        "warps_m": find_col(colmap, WARPS_M_ALIASES),
        "warps_n": find_col(colmap, WARPS_N_ALIASES),
        "warps_k": find_col(colmap, WARPS_K_ALIASES),
        "inst_m": find_col(colmap, INST_M_ALIASES),
        "inst_n": find_col(colmap, INST_N_ALIASES),
        "inst_k": find_col(colmap, INST_K_ALIASES),
    }


def infer_shape_from_row(row, cols, path: Path) -> Optional[Tuple[int, int, int]]:
    m = to_int(get_cell(row, cols["m"]))
    n = to_int(get_cell(row, cols["n"]))
    k = to_int(get_cell(row, cols["k"]))

    if m is not None and n is not None and k is not None:
        return m, n, k

    return filename_shape(path)


# -----------------------------------------------------------------------------
# Repo / extension loading
# -----------------------------------------------------------------------------

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


@dataclass
class AlgoIndex:
    by_key8: Dict[Tuple[Any, ...], int]
    by_key9: Dict[Tuple[Any, ...], int]
    inverse: Dict[int, Dict[str, Any]]
    has_stage_keys: bool


def make_key8(
    tile_m: int,
    tile_n: int,
    tile_k: int,
    cluster_m: int,
    cluster_n: int,
    cluster_k: int,
    mainloop: str,
    epilogue: str,
) -> Tuple[Any, ...]:
    return (
        int(tile_m),
        int(tile_n),
        int(tile_k),
        int(cluster_m),
        int(cluster_n),
        int(cluster_k),
        str(mainloop),
        str(epilogue),
    )


def make_key9(
    tile_m: int,
    tile_n: int,
    tile_k: int,
    cluster_m: int,
    cluster_n: int,
    cluster_k: int,
    stages: Any,
    mainloop: str,
    epilogue: str,
) -> Tuple[Any, ...]:
    return (
        int(tile_m),
        int(tile_n),
        int(tile_k),
        int(cluster_m),
        int(cluster_n),
        int(cluster_k),
        stage_to_key_value(stages),
        str(mainloop),
        str(epilogue),
    )


def item_to_algo_record(item: Dict[str, Any]) -> Dict[str, Any]:
    stage_value = (
        item.get("stages")
        if "stages" in item
        else item.get("stage_count", item.get("stage", None))
    )

    return {
        "algo": int(item["algo"]),
        "tile_m": int(item["tile_m"]),
        "tile_n": int(item["tile_n"]),
        "tile_k": int(item["tile_k"]),
        "cluster": [
            int(item["cluster"][0]),
            int(item["cluster"][1]),
            int(item["cluster"][2]),
        ],
        "stages": normalize_stage_value(stage_value),
        "mainloop": str(item["mainloop"]),
        "epilogue": str(item["epilogue"]),
    }


def load_algo_dict(path: Path) -> AlgoIndex:
    if not path.exists():
        raise FileNotFoundError(f"AlgoDictSm90 not found: {path}")

    by_key8: Dict[Tuple[Any, ...], int] = {}
    by_key9: Dict[Tuple[Any, ...], int] = {}
    inverse: Dict[int, Dict[str, Any]] = {}
    has_stage_keys = False

    if path.suffix == ".json":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        algorithms = data.get("algorithms", [])

        for item in algorithms:
            rec = item_to_algo_record(item)
            algo = int(rec["algo"])
            cm, cn, ck = rec["cluster"]

            key8 = make_key8(
                rec["tile_m"],
                rec["tile_n"],
                rec["tile_k"],
                cm,
                cn,
                ck,
                rec["mainloop"],
                rec["epilogue"],
            )

            by_key8.setdefault(key8, algo)

            if rec["stages"] is not None:
                has_stage_keys = True
                key9 = make_key9(
                    rec["tile_m"],
                    rec["tile_n"],
                    rec["tile_k"],
                    cm,
                    cn,
                    ck,
                    rec["stages"],
                    rec["mainloop"],
                    rec["epilogue"],
                )
                by_key9[key9] = algo

            inverse[algo] = rec

        return AlgoIndex(
            by_key8=by_key8,
            by_key9=by_key9,
            inverse=inverse,
            has_stage_keys=has_stage_keys,
        )

    if path.suffix == ".pt":
        raw = torch.load(path, map_location="cpu", weights_only=False)

        for key, value in raw.items():
            algo = int(value)
            key = tuple(key)

            if len(key) == 8:
                key8 = key
                by_key8[key8] = algo

                rec = {
                    "algo": algo,
                    "tile_m": int(key[0]),
                    "tile_n": int(key[1]),
                    "tile_k": int(key[2]),
                    "cluster": [int(key[3]), int(key[4]), int(key[5])],
                    "stages": None,
                    "mainloop": str(key[6]),
                    "epilogue": str(key[7]),
                }

            elif len(key) == 9:
                has_stage_keys = True
                key9 = (
                    int(key[0]),
                    int(key[1]),
                    int(key[2]),
                    int(key[3]),
                    int(key[4]),
                    int(key[5]),
                    stage_to_key_value(key[6]),
                    str(key[7]),
                    str(key[8]),
                )
                by_key9[key9] = algo

                key8 = make_key8(
                    key[0],
                    key[1],
                    key[2],
                    key[3],
                    key[4],
                    key[5],
                    key[7],
                    key[8],
                )
                by_key8.setdefault(key8, algo)

                rec = {
                    "algo": algo,
                    "tile_m": int(key[0]),
                    "tile_n": int(key[1]),
                    "tile_k": int(key[2]),
                    "cluster": [int(key[3]), int(key[4]), int(key[5])],
                    "stages": stage_to_key_value(key[6]),
                    "mainloop": str(key[7]),
                    "epilogue": str(key[8]),
                }

            else:
                continue

            inverse[algo] = rec

        return AlgoIndex(
            by_key8=by_key8,
            by_key9=by_key9,
            inverse=inverse,
            has_stage_keys=has_stage_keys,
        )

    raise ValueError(f"Unsupported AlgoDictSm90 format: {path}")


# -----------------------------------------------------------------------------
# CSV parsing / filtering
# -----------------------------------------------------------------------------

def find_csv(args, root: Path) -> Path:
    del root

    if args.csv is not None:
        p = Path(args.csv).expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(f"CSV not found: {p}")
        return p

    if args.csv_dir is None:
        raise ValueError("Pass either --csv or --csv-dir")

    csv_dir = Path(args.csv_dir).expanduser().resolve()

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


def csv_row_matches_layout_filters(row, cols, args) -> Tuple[bool, Optional[str]]:
    if not args.filter_csv_layouts:
        return True, None

    # A layout.
    if args.a_layout != "any" and cols["a"] is not None:
        got = normalize_layout_value(row[cols["a"]])
        if got is not None and got != args.a_layout:
            return False, f"A layout mismatch: got={got} want={args.a_layout}"

    # B layout.
    if args.b_layout != "any" and cols["b"] is not None:
        got = normalize_layout_value(row[cols["b"]])
        if got is not None and got != args.b_layout:
            return False, f"B layout mismatch: got={got} want={args.b_layout}"

    # Optional C filter. Default is any because CUTLASS CSV C is often source C,
    # and our epilogue can be C=void while D layout is implicit in the operation.
    if args.c_layout != "any" and cols["c"] is not None:
        got_dtype = normalize_dtype_value(row[cols["c"]])
        got_layout = normalize_layout_value(row[cols["c"]])

        if got_dtype == "void" and args.allow_void_c:
            pass
        elif got_layout is not None and got_layout != args.c_layout:
            return False, f"C layout mismatch: got={got_layout} want={args.c_layout}"

    # Optional D filter if the CSV has D explicitly.
    if args.d_layout != "any" and cols["d"] is not None:
        got = normalize_layout_value(row[cols["d"]])
        if got is not None and got != args.d_layout:
            return False, f"D layout mismatch: got={got} want={args.d_layout}"

    # Accumulator.
    if args.accum != "any" and cols["accum"] is not None:
        got = normalize_dtype_value(row[cols["accum"]])
        if got is not None and got != args.accum:
            return False, f"accum mismatch: got={got} want={args.accum}"

    # Optional operation regex.
    if args.operation_regex is not None and cols["operation"] is not None:
        op = str(row[cols["operation"]])
        if re.search(args.operation_regex, op) is None:
            return False, f"operation regex mismatch: {args.operation_regex}"

    return True, None


def parse_cutlass_csv(csv_path: Path, args) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    df = pd.read_csv(csv_path)
    cols = extract_columns(df)

    if cols["runtime"] is None:
        raise RuntimeError(
            f"No runtime column found in {csv_path}. Columns were: {list(df.columns)}"
        )

    records: List[Dict[str, Any]] = []
    skipped: List[Dict[str, Any]] = []

    filter_counts: Dict[str, int] = {
        "raw_rows": int(len(df)),
        "bad_runtime": 0,
        "failed_disposition": 0,
        "split_k": 0,
        "shape": 0,
        "tile_parse": 0,
        "csv_layout": 0,
        "kept": 0,
    }

    for idx, row in df.iterrows():
        runtime = to_float(get_cell(row, cols["runtime"]))
        if runtime is None:
            filter_counts["bad_runtime"] += 1
            skipped.append({"row": int(idx), "reason": "bad runtime"})
            continue

        if not passed_disposition(get_cell(row, cols["disposition"])):
            filter_counts["failed_disposition"] += 1
            continue

        split_k = to_int(get_cell(row, cols["split_k"]))
        if split_k is not None and split_k != 1 and not args.allow_split_k:
            filter_counts["split_k"] += 1
            continue

        shape = infer_shape_from_row(row, cols, csv_path)
        if shape is None:
            filter_counts["shape"] += 1
            skipped.append({"row": int(idx), "reason": "could not infer shape"})
            continue

        M, N, K = shape
        if M != args.m or N != args.n or K != args.k:
            filter_counts["shape"] += 1
            continue

        ok_layout, layout_reason = csv_row_matches_layout_filters(row, cols, args)
        if not ok_layout:
            filter_counts["csv_layout"] += 1
            if len(skipped) < 200:
                skipped.append({
                    "row": int(idx),
                    "reason": layout_reason,
                })
            continue

        tile_m = to_int(get_cell(row, cols["tile_m"]))
        tile_n = to_int(get_cell(row, cols["tile_n"]))
        tile_k = to_int(get_cell(row, cols["tile_k"]))

        if tile_m is None or tile_n is None or tile_k is None:
            filter_counts["tile_parse"] += 1
            skipped.append({
                "row": int(idx),
                "reason": "could not parse tile_m/tile_n/tile_k",
            })
            continue

        cluster_m = to_int(get_cell(row, cols["cluster_m"])) or 1
        cluster_n = to_int(get_cell(row, cols["cluster_n"])) or 1
        cluster_k = to_int(get_cell(row, cols["cluster_k"])) or 1

        stages = normalize_stage_value(get_cell(row, cols["stages"]))

        mainloop = map_mainloop(
            get_cell(row, cols["schedule"]),
            get_cell(row, cols["operation"]),
            args.default_mainloop,
        )

        epilogue = map_epilogue(
            get_cell(row, cols["epilogue"]),
            args.default_epilogue,
        )

        key8 = make_key8(
            tile_m,
            tile_n,
            tile_k,
            cluster_m,
            cluster_n,
            cluster_k,
            mainloop,
            epilogue,
        )

        key9 = make_key9(
            tile_m,
            tile_n,
            tile_k,
            cluster_m,
            cluster_n,
            cluster_k,
            stage_to_key_value(stages),
            mainloop,
            epilogue,
        )

        filter_counts["kept"] += 1

        records.append({
            "source_row": int(idx),
            "cutlass_runtime": float(runtime),

            "key8": key8,
            "key9": key9,

            "tile_m": int(tile_m),
            "tile_n": int(tile_n),
            "tile_k": int(tile_k),
            "cluster": [int(cluster_m), int(cluster_n), int(cluster_k)],
            "stages": stages,
            "mainloop": str(mainloop),
            "epilogue": str(epilogue),

            "split_k_slices": split_k,
            "a": str(get_cell(row, cols["a"])) if cols["a"] is not None else None,
            "b": str(get_cell(row, cols["b"])) if cols["b"] is not None else None,
            "c": str(get_cell(row, cols["c"])) if cols["c"] is not None else None,
            "d": str(get_cell(row, cols["d"])) if cols["d"] is not None else None,
            "accum": str(get_cell(row, cols["accum"])) if cols["accum"] is not None else None,

            "operation": str(get_cell(row, cols["operation"])) if cols["operation"] is not None else None,

            "swizzle_size": to_int(get_cell(row, cols["swizzle"])),
            "warps_m": to_int(get_cell(row, cols["warps_m"])),
            "warps_n": to_int(get_cell(row, cols["warps_n"])),
            "warps_k": to_int(get_cell(row, cols["warps_k"])),
            "inst_m": to_int(get_cell(row, cols["inst_m"])),
            "inst_n": to_int(get_cell(row, cols["inst_n"])),
            "inst_k": to_int(get_cell(row, cols["inst_k"])),
        })

    records.sort(key=lambda x: x["cutlass_runtime"])

    diagnostics = {
        "columns": {k: v for k, v in cols.items()},
        "filter_counts": filter_counts,
    }

    return records, skipped, diagnostics


def dedupe_by_match_key_keep_best(records: List[Dict[str, Any]], match_stages: bool) -> List[Dict[str, Any]]:
    best: Dict[Tuple[Any, ...], Dict[str, Any]] = {}

    key_name = "key9" if match_stages else "key8"

    for r in records:
        key = tuple(r[key_name])
        if key not in best or r["cutlass_runtime"] < best[key]["cutlass_runtime"]:
            best[key] = r

    out = list(best.values())
    out.sort(key=lambda x: x["cutlass_runtime"])
    return out


def resolve_match_stages(args, algo_index: AlgoIndex) -> bool:
    if args.match_stages == "exact":
        return True

    if args.match_stages == "ignore":
        return False

    # auto
    return bool(algo_index.has_stage_keys)


def match_algo(row: Dict[str, Any], algo_index: AlgoIndex, match_stages: bool) -> Tuple[Optional[int], str]:
    if match_stages:
        algo = algo_index.by_key9.get(tuple(row["key9"]))
        if algo is not None:
            return int(algo), "key9_exact_stage"

        return None, "missing_key9_exact_stage"

    algo = algo_index.by_key8.get(tuple(row["key8"]))
    if algo is not None:
        return int(algo), "key8_no_stage"

    return None, "missing_key8_no_stage"


# -----------------------------------------------------------------------------
# Reorder / segment helpers
# -----------------------------------------------------------------------------

def make_column_major_reorder(
    M: int,
    N: int,
    tile_m: int,
    tile_n: int,
    device,
) -> torch.Tensor:
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


def make_identity_reorder(
    M: int,
    N: int,
    tile_m: int,
    tile_n: int,
    device,
) -> torch.Tensor:
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


def resolve_layout(
    args,
    M: int,
    N: int,
    tile_m: int,
    tile_n: int,
) -> Tuple[str, int, Tuple[int, int], str]:
    tile_rows = M // tile_m
    tile_cols = N // tile_n
    num_tiles = tile_rows * tile_cols

    if args.layout == "normal":
        effective_reorder = "identity" if args.reorder is None else args.reorder
        if effective_reorder != "identity":
            raise ValueError("layout=normal requires reorder=identity")

        effective_reldn = tile_cols if args.reldn is None else int(args.reldn)
        if effective_reldn != tile_cols:
            raise ValueError(
                f"layout=normal requires reldn=tile_cols={tile_cols}, got {effective_reldn}"
            )

        output_shape = (M, N)
        metric_name = "normal_gemm_ms"
        return effective_reorder, effective_reldn, output_shape, metric_name

    if args.layout == "packed":
        effective_reorder = "column_major" if args.reorder is None else args.reorder
        effective_reldn = 1 if args.reldn is None else int(args.reldn)

        if effective_reldn <= 0:
            raise ValueError("reldn must be > 0")

        packed_tile_cols = effective_reldn
        packed_tile_rows = (num_tiles + packed_tile_cols - 1) // packed_tile_cols

        output_shape = (packed_tile_rows * tile_m, packed_tile_cols * tile_n)
        metric_name = "packed_signal_ms"
        return effective_reorder, effective_reldn, output_shape, metric_name

    raise ValueError(f"unknown layout={args.layout}")


# -----------------------------------------------------------------------------
# Benchmarking
# -----------------------------------------------------------------------------

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
) -> Tuple[Optional[float], Optional[float], Optional[str], Dict[str, Any]]:
    M, N, K = args.m, args.n, args.k

    tile_m = int(meta["tile_m"])
    tile_n = int(meta["tile_n"])

    if M % tile_m != 0 or N % tile_n != 0:
        return None, None, f"M/N not divisible by tile {tile_m}x{tile_n}", {}

    tile_rows = M // tile_m
    tile_cols = N // tile_n
    num_tiles = tile_rows * tile_cols

    try:
        effective_reorder, effective_reldn, output_shape, metric_name = resolve_layout(
            args,
            M,
            N,
            tile_m,
            tile_n,
        )
    except Exception as e:
        return None, None, str(e), {}

    if effective_reorder == "column_major":
        RA = make_column_major_reorder(M, N, tile_m, tile_n, device)
    elif effective_reorder == "identity":
        RA = make_identity_reorder(M, N, tile_m, tile_n, device)
    else:
        return None, None, f"unknown reorder={effective_reorder}", {}

    segments = make_segments(num_tiles, args.group_tiles)

    CommThr = torch.tensor(segments, device=device, dtype=torch.int32)
    MM = torch.empty((len(segments) + num_tiles,), device=device, dtype=torch.int32)

    A = torch.randn((M, K), device=device, dtype=torch.float16)

    # Torch sees B as [K, N].
    B_ref = torch.randn((K, N), device=device, dtype=torch.float16)

    # Our CUTLASS wrapper expects B physically as [N, K].
    B_packed = B_ref.t().contiguous()

    D = torch.empty(output_shape, device=device, dtype=torch.float16)

    monitor = False

    effective_reset_mm = args.reset_mm
    if effective_reset_mm is None:
        effective_reset_mm = args.layout == "packed"

    def run():
        if effective_reset_mm:
            MM.zero_()

        ext.gemm_signal_sm90(
            A,
            B_packed,
            D,
            MM,
            RA,
            CommThr,
            int(effective_reldn),
            int(algo),
            monitor,
        )

    # Make non-reset mode deterministic before timing.
    if not effective_reset_mm:
        MM.zero_()
        torch.cuda.synchronize()

    try:
        ms = time_cuda(run, args.warmup, args.iters)
    except Exception as e:
        torch.cuda.synchronize()
        return None, None, f"benchmark failed: {type(e).__name__}: {e}", {}

    max_abs_err = None

    if args.check:
        try:
            MM.zero_()
            ext.gemm_signal_sm90(
                A,
                B_packed,
                D,
                MM,
                RA,
                CommThr,
                int(effective_reldn),
                int(algo),
                monitor,
            )
            torch.cuda.synchronize()

            if args.layout == "normal":
                D_normal = D
            else:
                D_normal = unpack_packed_to_normal(
                    D,
                    RA,
                    M,
                    N,
                    tile_m,
                    tile_n,
                    effective_reldn,
                )

            ref = A @ B_ref
            max_abs_err = float((D_normal - ref).abs().max().item())
        except Exception as e:
            return ms, None, f"check failed: {type(e).__name__}: {e}", {}

    bench_info = {
        "layout": args.layout,
        "reorder": effective_reorder,
        "reldn": int(effective_reldn),
        "output_shape": [int(output_shape[0]), int(output_shape[1])],
        "num_tiles": int(num_tiles),
        "tile_rows": int(tile_rows),
        "tile_cols": int(tile_cols),
        "num_segments": int(len(segments)),
        "segments": [int(x) for x in segments],
        "reset_mm": bool(effective_reset_mm),
        "metric_name": metric_name,
    }

    return ms, max_abs_err, None, bench_info


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

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

    ap.add_argument(
        "--layout",
        choices=["packed", "normal"],
        default="packed",
        help=(
            "packed: FlashOverlap-style packed tile output. "
            "normal: normal [M,N] output, useful for base epilogue comparison."
        ),
    )

    ap.add_argument(
        "--reldn",
        type=int,
        default=None,
        help=(
            "Packed tile columns. Default: 1 for layout=packed, "
            "tile_cols for layout=normal."
        ),
    )

    ap.add_argument(
        "--group-tiles",
        type=int,
        default=0,
        help="0 means one full segment. Otherwise segment by this many tiles.",
    )

    ap.add_argument(
        "--reorder",
        choices=["column_major", "identity"],
        default=None,
        help=(
            "Default: column_major for layout=packed, identity for layout=normal."
        ),
    )

    ap.add_argument(
        "--reset-mm",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Whether to zero MM before each measured GEMM. "
            "Default: True for layout=packed, False for layout=normal."
        ),
    )

    ap.add_argument(
        "--filter-csv-layouts",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Filter CSV rows by A/B/C/D/accum assumptions. "
            "Default True."
        ),
    )

    ap.add_argument(
        "--a-layout",
        choices=["row", "column", "any"],
        default="row",
        help="Expected CSV A layout. Default row.",
    )

    ap.add_argument(
        "--b-layout",
        choices=["row", "column", "any"],
        default="column",
        help="Expected CSV B layout. Default column.",
    )

    ap.add_argument(
        "--c-layout",
        choices=["row", "column", "any"],
        default="any",
        help="Optional CSV C/source layout filter. Default any.",
    )

    ap.add_argument(
        "--d-layout",
        choices=["row", "column", "any"],
        default="any",
        help="Optional CSV D/output layout filter if D column exists. Default any.",
    )

    ap.add_argument(
        "--allow-void-c",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Allow CSV C operand to be void when filtering C layout.",
    )

    ap.add_argument(
        "--accum",
        choices=["f16", "f32", "any"],
        default="f32",
        help="Expected accumulator type in CSV. Default f32.",
    )

    ap.add_argument(
        "--operation-regex",
        type=str,
        default=None,
        help="Optional regex filter on CUTLASS Operation/procedural name.",
    )

    ap.add_argument(
        "--match-stages",
        choices=["auto", "exact", "ignore"],
        default="auto",
        help=(
            "auto: exact stage matching only if AlgoDict contains stages. "
            "exact: include CSV stages in key. "
            "ignore: use old no-stage key."
        ),
    )

    ap.add_argument(
        "--default-mainloop",
        choices=["ws", "pingpong", "cooperative"],
        default="ws",
    )

    ap.add_argument(
        "--default-epilogue",
        choices=["auto"],
        default="auto",
    )

    ap.add_argument("--allow-split-k", action="store_true")
    ap.add_argument("--check", action="store_true")

    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--missing-out", type=str, default=None)
    ap.add_argument("--failed-out", type=str, default=None)

    args = ap.parse_args()

    root = Path(args.root).expanduser().resolve() if args.root else repo_root_from_script()

    algo_dict_path = (
        Path(args.algo_dict).expanduser().resolve()
        if args.algo_dict
        else root / "configs" / "AlgoDictSm90.json"
    )

    algo_index = load_algo_dict(algo_dict_path)
    match_stages = resolve_match_stages(args, algo_index)

    csv_path = find_csv(args, root)

    records, skipped, csv_diag = parse_cutlass_csv(csv_path, args)

    records = dedupe_by_match_key_keep_best(records, match_stages)

    selected_csv = records[: args.top_csv]

    matched = []
    missing = []

    for r in selected_csv:
        algo, match_kind = match_algo(r, algo_index, match_stages)

        item = dict(r)
        item["key8"] = list(r["key8"])
        item["key9"] = list(r["key9"])
        item["match_kind"] = match_kind

        if algo is None:
            item["matched"] = False
            item["algo"] = None
            missing.append(item)
        else:
            item["matched"] = True
            item["algo"] = int(algo)
            matched.append(item)

    by_algo: Dict[int, Dict[str, Any]] = {}

    for r in matched:
        algo = int(r["algo"])
        if algo not in by_algo or r["cutlass_runtime"] < by_algo[algo]["cutlass_runtime"]:
            by_algo[algo] = r

    candidate_rows = sorted(by_algo.values(), key=lambda x: x["cutlass_runtime"])

    torch.cuda.set_device(args.device)
    device = torch.device(f"cuda:{args.device}")

    ext = load_ooverlap_ext(root)

    gpu_name = torch.cuda.get_device_properties(args.device).name
    gpu_slug = re.sub(r"[^a-zA-Z0-9]+", "_", gpu_name).strip("_").lower()

    layout_suffix = f"{args.layout}_sm90"

    out_path = Path(args.out).expanduser().resolve() if args.out else (
        root / "configs" / f"m{args.m}n{args.n}k{args.k}_{gpu_slug}_{layout_suffix}.json"
    )

    missing_path = Path(args.missing_out).expanduser().resolve() if args.missing_out else (
        root / "configs" / f"m{args.m}n{args.n}k{args.k}_{gpu_slug}_{layout_suffix}_missing.json"
    )

    failed_path = Path(args.failed_out).expanduser().resolve() if args.failed_out else (
        root / "configs" / f"m{args.m}n{args.n}k{args.k}_{gpu_slug}_{layout_suffix}_failed.json"
    )

    if not candidate_rows:
        missing_path.parent.mkdir(parents=True, exist_ok=True)

        with open(missing_path, "w", encoding="utf-8") as f:
            json.dump({
                "description": "CUTLASS CSV candidates that did not exist in AlgoDictSm90",
                "M": args.m,
                "N": args.n,
                "K": args.k,
                "layout": args.layout,
                "csv": str(csv_path),
                "algo_dict": str(algo_dict_path),
                "match_stages": match_stages,
                "csv_diagnostics": csv_diag,
                "missing": missing,
                "skipped_rows_sample": skipped[:200],
                "skipped_rows_count": len(skipped),
            }, f, indent=2)

        raise RuntimeError(
            f"No CSV candidates matched AlgoDictSm90. Wrote diagnostics to {missing_path}"
        )

    results = []
    failed = []

    print("========================================")
    print("gen_config_sm90")
    print(f"shape:          M={args.m} N={args.n} K={args.k}")
    print(f"layout:         {args.layout}")
    print(f"csv:            {csv_path}")
    print(f"algo_dict:      {algo_dict_path}")
    print(f"csv raw rows:   {csv_diag['filter_counts']['raw_rows']}")
    print(f"csv rows kept:  {csv_diag['filter_counts']['kept']}")
    print(f"top_csv:        {args.top_csv}")
    print(f"matched unique: {len(candidate_rows)}")
    print(f"missing rows:   {len(missing)}")
    print(f"match_stages:   {match_stages} ({args.match_stages})")
    print(f"filter_layouts: {args.filter_csv_layouts}")
    print(f"A/B/C/D layout: {args.a_layout}/{args.b_layout}/{args.c_layout}/{args.d_layout}")
    print(f"accum:          {args.accum}")
    print(f"reorder:        {args.reorder if args.reorder is not None else 'auto'}")
    print(f"reldn:          {args.reldn if args.reldn is not None else 'auto'}")
    print(f"group_tiles:    {args.group_tiles} (0 means full segment)")
    print(f"reset_mm:       {args.reset_mm if args.reset_mm is not None else 'auto'}")
    print("")
    print("CSV filter counts:")
    for k, v in csv_diag["filter_counts"].items():
        print(f"  {k:22s}: {v}")
    print("========================================")

    metric_name_seen = None

    for i, row in enumerate(candidate_rows):
        algo = int(row["algo"])
        meta = algo_index.inverse[algo]

        print(
            f"[{i + 1}/{len(candidate_rows)}] "
            f"algo={algo} "
            f"tile={meta['tile_m']}x{meta['tile_n']}x{meta['tile_k']} "
            f"cluster={meta['cluster']} "
            f"stages={meta.get('stages')} "
            f"mainloop={meta['mainloop']} "
            f"cutlass={row['cutlass_runtime']:.6f} "
            f"match={row['match_kind']}"
        )

        ms, max_abs_err, err, bench_info = benchmark_algo(
            ext,
            algo,
            meta,
            args,
            device,
        )

        out_item = {
            "algo": algo,
            "cutlass_runtime": row["cutlass_runtime"],
            "source_row": row["source_row"],

            "tile_m": int(meta["tile_m"]),
            "tile_n": int(meta["tile_n"]),
            "tile_k": int(meta["tile_k"]),
            "cluster": [int(x) for x in meta["cluster"]],
            "stages": meta.get("stages"),
            "mainloop": str(meta["mainloop"]),
            "epilogue": str(meta["epilogue"]),

            "csv_stages": row.get("stages"),
            "csv_a": row.get("a"),
            "csv_b": row.get("b"),
            "csv_c": row.get("c"),
            "csv_d": row.get("d"),
            "csv_accum": row.get("accum"),
            "csv_operation": row.get("operation"),
            "csv_swizzle_size": row.get("swizzle_size"),
            "csv_warps_m": row.get("warps_m"),
            "csv_warps_n": row.get("warps_n"),
            "csv_warps_k": row.get("warps_k"),
            "csv_inst_m": row.get("inst_m"),
            "csv_inst_n": row.get("inst_n"),
            "csv_inst_k": row.get("inst_k"),
            "match_kind": row.get("match_kind"),
        }

        if bench_info:
            out_item.update(bench_info)
            metric_name_seen = bench_info["metric_name"]

        if err is not None:
            out_item["error"] = err
            failed.append(out_item)
            print(f"  FAILED: {err}")
            continue

        metric_name = bench_info["metric_name"]
        out_item["measured_ms"] = float(ms)
        out_item[metric_name] = float(ms)
        out_item["max_abs_err"] = max_abs_err

        results.append(out_item)

        if max_abs_err is None:
            print(f"  {metric_name}={ms:.6f}")
        else:
            print(f"  {metric_name}={ms:.6f} max_abs_err={max_abs_err}")

    results.sort(key=lambda x: x["measured_ms"])
    top = results[: args.top_save]

    effective_metric = metric_name_seen or (
        "normal_gemm_ms" if args.layout == "normal" else "packed_signal_ms"
    )

    save_data = {
        "description": (
            "ooverlap SM90 configs selected from CUTLASS CSV then benchmarked "
            f"with layout={args.layout}"
        ),
        "M": args.m,
        "N": args.n,
        "K": args.k,
        "gpu": gpu_name,
        "csv": str(csv_path),
        "algo_dict": str(algo_dict_path),
        "layout": args.layout,
        "reorder_arg": args.reorder,
        "reldn_arg": args.reldn,
        "group_tiles": args.group_tiles,
        "reset_mm_arg": args.reset_mm,
        "metric": effective_metric,
        "warmup": args.warmup,
        "iters": args.iters,

        "match_stages": match_stages,
        "filter_csv_layouts": args.filter_csv_layouts,
        "a_layout": args.a_layout,
        "b_layout": args.b_layout,
        "c_layout": args.c_layout,
        "d_layout": args.d_layout,
        "accum": args.accum,
        "operation_regex": args.operation_regex,

        "csv_diagnostics": csv_diag,

        # FlashOverlap-compatible fields.
        "BM": [int(x["tile_m"]) for x in top],
        "BN": [int(x["tile_n"]) for x in top],
        "BK": [int(x["tile_k"]) for x in top],
        "Algo": [int(x["algo"]) for x in top],
        "dur": [float(x["measured_ms"]) for x in top],

        "top": top,
        "all_profiled": results,
    }

    missing_data = {
        "description": "CUTLASS CSV candidates that did not exist in AlgoDictSm90",
        "M": args.m,
        "N": args.n,
        "K": args.k,
        "layout": args.layout,
        "csv": str(csv_path),
        "algo_dict": str(algo_dict_path),
        "match_stages": match_stages,
        "filter_csv_layouts": args.filter_csv_layouts,
        "csv_diagnostics": csv_diag,
        "missing": missing,
        "skipped_rows_sample": skipped[:200],
        "skipped_rows_count": len(skipped),
    }

    failed_data = {
        "description": "Matched AlgoDictSm90 candidates that failed during ooverlap profiling",
        "M": args.m,
        "N": args.n,
        "K": args.k,
        "layout": args.layout,
        "csv": str(csv_path),
        "algo_dict": str(algo_dict_path),
        "match_stages": match_stages,
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

    if top:
        print("")
        print("Top selected configs:")
        for i, x in enumerate(top):
            print(
                f"  #{i + 1}: "
                f"algo={x['algo']} "
                f"tile={x['tile_m']}x{x['tile_n']}x{x['tile_k']} "
                f"cluster={x['cluster']} "
                f"stages={x.get('stages')} "
                f"mainloop={x['mainloop']} "
                f"{effective_metric}={x['measured_ms']:.6f} "
                f"cutlass_runtime={x['cutlass_runtime']:.6f} "
                f"match={x.get('match_kind')}"
            )
    else:
        print("")
        print("No successful configs.")

    print("========================================")


if __name__ == "__main__":
    main()
