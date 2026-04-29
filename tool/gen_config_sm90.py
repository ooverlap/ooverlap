#!/usr/bin/env python3
"""
Profile ooverlap SM90 GEMM-with-signal candidates against CUTLASS profiler CSV.

This version expects the new 9-field AlgoDictSm90 key:

  TileM, TileN, TileK,
  ClusterM, ClusterN, ClusterK,
  Stages,
  Mainloop,
  Epilogue

It can still run in --match-stages ignore mode, but exact matching is what we
need now.
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
WARPS_M_NAMES = ["warps_m"]
WARPS_N_NAMES = ["warps_n"]
WARPS_K_NAMES = ["warps_k"]
INST_M_NAMES = ["inst_m"]
INST_N_NAMES = ["inst_n"]
INST_K_NAMES = ["inst_k"]


def norm_col(x: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(x).strip().lower()).strip("_")


def col(df: pd.DataFrame, names: List[str]) -> Optional[str]:
    cmap = {norm_col(c): c for c in df.columns}
    for n in names:
        k = norm_col(n)
        if k in cmap:
            return cmap[k]
    return None


def as_int(x, default=None):
    try:
        if x is None:
            return default
        if isinstance(x, float) and math.isnan(x):
            return default
        s = str(x).strip()
        if not s:
            return default
        return int(float(s))
    except Exception:
        return default


def as_float(x, default=None):
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


def norm_layout(x) -> Optional[str]:
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


def norm_dtype(x) -> Optional[str]:
    if x is None:
        return None
    s = str(x).strip().lower()
    if ":" in s:
        s = s.split(":")[0]
    if s in ("half", "fp16", "float16"):
        return "f16"
    if s in ("float", "fp32", "float32"):
        return "f32"
    if s in ("void", "none"):
        return "void"
    return s or None


def parse_op_dtypes(op: str) -> Dict[str, Optional[str]]:
    """
    CUTLASS names usually contain:
      gemm_A_B_ACCUM_C_D_
    e.g.
      ...gemm_f16_f16_f32_void_f16_128x128x64...
      ...gemm_f16_f16_f32_f32_f32_128x128x64...
    """
    s = str(op)
    m = re.search(r"gemm_([^_]+)_([^_]+)_([^_]+)_([^_]+)_([^_]+)_", s)
    if not m:
        return {"a": None, "b": None, "accum": None, "c": None, "d": None}
    return {
        "a": m.group(1).lower(),
        "b": m.group(2).lower(),
        "accum": m.group(3).lower(),
        "c": m.group(4).lower(),
        "d": m.group(5).lower(),
    }


def map_mainloop(op: str, default: str) -> str:
    s = str(op).lower().replace("-", "_")
    if "pingpong" in s or "ping_pong" in s:
        return "pingpong"
    if "cooperative" in s or "coop" in s:
        return "cooperative"
    if "warpspecialized" in s or "warp_specialized" in s or "tma" in s:
        return default
    return default


def make_key8(tm, tn, tk, cm, cn, ck, mainloop, epilogue):
    return (int(tm), int(tn), int(tk), int(cm), int(cn), int(ck), str(mainloop), str(epilogue))


def make_key9(tm, tn, tk, cm, cn, ck, stages, mainloop, epilogue):
    return (int(tm), int(tn), int(tk), int(cm), int(cn), int(ck), int(stages), str(mainloop), str(epilogue))


def filename_shape(path: Path):
    m = re.search(r"m(\d+)n(\d+)k(\d+)", path.name, re.IGNORECASE)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2)), int(m.group(3))


def load_ext(root: Path):
    so = root / "build" / "lib" / "ooverlap_ext.so"
    if not so.exists():
        raise FileNotFoundError(f"Could not find {so}. Build first.")
    spec = importlib.util.spec_from_file_location("ooverlap_ext", str(so))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_algo_dict(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    by8 = {}
    by9 = {}
    inverse = {}

    for item in data.get("algorithms", []):
        algo = int(item["algo"])
        tm = int(item["tile_m"])
        tn = int(item["tile_n"])
        tk = int(item["tile_k"])
        cm, cn, ck = [int(x) for x in item["cluster"]]
        stages = item.get("stages", None)
        mainloop = str(item["mainloop"])
        epilogue = str(item.get("epilogue", "auto"))

        k8 = make_key8(tm, tn, tk, cm, cn, ck, mainloop, epilogue)
        by8.setdefault(k8, algo)

        if stages is not None and str(stages).lower() != "auto":
            k9 = make_key9(tm, tn, tk, cm, cn, ck, int(stages), mainloop, epilogue)
            by9[k9] = algo

        inverse[algo] = {
            "algo": algo,
            "tile_m": tm,
            "tile_n": tn,
            "tile_k": tk,
            "cluster": [cm, cn, ck],
            "stages": stages,
            "mainloop": mainloop,
            "epilogue": epilogue,
        }

    return by8, by9, inverse


def parse_csv(args, csv_path: Path):
    df = pd.read_csv(csv_path)

    c_runtime = col(df, RUNTIME_NAMES)
    c_op = col(df, OP_NAMES)
    c_a = col(df, A_NAMES)
    c_b = col(df, B_NAMES)
    c_c = col(df, C_NAMES)
    c_d = col(df, D_NAMES)
    c_accum = col(df, ACCUM_NAMES)
    c_split = col(df, SPLIT_K_NAMES)
    c_tm = col(df, CTA_M_NAMES)
    c_tn = col(df, CTA_N_NAMES)
    c_tk = col(df, CTA_K_NAMES)
    c_cm = col(df, CLUSTER_M_NAMES)
    c_cn = col(df, CLUSTER_N_NAMES)
    c_ck = col(df, CLUSTER_K_NAMES)
    c_stages = col(df, STAGES_NAMES)
    c_wm = col(df, WARPS_M_NAMES)
    c_wn = col(df, WARPS_N_NAMES)
    c_wk = col(df, WARPS_K_NAMES)
    c_im = col(df, INST_M_NAMES)
    c_in = col(df, INST_N_NAMES)
    c_ik = col(df, INST_K_NAMES)

    if c_runtime is None:
        raise RuntimeError(f"No runtime column found. CSV columns: {list(df.columns)}")

    shape_from_name = filename_shape(csv_path)

    records = []
    counts = {
        "raw_rows": int(len(df)),
        "bad_runtime": 0,
        "split_k": 0,
        "layout_or_dtype": 0,
        "tile_parse": 0,
        "kept": 0,
    }

    for i, r in df.iterrows():
        runtime = as_float(r[c_runtime])
        if runtime is None:
            counts["bad_runtime"] += 1
            continue

        split_k = as_int(r[c_split], 1) if c_split is not None else 1
        if split_k != 1 and not args.allow_split_k:
            counts["split_k"] += 1
            continue

        # Shape columns are not always present in CUTLASS profiler CSV.
        if shape_from_name is not None:
            M, N, K = shape_from_name
            if (M, N, K) != (args.m, args.n, args.k):
                continue

        tm = as_int(r[c_tm]) if c_tm is not None else None
        tn = as_int(r[c_tn]) if c_tn is not None else None
        tk = as_int(r[c_tk]) if c_tk is not None else None
        if tm is None or tn is None or tk is None:
            counts["tile_parse"] += 1
            continue

        cm = as_int(r[c_cm], 1) if c_cm is not None else 1
        cn = as_int(r[c_cn], 1) if c_cn is not None else 1
        ck = as_int(r[c_ck], 1) if c_ck is not None else 1
        stages = as_int(r[c_stages]) if c_stages is not None else None
        if stages is None:
            counts["tile_parse"] += 1
            continue

        op = str(r[c_op]) if c_op is not None else ""
        op_dt = parse_op_dtypes(op)

        # Our wrapper is A=f16 row, B=f16 column, accum=f32, D=f16.
        a_layout = norm_layout(r[c_a]) if c_a is not None else None
        b_layout = norm_layout(r[c_b]) if c_b is not None else None
        a_dtype = norm_dtype(r[c_a]) if c_a is not None else op_dt["a"]
        b_dtype = norm_dtype(r[c_b]) if c_b is not None else op_dt["b"]
        c_dtype = norm_dtype(r[c_c]) if c_c is not None else op_dt["c"]
        d_dtype = norm_dtype(r[c_d]) if c_d is not None else op_dt["d"]
        accum = norm_dtype(r[c_accum]) if c_accum is not None else op_dt["accum"]

        ok = True
        if args.filter_layouts:
            if a_layout is not None and a_layout != "row":
                ok = False
            if b_layout is not None and b_layout != "column":
                ok = False

        if a_dtype is not None and a_dtype != "f16":
            ok = False
        if b_dtype is not None and b_dtype != "f16":
            ok = False
        if accum is not None and accum != "f32":
            ok = False

        # Critical fix: reject profiler rows whose output D is f32.
        # Our kernel instantiates ElementOutput=cutlass::half_t.
        if d_dtype is not None and d_dtype != "f16":
            ok = False

        # C can be void or f16 because beta=0 and C is effectively unused.
        if c_dtype is not None and c_dtype not in ("void", "f16"):
            ok = False

        if not ok:
            counts["layout_or_dtype"] += 1
            continue

        mainloop = map_mainloop(op, args.default_mainloop)
        epilogue = "auto"

        key8 = make_key8(tm, tn, tk, cm, cn, ck, mainloop, epilogue)
        key9 = make_key9(tm, tn, tk, cm, cn, ck, stages, mainloop, epilogue)

        records.append({
            "source_row": int(i),
            "cutlass_runtime": float(runtime),
            "key8": list(key8),
            "key9": list(key9),
            "tile_m": tm,
            "tile_n": tn,
            "tile_k": tk,
            "cluster": [cm, cn, ck],
            "stages": stages,
            "mainloop": mainloop,
            "epilogue": epilogue,
            "a": str(r[c_a]) if c_a is not None else None,
            "b": str(r[c_b]) if c_b is not None else None,
            "c": str(r[c_c]) if c_c is not None else None,
            "d": str(r[c_d]) if c_d is not None else None,
            "accum": str(r[c_accum]) if c_accum is not None else None,
            "operation": op,
            "csv_warps_m": as_int(r[c_wm]) if c_wm is not None else None,
            "csv_warps_n": as_int(r[c_wn]) if c_wn is not None else None,
            "csv_warps_k": as_int(r[c_wk]) if c_wk is not None else None,
            "csv_inst_m": as_int(r[c_im]) if c_im is not None else None,
            "csv_inst_n": as_int(r[c_in]) if c_in is not None else None,
            "csv_inst_k": as_int(r[c_ik]) if c_ik is not None else None,
        })
        counts["kept"] += 1

    records.sort(key=lambda x: x["cutlass_runtime"])
    return records, counts


def match_algo(row, by8, by9, match_stages):
    k8 = tuple(row["key8"])
    k9 = tuple(row["key9"])
    if match_stages in ("exact", "auto"):
        if k9 in by9:
            return by9[k9], "key9_exact"
        if match_stages == "exact":
            return None, "missing_key9"
    if k8 in by8:
        return by8[k8], "key8_no_stage"
    return None, "missing_key8"


def dedupe_matched(rows):
    out = {}
    for r in rows:
        if r["algo"] not in out or r["cutlass_runtime"] < out[r["algo"]]["cutlass_runtime"]:
            out[r["algo"]] = r
    vals = list(out.values())
    vals.sort(key=lambda x: x["cutlass_runtime"])
    return vals


def make_ra(tile_rows, tile_cols, reorder, device):
    num_tiles = tile_rows * tile_cols
    if reorder == "identity":
        return torch.arange(num_tiles, device=device, dtype=torch.int32)

    if reorder == "column_major":
        vals = []
        for tm in range(tile_rows):
            for tn in range(tile_cols):
                logical = tm * tile_cols + tn
                packed = tn * tile_rows + tm
                vals.append((logical, packed))
        ra = torch.empty((num_tiles,), device=device, dtype=torch.int32)
        for logical, packed in vals:
            ra[logical] = packed
        return ra

    raise ValueError(f"unknown reorder={reorder}")


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


def benchmark_algo(ext, args, meta):
    torch.cuda.set_device(args.device)
    device = torch.device("cuda", args.device)

    M, N, K = args.m, args.n, args.k
    tm, tn = int(meta["tile_m"]), int(meta["tile_n"])
    assert M % tm == 0
    assert N % tn == 0

    tile_rows = M // tm
    tile_cols = N // tn
    num_tiles = tile_rows * tile_cols

    if args.layout == "normal":
        reorder = "identity" if args.reorder == "auto" else args.reorder
        reldn = tile_cols if args.reldn == 0 else args.reldn
        out_m = M
        out_n = N
    else:
        reorder = "column_major" if args.reorder == "auto" else args.reorder
        reldn = 1 if args.reldn == 0 else args.reldn
        packed_rows = (num_tiles + reldn - 1) // reldn
        out_m = packed_rows * tm
        out_n = reldn * tn

    A = torch.randn((M, K), device=device, dtype=torch.float16)
    B_ref = torch.randn((K, N), device=device, dtype=torch.float16)
    B_packed = B_ref.t().contiguous()

    D = torch.empty((out_m, out_n), device=device, dtype=torch.float16)

    RA = make_ra(tile_rows, tile_cols, reorder, device)
    CommThr = torch.tensor([num_tiles], device=device, dtype=torch.int32)
    MM = torch.zeros((1 + num_tiles,), device=device, dtype=torch.int32)

    def run():
        ext.gemm_signal_sm90(
            A,
            B_packed,
            D,
            MM,
            RA,
            CommThr,
            int(reldn),
            int(meta["algo"]),
            False,
        )

    # Prime initialization/cache outside timing.
    run()
    torch.cuda.synchronize()

    ms = time_cuda(run, args.warmup, args.iters)
    return float(ms)


def gpu_name_slug():
    name = torch.cuda.get_device_properties(torch.cuda.current_device()).name
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_"), name


def write_json(path: Path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


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
    ap.add_argument("--top-csv", type=int, default=40)
    ap.add_argument("--top-save", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--iters", type=int, default=1000)
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--match-stages", choices=["auto", "exact", "ignore"], default="auto")
    ap.add_argument("--default-mainloop", choices=["ws", "pingpong", "cooperative"], default="ws")
    ap.add_argument("--no-filter-layouts", dest="filter_layouts", action="store_false")
    ap.add_argument("--allow-split-k", action="store_true")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    csv_path = Path(args.csv).expanduser().resolve() if args.csv else None
    if csv_path is None:
        if args.csv_dir is None:
            raise ValueError("Pass --csv or --csv-dir")
        csv_path = Path(args.csv_dir).expanduser().resolve() / f"m{args.m}n{args.n}k{args.k}.gemm.csv"
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    algo_path = Path(args.algo_dict).expanduser().resolve() if args.algo_dict else root / "configs" / "AlgoDictSm90.json"

    records, counts = parse_csv(args, csv_path)
    by8, by9, inverse = load_algo_dict(algo_path)

    considered = records[: args.top_csv]
    matched = []
    missing = []

    for r in considered:
        algo, how = match_algo(r, by8, by9, args.match_stages)
        if algo is None:
            rr = dict(r)
            rr["match"] = how
            missing.append(rr)
            continue

        rr = dict(r)
        rr["algo"] = int(algo)
        rr["match"] = how
        rr.update({
            "algo_tile_m": inverse[algo]["tile_m"],
            "algo_tile_n": inverse[algo]["tile_n"],
            "algo_tile_k": inverse[algo]["tile_k"],
            "algo_cluster": inverse[algo]["cluster"],
            "algo_stages": inverse[algo].get("stages"),
            "algo_mainloop": inverse[algo]["mainloop"],
        })
        matched.append(rr)

    matched = dedupe_matched(matched)

    torch.cuda.set_device(args.device)
    ext = load_ext(root)

    ok = []
    failed = []

    print("========================================")
    print("gen_config_sm90")
    print(f"shape:          M={args.m} N={args.n} K={args.k}")
    print(f"layout:         {args.layout}")
    print(f"csv:            {csv_path}")
    print(f"algo_dict:      {algo_path}")
    print(f"csv raw rows:   {counts['raw_rows']}")
    print(f"csv rows kept:  {counts['kept']}")
    print(f"top_csv:        {args.top_csv}")
    print(f"matched unique: {len(matched)}")
    print(f"missing rows:   {len(missing)}")
    print(f"match_stages:   {args.match_stages}")
    print("")
    print("CSV filter counts:")
    for k, v in counts.items():
        print(f"  {k:20s}: {v}")
    print("========================================")

    for i, r in enumerate(matched, 1):
        print(
            f"[{i}/{len(matched)}] "
            f"algo={r['algo']} tile={r['tile_m']}x{r['tile_n']}x{r['tile_k']} "
            f"cluster={r['cluster']} stages={r['stages']} "
            f"mainloop={r['mainloop']} cutlass={r['cutlass_runtime']:.6f} "
            f"match={r['match']}"
        )
        try:
            meta = {
                "algo": r["algo"],
                "tile_m": r["algo_tile_m"],
                "tile_n": r["algo_tile_n"],
            }
            ms = benchmark_algo(ext, args, meta)
            r["measured_ms"] = ms
            r[f"{args.layout}_gemm_ms"] = ms
            ok.append(r)
            print(f"  {args.layout}_gemm_ms={ms:.6f}")
        except Exception as e:
            rr = dict(r)
            rr["error"] = repr(e)
            failed.append(rr)
            print(f"  FAILED: {repr(e)}")

    ok.sort(key=lambda x: x["measured_ms"])
    selected = ok[: args.top_save]

    slug, gpu_name = gpu_name_slug()
    base = root / "configs" / f"m{args.m}n{args.n}k{args.k}_{slug}_{args.layout}_sm90"

    result = {
        "description": "ooverlap SM90 configs selected from CUTLASS CSV then benchmarked",
        "M": args.m,
        "N": args.n,
        "K": args.k,
        "gpu": gpu_name,
        "csv": str(csv_path),
        "algo_dict": str(algo_path),
        "layout": args.layout,
        "warmup": args.warmup,
        "iters": args.iters,
        "match_stages": args.match_stages,
        "BM": [int(x["tile_m"]) for x in selected],
        "BN": [int(x["tile_n"]) for x in selected],
        "BK": [int(x["tile_k"]) for x in selected],
        "Stages": [int(x["stages"]) for x in selected],
        "Algo": [int(x["algo"]) for x in selected],
        "dur": [float(x["measured_ms"]) for x in selected],
        "top": selected,
        "all_profiled": ok,
    }

    missing_obj = {
        "description": "CUTLASS CSV rows that did not match AlgoDictSm90",
        "missing": missing,
    }

    write_json(base.with_suffix(".json"), result)
    write_json(Path(str(base) + "_missing.json"), missing_obj)
    write_json(Path(str(base) + "_failed.json"), {"failed": failed})

    print("")
    print("========================================")
    print("DONE")
    print(f"profiled ok: {len(ok)}")
    print(f"failed:      {len(failed)}")
    print(f"missing:     {len(missing)}")
    print("")
    print(f"wrote:       {base.with_suffix('.json')}")
    print(f"wrote:       {Path(str(base) + '_missing.json')}")
    print(f"wrote:       {Path(str(base) + '_failed.json')}")

    if selected:
        print("")
        print("Top selected configs:")
        for i, r in enumerate(selected, 1):
            print(
                f"  #{i}: algo={r['algo']} tile={r['tile_m']}x{r['tile_n']}x{r['tile_k']} "
                f"cluster={r['cluster']} stages={r['stages']} mainloop={r['mainloop']} "
                f"{args.layout}_gemm_ms={r['measured_ms']:.6f} "
                f"cutlass_runtime={r['cutlass_runtime']:.6f} match={r['match']}"
            )
    else:
        print("")
        print("No successful configs.")

    print("========================================")

    if not matched:
        raise RuntimeError(
            f"No CSV candidates matched AlgoDictSm90. Wrote diagnostics to {Path(str(base) + '_missing.json')}"
        )


if __name__ == "__main__":
    main()
