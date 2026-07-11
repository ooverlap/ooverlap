#!/usr/bin/env python3
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
A_NAMES, B_NAMES, C_NAMES, D_NAMES = ["a", "A"], ["b", "B"], ["c", "C"], ["d", "D"]
ACCUM_NAMES = ["accum", "Accum", "accumulator", "Accumulator"]
SPLIT_K_NAMES = ["split_k_slices", "split_k", "SplitK"]
CTA_M_NAMES, CTA_N_NAMES, CTA_K_NAMES = ["cta_m", "threadblock_m", "tile_m"], ["cta_n", "threadblock_n", "tile_n"], ["cta_k", "threadblock_k", "tile_k"]
CLUSTER_M_NAMES, CLUSTER_N_NAMES, CLUSTER_K_NAMES = ["cluster_m", "cluster_shape_m"], ["cluster_n", "cluster_shape_n"], ["cluster_k", "cluster_shape_k"]
STAGES_NAMES = ["stages", "Stages"]


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def norm_col(x: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(x).strip().lower()).strip("_")


def find_col(fields: Sequence[str], names: Sequence[str]) -> Optional[str]:
    m = {norm_col(c): c for c in fields}
    return next((m[norm_col(n)] for n in names if norm_col(n) in m), None)


def to_int(x: Any, default: Optional[int] = None) -> Optional[int]:
    try:
        s = "" if x is None else str(x).strip()
        return int(float(s)) if s else default
    except Exception:
        return default


def to_float(x: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        s = "" if x is None else str(x).strip()
        v = float(s) if s else default
        return default if v is None or math.isnan(v) else v
    except Exception:
        return default


def norm_layout(x: Any) -> Optional[str]:
    if x is None:
        return None
    s = str(x).strip().lower().split(":")[-1]
    if s in ("row", "row_major", "rowmajor", "n") or "row" in s:
        return "row"
    if s in ("column", "col", "column_major", "columnmajor", "t") or "col" in s:
        return "column"
    return None


def norm_dtype(x: Any) -> Optional[str]:
    if x is None:
        return None
    s = str(x).strip().lower().split(":")[0]
    if s in ("half", "fp16", "float16", "f16"):
        return "f16"
    if s in ("float", "fp32", "float32", "f32"):
        return "f32"
    if s in ("void", "none", "null"):
        return "void"
    return s or None


def op_dtypes(op: str) -> Dict[str, Optional[str]]:
    m = re.search(r"gemm_([^_]+)_([^_]+)_([^_]+)_([^_]+)_([^_]+)_", str(op))
    if not m:
        return {"a": None, "b": None, "accum": None, "c": None, "d": None}
    return {k: norm_dtype(v) for k, v in zip(("a", "b", "accum", "c", "d"), m.groups())}


def stage(x: Any) -> int:
    s = "" if x is None else str(x).strip().lower()
    return -1 if s in ("", "auto", "-1") else int(float(s))


def mainloop(op: str) -> str:
    s = str(op).lower().replace("-", "_")
    if "pingpong" in s or "ping_pong" in s:
        return "pingpong"
    if "cooperative" in s or "coop" in s:
        return "cooperative"
    return "ws"


def scheduler(op: str) -> str:
    s = str(op).lower().replace("-", "_")
    return "stream_k" if "stream_k" in s or "streamk" in s else "normal"


def filename_shape(path: Path) -> Optional[Tuple[int, int, int]]:
    m = re.search(r"m(\d+)n(\d+)k(\d+)", path.name, re.I)
    return (int(m.group(1)), int(m.group(2)), int(m.group(3))) if m else None


def key(meta: Dict[str, Any]) -> Tuple[Any, ...]:
    c = meta["cluster"]
    return (
        int(meta["tile_m"]), int(meta["tile_n"]), int(meta["tile_k"]),
        int(c[0]), int(c[1]), int(c[2]),
        stage(meta.get("stages", -1)),
        str(meta.get("mainloop", "ws")),
        str(meta.get("epilogue", "auto")),
        str(meta.get("scheduler", "normal")),
        int(meta.get("split_k", 1)),
    )


def load_ext(root: Path):
    so = root / "build" / "lib" / "ooverlap_ext.so"
    if not so.exists():
        raise FileNotFoundError(f"Could not find {so}. Build first.")
    if "ooverlap_ext" in sys.modules:
        return sys.modules["ooverlap_ext"]
    spec = importlib.util.spec_from_file_location("ooverlap_ext", str(so))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {so}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["ooverlap_ext"] = mod
    spec.loader.exec_module(mod)
    return mod


def load_algo_map(path: Path) -> Dict[int, Dict[str, Any]]:
    out = {}
    for item in json.loads(path.read_text()).get("algorithms", []):
        algo = int(item["algo"])
        cluster = item.get("cluster", [item.get("cluster_m", 1), item.get("cluster_n", 1), item.get("cluster_k", 1)])
        out[algo] = {
            "algo": algo,
            "tile_m": int(item["tile_m"]),
            "tile_n": int(item["tile_n"]),
            "tile_k": int(item["tile_k"]),
            "cluster": [int(x) for x in cluster],
            "stages": stage(item.get("stages", -1)),
            "mainloop": str(item.get("mainloop", "ws")),
            "epilogue": str(item.get("epilogue", "auto")),
            "scheduler": str(item.get("scheduler", "normal")),
            "split_k": int(item.get("split_k", item.get("split_k_slices", 1))),
        }
    return out


def csv_row_ok(args, a_dt, b_dt, c_dt, d_dt, acc, a_l, b_l, c_l, d_l) -> bool:
    checks = [
        a_dt is None or a_dt == args.csv_a_dtype,
        b_dt is None or b_dt == args.csv_b_dtype,
        acc is None or acc == args.csv_accum_dtype,
        args.csv_c_dtype == "any" or (c_dt is not None and c_dt == args.csv_c_dtype),
        args.csv_d_dtype == "any" or (d_dt is not None and d_dt == args.csv_d_dtype),
        args.csv_a_layout == "any" or a_l is None or a_l == args.csv_a_layout,
        args.csv_b_layout == "any" or b_l is None or b_l == args.csv_b_layout,
        args.csv_c_layout == "any" or c_l is None or c_l == args.csv_c_layout,
        args.csv_d_layout == "any" or d_l is None or d_l == args.csv_d_layout,
    ]
    return all(checks)


def parse_csv_rows(args, path: Path) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        rd = csv.DictReader(f)
        fields = rd.fieldnames or []

        cols = {
            "rt": find_col(fields, RUNTIME_NAMES),
            "op": find_col(fields, OP_NAMES),
            "a": find_col(fields, A_NAMES),
            "b": find_col(fields, B_NAMES),
            "c": find_col(fields, C_NAMES),
            "d": find_col(fields, D_NAMES),
            "acc": find_col(fields, ACCUM_NAMES),
            "split": find_col(fields, SPLIT_K_NAMES),
            "tm": find_col(fields, CTA_M_NAMES),
            "tn": find_col(fields, CTA_N_NAMES),
            "tk": find_col(fields, CTA_K_NAMES),
            "cm": find_col(fields, CLUSTER_M_NAMES),
            "cn": find_col(fields, CLUSTER_N_NAMES),
            "ck": find_col(fields, CLUSTER_K_NAMES),
            "st": find_col(fields, STAGES_NAMES),
        }
        if cols["rt"] is None:
            raise RuntimeError(f"No runtime column found. CSV columns: {fields}")

        shape0 = filename_shape(path)
        counts = {k: 0 for k in ("raw_rows", "bad_runtime", "shape", "layout_or_dtype", "tile_parse", "stream_k_rows_seen", "split_k_rows_seen", "kept")}
        records = []

        for i, row in enumerate(rd):
            counts["raw_rows"] += 1
            rt = to_float(row.get(cols["rt"]))
            if rt is None:
                counts["bad_runtime"] += 1
                continue

            if shape0 is not None and shape0 != (args.m, args.n, args.k):
                counts["shape"] += 1
                continue

            op = str(row.get(cols["op"], "")) if cols["op"] else ""
            sched = scheduler(op)
            split_k = to_int(row.get(cols["split"]), 1) if cols["split"] else 1
            counts["stream_k_rows_seen"] += int(sched == "stream_k")
            counts["split_k_rows_seen"] += int(split_k != 1)

            tm, tn, tk = (to_int(row.get(cols[x])) if cols[x] else None for x in ("tm", "tn", "tk"))
            st = stage(row.get(cols["st"])) if cols["st"] else None
            if tm is None or tn is None or tk is None or st is None:
                counts["tile_parse"] += 1
                continue

            cm = to_int(row.get(cols["cm"]), 1) if cols["cm"] else 1
            cn = to_int(row.get(cols["cn"]), 1) if cols["cn"] else 1
            ck = to_int(row.get(cols["ck"]), 1) if cols["ck"] else 1
            dt = op_dtypes(op)

            a_dt = norm_dtype(row.get(cols["a"])) if cols["a"] else dt["a"]
            b_dt = norm_dtype(row.get(cols["b"])) if cols["b"] else dt["b"]
            c_dt = norm_dtype(row.get(cols["c"])) if cols["c"] else dt["c"]
            d_dt = norm_dtype(row.get(cols["d"])) if cols["d"] else dt["d"]
            acc = norm_dtype(row.get(cols["acc"])) if cols["acc"] else dt["accum"]

            a_l = norm_layout(row.get(cols["a"])) if cols["a"] else None
            b_l = norm_layout(row.get(cols["b"])) if cols["b"] else None
            c_l = norm_layout(row.get(cols["c"])) if cols["c"] else None
            d_l = norm_layout(row.get(cols["d"])) if cols["d"] else None

            if not csv_row_ok(args, a_dt, b_dt, c_dt, d_dt, acc, a_l, b_l, c_l, d_l):
                counts["layout_or_dtype"] += 1
                continue

            r = {
                "source_row": i,
                "cutlass_runtime": float(rt),
                "tile_m": int(tm),
                "tile_n": int(tn),
                "tile_k": int(tk),
                "cluster": [int(cm), int(cn), int(ck)],
                "stages": int(st),
                "mainloop": mainloop(op),
                "epilogue": "auto",
                "scheduler": sched,
                "split_k": int(split_k),
                "operation": op,
            }
            r["key"] = list(key(r))
            records.append(r)
            counts["kept"] += 1

    records.sort(key=lambda x: float(x["cutlass_runtime"]))
    return records, counts


def match_records(records, algo_map):
    by_key = {key(meta): int(algo) for algo, meta in algo_map.items()}
    matched_by_algo, missing = {}, []
    for r in records:
        algo = by_key.get(tuple(r["key"]))
        if algo is None:
            missing.append({**r, "error": "no_signal_algo_for_key"})
            continue
        rr = {**r, "algo": int(algo)}
        rr.update({f"algo_{k}": v for k, v in algo_map[algo].items() if k != "algo"})
        if algo not in matched_by_algo or rr["cutlass_runtime"] < matched_by_algo[algo]["cutlass_runtime"]:
            matched_by_algo[algo] = rr
    out = list(matched_by_algo.values())
    out.sort(key=lambda x: float(x["cutlass_runtime"]))
    return out, missing


def make_ra(tile_rows: int, tile_cols: int, reorder: str, device) -> torch.Tensor:
    n = tile_rows * tile_cols
    if reorder == "identity":
        return torch.arange(n, device=device, dtype=torch.int32)
    if reorder != "column_major":
        raise ValueError(f"unknown reorder={reorder}")
    h, p = [0] * n, 0
    for tc in range(tile_cols):
        for tr in range(tile_rows):
            h[tr * tile_cols + tc] = p
            p += 1
    return torch.tensor(h, device=device, dtype=torch.int32)


def make_segments(num_tiles: int, group_tiles: int, device) -> torch.Tensor:
    if group_tiles <= 0:
        return torch.tensor([num_tiles], device=device, dtype=torch.int32)
    vals, left = [], num_tiles
    while left > 0:
        x = min(group_tiles, left)
        vals.append(x)
        left -= x
    return torch.tensor(vals, device=device, dtype=torch.int32)


def unpack(D, RA, M, N, tm, tn, reldn):
    tile_rows, tile_cols = M // tm, N // tn
    out = torch.empty((M, N), device=D.device, dtype=D.dtype)
    ra = RA.detach().cpu().tolist()
    for tr in range(tile_rows):
        for tc in range(tile_cols):
            packed = int(ra[tr * tile_cols + tc])
            pr, pc = packed // reldn, packed % reldn
            out[tr * tm:(tr + 1) * tm, tc * tn:(tc + 1) * tn].copy_(D[pr * tm:(pr + 1) * tm, pc * tn:(pc + 1) * tn])
    return out


def output_normal(D, layout, reorder, RA, M, N, tm, tn, reldn):
    return D[:M, :N] if layout == "normal" and reorder == "identity" else unpack(D, RA, M, N, tm, tn, reldn)


def time_cuda(fn: Callable[[], None], setup: Callable[[], None], warmup: int, iters: int) -> float:
    for _ in range(warmup):
        setup()
        fn()
    torch.cuda.synchronize()

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        setup()
        starts[i].record()
        fn()
        ends[i].record()
    torch.cuda.synchronize()
    return sum(s.elapsed_time(e) for s, e in zip(starts, ends)) / iters


def q1d(x: torch.Tensor, q: float, max_samples: int = 1_000_000) -> float:
    x = x.reshape(-1)
    if x.numel() > max_samples:
        x = x[::((x.numel() + max_samples - 1) // max_samples)][:max_samples]
    return float(torch.quantile(x, q).item())


def err_summary(x: torch.Tensor, y: torch.Tensor) -> Dict[str, float]:
    d = (x.float() - y.float()).abs().reshape(-1)
    return {"max_abs": float(d.max().item()), "mean_abs": float(d.mean().item()), "p99_abs": q1d(d, 0.99), "p999_abs": q1d(d, 0.999)}


def write_json(path: Path, obj: Any):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2) + "\n")


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]):
    cols = [
      "rank",
      "algo",
      "signal_gemm_ms",
      "cutlass_runtime",
      "tflops",
      "tile_m",
      "tile_n",
      "tile_k",
      "cluster",
      "stages",
      "mainloop",
      "scheduler",
      "split_k",
      "layout",
      "reorder",
      "reldn",
      "check_ok",
      "max_abs",
      "mean_abs",
      "p99_abs",
      "p999_abs",
      "source_row",
      "operation"
    ]
    
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=cols)
        wr.writeheader()
        for r in rows:
            wr.writerow({k: r.get(k, "") for k in cols})


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", type=int, required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--k", type=int, required=True)
    ap.add_argument("--csv", type=str)
    ap.add_argument("--csv-dir", type=str)
    ap.add_argument("--algo-dict", type=str)
    ap.add_argument("--layout", choices=["normal", "packed"], default="packed")
    ap.add_argument("--reorder", choices=["auto", "identity", "column_major"], default="auto")
    ap.add_argument("--reldn", type=int, default=0)
    ap.add_argument("--group-tiles", type=int, default=0)
    ap.add_argument("--top-csv", type=int, default=40)
    ap.add_argument("--top-save", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--check-atol", type=float, default=8.0)
    ap.add_argument("--reject-failed-check", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument(
        "--cpp-cublas-reference",
        action="store_true",
        help=(
            "Use the extension's baseline_gemm_col/cublasGemmEx reference. "
            "Default uses torch.mm so PyTorch owns cuBLAS/cuBLASLt end-to-end."
        ),
    )
    ap.add_argument("--csv-a-dtype", choices=["f16"], default="f16")
    ap.add_argument("--csv-b-dtype", choices=["f16"], default="f16")
    ap.add_argument("--csv-accum-dtype", choices=["f16", "f32"], default="f16")
    ap.add_argument("--csv-c-dtype", choices=["f16", "void", "any"], default="f16")
    ap.add_argument("--csv-d-dtype", choices=["f16", "f32", "any"], default="f16")
    ap.add_argument("--csv-a-layout", choices=["row", "column", "any"], default="row")
    ap.add_argument("--csv-b-layout", choices=["row", "column", "any"], default="column")
    ap.add_argument("--csv-c-layout", choices=["row", "column", "any"], default="row")
    ap.add_argument("--csv-d-layout", choices=["row", "column", "any"], default="row")
    return ap.parse_args()


def main():
    args, root = parse_args(), repo_root()
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
    matched, missing = match_records(records[:args.top_csv], algo_map)

    gpu = torch.cuda.get_device_properties(torch.cuda.current_device()).name
    slug = re.sub(r"[^a-z0-9]+", "_", gpu.lower()).strip("_")
    base = root / "configs" / f"m{args.m}n{args.n}k{args.k}_{slug}_{args.layout}_sm90"
    out_json, out_csv = base.with_suffix(".json"), base.with_suffix(".csv")
    out_missing, out_failed = Path(str(base) + "_missing.json"), Path(str(base) + "_failed.json")

    print("========================================")
    print("profile_signal_sm90")
    print(f"shape:          M={args.m} N={args.n} K={args.k}")
    print(f"gpu:            {gpu}")
    print(f"layout:         {args.layout}")
    print(f"csv:            {csv_path}")
    print(f"algo_dict:      {algo_path}")
    print(f"csv rows kept:  {counts['kept']} / {counts['raw_rows']}")
    print(f"top_csv:        {args.top_csv}")
    print(f"matched:        {len(matched)}")
    print(f"missing:        {len(missing)}")
    print("timing:         eager, MM reset outside measurement")
    reference_name = "baseline_gemm_col" if args.cpp_cublas_reference else "torch_mm"
    print(f"reference:      {reference_name}")
    for k, v in counts.items():
        print(f"  {k:18s}: {v}")
    print("========================================")

    if args.dry_run:
        write_json(out_missing, {"csv_filter_counts": counts, "missing": missing})
        return

    ext = load_ext(root)

    M, N, K = args.m, args.n, args.k
    A = torch.randn((M, K), device=device, dtype=torch.float16)
    B = torch.randn((N, K), device=device, dtype=torch.float16)
    D_ref = torch.empty((N, M), device=device, dtype=torch.float16)

    if args.cpp_cublas_reference:
        baseline = getattr(ext, "baseline_gemm_col", None)
        if baseline is None:
            raise RuntimeError("baseline_gemm_col not found in ooverlap_ext")
        baseline(A, B, D_ref)
    else:
        # Match baseline_gemm_col's physical output layout: D_ref is [N, M].
        # This avoids the extension's direct cublasGemmEx path, which can
        # segfault when its linked cuBLAS symbols do not match the cuBLAS
        # handle/runtime owned by PyTorch.
        torch.mm(B, A.t(), out=D_ref)

    torch.cuda.synchronize()
    ref = D_ref.t().contiguous()

    ok_rows: List[Dict[str, Any]] = []
    failed: List[Dict[str, Any]] = []
    flops = 2.0 * M * N * K

    for i, r in enumerate(matched, 1):
        algo, tm, tn = int(r["algo"]), int(r["tile_m"]), int(r["tile_n"])
        print(f"[{i}/{len(matched)}] algo={algo} tile={tm}x{tn}x{r['tile_k']} cluster={r['cluster']} stages={r['stages']} mainloop={r['mainloop']} scheduler={r['scheduler']} split_k={r['split_k']} cutlass={float(r['cutlass_runtime']):.6f}")

        try:
            if M % tm or N % tn:
                raise RuntimeError(f"M/N not divisible by tile {tm}x{tn}")

            tile_rows, tile_cols = M // tm, N // tn
            num_tiles = tile_rows * tile_cols
            if args.layout == "normal":
                reorder = "identity" if args.reorder == "auto" else args.reorder
                reldn, out_m, out_n = (tile_cols if args.reldn == 0 else args.reldn), M, N
            else:
                reorder = "column_major" if args.reorder == "auto" else args.reorder
                reldn = 1 if args.reldn == 0 else args.reldn
                out_m, out_n = ((num_tiles + reldn - 1) // reldn) * tm, reldn * tn

            RA = make_ra(tile_rows, tile_cols, reorder, device)
            CommThr = make_segments(num_tiles, args.group_tiles, device)
            MM = torch.zeros((int(CommThr.numel()) + num_tiles,), device=device, dtype=torch.int32)
            D = torch.empty((out_m, out_n), device=device, dtype=torch.float16)

            def setup():
                MM.zero_()

            def run():
                ext.gemm_signal_sm90(A, B, D, MM, RA, CommThr, int(reldn), algo, False)

            ms = time_cuda(run, setup, args.warmup, args.iters)
            err = err_summary(output_normal(D, args.layout, reorder, RA, M, N, tm, tn, reldn), ref)
            check_ok = err["max_abs"] <= args.check_atol
            if args.reject_failed_check and not check_ok:
                raise RuntimeError(f"check failed: max_abs={err['max_abs']} > {args.check_atol}")

            rr = {**r, **err, "signal_gemm_ms": float(ms), "measured_ms": float(ms), "timing_used": "eager", "layout": args.layout, "reorder": reorder, "reldn": int(reldn), "check_ref": reference_name, "check_ok": bool(check_ok)}
            ok_rows.append(rr)
            print(f"  signal={ms:.6f} ms err max={err['max_abs']:.6f} mean={err['mean_abs']:.6f} p99={err['p99_abs']:.6f} ok={check_ok}")
        except KeyboardInterrupt:
            raise
        except Exception as e:
            failed.append({**r, "error": repr(e)})
            print(f"  FAILED: {repr(e)}")

    ok_rows.sort(key=lambda x: float(x["signal_gemm_ms"]))
    for rank, r in enumerate(ok_rows, 1):
        r["rank"] = rank
        r["tflops"] = flops / (float(r["signal_gemm_ms"]) * 1.0e-3) / 1.0e12

    selected = ok_rows[:args.top_save]
    result = {
        "description": "ooverlap signal SM90 configs selected from CUTLASS CSV then benchmarked",
        "M": M, "N": N, "K": K, "gpu": gpu, "csv": str(csv_path), "algo_dict": str(algo_path),
        "layout": args.layout, "warmup": args.warmup, "iters": args.iters,
        "timing_mode": "eager", "reference": reference_name, "check_atol": args.check_atol,
        "reset_mm": True, "reset_mm_timing": "excluded",
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
    write_json(out_missing, {"csv_filter_counts": counts, "missing": missing})
    write_json(out_failed, {"failed": failed})

    print("\n========================================")
    print("DONE")
    print(f"profiled ok:   {len(ok_rows)}")
    print(f"failed:        {len(failed)}")
    print(f"missing:       {len(missing)}")
    print(f"wrote json:    {out_json}")
    print(f"wrote csv:     {out_csv}")
    print(f"wrote missing: {out_missing}")
    print(f"wrote failed:  {out_failed}")

    for i, r in enumerate(selected, 1):
        print(f"  #{i}: algo={r['algo']} tile={r['tile_m']}x{r['tile_n']}x{r['tile_k']} signal={float(r['signal_gemm_ms']):.6f} ms cutlass={float(r['cutlass_runtime']):.6f} ms TFLOP/s={float(r['tflops']):.2f} ok={r.get('check_ok')}")

    print("========================================")

    if not matched:
        raise RuntimeError(f"No CSV candidates matched generated SM90 signal algos. See {out_missing}")


if __name__ == "__main__":
    main()
