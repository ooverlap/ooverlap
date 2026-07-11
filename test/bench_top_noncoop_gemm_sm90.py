#!/usr/bin/env python3

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path


FLOAT_RE = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"


def parse_metrics(stdout):
    metrics = {}

    patterns = [
        (
            "baseline_gemm_col",
            re.compile(
                rf"baseline_gemm_col:\s+({FLOAT_RE})\s+ms\s+({FLOAT_RE})\s+TFLOP/s"
                rf"(?:\s+max=({FLOAT_RE})\s+mean=({FLOAT_RE}))?"
            ),
        ),
        (
            "plain_sm90",
            re.compile(
                rf"plain_sm90\s+algo=(\d+):\s+({FLOAT_RE})\s+ms\s+({FLOAT_RE})\s+TFLOP/s"
                rf"(?:\s+max=({FLOAT_RE})\s+mean=({FLOAT_RE}))?"
            ),
        ),
        (
            "signal_sm90",
            re.compile(
                rf"signal_sm90\s+algo=(\d+):\s+({FLOAT_RE})\s+ms\s+({FLOAT_RE})\s+TFLOP/s"
                rf"(?:\s+max=({FLOAT_RE})\s+mean=({FLOAT_RE}))?"
            ),
        ),
    ]

    for name, pat in patterns:
        m = pat.search(stdout)
        if not m:
            continue

        g = m.groups()

        if name == "baseline_gemm_col":
            metrics[name] = {
                "ms": float(g[0]),
                "tflops": float(g[1]),
            }
            if g[2] is not None:
                metrics[name]["max_abs"] = float(g[2])
                metrics[name]["mean_abs"] = float(g[3])
        else:
            metrics[name] = {
                "algo": int(g[0]),
                "ms": float(g[1]),
                "tflops": float(g[2]),
            }
            if g[3] is not None:
                metrics[name]["max_abs"] = float(g[3])
                metrics[name]["mean_abs"] = float(g[4])

    return metrics


def filename_dims(path):
    m = re.search(r"m(\d+)n(\d+)k(\d+)", path.name)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2)), int(m.group(3))


def is_real_config(path):
    name = path.name
    if not name.endswith(".json"):
        return False
    if name.startswith("solution_"):
        return False
    if name.startswith("bandwidth_") or name.startswith("AlgoDict"):
        return False
    if "_failed" in name or "_missing" in name:
        return False
    return re.match(r"m\d+n\d+k\d+_", name) is not None


def is_cooperative(entry):
    text = " ".join(
        str(entry.get(k, ""))
        for k in ("mainloop", "operation", "kernel", "name")
    ).lower()
    return "cooperative" in text


def metric_for_sort(entry, cfg, idx):
    for k in ("bench_ms", "dur", "cutlass_runtime", "runtime"):
        if k in entry:
            return float(entry[k])

    if idx < len(cfg.get("dur", [])):
        return float(cfg["dur"][idx])

    return float("inf")


def explicit_algo(entry):
    for k in ("algo", "Algo", "algo_id", "algoId", "AlgoId"):
        if k in entry:
            return int(entry[k])
    return None


def load_candidates(cfg_path, top_k):
    cfg = json.loads(cfg_path.read_text())

    dims = (
        int(cfg.get("M", 0)),
        int(cfg.get("N", 0)),
        int(cfg.get("K", 0)),
    )

    if 0 in dims:
        parsed = filename_dims(cfg_path)
        if parsed is None:
            raise ValueError(f"could not infer M/N/K from {cfg_path}")
        dims = parsed

    top = cfg.get("top", [])
    algos = cfg.get("Algo", [])

    candidates = []
    seen_algos = set()

    for idx, entry in enumerate(top):
        # if is_cooperative(entry):
            # continue

        algo = explicit_algo(entry)

        # In the current generated config files, top[i] corresponds to Algo[i].
        # Keep this as the simple/default mapping.
        if algo is None and idx < len(algos):
            algo = int(algos[idx])

        if algo is None:
            continue

        if algo in seen_algos:
            continue
        seen_algos.add(algo)

        candidates.append(
            {
                "top_index": idx,
                "algo": algo,
                "sort_ms": metric_for_sort(entry, cfg, idx),
                "cutlass_runtime": entry.get("cutlass_runtime"),
                "measured_config_ms": cfg.get("dur", [None] * (idx + 1))[idx]
                if idx < len(cfg.get("dur", []))
                else None,
                "tile_m": entry.get("tile_m", cfg.get("BM", [None] * (idx + 1))[idx] if idx < len(cfg.get("BM", [])) else None),
                "tile_n": entry.get("tile_n", cfg.get("BN", [None] * (idx + 1))[idx] if idx < len(cfg.get("BN", [])) else None),
                "tile_k": entry.get("tile_k", cfg.get("BK", [None] * (idx + 1))[idx] if idx < len(cfg.get("BK", [])) else None),
                "stages": entry.get("stages", cfg.get("Stages", [None] * (idx + 1))[idx] if idx < len(cfg.get("Stages", [])) else None),
                "mainloop": entry.get("mainloop"),
                "operation": entry.get("operation"),
                "source_row": entry.get("source_row"),
            }
        )

    candidates.sort(key=lambda x: x["sort_ms"])
    return cfg, dims, candidates[:top_k]


def run_one(repo_root, args, cfg_path, cfg, dims, cand):
    m, n, k = dims
    bench = repo_root / "test" / "bench_gemm_sm90.py"

    cmd = [
        sys.executable,
        str(bench),
        "--mode",
        "both",
        "--m",
        str(m),
        "--n",
        str(n),
        "--k",
        str(k),
        "--plain-algo",
        str(cand["algo"]),
        "--signal-algo",
        str(cand["algo"]),
        "--warmup",
        str(args.warmup),
        "--iters",
        str(args.iters),
        "--device",
        str(args.device),
        "--segments",
        str(args.segments),
        "--rldn",
        str(args.rldn),
    ]

    if cand.get("tile_m") is not None:
        cmd += ["--tile-m", str(int(cand["tile_m"]))]
    if cand.get("tile_n") is not None:
        cmd += ["--tile-n", str(int(cand["tile_n"]))]

    if args.with_baseline:
        cmd.append("--with-baseline")

    if not args.check:
        cmd.append("--no-check")

    env = os.environ.copy()

    proc = subprocess.run(
        cmd,
        cwd=str(repo_root),
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    row = {
        "config": str(cfg_path),
        "matrix": {"M": m, "N": n, "K": k},
        "layout": cfg.get("layout"),
        "algo": cand["algo"],
        "candidate": cand,
        "command": cmd,
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "metrics": parse_metrics(proc.stdout),
        "status": "ok" if proc.returncode == 0 else "error",
    }

    return row


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark top-k non-cooperative SM90 GEMM algos from generated config JSON files."
    )
    parser.add_argument("--repo-root", default=".")
    parser.add_argument(
        "--config-glob",
        default="configs/m*_nvidia_h100_nvl_*_sm90.json",
    )
    parser.add_argument("--config", action="append", default=[])
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--segments", type=int, default=1)
    parser.add_argument("--rldn", type=int, default=0)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--no-baseline", dest="with_baseline", action="store_false")
    parser.add_argument("--limit-configs", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--out",
        default="results/top_noncoop_gemm_sm90.json",
    )
    parser.set_defaults(with_baseline=True)
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    out_path = repo_root / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.config:
        config_paths = [Path(p) for p in args.config]
    else:
        config_paths = sorted(repo_root.glob(args.config_glob))
        config_paths = [p for p in config_paths if is_real_config(p)]

    if args.limit_configs > 0:
        config_paths = config_paths[: args.limit_configs]

    result = {
        "description": "Top-k non-cooperative SM90 GEMM algos benchmarked through test/bench_gemm_sm90.py",
        "top_k": args.top_k,
        "warmup": args.warmup,
        "iters": args.iters,
        "device": args.device,
        "check": args.check,
        "with_baseline": args.with_baseline,
        "config_count": len(config_paths),
        "runs": [],
        "skipped": [],
    }

    print(f"[info] configs: {len(config_paths)}")
    print(f"[info] output: {out_path}")

    for cfg_path in config_paths:
        if not cfg_path.is_absolute():
            cfg_path = repo_root / cfg_path

        try:
            cfg, dims, candidates = load_candidates(cfg_path, args.top_k)
        except Exception as e:
            result["skipped"].append(
                {
                    "config": str(cfg_path),
                    "reason": f"load failed: {e}",
                }
            )
            print(f"[skip] {cfg_path}: {e}")
            continue

        if not candidates:
            result["skipped"].append(
                {
                    "config": str(cfg_path),
                    "matrix": {"M": dims[0], "N": dims[1], "K": dims[2]},
                    "reason": "no non-cooperative candidates found",
                }
            )
            print(f"[skip] {cfg_path}: no non-cooperative candidates")
            continue

        print(f"[config] {cfg_path.name} M={dims[0]} N={dims[1]} K={dims[2]}")
        for cand in candidates:
            print(
                f"  algo={cand['algo']} "
                f"top_index={cand['top_index']} "
                f"mainloop={cand.get('mainloop')} "
                f"tile=({cand.get('tile_m')},{cand.get('tile_n')},{cand.get('tile_k')})"
            )

            if args.dry_run:
                result["runs"].append(
                    {
                        "config": str(cfg_path),
                        "matrix": {"M": dims[0], "N": dims[1], "K": dims[2]},
                        "algo": cand["algo"],
                        "candidate": cand,
                        "status": "dry_run",
                    }
                )
                continue

            row = run_one(repo_root, args, cfg_path, cfg, dims, cand)
            result["runs"].append(row)

            if row["status"] == "ok":
                p = row["metrics"].get("plain_sm90", {})
                s = row["metrics"].get("signal_sm90", {})
                print(
                    f"    ok plain={p.get('ms')} ms {p.get('tflops')} TF/s | "
                    f"signal={s.get('ms')} ms {s.get('tflops')} TF/s"
                )
            else:
                print(f"    ERROR returncode={row['returncode']}")
                print(row["stderr"][-1000:])

            out_path.write_text(json.dumps(result, indent=2))

    out_path.write_text(json.dumps(result, indent=2))
    print(f"[done] wrote {out_path}")


if __name__ == "__main__":
    main()
