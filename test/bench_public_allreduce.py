import argparse
import importlib.util
import json
import math
import os
from pathlib import Path

import matplotlib.pyplot as plt


TUNING_MODES = {
    "performance": 0,
    "best_performance": 0,
    "efficiency": 1,
    "best_efficiency": 1,
}


def parse_bytes(text):
    s = str(text).strip().lower()
    multipliers = {
        "b": 1,
        "kb": 1000,
        "kib": 1024,
        "mb": 1000**2,
        "mib": 1024**2,
        "gb": 1000**3,
        "gib": 1024**3,
    }

    for suffix in sorted(multipliers, key=len, reverse=True):
        if s.endswith(suffix):
            number = float(s[: -len(suffix)].strip())
            return int(number * multipliers[suffix])

    return int(float(s))


def bytes_label(num_bytes):
    num_bytes = float(num_bytes)

    if num_bytes >= 1024**3:
        return f"{num_bytes / 1024**3:.1f} GiB"
    if num_bytes >= 1024**2:
        return f"{num_bytes / 1024**2:.1f} MiB"
    if num_bytes >= 1024:
        return f"{num_bytes / 1024:.1f} KiB"

    return f"{num_bytes:.0f} B"


def repo_root():
    return Path(__file__).resolve().parents[1]


def load_ooverlap_ext():
    root = repo_root()
    so_path = root / "build" / "lib" / "ooverlap_ext.so"

    if not so_path.exists():
        raise RuntimeError(f"Extension not found: {so_path}")

    spec = importlib.util.spec_from_file_location("ooverlap_ext", so_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def parse_jsonl(text):
    rows = []
    for line_no, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Invalid JSONL line {line_no}: {exc}") from exc
    return rows


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


def run_one(ext, args, cta_cap):
    old = os.environ.get("OOVERLAP_MAX_CTAS")

    if cta_cap is None:
        os.environ.pop("OOVERLAP_MAX_CTAS", None)
    else:
        os.environ["OOVERLAP_MAX_CTAS"] = str(cta_cap)

    try:
        text = ext.benchmark_public_allreduce_2gpu_sm90(
            int(args.min_bytes),
            int(args.max_bytes),
            int(args.points),
            int(args.iters),
            int(args.warmup),
            int(TUNING_MODES[args.tuning_mode]),
            int(args.dev0),
            int(args.dev1),
        )
    finally:
        if old is None:
            os.environ.pop("OOVERLAP_MAX_CTAS", None)
        else:
            os.environ["OOVERLAP_MAX_CTAS"] = old

    rows = parse_jsonl(text)

    for row in rows:
        row["cta_cap"] = cta_cap

    return rows


def plot_rows(rows, out_dir, metric):
    cta_values = sorted({row.get("cta_cap") for row in rows}, key=lambda x: (-1 if x is None else x))

    written = []

    for cta in cta_values:
        subset = [row for row in rows if row.get("cta_cap") == cta]

        if not subset:
            continue

        plt.figure(figsize=(10, 6))

        for backend in ["ooverlap", "nccl"]:
            backend_rows = sorted(
                [row for row in subset if row["backend"] == backend],
                key=lambda r: int(r["bytes_per_rank"]),
            )

            if not backend_rows:
                continue

            xs = [int(row["bytes_per_rank"]) for row in backend_rows]
            ys = [float(row[metric]) for row in backend_rows]

            label = backend
            plt.plot(xs, ys, marker="o", linewidth=1.5, markersize=3, label=label)

        plt.xscale("log", base=2)
        plt.xlabel("Bytes per rank")

        if metric == "gbps_aggregate_2gpu":
            plt.ylabel("Aggregate payload bandwidth, 2 GPU (GB/s)")
            metric_label = "aggregate_bandwidth"
        elif metric == "gbps_per_rank":
            plt.ylabel("Per-rank payload bandwidth (GB/s)")
            metric_label = "per_rank_bandwidth"
        else:
            plt.ylabel(metric)
            metric_label = metric

        title_cta = "default CTA policy" if cta is None else f"OOVERLAP_MAX_CTAS={cta}"
        plt.title(f"Public allreduce vs NCCL: {metric_label}, {title_cta}")
        plt.grid(True, which="both", linestyle="--", linewidth=0.5)
        plt.legend()

        unique_xs = sorted({int(row["bytes_per_rank"]) for row in subset})
        if len(unique_xs) <= 24:
            plt.xticks(unique_xs, [bytes_label(x) for x in unique_xs], rotation=35, ha="right")

        plt.tight_layout()

        suffix = "default" if cta is None else f"ctas_{cta}"
        out_path = out_dir / f"public_allreduce_{metric_label}_{suffix}.png"
        plt.savefig(out_path, dpi=180)
        plt.close()

        written.append(out_path)

    return written


def main():
    parser = argparse.ArgumentParser("Benchmark public ooverlap allreduce API vs NCCL")
    parser.add_argument("--min-bytes", type=parse_bytes, required=True)
    parser.add_argument("--max-bytes", type=parse_bytes, required=True)
    parser.add_argument("--points", type=int, required=True)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--dev0", type=int, default=0)
    parser.add_argument("--dev1", type=int, default=1)
    parser.add_argument(
        "--tuning-mode",
        choices=sorted(TUNING_MODES.keys()),
        default="best_performance",
    )
    parser.add_argument(
        "--ctas",
        type=str,
        default="",
        help="Comma-separated OOVERLAP_MAX_CTAS caps, e.g. 2,4,8,16. Empty means default policy only.",
    )
    parser.add_argument(
        "--include-default-ctas",
        action="store_true",
        help="Also run once without OOVERLAP_MAX_CTAS when --ctas is set.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="results/public_allreduce_benchmark.jsonl",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="results",
    )
    parser.add_argument(
        "--metric",
        choices=["gbps_aggregate_2gpu", "gbps_per_rank"],
        default="gbps_aggregate_2gpu",
    )

    args = parser.parse_args()

    ext = load_ooverlap_ext()

    cta_caps = []

    if args.ctas.strip():
        cta_caps = [int(x.strip()) for x in args.ctas.split(",") if x.strip()]
        if args.include_default_ctas:
            cta_caps.insert(0, None)
    else:
        cta_caps = [None]

    print(f"[info] min_bytes={args.min_bytes} max_bytes={args.max_bytes} points={args.points}")
    print(f"[info] iters={args.iters} warmup={args.warmup}")
    print(f"[info] tuning_mode={args.tuning_mode}")
    print(f"[info] cta_caps={cta_caps}")

    all_rows = []

    for cta in cta_caps:
        label = "default" if cta is None else str(cta)
        print(f"[run] OOVERLAP_MAX_CTAS={label}")
        rows = run_one(ext, args, cta)
        all_rows.extend(rows)

    out_path = Path(args.out)
    write_jsonl(out_path, all_rows)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    plots = plot_rows(all_rows, out_dir, args.metric)

    print(f"[result] wrote {len(all_rows)} rows to {out_path}")
    for path in plots:
        print(f"[plot] {path}")

    print("PASS ✅ public allreduce benchmark complete")


if __name__ == "__main__":
    main()
