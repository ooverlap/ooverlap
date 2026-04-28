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

    if num_bytes >= 1024**2:
        value = num_bytes / 1024**2
        if abs(value - round(value)) < 1e-9:
            return f"{int(round(value))} MB"
        return f"{value:.1f} MB"

    if num_bytes >= 1024:
        value = num_bytes / 1024
        if abs(value - round(value)) < 1e-9:
            return f"{int(round(value))} KB"
        return f"{value:.1f} KB"

    return f"{num_bytes:.0f} B"

def set_size_axis_ticks(ax, min_x, max_x):
    """
    Use power-of-two tick locations, but label them as human sizes:

      2 KB, 4 KB, ..., 512 KB, 1 MB, 2 MB, ..., 512 MB

    This avoids matplotlib labels like 2^29 bytes.
    """
    min_x = max(1, int(min_x))
    max_x = max(min_x, int(max_x))

    start_exp = math.floor(math.log2(min_x))
    end_exp = math.ceil(math.log2(max_x))

    ticks = []

    for exp in range(start_exp, end_exp + 1):
        value = 1 << exp
        if min_x <= value <= max_x:
            ticks.append(value)

    if min_x not in ticks:
        ticks.insert(0, min_x)

    if max_x not in ticks:
        ticks.append(max_x)

    # Avoid unreadable axes when there are too many decades.
    if len(ticks) > 18:
        stride = math.ceil(len(ticks) / 18)
        ticks = ticks[::stride]
        if ticks[-1] != max_x:
            ticks.append(max_x)

    ax.set_xticks(ticks)
    ax.set_xticklabels(
        [bytes_label(x) for x in ticks],
        rotation=35,
        ha="right",
    )

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
    old_ooverlap_max_ctas = os.environ.get("OOVERLAP_MAX_CTAS")
    old_nccl_max_ctas = os.environ.get("NCCL_MAX_CTAS")

    if cta_cap is None:
        os.environ.pop("OOVERLAP_MAX_CTAS", None)
        os.environ.pop("NCCL_MAX_CTAS", None)
    else:
        os.environ["OOVERLAP_MAX_CTAS"] = str(cta_cap)
        os.environ["NCCL_MAX_CTAS"] = str(cta_cap)

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
        if old_ooverlap_max_ctas is None:
            os.environ.pop("OOVERLAP_MAX_CTAS", None)
        else:
            os.environ["OOVERLAP_MAX_CTAS"] = old_ooverlap_max_ctas

        if old_nccl_max_ctas is None:
            os.environ.pop("NCCL_MAX_CTAS", None)
        else:
            os.environ["NCCL_MAX_CTAS"] = old_nccl_max_ctas

    rows = parse_jsonl(text)

    for row in rows:
        row["cta_cap"] = cta_cap
        row["ooverlap_max_ctas_env"] = cta_cap
        row["nccl_max_ctas_env"] = cta_cap

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

        ax = plt.gca()
        ax.set_xscale("log", base=2)
        plt.xlabel("Data size per rank")

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
        if unique_xs:
            set_size_axis_ticks(ax, min(unique_xs), max(unique_xs))

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
