#!/usr/bin/env python3

import argparse
import importlib
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt


def parse_int_list(s):
    return [int(x) for x in s.replace(",", " ").split() if x]


def parse_str_list(s):
    return [x.strip() for x in s.replace(",", " ").split() if x.strip()]


def load_ext(build_dir):
    import sys

    build_lib = Path(build_dir) / "lib"
    if build_lib.exists():
        sys.path.insert(0, str(build_lib.resolve()))

    return importlib.import_module("ooverlap_ext")


def make_request(args):
    return {
        "numels": args.numels,
        "collectives": args.collectives,
        "kernels": args.kernels,
        "ctas": args.ctas,
        "iters": args.iters,
        "warmup": args.warmup,
        "dev0": args.dev0,
        "dev1": args.dev1,
        "efficiency_min_best_ratio": args.efficiency_min_best_ratio,
        "include_nccl": args.include_nccl,
        "variants": [
            {
                "name": "8k_x16",
                "chunk_bytes": 8 * 1024,
                "stage_depth": 16,
                "threads": args.threads,
                "window_chunks": args.window_chunks,
            },
            {
                "name": "16k_x8",
                "chunk_bytes": 16 * 1024,
                "stage_depth": 8,
                "threads": args.threads,
                "window_chunks": args.window_chunks,
            },
            {
                "name": "32k_x4",
                "chunk_bytes": 32 * 1024,
                "stage_depth": 4,
                "threads": args.threads,
                "window_chunks": args.window_chunks,
            },
        ],
    }


def label_from_row(row):
    rid = row.get("id", {})
    collective = rid.get("collective", row.get("collective", "unknown"))
    variant = rid.get("variant", "unknown")
    kernel = rid.get("kernel", row.get("kernel", "unknown"))
    return f"{collective}:{kernel}:{variant}"


def rows_for_cta(raw_rows, cta):
    out = []
    for row in raw_rows:
        if row.get("status") != "ok":
            continue
        if row.get("backend") != "ooverlap":
            continue
        rid = row.get("id", {})
        if rid.get("max_ctas") == cta:
            out.append(row)
    return out


def group_series(rows, y_key):
    series = {}

    for row in rows:
        rid = row.get("id", {})
        x = int(row["numel"])
        y = float(row[y_key])
        label = label_from_row(row)
        series.setdefault(label, []).append((x, y))

    for label in series:
        series[label].sort(key=lambda p: p[0])

    return series


def plot_metric(raw_rows, ctas, y_key, ylabel, title, out_path):
    fig, axes = plt.subplots(
        1,
        len(ctas),
        figsize=(5.5 * len(ctas), 4.2),
        squeeze=False,
        sharey=False,
    )

    axes = axes[0]

    for ax, cta in zip(axes, ctas):
        cta_rows = rows_for_cta(raw_rows, cta)
        series = group_series(cta_rows, y_key)

        for label, points in series.items():
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            ax.plot(xs, ys, marker="o", label=label)

        ax.set_xscale("log", base=2)
        ax.set_title(f"max_ctas={cta}")
        ax.set_xlabel("numel")
        ax.set_ylabel(ylabel)
        ax.grid(True, which="both", alpha=0.3)

    handles, labels = axes[-1].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=min(4, len(labels)))

    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0.12, 1, 0.92))
    fig.savefig(out_path, dpi=180)
    print(f"[plot] wrote {out_path}")


def print_summary(summary_rows):
    print("\n[summary across variants]")
    for row in summary_rows:
        best = row["best_performance"]
        eff = row["efficiency_near_best"]
        print(
            f"{row['collective']} numel={row['numel']} "
            f"best: cta={best.get('max_ctas')} "
            f"{best.get('effective_gbps_aggregate_2gpu'):.2f} GB/s "
            f"{best.get('latency_us'):.2f} us | "
            f"eff: cta={eff.get('max_ctas')} "
            f"{eff.get('effective_gbps_aggregate_2gpu'):.2f} GB/s "
            f"{eff.get('latency_us'):.2f} us "
            f"ratio={row.get('near_best_perf_ratio', 0.0):.3f}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-dir", default="build")
    parser.add_argument(
        "--numels",
        type=parse_int_list,
        default=parse_int_list("4096,65536,1048576,4194304,16777216,134217728"),
    )
    parser.add_argument(
        "--ctas",
        type=parse_int_list,
        default=parse_int_list("1,2,4,8,16"),
    )
    parser.add_argument(
        "--collectives",
        type=parse_str_list,
        default=parse_str_list("allreduce,reduce_scatter,all_gather"),
    )
    parser.add_argument(
        "--kernels",
        type=parse_str_list,
        default=parse_str_list("tma_copy"),
    )
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--dev0", type=int, default=0)
    parser.add_argument("--dev1", type=int, default=1)
    parser.add_argument("--threads", type=int, default=32)
    parser.add_argument("--window-chunks", type=int, default=128)
    parser.add_argument("--efficiency-min-best-ratio", type=float, default=0.95)
    parser.add_argument("--no-nccl", action="store_true")
    parser.add_argument("--out-dir", default="results/tma_efficiency")
    args = parser.parse_args()

    args.include_nccl = not args.no_nccl

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ext = load_ext(args.build_dir)
    request = make_request(args)

    print("[request]")
    print(json.dumps(request, indent=2))

    result_text = ext.benchmark_tma_efficiency_vs_best_2gpu_json(
        json.dumps(request)
    )
    result = json.loads(result_text)

    result_path = out_dir / "tma_efficiency_vs_best.json"
    result_path.write_text(json.dumps(result, indent=2))
    print(f"[json] wrote {result_path}")

    if not result.get("ok", False):
        print("[error] benchmark returned ok=false")
        print(json.dumps(result.get("errors", result.get("sweep_errors", [])), indent=2))
        raise SystemExit(1)

    raw_rows = result["raw_results"]

    print_summary(result.get("summary_across_variants", []))

    plot_metric(
        raw_rows,
        args.ctas,
        "effective_gbps_aggregate_2gpu",
        "aggregate bandwidth, 2 GPU (GB/s)",
        "Bandwidth vs size by CTA count",
        out_dir / "bandwidth_by_cta.png",
    )

    plot_metric(
        raw_rows,
        args.ctas,
        "latency_us",
        "latency (us)",
        "Latency vs size by CTA count",
        out_dir / "latency_by_cta.png",
    )


if __name__ == "__main__":
    main()
