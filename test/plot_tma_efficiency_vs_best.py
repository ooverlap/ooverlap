#!/usr/bin/env python3

import argparse
import csv
import importlib
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt


MODE_LABEL = {
    "best_performance": "Best performance",
    "best_efficiency": "Best efficiency",
}

COLLECTIVE_LABEL = {
    "allreduce": "All-reduce",
    "all_reduce": "All-reduce",
    "reduce_scatter": "Reduce-scatter",
    "all_gather": "All-gather",
}


def parse_int_list(s):
    if s is None or str(s).strip() == "":
        return []
    return [int(x) for x in str(s).replace(",", " ").split() if x.strip()]


def parse_str_list(s):
    if s is None or str(s).strip() == "":
        return []
    return [x.strip() for x in str(s).replace(",", " ").split() if x.strip()]


def load_ext(build_dir):
    build_lib = Path(build_dir) / "lib"
    if build_lib.exists():
        sys.path.insert(0, str(build_lib.resolve()))
    return importlib.import_module("ooverlap_ext")


def make_request(args):
    req = {
        "numels": args.numels,
        "collectives": args.collectives,
        "modes": args.modes,
        "iters": args.iters,
        "warmup": args.warmup,
        "dev0": args.dev0,
        "dev1": args.dev1,
        # NCCL is not useful for this specific plot. Keep it off by default.
        "include_nccl": bool(args.include_nccl and not args.no_nccl),
    }

    if args.ctas:
        req["ctas"] = args.ctas

    return req


def row_backend(row):
    return row.get("backend", "")


def row_mode(row):
    mode = row.get("tuning_mode")
    if mode is None:
        mode = row.get("id", {}).get("tuning_mode")
    return mode


def row_cta(row):
    cta = row.get("max_ctas")
    if cta is None:
        cta = row.get("id", {}).get("max_ctas")
    return cta


def row_collective(row):
    return row.get("collective") or row.get("id", {}).get("collective", "unknown")


def ok_rows(rows):
    return [r for r in rows if r.get("status") == "ok"]


def ooverlap_rows(rows):
    return [
        r for r in ok_rows(rows)
        if row_backend(r) in ("ooverlap_public", "ooverlap")
    ]


def available_ctas(rows, requested_ctas):
    if requested_ctas:
        return requested_ctas

    found = sorted(
        {row_cta(r) for r in ooverlap_rows(rows) if row_cta(r) is not None}
    )
    return found if found else [None]


def format_cta(cta):
    return "ambient" if cta is None else str(cta)


def format_size(nbytes):
    n = int(nbytes)
    for scale, suffix in ((1024**3, "GiB"), (1024**2, "MiB"), (1024, "KiB")):
        if n >= scale:
            value = n / scale
            return f"{int(value)} {suffix}" if value.is_integer() else f"{value:.1f} {suffix}"
    return f"{n} B"


def collective_label(name):
    return COLLECTIVE_LABEL.get(name, name.replace("_", "-"))


def mode_label(name):
    return MODE_LABEL.get(name, name or "unknown")


def label_for_row(row):
    return f"{collective_label(row_collective(row))}: {mode_label(row_mode(row))}"


def line_style_for_mode(mode):
    if mode == "best_performance":
        return {
            "linestyle": "-",
            "marker": "o",
            "linewidth": 1.8,
            "alpha": 0.90,
        }

    if mode == "best_efficiency":
        return {
            "linestyle": "--",
            "marker": "s",
            "linewidth": 2.2,
            "alpha": 0.95,
        }

    return {
        "linestyle": "-",
        "marker": "o",
        "linewidth": 1.8,
        "alpha": 0.90,
    }


def metric_value(row, metric):
    return float(row[metric])


def x_value_bytes(row):
    if "bytes_per_rank" in row:
        return int(row["bytes_per_rank"])

    # Fallback for older result files.
    return int(row["numel"]) * 2


def group_series(rows, metric, cta=None):
    series = {}

    for row in ooverlap_rows(rows):
        if row_cta(row) != cta:
            continue

        label = label_for_row(row)
        series.setdefault(label, []).append(
            (
                x_value_bytes(row),
                metric_value(row, metric),
                row_mode(row),
            )
        )

    for label in series:
        series[label].sort(key=lambda p: p[0])

    return series


def plot_metric(rows, ctas, metric, ylabel, title, out_path):
    fig, axes = plt.subplots(
        1,
        len(ctas),
        figsize=(5.8 * len(ctas), 4.5),
        squeeze=False,
        sharey=False,
    )
    axes = axes[0]

    legend = {}

    for ax, cta in zip(axes, ctas):
        series = group_series(rows, metric, cta=cta)

        xticks = set()

        for label, points in sorted(series.items()):
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            mode = points[0][2] if points else None
            xticks.update(xs)

            line, = ax.plot(
                xs,
                ys,
                label=label,
                **line_style_for_mode(mode),
            )

            if label not in legend:
                legend[label] = line

        ax.set_xscale("log", base=2)

        xticks = sorted(xticks)
        if xticks:
            ax.set_xticks(xticks)
            ax.set_xticklabels(
                [format_size(x) for x in xticks],
                rotation=30,
                ha="right",
            )

        ax.set_title(f"CTA limit = {format_cta(cta)}", fontsize=12)
        ax.set_xlabel("Message size per rank")
        ax.set_ylabel(ylabel)
        ax.grid(True, which="both", linestyle="--", alpha=0.35)

    if legend:
        fig.legend(
            list(legend.values()),
            list(legend.keys()),
            loc="upper center",
            bbox_to_anchor=(0.5, 0.93),
            ncol=min(3, max(1, len(legend))),
            fontsize=9,
            frameon=False,
        )

    fig.suptitle(title, fontsize=14, y=1.02)
    fig.tight_layout(rect=(0, 0, 1, 0.84))
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    print(f"[plot] wrote {out_path}")


def flatten_row(row):
    rid = row.get("id", {})
    out = {}

    for k, v in row.items():
        if k == "id":
            continue
        if isinstance(v, (str, int, float, bool)) or v is None:
            out[k] = v

    for k, v in rid.items():
        key = f"id_{k}"
        if isinstance(v, (str, int, float, bool)) or v is None:
            out[key] = v

    return out


def write_csv(rows, path):
    flat = [flatten_row(r) for r in rows]

    keys = []
    seen = set()
    preferred = [
        "status",
        "backend",
        "collective",
        "tuning_mode",
        "max_ctas",
        "numel",
        "bytes_per_rank",
        "avg_ms",
        "latency_us",
        "effective_gbps_per_rank",
        "effective_gbps_aggregate_2gpu",
        "iters",
        "warmup",
    ]

    for k in preferred:
        if any(k in r for r in flat):
            keys.append(k)
            seen.add(k)

    for row in flat:
        for k in sorted(row.keys()):
            if k not in seen:
                keys.append(k)
                seen.add(k)

    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(flat)

    print(f"[csv] wrote {path}")


def print_summary(rows):
    rows = ooverlap_rows(rows)

    by_key = {}
    for row in rows:
        key = (row_collective(row), int(row["numel"]), row_cta(row))
        by_key.setdefault(key, {})[row_mode(row)] = row

    print("\n[best_performance vs best_efficiency]")
    for key in sorted(by_key.keys(), key=lambda x: (x[0], x[1], str(x[2]))):
        collective, numel, cta = key
        modes = by_key[key]

        perf = modes.get("best_performance")
        eff = modes.get("best_efficiency")

        if not perf and not eff:
            continue

        print(
            f"{collective_label(collective)} "
            f"size={format_size((perf or eff).get('bytes_per_rank', int(numel) * 2))} "
            f"max_ctas={format_cta(cta)}"
        )

        for name, row in [("performance", perf), ("efficiency ", eff)]:
            if not row:
                continue

            print(
                f"  {name}: "
                f"{float(row['effective_gbps_per_rank']):8.2f} GB/s/rank "
                f"{float(row['latency_us']):8.2f} us"
            )

        if perf and eff:
            perf_bw = float(perf["effective_gbps_per_rank"])
            eff_bw = float(eff["effective_gbps_per_rank"])
            ratio = eff_bw / perf_bw if perf_bw > 0.0 else 0.0
            print(f"  efficiency/performance bandwidth ratio: {ratio:.3f}")

            if abs(1.0 - ratio) < 0.005:
                print("  note: curves may overlap; selected configs are effectively identical")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--build-dir", default="build")
    parser.add_argument("--out-dir", default="results/tma_efficiency")

    parser.add_argument(
        "--numels",
        type=parse_int_list,
        default=parse_int_list("4096,65536,1048576,4194304,16777216,134217728"),
    )
    parser.add_argument(
        "--ctas",
        type=parse_int_list,
        default=parse_int_list("1,2,4,8,16"),
        help="CTA constraints to test via OOVERLAP_MAX_CTAS. Use '' to leave ambient env alone.",
    )
    parser.add_argument(
        "--collectives",
        type=parse_str_list,
        default=parse_str_list("allreduce,reduce_scatter,all_gather"),
    )
    parser.add_argument(
        "--modes",
        type=parse_str_list,
        default=parse_str_list("best_performance,best_efficiency"),
    )

    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--dev0", type=int, default=0)
    parser.add_argument("--dev1", type=int, default=1)

    # Default: no NCCL. It is not useful for this comparison figure.
    parser.add_argument("--include-nccl", action="store_true")
    parser.add_argument("--no-nccl", action="store_true")

    args = parser.parse_args()

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
        print(json.dumps(result, indent=2))
        raise SystemExit(1)

    rows = result.get("raw_results", result.get("results", []))

    csv_path = out_dir / "tma_efficiency_vs_best.csv"
    write_csv(rows, csv_path)

    print_summary(rows)

    ctas = available_ctas(rows, args.ctas)

    plot_metric(
        rows,
        ctas,
        "effective_gbps_per_rank",
        "Effective bandwidth per rank (GB/s)",
        "Public Tuning Policy: Bandwidth vs. Message Size",
        out_dir / "bandwidth_by_cta.png",
    )

    plot_metric(
        rows,
        ctas,
        "latency_us",
        "Latency (us)",
        "Public Tuning Policy: Latency vs. Message Size",
        out_dir / "latency_by_cta.png",
    )


if __name__ == "__main__":
    main()
