#!/usr/bin/env python3

import argparse
import csv
import importlib
import json
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt


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
        "include_nccl": not args.no_nccl,
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


def nccl_rows(rows):
    return [r for r in ok_rows(rows) if row_backend(r) == "nccl"]


def available_ctas(rows, requested_ctas):
    if requested_ctas:
        return requested_ctas

    found = sorted(
        {row_cta(r) for r in ooverlap_rows(rows) if row_cta(r) is not None}
    )
    return found if found else [None]


def format_cta(cta):
    return "ambient" if cta is None else str(cta)


def label_for_row(row):
    collective = row_collective(row)
    backend = row_backend(row)

    if backend == "nccl":
        return f"{collective}:nccl"

    mode = row_mode(row) or "unknown_mode"
    return f"{collective}:{mode}"


def metric_value(row, metric):
    return float(row[metric])


def group_series(rows, metric, cta=None, include_nccl=False, nccl_by_collective=None):
    series = {}

    for row in ooverlap_rows(rows):
        if row_cta(row) != cta:
            continue

        label = label_for_row(row)
        series.setdefault(label, []).append(
            (int(row["numel"]), metric_value(row, metric))
        )

    if include_nccl and nccl_by_collective:
        for collective, points in nccl_by_collective.items():
            label = f"{collective}:nccl"
            series[label] = [(int(x), float(y[metric])) for x, y in points.items()]

    for label in series:
        series[label].sort(key=lambda p: p[0])

    return series


def make_nccl_lookup(rows):
    lookup = {}

    for row in nccl_rows(rows):
        collective = row_collective(row)
        numel = int(row["numel"])
        lookup.setdefault(collective, {})[numel] = row

    return lookup


def plot_metric(rows, ctas, metric, ylabel, title, out_path, include_nccl=True):
    nccl_lookup = make_nccl_lookup(rows)

    fig, axes = plt.subplots(
        1,
        len(ctas),
        figsize=(5.8 * len(ctas), 4.4),
        squeeze=False,
        sharey=False,
    )
    axes = axes[0]

    for ax, cta in zip(axes, ctas):
        series = group_series(
            rows,
            metric,
            cta=cta,
            include_nccl=include_nccl,
            nccl_by_collective=nccl_lookup,
        )

        for label, points in sorted(series.items()):
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            ax.plot(xs, ys, marker="o", label=label)

        ax.set_xscale("log", base=2)
        ax.set_title(f"max_ctas={format_cta(cta)}")
        ax.set_xlabel("numel")
        ax.set_ylabel(ylabel)
        ax.grid(True, which="both", alpha=0.3)

    handles, labels = axes[-1].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="lower center",
            ncol=min(4, max(1, len(labels))),
            fontsize=9,
        )

    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0.15, 1, 0.92))
    fig.savefig(out_path, dpi=180)
    print(f"[plot] wrote {out_path}")


def speedup_rows(rows):
    nccl = make_nccl_lookup(rows)
    out = []

    for row in ooverlap_rows(rows):
        collective = row_collective(row)
        numel = int(row["numel"])

        nccl_row = nccl.get(collective, {}).get(numel)
        if not nccl_row:
            continue

        nccl_ms = float(nccl_row["avg_ms"])
        row_ms = float(row["avg_ms"])
        speedup = nccl_ms / row_ms if row_ms > 0.0 else 0.0

        r = dict(row)
        r["speedup_over_nccl"] = speedup
        r["nccl_avg_ms"] = nccl_ms
        r["nccl_latency_us"] = float(nccl_row["latency_us"])
        r["nccl_effective_gbps_aggregate_2gpu"] = float(
            nccl_row["effective_gbps_aggregate_2gpu"]
        )
        out.append(r)

    return out


def plot_speedup(rows, ctas, out_path):
    rows = speedup_rows(rows)

    fig, axes = plt.subplots(
        1,
        len(ctas),
        figsize=(5.8 * len(ctas), 4.4),
        squeeze=False,
        sharey=False,
    )
    axes = axes[0]

    for ax, cta in zip(axes, ctas):
        series = {}

        for row in rows:
            if row_cta(row) != cta:
                continue

            label = label_for_row(row)
            series.setdefault(label, []).append(
                (int(row["numel"]), float(row["speedup_over_nccl"]))
            )

        for label in series:
            series[label].sort(key=lambda p: p[0])

        for label, points in sorted(series.items()):
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            ax.plot(xs, ys, marker="o", label=label)

        ax.axhline(1.0, linestyle="--", linewidth=1)
        ax.set_xscale("log", base=2)
        ax.set_title(f"max_ctas={format_cta(cta)}")
        ax.set_xlabel("numel")
        ax.set_ylabel("speedup over NCCL")
        ax.grid(True, which="both", alpha=0.3)

    handles, labels = axes[-1].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="lower center",
            ncol=min(4, max(1, len(labels))),
            fontsize=9,
        )

    fig.suptitle("Speedup over NCCL vs size by CTA constraint")
    fig.tight_layout(rect=(0, 0.15, 1, 0.92))
    fig.savefig(out_path, dpi=180)
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
    nccl = make_nccl_lookup(rows)

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

        print(f"{collective} numel={numel} max_ctas={format_cta(cta)}")

        for name, row in [("perf", perf), ("eff ", eff)]:
            if not row:
                continue

            speedup_text = ""
            nccl_row = make_nccl_lookup(ok_rows_global).get(collective, {}).get(numel)
            if nccl_row:
                speedup = float(nccl_row["avg_ms"]) / float(row["avg_ms"])
                speedup_text = f" speedup_vs_nccl={speedup:.3f}x"

            print(
                f"  {name}: "
                f"{float(row['effective_gbps_aggregate_2gpu']):8.2f} GB/s "
                f"{float(row['latency_us']):8.2f} us"
                f"{speedup_text}"
            )

        if perf and eff:
            perf_bw = float(perf["effective_gbps_aggregate_2gpu"])
            eff_bw = float(eff["effective_gbps_aggregate_2gpu"])
            ratio = eff_bw / perf_bw if perf_bw > 0.0 else 0.0
            print(f"  eff/perf bandwidth ratio: {ratio:.3f}")


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

    global ok_rows_global
    ok_rows_global = ok_rows(rows)

    csv_path = out_dir / "tma_efficiency_vs_best.csv"
    write_csv(rows, csv_path)

    print_summary(rows)

    ctas = available_ctas(rows, args.ctas)

    plot_metric(
        rows,
        ctas,
        "effective_gbps_aggregate_2gpu",
        "aggregate bandwidth, 2 GPU (GB/s)",
        "Public tuned bandwidth vs size by CTA constraint",
        out_dir / "bandwidth_by_cta.png",
        include_nccl=not args.no_nccl,
    )

    plot_metric(
        rows,
        ctas,
        "latency_us",
        "latency (us)",
        "Public tuned latency vs size by CTA constraint",
        out_dir / "latency_by_cta.png",
        include_nccl=not args.no_nccl,
    )

    if not args.no_nccl:
        plot_speedup(
            rows,
            ctas,
            out_dir / "speedup_over_nccl_by_cta.png",
        )


if __name__ == "__main__":
    ok_rows_global = []
    main()
