#!/usr/bin/env python3
"""Plot external-P2P collective latency and bandwidth from the sweep TSV."""

from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


COLLECTIVES = ("allreduce", "reduce_scatter", "all_gather")
COLLECTIVE_TITLES = {
    "allreduce": "All-Reduce",
    "reduce_scatter": "Reduce-Scatter",
    "all_gather": "All-Gather",
}

# OOVERLAP_EXTERNAL_PLOT_READABILITY_V1
# Keep the existing visual design, but make the curves survive paper scaling
# slightly better without making the figure look heavy.
PLOT_LINEWIDTH = 1.8
PLOT_MARKERSIZE = 6.5
PLOT_MARKEREDGEWIDTH = 0.8


class PlotInputError(ValueError):
    pass


def parse_optional_int(value: object) -> Optional[int]:
    text = "" if value is None else str(value).strip()
    if not text:
        return None
    try:
        return int(float(text))
    except ValueError as exc:
        raise PlotInputError(f"expected an integer, got {value!r}") from exc


def parse_float(value: object, column: str) -> float:
    text = "" if value is None else str(value).strip()
    if not text:
        return float("nan")
    try:
        parsed = float(text)
    except ValueError as exc:
        raise PlotInputError(f"invalid {column} value: {value!r}") from exc
    return parsed


def format_size_bytes(size_bytes: int) -> str:
    size_bytes = int(size_bytes)
    for scale, suffix in ((1024**3, "G"), (1024**2, "M"), (1024, "K")):
        if size_bytes >= scale:
            value = size_bytes / scale
            return f"{int(value)}{suffix}" if value.is_integer() else f"{value:.1f}{suffix}"
    return f"{size_bytes}B"


def infer_tp(path: Path) -> Optional[int]:
    for part in reversed(path.parts):
        match = re.fullmatch(r"tp(\d+)", part.lower())
        if match:
            value = int(match.group(1))
            return value if value > 0 else None
    return None


def load_rows(path: Path) -> List[Dict[str, object]]:
    if not path.is_file():
        raise PlotInputError(f"input text file does not exist: {path}")

    required = {
        "mode",
        "metric_set",
        "collective",
        "cta_limit",
        "bytes",
        "ooverlap_latency_us",
        "nccl_latency_us",
        "nccl_symmetric_latency_us",
        "ooverlap_bandwidth_gbps",
        "nccl_bandwidth_gbps",
        "nccl_symmetric_bandwidth_gbps",
    }

    rows: List[Dict[str, object]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        fields = set(reader.fieldnames or [])
        missing = sorted(required - fields)
        if missing:
            raise PlotInputError(
                f"{path} is missing required columns: {', '.join(missing)}"
            )

        for raw in reader:
            if str(raw.get("mode", "")).strip() != "bench":
                continue
            collective = str(raw.get("collective", "")).strip()
            metric_set = str(raw.get("metric_set", "")).strip()
            if collective not in COLLECTIVES or metric_set not in ("latency", "bandwidth"):
                continue

            rows.append(
                {
                    "metric_set": metric_set,
                    "collective": collective,
                    "cta_limit": parse_optional_int(raw.get("cta_limit")),
                    "bytes": int(parse_float(raw.get("bytes"), "bytes")),
                    "t_ccl_latency_us": parse_float(
                        raw.get("ooverlap_latency_us"), "ooverlap_latency_us"
                    ),
                    "nccl_latency_us": parse_float(
                        raw.get("nccl_latency_us"), "nccl_latency_us"
                    ),
                    "nccl_symmetric_latency_us": parse_float(
                        raw.get("nccl_symmetric_latency_us"),
                        "nccl_symmetric_latency_us",
                    ),
                    "t_ccl_bandwidth_gbps": parse_float(
                        raw.get("ooverlap_bandwidth_gbps"),
                        "ooverlap_bandwidth_gbps",
                    ),
                    "nccl_bandwidth_gbps": parse_float(
                        raw.get("nccl_bandwidth_gbps"), "nccl_bandwidth_gbps"
                    ),
                    "nccl_symmetric_bandwidth_gbps": parse_float(
                        raw.get("nccl_symmetric_bandwidth_gbps"),
                        "nccl_symmetric_bandwidth_gbps",
                    ),
                }
            )

    if not rows:
        raise PlotInputError(f"no benchmark latency or bandwidth rows found in {path}")
    return rows


def finite_positive(values: Iterable[float]) -> bool:
    values = list(values)
    return bool(values) and all(math.isfinite(value) and value > 0.0 for value in values)


def group_rows(
    rows: List[Dict[str, object]], metric: str
) -> Dict[str, Dict[Optional[int], List[Dict[str, object]]]]:
    grouped: Dict[str, Dict[Optional[int], List[Dict[str, object]]]] = {
        collective: {} for collective in COLLECTIVES
    }
    for row in rows:
        if row["metric_set"] != metric:
            continue
        collective = str(row["collective"])
        cta_limit = row["cta_limit"]
        grouped[collective].setdefault(cta_limit, []).append(row)

    for collective in COLLECTIVES:
        for cta_limit in grouped[collective]:
            grouped[collective][cta_limit].sort(key=lambda row: int(row["bytes"]))
        if not grouped[collective]:
            raise PlotInputError(f"no {metric} rows found for {collective}")
    return grouped


def cta_suffix(cta_limit: Optional[int], show: bool) -> str:
    if not show:
        return ""
    return " unrestricted" if cta_limit is None else f" ({cta_limit} CTAs)"


def plot_combined(
    rows: List[Dict[str, object]],
    tp: int,
    output_path: Path,
) -> None:
    del tp  # TP belongs in the paper caption, not in the figure title.

    grouped_by_metric = {
        "bandwidth": group_rows(rows, "bandwidth"),
        "latency": group_rows(rows, "latency"),
    }
    cta_values = sorted(
        {
            cta
            for grouped in grouped_by_metric.values()
            for collective_rows in grouped.values()
            for cta in collective_rows
        },
        key=lambda value: (-1 if value is None else value),
    )
    show_cta_in_legend = len(cta_values) > 1

    metric_specs = (
        (
            "bandwidth",
            "t_ccl_bandwidth_gbps",
            "nccl_bandwidth_gbps",
            "nccl_symmetric_bandwidth_gbps",
            "AlgoBW (GB/s)",
        ),
        (
            "latency",
            "t_ccl_latency_us",
            "nccl_latency_us",
            "nccl_symmetric_latency_us",
            "Latency (µs)",
        ),
    )

    fig, axes = plt.subplots(
        nrows=2,
        ncols=3,
        figsize=(15, 7.6),
        sharex=False,
    )

    legend_by_label = {}

    for row_idx, (
        metric,
        t_ccl_key,
        nccl_key,
        symmetric_key,
        ylabel,
    ) in enumerate(metric_specs):
        grouped = grouped_by_metric[metric]

        for col_idx, collective in enumerate(COLLECTIVES):
            axis = axes[row_idx][col_idx]
            collective_groups = grouped[collective]
            first_rows = next(iter(collective_groups.values()))

            for cta_limit, series in sorted(
                collective_groups.items(),
                key=lambda item: (-1 if item[0] is None else item[0]),
            ):
                x_values = [int(row["bytes"]) for row in series]
                t_ccl_values = [float(row[t_ccl_key]) for row in series]
                nccl_values = [float(row[nccl_key]) for row in series]
                symmetric_values = [float(row[symmetric_key]) for row in series]
                suffix = cta_suffix(cta_limit, show_cta_in_legend)

                axis.plot(
                    x_values,
                    t_ccl_values,
                    marker="o",
                    linewidth=PLOT_LINEWIDTH,
                    markersize=PLOT_MARKERSIZE,
                    markeredgewidth=PLOT_MARKEREDGEWIDTH,
                    label=f"T-CCL{suffix}",
                )
                axis.plot(
                    x_values,
                    nccl_values,
                    marker="s",
                    linestyle="--",
                    linewidth=PLOT_LINEWIDTH,
                    markersize=PLOT_MARKERSIZE,
                    markeredgewidth=PLOT_MARKEREDGEWIDTH,
                    label=f"NCCL{suffix}",
                )
                if finite_positive(symmetric_values):
                    axis.plot(
                        x_values,
                        symmetric_values,
                        marker="^",
                        linestyle=":",
                        linewidth=PLOT_LINEWIDTH,
                        markersize=PLOT_MARKERSIZE,
                        markeredgewidth=PLOT_MARKEREDGEWIDTH,
                        label=f"symmetric NCCL{suffix}",
                    )

            for handle, label in zip(*axis.get_legend_handles_labels()):
                legend_by_label.setdefault(label, handle)

            x_ticks = [int(row["bytes"]) for row in first_rows]
            labeled_tick_indices = set(range(0, len(x_ticks), 1))
            if x_ticks:
                labeled_tick_indices.add(len(x_ticks) - 1)

            axis.set_xscale("log", base=2)
            axis.set_xticks(x_ticks)
            axis.set_xticklabels(
                [
                    format_size_bytes(value)
                    if index in labeled_tick_indices
                    else ""
                    for index, value in enumerate(x_ticks)
                ],
                rotation=25,
                ha="right",
            )
            axis.grid(True, which="both", linestyle="--", alpha=0.35)

            if row_idx == 0:
                axis.set_title(COLLECTIVE_TITLES[collective], fontsize=12)
            if col_idx == 0:
                axis.set_ylabel(ylabel)

    fig.supxlabel("Buffer size", y=0.022)

    legend_labels = list(legend_by_label)
    legend_handles = [legend_by_label[label] for label in legend_labels]
    fig.legend(
        legend_handles,
        legend_labels,
        loc="upper center",
        ncol=min(len(legend_labels), 6),
        bbox_to_anchor=(0.5, 0.988),
        frameon=False,
        handlelength=2.0,
        columnspacing=1.25,
    )
    fig.subplots_adjust(
        left=0.06,
        right=0.995,
        bottom=0.12,
        top=0.90,
        wspace=0.18,
        hspace=0.22,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] wrote {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot external-P2P collective latency and bandwidth in one figure"
    )
    parser.add_argument("--text", required=True, help="Sweep tab-separated text file")
    parser.add_argument(
        "--tp",
        type=int,
        default=None,
        help="Tensor-parallel size. If omitted, infer it from a tpN path component.",
    )
    parser.add_argument(
        "--out-prefix",
        default=None,
        help="Output prefix. Defaults to the input path without its .txt suffix.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    text_path = Path(args.text).expanduser().resolve()
    tp = args.tp if args.tp is not None else infer_tp(text_path)
    if tp is None or tp < 2:
        raise PlotInputError("--tp must be at least 2 or inferable from a tpN path")

    out_prefix = (
        Path(args.out_prefix).expanduser().resolve()
        if args.out_prefix
        else text_path.with_suffix("")
    )
    rows = load_rows(text_path)

    plot_combined(rows, tp, Path(f"{out_prefix}.png"))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except PlotInputError as exc:
        raise SystemExit(f"error: {exc}")
