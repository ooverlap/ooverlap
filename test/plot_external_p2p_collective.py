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

# OOVERLAP_EXTERNAL_PLOT_PAPER_3X4_READABILITY_V1
# These values are used only by the dense 3x4 paper layout. They are slightly
# stronger than the standalone-figure defaults so curves remain distinguishable
# after the figure is embedded at two-column paper width.
PAPER_LINEWIDTH = 2.25
PAPER_MARKERSIZE = 7.4
PAPER_MARKEREDGEWIDTH = 1.0
PAPER_TICK_FONTSIZE = 8.0
PAPER_LABEL_FONTSIZE = 9.0
PAPER_TITLE_FONTSIZE = 10.0
PAPER_LEGEND_FONTSIZE = 8.5


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


# OOVERLAP_EXTERNAL_PLOT_PAIRED_TP_V1
METRIC_SPECS = (
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


def prepare_plot_data(rows: List[Dict[str, object]]):
    grouped_by_metric = {
        "bandwidth": group_rows(rows, "bandwidth"),
        "latency": group_rows(rows, "latency"),
    }
    cta_values = {
        cta
        for grouped in grouped_by_metric.values()
        for collective_rows in grouped.values()
        for cta in collective_rows
    }
    return grouped_by_metric, len(cta_values) > 1


def plot_tp_group(
    rows: List[Dict[str, object]],
    axes,
    legend_by_label: Dict[str, object],
    *,
    show_ylabels: bool,
) -> None:
    grouped_by_metric, show_cta_in_legend = prepare_plot_data(rows)

    for row_idx, (
        metric,
        t_ccl_key,
        nccl_key,
        symmetric_key,
        ylabel,
    ) in enumerate(METRIC_SPECS):
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
            axis.set_xscale("log", base=2)
            axis.set_xticks(x_ticks)
            axis.set_xticklabels(
                [format_size_bytes(value) for value in x_ticks],
                rotation=90,
                ha="center",
            )
            axis.grid(True, which="both", linestyle="--", alpha=0.35)

            if row_idx == 0:
                axis.set_title(COLLECTIVE_TITLES[collective], fontsize=12)
            if show_ylabels and col_idx == 0:
                axis.set_ylabel(ylabel)


def add_figure_legend(fig, legend_by_label: Dict[str, object], y: float) -> None:
    legend_labels = list(legend_by_label)
    legend_handles = [legend_by_label[label] for label in legend_labels]
    fig.legend(
        legend_handles,
        legend_labels,
        loc="upper center",
        ncol=min(len(legend_labels), 6),
        bbox_to_anchor=(0.5, y),
        frameon=False,
        handlelength=2.0,
        columnspacing=1.25,
    )


def save_figure(fig, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] wrote {output_path}")


def plot_combined(
    rows: List[Dict[str, object]],
    tp: int,
    output_path: Path,
) -> None:
    del tp  # TP belongs in the paper caption, not in the standalone figure title.

    fig, axes = plt.subplots(
        nrows=2,
        ncols=3,
        figsize=(15, 7.6),
        sharex=False,
    )
    legend_by_label: Dict[str, object] = {}
    plot_tp_group(rows, axes, legend_by_label, show_ylabels=True)

    fig.supxlabel("Buffer size", y=0.022)
    add_figure_legend(fig, legend_by_label, 0.988)
    fig.subplots_adjust(
        left=0.06,
        right=0.995,
        bottom=0.12,
        top=0.90,
        wspace=0.18,
        hspace=0.22,
    )
    save_figure(fig, output_path)


def plot_paired_tp(
    left_rows: List[Dict[str, object]],
    left_tp: int,
    right_rows: List[Dict[str, object]],
    right_tp: int,
    output_path: Path,
) -> None:
    fig = plt.figure(figsize=(15.5, 5.4))
    outer = fig.add_gridspec(nrows=1, ncols=2, wspace=0.065)

    group_axes = []
    for group_idx in range(2):
        inner = outer[group_idx].subgridspec(
            nrows=2,
            ncols=3,
            wspace=0.16,
            hspace=0.22,
        )
        group_axes.append(
            [
                [fig.add_subplot(inner[row_idx, col_idx]) for col_idx in range(3)]
                for row_idx in range(2)
            ]
        )

    legend_by_label: Dict[str, object] = {}
    plot_tp_group(
        left_rows,
        group_axes[0],
        legend_by_label,
        show_ylabels=False,
    )
    plot_tp_group(
        right_rows,
        group_axes[1],
        legend_by_label,
        show_ylabels=False,
    )

    fig.subplots_adjust(
        left=0.045,
        right=0.995,
        bottom=0.17,
        top=0.80,
    )

    for axes, tp in zip(group_axes, (left_tp, right_tp)):
        group_left = axes[0][0].get_position().x0
        group_right = axes[0][-1].get_position().x1
        fig.text(
            (group_left + group_right) / 2.0,
            0.865,
            f"TP={tp}",
            ha="center",
            va="center",
            fontsize=13,
            fontweight="semibold",
        )

    fig.text(
        0.012,
        0.64,
        "AlgoBW (GB/s)",
        rotation=90,
        ha="center",
        va="center",
    )
    fig.text(
        0.012,
        0.315,
        "Latency (µs)",
        rotation=90,
        ha="center",
        va="center",
    )
    fig.supxlabel("Buffer size", y=0.025)
    add_figure_legend(fig, legend_by_label, 0.995)
    save_figure(fig, output_path)


# OOVERLAP_EXTERNAL_PLOT_PAPER_3X4_V1
# OOVERLAP_EXTERNAL_PLOT_PAPER_3X4_LABELS_V2
def paper_tick_values(values: Iterable[int]) -> List[int]:
    """Return every distinct measured buffer size in increasing order."""
    return sorted({int(value) for value in values})
# OOVERLAP_EXTERNAL_PLOT_PAPER_3X4_Y_AUTOSCALE_V6
# OOVERLAP_EXTERNAL_PLOT_PAPER_3X4_Y_PADDING_V7
def configure_paper_y_axis(axis, values: Iterable[float]) -> None:
    """Add balanced space below the minimum and above the maximum.

    Keep one labelled major tick outside the data range on each side, then add
    a small fraction of one tick interval beyond those outer ticks. This keeps
    extreme markers away from the top and bottom spines without forcing zero
    into panels whose measurements live far above zero.
    """
    finite_values = [float(value) for value in values if math.isfinite(float(value))]
    if not finite_values:
        return

    min_value = min(finite_values)
    max_value = max(finite_values)

    # Start from Matplotlib's normal data-driven scale so its locator chooses a
    # sensible tick interval for this particular panel.
    axis.relim()
    axis.autoscale_view(scalex=False, scaley=True)

    major_ticks = sorted(
        {
            float(tick)
            for tick in axis.get_yticks()
            if math.isfinite(float(tick))
        }
    )
    if len(major_ticks) < 2:
        return

    positive_steps = [
        right - left
        for left, right in zip(major_ticks, major_ticks[1:])
        if right - left > 0.0
    ]
    if not positive_steps:
        return
    step = min(positive_steps)

    epsilon = 1.0e-10 * max(1.0, abs(min_value), abs(max_value))

    lower_tick = next(
        (tick for tick in reversed(major_ticks) if tick < min_value - epsilon),
        None,
    )
    if lower_tick is None:
        lower_tick = major_ticks[0]
        while lower_tick >= min_value - epsilon:
            lower_tick -= step
        major_ticks.append(lower_tick)

    upper_tick = next(
        (tick for tick in major_ticks if tick > max_value + epsilon),
        None,
    )
    if upper_tick is None:
        upper_tick = major_ticks[-1]
        while upper_tick <= max_value + epsilon:
            upper_tick += step
        major_ticks.append(upper_tick)

    visible_ticks = sorted(
        tick
        for tick in set(major_ticks)
        if tick >= lower_tick - epsilon and tick <= upper_tick + epsilon
    )

    # Leave a little whitespace outside the outer labelled grid lines. Increase
    # this value slightly (for example, to 0.20) for more padding.
    edge_padding = 0.15 * step
    axis.set_ylim(lower_tick - edge_padding, upper_tick + edge_padding)
    axis.set_yticks(visible_ticks)
def plot_paper_panel(
    grouped_by_metric,
    show_cta_in_legend: bool,
    metric_spec,
    collective: str,
    axis,
    legend_by_label: Dict[str, object],
    *,
    show_xlabels: bool,
) -> None:
    """Plot one collective for one TP/metric column."""
    metric, t_ccl_key, nccl_key, symmetric_key, _ylabel = metric_spec
    collective_groups = grouped_by_metric[metric][collective]
    plotted_y_values: List[float] = []

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
            linewidth=PAPER_LINEWIDTH,
            markersize=PAPER_MARKERSIZE,
            markeredgewidth=PAPER_MARKEREDGEWIDTH,
            label=f"T-CCL{suffix}",
        )
        axis.plot(
            x_values,
            nccl_values,
            marker="s",
            linestyle="--",
            linewidth=PAPER_LINEWIDTH,
            markersize=PAPER_MARKERSIZE,
            markeredgewidth=PAPER_MARKEREDGEWIDTH,
            label=f"NCCL{suffix}",
        )
        plotted_y_values.extend(t_ccl_values)
        plotted_y_values.extend(nccl_values)

        if finite_positive(symmetric_values):
            axis.plot(
                x_values,
                symmetric_values,
                marker="^",
                linestyle=":",
                linewidth=PAPER_LINEWIDTH,
                markersize=PAPER_MARKERSIZE,
                markeredgewidth=PAPER_MARKEREDGEWIDTH,
                label=f"Symmetric NCCL{suffix}",
            )
            plotted_y_values.extend(symmetric_values)

    for handle, label in zip(*axis.get_legend_handles_labels()):
        legend_by_label.setdefault(label, handle)

    all_x_ticks = paper_tick_values(
        int(row["bytes"])
        for series in collective_groups.values()
        for row in series
    )
    axis.set_xscale("log", base=2)
    axis.set_xticks(all_x_ticks)
    if show_xlabels:
        axis.set_xticklabels(
            [format_size_bytes(value) for value in all_x_ticks],
            rotation=90,
            ha="center",
            va="top",
        )
        axis.tick_params(
            axis="x",
            which="major",
            labelsize=PAPER_TICK_FONTSIZE,
            pad=7,
            direction="out",
        )
    else:
        axis.tick_params(axis="x", which="both", labelbottom=False)

    configure_paper_y_axis(axis, plotted_y_values)
    axis.tick_params(
        axis="y",
        labelsize=PAPER_TICK_FONTSIZE,
        pad=1,
        direction="out",
    )
    axis.grid(True, which="both", linestyle="--", linewidth=0.65, alpha=0.32)
    axis.margins(x=0.02)


def add_paper_figure_legend(legend_axis, legend_by_label: Dict[str, object]) -> None:
    """Draw the shared legend in its own row."""
    legend_axis.axis("off")
    legend_labels = list(legend_by_label)
    legend_handles = [legend_by_label[label] for label in legend_labels]
    legend_axis.legend(
        legend_handles,
        legend_labels,
        loc="center",
        ncol=min(3, len(legend_labels)),
        frameon=False,
        fontsize=PAPER_LEGEND_FONTSIZE,
        handlelength=2.25,
        handletextpad=0.55,
        columnspacing=1.05,
        borderaxespad=0.0,
    )


# OOVERLAP_EXTERNAL_PLOT_PAPER_3X4_SPACING_V4
# OOVERLAP_EXTERNAL_PLOT_PAPER_3X4_POLISH_V5
# OOVERLAP_EXTERNAL_PLOT_PAPER_3X4_METRIC_YLABELS_V8
def plot_paired_tp_3x4(
    left_rows: List[Dict[str, object]],
    left_tp: int,
    right_rows: List[Dict[str, object]],
    right_tp: int,
    output_path: Path,
) -> None:
    """Plot collective rows against TP/metric columns.

    Data rows:
      1. All-Reduce
      2. Reduce-Scatter
      3. All-Gather

    Data columns:
      1. smaller TP latency
      2. smaller TP bandwidth
      3. larger TP latency
      4. larger TP bandwidth
    """
    tp_groups = sorted(
        ((int(left_tp), left_rows), (int(right_tp), right_rows)),
        key=lambda item: item[0],
    )
    (first_tp, first_rows), (second_tp, second_rows) = tp_groups

    prepared_groups = []
    for tp, rows in ((first_tp, first_rows), (second_tp, second_rows)):
        grouped_by_metric, show_cta_in_legend = prepare_plot_data(rows)
        prepared_groups.append((tp, grouped_by_metric, show_cta_in_legend))

    bandwidth_spec = METRIC_SPECS[0]
    latency_spec = METRIC_SPECS[1]

    # Within each TP pair, latency comes first and bandwidth second. Metric names
    # are rendered as vertical y-axis labels on every panel rather than as titles.
    columns = (
        (prepared_groups[0], latency_spec),
        (prepared_groups[0], bandwidth_spec),
        (prepared_groups[1], latency_spec),
        (prepared_groups[1], bandwidth_spec),
    )

    fig = plt.figure(figsize=(12.8, 8.0))
    grid = fig.add_gridspec(
        nrows=5,
        ncols=4,
        height_ratios=(0.18, 0.15, 1.0, 1.0, 1.0),
        left=0.080,
        right=0.998,
        bottom=0.130,
        top=0.995,
        # The existing spacing is retained; compact label padding keeps the
        # per-panel metric labels inside the available inter-column gutters.
        wspace=0.20,
        hspace=0.10,
    )

    legend_axis = fig.add_subplot(grid[0, :])
    first_tp_axis = fig.add_subplot(grid[1, 0:2])
    second_tp_axis = fig.add_subplot(grid[1, 2:4])
    for group_axis, tp in ((first_tp_axis, first_tp), (second_tp_axis, second_tp)):
        group_axis.axis("off")
        group_axis.text(
            0.5,
            0.5,
            f"TP={tp}",
            ha="center",
            va="center",
            fontsize=PAPER_TITLE_FONTSIZE + 1.5,
            fontweight="semibold",
        )

    axes = []
    for row_idx, collective in enumerate(COLLECTIVES):
        grid_row = row_idx + 2
        axis_row = []
        for col_idx, (prepared, metric_spec) in enumerate(columns):
            _tp, grouped_by_metric, show_cta_in_legend = prepared
            axis = fig.add_subplot(grid[grid_row, col_idx])
            plot_paper_panel(
                grouped_by_metric,
                show_cta_in_legend,
                metric_spec,
                collective,
                axis,
                legend_by_label={},
                show_xlabels=(row_idx == len(COLLECTIVES) - 1),
            )

            metric_ylabel = metric_spec[4]
            axis.set_ylabel(
                metric_ylabel,
                rotation=90,
                ha="center",
                va="center",
                fontsize=PAPER_LABEL_FONTSIZE,
                fontweight="normal",
                labelpad=5,
            )

            # Column 1 needs both the metric label and the collective row label.
            # Keep the metric label next to its axis and place the collective name
            # farther left with a deliberate gap between the two vertical labels.
            if col_idx == 0:
                axis.annotate(
                    COLLECTIVE_TITLES[collective],
                    xy=(0.0, 0.5),
                    xycoords="axes fraction",
                    xytext=(-72, 0),
                    textcoords="offset points",
                    rotation=90,
                    ha="center",
                    va="center",
                    fontsize=PAPER_LABEL_FONTSIZE,
                    fontweight="medium",
                    annotation_clip=False,
                )

            axis_row.append(axis)
        axes.append(axis_row)

    legend_by_label: Dict[str, object] = {}
    for axis_row in axes:
        for axis in axis_row:
            for handle, label in zip(*axis.get_legend_handles_labels()):
                legend_by_label.setdefault(label, handle)
    add_paper_figure_legend(legend_axis, legend_by_label)

    # One figure-wide label is sufficient because every column uses buffer size.
    fig.supxlabel(
        "Buffer size",
        x=(0.080 + 0.998) / 2.0,
        y=0.018,
        fontsize=PAPER_LABEL_FONTSIZE,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    pdf_path = output_path.with_suffix(".pdf")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] wrote {output_path}")
    print(f"[plot] wrote {pdf_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot one external-P2P TP sweep or combine two TP sweeps in one figure"
        )
    )
    parser.add_argument(
        "--text",
        default=None,
        help="Single-TP sweep tab-separated text file.",
    )
    parser.add_argument(
        "--tp",
        type=int,
        default=None,
        help="Single-TP size. If omitted, infer it from a tpN path component.",
    )
    parser.add_argument(
        "--left-text",
        default=None,
        help="Left TP-group sweep text file for paired mode.",
    )
    parser.add_argument(
        "--left-tp",
        type=int,
        default=None,
        help="Left TP-group size. If omitted, infer it from the path.",
    )
    parser.add_argument(
        "--right-text",
        default=None,
        help="Right TP-group sweep text file for paired mode.",
    )
    parser.add_argument(
        "--right-tp",
        type=int,
        default=None,
        help="Right TP-group size. If omitted, infer it from the path.",
    )
    parser.add_argument(
        "--paired-layout",
        choices=["side-by-side", "paper-3x4", "paper-4x3"],
        default="side-by-side",
        help=(
            "Paired-TP layout. 'side-by-side' preserves the existing 2x6-style "
            "figure; 'paper-3x4' uses collective rows and TP/metric columns. "
            "The old 'paper-4x3' spelling is accepted as a compatibility alias."
        ),
    )
    parser.add_argument(
        "--out-prefix",
        default=None,
        help=(
            "Output prefix. In single mode, defaults to the input path without "
            "its suffix. In paired mode, defaults beside the left input."
        ),
    )
    return parser.parse_args()


def resolve_tp(path: Path, explicit_tp: Optional[int], option_name: str) -> int:
    tp = explicit_tp if explicit_tp is not None else infer_tp(path)
    if tp is None or tp < 2:
        raise PlotInputError(
            f"{option_name} must be at least 2 or inferable from a tpN path"
        )
    return tp


def main() -> int:
    args = parse_args()

    single_options_used = args.text is not None or args.tp is not None
    paired_options_used = any(
        value is not None
        for value in (
            args.left_text,
            args.left_tp,
            args.right_text,
            args.right_tp,
        )
    )

    if single_options_used and paired_options_used:
        raise PlotInputError(
            "use either --text/--tp or --left-text/--right-text, not both"
        )

    if args.text is not None:
        text_path = Path(args.text).expanduser().resolve()
        tp = resolve_tp(text_path, args.tp, "--tp")
        out_prefix = (
            Path(args.out_prefix).expanduser().resolve()
            if args.out_prefix
            else text_path.with_suffix("")
        )
        plot_combined(load_rows(text_path), tp, Path(f"{out_prefix}.png"))
        return 0

    if args.tp is not None:
        raise PlotInputError("--tp requires --text")

    if not paired_options_used:
        raise PlotInputError(
            "provide --text for single mode or both --left-text and --right-text "
            "for paired mode"
        )

    if args.left_text is None or args.right_text is None:
        raise PlotInputError(
            "paired mode requires both --left-text and --right-text"
        )

    left_path = Path(args.left_text).expanduser().resolve()
    right_path = Path(args.right_text).expanduser().resolve()
    left_tp = resolve_tp(left_path, args.left_tp, "--left-tp")
    right_tp = resolve_tp(right_path, args.right_tp, "--right-tp")

    paper_layout_requested = args.paired_layout in ("paper-3x4", "paper-4x3")
    default_layout_suffix = "paper_3x4" if paper_layout_requested else "combined"
    out_prefix = (
        Path(args.out_prefix).expanduser().resolve()
        if args.out_prefix
        else left_path.with_name(
            f"{left_path.stem}_tp{left_tp}_tp{right_tp}_{default_layout_suffix}"
        )
    )

    plot_function = plot_paired_tp_3x4 if paper_layout_requested else plot_paired_tp
    plot_function(
        load_rows(left_path),
        left_tp,
        load_rows(right_path),
        right_tp,
        Path(f"{out_prefix}.png"),
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except PlotInputError as exc:
        raise SystemExit(f"error: {exc}")
