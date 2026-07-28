#!/usr/bin/env python3
"""
Plot SM90 operator-level normalized speedup from eval_sm90/evaluation_result.json.

Layout:
  - rows: unique M values
  - columns: unique N values
  - x-axis inside each panel: K values
  - bars: no-overlap baseline, NCCL overlap, T-CCL overlap

Outputs:
  - sm90_operator_speedup_by_shape.png
  - sm90_operator_speedup_by_shape.pdf
  - sm90_operator_speedup_by_shape_data.json
  - sm90_operator_speedup_by_shape_data.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


@dataclass(frozen=True)
class ShapePlot:
    m: int
    n: int
    k: Union[str, int, List[int], Tuple[int, ...]] = "x"
    label: str = ""


# Keep this list for explicit report figures.
# Use --all-shapes to ignore this and plot every measured M,N in the JSON.
PLOT_SHAPES: List[ShapePlot] = [
    ShapePlot(16384, 2048, "x"),
    ShapePlot(16384, 4096, "x"),
    ShapePlot(16384, 8192, "x"),
    ShapePlot(32768, 2048, "x"),
    ShapePlot(32768, 4096, "x"),
    ShapePlot(32768, 8192, "x"),
    # ShapePlot(49152, 4096, "x"),
    # ShapePlot(49152, 8192, "x"),
]

DEFAULT_JSON = "results/eval_sm90/evaluation_result.json"
DEFAULT_OUT_DIR = "results/eval_sm90/operator_speedup"
DEFAULT_PLOT_NAME = "operator_overlap_speedup_by_shape"

DEFAULT_BASELINE_SOURCE = "nccl_cublas"
DEFAULT_BASELINE_AGGREGATION = "mean"
DEFAULT_TEST_MODE = "both"


# OOVERLAP_FLASHOVERLAP_PLOT_FONTS_V1
AXIS_LABEL_FONTSIZE = 15
TICK_LABEL_FONTSIZE = 13
LEGEND_FONTSIZE = 13
PANEL_LABEL_FONTSIZE = 12
SPEEDUP_LABEL_FONTSIZE = 10


# OOVERLAP_BASELINE_AGGREGATION_V1
@dataclass
class MetricPick:
    ms: float
    scenario_id: str
    test_mode: str
    backend: str
    comm_sm_slack: Optional[int]
    cseg: Optional[List[int]]


@dataclass
class ShapeKResult:
    m: int
    n: int
    k: int
    baseline_ms: Optional[float]
    nccl_overlap_ms: Optional[float]
    ooverlap_overlap_ms: Optional[float]
    baseline_speedup: Optional[float]
    nccl_speedup: Optional[float]
    ooverlap_speedup: Optional[float]
    baseline_pick: Optional[MetricPick]
    nccl_pick: Optional[MetricPick]
    ooverlap_pick: Optional[MetricPick]


def load_eval_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError(f"Expected top-level JSON object in {path}")

    if "scenarios" not in data or not isinstance(data["scenarios"], list):
        raise ValueError(f"Missing scenarios list in {path}")

    return data


def get_in(d: Dict[str, Any], keys: Iterable[str], default: Any = None) -> Any:
    cur: Any = d
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def as_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        y = float(x)
        if not math.isfinite(y):
            return None
        return y
    except Exception:
        return None


def parse_int_list(text: str) -> List[int]:
    if text is None or str(text).strip() == "":
        return []
    return [int(x) for x in str(text).replace(",", " ").split() if x.strip()]


def scenario_world_size(scenario: Dict[str, Any]) -> Optional[int]:
    for key in ("world_size", "tp_size", "tensor_parallel_size"):
        try:
            value = int(scenario.get(key))
        except Exception:
            continue
        if value > 0:
            return value

    devices = scenario.get("devices")
    if isinstance(devices, (list, tuple)) and devices:
        return len(devices)

    for part in str(scenario.get("scenario_id", "")).split("__"):
        if part.startswith("tp") and part[2:].isdigit():
            value = int(part[2:])
            if value > 0:
                return value

    return None


def scenarios_by_tp(
    scenarios: List[Dict[str, Any]],
) -> Tuple[Dict[int, List[Dict[str, Any]]], int]:
    grouped: Dict[int, List[Dict[str, Any]]] = {}
    unknown_count = 0

    for scenario in scenarios:
        tp = scenario_world_size(scenario)
        if tp is None:
            unknown_count += 1
            continue
        grouped.setdefault(tp, []).append(scenario)

    return dict(sorted(grouped.items())), unknown_count


def cseg_for_backend(scenario: Dict[str, Any], backend: str) -> Optional[List[int]]:
    cseg = get_in(scenario, ["solutions", backend, "cSeg"])
    if isinstance(cseg, list):
        try:
            return [int(x) for x in cseg]
        except Exception:
            return cseg
    return None


def make_pick(
    scenario: Dict[str, Any],
    test_mode: str,
    backend: str,
    ms: float,
) -> MetricPick:
    return MetricPick(
        ms=float(ms),
        scenario_id=str(scenario.get("scenario_id", "")),
        test_mode=test_mode,
        backend=backend,
        comm_sm_slack=scenario.get("comm_sm_slack"),
        cseg=cseg_for_backend(scenario, backend),
    )


def better_pick(old: Optional[MetricPick], new: Optional[MetricPick]) -> Optional[MetricPick]:
    if new is None:
        return old
    if old is None or new.ms < old.ms:
        return new
    return old


def aggregate_baseline_candidates(
    candidates: List[MetricPick],
    aggregation: str,
    baseline_source: str,
) -> Optional[MetricPick]:
    """Aggregate repeated baseline measurements for one exact TP/M/N/K shape."""
    if not candidates:
        return None

    ordered = sorted(candidates, key=lambda pick: float(pick.ms))
    values = [float(pick.ms) for pick in ordered]

    if aggregation == "fastest":
        return ordered[0]
    if aggregation == "slowest":
        return ordered[-1]
    if aggregation == "mean":
        aggregate_ms = sum(values) / len(values)
    elif aggregation == "median":
        midpoint = len(values) // 2
        if len(values) % 2:
            aggregate_ms = values[midpoint]
        else:
            aggregate_ms = 0.5 * (values[midpoint - 1] + values[midpoint])
    else:
        raise ValueError(f"Unknown baseline_aggregation={aggregation}")

    # Preserve the existing MetricPick-based JSON/CSV schema while making it
    # explicit that this value is an aggregate rather than one scenario pick.
    return MetricPick(
        ms=float(aggregate_ms),
        scenario_id=f"{aggregation}_of_{len(values)}_baseline_samples",
        test_mode="aggregate",
        backend=baseline_source,
        comm_sm_slack=None,
        cseg=None,
    )


def iter_test_modes(scenario: Dict[str, Any], test_mode_filter: str):
    tests = scenario.get("tests", {})
    if not isinstance(tests, dict):
        return

    modes = ["uncapped", "capped"] if test_mode_filter == "both" else [test_mode_filter]

    for mode in modes:
        test = tests.get(mode)
        if not isinstance(test, dict) or not test.get("ok", False):
            continue

        by_backend = get_in(test, ["parsed", "by_backend"], {})
        if isinstance(by_backend, dict):
            yield mode, by_backend


def baseline_candidates(
    scenario: Dict[str, Any],
    mode: str,
    by_backend: Dict[str, Any],
    baseline_source: str,
) -> List[MetricPick]:
    out: List[MetricPick] = []

    def add(backend: str, field: str) -> None:
        value = as_float(get_in(by_backend, [backend, field]))
        if value is not None:
            out.append(make_pick(scenario, mode, backend, value))

    if baseline_source == "nccl_plain":
        add("nccl", "plain_baseline_ms")
    elif baseline_source == "best_plain_any_backend":
        add("nccl", "plain_baseline_ms")
        add("ooverlap", "plain_baseline_ms")
    elif baseline_source == "nccl_cublas":
        add("nccl", "cublas_baseline_ms")
    elif baseline_source == "best_cublas_any_backend":
        add("nccl", "cublas_baseline_ms")
        add("ooverlap", "cublas_baseline_ms")
    else:
        raise ValueError(f"Unknown baseline_source={baseline_source}")

    return out


def measured_shape_pairs(scenarios: List[Dict[str, Any]]) -> List[ShapePlot]:
    pairs = set()

    for scenario in scenarios:
        if scenario.get("status") != "ok":
            continue

        shape = scenario.get("shape", {})
        try:
            m = int(shape["m"])
            n = int(shape["n"])
        except Exception:
            continue

        pairs.add((m, n))

    return [ShapePlot(m, n, "x") for m, n in sorted(pairs)]


def collect_measured_ks(
    scenarios: List[Dict[str, Any]],
    m: int,
    n: int,
) -> List[int]:
    ks = set()

    for scenario in scenarios:
        if scenario.get("status") != "ok":
            continue

        shape = scenario.get("shape", {})
        try:
            sm = int(shape.get("m", -1))
            sn = int(shape.get("n", -1))
            sk = int(shape.get("k", -1))
        except Exception:
            continue

        if sm == int(m) and sn == int(n):
            ks.add(sk)

    return sorted(ks)


def ks_for_shape_plot(
    scenarios: List[Dict[str, Any]],
    sp: ShapePlot,
) -> List[int]:
    if isinstance(sp.k, str):
        if sp.k.lower() != "x":
            raise ValueError(f'Unsupported k string: {sp.k}; use "x"')
        return collect_measured_ks(scenarios, sp.m, sp.n)

    if isinstance(sp.k, int):
        return [int(sp.k)]

    return sorted(int(x) for x in sp.k)


def collect_best_for_shape_k(
    scenarios: List[Dict[str, Any]],
    m: int,
    n: int,
    k: int,
    baseline_source: str,
    baseline_aggregation: str,
    test_mode_filter: str,
) -> ShapeKResult:
    baseline_samples: List[MetricPick] = []
    nccl_pick: Optional[MetricPick] = None
    ooverlap_pick: Optional[MetricPick] = None

    for scenario in scenarios:
        if scenario.get("status") != "ok":
            continue

        shape = scenario.get("shape", {})
        try:
            sm = int(shape.get("m", -1))
            sn = int(shape.get("n", -1))
            sk = int(shape.get("k", -1))
        except Exception:
            continue

        if sm != int(m) or sn != int(n) or sk != int(k):
            continue

        for mode, by_backend in iter_test_modes(scenario, test_mode_filter):
            baseline_samples.extend(
                baseline_candidates(scenario, mode, by_backend, baseline_source)
            )

            nccl_ms = as_float(get_in(by_backend, ["nccl", "overlap_dur_ms"]))
            if nccl_ms is not None:
                nccl_pick = better_pick(
                    nccl_pick,
                    make_pick(scenario, mode, "nccl", nccl_ms),
                )

            oo_ms = as_float(get_in(by_backend, ["ooverlap", "overlap_dur_ms"]))
            if oo_ms is not None:
                ooverlap_pick = better_pick(
                    ooverlap_pick,
                    make_pick(scenario, mode, "ooverlap", oo_ms),
                )

    baseline_pick = aggregate_baseline_candidates(
        baseline_samples,
        aggregation=baseline_aggregation,
        baseline_source=baseline_source,
    )
    baseline_ms = baseline_pick.ms if baseline_pick else None
    nccl_ms = nccl_pick.ms if nccl_pick else None
    oo_ms = ooverlap_pick.ms if ooverlap_pick else None

    def speedup(time_ms: Optional[float]) -> Optional[float]:
        if baseline_ms is None or time_ms is None or time_ms <= 0.0:
            return None
        return baseline_ms / time_ms

    return ShapeKResult(
        m=int(m),
        n=int(n),
        k=int(k),
        baseline_ms=baseline_ms,
        nccl_overlap_ms=nccl_ms,
        ooverlap_overlap_ms=oo_ms,
        baseline_speedup=1.0 if baseline_ms is not None else None,
        nccl_speedup=speedup(nccl_ms),
        ooverlap_speedup=speedup(oo_ms),
        baseline_pick=baseline_pick,
        nccl_pick=nccl_pick,
        ooverlap_pick=ooverlap_pick,
    )


def valid_for_plot(r: ShapeKResult) -> bool:
    return (
        r.baseline_speedup is not None
        and r.nccl_speedup is not None
        and r.ooverlap_speedup is not None
    )


def fmt_dim(x: int) -> str:
    return str(x)


def fmt_k(k: int) -> str:
    return f"K = {int(k)}"


def metric_pick_to_dict(x: Optional[MetricPick]) -> Optional[Dict[str, Any]]:
    if x is None:
        return None

    return {
        "ms": x.ms,
        "scenario_id": x.scenario_id,
        "test_mode": x.test_mode,
        "backend": x.backend,
        "comm_sm_slack": x.comm_sm_slack,
        "cSeg": x.cseg,
    }


def result_to_dict(x: ShapeKResult) -> Dict[str, Any]:
    return {
        "m": x.m,
        "n": x.n,
        "k": x.k,
        "baseline_ms": x.baseline_ms,
        "nccl_overlap_ms": x.nccl_overlap_ms,
        "ooverlap_overlap_ms": x.ooverlap_overlap_ms,
        "baseline_speedup": x.baseline_speedup,
        "nccl_speedup": x.nccl_speedup,
        "ooverlap_speedup": x.ooverlap_speedup,
        "baseline_pick": metric_pick_to_dict(x.baseline_pick),
        "nccl_pick": metric_pick_to_dict(x.nccl_pick),
        "ooverlap_pick": metric_pick_to_dict(x.ooverlap_pick),
    }


def build_results(
    scenarios: List[Dict[str, Any]],
    shape_plots: List[ShapePlot],
    baseline_source: str,
    baseline_aggregation: str,
    test_mode: str,
) -> Tuple[Dict[Tuple[int, int], List[ShapeKResult]], List[ShapeKResult]]:
    by_shape: Dict[Tuple[int, int], List[ShapeKResult]] = {}
    all_rows: List[ShapeKResult] = []

    for sp in shape_plots:
        ks = ks_for_shape_plot(scenarios, sp)
        rows = []

        for k in ks:
            r = collect_best_for_shape_k(
                scenarios=scenarios,
                m=sp.m,
                n=sp.n,
                k=k,
                baseline_source=baseline_source,
                baseline_aggregation=baseline_aggregation,
                test_mode_filter=test_mode,
            )
            rows.append(r)
            all_rows.append(r)

        by_shape[(sp.m, sp.n)] = rows

    return by_shape, all_rows


def global_ymax(rows: List[ShapeKResult]) -> float:
    vals = []
    for r in rows:
        if not valid_for_plot(r):
            continue
        vals.extend([
            float(r.baseline_speedup),
            float(r.nccl_speedup),
            float(r.ooverlap_speedup),
        ])

    if not vals:
        return 1.2

    ymax = max(vals)
    return max(1.2, ymax * 1.18)


# OOVERLAP_FLASHOVERLAP_PAPER_PLOT_STYLE_V2
def average_speedups(rows: List[ShapeKResult]) -> Dict[str, Any]:
    """Summarize the mean and observed range for every overlap backend."""
    valid = [r for r in rows if valid_for_plot(r)]

    def mean(values: List[float]) -> Optional[float]:
        return sum(values) / len(values) if values else None

    def minimum(values: List[float]) -> Optional[float]:
        return min(values) if values else None

    def maximum(values: List[float]) -> Optional[float]:
        return max(values) if values else None

    nccl_values = [float(r.nccl_speedup) for r in valid]
    t_ccl_values = [float(r.ooverlap_speedup) for r in valid]
    t_ccl_vs_nccl_values = [
        float(r.ooverlap_speedup) / float(r.nccl_speedup)
        for r in valid
        if float(r.nccl_speedup) > 0.0
    ]

    return {
        "aggregation": "arithmetic_mean_of_per_shape_speedups",
        "total_row_count": len(rows),
        "valid_row_count": len(valid),
        "mean_baseline_speedup": 1.0 if valid else None,
        "mean_nccl_speedup": mean(nccl_values),
        "min_nccl_speedup": minimum(nccl_values),
        "max_nccl_speedup": maximum(nccl_values),
        # Keep internal ooverlap keys for compatibility with existing JSON/CSV readers.
        "mean_ooverlap_speedup": mean(t_ccl_values),
        "min_ooverlap_speedup": minimum(t_ccl_values),
        "max_ooverlap_speedup": maximum(t_ccl_values),
        "mean_ooverlap_vs_nccl": mean(t_ccl_vs_nccl_values),
    }


def format_speedup(x: Optional[float]) -> str:
    return "NA" if x is None else f"{x:.3f}x"


def plot_results_grid(
    by_shape: Dict[Tuple[int, int], List[ShapeKResult]],
    out_png: Path,
    out_pdf: Optional[Path],
    annotate: bool,
) -> None:
    """Plot per-shape speedups without an overall figure title."""
    valid_count = sum(
        1
        for rows in by_shape.values()
        for r in rows
        if valid_for_plot(r)
    )
    if valid_count <= 0:
        raise RuntimeError("No valid shape/K rows to plot.")

    ms = sorted({m for m, _ in by_shape.keys()})
    ns = sorted({n for _, n in by_shape.keys()})

    nrows = len(ms)
    ncols = len(ns)
    fig_width = max(10.0, 4.2 * ncols)
    fig_height = max(4.0, 3.25 * nrows)

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(fig_width, fig_height),
        squeeze=False,
        sharey=True,
    )

    ymax = global_ymax([r for rows in by_shape.values() for r in rows])
    legend_handles = None
    legend_labels = None
    bar_width = 0.25

    for row_idx, m in enumerate(ms):
        for col_idx, n in enumerate(ns):
            ax = axes[row_idx][col_idx]
            rows = [r for r in by_shape.get((m, n), []) if valid_for_plot(r)]
            rows.sort(key=lambda r: r.k)

            if not rows:
                ax.axis("off")
                continue

            xs = list(range(len(rows)))
            k_labels = [fmt_k(r.k) for r in rows]
            baseline_vals = [float(r.baseline_speedup) for r in rows]
            nccl_vals = [float(r.nccl_speedup) for r in rows]
            t_ccl_vals = [float(r.ooverlap_speedup) for r in rows]

            bars0 = ax.bar(
                [x - bar_width for x in xs],
                baseline_vals,
                width=bar_width,
                label="Baseline",
                edgecolor="none",
                linewidth=0,
            )
            bars1 = ax.bar(
                xs,
                nccl_vals,
                width=bar_width,
                label="NCCL",
                edgecolor="none",
                linewidth=0,
            )
            bars2 = ax.bar(
                [x + bar_width for x in xs],
                t_ccl_vals,
                width=bar_width,
                label="T-CCL",
                edgecolor="none",
                linewidth=0,
            )

            if legend_handles is None:
                legend_handles, legend_labels = ax.get_legend_handles_labels()

            ax.axhline(1.0, linewidth=1.0, linestyle="--", alpha=0.75)
            ax.set_ylim(0.0, ymax)
            ax.set_xticks(xs)
            ax.set_xticklabels(k_labels, rotation=0)
            ax.tick_params(axis="both", labelsize=TICK_LABEL_FONTSIZE)
            ax.grid(True, axis="y", linestyle="--", linewidth=0.6, alpha=0.45)

            # Keep the panel name inside the axes so it cannot be clipped outside.
            ax.text(
                0.02,
                0.97,
                f"M = {fmt_dim(m)}, N = {fmt_dim(n)}",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=PANEL_LABEL_FONTSIZE,
            )

            if row_idx == nrows - 1:
                ax.set_xlabel("K dimension", fontsize=AXIS_LABEL_FONTSIZE)
            if col_idx == 0:
                ax.set_ylabel("Speedup", fontsize=AXIS_LABEL_FONTSIZE)

            if annotate:
                for bars in (bars0, bars1, bars2):
                    for bar in bars:
                        height = bar.get_height()
                        ax.text(
                            bar.get_x() + bar.get_width() / 2.0,
                            max(0.02 * ymax, height - 0.035 * ymax),
                            f"{height:.2f}",
                            ha="center",
                            va="top",
                            fontsize=SPEEDUP_LABEL_FONTSIZE,
                            rotation=90,
                        )

    if legend_handles is not None:
        fig.legend(
            legend_handles,
            legend_labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.985),
            ncols=3,
            frameon=False,
            fontsize=LEGEND_FONTSIZE,
        )

    fig.tight_layout(rect=(0.02, 0.02, 0.98, 0.94))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=240, bbox_inches="tight")
    if out_pdf is not None:
        fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


def plot_tp_average_speedups(
    averages: List[Dict[str, Any]],
    out_png: Path,
    out_pdf: Optional[Path],
) -> None:
    """Plot TP means with min/max endpoint markers and no figure title."""
    rows = [row for row in averages if int(row.get("valid_row_count", 0)) > 0]
    if not rows:
        raise RuntimeError("No valid per-TP averages to plot.")

    xs = list(range(len(rows)))
    bar_width = 0.25
    baseline_values = [1.0 for _ in rows]
    nccl_values = [float(row["mean_nccl_speedup"]) for row in rows]
    t_ccl_values = [float(row["mean_ooverlap_speedup"]) for row in rows]
    nccl_min_values = [float(row["min_nccl_speedup"]) for row in rows]
    nccl_max_values = [float(row["max_nccl_speedup"]) for row in rows]
    t_ccl_min_values = [float(row["min_ooverlap_speedup"]) for row in rows]
    t_ccl_max_values = [float(row["max_ooverlap_speedup"]) for row in rows]

    all_values = (
        baseline_values
        + nccl_values
        + t_ccl_values
        + nccl_min_values
        + nccl_max_values
        + t_ccl_min_values
        + t_ccl_max_values
    )
    # Leave room above the maximum endpoint for the mean-speedup label.
    ymax = max(1.2, max(all_values) * 1.32)

    fig_width = max(7.0, 2.4 * len(rows) + 2.5)
    fig, ax = plt.subplots(figsize=(fig_width, 4.8))

    bars0 = ax.bar(
        [x - bar_width for x in xs],
        baseline_values,
        width=bar_width,
        label="Baseline",
        edgecolor="none",
        linewidth=0,
    )
    bars1 = ax.bar(
        xs,
        nccl_values,
        width=bar_width,
        label="NCCL",
        edgecolor="none",
        linewidth=0,
    )
    bars2 = ax.bar(
        [x + bar_width for x in xs],
        t_ccl_values,
        width=bar_width,
        label="T-CCL",
        edgecolor="none",
        linewidth=0,
    )

    # A hollow endpoint is the minimum; a filled endpoint is the maximum.
    # NCCL uses circles and T-CCL uses diamonds.
    for x, low, high in zip(xs, nccl_min_values, nccl_max_values):
        ax.scatter([x], [low], marker="o", facecolors="none", edgecolors="black", zorder=5)
        ax.scatter([x], [high], marker="o", facecolors="black", edgecolors="black", zorder=5)

    for x, low, high in zip(xs, t_ccl_min_values, t_ccl_max_values):
        marker_x = x + bar_width
        ax.scatter(
            [marker_x], [low], marker="D", facecolors="none", edgecolors="black", zorder=5
        )
        ax.scatter(
            [marker_x], [high], marker="D", facecolors="black", edgecolors="black", zorder=5
        )

    ax.set_ylim(0.0, ymax)
    ax.axhline(1.0, linewidth=1.0, linestyle="--", alpha=0.75)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"TP={int(row['world_size'])}" for row in rows])
    ax.set_xlabel("Tensor parallel size", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel("Speedup", fontsize=AXIS_LABEL_FONTSIZE)
    ax.tick_params(axis="both", labelsize=TICK_LABEL_FONTSIZE)
    ax.grid(True, axis="y", linestyle="--", linewidth=0.6, alpha=0.45)
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncols=3,
        frameon=False,
        fontsize=LEGEND_FONTSIZE,
    )

    # Put the mean label above the larger of the mean bar and the observed
    # maximum. This avoids collisions with both hollow-min and filled-max
    # endpoint markers.
    label_groups = (
        (bars0, baseline_values),
        (bars1, nccl_max_values),
        (bars2, t_ccl_max_values),
    )
    label_pad = 0.025 * ymax
    for bars, upper_values in label_groups:
        for bar, upper in zip(bars, upper_values):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                max(height, upper) + label_pad,
                f"{height:.2f}x",
                ha="center",
                va="bottom",
                fontsize=SPEEDUP_LABEL_FONTSIZE,
                fontweight="bold",
                clip_on=False,
            )

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.92))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=240, bbox_inches="tight")
    if out_pdf is not None:
        fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


def write_summary_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_summary_csv(path: Path, rows: List[ShapeKResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    fields = [
        "m",
        "n",
        "k",
        "baseline_ms",
        "nccl_overlap_ms",
        "ooverlap_overlap_ms",
        "baseline_speedup",
        "nccl_speedup",
        "ooverlap_speedup",
        "baseline_scenario_id",
        "baseline_test_mode",
        "baseline_backend",
        "baseline_comm_sm_slack",
        "nccl_scenario_id",
        "nccl_test_mode",
        "nccl_comm_sm_slack",
        "ooverlap_scenario_id",
        "ooverlap_test_mode",
        "ooverlap_comm_sm_slack",
    ]

    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()

        for r in rows:
            w.writerow({
                "m": r.m,
                "n": r.n,
                "k": r.k,
                "baseline_ms": r.baseline_ms,
                "nccl_overlap_ms": r.nccl_overlap_ms,
                "ooverlap_overlap_ms": r.ooverlap_overlap_ms,
                "baseline_speedup": r.baseline_speedup,
                "nccl_speedup": r.nccl_speedup,
                "ooverlap_speedup": r.ooverlap_speedup,
                "baseline_scenario_id": r.baseline_pick.scenario_id if r.baseline_pick else "",
                "baseline_test_mode": r.baseline_pick.test_mode if r.baseline_pick else "",
                "baseline_backend": r.baseline_pick.backend if r.baseline_pick else "",
                "baseline_comm_sm_slack": r.baseline_pick.comm_sm_slack if r.baseline_pick else "",
                "nccl_scenario_id": r.nccl_pick.scenario_id if r.nccl_pick else "",
                "nccl_test_mode": r.nccl_pick.test_mode if r.nccl_pick else "",
                "nccl_comm_sm_slack": r.nccl_pick.comm_sm_slack if r.nccl_pick else "",
                "ooverlap_scenario_id": r.ooverlap_pick.scenario_id if r.ooverlap_pick else "",
                "ooverlap_test_mode": r.ooverlap_pick.test_mode if r.ooverlap_pick else "",
                "ooverlap_comm_sm_slack": r.ooverlap_pick.comm_sm_slack if r.ooverlap_pick else "",
            })


def write_summary_txt(path: Path, rows: List[ShapeKResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    def fmt_ms(x: Optional[float]) -> str:
        return "NA" if x is None else f"{x:.4f}"

    def fmt_speedup(x: Optional[float]) -> str:
        return "NA" if x is None else f"{x:.4f}x"

    ordered = sorted(rows, key=lambda r: (r.m, r.n, r.k))
    table_rows = []
    for r in ordered:
        t_ccl_vs_nccl = None
        if (
            r.nccl_overlap_ms is not None
            and r.ooverlap_overlap_ms is not None
            and r.ooverlap_overlap_ms > 0.0
        ):
            t_ccl_vs_nccl = r.nccl_overlap_ms / r.ooverlap_overlap_ms

        table_rows.append([
            str(r.m),
            str(r.n),
            str(r.k),
            fmt_ms(r.baseline_ms),
            fmt_ms(r.nccl_overlap_ms),
            fmt_ms(r.ooverlap_overlap_ms),
            fmt_speedup(r.nccl_speedup),
            fmt_speedup(r.ooverlap_speedup),
            fmt_speedup(t_ccl_vs_nccl),
        ])

    headers = [
        "M",
        "N",
        "K",
        "Baseline ms",
        "NCCL ms",
        "T-CCL ms",
        "NCCL vs baseline",
        "T-CCL vs baseline",
        "T-CCL vs NCCL",
    ]

    widths = [len(h) for h in headers]
    for row in table_rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def line(cells: List[str]) -> str:
        return "  ".join(cell.ljust(widths[i]) for i, cell in enumerate(cells))

    out = [
        "Operator-Level Communication-Computation Overlap Summary",
        "",
        "Speedups are normalized to the aggregated no-overlap baseline for the same M, N, K shape.",
        "The baseline aggregation policy is recorded in the companion JSON output.",
        "T-CCL vs NCCL is computed as NCCL overlap time divided by T-CCL overlap time.",
        "",
        line(headers),
        line(["-" * width for width in widths]),
    ]
    out.extend(line(row) for row in table_rows)
    path.write_text("\n".join(out) + "\n", encoding="utf-8")


def write_tp_average_csv(path: Path, averages: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "world_size",
        "total_row_count",
        "valid_row_count",
        "mean_baseline_speedup",
        "mean_nccl_speedup",
        "min_nccl_speedup",
        "max_nccl_speedup",
        "mean_ooverlap_speedup",
        "min_ooverlap_speedup",
        "max_ooverlap_speedup",
        "mean_ooverlap_vs_nccl",
        "aggregation",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(averages)


def write_tp_average_txt(path: Path, averages: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    headers = [
        "TP",
        "Valid/Total",
        "NCCL mean",
        "NCCL min-max",
        "T-CCL mean",
        "T-CCL min-max",
    ]
    table_rows = [
        [
            str(row["world_size"]),
            f"{row['valid_row_count']}/{row['total_row_count']}",
            format_speedup(row.get("mean_nccl_speedup")),
            f"{format_speedup(row.get('min_nccl_speedup'))} - "
            f"{format_speedup(row.get('max_nccl_speedup'))}",
            format_speedup(row.get("mean_ooverlap_speedup")),
            f"{format_speedup(row.get('min_ooverlap_speedup'))} - "
            f"{format_speedup(row.get('max_ooverlap_speedup'))}",
        ]
        for row in averages
    ]
    widths = [len(header) for header in headers]
    for row in table_rows:
        for index, cell in enumerate(row):
            widths[index] = max(widths[index], len(cell))

    def line(cells: List[str]) -> str:
        return "  ".join(cell.ljust(widths[index]) for index, cell in enumerate(cells))

    out = [
        "Average Operator Speedup by Tensor Parallel Size",
        "",
        "Arithmetic mean and observed min-max range over valid selected (M,N,K) rows.",
        "",
        line(headers),
        line(["-" * width for width in widths]),
    ]
    out.extend(line(row) for row in table_rows)
    path.write_text("\n".join(out) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()

    p.add_argument("--json", type=str, default=DEFAULT_JSON)
    p.add_argument("--out-dir", type=str, default=DEFAULT_OUT_DIR)
    p.add_argument("--name", type=str, default=DEFAULT_PLOT_NAME)

    p.add_argument(
        "--baseline-source",
        type=str,
        default=DEFAULT_BASELINE_SOURCE,
        choices=[
            "nccl_plain",
            "best_plain_any_backend",
            "nccl_cublas",
            "best_cublas_any_backend",
        ],
    )
    p.add_argument(
        "--baseline-aggregation",
        type=str,
        default=DEFAULT_BASELINE_AGGREGATION,
        choices=["mean", "median", "fastest", "slowest"],
        help=(
            "How to combine repeated no-overlap baseline measurements for the "
            "same TP/M/N/K. Default: mean."
        ),
    )
    p.add_argument(
        "--test-mode",
        type=str,
        default=DEFAULT_TEST_MODE,
        choices=["both", "uncapped", "capped"],
    )

    p.add_argument(
        "--all-shapes",
        action="store_true",
        help="Plot every measured M,N pair in the JSON instead of PLOT_SHAPES.",
    )
    p.add_argument(
        "--ms",
        type=parse_int_list,
        default=[],
        help="Optional M filter, for example: --ms 16384,32768",
    )
    p.add_argument(
        "--ns",
        type=parse_int_list,
        default=[],
        help="Optional N filter, for example: --ns 4096,8192",
    )

    p.add_argument(
        "--tps",
        type=parse_int_list,
        default=[],
        help="Optional TP/world-size filter, for example: --tps 2,4,8",
    )
    p.add_argument("--annotate", action="store_true")
    p.add_argument("--no-pdf", action="store_true")
    return p.parse_args()


def choose_shape_plots(
    scenarios: List[Dict[str, Any]],
    use_all_shapes: bool,
    ms_filter: List[int],
    ns_filter: List[int],
) -> List[ShapePlot]:
    shapes = measured_shape_pairs(scenarios) if use_all_shapes else list(PLOT_SHAPES)

    if ms_filter:
        mset = set(int(x) for x in ms_filter)
        shapes = [s for s in shapes if int(s.m) in mset]

    if ns_filter:
        nset = set(int(x) for x in ns_filter)
        shapes = [s for s in shapes if int(s.n) in nset]

    shapes = sorted(shapes, key=lambda s: (s.m, s.n))

    if not shapes:
        raise SystemExit("No shapes selected. Check PLOT_SHAPES, --all-shapes, --ms, or --ns.")

    return shapes


def main() -> int:
    args = parse_args()

    in_path = Path(args.json).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()

    data = load_eval_json(in_path)
    grouped, unknown_tp_count = scenarios_by_tp(data["scenarios"])
    source_has_multiple_tps = len(grouped) > 1

    if args.tps:
        selected_tps = set(int(tp) for tp in args.tps)
        grouped = {tp: scenarios for tp, scenarios in grouped.items() if tp in selected_tps}

    if not grouped:
        raise SystemExit(
            "No scenarios with usable TP metadata were selected. "
            "Expected scenario.world_size, scenario.devices, or __tpN__ in scenario_id."
        )

    tp_averages: List[Dict[str, Any]] = []

    for world_size, scenarios in grouped.items():
        shape_plots = choose_shape_plots(
            scenarios=scenarios,
            use_all_shapes=bool(args.all_shapes),
            ms_filter=args.ms,
            ns_filter=args.ns,
        )

        by_shape, all_rows = build_results(
            scenarios=scenarios,
            shape_plots=shape_plots,
            baseline_source=args.baseline_source,
            baseline_aggregation=args.baseline_aggregation,
            test_mode=args.test_mode,
        )

        average = average_speedups(all_rows)
        average["world_size"] = int(world_size)
        tp_averages.append(average)

        suffix = f"_tp{world_size}" if source_has_multiple_tps else ""
        output_stem = f"{args.name}{suffix}"
        out_png = out_dir / f"{output_stem}.png"
        out_pdf = None if args.no_pdf else out_dir / f"{output_stem}.pdf"
        out_json = out_dir / f"{output_stem}_data.json"
        out_csv = out_dir / f"{output_stem}_data.csv"
        out_txt = out_dir / f"{output_stem}_summary.txt"

        summary = {
            "input_json": str(in_path),
            "world_size": int(world_size),
            "baseline_source": args.baseline_source,
            "baseline_aggregation": args.baseline_aggregation,
            "test_mode": args.test_mode,
            "plot_layout": "rows=M, columns=N, x-axis=K",
            "average_speedup": average,
            "plot_shapes": [
                {
                    "m": sp.m,
                    "n": sp.n,
                    "k": sp.k,
                    "label": sp.label,
                }
                for sp in shape_plots
            ],
            "rows": [result_to_dict(r) for r in all_rows],
        }

        write_summary_json(out_json, summary)
        write_summary_csv(out_csv, all_rows)
        write_summary_txt(out_txt, all_rows)

        plot_results_grid(
            by_shape=by_shape,
            out_png=out_png,
            out_pdf=out_pdf,
            annotate=bool(args.annotate),
        )

        print(f"wrote: {out_png}")
        if out_pdf is not None:
            print(f"wrote: {out_pdf}")
        print(f"wrote: {out_json}")
        print(f"wrote: {out_csv}")
        print(f"wrote: {out_txt}")

        missing = [r for r in all_rows if not valid_for_plot(r)]
        if missing:
            print("")
            print(f"WARNING: TP={world_size} has rows missing one or more timings:")
            for r in missing:
                print(
                    f"  m={r.m} n={r.n} k={r.k} "
                    f"baseline={r.baseline_ms} "
                    f"nccl={r.nccl_overlap_ms} "
                    f"ooverlap={r.ooverlap_overlap_ms}"
                )

    tp_averages.sort(key=lambda row: int(row["world_size"]))
    average_stem = f"{args.name}_average_by_tp"
    average_png = out_dir / f"{average_stem}.png"
    average_pdf = None if args.no_pdf else out_dir / f"{average_stem}.pdf"
    average_json = out_dir / f"{average_stem}.json"
    average_csv = out_dir / f"{average_stem}.csv"
    average_txt = out_dir / f"{average_stem}.txt"

    average_summary = {
        "input_json": str(in_path),
        "baseline_source": args.baseline_source,
        "test_mode": args.test_mode,
        "aggregation": "arithmetic_mean_of_per_shape_speedups",
        "unknown_tp_scenario_count": unknown_tp_count,
        "rows": tp_averages,
    }
    write_summary_json(average_json, average_summary)
    write_tp_average_csv(average_csv, tp_averages)
    write_tp_average_txt(average_txt, tp_averages)
    plot_tp_average_speedups(
        averages=tp_averages,
        out_png=average_png,
        out_pdf=average_pdf,
    )

    print(f"wrote: {average_png}")
    if average_pdf is not None:
        print(f"wrote: {average_pdf}")
    print(f"wrote: {average_json}")
    print(f"wrote: {average_csv}")
    print(f"wrote: {average_txt}")
    if unknown_tp_count:
        print(f"WARNING: skipped {unknown_tp_count} scenarios without TP metadata")

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
