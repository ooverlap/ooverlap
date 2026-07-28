#!/usr/bin/env python3
"""Plot the vLLM decode batch-scaling benchmark.

Inputs are the per-run summary.csv files emitted by benchmark_vllm_paperlike.py.
The script creates PDF and PNG versions of batch_throughput.

OOVERLAP_VLLM_BATCH_ONLY_PLOT_V1
"""
from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, pstdev
from typing import Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# OOVERLAP_VLLM_PAPER_FONT_SIZES_V1
AXIS_LABEL_FONTSIZE = 15
TICK_LABEL_FONTSIZE = 13
LEGEND_FONTSIZE = 13
SPEEDUP_LABEL_FONTSIZE = 10

# OOVERLAP_VLLM_AUTO_BASELINE_SPEEDUP_V1
BACKEND_LABELS = {
    "auto": "vLLM Auto",
    "pynccl": "NCCL",
    "ooverlap": "T-CCL",
    "nccl_symm": "NCCL symmetric",
}
# OOVERLAP_VLLM_GROUPED_HATCHED_BAR_PLOTS_V1
BACKEND_HATCHES = {
    "auto": "",
    "pynccl": "///",
    "ooverlap": "xxx",
    "nccl_symm": r"\\",
}


class PlotInputError(ValueError):
    pass


@dataclass(frozen=True)
class Run:
    backend: str
    repetition: int
    input_len: int
    output_len: int
    max_num_seqs: int
    elapsed_s: float
    num_requests: int
    output_tokens_per_s: float

    @property
    def request_latency_ms(self) -> float:
        return 1000.0 * self.elapsed_s / self.num_requests


def csv_list(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_int(row: dict[str, str], key: str) -> int:
    try:
        return int(float(row[key]))
    except (KeyError, TypeError, ValueError) as exc:
        raise PlotInputError(f"invalid {key}: {row.get(key)!r}") from exc


def parse_float(row: dict[str, str], key: str) -> float:
    try:
        value = float(row[key])
    except (KeyError, TypeError, ValueError) as exc:
        raise PlotInputError(f"invalid {key}: {row.get(key)!r}") from exc
    if not math.isfinite(value):
        raise PlotInputError(f"non-finite {key}: {value}")
    return value


def load_runs(path: Path) -> list[Run]:
    if not path.is_file():
        raise PlotInputError(f"missing benchmark summary: {path}")
    runs: list[Run] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row.get("status") != "ok":
                continue
            num_requests = parse_int(row, "num_requests")
            if num_requests <= 0:
                continue
            runs.append(
                Run(
                    backend=str(row["backend"]),
                    repetition=parse_int(row, "repetition"),
                    input_len=parse_int(row, "input_len"),
                    output_len=parse_int(row, "output_len"),
                    max_num_seqs=parse_int(row, "max_num_seqs"),
                    elapsed_s=parse_float(row, "elapsed_s"),
                    num_requests=num_requests,
                    output_tokens_per_s=parse_float(row, "output_tokens_per_s"),
                )
            )
    if not runs:
        raise PlotInputError(f"no successful runs in {path}")
    return runs


def select_backends(
    runs: Iterable[Run], include: set[str], exclude: set[str]
) -> list[Run]:
    selected = [
        run
        for run in runs
        if (not include or run.backend in include) and run.backend not in exclude
    ]
    if not selected:
        raise PlotInputError("backend filtering removed every data point")
    return selected


def aggregate(values: Iterable[float]) -> tuple[float, float]:
    items = list(values)
    if not items:
        raise PlotInputError("cannot aggregate an empty series")
    return mean(items), pstdev(items) if len(items) > 1 else 0.0


def ordered_backends(runs: Iterable[Run]) -> list[str]:
    present = {run.backend for run in runs}
    preferred = ["auto", "pynccl", "ooverlap", "nccl_symm"]
    return [name for name in preferred if name in present] + sorted(present - set(preferred))


def label_for(backend: str) -> str:
    return BACKEND_LABELS.get(backend, backend)


def plot_series(
    output_base: Path,
    series: dict[str, list[tuple[int, float, float]]],
    xlabel: str,
    ylabel: str,
    *,
    xlog2: bool,
) -> None:
    # OOVERLAP_VLLM_GROUPED_HATCHED_BAR_PLOTS_V1
    # Use categorical spacing so every sweep is a grouped bar chart. The third
    # tuple item is the repetition standard deviation; it is intentionally not
    # drawn because these paper plots should not contain error bars.
    _ = xlog2
    fig, axis = plt.subplots(figsize=(7.4, 4.8))

    backends = list(series)
    all_xs = sorted(
        {point[0] for backend_points in series.values() for point in backend_points}
    )
    centers = list(range(len(all_xs)))
    center_by_x = {value: center for value, center in zip(all_xs, centers)}
    baseline_by_x = {
        point[0]: point[1]
        for point in series.get("auto", [])
    }

    group_width = 0.82
    bar_width = group_width / max(1, len(backends))

    for backend_index, backend in enumerate(backends):
        mean_by_x = {point[0]: point[1] for point in sorted(series[backend])}
        present_xs = [value for value in all_xs if value in mean_by_x]
        offset = (backend_index - (len(backends) - 1) / 2.0) * bar_width
        bar_positions = [center_by_x[value] + offset for value in present_xs]
        bar_values = [mean_by_x[value] for value in present_xs]

        bars = axis.bar(
            bar_positions,
            bar_values,
            width=bar_width * 0.9,
            hatch=BACKEND_HATCHES.get(backend, "..."),
            edgecolor="black",
            linewidth=0.8,
            label=label_for(backend),
        )

        if backend == "ooverlap":
            for bar, batch_size, throughput in zip(
                bars,
                present_xs,
                bar_values,
            ):
                baseline = baseline_by_x.get(batch_size)
                if baseline is None or baseline <= 0.0:
                    continue

                speedup = throughput / baseline
                axis.annotate(
                    f"{speedup:.2f}×",
                    xy=(
                        bar.get_x() + bar.get_width() / 2.0,
                        bar.get_height(),
                    ),
                    xytext=(2, 5),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    rotation=90,
                    # rotation_mode="anchor",
                    fontsize=SPEEDUP_LABEL_FONTSIZE,
                    fontweight="bold",
                )

    axis.set_xticks(centers)
    axis.set_xticklabels([str(value) for value in all_xs])
    axis.set_xlabel(xlabel, fontsize=AXIS_LABEL_FONTSIZE)
    axis.set_ylabel(ylabel, fontsize=AXIS_LABEL_FONTSIZE)
    axis.tick_params(axis="both", labelsize=TICK_LABEL_FONTSIZE)
    axis.margins(y=0.20)
    axis.grid(True, axis="y", linestyle="--", alpha=0.35)
    axis.legend(frameon=False, fontsize=LEGEND_FONTSIZE)
    fig.tight_layout()
    output_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_base.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] wrote {output_base.with_suffix('.png')}")
    print(f"[plot] wrote {output_base.with_suffix('.pdf')}")


def prefill_series(runs: list[Run]) -> dict[str, list[tuple[int, float, float]]]:
    grouped: dict[tuple[str, int], list[float]] = defaultdict(list)
    for run in runs:
        if run.output_len != 1 or run.max_num_seqs != 1:
            continue
        grouped[(run.backend, run.input_len)].append(run.request_latency_ms)
    result: dict[str, list[tuple[int, float, float]]] = defaultdict(list)
    for (backend, input_len), values in grouped.items():
        avg, std = aggregate(values)
        result[backend].append((input_len, avg, std))
    return dict(result)


def decode_series(runs: list[Run]) -> dict[str, list[tuple[int, float, float]]]:
    # Subtract the output_len=1 request from every longer request within the same
    # backend and repetition. Since output_len=1 has no decode forward, this
    # removes the fixed prefill/first-token component.
    by_key = {(r.backend, r.repetition, r.output_len): r for r in runs if r.max_num_seqs == 1}
    grouped: dict[tuple[str, int], list[float]] = defaultdict(list)
    for run in runs:
        if run.max_num_seqs != 1 or run.output_len <= 1:
            continue
        baseline = by_key.get((run.backend, run.repetition, 1))
        if baseline is None:
            continue
        incremental = (run.request_latency_ms - baseline.request_latency_ms) / (run.output_len - 1)
        if incremental >= 0.0:
            grouped[(run.backend, run.output_len)].append(incremental)
    result: dict[str, list[tuple[int, float, float]]] = defaultdict(list)
    for (backend, output_len), values in grouped.items():
        avg, std = aggregate(values)
        result[backend].append((output_len, avg, std))
    if not result:
        raise PlotInputError("decode sweep lacks paired output_len=1 baselines")
    return dict(result)


def batch_series(runs: list[Run]) -> dict[str, list[tuple[int, float, float]]]:
    grouped: dict[tuple[str, int], list[float]] = defaultdict(list)
    for run in runs:
        grouped[(run.backend, run.max_num_seqs)].append(run.output_tokens_per_s)
    result: dict[str, list[tuple[int, float, float]]] = defaultdict(list)
    for (backend, batch), values in grouped.items():
        avg, std = aggregate(values)
        result[backend].append((batch, avg, std))
    return dict(result)


def retain_order(series: dict[str, list[tuple[int, float, float]]], order: list[str]):
    return {backend: series[backend] for backend in order if backend in series}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot the vLLM decode batch-scaling benchmark")
    parser.add_argument("--root", required=True, help="TP result root containing batch_scaling")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--include-backends", default=None, help="Comma-separated backend allow-list")
    parser.add_argument("--exclude-backends", default=None, help="Comma-separated backend deny-list")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(args.root).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else root / "plots"
    include = set(csv_list(args.include_backends))
    exclude = set(csv_list(args.exclude_backends))

    batch = select_backends(load_runs(root / "batch_scaling" / "summary.csv"), include, exclude)
    order = ordered_backends(batch)

    plot_series(
        out_dir / "batch_throughput",
        retain_order(batch_series(batch), order),
        "Batch Size",
        "Tokens per Second",
        xlog2=True,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except PlotInputError as exc:
        raise SystemExit(f"error: {exc}")
