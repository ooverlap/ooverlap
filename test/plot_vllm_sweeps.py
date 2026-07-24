#!/usr/bin/env python3
"""Plot the three vLLM paper-evaluation sweeps.

Inputs are the per-run summary.csv files emitted by benchmark_vllm_paperlike.py.
The script creates PDF and PNG versions of:

* prefill_latency.{pdf,png}
* decode_latency.{pdf,png}
* batch_throughput.{pdf,png}
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

BACKEND_LABELS = {
    "pynccl": "NCCL",
    "nccl_symm": "NCCL symmetric",
    "ooverlap": "Ooverlap",
}
BACKEND_MARKERS = {
    "pynccl": "o",
    "nccl_symm": "^",
    "ooverlap": "s",
}
BACKEND_LINESTYLES = {
    "pynccl": "--",
    "nccl_symm": ":",
    "ooverlap": "-",
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
    preferred = ["pynccl", "nccl_symm", "ooverlap"]
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
    lower_is_better: bool,
) -> None:
    fig, axis = plt.subplots(figsize=(7.4, 4.8))
    for backend, points in series.items():
        points = sorted(points)
        xs = [point[0] for point in points]
        ys = [point[1] for point in points]
        errors = [point[2] for point in points]
        axis.errorbar(
            xs,
            ys,
            yerr=errors,
            marker=BACKEND_MARKERS.get(backend, "o"),
            linestyle=BACKEND_LINESTYLES.get(backend, "-"),
            linewidth=1.8,
            capsize=3,
            label=label_for(backend),
        )
    if xlog2:
        axis.set_xscale("log", base=2)
        all_xs = sorted({point[0] for points in series.values() for point in points})
        axis.set_xticks(all_xs)
        axis.set_xticklabels([str(value) for value in all_xs])
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    axis.grid(True, which="both", linestyle="--", alpha=0.35)
    axis.legend(frameon=False)
    direction = "Lower is better" if lower_is_better else "Higher is better"
    axis.text(0.99, 0.02, direction, transform=axis.transAxes, ha="right", va="bottom", fontsize=9)
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
    parser = argparse.ArgumentParser(description="Plot the three vLLM evaluation sweeps")
    parser.add_argument("--root", required=True, help="TP result root containing the three sweep directories")
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

    prefill = select_backends(load_runs(root / "prefill_length" / "summary.csv"), include, exclude)
    decode = select_backends(load_runs(root / "decode_length" / "summary.csv"), include, exclude)
    batch = select_backends(load_runs(root / "batch_scaling" / "summary.csv"), include, exclude)
    order = ordered_backends(prefill + decode + batch)

    plot_series(
        out_dir / "prefill_latency",
        retain_order(prefill_series(prefill), order),
        "Prompt length (tokens)",
        "Prefill latency (ms/request)",
        xlog2=True,
        lower_is_better=True,
    )
    plot_series(
        out_dir / "decode_latency",
        retain_order(decode_series(decode), order),
        "Output length (tokens)",
        "Incremental decode latency (ms/output token)",
        xlog2=True,
        lower_is_better=True,
    )
    plot_series(
        out_dir / "batch_throughput",
        retain_order(batch_series(batch), order),
        "Maximum active sequences",
        "Output throughput (tokens/s)",
        xlog2=True,
        lower_is_better=False,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except PlotInputError as exc:
        raise SystemExit(f"error: {exc}")
