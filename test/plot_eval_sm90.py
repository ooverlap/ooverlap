#!/usr/bin/env python3
"""
Plot Figure-11-style normalized speedup bars from eval_sm90/evaluation_result.json.

For each ShapePlot(M, N, "x"):
  - collect all measured K values for that M,N
  - for each K, scan all scenarios/slacks/test modes
  - choose:
      baseline_ms        = best plain GEMM + NCCL baseline by default
      nccl_overlap_ms    = best NCCL overlap time
      ooverlap_overlap_ms= best ooverlap overlap time
  - plot:
      Non-overlap baseline speedup = 1.0
      NCCL overlap speedup         = baseline_ms / nccl_overlap_ms
      ooverlap overlap speedup     = baseline_ms / ooverlap_overlap_ms

Outputs:
  - PNG/PDF figure
  - plot_data.json
  - plot_data.csv
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


# =============================================================================
# Edit these.
# =============================================================================

@dataclass(frozen=True)
class ShapePlot:
    m: int
    n: int
    k: Union[str, int, List[int], Tuple[int, ...]] = "x"
    label: str = ""


# Put one ShapePlot per M,N panel. Use k="x" to plot all measured K values.
PLOT_SHAPES: List[ShapePlot] = [
    # ShapePlot(8192, 2048, "x"),
    # ShapePlot(8192, 4096, "x"),
    # ShapePlot(8192, 8192, "x"),

    # Add more, for example:
    # ShapePlot(16384, 2048, "x"),
    ShapePlot(16384, 4096, "x"),
    ShapePlot(16384, 8192, "x"),
    # ShapePlot(32768, 2048, "x"),
    ShapePlot(32768, 4096, "x"),
    ShapePlot(32768, 8192, "x"),

    ShapePlot(49152, 4096, "x"),
    ShapePlot(49152, 8192, "x"),
]

DEFAULT_JSON = "results/eval_sm90/evaluation_result.json"
DEFAULT_OUT_DIR = "results/eval_sm90/plots"
DEFAULT_PLOT_NAME = "figure11_style_operator_speedup"

# Default matches what you asked: plain GEMM + NCCL baseline.
# Choices:
#   nccl_plain
#   best_plain_any_backend
#   nccl_cublas
#   best_cublas_any_backend
DEFAULT_BASELINE_SOURCE = "nccl_plain"

# Choices:
#   both
#   uncapped
#   capped
DEFAULT_TEST_MODE = "both"


# =============================================================================
# Data extraction.
# =============================================================================

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


def shape_id(m: int, n: int, k: int) -> str:
    return f"m{m}n{n}k{k}"


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


def iter_test_modes(scenario: Dict[str, Any], test_mode_filter: str):
    tests = scenario.get("tests", {})
    if not isinstance(tests, dict):
        return

    modes: List[str]
    if test_mode_filter == "both":
        modes = ["uncapped", "capped"]
    else:
        modes = [test_mode_filter]

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


def collect_measured_ks(
    scenarios: List[Dict[str, Any]],
    m: int,
    n: int,
) -> List[int]:
    ks = set()

    for s in scenarios:
        if s.get("status") != "ok":
            continue

        shape = s.get("shape", {})
        if int(shape.get("m", -1)) == int(m) and int(shape.get("n", -1)) == int(n):
            try:
                ks.add(int(shape["k"]))
            except Exception:
                pass

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
    test_mode_filter: str,
) -> ShapeKResult:
    baseline_pick: Optional[MetricPick] = None
    nccl_pick: Optional[MetricPick] = None
    ooverlap_pick: Optional[MetricPick] = None

    for scenario in scenarios:
        if scenario.get("status") != "ok":
            continue

        shape = scenario.get("shape", {})
        if (
            int(shape.get("m", -1)) != int(m)
            or int(shape.get("n", -1)) != int(n)
            or int(shape.get("k", -1)) != int(k)
        ):
            continue

        for mode, by_backend in iter_test_modes(scenario, test_mode_filter):
            for cand in baseline_candidates(scenario, mode, by_backend, baseline_source):
                baseline_pick = better_pick(baseline_pick, cand)

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


# =============================================================================
# Plotting.
# =============================================================================

def fmt_k(k: int) -> str:
    if k % 1024 == 0:
        return f"{k // 1024}K"
    return str(k)


def shape_label(sp: ShapePlot) -> str:
    if sp.label:
        return sp.label
    return f"{sp.m}x{sp.n}"


def valid_for_plot(r: ShapeKResult) -> bool:
    return (
        r.baseline_speedup is not None
        and r.nccl_speedup is not None
        and r.ooverlap_speedup is not None
    )


def plot_results(
    grouped: List[Tuple[ShapePlot, List[ShapeKResult]]],
    out_png: Path,
    out_pdf: Optional[Path],
    title: str,
    annotate: bool,
) -> None:
    flat_count = sum(len([r for r in rows if valid_for_plot(r)]) for _, rows in grouped)
    if flat_count <= 0:
        raise RuntimeError("No valid shape/K rows to plot.")

    fig_width = max(12.0, 1.05 * flat_count + 2.0)
    fig_height = 4.8

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    bar_width = 0.24
    intra_group_gap = 1.0
    inter_shape_gap = 0.9

    x_positions: List[float] = []
    x_labels: List[str] = []
    group_centers: List[Tuple[float, str]] = []

    baseline_vals: List[float] = []
    nccl_vals: List[float] = []
    oo_vals: List[float] = []

    x = 0.0

    for sp, rows in grouped:
        rows = [r for r in rows if valid_for_plot(r)]
        if not rows:
            continue

        start_x = x

        for r in rows:
            x_positions.append(x)
            x_labels.append(fmt_k(r.k))

            baseline_vals.append(float(r.baseline_speedup))
            nccl_vals.append(float(r.nccl_speedup))
            oo_vals.append(float(r.ooverlap_speedup))

            x += intra_group_gap

        end_x = x - intra_group_gap
        center = (start_x + end_x) / 2.0
        group_centers.append((center, shape_label(sp)))

        x += inter_shape_gap

    xs = x_positions

    bars0 = ax.bar(
        [v - bar_width for v in xs],
        baseline_vals,
        width=bar_width,
        label="Non-overlap baseline",
    )
    bars1 = ax.bar(
        xs,
        nccl_vals,
        width=bar_width,
        label="NCCL overlap",
    )
    bars2 = ax.bar(
        [v + bar_width for v in xs],
        oo_vals,
        width=bar_width,
        label="ooverlap overlap",
    )

    ax.axhline(1.0, linewidth=1.0, linestyle="--")

    ax.set_xticks(xs)
    ax.set_xticklabels(x_labels, rotation=0)
    ax.set_ylabel("Normalized speedup")
    ax.set_xlabel("K")
    ax.set_title(title)
    ax.legend(ncols=3, loc="upper center", bbox_to_anchor=(0.5, 1.18), frameon=False)

    ymin = min(0.0, min(baseline_vals + nccl_vals + oo_vals) - 0.08)
    ymax = max(1.05, max(baseline_vals + nccl_vals + oo_vals) + 0.12)
    ax.set_ylim(ymin, ymax)

    # MxN group labels under K ticks.
    trans = ax.get_xaxis_transform()
    for center, label in group_centers:
        ax.text(
            center,
            -0.17,
            label,
            ha="center",
            va="top",
            transform=trans,
        )

    # Vertical separators between ShapePlot groups.
    last_end = None
    for idx, (center, label) in enumerate(group_centers[:-1]):
        next_center = group_centers[idx + 1][0]
        sep = (center + next_center) / 2.0
        ax.axvline(sep, linewidth=0.6, alpha=0.35)

    if annotate:
        for bars in (bars0, bars1, bars2):
            for b in bars:
                h = b.get_height()
                ax.text(
                    b.get_x() + b.get_width() / 2.0,
                    h + 0.015,
                    f"{h:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    rotation=90,
                )

    fig.subplots_adjust(bottom=0.24, top=0.82, left=0.08, right=0.99)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=220)

    if out_pdf is not None:
        fig.savefig(out_pdf)

    plt.close(fig)


# =============================================================================
# Output data.
# =============================================================================

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


# =============================================================================
# CLI.
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()

    p.add_argument(
        "--json",
        type=str,
        default=DEFAULT_JSON,
        help="Path to evaluation_result.json.",
    )
    p.add_argument(
        "--out-dir",
        type=str,
        default=DEFAULT_OUT_DIR,
        help="Output directory.",
    )
    p.add_argument(
        "--name",
        type=str,
        default=DEFAULT_PLOT_NAME,
        help="Base output filename without extension.",
    )
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
        help="Which non-overlap baseline to normalize against.",
    )
    p.add_argument(
        "--test-mode",
        type=str,
        default=DEFAULT_TEST_MODE,
        choices=["both", "uncapped", "capped"],
        help="Which test.py result set to consider.",
    )
    p.add_argument(
        "--annotate",
        action="store_true",
        help="Write numeric values above bars.",
    )
    p.add_argument(
        "--no-pdf",
        action="store_true",
        help="Only write PNG, not PDF.",
    )
    p.add_argument(
        "--title",
        type=str,
        default="Operator-level normalized speedup",
    )

    return p.parse_args()


def main() -> int:
    args = parse_args()

    in_path = Path(args.json).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()

    data = load_eval_json(in_path)
    scenarios = data["scenarios"]

    if not PLOT_SHAPES:
        raise SystemExit("PLOT_SHAPES is empty. Add ShapePlot entries near the top.")

    grouped: List[Tuple[ShapePlot, List[ShapeKResult]]] = []
    all_rows: List[ShapeKResult] = []

    for sp in PLOT_SHAPES:
        ks = ks_for_shape_plot(scenarios, sp)
        rows: List[ShapeKResult] = []

        for k in ks:
            r = collect_best_for_shape_k(
                scenarios=scenarios,
                m=sp.m,
                n=sp.n,
                k=k,
                baseline_source=args.baseline_source,
                test_mode_filter=args.test_mode,
            )
            rows.append(r)
            all_rows.append(r)

        grouped.append((sp, rows))

    out_png = out_dir / f"{args.name}.png"
    out_pdf = None if args.no_pdf else out_dir / f"{args.name}.pdf"
    out_json = out_dir / f"{args.name}_plot_data.json"
    out_csv = out_dir / f"{args.name}_plot_data.csv"

    summary = {
        "input_json": str(in_path),
        "baseline_source": args.baseline_source,
        "test_mode": args.test_mode,
        "plot_shapes": [
            {
                "m": sp.m,
                "n": sp.n,
                "k": sp.k,
                "label": sp.label,
            }
            for sp in PLOT_SHAPES
        ],
        "rows": [result_to_dict(r) for r in all_rows],
    }

    write_summary_json(out_json, summary)
    write_summary_csv(out_csv, all_rows)

    plot_results(
        grouped=grouped,
        out_png=out_png,
        out_pdf=out_pdf,
        title=args.title,
        annotate=bool(args.annotate),
    )

    print(f"wrote: {out_png}")
    if out_pdf is not None:
        print(f"wrote: {out_pdf}")
    print(f"wrote: {out_json}")
    print(f"wrote: {out_csv}")

    missing = [r for r in all_rows if not valid_for_plot(r)]
    if missing:
        print("")
        print("WARNING: some rows were missing one or more timings:")
        for r in missing:
            print(
                f"  m={r.m} n={r.n} k={r.k} "
                f"baseline={r.baseline_ms} "
                f"nccl={r.nccl_overlap_ms} "
                f"ooverlap={r.ooverlap_overlap_ms}"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
