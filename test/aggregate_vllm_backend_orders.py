#!/usr/bin/env python3
"""Aggregate paper-like vLLM results across backend-order permutations.

Each input row is already the mean of the repetitions within one backend
order (``batch_scaling/summary_aggregate.csv``).  This script gives every
backend order equal weight, reports cross-order variation, and computes
speedups only from orders where both compared backends completed.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


CONFIG_FIELDS = (
    "model",
    "devices",
    "tensor_parallel_size",
    "workload",
    "input_len",
    "output_len",
    "max_num_seqs",
    "max_num_batched_tokens",
    "num_prompts",
    "num_warmups",
    "max_model_len",
    "gpu_memory_utilization",
)

CROSS_BATCH_FIELDS = (
    "model",
    "devices",
    "tensor_parallel_size",
    "workload",
    "input_len",
    "output_len",
    "max_num_batched_tokens",
    "max_model_len",
    "gpu_memory_utilization",
)

INTEGER_FIELDS = {
    "tensor_parallel_size",
    "input_len",
    "output_len",
    "max_num_seqs",
    "max_num_batched_tokens",
    "num_prompts",
    "num_warmups",
    "max_model_len",
    "repetitions_ok",
}

FLOAT_FIELDS = {
    "gpu_memory_utilization",
    "elapsed_mean_s",
    "elapsed_std_s",
    "requests_per_s_mean",
    "total_tokens_per_s_mean",
    "output_tokens_per_s_mean",
    "speedup_vs_base_mean",
    "change_vs_base_pct_mean",
}

METRIC_FIELDS = (
    "elapsed_mean_s",
    "requests_per_s_mean",
    "total_tokens_per_s_mean",
    "output_tokens_per_s_mean",
)

REQUIRED_FIELDS = set(CONFIG_FIELDS) | {
    "backend",
    "repetitions_ok",
    *METRIC_FIELDS,
}


class AggregationError(ValueError):
    pass


@dataclass(frozen=True)
class OrderSource:
    name: str
    directory: Path
    backend_order: tuple[str, ...]
    summary_path: Path


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def atomic_write_json(path: Path, value: Any) -> None:
    atomic_write_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def csv_value(value: Any) -> Any:
    if isinstance(value, (tuple, list, dict)):
        return json.dumps(value, sort_keys=True)
    return value


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        atomic_write_text(path, "")
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for field in row:
            if field not in seen:
                seen.add(field)
                fieldnames.append(field)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.parent.mkdir(parents=True, exist_ok=True)
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: csv_value(row.get(field)) for field in fieldnames})
    temporary.replace(path)


def parse_devices(value: str) -> tuple[int, ...]:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise AggregationError(f"invalid devices value {value!r}") from exc
    if not isinstance(parsed, list):
        raise AggregationError(f"devices must be a JSON list, got {value!r}")
    try:
        return tuple(int(device) for device in parsed)
    except (TypeError, ValueError) as exc:
        raise AggregationError(f"invalid devices value {value!r}") from exc


def parse_optional_number(value: str | None, converter: type[int] | type[float]) -> Any:
    if value is None or value.strip() == "":
        return None
    try:
        return converter(value)
    except ValueError as exc:
        raise AggregationError(f"invalid numeric value {value!r}") from exc


def normalize_row(raw: Mapping[str, str], source: OrderSource) -> dict[str, Any]:
    missing = sorted(field for field in REQUIRED_FIELDS if field not in raw)
    if missing:
        raise AggregationError(
            f"{source.summary_path} is missing columns: {', '.join(missing)}"
        )
    row: dict[str, Any] = dict(raw)
    row["devices"] = parse_devices(raw["devices"])
    for field in INTEGER_FIELDS:
        row[field] = parse_optional_number(raw.get(field), int)
    for field in FLOAT_FIELDS:
        row[field] = parse_optional_number(raw.get(field), float)
    backend = str(row.get("backend", "")).strip()
    if not backend:
        raise AggregationError(f"{source.summary_path} contains an empty backend")
    row["backend"] = backend
    row["order_directory"] = source.name
    row["backend_order"] = source.backend_order
    row["source_summary"] = str(source.summary_path)
    return row


def read_backend_order(path: Path) -> tuple[str, ...]:
    order_file = path / "backend_order.txt"
    if order_file.is_file():
        return tuple(
            backend.strip()
            for backend in order_file.read_text(encoding="utf-8").strip().split(",")
            if backend.strip()
        )
    pieces = path.name.split("__")[1:]
    return tuple(piece for piece in pieces if piece)


def discover_sources(root: Path) -> list[OrderSource]:
    sources: list[OrderSource] = []
    for order_dir in sorted(root.glob("order-*")):
        if not order_dir.is_dir():
            continue
        summary = order_dir / "batch_scaling" / "summary_aggregate.csv"
        if not summary.is_file() or summary.stat().st_size == 0:
            continue
        sources.append(
            OrderSource(
                name=order_dir.name,
                directory=order_dir,
                backend_order=read_backend_order(order_dir),
                summary_path=summary,
            )
        )
    return sources


def load_order_rows(sources: Sequence[OrderSource]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    for source in sources:
        with source.summary_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise AggregationError(f"empty CSV header: {source.summary_path}")
            for raw in reader:
                row = normalize_row(raw, source)
                identity = (source.name, configuration_key(row), row["backend"])
                if identity in seen:
                    raise AggregationError(
                        "duplicate backend/configuration row in "
                        f"{source.summary_path}: {row['backend']}"
                    )
                seen.add(identity)
                rows.append(row)
    return rows


def planned_order_count(root: Path) -> int | None:
    manifest = root.parent / "backend_orders.tsv"
    if not manifest.is_file():
        return None
    with manifest.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    return len(rows) or None


def configuration_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return tuple(row.get(field) for field in CONFIG_FIELDS)


def cross_batch_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return tuple(row.get(field) for field in CROSS_BATCH_FIELDS)


def fields_from_key(key: tuple[Any, ...], fields: Sequence[str]) -> dict[str, Any]:
    return dict(zip(fields, key))


def numeric_values(rows: Iterable[Mapping[str, Any]], field: str) -> list[float]:
    return [float(row[field]) for row in rows if row.get(field) is not None]


def mean_or_none(values: Sequence[float]) -> float | None:
    return statistics.mean(values) if values else None


def pstdev_or_zero(values: Sequence[float]) -> float:
    return statistics.pstdev(values) if len(values) > 1 else 0.0


def pair_speedups(
    rows: Sequence[dict[str, Any]], target_backend: str, baseline_backend: str
) -> dict[tuple[Any, ...], dict[str, Any]]:
    indexed = {
        (row["order_directory"], configuration_key(row), row["backend"]): row
        for row in rows
    }
    grouped: dict[tuple[Any, ...], list[tuple[str, float]]] = {}
    for row in rows:
        if row["backend"] != target_backend:
            continue
        key = configuration_key(row)
        baseline = indexed.get((row["order_directory"], key, baseline_backend))
        target_throughput = row.get("output_tokens_per_s_mean")
        baseline_throughput = (
            baseline.get("output_tokens_per_s_mean") if baseline is not None else None
        )
        if (
            target_throughput is None
            or baseline_throughput is None
            or float(baseline_throughput) <= 0.0
        ):
            continue
        grouped.setdefault(key, []).append(
            (
                row["order_directory"],
                float(target_throughput) / float(baseline_throughput),
            )
        )

    result: dict[tuple[Any, ...], dict[str, Any]] = {}
    for key, samples in grouped.items():
        ratios = [ratio for _, ratio in samples]
        result[key] = {
            "paired_orders": len(samples),
            "paired_order_directories": tuple(name for name, _ in samples),
            "speedup_mean": statistics.mean(ratios),
            "speedup_std": pstdev_or_zero(ratios),
            "speedup_min": min(ratios),
            "speedup_max": max(ratios),
        }
    return result


def aggregate_order_rows(
    rows: Sequence[dict[str, Any]],
    baseline_backend: str,
    orders_found: int,
    orders_planned: int,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((configuration_key(row), row["backend"]), []).append(row)

    speedups_by_backend: dict[str, dict[tuple[Any, ...], dict[str, Any]]] = {}
    for backend in {str(row["backend"]) for row in rows}:
        speedups_by_backend[backend] = pair_speedups(rows, backend, baseline_backend)

    aggregates: list[dict[str, Any]] = []
    for (key, backend), samples in grouped.items():
        result = fields_from_key(key, CONFIG_FIELDS)
        result.update(
            {
                "backend": backend,
                "baseline_backend": baseline_backend,
                "orders_ok": len(samples),
                "orders_found": orders_found,
                "orders_planned": orders_planned,
                "order_directories": tuple(
                    sorted(str(row["order_directory"]) for row in samples)
                ),
                "repetitions_ok": sum(
                    int(row.get("repetitions_ok") or 0) for row in samples
                ),
            }
        )
        for metric in METRIC_FIELDS:
            values = numeric_values(samples, metric)
            result[metric] = mean_or_none(values)
            result[metric.replace("_mean", "_order_std")] = pstdev_or_zero(values)

        paired = speedups_by_backend[backend].get(key)
        result["paired_orders_vs_base"] = paired["paired_orders"] if paired else 0
        result["paired_order_directories_vs_base"] = (
            paired["paired_order_directories"] if paired else ()
        )
        result["speedup_vs_base_mean"] = paired["speedup_mean"] if paired else None
        result["speedup_vs_base_order_std"] = (
            paired["speedup_std"] if paired else None
        )
        result["change_vs_base_pct_mean"] = (
            (float(paired["speedup_mean"]) - 1.0) * 100.0 if paired else None
        )
        aggregates.append(result)

    aggregates.sort(
        key=lambda row: (
            str(row.get("model")),
            int(row.get("tensor_parallel_size") or 0),
            str(row.get("workload")),
            int(row.get("max_num_seqs") or 0),
            int(row.get("max_num_batched_tokens") or 0),
            row.get("backend") != baseline_backend,
            str(row.get("backend")),
        )
    )
    return aggregates


def build_paired_speedup_rows(
    rows: Sequence[dict[str, Any]],
    target_backend: str,
    comparison_backends: Sequence[str],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for baseline_backend in comparison_backends:
        if baseline_backend == target_backend:
            continue
        for key, stats in pair_speedups(rows, target_backend, baseline_backend).items():
            item = fields_from_key(key, CONFIG_FIELDS)
            item.update(
                {
                    "target_backend": target_backend,
                    "baseline_backend": baseline_backend,
                    **stats,
                }
            )
            output.append(item)
    output.sort(
        key=lambda row: (
            cross_batch_key(row),
            int(row.get("max_num_seqs") or 0),
            str(row.get("baseline_backend")),
        )
    )
    return output


def build_cross_batch_rows(
    paired_rows: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in paired_rows:
        key = (
            cross_batch_key(row),
            row["target_backend"],
            row["baseline_backend"],
        )
        grouped.setdefault(key, []).append(row)

    output: list[dict[str, Any]] = []
    for (key, target, baseline), samples in grouped.items():
        samples = sorted(samples, key=lambda row: int(row["max_num_seqs"]))
        ratios = [float(row["speedup_mean"]) for row in samples]
        item = fields_from_key(key, CROSS_BATCH_FIELDS)
        item.update(
            {
                "target_backend": target,
                "baseline_backend": baseline,
                "batch_sizes": tuple(int(row["max_num_seqs"]) for row in samples),
                "paired_orders_by_batch": tuple(
                    int(row["paired_orders"]) for row in samples
                ),
                "points": len(ratios),
                "geometric_mean_speedup": statistics.geometric_mean(ratios),
                "arithmetic_mean_speedup": statistics.mean(ratios),
                "minimum_speedup": min(ratios),
                "maximum_speedup": max(ratios),
            }
        )
        item["geometric_mean_change_pct"] = (
            float(item["geometric_mean_speedup"]) - 1.0
        ) * 100.0
        output.append(item)
    output.sort(
        key=lambda row: (
            cross_batch_key(row),
            str(row.get("baseline_backend")),
        )
    )
    return output


def format_optional(value: Any, spec: str) -> str:
    return "N/A" if value is None else format(float(value), spec)


def backend_label(backend: str) -> str:
    return {"auto": "vLLM Auto", "pynccl": "NCCL", "ooverlap": "T-CCL"}.get(
        backend, backend
    )


def make_summary_text(
    root: Path,
    sources: Sequence[OrderSource],
    aggregates: Sequence[dict[str, Any]],
    cross_batch_rows: Sequence[dict[str, Any]],
    baseline_backend: str,
    orders_planned: int,
) -> str:
    lines = [
        "vLLM backend-order aggregate summary",
        f"Generated: {utc_now()}",
        f"Backend-order root: {root}",
        f"Orders with usable summaries: {len(sources)}",
        f"Orders planned by backend_orders.tsv: {orders_planned}",
        "Averaging: arithmetic mean of each order's repetition mean (equal weight per order).",
        "Speedups: mean of paired per-order output-throughput ratios; unpaired orders are excluded.",
        "",
        "Included orders:",
    ]
    for source in sources:
        order = ",".join(source.backend_order) or "unknown"
        lines.append(f"- {source.name}: {order}")
    lines.append("")

    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in aggregates:
        grouped.setdefault(configuration_key(row), []).append(row)

    incomplete = False
    for key in sorted(grouped, key=lambda value: tuple(str(item) for item in value)):
        values = fields_from_key(key, CONFIG_FIELDS)
        devices = ",".join(str(device) for device in values["devices"])
        lines.extend(
            [
                "=" * 104,
                f"Model: {values['model']}",
                f"Devices: {devices}  TP: {values['tensor_parallel_size']}",
                f"Workload: {values['workload']}  input={values['input_len']}  output={values['output_len']}",
                f"max_num_seqs={values['max_num_seqs']}  max_num_batched_tokens={values['max_num_batched_tokens']}",
                "-" * 104,
            ]
        )
        group_rows = sorted(
            grouped[key],
            key=lambda row: (
                row["backend"] != baseline_backend,
                str(row["backend"]),
            ),
        )
        for row in group_rows:
            orders_ok = int(row["orders_ok"])
            incomplete = incomplete or orders_ok < len(sources)
            marker = " [base]" if row["backend"] == baseline_backend else ""
            lines.append(
                f"{str(row['backend']):<16} "
                f"orders={orders_ok}/{len(sources)} "
                f"reps={int(row['repetitions_ok']):<3d} "
                f"elapsed={format_optional(row.get('elapsed_mean_s'), '.4f')}s "
                f"order_std={format_optional(row.get('elapsed_order_std_s'), '.4f')}s "
                f"req/s={format_optional(row.get('requests_per_s_mean'), '.2f')} "
                f"out_tok/s={format_optional(row.get('output_tokens_per_s_mean'), '.1f')} "
                f"speedup={format_optional(row.get('speedup_vs_base_mean'), '.4f')}x "
                f"pairs={int(row.get('paired_orders_vs_base') or 0)}"
                f"{marker}"
            )
        lines.append("")

    if incomplete:
        lines.extend(
            [
                "Coverage warning: at least one backend/batch has fewer samples than the",
                "number of discovered orders. Check orders= and pairs= before comparing means.",
                "",
            ]
        )

    lines.extend(
        [
            "Cross-batch paired-order speedup summary",
            "Headline: geometric mean across batches of each batch's mean paired-order ratio.",
            "",
        ]
    )
    if not cross_batch_rows:
        lines.append("No complete paired comparisons were found.")
    for row in cross_batch_rows:
        batches = ",".join(str(value) for value in row["batch_sizes"])
        pairs = ",".join(str(value) for value in row["paired_orders_by_batch"])
        lines.append(
            f"{backend_label(str(row['target_backend']))} vs "
            f"{backend_label(str(row['baseline_backend']))}: "
            f"geomean={float(row['geometric_mean_speedup']):.4f}x "
            f"({float(row['geometric_mean_change_pct']):+.2f}%), "
            f"arithmetic_mean={float(row['arithmetic_mean_speedup']):.4f}x, "
            f"range={float(row['minimum_speedup']):.4f}x-"
            f"{float(row['maximum_speedup']):.4f}x, "
            f"batches={batches}, paired_orders={pairs}"
        )
    return "\n".join(lines).rstrip() + "\n"


def aggregate(
    root: Path,
    output_dir: Path,
    baseline_backend: str,
    target_backend: str,
    comparison_backends: Sequence[str],
) -> tuple[int, int]:
    root = root.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    sources = discover_sources(root)
    if not sources:
        raise AggregationError(
            f"no order-*/batch_scaling/summary_aggregate.csv files found below {root}"
        )
    rows = load_order_rows(sources)
    if not rows:
        raise AggregationError(f"no aggregate rows found below {root}")

    planned = planned_order_count(root) or len(sources)
    aggregates = aggregate_order_rows(
        rows,
        baseline_backend=baseline_backend,
        orders_found=len(sources),
        orders_planned=planned,
    )
    paired_rows = build_paired_speedup_rows(
        rows,
        target_backend=target_backend,
        comparison_backends=comparison_backends,
    )
    cross_rows = build_cross_batch_rows(paired_rows)

    write_csv(output_dir / "order_samples.csv", rows)
    write_csv(output_dir / "summary.csv", aggregates)
    write_csv(output_dir / "paired_speedup_by_batch.csv", paired_rows)
    write_csv(output_dir / "speedup_summary.csv", cross_rows)
    atomic_write_text(
        output_dir / "summary.txt",
        make_summary_text(
            root,
            sources,
            aggregates,
            cross_rows,
            baseline_backend,
            planned,
        ),
    )
    atomic_write_json(
        output_dir / "metadata.json",
        {
            "generated_at": utc_now(),
            "backend_order_root": str(root),
            "baseline_backend": baseline_backend,
            "target_backend": target_backend,
            "comparison_backends": list(comparison_backends),
            "orders_found": len(sources),
            "orders_planned": planned,
            "source_summaries": [str(source.summary_path) for source in sources],
            "order_sample_rows": len(rows),
            "aggregate_rows": len(aggregates),
        },
    )
    return len(sources), len(aggregates)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Average vLLM benchmark results across backend-order permutations"
    )
    parser.add_argument(
        "--root",
        type=Path,
        required=True,
        help="Directory containing order-XX__*/batch_scaling results",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Default: ROOT/aggregate/batch_scaling",
    )
    parser.add_argument("--baseline-backend", default="pynccl")
    parser.add_argument("--target-backend", default="ooverlap")
    parser.add_argument("--comparison-backends", default="auto,pynccl")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.root.expanduser().resolve()
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else root / "aggregate" / "batch_scaling"
    )
    comparisons = tuple(
        backend.strip()
        for backend in args.comparison_backends.split(",")
        if backend.strip()
    )
    try:
        orders, aggregate_rows_count = aggregate(
            root=root,
            output_dir=output_dir,
            baseline_backend=args.baseline_backend,
            target_backend=args.target_backend,
            comparison_backends=comparisons,
        )
    except (AggregationError, OSError) as exc:
        print(f"ERROR: {exc}")
        return 2
    print(
        f"Aggregated {aggregate_rows_count} backend/configuration rows "
        f"from {orders} backend orders."
    )
    print(f"Wrote: {output_dir / 'summary.txt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
