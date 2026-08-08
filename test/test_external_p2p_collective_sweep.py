#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from time import sleep


COLLECTIVES = ("allreduce", "reduce_scatter", "all_gather")
FP16_BYTES = 2
OOVERLAP_CTA_ENV_VAR = "OOVERLAP_MAX_CTAS"
NCCL_CTA_ENV_VAR = "NCCL_MAX_CTAS"
REDUCE_TASK_CTA_ENV_VAR = "OOVERLAP_MAX_CTAS_PER_REDUCE_TASK"
TUNING_POLICY_ENV_VAR = "OOVERLAP_TUNING_POLICY"
# OOVERLAP_EXTERNAL_P2P_TUNING_POLICY_CLI_V1

OUTPUT_COLUMNS = [
    "timestamp_utc",
    "run_id",
    "mode",
    "metric_set",
    "collective",
    "collective_code",
    "cta_limit",
    "ooverlap_max_ctas",
    "nccl_max_ctas",
    "ooverlap_max_ctas_per_reduce_task",
    "devices",
    "world_size",
    "numel",
    "bytes",
    "local_shard_numel",
    "local_shard_bytes",
    "bandwidth_bytes_per_rank",
    "iters",
    "warmup",
    "verify",
    "ring_size",
    "oo_total_ms",
    "nccl_total_ms",
    "nccl_symmetric_total_ms",
    "ooverlap_ms",
    "nccl_ms",
    "nccl_symmetric_ms",
    "ooverlap_latency_us",
    "nccl_latency_us",
    "nccl_symmetric_latency_us",
    "ooverlap_bandwidth_gbps",
    "nccl_bandwidth_gbps",
    "nccl_symmetric_bandwidth_gbps",
]

TEXT_COLUMNS = [
    "mode",
    "metric_set",
    "collective",
    "cta_limit",
    "ooverlap_max_ctas_per_reduce_task",
    "ring_size",
    "bytes",
    "ooverlap_latency_us",
    "nccl_latency_us",
    "nccl_symmetric_latency_us",
    "ooverlap_bandwidth_gbps",
    "nccl_bandwidth_gbps",
    "nccl_symmetric_bandwidth_gbps",
]


def repository_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_ooverlap_ext():
    root = repository_root()
    so = root / "build" / "lib" / "ooverlap_ext.so"
    if not so.exists():
        raise FileNotFoundError(f"Could not find {so}. Build the extension first.")

    spec = importlib.util.spec_from_file_location("ooverlap_ext", str(so))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load extension spec for {so}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_devices(value: str) -> list[int]:
    devices = [int(item.strip()) for item in value.split(",") if item.strip()]
    if len(devices) < 2:
        raise ValueError("at least two device ids are required")
    if len(set(devices)) != len(devices):
        raise ValueError("device ids must not contain duplicates")
    if any(device < 0 for device in devices):
        raise ValueError("device ids must be non-negative")
    return devices


def parse_size_one(value: str) -> int:
    text = value.strip().lower()
    if not text:
        raise ValueError("empty size")

    multiplier = 1
    suffixes = (
        ("gib", 1024**3),
        ("gb", 1024**3),
        ("g", 1024**3),
        ("mib", 1024**2),
        ("mb", 1024**2),
        ("m", 1024**2),
        ("kib", 1024),
        ("kb", 1024),
        ("k", 1024),
        ("b", 1),
    )
    for suffix, scale in suffixes:
        if text.endswith(suffix):
            text = text[: -len(suffix)]
            multiplier = scale
            break

    if not text:
        raise ValueError(f"invalid size: {value!r}")

    size = float(text) * multiplier
    if size <= 0 or not size.is_integer():
        raise ValueError(f"size must resolve to a positive whole byte count: {value!r}")
    return int(size)


def parse_sizes(value: str) -> list[int]:
    sizes = [parse_size_one(item) for item in value.split(",") if item.strip()]
    if not sizes:
        raise ValueError("size list is empty")
    return sizes


def parse_ctas(value: str | None) -> list[int | None]:
    if value is None or not value.strip():
        return [None]

    text = value.strip().lower()
    if text == "all":
        return [None, 1, 2, 4, 8]

    parsed: list[int | None] = []
    for item in text.split(","):
        token = item.strip()
        if not token:
            continue
        if token in ("default", "none", "unset", "unlimited"):
            ctas = None
        else:
            ctas = int(token)
            if ctas <= 0:
                raise ValueError("--ctas values must be positive")
        if ctas not in parsed:
            parsed.append(ctas)

    return parsed or [None]


def parse_optional_positive_int(
    value: str | None,
    option_name: str,
) -> int | None:
    if value is None:
        return None

    token = value.strip().lower()
    if token in ("default", "none", "unset"):
        return None

    try:
        parsed = int(token)
    except ValueError as exc:
        raise ValueError(
            f"{option_name} must be a positive integer or 'default'"
        ) from exc

    if parsed <= 0:
        raise ValueError(f"{option_name} must be positive or 'default'")
    return parsed



def parse_optional_nccl_ctas(value: str | None) -> int | None:
    """Return None for -1/default, otherwise a positive NCCL CTA limit."""
    if value is None:
        return None

    token = value.strip().lower()
    if token in ("-1", "default", "none", "unset", "unlimited"):
        return None

    try:
        parsed = int(token)
    except ValueError as exc:
        raise ValueError(
            "--nccl-ctas must be a positive integer or -1 for NCCL default"
        ) from exc

    if parsed <= 0:
        raise ValueError(
            "--nccl-ctas must be a positive integer or -1 for NCCL default"
        )
    return parsed

def bytes_to_numel(size_bytes: int) -> int:
    if size_bytes <= 0:
        raise ValueError("buffer size must be positive")
    if size_bytes % FP16_BYTES != 0:
        raise ValueError(
            f"buffer size {size_bytes} is not divisible by {FP16_BYTES} for fp16"
        )
    return size_bytes // FP16_BYTES


def format_size(size_bytes: int) -> str:
    for scale, suffix in ((1024**3, "G"), (1024**2, "M"), (1024, "K")):
        if size_bytes >= scale:
            value = size_bytes / scale
            return f"{int(value)}{suffix}" if value.is_integer() else f"{value:.1f}{suffix}"
    return f"{size_bytes}B"


def selected_metrics(metric: str) -> list[str]:
    metrics: list[str] = []
    if metric in ("latency", "both", "all"):
        metrics.append("latency")
    if metric in ("bandwidth", "both", "all"):
        metrics.append("bandwidth")
    return metrics


def selected_collectives(value: str) -> list[str]:
    return list(COLLECTIVES) if value == "all" else [value]


def validate_sizes_for_collective(
    collective: str,
    sizes_bytes: list[int],
    world_size: int,
) -> None:
    numels = [bytes_to_numel(size_bytes) for size_bytes in sizes_bytes]
    if collective not in ("reduce_scatter", "all_gather"):
        return

    invalid = [
        sizes_bytes[index]
        for index, numel in enumerate(numels)
        if numel % world_size != 0
    ]
    if invalid:
        rendered = ", ".join(format_size(value) for value in invalid)
        raise ValueError(
            f"{collective} fp16 element counts must be divisible by world size "
            f"{world_size}: {rendered}"
        )


def metric_sizes(args: argparse.Namespace, metric: str) -> list[int]:
    override = getattr(args, f"{metric}_bytes", None)
    return parse_sizes(override if override else args.bytes)


def build_jobs(args: argparse.Namespace, devices: list[int]) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    collectives = selected_collectives(args.collective)

    if args.mode in ("smoke", "both"):
        smoke_sizes = [parse_size_one(args.smoke_bytes)]
        for collective in collectives:
            validate_sizes_for_collective(collective, smoke_sizes, len(devices))
            jobs.append(
                {
                    "mode": "smoke",
                    "metric_set": "smoke",
                    "collective": collective,
                    "sizes_bytes": smoke_sizes,
                    "iters": 1,
                    "warmup": 0,
                    "verify": True,
                }
            )

    if args.mode in ("bench", "both"):
        for metric in selected_metrics(args.metric):
            sizes = metric_sizes(args, metric)
            for collective in collectives:
                validate_sizes_for_collective(collective, sizes, len(devices))
                jobs.append(
                    {
                        "mode": "bench",
                        "metric_set": metric,
                        "collective": collective,
                        "sizes_bytes": sizes,
                        "iters": args.iters,
                        "warmup": args.warmup,
                        "verify": bool(args.verify),
                    }
                )

    if not jobs:
        raise ValueError("no benchmark jobs were selected")
    return jobs


def bandwidth_gbps(size_bytes: float, latency_ms: float) -> float:
    if size_bytes <= 0.0 or latency_ms <= 0.0:
        return 0.0
    return (size_bytes / 1.0e9) / (latency_ms / 1.0e3)


def normalize_row(
    raw_row: dict[str, Any],
    job: dict[str, Any],
    request: dict[str, Any],
) -> dict[str, Any]:
    row = {key: float(value) for key, value in raw_row.items()}
    collective_code = row.pop("collective", 0.0)

    oo_ms = float(row.get("ooverlap_ms", 0.0))
    nccl_ms = float(row.get("nccl_ms", 0.0))
    nccl_symmetric_ms = float(row.get("nccl_symmetric_ms", 0.0))
    normalized_bytes = float(
        row.get("bandwidth_bytes_per_rank", row.get("bytes", 0.0))
    )
    cta_limit = request.get("cta_limit")
    result: dict[str, Any] = {
        "timestamp_utc": request["timestamp_utc"],
        "run_id": request["run_id"],
        "mode": job["mode"],
        "metric_set": job["metric_set"],
        "collective": job["collective"],
        "collective_code": collective_code,
        "cta_limit": cta_limit,
        "ooverlap_max_ctas": os.environ.get("OOVERLAP_MAX_CTAS", ""),
        "nccl_max_ctas": os.environ.get("NCCL_MAX_CTAS", ""),
        "ooverlap_max_ctas_per_reduce_task": os.environ.get(
            REDUCE_TASK_CTA_ENV_VAR, ""
        ),
        "devices": ",".join(str(device) for device in request["devices"]),
        **row,
        "ooverlap_latency_us": oo_ms * 1000.0,
        "nccl_latency_us": nccl_ms * 1000.0,
        "nccl_symmetric_latency_us": nccl_symmetric_ms * 1000.0,
        "ooverlap_bandwidth_gbps": bandwidth_gbps(normalized_bytes, oo_ms),
        "nccl_bandwidth_gbps": bandwidth_gbps(normalized_bytes, nccl_ms),
        "nccl_symmetric_bandwidth_gbps": bandwidth_gbps(
            normalized_bytes, nccl_symmetric_ms
        ),
    }
    return result


def run_worker(request_path: Path, output_path: Path) -> None:
    request = json.loads(request_path.read_text(encoding="utf-8"))
    devices = [int(device) for device in request["devices"]]

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    if max(devices) >= torch.cuda.device_count():
        raise ValueError(
            f"requested devices {devices}, but CUDA device count is "
            f"{torch.cuda.device_count()}"
        )

    ext = load_ooverlap_ext()
    if not hasattr(ext, "benchmark_external_p2p_collective_sweep_sm90"):
        raise AttributeError(
            "ooverlap_ext does not expose "
            "benchmark_external_p2p_collective_sweep_sm90; rebuild the extension"
        )

    cta_label = request.get("cta_limit")
    nccl_cta_limit = request.get("nccl_cta_limit")
    reduce_task_ctas = request.get("max_ctas_per_reduce_task")
    tuning_policy = os.environ.get(TUNING_POLICY_ENV_VAR, "default")
    ring_size = os.environ.get("OOVERLAP_BENCH_RING_SIZE", "16")
    print(
        f"[worker] ooverlap_ctas={cta_label if cta_label is not None else 'default'} "
        f"nccl_ctas={nccl_cta_limit if nccl_cta_limit is not None else 'default'} "
        f"reduce_task_ctas={reduce_task_ctas if reduce_task_ctas is not None else 'default'} "
        f"tuning_policy={tuning_policy} "
        f"bandwidth_ring_size={ring_size} "
        f"devices={devices} torch={torch.__version__}",
        flush=True,
    )

    rows: list[dict[str, Any]] = []
    cache: dict[tuple[Any, ...], list[dict[str, Any]]] = {}

    for job in request["jobs"]:
        sleep(5)
        sizes_bytes = [int(value) for value in job["sizes_bytes"]]
        numels = [bytes_to_numel(value) for value in sizes_bytes]
        use_ring = (
            job["mode"] == "bench"
            and (
                bool(request.get("use_ring_for_all_metrics", False))
                or job["metric_set"] == "bandwidth"
            )
        )
        key = (
            use_ring,
            job["collective"],
            tuple(numels),
            int(job["iters"]),
            int(job["warmup"]),
            bool(job["verify"]),
        )

        print(
            f"[worker] mode={job['mode']} metric={job['metric_set']} "
            f"collective={job['collective']} "
            f"sizes={[format_size(value) for value in sizes_bytes]}",
            flush=True,
        )

        raw_rows = cache.get(key)
        if raw_rows is None:
            raw_rows = ext.benchmark_external_p2p_collective_sweep_sm90(
                job["collective"],
                numels,
                int(job["iters"]),
                int(job["warmup"]),
                devices,
                bool(job["verify"]),
                use_ring,
            )
            raw_rows = [dict(row) for row in raw_rows]
            cache[key] = raw_rows

        rows.extend(normalize_row(row, job, request) for row in raw_rows)

    for device in devices:
        torch.cuda.synchronize(device)

    output_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")


def worker_environment(
    cta_limit: int | None,
    nccl_cta_limit: int | None,
    max_ctas_per_reduce_task: int | None,
    tuning_policy: Path | None,
) -> dict[str, str]:
    env = os.environ.copy()

    # Clear inherited values first, then apply the independently requested limits.
    env.pop(OOVERLAP_CTA_ENV_VAR, None)
    env.pop(NCCL_CTA_ENV_VAR, None)
    env.pop(REDUCE_TASK_CTA_ENV_VAR, None)

    if cta_limit is not None:
        env[OOVERLAP_CTA_ENV_VAR] = str(cta_limit)
    if nccl_cta_limit is not None:
        env[NCCL_CTA_ENV_VAR] = str(nccl_cta_limit)
    if max_ctas_per_reduce_task is not None:
        env[REDUCE_TASK_CTA_ENV_VAR] = str(max_ctas_per_reduce_task)
    if tuning_policy is not None:
        env[TUNING_POLICY_ENV_VAR] = str(tuning_policy)
    return env


def run_cta_worker(
    cta_limit: int | None,
    nccl_cta_limit: int | None,
    max_ctas_per_reduce_task: int | None,
    tuning_policy: Path | None,
    use_ring_for_all_metrics: bool,
    jobs: list[dict[str, Any]],
    devices: list[int],
    run_id: str,
    timestamp_utc: str,
) -> list[dict[str, Any]]:
    label = "default" if cta_limit is None else str(cta_limit)
    environment = worker_environment(
        cta_limit,
        nccl_cta_limit,
        max_ctas_per_reduce_task,
        tuning_policy,
    )
    rows: list[dict[str, Any]] = []

    with tempfile.TemporaryDirectory(prefix="ooverlap-external-sweep-") as temp_dir:
        temp = Path(temp_dir)

        for job_index, job in enumerate(jobs, start=1):
            request = {
                "run_id": run_id,
                "timestamp_utc": timestamp_utc,
                "cta_limit": cta_limit,
                "nccl_cta_limit": nccl_cta_limit,
                "max_ctas_per_reduce_task": max_ctas_per_reduce_task,
                "use_ring_for_all_metrics": use_ring_for_all_metrics,
                "devices": devices,
                "jobs": [job],
            }

            request_path = temp / f"request-{job_index}.json"
            output_path = temp / f"rows-{job_index}.json"
            request_path.write_text(
                json.dumps(request, indent=2),
                encoding="utf-8",
            )

            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--_worker-request",
                str(request_path),
                "--_worker-output",
                str(output_path),
            ]
            job_label = (
                f"{job['mode']}/{job['metric_set']}/{job['collective']}"
            )
            print(
                f"[parent] starting fresh CTA/job worker: {label} "
                f"job={job_index}/{len(jobs)} {job_label}",
                flush=True,
            )
            subprocess.run(
                command,
                env=environment,
                check=True,
            )

            if not output_path.exists():
                raise RuntimeError(
                    f"CTA/job worker {label} {job_label} did not create "
                    f"{output_path}"
                )

            job_rows = json.loads(output_path.read_text(encoding="utf-8"))
            if not isinstance(job_rows, list):
                raise RuntimeError(
                    f"CTA/job worker {label} {job_label} returned a "
                    "non-list result"
                )
            rows.extend(job_rows)
            sleep(5)

    return rows

def csv_value(value: Any) -> Any:
    if value is None:
        return ""
    return value


def write_jsonl(path: Path, rows: list[dict[str, Any]], append: bool) -> None:
    mode = "a" if append else "w"
    with path.open(mode, encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True))
            handle.write("\n")


def write_csv(path: Path, rows: list[dict[str, Any]], append: bool) -> None:
    write_header = not append or not path.exists() or path.stat().st_size == 0
    mode = "a" if append else "w"
    with path.open(mode, encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        for row in rows:
            writer.writerow({key: csv_value(row.get(key, "")) for key in OUTPUT_COLUMNS})


def write_text(path: Path, rows: list[dict[str, Any]], append: bool) -> None:
    write_header = not append or not path.exists() or path.stat().st_size == 0
    mode = "a" if append else "w"
    with path.open(mode, encoding="utf-8") as handle:
        if write_header:
            handle.write("\t".join(TEXT_COLUMNS))
            handle.write("\n")
        for row in rows:
            values = [csv_value(row.get(key, "")) for key in TEXT_COLUMNS]
            handle.write("\t".join(str(value) for value in values))
            handle.write("\n")


def write_outputs(
    out_prefix: Path,
    rows: list[dict[str, Any]],
    append: bool,
) -> tuple[Path, Path, Path]:
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    csv_path = Path(f"{out_prefix}.csv")
    jsonl_path = Path(f"{out_prefix}.jsonl")
    text_path = Path(f"{out_prefix}.txt")

    write_csv(csv_path, rows, append)
    write_jsonl(jsonl_path, rows, append)
    write_text(text_path, rows, append)
    return csv_path, jsonl_path, text_path


# OOVERLAP_EXTERNAL_P2P_SPEEDUP_SUMMARY_V1
SPEEDUP_BASELINES = (
    ("nccl", "NCCL"),
    ("nccl_symmetric", "symmetric NCCL"),
)


def positive_finite_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed) or parsed <= 0.0:
        return None
    return parsed


def collect_speedup_points(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    points: list[dict[str, Any]] = []

    for row in rows:
        if str(row.get("mode", "")) != "bench":
            continue

        metric = str(row.get("metric_set", ""))
        if metric not in ("latency", "bandwidth"):
            continue

        if metric == "latency":
            ooverlap_value = positive_finite_float(
                row.get("ooverlap_latency_us")
            )
        else:
            ooverlap_value = positive_finite_float(
                row.get("ooverlap_bandwidth_gbps")
            )

        if ooverlap_value is None:
            continue

        for baseline_key, baseline_label in SPEEDUP_BASELINES:
            if metric == "latency":
                baseline_value = positive_finite_float(
                    row.get(f"{baseline_key}_latency_us")
                )
                speedup = (
                    baseline_value / ooverlap_value
                    if baseline_value is not None
                    else None
                )
            else:
                baseline_value = positive_finite_float(
                    row.get(f"{baseline_key}_bandwidth_gbps")
                )
                speedup = (
                    ooverlap_value / baseline_value
                    if baseline_value is not None
                    else None
                )

            if speedup is None or not math.isfinite(speedup) or speedup <= 0.0:
                continue

            points.append(
                {
                    "metric": metric,
                    "baseline_key": baseline_key,
                    "baseline_label": baseline_label,
                    "speedup": speedup,
                    "row": row,
                }
            )

    return points


def format_speedup_point(point: dict[str, Any]) -> str:
    speedup = float(point["speedup"])
    row = point["row"]
    metric = str(point["metric"])
    collective = str(row.get("collective", "unknown")).replace("_", "-")

    try:
        size_label = format_size(int(float(row.get("bytes", 0))))
    except (TypeError, ValueError):
        size_label = str(row.get("bytes", "unknown"))

    cta_limit = row.get("cta_limit")
    cta_label = "default" if cta_limit in (None, "") else str(cta_limit)
    improvement_percent = (speedup - 1.0) * 100.0

    return (
        f"{speedup:.4f}x ({improvement_percent:+.2f}%) "
        f"at metric={metric}, collective={collective}, "
        f"size={size_label}, OOverlap CTAs={cta_label}"
    )


def speedup_extreme_line(
    points: list[dict[str, Any]],
    baseline_key: str,
    baseline_label: str,
    *,
    metric: str | None = None,
    collective: str | None = None,
) -> str:
    selected = [
        point
        for point in points
        if point["baseline_key"] == baseline_key
        and (metric is None or point["metric"] == metric)
        and (
            collective is None
            or str(point["row"].get("collective", "")) == collective
        )
    ]

    if not selected:
        return f"  versus {baseline_label}: no valid measurements"

    minimum = min(selected, key=lambda point: float(point["speedup"]))
    maximum = max(selected, key=lambda point: float(point["speedup"]))
    return (
        f"  versus {baseline_label}:\n"
        f"    minimum: {format_speedup_point(minimum)}\n"
        f"    maximum: {format_speedup_point(maximum)}"
    )


def build_speedup_summary(
    rows: list[dict[str, Any]],
    *,
    run_id: str,
    devices: list[int],
) -> str:
    points = collect_speedup_points(rows)
    lines = [
        "External P2P collective speedup summary",
        f"run_id: {run_id}",
        f"devices: {','.join(str(device) for device in devices)}",
        f"world_size: {len(devices)}",
        "scope: current invocation only",
        "latency speedup: baseline latency / OOverlap latency",
        "bandwidth speedup: OOverlap bandwidth / baseline bandwidth",
        "",
        "Overall across latency and bandwidth points",
    ]

    for baseline_key, baseline_label in SPEEDUP_BASELINES:
        lines.append(
            speedup_extreme_line(
                points,
                baseline_key,
                baseline_label,
            )
        )

    for metric in ("latency", "bandwidth"):
        lines.extend(("", metric.capitalize()))
        for baseline_key, baseline_label in SPEEDUP_BASELINES:
            lines.append(
                speedup_extreme_line(
                    points,
                    baseline_key,
                    baseline_label,
                    metric=metric,
                )
            )

    lines.extend(("", "By collective"))
    for collective in COLLECTIVES:
        lines.append(collective.replace("_", "-").title())
        for metric in ("latency", "bandwidth"):
            lines.append(f"  {metric}")
            for baseline_key, baseline_label in SPEEDUP_BASELINES:
                detail = speedup_extreme_line(
                    points,
                    baseline_key,
                    baseline_label,
                    metric=metric,
                    collective=collective,
                )
                lines.append("  " + detail.replace("\n", "\n  "))

    return "\n".join(lines) + "\n"


def write_speedup_summary(
    out_prefix: Path,
    rows: list[dict[str, Any]],
    *,
    run_id: str,
    devices: list[int],
) -> tuple[Path, str]:
    summary_path = Path(f"{out_prefix}_speedup.txt")
    summary_text = build_speedup_summary(
        rows,
        run_id=run_id,
        devices=devices,
    )
    summary_path.write_text(summary_text, encoding="utf-8")
    return summary_path, summary_text


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Collect in-process external-P2P collective measurements without "
            "plotting. OOverlap and NCCL CTA limits are configured independently "
            "before each fresh worker process starts."
        )
    )
    parser.add_argument("--mode", choices=("smoke", "bench", "both"), default="bench")
    parser.add_argument(
        "--collective",
        choices=("all",) + COLLECTIVES,
        default="all",
    )
    parser.add_argument(
        "--metric",
        choices=("latency", "bandwidth", "both", "all"),
        default="all",
    )
    parser.add_argument(
        "--ctas",
        default=None,
        help=(
            "comma-separated OOverlap CTA limits, for example default,4,8. "
            "Values set OOVERLAP_MAX_CTAS only"
        ),
    )
    parser.add_argument(
        "--nccl-ctas",
        default="-1",
        help=(
            "NCCL_MAX_CTAS limit. Use a positive integer, or -1/default to "
            "leave NCCL_MAX_CTAS unset and use NCCL's default policy"
        ),
    )
    parser.add_argument(
        "--max-ctas-per-reduce-task",
        default="8",
        help=(
            "positive OOVERLAP_MAX_CTAS_PER_REDUCE_TASK value. Defaults to 8; "
            "use default or unset to leave the environment variable unset"
        ),
    )
    parser.add_argument(
        "--tuning-policy",
        default=None,
        help=(
            "optional path to the OOverlap tuning-policy JSON file. If it "
            "exists, the resolved path is passed to each fresh worker as "
            "OOVERLAP_TUNING_POLICY; otherwise a warning is printed and the "
            "runtime fallback is used"
        ),
    )
    parser.add_argument(
        "--bytes",
        default="1M,2M,4M,8M,16M,32M,64M,128M,256M,512M",
        help="default byte-size list used when a metric-specific list is omitted",
    )
    parser.add_argument("--latency-bytes", default=None)
    parser.add_argument("--bandwidth-bytes", default=None)
    parser.add_argument("--smoke-bytes", default="1M")
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument(
        "--use-ring-for-all-metrics",
        action="store_true",
        help=(
            "use the batched rotating-buffer benchmark for latency jobs as "
            "well as bandwidth jobs"
        ),
    )
    parser.add_argument(
        "--devices",
        default=None,
        help="comma-separated device ids, for example 0,1,2,3",
    )
    parser.add_argument("--dev0", type=int, default=0)
    parser.add_argument("--dev1", type=int, default=1)
    parser.add_argument(
        "--out-prefix",
        default="results/external_p2p_collective",
        help="output path prefix; .csv, .jsonl, and .txt are written",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="append to existing output files instead of replacing them",
    )

    parser.add_argument("--_worker-request", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--_worker-output", default=None, help=argparse.SUPPRESS)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args._worker_request is not None:
        if args._worker_output is None:
            parser.error("--_worker-output is required with --_worker-request")
        run_worker(Path(args._worker_request), Path(args._worker_output))
        return

    if args.iters <= 0:
        parser.error("--iters must be > 0")
    if args.warmup < 0:
        parser.error("--warmup must be >= 0")

    devices = (
        parse_devices(args.devices)
        if args.devices is not None
        else parse_devices(f"{args.dev0},{args.dev1}")
    )
    cta_values = parse_ctas(args.ctas)
    try:
        nccl_cta_limit = parse_optional_nccl_ctas(args.nccl_ctas)
        max_ctas_per_reduce_task = parse_optional_positive_int(
            args.max_ctas_per_reduce_task,
            "--max-ctas-per-reduce-task",
        )
    except ValueError as exc:
        parser.error(str(exc))

    tuning_policy: Path | None = None
    if args.tuning_policy is not None:
        requested_tuning_policy = Path(args.tuning_policy).expanduser().resolve()
        if requested_tuning_policy.is_file():
            tuning_policy = requested_tuning_policy
        else:
            print(
                "[warning] tuning policy not found: "
                f"{requested_tuning_policy}; continuing without "
                f"{TUNING_POLICY_ENV_VAR}",
                file=sys.stderr,
            )

    jobs = build_jobs(args, devices)

    timestamp_utc = datetime.now(timezone.utc).isoformat()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")

    print(f"[info] devices={devices} world_size={len(devices)}")
    print(f"[info] ooverlap_ctas={cta_values}")
    print(
        "[info] nccl_ctas="
        f"{nccl_cta_limit if nccl_cta_limit is not None else 'default'}"
    )
    print(
        "[info] max_ctas_per_reduce_task="
        f"{max_ctas_per_reduce_task if max_ctas_per_reduce_task is not None else 'default'}"
    )
    print(
        "[info] tuning_policy="
        f"{tuning_policy if tuning_policy is not None else 'inherited/default'}"
    )
    print(f"[info] jobs={len(jobs)} iters={args.iters} warmup={args.warmup}")
    print(
        "[info] benchmark_timing="
        + (
            "batched-ring-all-metrics"
            if args.use_ring_for_all_metrics
            else "isolated-latency/batched-ring-bandwidth"
        )
    )
    print("[info] no plotting; saving CSV, JSONL, and tab-separated text")

    all_rows: list[dict[str, Any]] = []
    for cta_limit in cta_values:
        all_rows.extend(
            run_cta_worker(
                cta_limit,
                nccl_cta_limit,
                max_ctas_per_reduce_task,
                tuning_policy,
                bool(args.use_ring_for_all_metrics),
                jobs,
                devices,
                run_id,
                timestamp_utc,
            )
        )

    out_prefix = Path(args.out_prefix)
    csv_path, jsonl_path, text_path = write_outputs(
        out_prefix,
        all_rows,
        args.append,
    )
    speedup_path, speedup_text = write_speedup_summary(
        out_prefix,
        all_rows,
        run_id=run_id,
        devices=devices,
    )
    print(f"[result] rows={len(all_rows)}")
    print(f"[result] csv={csv_path}")
    print(f"[result] jsonl={jsonl_path}")
    print(f"[result] text={text_path}")
    print(f"[result] speedup={speedup_path}")
    print()
    print(speedup_text.rstrip())


if __name__ == "__main__":
    main()
