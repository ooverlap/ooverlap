#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


COLLECTIVES = ("allreduce", "reduce_scatter", "all_gather")
FP16_BYTES = 2
CTA_ENV_VARS = ("OOVERLAP_MAX_CTAS", "NCCL_MAX_CTAS")

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
    "ooverlap_speedup_over_nccl",
    "ooverlap_speedup_over_nccl_symmetric",
    "nccl_symmetric_speedup_over_nccl",
]

TEXT_COLUMNS = [
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
    "ooverlap_speedup_over_nccl",
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
    if metric in ("speedup", "all"):
        metrics.append("speedup")
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


def safe_ratio(numerator: float, denominator: float) -> float:
    return 0.0 if denominator <= 0.0 else numerator / denominator


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
        "ooverlap_speedup_over_nccl": safe_ratio(nccl_ms, oo_ms),
        "ooverlap_speedup_over_nccl_symmetric": safe_ratio(
            nccl_symmetric_ms, oo_ms
        ),
        "nccl_symmetric_speedup_over_nccl": safe_ratio(
            nccl_ms, nccl_symmetric_ms
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
    print(
        f"[worker] ctas={cta_label if cta_label is not None else 'default'} "
        f"devices={devices} torch={torch.__version__}",
        flush=True,
    )

    rows: list[dict[str, Any]] = []
    cache: dict[tuple[Any, ...], list[dict[str, Any]]] = {}

    for job in request["jobs"]:
        sizes_bytes = [int(value) for value in job["sizes_bytes"]]
        numels = [bytes_to_numel(value) for value in sizes_bytes]
        key = (
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
            )
            raw_rows = [dict(row) for row in raw_rows]
            cache[key] = raw_rows

        rows.extend(normalize_row(row, job, request) for row in raw_rows)

    for device in devices:
        torch.cuda.synchronize(device)

    output_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")


def worker_environment(cta_limit: int | None) -> dict[str, str]:
    env = os.environ.copy()
    for name in CTA_ENV_VARS:
        env.pop(name, None)
    if cta_limit is not None:
        value = str(cta_limit)
        for name in CTA_ENV_VARS:
            env[name] = value
    return env


def run_cta_worker(
    cta_limit: int | None,
    jobs: list[dict[str, Any]],
    devices: list[int],
    run_id: str,
    timestamp_utc: str,
) -> list[dict[str, Any]]:
    request = {
        "run_id": run_id,
        "timestamp_utc": timestamp_utc,
        "cta_limit": cta_limit,
        "devices": devices,
        "jobs": jobs,
    }

    with tempfile.TemporaryDirectory(prefix="ooverlap-external-sweep-") as temp_dir:
        temp = Path(temp_dir)
        request_path = temp / "request.json"
        output_path = temp / "rows.json"
        request_path.write_text(json.dumps(request, indent=2), encoding="utf-8")

        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--_worker-request",
            str(request_path),
            "--_worker-output",
            str(output_path),
        ]
        label = "default" if cta_limit is None else str(cta_limit)
        print(f"[parent] starting fresh CTA worker: {label}", flush=True)
        subprocess.run(command, env=worker_environment(cta_limit), check=True)

        if not output_path.exists():
            raise RuntimeError(f"CTA worker {label} did not create {output_path}")
        return json.loads(output_path.read_text(encoding="utf-8"))


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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Collect in-process external-P2P collective measurements without "
            "plotting. Each CTA setting runs in a fresh process so Ooverlap and "
            "NCCL see the requested CTA environment before initialization."
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
        choices=("latency", "bandwidth", "speedup", "both", "all"),
        default="all",
    )
    parser.add_argument(
        "--ctas",
        default=None,
        help=(
            "comma-separated CTA limits, for example default,4,8. The same value "
            "is set in OOVERLAP_MAX_CTAS and NCCL_MAX_CTAS. Omit this option or "
            "use default to leave both variables unset"
        ),
    )
    parser.add_argument(
        "--bytes",
        default="1M,2M,4M,8M,16M,32M,64M,128M,256M,512M",
        help="default byte-size list used when a metric-specific list is omitted",
    )
    parser.add_argument("--latency-bytes", default=None)
    parser.add_argument("--bandwidth-bytes", default=None)
    parser.add_argument("--speedup-bytes", default=None)
    parser.add_argument("--smoke-bytes", default="1M")
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--verify", action="store_true")
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
    jobs = build_jobs(args, devices)

    timestamp_utc = datetime.now(timezone.utc).isoformat()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")

    print(f"[info] devices={devices} world_size={len(devices)}")
    print(f"[info] ctas={cta_values}")
    print(f"[info] jobs={len(jobs)} iters={args.iters} warmup={args.warmup}")
    print("[info] no plotting; saving CSV, JSONL, and tab-separated text")

    all_rows: list[dict[str, Any]] = []
    for cta_limit in cta_values:
        all_rows.extend(
            run_cta_worker(
                cta_limit,
                jobs,
                devices,
                run_id,
                timestamp_utc,
            )
        )

    csv_path, jsonl_path, text_path = write_outputs(
        Path(args.out_prefix),
        all_rows,
        args.append,
    )
    print(f"[result] rows={len(all_rows)}")
    print(f"[result] csv={csv_path}")
    print(f"[result] jsonl={jsonl_path}")
    print(f"[result] text={text_path}")


if __name__ == "__main__":
    main()
