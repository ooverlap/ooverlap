#!/usr/bin/env python3

"""Tune Ooverlap CTA limits and generate tma_collective_policy.json.

The parent process launches one fresh worker process for each
(max_ctas, max_ctas_per_reduce_task) candidate. A worker loads the extension
once and benchmarks every requested collective and message size in same-process
multi-GPU mode.

The final policy intentionally contains only:
  world_size, collective, bytes, ranked CTA pairs.

Detailed timing data is written separately for inspection.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence


DEFAULT_NUMELS = sorted(
    {
        32768,
        65536,
        131072,
        262144,
        524288,
        623104,
        741376,
        881664,
        1048576,
        1246720,
        1482752,
        1763328,
        2097152,
        2493440,
        2965504,
        3526656,
        4194304,
        4987392,
        5931520,
        7053824,
        8388608,
        9975296,
        11863040,
        14107648,
        16777216,
        19951104,
        23726080,
        28215296,
        33554432,
        39902720,
        47452672,
        56431104,
        67108864,
        79805952,
        94905856,
        112862720,
        134217728,
        159612416,
        189812224,
        225725952,
        268435456,
        536870912,
    }
)

DEFAULT_COLLECTIVES = (
    "allreduce",
    "reduce_scatter",
    "all_gather",
)

# Edit this list or pass --cta-candidates. Each tuple gets a fresh process.
DEFAULT_CTA_CANDIDATES = (
    (3, 1),
    (6, 2),
    (9, 3),
    (12, 4),
    (15, 5),
    (18, 6),
    (21, 7),
    (24, 8),
    (27, 9),

    (1, 1),
(2, 2),
(3, 3),
(4, 4),
(5, 5),
(6, 6),
(7, 7),
(8, 8),
(9, 9),
(10, 10),
(12, 12),
(15, 15),
(18, 18),
)

ENV_MAX_CTAS = "OOVERLAP_MAX_CTAS"
ENV_MAX_CTAS_PER_REDUCE_TASK = "OOVERLAP_MAX_CTAS_PER_REDUCE_TASK"
ENV_TUNING_POLICY = "OOVERLAP_TUNING_POLICY"
MAX_CTAS = 32
FP16_BYTES = 2


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_extension_path() -> Path:
    return repo_root() / "build" / "lib" / "ooverlap_ext.so"


def default_work_dir() -> Path:
    return repo_root() / "results" / "tma_collective_cta_tuning"


def default_policy_path() -> Path:
    # This matches the runtime loader's default relative path.
    return repo_root() / "tma_collective_policy.json"


def load_extension(path: Path):
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"could not find extension: {path}")

    spec = importlib.util.spec_from_file_location("ooverlap_ext", str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not create import spec for {path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    function_name = "benchmark_tma_collective_cta_sweep_sm90"
    if not hasattr(module, function_name):
        raise RuntimeError(
            f"{path} does not export {function_name}; rebuild the new-policy branch"
        )
    return module


def parse_csv_ints(text: str, name: str) -> list[int]:
    values: list[int] = []
    for token in text.split(","):
        token = token.strip()
        if not token:
            continue
        try:
            value = int(token)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                f"{name} contains a non-integer value: {token!r}"
            ) from exc
        values.append(value)

    if not values:
        raise argparse.ArgumentTypeError(f"{name} must not be empty")
    return values


def parse_devices(text: str) -> list[int]:
    devices = parse_csv_ints(text, "devices")
    if len(devices) < 2:
        raise argparse.ArgumentTypeError("at least two devices are required")
    if any(device < 0 for device in devices):
        raise argparse.ArgumentTypeError("device ids must be non-negative")
    if len(set(devices)) != len(devices):
        raise argparse.ArgumentTypeError("device ids must be unique")
    return devices


def parse_numels(text: str) -> list[int]:
    numels = parse_csv_ints(text, "numels")
    if any(numel <= 0 for numel in numels):
        raise argparse.ArgumentTypeError("all numels must be positive")
    return sorted(set(numels))


def normalize_collective(value: str) -> str:
    aliases = {
        "allreduce": "allreduce",
        "all_reduce": "allreduce",
        "all-reduce": "allreduce",
        "ar": "allreduce",
        "reduce_scatter": "reduce_scatter",
        "reduce-scatter": "reduce_scatter",
        "reducescatter": "reduce_scatter",
        "rs": "reduce_scatter",
        "all_gather": "all_gather",
        "all-gather": "all_gather",
        "allgather": "all_gather",
        "ag": "all_gather",
    }
    try:
        return aliases[value.strip().lower()]
    except KeyError as exc:
        raise argparse.ArgumentTypeError(
            f"unknown collective {value!r}; expected allreduce, "
            "reduce_scatter, or all_gather"
        ) from exc


def parse_collectives(text: str) -> list[str]:
    collectives: list[str] = []
    for token in text.split(","):
        token = token.strip()
        if token:
            collective = normalize_collective(token)
            if collective not in collectives:
                collectives.append(collective)
    if not collectives:
        raise argparse.ArgumentTypeError("collectives must not be empty")
    return collectives


def validate_candidate(max_ctas: int, reduce_ctas: int) -> tuple[int, int]:
    if max_ctas <= 0 or max_ctas > MAX_CTAS:
        raise argparse.ArgumentTypeError(
            f"max_ctas must be in [1, {MAX_CTAS}], got {max_ctas}"
        )
    if reduce_ctas <= 0 or reduce_ctas > max_ctas:
        raise argparse.ArgumentTypeError(
            "max_ctas_per_reduce_task must be positive and no greater than max_ctas"
        )
    return max_ctas, reduce_ctas


def parse_candidates(text: str) -> list[tuple[int, int]]:
    candidates: list[tuple[int, int]] = []
    for token in text.split(","):
        token = token.strip()
        if not token:
            continue
        fields = token.split(":")
        if len(fields) != 2:
            raise argparse.ArgumentTypeError(
                "CTA candidates must use MAX_CTAS:REDUCE_CTAS, for example 12:4"
            )
        try:
            candidate = validate_candidate(int(fields[0]), int(fields[1]))
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                f"invalid CTA candidate: {token!r}"
            ) from exc
        if candidate not in candidates:
            candidates.append(candidate)

    if not candidates:
        raise argparse.ArgumentTypeError("CTA candidates must not be empty")
    return candidates


def valid_numels_for_collective(
    collective: str,
    numels: Sequence[int],
    world_size: int,
) -> tuple[list[int], list[int]]:
    if collective == "allreduce":
        return list(numels), []

    valid: list[int] = []
    skipped: list[int] = []
    for numel in numels:
        if numel % world_size == 0:
            valid.append(numel)
        else:
            skipped.append(numel)
    return valid, skipped


def write_json_atomic(path: Path, payload: Any, *, pretty: bool) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")

    if pretty:
        text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    else:
        text = json.dumps(payload, separators=(",", ":"), sort_keys=False) + "\n"

    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def normalize_measurement(
    raw: dict[str, Any],
    collective: str,
    candidate: tuple[int, int],
    world_size: int,
) -> dict[str, Any]:
    max_ctas, reduce_ctas = candidate

    required = (
        "world_size",
        "numel",
        "bytes",
        "iters",
        "warmup",
        "max_ctas",
        "max_ctas_per_reduce_task",
        "total_ms",
        "avg_ms",
        "latency_us",
    )
    missing = [field for field in required if field not in raw]
    if missing:
        raise RuntimeError(
            f"C++ sweep row for {collective} is missing fields: {missing}"
        )

    row = {
        "collective": collective,
        "world_size": int(raw["world_size"]),
        "numel": int(raw["numel"]),
        "bytes": int(raw["bytes"]),
        "iters": int(raw["iters"]),
        "warmup": int(raw["warmup"]),
        "max_ctas": int(raw["max_ctas"]),
        "max_ctas_per_reduce_task": int(
            raw["max_ctas_per_reduce_task"]
        ),
        "total_ms": float(raw["total_ms"]),
        "avg_ms": float(raw["avg_ms"]),
        "latency_us": float(raw["latency_us"]),
    }

    if row["world_size"] != world_size:
        raise RuntimeError(
            f"C++ sweep returned world_size={row['world_size']}, expected {world_size}"
        )
    if (row["max_ctas"], row["max_ctas_per_reduce_task"]) != candidate:
        raise RuntimeError(
            "C++ sweep did not use the requested CTA candidate: "
            f"requested={candidate}, returned="
            f"({row['max_ctas']}, {row['max_ctas_per_reduce_task']})"
        )
    if row["bytes"] != row["numel"] * FP16_BYTES:
        raise RuntimeError("C++ sweep returned an unexpected fp16 byte count")
    if not math.isfinite(row["avg_ms"]) or row["avg_ms"] <= 0.0:
        raise RuntimeError(f"invalid avg_ms in C++ sweep row: {row['avg_ms']}")

    return row


def worker_main(request_path: Path, result_path: Path) -> int:
    request = json.loads(request_path.read_text(encoding="utf-8"))

    candidate = (
        int(os.environ[ENV_MAX_CTAS]),
        int(os.environ[ENV_MAX_CTAS_PER_REDUCE_TASK]),
    )
    validate_candidate(*candidate)

    extension = load_extension(Path(request["extension"]))
    benchmark = extension.benchmark_tma_collective_cta_sweep_sm90

    devices = [int(value) for value in request["devices"]]
    numels = [int(value) for value in request["numels"]]
    collectives = [normalize_collective(value) for value in request["collectives"]]
    iters = int(request["iters"])
    warmup = int(request["warmup"])
    verify = bool(request["verify"])
    world_size = len(devices)

    response: dict[str, Any] = {
        "ok": True,
        "candidate": {
            "max_ctas": candidate[0],
            "max_ctas_per_reduce_task": candidate[1],
        },
        "devices": devices,
        "results": [],
        "skipped": [],
        "errors": [],
    }

    print(
        "[worker] candidate="
        f"({candidate[0]}, {candidate[1]}) devices={devices} "
        f"collectives={collectives} sizes={len(numels)}"
    )

    for collective in collectives:
        valid_numels, skipped_numels = valid_numels_for_collective(
            collective,
            numels,
            world_size,
        )

        for numel in skipped_numels:
            response["skipped"].append(
                {
                    "collective": collective,
                    "world_size": world_size,
                    "numel": numel,
                    "bytes": numel * FP16_BYTES,
                    "reason": "numel is not divisible by world_size",
                }
            )

        if not valid_numels:
            continue

        try:
            raw_rows = benchmark(
                collective,
                valid_numels,
                iters,
                warmup,
                devices,
                verify,
            )

            if len(raw_rows) != len(valid_numels):
                raise RuntimeError(
                    f"C++ sweep returned {len(raw_rows)} rows for "
                    f"{len(valid_numels)} requested sizes"
                )

            normalized = [
                normalize_measurement(
                    dict(raw),
                    collective,
                    candidate,
                    world_size,
                )
                for raw in raw_rows
            ]
            response["results"].extend(normalized)
        except Exception as exc:  # Worker must preserve failures for the parent.
            response["ok"] = False
            response["errors"].append(
                {
                    "collective": collective,
                    "candidate": list(candidate),
                    "error": str(exc),
                }
            )
            print(f"[worker] ERROR collective={collective}: {exc}", file=sys.stderr)

    write_json_atomic(result_path, response, pretty=True)
    return 0 if response["ok"] else 1


def candidate_slug(candidate: tuple[int, int]) -> str:
    return f"ctas_{candidate[0]}__reduce_{candidate[1]}"


def run_candidate_process(
    candidate: tuple[int, int],
    request_path: Path,
    workers_dir: Path,
    logs_dir: Path,
) -> dict[str, Any]:
    slug = candidate_slug(candidate)
    result_path = workers_dir / f"{slug}.json"
    log_path = logs_dir / f"{slug}.log"

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env[ENV_MAX_CTAS] = str(candidate[0])
    env[ENV_MAX_CTAS_PER_REDUCE_TASK] = str(candidate[1])
    # The C++ sweep bypasses policy selection, but clearing this avoids accidental
    # interaction with other extension initialization paths.
    env.pop(ENV_TUNING_POLICY, None)

    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--request",
        str(request_path),
        "--result",
        str(result_path),
    ]

    print(
        f"[parent] run candidate={candidate} "
        f"result={result_path.name} log={log_path.name}"
    )

    process = subprocess.run(
        command,
        cwd=str(repo_root()),
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    log_path.write_text(process.stdout, encoding="utf-8")

    if result_path.is_file():
        try:
            response = json.loads(result_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            response = {
                "ok": False,
                "candidate": {
                    "max_ctas": candidate[0],
                    "max_ctas_per_reduce_task": candidate[1],
                },
                "results": [],
                "skipped": [],
                "errors": [{"error": f"invalid worker JSON: {exc}"}],
            }
    else:
        response = {
            "ok": False,
            "candidate": {
                "max_ctas": candidate[0],
                "max_ctas_per_reduce_task": candidate[1],
            },
            "results": [],
            "skipped": [],
            "errors": [
                {
                    "error": (
                        f"worker exited with code {process.returncode}; "
                        f"see {log_path}"
                    )
                }
            ],
        }

    response["returncode"] = process.returncode
    response["log"] = str(log_path)
    if process.returncode != 0:
        response["ok"] = False

    print(
        f"[parent] candidate={candidate} returncode={process.returncode} "
        f"rows={len(response.get('results', []))}"
    )
    return response


def build_policy(measurements: Iterable[dict[str, Any]]) -> dict[str, Any]:
    # key -> candidate -> best measured avg_ms
    grouped: dict[
        tuple[int, str, int],
        dict[tuple[int, int], float],
    ] = defaultdict(dict)

    for row in measurements:
        world_size = int(row["world_size"])
        collective = normalize_collective(str(row["collective"]))
        bytes_count = int(row["bytes"])
        candidate = (
            int(row["max_ctas"]),
            int(row["max_ctas_per_reduce_task"]),
        )
        avg_ms = float(row["avg_ms"])

        if world_size <= 0 or bytes_count <= 0:
            continue
        if not math.isfinite(avg_ms) or avg_ms <= 0.0:
            continue

        key = (world_size, collective, bytes_count)
        previous = grouped[key].get(candidate)
        if previous is None or avg_ms < previous:
            grouped[key][candidate] = avg_ms

    collective_order = {
        "allreduce": 0,
        "reduce_scatter": 1,
        "all_gather": 2,
    }

    entries: list[dict[str, Any]] = []
    sorted_keys = sorted(
        grouped,
        key=lambda key: (
            key[0],
            collective_order[key[1]],
            key[2],
        ),
    )

    for world_size, collective, bytes_count in sorted_keys:
        candidates = grouped[(world_size, collective, bytes_count)]
        ranked = [
            [candidate[0], candidate[1]]
            for candidate, _ in sorted(
                candidates.items(),
                key=lambda item: (
                    item[1],
                    item[0][0],
                    item[0][1],
                ),
            )
        ]
        if not ranked:
            continue

        entries.append(
            {
                "world_size": world_size,
                "collective": collective,
                "bytes": bytes_count,
                "ranked": ranked,
            }
        )

    return {
        "version": 1,
        "entries": entries,
    }


def requested_policy_keys(request: dict[str, Any]) -> set[tuple[int, str, int]]:
    devices = [int(value) for value in request["devices"]]
    world_size = len(devices)
    numels = [int(value) for value in request["numels"]]

    keys: set[tuple[int, str, int]] = set()
    for collective_value in request["collectives"]:
        collective = normalize_collective(collective_value)
        valid_numels, _ = valid_numels_for_collective(
            collective,
            numels,
            world_size,
        )
        for numel in valid_numels:
            keys.add((world_size, collective, numel * FP16_BYTES))
    return keys


def produced_policy_keys(policy: dict[str, Any]) -> set[tuple[int, str, int]]:
    return {
        (
            int(entry["world_size"]),
            normalize_collective(str(entry["collective"])),
            int(entry["bytes"]),
        )
        for entry in policy.get("entries", [])
    }


def parent_main(args: argparse.Namespace) -> int:
    work_dir = args.work_dir.resolve()
    workers_dir = work_dir / "workers"
    logs_dir = work_dir / "logs"
    workers_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    request = {
        "extension": str(args.extension.resolve()),
        "devices": args.devices,
        "collectives": args.collectives,
        "numels": args.numels,
        "iters": args.iters,
        "warmup": args.warmup,
        "verify": args.verify,
    }
    request_path = work_dir / "request.json"
    write_json_atomic(request_path, request, pretty=True)

    responses: list[dict[str, Any]] = []
    for candidate in args.cta_candidates:
        responses.append(
            run_candidate_process(
                candidate,
                request_path,
                workers_dir,
                logs_dir,
            )
        )

    measurements = [
        row
        for response in responses
        for row in response.get("results", [])
    ]
    skipped = [
        row
        for response in responses
        for row in response.get("skipped", [])
    ]
    errors = [
        {
            "candidate": response.get("candidate"),
            **error,
        }
        for response in responses
        for error in response.get("errors", [])
    ]

    raw_payload = {
        "version": 1,
        "request": request,
        "candidates": [list(candidate) for candidate in args.cta_candidates],
        "measurements": measurements,
        "skipped": skipped,
        "errors": errors,
        "workers": responses,
    }
    write_json_atomic(args.raw_out, raw_payload, pretty=True)

    policy = build_policy(measurements)
    write_json_atomic(args.policy_out, policy, pretty=args.pretty_policy)

    missing = requested_policy_keys(request) - produced_policy_keys(policy)
    failed_workers = [response for response in responses if not response.get("ok", False)]

    print(f"[result] workers:       {len(responses)}")
    print(f"[result] failed workers:{len(failed_workers):>8}")
    print(f"[result] measurements:  {len(measurements)}")
    print(f"[result] policy entries:{len(policy['entries']):>8}")
    print(f"[result] missing points:{len(missing):>8}")
    print(f"[result] raw:           {args.raw_out.resolve()}")
    print(f"[result] policy:        {args.policy_out.resolve()}")

    if missing:
        print("[missing]")
        for world_size, collective, bytes_count in sorted(missing)[:50]:
            print(
                f"  world_size={world_size} collective={collective} "
                f"bytes={bytes_count}"
            )

    if errors:
        print("[errors]")
        for error in errors[:50]:
            print(json.dumps(error, sort_keys=True))

    if failed_workers or missing:
        print("FAIL")
        return 1

    print("PASS")
    return 0


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark CTA-limit candidates in fresh processes and generate "
            "the simplified Ooverlap tuning policy"
        )
    )

    parser.add_argument(
        "--devices",
        type=parse_devices,
        default=[0, 1],
        help="comma-separated CUDA device ids (default: 0,1)",
    )
    parser.add_argument(
        "--collectives",
        type=parse_collectives,
        default=list(DEFAULT_COLLECTIVES),
        help="comma-separated collectives",
    )
    parser.add_argument(
        "--numels",
        type=parse_numels,
        default=list(DEFAULT_NUMELS),
        help="comma-separated full logical fp16 element counts",
    )
    parser.add_argument(
        "--cta-candidates",
        type=parse_candidates,
        default=list(DEFAULT_CTA_CANDIDATES),
        help=(
            "comma-separated MAX_CTAS:REDUCE_CTAS pairs, for example "
            "4:2,8:4,12:4,16:8"
        ),
    )
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument(
        "--extension",
        type=Path,
        default=default_extension_path(),
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=default_work_dir(),
    )
    parser.add_argument(
        "--raw-out",
        type=Path,
        default=default_work_dir() / "measurements.json",
    )
    parser.add_argument(
        "--policy-out",
        type=Path,
        default=default_policy_path(),
    )
    parser.add_argument(
        "--pretty-policy",
        action="store_true",
        help="pretty-print the runtime policy instead of compact JSON",
    )

    # Internal worker arguments.
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--request", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--result", type=Path, help=argparse.SUPPRESS)
    return parser


def validate_cli(args: argparse.Namespace) -> None:
    if args.iters <= 0:
        raise SystemExit("--iters must be > 0")
    if args.warmup < 0:
        raise SystemExit("--warmup must be >= 0")

    if args.worker:
        if args.request is None or args.result is None:
            raise SystemExit("--worker requires --request and --result")
        return

    if not args.extension.is_file():
        raise SystemExit(
            f"extension not found: {args.extension}; build ooverlap_ext first"
        )


def main() -> int:
    parser = build_argument_parser()
    args = parser.parse_args()
    validate_cli(args)

    if args.worker:
        return worker_main(args.request, args.result)
    return parent_main(args)


if __name__ == "__main__":
    raise SystemExit(main())
