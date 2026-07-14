import csv
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

OUT = Path(
    os.environ.get(
        "OUT",
        Path(os.environ["OOTMP"]) / "vllm_eager_paperlike",
    )
)

BASE_BACKEND = os.environ.get("BASE_BACKEND", "auto")

cases = {
    "decode": {"input_len": 16, "output_len": 256},
    "mixed": {"input_len": 256, "output_len": 128},
    "long": {"input_len": 1024, "output_len": 128},
    "prefill": {"input_len": 1024, "output_len": 1},
}

RESULT_RE = re.compile(
    r"^(?P<kind>.+)_backend_(?P<backend>.+)_bsz_(?P<bsz>\d+)\.json$"
)


def discover_results() -> dict[tuple[str, str, int], Path]:
    results: dict[tuple[str, str, int], Path] = {}

    for path in OUT.glob("*_backend_*_bsz_*.json"):
        match = RESULT_RE.match(path.name)
        if match is None:
            continue

        kind = match.group("kind")
        backend = match.group("backend")
        bsz = int(match.group("bsz"))

        if kind not in cases:
            continue

        results[(kind, backend, bsz)] = path

    return results


result_files = discover_results()

if not result_files:
    raise SystemExit(f"No benchmark JSON files found under: {OUT}")

discovered_backends = sorted(
    {backend for _, backend, _ in result_files},
    key=lambda backend: (backend != BASE_BACKEND, backend),
)

backends_env = os.environ.get("BACKENDS", "").strip()
if backends_env:
    backends = backends_env.split()
else:
    backends = discovered_backends

if BASE_BACKEND not in backends:
    backends.insert(0, BASE_BACKEND)

backends = list(dict.fromkeys(backends))

bszs_env = os.environ.get("BSZS", "").strip()
if bszs_env:
    bszs = [int(value) for value in bszs_env.split()]
else:
    bszs = sorted({bsz for _, _, bsz in result_files})


def result_path(kind: str, backend: str, bsz: int) -> Path:
    return OUT / f"{kind}_backend_{backend}_bsz_{bsz}.json"


def load(kind: str, backend: str, bsz: int) -> dict[str, Any] | None:
    path = result_files.get((kind, backend, bsz), result_path(kind, backend, bsz))

    if not path.is_file():
        return None

    try:
        with path.open() as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"WARNING: could not read {path}: {exc}", file=sys.stderr)
        return None

    required = {
        "elapsed_time",
        "num_requests",
        "requests_per_second",
        "tokens_per_second",
    }
    missing = required.difference(data)
    if missing:
        print(
            f"WARNING: skipping {path}; missing fields: "
            f"{', '.join(sorted(missing))}",
            file=sys.stderr,
        )
        return None

    return data


rows: list[dict[str, Any]] = []
missing_results: list[str] = []

for kind, cfg in cases.items():
    for bsz in bszs:
        base = load(kind, BASE_BACKEND, bsz)
        if base is None:
            missing_results.append(
                f"{kind}: baseline={BASE_BACKEND}, max_seqs={bsz}"
            )
            continue

        base_elapsed = float(base["elapsed_time"])

        for backend in backends:
            result = load(kind, backend, bsz)
            if result is None:
                missing_results.append(
                    f"{kind}: backend={backend}, max_seqs={bsz}"
                )
                continue

            num_requests = int(result["num_requests"])
            elapsed = float(result["elapsed_time"])
            output_len = int(cfg["output_len"])
            input_len = int(cfg["input_len"])
            speedup = base_elapsed / elapsed

            rows.append(
                {
                    "kind": kind,
                    "backend": backend,
                    "baseline_backend": BASE_BACKEND,
                    "max_seqs": bsz,
                    "input_len": input_len,
                    "output_len": output_len,
                    "elapsed_s": elapsed,
                    "speedup_vs_base": speedup,
                    "change_vs_base_pct": (speedup - 1.0) * 100.0,
                    "num_requests": num_requests,
                    "requests_per_s": float(result["requests_per_second"]),
                    "total_tokens_per_s": float(result["tokens_per_second"]),
                    "output_tokens_per_s": (
                        num_requests * output_len
                    ) / elapsed,
                }
            )

if not rows:
    raise SystemExit(
        "No complete benchmark rows were found. "
        f"Baseline backend is {BASE_BACKEND!r}."
    )

csv_path = OUT / "summary_all_backends.csv"
with csv_path.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)

print(f"Baseline backend: {BASE_BACKEND}")
print(f"Backends: {' '.join(backends)}")
print(f"Batch sizes: {' '.join(map(str, bszs))}")
print(f"Wrote: {csv_path}")
print()

backend_width = max(8, max(len(backend) for backend in backends))

for kind in cases:
    print(f"================ {kind.upper()} ================")

    for bsz in bszs:
        case_rows = [
            row
            for row in rows
            if row["kind"] == kind and row["max_seqs"] == bsz
        ]

        if not case_rows:
            continue

        print(f"\nmax_seqs={bsz}")

        by_backend = {row["backend"]: row for row in case_rows}
        for backend in backends:
            row = by_backend.get(backend)
            if row is None:
                print(f"{backend:<{backend_width}s} MISSING")
                continue

            marker = " [base]" if backend == BASE_BACKEND else ""
            print(
                f"{backend:<{backend_width}s} "
                f"elapsed={row['elapsed_s']:.4f}s "
                f"req/s={row['requests_per_s']:.2f} "
                f"out_tok/s={row['output_tokens_per_s']:.1f} "
                f"speedup={row['speedup_vs_base']:.4f}x "
                f"delta={row['change_vs_base_pct']:+.2f}%"
                f"{marker}"
            )

if missing_results:
    print()
    print("Missing or invalid results:")
    for item in missing_results:
        print(f"  - {item}")
