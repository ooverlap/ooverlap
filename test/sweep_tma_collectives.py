#!/usr/bin/env python3

import argparse
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path


# =============================================================================
# Edit these arrays.
# =============================================================================

NUMELS = [
    32768,
    65536,
    131072,
    262144,
    524288,
    1048576,
    2097152,
    4194304,
    8388608,
    16777216,
    33554432,
    67108864,
    134217728,
    268435456,
    536870912,
    623104,
    741376,
    881664,
    1246720,
    1482752,
    1763328,
    2493440,
    2965504,
    3526656,
    4987392,
    5931520,
    7053824,
    9975296,
    11863040,
    14107648,
    19951104,
    23726080,
    28215296,
    39902720,
    47452672,
    56431104,
    79805952,
    94905856,
    112862720,
    159612416,
    189812224,
    225725952,
]

COLLECTIVES = [
    "allreduce",
    "reduce_scatter",
    "all_gather",
]

# Kernel names can be different per collective.
# These strings should match what your C++ launch config parser accepts.
KERNELS_BY_COLLECTIVE = {
    "allreduce": [
        "tma_copy",
        "seq_fast_gmem",
        "overlap_fast_gmem",
    ],
    "reduce_scatter": [
        "tma_copy",
        "seq_fast_gmem",
    ],
    "all_gather": [
        "tma_copy",
        "seq_fast_gmem",
    ],
}

THREADS = [
    128,
    256,
    512,
    1024,
]

# Each value here runs in its own subprocess.
# In that subprocess we set:
#   OOVERLAP_MAX_CTAS=<ctas>
#   NCCL_MAX_CTAS=<ctas>
CTA_LIMITS = [
    2,
    4,
    8,
    16,
]

WINDOW_CHUNKS = [
    16,
    32,
    64,
    128,
    256,
    512,
]

# Each pair is:
#   (chunk_bytes, stage_depth)
CHUNK_STAGE_PAIRS = [
    (2 * 1024, 64),
    (4 * 1024, 32),
    (8 * 1024, 16),
    (16 * 1024, 8),
    (32 * 1024, 4),
    (64 * 1024, 2),
]

ITERS = 100
WARMUP = 20
DEV0 = 0
DEV1 = 1

INCLUDE_NCCL_BASELINE = True

OUT_DIR = Path("results")
REQUESTS_DIR = OUT_DIR / "tma_collective_sweep_requests"
LOGS_DIR = OUT_DIR / "tma_collective_sweep_logs"

MERGED_RESULT_JSON = OUT_DIR / "tma_collective_sweep_result.json"
SKIPPED_JSON = OUT_DIR / "tma_collective_sweep_skipped.json"
BEST_JSON = OUT_DIR / "tma_collective_sweep_best.json"

# Extra NCCL knobs. Leave None to not set.
NCCL_ALGO = None
NCCL_PROTO = None


# =============================================================================
# Put your custom skip logic here.
# Return None to run.
# Return a string to skip.
# =============================================================================

def skip_scenario(s):
    """
    s example:

    {
        "backend": "ooverlap",
        "collective": "allreduce",
        "kernel": "seq_fast_gmem",
        "numel": 1048576,
        "threads": 1024,
        "max_ctas": 4,
        "window_chunks": 32,
        "chunk_bytes": 32768,
        "stage_depth": 4,
    }
    """

    if s["collective"] in ("reduce_scatter", "all_gather"):
        if s["numel"] % 2 != 0:
            return "reduce_scatter/all_gather need even numel for 2 GPUs"

    if s["backend"] == "nccl":
        return None

    if s["threads"] <= 0 or s["threads"] % 32 != 0:
        return "threads must be positive and warp aligned"

    if s["threads"] > 1024:
        return "threads > 1024"

    if s["max_ctas"] <= 0:
        return "max_ctas must be positive"

    if s["window_chunks"] <= 0:
        return "window_chunks must be positive"

    if s["chunk_bytes"] <= 0 or s["chunk_bytes"] % 16 != 0:
        return "chunk_bytes must be positive and 16-byte aligned"

    if s["stage_depth"] <= 0:
        return "stage_depth must be positive"

    if s["window_chunks"] < s["stage_depth"]:
        return "window_chunks < stage_depth"

    if s["kernel"] == "tma_copy" and s["threads"] > 32:
        return "address is already 16-bit aligned we dont need tons of threads"

    if s["numel"] > 67108864:
        if (
            s["kernel"] in ("seq_fast_gmem", "overlap_fast_gmem")
            and s["threads"] < 512
        ):
            return "for huge numel numbers threads should be above 512"

        if s["max_ctas"] < 8:
            return "for these huge numel numbers lower cta does not give anything back"

    if s["numel"] < 16777216:
        if s["window_chunks"] > 64:
            return "windows with huge chunks is not needed"

        if (
            s["kernel"] in ("seq_fast_gmem", "overlap_fast_gmem")
            and s["threads"] > 512
        ):
            return "we dont need that much threads"

    return None


# =============================================================================
# No need to edit below this line most of the time.
# =============================================================================

def repo_root():
    return Path(__file__).resolve().parents[1]


def extension_path():
    return repo_root() / "build" / "lib" / "ooverlap_ext.so"


def load_extension():
    so = extension_path()

    if not so.exists():
        raise FileNotFoundError(f"Could not find {so}. Build first.")

    spec = importlib.util.spec_from_file_location("ooverlap_ext", str(so))
    mod = importlib.util.module_from_spec(spec)

    if spec.loader is None:
        raise RuntimeError(f"Could not load extension from {so}")

    spec.loader.exec_module(mod)
    return mod


def make_nccl_scenario(collective, numel, max_ctas):
    return {
        "id": f"nccl__{collective}__n{numel}__ctas{max_ctas}",
        "backend": "nccl",
        "collective": collective,
        "kernel": "nccl",
        "numel": numel,
        "max_ctas": max_ctas,
    }


def make_ooverlap_scenario(
    collective,
    kernel,
    numel,
    threads,
    max_ctas,
    window_chunks,
    chunk_bytes,
    stage_depth,
):
    return {
        "id": (
            f"oo__{collective}"
            f"__{kernel}"
            f"__n{numel}"
            f"__thr{threads}"
            f"__ctas{max_ctas}"
            f"__win{window_chunks}"
            f"__chunk{chunk_bytes}"
            f"__stage{stage_depth}"
        ),
        "backend": "ooverlap",
        "collective": collective,
        "kernel": kernel,
        "numel": numel,
        "threads": threads,
        "max_ctas": max_ctas,
        "window_chunks": window_chunks,
        "chunk_bytes": chunk_bytes,
        "stage_depth": stage_depth,
    }


def generate_scenarios_for_cta(max_ctas):
    scenarios = []
    skipped = []

    for collective in COLLECTIVES:
        for numel in NUMELS:
            if INCLUDE_NCCL_BASELINE:
                s = make_nccl_scenario(collective, numel, max_ctas)
                reason = skip_scenario(s)

                if reason is None:
                    scenarios.append(s)
                else:
                    skipped.append({**s, "reason": reason})

            kernels = KERNELS_BY_COLLECTIVE.get(collective, [])

            for kernel in kernels:
                for threads in THREADS:
                    for window_chunks in WINDOW_CHUNKS:
                        for chunk_bytes, stage_depth in CHUNK_STAGE_PAIRS:
                            s = make_ooverlap_scenario(
                                collective=collective,
                                kernel=kernel,
                                numel=numel,
                                threads=threads,
                                max_ctas=max_ctas,
                                window_chunks=window_chunks,
                                chunk_bytes=chunk_bytes,
                                stage_depth=stage_depth,
                            )

                            reason = skip_scenario(s)

                            if reason is None:
                                scenarios.append(s)
                            else:
                                skipped.append({**s, "reason": reason})

    return scenarios, skipped


def make_request(scenarios):
    return {
        "iters": ITERS,
        "warmup": WARMUP,
        "dev0": DEV0,
        "dev1": DEV1,
        "scenarios": scenarios,
    }


def worker_main(request_path, result_path):
    request_path = Path(request_path)
    result_path = Path(result_path)

    request = json.loads(request_path.read_text())

    print(f"[worker] request={request_path}")
    print(f"[worker] result={result_path}")
    print(f"[worker] OOVERLAP_MAX_CTAS={os.environ.get('OOVERLAP_MAX_CTAS')}")
    print(f"[worker] NCCL_MAX_CTAS={os.environ.get('NCCL_MAX_CTAS')}")

    ext = load_extension()
    run_sweep = ext.benchmark_tma_two_gpu_collective_sweep_json

    response_text = run_sweep(json.dumps(request))
    response = json.loads(response_text)

    response["env"] = {
        "OOVERLAP_MAX_CTAS": os.environ.get("OOVERLAP_MAX_CTAS"),
        "NCCL_MAX_CTAS": os.environ.get("NCCL_MAX_CTAS"),
    }

    result_path.write_text(json.dumps(response, indent=2, sort_keys=True) + "\n")

    if not response.get("ok", False):
        return 1

    return 0


def run_one_cta_process(max_ctas, scenarios):
    request_path = REQUESTS_DIR / f"request_ctas{max_ctas}.json"
    result_path = REQUESTS_DIR / f"result_ctas{max_ctas}.json"
    log_path = LOGS_DIR / f"log_ctas{max_ctas}.txt"

    request = make_request(scenarios)
    request_path.write_text(json.dumps(request, indent=2, sort_keys=True) + "\n")

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    env["OOVERLAP_MAX_CTAS"] = str(max_ctas)
    env["NCCL_MAX_CTAS"] = str(max_ctas)

    if NCCL_ALGO is not None:
        env["NCCL_ALGO"] = str(NCCL_ALGO)

    if NCCL_PROTO is not None:
        env["NCCL_PROTO"] = str(NCCL_PROTO)

    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--request",
        str(request_path),
        "--result",
        str(result_path),
    ]

    print(
        f"[parent] run ctas={max_ctas} "
        f"scenarios={len(scenarios)} "
        f"OOVERLAP_MAX_CTAS={env['OOVERLAP_MAX_CTAS']} "
        f"NCCL_MAX_CTAS={env['NCCL_MAX_CTAS']}"
    )

    proc = subprocess.run(
        cmd,
        cwd=str(repo_root()),
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    log_path.write_text(proc.stdout)

    print(f"[parent] ctas={max_ctas} returncode={proc.returncode}")
    print(f"[parent] ctas={max_ctas} log={log_path}")

    if proc.returncode != 0:
        return {
            "ok": False,
            "results": [],
            "errors": [
                {
                    "id": None,
                    "index": None,
                    "max_ctas": max_ctas,
                    "error": f"subprocess failed; see {log_path}",
                }
            ],
            "env": {
                "OOVERLAP_MAX_CTAS": str(max_ctas),
                "NCCL_MAX_CTAS": str(max_ctas),
            },
        }

    return json.loads(result_path.read_text())


def tag_rows_with_cta(response, max_ctas):
    for row in response.get("results", []):
        row["process_max_ctas"] = max_ctas
        row["ooverlap_max_ctas_env"] = max_ctas
        row["nccl_max_ctas_env"] = max_ctas

    for err in response.get("errors", []):
        err["process_max_ctas"] = max_ctas


def load_json_if_exists(path, default):
    path = Path(path)

    if not path.exists() or path.stat().st_size == 0:
        return default

    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        backup = path.with_suffix(path.suffix + ".bad")
        backup.write_text(path.read_text())
        print(f"[warn] could not parse {path}; backed it up to {backup}")
        return default


def row_key(row, fallback_prefix, index):
    row_id = row.get("id")
    if row_id is not None:
        return ("id", str(row_id))

    return (
        fallback_prefix,
        row.get("backend"),
        row.get("collective"),
        row.get("kernel"),
        row.get("numel"),
        row.get("process_max_ctas"),
        row.get("max_ctas"),
        row.get("threads"),
        row.get("window_chunks"),
        row.get("chunk_bytes"),
        row.get("stage_depth"),
        index,
    )


def merge_rows(old_rows, new_rows, fallback_prefix):
    merged = {}
    order = []

    for i, row in enumerate(old_rows or []):
        key = row_key(row, fallback_prefix, i)
        if key not in merged:
            order.append(key)
        merged[key] = row

    for i, row in enumerate(new_rows or []):
        key = row_key(row, fallback_prefix, i)
        if key not in merged:
            order.append(key)

        # New run wins for the same scenario id/config.
        merged[key] = row

    return [merged[k] for k in order]


def extract_result_rows(payload):
    if isinstance(payload, list):
        return payload

    if isinstance(payload, dict):
        rows = payload.get("results", [])
        if isinstance(rows, list):
            return rows

    return []


def extract_skipped_rows(payload):
    if isinstance(payload, list):
        return payload

    if isinstance(payload, dict):
        rows = payload.get("skipped", [])
        if isinstance(rows, list):
            return rows

    return []


def write_json_atomic(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def best_rows(results):
    best = {}

    for row in results:
        if row.get("status") != "ok":
            continue

        key = (
            row.get("backend"),
            row.get("collective"),
            row.get("kernel"),
            int(row.get("numel", 0)),
        )

        old = best.get(key)

        if old is None or float(row.get("avg_ms", 1e100)) < float(old.get("avg_ms", 1e100)):
            best[key] = row

    return [best[k] for k in sorted(best)]


def parent_main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REQUESTS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    current_results = []
    current_errors = []
    current_skipped = []
    current_ok = True

    total_generated = 0
    total_skipped = 0

    for max_ctas in CTA_LIMITS:
        scenarios, skipped = generate_scenarios_for_cta(max_ctas)

        total_generated += len(scenarios)
        total_skipped += len(skipped)

        current_skipped.extend(skipped)

        if not scenarios:
            print(f"[parent] ctas={max_ctas} has no scenarios")
            continue

        response = run_one_cta_process(max_ctas, scenarios)
        tag_rows_with_cta(response, max_ctas)

        if not response.get("ok", False):
            current_ok = False

        current_results.extend(response.get("results", []))
        current_errors.extend(response.get("errors", []))

    old_result_payload = load_json_if_exists(
        MERGED_RESULT_JSON,
        {
            "ok": True,
            "results": [],
        },
    )

    old_skipped_payload = load_json_if_exists(SKIPPED_JSON, [])

    old_results = extract_result_rows(old_result_payload)
    old_skipped = extract_skipped_rows(old_skipped_payload)

    merged_results = merge_rows(old_results, current_results, "result")
    merged_skipped = merge_rows(old_skipped, current_skipped, "skipped")

    # Important: result JSON intentionally contains only measured result rows.
    # Skipped rows stay in SKIPPED_JSON. Errors are printed and kept in per-CTA logs.
    result_payload = {
        "ok": bool(old_result_payload.get("ok", True)) and current_ok,
        "results": merged_results,
    }

    write_json_atomic(MERGED_RESULT_JSON, result_payload)
    write_json_atomic(SKIPPED_JSON, merged_skipped)

    best = best_rows(merged_results)
    write_json_atomic(BEST_JSON, best)

    ok_rows = [
        r for r in merged_results
        if r.get("status") == "ok"
    ]

    cpp_skipped = [
        r for r in merged_results
        if r.get("status") == "skipped"
    ]

    print(f"[result] generated scenarios this run: {total_generated}")
    print(f"[result] python skipped this run:      {total_skipped}")
    print(f"[result] merged result rows:           {len(merged_results)}")
    print(f"[result] ok rows:                       {len(ok_rows)}")
    print(f"[result] cpp skipped in results:        {len(cpp_skipped)}")
    print(f"[result] merged python skipped rows:    {len(merged_skipped)}")
    print(f"[result] errors this run:               {len(current_errors)}")
    print(f"[result] merged:                        {MERGED_RESULT_JSON}")
    print(f"[result] skipped:                       {SKIPPED_JSON}")
    print(f"[result] best:                          {BEST_JSON}")

    if current_errors:
        print("[errors]")
        for err in current_errors[:20]:
            print(json.dumps(err, sort_keys=True))

    if best:
        print("[best]")
        for row in best[:80]:
            print(
                f"  backend={row.get('backend'):>8} "
                f"collective={row.get('collective'):>14} "
                f"kernel={row.get('kernel'):>18} "
                f"numel={int(row.get('numel', 0)):>12} "
                f"ctas={row.get('process_max_ctas')} "
                f"avg_ms={float(row.get('avg_ms', 0.0)):.6f} "
                f"id={row.get('id')}"
            )

    if current_errors or not current_ok:
        print("FAIL")
        return 1

    print("PASS")
    return 0


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--request", type=str, default="")
    parser.add_argument("--result", type=str, default="")

    return parser.parse_args()


def main():
    args = parse_args()

    if args.worker:
        if not args.request or not args.result:
            raise SystemExit("--worker requires --request and --result")

        return worker_main(args.request, args.result)

    return parent_main()


if __name__ == "__main__":
    raise SystemExit(main())
