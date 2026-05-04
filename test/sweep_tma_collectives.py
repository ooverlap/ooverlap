#!/usr/bin/env python3

import importlib.util
import json
import os
from pathlib import Path


# ---------------------------------------------------------------------------
# Edit these arrays directly.
# ---------------------------------------------------------------------------

NUMELS = [
    1 << 20,
    1 << 21,
    1 << 22,
    1 << 23,
    1 << 24,
]

COLLECTIVES = [
    "allreduce",
    "reduce_scatter",
    "all_gather",
]

KERNELS = [
    "tma_copy",
    "seq_fast_gmem",
    "overlap_fast_gmem",
]

THREADS = [
    1024,
]

MAX_CTAS = [
    2,
    4,
    8,
]

WINDOW_CHUNKS = [
    16,
    32,
    64,
]

# Each pair is: (chunk_bytes, stage_depth)
VARIANTS = [
    (16 * 1024, 8),
    (32 * 1024, 4),
    (64 * 1024, 2),
]

ITERS = 100
WARMUP = 20
DEV0 = 0
DEV1 = 1

INCLUDE_NCCL = True

OUT_DIR = Path("results")
REQUEST_JSON = OUT_DIR / "tma_collective_sweep_request.json"
RESULT_JSON = OUT_DIR / "tma_collective_sweep_result.json"
SKIPPED_JSON = OUT_DIR / "tma_collective_sweep_skipped.json"


# Optional env knobs.
# Leave as None if you do not want to set them here.
NCCL_MIN_CTAS = None
NCCL_MAX_CTAS = None
NCCL_ALGO = None
NCCL_PROTO = None


# ---------------------------------------------------------------------------
# Put your custom skip logic here.
# Return None to keep.
# Return a string to skip.
# ---------------------------------------------------------------------------

def skip_combo(s):
    """
    s is a dict like:

    {
      "backend": "ooverlap",
      "collective": "allreduce",
      "kernel": "tma_copy",
      "numel": ...,
      "threads": ...,
      "max_ctas": ...,
      "window_chunks": ...,
      "chunk_bytes": ...,
      "stage_depth": ...
    }
    """

    # Basic correctness/sanity filters.
    if s["collective"] in ("reduce_scatter", "all_gather"):
        if s["numel"] % 2 != 0:
            return "reduce_scatter/all_gather need even numel for 2 GPUs"

    if s["backend"] == "nccl":
        return None

    if s["threads"] % 32 != 0:
        return "threads must be warp aligned"

    if s["threads"] > 1024:
        return "threads > 1024"

    if s["chunk_bytes"] % 16 != 0:
        return "chunk_bytes must be 16-byte aligned"

    if s["stage_depth"] <= 0:
        return "stage_depth must be positive"

    if s["window_chunks"] < s["stage_depth"]:
        return "window_chunks < stage_depth"

    # Example custom pruning:
    # overlap_fast_gmem was mainly useful for allreduce tuning.
    # Remove this if you want to test it for every collective.
    if s["kernel"] == "overlap_fast_gmem" and s["collective"] != "allreduce":
        return "skip overlap_fast_gmem for non-allreduce"

    # Example custom pruning:
    # if s["max_ctas"] == 2 and s["kernel"] == "overlap_fast_gmem":
    #     return "skip overlap with too few CTAs"

    return None


def load_ooverlap_ext():
    root = Path(__file__).resolve().parents[1]
    so = root / "build" / "lib" / "ooverlap_ext.so"

    if not so.exists():
        raise FileNotFoundError(f"Could not find {so}. Build first.")

    spec = importlib.util.spec_from_file_location("ooverlap_ext", str(so))
    mod = importlib.util.module_from_spec(spec)

    if spec.loader is None:
        raise RuntimeError(f"Could not load extension spec for {so}")

    spec.loader.exec_module(mod)
    return mod


def set_env(name, value):
    if value is not None:
        os.environ[name] = str(value)


def make_nccl_scenario(collective, numel):
    return {
        "id": f"nccl__{collective}__n{numel}",
        "backend": "nccl",
        "collective": collective,
        "numel": numel,
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


def generate_scenarios():
    scenarios = []
    skipped = []

    for collective in COLLECTIVES:
        for numel in NUMELS:
            if INCLUDE_NCCL:
                s = make_nccl_scenario(collective, numel)
                reason = skip_combo(s)

                if reason is None:
                    scenarios.append(s)
                else:
                    skipped.append({**s, "reason": reason})

            for kernel in KERNELS:
                for threads in THREADS:
                    for max_ctas in MAX_CTAS:
                        for window_chunks in WINDOW_CHUNKS:
                            for chunk_bytes, stage_depth in VARIANTS:
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

                                reason = skip_combo(s)

                                if reason is None:
                                    scenarios.append(s)
                                else:
                                    skipped.append({**s, "reason": reason})

    return scenarios, skipped


def best_rows(results):
    best = {}

    for row in results:
        if row.get("status") != "ok":
            continue

        if row.get("backend") != "ooverlap":
            continue

        key = (
            row.get("collective"),
            row.get("kernel"),
        )

        old = best.get(key)

        if old is None or float(row["avg_ms"]) < float(old["avg_ms"]):
            best[key] = row

    return best


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    set_env("NCCL_MIN_CTAS", NCCL_MIN_CTAS)
    set_env("NCCL_MAX_CTAS", NCCL_MAX_CTAS)
    set_env("NCCL_ALGO", NCCL_ALGO)
    set_env("NCCL_PROTO", NCCL_PROTO)

    scenarios, skipped = generate_scenarios()

    request = {
        "iters": ITERS,
        "warmup": WARMUP,
        "dev0": DEV0,
        "dev1": DEV1,
        "scenarios": scenarios,
    }

    REQUEST_JSON.write_text(json.dumps(request, indent=2, sort_keys=True) + "\n")
    SKIPPED_JSON.write_text(json.dumps(skipped, indent=2, sort_keys=True) + "\n")

    print(f"[info] generated scenarios: {len(scenarios)}")
    print(f"[info] skipped scenarios:   {len(skipped)}")
    print(f"[info] request: {REQUEST_JSON}")
    print(f"[info] skipped: {SKIPPED_JSON}")
    print(f"[info] NCCL_MIN_CTAS={os.environ.get('NCCL_MIN_CTAS')}")
    print(f"[info] NCCL_MAX_CTAS={os.environ.get('NCCL_MAX_CTAS')}")
    print(f"[info] NCCL_ALGO={os.environ.get('NCCL_ALGO')}")
    print(f"[info] NCCL_PROTO={os.environ.get('NCCL_PROTO')}")

    ext = load_ooverlap_ext()

    run_sweep = ext.benchmark_tma_two_gpu_collective_sweep_json

    response_text = run_sweep(json.dumps(request))
    response = json.loads(response_text)

    response["skipped_by_python"] = skipped

    RESULT_JSON.write_text(json.dumps(response, indent=2, sort_keys=True) + "\n")
    print(f"[result] wrote: {RESULT_JSON}")

    results = response.get("results", [])
    errors = response.get("errors", [])

    ok_rows = [r for r in results if r.get("status") == "ok"]
    cpp_skipped = [r for r in results if r.get("status") == "skipped"]

    print(f"[result] ok rows:      {len(ok_rows)}")
    print(f"[result] cpp skipped:  {len(cpp_skipped)}")
    print(f"[result] errors:       {len(errors)}")

    if errors:
        print("[errors]")
        for e in errors[:20]:
            print(json.dumps(e, sort_keys=True))

    best = best_rows(results)

    if best:
        print("[best]")
        for key in sorted(best):
            row = best[key]
            print(
                f"  {key[0]:>14} {key[1]:>18} "
                f"avg_ms={float(row['avg_ms']):.6f} "
                f"lat_us={float(row.get('latency_us', 0.0)):.2f} "
                f"gbps={float(row['effective_gbps_per_rank']):.2f} "
                f"id={row.get('id')}"
            )

    print("PASS")


if __name__ == "__main__":
    main()
