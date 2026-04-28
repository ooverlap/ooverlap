import argparse
import glob
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


DEFAULT_KERNELS = [
    "tma_copy",
    "seq_fast_gmem",
    "overlap_fast_gmem",
]

COMPILE_FIELDS = [
    "compile_chunk_bytes",
    "compile_reduce_stage_depth",
    "compile_reduce_stage_gap",
    "compile_copy_stage_depth",
    "compile_copy_stage_gap",
    "compile_fast_copy_unroll",
]


def parse_kernel_list(text):
    kernels = []
    for item in str(text).split(","):
        item = item.strip()
        if item:
            kernels.append(item)

    if not kernels:
        raise ValueError("kernel list is empty")

    return kernels


def expand_inputs(inputs):
    paths = []

    for item in inputs:
        matches = glob.glob(item)

        if matches:
            for match in matches:
                p = Path(match)
                if p.is_dir():
                    paths.extend(sorted(p.glob("*.jsonl")))
                else:
                    paths.append(p)
            continue

        p = Path(item)
        if p.is_dir():
            paths.extend(sorted(p.glob("*.jsonl")))
        else:
            paths.append(p)

    unique = []
    seen = set()

    for path in paths:
        resolved = str(path)
        if resolved not in seen:
            seen.add(resolved)
            unique.append(path)

    if not unique:
        raise FileNotFoundError("No input JSONL files found")

    return unique


def load_jsonl_files(paths):
    rows = []

    for path in paths:
        with Path(path).open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue

                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise RuntimeError(
                        f"Invalid JSON in {path}:{line_no}: {exc}"
                    ) from exc

                row["_source_file"] = str(path)
                row["_source_line"] = line_no
                rows.append(row)

    return rows


def as_int(value, default=None):
    if value is None:
        return default

    if isinstance(value, bool):
        return int(value)

    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def as_float(value, default=None):
    if value is None:
        return default

    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def bytes_label(num_bytes):
    num_bytes = float(num_bytes)

    if num_bytes >= 1024**3:
        return f"{num_bytes / 1024**3:.1f} GiB"
    if num_bytes >= 1024**2:
        return f"{num_bytes / 1024**2:.1f} MiB"
    if num_bytes >= 1024:
        return f"{num_bytes / 1024:.1f} KiB"

    return f"{num_bytes:.0f} B"


def normalize_row(row, allowed_kernels):
    kernel = row.get("kernel")

    if kernel == "nccl":
        return None

    if kernel not in allowed_kernels:
        return None

    avg_ms = as_float(row.get("avg_ms"))
    if avg_ms is None or not math.isfinite(avg_ms) or avg_ms <= 0.0:
        return None

    numel = as_int(row.get("numel"))
    bytes_per_rank = as_int(row.get("bytes_per_rank"))
    threads = as_int(row.get("threads"))
    max_ctas = as_int(row.get("max_ctas"))
    window_chunks = as_int(row.get("window_chunks"))

    if (
        numel is None
        or bytes_per_rank is None
        or threads is None
        or max_ctas is None
        or window_chunks is None
    ):
        return None

    compact = {
        "kernel": kernel,
        "numel": numel,
        "bytes_per_rank": bytes_per_rank,
        "threads": threads,
        "max_ctas": max_ctas,
        "window_chunks": window_chunks,
        "avg_ms": avg_ms,
        "total_ms": as_float(row.get("total_ms")),
        "effective_gbps_per_rank": as_float(row.get("effective_gbps_per_rank")),
        "effective_gbps_aggregate_2gpu": as_float(
            row.get("effective_gbps_aggregate_2gpu")
        ),
        "speedup_vs_nccl": as_float(row.get("speedup_vs_nccl")),
        "iters": as_int(row.get("iters")),
        "warmup": as_int(row.get("warmup")),
        "source_file": row.get("_source_file"),
        "source_line": row.get("_source_line"),
    }

    for field in COMPILE_FIELDS:
        compact[field] = as_int(row.get(field))

    return compact


def config_identity(row):
    return (
        row["numel"],
        row["bytes_per_rank"],
        row["kernel"],
        row["threads"],
        row["max_ctas"],
        row["window_chunks"],
        tuple(row.get(field) for field in COMPILE_FIELDS),
    )


def dedupe_best_exact_config(rows):
    """
    If the same exact config appears multiple times, keep the fastest observed
    avg_ms. This is useful when several runs are appended into one result set.
    """
    best = {}

    for row in rows:
        key = config_identity(row)
        old = best.get(key)

        if old is None or row["avg_ms"] < old["avg_ms"]:
            best[key] = row

    return list(best.values())


def with_delta(row, best_ms):
    out = dict(row)
    out["slower_than_best_ms"] = row["avg_ms"] - best_ms
    out["slower_than_best_percent"] = (
        ((row["avg_ms"] / best_ms) - 1.0) * 100.0 if best_ms > 0.0 else None
    )
    return out


def choose_best_and_near(rows, tolerance_fraction):
    if not rows:
        return None, []

    ordered = sorted(rows, key=lambda r: (r["avg_ms"], r["kernel"], r["threads"]))
    best = ordered[0]
    best_ms = best["avg_ms"]
    threshold = best_ms * (1.0 + tolerance_fraction)

    near = [
        with_delta(row, best_ms)
        for row in ordered
        if row["avg_ms"] <= threshold
    ]

    return with_delta(best, best_ms), near


def build_policy_by_cta(rows, tolerance_fraction):
    groups = defaultdict(list)

    for row in rows:
        key = (row["max_ctas"], row["numel"], row["bytes_per_rank"])
        groups[key].append(row)

    by_cta = defaultdict(list)

    for (max_ctas, numel, bytes_per_rank), group_rows in sorted(groups.items()):
        best, near = choose_best_and_near(group_rows, tolerance_fraction)
        if best is None:
            continue

        by_cta[str(max_ctas)].append(
            {
                "numel": numel,
                "bytes_per_rank": bytes_per_rank,
                "max_ctas": max_ctas,
                "best": best,
                "near_best": near,
            }
        )

    for key in by_cta:
        by_cta[key].sort(key=lambda item: item["bytes_per_rank"])

    return dict(sorted(by_cta.items(), key=lambda kv: int(kv[0])))


def build_policy_by_size(rows, tolerance_fraction):
    groups = defaultdict(list)

    for row in rows:
        key = (row["numel"], row["bytes_per_rank"])
        groups[key].append(row)

    by_size = []

    for (numel, bytes_per_rank), group_rows in sorted(
        groups.items(),
        key=lambda kv: kv[0][1],
    ):
        best, near = choose_best_and_near(group_rows, tolerance_fraction)
        if best is None:
            continue

        by_size.append(
            {
                "numel": numel,
                "bytes_per_rank": bytes_per_rank,
                "best": best,
                "near_best": near,
            }
        )

    return by_size


def policy_summary(policy):
    by_cta_count = sum(len(v) for v in policy["by_cta"].values())
    by_size_count = len(policy["by_size"])

    return {
        "by_cta_entries": by_cta_count,
        "by_size_entries": by_size_count,
    }


def write_policy_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)

    return path


def plot_best_overall_by_size(policy, output_dir):
    items = policy["by_size"]
    if not items:
        return None

    xs = [item["bytes_per_rank"] for item in items]
    ys = [item["best"]["avg_ms"] for item in items]

    labels = [
        f'{item["best"]["kernel"]}\nctas={item["best"]["max_ctas"]}'
        for item in items
    ]

    plt.figure(figsize=(11, 6))
    plt.plot(xs, ys, marker="o", label="best overall")

    plt.xscale("log", base=2)
    plt.yscale("log")
    plt.xlabel("Bytes per rank")
    plt.ylabel("Best average latency (ms)")
    plt.title("Best allreduce config by transfer size")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.legend()

    for x, y, label in zip(xs, ys, labels):
        plt.annotate(
            label,
            (x, y),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
            fontsize=7,
        )

    plt.xticks(xs, [bytes_label(x) for x in xs], rotation=35, ha="right")
    plt.tight_layout()

    out_path = Path(output_dir) / "policy_best_overall_by_size.png"
    plt.savefig(out_path, dpi=180)
    plt.close()
    return out_path


def plot_best_by_cta(policy, output_dir):
    by_cta = policy["by_cta"]
    if not by_cta:
        return None

    plt.figure(figsize=(11, 6))

    plotted = False

    for max_ctas, items in by_cta.items():
        if not items:
            continue

        xs = [item["bytes_per_rank"] for item in items]
        ys = [item["best"]["avg_ms"] for item in items]

        plotted = True
        plt.plot(xs, ys, marker="o", label=f"ctas={max_ctas}")

    if not plotted:
        plt.close()
        return None

    all_bytes = sorted(
        {
            item["bytes_per_rank"]
            for items in by_cta.values()
            for item in items
        }
    )

    plt.xscale("log", base=2)
    plt.yscale("log")
    plt.xlabel("Bytes per rank")
    plt.ylabel("Best average latency within CTA group (ms)")
    plt.title("Best allreduce config by transfer size and CTA count")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.legend()

    plt.xticks(all_bytes, [bytes_label(x) for x in all_bytes], rotation=35, ha="right")
    plt.tight_layout()

    out_path = Path(output_dir) / "policy_best_by_cta.png"
    plt.savefig(out_path, dpi=180)
    plt.close()
    return out_path


def plot_near_best_count(policy, output_dir):
    items = policy["by_size"]
    if not items:
        return None

    xs = [item["bytes_per_rank"] for item in items]
    ys = [len(item["near_best"]) for item in items]

    plt.figure(figsize=(11, 5))
    plt.bar([bytes_label(x) for x in xs], ys)

    plt.xlabel("Bytes per rank")
    plt.ylabel("Configs within tolerance")
    plt.title("Number of near-best configs per transfer size")
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()

    out_path = Path(output_dir) / "policy_near_best_count_by_size.png"
    plt.savefig(out_path, dpi=180)
    plt.close()
    return out_path


def write_flat_jsonl(path, policy):
    """
    Convenience output for quick grepping. This is not the main policy file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        for item in policy["by_size"]:
            row = {
                "policy": "by_size",
                "numel": item["numel"],
                "bytes_per_rank": item["bytes_per_rank"],
                "best": item["best"],
                "near_best_count": len(item["near_best"]),
            }
            f.write(json.dumps(row, sort_keys=True) + "\n")

        for max_ctas, items in policy["by_cta"].items():
            for item in items:
                row = {
                    "policy": "by_cta",
                    "max_ctas": int(max_ctas),
                    "numel": item["numel"],
                    "bytes_per_rank": item["bytes_per_rank"],
                    "best": item["best"],
                    "near_best_count": len(item["near_best"]),
                }
                f.write(json.dumps(row, sort_keys=True) + "\n")

    return path


def main():
    parser = argparse.ArgumentParser(
        "Build TMA allreduce policy JSON from one or more sweep JSONL files"
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="Input JSONL files, directories, or glob patterns",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="results/tma_allreduce_policy.json",
        help="Output policy JSON",
    )
    parser.add_argument(
        "--flat-out",
        type=str,
        default=None,
        help="Optional flat JSONL summary output",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="results",
        help="Output directory for plots",
    )
    parser.add_argument(
        "--tolerance-percent",
        type=float,
        default=5.0,
        help="Keep configs with avg_ms <= best_avg_ms * (1 + tolerance_percent / 100)",
    )
    parser.add_argument(
        "--kernels",
        type=str,
        default=",".join(DEFAULT_KERNELS),
        help="Comma-separated kernels to include",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Disable plot generation",
    )

    args = parser.parse_args()

    if args.tolerance_percent < 0.0:
        raise ValueError("--tolerance-percent must be non-negative")

    input_paths = expand_inputs(args.inputs)
    allowed_kernels = parse_kernel_list(args.kernels)
    tolerance_fraction = args.tolerance_percent / 100.0

    raw_rows = load_jsonl_files(input_paths)

    normalized = []
    skipped = 0

    for row in raw_rows:
        compact = normalize_row(row, allowed_kernels)
        if compact is None:
            skipped += 1
            continue
        normalized.append(compact)

    if not normalized:
        raise RuntimeError("No usable non-NCCL TMA rows found")

    deduped = dedupe_best_exact_config(normalized)

    by_cta = build_policy_by_cta(deduped, tolerance_fraction)
    by_size = build_policy_by_size(deduped, tolerance_fraction)

    compile_variants = sorted(
        {
            tuple(row.get(field) for field in COMPILE_FIELDS)
            for row in deduped
        }
    )

    max_ctas_values = sorted({row["max_ctas"] for row in deduped})
    numels = sorted({row["numel"] for row in deduped})
    bytes_values = sorted({row["bytes_per_rank"] for row in deduped})

    policy = {
        "metadata": {
            "kind": "tma_allreduce_policy",
            "tolerance_percent": args.tolerance_percent,
            "tolerance_factor": 1.0 + tolerance_fraction,
            "input_files": [str(p) for p in input_paths],
            "raw_rows": len(raw_rows),
            "usable_rows": len(normalized),
            "deduped_exact_configs": len(deduped),
            "skipped_rows": skipped,
            "kernels": allowed_kernels,
            "max_ctas_values": max_ctas_values,
            "numels": numels,
            "bytes_per_rank_values": bytes_values,
            "compile_fields": COMPILE_FIELDS,
            "compile_variants": [
                dict(zip(COMPILE_FIELDS, variant))
                for variant in compile_variants
            ],
        },
        "summary": {},
        "by_cta": by_cta,
        "by_size": by_size,
    }

    policy["summary"] = policy_summary(policy)

    out_path = write_policy_json(args.out, policy)

    flat_path = None
    if args.flat_out is not None:
        flat_path = write_flat_jsonl(args.flat_out, policy)

    written_plots = []
    if not args.no_plots:
        output_dir = Path(args.out_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        for plot_fn in (
            plot_best_overall_by_size,
            plot_best_by_cta,
            plot_near_best_count,
        ):
            plot_path = plot_fn(policy, output_dir)
            if plot_path is not None:
                written_plots.append(plot_path)

    print(f"[info] input files: {len(input_paths)}")
    for path in input_paths:
        print(f"[input] {path}")

    print(f"[info] raw rows: {len(raw_rows)}")
    print(f"[info] usable non-NCCL rows: {len(normalized)}")
    print(f"[info] deduped exact configs: {len(deduped)}")
    print(f"[info] skipped rows: {skipped}")
    print(f"[info] tolerance percent: {args.tolerance_percent}")
    print(f"[info] max_ctas values: {max_ctas_values}")
    print(f"[info] transfer sizes: {[bytes_label(x) for x in bytes_values]}")
    print(f"[policy] {out_path}")

    if flat_path is not None:
        print(f"[flat] {flat_path}")

    for path in written_plots:
        print(f"[plot] {path}")

    print("PASS ✅ built TMA allreduce policy")


if __name__ == "__main__":
    main()
