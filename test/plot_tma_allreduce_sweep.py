import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


KERNEL_ORDER = [
    "nccl",
    "tma_copy",
    "seq_fast_gmem",
    "overlap_fast_gmem",
]

DISPLAY_NAMES = {
    "nccl": "NCCL all-reduce",
    "tma_copy": "TMA reduce + TMA copy",
    "seq_fast_gmem": "TMA reduce + gmem copy",
    "overlap_fast_gmem": "TMA reduce + overlapped gmem copy",
}


def load_jsonl(path):
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Invalid JSON on line {line_no}: {exc}") from exc

    return rows


def bytes_label(num_bytes):
    num_bytes = float(num_bytes)

    if num_bytes >= 1024**3:
        return f"{num_bytes / 1024**3:.1f} GiB"
    if num_bytes >= 1024**2:
        return f"{num_bytes / 1024**2:.1f} MiB"
    if num_bytes >= 1024:
        return f"{num_bytes / 1024:.1f} KiB"

    return f"{num_bytes:.0f} B"


def parse_optional_int(value):
    if value is None:
        return None

    if isinstance(value, int):
        return value

    if isinstance(value, float):
        return int(value)

    value = str(value).strip()
    if not value or value.lower() in ("none", "null"):
        return None

    return int(value)


def row_nccl_ctas(row):
    return parse_optional_int(row.get("nccl_max_ctas_env"))


def make_best_rows(rows):
    """
    TMA:
      fastest row for each (max_ctas, kernel, numel)
      across threads/window_chunks.

    NCCL:
      fastest row for each (NCCL_MAX_CTAS, numel).
    """
    best_tma = {}
    best_nccl = {}

    for row in rows:
        kernel = row.get("kernel")
        numel = int(row["numel"])
        avg_ms = float(row["avg_ms"])

        if kernel == "nccl":
            nccl_ctas = row_nccl_ctas(row)
            if nccl_ctas is None:
                continue

            key = (nccl_ctas, numel)
            old = best_nccl.get(key)

            if old is None or avg_ms < float(old["avg_ms"]):
                best_nccl[key] = row

            continue

        max_ctas = int(row["max_ctas"])
        key = (max_ctas, kernel, numel)

        old = best_tma.get(key)
        if old is None or avg_ms < float(old["avg_ms"]):
            best_tma[key] = row

    return best_tma, best_nccl


def collect_max_ctas(rows):
    values = set()

    for row in rows:
        if row.get("kernel") == "nccl":
            nccl_ctas = row_nccl_ctas(row)
            if nccl_ctas is not None:
                values.add(nccl_ctas)
            continue

        if row.get("max_ctas") is not None:
            values.add(int(row["max_ctas"]))

    return sorted(values)


def collect_numels(rows):
    return sorted({int(row["numel"]) for row in rows})


def find_bytes_for_numel(numel, rows):
    for row in rows:
        if int(row["numel"]) == int(numel):
            return int(row["bytes_per_rank"])

    return int(numel) * 2


def matched_value_for_row(row, nccl_row, y_metric):
    if row is None:
        return None

    if y_metric == "avg_ms":
        return float(row["avg_ms"])

    if y_metric == "speedup_vs_nccl":
        if row.get("kernel") == "nccl":
            return 1.0

        if nccl_row is None:
            return None

        avg_ms = float(row["avg_ms"])
        nccl_avg_ms = float(nccl_row["avg_ms"])

        if avg_ms <= 0.0:
            return None

        return nccl_avg_ms / avg_ms

    return float(row[y_metric])


def plot_for_max_ctas(
    max_ctas,
    numels,
    rows,
    best_tma,
    best_nccl,
    output_dir,
    y_metric,
):
    plt.figure(figsize=(10, 6))

    plotted_any = False

    for kernel in KERNEL_ORDER:
        xs = []
        ys = []

        for numel in numels:
            nccl_row = best_nccl.get((max_ctas, numel))

            if kernel == "nccl":
                row = nccl_row
            else:
                row = best_tma.get((max_ctas, kernel, numel))

            value = matched_value_for_row(row, nccl_row, y_metric)
            if value is None:
                continue

            bytes_per_rank = int(row["bytes_per_rank"])

            xs.append(bytes_per_rank)
            ys.append(value)

        if xs:
            plotted_any = True
            plt.plot(xs, ys, marker="o", label=DISPLAY_NAMES.get(kernel, kernel))

    if not plotted_any:
        plt.close()
        return None

    plt.xscale("log", base=2)

    if y_metric == "avg_ms":
        plt.yscale("log")
        plt.ylabel("Average latency per allreduce (ms)")
        title_metric = "latency"
    elif y_metric == "speedup_vs_nccl":
        plt.yscale("linear")
        plt.ylabel("Speedup vs matched NCCL")
        title_metric = "speedup"
        plt.axhline(1.0, linestyle="--", linewidth=1)
    else:
        plt.yscale("log")
        plt.ylabel(y_metric)
        title_metric = y_metric

    plt.xlabel("Bytes per rank")
    plt.title(
        f"TMA allreduce sweep: {title_metric}, "
        f"TMA max_ctas={max_ctas}, NCCL_MAX_CTAS={max_ctas}"
    )
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.legend()

    ticks = []
    tick_labels = []

    for numel in numels:
        b = find_bytes_for_numel(numel, rows)
        ticks.append(b)
        tick_labels.append(bytes_label(b))

    plt.xticks(ticks, tick_labels, rotation=35, ha="right")
    plt.tight_layout()

    out_path = output_dir / f"tma_allreduce_{y_metric}_matched_ctas_{max_ctas}.png"
    plt.savefig(out_path, dpi=180)
    plt.close()

    return out_path


def plot_cta_scaling_for_kernel(
    kernel,
    max_ctas_values,
    numels,
    rows,
    best_tma,
    best_nccl,
    output_dir,
):
    plt.figure(figsize=(10, 6))

    plotted_any = False

    for max_ctas in max_ctas_values:
        xs = []
        ys = []

        for numel in numels:
            if kernel == "nccl":
                row = best_nccl.get((max_ctas, numel))
            else:
                row = best_tma.get((max_ctas, kernel, numel))

            if row is None:
                continue

            xs.append(int(row["bytes_per_rank"]))
            ys.append(float(row["avg_ms"]))

        if xs:
            plotted_any = True
            plt.plot(xs, ys, marker="o", label=f"ctas={max_ctas}")

    if not plotted_any:
        plt.close()
        return None

    plt.xscale("log", base=2)
    plt.yscale("log")

    plt.xlabel("Bytes per rank")
    plt.ylabel("Average latency per allreduce (ms)")
    plt.title(f"CTA scaling: {DISPLAY_NAMES.get(kernel, kernel)}")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.legend()

    ticks = []
    tick_labels = []

    for numel in numels:
        b = find_bytes_for_numel(numel, rows)
        ticks.append(b)
        tick_labels.append(bytes_label(b))

    plt.xticks(ticks, tick_labels, rotation=35, ha="right")
    plt.tight_layout()

    safe_kernel = kernel.replace("/", "_").replace(" ", "_")
    out_path = output_dir / f"tma_allreduce_cta_scaling_{safe_kernel}.png"
    plt.savefig(out_path, dpi=180)
    plt.close()

    return out_path


def write_best_config_summary(rows, best_tma, best_nccl, output_dir):
    out_path = output_dir / "tma_allreduce_best_configs.jsonl"

    max_ctas_values = collect_max_ctas(rows)
    numels = collect_numels(rows)

    with out_path.open("w", encoding="utf-8") as f:
        for max_ctas in max_ctas_values:
            for kernel in KERNEL_ORDER:
                if kernel == "nccl":
                    continue

                for numel in numels:
                    row = best_tma.get((max_ctas, kernel, numel))
                    if row is None:
                        continue

                    nccl_row = best_nccl.get((max_ctas, numel))

                    matched_nccl_avg_ms = (
                        float(nccl_row["avg_ms"]) if nccl_row is not None else None
                    )

                    avg_ms = float(row["avg_ms"])
                    speedup_vs_matched_nccl = (
                        matched_nccl_avg_ms / avg_ms
                        if matched_nccl_avg_ms is not None and avg_ms > 0.0
                        else None
                    )

                    compact = {
                        "kernel": row["kernel"],
                        "numel": int(row["numel"]),
                        "bytes_per_rank": int(row["bytes_per_rank"]),
                        "max_ctas": int(row["max_ctas"]),
                        "matched_nccl_max_ctas": max_ctas,
                        "threads": int(row["threads"]),
                        "window_chunks": int(row["window_chunks"]),
                        "avg_ms": avg_ms,
                        "matched_nccl_avg_ms": matched_nccl_avg_ms,
                        "speedup_vs_matched_nccl": speedup_vs_matched_nccl,
                        "compile_chunk_bytes": int(row["compile_chunk_bytes"]),
                        "compile_reduce_stage_depth": int(row["compile_reduce_stage_depth"]),
                        "compile_reduce_stage_gap": int(row["compile_reduce_stage_gap"]),
                        "compile_copy_stage_depth": int(row["compile_copy_stage_depth"]),
                        "compile_copy_stage_gap": int(row["compile_copy_stage_gap"]),
                        "compile_fast_copy_unroll": int(row["compile_fast_copy_unroll"]),
                    }

                    f.write(json.dumps(compact, sort_keys=True) + "\n")

    return out_path


def warn_missing_nccl(rows, max_ctas_values, numels, best_nccl):
    missing = []

    for max_ctas in max_ctas_values:
        for numel in numels:
            if (max_ctas, numel) not in best_nccl:
                missing.append((max_ctas, numel))

    if not missing:
        return

    print("[warn] Missing matched NCCL rows for some (max_ctas, numel) pairs.")
    print("[warn] The affected NCCL line or matched speedup point will be omitted.")
    print("[warn] Rerun sweep like:")
    print("[warn]   --max-ctas 4  --nccl-max-ctas 4")
    print("[warn]   --max-ctas 8  --nccl-max-ctas 8")
    print("[warn]   --max-ctas 16 --nccl-max-ctas 16")

    for max_ctas, numel in missing[:10]:
        b = find_bytes_for_numel(numel, rows)
        print(f"[warn] missing NCCL_MAX_CTAS={max_ctas}, numel={numel}, bytes={b}")

    if len(missing) > 10:
        print(f"[warn] ... and {len(missing) - 10} more")


def main():
    parser = argparse.ArgumentParser("Plot TMA allreduce sweep JSONL output")
    parser.add_argument(
        "--input",
        type=str,
        default="results/tma_allreduce_sweep.jsonl",
        help="Input JSONL file from sweep_tma_allreduce.py",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="results",
        help="Output directory for plots",
    )
    parser.add_argument(
        "--metric",
        choices=["avg_ms", "speedup_vs_nccl", "both"],
        default="avg_ms",
        help="Which matched-NCCL metric to plot",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Also write best-config JSONL summary",
    )
    parser.add_argument(
        "--no-cta-scaling",
        action="store_true",
        help="Disable per-kernel CTA scaling plots",
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_jsonl(input_path)
    if not rows:
        raise RuntimeError(f"No rows found in {input_path}")

    best_tma, best_nccl = make_best_rows(rows)
    max_ctas_values = collect_max_ctas(rows)
    numels = collect_numels(rows)

    if not max_ctas_values:
        raise RuntimeError("No max_ctas values found")

    warn_missing_nccl(
        rows=rows,
        max_ctas_values=max_ctas_values,
        numels=numels,
        best_nccl=best_nccl,
    )

    metrics = ["avg_ms", "speedup_vs_nccl"] if args.metric == "both" else [args.metric]

    written = []

    for metric in metrics:
        for max_ctas in max_ctas_values:
            out_path = plot_for_max_ctas(
                max_ctas=max_ctas,
                numels=numels,
                rows=rows,
                best_tma=best_tma,
                best_nccl=best_nccl,
                output_dir=output_dir,
                y_metric=metric,
            )

            if out_path is not None:
                written.append(out_path)

    if not args.no_cta_scaling:
        for kernel in KERNEL_ORDER:
            out_path = plot_cta_scaling_for_kernel(
                kernel=kernel,
                max_ctas_values=max_ctas_values,
                numels=numels,
                rows=rows,
                best_tma=best_tma,
                best_nccl=best_nccl,
                output_dir=output_dir,
            )

            if out_path is not None:
                written.append(out_path)

    summary_path = None
    if args.summary:
        summary_path = write_best_config_summary(
            rows=rows,
            best_tma=best_tma,
            best_nccl=best_nccl,
            output_dir=output_dir,
        )

    print(f"[info] read {len(rows)} rows from {input_path}")
    print(f"[info] max_ctas values: {max_ctas_values}")
    print(f"[info] numels: {numels}")
    print(f"[info] matched NCCL rows: {len(best_nccl)}")

    for path in written:
        print(f"[plot] {path}")

    if summary_path is not None:
        print(f"[summary] {summary_path}")

    print("PASS ✅ plotted TMA allreduce sweep")


if __name__ == "__main__":
    main()
