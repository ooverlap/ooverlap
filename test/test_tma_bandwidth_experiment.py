#!/usr/bin/env python3

import argparse
import importlib.util
from pathlib import Path

import matplotlib.pyplot as plt
import torch


SCENARIO_NAME = {
    0: "local_to_peer",
    1: "peer_to_local",
    2: "same_dev",
}

METHOD_NAME = {
    0: "tma_copy",
    1: "tma_reduce_add_f16",
    2: "mem_async_copy",
}


def load_ooverlap_ext():
    root = Path(__file__).resolve().parents[1]
    so_path = root / "build" / "lib" / "ooverlap_ext.so"
    if not so_path.exists():
        raise FileNotFoundError(f"Extension not found: {so_path}")

    spec = importlib.util.spec_from_file_location("ooverlap_ext", so_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def normalize_rows(rows):
    out = []
    for row in rows:
        r = dict(row)
        r["scenario_name"] = SCENARIO_NAME[int(r["scenario"])]
        r["method_name"] = METHOD_NAME[int(r["method"])]
        r["mib"] = r["bytes"] / (1024.0 * 1024.0)
        out.append(r)
    return out


def print_table(rows):
    print(
        f"{'scenario':>15} "
        f"{'method':>20} "
        f"{'MiB':>10} "
        f"{'GB/s':>12} "
        f"{'src':>4} "
        f"{'dst':>4} "
        f"{'ker':>4} "
        f"{'blocks':>8}"
    )
    for r in rows:
        print(
            f"{r['scenario_name']:>15} "
            f"{r['method_name']:>20} "
            f"{r['mib']:10.1f} "
            f"{r['gbps']:12.2f} "
            f"{int(r['src_device']):4d} "
            f"{int(r['dst_device']):4d} "
            f"{int(r['kernel_device']):4d} "
            f"{int(r['num_blocks']):8d}"
        )


def ordered_methods(rows):
    present = {r["method_name"] for r in rows}
    preferred = ["tma_copy", "tma_reduce_add_f16"]
    return [m for m in preferred if m in present]


def plot_rows(rows, output_prefix):
    scenarios = ["local_to_peer", "peer_to_local", "same_dev"]
    methods = ordered_methods(rows)

    for scenario in scenarios:
        subset = [r for r in rows if r["scenario_name"] == scenario]
        if not subset:
            continue

        plt.figure()

        for method in methods:
            data = sorted(
                [r for r in subset if r["method_name"] == method],
                key=lambda x: x["bytes"],
            )
            if not data:
                continue

            xs = [r["mib"] for r in data]
            ys = [r["gbps"] for r in data]
            plt.plot(xs, ys, marker="o", label=method)

        plt.xscale("log", base=2)
        plt.xlabel("Transfer size (MiB)")
        plt.ylabel("Effective bandwidth (GB/s)")

        if scenario == "local_to_peer":
            title = "TMA bandwidth: local -> peer"
        elif scenario == "peer_to_local":
            title = "TMA bandwidth: peer -> local"
        else:
            title = "TMA bandwidth: same device"

        plt.title(title)
        plt.grid(True, which="both")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{output_prefix}_{scenario}_bandwidth.png", dpi=200)
        plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-bytes", type=int, default=512 * 1024)
    parser.add_argument("--max-bytes", type=int, default=1024 * 1024 * 1024)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--num-blocks", type=int, default=8)
    parser.add_argument("--dev0", type=int, default=0)
    parser.add_argument("--dev1", type=int, default=1)
    parser.add_argument("--output-prefix", type=str, default="tma_bandwidth_experiment")
    parser.add_argument(
        "--include-mem-async",
        action="store_true",
        help="Still benchmarks mem_async_copy, but plots only TMA copy/reduce.",
    )
    args = parser.parse_args()

    print(f"[info] torch {torch.__version__}")
    print(f"[info] cuda available: {torch.cuda.is_available()}")
    print(f"[info] cuda device count: {torch.cuda.device_count()}")
    print(
        f"[info] min_bytes={args.min_bytes} max_bytes={args.max_bytes} "
        f"iters={args.iters} warmup={args.warmup} blocks={args.num_blocks} "
        f"dev0={args.dev0} dev1={args.dev1} "
        f"include_mem_async={args.include_mem_async}"
    )

    ext = load_ooverlap_ext()

    rows = ext.benchmark_tma_bandwidth_experiment_sm90(
        args.min_bytes,
        args.max_bytes,
        args.iters,
        args.warmup,
        args.num_blocks,
        args.dev0,
        args.dev1,
        args.include_mem_async,
    )

    rows = normalize_rows(rows)
    print_table(rows)
    plot_rows(rows, args.output_prefix)

    print("[done] wrote:")
    for scenario in ["local_to_peer", "peer_to_local", "same_dev"]:
        print(f"  {args.output_prefix}_{scenario}_bandwidth.png")


if __name__ == "__main__":
    main()
