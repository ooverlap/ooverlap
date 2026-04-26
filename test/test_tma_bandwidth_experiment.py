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
    3: "gmem_copy_u32",
    4: "nccl_sendrecv",
    5: "gmem_copy_u64",
    6: "gmem_copy_u128",
    7: "fast_add_f16_u128",
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
        f"{'method':>27} "
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
            f"{r['method_name']:>27} "
            f"{r['mib']:10.1f} "
            f"{r['gbps']:12.2f} "
            f"{int(r['src_device']):4d} "
            f"{int(r['dst_device']):4d} "
            f"{int(r['kernel_device']):4d} "
            f"{int(r['num_blocks']):8d}"
        )


def present_methods(rows, preferred):
    present = {r["method_name"] for r in rows}
    return [m for m in preferred if m in present]


def plot_one_group(rows, scenario, methods, title, output_path):
    subset = [r for r in rows if r["scenario_name"] == scenario]
    if not subset:
        return False

    plotted = False
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
        plotted = True

    if not plotted:
        plt.close()
        return False

    plt.xscale("log", base=2)
    plt.xlabel("Transfer size (MiB)")
    plt.ylabel("Effective bandwidth (GB/s)")
    plt.title(title)
    plt.grid(True, which="both")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()

    return True


def plot_rows(rows, output_prefix):
    scenarios = ["local_to_peer", "peer_to_local", "same_dev"]

    copy_methods = present_methods(
        rows,
        [
            "tma_copy",
            "gmem_copy_u32",
            "gmem_copy_u64",
            "gmem_copy_u128",
            "nccl_sendrecv",
            "mem_async_copy",
        ],
    )

    reduce_methods = present_methods(
        rows,
        [
            "tma_reduce_add_f16",
            "fast_add_f16_u128",
        ],
    )

    written = []

    for scenario in scenarios:
        if scenario == "local_to_peer":
            scenario_title = "local -> peer"
        elif scenario == "peer_to_local":
            scenario_title = "peer -> local"
        else:
            scenario_title = "same device"

        copy_path = f"{output_prefix}_{scenario}_copy_bandwidth.png"
        if plot_one_group(
            rows,
            scenario,
            copy_methods,
            f"Copy bandwidth: {scenario_title}",
            copy_path,
        ):
            written.append(copy_path)

        reduce_path = f"{output_prefix}_{scenario}_reduce_vs_fast_add_bandwidth.png"
        if plot_one_group(
            rows,
            scenario,
            reduce_methods,
            f"Reduce/add bandwidth: {scenario_title}",
            reduce_path,
        ):
            written.append(reduce_path)

    return written


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
        help="Include slow cuda::memcpy_async baseline.",
    )
    parser.add_argument(
        "--include-nccl",
        action="store_true",
        help="Include NCCL ncclSend/ncclRecv one-way copy baseline.",
    )
    args = parser.parse_args()

    print(f"[info] torch {torch.__version__}")
    print(f"[info] cuda available: {torch.cuda.is_available()}")
    print(f"[info] cuda device count: {torch.cuda.device_count()}")
    print(
        f"[info] min_bytes={args.min_bytes} max_bytes={args.max_bytes} "
        f"iters={args.iters} warmup={args.warmup} blocks={args.num_blocks} "
        f"dev0={args.dev0} dev1={args.dev1} "
        f"include_mem_async={args.include_mem_async} "
        f"include_nccl={args.include_nccl}"
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
        args.include_nccl,
    )

    rows = normalize_rows(rows)
    print_table(rows)
    written = plot_rows(rows, args.output_prefix)

    print("[done] wrote:")
    for path in written:
        print(f"  {path}")


if __name__ == "__main__":
    main()
