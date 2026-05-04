#!/usr/bin/env python3

import argparse
import importlib.util
from pathlib import Path

import matplotlib.pyplot as plt
import torch


EXPERIMENT = {
    0: "copy",
    1: "reduce_add_f16",
}

SCENARIO = {
    0: "local_to_peer",
    1: "peer_to_local",
}

SCENARIO_TITLE = {
    "local_to_peer": "local -> peer",
    "peer_to_local": "peer -> local",
}

METHOD = {
    0: "tma_copy",
    1: "fast_copy_u128",
    2: "nccl_sendrecv",
    3: "tma_reduce_add_f16",
    4: "fast_add_f16_u128",
}

COPY_METHODS = [
    "tma_copy",
    "fast_copy_u128",
    "nccl_sendrecv",
]

REDUCE_METHODS = [
    "tma_reduce_add_f16",
    "fast_add_f16_u128",
]


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


def parse_size_one(s: str) -> int:
    text = s.strip().lower()
    mult = 1

    if text.endswith("kb") or text.endswith("k"):
        mult = 1024
        text = text.rstrip("b").rstrip("k")
    elif text.endswith("mb") or text.endswith("m"):
        mult = 1024**2
        text = text.rstrip("b").rstrip("m")
    elif text.endswith("gb") or text.endswith("g"):
        mult = 1024**3
        text = text.rstrip("b").rstrip("g")

    return int(float(text) * mult)


def parse_sizes(s: str) -> list[int]:
    out = [parse_size_one(x) for x in s.split(",") if x.strip()]
    if not out:
        raise ValueError("empty size list")
    return out


def parse_ints(s: str) -> list[int]:
    out = [int(x) for x in s.split(",") if x.strip()]
    if not out:
        raise ValueError("empty integer list")
    if any(x <= 0 for x in out):
        raise ValueError("CTA counts must be positive")
    return out


def format_size(n: int) -> str:
    n = int(n)
    for scale, suffix in ((1024**3, "G"), (1024**2, "M"), (1024, "K")):
        if n >= scale:
            value = n / scale
            return f"{int(value)}{suffix}" if value.is_integer() else f"{value:.1f}{suffix}"
    return f"{n}B"


def normalize_rows(rows):
    out = []

    for row in rows:
        r = dict(row)
        r["experiment_name"] = EXPERIMENT[int(r["experiment"])]
        r["scenario_name"] = SCENARIO[int(r["scenario"])]
        r["method_name"] = METHOD[int(r["method"])]
        r["bytes"] = int(r["bytes"])
        r["num_blocks"] = int(r["num_blocks"])
        r["gbps"] = float(r["gbps"])
        r["latency_ms"] = float(r["latency_ms"])
        out.append(r)

    return out


def benchmark(ext, sizes, ctas, iters, warmup, dev0, dev1, include_nccl):
    if hasattr(ext, "benchmark_tma_bandwidth_experiment_sweep_sm90"):
        rows = ext.benchmark_tma_bandwidth_experiment_sweep_sm90(
            sizes,
            ctas,
            int(iters),
            int(warmup),
            int(dev0),
            int(dev1),
            bool(include_nccl),
        )
        return normalize_rows(rows)

    # Fallback for the current binding if only the old wrapper is exposed.
    rows = []
    for cta in ctas:
        for size in sizes:
            part = ext.benchmark_tma_bandwidth_experiment_sm90(
                int(size),
                int(size),
                int(iters),
                int(warmup),
                int(cta),
                int(dev0),
                int(dev1),
                False,
                bool(include_nccl),
            )
            rows.extend(part)

    return normalize_rows(rows)


def print_table(rows):
    print(
        f"{'experiment':>15} "
        f"{'scenario':>15} "
        f"{'method':>22} "
        f"{'size':>8} "
        f"{'ctas':>6} "
        f"{'GB/s':>10} "
        f"{'us':>10}"
    )

    for r in rows:
        print(
            f"{r['experiment_name']:>15} "
            f"{r['scenario_name']:>15} "
            f"{r['method_name']:>22} "
            f"{format_size(r['bytes']):>8} "
            f"{r['num_blocks']:6d} "
            f"{r['gbps']:10.2f} "
            f"{r['latency_ms'] * 1000.0:10.2f}"
        )


def plot_group(rows, experiment_name, methods, out_path):
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(13, 4), sharex=True)
    legend_handles, legend_labels = None, None

    for ax, scenario in zip(axes, ("local_to_peer", "peer_to_local")):
        subset = [
            r for r in rows
            if r["experiment_name"] == experiment_name
            and r["scenario_name"] == scenario
        ]

        for method in methods:
            method_rows = [r for r in subset if r["method_name"] == method]
            ctas = sorted({r["num_blocks"] for r in method_rows})

            for cta in ctas:
                data = sorted(
                    [r for r in method_rows if r["num_blocks"] == cta],
                    key=lambda x: x["bytes"],
                )
                if not data:
                    continue

                x = [r["bytes"] for r in data]
                y = [r["gbps"] for r in data]
                ax.plot(x, y, marker="o", label=f"{method}, {cta} CTAs")

        if legend_handles is None:
            legend_handles, legend_labels = ax.get_legend_handles_labels()

        xticks = sorted({r["bytes"] for r in subset})
        ax.set_xscale("log", base=2)
        ax.set_xticks(xticks)
        ax.set_xticklabels([format_size(v) for v in xticks], rotation=30, ha="right")
        ax.set_title(SCENARIO_TITLE[scenario])
        ax.grid(True, which="both", linestyle="--", alpha=0.35)

    title = "TMA copy experiment" if experiment_name == "copy" else "TMA reduce/add experiment"
    fig.suptitle(title, y=1.08)
    fig.supylabel("bandwidth (GB/s)")
    fig.supxlabel("buffer size")
    fig.legend(
        legend_handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=min(len(legend_labels or []), 4),
        frameon=False,
    )
    fig.tight_layout(rect=(0.02, 0.02, 1.0, 0.88))
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    print(f"[plot] wrote {out_path}")


def main():
    parser = argparse.ArgumentParser("TMA copy/reduce bandwidth experiment")
    parser.add_argument("--bytes", default="1M,2M,4M,8M,16M,32M,64M,128M")
    parser.add_argument("--ctas", default="1,2,4,8")
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--dev0", type=int, default=0)
    parser.add_argument("--dev1", type=int, default=1)
    parser.add_argument("--no-nccl", action="store_true")
    parser.add_argument("--print-table", action="store_true")
    parser.add_argument("--out-prefix", default="tma_bandwidth")
    args = parser.parse_args()

    assert torch.cuda.is_available(), "torch.cuda.is_available() is False"
    assert torch.cuda.device_count() >= 2, "Need at least 2 GPUs"
    if args.dev0 == args.dev1:
        raise ValueError("--dev0 and --dev1 must be different")

    sizes = parse_sizes(args.bytes)
    ctas = parse_ints(args.ctas)

    print(f"[info] torch={torch.__version__}")
    print(f"[info] dev0={args.dev0} dev1={args.dev1}")
    print(f"[info] sizes={[format_size(x) for x in sizes]}")
    print(f"[info] ctas={ctas}")
    print(f"[info] iters={args.iters} warmup={args.warmup}")
    print(f"[info] include_nccl={not args.no_nccl}")

    ext = load_ooverlap_ext()

    rows = benchmark(
        ext=ext,
        sizes=sizes,
        ctas=ctas,
        iters=args.iters,
        warmup=args.warmup,
        dev0=args.dev0,
        dev1=args.dev1,
        include_nccl=not args.no_nccl,
    )

    if args.print_table:
        print_table(rows)

    plot_group(
        rows,
        experiment_name="copy",
        methods=COPY_METHODS if not args.no_nccl else COPY_METHODS[:2],
        out_path=Path(f"{args.out_prefix}_copy.png"),
    )

    plot_group(
        rows,
        experiment_name="reduce_add_f16",
        methods=REDUCE_METHODS,
        out_path=Path(f"{args.out_prefix}_reduce.png"),
    )

    torch.cuda.synchronize(args.dev0)
    torch.cuda.synchronize(args.dev1)
    print("PASS")


if __name__ == "__main__":
    main()
