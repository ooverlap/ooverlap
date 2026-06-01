#!/usr/bin/env python3

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt


EXPERIMENT = {
    0: "copy",
    1: "reduce_add_f16",
}

SCENARIO = {
    0: "local_to_peer",
    1: "peer_to_local",
}

SCENARIO_TITLE = {
    "local_to_peer": "Local-to-Remote Transfer",
    "peer_to_local": "Remote-to-Local Transfer",
}

METHOD = {
    0: "tma_copy",
    1: "fast_copy_u128",
    2: "nccl_sendrecv",
    3: "tma_reduce_add_f16",
    4: "fast_add_f16_u128",
}

METHOD_TITLE = {
    "tma_copy": "TMA Copy",
    "fast_copy_u128": "Vectorized Copy",
    "nccl_sendrecv": "NCCL Send/Recv",
    "tma_reduce_add_f16": "TMA FP16 Reduction",
    "fast_add_f16_u128": "Vectorized FP16 Reduction",
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
    for scale, suffix in ((1024**3, "GiB"), (1024**2, "MiB"), (1024, "KiB")):
        if n >= scale:
            value = n / scale
            return f"{int(value)} {suffix}" if value.is_integer() else f"{value:.1f} {suffix}"
    return f"{n} B"


def normalize_rows(rows, requested_cta=None):
    out = []

    for row in rows:
        r = dict(row)
        r["experiment_name"] = EXPERIMENT[int(r["experiment"])]
        r["scenario_name"] = SCENARIO[int(r["scenario"])]
        r["method_name"] = METHOD[int(r["method"])]
        r["bytes"] = int(r["bytes"])
        r["num_blocks"] = int(r["num_blocks"])
        r["requested_cta"] = int(requested_cta if requested_cta is not None else r["num_blocks"])
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
        return rows

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

    return rows


def run_worker(args):
    import torch

    assert torch.cuda.is_available(), "torch.cuda.is_available() is False"
    assert torch.cuda.device_count() >= 2, "Need at least 2 GPUs"
    if args.dev0 == args.dev1:
        raise ValueError("--dev0 and --dev1 must be different")

    sizes = parse_sizes(args.bytes)
    cta = int(args.worker_cta)

    print(f"[worker] NCCL_MAX_CTAS={os.environ.get('NCCL_MAX_CTAS')}")
    print(f"[worker] benchmark CTA={cta}")

    ext = load_ooverlap_ext()
    rows = benchmark(
        ext=ext,
        sizes=sizes,
        ctas=[cta],
        iters=args.iters,
        warmup=args.warmup,
        dev0=args.dev0,
        dev1=args.dev1,
        include_nccl=not args.no_nccl,
    )

    rows = normalize_rows(rows, requested_cta=cta)

    out_path = Path(args.worker_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(rows, indent=2))

    torch.cuda.synchronize(args.dev0)
    torch.cuda.synchronize(args.dev1)


def run_cta_in_child(args, cta, out_dir):
    out_path = out_dir / f"rows_cta_{cta}.json"

    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker-cta",
        str(cta),
        "--worker-out",
        str(out_path),
        "--bytes",
        args.bytes,
        "--ctas",
        str(cta),
        "--iters",
        str(args.iters),
        "--warmup",
        str(args.warmup),
        "--dev0",
        str(args.dev0),
        "--dev1",
        str(args.dev1),
    ]

    if args.no_nccl:
        cmd.append("--no-nccl")

    env = os.environ.copy()
    env["NCCL_MAX_CTAS"] = str(cta)

    print(f"[run] CTA={cta}: NCCL_MAX_CTAS={cta}")
    subprocess.run(cmd, check=True, env=env)

    return json.loads(out_path.read_text())


def print_table(rows):
    print(
        f"{'experiment':>18} "
        f"{'direction':>24} "
        f"{'method':>34} "
        f"{'size':>10} "
        f"{'CTA limit':>10} "
        f"{'GB/s':>10} "
        f"{'latency us':>12}"
    )

    for r in rows:
        print(
            f"{r['experiment_name']:>18} "
            f"{SCENARIO_TITLE[r['scenario_name']]:>24} "
            f"{METHOD_TITLE[r['method_name']]:>34} "
            f"{format_size(r['bytes']):>10} "
            f"{r['requested_cta']:10d} "
            f"{r['gbps']:10.2f} "
            f"{r['latency_ms'] * 1000.0:12.2f}"
        )


def plot_group(rows, experiment_name, methods, ctas, out_path):
    fig, axes = plt.subplots(
        nrows=2,
        ncols=len(ctas),
        figsize=(5.2 * len(ctas), 7.0),
        sharex=True,
        sharey=True,
        squeeze=False,
    )

    legend_handles = []
    legend_labels = []

    for col, cta in enumerate(ctas):
        for row_idx, scenario in enumerate(("local_to_peer", "peer_to_local")):
            ax = axes[row_idx][col]

            subset = [
                r for r in rows
                if r["experiment_name"] == experiment_name
                and r["scenario_name"] == scenario
                and r["requested_cta"] == cta
            ]

            for method in methods:
                data = sorted(
                    [r for r in subset if r["method_name"] == method],
                    key=lambda x: x["bytes"],
                )
                if not data:
                    continue

                x = [r["bytes"] for r in data]
                y = [r["gbps"] for r in data]
                line, = ax.plot(
                    x,
                    y,
                    marker="o",
                    label=METHOD_TITLE[method],
                )

                if col == 0 and row_idx == 0:
                    legend_handles.append(line)
                    legend_labels.append(METHOD_TITLE[method])

            xticks = sorted({r["bytes"] for r in subset})
            ax.set_xscale("log", base=2)
            if xticks:
                ax.set_xticks(xticks)
                ax.set_xticklabels(
                    [format_size(v) for v in xticks],
                    rotation=30,
                    ha="right",
                )

            if row_idx == 0:
                ax.set_title(f"Maximum CTAs = {cta}", fontsize=12)

            if col == 0:
                ax.set_ylabel(
                    f"{SCENARIO_TITLE[scenario]}\nEffective Bandwidth (GB/s)"
                )

            ax.grid(True, which="both", linestyle="--", alpha=0.35)

    if experiment_name == "copy":
        title = "Two-GPU Copy Bandwidth under CTA Limits"
    else:
        title = "Two-GPU FP16 Reduction Bandwidth under CTA Limits"

    fig.suptitle(title, fontsize=14)
    fig.supxlabel("Message Size")

    if legend_handles:
        fig.legend(
            legend_handles,
            legend_labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.955),
            ncol=min(len(legend_labels), 3),
            frameon=False,
        )

    fig.tight_layout(rect=(0.02, 0.02, 1.0, 0.90))
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
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

    parser.add_argument("--worker-cta", type=int, default=None)
    parser.add_argument("--worker-out", default=None)

    args = parser.parse_args()

    if args.worker_cta is not None:
        if args.worker_out is None:
            raise ValueError("--worker-out is required with --worker-cta")
        run_worker(args)
        return

    import torch

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
    print("[info] running one child process per CTA so NCCL_MAX_CTAS is isolated")

    all_rows = []
    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="tma_bandwidth_") as tmp:
        tmp_dir = Path(tmp)
        for cta in ctas:
            all_rows.extend(run_cta_in_child(args, cta, tmp_dir))

    json_path = out_prefix.with_suffix(".json")
    json_path.write_text(json.dumps(all_rows, indent=2))
    print(f"[json] wrote {json_path}")

    if args.print_table:
        print_table(all_rows)

    plot_group(
        all_rows,
        experiment_name="copy",
        methods=COPY_METHODS if not args.no_nccl else COPY_METHODS[:2],
        ctas=ctas,
        out_path=out_prefix.parent / f"{out_prefix.name}_copy.png",
    )

    plot_group(
        all_rows,
        experiment_name="reduce_add_f16",
        methods=REDUCE_METHODS,
        ctas=ctas,
        out_path=out_prefix.parent / f"{out_prefix.name}_reduce.png",
    )

    torch.cuda.synchronize(args.dev0)
    torch.cuda.synchronize(args.dev1)
    print("PASS")


if __name__ == "__main__":
    main()
