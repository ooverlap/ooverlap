import argparse
import importlib.util
import multiprocessing as mp
import os
import uuid
from pathlib import Path

import matplotlib.pyplot as plt
import torch


COLLECTIVES = ("allreduce", "reduce_scatter", "all_gather")
FP16_BYTES = 2


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
        raise ValueError("size list is empty")
    return out

def parse_ints(s: str | None) -> list[int | None]:
    if s is None or not s.strip():
        return [None]

    text = s.strip().lower()

    if text == "all":
        return [None, 1, 2, 4, 8]

    vals = []
    for x in text.split(","):
        x = x.strip()
        if not x:
            continue

        if x in ("default", "none", "unset"):
            vals.append(None)
        else:
            v = int(x)
            if v <= 0:
                raise ValueError("--ctas values must be positive")
            vals.append(v)

    if not vals:
        return [None]

    return vals

def format_size_bytes(n: int) -> str:
    n = int(n)
    for scale, suffix in ((1024**3, "G"), (1024**2, "M"), (1024, "K")):
        if n >= scale:
            value = n / scale
            return f"{int(value)}{suffix}" if value.is_integer() else f"{value:.1f}{suffix}"
    return f"{n}B"


def bytes_to_numel(size_bytes: int) -> int:
    if size_bytes <= 0:
        raise ValueError("buffer size must be > 0")
    if size_bytes % FP16_BYTES != 0:
        raise ValueError(f"buffer size must be divisible by {FP16_BYTES} for fp16")
    return size_bytes // FP16_BYTES


def set_cta_env(ctas: int | None):
    for k in ("OOVERLAP_MAX_CTAS", "NCCL_MIN_CTAS", "NCCL_MAX_CTAS"):
        os.environ.pop(k, None)
    if ctas is not None:
        os.environ["OOVERLAP_MAX_CTAS"] = str(ctas)
        os.environ["NCCL_MIN_CTAS"] = str(ctas)
        os.environ["NCCL_MAX_CTAS"] = str(ctas)



def cta_label(ctas: int | None) -> str:
    return "" if ctas is None else f"{ctas} CTAs"


def cta_log_label(ctas: int | None) -> str:
    return "unrestricted" if ctas is None else f"{ctas} CTAs"


def cta_file_suffix(ctas: int | None) -> str:
    return "" if ctas is None else f"_{ctas}ctas"


def name_with_cta_suffix(path: Path, ctas: int | None) -> Path:
    return path.with_name(f"{path.stem}{cta_file_suffix(ctas)}{path.suffix}")


def label_with_cta(base: str, ctas: int | None) -> str:
    suffix = cta_label(ctas)
    return base if not suffix else f"{base} {suffix}"

def rank_worker(q, collective, sizes_bytes, local_rank, dev0, dev1, broker_key,
                nccl_id, iters, warmup, verify, ctas):
    try:
        set_cta_env(ctas)
        ext = load_ooverlap_ext()
        rows = ext.benchmark_ipc_collective_rank_sm90(
            collective,
            [bytes_to_numel(x) for x in sizes_bytes],
            int(local_rank),
            int(dev0),
            int(dev1),
            broker_key,
            nccl_id,
            int(iters),
            int(warmup),
            bool(verify),
        )
        q.put((local_rank, rows, None))
    except BaseException as exc:
        q.put((local_rank, None, repr(exc)))


def run_two_rank_collective(collective, sizes_bytes, dev0, dev1, iters, warmup,
                            verify, ctas):
    set_cta_env(ctas)
    ext = load_ooverlap_ext()
    nccl_id = ext.generate_nccl_id()
    broker_key = f"ooverlap-ipc-{collective}-{uuid.uuid4().hex}"

    ctx = mp.get_context("spawn")
    q = ctx.Queue()

    procs = [
        ctx.Process(
            target=rank_worker,
            args=(q, collective, sizes_bytes, rank, dev0, dev1, broker_key,
                  nccl_id, iters, warmup, verify, ctas),
        )
        for rank in (0, 1)
    ]

    for p in procs:
        p.start()

    results, errors = {}, []
    for _ in procs:
        rank, rows, err = q.get()
        if err is None:
            results[rank] = rows
        else:
            errors.append(f"rank{rank}: {err}")

    for p in procs:
        p.join()
        if p.exitcode != 0:
            errors.append(f"process pid={p.pid} exitcode={p.exitcode}")

    if errors:
        raise RuntimeError("\n".join(errors))
    return results


def combine_rank_rows(rank_rows):
    rows = []
    for a, b in zip(rank_rows[0], rank_rows[1]):
        iters = int(a["iters"])
        rows.append(
            {
                "bytes": int(a["bytes"]),
                "iters": iters,
                "oo_latency_ms": max(float(a["oo_total_ms"]), float(b["oo_total_ms"])) / iters,
                "nccl_latency_ms": max(float(a["nccl_total_ms"]), float(b["nccl_total_ms"])) / iters,
            }
        )
    return rows


def bandwidth_gbps(size_bytes: int, latency_ms: float) -> float:
    return 0.0 if latency_ms <= 0.0 else (size_bytes / 1e9) / (latency_ms / 1e3)


def metric_values(metric: str, rows):
    if metric == "latency":
        return (
            [r["oo_latency_ms"] * 1000.0 for r in rows],
            [r["nccl_latency_ms"] * 1000.0 for r in rows],
            "latency (us)",
        )
    if metric == "bandwidth":
        return (
            [bandwidth_gbps(r["bytes"], r["oo_latency_ms"]) for r in rows],
            [bandwidth_gbps(r["bytes"], r["nccl_latency_ms"]) for r in rows],
            "bandwidth (GB/s)",
        )
    if metric == "speedup":
        return (
            [
                r["nccl_latency_ms"] / r["oo_latency_ms"]
                if r["oo_latency_ms"] > 0.0 else 0.0
                for r in rows
            ],
            None,
            "speedup vs NCCL (x)",
        )
    raise ValueError(f"unknown metric: {metric}")


def run_smoke(smoke_bytes, dev0, dev1):
    print(f"[smoke] size_bytes={smoke_bytes}")
    for collective in COLLECTIVES:
        print(f"[smoke] {collective}")
        run_two_rank_collective(
            collective, [smoke_bytes], dev0, dev1,
            iters=1, warmup=0, verify=True, ctas=None
        )
    print("[smoke] PASS")



def run_metric_suite(metric, sizes_bytes, cta_values, dev0, dev1, iters, warmup,
                     verify, out_path):
    all_rows = {c: {} for c in COLLECTIVES}

    for ctas in cta_values:
        for collective in COLLECTIVES:
            print(
                f"[bench] metric={metric} collective={collective} "
                f"ctas={cta_log_label(ctas)} sizes={sizes_bytes}"
            )
            rank_rows = run_two_rank_collective(
                collective, sizes_bytes, dev0, dev1, iters, warmup, verify, ctas
            )
            all_rows[collective][ctas] = combine_rank_rows(rank_rows)

    if metric == "speedup":
        for ctas in cta_values:
            one_cta_rows = {
                collective: {ctas: all_rows[collective][ctas]}
                for collective in COLLECTIVES
            }
            plot_metric(metric, one_cta_rows, name_with_cta_suffix(out_path, ctas))
    else:
        plot_metric(metric, all_rows, out_path)


def plot_metric(metric, all_rows, out_path: Path):
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 4), sharex=True)
    legend_handles, legend_labels, ylabel = None, None, ""

    for ax, collective in zip(axes, COLLECTIVES):
        for ctas, rows in all_rows[collective].items():
            x = [r["bytes"] for r in rows]
            y_oo, y_nccl, ylabel = metric_values(metric, rows)

            if metric == "speedup":
                ax.plot(x, y_oo, marker="o", label="ooverlap / nccl")
            else:
                ax.plot(x, y_oo, marker="o", label=label_with_cta("ooverlap", ctas))
                ax.plot(x, y_nccl, marker="s", linestyle="--", label=label_with_cta("nccl", ctas))

        if metric == "speedup":
            ax.axhline(1.0, color="gray", linestyle="--", linewidth=1.0, label="nccl baseline")

        if legend_handles is None:
            legend_handles, legend_labels = ax.get_legend_handles_labels()

        xticks = [r["bytes"] for r in next(iter(all_rows[collective].values()))]
        ax.set_xscale("log", base=2)
        ax.set_xticks(xticks)
        ax.set_xticklabels([format_size_bytes(v) for v in xticks], rotation=30, ha="right")
        ax.set_title(collective)
        ax.grid(True, which="both", linestyle="--", alpha=0.35)

    fig.supylabel(ylabel)
    fig.supxlabel("buffer size")
    fig.legend(
        legend_handles,
        legend_labels,
        loc="upper center",
        ncol=min(len(legend_labels), 6),
        bbox_to_anchor=(0.5, 1.08),
        frameon=False,
    )
    fig.tight_layout(rect=(0.02, 0.02, 1.0, 0.88))
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    print(f"[plot] wrote {out_path}")

def main():
    parser = argparse.ArgumentParser("IPC collective smoke + benchmark plotter")
    parser.add_argument("--mode", choices=["smoke", "bench", "both"], default="both")
    parser.add_argument("--dev0", type=int, default=0)
    parser.add_argument("--dev1", type=int, default=1)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--smoke-bytes", default="1M")
    parser.add_argument("--metric", choices=["latency", "bandwidth", "speedup", "both", "all"], default="bandwidth")
    parser.add_argument("--bytes", default="1M,2M,4M,8M,16M,32M,64M,128M,256M")
    parser.add_argument("--latency-bytes", default=None)
    parser.add_argument("--bandwidth-bytes", default=None)
    parser.add_argument("--speedup-bytes", default=None)
    parser.add_argument("--ctas", default=None, help="comma-separated communication CTA counts, e.g. 1,2,4,8")
    parser.add_argument("--out-prefix", default="ipc_collective")

    args = parser.parse_args()

    assert torch.cuda.is_available(), "torch.cuda.is_available() is False"
    assert torch.cuda.device_count() >= 2, "Need at least 2 GPUs"
    if args.dev0 == args.dev1:
        raise ValueError("--dev0 and --dev1 must be different")

    print(f"[info] torch={torch.__version__}")
    print(f"[info] devices={torch.cuda.device_count()} dev0={args.dev0} dev1={args.dev1}")
    print(f"[info] iters={args.iters} warmup={args.warmup} ctas={args.ctas or 'default'}")

    if args.mode in ("smoke", "both"):
        run_smoke(parse_size_one(args.smoke_bytes), args.dev0, args.dev1)

    if args.mode in ("bench", "both"):
        default_sizes = parse_sizes(args.bytes)
        cta_values = parse_ints(args.ctas)

        metrics = []
        if args.metric in ("latency", "both", "all"):
            metrics.append("latency")
        if args.metric in ("bandwidth", "both", "all"):
            metrics.append("bandwidth")
        if args.metric in ("speedup", "all"):
            metrics.append("speedup")

        for metric in metrics:
            override = getattr(args, f"{metric}_bytes", None)
            sizes = parse_sizes(override) if override else default_sizes
            run_metric_suite(
                metric=metric,
                sizes_bytes=sizes,
                cta_values=cta_values,
                dev0=args.dev0,
                dev1=args.dev1,
                iters=args.iters,
                warmup=args.warmup,
                verify=args.verify,
                out_path=Path(f"{args.out_prefix}_{metric}.png"),
            )

    torch.cuda.synchronize(args.dev0)
    torch.cuda.synchronize(args.dev1)
    print("PASS")


if __name__ == "__main__":
    main()
