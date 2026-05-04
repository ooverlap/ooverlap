import argparse
import importlib.util
import multiprocessing as mp
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


def bytes_to_numel(size_bytes: int) -> int:
    if size_bytes <= 0:
        raise ValueError("buffer size must be > 0")
    if size_bytes % FP16_BYTES != 0:
        raise ValueError(f"buffer size must be divisible by {FP16_BYTES} for fp16")
    return size_bytes // FP16_BYTES


def rank_worker(
    q,
    collective: str,
    sizes_bytes: list[int],
    local_rank: int,
    dev0: int,
    dev1: int,
    broker_key: str,
    nccl_id,
    iters: int,
    warmup: int,
    verify: bool,
):
    try:
        ext = load_ooverlap_ext()
        sizes_numel = [bytes_to_numel(x) for x in sizes_bytes]

        rows = ext.benchmark_ipc_collective_rank_sm90(
            collective,
            sizes_numel,
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


def run_two_rank_collective(
    collective: str,
    sizes_bytes: list[int],
    dev0: int,
    dev1: int,
    iters: int,
    warmup: int,
    verify: bool,
):
    ext = load_ooverlap_ext()
    nccl_id = ext.generate_nccl_id()
    broker_key = f"ooverlap-ipc-{collective}-{uuid.uuid4().hex}"

    ctx = mp.get_context("spawn")
    q = ctx.Queue()

    procs = [
        ctx.Process(
            target=rank_worker,
            args=(
                q,
                collective,
                sizes_bytes,
                rank,
                dev0,
                dev1,
                broker_key,
                nccl_id,
                iters,
                warmup,
                verify,
            ),
        )
        for rank in (0, 1)
    ]

    for p in procs:
        p.start()

    results = {}
    errors = []

    for _ in procs:
        rank, rows, err = q.get()
        if err is not None:
            errors.append(f"rank{rank}: {err}")
        else:
            results[rank] = rows

    for p in procs:
        p.join()

    for p in procs:
        if p.exitcode != 0:
            errors.append(f"process pid={p.pid} exitcode={p.exitcode}")

    if errors:
        raise RuntimeError("\n".join(errors))

    return results


def combine_rank_rows(rank_rows):
    rows0 = rank_rows[0]
    rows1 = rank_rows[1]

    if len(rows0) != len(rows1):
        raise RuntimeError("rank row count mismatch")

    rows = []

    for a, b in zip(rows0, rows1):
        numel = int(a["numel"])
        size_bytes = int(a["bytes"])
        iters = int(a["iters"])

        oo_total_ms = max(float(a["oo_total_ms"]), float(b["oo_total_ms"]))
        nccl_total_ms = max(float(a["nccl_total_ms"]), float(b["nccl_total_ms"]))

        rows.append(
            {
                "numel": numel,
                "bytes": size_bytes,
                "iters": iters,
                "oo_latency_ms": oo_total_ms / iters,
                "nccl_latency_ms": nccl_total_ms / iters,
            }
        )

    return rows


def bandwidth_gbps(size_bytes: int, latency_ms: float) -> float:
    if latency_ms <= 0.0:
        return 0.0
    return (size_bytes / 1e9) / (latency_ms / 1e3)


def run_smoke(smoke_bytes: int, dev0: int, dev1: int):
    print(f"[smoke] size_bytes={smoke_bytes}")

    for collective in COLLECTIVES:
        print(f"[smoke] {collective}")
        run_two_rank_collective(
            collective=collective,
            sizes_bytes=[smoke_bytes],
            dev0=dev0,
            dev1=dev1,
            iters=1,
            warmup=0,
            verify=True,
        )

    print("[smoke] PASS")


def run_metric_suite(
    metric: str,
    sizes_bytes: list[int],
    dev0: int,
    dev1: int,
    iters: int,
    warmup: int,
    verify: bool,
    out_path: Path,
):
    all_rows = {}

    for collective in COLLECTIVES:
        print(f"[bench] metric={metric} collective={collective} sizes={sizes_bytes}")

        rank_rows = run_two_rank_collective(
            collective=collective,
            sizes_bytes=sizes_bytes,
            dev0=dev0,
            dev1=dev1,
            iters=iters,
            warmup=warmup,
            verify=verify,
        )

        all_rows[collective] = combine_rank_rows(rank_rows)

    plot_metric(metric, all_rows, out_path)


def plot_metric(metric: str, all_rows, out_path: Path):
    fig, axes = plt.subplots(
        nrows=3,
        ncols=1,
        figsize=(8, 10),
        sharex=True,
    )

    for ax, collective in zip(axes, COLLECTIVES):
        rows = all_rows[collective]
        x = [r["bytes"] for r in rows]

        if metric == "latency":
            y_oo = [r["oo_latency_ms"] for r in rows]
            y_nccl = [r["nccl_latency_ms"] for r in rows]
            ylabel = "latency (ms)"
        elif metric == "bandwidth":
            y_oo = [bandwidth_gbps(r["bytes"], r["oo_latency_ms"]) for r in rows]
            y_nccl = [bandwidth_gbps(r["bytes"], r["nccl_latency_ms"]) for r in rows]
            ylabel = "bandwidth (GB/s)"
        else:
            raise ValueError(f"unknown metric: {metric}")

        ax.plot(x, y_oo, marker="o", label="ooverlap")
        ax.plot(x, y_nccl, marker="o", label="nccl")

        ax.set_xscale("log", base=2)
        ax.set_title(collective)
        ax.set_ylabel(ylabel)
        ax.grid(True, which="both", linestyle="--", alpha=0.35)
        ax.legend()

    axes[-1].set_xlabel("buffer size (bytes)")

    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    print(f"[plot] wrote {out_path}")


def main():
    parser = argparse.ArgumentParser("IPC collective smoke + benchmark plotter")

    parser.add_argument("--mode", choices=["smoke", "bench", "both"], default="both")
    parser.add_argument("--dev0", type=int, default=0)
    parser.add_argument("--dev1", type=int, default=1)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--verify", action="store_true")

    parser.add_argument(
        "--smoke-bytes",
        default="1M",
        help="single buffer size for smoke verification, e.g. 1M",
    )

    parser.add_argument(
        "--metric",
        choices=["latency", "bandwidth", "both"],
        default="bandwidth",
    )

    parser.add_argument(
        "--bytes",
        default="1M,2M,4M,8M,16M,32M,64M,128M,256M",
        help="default buffer sizes for selected metric(s)",
    )

    parser.add_argument(
        "--latency-bytes",
        default=None,
        help="override sizes for latency plot",
    )

    parser.add_argument(
        "--bandwidth-bytes",
        default=None,
        help="override sizes for bandwidth plot",
    )

    parser.add_argument(
        "--out-prefix",
        default="ipc_collective",
    )

    args = parser.parse_args()

    assert torch.cuda.is_available(), "torch.cuda.is_available() is False"
    assert torch.cuda.device_count() >= 2, "Need at least 2 GPUs"

    if args.dev0 == args.dev1:
        raise ValueError("--dev0 and --dev1 must be different")

    print(f"[info] torch={torch.__version__}")
    print(f"[info] devices={torch.cuda.device_count()} dev0={args.dev0} dev1={args.dev1}")
    print(f"[info] iters={args.iters} warmup={args.warmup}")

    if args.mode in ("smoke", "both"):
        run_smoke(
            smoke_bytes=parse_size_one(args.smoke_bytes),
            dev0=args.dev0,
            dev1=args.dev1,
        )

    if args.mode in ("bench", "both"):
        default_sizes = parse_sizes(args.bytes)

        metrics = []
        if args.metric in ("latency", "both"):
            metrics.append("latency")
        if args.metric in ("bandwidth", "both"):
            metrics.append("bandwidth")

        for metric in metrics:
            if metric == "latency" and args.latency_bytes is not None:
                sizes = parse_sizes(args.latency_bytes)
            elif metric == "bandwidth" and args.bandwidth_bytes is not None:
                sizes = parse_sizes(args.bandwidth_bytes)
            else:
                sizes = default_sizes

            run_metric_suite(
                metric=metric,
                sizes_bytes=sizes,
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
