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


# OOVERLAP_IPC_COLLECTIVE_MULTI_GPU_V1
def parse_devices(value: str) -> list[int]:
    devices = [int(item.strip()) for item in value.split(",") if item.strip()]
    if len(devices) < 2:
        raise ValueError("--devices must contain at least two device ids")
    if len(set(devices)) != len(devices):
        raise ValueError("--devices must not contain duplicates")
    if any(device < 0 for device in devices):
        raise ValueError("--devices must contain non-negative device ids")
    return devices


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
    for k in ("OOVERLAP_MAX_CTAS", "NCCL_MAX_CTAS"):
        os.environ.pop(k, None)
    if ctas is not None:
        os.environ["OOVERLAP_MAX_CTAS"] = str(ctas)
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

def rank_worker(q, collective, sizes_bytes, local_rank, devices, broker_key,
                nccl_id, iters, warmup, verify, ctas):
    try:
        set_cta_env(ctas)
        ext = load_ooverlap_ext()
        rows = ext.benchmark_ipc_collective_rank_sm90(
            collective,
            [bytes_to_numel(x) for x in sizes_bytes],
            int(local_rank),
            devices,
            broker_key,
            nccl_id,
            int(iters),
            int(warmup),
            bool(verify),
        )
        q.put((local_rank, rows, None))
    except BaseException as exc:
        q.put((local_rank, None, repr(exc)))


def run_rank_collective(collective, sizes_bytes, devices, iters, warmup,
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
            args=(q, collective, sizes_bytes, rank, devices, broker_key,
                  nccl_id, iters, warmup, verify, ctas),
        )
        for rank in range(len(devices))
    ]

    for proc in procs:
        proc.start()

    results, errors = {}, []
    for _ in procs:
        rank, rows, err = q.get()
        if err is None:
            results[rank] = rows
        else:
            errors.append(f"rank{rank}: {err}")

    for proc in procs:
        proc.join()
        if proc.exitcode != 0:
            errors.append(f"process pid={proc.pid} exitcode={proc.exitcode}")

    if errors:
        raise RuntimeError("\n".join(errors))
    return results


def per_rank_bandwidth_bytes(collective: str, rows) -> int:
    """Return the byte count used for per-rank bandwidth normalization."""
    full_bytes = int(rows[0]["bytes"])

    if collective == "allreduce":
        return full_bytes

    if collective in ("reduce_scatter", "all_gather"):
        shard_values = [
            int(float(row["local_shard_bytes"]))
            for row in rows
            if "local_shard_bytes" in row
        ]
        if len(shard_values) == len(rows):
            return max(shard_values)

        world_size = int(round(float(rows[0].get("world_size", len(rows)))))
        return full_bytes // max(world_size, 1)

    return full_bytes


def combine_rank_rows(collective, rank_rows):
    expected_ranks = set(range(len(rank_rows)))
    if set(rank_rows) != expected_ranks:
        raise RuntimeError(
            f"expected rank results for {sorted(expected_ranks)}, "
            f"got {sorted(rank_rows)}"
        )

    ordered = [rank_rows[rank] for rank in range(len(rank_rows))]
    row_counts = {len(rows) for rows in ordered}
    if len(row_counts) != 1:
        raise RuntimeError(f"rank result lengths differ: {sorted(row_counts)}")

    combined = []
    for index in range(len(ordered[0])):
        rank_values = [rows[index] for rows in ordered]
        first = rank_values[0]
        iters = int(first["iters"])

        row = {
            "bytes": int(first["bytes"]),
            "world_size": len(rank_values),
            "bandwidth_bytes_per_rank": per_rank_bandwidth_bytes(
                collective, rank_values
            ),
            "iters": iters,
            "oo_latency_ms": max(
                float(value["oo_total_ms"]) for value in rank_values
            ) / iters,
            "nccl_latency_ms": max(
                float(value["nccl_total_ms"]) for value in rank_values
            ) / iters,
        }

        if all("nccl_symmetric_total_ms" in value for value in rank_values):
            row["nccl_symmetric_latency_ms"] = max(
                float(value["nccl_symmetric_total_ms"])
                for value in rank_values
            ) / iters

        combined.append(row)

    return combined


def bandwidth_gbps(size_bytes: int, latency_ms: float) -> float:
    return 0.0 if latency_ms <= 0.0 else (size_bytes / 1e9) / (latency_ms / 1e3)


def metric_values(metric: str, rows):
    have_symmetric = all("nccl_symmetric_latency_ms" in r for r in rows)

    if metric == "latency":
        return (
            [r["oo_latency_ms"] * 1000.0 for r in rows],
            [r["nccl_latency_ms"] * 1000.0 for r in rows],
            [r["nccl_symmetric_latency_ms"] * 1000.0 for r in rows] if have_symmetric else None,
            "Latency (µs)",
        )
    if metric == "bandwidth":
        bytes_per_rank = [r.get("bandwidth_bytes_per_rank", r["bytes"]) for r in rows]
        return (
            [bandwidth_gbps(b, r["oo_latency_ms"]) for b, r in zip(bytes_per_rank, rows)],
            [bandwidth_gbps(b, r["nccl_latency_ms"]) for b, r in zip(bytes_per_rank, rows)],
            [bandwidth_gbps(b, r["nccl_symmetric_latency_ms"]) for b, r in zip(bytes_per_rank, rows)]
            if have_symmetric else None,
            "Per-rank bandwidth (GB/s)",
        )
    if metric == "speedup":
        return (
            [
                r["nccl_latency_ms"] / r["oo_latency_ms"]
                if r["oo_latency_ms"] > 0.0 else 0.0
                for r in rows
            ],
            None,
            [
                r["nccl_latency_ms"] / r["nccl_symmetric_latency_ms"]
                if r.get("nccl_symmetric_latency_ms", 0.0) > 0.0 else 0.0
                for r in rows
            ] if have_symmetric else None,
            "Speedup relative to normal NCCL (×)",
        )
    raise ValueError(f"unknown metric: {metric}")


def validate_sizes_for_collective(collective, sizes_bytes, world_size):
    if collective not in ("reduce_scatter", "all_gather"):
        return

    invalid = [
        size_bytes
        for size_bytes in sizes_bytes
        if bytes_to_numel(size_bytes) % world_size != 0
    ]
    if invalid:
        rendered = ", ".join(format_size_bytes(value) for value in invalid)
        raise ValueError(
            f"{collective} sizes must be divisible by world size {world_size}: "
            f"{rendered}"
        )


def run_smoke(smoke_bytes, devices):
    print(f"[smoke] size_bytes={smoke_bytes} devices={devices}")
    for collective in COLLECTIVES:
        validate_sizes_for_collective(collective, [smoke_bytes], len(devices))
        print(f"[smoke] {collective}")
        run_rank_collective(
            collective, [smoke_bytes], devices,
            iters=1, warmup=0, verify=True, ctas=None
        )
    print("[smoke] PASS")


def run_metric_suite(metric, sizes_bytes, cta_values, devices, iters, warmup,
                     verify, out_path):
    all_rows = {collective: {} for collective in COLLECTIVES}

    for ctas in cta_values:
        for collective in COLLECTIVES:
            validate_sizes_for_collective(
                collective, sizes_bytes, len(devices)
            )
            print(
                f"[bench] metric={metric} collective={collective} "
                f"ctas={cta_log_label(ctas)} devices={devices} "
                f"sizes={sizes_bytes}"
            )
            rank_rows = run_rank_collective(
                collective, sizes_bytes, devices, iters, warmup, verify, ctas
            )
            all_rows[collective][ctas] = combine_rank_rows(
                collective, rank_rows
            )

    if metric == "speedup":
        for ctas in cta_values:
            one_cta_rows = {
                collective: {ctas: all_rows[collective][ctas]}
                for collective in COLLECTIVES
            }
            plot_metric(
                metric,
                one_cta_rows,
                name_with_cta_suffix(out_path, ctas),
                len(devices),
            )
    else:
        plot_metric(metric, all_rows, out_path, len(devices))


def plot_metric(metric, all_rows, out_path: Path, world_size: int):
    COLLECTIVE_TITLES = {
        "allreduce": "All-Reduce",
        "reduce_scatter": "Reduce-Scatter",
        "all_gather": "All-Gather",
    }
    
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 4), sharex=True)
    legend_handles, legend_labels, ylabel = None, None, ""

    cta_values = list(next(iter(all_rows.values())).keys())
    single_cta_setting = len(cta_values) == 1
    only_cta = cta_values[0] if single_cta_setting else None
    has_cta_limit = any(c is not None for c in cta_values)

    for ax, collective in zip(axes, COLLECTIVES):
        for ctas, rows in all_rows[collective].items():
            x = [r["bytes"] for r in rows]
            y_oo, y_nccl, y_nccl_symmetric, ylabel = metric_values(metric, rows)

            if metric == "speedup":
                ax.plot(x, y_oo, marker="o", label="OOverlap / normal NCCL")
                if y_nccl_symmetric is not None:
                    ax.plot(
                        x,
                        y_nccl_symmetric,
                        marker="^",
                        linestyle=":",
                        label="symmetric NCCL / normal NCCL",
                    )
            else:
                ax.plot(x, y_oo, marker="o", label=label_with_cta("OOverlap", ctas))
                ax.plot(x, y_nccl, marker="s", linestyle="--", label=label_with_cta("NCCL", ctas))
                if y_nccl_symmetric is not None:
                    ax.plot(
                        x,
                        y_nccl_symmetric,
                        marker="^",
                        linestyle=":",
                        label=label_with_cta("symmetric NCCL", ctas),
                    )

        if metric == "speedup":
            ax.axhline(1.0, color="gray", linestyle="--", linewidth=1.0, label="normal NCCL baseline")

        if legend_handles is None:
            legend_handles, legend_labels = ax.get_legend_handles_labels()

        xticks = [r["bytes"] for r in next(iter(all_rows[collective].values()))]
        ax.set_xscale("log", base=2)
        ax.set_xticks(xticks)
        ax.set_xticklabels([format_size_bytes(v) for v in xticks], rotation=30, ha="right")
        ax.set_title(COLLECTIVE_TITLES.get(collective, collective), fontsize=12)
        ax.grid(True, which="both", linestyle="--", alpha=0.35)

    def figure_title(metric: str, only_cta, single_cta_setting: bool, has_cta_limit: bool):
        metric_titles = {
            "bandwidth": f"{world_size}-GPU Collective Bandwidth",
            "latency": f"{world_size}-GPU Collective Latency",
            "speedup": f"{world_size}-GPU Collective Speedup Relative to NCCL",
        }
    
        base = metric_titles[metric]
    
        if single_cta_setting:
            if only_cta is None:
                return f"{base}\nNo CTA Limit"
            return f"{base}\nCTA Limit: {only_cta}"
    
        if has_cta_limit:
            return f"{base}\nUnder CTA Limits"
    
        return f"{base}\nNo CTA Limit"

    if metric == "speedup":
        title_y = 1.12
        legend_y = 1.02
        layout_top = 0.84
    else:
        title_y = 1.08
        legend_y = 1.00
        layout_top = 0.88

    fig.suptitle(
        figure_title(metric, only_cta, single_cta_setting, has_cta_limit),
        y=title_y,
        fontsize=14,
    )
    
    fig.supylabel(ylabel)
    fig.supxlabel("Buffer size")
    
    fig.legend(
        legend_handles,
        legend_labels,
        loc="upper center",
        ncol=min(len(legend_labels), 6),
        bbox_to_anchor=(0.5, legend_y),
        frameon=False,
    )
    
    fig.tight_layout(rect=(0.02, 0.02, 1.0, layout_top))
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    print(f"[plot] wrote {out_path}")

def main():
    parser = argparse.ArgumentParser("IPC collective smoke + benchmark plotter")
    parser.add_argument("--mode", choices=["smoke", "bench", "both"], default="both")
    parser.add_argument(
        "--devices",
        default=None,
        help="comma-separated device ids, for example 0,1,2,3",
    )
    parser.add_argument("--dev0", type=int, default=0)
    parser.add_argument("--dev1", type=int, default=1)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--smoke-bytes", default="1M")
    parser.add_argument(
        "--metric",
        choices=["latency", "bandwidth", "speedup", "both", "all"],
        default="bandwidth",
    )
    parser.add_argument("--bytes", default="1M,2M,4M,8M,16M,32M,64M,128M,256M")
    parser.add_argument("--latency-bytes", default=None)
    parser.add_argument("--bandwidth-bytes", default=None)
    parser.add_argument("--speedup-bytes", default=None)
    parser.add_argument(
        "--ctas",
        default=None,
        help="comma-separated communication CTA counts, e.g. 1,2,4,8",
    )
    parser.add_argument("--out-prefix", default="ipc_collective")

    args = parser.parse_args()
    devices = (
        parse_devices(args.devices)
        if args.devices is not None
        else parse_devices(f"{args.dev0},{args.dev1}")
    )

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    if max(devices) >= torch.cuda.device_count():
        raise ValueError(
            f"requested devices {devices}, but CUDA device count is "
            f"{torch.cuda.device_count()}"
        )
    if args.iters <= 0:
        raise ValueError("--iters must be > 0")
    if args.warmup < 0:
        raise ValueError("--warmup must be >= 0")

    print(f"[info] torch={torch.__version__}")
    print(f"[info] cuda_device_count={torch.cuda.device_count()}")
    print(f"[info] devices={devices} world_size={len(devices)}")
    print(f"[info] iters={args.iters} warmup={args.warmup} ctas={args.ctas or 'default'}")

    if args.mode in ("smoke", "both"):
        run_smoke(parse_size_one(args.smoke_bytes), devices)

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
                devices=devices,
                iters=args.iters,
                warmup=args.warmup,
                verify=args.verify,
                out_path=Path(f"{args.out_prefix}_{metric}.png"),
            )

    for device in devices:
        torch.cuda.synchronize(device)
    print("PASS")


if __name__ == "__main__":
    main()
