import argparse
import importlib.util
from pathlib import Path

import torch


def load_ooverlap_ext():
    root = Path(__file__).resolve().parents[1]
    so = root / "build" / "lib" / "ooverlap_ext.so"
    if not so.exists():
        raise FileNotFoundError(f"Could not find {so}. Build first.")
    spec = importlib.util.spec_from_file_location("ooverlap_ext", str(so))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def parse_devices(devices_str: str):
    if devices_str.strip() == "":
        return []
    return [int(x) for x in devices_str.split(",") if x.strip() != ""]


def print_metrics(title, metrics):
    print(f"\n[{title}]")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.6f}")
        else:
            print(f"  {k}: {v}")


def main():
    parser = argparse.ArgumentParser("Benchmark bulk-TMA copy and basic N-GPU collectives")
    parser.add_argument("--mode", choices=["copy", "collective", "all"], default="all")
    parser.add_argument("--devices", type=str, default="",
                        help="comma-separated CUDA device ids, e.g. 0,1,2,3")
    parser.add_argument("--numel", type=int, default=1 << 20,
                        help="numel for copy and allreduce/reducescatter")
    parser.add_argument("--shard-numel", type=int, default=1 << 18,
                        help="shard numel for allgather")
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=20)
    args = parser.parse_args()

    assert torch.cuda.is_available(), "torch.cuda.is_available() is False"
    ndev = torch.cuda.device_count()
    assert ndev >= 2, f"Need at least 2 GPUs, found {ndev}"

    devices = parse_devices(args.devices)
    if not devices:
        devices = list(range(ndev))

    print(f"[info] torch {torch.__version__}")
    print(f"[info] cuda available: {torch.cuda.is_available()}")
    print(f"[info] cuda device count: {ndev}")
    print(f"[info] devices: {devices}")
    print(f"[info] iters={args.iters} warmup={args.warmup}")
    print(f"[info] numel={args.numel} shard_numel={args.shard_numel}")

    ext = load_ooverlap_ext()
    print("[info] loaded extension:", ext)

    if args.mode in ("copy", "all"):
        assert len(devices) >= 2, "copy benchmark needs at least 2 devices"
        metrics = ext.benchmark_2gpu_copy_sm90(
            int(args.numel),
            int(args.iters),
            int(args.warmup),
            int(devices[0]),
            int(devices[1]),
        )
        print_metrics("2-GPU copy benchmark", metrics)

    if args.mode in ("collective", "all"):
        if args.numel % len(devices) != 0:
            raise ValueError(
                f"numel={args.numel} must be divisible by world_size={len(devices)} "
                f"for reduce-scatter/all-reduce"
            )

        metrics = ext.benchmark_basic_ngpu_collective_sm90(
            "reducescatter",
            int(args.numel),
            devices,
            int(args.iters),
            int(args.warmup),
        )
        print_metrics("Basic N-GPU reduce-scatter vs NCCL", metrics)

        metrics = ext.benchmark_basic_ngpu_collective_sm90(
            "allgather",
            int(args.shard_numel),
            devices,
            int(args.iters),
            int(args.warmup),
        )
        print_metrics("Basic N-GPU all-gather vs NCCL", metrics)

        metrics = ext.benchmark_basic_ngpu_collective_sm90(
            "allreduce",
            int(args.numel),
            devices,
            int(args.iters),
            int(args.warmup),
        )
        print_metrics("Basic N-GPU all-reduce vs NCCL", metrics)

    for d in devices:
        torch.cuda.synchronize(d)


if __name__ == "__main__":
    main()
