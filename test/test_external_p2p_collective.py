import argparse
import importlib.util
from pathlib import Path

import torch


VALID_COLLECTIVES = ("allreduce", "reduce_scatter", "all_gather")


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


def parse_devices(value: str) -> list[int]:
    devices = [int(item.strip()) for item in value.split(",") if item.strip()]
    if len(devices) < 2:
        raise ValueError("--devices must contain at least two device ids")
    if len(set(devices)) != len(devices):
        raise ValueError("--devices must not contain duplicates")
    if any(device < 0 for device in devices):
        raise ValueError("--devices must contain non-negative device ids")
    return devices


def print_speedup_line(label: str, baseline_label: str, baseline: float, value: float):
    speedup = baseline / value
    pct = (speedup - 1.0) * 100.0
    sign = "+" if pct >= 0.0 else ""
    print(f"  {label}_speedup_over_{baseline_label}: {speedup:.6f}x ({sign}{pct:.2f}%)")


def print_metrics(title, metrics):
    print(f"\n[{title}]")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.6f}")
        else:
            print(f"  {key}: {value}")

    nccl_ms = metrics.get("nccl_ms")
    nccl_symmetric_ms = metrics.get("nccl_symmetric_ms")
    ooverlap_ms = metrics.get("ooverlap_ms")

    if isinstance(nccl_ms, float) and nccl_ms > 0.0:
        print("\n[Speedup over normal NCCL]")
        if isinstance(ooverlap_ms, float) and ooverlap_ms > 0.0:
            print_speedup_line("ooverlap", "nccl", nccl_ms, ooverlap_ms)
        if isinstance(nccl_symmetric_ms, float) and nccl_symmetric_ms > 0.0:
            print_speedup_line("nccl_symmetric", "nccl", nccl_ms, nccl_symmetric_ms)

    if (
        isinstance(nccl_symmetric_ms, float)
        and nccl_symmetric_ms > 0.0
        and isinstance(ooverlap_ms, float)
        and ooverlap_ms > 0.0
    ):
        print("\n[Speedup over symmetric NCCL]")
        print_speedup_line("ooverlap", "nccl_symmetric", nccl_symmetric_ms, ooverlap_ms)


def collective_display_name(collective: str) -> str:
    return {
        "allreduce": "all-reduce",
        "reduce_scatter": "reduce-scatter",
        "all_gather": "all-gather",
    }.get(collective, collective)


def run_smoke(ext, collective: str, numel: int, devices: list[int]) -> bool:
    if hasattr(ext, "external_p2p_collective_smoke_test"):
        return ext.external_p2p_collective_smoke_test(
            collective,
            int(numel),
            devices,
        )

    if len(devices) == 2 and hasattr(ext, "external_p2p_two_gpu_collective_smoke_test"):
        return ext.external_p2p_two_gpu_collective_smoke_test(
            collective,
            int(numel),
            int(devices[0]),
            int(devices[1]),
        )

    raise AttributeError(
        "Extension does not expose external_p2p_collective_smoke_test. Rebuild ooverlap_ext."
    )


def run_benchmark(
    ext,
    collective: str,
    numel: int,
    iters: int,
    warmup: int,
    devices: list[int],
):
    if hasattr(ext, "benchmark_external_p2p_collective_sm90"):
        return ext.benchmark_external_p2p_collective_sm90(
            collective,
            int(numel),
            int(iters),
            int(warmup),
            devices,
        )

    if len(devices) == 2 and hasattr(ext, "benchmark_external_p2p_two_gpu_collective_sm90"):
        return ext.benchmark_external_p2p_two_gpu_collective_sm90(
            collective,
            int(numel),
            int(iters),
            int(warmup),
            int(devices[0]),
            int(devices[1]),
        )

    raise AttributeError(
        "Extension does not expose benchmark_external_p2p_collective_sm90. Rebuild ooverlap_ext."
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="External-buffer multi-GPU collective test/benchmark"
    )
    parser.add_argument("--mode", choices=("smoke", "bench", "both"), default="both")
    parser.add_argument("--collective", choices=VALID_COLLECTIVES, default="allreduce")
    parser.add_argument("--numel", type=int, default=1 << 20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument(
        "--devices",
        default=None,
        help="comma-separated device ids, for example 0,1,2,3",
    )
    parser.add_argument("--dev0", type=int, default=0)
    parser.add_argument("--dev1", type=int, default=1)
    return parser.parse_args()


def main():
    args = parse_args()
    devices = (
        parse_devices(args.devices)
        if args.devices is not None
        else [args.dev0, args.dev1]
    )
    if len(set(devices)) != len(devices):
        raise ValueError("device ids must be unique")

    if args.numel <= 0:
        raise ValueError("--numel must be > 0")
    if args.iters <= 0:
        raise ValueError("--iters must be > 0")
    if args.warmup < 0:
        raise ValueError("--warmup must be >= 0")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    if max(devices) >= torch.cuda.device_count():
        raise ValueError(
            f"requested devices {devices}, but CUDA device count is {torch.cuda.device_count()}"
        )

    display_collective = collective_display_name(args.collective)

    print(f"[info] torch {torch.__version__}")
    print(f"[info] cuda device count: {torch.cuda.device_count()}")
    print(f"[info] collective={args.collective} ({display_collective})")
    print("[info] memory mode=external cudaMalloc buffers + oo_buffer_wrap + oo_group_create_p2p")
    print("[info] NCCL modes=normal cudaMalloc, symmetric ncclMemAlloc + ncclCommWindowRegister")
    print(f"[info] devices={devices} world_size={len(devices)}")
    print(f"[info] numel={args.numel} iters={args.iters} warmup={args.warmup}")

    ext = load_ooverlap_ext()
    print("[info] loaded extension:", ext)

    if args.mode in ("smoke", "both"):
        ok = run_smoke(ext, args.collective, args.numel, devices)
        print(f"[result] external P2P {display_collective} smoke: {ok}")
        assert ok is True

    if args.mode in ("bench", "both"):
        metrics = run_benchmark(
            ext,
            args.collective,
            args.numel,
            args.iters,
            args.warmup,
            devices,
        )
        print_metrics(
            f"External-buffer P2P {len(devices)}-GPU {display_collective}: ooverlap vs NCCL",
            metrics,
        )

    for device in devices:
        torch.cuda.synchronize(device)

    print(f"PASS external-buffer P2P {len(devices)}-GPU {display_collective} path")


if __name__ == "__main__":
    main()
