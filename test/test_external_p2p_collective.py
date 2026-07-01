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


def print_metrics(title, metrics):
    print(f"\n[{title}]")

    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.6f}")
        else:
            print(f"  {k}: {v}")

    nccl_ms = metrics.get("nccl_ms", None)

    if not isinstance(nccl_ms, float) or nccl_ms <= 0.0:
        return

    print("\n[Speedup over NCCL]")

    speedup_keys = [
        ("normal_ms", "normal"),
        ("efficiency_ms", "efficiency"),
    ]

    for metric_key, label in speedup_keys:
        value = metrics.get(metric_key, None)

        if not isinstance(value, float) or value <= 0.0:
            continue

        speedup = nccl_ms / value
        pct = (speedup - 1.0) * 100.0

        if pct >= 0.0:
            print(f"  {label}_speedup_over_nccl: {speedup:.6f}x (+{pct:.2f}%)")
        else:
            print(f"  {label}_speedup_over_nccl: {speedup:.6f}x ({pct:.2f}%)")


def collective_display_name(collective: str) -> str:
    if collective == "allreduce":
        return "all-reduce"
    if collective == "reduce_scatter":
        return "reduce-scatter"
    if collective == "all_gather":
        return "all-gather"
    return collective


def run_smoke(ext, collective: str, numel: int, dev0: int, dev1: int) -> bool:
    if hasattr(ext, "external_p2p_two_gpu_collective_smoke_test"):
        return ext.external_p2p_two_gpu_collective_smoke_test(
            collective,
            int(numel),
            int(dev0),
            int(dev1),
        )

    if collective == "allreduce" and hasattr(ext, "external_p2p_two_gpu_allreduce_smoke_test"):
        return ext.external_p2p_two_gpu_allreduce_smoke_test(
            int(numel),
            int(dev0),
            int(dev1),
        )

    raise AttributeError(
        "Extension does not expose external_p2p_two_gpu_collective_smoke_test. "
        "Rebuild after adding the pybind binding and adding the external P2P test source to CMake."
    )


def run_benchmark(
    ext,
    collective: str,
    numel: int,
    iters: int,
    warmup: int,
    dev0: int,
    dev1: int,
):
    if hasattr(ext, "benchmark_external_p2p_two_gpu_collective_sm90"):
        return ext.benchmark_external_p2p_two_gpu_collective_sm90(
            collective,
            int(numel),
            int(iters),
            int(warmup),
            int(dev0),
            int(dev1),
        )

    if collective == "allreduce" and hasattr(ext, "benchmark_external_p2p_two_gpu_allreduce_sm90"):
        return ext.benchmark_external_p2p_two_gpu_allreduce_sm90(
            int(numel),
            int(iters),
            int(warmup),
            int(dev0),
            int(dev1),
        )

    raise AttributeError(
        "Extension does not expose benchmark_external_p2p_two_gpu_collective_sm90. "
        "Rebuild after adding the pybind binding and adding the external P2P test source to CMake."
    )


def main():
    parser = argparse.ArgumentParser(
        "External-buffer P2P 2-GPU collective smoke + benchmark"
    )

    parser.add_argument(
        "--collective",
        choices=VALID_COLLECTIVES,
        default="allreduce",
        help="Collective to benchmark: allreduce, reduce_scatter, or all_gather.",
    )

    parser.add_argument(
        "--mode",
        choices=["smoke", "bench", "both"],
        default="both",
    )

    parser.add_argument("--numel", type=int, default=1 << 20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--dev0", type=int, default=0)
    parser.add_argument("--dev1", type=int, default=1)

    args = parser.parse_args()

    assert torch.cuda.is_available(), "torch.cuda.is_available() is False"
    assert torch.cuda.device_count() >= 2, "Need at least 2 GPUs"

    if args.dev0 == args.dev1:
        raise ValueError("--dev0 and --dev1 must be different")

    if args.dev0 < 0 or args.dev0 >= torch.cuda.device_count():
        raise ValueError(f"--dev0={args.dev0} is outside available CUDA devices")

    if args.dev1 < 0 or args.dev1 >= torch.cuda.device_count():
        raise ValueError(f"--dev1={args.dev1} is outside available CUDA devices")

    if args.numel <= 0:
        raise ValueError("--numel must be > 0")

    if args.iters <= 0:
        raise ValueError("--iters must be > 0")

    if args.warmup < 0:
        raise ValueError("--warmup must be >= 0")

    if args.collective in ("reduce_scatter", "all_gather") and (args.numel % 2) != 0:
        raise ValueError(
            f"--numel must be divisible by 2 for {args.collective} "
            "because the NCCL comparison uses equal 2-GPU shards"
        )

    display_collective = collective_display_name(args.collective)

    print(f"[info] torch {torch.__version__}")
    print(f"[info] cuda available: {torch.cuda.is_available()}")
    print(f"[info] cuda device count: {torch.cuda.device_count()}")
    print(f"[info] collective={args.collective} ({display_collective})")
    print("[info] memory mode=external cudaMalloc buffers + oo_buffer_wrap + oo_group_create_p2p")
    print("[info] NCCL mode=normal ncclCommInitAll + cudaMalloc buffers; no ncclMemAlloc/window register")
    print(f"[info] dev0={args.dev0} dev1={args.dev1}")
    print(f"[info] numel={args.numel} iters={args.iters} warmup={args.warmup}")

    ext = load_ooverlap_ext()
    print("[info] loaded extension:", ext)

    if args.mode in ("smoke", "both"):
        ok = run_smoke(
            ext,
            args.collective,
            args.numel,
            args.dev0,
            args.dev1,
        )

        print(f"[result] external P2P {display_collective} smoke: {ok}")
        assert ok is True

    if args.mode in ("bench", "both"):
        metrics = run_benchmark(
            ext,
            args.collective,
            args.numel,
            args.iters,
            args.warmup,
            args.dev0,
            args.dev1,
        )

        print_metrics(
            f"External-buffer P2P 2-GPU {display_collective} vs NCCL",
            metrics,
        )

    torch.cuda.synchronize(args.dev0)
    torch.cuda.synchronize(args.dev1)

    print(f"PASS ✅ external-buffer P2P 2-GPU {display_collective} path")


if __name__ == "__main__":
    main()
