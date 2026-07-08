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


def print_speedup_line(label: str, baseline_label: str, baseline: float, value: float):
    speedup = baseline / value
    pct = (speedup - 1.0) * 100.0

    if pct >= 0.0:
        print(f"  {label}_speedup_over_{baseline_label}: {speedup:.6f}x (+{pct:.2f}%)")
    else:
        print(f"  {label}_speedup_over_{baseline_label}: {speedup:.6f}x ({pct:.2f}%)")


def print_metrics(title, metrics):
    print(f"\n[{title}]")

    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.6f}")
        else:
            print(f"  {k}: {v}")

    nccl_ms = metrics.get("nccl_ms", None)

    if isinstance(nccl_ms, float) and nccl_ms > 0.0:
        print("\n[Speedup over NCCL]")

        ooverlap_ms = metrics.get("ooverlap_ms", None)
        if isinstance(ooverlap_ms, float) and ooverlap_ms > 0.0:
            print_speedup_line("ooverlap", "nccl", nccl_ms, ooverlap_ms)

        nccl_registered_ms = metrics.get("nccl_registered_ms", None)
        if isinstance(nccl_registered_ms, float) and nccl_registered_ms > 0.0:
            print_speedup_line(
                "nccl_registered",
                "nccl",
                nccl_ms,
                nccl_registered_ms,
            )

    nccl_registered_ms = metrics.get("nccl_registered_ms", None)
    ooverlap_ms = metrics.get("ooverlap_ms", None)

    if (isinstance(nccl_registered_ms, float) and nccl_registered_ms > 0.0 and
            isinstance(ooverlap_ms, float) and ooverlap_ms > 0.0):
        print("\n[Speedup over registered NCCL]")
        print_speedup_line(
            "ooverlap",
            "nccl_registered",
            nccl_registered_ms,
            ooverlap_ms,
        )


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


def parse_args():
    parser = argparse.ArgumentParser(
        description="External-buffer two-GPU collective test/benchmark"
    )

    parser.add_argument(
        "--mode",
        choices=("smoke", "bench", "both"),
        default="both",
    )
    parser.add_argument(
        "--collective",
        choices=VALID_COLLECTIVES,
        default="allreduce",
    )
    parser.add_argument(
        "--numel",
        type=int,
        default=1 << 20,
    )
    parser.add_argument(
        "--iters",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=20,
    )
    parser.add_argument(
        "--dev0",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--dev1",
        type=int,
        default=1,
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if args.numel <= 0:
        raise ValueError("--numel must be > 0")

    if args.iters <= 0:
        raise ValueError("--iters must be > 0")

    if args.warmup < 0:
        raise ValueError("--warmup must be >= 0")

    if args.dev0 == args.dev1:
        raise ValueError("--dev0 and --dev1 must differ")

    display_collective = collective_display_name(args.collective)

    print(f"[info] torch {torch.__version__}")
    print(f"[info] cuda available: {torch.cuda.is_available()}")
    print(f"[info] cuda device count: {torch.cuda.device_count()}")
    print(f"[info] collective={args.collective} ({display_collective})")
    print("[info] memory mode=external cudaMalloc buffers + oo_buffer_wrap + oo_group_create_p2p")
    print("[info] NCCL modes=normal cudaMalloc, plus optional ncclMemAlloc + ncclCommRegister")
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
            f"External-buffer P2P 2-GPU {display_collective}: ooverlap vs NCCL",
            metrics,
        )

    torch.cuda.synchronize(args.dev0)
    torch.cuda.synchronize(args.dev1)

    print(f"PASS ✅ external-buffer P2P 2-GPU {display_collective} path")


if __name__ == "__main__":
    main()
