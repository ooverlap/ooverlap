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


def print_metrics(title, metrics):
    print(f"\n[{title}]")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.6f}")
        else:
            print(f"  {k}: {v}")


def main():
    parser = argparse.ArgumentParser("Persistent 2-GPU all-reduce smoke + benchmark")
    parser.add_argument("--mode", choices=["smoke", "bench", "both"], default="both")
    parser.add_argument("--numel", type=int, default=1 << 20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--dev0", type=int, default=0)
    parser.add_argument("--dev1", type=int, default=1)
    args = parser.parse_args()

    assert torch.cuda.is_available(), "torch.cuda.is_available() is False"
    assert torch.cuda.device_count() >= 2, "Need at least 2 GPUs"

    print(f"[info] torch {torch.__version__}")
    print(f"[info] cuda available: {torch.cuda.is_available()}")
    print(f"[info] cuda device count: {torch.cuda.device_count()}")
    print(f"[info] dev0={args.dev0} dev1={args.dev1}")
    print(f"[info] numel={args.numel} iters={args.iters} warmup={args.warmup}")

    ext = load_ooverlap_ext()
    print("[info] loaded extension:", ext)

    if args.mode in ("smoke", "both"):
        ok = ext.tma_persistent_two_gpu_allreduce_smoke_test(
            int(args.numel), int(args.dev0), int(args.dev1)
        )
        print(f"[result] persistent smoke: {ok}")
        assert ok is True

    if args.mode in ("bench", "both"):
        metrics = ext.benchmark_persistent_two_gpu_allreduce_sm90(
            int(args.numel),
            int(args.iters),
            int(args.warmup),
            int(args.dev0),
            int(args.dev1),
        )
        print_metrics("Persistent 2-GPU all-reduce vs basic vs NCCL", metrics)

    torch.cuda.synchronize(args.dev0)
    torch.cuda.synchronize(args.dev1)
    print("PASS ✅ persistent 2-GPU all-reduce path")


if __name__ == "__main__":
    main()
