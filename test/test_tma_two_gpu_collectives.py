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


def main():
    parser = argparse.ArgumentParser("2-GPU collectives above bulk-TMA")
    parser.add_argument(
        "--op",
        choices=["allreduce", "allgather", "both"],
        default="both",
        help="which smoke test to run",
    )
    parser.add_argument(
        "--numel",
        type=int,
        default=1 << 20,
        help="all-reduce numel or all-gather shard_numel",
    )
    parser.add_argument("--dev0", type=int, default=0)
    parser.add_argument("--dev1", type=int, default=1)
    args = parser.parse_args()

    assert torch.cuda.is_available(), "torch.cuda.is_available() is False"
    ndev = torch.cuda.device_count()
    assert ndev >= 2, f"Need at least 2 GPUs, found {ndev}"

    print(f"[info] torch {torch.__version__}")
    print(f"[info] cuda available: {torch.cuda.is_available()}")
    print(f"[info] cuda device count: {ndev}")
    print(f"[info] dev0={args.dev0} dev1={args.dev1} numel={args.numel}")

    ext = load_ooverlap_ext()
    print("[info] loaded extension:", ext)

    if args.op in ("allreduce", "both"):
        print("[run] tma_two_gpu_all_reduce_smoke_test")
        ok = ext.tma_two_gpu_all_reduce_smoke_test(
            int(args.numel), int(args.dev0), int(args.dev1)
        )
        print("[result] all-reduce:", ok)
        assert ok is True

    if args.op in ("allgather", "both"):
        print("[run] tma_two_gpu_all_gather_smoke_test")
        ok = ext.tma_two_gpu_all_gather_smoke_test(
            int(args.numel), int(args.dev0), int(args.dev1)
        )
        print("[result] all-gather:", ok)
        assert ok is True

    torch.cuda.synchronize(args.dev0)
    torch.cuda.synchronize(args.dev1)
    print("PASS ✅  2-GPU collectives above bulk-TMA")


if __name__ == "__main__":
    main()
