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


def main():
    parser = argparse.ArgumentParser("Basic same-process N-GPU collectives above bulk-TMA")
    parser.add_argument(
        "--op",
        choices=["reducescatter", "allgather", "allreduce", "all"],
        default="all",
    )
    parser.add_argument(
        "--devices",
        type=str,
        default="",
        help="comma-separated CUDA device ids, e.g. 0,1,2,3 ; empty means all visible GPUs",
    )
    parser.add_argument(
        "--full-numel",
        type=int,
        default=1 << 20,
        help="full_numel for reduce-scatter and all-reduce",
    )
    parser.add_argument(
        "--shard-numel",
        type=int,
        default=1 << 18,
        help="shard_numel for all-gather",
    )
    args = parser.parse_args()

    assert torch.cuda.is_available(), "torch.cuda.is_available() is False"
    ndev = torch.cuda.device_count()
    assert ndev >= 2, f"Need at least 2 GPUs, found {ndev}"

    devices = parse_devices(args.devices)
    if not devices:
        devices = list(range(ndev))

    assert len(devices) >= 2, "Need at least 2 devices"
    assert all(0 <= d < ndev for d in devices), f"Invalid device list: {devices}"

    print(f"[info] torch {torch.__version__}")
    print(f"[info] cuda available: {torch.cuda.is_available()}")
    print(f"[info] cuda device count: {ndev}")
    print(f"[info] devices: {devices}")
    print(f"[info] full_numel={args.full_numel} shard_numel={args.shard_numel}")

    ext = load_ooverlap_ext()
    print("[info] loaded extension:", ext)

    if args.op in ("reducescatter", "all"):
        assert args.full_numel % len(devices) == 0, \
            f"full_numel={args.full_numel} must be divisible by world_size={len(devices)}"
        print("[run] tma_basic_ngpu_reduce_scatter_smoke_test")
        ok = ext.tma_basic_ngpu_reduce_scatter_smoke_test(
            int(args.full_numel), devices
        )
        print("[result] reduce-scatter:", ok)
        assert ok is True

    if args.op in ("allgather", "all"):
        print("[run] tma_basic_ngpu_all_gather_smoke_test")
        ok = ext.tma_basic_ngpu_all_gather_smoke_test(
            int(args.shard_numel), devices
        )
        print("[result] all-gather:", ok)
        assert ok is True

    if args.op in ("allreduce", "all"):
        assert args.full_numel % len(devices) == 0, \
            f"full_numel={args.full_numel} must be divisible by world_size={len(devices)}"
        print("[run] tma_basic_ngpu_all_reduce_smoke_test")
        ok = ext.tma_basic_ngpu_all_reduce_smoke_test(
            int(args.full_numel), devices
        )
        print("[result] all-reduce:", ok)
        assert ok is True

    for d in devices:
        torch.cuda.synchronize(d)

    print("PASS ✅  basic same-process N-GPU collectives above bulk-TMA")


if __name__ == "__main__":
    main()
