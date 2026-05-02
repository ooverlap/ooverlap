import argparse
import importlib.util
import math
import os
import sys
from pathlib import Path
from time import sleep

import torch
import torch.multiprocessing as mp
import matplotlib.pyplot as plt


def repo_root():
    return Path(__file__).resolve().parents[1]


def load_ooverlap_ext():
    so = repo_root() / "build" / "lib" / "ooverlap_ext.so"
    if not so.exists():
        raise FileNotFoundError(f"Could not find {so}. Build first.")

    module_name = "ooverlap_ext"
    if module_name in sys.modules:
        return sys.modules[module_name]

    spec = importlib.util.spec_from_file_location(module_name, str(so))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


def make_sizes():
    # Same style as FlashOverlap: element counts, not bytes.
    return [(int(2 ** (20 + 0.25 * i)) // 1024 * 1024) for i in range(36)]


def p90_from_tensor(x):
    # Nearest-rank p90: sort samples and keep the 90th-percentile latency.
    # This is intentionally simple and deterministic.
    x_sorted = torch.sort(x.flatten()).values
    n = int(x_sorted.numel())
    idx = int(math.ceil(0.90 * n)) - 1
    idx = max(0, min(idx, n - 1))
    return float(x_sorted[idx].item())


def comm_backend_label(comm_backend: str) -> str:
    if comm_backend == "ooverlap":
        return "ooverlap"
    if comm_backend == "nccl":
        return "nccl"
    raise ValueError(f"Unsupported comm_backend={comm_backend}")


def call_comm(comm_class, comm_backend: str, comm_op: str, C, D=None):
    if comm_backend == "nccl":
        if comm_op == "all_reduce":
            comm_class.nccl_allreduce(C)
        elif comm_op == "reduce_scatter":
            if D is None:
                raise RuntimeError("D must be provided for NCCL reduce_scatter")
            comm_class.nccl_reducescatter(C, D)
        else:
            raise ValueError(f"Unsupported comm_op={comm_op}")
    elif comm_backend == "ooverlap":
        if comm_op != "all_reduce":
            raise ValueError("ooverlap backend currently supports only all_reduce")
        comm_class.ooverlap_allreduce(C)
    else:
        raise ValueError(f"Unsupported comm_backend={comm_backend}")


def perf_comm_process(
    rank,
    world_size,
    nccl_id,
    broker_key,
    comm_backend,
    comm_op,
    sizes,
    warmup,
    iters,
    sleep_seconds,
    barrier,
    result_dict,
):
    torch.cuda.set_device(rank)

    ext = load_ooverlap_ext()

    comm_class = ext.OverlapImpl()
    comm_class.cutlass_init()

    if comm_backend == "nccl":
        comm_class.nccl_init(rank, world_size, nccl_id)
    elif comm_backend == "ooverlap":
        if world_size != 2:
            raise RuntimeError("ooverlap backend currently supports exactly 2 GPUs/ranks")
        # Device ids are local CUDA-visible ids. With CUDA_VISIBLE_DEVICES=0,1 this is [0, 1].
        comm_class.ooverlap_ipc_init(rank, world_size, list(range(world_size)), broker_key)
    else:
        raise ValueError(f"Unsupported comm_backend={comm_backend}")

    torch.cuda.synchronize()
    barrier.wait()

    rank_results = []
    if rank == 0:
        print(f"sizes={sizes}")

    for size in sizes:
        M = 1024
        N = size // 1024

        C = torch.empty((M, N), dtype=torch.float16, device="cuda").normal_(
            mean=0.0,
            std=0.5,
        )

        D = None
        if comm_backend == "nccl" and comm_op == "reduce_scatter":
            if C.numel() % world_size != 0:
                raise RuntimeError(
                    f"C.numel()={C.numel()} must be divisible by world_size={world_size}"
                )
            # OverlapImpl::NcclReduceScatter expects D.numel() == C.numel()/world_size.
            D = torch.empty((C.numel() // world_size,), dtype=torch.float16, device="cuda")

        # Let the link rest between points, matching the old script behavior.
        if sleep_seconds > 0:
            sleep(sleep_seconds)

        torch.cuda.synchronize()
        barrier.wait()

        for _ in range(warmup):
            call_comm(comm_class, comm_backend, comm_op, C, D)

        torch.cuda.synchronize()
        barrier.wait()

        start_event = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        end_event = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]

        for i in range(iters):
            start_event[i].record()
            call_comm(comm_class, comm_backend, comm_op, C, D)
            end_event[i].record()

        torch.cuda.synchronize()
        barrier.wait()

        dur_ms = torch.tensor(
            [s.elapsed_time(e) for s, e in zip(start_event, end_event)],
            dtype=torch.float32,
        )

        print(
            f"rank={rank} "
            f"backend={comm_backend} "
            f"comm_op={comm_op} "
            f"size={size} "
            f"dur_ms={dur_ms}"
        )

        rank_results.append(
            {
                "size": int(size),
                "bytes": int(size * 2),
                "mean_ms": float(torch.mean(dur_ms).item()),
                "median_ms": float(torch.median(dur_ms).item()),
                "p90_ms": p90_from_tensor(dur_ms),
                "min_ms": float(torch.min(dur_ms).item()),
                "max_ms": float(torch.max(dur_ms).item()),
            }
        )

        del C
        if D is not None:
            del D

        torch.cuda.synchronize()
        barrier.wait()

    if comm_backend == "ooverlap":
        torch.cuda.synchronize()
        barrier.wait()
        comm_class.ooverlap_release()
        torch.cuda.synchronize()
        barrier.wait()

    result_dict[rank] = rank_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--comm_backend",
        type=str,
        default="ooverlap",
        choices=["ooverlap", "nccl"],
        help="Communication backend to benchmark. This file defaults to ooverlap.",
    )
    parser.add_argument(
        "--comm_op",
        type=str,
        default="all_reduce",
        choices=["all_reduce", "reduce_scatter"],
    )
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--sleep-seconds", type=float, default=5.0)
    parser.add_argument("--use-median", action="store_true")
    parser.add_argument("--use-p90", action="store_true")
    args = parser.parse_args()

    if args.comm_backend == "ooverlap" and args.comm_op != "all_reduce":
        raise RuntimeError("ooverlap backend currently supports only --comm_op all_reduce")

    world_size = torch.cuda.device_count()
    if world_size < 2:
        raise RuntimeError("At least 2 GPUs are required.")

    if args.comm_backend == "ooverlap" and world_size != 2:
        raise RuntimeError("ooverlap backend currently supports exactly 2 GPUs/ranks")

    sizes = make_sizes()

    ext = load_ooverlap_ext()
    nccl_id = ext.generate_nccl_id() if args.comm_backend == "nccl" else []
    broker_key = f"ooverlap_bw_{os.getpid()}_{int(torch.empty((), device='cpu').new_empty(()).random_().item())}"

    manager = mp.Manager()
    result_dict = manager.dict()
    barrier = manager.Barrier(world_size)

    mp.spawn(
        perf_comm_process,
        args=(
            world_size,
            nccl_id,
            broker_key,
            args.comm_backend,
            args.comm_op,
            sizes,
            args.warmup,
            args.iters,
            args.sleep_seconds,
            barrier,
            result_dict,
        ),
        nprocs=world_size,
        join=True,
    )

    backend = comm_backend_label(args.comm_backend)

    bandwidths = []
    comm_array = torch.zeros((len(sizes), 2), dtype=torch.float32)

    print("========================================")
    print("bandwidth_ooverlap.py")
    print(f"comm_backend: {backend}")
    print(f"comm_op:      {args.comm_op}")
    print(f"gpus:         {world_size}")
    print("dtype:        fp16")
    print(f"warmup:       {args.warmup}")
    print(f"iters:        {args.iters}")
    print(f"sleep_sec:    {args.sleep_seconds}")
    print(f"use_median:   {args.use_median}")
    print(f"use_p90:      {args.use_p90}")
    print("========================================")

    for i, size in enumerate(sizes):
        per_rank = [result_dict[r][i] for r in range(world_size)]

        mean_ms = max(x["mean_ms"] for x in per_rank)
        median_ms = max(x["median_ms"] for x in per_rank)
        p90_ms = max(x["p90_ms"] for x in per_rank)
        min_ms = max(x["min_ms"] for x in per_rank)
        max_ms = max(x["max_ms"] for x in per_rank)

        if args.use_p90:
            latency_ms = p90_ms
        elif args.use_median:
            latency_ms = median_ms
        else:
            latency_ms = mean_ms

        latency_s = latency_ms * 1.0e-3
        data_size_bytes = size * 2  # fp16

        if args.comm_op == "all_reduce":
            total_data_transferred = data_size_bytes * 2 * (world_size - 1)
        elif args.comm_op == "reduce_scatter":
            total_data_transferred = data_size_bytes * (world_size - 1)
        else:
            raise ValueError(f"Unsupported comm_op={args.comm_op}")

        flash_bw = total_data_transferred / latency_s / (1024 ** 3)
        alg_bw = data_size_bytes / latency_s / (1024 ** 3)

        bandwidths.append(flash_bw)
        comm_array[i, 0] = float(size)
        comm_array[i, 1] = float(flash_bw)

        print(
            f"backend={backend:9s} "
            f"op={args.comm_op:14s} "
            f"size={size:12d} elems "
            f"bytes={data_size_bytes:12d} "
            f"mean={mean_ms:9.5f} ms "
            f"median={median_ms:9.5f} ms "
            f"p90={p90_ms:9.5f} ms "
            f"min={min_ms:9.5f} ms "
            f"max={max_ms:9.5f} ms "
            f"flash_bw={flash_bw:9.2f} GB/s "
            f"alg_bw={alg_bw:9.2f} GB/s"
        )

    out_dir = repo_root() / "configs"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_pt = out_dir / f"bandwidth_{backend}_{args.comm_op}_tp{world_size}.pt"
    torch.save(comm_array, out_pt)

    out_png = out_dir / f"bandwidth_{backend}_{args.comm_op}_tp{world_size}.png"
    plt.figure()
    plt.plot(sizes, bandwidths, marker="o")
    plt.xlabel("Data Size (elements)")
    plt.ylabel("Bandwidth (GB/s)")
    plt.title(f"Bandwidth vs Data Size ({backend}, {args.comm_op}, tp={world_size})")
    plt.grid(True)
    plt.savefig(out_png, dpi=300, bbox_inches="tight")

    print("========================================")
    print(f"saved: {out_pt}")
    print(f"saved: {out_png}")
    print("========================================")


if __name__ == "__main__":
    main()
