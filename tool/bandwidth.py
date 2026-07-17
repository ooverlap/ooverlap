#!/usr/bin/env python3

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
    x_sorted = torch.sort(x.flatten()).values
    n = int(x_sorted.numel())
    idx = int(math.ceil(0.90 * n)) - 1
    idx = max(0, min(idx, n - 1))
    return float(x_sorted[idx].item())


def parse_devices(s):
    devices = [int(x) for x in str(s).split(",") if x.strip() != ""]
    if len(devices) < 2:
        raise ValueError("--devices must contain at least two CUDA device ids, e.g. --devices 0,1")
    if len(set(devices)) != len(devices):
        raise ValueError("--devices must not contain duplicate CUDA device ids")
    if any(device < 0 for device in devices):
        raise ValueError("--devices must contain non-negative CUDA device ids")
    return devices


def backend_label(backend):
    if backend == "nccl":
        return "nccl"
    if backend == "ooverlap":
        return "ooverlap"
    raise ValueError(f"Unsupported comm_backend={backend}")


def init_comm(ext, rank, world_size, devices, backend, nccl_id, broker_key):
    if backend == "nccl":
        comm = ext.BaselineImpl()
        comm.nccl_init(rank, world_size, nccl_id)
        return comm

    if backend == "ooverlap":
        comm = ext.OverlapImpl()
        comm.cutlass_init()
        comm.ooverlap_ipc_init(rank, world_size, devices, broker_key)
        return comm

    raise ValueError(f"Unsupported comm_backend={backend}")


def release_comm(comm, backend):
    if backend == "ooverlap":
        comm.ooverlap_release()


def call_comm(comm, backend, comm_op, C):
    if backend == "nccl":
        if comm_op == "all_reduce":
            comm.nccl_allreduce(C)
        elif comm_op == "reduce_scatter":
            comm.nccl_reducescatter(C)
        else:
            raise ValueError(f"Unsupported comm_op={comm_op}")
        return

    if backend == "ooverlap":
        if comm_op != "all_reduce":
            raise RuntimeError("ooverlap backend currently supports only all_reduce")
        comm.ooverlap_allreduce(C)
        return

    raise ValueError(f"Unsupported comm_backend={backend}")


def perf_comm_process(
    rank,
    world_size,
    devices,
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
    torch.cuda.set_device(devices[rank])

    ext = load_ooverlap_ext()
    comm = init_comm(
        ext=ext,
        rank=rank,
        world_size=world_size,
        devices=devices,
        backend=comm_backend,
        nccl_id=nccl_id,
        broker_key=broker_key,
    )

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

        if sleep_seconds > 0:
            sleep(sleep_seconds)

        torch.cuda.synchronize()
        barrier.wait()

        for _ in range(warmup):
            call_comm(comm, comm_backend, comm_op, C)

        torch.cuda.synchronize()
        barrier.wait()

        start_event = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        end_event = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]

        for i in range(iters):
            start_event[i].record()
            call_comm(comm, comm_backend, comm_op, C)
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
            f"op={comm_op} "
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
        torch.cuda.synchronize()
        barrier.wait()

    torch.cuda.synchronize()
    barrier.wait()

    release_comm(comm, comm_backend)

    torch.cuda.synchronize()
    barrier.wait()

    result_dict[rank] = rank_results


def perf_comm(comm_backend, comm_op, devices, sizes, warmup, iters, sleep_seconds):
    world_size = len(devices)

    if world_size < 2:
        raise RuntimeError("At least 2 GPUs are required.")

    if comm_backend == "ooverlap" and comm_op != "all_reduce":
        raise RuntimeError("ooverlap backend currently supports only all_reduce")

    ext = load_ooverlap_ext()

    nccl_id = ext.generate_nccl_id() if comm_backend == "nccl" else []
    broker_key = f"ooverlap_bw_{os.getpid()}_{int(torch.empty((), device='cpu').random_().item())}"

    manager = mp.Manager()
    result_dict = manager.dict()
    barrier = manager.Barrier(world_size)

    mp.spawn(
        perf_comm_process,
        args=(
            world_size,
            devices,
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
        ),
        nprocs=world_size,
        join=True,
    )

    return result_dict


def choose_latency(mean_ms, median_ms, p90_ms, use_median, use_p90):
    if use_p90:
        return p90_ms
    if use_median:
        return median_ms
    return mean_ms


def bandwidth_for(comm_op, world_size, data_size_bytes, latency_ms):
    latency_s = latency_ms * 1.0e-3

    if comm_op == "all_reduce":
        total_data_transferred = data_size_bytes * 2 * (world_size - 1)
    elif comm_op == "reduce_scatter":
        total_data_transferred = data_size_bytes * (world_size - 1)
    else:
        raise ValueError(f"Unsupported comm_op={comm_op}")

    flash_bw = total_data_transferred / latency_s / (1024 ** 3)
    alg_bw = data_size_bytes / latency_s / (1024 ** 3)
    return flash_bw, alg_bw


def summarize_results(
    comm_backend,
    comm_op,
    devices,
    sizes,
    result_dict,
    warmup,
    iters,
    sleep_seconds,
    use_median,
    use_p90,
):
    world_size = len(devices)
    backend = backend_label(comm_backend)

    bandwidths = []
    comm_array = torch.zeros((len(sizes), 2), dtype=torch.float32)

    print("========================================")
    print("bandwidth.py")
    print(f"comm_backend: {backend}")
    print(f"comm_op:      {comm_op}")
    print(f"devices:      {devices}")
    print(f"gpus:         {world_size}")
    print("dtype:        fp16")
    print(f"warmup:       {warmup}")
    print(f"iters:        {iters}")
    print(f"sleep_sec:    {sleep_seconds}")
    print(f"use_median:   {use_median}")
    print(f"use_p90:      {use_p90}")
    print("========================================")

    for i, size in enumerate(sizes):
        per_rank = [result_dict[r][i] for r in range(world_size)]

        mean_ms = max(x["mean_ms"] for x in per_rank)
        median_ms = max(x["median_ms"] for x in per_rank)
        p90_ms = max(x["p90_ms"] for x in per_rank)
        min_ms = max(x["min_ms"] for x in per_rank)
        max_ms = max(x["max_ms"] for x in per_rank)

        latency_ms = choose_latency(
            mean_ms=mean_ms,
            median_ms=median_ms,
            p90_ms=p90_ms,
            use_median=use_median,
            use_p90=use_p90,
        )

        data_size_bytes = size * 2  # fp16
        flash_bw, alg_bw = bandwidth_for(
            comm_op=comm_op,
            world_size=world_size,
            data_size_bytes=data_size_bytes,
            latency_ms=latency_ms,
        )

        bandwidths.append(flash_bw)
        comm_array[i, 0] = float(size)
        comm_array[i, 1] = float(flash_bw)

        print(
            f"backend={backend:9s} "
            f"op={comm_op:14s} "
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

    return comm_array, bandwidths


def save_outputs(comm_backend, comm_op, world_size, sizes, comm_array, bandwidths):
    backend = backend_label(comm_backend)

    out_dir = repo_root() / "configs"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_pt = out_dir / f"bandwidth_{backend}_{comm_op}_tp{world_size}.pt"
    torch.save(comm_array, out_pt)

    out_png = out_dir / f"bandwidth_{backend}_{comm_op}_tp{world_size}.png"

    plt.figure()
    plt.plot(sizes, bandwidths, marker="o")
    plt.xlabel("Data Size (elements)")
    plt.ylabel("Bandwidth (GB/s)")
    plt.title(f"Bandwidth vs Data Size ({backend}, {comm_op}, tp={world_size})")
    plt.grid(True)
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()

    print("========================================")
    print(f"saved: {out_pt}")
    print(f"saved: {out_png}")
    print("========================================")


def run_backend(args, comm_backend):
    devices = parse_devices(args.devices)
    sizes = make_sizes()

    result_dict = perf_comm(
        comm_backend=comm_backend,
        comm_op=args.comm_op,
        devices=devices,
        sizes=sizes,
        warmup=args.warmup,
        iters=args.iters,
        sleep_seconds=args.sleep_seconds,
    )

    comm_array, bandwidths = summarize_results(
        comm_backend=comm_backend,
        comm_op=args.comm_op,
        devices=devices,
        sizes=sizes,
        result_dict=result_dict,
        warmup=args.warmup,
        iters=args.iters,
        sleep_seconds=args.sleep_seconds,
        use_median=args.use_median,
        use_p90=args.use_p90,
    )

    save_outputs(
        comm_backend=comm_backend,
        comm_op=args.comm_op,
        world_size=len(devices),
        sizes=sizes,
        comm_array=comm_array,
        bandwidths=bandwidths,
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--comm_backend",
        type=str,
        default="nccl",
        choices=["nccl", "ooverlap", "both"],
        help="Backend to benchmark.",
    )

    parser.add_argument(
        "--comm_op",
        type=str,
        default="all_reduce",
        choices=["all_reduce", "reduce_scatter"],
    )

    parser.add_argument(
        "--devices",
        type=str,
        default="0,1",
        help="Comma-separated CUDA device ids. Default: 0,1",
    )

    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--sleep-seconds", type=float, default=5.0)
    parser.add_argument("--use-median", action="store_true")
    parser.add_argument("--use-p90", action="store_true")
    parser.add_argument("--sizes", action="store_true")

    args = parser.parse_args()

    if args.sizes:
        print(f"sizes for test are: {make_sizes()}")
        return

    if args.warmup < 0:
        raise ValueError("--warmup must be >= 0")
    if args.iters <= 0:
        raise ValueError("--iters must be > 0")

    if args.comm_backend in ("ooverlap", "both") and args.comm_op != "all_reduce":
        raise RuntimeError("ooverlap backend currently supports only --comm_op all_reduce")

    if args.comm_backend == "both":
        run_backend(args, "nccl")
        run_backend(args, "ooverlap")
    else:
        run_backend(args, args.comm_backend)


if __name__ == "__main__":
    main()
