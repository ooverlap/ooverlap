import argparse
import importlib.util
import math
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

    spec = importlib.util.spec_from_file_location("ooverlap_ext", str(so))
    mod = importlib.util.module_from_spec(spec)
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


def perf_comm_process(
    rank,
    world_size,
    nccl_id,
    comm_op,
    sizes,
    warmup,
    iters,
    barrier,
    result_dict,
):
    torch.cuda.set_device(rank)

    ext = load_ooverlap_ext()

    comm_class = ext.OverlapImpl()
    comm_class.nccl_init(rank, world_size, nccl_id)
    comm_class.cutlass_init()

    rank_results = []
    print(sizes)

    for size in sizes:
        M = 1024
        N = size // 1024

        C = torch.empty((M, N), dtype=torch.float16, device="cuda").normal_(
            mean=0.0,
            std=0.5,
        )

        # sleep to let the link rest?!
        sleep(5)
        torch.cuda.synchronize()
        barrier.wait()

        if comm_op == "all_reduce":
            for _ in range(warmup):
                comm_class.nccl_allreduce(C)

            torch.cuda.synchronize()
            barrier.wait()

            start_event = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
            end_event = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]

            for i in range(iters):
                start_event[i].record()
                comm_class.nccl_allreduce(C)
                end_event[i].record()

        elif comm_op == "reduce_scatter":
            for _ in range(warmup):
                comm_class.nccl_reducescatter(C)

            torch.cuda.synchronize()
            barrier.wait()

            start_event = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
            end_event = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]

            for i in range(iters):
                start_event[i].record()
                comm_class.nccl_reducescatter(C)
                end_event[i].record()

        else:
            raise ValueError(f"Unsupported comm_op={comm_op}")

        torch.cuda.synchronize()
        barrier.wait()

        dur_ms = torch.tensor(
            [s.elapsed_time(e) for s, e in zip(start_event, end_event)],
            dtype=torch.float32,
        )

        s = (f"====================size is {size}============================"
                f"{dur_ms}\n"
        "---------------------------------------------------------------")
        print(s)

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

    result_dict[rank] = rank_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--comm_op", type=str, default="all_reduce",
                        choices=["all_reduce", "reduce_scatter"])
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--use-median", action="store_true")
    parser.add_argument("--use-p90", action="store_true")
    args = parser.parse_args()

    world_size = torch.cuda.device_count()
    if world_size < 2:
        raise RuntimeError("At least 2 GPUs are required.")

    sizes = make_sizes()

    ext = load_ooverlap_ext()
    nccl_id = ext.generate_nccl_id()

    manager = mp.Manager()
    result_dict = manager.dict()
    barrier = manager.Barrier(world_size)

    mp.spawn(
        perf_comm_process,
        args=(
            world_size,
            nccl_id,
            args.comm_op,
            sizes,
            args.warmup,
            args.iters,
            barrier,
            result_dict,
        ),
        nprocs=world_size,
        join=True,
    )

    bandwidths = []
    comm_array = torch.zeros((len(sizes), 2), dtype=torch.float32)

    print("========================================")
    print("bandwidth.py")
    print(f"comm_op:    {args.comm_op}")
    print(f"gpus:       {world_size}")
    print("dtype:      fp16")
    print(f"warmup:     {args.warmup}")
    print(f"iters:      {args.iters}")
    print(f"use_median: {args.use_median}")
    print(f"use_p90:    {args.use_p90}")
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

    out_pt = out_dir / f"bandwidth_{args.comm_op}_tp{world_size}.pt"
    torch.save(comm_array, out_pt)

    plt.plot(sizes, bandwidths, marker="o")
    plt.xlabel("Data Size (elements)")
    plt.ylabel("Bandwidth (GB/s)")
    plt.title(f"Bandwidth vs Data Size ({args.comm_op}, tp={world_size})")
    plt.grid(True)
    plt.savefig("bandwidth.png", dpi=300, bbox_inches="tight")

    print("========================================")
    print(f"saved: {out_pt}")
    print("saved: bandwidth.png")
    print("========================================")


if __name__ == "__main__":
    main()
