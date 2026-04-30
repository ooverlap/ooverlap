import argparse
import importlib.util
from pathlib import Path

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


def perf_comm_process(rank, world_size, nccl_id, M, N, comm_op, result_dict):
    torch.cuda.set_device(rank)

    ext = load_ooverlap_ext()

    comm_class = ext.OverlapImpl()
    comm_class.nccl_init(rank, world_size, nccl_id)
    comm_class.cutlass_init()

    C = torch.empty((M, N), dtype=torch.float16, device="cuda").normal_(mean=0.0, std=0.5)

    if comm_op == "all_reduce":
        for _ in range(20):
            comm_class.nccl_allreduce(C)

        start_event = [torch.cuda.Event(enable_timing=True) for _ in range(200)]
        end_event = [torch.cuda.Event(enable_timing=True) for _ in range(200)]

        for i in range(200):
            start_event[i].record()
            comm_class.nccl_allreduce(C)
            end_event[i].record()

        torch.cuda.synchronize()
        dur = torch.tensor(
            [s.elapsed_time(e) for s, e in zip(start_event, end_event)],
            dtype=torch.float,
        )

    elif comm_op == "reduce_scatter":
        for _ in range(20):
            comm_class.nccl_reducescatter(C)

        start_event = [torch.cuda.Event(enable_timing=True) for _ in range(200)]
        end_event = [torch.cuda.Event(enable_timing=True) for _ in range(200)]

        for i in range(200):
            start_event[i].record()
            comm_class.nccl_reducescatter(C)
            end_event[i].record()

        torch.cuda.synchronize()
        dur = torch.tensor(
            [s.elapsed_time(e) for s, e in zip(start_event, end_event)],
            dtype=torch.float,
        )

    else:
        raise ValueError(f"Unsupported communication operation: {comm_op}")

    result_dict[rank] = torch.mean(dur).item()


def perf_comm(M: int, N: int, comm_op: str):
    world_size = torch.cuda.device_count()
    if world_size < 2:
        raise RuntimeError("At least 2 GPUs are required!")

    ext = load_ooverlap_ext()
    nccl_id = ext.generate_nccl_id()

    torch.cuda.synchronize()

    manager = mp.Manager()
    result_dict = manager.dict()

    mp.spawn(
        perf_comm_process,
        args=(world_size, nccl_id, M, N, comm_op, result_dict),
        nprocs=world_size,
    )

    return result_dict[0]


def main():
    world_size = torch.cuda.device_count()

    parser = argparse.ArgumentParser()
    parser.add_argument("--comm_op", type=str, default="all_reduce")
    args = parser.parse_args()

    data_sizes = [(int(2 ** (20 + 0.25 * i)) // 1024 * 1024) for i in range(36)]

    bandwidths = []
    size_len = len(data_sizes)
    comm_array = torch.zeros((size_len, 2))

    for i, size in enumerate(data_sizes):
        input_data = torch.randn(size, dtype=torch.float16, device="cuda")

        avg_time_ms = perf_comm(1024, size // 1024, args.comm_op)

        data_size_bytes = input_data.numel() * input_data.element_size()

        if args.comm_op == "all_reduce":
            total_data_transferred = data_size_bytes * 2 * (world_size - 1)
        elif args.comm_op == "reduce_scatter":
            total_data_transferred = data_size_bytes * (world_size - 1)
        else:
            raise ValueError("Unsupported communication operation")

        # CUDA event elapsed_time is milliseconds.
        # FlashOverlap code forgot this conversion. Without 1e-3, bandwidth is 1000x too small.
        bandwidth = (total_data_transferred / avg_time_ms) / (1024 ** 3)

        bandwidths.append(bandwidth)

        comm_array[i, 0] = size
        comm_array[i, 1] = bandwidth

        print(
            f"size={size:12d} elems "
            f"bytes={data_size_bytes:12d} "
            f"time={avg_time_ms:9.5f} ms "
            f"bandwidth={bandwidth:9.2f} GB/s"
        )

    plt.plot(data_sizes, bandwidths, marker="o")
    plt.xlabel("Data Size (elements)")
    plt.ylabel("Bandwidth (GB/s)")
    plt.title("Bandwidth vs Data Size")
    plt.grid(True)
    plt.savefig("bandwidth.png", dpi=300, bbox_inches="tight")
    plt.show()

    out_dir = repo_root() / "configs"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / f"bandwidth_{args.comm_op}_tp{world_size}.pt"
    torch.save(comm_array, out_path)

    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
