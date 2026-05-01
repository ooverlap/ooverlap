import argparse
import importlib.util
import json
import socket
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def repo_root():
    return Path(__file__).resolve().parents[1]


def load_ooverlap_ext():
    root = repo_root()
    so = root / "build" / "lib" / "ooverlap_ext.so"
    if not so.exists():
        raise FileNotFoundError(f"Could not find {so}. Build first.")

    spec = importlib.util.spec_from_file_location("ooverlap_ext", str(so))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_algo_meta(algo, algo_dict_path=None):
    root = repo_root()

    if algo_dict_path is None:
        algo_dict_path = root / "configs" / "AlgoDictSm90.json"
    else:
        algo_dict_path = Path(algo_dict_path).expanduser().resolve()

    if not algo_dict_path.exists():
        raise FileNotFoundError(f"Could not find algo dict: {algo_dict_path}")

    with open(algo_dict_path, "r") as f:
        data = json.load(f)

    for item in data.get("algorithms", []):
        if int(item["algo"]) == int(algo):
            return item, algo_dict_path

    raise ValueError(f"algo={algo} not found in {algo_dict_path}")


def find_free_port():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


def make_column_major_reorder(M, N, tile_m, tile_n, device="cuda"):
    assert M % tile_m == 0
    assert N % tile_n == 0

    tile_rows = M // tile_m
    tile_cols = N // tile_n
    num_tiles = tile_rows * tile_cols

    ra = torch.empty((num_tiles,), dtype=torch.int32, device=device)

    packed_pos = 0
    for tile_col in range(tile_cols):
        for tile_row in range(tile_rows):
            logical_tile = tile_row * tile_cols + tile_col
            ra[logical_tile] = packed_pos
            packed_pos += 1

    return ra


def make_identity_reorder(M, N, tile_m, tile_n, device="cuda"):
    assert M % tile_m == 0
    assert N % tile_n == 0

    tile_rows = M // tile_m
    tile_cols = N // tile_n
    num_tiles = tile_rows * tile_cols

    return torch.arange(num_tiles, device=device, dtype=torch.int32)


def make_segments(num_tiles, group_tiles):
    # group_tiles=0 means no segmentation / one full segment.
    if group_tiles <= 0 or group_tiles >= num_tiles:
        return [num_tiles]

    segs = []
    remaining = num_tiles
    while remaining > 0:
        x = min(group_tiles, remaining)
        segs.append(x)
        remaining -= x

    return segs


def sync_all(device_index=None):
    torch.cuda.synchronize()

    if dist.is_initialized():
        if device_index is None:
            dist.barrier()
        else:
            # Avoid ProcessGroupNCCL "devices used by this process are unknown" warning.
            dist.barrier(device_ids=[int(device_index)])

    torch.cuda.synchronize()


def reduce_max_float(x, device):
    t = torch.tensor([float(x)], device=device, dtype=torch.float32)
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return float(t.item())


def time_cuda_max(fn, warmup, iters, device, device_index):
    # Make sure both ranks enter this benchmark section together.
    sync_all(device_index)

    for _ in range(warmup):
        fn()

    # Make sure warmup is fully done on both ranks.
    sync_all(device_index)

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(iters):
        fn()
    end.record()

    torch.cuda.synchronize()

    local_ms = start.elapsed_time(end) / iters
    max_ms = reduce_max_float(local_ms, device)

    # Do not let one rank start the next section early.
    sync_all(device_index)

    return max_ms, local_ms


def worker(
    rank,
    world,
    dist_url,
    nccl_id,
    M,
    N,
    K,
    reldn,
    group_tiles,
    warmup,
    iters,
    reorder,
    algo,
    skip_extra,
    algo_dict_path,
):
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    dist.init_process_group(
        backend="nccl",
        init_method=dist_url,
        rank=rank,
        world_size=world,
    )

    try:
        torch.manual_seed(1234 + rank)

        ext = load_ooverlap_ext()
        algo_meta, algo_dict_path_resolved = load_algo_meta(algo, algo_dict_path)

        tile_m = int(algo_meta["tile_m"])
        tile_n = int(algo_meta["tile_n"])
        tile_k = int(algo_meta["tile_k"])
        cluster = algo_meta.get("cluster", None)
        stages = algo_meta.get("stages", None)
        mainloop = algo_meta.get("mainloop", None)
        epilogue = algo_meta.get("epilogue", None)

        assert M % tile_m == 0, (
            f"M={M} must be divisible by tile_m={tile_m} for algo={algo}"
        )
        assert N % tile_n == 0, (
            f"N={N} must be divisible by tile_n={tile_n} for algo={algo}"
        )
        assert reldn > 0, f"reldn must be positive, got {reldn}"

        tile_rows = M // tile_m
        tile_cols = N // tile_n
        num_tiles = tile_rows * tile_cols

        packed_tile_cols = int(reldn)
        packed_tile_rows = (num_tiles + packed_tile_cols - 1) // packed_tile_cols
        packed_M = packed_tile_rows * tile_m
        packed_N = packed_tile_cols * tile_n

        overlap_cseg = make_segments(num_tiles, group_tiles)
        full_cseg = [num_tiles]

        ov = ext.OverlapImpl()
        ov.nccl_init(rank, world, nccl_id)
        ov.cutlass_init()
        ov.overlap_init()

        A = torch.randn((M, K), device=device, dtype=torch.float16)

        # Torch baseline uses B_ref as [K, N].
        B_ref = torch.randn((K, N), device=device, dtype=torch.float16)

        # Our wrapper expects packed/column-major style B as [N, K].
        B_packed = B_ref.t().contiguous()

        C_packed = torch.empty((packed_M, packed_N), device=device, dtype=torch.float16)
        C_packed_full = torch.empty((packed_M, packed_N), device=device, dtype=torch.float16)

        C_torch = torch.empty((M, N), device=device, dtype=torch.float16)
        C_nccl_only = torch.randn((packed_M, packed_N), device=device, dtype=torch.float16)

        if reorder == "column_major":
            RA = make_column_major_reorder(M, N, tile_m, tile_n, device=device)
        elif reorder == "identity":
            RA = make_identity_reorder(M, N, tile_m, tile_n, device=device)
        else:
            raise ValueError(f"unknown reorder={reorder}")

        overlap_cseg_cpu = torch.tensor(overlap_cseg, dtype=torch.int32)
        overlap_cseg_gpu = overlap_cseg_cpu.to(device=device)

        full_cseg_cpu = torch.tensor(full_cseg, dtype=torch.int32)
        full_cseg_gpu = full_cseg_cpu.to(device=device)

        # Layout:
        #   MM[0:num_segments]               = segment counters
        #   MM[num_segments:num_segments+T]  = per-tile arrival counters
        MM_overlap = torch.empty(
            (len(overlap_cseg) + num_tiles,),
            device=device,
            dtype=torch.int32,
        )
        MM_full = torch.empty(
            (len(full_cseg) + num_tiles,),
            device=device,
            dtype=torch.int32,
        )

        monitor = False

        def run_packed_overlap():
            MM_overlap.zero_()
            ov.gemm_allreduce_overlap(
                A,
                B_packed,
                C_packed,
                MM_overlap,
                RA,
                int(reldn),
                overlap_cseg_cpu,
                overlap_cseg_gpu,
                int(algo),
                monitor,
            )

        def run_packed_full_segment():
            MM_full.zero_()
            ov.gemm_allreduce_overlap(
                A,
                B_packed,
                C_packed_full,
                MM_full,
                RA,
                int(reldn),
                full_cseg_cpu,
                full_cseg_gpu,
                int(algo),
                monitor,
            )

        def run_torch_matmul_plus_nccl():
            torch.matmul(A, B_ref, out=C_torch)
            ov.nccl_allreduce(C_torch)

        def run_nccl_only_full():
            ov.nccl_allreduce(C_nccl_only)

        MM_overlap.zero_()
        ext.gemm_signal_sm90(
            A,
            B_packed,
            C_packed,
            MM_overlap,
            RA,
            overlap_cseg_gpu,
            int(reldn),
            int(algo),
            False,
        )
        torch.cuda.synchronize()
        
        if rank == 0:
            seg_counts = MM_overlap[:len(overlap_cseg)].detach().cpu().tolist()
            tile_arrivals = MM_overlap[len(overlap_cseg):].detach().cpu()
            print("expected segments:", overlap_cseg, flush=True)
            print("actual segment counts:", seg_counts, flush=True)
            print("tile_done unique:", torch.unique(tile_arrivals, return_counts=True), flush=True)

        overlap_ms, overlap_local_ms = time_cuda_max(
            run_packed_overlap,
            warmup,
            iters,
            device,
            rank,
        )

        full_segment_ms, full_segment_local_ms = time_cuda_max(
            run_packed_full_segment,
            warmup,
            iters,
            device,
            rank,
        )

        torch_baseline_ms = None
        torch_baseline_local_ms = None
        nccl_only_ms = None
        nccl_only_local_ms = None

        if not skip_extra:
            torch_baseline_ms, torch_baseline_local_ms = time_cuda_max(
                run_torch_matmul_plus_nccl,
                warmup,
                iters,
                device,
                rank,
            )

            nccl_only_ms, nccl_only_local_ms = time_cuda_max(
                run_nccl_only_full,
                warmup,
                iters,
                device,
                rank,
            )

        if rank == 0:
            speedup_vs_full = (
                full_segment_ms / overlap_ms if overlap_ms > 0 else float("nan")
            )

            print("========================================")
            print(f"M={M} N={N} K={K}")
            print(f"tile_rows={tile_rows} tile_cols={tile_cols} num_tiles={num_tiles}")
            print(f"reldn={reldn} packed_shape=({packed_M}, {packed_N})")
            print(f"reorder={reorder}")
            print(
                f"algo={algo} tile={tile_m}x{tile_n}x{tile_k} "
                f"cluster={cluster} stages={stages} "
                f"mainloop={mainloop} epilogue={epilogue}"
            )
            print(f"algo_dict={algo_dict_path_resolved}")
            print("")
            print(f"overlap group_tiles={group_tiles}")
            print(f"overlap num_segments={len(overlap_cseg)}")
            print(
                f"overlap segments={overlap_cseg[:16]}"
                f"{' ...' if len(overlap_cseg) > 16 else ''}"
            )
            print("")
            print("All reported latencies below are MAX across ranks.")
            print(f"packed overlap latency:        {overlap_ms:.4f} ms")
            print(f"packed full-segment latency:   {full_segment_ms:.4f} ms")

            if torch_baseline_ms is not None:
                print(f"torch matmul + NCCL latency:   {torch_baseline_ms:.4f} ms")
            if nccl_only_ms is not None:
                print(f"NCCL-only full-buffer latency: {nccl_only_ms:.4f} ms")

            print("")
            print(f"rank0 packed overlap local:    {overlap_local_ms:.4f} ms")
            print(f"rank0 full-segment local:      {full_segment_local_ms:.4f} ms")

            if torch_baseline_local_ms is not None:
                print(f"rank0 torch+NCCL local:        {torch_baseline_local_ms:.4f} ms")
            if nccl_only_local_ms is not None:
                print(f"rank0 NCCL-only local:         {nccl_only_local_ms:.4f} ms")

            print("")
            print(f"speedup vs packed full-seg:    {speedup_vs_full:.4f}x")
            print("========================================")

    finally:
        try:
            sync_all(rank)
        finally:
            if dist.is_initialized():
                dist.destroy_process_group()


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--gpus", type=int, default=2)
    ap.add_argument("--m", type=int, default=1024)
    ap.add_argument("--n", type=int, default=1024)
    ap.add_argument("--k", type=int, default=4096)

    ap.add_argument(
        "--reldn",
        type=int,
        default=1,
        help="Packed tile columns. reldn=1 gives shape (num_tiles * tile_m, tile_n).",
    )

    ap.add_argument(
        "--group-tiles",
        type=int,
        default=8,
        help="Tiles per communication segment. Use 0 for one full segment.",
    )

    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--iters", type=int, default=200)

    ap.add_argument(
        "--reorder",
        choices=["column_major", "identity"],
        default="column_major",
    )

    ap.add_argument("--algo", type=int, default=8)

    ap.add_argument(
        "--algo-dict",
        type=str,
        default=None,
        help="Path to AlgoDictSm90.json. Defaults to configs/AlgoDictSm90.json.",
    )

    ap.add_argument(
        "--skip-extra",
        action="store_true",
        help="Skip torch+NCCL and NCCL-only baselines.",
    )

    args = ap.parse_args()

    assert torch.cuda.is_available()
    assert args.gpus >= 1
    assert torch.cuda.device_count() >= args.gpus

    ext = load_ooverlap_ext()
    nccl_id = ext.generate_nccl_id()

    port = find_free_port()
    dist_url = f"tcp://127.0.0.1:{port}"

    mp.spawn(
        worker,
        args=(
            args.gpus,
            dist_url,
            nccl_id,
            args.m,
            args.n,
            args.k,
            args.reldn,
            args.group_tiles,
            args.warmup,
            args.iters,
            args.reorder,
            args.algo,
            args.skip_extra,
            args.algo_dict,
        ),
        nprocs=args.gpus,
        join=True,
    )


if __name__ == "__main__":
    main()
