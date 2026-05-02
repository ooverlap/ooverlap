'''
    Using multiprocessing for distributed running,
    please specify the GPUs via CUDA_VISIBLE_DEVICES:
        CUDA_VISIBLE_DEVICES=0,1 python3 test.py --m 4096 --n 8192 --k 4096
'''

import torch
import time
import json
from pathlib import Path
import torch.multiprocessing as mp
import pandas as pd
import argparse
import os
import importlib.util
import sys


WARM_UP=20
REP=200


def repo_root():
    p = Path(__file__).resolve()
    for q in [p.parent, *p.parents]:
        if (q / "build" / "lib" / "ooverlap_ext.so").exists():
            return q
    return Path(__file__).resolve().parents[1]


def load_ooverlap_ext():
    root = repo_root()
    so = root / "build" / "lib" / "ooverlap_ext.so"

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


ext = load_ooverlap_ext()


def div_up(x: int, y: int):
    return (x + y - 1) // y


def gpu_config_name():
    device = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device)

    # Match gen_config_file/search.py naming:
    # "NVIDIA H100 NVL" -> "nvidia_h100_nvl"
    return props.name.lower().replace(" ", "_")


def solution_json_path(M: int, N: int, K: int):
    gpu_name = gpu_config_name()
    return repo_root() / "configs" / f"solution_m{M}n{N}k{K}_{gpu_name}_packed_sm90.json"


def packed_shape(M: int, N: int, BM: int, BN: int, rLDN: int = 1):
    TileNum = div_up(M, BM) * div_up(N, BN)
    packed_tile_rows = div_up(TileNum, rLDN)
    return packed_tile_rows * BM, rLDN * BN


def monitor_size(TileNum: int, seg_size: int, if_monitor: bool):
    if if_monitor:
        return seg_size + TileNum + 1 + TileNum
    return seg_size + TileNum


def reset_monitor_matrix(MonitoredMatrix, TileNum: int, seg_size: int, if_monitor: bool):
    if if_monitor:
        # Reset segment counters, per-tile epilogue counters, and global monitor counter.
        MonitoredMatrix[: seg_size + TileNum + 1] = 0
    else:
        # Reset segment counters and per-tile epilogue counters.
        MonitoredMatrix[: seg_size + TileNum] = 0


def reorder_indices(S, hint):
    # Generate the original array of indices [0, 1, ..., S-1]
    original = list(range(S))

    # Create an empty list to store the new order of indices
    new_order = [-1] * S

    # Place the indices of the hint list in the first positions of the new order
    for i, element in enumerate(hint):
        new_order[element] = i

    # Place the remaining indices in the new order
    remaining_elements = [x for x in original if x not in hint]
    for i, element in enumerate(remaining_elements, start=len(hint)):
        new_order[element] = i

    return torch.tensor(new_order, dtype=torch.int, device="cuda")


def make_reordered_array(TileNum: int, hint: list, reorder_map=None):
    if reorder_map is not None:
        if len(reorder_map) != TileNum:
            raise ValueError(
                f"reorder_map length mismatch: got {len(reorder_map)}, expected {TileNum}"
            )
        return torch.tensor([int(x) for x in reorder_map], dtype=torch.int, device="cuda")

    return reorder_indices(TileNum, hint)


def generate_row_remap_array(
    M, N, BM, BN, S_list, world_size, device="cuda"
):
    total_tiles = (M * N) // (BM * BN)
    assert sum(S_list) == total_tiles, "sum(S_list) must equal total number of tiles"

    original_row_ids = torch.arange(M * N // BN, dtype=torch.int, device=device)
    reordered_row_id = torch.empty_like(original_row_ids)

    current_row = 0
    for S in S_list:
        chunk_size = S * BM
        chunk_row_ids = original_row_ids[current_row : current_row + chunk_size]

        # Compute row_id % world_size for the current chunk
        mod_values = chunk_row_ids % world_size

        # Sort the chunk based on mod_values (stable sort)
        _, sorted_indices = torch.sort(mod_values, stable=True)
        reordered_chunk = chunk_row_ids[sorted_indices]

        reordered_row_id[current_row : current_row + chunk_size] = reordered_chunk
        current_row += chunk_size

    # Compute remap: remap[original_row_id] = new_row_id
    remap = torch.empty_like(original_row_ids)
    remap[reordered_row_id] = torch.arange(len(reordered_row_id), dtype=torch.int, device=device)

    return remap


def perf_running_process(rank, world_size, nccl_id, broker_key, comm_backend,
    M: int, N: int, K: int,
    BM: int, BN: int, Algo: int, cSeg: list, hint: list, reorder_map,
    comm_op: str,
    active_sm_count: int,
    result_dict):

    cSeg_CPU = torch.tensor(cSeg, dtype=torch.int32)
    cSeg_GPU = cSeg_CPU.cuda(rank)

    TileNum = div_up(M, BM) * div_up(N, BN)

    torch.cuda.set_device(rank)

    gemm_class = ext.OverlapImpl()

    # Keep NCCL initialized for normal paths and for fallback.
    gemm_class.nccl_init(rank, world_size, nccl_id)

    if comm_backend == "ooverlap":
        raise RuntimeError("ooverlap IPC backend is not wired in this SM90 packed port yet")

    gemm_class.cutlass_init()
    gemm_class.overlap_init()

    A = torch.empty((M, K), dtype=torch.float16, device="cuda").normal_(mean=0., std=0.5)
    B = torch.empty((N, K), dtype=torch.float16, device="cuda").normal_(mean=0., std=0.5)

    packed_M, packed_N = packed_shape(M, N, BM, BN, 1)
    C = torch.empty((packed_M, packed_N), dtype=torch.float16, device="cuda")

    MonitoredMatrix = torch.zeros((monitor_size(TileNum, len(cSeg), False),), dtype=torch.int, device="cuda")
    ReorderedArray = make_reordered_array(TileNum, hint, reorder_map).reshape(((M+BM-1)//BM, (N+BN-1)//BN))

    if comm_op == "reduce_scatter":
        D = torch.empty((M // world_size, N), dtype=torch.float16, device="cuda")
        RowArray = generate_row_remap_array(M, N, BM, BN, cSeg, world_size)

    _warm_up = WARM_UP
    _freq = REP

    if len(cSeg) == 1:
        # No overlapping
        # In this SM90 packed port, use the overlap wrapper with one full segment.
        if comm_op == "all_reduce":
            for _ in range(_warm_up):
                reset_monitor_matrix(MonitoredMatrix, TileNum, len(cSeg), False)
                gemm_class.gemm_allreduce_overlap(A, B, C, MonitoredMatrix, ReorderedArray, 1, cSeg_CPU, cSeg_GPU, Algo, int(active_sm_count), False)

            reset_monitor_matrix(MonitoredMatrix, TileNum, len(cSeg), False)
            gemm_class.gemm_allreduce_overlap(A, B, C, MonitoredMatrix, ReorderedArray, 1, cSeg_CPU, cSeg_GPU, Algo, int(active_sm_count), False)

            start_event = [torch.cuda.Event(enable_timing=True) for i in range(_freq)]
            end_event = [torch.cuda.Event(enable_timing=True) for i in range(_freq)]
            for i in range(_freq):
                reset_monitor_matrix(MonitoredMatrix, TileNum, len(cSeg), False)
                start_event[i].record()
                gemm_class.gemm_allreduce_overlap(A, B, C, MonitoredMatrix, ReorderedArray, 1, cSeg_CPU, cSeg_GPU, Algo, int(active_sm_count), False)
                end_event[i].record()
            torch.cuda.synchronize()
            dur = torch.tensor([s.elapsed_time(e) for s, e in zip(start_event, end_event)], dtype=torch.float)
        elif comm_op == "reduce_scatter":
            for _ in range(_warm_up):
                reset_monitor_matrix(MonitoredMatrix, TileNum, len(cSeg), False)
                gemm_class.gemm_reducescatter_overlap(A, B, C, D, MonitoredMatrix, ReorderedArray, RowArray, 1, cSeg_CPU, cSeg_GPU, Algo, False)

            reset_monitor_matrix(MonitoredMatrix, TileNum, len(cSeg), False)
            gemm_class.gemm_reducescatter_overlap(A, B, C, D, MonitoredMatrix, ReorderedArray, RowArray, 1, cSeg_CPU, cSeg_GPU, Algo, False)

            start_event = [torch.cuda.Event(enable_timing=True) for i in range(_freq)]
            end_event = [torch.cuda.Event(enable_timing=True) for i in range(_freq)]
            for i in range(_freq):
                reset_monitor_matrix(MonitoredMatrix, TileNum, len(cSeg), False)
                start_event[i].record()
                gemm_class.gemm_reducescatter_overlap(A, B, C, D, MonitoredMatrix, ReorderedArray, RowArray, 1, cSeg_CPU, cSeg_GPU, Algo, False)
                end_event[i].record()
            torch.cuda.synchronize()
            dur = torch.tensor([s.elapsed_time(e) for s, e in zip(start_event, end_event)], dtype=torch.float)

    else:
        if comm_op == "all_reduce":
            for _ in range(_warm_up):
                reset_monitor_matrix(MonitoredMatrix, TileNum, len(cSeg), False)
                gemm_class.gemm_allreduce_overlap(A, B, C, MonitoredMatrix, ReorderedArray, 1, cSeg_CPU, cSeg_GPU, Algo, int(active_sm_count), False)

            start_event = [torch.cuda.Event(enable_timing=True) for i in range(_freq)]
            end_event = [torch.cuda.Event(enable_timing=True) for i in range(_freq)]
            for i in range(_freq):
                reset_monitor_matrix(MonitoredMatrix, TileNum, len(cSeg), False)
                start_event[i].record()
                gemm_class.gemm_allreduce_overlap(A, B, C, MonitoredMatrix, ReorderedArray, 1, cSeg_CPU, cSeg_GPU, Algo, int(active_sm_count), False)
                end_event[i].record()
            torch.cuda.synchronize()
            dur = torch.tensor([s.elapsed_time(e) for s, e in zip(start_event, end_event)], dtype=torch.float)
        elif comm_op == "reduce_scatter":
            for _ in range(_warm_up):
                reset_monitor_matrix(MonitoredMatrix, TileNum, len(cSeg), False)
                gemm_class.gemm_reducescatter_overlap(A, B, C, D, MonitoredMatrix, ReorderedArray, RowArray, 1, cSeg_CPU, cSeg_GPU, Algo, False)

            start_event = [torch.cuda.Event(enable_timing=True) for i in range(_freq)]
            end_event = [torch.cuda.Event(enable_timing=True) for i in range(_freq)]
            for i in range(_freq):
                reset_monitor_matrix(MonitoredMatrix, TileNum, len(cSeg), False)
                start_event[i].record()
                gemm_class.gemm_reducescatter_overlap(A, B, C, D, MonitoredMatrix, ReorderedArray, RowArray, 1, cSeg_CPU, cSeg_GPU, Algo, False)
                end_event[i].record()
            torch.cuda.synchronize()
            dur = torch.tensor([s.elapsed_time(e) for s, e in zip(start_event, end_event)], dtype=torch.float)
        else:
            dur = torch.zeros((_freq))

    result_dict[rank] = torch.mean(dur).item()


def perf_running(M: int, N: int, K: int,
    BM: int, BN: int, Algo: int,
    cSeg: list, hint: list, comm_op: str,
    comm_backend: str = "nccl",
    active_sm_count: int = 0,
    reorder_map=None):
    world_size = torch.cuda.device_count()
    if world_size < 2:
        raise RuntimeError("At least 2 GPUs are required for this program.")

    nccl_id = ext.generate_nccl_id()
    broker_key = f"ooverlap_oo_{os.getpid()}_{int(time.time() * 1000000)}"
    torch.cuda.synchronize()
    # print(f"NCCL ID generated: {nccl_id[0]}")

    manager = mp.Manager()
    result_dict = manager.dict()

    mp.spawn(
            perf_running_process,
            args=(world_size, nccl_id, broker_key, comm_backend, M, N, K, BM, BN, Algo, cSeg, hint, reorder_map, comm_op, active_sm_count, result_dict),
            nprocs=world_size
        )

    dur = torch.empty((world_size))
    for i in range(world_size):
        dur[i] = result_dict[i]

    return dur.max()


# Function to initialize NCCL in each process
def perf_comm_process(rank, world_size, nccl_id, M, N, comm_type, result_dict):
    torch.cuda.set_device(rank)

    comm_class = ext.OverlapImpl()

    comm_class.nccl_init(rank, world_size, nccl_id)
    comm_class.cutlass_init()
    comm_class.overlap_init()

    C = torch.empty((M, N), dtype=torch.float16, device="cuda")
    if comm_type == "reduce_scatter":
        D = torch.empty((M // world_size, N), dtype=torch.float16, device="cuda")

    if comm_type == "all_reduce":
        for _ in range(WARM_UP):
            comm_class.nccl_allreduce(C)
        start_event = [torch.cuda.Event(enable_timing=True) for i in range(REP)]
        end_event = [torch.cuda.Event(enable_timing=True) for i in range(REP)]
        for i in range(REP):
            start_event[i].record()
            comm_class.nccl_allreduce(C)
            end_event[i].record()
        torch.cuda.synchronize()
        dur = torch.tensor([s.elapsed_time(e) for s, e in zip(start_event, end_event)], dtype=torch.float)
    elif comm_type == "reduce_scatter":
        for _ in range(WARM_UP):
            comm_class.nccl_reducescatter(C, D)
        start_event = [torch.cuda.Event(enable_timing=True) for i in range(REP)]
        end_event = [torch.cuda.Event(enable_timing=True) for i in range(REP)]
        for i in range(REP):
            start_event[i].record()
            comm_class.nccl_reducescatter(C, D)
            end_event[i].record()
        torch.cuda.synchronize()
        dur = torch.tensor([s.elapsed_time(e) for s, e in zip(start_event, end_event)], dtype=torch.float)
    else:
        dur = torch.zeros((REP))

    result_dict[rank] = torch.mean(dur).item()


def perf_comm(M: int, N: int, comm_type: str):
    world_size = torch.cuda.device_count()
    if world_size < 2:
        raise RuntimeError("At least 2 GPUs are required!")

    nccl_id = ext.generate_nccl_id()
    torch.cuda.synchronize()
    # print(f"NCCL ID generated: {nccl_id[0]}")

    manager = mp.Manager()
    result_dict = manager.dict()

    # get the all reduce time
    mp.spawn(
            perf_comm_process,
            args=(world_size, nccl_id, M, N, comm_type, result_dict),
            nprocs=world_size
        )

    return result_dict[0]


# Function to initialize NCCL in each process
def perf_baseline_process(rank, world_size, nccl_id, M, N, K, comm_op, result_dict):
    torch.cuda.set_device(rank)

    A = torch.empty((M, K), dtype=torch.float16, device="cuda").normal_(mean=0., std=0.5)
    B = torch.empty((N, K), dtype=torch.float16, device="cuda").normal_(mean=0., std=0.5)
    C = torch.empty((M, N), dtype=torch.float16, device="cuda")

    if comm_op == "reduce_scatter":
        D = torch.empty((M // world_size, N), dtype=torch.float16, device="cuda")

    # **** Init Baseline Class **** #
    gemm_comm = ext.BaselineImpl()
    gemm_comm.nccl_init(rank, world_size, nccl_id)
    gemm_comm.cublas_init()

    if comm_op == "all_reduce":
        for _ in range(WARM_UP):
            gemm_comm.gemm_allreduce(A, B, C)
        start_event = [torch.cuda.Event(enable_timing=True) for i in range(REP)]
        end_event = [torch.cuda.Event(enable_timing=True) for i in range(REP)]
        for i in range(REP):
            start_event[i].record()
            # torch.cuda.cudart().cudaProfilerStart()
            gemm_comm.gemm_allreduce(A, B, C)
            # torch.cuda.cudart().cudaProfilerStop()
            end_event[i].record()
        torch.cuda.synchronize()
        dur = torch.tensor([s.elapsed_time(e) for s, e in zip(start_event, end_event)], dtype=torch.float)
    elif comm_op == "reduce_scatter":
        for _ in range(WARM_UP):
            gemm_comm.gemm_reducescatter(A, B, C, D)
        start_event = [torch.cuda.Event(enable_timing=True) for i in range(REP)]
        end_event = [torch.cuda.Event(enable_timing=True) for i in range(REP)]
        for i in range(REP):
            start_event[i].record()
            # torch.cuda.cudart().cudaProfilerStart()
            gemm_comm.gemm_reducescatter(A, B, C, D)
            # torch.cuda.cudart().cudaProfilerStop()
            end_event[i].record()
        torch.cuda.synchronize()
        dur = torch.tensor([s.elapsed_time(e) for s, e in zip(start_event, end_event)], dtype=torch.float)
    else:
        dur = torch.zeros((REP))

    result_dict[rank] = torch.mean(dur).item()


def perf_baseline(M: int, N: int, K: int, comm_op: str):
    world_size = torch.cuda.device_count()
    if world_size < 2:
        raise RuntimeError("At least 2 GPUs are required for this program.")
    # Use the custom NCCL initialization wrapper to get a unique NCCL ID
    nccl_id = ext.generate_nccl_id()
    torch.cuda.synchronize()

    manager = mp.Manager()
    result_dict = manager.dict()

    # Spawn processes
    mp.spawn(
            perf_baseline_process,
            args=(world_size, nccl_id, M, N, K, comm_op, result_dict),
            nprocs=world_size
        )

    dur = torch.empty((world_size))
    for i in range(world_size):
        dur[i] = result_dict[i]

    return dur.max()


def main():
    device = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device)
    gpu_name = gpu_config_name()
    sm_count = props.multi_processor_count
    wave_size = sm_count - 2

    parser = argparse.ArgumentParser()
    parser.add_argument('--m', type=int, default=4096)
    parser.add_argument('--k', type=int, default=8192)
    parser.add_argument('--n', type=int, default=8192)
    parser.add_argument('--comm_op', type=str, default='all_reduce')
    parser.add_argument(
        "--comm_backend",
        type=str,
        default="nccl",
        choices=["nccl", "ooverlap"],
    )
    parser.add_argument(
        "--baseline_impl",
        type=str,
        default="same_algo",
        choices=["same_algo", "cublas"],
        help=(
            "Baseline implementation. same_algo uses the same SM90 CUTLASS "
            "Algo/BM/BN as overlap with cSeg=[tile_num]. cublas keeps the old baseline."
        ),
    )
    parser.add_argument(
        "--baseline_same_algo_active_sms",
        type=str,
        default="all",
        choices=["all", "solution"],
        help=(
            "For --baseline_impl same_algo: use all physical SMs, or use the "
            "solution active_sm_count/compute_sms."
        ),
    )
    args = parser.parse_args()

    comm_op = args.comm_op
    comm_backend = args.comm_backend
    baseline_impl = args.baseline_impl
    print(f"comm_backend: {comm_backend}")
    print(f"baseline_impl: {baseline_impl}")

    m, n, k = args.m, args.n, args.k

    file_path = solution_json_path(m, n, k)

    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    tile_num = m // data["BM"] * n // data["BN"]
    wave_num = (tile_num + wave_size - 1) // wave_size

    active_sm_count = int(data.get("compute_sms", sm_count))
    comm_sm_slack = int(data.get("comm_sm_slack", sm_count - active_sm_count))
    cseg_sum = int(sum(data["cSeg"]))
    reorder_map = data.get("reorder_map")

    print("Loaded solution:", file_path)
    print("Solution debug:")
    print(f"  Algo={data['Algo']} BM={data['BM']} BN={data['BN']}")
    print(f"  sm_count={sm_count} json_sm_count={data.get('sm_count')}")
    print(f"  comm_sm_slack={comm_sm_slack} active_sm_count={active_sm_count}")
    print(f"  cSeg={data['cSeg']}")
    print(f"  len(cSeg)={len(data['cSeg'])} sum(cSeg)={cseg_sum} tile_num={tile_num}")
    print(f"  hint_len={len(data['hint'])}")
    print(f"  has_reorder_map={'reorder_map' in data} reorder_map_len={len(data.get('reorder_map', []))}")
    print(f"  baseline_impl={baseline_impl}")
    if baseline_impl == "same_algo":
        print(f"  baseline_same_algo_active_sms={args.baseline_same_algo_active_sms}")

    assert cseg_sum == tile_num, f"sum(cSeg)={cseg_sum} must equal tile_num={tile_num}"

    gemm_dur = data["dur"]
    comm_dur = perf_comm(m, n, comm_op)

    overlap_dur = perf_running(
        m, n, k,
        data["BM"], data["BN"], data["Algo"],
        data["cSeg"], data["hint"],
        comm_op,
        comm_backend,
        active_sm_count,
        reorder_map,
    )

    if baseline_impl == "same_algo":
        baseline_cSeg = [tile_num]
        baseline_active_sm_count = (
            sm_count if args.baseline_same_algo_active_sms == "all" else active_sm_count
        )

        # Same CUTLASS SM90 Algo/BM/BN as overlap, but no overlap segmentation:
        # one full segment, then one full communication.
        baseline_dur = perf_running(
            m, n, k,
            data["BM"], data["BN"], data["Algo"],
            baseline_cSeg, data["hint"],
            comm_op,
            comm_backend,
            baseline_active_sm_count,
            reorder_map,
        )
    else:
        baseline_cSeg = ["cublas"]
        baseline_active_sm_count = "cublas"
        baseline_dur = perf_baseline(m, n, k, comm_op)

    speedup = baseline_dur / overlap_dur

    print(f"""
        {'Item':<20} {'Value':>15}
        {'-----':<20} {'-----':>15}
        {'m':<20} {m:>15}
        {'n':<20} {n:>15}
        {'k':<20} {k:>15}
        {'tile_num':<20} {tile_num:>15}
        {'cSeg':<20} {str(data["cSeg"]):>15}
        {'cSeg_len':<20} {len(data["cSeg"]):>15}
        {'active_sm_count':<20} {active_sm_count:>15}
        {'comm_sm_slack':<20} {comm_sm_slack:>15}
        {'baseline_impl':<20} {baseline_impl:>15}
        {'baseline_cSeg':<20} {str(baseline_cSeg):>15}
        {'baseline_sms':<20} {str(baseline_active_sm_count):>15}
        {'gemm_dur (ms)':<20} {gemm_dur:>15.4f}
        {'comm_dur (ms)':<20} {comm_dur:>15.4f}
        {'baseline_dur (ms)':<20} {baseline_dur:>15.4f}
        {'overlap_dur (ms)':<20} {overlap_dur:>15.4f}
        {'speedup':<20} {speedup:>15.4f}
        """)


if __name__ == "__main__":
    main()
