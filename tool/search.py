'''
    Using multiprocessing for distributed running,
    please specify the GPUs via CUDA_VISIBLE_DEVICES:
        e.g., CUDA_VISIBLE_DEVICES=0,1 python3 search_sm90_packed.py --m 4096 --n 8192 --k 4096 --comm_op all_reduce
'''

import torch
import argparse
import pandas as pd
import json
from pathlib import Path
import torch.multiprocessing as mp
import numpy as np
import importlib.util
import sys


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


def effective_compute_sms(sm_count: int, comm_sm_slack: int):
    assert comm_sm_slack >= 0, "comm_sm_slack must be >= 0"
    compute_sms = sm_count - comm_sm_slack
    assert compute_sms > 0, f"Invalid comm_sm_slack={comm_sm_slack} for sm_count={sm_count}"
    return compute_sms


def load_algo_dict():
    file_path = repo_root() / "configs" / "AlgoDictSm90.json"

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    ret = {}
    for item in data["algorithms"]:
        ret[int(item["algo"])] = item
    return ret


def is_cooperative_algo(Algo: int):
    algo_dict = load_algo_dict()
    item = algo_dict[int(Algo)]
    return item.get("mainloop", "") == "cooperative"


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
        # Do NOT reset monitor_order output, matching FlashOverlap behavior.
        MonitoredMatrix[: seg_size + TileNum + 1] = 0
    else:
        # Reset segment counters and per-tile epilogue counters.
        MonitoredMatrix[: seg_size + TileNum] = 0


def monitor_order_view(MonitoredMatrix, TileNum: int, seg_size: int):
    offset = seg_size + TileNum + 1
    return MonitoredMatrix[offset: offset + TileNum]

def gpu_config_name():
    device = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device)

    # Match gen_config_file naming:
    # "NVIDIA H100 NVL" -> "nvidia_h100_nvl"
    return props.name.lower().replace(" ", "_")


def solution_json_path(M: int, N: int, K: int):
    gpu_name = gpu_config_name()
    return repo_root() / "configs" / f"solution_m{M}n{N}k{K}_{gpu_name}_packed_sm90.json"


def shape_json_path(M: int, N: int, K: int, layout: str = "packed"):
    gpu_name = gpu_config_name()
    return repo_root() / "configs" / f"m{M}n{N}k{K}_{gpu_name}_{layout}_sm90.json"


def load_json(M: int, N: int, K: int):
    # Search uses packed output/reorder path, so prefer packed config.
    packed_path = shape_json_path(M, N, K, "packed")

    # Optional: prefer packed_custom if you intentionally generated custom configs.
    packed_custom_path = repo_root() / "configs" / f"m{M}n{N}k{K}_{gpu_config_name()}_packed_custom_sm90.json"

    if Path(packed_custom_path).exists():
        file_path = packed_custom_path
    elif Path(packed_path).exists():
        file_path = packed_path
    else:
        raise FileNotFoundError(
            "Could not find packed SM90 config file. Tried:\n"
            f"  {packed_custom_path}\n"
            f"  {packed_path}\n"
            "Run gen_config_file/profile for packed_sm90 first."
        )

    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"Loaded shape config: {file_path}")

    return data["BM"], data["BN"], data["dur"], data["Algo"]

def save_solution(M: int, N: int, K: int, BM: int, BN: int, gemm_dur: float, Algo: int, hint: list, cSeg: list, comm_sm_slack: int):
    out_path = solution_json_path(M, N, K)

    device = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device)
    sm_count = props.multi_processor_count
    compute_sms = effective_compute_sms(sm_count, comm_sm_slack)

    data = {
        "M": int(M),
        "N": int(N),
        "K": int(K),
        "hint": [int(x) for x in hint],
        "cSeg": [int(x) for x in cSeg],
        "rLDN": 1,
        "BM": int(BM),
        "BN": int(BN),
        "dur": float(gemm_dur),
        "Algo": int(Algo),
        "sm_count": int(sm_count),
        "comm_sm_slack": int(comm_sm_slack),
        "compute_sms": int(compute_sms),
        "source": "tool/search.py",
        "note": "Generated by SM90 packed overlap search. AlgoDictSm90.json is not modified."
    }

    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4)

    print(f"Solution saved to {out_path}")


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


def compute_hint_process(rank, world_size, nccl_id,
    M: int, N: int, K: int,
    BM: int, BN: int, Algo: list, wSize: int, comm_op: str,
    result_dict):

    TileNum = div_up(M, BM) * div_up(N, BN)
    WaveNum = div_up(TileNum, wSize)

    cSeg = []
    for i in range(WaveNum):
        this_seg = min(wSize, TileNum - i * wSize)
        cSeg = cSeg + [this_seg]

    cSeg_CPU = torch.tensor(cSeg, dtype=torch.int32)
    cSeg_GPU = cSeg_CPU.cuda(rank)

    torch.cuda.set_device(rank)

    gemm_class = ext.OverlapImpl()

    gemm_class.nccl_init(rank, world_size, nccl_id)
    gemm_class.cutlass_init()
    gemm_class.overlap_init()

    A = torch.empty((M, K), dtype=torch.float16, device="cuda").normal_(mean=0., std=0.5)
    B = torch.empty((N, K), dtype=torch.float16, device="cuda").normal_(mean=0., std=0.5)

    packed_M, packed_N = packed_shape(M, N, BM, BN, 1)
    C = torch.empty((packed_M, packed_N), dtype=torch.float16, device="cuda")

    MonitoredMatrix = torch.zeros((monitor_size(TileNum, len(cSeg), True),), dtype=torch.int, device="cuda")
    ReorderedArray = torch.arange(0, TileNum, dtype=torch.int, device="cuda").reshape(((M+BM-1)//BM, (N+BN-1)//BN))

    if comm_op == "reduce_scatter":
        D = torch.empty((M // world_size, N), dtype=torch.float16, device="cuda")
        RowArray = generate_row_remap_array(M, N, BM, BN, cSeg, world_size)

    _warm_up = 100
    _sample = 10

    if comm_op == "all_reduce":
        for _ in range(_warm_up):
            reset_monitor_matrix(MonitoredMatrix, TileNum, len(cSeg), True)
            gemm_class.gemm_allreduce_overlap(A, B, C, MonitoredMatrix, ReorderedArray, 1, cSeg_CPU, cSeg_GPU, Algo, True)

        samples = torch.empty((_sample, TileNum), dtype=torch.int, device="cuda")
        for i in range(_sample):
            reset_monitor_matrix(MonitoredMatrix, TileNum, len(cSeg), True)
            gemm_class.gemm_allreduce_overlap(A, B, C, MonitoredMatrix, ReorderedArray, 1, cSeg_CPU, cSeg_GPU, Algo, True)
            samples[i, :] = monitor_order_view(MonitoredMatrix, TileNum, len(cSeg))

    elif comm_op == "reduce_scatter":
        for _ in range(_warm_up):
            reset_monitor_matrix(MonitoredMatrix, TileNum, len(cSeg), True)
            gemm_class.gemm_reducescatter_overlap(A, B, C, D, MonitoredMatrix, ReorderedArray, RowArray, 1, cSeg_CPU, cSeg_GPU, Algo, True)

        samples = torch.empty((_sample, TileNum), dtype=torch.int, device="cuda")
        for i in range(_sample):
            reset_monitor_matrix(MonitoredMatrix, TileNum, len(cSeg), True)
            gemm_class.gemm_reducescatter_overlap(A, B, C, D, MonitoredMatrix, ReorderedArray, RowArray, 1, cSeg_CPU, cSeg_GPU, Algo, True)
            samples[i, :] = monitor_order_view(MonitoredMatrix, TileNum, len(cSeg))

    else:
        assert comm_op in ["all_reduce", "reduce_scatter"], \
            f"comm_op must be 'all_reduce' or 'reduce_scatter', but got '{comm_op}'"

    hint = []
    is_consistency = True
    for w in range(WaveNum):
        index = torch.where(((samples >= w * wSize) * (samples < (w + 1) * wSize)).sum(dim=0) == 10)

        if w < WaveNum - 1:
            if index[0].shape[0] < wSize:
                is_consistency = False
                break

        hint = hint + index[0].tolist()

    result_dict[rank] = (is_consistency, hint)


def compute_hint(M: int, N: int, K: int,
    BM: int, BN: int, Algo: list, wSize: int, comm_op: str):
    world_size = torch.cuda.device_count()
    if world_size < 2:
        raise RuntimeError("At least 2 GPUs are required for this program.")

    nccl_id = ext.generate_nccl_id()
    torch.cuda.synchronize()

    manager = mp.Manager()
    result_dict = manager.dict()

    mp.spawn(
            compute_hint_process,
            args=(world_size, nccl_id, M, N, K, BM, BN, Algo, wSize, comm_op, result_dict),
            nprocs=world_size
        )

    return result_dict[0]


def interpolate_latency(samples, x, comm_op):
    world_size = torch.cuda.device_count()
    # 确保输入是 PyTorch 张量
    if not isinstance(samples, torch.Tensor):
        samples = torch.tensor(samples, dtype=torch.float32)
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x, dtype=torch.float32)

    # 将数据转换为 NumPy 数组
    data_sizes = samples[:, 0].numpy()  # 数据量
    bandwidths = samples[:, 1].numpy()  # 带宽
    x_np = x.numpy()  # 需要插值的数据量

    # 使用 NumPy 的 interp 函数进行线性插值
    y_np = np.interp(x_np, data_sizes, bandwidths)

    # 将结果转换回 PyTorch 张量
    y = torch.tensor(y_np, dtype=torch.float32).item()

    # 使用 torch.interp 进行线性插值
    if comm_op == "all_reduce":
        latency = x * 2 * 2 * (world_size - 1) / y / (1024 ** 3)
    elif comm_op == "reduce_scatter":
        latency = x * 2 * (world_size - 1) / y / (1024 ** 3)

    return latency.item()


def predict_lat(M: int, N: int, gemm_dur: float,
    comm_array: torch.Tensor, gp: list, tile_num: int, comm_op: str,
    comm_sm_slack: int):

    device = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device)
    sm_count = props.multi_processor_count
    compute_sms = effective_compute_sms(sm_count, comm_sm_slack)

    acc_comm_dur = 0
    acc_comp_dur = 0
    iter_num = len(gp)

    if iter_num == 1:
        acc_comm_dur = interpolate_latency(comm_array, M*N // tile_num * gp[0], comm_op) + gemm_dur
        return acc_comm_dur

    old_wave_num = div_up(tile_num, sm_count)
    new_wave_num = div_up(tile_num, compute_sms)
    gemm_dur = gemm_dur / old_wave_num * new_wave_num

    for i in range(iter_num):
        if i == 0:
            comm_dur = 0
        else:
            comm_dur = interpolate_latency(comm_array, M*N // tile_num * gp[i - 1], comm_op)
        acc_comm_dur = max(acc_comp_dur, acc_comm_dur) + comm_dur
        acc_comp_dur += gemm_dur / new_wave_num * div_up(gp[i], compute_sms)
    acc_comm_dur = max(acc_comp_dur, acc_comm_dur) + interpolate_latency(comm_array, M*N // tile_num * gp[-1], comm_op)

    return acc_comm_dur


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


def perf_running_process(rank, world_size, nccl_id,
    M: int, N: int, K: int,
    BM: int, BN: int, Algo: int, cSeg: list, hint: list,
    comm_op: str,
    result_dict):

    cSeg_CPU = torch.tensor(cSeg, dtype=torch.int32)
    cSeg_GPU = cSeg_CPU.cuda(rank)

    TileNum = div_up(M, BM) * div_up(N, BN)

    torch.cuda.set_device(rank)

    gemm_class = ext.OverlapImpl()

    gemm_class.nccl_init(rank, world_size, nccl_id)
    gemm_class.cutlass_init()
    gemm_class.overlap_init()

    A = torch.empty((M, K), dtype=torch.float16, device="cuda").normal_(mean=0., std=0.5)
    B = torch.empty((N, K), dtype=torch.float16, device="cuda").normal_(mean=0., std=0.5)

    packed_M, packed_N = packed_shape(M, N, BM, BN, 1)
    C = torch.empty((packed_M, packed_N), dtype=torch.float16, device="cuda")

    MonitoredMatrix = torch.zeros((monitor_size(TileNum, len(cSeg), False),), dtype=torch.int, device="cuda")
    ReorderedArray = reorder_indices(TileNum, hint).reshape(((M+BM-1)//BM, (N+BN-1)//BN))

    if comm_op == "reduce_scatter":
        D = torch.empty((M // world_size, N), dtype=torch.float16, device="cuda")
        RowArray = generate_row_remap_array(M, N, BM, BN, cSeg, world_size)

    _warm_up = 20
    _freq = 200

    if len(cSeg) == 1:
        # No overlapping
        # In this SM90 packed port, we still use gemm_*_overlap with one segment.
        if comm_op == "all_reduce":
            for _ in range(_warm_up):
                reset_monitor_matrix(MonitoredMatrix, TileNum, len(cSeg), False)
                gemm_class.gemm_allreduce_overlap(A, B, C, MonitoredMatrix, ReorderedArray, 1, cSeg_CPU, cSeg_GPU, Algo, False)

            reset_monitor_matrix(MonitoredMatrix, TileNum, len(cSeg), False)
            gemm_class.gemm_allreduce_overlap(A, B, C, MonitoredMatrix, ReorderedArray, 1, cSeg_CPU, cSeg_GPU, Algo, False)

            start_event = [torch.cuda.Event(enable_timing=True) for i in range(_freq)]
            end_event = [torch.cuda.Event(enable_timing=True) for i in range(_freq)]
            for i in range(_freq):
                reset_monitor_matrix(MonitoredMatrix, TileNum, len(cSeg), False)
                start_event[i].record()
                gemm_class.gemm_allreduce_overlap(A, B, C, MonitoredMatrix, ReorderedArray, 1, cSeg_CPU, cSeg_GPU, Algo, False)
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
                gemm_class.gemm_allreduce_overlap(A, B, C, MonitoredMatrix, ReorderedArray, 1, cSeg_CPU, cSeg_GPU, Algo, False)

            start_event = [torch.cuda.Event(enable_timing=True) for i in range(_freq)]
            end_event = [torch.cuda.Event(enable_timing=True) for i in range(_freq)]
            for i in range(_freq):
                reset_monitor_matrix(MonitoredMatrix, TileNum, len(cSeg), False)
                start_event[i].record()
                gemm_class.gemm_allreduce_overlap(A, B, C, MonitoredMatrix, ReorderedArray, 1, cSeg_CPU, cSeg_GPU, Algo, False)
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
    cSeg: list, hint: list, comm_op: str):
    world_size = torch.cuda.device_count()
    if world_size < 2:
        raise RuntimeError("At least 2 GPUs are required for this program.")

    nccl_id = ext.generate_nccl_id()
    torch.cuda.synchronize()

    manager = mp.Manager()
    result_dict = manager.dict()

    mp.spawn(
            perf_running_process,
            args=(world_size, nccl_id, M, N, K, BM, BN, Algo, cSeg, hint, comm_op, result_dict),
            nprocs=world_size
        )

    dur = torch.empty((world_size))
    for i in range(world_size):
        dur[i] = result_dict[i]

    return dur.max()


def integer_partitions(n):
    result = []
    def helper(remaining, path):
        if remaining == 0:
            result.append(path)
            return
        for i in range(1, remaining + 1):
            helper(remaining - i, path + [i])
    helper(n, [])
    return result


def exhaustive_search(M: int, N: int, K: int, comm_op: str, comm_sm_slack: int):
    # load the .json file
    BM_list, BN_list, gemm_dur_list, Algo_list = load_json(M, N, K)

    # get the SM count
    device = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device)
    sm_count = props.multi_processor_count
    compute_sms = effective_compute_sms(sm_count, comm_sm_slack)

    print(f"SM count={sm_count}, comm_sm_slack={comm_sm_slack}, compute_sms={compute_sms}")

    hint = None
    for t in range(5):
        BM = BM_list[t]
        BN = BN_list[t]
        gemm_dur = gemm_dur_list[t]
        Algo = Algo_list[t]

        if is_cooperative_algo(Algo):
            print(f"Skip cooperative algo={Algo}")
            continue

        tile_num = div_up(M, BM) * div_up(N, BN)
        wave_num = div_up(tile_num, compute_sms)

        #compute hint
        result = compute_hint(M, N, K, BM, BN, Algo, compute_sms, comm_op)

        if result[0] == True:
            hint = result[1]
            break

    assert hint != None, "Tuning fails! Try to increase min_group_size manully."
    print("Start exhaustive searching.")

    min_dur = 1e5

    group_size_list = integer_partitions(wave_num)
    group_choice = len(group_size_list)
    for i in range(group_choice):
        gp = group_size_list[i]
        iter_num = len(gp)
        acc = 0
        for j in range(iter_num):
            if j < iter_num - 1:
                gp[j] = gp[j] * compute_sms
                acc += gp[j]
            else:
                gp[j] = min(gp[j] * compute_sms, tile_num - acc)
        dur = perf_running(M, N, K, BM, BN, Algo, gp, hint, comm_op)
        print(gp, "%.4f" % (dur))

        if dur < min_dur:
            min_dur = dur
            cSeg = gp

    print("Best solution: ", cSeg)
    save_solution(M, N, K, BM, BN, gemm_dur, Algo, hint, cSeg, comm_sm_slack)
    print("Solution saved.")


def fast_search(M: int, N: int, K: int, comm_array: torch.Tensor, comm_op: str,
    comm_sm_slack: int, min_group_size_override):
    # load the .json file
    BM_list, BN_list, gemm_dur_list, Algo_list = load_json(M, N, K)

    # get the SM count
    device = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device)
    sm_count = props.multi_processor_count
    compute_sms = effective_compute_sms(sm_count, comm_sm_slack)

    print(f"SM count={sm_count}, comm_sm_slack={comm_sm_slack}, compute_sms={compute_sms}")

    hint = None
    for t in range(10):
        BM = BM_list[t]
        BN = BN_list[t]
        gemm_dur = gemm_dur_list[t]
        Algo = Algo_list[t]

        if is_cooperative_algo(Algo):
            print(f"Skip cooperative algo={Algo}")
            continue

        tile_num = div_up(M, BM) * div_up(N, BN)
        wave_num = div_up(tile_num, compute_sms)

        if min_group_size_override is None:
            min_group_size = div_up(wave_num, 10)
        else:
            min_group_size = min_group_size_override

        #compute hint
        result = compute_hint(M, N, K, BM, BN, Algo, min_group_size * compute_sms, comm_op)

        if result[0] == True:
            hint = result[1]
            break

    assert hint != None, "Tuning fails! Try to increase min_group_size manully."
    print("Start predictive searching.")

    min_dur = 1e5
    normalized_wave_num = div_up(wave_num, min_group_size)
    group_size_list = integer_partitions(normalized_wave_num)

    group_choice = len(group_size_list)
    for i in range(group_choice):
        gp = group_size_list[i]
        iter_num = len(gp)
        acc = 0
        # avoid cold start
        if iter_num > 5 and gp[0] > 2:
            continue
        for j in range(iter_num):
            if j < iter_num - 1:
                gp[j] = gp[j] * compute_sms * min_group_size
                acc += gp[j]
            else:
                gp[j] = min(gp[j] * compute_sms * min_group_size, tile_num - acc)
        est_dur = predict_lat(M, N, gemm_dur, comm_array, gp, tile_num, comm_op, comm_sm_slack)

        if est_dur < min_dur:
            min_dur = est_dur
            cSeg = gp
    print("Search process finished.")

    searched_lat = perf_running(M, N, K, BM, BN, Algo, cSeg, hint, comm_op)
    print("Searched latency: %.4f" % searched_lat)
    print("Best solution: ", cSeg)
    save_solution(M, N, K, BM, BN, gemm_dur, Algo, hint, cSeg, comm_sm_slack)
    print("Solution saved.")


# Define the main function
def main():
    world_size = torch.cuda.device_count()

    # pass the problem size M, N, K via parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--m', type=int, default=4096)
    parser.add_argument('--k', type=int, default=8192)
    parser.add_argument('--n', type=int, default=8192)
    parser.add_argument('--comm_op', type=str, default='all_reduce')
    parser.add_argument('--predictive_search', action='store_true')
    parser.add_argument('--comm_sm_slack', type=int, default=2,
                        help='Number of SMs to leave as communication slack in the search model. Original FlashOverlap uses 2.')
    parser.add_argument('--min_group_size', type=int, default=None,
                        help='Override predictive-search min_group_size. If omitted, uses div_up(wave_num, 10).')
    args = parser.parse_args()

    # Force to use predictive search if the workload is large
    if args.predictive_search or args.m * args.n > 33554432:
        comm_array = torch.load(repo_root() / "configs" / f"bandwidth_{args.comm_op}_tp{world_size}.pt")
        print("Bandwidth curve captured.")
        fast_search(args.m, args.n, args.k, comm_array, args.comm_op,
                    args.comm_sm_slack, args.min_group_size)
    else:
        # compute the optimal solution
        exhaustive_search(args.m, args.n, args.k, args.comm_op,
                          args.comm_sm_slack)


if __name__ == "__main__":
    main()
