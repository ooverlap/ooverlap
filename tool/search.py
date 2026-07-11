#!/usr/bin/env python3
"""
Search SM90 packed overlap solutions.

Example:
  CUDA_VISIBLE_DEVICES=0,1 python tool/search.py \
    --m 4096 --n 8192 --k 4096 \
    --comm_op all_reduce \
    --comm_backend nccl \
    --predictive_search

This keeps the FlashOverlap-style search logic, but supports either NCCL or
ooverlap initialization for the segmented communication backend.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import uuid
from pathlib import Path

import numpy as np
import torch
import torch.multiprocessing as mp


DEFAULT_EXHAUSTIVE_ALGOS = 5
DEFAULT_PREDICTIVE_ALGOS = 10


def repo_root() -> Path:
    p = Path(__file__).resolve()
    for q in [p.parent, *p.parents]:
        if (q / "build" / "lib" / "ooverlap_ext.so").exists():
            return q
    return Path(__file__).resolve().parents[1]


def load_ooverlap_ext():
    so = repo_root() / "build" / "lib" / "ooverlap_ext.so"
    if not so.exists():
        raise FileNotFoundError(f"Could not find {so}. Build first.")

    name = "ooverlap_ext"
    if name in sys.modules:
        return sys.modules[name]

    spec = importlib.util.spec_from_file_location(name, str(so))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load extension from {so}")

    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


ext = load_ooverlap_ext()


def div_up(x: int, y: int) -> int:
    return (int(x) + int(y) - 1) // int(y)


def compute_sms_from_slack(sm_count: int, comm_sm_slack: int) -> int:
    return int(sm_count) - int(comm_sm_slack)


def make_broker_key() -> str:
    pid = format(os.getpid() & 0xFFFF, "04x")
    rnd = uuid.uuid4().hex[:8]
    return f"oo{pid}{rnd}"


def init_overlap_backend(gemm_class, rank, world_size, nccl_id, broker_key, comm_backend):
    if comm_backend == "nccl":
        gemm_class.nccl_init(rank, world_size, nccl_id)
    elif comm_backend == "ooverlap":
        gemm_class.ooverlap_ipc_init(rank, world_size, list(range(world_size)), broker_key)
    else:
        raise ValueError(f"Unknown comm_backend={comm_backend}")


def release_overlap_backend(gemm_class, comm_backend):
    if comm_backend == "ooverlap":
        gemm_class.ooverlap_release()


_ALGO_DICT_CACHE = None


def load_algo_dict():
    global _ALGO_DICT_CACHE
    if _ALGO_DICT_CACHE is not None:
        return _ALGO_DICT_CACHE

    path = repo_root() / "configs" / "AlgoDictSm90.json"
    data = json.loads(path.read_text())

    _ALGO_DICT_CACHE = {
        int(item["algo"]): item
        for item in data["algorithms"]
    }
    return _ALGO_DICT_CACHE


def is_cooperative_algo(algo: int) -> bool:
    item = load_algo_dict()[int(algo)]
    return item.get("mainloop", "") == "cooperative"


def normalize_loaded_candidates(BM_list, BN_list, gemm_dur_list, Algo_list):
    algo_dict = load_algo_dict()
    out_BM, out_BN, out_dur, out_Algo = [], [], [], []

    for BM, BN, dur, Algo in zip(BM_list, BN_list, gemm_dur_list, Algo_list):
        Algo = int(Algo)
        if Algo not in algo_dict:
            print(f"Skip unknown algo={Algo}")
            continue

        true_BM = int(algo_dict[Algo]["tile_m"])
        true_BN = int(algo_dict[Algo]["tile_n"])

        if int(BM) != true_BM or int(BN) != true_BN:
            print(
                f"Fix config tile mismatch for algo={Algo}: "
                f"config BM/BN={BM}x{BN}, algo BM/BN={true_BM}x{true_BN}"
            )

        out_BM.append(true_BM)
        out_BN.append(true_BN)
        out_dur.append(float(dur))
        out_Algo.append(Algo)

    return out_BM, out_BN, out_dur, out_Algo


def filter_candidates_by_algo_id(BM_list, BN_list, gemm_dur_list, Algo_list, algo_id):
    if algo_id is None:
        return BM_list, BN_list, gemm_dur_list, Algo_list

    algo_id = int(algo_id)
    out_BM, out_BN, out_dur, out_Algo = [], [], [], []

    for BM, BN, dur, Algo in zip(BM_list, BN_list, gemm_dur_list, Algo_list):
        if int(Algo) == algo_id:
            out_BM.append(BM)
            out_BN.append(BN)
            out_dur.append(dur)
            out_Algo.append(Algo)

    if not out_Algo:
        available = ", ".join(str(int(x)) for x in Algo_list)
        raise ValueError(
            f"Requested --algo_id {algo_id}, but it was not found. "
            f"Available algos: {available}"
        )

    print(f"Using requested algo_id={algo_id}.")
    return out_BM, out_BN, out_dur, out_Algo


def candidate_count(total: int, default_limit: int, algo_id) -> int:
    if algo_id is not None:
        return total
    return min(int(default_limit), int(total))


def gpu_config_name() -> str:
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    return props.name.lower().replace(" ", "_")


def shape_json_path(M: int, N: int, K: int, layout: str = "packed") -> Path:
    return repo_root() / "configs" / f"m{M}n{N}k{K}_{gpu_config_name()}_{layout}_sm90.json"


def solution_json_path(M: int, N: int, K: int, comm_backend: str) -> Path:
    return repo_root() / "configs" / f"solution_{comm_backend}_m{M}n{N}k{K}_{gpu_config_name()}_packed_sm90.json"


def legacy_solution_json_path(M: int, N: int, K: int) -> Path:
    return repo_root() / "configs" / f"solution_m{M}n{N}k{K}_{gpu_config_name()}_packed_sm90.json"


def bandwidth_curve_path(comm_backend: str, comm_op: str, world_size: int) -> Path:
    if comm_backend == "nccl":
        return repo_root() / "configs" / f"bandwidth_{comm_op}_tp{world_size}.pt"
    return repo_root() / "configs" / f"bandwidth_ooverlap_{comm_op}_tp{world_size}.pt"


def load_comm_array(comm_backend: str, comm_op: str, world_size: int, bandwidth_path: str = ""):
    path = Path(bandwidth_path) if bandwidth_path else bandwidth_curve_path(comm_backend, comm_op, world_size)

    if not path.exists():
        raise FileNotFoundError(
            f"Could not find bandwidth curve:\n"
            f"  {path}\n"
            f"Generate it first with tool/bandwidth.py."
        )

    print(f"Bandwidth curve captured from: {path}")
    return torch.load(path), path


def load_shape_config(M: int, N: int, K: int):
    path = shape_json_path(M, N, K, "packed")
    if not path.exists():
        raise FileNotFoundError(
            f"Could not find packed SM90 config file:\n"
            f"  {path}\n"
            f"Run the SM90 profiling config generator first."
        )

    data = json.loads(path.read_text())
    print(f"Loaded shape config: {path}")
    return normalize_loaded_candidates(data["BM"], data["BN"], data["dur"], data["Algo"])


def packed_shape(M: int, N: int, BM: int, BN: int, rLDN: int = 1):
    tile_num = div_up(M, BM) * div_up(N, BN)
    packed_tile_rows = div_up(tile_num, rLDN)
    return packed_tile_rows * BM, rLDN * BN


def monitor_size(tile_num: int, seg_size: int, monitor: bool):
    if monitor:
        return seg_size + tile_num + 1 + tile_num
    return seg_size + tile_num


def reset_monitor_matrix(MM, tile_num: int, seg_size: int, monitor: bool):
    if monitor:
        MM[: seg_size + tile_num + 1] = 0
    else:
        MM[: seg_size + tile_num] = 0


def monitor_order_view(MM, tile_num: int, seg_size: int):
    offset = seg_size + tile_num + 1
    return MM[offset: offset + tile_num]


def generate_row_remap_array(M, N, BM, BN, S_list, world_size, device="cuda"):
    total_tiles = (M * N) // (BM * BN)
    if sum(S_list) != total_tiles:
        raise RuntimeError("sum(S_list) must equal total number of tiles")

    original_row_ids = torch.arange(M * N // BN, dtype=torch.int, device=device)
    reordered_row_id = torch.empty_like(original_row_ids)

    current_row = 0
    for S in S_list:
        chunk_size = S * BM
        chunk_row_ids = original_row_ids[current_row: current_row + chunk_size]
        mod_values = chunk_row_ids % world_size
        _, idx = torch.sort(mod_values, stable=True)
        reordered_row_id[current_row: current_row + chunk_size] = chunk_row_ids[idx]
        current_row += chunk_size

    remap = torch.empty_like(original_row_ids)
    remap[reordered_row_id] = torch.arange(len(reordered_row_id), dtype=torch.int, device=device)
    return remap


def reorder_indices(tile_num: int, hint):
    new_order = [-1] * tile_num

    for i, tile in enumerate(hint):
        new_order[int(tile)] = i

    pos = len(hint)
    for tile in range(tile_num):
        if new_order[tile] < 0:
            new_order[tile] = pos
            pos += 1

    return torch.tensor(new_order, dtype=torch.int, device="cuda")


def build_full_order_and_reorder_map(tile_num: int, hint):
    used = [False] * tile_num
    stable_prefix = []

    for x in hint:
        x = int(x)
        if x < 0 or x >= tile_num:
            raise ValueError(f"Invalid tile id in hint: {x}")
        if not used[x]:
            stable_prefix.append(x)
            used[x] = True

    unstable_tail = [x for x in range(tile_num) if not used[x]]
    full_tile_order = stable_prefix + unstable_tail

    reorder_map = [-1] * tile_num
    for new_pos, tile_id in enumerate(full_tile_order):
        reorder_map[tile_id] = new_pos

    return full_tile_order, reorder_map, unstable_tail


def save_solution(
    M,
    N,
    K,
    BM,
    BN,
    gemm_dur,
    Algo,
    hint,
    cSeg,
    comm_sm_slack,
    comm_backend,
    comm_op,
    bandwidth_path="",
    predicted_latency_ms=None,
    searched_latency_ms=None,
    write_legacy_solution=False,
):
    out_path = solution_json_path(M, N, K, comm_backend)

    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    sm_count = int(props.multi_processor_count)
    compute_sms = compute_sms_from_slack(sm_count, comm_sm_slack)

    tile_num = div_up(M, BM) * div_up(N, BN)
    full_tile_order, reorder_map, unstable_tail = build_full_order_and_reorder_map(tile_num, hint)

    data = {
        "M": int(M),
        "N": int(N),
        "K": int(K),
        "comm_backend": str(comm_backend),
        "comm_op": str(comm_op),
        "bandwidth_path": str(bandwidth_path) if bandwidth_path else "",
        "selected_predicted_latency_ms": None if predicted_latency_ms is None else float(predicted_latency_ms),
        "searched_latency_ms": None if searched_latency_ms is None else float(searched_latency_ms),

        "hint": [int(x) for x in hint],
        "hint_stable_count": int(len(hint)),
        "unstable_tail_count": int(len(unstable_tail)),
        "unstable_tail": [int(x) for x in unstable_tail],
        "full_tile_order": [int(x) for x in full_tile_order],
        "reorder_map": [int(x) for x in reorder_map],

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
        "note": "Generated by SM90 packed overlap search. AlgoDictSm90.json is not modified.",
    }

    out_path.write_text(json.dumps(data, indent=4) + "\n")
    print(f"Solution saved to {out_path}")

    if write_legacy_solution:
        legacy_path = legacy_solution_json_path(M, N, K)
        legacy_path.write_text(json.dumps(data, indent=4) + "\n")
        print(f"Legacy solution saved to {legacy_path}")


def count_exact_str(counts, sample_count: int, min_count: int = 2):
    return " ".join(
        f"{c}/{sample_count}:{int((counts == c).sum().item())}"
        for c in range(sample_count, min_count - 1, -1)
    )


def count_at_least_str(counts, sample_count: int, min_count: int = 2):
    return " ".join(
        f">={c}/{sample_count}:{int((counts >= c).sum().item())}"
        for c in range(sample_count, min_count - 1, -1)
    )


def neighbor_membership_str(count_in_window, prev_count, next_count, exact_c, sample_count, max_patterns=12):
    tiles = torch.where(count_in_window == exact_c)[0]
    if int(tiles.numel()) == 0:
        return ""

    prev_hits = prev_count[tiles]
    next_hits = next_count[tiles]

    prev_any = int((prev_hits > 0).sum().item())
    next_any = int((next_hits > 0).sum().item())
    both_any = int(((prev_hits > 0) & (next_hits > 0)).sum().item())
    prev_only = int(((prev_hits > 0) & (next_hits == 0)).sum().item())
    next_only = int(((prev_hits == 0) & (next_hits > 0)).sum().item())
    neither = int(((prev_hits == 0) & (next_hits == 0)).sum().item())

    pattern_counts = {}
    for tile in tiles.tolist():
        p = int(prev_count[tile].item())
        n = int(next_count[tile].item())
        other = sample_count - exact_c - p - n
        key = (p, n, other)
        pattern_counts[key] = pattern_counts.get(key, 0) + 1

    patterns = sorted(pattern_counts.items(), key=lambda kv: (-kv[1], -kv[0][0], -kv[0][1], kv[0][2]))
    pattern_str = " | ".join(
        f"prev={p} next={n} other={o}:{num}"
        for (p, n, o), num in patterns[:max_patterns]
    )
    if len(patterns) > max_patterns:
        pattern_str += f" | ... +{len(patterns) - max_patterns} more"

    return (
        f"current={exact_c}/{sample_count}: tiles={int(tiles.numel())} "
        f"prev_any={prev_any} next_any={next_any} both={both_any} "
        f"prev_only={prev_only} next_only={next_only} neither_adjacent={neither} "
        f"patterns[{pattern_str}]"
    )


def debug_hint_samples(samples, tile_num: int, wSize: int, rank: int, Algo: int, BM: int, BN: int, label: str = ""):
    if rank != 0:
        return

    samples_cpu = samples.detach().cpu()
    sample_count = samples_cpu.shape[0]
    wave_num = div_up(tile_num, wSize)

    print("")
    print("========================================")
    print("DEBUG compute_hint samples")
    print(f"label={label}")
    print(f"rank={rank}")
    print(f"Algo={Algo} BM={BM} BN={BN}")
    print(f"TileNum={tile_num} wSize={wSize} WaveNum={wave_num} sample_count={sample_count}")
    print("========================================")

    for i in range(sample_count):
        row = samples_cpu[i]
        unique_count = int(torch.unique(row).numel())
        print(
            f"sample[{i}] min={int(row.min().item())} max={int(row.max().item())} "
            f"unique={unique_count}/{tile_num} duplicate_or_missing={tile_num - unique_count}"
        )

    print("----------------------------------------")
    print("Per-window membership counts:")

    total_exact = 0
    for w in range(wave_num):
        lo = w * wSize
        hi = min((w + 1) * wSize, tile_num)

        in_window_count = ((samples_cpu >= lo) & (samples_cpu < hi)).sum(dim=0)
        exact = torch.where(in_window_count == sample_count)[0]
        total_exact += int(exact.numel())

        print(f"window {w:03d}: range=[{lo}, {hi}) expected={hi - lo}")
        print(f"  exact:      {count_exact_str(in_window_count, sample_count)}")
        print(f"  cumulative: {count_at_least_str(in_window_count, sample_count)}")

        prev_count = torch.zeros_like(in_window_count)
        next_count = torch.zeros_like(in_window_count)

        if w > 0:
            prev_count = ((samples_cpu >= (w - 1) * wSize) & (samples_cpu < w * wSize)).sum(dim=0)

        if w + 1 < wave_num:
            next_count = ((samples_cpu >= (w + 1) * wSize) & (samples_cpu < min((w + 2) * wSize, tile_num))).sum(dim=0)

        lines = [
            neighbor_membership_str(in_window_count, prev_count, next_count, c, sample_count)
            for c in (4, 3, 2)
        ]
        lines = [x for x in lines if x]
        if lines:
            print("  neighbor distribution for low-count current-window tiles:")
            for line in lines:
                print(f"    {line}")

    print(f"total exact-stable tiles collected={total_exact}/{tile_num}")
    print("----------------------------------------")
    print("Boundary instability check:")

    for w in range(1, wave_num):
        boundary = w * wSize
        before_count = (samples_cpu < boundary).sum(dim=0)
        crosses = torch.where((before_count > 0) & (before_count < sample_count))[0]

        if crosses.numel() == 0:
            print(f"boundary {w:03d} at order={boundary}: crossing_tiles=0")
            continue

        hist = []
        crossing_before_count = before_count[crosses]
        for c in range(sample_count - 1, 0, -1):
            n = int((crossing_before_count == c).sum().item())
            hist.append(f"left={c}/{sample_count}:right={sample_count-c}/{sample_count}:{n}")

        tile_ids = crosses[:20].tolist()
        print(f"boundary {w:03d} at order={boundary}: crossing_tiles={int(crosses.numel())}")
        print("  crossing histogram:", " ".join(hist))
        print(f"  first20={tile_ids}")

    print("========================================")
    print("")


def collect_monitor_samples(
    gemm_class,
    M,
    N,
    K,
    BM,
    BN,
    Algo,
    tile_num,
    cSeg,
    cSeg_CPU,
    cSeg_GPU,
    A,
    B,
    C,
    MM,
    RA,
    comm_op,
    sample_count,
    active_sm_count,
    D=None,
    RowArray=None,
):
    samples = torch.empty((sample_count, tile_num), dtype=torch.int, device="cuda")

    for i in range(sample_count):
        reset_monitor_matrix(MM, tile_num, len(cSeg), True)

        if comm_op == "all_reduce":
            gemm_class.gemm_allreduce_overlap(
                A,
                B,
                C,
                MM,
                RA,
                1,
                cSeg_CPU,
                cSeg_GPU,
                Algo,
                int(active_sm_count),
                True,
            )
        elif comm_op == "reduce_scatter":
            gemm_class.gemm_reducescatter_overlap(
                A,
                B,
                C,
                D,
                MM,
                RA,
                RowArray,
                1,
                cSeg_CPU,
                cSeg_GPU,
                Algo,
                True,
            )
        else:
            raise ValueError(f"Unknown comm_op={comm_op}")

        samples[i, :] = monitor_order_view(MM, tile_num, len(cSeg))

    return samples


def compute_stable_hint_from_rounds(
    sample_rounds,
    tile_num: int,
    wSize: int,
    compute_sms: int,
    nominal_min_group_size: int,
    min_effective_group_size: int,
    rank: int,
):
    """
    Exact-stable effective group logic.

    A tile is stable for a window only if it lands in that same nominal window
    in every sample of every confirmation round.

    The effective group is:
        min over full windows of floor(stable_count / compute_sms)

    This is the old behavior that worked: the nominal window stays fixed, and
    effective_group describes how many waves inside that same window were stable.
    """
    if not sample_rounds:
        raise RuntimeError("sample_rounds is empty")

    wave_num = div_up(tile_num, wSize)
    device = sample_rounds[0].device

    stable_per_window = []
    effective_group = int(nominal_min_group_size)
    full_window_seen = False

    if rank == 0:
        print("")
        print("----------------------------------------")
        print("Effective hint stability summary:")
        print(
            f"nominal_min_group_size={nominal_min_group_size} "
            f"nominal_wSize={wSize} compute_sms={compute_sms} "
            f"min_effective_group_size={min_effective_group_size}"
        )
        print(
            "A tile is final-stable for a window only if it is exact-stable "
            "in that window for every confirmation round."
        )

    for w in range(wave_num):
        lo = w * wSize
        hi = min((w + 1) * wSize, tile_num)
        expected = hi - lo

        stable_mask = torch.ones((tile_num,), dtype=torch.bool, device=device)
        round_exact_counts = []

        for samples in sample_rounds:
            sample_count = samples.shape[0]
            count_in_window = ((samples >= lo) & (samples < hi)).sum(dim=0)
            exact_mask = count_in_window == sample_count
            stable_mask &= exact_mask
            round_exact_counts.append(int(exact_mask.sum().item()))

        stable_tiles = torch.where(stable_mask)[0]
        stable_count = int(stable_tiles.numel())
        stable_per_window.append(stable_tiles)

        window_effective_group = min(stable_count // compute_sms, nominal_min_group_size)

        if expected == wSize:
            full_window_seen = True
            effective_group = min(effective_group, window_effective_group)

        if rank == 0:
            round_counts = " ".join(
                f"round{idx}_exact={cnt}"
                for idx, cnt in enumerate(round_exact_counts)
            )
            print(
                f"window {w:03d}: range=[{lo}, {hi}) expected={expected} "
                f"{round_counts} final_stable={stable_count} "
                f"window_effective_group={window_effective_group}"
            )

    if not full_window_seen:
        effective_group = min(nominal_min_group_size, div_up(tile_num, compute_sms))

    if rank == 0:
        print(
            f"chosen effective_min_group_size={effective_group} "
            f"(requires >= {min_effective_group_size})"
        )

    if effective_group < min_effective_group_size:
        if rank == 0:
            print(
                f"compute_hint rejected: effective_min_group_size={effective_group} "
                f"is below min_effective_group_size={min_effective_group_size}"
            )
        return False, [], effective_group

    hint = []
    used = torch.zeros((tile_num,), dtype=torch.bool, device=device)

    for stable_tiles in stable_per_window:
        for tile in stable_tiles.tolist():
            if not bool(used[tile].item()):
                hint.append(int(tile))
                used[tile] = True

    if rank == 0:
        print(
            f"hint stable tiles={len(hint)}/{tile_num}; "
            f"unstable tiles left for reorder tail={tile_num - len(hint)}"
        )
        print("----------------------------------------")
        print("")

    return True, hint, effective_group


def compute_hint_process(
    rank,
    world_size,
    nccl_id,
    broker_key,
    comm_backend,
    M,
    N,
    K,
    BM,
    BN,
    Algo,
    wSize,
    comm_op,
    compute_sms,
    nominal_min_group_size,
    min_effective_group_size,
    effective_hint_confirm,
    debug_hint,
    debug_hint_dump,
    barrier,
    result_dict,
):
    torch.cuda.set_device(rank)

    tile_num = div_up(M, BM) * div_up(N, BN)
    wave_num = div_up(tile_num, wSize)
    cSeg = [min(wSize, tile_num - i * wSize) for i in range(wave_num)]

    cSeg_CPU = torch.tensor(cSeg, dtype=torch.int32)
    cSeg_GPU = cSeg_CPU.cuda(rank)

    gemm_class = ext.OverlapImpl()
    init_overlap_backend(gemm_class, rank, world_size, nccl_id, broker_key, comm_backend)
    gemm_class.cutlass_init()
    gemm_class.overlap_init()

    torch.cuda.synchronize()
    barrier.wait()

    try:
        A = torch.empty((M, K), dtype=torch.float16, device="cuda").normal_(mean=0.0, std=0.5)
        B = torch.empty((N, K), dtype=torch.float16, device="cuda").normal_(mean=0.0, std=0.5)

        packed_M, packed_N = packed_shape(M, N, BM, BN, 1)
        C = torch.empty((packed_M, packed_N), dtype=torch.float16, device="cuda")

        MM = torch.zeros((monitor_size(tile_num, len(cSeg), True),), dtype=torch.int, device="cuda")
        RA = torch.arange(0, tile_num, dtype=torch.int, device="cuda").reshape(
            (div_up(M, BM), div_up(N, BN))
        )

        D = None
        RowArray = None
        if comm_op == "reduce_scatter":
            D = torch.empty((M // world_size, N), dtype=torch.float16, device="cuda")
            RowArray = generate_row_remap_array(M, N, BM, BN, cSeg, world_size)

        warmup = 100
        sample_count = 10

        for _ in range(warmup):
            reset_monitor_matrix(MM, tile_num, len(cSeg), True)
            if comm_op == "all_reduce":
                gemm_class.gemm_allreduce_overlap(
                    A, B, C, MM, RA, 1, cSeg_CPU, cSeg_GPU, Algo, int(compute_sms), True
                )
            elif comm_op == "reduce_scatter":
                gemm_class.gemm_reducescatter_overlap(
                    A, B, C, D, MM, RA, RowArray, 1, cSeg_CPU, cSeg_GPU, Algo, True
                )
            else:
                raise ValueError(f"Unknown comm_op={comm_op}")

        samples_first = collect_monitor_samples(
            gemm_class,
            M,
            N,
            K,
            BM,
            BN,
            Algo,
            tile_num,
            cSeg,
            cSeg_CPU,
            cSeg_GPU,
            A,
            B,
            C,
            MM,
            RA,
            comm_op,
            sample_count,
            int(compute_sms),
            D=D,
            RowArray=RowArray,
        )

        torch.cuda.synchronize()
        sample_rounds = [samples_first]

        if debug_hint:
            debug_hint_samples(samples_first, tile_num, wSize, rank, Algo, BM, BN, f"M{M}N{N}K{K}_round0")

        if effective_hint_confirm:
            samples_confirm = collect_monitor_samples(
                gemm_class,
                M,
                N,
                K,
                BM,
                BN,
                Algo,
                tile_num,
                cSeg,
                cSeg_CPU,
                cSeg_GPU,
                A,
                B,
                C,
                MM,
                RA,
                comm_op,
                sample_count,
                int(compute_sms),
                D=D,
                RowArray=RowArray,
            )

            torch.cuda.synchronize()
            sample_rounds.append(samples_confirm)

            if debug_hint:
                combined = torch.cat([samples_first, samples_confirm], dim=0)
                debug_hint_samples(combined, tile_num, wSize, rank, Algo, BM, BN, f"M{M}N{N}K{K}_combined")

        if debug_hint_dump and rank == 0:
            dump_path = Path(debug_hint_dump)
            dump_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "samples": samples_first.detach().cpu(),
                    "sample_rounds": [x.detach().cpu() for x in sample_rounds],
                    "TileNum": tile_num,
                    "WaveNum": wave_num,
                    "wSize": wSize,
                    "M": M,
                    "N": N,
                    "K": K,
                    "BM": BM,
                    "BN": BN,
                    "Algo": int(Algo),
                    "comm_backend": comm_backend,
                    "comm_op": comm_op,
                    "compute_sms": int(compute_sms),
                    "nominal_min_group_size": int(nominal_min_group_size),
                    "min_effective_group_size": int(min_effective_group_size),
                    "effective_hint_confirm": bool(effective_hint_confirm),
                },
                dump_path,
            )
            print(f"DEBUG hint samples dumped to: {dump_path}")

        result_dict[rank] = compute_stable_hint_from_rounds(
            sample_rounds,
            tile_num,
            wSize,
            compute_sms,
            nominal_min_group_size,
            min_effective_group_size,
            rank,
        )

    finally:
        torch.cuda.synchronize()
        barrier.wait()
        release_overlap_backend(gemm_class, comm_backend)
        torch.cuda.synchronize()
        barrier.wait()


def compute_hint(
    M,
    N,
    K,
    BM,
    BN,
    Algo,
    wSize,
    comm_op,
    compute_sms,
    nominal_min_group_size,
    min_effective_group_size,
    comm_backend,
    effective_hint_confirm=False,
    debug_hint=False,
    debug_hint_dump="",
):
    world_size = torch.cuda.device_count()
    if world_size < 2:
        raise RuntimeError("At least 2 GPUs are required.")

    nccl_id = ext.generate_nccl_id() if comm_backend == "nccl" else []
    broker_key = make_broker_key() if comm_backend == "ooverlap" else ""

    manager = mp.Manager()
    result_dict = manager.dict()
    barrier = manager.Barrier(world_size)

    mp.spawn(
        compute_hint_process,
        args=(
            world_size,
            nccl_id,
            broker_key,
            comm_backend,
            M,
            N,
            K,
            BM,
            BN,
            Algo,
            wSize,
            comm_op,
            compute_sms,
            nominal_min_group_size,
            min_effective_group_size,
            effective_hint_confirm,
            debug_hint,
            debug_hint_dump,
            barrier,
            result_dict,
        ),
        nprocs=world_size,
    )

    return result_dict[0]


def interpolate_latency(samples, x, comm_op):
    world_size = torch.cuda.device_count()

    if not isinstance(samples, torch.Tensor):
        samples = torch.tensor(samples, dtype=torch.float32)
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x, dtype=torch.float32)

    data_sizes = samples[:, 0].detach().cpu().numpy()
    bandwidths = samples[:, 1].detach().cpu().numpy()
    x_np = x.detach().cpu().numpy()

    bw = torch.tensor(np.interp(x_np, data_sizes, bandwidths), dtype=torch.float32).item()

    if comm_op == "all_reduce":
        latency_sec = x * 2 * 2 * (world_size - 1) / bw / (1024 ** 3)
    elif comm_op == "reduce_scatter":
        latency_sec = x * 2 * (world_size - 1) / bw / (1024 ** 3)
    else:
        raise ValueError(f"Unknown comm_op={comm_op}")

    return latency_sec.item() * 1000.0


def predict_lat(M, N, gemm_dur, comm_array, gp, tile_num, comm_op, comm_sm_slack):
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    sm_count = props.multi_processor_count
    compute_sms = compute_sms_from_slack(sm_count, comm_sm_slack)

    if len(gp) == 1:
        return interpolate_latency(comm_array, M * N // tile_num * gp[0], comm_op) + gemm_dur

    old_wave_num = div_up(tile_num, sm_count)
    new_wave_num = div_up(tile_num, compute_sms)
    scaled_gemm_dur = gemm_dur / old_wave_num * new_wave_num

    acc_comm_dur = 0.0
    acc_comp_dur = 0.0

    for i in range(len(gp)):
        comm_dur = 0.0 if i == 0 else interpolate_latency(comm_array, M * N // tile_num * gp[i - 1], comm_op)
        acc_comm_dur = max(acc_comp_dur, acc_comm_dur) + comm_dur
        acc_comp_dur += scaled_gemm_dur / new_wave_num * div_up(gp[i], compute_sms)

    acc_comm_dur = max(acc_comp_dur, acc_comm_dur) + interpolate_latency(
        comm_array,
        M * N // tile_num * gp[-1],
        comm_op,
    )

    return acc_comm_dur


def predict_lat_with_debug(M, N, gemm_dur, comm_array, gp, tile_num, comm_op, comm_sm_slack):
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    sm_count = props.multi_processor_count
    compute_sms = compute_sms_from_slack(sm_count, comm_sm_slack)
    elems_per_tile = M * N // tile_num

    lines = [
        f"gp={gp} tile_num={tile_num} sm_count={sm_count} compute_sms={compute_sms} elems_per_tile={elems_per_tile}"
    ]

    if len(gp) == 1:
        comm_x = elems_per_tile * gp[0]
        comm_dur = interpolate_latency(comm_array, comm_x, comm_op)
        total = comm_dur + gemm_dur
        lines.append(
            f"one-segment: comm_x={comm_x} comm_dur={comm_dur:.6f} "
            f"gemm_dur={gemm_dur:.6f} total={total:.6f}"
        )
        return total, lines

    old_wave_num = div_up(tile_num, sm_count)
    new_wave_num = div_up(tile_num, compute_sms)
    scaled_gemm_dur = gemm_dur / old_wave_num * new_wave_num

    lines.append(
        f"multi-segment: old_wave_num={old_wave_num} new_wave_num={new_wave_num} "
        f"scaled_gemm_dur={scaled_gemm_dur:.6f}"
    )

    acc_comm_dur = 0.0
    acc_comp_dur = 0.0

    for i in range(len(gp)):
        if i == 0:
            comm_tiles = 0
            comm_x = 0
            comm_dur = 0.0
        else:
            comm_tiles = gp[i - 1]
            comm_x = elems_per_tile * comm_tiles
            comm_dur = interpolate_latency(comm_array, comm_x, comm_op)

        comp_waves = div_up(gp[i], compute_sms)
        comp_dur = scaled_gemm_dur / new_wave_num * comp_waves

        prev_comm = acc_comm_dur
        prev_comp = acc_comp_dur

        acc_comm_dur = max(acc_comp_dur, acc_comm_dur) + comm_dur
        acc_comp_dur += comp_dur

        lines.append(
            f"iter={i}: seg_tiles={gp[i]} comp_waves={comp_waves} comp_dur={comp_dur:.6f} "
            f"prev_comp={prev_comp:.6f} prev_comm={prev_comm:.6f} "
            f"comm_tiles_from_prev={comm_tiles} comm_x={comm_x} comm_dur={comm_dur:.6f} "
            f"new_comp={acc_comp_dur:.6f} new_comm={acc_comm_dur:.6f}"
        )

    final_x = elems_per_tile * gp[-1]
    final_comm = interpolate_latency(comm_array, final_x, comm_op)
    total = max(acc_comp_dur, acc_comm_dur) + final_comm

    lines.append(
        f"final_comm: tiles={gp[-1]} comm_x={final_x} final_comm={final_comm:.6f} total={total:.6f}"
    )

    return total, lines


def perf_running_process(
    rank,
    world_size,
    nccl_id,
    broker_key,
    comm_backend,
    M,
    N,
    K,
    BM,
    BN,
    Algo,
    cSeg,
    hint,
    comm_op,
    active_sm_count,
    barrier,
    result_dict,
):
    torch.cuda.set_device(rank)

    cSeg_CPU = torch.tensor(cSeg, dtype=torch.int32)
    cSeg_GPU = cSeg_CPU.cuda(rank)
    tile_num = div_up(M, BM) * div_up(N, BN)

    gemm_class = ext.OverlapImpl()
    init_overlap_backend(gemm_class, rank, world_size, nccl_id, broker_key, comm_backend)
    gemm_class.cutlass_init()
    gemm_class.overlap_init()

    torch.cuda.synchronize()
    barrier.wait()

    try:
        A = torch.empty((M, K), dtype=torch.float16, device="cuda").normal_(mean=0.0, std=0.5)
        B = torch.empty((N, K), dtype=torch.float16, device="cuda").normal_(mean=0.0, std=0.5)

        packed_M, packed_N = packed_shape(M, N, BM, BN, 1)
        C = torch.empty((packed_M, packed_N), dtype=torch.float16, device="cuda")

        MM = torch.zeros((monitor_size(tile_num, len(cSeg), False),), dtype=torch.int, device="cuda")
        RA = reorder_indices(tile_num, hint).reshape((div_up(M, BM), div_up(N, BN)))

        D = None
        RowArray = None
        if comm_op == "reduce_scatter":
            D = torch.empty((M // world_size, N), dtype=torch.float16, device="cuda")
            RowArray = generate_row_remap_array(M, N, BM, BN, cSeg, world_size)

        warmup = 20
        iters = 200

        for _ in range(warmup):
            reset_monitor_matrix(MM, tile_num, len(cSeg), False)
            if comm_op == "all_reduce":
                gemm_class.gemm_allreduce_overlap(
                    A, B, C, MM, RA, 1, cSeg_CPU, cSeg_GPU, Algo, int(active_sm_count), False
                )
            elif comm_op == "reduce_scatter":
                gemm_class.gemm_reducescatter_overlap(
                    A, B, C, D, MM, RA, RowArray, 1, cSeg_CPU, cSeg_GPU, Algo, False
                )
            else:
                raise ValueError(f"Unknown comm_op={comm_op}")

        starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]

        for i in range(iters):
            reset_monitor_matrix(MM, tile_num, len(cSeg), False)
            starts[i].record()

            if comm_op == "all_reduce":
                gemm_class.gemm_allreduce_overlap(
                    A, B, C, MM, RA, 1, cSeg_CPU, cSeg_GPU, Algo, int(active_sm_count), False
                )
            elif comm_op == "reduce_scatter":
                gemm_class.gemm_reducescatter_overlap(
                    A, B, C, D, MM, RA, RowArray, 1, cSeg_CPU, cSeg_GPU, Algo, False
                )

            ends[i].record()

        torch.cuda.synchronize()
        dur = torch.tensor([s.elapsed_time(e) for s, e in zip(starts, ends)], dtype=torch.float)
        result_dict[rank] = torch.mean(dur).item()

    finally:
        torch.cuda.synchronize()
        barrier.wait()
        release_overlap_backend(gemm_class, comm_backend)
        torch.cuda.synchronize()
        barrier.wait()


def perf_running(M, N, K, BM, BN, Algo, cSeg, hint, comm_op, active_sm_count, comm_backend):
    world_size = torch.cuda.device_count()
    if world_size < 2:
        raise RuntimeError("At least 2 GPUs are required.")

    nccl_id = ext.generate_nccl_id() if comm_backend == "nccl" else []
    broker_key = make_broker_key() if comm_backend == "ooverlap" else ""

    manager = mp.Manager()
    result_dict = manager.dict()
    barrier = manager.Barrier(world_size)

    mp.spawn(
        perf_running_process,
        args=(
            world_size,
            nccl_id,
            broker_key,
            comm_backend,
            M,
            N,
            K,
            BM,
            BN,
            Algo,
            cSeg,
            hint,
            comm_op,
            active_sm_count,
            barrier,
            result_dict,
        ),
        nprocs=world_size,
    )

    dur = torch.empty((world_size,))
    for i in range(world_size):
        dur[i] = result_dict[i]

    return dur.max()


def integer_partitions(n: int):
    out = []

    def rec(left, path):
        if left == 0:
            out.append(path)
            return
        for x in range(1, left + 1):
            rec(left - x, path + [x])

    rec(int(n), [])
    return out


def expand_partition_to_cseg(gp_units, compute_sms: int, search_group_size: int, tile_num: int):
    gp = list(gp_units)
    acc = 0

    for i in range(len(gp)):
        if i < len(gp) - 1:
            gp[i] = gp[i] * compute_sms * search_group_size
            acc += gp[i]
        else:
            gp[i] = min(gp[i] * compute_sms * search_group_size, tile_num - acc)

    return gp


def choose_candidate(
    M,
    N,
    K,
    comm_op,
    comm_sm_slack,
    comm_backend,
    predictive,
    min_group_size_override,
    min_effective_group_size_override,
    effective_hint_confirm,
    debug_hint,
    debug_hint_dump,
    algo_id,
):
    BM_list, BN_list, gemm_dur_list, Algo_list = load_shape_config(M, N, K)
    BM_list, BN_list, gemm_dur_list, Algo_list = filter_candidates_by_algo_id(
        BM_list, BN_list, gemm_dur_list, Algo_list, algo_id
    )

    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    sm_count = int(props.multi_processor_count)
    compute_sms = compute_sms_from_slack(sm_count, comm_sm_slack)

    print(f"comm_backend={comm_backend}")
    print(f"SM count={sm_count}, comm_sm_slack={comm_sm_slack}, compute_sms={compute_sms}")

    limit = DEFAULT_PREDICTIVE_ALGOS if predictive else DEFAULT_EXHAUSTIVE_ALGOS
    tries = candidate_count(len(Algo_list), limit, algo_id)

    print(f"Trying {tries}/{len(Algo_list)} candidate algos.")

    for t in range(tries):
        BM = int(BM_list[t])
        BN = int(BN_list[t])
        gemm_dur = float(gemm_dur_list[t])
        Algo = int(Algo_list[t])

        # if is_cooperative_algo(Algo):
            # print(f"Skip cooperative algo={Algo}")
            # continue

        tile_num = div_up(M, BM) * div_up(N, BN)
        wave_num = div_up(tile_num, compute_sms)

        if predictive:
            min_group_size = div_up(wave_num, 10) if min_group_size_override is None else int(min_group_size_override)
            min_effective_group_size = (
                max(1, min_group_size - 3)
                if min_effective_group_size_override is None
                else int(min_effective_group_size_override)
            )
            wSize = min_group_size * compute_sms
        else:
            min_group_size = 1
            min_effective_group_size = 1
            wSize = compute_sms

        print(
            f"Try candidate {t + 1}/{tries}: algo={Algo} BM={BM} BN={BN} "
            f"tile_num={tile_num} wave_num={wave_num} "
            f"min_group_size={min_group_size} min_effective_group_size={min_effective_group_size}"
        )

        dump = ""
        if debug_hint_dump:
            dump = debug_hint_dump.replace(".pt", f"_{comm_backend}_algo{Algo}_bm{BM}_bn{BN}.pt")

        try:
            ok, hint, effective_min_group_size = compute_hint(
                M,
                N,
                K,
                BM,
                BN,
                Algo,
                wSize,
                comm_op,
                compute_sms,
                min_group_size,
                min_effective_group_size,
                comm_backend,
                effective_hint_confirm=effective_hint_confirm if predictive else False,
                debug_hint=debug_hint,
                debug_hint_dump=dump,
            )
        except Exception as e:
            print(f"compute_hint failed for algo={Algo}; trying next candidate.")
            print(f"  {type(e).__name__}: {e}")
            continue

        if ok:
            print(
                f"Selected algo={Algo}. "
                f"nominal_min_group_size={min_group_size} "
                f"effective_min_group_size={effective_min_group_size}"
            )
            return {
                "BM": BM,
                "BN": BN,
                "gemm_dur": gemm_dur,
                "Algo": Algo,
                "tile_num": tile_num,
                "wave_num": wave_num,
                "hint": hint,
                "effective_min_group_size": int(effective_min_group_size),
                "selected_min_group_size": int(min_group_size),
                "compute_sms": int(compute_sms),
            }

        print(f"compute_hint inconsistent for algo={Algo}; trying next candidate.")

    raise RuntimeError("Tuning failed. Try a specific --algo_id or adjust --min_group_size.")


def exhaustive_search(
    M,
    N,
    K,
    comm_op,
    comm_sm_slack,
    comm_backend,
    debug_hint=False,
    debug_hint_dump="",
    algo_id=None,
    write_legacy_solution=False,
):
    selected = choose_candidate(
        M,
        N,
        K,
        comm_op,
        comm_sm_slack,
        comm_backend,
        predictive=False,
        min_group_size_override=None,
        min_effective_group_size_override=None,
        effective_hint_confirm=False,
        debug_hint=debug_hint,
        debug_hint_dump=debug_hint_dump,
        algo_id=algo_id,
    )

    BM = selected["BM"]
    BN = selected["BN"]
    Algo = selected["Algo"]
    gemm_dur = selected["gemm_dur"]
    tile_num = selected["tile_num"]
    wave_num = selected["wave_num"]
    hint = selected["hint"]
    compute_sms = selected["compute_sms"]

    print("Start exhaustive searching.")

    min_dur = 1e100
    best_cSeg = None

    for gp_units in integer_partitions(wave_num):
        gp = expand_partition_to_cseg(gp_units, compute_sms, 1, tile_num)
        dur = perf_running(M, N, K, BM, BN, Algo, gp, hint, comm_op, compute_sms, comm_backend)

        print(gp, "%.4f" % dur)

        if float(dur) < min_dur:
            min_dur = float(dur)
            best_cSeg = gp

    print("Best solution:", best_cSeg)

    save_solution(
        M,
        N,
        K,
        BM,
        BN,
        gemm_dur,
        Algo,
        hint,
        best_cSeg,
        comm_sm_slack,
        comm_backend,
        comm_op,
        searched_latency_ms=min_dur,
        write_legacy_solution=write_legacy_solution,
    )


def fast_search(
    M,
    N,
    K,
    comm_array,
    comm_op,
    comm_sm_slack,
    min_group_size_override,
    comm_backend,
    bandwidth_path="",
    debug_hint=False,
    debug_hint_dump="",
    algo_id=None,
    min_effective_group_size_override=None,
    effective_hint_confirm=True,
    debug_search=False,
    debug_search_topk=20,
    write_legacy_solution=False,
):
    selected = choose_candidate(
        M,
        N,
        K,
        comm_op,
        comm_sm_slack,
        comm_backend,
        predictive=True,
        min_group_size_override=min_group_size_override,
        min_effective_group_size_override=min_effective_group_size_override,
        effective_hint_confirm=effective_hint_confirm,
        debug_hint=debug_hint,
        debug_hint_dump=debug_hint_dump,
        algo_id=algo_id,
    )

    BM = selected["BM"]
    BN = selected["BN"]
    Algo = selected["Algo"]
    gemm_dur = selected["gemm_dur"]
    tile_num = selected["tile_num"]
    wave_num = selected["wave_num"]
    hint = selected["hint"]
    compute_sms = selected["compute_sms"]
    search_group_size = selected["effective_min_group_size"]

    print(
        "Start predictive searching. "
        f"Using effective_min_group_size={search_group_size} "
        f"instead of nominal_min_group_size={selected['selected_min_group_size']}."
    )

    normalized_wave_num = div_up(wave_num, search_group_size)
    partitions = integer_partitions(normalized_wave_num)

    print(
        f"wave_num={wave_num} normalized_wave_num={normalized_wave_num} "
        f"compute_sms={compute_sms} search_group_size={search_group_size} "
        f"partition_count={len(partitions)}"
    )

    min_dur = 1e100
    best_cSeg = None
    debug_rows = []
    skipped_cold_start = 0
    evaluated = 0

    for gp_units in partitions:
        gp_units = list(gp_units)
        iter_num = len(gp_units)

        if iter_num > 5 and gp_units[0] > 2:
            skipped_cold_start += 1
            continue

        cSeg = expand_partition_to_cseg(gp_units, compute_sms, search_group_size, tile_num)

        if debug_search:
            est_dur, trace = predict_lat_with_debug(
                M, N, gemm_dur, comm_array, cSeg, tile_num, comm_op, comm_sm_slack
            )
        else:
            est_dur = predict_lat(M, N, gemm_dur, comm_array, cSeg, tile_num, comm_op, comm_sm_slack)
            trace = []

        evaluated += 1
        debug_rows.append({
            "est_dur": float(est_dur),
            "gp_units": gp_units,
            "cSeg": cSeg,
            "iter_num": iter_num,
            "trace": trace,
        })

        if float(est_dur) < min_dur:
            min_dur = float(est_dur)
            best_cSeg = cSeg
            if debug_search:
                print(f"New best: est={float(est_dur):.6f} gp_units={gp_units} cSeg={cSeg}")

    if best_cSeg is None:
        raise RuntimeError("Predictive search produced no candidate cSeg.")

    if debug_search:
        print("")
        print("----------------------------------------")
        print("Predictive search debug summary:")
        print(f"partition_count={len(partitions)} evaluated_count={evaluated} skipped_cold_start={skipped_cold_start}")
        print(f"selected_cSeg={best_cSeg} selected_predicted_latency={float(min_dur):.6f}")

        sorted_rows = sorted(debug_rows, key=lambda x: x["est_dur"])
        print("")
        print(f"Top {min(debug_search_topk, len(sorted_rows))} predicted candidates:")
        for idx, row in enumerate(sorted_rows[:debug_search_topk], start=1):
            print(
                f"rank={idx:03d} est={row['est_dur']:.6f} "
                f"iter_num={row['iter_num']} gp_units={row['gp_units']} cSeg={row['cSeg']}"
            )

        print("")
        print("Detailed trace for top predicted candidates:")
        for idx, row in enumerate(sorted_rows[:debug_search_topk], start=1):
            print(f"--- rank={idx:03d} est={row['est_dur']:.6f} gp_units={row['gp_units']} cSeg={row['cSeg']} ---")
            for line in row["trace"]:
                print(f"  {line}")
        print("----------------------------------------")
        print("")

    print("Search process finished.")

    searched_lat = perf_running(
        M,
        N,
        K,
        BM,
        BN,
        Algo,
        best_cSeg,
        hint,
        comm_op,
        compute_sms,
        comm_backend,
    )

    print("Searched latency: %.4f" % searched_lat)
    print("Best solution:", best_cSeg)

    save_solution(
        M,
        N,
        K,
        BM,
        BN,
        gemm_dur,
        Algo,
        hint,
        best_cSeg,
        comm_sm_slack,
        comm_backend,
        comm_op,
        bandwidth_path=bandwidth_path,
        predicted_latency_ms=min_dur,
        searched_latency_ms=float(searched_lat),
        write_legacy_solution=write_legacy_solution,
    )


def run_for_backend(args, comm_backend: str):
    world_size = torch.cuda.device_count()

    print("")
    print("########################################")
    print(f"# Starting search for comm_backend={comm_backend}")
    print("########################################")
    print("")

    if args.comm_backend == "both" and args.bandwidth_path:
        raise ValueError("--bandwidth_path is ambiguous with --comm_backend both.")

    use_predictive = args.predictive_search or args.m * args.n > 33554432

    if use_predictive:
        comm_array, used_path = load_comm_array(comm_backend, args.comm_op, world_size, args.bandwidth_path)

        fast_search(
            args.m,
            args.n,
            args.k,
            comm_array,
            args.comm_op,
            args.comm_sm_slack,
            args.min_group_size,
            comm_backend=comm_backend,
            bandwidth_path=str(used_path),
            debug_hint=args.debug_hint,
            debug_hint_dump=args.debug_hint_dump,
            algo_id=args.algo_id,
            min_effective_group_size_override=args.min_effective_group_size,
            effective_hint_confirm=not args.no_effective_hint_confirm,
            debug_search=args.debug_search,
            debug_search_topk=args.debug_search_topk,
            write_legacy_solution=args.write_legacy_solution,
        )
    else:
        exhaustive_search(
            args.m,
            args.n,
            args.k,
            args.comm_op,
            args.comm_sm_slack,
            comm_backend=comm_backend,
            debug_hint=args.debug_hint,
            debug_hint_dump=args.debug_hint_dump,
            algo_id=args.algo_id,
            write_legacy_solution=args.write_legacy_solution,
        )


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--m", type=int, default=4096)
    p.add_argument("--n", type=int, default=8192)
    p.add_argument("--k", type=int, default=8192)

    p.add_argument("--comm_op", type=str, default="all_reduce", choices=["all_reduce", "reduce_scatter"])
    p.add_argument("--comm_backend", type=str, default="nccl", choices=["nccl", "ooverlap", "both"])
    p.add_argument("--bandwidth_path", type=str, default="")

    p.add_argument("--predictive_search", action="store_true")
    p.add_argument("--comm_sm_slack", type=int, default=2)
    p.add_argument("--min_group_size", type=int, default=None)
    p.add_argument("--min_effective_group_size", type=int, default=None)
    p.add_argument("--no_effective_hint_confirm", action="store_true")

    p.add_argument("--algo_id", type=int, default=None)

    p.add_argument("--debug_hint", action="store_true")
    p.add_argument("--debug_hint_dump", type=str, default="")
    p.add_argument("--debug_search", action="store_true")
    p.add_argument("--debug_search_topk", type=int, default=20)

    p.add_argument("--write_legacy_solution", action="store_true")

    return p.parse_args()


def main():
    args = parse_args()

    world_size = torch.cuda.device_count()
    if world_size < 2:
        raise RuntimeError("At least 2 GPUs are required.")

    backends = ["nccl", "ooverlap"] if args.comm_backend == "both" else [args.comm_backend]

    for backend in backends:
        run_for_backend(args, backend)


if __name__ == "__main__":
    main()
