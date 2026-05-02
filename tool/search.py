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


def algo_attempt_count(total: int, default_limit: int, try_all_algos: bool, algo_limit):
    if algo_limit is not None:
        return min(int(algo_limit), total)
    if try_all_algos:
        return total
    return min(default_limit, total)


def filter_candidates_by_algo_id(BM_list, BN_list, gemm_dur_list, Algo_list, algo_id):
    if algo_id is None:
        return BM_list, BN_list, gemm_dur_list, Algo_list

    algo_id = int(algo_id)

    out_BM = []
    out_BN = []
    out_dur = []
    out_Algo = []

    for BM, BN, dur, Algo in zip(BM_list, BN_list, gemm_dur_list, Algo_list):
        if int(Algo) == algo_id:
            out_BM.append(BM)
            out_BN.append(BN)
            out_dur.append(dur)
            out_Algo.append(Algo)

    if len(out_Algo) == 0:
        available = ", ".join(str(int(x)) for x in Algo_list)
        raise ValueError(
            f"Requested --algo_id {algo_id}, but it was not found in the loaded config. "
            f"Available algos: {available}"
        )

    print(f"Using requested algo_id={algo_id}.")
    return out_BM, out_BN, out_dur, out_Algo


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


def algo_tile_shape(Algo: int):
    algo_dict = load_algo_dict()
    item = algo_dict[int(Algo)]
    return int(item["tile_m"]), int(item["tile_n"])


def normalize_loaded_candidates(BM_list, BN_list, gemm_dur_list, Algo_list):
    algo_dict = load_algo_dict()

    out_BM = []
    out_BN = []
    out_dur = []
    out_Algo = []

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


def count_exact_str(counts, sample_count: int, min_count: int = 2):
    parts = []
    for c in range(sample_count, min_count - 1, -1):
        parts.append(f"{c}/{sample_count}:{int((counts == c).sum().item())}")
    return " ".join(parts)


def count_at_least_str(counts, sample_count: int, min_count: int = 2):
    parts = []
    for c in range(sample_count, min_count - 1, -1):
        parts.append(f">={c}/{sample_count}:{int((counts >= c).sum().item())}")
    return " ".join(parts)


def neighbor_membership_str(
    count_in_window,
    prev_count,
    next_count,
    exact_c: int,
    sample_count: int,
    max_patterns: int = 12,
):
    tiles = torch.where(count_in_window == exact_c)[0]
    total = int(tiles.numel())

    if total == 0:
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

    patterns = sorted(
        pattern_counts.items(),
        key=lambda kv: (-kv[1], -kv[0][0], -kv[0][1], kv[0][2]),
    )

    pattern_str = " | ".join(
        f"prev={p} next={n} other={o}:{num}"
        for (p, n, o), num in patterns[:max_patterns]
    )

    if len(patterns) > max_patterns:
        pattern_str += f" | ... +{len(patterns) - max_patterns} more"

    return (
        f"current={exact_c}/{sample_count}: tiles={total} "
        f"prev_any={prev_any} next_any={next_any} both={both_any} "
        f"prev_only={prev_only} next_only={next_only} neither_adjacent={neither} "
        f"patterns[{pattern_str}]"
    )


def debug_hint_samples(samples, TileNum: int, wSize: int, rank: int, Algo: int, BM: int, BN: int, label: str = ""):
    if rank != 0:
        return

    samples_cpu = samples.detach().cpu()
    sample_count = samples_cpu.shape[0]
    WaveNum = div_up(TileNum, wSize)

    print("")
    print("========================================")
    print("DEBUG compute_hint samples")
    print(f"label={label}")
    print(f"rank={rank}")
    print(f"Algo={Algo} BM={BM} BN={BN}")
    print(f"TileNum={TileNum} wSize={wSize} WaveNum={WaveNum} sample_count={sample_count}")
    print("========================================")

    # Basic validity checks.
    for i in range(sample_count):
        row = samples_cpu[i]
        min_v = int(row.min().item())
        max_v = int(row.max().item())
        unique_count = int(torch.unique(row).numel())
        missing = TileNum - unique_count

        print(
            f"sample[{i}] min={min_v} max={max_v} "
            f"unique={unique_count}/{TileNum} duplicate_or_missing={missing}"
        )

    print("----------------------------------------")
    print("Per-window membership counts:")
    print("For each window, this shows how many tile ids landed in that window")
    print("exactly N/N, N-1/N, N-2/N, ... down to 2/N runs.")
    print("The cumulative line shows >=N/N, >=N-1/N, >=N-2/N, ...")

    total_exact = 0
    for w in range(WaveNum):
        lo = w * wSize
        hi = min((w + 1) * wSize, TileNum)

        in_window_count = ((samples_cpu >= lo) & (samples_cpu < hi)).sum(dim=0)
        exact = torch.where(in_window_count == sample_count)[0]

        expected = hi - lo
        total_exact += int(exact.numel())

        print(f"window {w:03d}: range=[{lo}, {hi}) expected={expected}")
        print(f"  exact:      {count_exact_str(in_window_count, sample_count, min_count=2)}")
        print(f"  cumulative: {count_at_least_str(in_window_count, sample_count, min_count=2)}")

        # These are same-shape per-tile counts for adjacent windows.
        prev_count = torch.zeros_like(in_window_count)
        next_count = torch.zeros_like(in_window_count)

        if w > 0:
            prev_lo = (w - 1) * wSize
            prev_hi = w * wSize
            prev_count = ((samples_cpu >= prev_lo) & (samples_cpu < prev_hi)).sum(dim=0)

        if w + 1 < WaveNum:
            next_lo = (w + 1) * wSize
            next_hi = min((w + 2) * wSize, TileNum)
            next_count = ((samples_cpu >= next_lo) & (samples_cpu < next_hi)).sum(dim=0)

        neighbor_lines = []
        for exact_c in [4, 3, 2]:
            line = neighbor_membership_str(
                in_window_count,
                prev_count,
                next_count,
                exact_c,
                sample_count,
            )
            if line:
                neighbor_lines.append(line)

        if neighbor_lines:
            print("  neighbor distribution for low-count current-window tiles:")
            print("  Format: current=x/N means tile landed in this window x times.")
            print("  prev/next show how the remaining samples split into adjacent windows.")
            print("  other means not current, not previous, and not next window.")
            for line in neighbor_lines:
                print(f"    {line}")

    print(f"total exact-stable tiles collected={total_exact}/{TileNum}")

    print("----------------------------------------")
    print("Boundary instability check:")
    print("For each boundary, crossing_tiles are tiles that were sometimes before")
    print("the boundary and sometimes after/equal to the boundary across samples.")
    print("The histogram shows, among crossing tiles only, how often they landed")
    print("on the left side of the boundary.")

    for w in range(1, WaveNum):
        boundary = w * wSize

        before_count = (samples_cpu < boundary).sum(dim=0)
        crosses = torch.where((before_count > 0) & (before_count < sample_count))[0]

        if crosses.numel() > 0:
            crossing_before_count = before_count[crosses]

            hist_parts = []
            for c in range(sample_count - 1, 0, -1):
                n = int((crossing_before_count == c).sum().item())
                hist_parts.append(f"left={c}/{sample_count}:right={sample_count-c}/{sample_count}:{n}")

            tile_ids = crosses[:20].tolist()

            print(
                f"boundary {w:03d} at order={boundary}: "
                f"crossing_tiles={int(crosses.numel())}"
            )
            print("  crossing histogram:", " ".join(hist_parts))
            print(f"  first20={tile_ids}")

            for tile in tile_ids[:5]:
                vals = samples_cpu[:, tile].tolist()
                left_hits = int((samples_cpu[:, tile] < boundary).sum().item())
                right_hits = sample_count - left_hits
                print(
                    f"  tile {tile}: orders={vals} "
                    f"left={left_hits}/{sample_count} right={right_hits}/{sample_count}"
                )
        else:
            print(f"boundary {w:03d} at order={boundary}: crossing_tiles=0")

    print("----------------------------------------")
    print("First 64 completion orders per sample:")
    print("This prints tile ids sorted by completion order for each run.")

    for i in range(sample_count):
        row = samples_cpu[i]
        order_to_tile = torch.argsort(row)
        print(f"sample[{i}] first64 tile_ids:", order_to_tile[:64].tolist())

    print("----------------------------------------")
    print("Last 64 completion orders per sample:")

    for i in range(sample_count):
        row = samples_cpu[i]
        order_to_tile = torch.argsort(row)
        print(f"sample[{i}] last64 tile_ids:", order_to_tile[-64:].tolist())

    print("========================================")
    print("")


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
    packed_path = shape_json_path(M, N, K, "packed")

    if not Path(packed_path).exists():
        raise FileNotFoundError(
            f"Could not find packed SM90 config file:\n"
            f"  {packed_path}\n"
            "Run gen_config_sm90.py with --layout packed first."
        )

    with open(packed_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"Loaded shape config: {packed_path}")

    return normalize_loaded_candidates(
        data["BM"],
        data["BN"],
        data["dur"],
        data["Algo"],
    )

def build_full_order_and_reorder_map(TileNum: int, hint: list):
    used = [False] * TileNum

    stable_prefix = []
    for x in hint:
        x = int(x)
        if x < 0 or x >= TileNum:
            raise ValueError(f"Invalid tile id in hint: {x}")
        if not used[x]:
            stable_prefix.append(x)
            used[x] = True

    unstable_tail = [x for x in range(TileNum) if not used[x]]
    full_tile_order = stable_prefix + unstable_tail

    reorder_map = [-1] * TileNum
    for new_pos, tile_id in enumerate(full_tile_order):
        reorder_map[tile_id] = new_pos

    return full_tile_order, reorder_map, unstable_tail

def save_solution(M: int, N: int, K: int, BM: int, BN: int, gemm_dur: float, Algo: int, hint: list, cSeg: list, comm_sm_slack: int):
    out_path = solution_json_path(M, N, K)

    device = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device)
    sm_count = props.multi_processor_count
    compute_sms = effective_compute_sms(sm_count, comm_sm_slack)

    TileNum = div_up(M, BM) * div_up(N, BN)
    full_tile_order, reorder_map, unstable_tail = build_full_order_and_reorder_map(TileNum, hint)
    
    data = {
        "M": int(M),
        "N": int(N),
        "K": int(K),
    
        # Existing field. Stable tiles only.
        "hint": [int(x) for x in hint],
    
        # New explicit fields.
        "hint_stable_count": int(len(hint)),
        "unstable_tail_count": int(len(unstable_tail)),
        "unstable_tail": [int(x) for x in unstable_tail],
    
        # Original tile ids in final desired order:
        # stable tiles first, unstable tiles at the end.
        "full_tile_order": [int(x) for x in full_tile_order],
    
        # This is the actual map equivalent to reorder_indices(TileNum, hint):
        # reorder_map[original_tile_id] = new_position
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

        mod_values = chunk_row_ids % world_size

        _, sorted_indices = torch.sort(mod_values, stable=True)
        reordered_chunk = chunk_row_ids[sorted_indices]

        reordered_row_id[current_row : current_row + chunk_size] = reordered_chunk
        current_row += chunk_size

    remap = torch.empty_like(original_row_ids)
    remap[reordered_row_id] = torch.arange(len(reordered_row_id), dtype=torch.int, device=device)

    return remap


def collect_monitor_samples(
    gemm_class,
    rank: int,
    M: int,
    N: int,
    K: int,
    BM: int,
    BN: int,
    Algo: int,
    TileNum: int,
    cSeg: list,
    cSeg_CPU,
    cSeg_GPU,
    A,
    B,
    C,
    MonitoredMatrix,
    ReorderedArray,
    comm_op: str,
    sample_count: int,
    active_sm_count: int,
    D=None,
    RowArray=None,
):
    samples = torch.empty((sample_count, TileNum), dtype=torch.int, device="cuda")

    if comm_op == "all_reduce":
        for i in range(sample_count):
            reset_monitor_matrix(MonitoredMatrix, TileNum, len(cSeg), True)
            gemm_class.gemm_allreduce_overlap(
                A,
                B,
                C,
                MonitoredMatrix,
                ReorderedArray,
                1,
                cSeg_CPU,
                cSeg_GPU,
                Algo,
                int(active_sm_count),
                True,
            )
            samples[i, :] = monitor_order_view(MonitoredMatrix, TileNum, len(cSeg))

    elif comm_op == "reduce_scatter":
        for i in range(sample_count):
            reset_monitor_matrix(MonitoredMatrix, TileNum, len(cSeg), True)
            gemm_class.gemm_reducescatter_overlap(
                A,
                B,
                C,
                D,
                MonitoredMatrix,
                ReorderedArray,
                RowArray,
                1,
                cSeg_CPU,
                cSeg_GPU,
                Algo,
                True,
            )
            samples[i, :] = monitor_order_view(MonitoredMatrix, TileNum, len(cSeg))

    else:
        raise ValueError(f"Unknown comm_op={comm_op}")

    return samples


def compute_stable_hint_from_rounds(
    sample_rounds,
    TileNum: int,
    wSize: int,
    compute_sms: int,
    nominal_min_group_size: int,
    min_effective_group_size: int,
    rank: int,
):
    """
    Build a stable hint from one or more sampling rounds.

    Each nominal window has size:
        wSize = nominal_min_group_size * compute_sms

    A tile is considered stable for a window only if it lands in that same
    nominal window in every sample of every round.

    effective_min_group_size is chosen by:
        floor(min_stable_count_across_full_windows / compute_sms)

    This lets us accept a nominal min_group_size=20 run even if only 19 waves
    worth of tiles are perfectly stable inside each nominal 20-wave window.
    """
    assert len(sample_rounds) > 0
    WaveNum = div_up(TileNum, wSize)
    device = sample_rounds[0].device

    stable_per_window = []
    effective_group = int(nominal_min_group_size)

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

    full_window_seen = False

    for w in range(WaveNum):
        lo = w * wSize
        hi = min((w + 1) * wSize, TileNum)
        expected = hi - lo

        stable_mask = torch.ones((TileNum,), dtype=torch.bool, device=device)
        round_exact_counts = []

        for samples in sample_rounds:
            sample_count = samples.shape[0]
            count_in_window = ((samples >= lo) & (samples < hi)).sum(dim=0)
            exact_mask = count_in_window == sample_count
            stable_mask = stable_mask & exact_mask
            round_exact_counts.append(int(exact_mask.sum().item()))

        stable_tiles = torch.where(stable_mask)[0]
        stable_count = int(stable_tiles.numel())
        stable_per_window.append(stable_tiles)

        if expected == wSize:
            full_window_seen = True
            window_effective_group = stable_count // compute_sms
            window_effective_group = min(window_effective_group, nominal_min_group_size)
            effective_group = min(effective_group, window_effective_group)
        else:
            window_effective_group = stable_count // compute_sms
            window_effective_group = min(window_effective_group, nominal_min_group_size)

        if rank == 0:
            round_counts_str = " ".join(
                f"round{idx}_exact={cnt}"
                for idx, cnt in enumerate(round_exact_counts)
            )
            print(
                f"window {w:03d}: range=[{lo}, {hi}) expected={expected} "
                f"{round_counts_str} final_stable={stable_count} "
                f"window_effective_group={window_effective_group}"
            )

    if not full_window_seen:
        effective_group = min(nominal_min_group_size, div_up(TileNum, compute_sms))

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
    used = torch.zeros((TileNum,), dtype=torch.bool, device=device)

    for stable_tiles in stable_per_window:
        for tile in stable_tiles.tolist():
            if not bool(used[tile].item()):
                hint.append(int(tile))
                used[tile] = True

    stable_total = len(hint)
    unstable_total = TileNum - stable_total

    if rank == 0:
        print(
            f"hint stable tiles={stable_total}/{TileNum}; "
            f"unstable tiles left for reorder tail={unstable_total}"
        )
        print("----------------------------------------")
        print("")

    return True, hint, effective_group


def compute_hint_process(rank, world_size, nccl_id,
    M: int, N: int, K: int,
    BM: int, BN: int, Algo: int, wSize: int, comm_op: str,
    compute_sms: int,
    nominal_min_group_size: int,
    min_effective_group_size: int,
    effective_hint_confirm: bool,
    debug_hint: bool, debug_hint_dump: str,
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
    ReorderedArray = torch.arange(0, TileNum, dtype=torch.int, device="cuda").reshape(((M + BM - 1) // BM, (N + BN - 1) // BN))

    D = None
    RowArray = None
    if comm_op == "reduce_scatter":
        D = torch.empty((M // world_size, N), dtype=torch.float16, device="cuda")
        RowArray = generate_row_remap_array(M, N, BM, BN, cSeg, world_size)

    _warm_up = 100
    _sample = 10

    if comm_op == "all_reduce":
        for _ in range(_warm_up):
            reset_monitor_matrix(MonitoredMatrix, TileNum, len(cSeg), True)
            gemm_class.gemm_allreduce_overlap(
                A,
                B,
                C,
                MonitoredMatrix,
                ReorderedArray,
                1,
                cSeg_CPU,
                cSeg_GPU,
                Algo,
                int(compute_sms),
                True,
            )
    elif comm_op == "reduce_scatter":
        for _ in range(_warm_up):
            reset_monitor_matrix(MonitoredMatrix, TileNum, len(cSeg), True)
            gemm_class.gemm_reducescatter_overlap(
                A,
                B,
                C,
                D,
                MonitoredMatrix,
                ReorderedArray,
                RowArray,
                1,
                cSeg_CPU,
                cSeg_GPU,
                Algo,
                True,
            )
    else:
        raise ValueError(f"Unknown comm_op={comm_op}")

    samples_first = collect_monitor_samples(
        gemm_class,
        rank,
        M,
        N,
        K,
        BM,
        BN,
        Algo,
        TileNum,
        cSeg,
        cSeg_CPU,
        cSeg_GPU,
        A,
        B,
        C,
        MonitoredMatrix,
        ReorderedArray,
        comm_op,
        _sample,
        int(compute_sms),
        D=D,
        RowArray=RowArray,
    )

    torch.cuda.synchronize()

    sample_rounds = [samples_first]

    if debug_hint:
        debug_hint_samples(
            samples_first,
            TileNum,
            wSize,
            rank,
            Algo,
            BM,
            BN,
            label=f"M{M}N{N}K{K}_round0"
        )

    if effective_hint_confirm:
        samples_confirm = collect_monitor_samples(
            gemm_class,
            rank,
            M,
            N,
            K,
            BM,
            BN,
            Algo,
            TileNum,
            cSeg,
            cSeg_CPU,
            cSeg_GPU,
            A,
            B,
            C,
            MonitoredMatrix,
            ReorderedArray,
            comm_op,
            _sample,
            int(compute_sms),
            D=D,
            RowArray=RowArray,
        )

        torch.cuda.synchronize()
        sample_rounds.append(samples_confirm)

        if debug_hint:
            combined = torch.cat([samples_first, samples_confirm], dim=0)
            debug_hint_samples(
                combined,
                TileNum,
                wSize,
                rank,
                Algo,
                BM,
                BN,
                label=f"M{M}N{N}K{K}_combined_confirm"
            )

    if debug_hint_dump and rank == 0:
        dump_path = Path(debug_hint_dump)
        dump_path.parent.mkdir(parents=True, exist_ok=True)

        dump_data = {
            "samples": samples_first.detach().cpu(),
            "sample_rounds": [x.detach().cpu() for x in sample_rounds],
            "TileNum": TileNum,
            "WaveNum": WaveNum,
            "wSize": wSize,
            "M": M,
            "N": N,
            "K": K,
            "BM": BM,
            "BN": BN,
            "Algo": int(Algo),
            "comm_op": comm_op,
            "compute_sms": int(compute_sms),
            "nominal_min_group_size": int(nominal_min_group_size),
            "min_effective_group_size": int(min_effective_group_size),
            "effective_hint_confirm": bool(effective_hint_confirm),
        }

        torch.save(dump_data, dump_path)
        print(f"DEBUG hint samples dumped to: {dump_path}")

    is_consistency, hint, effective_min_group_size = compute_stable_hint_from_rounds(
        sample_rounds,
        TileNum,
        wSize,
        compute_sms,
        nominal_min_group_size,
        min_effective_group_size,
        rank,
    )

    result_dict[rank] = (is_consistency, hint, effective_min_group_size)


def compute_hint(M: int, N: int, K: int,
    BM: int, BN: int, Algo: int, wSize: int, comm_op: str,
    compute_sms: int,
    nominal_min_group_size: int,
    min_effective_group_size: int,
    effective_hint_confirm: bool = False,
    debug_hint: bool = False,
    debug_hint_dump: str = ""):
    world_size = torch.cuda.device_count()
    if world_size < 2:
        raise RuntimeError("At least 2 GPUs are required for this program.")

    nccl_id = ext.generate_nccl_id()
    torch.cuda.synchronize()

    manager = mp.Manager()
    result_dict = manager.dict()

    mp.spawn(
        compute_hint_process,
        args=(
            world_size,
            nccl_id,
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
            result_dict,
        ),
        nprocs=world_size
    )

    return result_dict[0]


def interpolate_latency(samples, x, comm_op):
    world_size = torch.cuda.device_count()

    if not isinstance(samples, torch.Tensor):
        samples = torch.tensor(samples, dtype=torch.float32)
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x, dtype=torch.float32)

    data_sizes = samples[:, 0].numpy()
    bandwidths = samples[:, 1].numpy()
    x_np = x.numpy()

    y_np = np.interp(x_np, data_sizes, bandwidths)
    y = torch.tensor(y_np, dtype=torch.float32).item()

    if comm_op == "all_reduce":
        latency_sec = x * 2 * 2 * (world_size - 1) / y / (1024 ** 3)
    elif comm_op == "reduce_scatter":
        latency_sec = x * 2 * (world_size - 1) / y / (1024 ** 3)
    else:
        raise ValueError(f"Unknown comm_op={comm_op}")

    # gemm_dur and CUDA event timings are in milliseconds.
    # Bandwidth formula above returns seconds, so convert to ms.
    return latency_sec.item() * 1000.0

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
        acc_comm_dur = interpolate_latency(comm_array, M * N // tile_num * gp[0], comm_op) + gemm_dur
        return acc_comm_dur

    old_wave_num = div_up(tile_num, sm_count)
    new_wave_num = div_up(tile_num, compute_sms)
    gemm_dur = gemm_dur / old_wave_num * new_wave_num

    for i in range(iter_num):
        if i == 0:
            comm_dur = 0
        else:
            comm_dur = interpolate_latency(comm_array, M * N // tile_num * gp[i - 1], comm_op)

        acc_comm_dur = max(acc_comp_dur, acc_comm_dur) + comm_dur
        acc_comp_dur += gemm_dur / new_wave_num * div_up(gp[i], compute_sms)

    acc_comm_dur = max(acc_comp_dur, acc_comm_dur) + interpolate_latency(
        comm_array,
        M * N // tile_num * gp[-1],
        comm_op,
    )

    return acc_comm_dur


def reorder_indices(S, hint):
    original = list(range(S))
    new_order = [-1] * S

    for i, element in enumerate(hint):
        new_order[element] = i

    remaining_elements = [x for x in original if x not in hint]
    for i, element in enumerate(remaining_elements, start=len(hint)):
        new_order[element] = i

    return torch.tensor(new_order, dtype=torch.int, device="cuda")


def perf_running_process(rank, world_size, nccl_id,
    M: int, N: int, K: int,
    BM: int, BN: int, Algo: int, cSeg: list, hint: list,
    comm_op: str,
    active_sm_count: int,
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
    ReorderedArray = reorder_indices(TileNum, hint).reshape(((M + BM - 1) // BM, (N + BN - 1) // BN))

    if comm_op == "reduce_scatter":
        D = torch.empty((M // world_size, N), dtype=torch.float16, device="cuda")
        RowArray = generate_row_remap_array(M, N, BM, BN, cSeg, world_size)

    _warm_up = 20
    _freq = 200

    if len(cSeg) == 1:
        if comm_op == "all_reduce":
            for _ in range(_warm_up):
                reset_monitor_matrix(MonitoredMatrix, TileNum, len(cSeg), False)
                gemm_class.gemm_allreduce_overlap(A, B, C, MonitoredMatrix, ReorderedArray, 1, cSeg_CPU, cSeg_GPU, Algo, int(active_sm_count), False)

            reset_monitor_matrix(MonitoredMatrix, TileNum, len(cSeg), False)
            gemm_class.gemm_allreduce_overlap(A, B, C, MonitoredMatrix, ReorderedArray, 1, cSeg_CPU, cSeg_GPU, Algo, int(active_sm_count), False)

            start_event = [torch.cuda.Event(enable_timing=True) for _ in range(_freq)]
            end_event = [torch.cuda.Event(enable_timing=True) for _ in range(_freq)]
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

            start_event = [torch.cuda.Event(enable_timing=True) for _ in range(_freq)]
            end_event = [torch.cuda.Event(enable_timing=True) for _ in range(_freq)]
            for i in range(_freq):
                reset_monitor_matrix(MonitoredMatrix, TileNum, len(cSeg), False)
                start_event[i].record()
                gemm_class.gemm_reducescatter_overlap(A, B, C, D, MonitoredMatrix, ReorderedArray, RowArray, 1, cSeg_CPU, cSeg_GPU, Algo, False)
                end_event[i].record()
            torch.cuda.synchronize()
            dur = torch.tensor([s.elapsed_time(e) for s, e in zip(start_event, end_event)], dtype=torch.float)
        else:
            dur = torch.zeros((_freq))

    else:
        if comm_op == "all_reduce":
            for _ in range(_warm_up):
                reset_monitor_matrix(MonitoredMatrix, TileNum, len(cSeg), False)
                gemm_class.gemm_allreduce_overlap(A, B, C, MonitoredMatrix, ReorderedArray, 1, cSeg_CPU, cSeg_GPU, Algo, int(active_sm_count), False)

            start_event = [torch.cuda.Event(enable_timing=True) for _ in range(_freq)]
            end_event = [torch.cuda.Event(enable_timing=True) for _ in range(_freq)]
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

            start_event = [torch.cuda.Event(enable_timing=True) for _ in range(_freq)]
            end_event = [torch.cuda.Event(enable_timing=True) for _ in range(_freq)]
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
    active_sm_count: int = 0):
    world_size = torch.cuda.device_count()
    if world_size < 2:
        raise RuntimeError("At least 2 GPUs are required for this program.")

    nccl_id = ext.generate_nccl_id()
    torch.cuda.synchronize()

    manager = mp.Manager()
    result_dict = manager.dict()

    mp.spawn(
        perf_running_process,
        args=(world_size, nccl_id, M, N, K, BM, BN, Algo, cSeg, hint, comm_op, active_sm_count, result_dict),
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

def expand_partition_to_cseg(gp_units, compute_sms: int, search_group_size: int, tile_num: int):
    gp = list(gp_units)
    acc = 0

    for j in range(len(gp)):
        if j < len(gp) - 1:
            gp[j] = gp[j] * compute_sms * search_group_size
            acc += gp[j]
        else:
            gp[j] = min(gp[j] * compute_sms * search_group_size, tile_num - acc)

    return gp


def predict_lat_with_debug(
    M: int,
    N: int,
    gemm_dur: float,
    comm_array: torch.Tensor,
    gp: list,
    tile_num: int,
    comm_op: str,
    comm_sm_slack: int,
):
    device = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device)
    sm_count = props.multi_processor_count
    compute_sms = effective_compute_sms(sm_count, comm_sm_slack)

    bytes_or_elems_per_tile = M * N // tile_num
    iter_num = len(gp)

    lines = []
    lines.append(
        f"gp={gp} iter_num={iter_num} tile_num={tile_num} "
        f"sm_count={sm_count} compute_sms={compute_sms} "
        f"bytes_or_elems_per_tile={bytes_or_elems_per_tile}"
    )

    if iter_num == 1:
        comm_x = bytes_or_elems_per_tile * gp[0]
        comm_dur = interpolate_latency(comm_array, comm_x, comm_op)
        total = comm_dur + gemm_dur

        lines.append(
            "one-segment/no-overlap path: "
            f"comm_x={comm_x} comm_dur={comm_dur:.6f} "
            f"gemm_dur={gemm_dur:.6f} total={total:.6f}"
        )
        lines.append(
            "NOTE: this path uses the original gemm_dur directly, matching predict_lat()."
        )

        return total, lines

    old_wave_num = div_up(tile_num, sm_count)
    new_wave_num = div_up(tile_num, compute_sms)
    scaled_gemm_dur = gemm_dur / old_wave_num * new_wave_num

    lines.append(
        "multi-segment/overlap path: "
        f"old_wave_num={old_wave_num} new_wave_num={new_wave_num} "
        f"original_gemm_dur={gemm_dur:.6f} "
        f"scaled_gemm_dur={scaled_gemm_dur:.6f}"
    )

    acc_comm_dur = 0.0
    acc_comp_dur = 0.0

    for i in range(iter_num):
        if i == 0:
            comm_tiles = 0
            comm_x = 0
            comm_dur = 0.0
        else:
            comm_tiles = gp[i - 1]
            comm_x = bytes_or_elems_per_tile * comm_tiles
            comm_dur = interpolate_latency(comm_array, comm_x, comm_op)

        comp_waves = div_up(gp[i], compute_sms)
        comp_dur = scaled_gemm_dur / new_wave_num * comp_waves

        prev_acc_comm_dur = acc_comm_dur
        prev_acc_comp_dur = acc_comp_dur

        acc_comm_dur = max(acc_comp_dur, acc_comm_dur) + comm_dur
        acc_comp_dur += comp_dur

        lines.append(
            f"iter={i}: segment_tiles={gp[i]} comp_waves={comp_waves} "
            f"comp_dur={comp_dur:.6f} "
            f"prev_acc_comp={prev_acc_comp_dur:.6f} "
            f"prev_acc_comm={prev_acc_comm_dur:.6f} "
            f"comm_tiles_from_prev={comm_tiles} comm_x={comm_x} "
            f"comm_dur={comm_dur:.6f} "
            f"new_acc_comp={acc_comp_dur:.6f} "
            f"new_acc_comm={acc_comm_dur:.6f}"
        )

    final_comm_x = bytes_or_elems_per_tile * gp[-1]
    final_comm_dur = interpolate_latency(comm_array, final_comm_x, comm_op)
    total = max(acc_comp_dur, acc_comm_dur) + final_comm_dur

    lines.append(
        f"final_comm: tiles={gp[-1]} comm_x={final_comm_x} "
        f"final_comm_dur={final_comm_dur:.6f} "
        f"max(acc_comp={acc_comp_dur:.6f}, acc_comm={acc_comm_dur:.6f}) "
        f"total={total:.6f}"
    )

    return total, lines


def exhaustive_search(M: int, N: int, K: int, comm_op: str, comm_sm_slack: int,
    debug_hint: bool = False, debug_hint_dump: str = "",
    try_all_algos: bool = False, algo_limit=None, algo_id=None):

    BM_list, BN_list, gemm_dur_list, Algo_list = load_json(M, N, K)
    BM_list, BN_list, gemm_dur_list, Algo_list = filter_candidates_by_algo_id(
        BM_list, BN_list, gemm_dur_list, Algo_list, algo_id
    )

    device = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device)
    sm_count = props.multi_processor_count
    compute_sms = effective_compute_sms(sm_count, comm_sm_slack)

    print(f"SM count={sm_count}, comm_sm_slack={comm_sm_slack}, compute_sms={compute_sms}")

    hint = None
    effective_min_group_size = 1
    candidate_count = algo_attempt_count(len(Algo_list), 5, try_all_algos, algo_limit)
    print(f"Trying {candidate_count}/{len(Algo_list)} candidate algos for exhaustive search.")

    for t in range(candidate_count):
        BM = BM_list[t]
        BN = BN_list[t]
        gemm_dur = gemm_dur_list[t]
        Algo = Algo_list[t]

        if is_cooperative_algo(Algo):
            print(f"Skip cooperative algo={Algo}")
            continue

        tile_num = div_up(M, BM) * div_up(N, BN)
        wave_num = div_up(tile_num, compute_sms)

        print(
            f"Try candidate {t + 1}/{candidate_count}: "
            f"algo={Algo} BM={BM} BN={BN} tile_num={tile_num} wave_num={wave_num}"
        )

        debug_dump_this_algo = ""
        if debug_hint_dump:
            debug_dump_this_algo = debug_hint_dump.replace(".pt", f"_algo{Algo}_bm{BM}_bn{BN}.pt")

        try:
            result = compute_hint(
                M,
                N,
                K,
                BM,
                BN,
                Algo,
                compute_sms,
                comm_op,
                compute_sms=compute_sms,
                nominal_min_group_size=1,
                min_effective_group_size=1,
                effective_hint_confirm=False,
                debug_hint=debug_hint,
                debug_hint_dump=debug_dump_this_algo,
            )
        except Exception as e:
            print(f"compute_hint failed for algo={Algo}; trying next candidate.")
            print(f"  {type(e).__name__}: {e}")
            continue

        if result[0] == True:
            hint = result[1]
            effective_min_group_size = result[2]
            print(
                f"Selected algo={Algo} after successful compute_hint. "
                f"effective_min_group_size={effective_min_group_size}"
            )
            break

        print(f"compute_hint inconsistent for algo={Algo}; trying next candidate.")

    assert hint is not None, "Tuning fails! Try to increase min_group_size manually or use --try_all_algos."
    print("Start exhaustive searching.")

    min_dur = 1e5
    cSeg = None

    group_size_list = integer_partitions(wave_num)
    for gp in group_size_list:
        gp = list(gp)
        iter_num = len(gp)
        acc = 0
        for j in range(iter_num):
            if j < iter_num - 1:
                gp[j] = gp[j] * compute_sms
                acc += gp[j]
            else:
                gp[j] = min(gp[j] * compute_sms, tile_num - acc)

        dur = perf_running(M, N, K, BM, BN, Algo, gp, hint, comm_op, compute_sms)
        print(gp, "%.4f" % dur)

        if dur < min_dur:
            min_dur = dur
            cSeg = gp

    print("Best solution: ", cSeg)
    save_solution(M, N, K, BM, BN, gemm_dur, Algo, hint, cSeg, comm_sm_slack)
    print("Solution saved.")

def fast_search(M: int, N: int, K: int, comm_array: torch.Tensor, comm_op: str,
    comm_sm_slack: int, min_group_size_override,
    debug_hint: bool = False, debug_hint_dump: str = "",
    try_all_algos: bool = False, algo_limit=None, algo_id=None,
    min_effective_group_size_override=None,
    effective_hint_confirm: bool = True,
    debug_search: bool = False,
    debug_search_topk: int = 20):

    BM_list, BN_list, gemm_dur_list, Algo_list = load_json(M, N, K)
    BM_list, BN_list, gemm_dur_list, Algo_list = filter_candidates_by_algo_id(
        BM_list, BN_list, gemm_dur_list, Algo_list, algo_id
    )

    device = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device)
    sm_count = props.multi_processor_count
    compute_sms = effective_compute_sms(sm_count, comm_sm_slack)

    print(f"SM count={sm_count}, comm_sm_slack={comm_sm_slack}, compute_sms={compute_sms}")

    hint = None
    effective_min_group_size = None
    selected_min_group_size = None

    candidate_count = algo_attempt_count(len(Algo_list), 10, try_all_algos, algo_limit)
    print(f"Trying {candidate_count}/{len(Algo_list)} candidate algos for predictive search.")

    for t in range(candidate_count):
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
            min_group_size = int(min_group_size_override)

        if min_effective_group_size_override is None:
            min_effective_group_size = max(1, min_group_size - 3)
        else:
            min_effective_group_size = int(min_effective_group_size_override)

        print(
            f"Try candidate {t + 1}/{candidate_count}: "
            f"algo={Algo} BM={BM} BN={BN} tile_num={tile_num} "
            f"wave_num={wave_num} min_group_size={min_group_size} "
            f"min_effective_group_size={min_effective_group_size}"
        )

        debug_dump_this_algo = ""
        if debug_hint_dump:
            debug_dump_this_algo = debug_hint_dump.replace(".pt", f"_algo{Algo}_bm{BM}_bn{BN}.pt")

        try:
            result = compute_hint(
                M,
                N,
                K,
                BM,
                BN,
                Algo,
                min_group_size * compute_sms,
                comm_op,
                compute_sms=compute_sms,
                nominal_min_group_size=min_group_size,
                min_effective_group_size=min_effective_group_size,
                effective_hint_confirm=effective_hint_confirm,
                debug_hint=debug_hint,
                debug_hint_dump=debug_dump_this_algo,
            )
        except Exception as e:
            print(f"compute_hint failed for algo={Algo}; trying next candidate.")
            print(f"  {type(e).__name__}: {e}")
            continue

        if result[0] == True:
            hint = result[1]
            effective_min_group_size = result[2]
            selected_min_group_size = min_group_size
            print(
                f"Selected algo={Algo} after successful compute_hint. "
                f"nominal_min_group_size={selected_min_group_size} "
                f"effective_min_group_size={effective_min_group_size}"
            )
            break

        print(f"compute_hint inconsistent for algo={Algo}; trying next candidate.")

    assert hint is not None, "Tuning fails! Try to increase min_group_size manually or use --try_all_algos."

    search_group_size = int(effective_min_group_size)
    assert search_group_size > 0

    print(
        "Start predictive searching. "
        f"Using effective_min_group_size={search_group_size} "
        f"instead of nominal_min_group_size={selected_min_group_size}."
    )

    min_dur = 1e5
    cSeg = None

    normalized_wave_num = div_up(wave_num, search_group_size)
    group_size_list = integer_partitions(normalized_wave_num)

    print(
        f"wave_num={wave_num} normalized_wave_num={normalized_wave_num} "
        f"compute_sms={compute_sms} search_group_size={search_group_size} "
        f"partition_count={len(group_size_list)}"
    )

    debug_rows = []
    skipped_cold_start = 0
    evaluated_count = 0

    for gp_units in group_size_list:
        gp_units = list(gp_units)
        iter_num = len(gp_units)

        # avoid cold start
        if iter_num > 5 and gp_units[0] > 2:
            skipped_cold_start += 1
            if debug_search:
                print(
                    f"Skip partition due to cold-start rule: "
                    f"gp_units={gp_units} iter_num={iter_num} first_unit={gp_units[0]}"
                )
            continue

        gp = expand_partition_to_cseg(
            gp_units,
            compute_sms,
            search_group_size,
            tile_num,
        )

        if debug_search:
            est_dur, trace_lines = predict_lat_with_debug(
                M,
                N,
                gemm_dur,
                comm_array,
                gp,
                tile_num,
                comm_op,
                comm_sm_slack,
            )
        else:
            est_dur = predict_lat(M, N, gemm_dur, comm_array, gp, tile_num, comm_op, comm_sm_slack)
            trace_lines = []

        evaluated_count += 1

        debug_rows.append({
            "est_dur": float(est_dur),
            "gp_units": list(gp_units),
            "cSeg": list(gp),
            "iter_num": iter_num,
            "trace": trace_lines,
        })

        if est_dur < min_dur:
            old_min = min_dur
            min_dur = est_dur
            cSeg = gp

            if debug_search:
                if old_min == 1e5:
                    print(
                        f"New best initial: est={float(est_dur):.6f} "
                        f"gp_units={gp_units} cSeg={gp}"
                    )
                else:
                    print(
                        f"New best: est={float(est_dur):.6f} "
                        f"old_best={float(old_min):.6f} "
                        f"gp_units={gp_units} cSeg={gp}"
                    )

    assert cSeg is not None, "Predictive search produced no candidate cSeg."

    if debug_search:
        print("")
        print("----------------------------------------")
        print("Predictive search debug summary:")
        print(
            f"partition_count={len(group_size_list)} "
            f"evaluated_count={evaluated_count} "
            f"skipped_cold_start={skipped_cold_start}"
        )
        print(
            f"selected_cSeg={cSeg} selected_predicted_latency={float(min_dur):.6f}"
        )

        sorted_rows = sorted(debug_rows, key=lambda x: x["est_dur"])

        one_segment_rank = None
        for idx, row in enumerate(sorted_rows):
            if len(row["cSeg"]) == 1:
                one_segment_rank = idx + 1
                break

        if one_segment_rank is not None:
            one_seg_row = sorted_rows[one_segment_rank - 1]
            print(
                f"one-segment candidate rank={one_segment_rank}/{len(sorted_rows)} "
                f"est={one_seg_row['est_dur']:.6f} cSeg={one_seg_row['cSeg']}"
            )
        else:
            print("one-segment candidate was not evaluated.")

        print("")
        print(f"Top {min(debug_search_topk, len(sorted_rows))} predicted candidates:")
        for idx, row in enumerate(sorted_rows[:debug_search_topk], start=1):
            print(
                f"rank={idx:03d} est={row['est_dur']:.6f} "
                f"iter_num={row['iter_num']} "
                f"gp_units={row['gp_units']} cSeg={row['cSeg']}"
            )

        print("")
        print("Detailed trace for top predicted candidates:")
        for idx, row in enumerate(sorted_rows[:debug_search_topk], start=1):
            print(
                f"--- rank={idx:03d} est={row['est_dur']:.6f} "
                f"gp_units={row['gp_units']} cSeg={row['cSeg']} ---"
            )
            for line in row["trace"]:
                print(f"  {line}")

        print("----------------------------------------")
        print("")

    print("Search process finished.")

    searched_lat = perf_running(M, N, K, BM, BN, Algo, cSeg, hint, comm_op, compute_sms)
    print("Searched latency: %.4f" % searched_lat)
    print("Best solution: ", cSeg)
    save_solution(M, N, K, BM, BN, gemm_dur, Algo, hint, cSeg, comm_sm_slack)
    print("Solution saved.")

def main():
    world_size = torch.cuda.device_count()

    parser = argparse.ArgumentParser()
    parser.add_argument('--m', type=int, default=4096)
    parser.add_argument('--k', type=int, default=8192)
    parser.add_argument('--n', type=int, default=8192)
    parser.add_argument('--comm_op', type=str, default='all_reduce')
    parser.add_argument('--predictive_search', action='store_true')
    parser.add_argument('--comm_sm_slack', type=int, default=2,
                        help='Number of SMs to leave as communication slack in the search model. Original FlashOverlap uses 2.')
    parser.add_argument('--min_group_size', type=int, default=None,
                        help='Override predictive-search nominal min_group_size. If omitted, uses div_up(wave_num, 10).')
    parser.add_argument('--min_effective_group_size', type=int, default=None,
                        help='Minimum accepted effective group size. If omitted, uses max(1, min_group_size - 3).')
    parser.add_argument('--no_effective_hint_confirm', action='store_true',
                        help='Disable the second 10-sample confirmation round for effective hint generation.')
    parser.add_argument('--debug_hint', action='store_true',
                        help='Print detailed compute_hint monitor-order diagnostics.')
    parser.add_argument('--debug_hint_dump', type=str, default="",
                        help='Optional .pt path to dump compute_hint samples for offline inspection.')
    parser.add_argument('--try_all_algos', action='store_true',
                        help='Try every loaded non-cooperative algo if earlier candidates fail. Default keeps original top-5/top-10 behavior.')
    parser.add_argument('--algo_limit', type=int, default=None,
                        help='Try at most this many candidates from the loaded config. Overrides --try_all_algos if provided.')
    parser.add_argument('--algo_id', type=int, default=None,
                        help='Try only this specific algo id from the loaded config.')
    parser.add_argument('--debug_search', action='store_true',
                    help='Print predictive-search candidate estimates and why the selected cSeg won.')
    parser.add_argument('--debug_search_topk', type=int, default=20,
                    help='How many top predictive-search candidates to print when --debug_search is enabled.')
    args = parser.parse_args()

    if args.predictive_search or args.m * args.n > 33554432:
        comm_array = torch.load(repo_root() / "configs" / f"bandwidth_{args.comm_op}_tp{world_size}.pt")
        print("Bandwidth curve captured.")
        fast_search(
            args.m,
            args.n,
            args.k,
            comm_array,
            args.comm_op,
            args.comm_sm_slack,
            args.min_group_size,
            debug_hint=args.debug_hint,
            debug_hint_dump=args.debug_hint_dump,
            try_all_algos=args.try_all_algos,
            algo_limit=args.algo_limit,
            algo_id=args.algo_id,
            min_effective_group_size_override=args.min_effective_group_size,
            effective_hint_confirm=not args.no_effective_hint_confirm,
            debug_search=args.debug_search,
            debug_search_topk=args.debug_search_topk,
        )
    else:
        exhaustive_search(
            args.m,
            args.n,
            args.k,
            args.comm_op,
            args.comm_sm_slack,
            debug_hint=args.debug_hint,
            debug_hint_dump=args.debug_hint_dump,
            try_all_algos=args.try_all_algos,
            algo_limit=args.algo_limit,
            algo_id=args.algo_id,
        )


if __name__ == "__main__":
    main()
