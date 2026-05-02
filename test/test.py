'''
    Using multiprocessing for distributed running,
    please specify the GPUs via CUDA_VISIBLE_DEVICES:
        CUDA_VISIBLE_DEVICES=0,1 python test/test.py --m 4096 --n 8192 --k 4096

    This test can run:
      - NCCL overlap solution
      - ooverlap overlap solution
      - both backends in one run

    It compares each backend overlap result against that backend's own baseline:
      - default: same SM90 Algo/BM/BN with cSeg=[tile_num], no overlap segmentation
      - optional: cublas baseline
'''

import argparse
import importlib.util
import json
import os
import sys
import uuid
from contextlib import contextmanager
from pathlib import Path

import torch
import torch.multiprocessing as mp


WARM_UP = 20
REP = 200


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


@contextmanager
def temporary_env(env_updates):
    old_values = {}
    missing = set()

    for key, value in env_updates.items():
        if key in os.environ:
            old_values[key] = os.environ[key]
        else:
            missing.add(key)
        os.environ[key] = str(value)

    try:
        yield
    finally:
        for key in env_updates:
            if key in old_values:
                os.environ[key] = old_values[key]
            elif key in missing:
                os.environ.pop(key, None)


def div_up(x: int, y: int):
    return (x + y - 1) // y


def gpu_config_name():
    device = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device)
    return props.name.lower().replace(" ", "_")


def make_broker_key(prefix: str = "oo"):
    # Broker key is used in a Unix-domain socket path. Keep it short.
    _ = prefix
    pid_part = format(os.getpid() & 0xffff, "04x")
    rand_part = uuid.uuid4().hex[:8]
    return f"oo{pid_part}{rand_part}"


def validate_comm_backend(comm_backend: str, comm_op: str, world_size: int):
    if comm_backend not in ["nccl", "ooverlap"]:
        raise ValueError(f"Unsupported comm_backend={comm_backend}")

    if comm_op not in ["all_reduce", "reduce_scatter"]:
        raise ValueError(f"Unsupported comm_op={comm_op}")

    if comm_backend == "ooverlap":
        if comm_op != "all_reduce":
            raise ValueError("ooverlap backend currently supports only --comm_op all_reduce")
        if world_size != 2:
            raise ValueError("ooverlap backend currently supports exactly 2 visible GPUs")


def init_overlap_backend(gemm_class, rank: int, world_size: int, nccl_id, broker_key: str,
                         comm_backend: str, comm_op: str):
    validate_comm_backend(comm_backend, comm_op, world_size)

    if comm_backend == "nccl":
        gemm_class.nccl_init(rank, world_size, nccl_id)
    elif comm_backend == "ooverlap":
        gemm_class.ooverlap_ipc_init(rank, world_size, list(range(world_size)), broker_key)
    else:
        raise ValueError(f"Unsupported comm_backend={comm_backend}")


def release_overlap_backend(gemm_class, comm_backend: str):
    if comm_backend == "ooverlap":
        gemm_class.ooverlap_release()


def solution_json_path(M: int, N: int, K: int, comm_backend: str):
    gpu_name = gpu_config_name()
    return repo_root() / "configs" / f"solution_{comm_backend}_m{M}n{N}k{K}_{gpu_name}_packed_sm90.json"


def legacy_solution_json_path(M: int, N: int, K: int):
    gpu_name = gpu_config_name()
    return repo_root() / "configs" / f"solution_m{M}n{N}k{K}_{gpu_name}_packed_sm90.json"


def find_solution_json(M: int, N: int, K: int, comm_backend: str):
    path = solution_json_path(M, N, K, comm_backend)
    if path.exists():
        return path

    if comm_backend == "nccl":
        legacy = legacy_solution_json_path(M, N, K)
        if legacy.exists():
            return legacy

    raise FileNotFoundError(
        f"Could not find solution JSON for backend={comm_backend}:\n"
        f"  {path}\n"
        "Run tool/search.py for this backend first."
    )


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
        MonitoredMatrix[: seg_size + TileNum + 1] = 0
    else:
        MonitoredMatrix[: seg_size + TileNum] = 0


def reorder_indices(S, hint):
    original = list(range(S))
    new_order = [-1] * S

    for i, element in enumerate(hint):
        new_order[int(element)] = i

    hint_set = set(int(x) for x in hint)
    remaining_elements = [x for x in original if x not in hint_set]
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


def generate_row_remap_array(M, N, BM, BN, S_list, world_size, device="cuda"):
    total_tiles = (M * N) // (BM * BN)
    assert sum(S_list) == total_tiles, "sum(S_list) must equal total number of tiles"

    original_row_ids = torch.arange(M * N // BN, dtype=torch.int, device=device)
    reordered_row_id = torch.empty_like(original_row_ids)

    current_row = 0
    for S in S_list:
        chunk_size = S * BM
        chunk_row_ids = original_row_ids[current_row: current_row + chunk_size]

        mod_values = chunk_row_ids % world_size
        _, sorted_indices = torch.sort(mod_values, stable=True)
        reordered_chunk = chunk_row_ids[sorted_indices]

        reordered_row_id[current_row: current_row + chunk_size] = reordered_chunk
        current_row += chunk_size

    remap = torch.empty_like(original_row_ids)
    remap[reordered_row_id] = torch.arange(len(reordered_row_id), dtype=torch.int, device=device)

    return remap


def perf_running_process(rank, world_size, nccl_id, broker_key, comm_backend,
    M: int, N: int, K: int,
    BM: int, BN: int, Algo: int, cSeg: list, hint: list, reorder_map,
    comm_op: str,
    active_sm_count: int,
    set_ooverlap_comm_ctas: bool,
    ooverlap_comm_ctas: int,
    result_dict):

    torch.cuda.set_device(rank)
    validate_comm_backend(comm_backend, comm_op, world_size)

    if set_ooverlap_comm_ctas:
        os.environ["OOVERLAP_MAX_CTAS"] = str(int(ooverlap_comm_ctas))
        os.environ["NCCL_MAX_CTAS"] = str(int(ooverlap_comm_ctas))

    cSeg_CPU = torch.tensor(cSeg, dtype=torch.int32)
    cSeg_GPU = cSeg_CPU.cuda(rank)

    TileNum = div_up(M, BM) * div_up(N, BN)

    gemm_class = ext.OverlapImpl()

    try:
        init_overlap_backend(
            gemm_class,
            rank,
            world_size,
            nccl_id,
            broker_key,
            comm_backend,
            comm_op,
        )

        gemm_class.cutlass_init()
        gemm_class.overlap_init()

        A = torch.empty((M, K), dtype=torch.float16, device="cuda").normal_(mean=0.0, std=0.5)
        B = torch.empty((N, K), dtype=torch.float16, device="cuda").normal_(mean=0.0, std=0.5)

        packed_M, packed_N = packed_shape(M, N, BM, BN, 1)
        C = torch.empty((packed_M, packed_N), dtype=torch.float16, device="cuda")

        MonitoredMatrix = torch.zeros(
            (monitor_size(TileNum, len(cSeg), False),),
            dtype=torch.int,
            device="cuda",
        )

        ReorderedArray = make_reordered_array(TileNum, hint, reorder_map).reshape(
            ((M + BM - 1) // BM, (N + BN - 1) // BN)
        )

        D = None
        RowArray = None
        if comm_op == "reduce_scatter":
            D = torch.empty((M // world_size, N), dtype=torch.float16, device="cuda")
            RowArray = generate_row_remap_array(M, N, BM, BN, cSeg, world_size)

        if comm_op == "all_reduce":
            for _ in range(WARM_UP):
                reset_monitor_matrix(MonitoredMatrix, TileNum, len(cSeg), False)
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
                    False,
                )

            start_event = [torch.cuda.Event(enable_timing=True) for _ in range(REP)]
            end_event = [torch.cuda.Event(enable_timing=True) for _ in range(REP)]

            for i in range(REP):
                reset_monitor_matrix(MonitoredMatrix, TileNum, len(cSeg), False)
                start_event[i].record()
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
                    False,
                )
                end_event[i].record()

            torch.cuda.synchronize()
            dur = torch.tensor(
                [s.elapsed_time(e) for s, e in zip(start_event, end_event)],
                dtype=torch.float,
            )

        elif comm_op == "reduce_scatter":
            for _ in range(WARM_UP):
                reset_monitor_matrix(MonitoredMatrix, TileNum, len(cSeg), False)
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
                    False,
                )

            start_event = [torch.cuda.Event(enable_timing=True) for _ in range(REP)]
            end_event = [torch.cuda.Event(enable_timing=True) for _ in range(REP)]

            for i in range(REP):
                reset_monitor_matrix(MonitoredMatrix, TileNum, len(cSeg), False)
                start_event[i].record()
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
                    False,
                )
                end_event[i].record()

            torch.cuda.synchronize()
            dur = torch.tensor(
                [s.elapsed_time(e) for s, e in zip(start_event, end_event)],
                dtype=torch.float,
            )

        else:
            raise ValueError(f"Unknown comm_op={comm_op}")

        result_dict[rank] = torch.mean(dur).item()

    finally:
        release_overlap_backend(gemm_class, comm_backend)


def perf_running(M: int, N: int, K: int,
    BM: int, BN: int, Algo: int,
    cSeg: list, hint: list, comm_op: str,
    comm_backend: str,
    active_sm_count: int,
    reorder_map=None,
    set_ooverlap_comm_ctas: bool = False,
    ooverlap_comm_ctas: int = 0):

    world_size = torch.cuda.device_count()
    if world_size < 2:
        raise RuntimeError("At least 2 GPUs are required for this program.")

    validate_comm_backend(comm_backend, comm_op, world_size)

    if set_ooverlap_comm_ctas and comm_backend != "ooverlap":
        raise ValueError("set_ooverlap_comm_ctas can only be used with comm_backend=ooverlap")

    if set_ooverlap_comm_ctas and int(ooverlap_comm_ctas) <= 0:
        raise ValueError(f"ooverlap_comm_ctas must be > 0, got {ooverlap_comm_ctas}")

    nccl_id = ext.generate_nccl_id()
    broker_key = make_broker_key(f"test_{comm_backend}")
    torch.cuda.synchronize()

    manager = mp.Manager()
    result_dict = manager.dict()

    env_updates = {}
    if set_ooverlap_comm_ctas:
        env_updates = {
            "OOVERLAP_MAX_CTAS": str(int(ooverlap_comm_ctas)),
            "NCCL_MAX_CTAS": str(int(ooverlap_comm_ctas)),
        }

    with temporary_env(env_updates):
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
                reorder_map,
                comm_op,
                active_sm_count,
                set_ooverlap_comm_ctas,
                int(ooverlap_comm_ctas),
                result_dict,
            ),
            nprocs=world_size,
        )

    dur = torch.empty((world_size))
    for i in range(world_size):
        dur[i] = result_dict[i]

    return dur.max().item()


def perf_comm_process(rank, world_size, nccl_id, broker_key, comm_backend,
                      M, N, comm_op, result_dict):
    torch.cuda.set_device(rank)
    validate_comm_backend(comm_backend, comm_op, world_size)

    comm_class = ext.OverlapImpl()

    try:
        init_overlap_backend(
            comm_class,
            rank,
            world_size,
            nccl_id,
            broker_key,
            comm_backend,
            comm_op,
        )

        comm_class.cutlass_init()
        comm_class.overlap_init()

        C = torch.empty((M, N), dtype=torch.float16, device="cuda").normal_(mean=0.0, std=0.5)

        if comm_op == "reduce_scatter":
            D = torch.empty((M // world_size, N), dtype=torch.float16, device="cuda")
        else:
            D = None

        if comm_op == "all_reduce":
            if comm_backend == "nccl":
                for _ in range(WARM_UP):
                    comm_class.nccl_allreduce(C)

                start_event = [torch.cuda.Event(enable_timing=True) for _ in range(REP)]
                end_event = [torch.cuda.Event(enable_timing=True) for _ in range(REP)]

                for i in range(REP):
                    start_event[i].record()
                    comm_class.nccl_allreduce(C)
                    end_event[i].record()

            elif comm_backend == "ooverlap":
                for _ in range(WARM_UP):
                    comm_class.ooverlap_allreduce(C)

                start_event = [torch.cuda.Event(enable_timing=True) for _ in range(REP)]
                end_event = [torch.cuda.Event(enable_timing=True) for _ in range(REP)]

                for i in range(REP):
                    start_event[i].record()
                    comm_class.ooverlap_allreduce(C)
                    end_event[i].record()

            else:
                raise ValueError(f"Unknown comm_backend={comm_backend}")

        elif comm_op == "reduce_scatter":
            for _ in range(WARM_UP):
                comm_class.nccl_reducescatter(C, D)

            start_event = [torch.cuda.Event(enable_timing=True) for _ in range(REP)]
            end_event = [torch.cuda.Event(enable_timing=True) for _ in range(REP)]

            for i in range(REP):
                start_event[i].record()
                comm_class.nccl_reducescatter(C, D)
                end_event[i].record()

        else:
            raise ValueError(f"Unknown comm_op={comm_op}")

        torch.cuda.synchronize()
        dur = torch.tensor(
            [s.elapsed_time(e) for s, e in zip(start_event, end_event)],
            dtype=torch.float,
        )
        result_dict[rank] = torch.mean(dur).item()

    finally:
        release_overlap_backend(comm_class, comm_backend)


def perf_comm(M: int, N: int, comm_op: str, comm_backend: str):
    world_size = torch.cuda.device_count()
    if world_size < 2:
        raise RuntimeError("At least 2 GPUs are required.")

    validate_comm_backend(comm_backend, comm_op, world_size)

    nccl_id = ext.generate_nccl_id()
    broker_key = make_broker_key(f"comm_{comm_backend}")
    torch.cuda.synchronize()

    manager = mp.Manager()
    result_dict = manager.dict()

    mp.spawn(
        perf_comm_process,
        args=(world_size, nccl_id, broker_key, comm_backend, M, N, comm_op, result_dict),
        nprocs=world_size,
    )

    dur = torch.empty((world_size))
    for i in range(world_size):
        dur[i] = result_dict[i]

    return dur.max().item()


def perf_cublas_baseline_process(rank, world_size, nccl_id, broker_key, comm_backend,
                                 M, N, K, comm_op, result_dict):
    torch.cuda.set_device(rank)
    validate_comm_backend(comm_backend, comm_op, world_size)

    A = torch.empty((M, K), dtype=torch.float16, device="cuda").normal_(mean=0.0, std=0.5)
    B = torch.empty((N, K), dtype=torch.float16, device="cuda").normal_(mean=0.0, std=0.5)

    if comm_backend == "nccl":
        C = torch.empty((M, N), dtype=torch.float16, device="cuda")

        if comm_op == "reduce_scatter":
            D = torch.empty((M // world_size, N), dtype=torch.float16, device="cuda")
        else:
            D = None

        gemm_comm = ext.BaselineImpl()
        gemm_comm.nccl_init(rank, world_size, nccl_id)
        gemm_comm.cublas_init()

        if comm_op == "all_reduce":
            for _ in range(WARM_UP):
                gemm_comm.gemm_allreduce(A, B, C)

            start_event = [torch.cuda.Event(enable_timing=True) for _ in range(REP)]
            end_event = [torch.cuda.Event(enable_timing=True) for _ in range(REP)]

            for i in range(REP):
                start_event[i].record()
                gemm_comm.gemm_allreduce(A, B, C)
                end_event[i].record()

        elif comm_op == "reduce_scatter":
            for _ in range(WARM_UP):
                gemm_comm.gemm_reducescatter(A, B, C, D)

            start_event = [torch.cuda.Event(enable_timing=True) for _ in range(REP)]
            end_event = [torch.cuda.Event(enable_timing=True) for _ in range(REP)]

            for i in range(REP):
                start_event[i].record()
                gemm_comm.gemm_reducescatter(A, B, C, D)
                end_event[i].record()

        else:
            raise ValueError(f"Unknown comm_op={comm_op}")

    elif comm_backend == "ooverlap":
        if comm_op != "all_reduce":
            raise ValueError("ooverlap cublas baseline supports only all_reduce")

        # baseline_gemm_col expects physical C shape [N, M].
        C = torch.empty((N, M), dtype=torch.float16, device="cuda")

        comm_class = ext.OverlapImpl()
        try:
            comm_class.ooverlap_ipc_init(rank, world_size, list(range(world_size)), broker_key)
            comm_class.cutlass_init()

            for _ in range(WARM_UP):
                ext.baseline_gemm_col(A, B, C)
                comm_class.ooverlap_allreduce(C)

            start_event = [torch.cuda.Event(enable_timing=True) for _ in range(REP)]
            end_event = [torch.cuda.Event(enable_timing=True) for _ in range(REP)]

            for i in range(REP):
                start_event[i].record()
                ext.baseline_gemm_col(A, B, C)
                comm_class.ooverlap_allreduce(C)
                end_event[i].record()

        finally:
            comm_class.ooverlap_release()

    else:
        raise ValueError(f"Unknown comm_backend={comm_backend}")

    torch.cuda.synchronize()
    dur = torch.tensor(
        [s.elapsed_time(e) for s, e in zip(start_event, end_event)],
        dtype=torch.float,
    )
    result_dict[rank] = torch.mean(dur).item()


def perf_cublas_baseline(M: int, N: int, K: int, comm_op: str, comm_backend: str):
    world_size = torch.cuda.device_count()
    if world_size < 2:
        raise RuntimeError("At least 2 GPUs are required.")

    validate_comm_backend(comm_backend, comm_op, world_size)

    nccl_id = ext.generate_nccl_id()
    broker_key = make_broker_key(f"cublas_{comm_backend}")
    torch.cuda.synchronize()

    manager = mp.Manager()
    result_dict = manager.dict()

    mp.spawn(
        perf_cublas_baseline_process,
        args=(world_size, nccl_id, broker_key, comm_backend, M, N, K, comm_op, result_dict),
        nprocs=world_size,
    )

    dur = torch.empty((world_size))
    for i in range(world_size):
        dur[i] = result_dict[i]

    return dur.max().item()


def load_solution(m: int, n: int, k: int, comm_backend: str):
    file_path = find_solution_json(m, n, k, comm_backend)

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    json_backend = data.get("comm_backend")
    if json_backend is not None and json_backend != comm_backend:
        print(
            f"WARNING: loaded solution has comm_backend={json_backend}, "
            f"but requested backend={comm_backend}"
        )

    return file_path, data


def run_backend(args, comm_backend: str):
    world_size = torch.cuda.device_count()
    validate_comm_backend(comm_backend, args.comm_op, world_size)

    m, n, k = args.m, args.n, args.k

    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    sm_count = props.multi_processor_count

    file_path, data = load_solution(m, n, k, comm_backend)

    BM = int(data["BM"])
    BN = int(data["BN"])
    Algo = int(data["Algo"])

    tile_num = div_up(m, BM) * div_up(n, BN)
    cSeg = [int(x) for x in data["cSeg"]]
    hint = [int(x) for x in data["hint"]]
    reorder_map = data.get("reorder_map")

    active_sm_count = int(data.get("compute_sms", sm_count))
    comm_sm_slack = int(data.get("comm_sm_slack", sm_count - active_sm_count))
    cseg_sum = int(sum(cSeg))

    assert cseg_sum == tile_num, f"sum(cSeg)={cseg_sum} must equal tile_num={tile_num}"

    set_overlap_comm_ctas = (
        bool(args.set_ooverlap_comm_ctas_to_comm_sms)
        and comm_backend == "ooverlap"
    )

    print("")
    print("########################################")
    print(f"# Running test for comm_backend={comm_backend}")
    print("########################################")
    print(f"Loaded solution: {file_path}")
    print("Solution debug:")
    print(f"  Algo={Algo} BM={BM} BN={BN}")
    print(f"  sm_count={sm_count} json_sm_count={data.get('sm_count')}")
    print(f"  comm_sm_slack={comm_sm_slack} active_sm_count={active_sm_count}")
    print(f"  cSeg={cSeg}")
    print(f"  len(cSeg)={len(cSeg)} sum(cSeg)={cseg_sum} tile_num={tile_num}")
    print(f"  hint_len={len(hint)}")
    print(f"  has_reorder_map={'reorder_map' in data} reorder_map_len={len(reorder_map or [])}")
    print(f"  baseline_impl={args.baseline_impl}")
    if set_overlap_comm_ctas:
        print(
            "  ooverlap overlap env override: "
            f"OOVERLAP_MAX_CTAS={comm_sm_slack} NCCL_MAX_CTAS={comm_sm_slack}"
        )

    config_gemm_dur = float(data["dur"])
    comm_dur = perf_comm(m, n, args.comm_op, comm_backend)

    overlap_dur = perf_running(
        m,
        n,
        k,
        BM,
        BN,
        Algo,
        cSeg,
        hint,
        args.comm_op,
        comm_backend,
        active_sm_count,
        reorder_map,
        set_ooverlap_comm_ctas=set_overlap_comm_ctas,
        ooverlap_comm_ctas=comm_sm_slack,
    )

    if args.baseline_impl == "same_algo":
        baseline_cSeg = [tile_num]
        baseline_active_sm_count = (
            sm_count if args.baseline_same_algo_active_sms == "all" else active_sm_count
        )

        # Important:
        # Do NOT set OOVERLAP_MAX_CTAS/NCCL_MAX_CTAS for this baseline call.
        # The requested env override is only for the real ooverlap overlap scenario.
        baseline_dur = perf_running(
            m,
            n,
            k,
            BM,
            BN,
            Algo,
            baseline_cSeg,
            hint,
            args.comm_op,
            comm_backend,
            baseline_active_sm_count,
            reorder_map,
            set_ooverlap_comm_ctas=False,
            ooverlap_comm_ctas=0,
        )

        baseline_desc = "same_algo_full_segment"
        baseline_sms = str(baseline_active_sm_count)

    elif args.baseline_impl == "cublas":
        baseline_cSeg = ["cublas"]
        baseline_dur = perf_cublas_baseline(m, n, k, args.comm_op, comm_backend)
        baseline_desc = "cublas"
        baseline_sms = "cublas"

    else:
        raise ValueError(f"Unknown baseline_impl={args.baseline_impl}")

    speedup = baseline_dur / overlap_dur

    env_override_desc = (
        f"OOVERLAP_MAX_CTAS=NCCL_MAX_CTAS={comm_sm_slack}"
        if set_overlap_comm_ctas
        else "off"
    )

    print(f"""
        {'Item':<28} {'Value':>18}
        {'-----':<28} {'-----':>18}
        {'comm_backend':<28} {comm_backend:>18}
        {'comm_op':<28} {args.comm_op:>18}
        {'m':<28} {m:>18}
        {'n':<28} {n:>18}
        {'k':<28} {k:>18}
        {'tile_num':<28} {tile_num:>18}
        {'Algo':<28} {Algo:>18}
        {'BM':<28} {BM:>18}
        {'BN':<28} {BN:>18}
        {'cSeg':<28} {str(cSeg):>18}
        {'cSeg_len':<28} {len(cSeg):>18}
        {'active_sm_count':<28} {active_sm_count:>18}
        {'comm_sm_slack':<28} {comm_sm_slack:>18}
        {'overlap_comm_ctas_env':<28} {env_override_desc:>18}
        {'baseline_impl':<28} {baseline_desc:>18}
        {'baseline_cSeg':<28} {str(baseline_cSeg):>18}
        {'baseline_sms':<28} {baseline_sms:>18}
        {'config_gemm_dur (ms)':<28} {config_gemm_dur:>18.4f}
        {'comm_dur (ms)':<28} {comm_dur:>18.4f}
        {'baseline_dur (ms)':<28} {baseline_dur:>18.4f}
        {'overlap_dur (ms)':<28} {overlap_dur:>18.4f}
        {'speedup':<28} {speedup:>18.4f}
    """)

    return {
        "comm_backend": comm_backend,
        "comm_op": args.comm_op,
        "baseline_impl": baseline_desc,
        "baseline_dur_ms": float(baseline_dur),
        "overlap_dur_ms": float(overlap_dur),
        "speedup": float(speedup),
        "comm_dur_ms": float(comm_dur),
        "config_gemm_dur_ms": float(config_gemm_dur),
        "cSeg": cSeg,
        "active_sm_count": int(active_sm_count),
        "comm_sm_slack": int(comm_sm_slack),
        "overlap_comm_ctas_env": env_override_desc,
        "solution": str(file_path),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, default=4096)
    parser.add_argument("--k", type=int, default=8192)
    parser.add_argument("--n", type=int, default=8192)
    parser.add_argument("--comm_op", type=str, default="all_reduce", choices=["all_reduce", "reduce_scatter"])
    parser.add_argument(
        "--comm_backend",
        type=str,
        default="both",
        choices=["both", "nccl", "ooverlap"],
        help="Run one backend or both. ooverlap currently supports all_reduce only.",
    )
    parser.add_argument(
        "--baseline_impl",
        type=str,
        default="same_algo",
        choices=["same_algo", "cublas"],
        help=(
            "same_algo uses the same SM90 CUTLASS Algo/BM/BN as overlap with "
            "cSeg=[tile_num]. cublas uses the older cuBLAS GEMM baseline."
        ),
    )
    parser.add_argument(
        "--baseline_same_algo_active_sms",
        type=str,
        default="all",
        choices=["all", "solution"],
        help=(
            "For --baseline_impl same_algo: use all physical SMs for the "
            "one-segment baseline, or use the solution compute_sms."
        ),
    )
    parser.add_argument(
        "--set_ooverlap_comm_ctas_to_comm_sms",
        action="store_true",
        help=(
            "Only for the real ooverlap overlap run: set both OOVERLAP_MAX_CTAS "
            "and NCCL_MAX_CTAS to comm_sm_slack before spawning the worker "
            "processes. This is intentionally not applied to baseline runs."
        ),
    )

    args = parser.parse_args()

    world_size = torch.cuda.device_count()
    if world_size < 2:
        raise RuntimeError("At least 2 GPUs are required.")

    if args.comm_backend == "both":
        backends = ["nccl", "ooverlap"]
    else:
        backends = [args.comm_backend]

    results = []

    for backend in backends:
        if backend == "ooverlap" and args.comm_op != "all_reduce":
            print("")
            print("Skipping ooverlap: ooverlap backend currently supports only all_reduce.")
            continue

        results.append(run_backend(args, backend))

    if len(results) > 1:
        print("")
        print("########################################")
        print("# Summary")
        print("########################################")
        print(
            f"{'backend':<12} {'baseline(ms)':>14} {'overlap(ms)':>14} "
            f"{'speedup':>10} {'comm(ms)':>12} {'ctas_env':>20} {'cSeg':>24}"
        )
        print(
            f"{'-------':<12} {'------------':>14} {'-----------':>14} "
            f"{'-------':>10} {'--------':>12} {'--------':>20} {'----':>24}"
        )

        for row in results:
            print(
                f"{row['comm_backend']:<12} "
                f"{row['baseline_dur_ms']:>14.4f} "
                f"{row['overlap_dur_ms']:>14.4f} "
                f"{row['speedup']:>10.4f} "
                f"{row['comm_dur_ms']:>12.4f} "
                f"{row['overlap_comm_ctas_env']:>20} "
                f"{str(row['cSeg']):>24}"
            )


if __name__ == "__main__":
    main()
