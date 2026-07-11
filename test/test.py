#!/usr/bin/env python3
"""
Run overlap tests for NCCL, ooverlap, or both.

Baselines, both enabled by default:
  - cublas: cuBLAS GEMM + communication
  - plain:  gemm_plain_sm90 with the same Algo + communication

CTA env behavior:
  - --set_nccl_comm_ctas_to_comm_sms applies only to the NCCL overlap run.
  - --set_ooverlap_comm_ctas_to_comm_sms applies only to the ooverlap overlap run.
  - Baselines always clear OOVERLAP_MAX_CTAS and NCCL_MAX_CTAS.
  - Standalone comm timing always clears OOVERLAP_MAX_CTAS and NCCL_MAX_CTAS.
"""

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


WARM_UP = int(os.environ.get("OOVERLAP_TEST_WARMUP", "20"))
REP = int(os.environ.get("OOVERLAP_TEST_REP", "200"))
SYNC_EACH_ITER = os.environ.get("OOVERLAP_TEST_SYNC_EACH_ITER", "1") != "0"
COMM_CTA_ENV_KEYS = ("OOVERLAP_MAX_CTAS", "NCCL_MAX_CTAS")


def repo_root():
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


ext = None

def ensure_ext():
    global ext
    if ext is None:
        ext = load_ooverlap_ext()
    return ext


@contextmanager
def scoped_env(set_values=None, unset_keys=None):
    set_values = dict(set_values or {})
    unset_keys = set(unset_keys or [])

    touched = set(set_values.keys()) | set(unset_keys)
    old_present = {k: k in os.environ for k in touched}
    old_values = {k: os.environ.get(k) for k in touched}

    for k in unset_keys:
        os.environ.pop(k, None)

    for k, v in set_values.items():
        os.environ[k] = str(v)

    try:
        yield
    finally:
        for k in touched:
            if old_present[k]:
                os.environ[k] = old_values[k]
            else:
                os.environ.pop(k, None)


def clear_comm_cta_env_in_child():
    for k in COMM_CTA_ENV_KEYS:
        os.environ.pop(k, None)


def set_comm_cta_env_in_child(comm_sm_slack: int):
    v = str(int(comm_sm_slack))
    os.environ["OOVERLAP_MAX_CTAS"] = v
    os.environ["NCCL_MAX_CTAS"] = v


def comm_cta_env_values(comm_sm_slack: int):
    v = str(int(comm_sm_slack))
    return {
        "OOVERLAP_MAX_CTAS": v,
        "NCCL_MAX_CTAS": v,
    }


def div_up(x: int, y: int):
    return (x + y - 1) // y


def gpu_config_name():
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    return props.name.lower().replace(" ", "_")


def make_broker_key():
    # Broker key becomes part of a Unix-domain socket path. Keep it short.
    pid_part = format(os.getpid() & 0xffff, "04x")
    rand_part = uuid.uuid4().hex[:8]
    return f"oo{pid_part}{rand_part}"


def validate_backend(comm_backend: str, comm_op: str, world_size: int):
    if comm_backend not in ("nccl", "ooverlap"):
        raise ValueError(f"Unsupported comm_backend={comm_backend}")

    if comm_op not in ("all_reduce", "reduce_scatter"):
        raise ValueError(f"Unsupported comm_op={comm_op}")

    if comm_backend == "ooverlap":
        if comm_op != "all_reduce":
            raise ValueError("ooverlap backend currently supports only all_reduce")
        if world_size != 2:
            raise ValueError("ooverlap backend currently supports exactly 2 visible GPUs")


def method(obj, names):
    for name in names:
        if hasattr(obj, name):
            return getattr(obj, name)
    raise AttributeError(
        f"{type(obj).__name__} has none of these methods: {', '.join(names)}"
    )


def init_overlap_backend(obj, rank, world_size, nccl_id, broker_key, comm_backend, comm_op):
    validate_backend(comm_backend, comm_op, world_size)

    if comm_backend == "nccl":
        method(obj, ("nccl_init",))(rank, world_size, nccl_id)
    else:
        method(obj, ("ooverlap_ipc_init",))(rank, world_size, list(range(world_size)), broker_key)


def release_overlap_backend(obj, comm_backend: str):
    if comm_backend == "ooverlap" and hasattr(obj, "ooverlap_release"):
        obj.ooverlap_release()

def sync_and_release_overlap_backend(obj, comm_backend: str):
    try:
        torch.cuda.synchronize()
    finally:
        release_overlap_backend(obj, comm_backend)


def init_baseline_nccl(obj, rank, world_size, nccl_id):
    method(obj, ("nccl_init",))(rank, world_size, nccl_id)

    if hasattr(obj, "cublas_init"):
        obj.cublas_init()


def solution_json_path(M: int, N: int, K: int, comm_backend: str):
    gpu = gpu_config_name()
    return repo_root() / "configs" / f"solution_{comm_backend}_m{M}n{N}k{K}_{gpu}_packed_sm90.json"


def legacy_solution_json_path(M: int, N: int, K: int):
    gpu = gpu_config_name()
    return repo_root() / "configs" / f"solution_m{M}n{N}k{K}_{gpu}_packed_sm90.json"


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
        "Run tool/search.py first."
    )


def load_solution(M: int, N: int, K: int, comm_backend: str):
    path = find_solution_json(M, N, K, comm_backend)
    data = json.loads(path.read_text())

    json_backend = data.get("comm_backend")
    if json_backend is not None and json_backend != comm_backend:
        print(f"WARNING: loaded solution has comm_backend={json_backend}, requested={comm_backend}")

    return path, data


def packed_shape(M: int, N: int, BM: int, BN: int, rLDN: int = 1):
    tile_num = div_up(M, BM) * div_up(N, BN)
    packed_tile_rows = div_up(tile_num, rLDN)
    return packed_tile_rows * BM, rLDN * BN


def monitor_size(tile_num: int, seg_size: int, monitor: bool):
    if monitor:
        return seg_size + tile_num + 1 + tile_num
    return seg_size + tile_num


def reset_monitor_matrix(mm, tile_num: int, seg_size: int, monitor: bool):
    if monitor:
        mm[: seg_size + tile_num + 1] = 0
    else:
        mm[: seg_size + tile_num] = 0


def reorder_indices(tile_num: int, hint):
    new_order = [-1] * tile_num

    for i, x in enumerate(hint):
        new_order[int(x)] = i

    used = set(int(x) for x in hint)
    tail = [x for x in range(tile_num) if x not in used]

    for i, x in enumerate(tail, start=len(hint)):
        new_order[x] = i

    return torch.tensor(new_order, dtype=torch.int, device="cuda")


def make_reordered_array(tile_num: int, hint, reorder_map=None):
    if reorder_map is not None:
        if len(reorder_map) != tile_num:
            raise ValueError(f"reorder_map length={len(reorder_map)}, expected={tile_num}")
        return torch.tensor([int(x) for x in reorder_map], dtype=torch.int, device="cuda")

    return reorder_indices(tile_num, hint)


def generate_row_remap_array(M, N, BM, BN, cSeg, world_size, device="cuda"):
    total_tiles = (M * N) // (BM * BN)
    assert sum(cSeg) == total_tiles, "sum(cSeg) must equal total number of tiles"

    original = torch.arange(M * N // BN, dtype=torch.int, device=device)
    reordered = torch.empty_like(original)

    cur = 0
    for S in cSeg:
        chunk_size = int(S) * BM
        chunk = original[cur: cur + chunk_size]
        _, idx = torch.sort(chunk % world_size, stable=True)
        reordered[cur: cur + chunk_size] = chunk[idx]
        cur += chunk_size

    remap = torch.empty_like(original)
    remap[reordered] = torch.arange(len(reordered), dtype=torch.int, device=device)
    return remap


def mean_timed(fn, pre_fn = None):
    for _ in range(WARM_UP):
        if pre_fn != None:
            pre_fn()
        fn()
        if SYNC_EACH_ITER:
            torch.cuda.synchronize()

    torch.cuda.synchronize()

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(REP)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(REP)]

    for i in range(REP):
        if pre_fn != None:
            pre_fn()
        starts[i].record()
        fn()
        ends[i].record()

        # Keep only one benchmark iteration in flight.  This is important for
        # the ooverlap IPC backend: its collectives/imported IPC resources are
        # stream-ordered, but queuing many API calls before any host/device sync
        # can leave multiple IPC collective generations alive at once.
        if SYNC_EACH_ITER:
            torch.cuda.synchronize()

    torch.cuda.synchronize()

    vals = [s.elapsed_time(e) for s, e in zip(starts, ends)]
    if os.environ.get("OOVERLAP_TEST_PRINT_STATS", "0") == "1":
        sv = sorted(vals)

        def pct(p):
            if len(sv) == 1:
                return sv[0]
            idx = int(round((p / 100.0) * (len(sv) - 1)))
            return sv[max(0, min(idx, len(sv) - 1))]

        print(
            "timing_stats "
            f"n={len(sv)} "
            f"mean={sum(sv) / len(sv):.4f} "
            f"min={sv[0]:.4f} "
            f"p50={pct(50):.4f} "
            f"p90={pct(90):.4f} "
            f"p99={pct(99):.4f} "
            f"max={sv[-1]:.4f}",
            flush=True,
        )

    return float(sum(vals)) / float(len(vals))


def call_nccl_allreduce(obj, C):
    method(obj, ("nccl_allreduce",))(C)


def call_nccl_reducescatter(obj, C, D):
    fn = method(obj, ("nccl_reducescatter",))
    try:
        fn(C, D)
    except TypeError:
        fn(C)


def call_ooverlap_allreduce(obj, C):
    method(obj, ("ooverlap_allreduce",))(C)


def call_module_cublas_gemm(A, B, C_col):
    ensure_ext()
    fn = getattr(ext, "baseline_gemm_col", None)
    if fn is None:
        raise RuntimeError("ooverlap_ext.baseline_gemm_col was not found")
    fn(A, B, C_col)


def call_module_plain_gemm(A, B, C_col, Algo):
    ensure_ext()
    fn = getattr(ext, "gemm_plain_sm90", None)
    if fn is None:
        raise RuntimeError("ooverlap_ext.gemm_plain_sm90 was not found")
    fn(A, B, C_col, int(Algo))


def call_overlap_allreduce(obj, A, B, C, MM, RA, cSeg_CPU, cSeg_GPU, Algo, active_sm_count):
    fn = method(obj, ("gemm_allreduce_overlap",))
    try:
        fn(
            A,
            B,
            C,
            MM,
            RA,
            1,
            cSeg_CPU,
            cSeg_GPU,
            int(Algo),
            int(active_sm_count),
            False,
        )
    except TypeError:
        fn(
            A,
            B,
            C,
            MM,
            RA,
            1,
            cSeg_CPU,
            cSeg_GPU,
            int(Algo),
            False,
        )


def call_overlap_reducescatter(obj, A, B, C, D, MM, RA, RowArray, cSeg_CPU, cSeg_GPU, Algo):
    fn = method(obj, ("gemm_reducescatter_overlap",))
    fn(
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
        int(Algo),
        False,
    )


def call_baseline_cublas_allreduce(obj, A, B, C):
    method(obj, ("gemm_allreduce",))(A, B, C)


def call_baseline_cublas_reducescatter(obj, A, B, C, D):
    method(obj, ("gemm_reducescatter",))(A, B, C, D)


def call_baseline_plain_allreduce(obj, A, B, C, Algo):
    method(obj, ("gemm_plain_allreduce",))(A, B, C, int(Algo))


def call_baseline_plain_reducescatter(obj, A, B, C, D, Algo):
    method(obj, ("gemm_plain_reducescatter",))(A, B, C, D, int(Algo))


def perf_comm_process(rank, world_size, nccl_id, broker_key, comm_backend, M, N, comm_op, result_dict):
    torch.cuda.set_device(rank)
    ensure_ext()
    validate_backend(comm_backend, comm_op, world_size)
    clear_comm_cta_env_in_child()

    if comm_backend == "nccl":
        obj = ext.BaselineImpl()
        init_baseline_nccl(obj, rank, world_size, nccl_id)

        C = torch.empty((M, N), dtype=torch.float16, device="cuda").normal_(mean=0.0, std=0.5)
        D = torch.empty((C.numel() // world_size,), dtype=torch.float16, device="cuda")

        if comm_op == "all_reduce":
            result_dict[rank] = mean_timed(lambda: call_nccl_allreduce(obj, C))
        elif comm_op == "reduce_scatter":
            result_dict[rank] = mean_timed(lambda: call_nccl_reducescatter(obj, C, D))
        else:
            raise ValueError(f"Unknown comm_op={comm_op}")

    else:
        obj = ext.OverlapImpl()
        try:
            init_overlap_backend(obj, rank, world_size, nccl_id, broker_key, comm_backend, comm_op)
            obj.cutlass_init()

            C = torch.empty((M, N), dtype=torch.float16, device="cuda").normal_(mean=0.0, std=0.5)
            result_dict[rank] = mean_timed(lambda: call_ooverlap_allreduce(obj, C))

        finally:
            sync_and_release_overlap_backend(obj, comm_backend)


def perf_comm(M, N, comm_op, comm_backend):
    world_size = torch.cuda.device_count()
    validate_backend(comm_backend, comm_op, world_size)

    ensure_ext()
    nccl_id = ext.generate_nccl_id() if comm_backend == "nccl" else []
    broker_key = make_broker_key() if comm_backend == "ooverlap" else ""

    with mp.Manager() as manager:
        result_dict = manager.dict()

        with scoped_env(unset_keys=COMM_CTA_ENV_KEYS):
            mp.spawn(
                perf_comm_process,
                args=(world_size, nccl_id, broker_key, comm_backend, M, N, comm_op, result_dict),
                nprocs=world_size,
            )

        return max(float(result_dict[r]) for r in range(world_size))


def perf_overlap_process(
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
    reorder_map,
    comm_op,
    active_sm_count,
    set_overlap_comm_ctas,
    overlap_comm_ctas,
    result_dict,
):
    torch.cuda.set_device(rank)
    ensure_ext()
    validate_backend(comm_backend, comm_op, world_size)

    if set_overlap_comm_ctas:
        set_comm_cta_env_in_child(overlap_comm_ctas)
    else:
        clear_comm_cta_env_in_child()

    obj = ext.OverlapImpl()

    try:
        init_overlap_backend(obj, rank, world_size, nccl_id, broker_key, comm_backend, comm_op)
        obj.cutlass_init()
        obj.overlap_init()

        tile_num = div_up(M, BM) * div_up(N, BN)
        cSeg_CPU = torch.tensor(cSeg, dtype=torch.int32)
        cSeg_GPU = cSeg_CPU.cuda(rank)

        A = torch.empty((M, K), dtype=torch.float16, device="cuda").normal_(mean=0.0, std=0.5)
        B = torch.empty((N, K), dtype=torch.float16, device="cuda").normal_(mean=0.0, std=0.5)

        packed_M, packed_N = packed_shape(M, N, BM, BN, 1)
        C = torch.empty((packed_M, packed_N), dtype=torch.float16, device="cuda")

        MM = torch.zeros(
            (monitor_size(tile_num, len(cSeg), False),),
            dtype=torch.int,
            device="cuda",
        )

        RA = make_reordered_array(tile_num, hint, reorder_map).reshape(
            (div_up(M, BM), div_up(N, BN))
        )

        D = None
        RowArray = None
        if comm_op == "reduce_scatter":
            D = torch.empty((M // world_size, N), dtype=torch.float16, device="cuda")
            RowArray = generate_row_remap_array(M, N, BM, BN, cSeg, world_size)

        def pre_run():
            reset_monitor_matrix(MM, tile_num, len(cSeg), False)
            
        def run():

            if comm_op == "all_reduce":
                call_overlap_allreduce(
                    obj,
                    A,
                    B,
                    C,
                    MM,
                    RA,
                    cSeg_CPU,
                    cSeg_GPU,
                    Algo,
                    active_sm_count,
                )
            elif comm_op == "reduce_scatter":
                call_overlap_reducescatter(
                    obj,
                    A,
                    B,
                    C,
                    D,
                    MM,
                    RA,
                    RowArray,
                    cSeg_CPU,
                    cSeg_GPU,
                    Algo,
                )
            else:
                raise ValueError(f"Unknown comm_op={comm_op}")

        result_dict[rank] = mean_timed(run, pre_run)

    finally:
        sync_and_release_overlap_backend(obj, comm_backend)


def perf_overlap(
    M,
    N,
    K,
    BM,
    BN,
    Algo,
    cSeg,
    hint,
    comm_op,
    comm_backend,
    active_sm_count,
    reorder_map=None,
    set_overlap_comm_ctas=False,
    overlap_comm_ctas=0,
):
    world_size = torch.cuda.device_count()
    validate_backend(comm_backend, comm_op, world_size)

    if set_overlap_comm_ctas and int(overlap_comm_ctas) <= 0:
        raise ValueError(f"overlap_comm_ctas must be > 0, got {overlap_comm_ctas}")

    ensure_ext()
    nccl_id = ext.generate_nccl_id() if comm_backend == "nccl" else []
    broker_key = make_broker_key() if comm_backend == "ooverlap" else ""

    with mp.Manager() as manager:
        result_dict = manager.dict()

        if set_overlap_comm_ctas:
            set_values = comm_cta_env_values(overlap_comm_ctas)
            unset_keys = []
        else:
            set_values = {}
            unset_keys = COMM_CTA_ENV_KEYS

        with scoped_env(set_values=set_values, unset_keys=unset_keys):
            mp.spawn(
                perf_overlap_process,
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
                    set_overlap_comm_ctas,
                    int(overlap_comm_ctas),
                    result_dict,
                ),
                nprocs=world_size,
            )

        return max(float(result_dict[r]) for r in range(world_size))


def perf_baseline_process(
    rank,
    world_size,
    nccl_id,
    broker_key,
    comm_backend,
    baseline_kind,
    M,
    N,
    K,
    Algo,
    comm_op,
    result_dict,
):
    torch.cuda.set_device(rank)
    ensure_ext()
    validate_backend(comm_backend, comm_op, world_size)
    clear_comm_cta_env_in_child()

    A = torch.empty((M, K), dtype=torch.float16, device="cuda").normal_(mean=0.0, std=0.5)
    B = torch.empty((N, K), dtype=torch.float16, device="cuda").normal_(mean=0.0, std=0.5)

    if comm_backend == "nccl":
        obj = ext.BaselineImpl()
        init_baseline_nccl(obj, rank, world_size, nccl_id)

        if baseline_kind == "cublas":
            # BaselineImpl::GemmAllReduce/GemmReduceScatter uses its original C layout.
            C = torch.empty((M, N), dtype=torch.float16, device="cuda")
            D = torch.empty((M // world_size, N), dtype=torch.float16, device="cuda")

            if comm_op == "all_reduce":
                result_dict[rank] = mean_timed(lambda: call_baseline_cublas_allreduce(obj, A, B, C))
            elif comm_op == "reduce_scatter":
                result_dict[rank] = mean_timed(lambda: call_baseline_cublas_reducescatter(obj, A, B, C, D))
            else:
                raise ValueError(f"Unknown comm_op={comm_op}")

        elif baseline_kind == "plain":
            # Plain SM90 path uses the module/plain physical D_col convention [N, M].
            C = torch.empty((N, M), dtype=torch.float16, device="cuda")
            D = torch.empty((C.numel() // world_size,), dtype=torch.float16, device="cuda")

            if comm_op == "all_reduce":
                result_dict[rank] = mean_timed(lambda: call_baseline_plain_allreduce(obj, A, B, C, Algo))
            elif comm_op == "reduce_scatter":
                result_dict[rank] = mean_timed(lambda: call_baseline_plain_reducescatter(obj, A, B, C, D, Algo))
            else:
                raise ValueError(f"Unknown comm_op={comm_op}")

        else:
            raise ValueError(f"Unknown baseline_kind={baseline_kind}")

    else:
        obj = ext.OverlapImpl()
        try:
            init_overlap_backend(obj, rank, world_size, nccl_id, broker_key, comm_backend, comm_op)
            obj.cutlass_init()

            # Module-level baseline_gemm_col/gemm_plain_sm90 both use D_col [N, M].
            C = torch.empty((N, M), dtype=torch.float16, device="cuda")

            def run():
                if baseline_kind == "cublas":
                    call_module_cublas_gemm(A, B, C)
                elif baseline_kind == "plain":
                    call_module_plain_gemm(A, B, C, Algo)
                else:
                    raise ValueError(f"Unknown baseline_kind={baseline_kind}")

                call_ooverlap_allreduce(obj, C)

            result_dict[rank] = mean_timed(run)

        finally:
            sync_and_release_overlap_backend(obj, comm_backend)


def perf_baseline(M, N, K, Algo, comm_op, comm_backend, baseline_kind):
    world_size = torch.cuda.device_count()
    validate_backend(comm_backend, comm_op, world_size)

    ensure_ext()
    nccl_id = ext.generate_nccl_id() if comm_backend == "nccl" else []
    broker_key = make_broker_key() if comm_backend == "ooverlap" else ""

    with mp.Manager() as manager:
        result_dict = manager.dict()

        with scoped_env(unset_keys=COMM_CTA_ENV_KEYS):
            mp.spawn(
                perf_baseline_process,
                args=(
                    world_size,
                    nccl_id,
                    broker_key,
                    comm_backend,
                    baseline_kind,
                    M,
                    N,
                    K,
                    Algo,
                    comm_op,
                    result_dict,
                ),
                nprocs=world_size,
            )

        return max(float(result_dict[r]) for r in range(world_size))


def should_set_overlap_comm_ctas(args, comm_backend: str):
    if comm_backend == "nccl":
        return bool(args.set_nccl_comm_ctas_to_comm_sms)
    if comm_backend == "ooverlap":
        return bool(args.set_ooverlap_comm_ctas_to_comm_sms)
    return False


def run_backend(args, comm_backend: str):
    world_size = torch.cuda.device_count()
    validate_backend(comm_backend, args.comm_op, world_size)

    M, N, K = args.m, args.n, args.k
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    sm_count = props.multi_processor_count

    path, data = load_solution(M, N, K, comm_backend)

    BM = int(data["BM"])
    BN = int(data["BN"])
    Algo = int(data["Algo"])

    tile_num = div_up(M, BM) * div_up(N, BN)
    cSeg = [int(x) for x in data["cSeg"]]
    hint = [int(x) for x in data["hint"]]
    reorder_map = data.get("reorder_map")

    active_sm_count = int(data.get("compute_sms", sm_count))
    comm_sm_slack = int(data.get("comm_sm_slack", sm_count - active_sm_count))
    config_gemm_dur = float(data["dur"])

    cseg_sum = int(sum(cSeg))
    if cseg_sum != tile_num:
        raise RuntimeError(f"sum(cSeg)={cseg_sum} must equal tile_num={tile_num}")

    set_overlap_ctas = should_set_overlap_comm_ctas(args, comm_backend)

    print("")
    print("########################################")
    print(f"# Running backend={comm_backend}")
    print("########################################")
    print(f"solution:          {path}")
    print(f"shape:             M={M} N={N} K={K}")
    print(f"algo:              {Algo}")
    print(f"tile:              BM={BM} BN={BN}")
    print(f"tile_num:          {tile_num}")
    print(f"cSeg:              {cSeg}")
    print(f"hint_len:          {len(hint)}")
    print(f"reorder_map:       {'yes' if reorder_map is not None else 'no'}")
    print(f"sm_count:          {sm_count}")
    print(f"active_sm_count:   {active_sm_count}")
    print(f"comm_sm_slack:     {comm_sm_slack}")
    print(f"config_gemm_dur:   {config_gemm_dur:.4f} ms")

    if set_overlap_ctas:
        print(f"overlap CTA env:   OOVERLAP_MAX_CTAS=NCCL_MAX_CTAS={comm_sm_slack}")
    else:
        print("overlap CTA env:   cleared")

    print("[phase] perf_comm", flush=True)
    comm_dur = perf_comm(M, N, args.comm_op, comm_backend)

    print("[phase] perf_overlap", flush=True)
    overlap_dur = perf_overlap(
        M,
        N,
        K,
        BM,
        BN,
        Algo,
        cSeg,
        hint,
        args.comm_op,
        comm_backend,
        active_sm_count,
        reorder_map=reorder_map,
        set_overlap_comm_ctas=set_overlap_ctas,
        overlap_comm_ctas=comm_sm_slack,
    )

    baselines = {}

    if args.run_cublas_baseline:
        print("[phase] cublas baseline", flush=True)
        baselines["cublas"] = perf_baseline(
            M,
            N,
            K,
            Algo,
            args.comm_op,
            comm_backend,
            "cublas",
        )

    if args.run_plain_baseline:
        print("[phase] plain baseline", flush=True)
        baselines["plain"] = perf_baseline(
            M,
            N,
            K,
            Algo,
            args.comm_op,
            comm_backend,
            "plain",
        )

    speedups = {
        name: float(dur) / float(overlap_dur)
        for name, dur in baselines.items()
    }

    print("")
    print(f"{'Item':<30} {'Value':>18}")
    print(f"{'----':<30} {'-----':>18}")
    print(f"{'backend':<30} {comm_backend:>18}")
    print(f"{'comm_op':<30} {args.comm_op:>18}")
    print(f"{'comm_dur_ms':<30} {comm_dur:>18.4f}")
    print(f"{'overlap_dur_ms':<30} {overlap_dur:>18.4f}")

    for name in ("cublas", "plain"):
        if name in baselines:
            print(f"{name + '_baseline_ms':<30} {baselines[name]:>18.4f}")
            print(f"{'speedup_vs_' + name:<30} {speedups[name]:>18.4f}")

    return {
        "comm_backend": comm_backend,
        "comm_op": args.comm_op,
        "solution": str(path),
        "Algo": int(Algo),
        "BM": int(BM),
        "BN": int(BN),
        "cSeg": cSeg,
        "active_sm_count": int(active_sm_count),
        "comm_sm_slack": int(comm_sm_slack),
        "comm_dur_ms": float(comm_dur),
        "overlap_dur_ms": float(overlap_dur),
        "config_gemm_dur_ms": float(config_gemm_dur),
        "baselines": {k: float(v) for k, v in baselines.items()},
        "speedups": speedups,
        "overlap_cta_env": (
            f"OOVERLAP_MAX_CTAS=NCCL_MAX_CTAS={comm_sm_slack}"
            if set_overlap_ctas
            else "cleared"
        ),
    }


def print_summary(results):
    if not results:
        return

    print("")
    print("########################################")
    print("# Summary")
    print("########################################")
    print(
        f"{'backend':<10} "
        f"{'overlap(ms)':>12} "
        f"{'comm(ms)':>10} "
        f"{'cublas_base':>12} "
        f"{'x_cublas':>9} "
        f"{'plain_base':>12} "
        f"{'x_plain':>9} "
        f"{'CTA env':>26} "
        f"{'cSeg':>24}"
    )
    print(
        f"{'-------':<10} "
        f"{'-----------':>12} "
        f"{'--------':>10} "
        f"{'-----------':>12} "
        f"{'--------':>9} "
        f"{'----------':>12} "
        f"{'-------':>9} "
        f"{'-------':>26} "
        f"{'----':>24}"
    )

    for r in results:
        cublas = r["baselines"].get("cublas")
        plain = r["baselines"].get("plain")

        print(
            f"{r['comm_backend']:<10} "
            f"{r['overlap_dur_ms']:>12.4f} "
            f"{r['comm_dur_ms']:>10.4f} "
            f"{(cublas if cublas is not None else float('nan')):>12.4f} "
            f"{r['speedups'].get('cublas', float('nan')):>9.4f} "
            f"{(plain if plain is not None else float('nan')):>12.4f} "
            f"{r['speedups'].get('plain', float('nan')):>9.4f} "
            f"{r['overlap_cta_env']:>26} "
            f"{str(r['cSeg']):>24}"
        )

    by_backend = {r["comm_backend"]: r for r in results}

    if "nccl" not in by_backend or "ooverlap" not in by_backend:
        return

    nccl = by_backend["nccl"]
    oo = by_backend["ooverlap"]

    print("")
    print("########################################")
    print("# Cross-backend speedups")
    print("########################################")
    print(
        f"{'comparison':<52} "
        f"{'baseline(ms)':>14} "
        f"{'overlap(ms)':>14} "
        f"{'speedup':>10}"
    )
    print(
        f"{'----------':<52} "
        f"{'------------':>14} "
        f"{'-----------':>14} "
        f"{'-------':>10}"
    )

    for base_name in ("cublas", "plain"):
        if base_name in oo["baselines"]:
            speedup = oo["baselines"][base_name] / nccl["overlap_dur_ms"]
            print(
                f"{'nccl overlap vs ooverlap ' + base_name + ' baseline':<52} "
                f"{oo['baselines'][base_name]:>14.4f} "
                f"{nccl['overlap_dur_ms']:>14.4f} "
                f"{speedup:>10.4f}"
            )

        if base_name in nccl["baselines"]:
            speedup = nccl["baselines"][base_name] / oo["overlap_dur_ms"]
            print(
                f"{'ooverlap overlap vs nccl ' + base_name + ' baseline':<52} "
                f"{nccl['baselines'][base_name]:>14.4f} "
                f"{oo['overlap_dur_ms']:>14.4f} "
                f"{speedup:>10.4f}"
            )

    print("")
    print("########################################")
    print("# Overlap-to-overlap")
    print("########################################")
    print(
        f"{'comparison':<52} "
        f"{'reference(ms)':>14} "
        f"{'target(ms)':>14} "
        f"{'ratio':>10}"
    )
    print(
        f"{'ooverlap overlap vs nccl overlap':<52} "
        f"{nccl['overlap_dur_ms']:>14.4f} "
        f"{oo['overlap_dur_ms']:>14.4f} "
        f"{nccl['overlap_dur_ms'] / oo['overlap_dur_ms']:>10.4f}"
    )


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--m", type=int, default=4096)
    p.add_argument("--n", type=int, default=8192)
    p.add_argument("--k", type=int, default=8192)

    p.add_argument(
        "--comm_op",
        type=str,
        default="all_reduce",
        choices=["all_reduce", "reduce_scatter"],
    )

    p.add_argument(
        "--comm_backend",
        type=str,
        default="both",
        choices=["both", "nccl", "ooverlap"],
    )

    p.add_argument(
        "--no_cublas_baseline",
        dest="run_cublas_baseline",
        action="store_false",
        help="Disable cuBLAS GEMM + communication baseline.",
    )

    p.add_argument(
        "--no_plain_baseline",
        dest="run_plain_baseline",
        action="store_false",
        help="Disable gemm_plain_sm90 + communication baseline.",
    )

    p.add_argument(
        "--set_nccl_comm_ctas_to_comm_sms",
        action="store_true",
        help="For NCCL overlap only, set OOVERLAP_MAX_CTAS and NCCL_MAX_CTAS to comm_sm_slack.",
    )

    p.add_argument(
        "--set_ooverlap_comm_ctas_to_comm_sms",
        action="store_true",
        help="For ooverlap overlap only, set OOVERLAP_MAX_CTAS and NCCL_MAX_CTAS to comm_sm_slack.",
    )

    p.set_defaults(run_cublas_baseline=True, run_plain_baseline=True)
    return p.parse_args()


def main():
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    args = parse_args()

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

    print_summary(results)


if __name__ == "__main__":
    main()
