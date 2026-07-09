import argparse
import ctypes
import ctypes.util
import importlib.util
import json
import os
import subprocess
import sys
import uuid
from pathlib import Path

import torch

VALID_COLLECTIVES = ("allreduce", "reduce_scatter", "all_gather")
RESULT_PREFIX = "OOVERLAP_IPC_EXTERNAL_P2P_RESULT_JSON="


def load_ooverlap_ext():
    root = Path(__file__).resolve().parents[1]
    so = root / "build" / "lib" / "ooverlap_ext.so"
    if not so.exists():
        raise FileNotFoundError(f"Could not find {so}. Build first.")

    spec = importlib.util.spec_from_file_location("ooverlap_ext", str(so))
    mod = importlib.util.module_from_spec(spec)

    if spec.loader is None:
        raise RuntimeError(f"Could not load extension spec for {so}")

    spec.loader.exec_module(mod)
    return mod


class NcclUniqueId(ctypes.Structure):
    _fields_ = [("internal", ctypes.c_char * 128)]


def make_nccl_id_bytes():
    libname = ctypes.util.find_library("nccl") or "libnccl.so"
    nccl = ctypes.CDLL(libname)
    uid = NcclUniqueId()
    rc = nccl.ncclGetUniqueId(ctypes.byref(uid))
    if rc != 0:
        raise RuntimeError(f"ncclGetUniqueId failed: {rc}")
    return list(bytearray(uid.internal))


def print_speedup_line(label: str, baseline_label: str, baseline: float, value: float):
    speedup = baseline / value
    pct = (speedup - 1.0) * 100.0

    if pct >= 0.0:
        print(f"  {label}_speedup_over_{baseline_label}: {speedup:.6f}x (+{pct:.2f}%)")
    else:
        print(f"  {label}_speedup_over_{baseline_label}: {speedup:.6f}x ({pct:.2f}%)")


def print_metrics(title, metrics):
    print(f"\n[{title}]")

    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.6f}")
        else:
            print(f"  {k}: {v}")

    nccl_ms = metrics.get("nccl_ms", None)
    nccl_symmetric_ms = metrics.get("nccl_symmetric_ms", None)
    ooverlap_ms = metrics.get("ooverlap_ms", None)

    if isinstance(nccl_ms, float) and nccl_ms > 0.0:
        print("\n[Speedup over normal NCCL]")

        if isinstance(ooverlap_ms, float) and ooverlap_ms > 0.0:
            print_speedup_line("ooverlap", "nccl", nccl_ms, ooverlap_ms)

        if isinstance(nccl_symmetric_ms, float) and nccl_symmetric_ms > 0.0:
            print_speedup_line("nccl_symmetric", "nccl", nccl_ms, nccl_symmetric_ms)

    if (isinstance(nccl_symmetric_ms, float) and nccl_symmetric_ms > 0.0 and
            isinstance(ooverlap_ms, float) and ooverlap_ms > 0.0):
        print("\n[Speedup over symmetric NCCL]")
        print_speedup_line("ooverlap", "nccl_symmetric", nccl_symmetric_ms, ooverlap_ms)


def collective_display_name(collective: str) -> str:
    if collective == "allreduce":
        return "all-reduce"
    if collective == "reduce_scatter":
        return "reduce-scatter"
    if collective == "all_gather":
        return "all-gather"
    return collective


def aggregate_rank_metrics(rank_results):
    if len(rank_results) != 2:
        raise RuntimeError(f"expected 2 rank results, got {len(rank_results)}")

    by_rank = {int(row["rank"]): row for row in rank_results}
    if set(by_rank) != {0, 1}:
        raise RuntimeError(f"expected rank results for ranks 0 and 1, got {sorted(by_rank)}")

    r0 = by_rank[0]
    r1 = by_rank[1]

    out = {
        "collective": r0["collective"],
        "world_size": 2.0,
        "numel": r0["numel"],
        "bytes": r0["bytes"],
        "local_shard_numel": max(r0["local_shard_numel"], r1["local_shard_numel"]),
        "local_shard_bytes": max(r0["local_shard_bytes"], r1["local_shard_bytes"]),
        "ooverlap_ms": max(r0["ooverlap_ms"], r1["ooverlap_ms"]),
        "nccl_ms": max(r0["nccl_ms"], r1["nccl_ms"]),
        "nccl_symmetric_ms": max(r0["nccl_symmetric_ms"], r1["nccl_symmetric_ms"]),
        "iters": r0["iters"],
        "warmup": r0["warmup"],
        "rank0_ooverlap_ms": r0["ooverlap_ms"],
        "rank1_ooverlap_ms": r1["ooverlap_ms"],
        "rank0_nccl_ms": r0["nccl_ms"],
        "rank1_nccl_ms": r1["nccl_ms"],
        "rank0_nccl_symmetric_ms": r0["nccl_symmetric_ms"],
        "rank1_nccl_symmetric_ms": r1["nccl_symmetric_ms"],
    }

    return out


def run_child():
    ext = load_ooverlap_ext()

    collective = os.environ["OOVERLAP_COLLECTIVE"]
    numel = int(os.environ["OOVERLAP_NUMEL"])
    iters = int(os.environ["OOVERLAP_ITERS"])
    warmup = int(os.environ["OOVERLAP_WARMUP"])
    dev0 = int(os.environ["OOVERLAP_DEV0"])
    dev1 = int(os.environ["OOVERLAP_DEV1"])
    rank = int(os.environ["OOVERLAP_LOCAL_RANK"])
    broker_key = os.environ["OOVERLAP_BROKER_KEY"]
    verify = os.environ.get("OOVERLAP_VERIFY", "1") == "1"
    nccl_id_bytes = json.loads(os.environ["OOVERLAP_NCCL_ID_BYTES"])
    mode = os.environ.get("OOVERLAP_MODE", "bench")

    if mode == "smoke":
        ok = ext.ipc_external_p2p_two_gpu_collective_smoke_rank_sm90(
            collective,
            int(numel),
            int(rank),
            int(dev0),
            int(dev1),
            broker_key,
            nccl_id_bytes,
            bool(verify),
        )
        metrics = {
            "rank": float(rank),
            "ok": 1.0 if ok else 0.0,
            "collective": float(VALID_COLLECTIVES.index(collective)),
            "numel": float(numel),
        }
    else:
        metrics = ext.benchmark_ipc_external_p2p_two_gpu_collective_rank_sm90(
            collective,
            int(numel),
            int(rank),
            int(dev0),
            int(dev1),
            broker_key,
            nccl_id_bytes,
            int(iters),
            int(warmup),
            bool(verify),
        )

    print(RESULT_PREFIX + json.dumps(metrics, sort_keys=True), flush=True)


def spawn_two_ranks(args, mode):
    env_base = os.environ.copy()
    env_base["OOVERLAP_MODE"] = mode
    env_base["OOVERLAP_COLLECTIVE"] = args.collective
    env_base["OOVERLAP_NUMEL"] = str(args.numel)
    env_base["OOVERLAP_ITERS"] = str(args.iters)
    env_base["OOVERLAP_WARMUP"] = str(args.warmup)
    env_base["OOVERLAP_DEV0"] = str(args.dev0)
    env_base["OOVERLAP_DEV1"] = str(args.dev1)
    env_base["OOVERLAP_VERIFY"] = "1" if args.verify else "0"
    env_base["OOVERLAP_BROKER_KEY"] = (
        args.broker_key or f"ooverlap_ipc_external_p2p_{os.getpid()}_{uuid.uuid4().hex}"
    )
    env_base["OOVERLAP_NCCL_ID_BYTES"] = json.dumps(make_nccl_id_bytes())

    procs = []
    for rank in (0, 1):
        env = env_base.copy()
        env["OOVERLAP_LOCAL_RANK"] = str(rank)
        procs.append(
            subprocess.Popen(
                [sys.executable, __file__, "--child"],
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
        )

    rank_results = []
    rc = 0

    for rank, proc in enumerate(procs):
        assert proc.stdout is not None
        for line in proc.stdout:
            line = line.rstrip()
            print(f"[rank{rank}] {line}")
            if line.startswith(RESULT_PREFIX):
                rank_results.append(json.loads(line[len(RESULT_PREFIX):]))

    for proc in procs:
        rc |= proc.wait()

    if rc != 0:
        raise RuntimeError(f"one or more rank processes failed: rc={rc}")

    return rank_results


def parse_args():
    parser = argparse.ArgumentParser(
        description="Multiprocess IPC external-buffer P2P collective benchmark"
    )

    parser.add_argument(
        "--child",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--mode",
        choices=("smoke", "bench", "both"),
        default="both",
    )
    parser.add_argument(
        "--collective",
        choices=VALID_COLLECTIVES,
        default="allreduce",
    )
    parser.add_argument(
        "--numel",
        type=int,
        default=1 << 20,
    )
    parser.add_argument(
        "--iters",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=20,
    )
    parser.add_argument(
        "--dev0",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--dev1",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--broker-key",
        default=None,
    )
    parser.add_argument(
        "--no-verify",
        dest="verify",
        action="store_false",
    )
    parser.set_defaults(verify=True)

    return parser.parse_args()


def main():
    args = parse_args()

    if args.child:
        run_child()
        return

    if args.numel <= 0:
        raise ValueError("--numel must be > 0")
    if args.iters <= 0:
        raise ValueError("--iters must be > 0")
    if args.warmup < 0:
        raise ValueError("--warmup must be >= 0")
    if args.dev0 == args.dev1:
        raise ValueError("--dev0 and --dev1 must differ")

    display_collective = collective_display_name(args.collective)

    print(f"[info] torch {torch.__version__}")
    print(f"[info] cuda available: {torch.cuda.is_available()}")
    print(f"[info] cuda device count: {torch.cuda.device_count()}")
    print(f"[info] collective={args.collective} ({display_collective})")
    print("[info] memory mode=multiprocess cudaMalloc buffers + oo_buffer_wrap + oo_group_create_ipc")
    print("[info] ooverlap mode=legacy CUDA IPC export/import inside prepare_collective_launch")
    print("[info] NCCL modes=multiprocess ncclCommInitRank normal cudaMalloc and symmetric ncclMemAlloc")
    print(f"[info] dev0={args.dev0} dev1={args.dev1}")
    print(f"[info] numel={args.numel} iters={args.iters} warmup={args.warmup}")
    print(f"[info] verify={args.verify}")

    if args.mode in ("smoke", "both"):
        smoke_rows = spawn_two_ranks(args, "smoke")
        ok = all(float(row.get("ok", 0.0)) == 1.0 for row in smoke_rows)
        print(f"[result] IPC external P2P {display_collective} smoke: {ok}")
        assert ok is True

    if args.mode in ("bench", "both"):
        rows = spawn_two_ranks(args, "bench")
        metrics = aggregate_rank_metrics(rows)
        print_metrics(
            f"IPC external-buffer P2P 2-GPU {display_collective}: ooverlap vs NCCL",
            metrics,
        )

    torch.cuda.synchronize(args.dev0)
    torch.cuda.synchronize(args.dev1)

    print(f"PASS ✅ IPC external-buffer P2P 2-GPU {display_collective} path")


if __name__ == "__main__":
    main()
