import argparse
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


def parse_devices(value: str) -> list[int]:
    devices = [int(item.strip()) for item in value.split(",") if item.strip()]
    if len(devices) < 2:
        raise ValueError("--devices must contain at least two device ids")
    if len(set(devices)) != len(devices):
        raise ValueError("--devices must not contain duplicates")
    if any(device < 0 for device in devices):
        raise ValueError("--devices must contain non-negative device ids")
    return devices


def make_nccl_id_bytes():
    ext = load_ooverlap_ext()
    return [int(value) for value in ext.generate_nccl_id()]


def print_speedup_line(label: str, baseline_label: str, baseline: float, value: float):
    speedup = baseline / value
    pct = (speedup - 1.0) * 100.0
    sign = "+" if pct >= 0.0 else ""
    print(f"  {label}_speedup_over_{baseline_label}: {speedup:.6f}x ({sign}{pct:.2f}%)")


def print_metrics(title, metrics):
    print(f"\n[{title}]")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.6f}")
        else:
            print(f"  {key}: {value}")

    nccl_ms = metrics.get("nccl_ms")
    nccl_symmetric_ms = metrics.get("nccl_symmetric_ms")
    ooverlap_ms = metrics.get("ooverlap_ms")

    if isinstance(nccl_ms, float) and nccl_ms > 0.0:
        print("\n[Speedup over normal NCCL]")
        if isinstance(ooverlap_ms, float) and ooverlap_ms > 0.0:
            print_speedup_line("ooverlap", "nccl", nccl_ms, ooverlap_ms)
        if isinstance(nccl_symmetric_ms, float) and nccl_symmetric_ms > 0.0:
            print_speedup_line("nccl_symmetric", "nccl", nccl_ms, nccl_symmetric_ms)

    if (
        isinstance(nccl_symmetric_ms, float)
        and nccl_symmetric_ms > 0.0
        and isinstance(ooverlap_ms, float)
        and ooverlap_ms > 0.0
    ):
        print("\n[Speedup over symmetric NCCL]")
        print_speedup_line("ooverlap", "nccl_symmetric", nccl_symmetric_ms, ooverlap_ms)


def collective_display_name(collective: str) -> str:
    return {
        "allreduce": "all-reduce",
        "reduce_scatter": "reduce-scatter",
        "all_gather": "all-gather",
    }.get(collective, collective)


def aggregate_rank_metrics(rank_results, world_size: int):
    if len(rank_results) != world_size:
        raise RuntimeError(
            f"expected {world_size} rank results, got {len(rank_results)}"
        )

    by_rank = {int(row["rank"]): row for row in rank_results}
    expected_ranks = set(range(world_size))
    if set(by_rank) != expected_ranks:
        raise RuntimeError(
            f"expected rank results for {sorted(expected_ranks)}, got {sorted(by_rank)}"
        )

    rows = [by_rank[rank] for rank in range(world_size)]
    first = rows[0]
    out = {
        "collective": first["collective"],
        "world_size": float(world_size),
        "numel": first["numel"],
        "bytes": first["bytes"],
        "local_shard_numel": max(row["local_shard_numel"] for row in rows),
        "local_shard_bytes": max(row["local_shard_bytes"] for row in rows),
        "ooverlap_ms": max(row["ooverlap_ms"] for row in rows),
        "nccl_ms": max(row["nccl_ms"] for row in rows),
        "nccl_symmetric_ms": max(row["nccl_symmetric_ms"] for row in rows),
        "iters": first["iters"],
        "warmup": first["warmup"],
    }

    for rank, row in enumerate(rows):
        out[f"rank{rank}_ooverlap_ms"] = row["ooverlap_ms"]
        out[f"rank{rank}_nccl_ms"] = row["nccl_ms"]
        out[f"rank{rank}_nccl_symmetric_ms"] = row["nccl_symmetric_ms"]

    return out


def run_child():
    ext = load_ooverlap_ext()

    collective = os.environ["OOVERLAP_COLLECTIVE"]
    numel = int(os.environ["OOVERLAP_NUMEL"])
    iters = int(os.environ["OOVERLAP_ITERS"])
    warmup = int(os.environ["OOVERLAP_WARMUP"])
    devices = [int(value) for value in json.loads(os.environ["OOVERLAP_DEVICES"])]
    rank = int(os.environ["OOVERLAP_LOCAL_RANK"])
    broker_key = os.environ["OOVERLAP_BROKER_KEY"]
    verify = os.environ.get("OOVERLAP_VERIFY", "1") == "1"
    nccl_id_bytes = json.loads(os.environ["OOVERLAP_NCCL_ID_BYTES"])
    mode = os.environ.get("OOVERLAP_MODE", "bench")

    if mode == "smoke":
        if hasattr(ext, "ipc_external_p2p_collective_smoke_rank_sm90"):
            ok = ext.ipc_external_p2p_collective_smoke_rank_sm90(
                collective,
                int(numel),
                int(rank),
                devices,
                broker_key,
                nccl_id_bytes,
                bool(verify),
            )
        elif len(devices) == 2 and hasattr(
            ext, "ipc_external_p2p_two_gpu_collective_smoke_rank_sm90"
        ):
            ok = ext.ipc_external_p2p_two_gpu_collective_smoke_rank_sm90(
                collective,
                int(numel),
                int(rank),
                int(devices[0]),
                int(devices[1]),
                broker_key,
                nccl_id_bytes,
                bool(verify),
            )
        else:
            raise AttributeError(
                "Extension does not expose ipc_external_p2p_collective_smoke_rank_sm90. "
                "Rebuild ooverlap_ext."
            )

        metrics = {
            "rank": float(rank),
            "world_size": float(len(devices)),
            "ok": 1.0 if ok else 0.0,
            "collective": float(VALID_COLLECTIVES.index(collective)),
            "numel": float(numel),
        }
    else:
        if hasattr(ext, "benchmark_ipc_external_p2p_collective_rank_sm90"):
            metrics = ext.benchmark_ipc_external_p2p_collective_rank_sm90(
                collective,
                int(numel),
                int(rank),
                devices,
                broker_key,
                nccl_id_bytes,
                int(iters),
                int(warmup),
                bool(verify),
            )
        elif len(devices) == 2 and hasattr(
            ext, "benchmark_ipc_external_p2p_two_gpu_collective_rank_sm90"
        ):
            metrics = ext.benchmark_ipc_external_p2p_two_gpu_collective_rank_sm90(
                collective,
                int(numel),
                int(rank),
                int(devices[0]),
                int(devices[1]),
                broker_key,
                nccl_id_bytes,
                int(iters),
                int(warmup),
                bool(verify),
            )
        else:
            raise AttributeError(
                "Extension does not expose benchmark_ipc_external_p2p_collective_rank_sm90. "
                "Rebuild ooverlap_ext."
            )

    print(RESULT_PREFIX + json.dumps(metrics, sort_keys=True), flush=True)


def spawn_ranks(args, mode: str, devices: list[int]):
    env_base = os.environ.copy()
    env_base["OOVERLAP_MODE"] = mode
    env_base["OOVERLAP_COLLECTIVE"] = args.collective
    env_base["OOVERLAP_NUMEL"] = str(args.numel)
    env_base["OOVERLAP_ITERS"] = str(args.iters)
    env_base["OOVERLAP_WARMUP"] = str(args.warmup)
    env_base["OOVERLAP_DEVICES"] = json.dumps(devices)
    env_base["OOVERLAP_VERIFY"] = "1" if args.verify else "0"
    env_base["OOVERLAP_BROKER_KEY"] = (
        args.broker_key or f"ooipc_{os.getpid():x}_{uuid.uuid4().hex[:8]}"
    )
    env_base["OOVERLAP_NCCL_ID_BYTES"] = json.dumps(make_nccl_id_bytes())

    procs = []
    for rank in range(len(devices)):
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
        description="Multiprocess IPC external-buffer multi-GPU collective benchmark"
    )
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--mode", choices=("smoke", "bench", "both"), default="both")
    parser.add_argument("--collective", choices=VALID_COLLECTIVES, default="allreduce")
    parser.add_argument("--numel", type=int, default=1 << 20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument(
        "--devices",
        default=None,
        help="comma-separated device ids, for example 0,1,2,3",
    )
    parser.add_argument("--dev0", type=int, default=0)
    parser.add_argument("--dev1", type=int, default=1)
    parser.add_argument("--broker-key", default=None)
    parser.add_argument("--no-verify", dest="verify", action="store_false")
    parser.set_defaults(verify=True)
    return parser.parse_args()


def main():
    args = parse_args()

    if args.child:
        run_child()
        return

    devices = (
        parse_devices(args.devices)
        if args.devices is not None
        else parse_devices(f"{args.dev0},{args.dev1}")
    )

    if args.numel <= 0:
        raise ValueError("--numel must be > 0")
    if args.iters <= 0:
        raise ValueError("--iters must be > 0")
    if args.warmup < 0:
        raise ValueError("--warmup must be >= 0")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    if max(devices) >= torch.cuda.device_count():
        raise ValueError(
            f"requested devices {devices}, but CUDA device count is {torch.cuda.device_count()}"
        )

    if args.collective in ("reduce_scatter", "all_gather"):
        if args.numel % len(devices) != 0:
            raise ValueError(
                f"--numel must be divisible by world size {len(devices)} "
                f"for {args.collective}"
            )

    display_collective = collective_display_name(args.collective)

    print(f"[info] torch {torch.__version__}")
    print(f"[info] cuda device count: {torch.cuda.device_count()}")
    print(f"[info] collective={args.collective} ({display_collective})")
    print("[info] memory mode=multiprocess cudaMalloc buffers + oo_buffer_wrap + oo_group_create_ipc")
    print("[info] ooverlap mode=legacy CUDA IPC export/import inside prepare_collective_launch")
    print("[info] NCCL modes=multiprocess ncclCommInitRank normal cudaMalloc and symmetric ncclMemAlloc")
    print(f"[info] devices={devices} world_size={len(devices)}")
    print(f"[info] numel={args.numel} iters={args.iters} warmup={args.warmup}")
    print(f"[info] verify={args.verify}")

    if args.mode in ("smoke", "both"):
        smoke_rows = spawn_ranks(args, "smoke", devices)
        ok = (
            len(smoke_rows) == len(devices)
            and all(float(row.get("ok", 0.0)) == 1.0 for row in smoke_rows)
        )
        print(f"[result] IPC external P2P {display_collective} smoke: {ok}")
        assert ok is True

    if args.mode in ("bench", "both"):
        rows = spawn_ranks(args, "bench", devices)
        metrics = aggregate_rank_metrics(rows, len(devices))
        print_metrics(
            f"IPC external-buffer P2P {len(devices)}-GPU {display_collective}: "
            "ooverlap vs NCCL",
            metrics,
        )

    for device in devices:
        torch.cuda.synchronize(device)

    print(f"PASS IPC external-buffer P2P {len(devices)}-GPU {display_collective} path")


if __name__ == "__main__":
    main()
