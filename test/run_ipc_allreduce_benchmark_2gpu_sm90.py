#!/usr/bin/env python3

import argparse
import faulthandler
import multiprocessing as mp
import os
import queue
import sys
import time
import traceback
import uuid


def _worker(
    rank,
    module_dir,
    numel,
    iters,
    warmup,
    dev0,
    dev1,
    broker_key,
    nccl_unique_id,
    verify,
    verbose,
    result_q,
):
    faulthandler.enable(all_threads=True)

    try:
        if module_dir:
            sys.path.insert(0, module_dir)

        import ooverlap_ext

        if verbose:
            print(
                f"[ipc-bench][rank={rank}][pid={os.getpid()}] start "
                f"numel={numel} iters={iters} warmup={warmup} "
                f"devs=({dev0},{dev1}) key={broker_key}",
                flush=True,
            )

        result = ooverlap_ext.benchmark_ipc_two_gpu_allreduce_rank_sm90(
            numel,
            rank,
            dev0,
            dev1,
            broker_key,
            nccl_unique_id,
            iters,
            warmup,
            verify,
        )

        if verbose:
            print(
                f"[ipc-bench][rank={rank}][pid={os.getpid()}] result={result}",
                flush=True,
            )

        result_q.put((rank, True, result, ""))

    except BaseException:
        tb = traceback.format_exc()
        print(
            f"[ipc-bench][rank={rank}][pid={os.getpid()}] exception:\n{tb}",
            flush=True,
        )
        result_q.put((rank, False, {}, tb))


def main():
    faulthandler.enable(all_threads=True)

    parser = argparse.ArgumentParser(
        description="Benchmark ooverlap IPC allreduce vs NCCL multiprocess allreduce."
    )
    parser.add_argument("--numel", type=int, default=1 << 20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--dev0", type=int, default=0)
    parser.add_argument("--dev1", type=int, default=1)
    parser.add_argument(
        "--module-dir",
        type=str,
        default=os.environ.get("OOVERLAP_EXT_DIR", ""),
        help="Directory containing ooverlap_ext.so. Optional if PYTHONPATH already points to it.",
    )
    parser.add_argument(
        "--timeout-s",
        type=float,
        default=300.0,
        help="Timeout for the two child processes.",
    )
    parser.add_argument(
        "--broker-key",
        type=str,
        default="",
        help="Unique broker key. Default: generated per run.",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify final ooverlap IPC and NCCL results.",
    )
    parser.add_argument(
        "--cuda-launch-blocking",
        action="store_true",
        help="Set CUDA_LAUNCH_BLOCKING=1 in child processes.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print runner progress logs.",
    )

    args = parser.parse_args()

    if args.numel <= 0:
        raise ValueError("--numel must be > 0")
    if args.iters <= 0:
        raise ValueError("--iters must be > 0")
    if args.warmup < 0:
        raise ValueError("--warmup must be >= 0")
    if args.dev0 == args.dev1:
        raise ValueError("--dev0 and --dev1 must differ")

    if args.cuda_launch_blocking:
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    if args.module_dir:
        sys.path.insert(0, args.module_dir)

    import ooverlap_ext

    nccl_unique_id = ooverlap_ext.generate_nccl_id()

    broker_key = args.broker_key
    if not broker_key:
        broker_key = f"ipc_bench_ar_{os.getpid()}_{uuid.uuid4().hex[:12]}"

    if args.verbose:
        print(
            "[ipc-bench] start "
            f"numel={args.numel} iters={args.iters} warmup={args.warmup} "
            f"devs=({args.dev0},{args.dev1}) key={broker_key} "
            f"module_dir={args.module_dir or '<PYTHONPATH>'} "
            f"verify={args.verify} "
            f"CUDA_LAUNCH_BLOCKING={os.environ.get('CUDA_LAUNCH_BLOCKING', '<unset>')}",
            flush=True,
        )

    ctx = mp.get_context("spawn")
    result_q = ctx.Queue()

    procs = []
    for rank in (0, 1):
        p = ctx.Process(
            target=_worker,
            args=(
                rank,
                args.module_dir,
                args.numel,
                args.iters,
                args.warmup,
                args.dev0,
                args.dev1,
                broker_key,
                nccl_unique_id,
                args.verify,
                args.verbose,
                result_q,
            ),
            daemon=False,
        )
        p.start()
        procs.append(p)

        if args.verbose:
            print(
                f"[ipc-bench] spawned rank={rank} pid={p.pid}",
                flush=True,
            )

    deadline = time.time() + args.timeout_s
    results = {}
    reported_exits = set()

    while len(results) < 2 and time.time() < deadline:
        try:
            rank, ok, result, msg = result_q.get(timeout=0.25)
            results[rank] = (ok, result, msg)

            if args.verbose:
                print(
                    f"[ipc-bench] got result rank={rank} ok={ok}",
                    flush=True,
                )
        except queue.Empty:
            pass

        for rank, p in enumerate(procs):
            if (
                p.exitcode is not None
                and rank not in results
                and rank not in reported_exits
            ):
                print(
                    f"[ipc-bench] rank={rank} pid={p.pid} exitcode={p.exitcode} "
                    "without queue result",
                    flush=True,
                )
                reported_exits.add(rank)

        if len(reported_exits) == 2 and len(results) == 0:
            break

    for rank, p in enumerate(procs):
        remaining = max(0.0, deadline - time.time())
        p.join(timeout=remaining)

    timed_out = [rank for rank, p in enumerate(procs) if p.is_alive()]
    if timed_out:
        for rank, p in enumerate(procs):
            if p.is_alive():
                print(
                    f"[ipc-bench] terminating timed-out rank={rank} pid={p.pid}",
                    flush=True,
                )
                p.terminate()
        for p in procs:
            p.join(timeout=5)
        raise RuntimeError(f"Timed out waiting for ranks: {timed_out}")

    errors = []
    rank_results = {}

    for rank in (0, 1):
        if rank not in results:
            errors.append(
                f"rank {rank}: no result returned, exitcode={procs[rank].exitcode}"
            )
            continue

        ok, result, msg = results[rank]
        if not ok:
            errors.append(f"rank {rank} failed:\n{msg}")
        else:
            rank_results[rank] = result

    for rank, p in enumerate(procs):
        if p.exitcode != 0:
            errors.append(f"rank {rank}: process exitcode={p.exitcode}")

    if errors:
        raise RuntimeError("\n\n".join(errors))

    r0 = rank_results[0]
    r1 = rank_results[1]

    oo_ms = max(float(r0["avg_ms_oo_ipc"]), float(r1["avg_ms_oo_ipc"]))
    nccl_ms = max(float(r0["avg_ms_nccl_ipc"]), float(r1["avg_ms_nccl_ipc"]))
    speedup = nccl_ms / oo_ms if oo_ms > 0.0 else float("inf")

    bytes_per_rank = args.numel * 2
    gib = bytes_per_rank / (1024.0 ** 3)

    oo_gib_s = gib / (oo_ms / 1000.0) if oo_ms > 0.0 else 0.0
    nccl_gib_s = gib / (nccl_ms / 1000.0) if nccl_ms > 0.0 else 0.0

    print("IPC 2-GPU allreduce benchmark")
    print(f"  numel:             {args.numel}")
    print(f"  bytes/rank:        {bytes_per_rank}")
    print(f"  iters:             {args.iters}")
    print(f"  warmup:            {args.warmup}")
    print(f"  devices:           ({args.dev0}, {args.dev1})")
    print(f"  verify:            {args.verify}")
    print("")
    print(f"  ooverlap IPC ms:   {oo_ms:.6f}")
    print(f"  NCCL IPC ms:       {nccl_ms:.6f}")
    print(f"  NCCL/OO speedup:   {speedup:.6f}")
    print("")
    print(f"  ooverlap GiB/s:    {oo_gib_s:.3f}")
    print(f"  NCCL GiB/s:        {nccl_gib_s:.3f}")
    print("")
    print(f"  rank0 raw:         {r0}")
    print(f"  rank1 raw:         {r1}")


if __name__ == "__main__":
    main()
