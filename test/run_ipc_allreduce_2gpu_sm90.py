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


def _worker(rank, module_dir, numel, iters, dev0, dev1, broker_key, result_q):
    faulthandler.enable(all_threads=True)

    try:
        print(
            f"[ipc-runner][rank={rank}][pid={os.getpid()}] worker start "
            f"numel={numel} iters={iters} devs=({dev0},{dev1}) key={broker_key}",
            flush=True,
        )

        if module_dir:
            sys.path.insert(0, module_dir)

        print(
            f"[ipc-runner][rank={rank}][pid={os.getpid()}] importing ooverlap_ext",
            flush=True,
        )
        import ooverlap_ext

        print(
            f"[ipc-runner][rank={rank}][pid={os.getpid()}] calling extension",
            flush=True,
        )

        ok = ooverlap_ext.tma_ipc_two_gpu_allreduce_rank_smoke_test(
            numel,
            rank,
            dev0,
            dev1,
            broker_key,
            iters,
        )

        print(
            f"[ipc-runner][rank={rank}][pid={os.getpid()}] extension returned ok={ok}",
            flush=True,
        )

        result_q.put((rank, bool(ok), ""))

    except BaseException:
        tb = traceback.format_exc()
        print(
            f"[ipc-runner][rank={rank}][pid={os.getpid()}] exception:\n{tb}",
            flush=True,
        )
        result_q.put((rank, False, tb))


def main():
    faulthandler.enable(all_threads=True)

    parser = argparse.ArgumentParser(
        description="Run ooverlap 2-process CUDA IPC allreduce smoke test."
    )
    parser.add_argument("--numel", type=int, default=1 << 20)
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument("--dev0", type=int, default=0)
    parser.add_argument("--dev1", type=int, default=1)
    parser.add_argument(
        "--module-dir",
        type=str,
        default=os.environ.get("OOVERLAP_EXT_DIR", ""),
        help="Directory containing ooverlap_ext.so. "
             "Optional if PYTHONPATH already points to it.",
    )
    parser.add_argument(
        "--timeout-s",
        type=float,
        default=300.0,
        help="Timeout for each child process.",
    )
    parser.add_argument(
        "--broker-key",
        type=str,
        default="",
        help="Unique broker key. Default: generated per run.",
    )
    parser.add_argument(
        "--cuda-launch-blocking",
        action="store_true",
        help="Set CUDA_LAUNCH_BLOCKING=1 in child processes.",
    )

    args = parser.parse_args()

    if args.dev0 == args.dev1:
        raise ValueError("--dev0 and --dev1 must differ")
    if args.numel <= 0:
        raise ValueError("--numel must be > 0")
    if args.iters <= 0:
        raise ValueError("--iters must be > 0")

    if args.cuda_launch_blocking:
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    broker_key = args.broker_key
    if not broker_key:
        broker_key = f"ipc_ar_{os.getpid()}_{uuid.uuid4().hex[:12]}"

    print(
        "[ipc-runner] start "
        f"numel={args.numel} iters={args.iters} "
        f"devs=({args.dev0},{args.dev1}) key={broker_key} "
        f"module_dir={args.module_dir or '<PYTHONPATH>'} "
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
                args.dev0,
                args.dev1,
                broker_key,
                result_q,
            ),
            daemon=False,
        )
        p.start()
        print(
            f"[ipc-runner] spawned rank={rank} pid={p.pid}",
            flush=True,
        )
        procs.append(p)

    deadline = time.time() + args.timeout_s
    results = {}

    while len(results) < 2 and time.time() < deadline:
        try:
            rank, ok, msg = result_q.get(timeout=0.25)
            print(
                f"[ipc-runner] got queue result rank={rank} ok={ok}",
                flush=True,
            )
            results[rank] = (ok, msg)
        except queue.Empty:
            pass

        for rank, p in enumerate(procs):
            if p.exitcode is not None and rank not in results:
                print(
                    f"[ipc-runner] rank={rank} pid={p.pid} exitcode={p.exitcode} "
                    "without queue result yet",
                    flush=True,
                )

    for rank, p in enumerate(procs):
        remaining = max(0.0, deadline - time.time())
        print(
            f"[ipc-runner] joining rank={rank} pid={p.pid} remaining={remaining:.1f}s",
            flush=True,
        )
        p.join(timeout=remaining)

    timed_out = [idx for idx, p in enumerate(procs) if p.is_alive()]
    if timed_out:
        for idx, p in enumerate(procs):
            if p.is_alive():
                print(
                    f"[ipc-runner] terminating timed-out rank={idx} pid={p.pid}",
                    flush=True,
                )
                p.terminate()
        for p in procs:
            p.join(timeout=5)
        raise RuntimeError(f"Timed out waiting for ranks: {timed_out}")

    errors = []

    for rank in (0, 1):
        if rank not in results:
            errors.append(
                f"rank {rank}: no result returned, exitcode={procs[rank].exitcode}"
            )
            continue

        ok, msg = results[rank]
        if not ok:
            errors.append(f"rank {rank} failed:\n{msg}")

    for rank, p in enumerate(procs):
        if p.exitcode != 0:
            errors.append(f"rank {rank}: process exitcode={p.exitcode}")

    if errors:
        raise RuntimeError("\n\n".join(errors))

    print(
        "PASS: IPC allreduce smoke test "
        f"numel={args.numel} iters={args.iters} "
        f"devices=({args.dev0},{args.dev1}) broker_key={broker_key}",
        flush=True,
    )


if __name__ == "__main__":
    main()
