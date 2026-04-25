#!/usr/bin/env python3

import argparse
import multiprocessing as mp
import os
import queue
import sys
import time
import traceback
import uuid


def _worker(rank, module_dir, numel, iters, dev0, dev1, broker_key, result_q):
    try:
        if module_dir:
            sys.path.insert(0, module_dir)

        import ooverlap_ext

        ok = ooverlap_ext.tma_ipc_two_gpu_allreduce_rank_smoke_test(
            numel,
            rank,
            dev0,
            dev1,
            broker_key,
            iters,
        )

        result_q.put((rank, bool(ok), ""))

    except Exception:
        result_q.put((rank, False, traceback.format_exc()))


def main():
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
        default=120.0,
        help="Timeout for each child process.",
    )
    parser.add_argument(
        "--broker-key",
        type=str,
        default="",
        help="Unique broker key. Default: generated per run.",
    )

    args = parser.parse_args()

    if args.dev0 == args.dev1:
        raise ValueError("--dev0 and --dev1 must differ")
    if args.numel <= 0:
        raise ValueError("--numel must be > 0")
    if args.iters <= 0:
        raise ValueError("--iters must be > 0")

    broker_key = args.broker_key
    if not broker_key:
        broker_key = f"ipc_ar_{os.getpid()}_{uuid.uuid4().hex[:12]}"

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
        procs.append(p)

    deadline = time.time() + args.timeout_s

    results = {}
    while len(results) < 2 and time.time() < deadline:
        try:
            rank, ok, msg = result_q.get(timeout=0.25)
            results[rank] = (ok, msg)
        except queue.Empty:
            pass

        for p in procs:
            if p.exitcode is not None and p.exitcode != 0:
                # Keep collecting queue output if available, but remember that
                # a nonzero exit is failure even without a Python traceback.
                pass

    for p in procs:
        remaining = max(0.0, deadline - time.time())
        p.join(timeout=remaining)

    timed_out = [idx for idx, p in enumerate(procs) if p.is_alive()]
    if timed_out:
        for p in procs:
            if p.is_alive():
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
        f"devices=({args.dev0},{args.dev1}) broker_key={broker_key}"
    )


if __name__ == "__main__":
    main()
