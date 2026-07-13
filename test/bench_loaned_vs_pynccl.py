#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import os
import re
import sys
import time
from pathlib import Path
from typing import Callable

import torch
import torch.distributed as dist


def _load_ext(path: str | None):
    if path is None:
        path = os.environ.get("VLLM_OOVERLAP_TORCH_EXT") or os.environ.get("OOVERLAP_TORCH_EXT")
    if not path:
        raise RuntimeError("pass --extension-so or set VLLM_OOVERLAP_TORCH_EXT/OOVERLAP_TORCH_EXT")
    so = Path(path).expanduser().resolve()
    if not so.exists():
        raise FileNotFoundError(f"extension not found: {so}")
    spec = importlib.util.spec_from_file_location("ooverlap_torch_ext", str(so))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load extension spec: {so}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["ooverlap_torch_ext"] = mod
    spec.loader.exec_module(mod)
    return mod


def parse_numels(s: str) -> list[int]:
    out = [int(x) for x in s.split(",") if x.strip()]
    if not out or any(x <= 0 for x in out):
        raise ValueError("--numels must be positive comma-separated ints")
    return out


def parse_dtypes(s: str) -> list[torch.dtype]:
    m = {
        "fp16": torch.float16,
        "f16": torch.float16,
        "float16": torch.float16,
        "half": torch.float16,
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp32": torch.float32,
        "f32": torch.float32,
        "float32": torch.float32,
        "float": torch.float32,
    }
    out = []
    for raw in s.split(","):
        k = raw.strip().lower()
        if not k:
            continue
        if k not in m:
            raise ValueError(f"bad dtype {raw!r}; choices={sorted(m)}")
        out.append(m[k])
    if not out:
        raise ValueError("empty --dtypes")
    return out


def dtype_label(dtype: torch.dtype) -> str:
    if dtype is torch.float16:
        return "fp16"
    if dtype is torch.bfloat16:
        return "bf16"
    if dtype is torch.float32:
        return "fp32"
    return str(dtype).replace("torch.", "")


def dtype_spec(dtype: torch.dtype) -> str:
    if dtype is torch.float16:
        return "fp16"
    if dtype is torch.bfloat16:
        return "bf16"
    if dtype is torch.float32:
        return "fp32"
    raise ValueError(dtype)


def sanitize_key(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", s)[:180]


def local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", "0"))


def local_world_size() -> int:
    return int(os.environ.get("LOCAL_WORLD_SIZE", "1"))


def expected_sum(world: int, base: float) -> float:
    return world * base + (world * (world - 1)) / 2.0


def assert_close(name: str, y: torch.Tensor, expected: float):
    exp = torch.full_like(y, expected)
    if y.dtype is torch.float32:
        rtol = atol = 1e-5
    else:
        rtol = atol = 1e-2
    torch.testing.assert_close(y, exp, rtol=rtol, atol=atol)
    if dist.get_rank() == 0:
        print(f"[pass] {name}: first={y.flatten()[0].item()} shape={tuple(y.shape)} dtype={y.dtype}", flush=True)


class VllmPyNcclAllReduce:
    def __init__(self, cpu_group, device: torch.device):
        try:
            from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
        except Exception as exc:
            raise RuntimeError("Could not import vLLM PyNcclCommunicator") from exc
        self.pynccl_comm = PyNcclCommunicator(group=cpu_group, device=device)
        if getattr(self.pynccl_comm, "disabled", True):
            raise RuntimeError("vLLM PyNcclCommunicator is disabled")

    def all_reduce(self, x: torch.Tensor) -> torch.Tensor:
        out = self.pynccl_comm.all_reduce(x)
        if out is None:
            raise RuntimeError("vLLM PyNcclCommunicator returned None")
        return out

    def destroy(self):
        if getattr(self, "pynccl_comm", None) is not None:
            self.pynccl_comm.destroy()
            self.pynccl_comm = None


def bench(label: str, fn: Callable[[torch.Tensor], torch.Tensor], x: torch.Tensor, device: torch.device, warmup: int, iters: int) -> tuple[float, float]:
    for _ in range(warmup):
        _ = fn(x)
    torch.cuda.synchronize(device)

    start_ev = torch.cuda.Event(enable_timing=True)
    end_ev = torch.cuda.Event(enable_timing=True)
    start_wall = time.perf_counter()
    start_ev.record()
    for _ in range(iters):
        _ = fn(x)
    end_ev.record()
    torch.cuda.synchronize(device)
    end_wall = time.perf_counter()
    cuda_ms = start_ev.elapsed_time(end_ev) / float(iters)
    wall_ms = (end_wall - start_wall) * 1000.0 / float(iters)
    return cuda_ms, wall_ms


def make_comm(ext, devices: list[int], rank: int, base_key: str, suffix: str):
    key = sanitize_key(f"{base_key}_{suffix}")
    return ext.Communicator(devices, rank, key)


def init_loaned_exact(comm, dtype: torch.dtype, nbytes: int, count: int):
    spec = f"{dtype_spec(dtype)}:{nbytes}:{count}"
    comm.init_loaned_slots(spec)


def main():
    p = argparse.ArgumentParser(description="Benchmark ooverlap loaned slots vs vLLM PyNCCL")
    p.add_argument("--extension-so", default=None)
    p.add_argument("--dist-backend", default="gloo", choices=("gloo", "nccl"))
    p.add_argument("--devices", default=None, help="visible device ids, e.g. 0,1; default 0..LOCAL_WORLD_SIZE-1")
    p.add_argument("--broker-key", default=None)
    p.add_argument("--numels", default="3584,14336,57344,229376,1048576,2097152,3670016")
    p.add_argument("--dtypes", default="bf16")
    p.add_argument("--bench-warmup", type=int, default=20)
    p.add_argument("--bench-iters", type=int, default=100)
    p.add_argument("--no-safe", action="store_true", help="skip ooverlap all_reduce_out safe path")
    p.add_argument("--enable-loaned-sync-replenish", action="store_true", help="after each loaned call, replenish one slot synchronously; includes registration cost")
    p.add_argument("--enable-loaned-bg", action="store_true", help="start C++ background replenisher and benchmark all_reduce_loaned")
    p.add_argument("--loaned-bg-initial", type=int, default=8)
    p.add_argument("--allow-bg-fallback", action="store_true", help="allow bg loaned mode to fall back to all_reduce_out if empty")
    args = p.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    if not dist.is_initialized():
        dist.init_process_group(backend=args.dist_backend)

    grank = dist.get_rank()
    world = dist.get_world_size()
    lrank = local_rank()
    lworld = local_world_size()
    if world != lworld:
        raise RuntimeError(f"single-node only: WORLD_SIZE={world}, LOCAL_WORLD_SIZE={lworld}")

    devices = list(range(lworld)) if not args.devices else [int(x) for x in args.devices.split(",") if x.strip()]
    if len(devices) != lworld:
        raise RuntimeError(f"devices={devices} length does not match LOCAL_WORLD_SIZE={lworld}")
    torch.cuda.set_device(devices[lrank])
    device = torch.device("cuda", devices[lrank])

    ext = _load_ext(args.extension_so)
    base_key = args.broker_key or sanitize_key(f"loaned_vs_pynccl_{os.environ.get('MASTER_ADDR','localhost')}_{os.environ.get('MASTER_PORT','0')}_{world}")

    if grank == 0:
        print("[info] extension:", ext, flush=True)
        print("[info] devices:", devices, "base_key:", base_key, flush=True)
        print("[info] numels:", parse_numels(args.numels), "dtypes:", args.dtypes, flush=True)
        print("[info] warmup/iters:", args.bench_warmup, args.bench_iters, flush=True)
        print("[info] modes: pynccl" + ("" if args.no_safe else ", safe_out") + ", loaned_once" + (", loaned_sync" if args.enable_loaned_sync_replenish else "") + (", loaned_bg" if args.enable_loaned_bg else ""), flush=True)

    dist.barrier()
    pynccl = VllmPyNcclAllReduce(dist.group.WORLD, device)

    try:
        for dtype in parse_dtypes(args.dtypes):
            for numel in parse_numels(args.numels):
                x = torch.full((numel,), 1.0 + lrank, device=device, dtype=dtype)
                nbytes = x.numel() * x.element_size()
                modes: list[tuple[str, Callable[[torch.Tensor], torch.Tensor], object | None]] = []

                # Correctness and benchmark pynccl.
                y = pynccl.all_reduce(x)
                torch.cuda.synchronize(device)
                assert_close(f"pynccl numel={numel} dtype={dtype_label(dtype)}", y, expected_sum(world, 1.0))
                modes.append(("pynccl", lambda t, pynccl=pynccl: pynccl.all_reduce(t), None))

                safe_comm = None
                if not args.no_safe:
                    safe_comm = make_comm(ext, devices, lrank, base_key, f"safe_{dtype_label(dtype)}_{numel}")
                    def safe_fn(t, comm=safe_comm):
                        out = torch.empty_like(t)
                        return comm.all_reduce_out(t, out)
                    y = safe_fn(x)
                    torch.cuda.synchronize(device)
                    assert_close(f"ooverlap_safe_out numel={numel} dtype={dtype_label(dtype)}", y, expected_sum(world, 1.0))
                    modes.append(("ooverlap_safe_out", safe_fn, safe_comm))

                # loaned_once needs enough one-shot slots for correctness + warmup + timed iters.
                loaned_once_comm = make_comm(ext, devices, lrank, base_key, f"loaned_once_{dtype_label(dtype)}_{numel}")
                init_loaned_exact(loaned_once_comm, dtype, nbytes, args.bench_warmup + args.bench_iters + 2)
                def loaned_once_fn(t, comm=loaned_once_comm):
                    return comm.all_reduce_loaned(t, False)
                y = loaned_once_fn(x)
                torch.cuda.synchronize(device)
                assert_close(f"ooverlap_loaned_once numel={numel} dtype={dtype_label(dtype)}", y, expected_sum(world, 1.0))
                modes.append(("ooverlap_loaned_once", loaned_once_fn, loaned_once_comm))

                sync_comm = None
                if args.enable_loaned_sync_replenish:
                    sync_comm = make_comm(ext, devices, lrank, base_key, f"loaned_sync_{dtype_label(dtype)}_{numel}")
                    init_loaned_exact(sync_comm, dtype, nbytes, 2)
                    def loaned_sync_fn(t, comm=sync_comm):
                        y = comm.all_reduce_loaned(t, False)
                        # This is collective registration through the same sequence on all ranks.
                        comm.replenish_loaned_slots_sync(1)
                        return y
                    y = loaned_sync_fn(x)
                    torch.cuda.synchronize(device)
                    assert_close(f"ooverlap_loaned_sync_replenish numel={numel} dtype={dtype_label(dtype)}", y, expected_sum(world, 1.0))
                    modes.append(("ooverlap_loaned_sync_replenish", loaned_sync_fn, sync_comm))

                bg_comm = None
                if args.enable_loaned_bg:
                    bg_comm = make_comm(ext, devices, lrank, base_key, f"loaned_bg_{dtype_label(dtype)}_{numel}")
                    init_loaned_exact(bg_comm, dtype, nbytes, args.loaned_bg_initial)
                    bg_comm.start_loaned_replenisher()
                    def loaned_bg_fn(t, comm=bg_comm):
                        return comm.all_reduce_loaned(t, bool(args.allow_bg_fallback))
                    y = loaned_bg_fn(x)
                    torch.cuda.synchronize(device)
                    assert_close(f"ooverlap_loaned_bg numel={numel} dtype={dtype_label(dtype)}", y, expected_sum(world, 1.0))
                    modes.append(("ooverlap_loaned_bg", loaned_bg_fn, bg_comm))

                results: dict[str, tuple[float, float]] = {}
                for label, fn, _owner in modes:
                    dist.barrier()
                    cuda_ms, wall_ms = bench(label, fn, x, device, args.bench_warmup, args.bench_iters)
                    dist.barrier()
                    results[label] = (cuda_ms, wall_ms)

                if grank == 0:
                    print(f"\n[bench] numel={numel} dtype={dtype_label(dtype)} bytes_per_rank={nbytes} warmup={args.bench_warmup} iters={args.bench_iters}")
                    for label, (cuda_ms, wall_ms) in results.items():
                        print(f"  {label}: cuda_ms={cuda_ms:.6f} wall_ms={wall_ms:.6f}")
                    base = results.get("pynccl")
                    if base:
                        pynccl_cuda, pynccl_wall = base
                        for label, (cuda_ms, wall_ms) in results.items():
                            if label == "pynccl":
                                continue
                            print(f"  {label}_speedup_vs_pynccl: cuda={pynccl_cuda/cuda_ms:.6f}x wall={pynccl_wall/wall_ms:.6f}x")
                    for label, _fn, owner in modes:
                        if owner is not None and hasattr(owner, "loaned_ready_count"):
                            try:
                                print(f"  {label}_ready={owner.loaned_ready_count()} retired={owner.loaned_retired_count()} queue={owner.loaned_replenish_queue_count() if hasattr(owner, 'loaned_replenish_queue_count') else 'NA'}")
                            except Exception:
                                pass
                    print(flush=True)

                # Cleanup communicators for this shape before moving on.
                for _label, _fn, owner in reversed(modes):
                    if owner is not None:
                        if hasattr(owner, "stop_loaned_replenisher"):
                            try:
                                owner.stop_loaned_replenisher()
                            except Exception:
                                pass
                        if hasattr(owner, "destroy"):
                            owner.destroy()
                dist.barrier()

        if grank == 0:
            print("[done] PASS", flush=True)
    finally:
        try:
            pynccl.destroy()
        except Exception:
            pass
        dist.barrier()


if __name__ == "__main__":
    main()
