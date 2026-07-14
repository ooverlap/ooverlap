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


def load_extension(path: str | None):
    path = path or os.environ.get("VLLM_OOVERLAP_TORCH_EXT") or os.environ.get("OOVERLAP_TORCH_EXT")
    if not path:
        raise RuntimeError("pass --extension-so or set VLLM_OOVERLAP_TORCH_EXT")
    so = Path(path).expanduser().resolve()
    if not so.is_file():
        raise FileNotFoundError(f"extension not found: {so}")
    spec = importlib.util.spec_from_file_location("ooverlap_torch_ext", str(so))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load extension: {so}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["ooverlap_torch_ext"] = mod
    spec.loader.exec_module(mod)
    return mod


def parse_numels(text: str) -> list[int]:
    values = [int(x.strip()) for x in text.split(",") if x.strip()]
    if not values or any(x <= 0 for x in values):
        raise ValueError("--numels must contain positive comma-separated integers")
    return values


def parse_devices(text: str) -> list[int]:
    values = [int(x.strip()) for x in text.split(",") if x.strip()]
    if not values or any(x < 0 for x in values):
        raise ValueError("--devices must contain non-negative comma-separated integers")
    return values


def parse_dtypes(text: str) -> list[torch.dtype]:
    aliases = {
        "fp16": torch.float16,
        "f16": torch.float16,
        "float16": torch.float16,
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp32": torch.float32,
        "f32": torch.float32,
        "float32": torch.float32,
    }
    result: list[torch.dtype] = []
    for raw in text.split(","):
        key = raw.strip().lower()
        if not key:
            continue
        if key not in aliases:
            raise ValueError(f"unsupported dtype: {raw!r}")
        result.append(aliases[key])
    if not result:
        raise ValueError("--dtypes cannot be empty")
    return result


def dtype_name(dtype: torch.dtype) -> str:
    if dtype == torch.float16:
        return "fp16"
    if dtype == torch.bfloat16:
        return "bf16"
    if dtype == torch.float32:
        return "fp32"
    raise ValueError(dtype)


def sanitize_key(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", text)[:160]


def expected_sum(world_size: int) -> float:
    # Rank r contributes 1 + r.
    return world_size + world_size * (world_size - 1) / 2.0


def check_result(label: str, result: torch.Tensor, expected: float) -> None:
    torch.cuda.synchronize(result.device)
    reference = torch.full_like(result, expected)
    tol = 1e-5 if result.dtype == torch.float32 else 1e-2
    torch.testing.assert_close(result, reference, rtol=tol, atol=tol)
    if dist.get_rank() == 0:
        print(
            f"[pass] {label}: first={result.flatten()[0].item()} "
            f"shape={tuple(result.shape)} dtype={result.dtype}",
            flush=True,
        )


class VllmPyNccl:
    def __init__(self, group, device: torch.device):
        try:
            from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
        except Exception as exc:
            raise RuntimeError("could not import vLLM PyNcclCommunicator") from exc
        self.comm = PyNcclCommunicator(group=group, device=device)
        if getattr(self.comm, "disabled", True):
            raise RuntimeError("vLLM PyNcclCommunicator is disabled")

    def all_reduce(self, tensor: torch.Tensor) -> torch.Tensor:
        # Exact vLLM API: output allocation and return happen inside this call.
        result = self.comm.all_reduce(tensor)
        if result is None:
            raise RuntimeError("vLLM PyNcclCommunicator returned None")
        return result

    def destroy(self) -> None:
        if self.comm is not None:
            self.comm.destroy()
            self.comm = None


def benchmark(
    fn: Callable[[torch.Tensor], torch.Tensor],
    tensor: torch.Tensor,
    warmup: int,
    iterations: int,
) -> tuple[float, float]:
    # Match normal API usage: each call returns a tensor and the local Python
    # reference is replaced by the next call. No copy or allocation is added
    # outside either implementation.
    result: torch.Tensor | None = None
    for _ in range(warmup):
        result = fn(tensor)
    torch.cuda.synchronize(tensor.device)

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_wall = time.perf_counter()
    start_event.record()
    for _ in range(iterations):
        result = fn(tensor)
    end_event.record()
    torch.cuda.synchronize(tensor.device)
    end_wall = time.perf_counter()

    cuda_ms = start_event.elapsed_time(end_event) / iterations
    wall_ms = (end_wall - start_wall) * 1000.0 / iterations
    # Keep the final returned tensor alive until after synchronization above.
    assert result is not None
    return cuda_ms, wall_ms


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark fixed ooverlap round-robin IPC buffers vs vLLM PyNCCL"
    )
    parser.add_argument("--extension-so", default=None)
    parser.add_argument("--dist-backend", default="gloo", choices=("gloo", "nccl"))
    parser.add_argument("--devices", default=None, help="physical CUDA IDs, e.g. 0,1")
    parser.add_argument("--broker-key", default=None)
    parser.add_argument(
        "--numels",
        default="3584,14336,57344,229376,1048576,2097152,3670016",
    )
    parser.add_argument("--dtypes", default="bf16")
    parser.add_argument("--slots", type=int, default=8)
    parser.add_argument(
        "--capacity-bytes",
        type=int,
        default=0,
        help="0 uses the largest requested tensor for each dtype",
    )
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    args = parser.parse_args()

    if args.slots <= 0:
        raise ValueError("--slots must be positive")
    if args.capacity_bytes < 0:
        raise ValueError("--capacity-bytes cannot be negative")
    if args.warmup < 0 or args.iters <= 0:
        raise ValueError("warmup must be >= 0 and iters must be > 0")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    if not dist.is_initialized():
        dist.init_process_group(backend=args.dist_backend)

    rank = dist.get_rank()
    world = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    local_world = int(os.environ.get("LOCAL_WORLD_SIZE", str(world)))
    if world != local_world:
        raise RuntimeError("single-node benchmark only")

    devices = list(range(local_world)) if args.devices is None else parse_devices(args.devices)
    if len(devices) != local_world:
        raise RuntimeError(f"devices={devices} does not match LOCAL_WORLD_SIZE={local_world}")

    torch.cuda.set_device(devices[local_rank])
    device = torch.device("cuda", devices[local_rank])

    ext = load_extension(args.extension_so)
    for method in ("init_round_robin_slots", "all_reduce_round_robin"):
        if not hasattr(ext.Communicator, method):
            raise RuntimeError(f"extension is missing Communicator.{method}; rebuild after patching")

    numels = parse_numels(args.numels)
    dtypes = parse_dtypes(args.dtypes)
    base_key = args.broker_key or sanitize_key(
        f"rr_pynccl_{os.environ.get('MASTER_ADDR', 'localhost')}_"
        f"{os.environ.get('MASTER_PORT', '0')}_{world}"
    )

    if rank == 0:
        print(f"[info] extension={ext}", flush=True)
        print(f"[info] devices={devices} broker_key={base_key}", flush=True)
        print(f"[info] numels={numels} dtypes={[dtype_name(x) for x in dtypes]}", flush=True)
        print(f"[info] slots={args.slots} warmup={args.warmup} iters={args.iters}", flush=True)
        print("[info] timed APIs:", flush=True)
        print("  pynccl: PyNcclCommunicator.all_reduce(input)", flush=True)
        print("  ooverlap: Communicator.all_reduce_round_robin(input)", flush=True)

    dist.barrier()
    pynccl = VllmPyNccl(dist.group.WORLD, device)

    try:
        for dtype in dtypes:
            elem_size = torch.empty((), dtype=dtype).element_size()
            required_capacity = max(numels) * elem_size
            capacity = args.capacity_bytes or required_capacity
            if capacity < required_capacity:
                raise ValueError(
                    f"capacity {capacity} is smaller than required {required_capacity} bytes"
                )
            if capacity % elem_size != 0:
                raise ValueError("capacity must be divisible by dtype element size")

            key = sanitize_key(f"{base_key}_{dtype_name(dtype)}")
            round_robin = ext.Communicator(devices, local_rank, key)
            round_robin.init_round_robin_slots(dtype_name(dtype), capacity, args.slots)

            if rank == 0:
                print(
                    f"\n[dtype] {dtype_name(dtype)} capacity_bytes={capacity} "
                    f"slots={args.slots}",
                    flush=True,
                )

            try:
                for numel in numels:
                    tensor = torch.full(
                        (numel,),
                        1.0 + rank,
                        device=device,
                        dtype=dtype,
                    )
                    expected = expected_sum(world)
                    nbytes = tensor.numel() * tensor.element_size()

                    dist.barrier()
                    y_pynccl = pynccl.all_reduce(tensor)
                    check_result(
                        f"pynccl_vllm numel={numel} dtype={dtype_name(dtype)}",
                        y_pynccl,
                        expected,
                    )

                    dist.barrier()
                    y_rr = round_robin.all_reduce_round_robin(tensor)
                    check_result(
                        f"ooverlap_round_robin numel={numel} dtype={dtype_name(dtype)}",
                        y_rr,
                        expected,
                    )

                    dist.barrier()
                    pynccl_cuda, pynccl_wall = benchmark(
                        pynccl.all_reduce, tensor, args.warmup, args.iters
                    )
                    dist.barrier()
                    rr_cuda, rr_wall = benchmark(
                        round_robin.all_reduce_round_robin,
                        tensor,
                        args.warmup,
                        args.iters,
                    )
                    dist.barrier()

                    if rank == 0:
                        print(
                            f"\n[bench] numel={numel} dtype={dtype_name(dtype)} "
                            f"bytes_per_rank={nbytes}",
                            flush=True,
                        )
                        print(
                            f"  pynccl_vllm_api: cuda_ms={pynccl_cuda:.6f} "
                            f"wall_ms={pynccl_wall:.6f}",
                            flush=True,
                        )
                        print(
                            f"  ooverlap_round_robin: cuda_ms={rr_cuda:.6f} "
                            f"wall_ms={rr_wall:.6f}",
                            flush=True,
                        )
                        print(
                            "  ooverlap_speedup_vs_pynccl: "
                            f"cuda={pynccl_cuda / rr_cuda:.6f}x "
                            f"wall={pynccl_wall / rr_wall:.6f}x",
                            flush=True,
                        )
                        if hasattr(round_robin, "round_robin_next_slot"):
                            print(
                                f"  round_robin_next_slot={round_robin.round_robin_next_slot()}",
                                flush=True,
                            )
            finally:
                round_robin.destroy()
                dist.barrier()

        if rank == 0:
            print("\n[done] PASS", flush=True)
    finally:
        try:
            pynccl.destroy()
        finally:
            dist.barrier()
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
