#!/usr/bin/env python3
"""
OOVERLAP_WRAPPER_TEST_V2_WITH_VLLM_NCCL_SYMMETRIC

Smoke/correctness + benchmark for the minimal ooverlap PyTorch wrapper.

This file intentionally benchmarks ONLY:
  1. ooverlap_torch_ext.Communicator.all_reduce(...)
  2. Optional vLLM NCCL symmetric-memory all_reduce_with_copy(...)

It does NOT benchmark normal torch.distributed NCCL.

Expected ooverlap extension API:
    ooverlap_torch_ext.Communicator(devices: list[int], local_rank: int, broker_key: str)
    comm.all_reduce(tensor: torch.Tensor) -> torch.Tensor
    optional: comm.all_reduce_inplace(tensor: torch.Tensor) -> torch.Tensor
    optional: comm.destroy() -> None

Important:
    vLLM registers the custom op under torch.ops.vllm, not torch.vllm.
    The op is not available just because vLLM is installed. This test creates a
    vLLM PyNcclCommunicator and calls register_nccl_symmetric_ops(...), which
    registers torch.ops.vllm.all_reduce_symmetric_with_copy at runtime.

Run correctness only:
    CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 \
      test/test_ooverlap_torch_collectives_vllm_symm.py

Run ooverlap benchmark only:
    CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 \
      test/test_ooverlap_torch_collectives_vllm_symm.py \
      --numels 1024,1048576,16777216 \
      --bench-iters 100 \
      --bench-warmup 20

Run ooverlap vs vLLM NCCL symmetric-memory:
    CUDA_VISIBLE_DEVICES=0,1 VLLM_USE_NCCL_SYMM_MEM=1 torchrun --standalone --nproc_per_node=2 \
      test/test_ooverlap_torch_collectives_vllm_symm.py \
      --numels 1048576,16777216 \
      --bench-iters 100 \
      --bench-warmup 20 \
      --enable-vllm-nccl-symm

If the ooverlap extension is not at build/lib/ooverlap_torch_ext.so:
    --extension-so /path/to/ooverlap_torch_ext.so
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import os
import re
import sys
from pathlib import Path
from typing import Callable

import torch
import torch.distributed as dist
from time import sleep


def _repo_root_from_this_file() -> Path:
    p = Path(__file__).resolve()
    if p.parent.name == "test":
        return p.parents[1]
    return Path.cwd().resolve()


def _candidate_extension_paths(root: Path) -> list[Path]:
    paths: list[Path] = []
    env_path = os.environ.get("OOVERLAP_TORCH_EXT")
    if env_path:
        paths.append(Path(env_path).expanduser())
    paths.append(root / "build" / "lib" / "ooverlap_torch_ext.so")
    build_lib = root / "build" / "lib"
    if build_lib.exists():
        paths.extend(sorted(build_lib.glob("ooverlap_torch_ext*.so")))
    return paths


def _load_extension_from_path(so: Path):
    spec = importlib.util.spec_from_file_location("ooverlap_torch_ext", str(so))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not create import spec for {so}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["ooverlap_torch_ext"] = mod
    spec.loader.exec_module(mod)
    return mod


def load_ooverlap_torch_ext(extension_so: str | None):
    if extension_so:
        so = Path(extension_so).expanduser().resolve()
        if not so.exists():
            raise FileNotFoundError(f"--extension-so does not exist: {so}")
        return _load_extension_from_path(so)

    root = _repo_root_from_this_file()
    for so in _candidate_extension_paths(root):
        if so.exists():
            return _load_extension_from_path(so.resolve())

    try:
        return importlib.import_module("ooverlap_torch_ext")
    except ImportError as exc:
        candidates = "\n".join(f"  - {p}" for p in _candidate_extension_paths(root))
        raise FileNotFoundError(
            "Could not find ooverlap_torch_ext. Build it first, or pass --extension-so.\n"
            f"Looked in:\n{candidates}"
        ) from exc


def parse_devices(text: str | None, local_world_size: int) -> list[int]:
    if text is None or text.strip() == "":
        return list(range(local_world_size))
    devices = [int(x) for x in text.split(",") if x.strip() != ""]
    if len(devices) != local_world_size:
        raise ValueError(
            f"--devices must contain LOCAL_WORLD_SIZE={local_world_size} entries, got {devices}"
        )
    return devices


def parse_numels(text: str) -> list[int]:
    values = [int(x) for x in text.split(",") if x.strip()]
    if not values or any(v <= 0 for v in values):
        raise ValueError("--numels must be a comma-separated list of positive integers")
    return values


def parse_dtypes(text: str) -> list[torch.dtype]:
    mapping = {
        "fp16": torch.float16,
        "float16": torch.float16,
        "half": torch.float16,
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    out: list[torch.dtype] = []
    for raw in text.split(","):
        key = raw.strip().lower()
        if not key:
            continue
        if key not in mapping:
            raise ValueError(f"unsupported dtype '{raw}', expected one of {sorted(mapping)}")
        out.append(mapping[key])
    if not out:
        raise ValueError("--dtypes produced an empty dtype list")
    return out


def dtype_label(dtype: torch.dtype) -> str:
    if dtype is torch.float16:
        return "fp16"
    if dtype is torch.bfloat16:
        return "bf16"
    if dtype is torch.float32:
        return "fp32"
    return str(dtype)


def sanitize_broker_key(s: str) -> str:
    s = re.sub(r"[^A-Za-z0-9_.-]", "_", s)
    return s[:180] if len(s) > 180 else s


def default_broker_key() -> str:
    master_addr = os.environ.get("MASTER_ADDR", "localhost")
    master_port = os.environ.get("MASTER_PORT", "0")
    local_world = os.environ.get("LOCAL_WORLD_SIZE", "unknown")
    return sanitize_broker_key(f"oo_torch_test_{master_addr}_{master_port}_{local_world}")


def init_dist(backend: str) -> None:
    if dist.is_available() and not dist.is_initialized():
        dist.init_process_group(backend=backend)


def barrier() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def get_local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", "0"))


def get_local_world_size() -> int:
    if "LOCAL_WORLD_SIZE" in os.environ:
        return int(os.environ["LOCAL_WORLD_SIZE"])
    if dist.is_initialized():
        return dist.get_world_size()
    return 1


def expected_rank_sum(world_size: int, base: float) -> float:
    # Each rank uses value = base + rank.
    return world_size * base + (world_size * (world_size - 1)) / 2.0


def assert_allclose(name: str, actual: torch.Tensor, expected_value: float) -> None:
    expected = torch.full_like(actual, expected_value)
    if actual.dtype == torch.float32:
        rtol, atol = 1e-5, 1e-5
    else:
        rtol, atol = 1e-2, 1e-2

    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)
    first = actual.flatten()[0].item() if actual.numel() > 0 else "<empty>"
    print(f"[pass] {name}: dtype={actual.dtype}, shape={tuple(actual.shape)}, first={first}")


def run_ooverlap_correctness(
    comm,
    numel: int,
    dtype: torch.dtype,
    rank: int,
    world: int,
    device: torch.device,
) -> None:
    base = 1.0
    x = torch.full((numel,), base + rank, device=device, dtype=dtype)
    y = comm.all_reduce(x)
    torch.cuda.synchronize(device)

    assert y.is_cuda, "all_reduce output must be CUDA"
    assert y.numel() == x.numel(), f"output numel mismatch: {y.numel()} vs {x.numel()}"
    assert y.dtype == x.dtype, f"output dtype mismatch: {y.dtype} vs {x.dtype}"
    assert_allclose(f"ooverlap all_reduce numel={numel}", y, expected_rank_sum(world, base))

    # Second call with same shape/dtype should hit cached scratch if your wrapper caches it.
    base = 10.0
    x2 = torch.full((numel,), base + rank, device=device, dtype=dtype)
    y2 = comm.all_reduce(x2)
    torch.cuda.synchronize(device)
    assert_allclose(f"ooverlap all_reduce cached numel={numel}", y2, expected_rank_sum(world, base))


def run_ooverlap_inplace_correctness(
    comm,
    numel: int,
    dtype: torch.dtype,
    rank: int,
    world: int,
    device: torch.device,
) -> None:
    if not hasattr(comm, "all_reduce_inplace"):
        print("[skip] comm.all_reduce_inplace is not exposed")
        return

    base = 3.0
    z = torch.full((numel,), base + rank, device=device, dtype=dtype)
    out = comm.all_reduce_inplace(z)
    torch.cuda.synchronize(device)
    check = z if out is None else out
    assert_allclose(f"ooverlap all_reduce_inplace numel={numel}", check, expected_rank_sum(world, base))


def torch_ops_has_vllm_symm_ar() -> bool:
    try:
        return hasattr(torch.ops, "vllm") and hasattr(torch.ops.vllm, "all_reduce_symmetric_with_copy")
    except Exception:
        return False


class VllmNcclSymmetricAllReduce:
    """Runtime wrapper around vLLM's torch.ops.vllm.all_reduce_symmetric_with_copy."""

    def __init__(self, cpu_group, device: torch.device):
        # This must be set before importing vllm.envs / pynccl_allocator.
        os.environ.setdefault("VLLM_USE_NCCL_SYMM_MEM", "1")

        try:
            import vllm  # noqa: F401
        except Exception as exc:
            raise RuntimeError(
                "Could not import vllm. Install a recent vLLM first."
            ) from exc

        try:
            from vllm.distributed.device_communicators.pynccl import (
                PyNcclCommunicator,
                register_nccl_symmetric_ops,
            )
            from vllm.distributed.device_communicators.pynccl_allocator import (
                is_symmetric_memory_enabled,
            )
        except Exception as exc:
            raise RuntimeError(
                "Installed vLLM does not expose the NCCL symmetric-memory helper APIs. "
                "Install a current vLLM nightly or source checkout."
            ) from exc

        self._is_symmetric_memory_enabled = is_symmetric_memory_enabled
        self.pynccl_comm = PyNcclCommunicator(group=cpu_group, device=device)
        if self.pynccl_comm.disabled:
            raise RuntimeError("vLLM PyNcclCommunicator is disabled")

        register_nccl_symmetric_ops(self.pynccl_comm)

        if not torch_ops_has_vllm_symm_ar():
            raise RuntimeError(
                "vLLM did not register torch.ops.vllm.all_reduce_symmetric_with_copy. "
                "Your vLLM version is too old for this benchmark."
            )

        # Trigger allocator compile/registration path once.
        probe = torch.empty(1, device=device, dtype=torch.float16)
        _ = torch.ops.vllm.all_reduce_symmetric_with_copy(probe)
        torch.cuda.synchronize(device)

        if not self._is_symmetric_memory_enabled():
            raise RuntimeError(
                "vLLM NCCL symmetric memory is not enabled. Check "
                "VLLM_USE_NCCL_SYMM_MEM=1, NCCL headers, Torch >= 2.8.0a0, "
                "and NCCL >= 2.27.3."
            )

    def all_reduce(self, x: torch.Tensor) -> torch.Tensor:
        return torch.ops.vllm.all_reduce_symmetric_with_copy(x)

    def destroy(self) -> None:
        if getattr(self, "pynccl_comm", None) is not None:
            self.pynccl_comm.destroy()


def maybe_make_vllm_nccl_symmetric(enabled: bool, device: torch.device):
    if not enabled:
        return None
    try:
        return VllmNcclSymmetricAllReduce(dist.group.WORLD, device)
    except Exception as exc:
        rank = dist.get_rank() if dist.is_initialized() else 0
        print(f"[rank {rank}] [skip] vLLM NCCL symmetric benchmark unavailable: {exc}")
        return None


def run_vllm_symm_correctness(
    vllm_symm,
    numel: int,
    dtype: torch.dtype,
    rank: int,
    world: int,
    device: torch.device,
) -> None:
    if vllm_symm is None:
        return
    base = 5.0
    x = torch.full((numel,), base + rank, device=device, dtype=dtype)
    y = vllm_symm.all_reduce(x)
    torch.cuda.synchronize(device)
    assert_allclose(f"vLLM NCCL symmetric all_reduce numel={numel}", y, expected_rank_sum(world, base))


def benchmark_callable(
    label: str,
    fn: Callable[[torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    device: torch.device,
    warmup: int,
    iters: int,
) -> float:
    for _ in range(warmup):
        _ = fn(x)
    torch.cuda.synchronize(device)

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(iters):
        _ = fn(x)
    end.record()

    torch.cuda.synchronize(device)
    _ = label
    return start.elapsed_time(end) / float(iters)


def run_benchmarks(
    comm,
    vllm_symm,
    numel: int,
    dtype: torch.dtype,
    rank: int,
    device: torch.device,
    warmup: int,
    iters: int,
) -> None:
    if iters <= 0:
        return

    x = torch.full((numel,), 1.0 + rank, device=device, dtype=dtype)
    bytes_per_rank = x.numel() * x.element_size()

    results: dict[str, float] = {}
    results["ooverlap_ms"] = benchmark_callable(
        "ooverlap",
        lambda t: comm.all_reduce(t),
        x,
        device,
        warmup,
        iters,
    )

    if vllm_symm is not None:
        results["vllm_nccl_symmetric_copy_ms"] = benchmark_callable(
            "vllm_nccl_symmetric_copy",
            lambda t: vllm_symm.all_reduce(t),
            x,
            device,
            warmup,
            iters,
        )

    rank0 = dist.get_rank() == 0 if dist.is_initialized() else True
    if rank0:
        print(
            f"\n[bench] all_reduce numel={numel} dtype={dtype_label(dtype)} "
            f"bytes_per_rank={bytes_per_rank} iters={iters} warmup={warmup}"
        )
        for key, value in results.items():
            print(f"  {key}: {value:.6f}")

        symm = results.get("vllm_nccl_symmetric_copy_ms")
        if symm is not None and symm > 0:
            print(f"  ooverlap_speedup_over_vllm_nccl_symmetric_copy: {symm / results['ooverlap_ms']:.6f}x")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test/benchmark ooverlap_torch_ext vs optional vLLM NCCL symmetric-memory path"
    )
    parser.add_argument("--extension-so", default=None, help="Path to ooverlap_torch_ext.so")
    parser.add_argument("--dist-backend", default="gloo", choices=("gloo", "nccl"))
    parser.add_argument("--devices", default=None, help="Visible device ids, e.g. '0,1'. Default: 0..LOCAL_WORLD_SIZE-1")
    parser.add_argument("--broker-key", default=None, help="Shared ooverlap Broker key. Default derives from MASTER_ADDR/PORT.")
    parser.add_argument("--numels", default="1024,1048576", help="Comma-separated element counts")
    parser.add_argument("--dtypes", default="fp16", help="Comma-separated dtypes: fp16,bf16,fp32")
    parser.add_argument("--test-inplace", action="store_true", help="Also test comm.all_reduce_inplace if available")
    parser.add_argument("--bench-iters", type=int, default=0, help="If >0, run timing loops")
    parser.add_argument("--bench-warmup", type=int, default=10)
    parser.add_argument(
        "--enable-vllm-nccl-symm",
        action="store_true",
        help="Benchmark vLLM NCCL symmetric-memory all_reduce_with_copy if available",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this test")

    init_dist(args.dist_backend)

    local_rank = get_local_rank()
    local_world_size = get_local_world_size()
    global_rank = dist.get_rank() if dist.is_initialized() else 0
    global_world = dist.get_world_size() if dist.is_initialized() else 1

    if global_world != local_world_size:
        raise RuntimeError(
            "This first wrapper test assumes one node only: "
            f"global_world={global_world}, local_world_size={local_world_size}"
        )

    devices = parse_devices(args.devices, local_world_size)
    device_id = devices[local_rank]
    torch.cuda.set_device(device_id)
    device = torch.device("cuda", device_id)

    broker_key = sanitize_broker_key(args.broker_key or default_broker_key())

    ext = load_ooverlap_torch_ext(args.extension_so)
    if not hasattr(ext, "Communicator"):
        raise AttributeError("ooverlap_torch_ext does not expose Communicator")

    if global_rank == 0:
        print("[info] script marker: OOVERLAP_WRAPPER_TEST_V2_WITH_VLLM_NCCL_SYMMETRIC")
        print("[info] loaded extension:", ext)
        print("[info] global_world_size:", global_world)
        print("[info] local_world_size:", local_world_size)
        print("[info] devices:", devices)
        print("[info] broker_key:", broker_key)
        print("[info] numels:", parse_numels(args.numels))
        print("[info] dtypes:", args.dtypes)
        print("[info] dist_backend:", args.dist_backend)
        print("[info] enable_vllm_nccl_symm:", args.enable_vllm_nccl_symm)

    barrier()

    vllm_symm = maybe_make_vllm_nccl_symmetric(args.enable_vllm_nccl_symm, device)

    if global_rank == 0:
        print(
            "[info] torch.ops.vllm.all_reduce_symmetric_with_copy available:",
            torch_ops_has_vllm_symm_ar(),
        )

    barrier()

    comm = ext.Communicator(devices, local_rank, broker_key)

    try:
        for dtype in parse_dtypes(args.dtypes):
            for numel in parse_numels(args.numels):
                run_ooverlap_correctness(comm, numel, dtype, local_rank, local_world_size, device)
                run_vllm_symm_correctness(vllm_symm, numel, dtype, local_rank, local_world_size, device)

                if args.test_inplace:
                    run_ooverlap_inplace_correctness(comm, numel, dtype, local_rank, local_world_size, device)

                run_benchmarks(
                    comm=comm,
                    vllm_symm=vllm_symm,
                    numel=numel,
                    dtype=dtype,
                    rank=local_rank,
                    device=device,
                    warmup=args.bench_warmup,
                    iters=args.bench_iters,
                )

        barrier()
        print(f"[rank {global_rank}] PASS")
    finally:
        if hasattr(comm, "destroy"):
            comm.destroy()
        if vllm_symm is not None:
            vllm_symm.destroy()

    barrier()


if __name__ == "__main__":
    main()
