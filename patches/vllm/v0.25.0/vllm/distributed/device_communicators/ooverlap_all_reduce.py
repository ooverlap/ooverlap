# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# OOVERLAP_FORCE_ALLREDUCE_BACKEND_PATCH_V1
# OOVERLAP_ROUND_ROBIN_SLOT_ADAPTER_V1
# OOVERLAP_ALL_GATHER_SLOT_ADAPTER_V1
# OOVERLAP_ALL_GATHER_ARBITRARY_DIM_V2
# OOVERLAP_QWEN_DIRECT_SLOT_GEMM_V2

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import os
import re
import sys
import uuid
from pathlib import Path
from typing import Final

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

from vllm.logger import init_logger

logger = init_logger(__name__)

_TRUE_VALUES: Final = ("1", "true", "yes", "on")


def _oo_dbg(msg: str) -> None:
    if os.getenv("VLLM_OOVERLAP_DEBUG", "").lower() not in _TRUE_VALUES:
        return
    print(f"[OOVERLAP_DEBUG pid={os.getpid()}] {msg}", flush=True)


def _sanitize_broker_key(text: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]", "_", text)
    return text[:180]


def _load_ooverlap_torch_ext():
    explicit = os.getenv("VLLM_OOVERLAP_TORCH_EXT")
    if explicit:
        so = Path(explicit).expanduser().resolve()
        if not so.exists():
            raise FileNotFoundError(
                f"VLLM_OOVERLAP_TORCH_EXT does not exist: {so}"
            )

        spec = importlib.util.spec_from_file_location(
            "ooverlap_torch_ext",
            str(so),
        )
        if spec is None or spec.loader is None:
            raise RuntimeError(
                f"Could not load ooverlap_torch_ext from {so}"
            )

        mod = importlib.util.module_from_spec(spec)
        sys.modules["ooverlap_torch_ext"] = mod
        spec.loader.exec_module(mod)
        return mod

    return importlib.import_module("ooverlap_torch_ext")


def _rank_and_world(group: ProcessGroup) -> tuple[int, int]:
    try:
        return dist.get_rank(group), dist.get_world_size(group)
    except Exception:
        return group.rank(), group.size()


def _group_ranks(group: ProcessGroup, world_size: int) -> list[int]:
    try:
        return list(dist.get_process_group_ranks(group))
    except Exception:
        return list(range(world_size))


def _short_broker_key(raw: str) -> str:
    sanitized = _sanitize_broker_key(raw)
    digest = hashlib.sha1(
        sanitized.encode("utf-8"),
        usedforsecurity=False,
    ).hexdigest()[:12]

    # Keep this short. ooverlap's Unix socket prefix has a hard <90 char limit.
    return f"vllm_oo_{digest}"


def _shared_broker_launch_id(
    group: ProcessGroup,
    world_size: int,
) -> str:
    """Create and share one fresh launch ID across communicator ranks."""
    override = os.getenv("VLLM_OOVERLAP_BROKER_LAUNCH_ID")
    if override:
        return _sanitize_broker_key(override)

    group_rank, _ = _rank_and_world(group)
    ranks = _group_ranks(group, world_size)
    src_rank = ranks[0]

    payload: list[str | None] = [
        uuid.uuid4().hex if group_rank == 0 else None
    ]
    dist.broadcast_object_list(payload, src=src_rank, group=group)

    launch_id = payload[0]
    if not launch_id:
        raise RuntimeError(
            "Failed to broadcast ooverlap broker launch id"
        )

    return _sanitize_broker_key(str(launch_id))


def _auto_broker_key(
    group: ProcessGroup,
    unique_name: str,
    world_size: int,
) -> str:
    override = os.getenv("VLLM_OOVERLAP_BROKER_KEY")
    if override:
        return _short_broker_key(override)

    ranks = ",".join(
        str(rank) for rank in _group_ranks(group, world_size)
    )
    master_addr = os.getenv("MASTER_ADDR", "localhost")
    master_port = os.getenv("MASTER_PORT", "0")
    local_world = os.getenv("LOCAL_WORLD_SIZE", str(world_size))
    launch_id = _shared_broker_launch_id(group, world_size)

    raw = (
        f"{master_addr}:{master_port}:{local_world}:"
        f"{unique_name}:{ranks}:{launch_id}"
    )
    return _short_broker_key(raw)


def _device_index(device: torch.device) -> int:
    if device.type != "cuda":
        raise RuntimeError(
            f"ooverlap requires CUDA device, got {device}"
        )

    if device.index is not None:
        return int(device.index)

    return int(torch.cuda.current_device())


def _gather_visible_devices(
    group: ProcessGroup,
    local_device: int,
    world_size: int,
) -> list[int]:
    devices: list[int | None] = [None for _ in range(world_size)]
    dist.all_gather_object(
        devices,
        int(local_device),
        group=group,
    )

    if any(device is None for device in devices):
        raise RuntimeError(
            f"Failed to gather ooverlap device list: {devices}"
        )

    return [int(device) for device in devices]


def _positive_env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default

    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(
            f"{name} must be an integer, got {raw!r}"
        ) from exc

    if value <= 0:
        raise ValueError(
            f"{name} must be positive, got {value}"
        )

    return value


def _normalize_rr_dtype(
    name: str,
    env_name: str = "VLLM_OOVERLAP_RR_DTYPE",
) -> tuple[str, torch.dtype]:
    normalized = name.strip().lower()

    aliases: dict[str, tuple[str, torch.dtype]] = {
        "bf16": ("bf16", torch.bfloat16),
        "bfloat16": ("bf16", torch.bfloat16),
        "torch.bfloat16": ("bf16", torch.bfloat16),
        "fp16": ("fp16", torch.float16),
        "float16": ("fp16", torch.float16),
        "half": ("fp16", torch.float16),
        "torch.float16": ("fp16", torch.float16),
        "fp32": ("fp32", torch.float32),
        "float32": ("fp32", torch.float32),
        "float": ("fp32", torch.float32),
        "torch.float32": ("fp32", torch.float32),
    }

    try:
        return aliases[normalized]
    except KeyError as exc:
        supported = "bf16, fp16, fp32"
        raise ValueError(
            f"{env_name} must be one of {supported}; "
            f"got {name!r}"
        ) from exc


class OoverlapAllReduce:
    """vLLM adapter for ooverlap's fixed round-robin IPC slot pools.

    All-reduce and all-gather return selected slot views directly without
    cloning. The caller must preserve deterministic same-stream slot reuse.
    """

    def __init__(
        self,
        group: ProcessGroup,
        device: torch.device,
        unique_name: str = "",
    ) -> None:
        self.group = group
        self.device_index = _device_index(device)
        self.device = torch.device("cuda", self.device_index)
        self.unique_name = unique_name

        self.disabled = True
        self.comm = None
        self.qwen_direct_slot_enabled = False

        self.rr_dtype_name = ""
        self.rr_torch_dtype: torch.dtype | None = None
        self.rr_slot_count = 0
        self.rr_capacity_bytes = 0

        self.ag_dtype_name = ""
        self.ag_torch_dtype: torch.dtype | None = None
        self.ag_slot_count = 0
        self.ag_capacity_bytes = 0

        if not dist.is_initialized():
            logger.warning(
                "OoverlapAllReduce disabled: "
                "torch.distributed is not initialized"
            )
            return

        if not torch.cuda.is_available():
            logger.warning(
                "OoverlapAllReduce disabled: CUDA is unavailable"
            )
            return

        self.rank, self.world_size = _rank_and_world(group)

        try:
            ext = _load_ooverlap_torch_ext()
        except Exception as exc:
            logger.warning(
                "OoverlapAllReduce disabled: "
                "could not load ooverlap_torch_ext: %s",
                exc,
            )
            return

        if not hasattr(ext, "Communicator"):
            logger.warning(
                "OoverlapAllReduce disabled: "
                "ooverlap_torch_ext lacks Communicator"
            )
            return

        try:
            _oo_dbg(
                f"enter init unique_name={unique_name!r} "
                f"rank={self.rank} world_size={self.world_size} "
                f"device_index={self.device_index}"
            )

            devices = _gather_visible_devices(
                group,
                self.device_index,
                self.world_size,
            )
            _oo_dbg(
                f"gathered devices rank={self.rank} devices={devices}"
            )

            broker_key = _auto_broker_key(
                group,
                unique_name,
                self.world_size,
            )
            _oo_dbg(
                f"before ext.Communicator rank={self.rank} "
                f"broker_key={broker_key!r}"
            )

            self.comm = ext.Communicator(
                devices,
                self.rank,
                broker_key,
            )
            _oo_dbg(
                f"after ext.Communicator rank={self.rank}"
            )

            if not hasattr(self.comm, "init_round_robin_slots"):
                raise RuntimeError(
                    "ooverlap_torch_ext.Communicator lacks "
                    "init_round_robin_slots(); rebuild the patched extension"
                )

            if not hasattr(self.comm, "all_reduce_round_robin"):
                raise RuntimeError(
                    "ooverlap_torch_ext.Communicator lacks "
                    "all_reduce_round_robin(); rebuild the patched extension"
                )

            if not hasattr(self.comm, "init_all_gather_round_robin_slots"):
                raise RuntimeError(
                    "ooverlap_torch_ext.Communicator lacks "
                    "init_all_gather_round_robin_slots(); rebuild the extension"
                )

            if not hasattr(self.comm, "all_gather_round_robin"):
                raise RuntimeError(
                    "ooverlap_torch_ext.Communicator lacks "
                    "all_gather_round_robin(); rebuild the extension"
                )

            rr_dtype_name, rr_torch_dtype = _normalize_rr_dtype(
                os.getenv("VLLM_OOVERLAP_RR_DTYPE", "bf16")
            )
            rr_slot_count = _positive_env_int(
                "VLLM_OOVERLAP_RR_SLOTS",
                32,
            )
            rr_capacity_bytes = _positive_env_int(
                "VLLM_OOVERLAP_RR_CAPACITY_BYTES",
                8 * 1024 * 1024,
            )

            ag_dtype_name, ag_torch_dtype = _normalize_rr_dtype(
                os.getenv("VLLM_OOVERLAP_AG_DTYPE", rr_dtype_name),
                "VLLM_OOVERLAP_AG_DTYPE",
            )
            ag_slot_count = _positive_env_int(
                "VLLM_OOVERLAP_AG_SLOTS",
                4,
            )
            ag_capacity_bytes = _positive_env_int(
                "VLLM_OOVERLAP_AG_CAPACITY_BYTES",
                64 * 1024 * 1024,
            )

            element_size = torch.empty(
                (),
                dtype=rr_torch_dtype,
            ).element_size()

            if rr_capacity_bytes % element_size != 0:
                raise ValueError(
                    "VLLM_OOVERLAP_RR_CAPACITY_BYTES must be divisible "
                    f"by the {element_size}-byte element size for "
                    f"{rr_dtype_name}; got {rr_capacity_bytes}"
                )

            ag_element_size = torch.empty(
                (),
                dtype=ag_torch_dtype,
            ).element_size()
            if ag_capacity_bytes % ag_element_size != 0:
                raise ValueError(
                    "VLLM_OOVERLAP_AG_CAPACITY_BYTES must be divisible "
                    f"by the {ag_element_size}-byte element size for "
                    f"{ag_dtype_name}; got {ag_capacity_bytes}"
                )

            _oo_dbg(
                f"before init_round_robin_slots rank={self.rank} "
                f"dtype={rr_dtype_name} "
                f"capacity_bytes={rr_capacity_bytes} "
                f"slot_count={rr_slot_count}"
            )

            self.comm.init_round_robin_slots(
                rr_dtype_name,
                rr_capacity_bytes,
                rr_slot_count,
            )

            self.rr_dtype_name = rr_dtype_name
            self.rr_torch_dtype = rr_torch_dtype
            self.rr_slot_count = rr_slot_count
            self.rr_capacity_bytes = rr_capacity_bytes

            _oo_dbg(
                f"after init_round_robin_slots rank={self.rank}"
            )

            _oo_dbg(
                f"before init_all_gather_round_robin_slots rank={self.rank} "
                f"dtype={ag_dtype_name} "
                f"capacity_bytes={ag_capacity_bytes} "
                f"slot_count={ag_slot_count}"
            )

            self.comm.init_all_gather_round_robin_slots(
                ag_dtype_name,
                ag_capacity_bytes,
                ag_slot_count,
            )

            self.ag_dtype_name = ag_dtype_name
            self.ag_torch_dtype = ag_torch_dtype
            self.ag_slot_count = ag_slot_count
            self.ag_capacity_bytes = ag_capacity_bytes

            _oo_dbg(
                f"after init_all_gather_round_robin_slots rank={self.rank}"
            )

        except Exception as exc:
            logger.warning(
                "OoverlapAllReduce disabled: "
                "communicator or slot-pool init failed: %s",
                exc,
            )

            if self.comm is not None and hasattr(self.comm, "destroy"):
                try:
                    self.comm.destroy()
                except Exception:
                    pass

            self.comm = None
            return

        self.disabled = False
        self.qwen_direct_slot_enabled = all(
            hasattr(self.comm, name)
            for name in (
                "acquire_all_reduce_slot",
                "all_reduce_preloaded_slot",
            )
        )
        if not self.qwen_direct_slot_enabled:
            _oo_dbg(
                "Qwen direct-slot path disabled: matching Ooverlap extension "
                "methods are unavailable"
            )

        logger.info_once(
            "Ooverlap communicator enabled for group '%s': "
            "rank=%d world_size=%d device=%s "
            "rr_dtype=%s rr_slots=%d rr_capacity_bytes=%d "
            "ag_dtype=%s ag_slots=%d ag_capacity_bytes=%d",
            unique_name or "<unnamed>",
            self.rank,
            self.world_size,
            self.device,
            self.rr_dtype_name,
            self.rr_slot_count,
            self.rr_capacity_bytes,
            self.ag_dtype_name,
            self.ag_slot_count,
            self.ag_capacity_bytes,
            scope="global",
        )

    def should_ooverlap_ar(self, inp: torch.Tensor) -> bool:
        if self.disabled or self.comm is None:
            return False

        if not inp.is_cuda:
            return False

        if inp.get_device() != self.device_index:
            return False

        if not inp.is_contiguous():
            return False

        if inp.numel() <= 0:
            return False

        if self.rr_torch_dtype is None:
            return False

        if inp.dtype != self.rr_torch_dtype:
            return False

        if inp.nbytes > self.rr_capacity_bytes:
            _oo_dbg(
                f"reject all_reduce rank={self.rank}: "
                f"input_bytes={inp.nbytes} exceeds "
                f"rr_capacity_bytes={self.rr_capacity_bytes}"
            )
            return False

        return True

    def qwen_row_parallel(
        self,
        inp: torch.Tensor,
        weight: torch.Tensor,
    ) -> torch.Tensor:
        """Write a Qwen row-parallel GEMM directly into the next AR slot."""
        if (
            not self.qwen_direct_slot_enabled
            or self.disabled
            or self.comm is None
        ):
            raise RuntimeError("Ooverlap Qwen direct-slot path is unavailable")

        if inp.dim() != 2 or weight.dim() != 2:
            raise RuntimeError(
                "Qwen direct-slot path requires 2D input and weight; "
                f"got input={tuple(inp.shape)} weight={tuple(weight.shape)}"
            )
        if not inp.is_cuda or not weight.is_cuda:
            raise RuntimeError("Qwen direct-slot path requires CUDA tensors")
        if inp.get_device() != self.device_index:
            raise RuntimeError(
                "Qwen direct-slot input is on the wrong device: "
                f"got={inp.device} expected={self.device}"
            )
        if weight.get_device() != self.device_index:
            raise RuntimeError(
                "Qwen direct-slot weight is on the wrong device: "
                f"got={weight.device} expected={self.device}"
            )
        if not inp.is_contiguous():
            raise RuntimeError(
                "Qwen direct-slot input must be contiguous; "
                f"shape={tuple(inp.shape)} stride={tuple(inp.stride())}"
            )
        if not weight.is_contiguous():
            raise RuntimeError(
                "Qwen direct-slot weight must be contiguous; "
                f"shape={tuple(weight.shape)} stride={tuple(weight.stride())}"
            )
        if self.rr_torch_dtype is None:
            raise RuntimeError("Ooverlap all-reduce slot dtype is unavailable")
        if inp.dtype != self.rr_torch_dtype or weight.dtype != inp.dtype:
            raise RuntimeError(
                "Qwen direct-slot dtype mismatch: "
                f"input={inp.dtype} weight={weight.dtype} "
                f"slot={self.rr_torch_dtype}"
            )
        if inp.shape[1] != weight.shape[1]:
            raise RuntimeError(
                "Qwen direct-slot GEMM K mismatch: "
                f"input={tuple(inp.shape)} weight={tuple(weight.shape)}"
            )

        output_shape = (inp.shape[0], weight.shape[0])
        output_numel = output_shape[0] * output_shape[1]
        output_bytes = output_numel * inp.element_size()
        if output_bytes > self.rr_capacity_bytes:
            raise RuntimeError(
                "Qwen direct-slot output exceeds Ooverlap slot capacity: "
                f"shape={output_shape} bytes={output_bytes} "
                f"capacity={self.rr_capacity_bytes}"
            )

        slot = self.comm.acquire_all_reduce_slot(list(output_shape))
        if tuple(slot.shape) != output_shape:
            raise RuntimeError(
                "Ooverlap returned a slot with the wrong shape: "
                f"got={tuple(slot.shape)} expected={output_shape}"
            )
        if slot.dtype != inp.dtype or slot.device != inp.device:
            raise RuntimeError(
                "Ooverlap returned a slot with the wrong dtype/device: "
                f"slot={slot.dtype}/{slot.device} "
                f"input={inp.dtype}/{inp.device}"
            )

        # Replaces F.linear for this row-parallel projection. The GEMM writes
        # the local partial result directly into Ooverlap-owned slot storage.
        torch.mm(inp, weight.t(), out=slot)
        return self.comm.all_reduce_preloaded_slot(slot)

    def all_reduce(
        self,
        inp: torch.Tensor,
    ) -> torch.Tensor | None:
        if not self.should_ooverlap_ar(inp):
            return None

        assert self.comm is not None

        #return self.comm.all_reduce_inplace(inp)

        # Zero-copy result: the returned tensor aliases the selected IPC slot.
        # Do not clone here. The patched ooverlap core protects asynchronous
        # launch-plan reuse with a per-slot plan-scratch ring.
        return self.comm.all_reduce_round_robin(inp)

    def all_gather(
        self,
        inp: torch.Tensor,
        dim: int = -1,
    ) -> torch.Tensor:
        if self.disabled or self.comm is None:
            raise RuntimeError("Forced ooverlap all-gather communicator is unavailable")

        if inp.dim() <= 0:
            raise RuntimeError(
                "Forced ooverlap all-gather requires an input with at least "
                f"one dimension, got shape={tuple(inp.shape)}"
            )

        normalized_dim = dim
        if normalized_dim < 0:
            normalized_dim += inp.dim()
        if normalized_dim < 0 or normalized_dim >= inp.dim():
            raise RuntimeError(
                f"Forced ooverlap all-gather received invalid dim={dim} "
                f"for shape={tuple(inp.shape)}"
            )
        if not inp.is_cuda:
            raise RuntimeError(
                f"Forced ooverlap all-gather requires a CUDA tensor, got {inp.device}"
            )
        if inp.get_device() != self.device_index:
            raise RuntimeError(
                "Forced ooverlap all-gather tensor is on the wrong device: "
                f"got={inp.device} expected={self.device}"
            )
        if not inp.is_contiguous():
            raise RuntimeError(
                "Forced ooverlap all-gather requires a contiguous tensor; "
                f"shape={tuple(inp.shape)} stride={tuple(inp.stride())}"
            )
        if inp.numel() <= 0:
            raise RuntimeError("Forced ooverlap all-gather requires a non-empty tensor")
        if self.ag_torch_dtype is None or inp.dtype != self.ag_torch_dtype:
            raise RuntimeError(
                "Forced ooverlap all-gather dtype does not match its slot pool: "
                f"input={inp.dtype} pool={self.ag_torch_dtype}"
            )

        output_bytes = inp.nbytes * self.world_size
        if output_bytes > self.ag_capacity_bytes:
            raise RuntimeError(
                "Forced ooverlap all-gather output exceeds slot capacity: "
                f"input_shape={tuple(inp.shape)} input_bytes={inp.nbytes} "
                f"world_size={self.world_size} output_bytes={output_bytes} "
                f"capacity_bytes={self.ag_capacity_bytes}"
            )

        _oo_dbg(
            f"all_gather rank={self.rank} shape={tuple(inp.shape)} "
            f"dtype={inp.dtype} dim={dim} normalized_dim={normalized_dim} "
            f"output_bytes={output_bytes}"
        )

        # ooverlap returns the same rank-major raw layout as
        # torch.distributed.all_gather_into_tensor: [rank, *input_shape],
        # flattened across the first two dimensions. Reproduce vLLM's base
        # communicator reshape for the requested concatenation dimension.
        gathered = self.comm.all_gather_round_robin(inp)
        if normalized_dim == 0:
            return gathered

        input_size = tuple(inp.size())
        output = gathered.reshape((self.world_size,) + input_size)
        output = output.movedim(0, normalized_dim)
        output_size = (
            input_size[:normalized_dim]
            + (self.world_size * input_size[normalized_dim],)
            + input_size[normalized_dim + 1 :]
        )
        return output.reshape(output_size)

    def destroy(self) -> None:
        if self.comm is not None and hasattr(self.comm, "destroy"):
            self.comm.destroy()

        self.comm = None
        self.disabled = True
        self.qwen_direct_slot_enabled = False
