#include "comm/collective/operation.h"

#include "ooverlap/system/runtime_utils.cuh"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace ooverlap {
namespace comm {
namespace collective {

bool operation_desc_build_ring_allreduce(
    OperationDesc* out,
    int rank,
    int world_size,
    size_t total_bytes,
    size_t chunk_bytes,
    exec::ChunkOpKind op,
    void* accum_ptr,
    void* next_accum_ptr,
    void* chunk_steps_ptr,
    void* chunk_done_ptr,
    ChunkState* chunk_states_ptr,
    uint32_t op_id,
    uint32_t epoch,
    bool enabled,
    uint64_t user_tag) {
    if (out == nullptr) {
        throw std::invalid_argument("operation_desc_build_ring_allreduce: out is null");
    }
    if (rank < 0) {
        throw std::invalid_argument("operation_desc_build_ring_allreduce: invalid rank");
    }
    if (world_size <= 0) {
        throw std::invalid_argument("operation_desc_build_ring_allreduce: invalid world_size");
    }
    if (total_bytes == 0) {
        throw std::invalid_argument("operation_desc_build_ring_allreduce: total_bytes must be > 0");
    }
    if (chunk_bytes == 0) {
        throw std::invalid_argument("operation_desc_build_ring_allreduce: chunk_bytes must be > 0");
    }
    if (op == exec::ChunkOpKind::kInvalid) {
        throw std::invalid_argument("operation_desc_build_ring_allreduce: invalid op");
    }
    if (accum_ptr == nullptr) {
        throw std::invalid_argument("operation_desc_build_ring_allreduce: accum_ptr is null");
    }
    if (world_size > 1 && next_accum_ptr == nullptr) {
        throw std::invalid_argument("operation_desc_build_ring_allreduce: next_accum_ptr is null");
    }
    if (chunk_steps_ptr == nullptr) {
        throw std::invalid_argument("operation_desc_build_ring_allreduce: chunk_steps_ptr is null");
    }
    if (chunk_done_ptr == nullptr) {
        throw std::invalid_argument("operation_desc_build_ring_allreduce: chunk_done_ptr is null");
    }
    if (chunk_states_ptr == nullptr) {
        throw std::invalid_argument("operation_desc_build_ring_allreduce: chunk_states_ptr is null");
    }
    if (op_id == 0) {
        throw std::invalid_argument("operation_desc_build_ring_allreduce: op_id must be > 0");
    }

    operation_desc_clear(out);

    out->op_id = op_id;
    out->epoch = epoch;
    out->flags = enabled ? kOperationFlagEnabled : 0u;

    out->rank = rank;
    out->world_size = world_size;
    out->prev_rank = (rank - 1 + world_size) % world_size;
    out->next_rank = (rank + 1) % world_size;

    out->total_bytes = total_bytes;
    out->chunk_bytes = chunk_bytes;
    out->num_chunks = operation_desc_compute_num_chunks(total_bytes, chunk_bytes);

    out->op = op;
    out->user_tag = user_tag;

    out->accum_ptr = reinterpret_cast<uint64_t>(accum_ptr);
    out->accum_bytes = total_bytes;

    out->next_accum_ptr = reinterpret_cast<uint64_t>(next_accum_ptr);
    out->next_accum_bytes = total_bytes;

    out->chunk_steps_ptr = reinterpret_cast<uint64_t>(chunk_steps_ptr);
    out->chunk_done_ptr = reinterpret_cast<uint64_t>(chunk_done_ptr);
    out->chunk_states_ptr = reinterpret_cast<uint64_t>(chunk_states_ptr);

    return operation_desc_is_valid(out);
}

bool operation_desc_reset_local_chunk_state(
    int device,
    const OperationDesc* desc) {
    if (device < 0) {
        throw std::invalid_argument("operation_desc_reset_local_chunk_state: invalid device");
    }
    if (!operation_desc_is_valid(desc)) {
        throw std::invalid_argument("operation_desc_reset_local_chunk_state: invalid desc");
    }

    system::runtime::set_device(device);
    system::runtime::check_cuda(
        cudaMemset(
            reinterpret_cast<void*>(desc->chunk_states_ptr),
            0,
            static_cast<size_t>(desc->num_chunks) * sizeof(ChunkState)),
        "cudaMemset(operation local chunk state)");

    return true;
}

bool operation_desc_reset_shared_progress(
    int device,
    const OperationDesc* desc) {
    if (device < 0) {
        throw std::invalid_argument("operation_desc_reset_shared_progress: invalid device");
    }
    if (!operation_desc_is_valid(desc)) {
        throw std::invalid_argument("operation_desc_reset_shared_progress: invalid desc");
    }

    system::runtime::set_device(device);
    system::runtime::check_cuda(
        cudaMemset(
            reinterpret_cast<void*>(desc->chunk_steps_ptr),
            0,
            static_cast<size_t>(desc->num_chunks) * sizeof(uint32_t)),
        "cudaMemset(operation chunk steps)");
    system::runtime::check_cuda(
        cudaMemset(
            reinterpret_cast<void*>(desc->chunk_done_ptr),
            0,
            static_cast<size_t>(desc->num_chunks) * sizeof(uint32_t)),
        "cudaMemset(operation chunk done)");

    return true;
}

bool device_operation_desc_create(
    DeviceOperationDesc* storage,
    int device,
    const OperationDesc* host_desc) {
    if (storage == nullptr) {
        throw std::invalid_argument("device_operation_desc_create: storage is null");
    }
    if (device < 0) {
        throw std::invalid_argument("device_operation_desc_create: invalid device");
    }
    if (!operation_desc_is_valid(host_desc)) {
        throw std::invalid_argument("device_operation_desc_create: host_desc is invalid");
    }

    device_operation_desc_destroy(storage);

    storage->device = device;
    system::runtime::set_device(device);
    system::runtime::check_cuda(
        cudaMalloc(&storage->ptr, sizeof(OperationDesc)),
        "cudaMalloc(device operation desc)");

    return device_operation_desc_write(storage, host_desc);
}

bool device_operation_desc_write(
    DeviceOperationDesc* storage,
    const OperationDesc* host_desc) {
    if (!device_operation_desc_is_configured(storage)) {
        throw std::invalid_argument("device_operation_desc_write: storage not configured");
    }
    if (!operation_desc_is_valid(host_desc)) {
        throw std::invalid_argument("device_operation_desc_write: host_desc is invalid");
    }

    system::runtime::set_device(storage->device);
    system::runtime::check_cuda(
        cudaMemcpy(
            storage->ptr,
            host_desc,
            sizeof(OperationDesc),
            cudaMemcpyHostToDevice),
        "cudaMemcpy(device operation desc)");

    return true;
}

void device_operation_desc_destroy(
    DeviceOperationDesc* storage) {
    if (storage == nullptr) {
        return;
    }

    if (storage->device >= 0) {
        system::runtime::set_device(storage->device);
    }

    if (storage->ptr != nullptr) {
        system::runtime::check_cuda(
            cudaFree(storage->ptr),
            "cudaFree(device operation desc)");
    }

    storage->ptr = nullptr;
    storage->device = -1;
}

bool chunk_state_table_init(
    ChunkStateTable* table,
    int device,
    uint32_t capacity) {
    if (table == nullptr) {
        throw std::invalid_argument("chunk_state_table_init: table is null");
    }
    if (device < 0) {
        throw std::invalid_argument("chunk_state_table_init: invalid device");
    }
    if (capacity == 0) {
        throw std::invalid_argument("chunk_state_table_init: capacity must be > 0");
    }

    chunk_state_table_destroy(table);

    table->capacity = capacity;
    table->device = device;

    system::runtime::set_device(device);
    system::runtime::check_cuda(
        cudaMalloc(
            &table->records,
            static_cast<size_t>(capacity) * sizeof(ChunkState)),
        "cudaMalloc(chunk state table records)");

    chunk_state_table_reset(table);
    return true;
}

void chunk_state_table_reset(
    ChunkStateTable* table) {
    if (!chunk_state_table_is_configured(table)) {
        throw std::invalid_argument("chunk_state_table_reset: table not configured");
    }

    system::runtime::set_device(table->device);
    system::runtime::check_cuda(
        cudaMemset(
            table->records,
            0,
            static_cast<size_t>(table->capacity) * sizeof(ChunkState)),
        "cudaMemset(chunk state table records)");
}

void chunk_state_table_destroy(
    ChunkStateTable* table) {
    if (table == nullptr) {
        return;
    }

    if (table->device >= 0) {
        system::runtime::set_device(table->device);
    }

    if (table->records != nullptr) {
        system::runtime::check_cuda(
            cudaFree(table->records),
            "cudaFree(chunk state table records)");
    }

    table->records = nullptr;
    table->capacity = 0;
    table->device = -1;
}

} // namespace collective
} // namespace comm
} // namespace ooverlap
