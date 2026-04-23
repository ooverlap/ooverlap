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
    void* ready_queue_ptr,
    void* next_ready_queue_ptr,
    void* progress_ptr,
    void* prev_progress_ptr,
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
    if (progress_ptr == nullptr) {
        throw std::invalid_argument("operation_desc_build_ring_allreduce: progress_ptr is null");
    }
    if (world_size > 1 && prev_progress_ptr == nullptr) {
        throw std::invalid_argument("operation_desc_build_ring_allreduce: prev_progress_ptr is null");
    }
    if (chunk_states_ptr == nullptr) {
        throw std::invalid_argument("operation_desc_build_ring_allreduce: chunk_states_ptr is null");
    }
    if (op_id == 0) {
        throw std::invalid_argument("operation_desc_build_ring_allreduce: op_id must be > 0");
    }

    operation_desc_clear(out);

    const uint32_t num_chunks =
        operation_desc_compute_num_chunks(total_bytes, chunk_bytes);
    const size_t queue_bytes =
        2 * sizeof(uint32_t) + static_cast<size_t>(num_chunks) * sizeof(ReadyItem);
    const size_t progress_bytes =
        static_cast<size_t>(num_chunks) * sizeof(uint32_t);

    out->op_id = op_id;
    out->epoch = epoch;
    out->flags = enabled ? kOperationFlagEnabled : 0u;

    out->rank = rank;
    out->world_size = world_size;
    out->prev_rank = (rank - 1 + world_size) % world_size;
    out->next_rank = (rank + 1) % world_size;

    out->total_bytes = total_bytes;
    out->chunk_bytes = chunk_bytes;
    out->num_chunks = num_chunks;

    out->op = op;
    out->user_tag = user_tag;

    out->accum_ptr = reinterpret_cast<uint64_t>(accum_ptr);
    out->accum_bytes = total_bytes;

    out->next_accum_ptr = reinterpret_cast<uint64_t>(next_accum_ptr);
    out->next_accum_bytes = total_bytes;

    // Retained only for compatibility during migration.
    out->ready_queue_ptr = reinterpret_cast<uint64_t>(ready_queue_ptr);
    out->ready_queue_bytes = (ready_queue_ptr != nullptr) ? queue_bytes : 0u;

    out->next_ready_queue_ptr = reinterpret_cast<uint64_t>(next_ready_queue_ptr);
    out->next_ready_queue_bytes = (next_ready_queue_ptr != nullptr) ? queue_bytes : 0u;

    // New signaling semantics.
    out->done_ptr = reinterpret_cast<uint64_t>(progress_ptr);
    out->done_bytes = progress_bytes;

    out->next_done_ptr = reinterpret_cast<uint64_t>(prev_progress_ptr);
    out->next_done_bytes = progress_bytes;

    out->chunk_states_ptr = reinterpret_cast<uint64_t>(chunk_states_ptr);

    return operation_desc_is_valid(out);
}

bool operation_desc_reset_local_state(
    int device,
    const OperationDesc* desc) {
    if (device < 0) {
        throw std::invalid_argument("operation_desc_reset_local_state: invalid device");
    }
    if (!operation_desc_is_valid(desc)) {
        throw std::invalid_argument("operation_desc_reset_local_state: invalid desc");
    }

    const uint32_t total_steps = operation_desc_total_ring_steps(desc);
    const size_t progress_bytes = operation_desc_progress_storage_bytes(desc);

    system::runtime::set_device(device);

    if (desc->ready_queue_ptr != 0 && desc->ready_queue_bytes != 0) {
        system::runtime::check_cuda(
            cudaMemset(
                reinterpret_cast<void*>(desc->ready_queue_ptr),
                0,
                desc->ready_queue_bytes),
            "cudaMemset(operation ready_queue reset)");
    }

    system::runtime::check_cuda(
        cudaMemset(
            reinterpret_cast<void*>(desc->done_ptr),
            0,
            progress_bytes),
        "cudaMemset(operation progress reset)");

    system::runtime::check_cuda(
        cudaMemset(
            reinterpret_cast<void*>(desc->chunk_states_ptr),
            0,
            static_cast<size_t>(desc->num_chunks) * sizeof(ChunkState)),
        "cudaMemset(operation chunk_states reset)");

    uint32_t host_completion_count = 0u;
    uint32_t host_completion_flag = 0u;

    if (total_steps == 0u) {
        host_completion_count = desc->completion_target;
        host_completion_flag = (desc->completion_target > 0u) ? 1u : 0u;
    }

    if (desc->completion_count_ptr != 0) {
        system::runtime::check_cuda(
            cudaMemcpy(
                reinterpret_cast<void*>(desc->completion_count_ptr),
                &host_completion_count,
                sizeof(uint32_t),
                cudaMemcpyHostToDevice),
            "cudaMemcpy(operation completion count reset)");
    }

    if (desc->completion_flag_ptr != 0) {
        system::runtime::check_cuda(
            cudaMemcpy(
                reinterpret_cast<void*>(desc->completion_flag_ptr),
                &host_completion_flag,
                sizeof(uint32_t),
                cudaMemcpyHostToDevice),
            "cudaMemcpy(operation completion flag reset)");
    }

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
