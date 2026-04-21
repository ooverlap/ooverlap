#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

#include "comm/exec/chunk.h"

namespace ooverlap {
namespace comm {
namespace collective {

enum : uint32_t {
    kOperationFlagEnabled = 1u << 0,
};

struct OperationDesc {
    uint32_t op_id = 0;
    uint32_t queue_id = 0;

    uint32_t epoch = 0;
    uint32_t flags = 0;

    int src_rank = -1;
    int dst_rank = -1;

    // Final destination / accumulation region for this operation.
    uint64_t dst_ptr = 0;
    size_t dst_bytes = 0;

    // For reduce-scatter style completion tracking later.
    uint32_t expected_contributions = 0;
    uint32_t reserved0 = 0;

    exec::ChunkOpKind op = exec::ChunkOpKind::kInvalid;
};

struct OperationTable {
    OperationDesc* records = nullptr;  // device pointer
    uint32_t capacity = 0;
    int device = -1;
};

__host__ __device__ __forceinline__ void operation_desc_clear(
    OperationDesc* op) {
    op->op_id = 0;
    op->queue_id = 0;
    op->epoch = 0;
    op->flags = 0;
    op->src_rank = -1;
    op->dst_rank = -1;
    op->dst_ptr = 0;
    op->dst_bytes = 0;
    op->expected_contributions = 0;
    op->reserved0 = 0;
    op->op = exec::ChunkOpKind::kInvalid;
}

__host__ __device__ __forceinline__ bool operation_desc_is_enabled(
    const OperationDesc* op) {
    return op != nullptr &&
           (op->flags & kOperationFlagEnabled) != 0u;
}

__host__ __device__ __forceinline__ bool operation_desc_is_valid(
    const OperationDesc* op) {
    return op != nullptr &&
           op->op_id != 0 &&
           op->src_rank >= 0 &&
           op->dst_rank >= 0 &&
           op->dst_ptr != 0 &&
           op->dst_bytes > 0 &&
           op->expected_contributions > 0 &&
           op->op != exec::ChunkOpKind::kInvalid;
}

__host__ __device__ __forceinline__ bool operation_desc_is_active(
    const OperationDesc* op) {
    return operation_desc_is_valid(op) &&
           operation_desc_is_enabled(op);
}

__host__ __device__ __forceinline__ unsigned char* operation_desc_dst_base(
    const OperationDesc* op) {
    return reinterpret_cast<unsigned char*>(op->dst_ptr);
}

__host__ __device__ __forceinline__ bool operation_table_is_configured(
    const OperationTable* table) {
    return table != nullptr &&
           table->records != nullptr &&
           table->capacity > 0 &&
           table->device >= 0;
}

__host__ __device__ __forceinline__ OperationDesc* operation_table_records(
    OperationTable* table) {
    return table->records;
}

__host__ __device__ __forceinline__ const OperationDesc* operation_table_records(
    const OperationTable* table) {
    return table->records;
}

__host__ __device__ __forceinline__ OperationDesc* operation_table_record_ptr(
    OperationTable* table,
    uint32_t idx) {
    return (table != nullptr && idx < table->capacity)
        ? &table->records[idx]
        : nullptr;
}

__host__ __device__ __forceinline__ const OperationDesc* operation_table_record_ptr(
    const OperationTable* table,
    uint32_t idx) {
    return (table != nullptr && idx < table->capacity)
        ? &table->records[idx]
        : nullptr;
}

bool operation_table_init(
    OperationTable* table,
    int device,
    uint32_t capacity);

void operation_table_reset(
    OperationTable* table);

void operation_table_destroy(
    OperationTable* table);

bool operation_table_write(
    OperationTable* table,
    uint32_t idx,
    const OperationDesc* desc);

bool operation_table_clear_record(
    OperationTable* table,
    uint32_t idx);

} // namespace collective
} // namespace comm
} // namespace ooverlap
