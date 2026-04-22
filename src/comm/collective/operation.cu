#include "comm/collective/operation.h"

#include "ooverlap/system/runtime_utils.cuh"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace ooverlap {
namespace comm {
namespace collective {

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
