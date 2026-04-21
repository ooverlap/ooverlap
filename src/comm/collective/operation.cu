#include "comm/collective/operation.h"

#include "ooverlap/system/runtime_utils.cuh"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace ooverlap {
namespace comm {
namespace collective {

bool operation_table_init(
    OperationTable* table,
    int device,
    uint32_t capacity) {
    if (table == nullptr) {
        throw std::invalid_argument("operation_table_init: table is null");
    }
    if (device < 0) {
        throw std::invalid_argument("operation_table_init: invalid device");
    }
    if (capacity == 0) {
        throw std::invalid_argument("operation_table_init: capacity must be > 0");
    }

    operation_table_destroy(table);

    table->capacity = capacity;
    table->device = device;

    system::runtime::set_device(device);
    system::runtime::check_cuda(
        cudaMalloc(&table->records, static_cast<size_t>(capacity) * sizeof(OperationDesc)),
        "cudaMalloc(operation table records)");

    operation_table_reset(table);
    return true;
}

void operation_table_reset(
    OperationTable* table) {
    if (!operation_table_is_configured(table)) {
        throw std::invalid_argument("operation_table_reset: table not configured");
    }

    system::runtime::set_device(table->device);
    system::runtime::check_cuda(
        cudaMemset(
            table->records,
            0,
            static_cast<size_t>(table->capacity) * sizeof(OperationDesc)),
        "cudaMemset(operation table records)");
}

void operation_table_destroy(
    OperationTable* table) {
    if (table == nullptr) {
        return;
    }

    if (table->device >= 0) {
        system::runtime::set_device(table->device);
    }

    if (table->records != nullptr) {
        system::runtime::check_cuda(
            cudaFree(table->records),
            "cudaFree(operation table records)");
    }

    table->records = nullptr;
    table->capacity = 0;
    table->device = -1;
}

bool operation_table_write(
    OperationTable* table,
    uint32_t idx,
    const OperationDesc* desc) {
    if (!operation_table_is_configured(table)) {
        throw std::invalid_argument("operation_table_write: table not configured");
    }
    if (desc == nullptr) {
        throw std::invalid_argument("operation_table_write: desc is null");
    }
    if (idx >= table->capacity) {
        throw std::invalid_argument("operation_table_write: idx out of range");
    }
    if (!operation_desc_is_valid(desc)) {
        throw std::invalid_argument("operation_table_write: desc is invalid");
    }

    system::runtime::set_device(table->device);
    system::runtime::check_cuda(
        cudaMemcpy(
            &table->records[idx],
            desc,
            sizeof(OperationDesc),
            cudaMemcpyHostToDevice),
        "cudaMemcpy(operation table write)");

    return true;
}

bool operation_table_clear_record(
    OperationTable* table,
    uint32_t idx) {
    if (!operation_table_is_configured(table)) {
        throw std::invalid_argument("operation_table_clear_record: table not configured");
    }
    if (idx >= table->capacity) {
        throw std::invalid_argument("operation_table_clear_record: idx out of range");
    }

    OperationDesc cleared{};
    operation_desc_clear(&cleared);

    system::runtime::set_device(table->device);
    system::runtime::check_cuda(
        cudaMemcpy(
            &table->records[idx],
            &cleared,
            sizeof(OperationDesc),
            cudaMemcpyHostToDevice),
        "cudaMemcpy(operation table clear)");

    return true;
}

} // namespace collective
} // namespace comm
} // namespace ooverlap
