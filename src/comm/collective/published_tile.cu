#include "comm/collective/published_tile.h"

#include "ooverlap/system/runtime_utils.cuh"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace ooverlap {
namespace comm {
namespace collective {

bool ready_tile_queue_init(
    ReadyTileQueue* q,
    int device,
    uint32_t capacity) {
    if (q == nullptr) {
        throw std::invalid_argument("ready_tile_queue_init: q is null");
    }
    if (device < 0) {
        throw std::invalid_argument("ready_tile_queue_init: invalid device");
    }
    if (capacity == 0) {
        throw std::invalid_argument("ready_tile_queue_init: capacity must be > 0");
    }

    ready_tile_queue_destroy(q);

    q->capacity = capacity;
    q->device = device;

    system::runtime::set_device(device);

    system::runtime::check_cuda(
        cudaMalloc(&q->records, static_cast<size_t>(capacity) * sizeof(ReadyTile)),
        "cudaMalloc(ready tile queue records)");
    system::runtime::check_cuda(
        cudaMalloc(&q->head, sizeof(uint64_t)),
        "cudaMalloc(ready tile queue head)");
    system::runtime::check_cuda(
        cudaMalloc(&q->tail, sizeof(uint64_t)),
        "cudaMalloc(ready tile queue tail)");

    ready_tile_queue_reset(q);
    return true;
}

void ready_tile_queue_reset(
    ReadyTileQueue* q) {
    if (!ready_tile_queue_is_configured(q)) {
        throw std::invalid_argument("ready_tile_queue_reset: queue not configured");
    }

    system::runtime::set_device(q->device);

    system::runtime::check_cuda(
        cudaMemset(q->records, 0, static_cast<size_t>(q->capacity) * sizeof(ReadyTile)),
        "cudaMemset(ready tile queue records)");
    system::runtime::check_cuda(
        cudaMemset(q->head, 0, sizeof(uint64_t)),
        "cudaMemset(ready tile queue head)");
    system::runtime::check_cuda(
        cudaMemset(q->tail, 0, sizeof(uint64_t)),
        "cudaMemset(ready tile queue tail)");
}

void ready_tile_queue_destroy(
    ReadyTileQueue* q) {
    if (q == nullptr) {
        return;
    }

    if (q->device >= 0) {
        system::runtime::set_device(q->device);
    }

    if (q->records != nullptr) {
        system::runtime::check_cuda(
            cudaFree(q->records),
            "cudaFree(ready tile queue records)");
    }
    if (q->head != nullptr) {
        system::runtime::check_cuda(
            cudaFree(q->head),
            "cudaFree(ready tile queue head)");
    }
    if (q->tail != nullptr) {
        system::runtime::check_cuda(
            cudaFree(q->tail),
            "cudaFree(ready tile queue tail)");
    }

    q->records = nullptr;
    q->head = nullptr;
    q->tail = nullptr;
    q->capacity = 0;
    q->device = -1;
}

} // namespace collective
} // namespace comm
} // namespace ooverlap
