#include "comm/transport/dispatch_queue.h"

#include "ooverlap/system/runtime_utils.cuh"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace ooverlap {
namespace comm {
namespace transport {
namespace {

uint64_t read_u64_from_device(
    const std::vector<int>& devices,
    int owner_rank,
    const void* ptr,
    const char* what) {
    uint64_t out = 0;
    system::runtime::set_device(devices[static_cast<size_t>(owner_rank)]);
    system::runtime::check_cuda(
        cudaMemcpy(&out, ptr, sizeof(uint64_t), cudaMemcpyDeviceToHost),
        what);
    return out;
}

void write_u64_to_device(
    const std::vector<int>& devices,
    int owner_rank,
    void* ptr,
    uint64_t value,
    const char* what) {
    system::runtime::set_device(devices[static_cast<size_t>(owner_rank)]);
    system::runtime::check_cuda(
        cudaMemcpy(ptr, &value, sizeof(uint64_t), cudaMemcpyHostToDevice),
        what);
}

void write_u32_to_device(
    const std::vector<int>& devices,
    int owner_rank,
    void* ptr,
    uint32_t value,
    const char* what) {
    system::runtime::set_device(devices[static_cast<size_t>(owner_rank)]);
    system::runtime::check_cuda(
        cudaMemcpy(ptr, &value, sizeof(uint32_t), cudaMemcpyHostToDevice),
        what);
}

void write_record_to_device(
    const std::vector<int>& devices,
    int owner_rank,
    void* ptr,
    const DispatchRecord* rec,
    const char* what) {
    system::runtime::set_device(devices[static_cast<size_t>(owner_rank)]);
    system::runtime::check_cuda(
        cudaMemcpy(ptr, rec, sizeof(DispatchRecord), cudaMemcpyHostToDevice),
        what);
}

} // namespace

bool dispatch_queue_init(
    const std::vector<int>& devices,
    DispatchQueue* q,
    int owner_rank,
    int src_rank,
    int dst_rank,
    uint32_t capacity) {

    if (q == nullptr) {
        throw std::invalid_argument("dispatch_queue_init: q is null");
    }
    if (owner_rank < 0 || owner_rank >= static_cast<int>(devices.size())) {
        throw std::invalid_argument("dispatch_queue_init: invalid owner_rank");
    }
    if (capacity == 0) {
        throw std::invalid_argument("dispatch_queue_init: capacity must be > 0");
    }

    dispatch_queue_destroy(devices, q);

    q->capacity = capacity;
    q->owner_rank = owner_rank;
    q->src_rank = src_rank;
    q->dst_rank = dst_rank;

    q->records_buffer = alloc_local_buffer_for_rank(
        devices,
        owner_rank,
        static_cast<size_t>(capacity) * sizeof(DispatchRecord));
    q->head_buffer = alloc_local_buffer_for_rank(devices, owner_rank, sizeof(uint64_t));
    q->tail_buffer = alloc_local_buffer_for_rank(devices, owner_rank, sizeof(uint64_t));
    q->overflow_buffer = alloc_local_buffer_for_rank(devices, owner_rank, sizeof(uint32_t));

    dispatch_queue_reset(devices, q);
    return true;
}

void dispatch_queue_destroy(
    const std::vector<int>& devices,
    DispatchQueue* q) {
    if (q == nullptr) {
        return;
    }

    free_comm_buffer(devices, q->records_buffer);
    free_comm_buffer(devices, q->head_buffer);
    free_comm_buffer(devices, q->tail_buffer);
    free_comm_buffer(devices, q->overflow_buffer);

    q->capacity = 0;
    q->owner_rank = -1;
    q->src_rank = -1;
    q->dst_rank = -1;
}

void dispatch_queue_reset(
    const std::vector<int>& devices,
    DispatchQueue* q) {
    if (!dispatch_queue_is_configured(q)) {
        throw std::invalid_argument("dispatch_queue_reset: queue not configured");
    }

    system::runtime::set_device(devices[static_cast<size_t>(q->owner_rank)]);
    system::runtime::check_cuda(
        cudaMemset(q->records_buffer.ptr, 0, q->records_buffer.bytes),
        "cudaMemset(dispatch records)");
    system::runtime::check_cuda(
        cudaMemset(q->head_buffer.ptr, 0, q->head_buffer.bytes),
        "cudaMemset(dispatch head)");
    system::runtime::check_cuda(
        cudaMemset(q->tail_buffer.ptr, 0, q->tail_buffer.bytes),
        "cudaMemset(dispatch tail)");
    system::runtime::check_cuda(
        cudaMemset(q->overflow_buffer.ptr, 0, q->overflow_buffer.bytes),
        "cudaMemset(dispatch overflow)");
}

bool dispatch_queue_push_host_blocking(
    const std::vector<int>& devices,
    DispatchQueue* q,
    const DispatchRecord* rec) {

    if (!dispatch_queue_is_configured(q)) {
        throw std::invalid_argument("dispatch_queue_push_host_blocking: queue not configured");
    }
    if (rec == nullptr) {
        throw std::invalid_argument("dispatch_queue_push_host_blocking: rec is null");
    }

    while (true) {
        const uint64_t head_ticket = read_u64_from_device(
            devices, q->owner_rank, q->head_buffer.ptr, "cudaMemcpy(dispatch head)");
        const uint64_t tail_ticket = read_u64_from_device(
            devices, q->owner_rank, q->tail_buffer.ptr, "cudaMemcpy(dispatch tail)");

        if ((tail_ticket - head_ticket) >= static_cast<uint64_t>(q->capacity)) {
            const uint32_t overflow = 1u;
            write_u32_to_device(
                devices,
                q->owner_rank,
                q->overflow_buffer.ptr,
                overflow,
                "cudaMemcpy(dispatch overflow)");
            continue;
        }

        DispatchRecord staged = *rec;
        staged.queue_ticket = tail_ticket;
        staged.flags |= kDispatchRecordFlagReady;

        DispatchRecord* records = dispatch_queue_records(q);
        DispatchRecord* dst =
            &records[static_cast<size_t>(tail_ticket % static_cast<uint64_t>(q->capacity))];

        write_record_to_device(
            devices,
            q->owner_rank,
            dst,
            &staged,
            "cudaMemcpy(dispatch record)");

        const uint64_t new_tail = tail_ticket + 1;
        write_u64_to_device(
            devices,
            q->owner_rank,
            q->tail_buffer.ptr,
            new_tail,
            "cudaMemcpy(dispatch tail update)");
        return true;
    }
}

DeviceDispatchQueueHandle dispatch_queue_get_device_handle(
    const DispatchQueue* q) {
    if (!dispatch_queue_is_configured(q)) {
        throw std::invalid_argument("dispatch_queue_get_device_handle: queue not configured");
    }

    DeviceDispatchQueueHandle out{};
    out.records = reinterpret_cast<DispatchRecord*>(q->records_buffer.ptr);
    out.head = reinterpret_cast<uint64_t*>(q->head_buffer.ptr);
    out.tail = reinterpret_cast<uint64_t*>(q->tail_buffer.ptr);
    out.overflow = reinterpret_cast<uint32_t*>(q->overflow_buffer.ptr);
    out.capacity = q->capacity;
    out.owner_rank = q->owner_rank;
    out.src_rank = q->src_rank;
    out.dst_rank = q->dst_rank;
    return out;
}

} // namespace transport
} // namespace comm
} // namespace ooverlap
