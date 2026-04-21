#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <vector>

#include "comm/transport/buffer.h"

namespace ooverlap {
namespace comm {
namespace transport {

enum class SlotLifecycleState : uint32_t {
    kFree = 0,
    kReserved = 1,
    kFilling = 2,
    kReady = 3,
    kConsumed = 4,
};

struct ChannelSlotControl {
    CommBuffer state_buffer;          // uint32_t SlotLifecycleState
    CommBuffer reserve_ticket_buffer; // uint64_t
    CommBuffer ready_ticket_buffer;   // uint64_t
    CommBuffer consume_ticket_buffer; // uint64_t
    CommBuffer ack_ticket_buffer;     // uint64_t

    int owner_rank = -1;
    int peer_rank = -1;
    uint32_t slot_id = 0;
};

struct DeviceChannelSlotControlHandle {
    uint32_t* state = nullptr;
    uint64_t* reserve_ticket = nullptr;
    uint64_t* ready_ticket = nullptr;
    uint64_t* consume_ticket = nullptr;
    uint64_t* ack_ticket = nullptr;

    int owner_rank = -1;
    int peer_rank = -1;
    uint32_t slot_id = 0;
};

struct DirectReduceControlPlane {
    CommBuffer ready_ticket_buffer;       // uint64_t
    CommBuffer completion_ticket_buffer;  // uint64_t
    CommBuffer window_owner_rank_buffer;  // int32_t
    CommBuffer window_idx_buffer;         // uint32_t

    int owner_rank = -1;
    int src_rank = -1;
    int dst_rank = -1;
};

struct DeviceDirectReduceControlHandle {
    uint64_t* ready_ticket = nullptr;
    uint64_t* completion_ticket = nullptr;
    int32_t* window_owner_rank = nullptr;
    uint32_t* window_idx = nullptr;

    int owner_rank = -1;
    int src_rank = -1;
    int dst_rank = -1;
};

bool channel_slot_control_init(
    const std::vector<int>& devices,
    ChannelSlotControl* ctrl,
    int owner_rank,
    int peer_rank,
    uint32_t slot_id);

void channel_slot_control_destroy(
    const std::vector<int>& devices,
    ChannelSlotControl* ctrl);

void channel_slot_control_reset(
    const std::vector<int>& devices,
    ChannelSlotControl* ctrl);

DeviceChannelSlotControlHandle channel_slot_control_get_device_handle(
    const ChannelSlotControl* ctrl);

bool direct_reduce_control_init(
    const std::vector<int>& devices,
    DirectReduceControlPlane* ctrl,
    int owner_rank,
    int src_rank,
    int dst_rank);

void direct_reduce_control_destroy(
    const std::vector<int>& devices,
    DirectReduceControlPlane* ctrl);

void direct_reduce_control_reset(
    const std::vector<int>& devices,
    DirectReduceControlPlane* ctrl);

DeviceDirectReduceControlHandle direct_reduce_control_get_device_handle(
    const DirectReduceControlPlane* ctrl);

__host__ __device__ __forceinline__ SlotLifecycleState device_slot_state(
    const DeviceChannelSlotControlHandle* ctrl) {
    if (ctrl == nullptr || ctrl->state == nullptr) {
        return SlotLifecycleState::kFree;
    }
    return static_cast<SlotLifecycleState>(*(ctrl->state));
}

__device__ __forceinline__ bool device_slot_try_reserve(
    DeviceChannelSlotControlHandle* ctrl,
    uint64_t reserve_ticket) {
    if (ctrl == nullptr || ctrl->state == nullptr || ctrl->reserve_ticket == nullptr) {
        return false;
    }

    const auto expected = static_cast<uint32_t>(SlotLifecycleState::kFree);
    const auto desired = static_cast<uint32_t>(SlotLifecycleState::kReserved);

    if (atomicCAS(reinterpret_cast<unsigned int*>(ctrl->state), expected, desired) != expected) {
        return false;
    }

    *(ctrl->reserve_ticket) = reserve_ticket;
    __threadfence_system();
    return true;
}

__device__ __forceinline__ void device_slot_mark_filling(
    DeviceChannelSlotControlHandle* ctrl) {
    if (ctrl == nullptr || ctrl->state == nullptr) {
        return;
    }
    atomicExch(reinterpret_cast<unsigned int*>(ctrl->state),
               static_cast<uint32_t>(SlotLifecycleState::kFilling));
}

__device__ __forceinline__ void device_slot_publish_ready(
    DeviceChannelSlotControlHandle* ctrl,
    uint64_t ready_ticket) {
    if (ctrl == nullptr || ctrl->state == nullptr || ctrl->ready_ticket == nullptr) {
        return;
    }
    *(ctrl->ready_ticket) = ready_ticket;
    __threadfence_system();
    atomicExch(reinterpret_cast<unsigned int*>(ctrl->state),
               static_cast<uint32_t>(SlotLifecycleState::kReady));
}

__device__ __forceinline__ bool device_slot_try_consume(
    DeviceChannelSlotControlHandle* ctrl,
    uint64_t* out_ready_ticket) {
    if (ctrl == nullptr || ctrl->state == nullptr || ctrl->ready_ticket == nullptr) {
        return false;
    }

    const auto expected = static_cast<uint32_t>(SlotLifecycleState::kReady);
    const auto desired = static_cast<uint32_t>(SlotLifecycleState::kConsumed);

    if (atomicCAS(reinterpret_cast<unsigned int*>(ctrl->state), expected, desired) != expected) {
        return false;
    }

    if (out_ready_ticket != nullptr) {
        *out_ready_ticket = *(ctrl->ready_ticket);
    }
    if (ctrl->consume_ticket != nullptr) {
        *(ctrl->consume_ticket) = *(ctrl->ready_ticket);
    }
    __threadfence_system();
    return true;
}

__device__ __forceinline__ void device_slot_ack_and_free(
    DeviceChannelSlotControlHandle* ctrl,
    uint64_t ack_ticket) {
    if (ctrl == nullptr || ctrl->state == nullptr || ctrl->ack_ticket == nullptr) {
        return;
    }

    *(ctrl->ack_ticket) = ack_ticket;
    __threadfence_system();
    atomicExch(reinterpret_cast<unsigned int*>(ctrl->state),
               static_cast<uint32_t>(SlotLifecycleState::kFree));
}

__device__ __forceinline__ void device_direct_reduce_publish(
    DeviceDirectReduceControlHandle* ctrl,
    uint64_t ready_ticket,
    int32_t window_owner_rank,
    uint32_t window_idx) {
    if (ctrl == nullptr ||
        ctrl->ready_ticket == nullptr ||
        ctrl->window_owner_rank == nullptr ||
        ctrl->window_idx == nullptr) {
        return;
    }

    *(ctrl->window_owner_rank) = window_owner_rank;
    *(ctrl->window_idx) = window_idx;
    __threadfence_system();
    *(ctrl->ready_ticket) = ready_ticket;
    __threadfence_system();
}

__device__ __forceinline__ void device_direct_reduce_mark_complete(
    DeviceDirectReduceControlHandle* ctrl,
    uint64_t completed_ticket) {
    if (ctrl == nullptr || ctrl->completion_ticket == nullptr) {
        return;
    }
    *(ctrl->completion_ticket) = completed_ticket;
    __threadfence_system();
}

} // namespace transport
} // namespace comm
} // namespace ooverlap
