#include "comm/transport/control_plane.h"

#include "ooverlap/system/runtime_utils.cuh"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace ooverlap {
namespace comm {
namespace transport {
namespace {

void memset_owner(
    const std::vector<int>& devices,
    int owner_rank,
    void* ptr,
    size_t bytes,
    const char* what) {
    system::runtime::set_device(devices[static_cast<size_t>(owner_rank)]);
    system::runtime::check_cuda(cudaMemset(ptr, 0, bytes), what);
}

void write_u32_owner(
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

} // namespace

bool channel_slot_control_init(
    const std::vector<int>& devices,
    ChannelSlotControl* ctrl,
    int owner_rank,
    int peer_rank,
    uint32_t slot_id) {
    if (ctrl == nullptr) {
        throw std::invalid_argument("channel_slot_control_init: ctrl is null");
    }
    if (owner_rank < 0 || owner_rank >= static_cast<int>(devices.size())) {
        throw std::invalid_argument("channel_slot_control_init: invalid owner_rank");
    }

    channel_slot_control_destroy(devices, ctrl);

    ctrl->owner_rank = owner_rank;
    ctrl->peer_rank = peer_rank;
    ctrl->slot_id = slot_id;

    ctrl->state_buffer = alloc_peer_visible_buffer_for_rank(devices, owner_rank, sizeof(uint32_t));
    ctrl->reserve_ticket_buffer = alloc_peer_visible_buffer_for_rank(devices, owner_rank, sizeof(uint64_t));
    ctrl->ready_ticket_buffer = alloc_peer_visible_buffer_for_rank(devices, owner_rank, sizeof(uint64_t));
    ctrl->consume_ticket_buffer = alloc_peer_visible_buffer_for_rank(devices, owner_rank, sizeof(uint64_t));
    ctrl->ack_ticket_buffer = alloc_peer_visible_buffer_for_rank(devices, owner_rank, sizeof(uint64_t));

    channel_slot_control_reset(devices, ctrl);
    return true;
}

void channel_slot_control_destroy(
    const std::vector<int>& devices,
    ChannelSlotControl* ctrl) {
    if (ctrl == nullptr) {
        return;
    }

    free_comm_buffer(devices, ctrl->state_buffer);
    free_comm_buffer(devices, ctrl->reserve_ticket_buffer);
    free_comm_buffer(devices, ctrl->ready_ticket_buffer);
    free_comm_buffer(devices, ctrl->consume_ticket_buffer);
    free_comm_buffer(devices, ctrl->ack_ticket_buffer);

    ctrl->owner_rank = -1;
    ctrl->peer_rank = -1;
    ctrl->slot_id = 0;
}

void channel_slot_control_reset(
    const std::vector<int>& devices,
    ChannelSlotControl* ctrl) {
    if (ctrl == nullptr || ctrl->owner_rank < 0) {
        throw std::invalid_argument("channel_slot_control_reset: ctrl not configured");
    }

    memset_owner(devices, ctrl->owner_rank, ctrl->state_buffer.ptr, ctrl->state_buffer.bytes,
                 "cudaMemset(slot control state)");
    memset_owner(devices, ctrl->owner_rank, ctrl->reserve_ticket_buffer.ptr, ctrl->reserve_ticket_buffer.bytes,
                 "cudaMemset(slot control reserve_ticket)");
    memset_owner(devices, ctrl->owner_rank, ctrl->ready_ticket_buffer.ptr, ctrl->ready_ticket_buffer.bytes,
                 "cudaMemset(slot control ready_ticket)");
    memset_owner(devices, ctrl->owner_rank, ctrl->consume_ticket_buffer.ptr, ctrl->consume_ticket_buffer.bytes,
                 "cudaMemset(slot control consume_ticket)");
    memset_owner(devices, ctrl->owner_rank, ctrl->ack_ticket_buffer.ptr, ctrl->ack_ticket_buffer.bytes,
                 "cudaMemset(slot control ack_ticket)");

    const uint32_t free_state = static_cast<uint32_t>(SlotLifecycleState::kFree);
    write_u32_owner(devices, ctrl->owner_rank, ctrl->state_buffer.ptr, free_state,
                    "cudaMemcpy(slot control state=free)");
}

DeviceChannelSlotControlHandle channel_slot_control_get_device_handle(
    const ChannelSlotControl* ctrl) {
    DeviceChannelSlotControlHandle out{};
    if (ctrl == nullptr) {
        return out;
    }

    out.state = reinterpret_cast<uint32_t*>(ctrl->state_buffer.ptr);
    out.reserve_ticket = reinterpret_cast<uint64_t*>(ctrl->reserve_ticket_buffer.ptr);
    out.ready_ticket = reinterpret_cast<uint64_t*>(ctrl->ready_ticket_buffer.ptr);
    out.consume_ticket = reinterpret_cast<uint64_t*>(ctrl->consume_ticket_buffer.ptr);
    out.ack_ticket = reinterpret_cast<uint64_t*>(ctrl->ack_ticket_buffer.ptr);
    out.owner_rank = ctrl->owner_rank;
    out.peer_rank = ctrl->peer_rank;
    out.slot_id = ctrl->slot_id;
    return out;
}

bool direct_reduce_control_init(
    const std::vector<int>& devices,
    DirectReduceControlPlane* ctrl,
    int owner_rank,
    int src_rank,
    int dst_rank) {
    if (ctrl == nullptr) {
        throw std::invalid_argument("direct_reduce_control_init: ctrl is null");
    }
    if (owner_rank < 0 || owner_rank >= static_cast<int>(devices.size())) {
        throw std::invalid_argument("direct_reduce_control_init: invalid owner_rank");
    }

    direct_reduce_control_destroy(devices, ctrl);

    ctrl->owner_rank = owner_rank;
    ctrl->src_rank = src_rank;
    ctrl->dst_rank = dst_rank;

    ctrl->ready_ticket_buffer = alloc_peer_visible_buffer_for_rank(devices, owner_rank, sizeof(uint64_t));
    ctrl->completion_ticket_buffer = alloc_peer_visible_buffer_for_rank(devices, owner_rank, sizeof(uint64_t));
    ctrl->window_owner_rank_buffer = alloc_peer_visible_buffer_for_rank(devices, owner_rank, sizeof(int32_t));
    ctrl->window_idx_buffer = alloc_peer_visible_buffer_for_rank(devices, owner_rank, sizeof(uint32_t));

    direct_reduce_control_reset(devices, ctrl);
    return true;
}

void direct_reduce_control_destroy(
    const std::vector<int>& devices,
    DirectReduceControlPlane* ctrl) {
    if (ctrl == nullptr) {
        return;
    }

    free_comm_buffer(devices, ctrl->ready_ticket_buffer);
    free_comm_buffer(devices, ctrl->completion_ticket_buffer);
    free_comm_buffer(devices, ctrl->window_owner_rank_buffer);
    free_comm_buffer(devices, ctrl->window_idx_buffer);

    ctrl->owner_rank = -1;
    ctrl->src_rank = -1;
    ctrl->dst_rank = -1;
}

void direct_reduce_control_reset(
    const std::vector<int>& devices,
    DirectReduceControlPlane* ctrl) {
    if (ctrl == nullptr || ctrl->owner_rank < 0) {
        throw std::invalid_argument("direct_reduce_control_reset: ctrl not configured");
    }

    memset_owner(devices, ctrl->owner_rank, ctrl->ready_ticket_buffer.ptr, ctrl->ready_ticket_buffer.bytes,
                 "cudaMemset(direct control ready_ticket)");
    memset_owner(devices, ctrl->owner_rank, ctrl->completion_ticket_buffer.ptr, ctrl->completion_ticket_buffer.bytes,
                 "cudaMemset(direct control completion_ticket)");
    memset_owner(devices, ctrl->owner_rank, ctrl->window_owner_rank_buffer.ptr, ctrl->window_owner_rank_buffer.bytes,
                 "cudaMemset(direct control window_owner_rank)");
    memset_owner(devices, ctrl->owner_rank, ctrl->window_idx_buffer.ptr, ctrl->window_idx_buffer.bytes,
                 "cudaMemset(direct control window_idx)");
}

DeviceDirectReduceControlHandle direct_reduce_control_get_device_handle(
    const DirectReduceControlPlane* ctrl) {
    DeviceDirectReduceControlHandle out{};
    if (ctrl == nullptr) {
        return out;
    }

    out.ready_ticket = reinterpret_cast<uint64_t*>(ctrl->ready_ticket_buffer.ptr);
    out.completion_ticket = reinterpret_cast<uint64_t*>(ctrl->completion_ticket_buffer.ptr);
    out.window_owner_rank = reinterpret_cast<int32_t*>(ctrl->window_owner_rank_buffer.ptr);
    out.window_idx = reinterpret_cast<uint32_t*>(ctrl->window_idx_buffer.ptr);
    out.owner_rank = ctrl->owner_rank;
    out.src_rank = ctrl->src_rank;
    out.dst_rank = ctrl->dst_rank;
    return out;
}

} // namespace transport
} // namespace comm
} // namespace ooverlap
