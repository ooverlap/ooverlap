#pragma once

#include "comm/transport/buffer.h"
#include "comm/transport/control_plane.h"
#include "comm/transport/dispatch_queue.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace ooverlap {
namespace comm {

enum class ChannelMode : uint8_t {
    kSlotQueue = 0,
    kDirectReduce = 1,
};

struct ChannelSlot {
    transport::CommBuffer buffer;
    transport::CommBuffer signal_buffer;
    transport::ChannelSlotControl control;
    uint64_t seq = 0;
    uint32_t slot_id = 0;
};

struct Channel {
    int src_rank = -1;
    int dst_rank = -1;
    int src_device = -1;
    int dst_device = -1;

    ChannelMode mode = ChannelMode::kSlotQueue;

    size_t slot_bytes = 0;
    int num_slots = 0;
    std::vector<ChannelSlot> slots;

    transport::DispatchQueue dispatch_queue;
    uint32_t dispatch_queue_capacity = 0;
    size_t dispatch_chunk_bytes = 0;

    transport::DirectReduceControlPlane direct_control;
};

int channel_index(
    int world_size,
    int src_rank,
    int dst_rank);

void validate_rank_or_throw(
    int world_size,
    int rank,
    const char* what);

transport::CommBuffer* channel_get_slot_buffer(
    Channel* ch,
    int slot_idx);

const transport::CommBuffer* channel_get_slot_buffer(
    const Channel* ch,
    int slot_idx);

transport::CommBuffer* channel_get_slot_signal_buffer(
    Channel* ch,
    int slot_idx);

const transport::CommBuffer* channel_get_slot_signal_buffer(
    const Channel* ch,
    int slot_idx);

transport::ChannelSlotControl* channel_get_slot_control(
    Channel* ch,
    int slot_idx);

const transport::ChannelSlotControl* channel_get_slot_control(
    const Channel* ch,
    int slot_idx);

transport::DispatchQueue* channel_get_dispatch_queue(
    Channel* ch);

const transport::DispatchQueue* channel_get_dispatch_queue(
    const Channel* ch);

transport::DirectReduceControlPlane* channel_get_direct_control(
    Channel* ch);

const transport::DirectReduceControlPlane* channel_get_direct_control(
    const Channel* ch);

inline bool channel_is_direct_reduce(const Channel* ch) {
    return ch != nullptr && ch->mode == ChannelMode::kDirectReduce;
}

using CommSlot = ChannelSlot;
using CommChannel = Channel;

} // namespace comm
} // namespace ooverlap
