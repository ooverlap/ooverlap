#pragma once

#include "comm/buffer.h"

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
    CommBuffer buffer;
    CommBuffer signal_buffer;   // uint64_t flag/sequence storage, peer-visible
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
};

int channel_index(
    int world_size,
    int src_rank,
    int dst_rank);

void validate_rank_or_throw(
    int world_size,
    int rank,
    const char* what);

CommBuffer* channel_get_slot_buffer(
    Channel* ch,
    int slot_idx);

const CommBuffer* channel_get_slot_buffer(
    const Channel* ch,
    int slot_idx);

CommBuffer* channel_get_slot_signal_buffer(
    Channel* ch,
    int slot_idx);

const CommBuffer* channel_get_slot_signal_buffer(
    const Channel* ch,
    int slot_idx);

inline bool channel_is_direct_reduce(const Channel* ch) {
    return ch != nullptr && ch->mode == ChannelMode::kDirectReduce;
}

// Compatibility aliases for current code.
using CommSlot = ChannelSlot;
using CommChannel = Channel;

} // namespace comm
} // namespace ooverlap
