#pragma once

#include "comm/buffer.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace ooverlap {
namespace comm {

struct CommSlot {
    CommBuffer buffer;
    CommBuffer signal_buffer;   // uint64_t flag/sequence storage, peer-visible
    uint64_t seq = 0;
};

struct CommChannel {
    int src_rank = -1;
    int dst_rank = -1;
    size_t slot_bytes = 0;
    int num_slots = 0;
    std::vector<CommSlot> slots;
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
    CommChannel* ch,
    int slot_idx);

const CommBuffer* channel_get_slot_buffer(
    const CommChannel* ch,
    int slot_idx);

CommBuffer* channel_get_slot_signal_buffer(
    CommChannel* ch,
    int slot_idx);

const CommBuffer* channel_get_slot_signal_buffer(
    const CommChannel* ch,
    int slot_idx);

// Compatibility aliases for current code.
using ChannelSlot = CommSlot;
using Channel = CommChannel;

} // namespace comm
} // namespace ooverlap
