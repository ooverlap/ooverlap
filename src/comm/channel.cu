#include "comm/channel.h"

#include <stdexcept>

namespace ooverlap {
namespace comm {

int channel_index(
    int world_size,
    int src_rank,
    int dst_rank) {
    return src_rank * world_size + dst_rank;
}

void validate_rank_or_throw(
    int world_size,
    int rank,
    const char* what) {
    if (rank < 0 || rank >= world_size) {
        throw std::invalid_argument(what);
    }
}

transport::CommBuffer* channel_get_slot_buffer(
    CommChannel* ch,
    int slot_idx) {
    if (ch == nullptr) {
        return nullptr;
    }
    if (slot_idx < 0 || slot_idx >= ch->num_slots) {
        throw std::invalid_argument("channel_get_slot_buffer: invalid slot_idx");
    }
    return &ch->slots[static_cast<size_t>(slot_idx)].buffer;
}

const transport::CommBuffer* channel_get_slot_buffer(
    const CommChannel* ch,
    int slot_idx) {
    if (ch == nullptr) {
        return nullptr;
    }
    if (slot_idx < 0 || slot_idx >= ch->num_slots) {
        throw std::invalid_argument("channel_get_slot_buffer: invalid slot_idx");
    }
    return &ch->slots[static_cast<size_t>(slot_idx)].buffer;
}

transport::CommBuffer* channel_get_slot_signal_buffer(
    CommChannel* ch,
    int slot_idx) {
    if (ch == nullptr) {
        return nullptr;
    }
    if (slot_idx < 0 || slot_idx >= ch->num_slots) {
        throw std::invalid_argument("channel_get_slot_signal_buffer: invalid slot_idx");
    }
    return &ch->slots[static_cast<size_t>(slot_idx)].signal_buffer;
}

const transport::CommBuffer* channel_get_slot_signal_buffer(
    const CommChannel* ch,
    int slot_idx) {
    if (ch == nullptr) {
        return nullptr;
    }
    if (slot_idx < 0 || slot_idx >= ch->num_slots) {
        throw std::invalid_argument("channel_get_slot_signal_buffer: invalid slot_idx");
    }
    return &ch->slots[static_cast<size_t>(slot_idx)].signal_buffer;
}

transport::DispatchQueue* channel_get_dispatch_queue(
    Channel* ch) {
    if (ch == nullptr) {
        return nullptr;
    }
    return &ch->dispatch_queue;
}

const transport::DispatchQueue* channel_get_dispatch_queue(
    const Channel* ch) {
    if (ch == nullptr) {
        return nullptr;
    }
    return &ch->dispatch_queue;
}

} // namespace comm
} // namespace ooverlap
