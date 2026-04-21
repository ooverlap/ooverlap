#pragma once

#include "comm/transport/buffer.h"

#include <cstddef>

namespace ooverlap {
namespace comm {

struct Channel {
    int src_rank = -1;
    int dst_rank = -1;
    int src_device = -1;
    int dst_device = -1;

    size_t bytes = 0;
    transport::CommBuffer buffer;
};

int channel_index(
    int world_size,
    int src_rank,
    int dst_rank);

void validate_rank_or_throw(
    int world_size,
    int rank,
    const char* what);

transport::CommBuffer* channel_get_buffer(
    Channel* ch);

const transport::CommBuffer* channel_get_buffer(
    const Channel* ch);

inline bool channel_is_configured(const Channel* ch) {
    return ch != nullptr &&
           ch->src_rank >= 0 &&
           ch->dst_rank >= 0 &&
           ch->src_device >= 0 &&
           ch->dst_device >= 0 &&
           ch->bytes > 0 &&
           ch->buffer.ptr != nullptr;
}

} // namespace comm
} // namespace ooverlap
