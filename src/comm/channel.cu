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

transport::CommBuffer* channel_get_buffer(
    Channel* ch) {
    if (ch == nullptr) {
        return nullptr;
    }
    return &ch->buffer;
}

const transport::CommBuffer* channel_get_buffer(
    const Channel* ch) {
    if (ch == nullptr) {
        return nullptr;
    }
    return &ch->buffer;
}

} // namespace comm
} // namespace ooverlap
