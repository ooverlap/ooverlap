#include "comm/buffer.h"

#include <stdexcept>

namespace ooverlap {
namespace comm {

void buffer_init(
    Buffer* buf,
    int owner_rank,
    int owner_device,
    size_t bytes,
    const std::vector<int>& visible_devices) {
    if (buf == nullptr) {
        throw std::invalid_argument("buffer_init: buffer is null");
    }
    if (bytes == 0) {
        throw std::invalid_argument("buffer_init: bytes must be > 0");
    }
    if (visible_devices.empty()) {
        throw std::invalid_argument("buffer_init: visible_devices must not be empty");
    }

    buffer_destroy(buf);

    buf->owner_rank = owner_rank;
    buf->owner_device = owner_device;
    buf->bytes = bytes;
    buf->mapped = system::alloc_peer_visible_buffer(
        bytes,
        owner_device,
        visible_devices);
}

void buffer_destroy(
    Buffer* buf) {
    if (buf == nullptr) {
        return;
    }

    system::free_peer_visible_buffer(buf->mapped);
    buf->owner_rank = -1;
    buf->owner_device = -1;
    buf->bytes = 0;
}

} // namespace comm
} // namespace ooverlap
