#pragma once

#include "ooverlap/system/peer_buffer.cuh"

#include <cstddef>
#include <vector>

namespace ooverlap {
namespace comm {

struct Buffer {
    int owner_rank = -1;
    int owner_device = -1;
    size_t bytes = 0;
    system::mapped_peer_buffer mapped{};
};

void buffer_init(
    Buffer* buf,
    int owner_rank,
    int owner_device,
    size_t bytes,
    const std::vector<int>& visible_devices);

void buffer_destroy(
    Buffer* buf);

inline void* buffer_ptr(Buffer* buf) {
    return (buf != nullptr) ? buf->mapped.ptr : nullptr;
}

inline const void* buffer_ptr(const Buffer* buf) {
    return (buf != nullptr) ? buf->mapped.ptr : nullptr;
}

} // namespace comm
} // namespace ooverlap
