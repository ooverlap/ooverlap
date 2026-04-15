#include "comm/buffer.h"

#include "ooverlap/system/runtime_utils.cuh"
#include "ooverlap/system/peer_buffer.cuh"

#include <cuda_runtime.h>

#include <stdexcept>
#include <vector>

namespace ooverlap {
namespace comm {

CommBuffer alloc_peer_visible_buffer_for_rank(
    const std::vector<int>& devices,
    int owner_rank,
    size_t bytes) {

    if (owner_rank < 0 || owner_rank >= static_cast<int>(devices.size())) {
        throw std::invalid_argument("alloc_peer_visible_buffer_for_rank: invalid owner_rank");
    }

    system::mapped_peer_buffer mapped =
        system::alloc_peer_visible_buffer(bytes, devices[owner_rank], devices);

    CommBuffer out{};
    out.ptr = mapped.ptr;
    out.bytes = mapped.mapped_size;
    out.owner_rank = owner_rank;
    out.peer_visible = true;
    return out;
}

CommBuffer alloc_local_buffer_for_rank(
    const std::vector<int>& devices,
    int owner_rank,
    size_t bytes) {

    if (owner_rank < 0 || owner_rank >= static_cast<int>(devices.size())) {
        throw std::invalid_argument("alloc_local_buffer_for_rank: invalid owner_rank");
    }

    CommBuffer out{};
    out.bytes = bytes;
    out.owner_rank = owner_rank;
    out.peer_visible = false;

    system::runtime::set_device(devices[owner_rank]);
    system::runtime::check_cuda(cudaMalloc(&out.ptr, bytes), "cudaMalloc(local buffer)");
    return out;
}

void free_comm_buffer(
    const std::vector<int>& devices,
    CommBuffer& buf) {

    if (buf.ptr == nullptr) {
        return;
    }

    if (buf.peer_visible) {
        system::mapped_peer_buffer mapped{};
        mapped.ptr = buf.ptr;
        mapped.mapped_size = buf.bytes;
        system::free_peer_visible_buffer(mapped);
    } else {
        if (buf.owner_rank < 0 || buf.owner_rank >= static_cast<int>(devices.size())) {
            throw std::invalid_argument("free_comm_buffer: invalid owner_rank");
        }
        system::runtime::set_device(devices[buf.owner_rank]);
        system::runtime::check_cuda(cudaFree(buf.ptr), "cudaFree(local buffer)");
    }

    buf.ptr = nullptr;
    buf.bytes = 0;
    buf.owner_rank = -1;
    buf.peer_visible = false;
}

} // namespace comm
} // namespace ooverlap
