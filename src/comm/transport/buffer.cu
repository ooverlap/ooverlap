#include "comm/transport/buffer.h"

#include "ooverlap/system/peer_buffer.cuh"
#include "ooverlap/system/runtime_utils.cuh"
#include "ooverlap/system/vmm.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <stdexcept>
#include <vector>

namespace ooverlap {
namespace comm {
namespace transport {
namespace {

void free_peer_visible_mappings(
    CommBuffer* buf) {
    if (buf == nullptr || buf->ptr == nullptr || buf->bytes == 0) {
        return;
    }
    system::vmm::vm_unmap(buf->ptr, buf->bytes);
}

} // namespace

CommBuffer alloc_peer_visible_buffer_for_rank(
    const std::vector<int>& devices,
    int owner_rank,
    size_t bytes) {

    if (owner_rank < 0 || owner_rank >= static_cast<int>(devices.size())) {
        throw std::invalid_argument("alloc_peer_visible_buffer_for_rank: invalid owner_rank");
    }
    if (bytes == 0) {
        throw std::invalid_argument("alloc_peer_visible_buffer_for_rank: bytes must be > 0");
    }

    auto mapped =
        system::alloc_peer_visible_buffer(bytes, devices[owner_rank], devices);

    CommBuffer out{};
    out.ptr = mapped.ptr;
    out.bytes = mapped.mapped_size;
    out.owner_rank = owner_rank;
    out.peer_visible = true;
    out.device_ptrs.resize(devices.size(), mapped.ptr);
    return out;
}

CommBuffer alloc_local_buffer_for_rank(
    const std::vector<int>& devices,
    int owner_rank,
    size_t bytes) {

    if (owner_rank < 0 || owner_rank >= static_cast<int>(devices.size())) {
        throw std::invalid_argument("alloc_local_buffer_for_rank: invalid owner_rank");
    }
    if (bytes == 0) {
        throw std::invalid_argument("alloc_local_buffer_for_rank: bytes must be > 0");
    }

    CommBuffer out{};
    out.bytes = bytes;
    out.owner_rank = owner_rank;
    out.peer_visible = false;
    out.device_ptrs.resize(devices.size(), nullptr);

    system::runtime::set_device(devices[owner_rank]);
    system::runtime::check_cuda(cudaMalloc(&out.ptr, bytes), "cudaMalloc(local buffer)");

    out.device_ptrs[static_cast<size_t>(owner_rank)] = out.ptr;
    return out;
}

void free_comm_buffer(
    const std::vector<int>& devices,
    CommBuffer& buf) {

    if (buf.ptr == nullptr && buf.device_ptrs.empty()) {
        return;
    }

    if (buf.peer_visible) {
        free_peer_visible_mappings(&buf);
    } else {
        if (buf.owner_rank < 0 || buf.owner_rank >= static_cast<int>(devices.size())) {
            throw std::invalid_argument("free_comm_buffer: invalid owner_rank");
        }

        void* owner_ptr = nullptr;
        if (!buf.device_ptrs.empty()) {
            owner_ptr = buf.device_ptrs[static_cast<size_t>(buf.owner_rank)];
        } else {
            owner_ptr = buf.ptr;
        }

        if (owner_ptr != nullptr) {
            system::runtime::set_device(devices[buf.owner_rank]);
            system::runtime::check_cuda(cudaFree(owner_ptr), "cudaFree(local buffer)");
        }
    }

    buf.ptr = nullptr;
    buf.device_ptrs.clear();
    buf.bytes = 0;
    buf.owner_rank = -1;
    buf.peer_visible = false;
}

} // namespace transport
} // namespace comm
} // namespace ooverlap
