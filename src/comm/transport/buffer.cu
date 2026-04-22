#include "comm/transport/buffer.h"

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
    if (buf == nullptr) {
        return;
    }

    std::vector<void*> seen{};
    seen.reserve(buf->device_ptrs.size());

    for (void* p : buf->device_ptrs) {
        if (p == nullptr) {
            continue;
        }

        const bool already_seen =
            std::find(seen.begin(), seen.end(), p) != seen.end();
        if (already_seen) {
            continue;
        }

        system::vmm::vm_unmap(p, buf->bytes);
        seen.push_back(p);
    }
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

    system::vmm::handle handle{};
    size_t mapped_size = 0;

    system::vmm::vm_alloc(
        &handle,
        &mapped_size,
        bytes,
        devices[owner_rank]);

    CommBuffer out{};
    out.bytes = mapped_size;
    out.owner_rank = owner_rank;
    out.peer_visible = true;
    out.device_ptrs.resize(devices.size(), nullptr);

    try {
        for (size_t rank = 0; rank < devices.size(); ++rank) {
            system::runtime::set_device(devices[rank]);

            void* mapped_ptr = nullptr;
            system::vmm::vm_map(
                &mapped_ptr,
                handle,
                mapped_size);

            system::vmm::vm_set_access(
                mapped_ptr,
                mapped_size,
                devices);

            out.device_ptrs[rank] = mapped_ptr;
        }

        out.ptr = out.device_ptrs[static_cast<size_t>(owner_rank)];

        // After all mappings are established, releasing the handle is fine.
        system::vmm::vm_free(handle);
        return out;
    } catch (...) {
        try {
            free_peer_visible_mappings(&out);
        } catch (...) {
        }

        try {
            system::vmm::vm_free(handle);
        } catch (...) {
        }

        throw;
    }
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
