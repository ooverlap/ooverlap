#pragma once

#include <cstddef>
#include <vector>

#include "vmm.cuh"

namespace ooverlap {
namespace system {

struct mapped_peer_buffer {
    void* ptr = nullptr;
    size_t mapped_size = 0;
};

inline mapped_peer_buffer alloc_peer_visible_buffer(
    size_t bytes,
    int owner_device,
    const std::vector<int>& access_devices) {
    mapped_peer_buffer out{};
    vmm::vm_alloc_map_set_access(
        &out.ptr,
        &out.mapped_size,
        bytes,
        owner_device,
        access_devices);
    return out;
}

inline void free_peer_visible_buffer(mapped_peer_buffer& buf) {
    if (buf.ptr != nullptr && buf.mapped_size != 0) {
        vmm::vm_unmap(buf.ptr, buf.mapped_size);
        buf.ptr = nullptr;
        buf.mapped_size = 0;
    }
}

} // namespace system
} // namespace ooverlap
