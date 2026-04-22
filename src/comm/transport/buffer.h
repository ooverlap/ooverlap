#pragma once

#include <cuda_fp16.h>

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace ooverlap {
namespace comm {
namespace transport {

struct CommBuffer {
    // Default pointer. For peer-visible buffers this is the owner's device view.
    void* ptr = nullptr;

    // Device-visible pointer view per rank/device in the group.
    // For ranks that do not have access, the entry is nullptr.
    std::vector<void*> device_ptrs{};

    size_t bytes = 0;
    int owner_rank = -1;
    bool peer_visible = false;

    inline bool has_device_ptr_for_rank(size_t rank) const {
        return rank < device_ptrs.size() && device_ptrs[rank] != nullptr;
    }

    inline void* device_ptr_for_rank(size_t rank) const {
        if (rank >= device_ptrs.size()) {
            throw std::out_of_range("CommBuffer::device_ptr_for_rank: rank out of range");
        }
        if (device_ptrs[rank] == nullptr) {
            throw std::invalid_argument("CommBuffer::device_ptr_for_rank: pointer is null for this rank");
        }
        return device_ptrs[rank];
    }
};

CommBuffer alloc_peer_visible_buffer_for_rank(
    const std::vector<int>& devices,
    int owner_rank,
    size_t bytes);

// New: allocate owner-local VMM buffer but expose it only to a subset of ranks.
CommBuffer alloc_peer_visible_buffer_for_rank_with_access_ranks(
    const std::vector<int>& devices,
    int owner_rank,
    const std::vector<int>& access_ranks,
    size_t bytes);

CommBuffer alloc_local_buffer_for_rank(
    const std::vector<int>& devices,
    int owner_rank,
    size_t bytes);

void free_comm_buffer(
    const std::vector<int>& devices,
    CommBuffer& buf);

inline half* buffer_as_half(CommBuffer* buf) {
    return reinterpret_cast<half*>(buf->ptr);
}

inline const half* buffer_as_half(const CommBuffer* buf) {
    return reinterpret_cast<const half*>(buf->ptr);
}

inline uint64_t* buffer_as_u64(CommBuffer* buf) {
    return reinterpret_cast<uint64_t*>(buf->ptr);
}

inline const uint64_t* buffer_as_u64(const CommBuffer* buf) {
    return reinterpret_cast<const uint64_t*>(buf->ptr);
}

} // namespace transport
} // namespace comm
} // namespace ooverlap
