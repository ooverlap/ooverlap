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

std::vector<int> normalize_access_ranks(
    int world_size,
    int owner_rank,
    const std::vector<int>& access_ranks) {
    if (world_size <= 0) {
        throw std::invalid_argument("normalize_access_ranks: invalid world_size");
    }
    if (owner_rank < 0 || owner_rank >= world_size) {
        throw std::invalid_argument("normalize_access_ranks: invalid owner_rank");
    }

    std::vector<int> out;
    out.reserve(access_ranks.size() + 1);

    auto push_unique = [&out, world_size](int r) {
        if (r < 0 || r >= world_size) {
            throw std::invalid_argument("normalize_access_ranks: invalid access rank");
        }
        if (std::find(out.begin(), out.end(), r) == out.end()) {
            out.push_back(r);
        }
    };

    push_unique(owner_rank);
    for (int r : access_ranks) {
        push_unique(r);
    }

    return out;
}

std::vector<int> ranks_to_devices(
    const std::vector<int>& devices,
    const std::vector<int>& ranks) {
    std::vector<int> out;
    out.reserve(ranks.size());
    for (int r : ranks) {
        out.push_back(devices[static_cast<size_t>(r)]);
    }
    return out;
}

} // namespace

CommBuffer alloc_peer_visible_buffer_for_rank(
    const std::vector<int>& devices,
    int owner_rank,
    size_t bytes) {

    std::vector<int> all_ranks(devices.size());
    for (size_t i = 0; i < devices.size(); ++i) {
        all_ranks[i] = static_cast<int>(i);
    }

    return alloc_peer_visible_buffer_for_rank_with_access_ranks(
        devices,
        owner_rank,
        all_ranks,
        bytes);
}

CommBuffer alloc_peer_visible_buffer_for_rank_with_access_ranks(
    const std::vector<int>& devices,
    int owner_rank,
    const std::vector<int>& access_ranks,
    size_t bytes) {

    if (owner_rank < 0 || owner_rank >= static_cast<int>(devices.size())) {
        throw std::invalid_argument("alloc_peer_visible_buffer_for_rank_with_access_ranks: invalid owner_rank");
    }
    if (bytes == 0) {
        throw std::invalid_argument("alloc_peer_visible_buffer_for_rank_with_access_ranks: bytes must be > 0");
    }

    const std::vector<int> normalized_ranks =
        normalize_access_ranks(static_cast<int>(devices.size()), owner_rank, access_ranks);
    const std::vector<int> access_devices =
        ranks_to_devices(devices, normalized_ranks);

    auto mapped =
        system::alloc_peer_visible_buffer(bytes, devices[owner_rank], access_devices);

    CommBuffer out{};
    out.ptr = mapped.ptr;
    out.bytes = mapped.mapped_size;
    out.owner_rank = owner_rank;
    out.peer_visible = true;
    out.device_ptrs.resize(devices.size(), nullptr);

    for (int r : normalized_ranks) {
        out.device_ptrs[static_cast<size_t>(r)] = mapped.ptr;
    }

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
