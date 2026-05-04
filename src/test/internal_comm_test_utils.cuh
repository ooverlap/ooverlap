#pragma once

#include "comm/ooverlap_comm_internal.h"
#include "ooverlap/system/runtime_utils.cuh"

#include <cuda_runtime.h>

#include <stdexcept>

namespace ooverlap {
namespace testing {

inline int* ready_signal_ptr(
    oo_group_t* group,
    int rank) {
    if (group == nullptr ||
        rank < 0 ||
        rank >= group->num_devices ||
        group->ready_signal_slots[rank].ptr == nullptr) {
        throw std::runtime_error("ready_signal_ptr: invalid ready signal");
    }

    return reinterpret_cast<int*>(group->ready_signal_slots[rank].ptr);
}

inline void reset_ready_signals(oo_group_t* group) {
    if (group == nullptr) {
        return;
    }

    for (int r = 0; r < group->num_devices; ++r) {
        oo_ready_signal& slot = group->ready_signal_slots[r];

        if (slot.ptr == nullptr || slot.owner_device < 0) {
            continue;
        }

        system::runtime::set_device(slot.owner_device);
        system::runtime::check_cuda(
            cudaMemset(slot.ptr, 0, sizeof(int)),
            "cudaMemset(ready signal)");
    }
}

inline void destroy_oo_buffer(oo_buffer_t*& buffer) {
    if (buffer == nullptr) {
        return;
    }

    oo_buffer_destroy(buffer);
    buffer = nullptr;
}

inline void destroy_oo_node(oo_node_t*& node) {
    if (node == nullptr) {
        return;
    }

    oo_node_destroy(node);
    node = nullptr;
}

inline void destroy_oo_group(oo_group_t*& group) {
    if (group == nullptr) {
        return;
    }

    oo_group_destroy(group);
    group = nullptr;
}

} // namespace testing
} // namespace ooverlap
