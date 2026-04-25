#pragma once

#include "ooverlap/comm.h"
#include "ooverlap/system/peer_buffer.cuh"

#include <cstddef>

constexpr int kOoMaxLocalDevices = 16;

struct oo_group {
    int num_devices = 0;
    int devices[kOoMaxLocalDevices] = {};

    // One peer-visible ready signal per rank/device.
    // ready_signals[r] is physically owned by devices[r], visible to all
    // devices in this group.
    ooverlap::system::mapped_peer_buffer ready_signals[kOoMaxLocalDevices] = {};
};

struct oo_node {
    oo_group_t* group = nullptr;
    int rank = -1;
    int device = -1;

    // Monotonic per-node collective sequence.
    // Rank-local calls must be issued in matching order, same as NCCL.
    int collective_epoch = 0;
};

struct oo_buffer {
    void* ptr = nullptr;
    size_t bytes = 0;
    size_t mapped_bytes = 0;
    oo_buffer_kind_t kind = OO_BUFFER_KIND_WRAPPED;

    // Internal validation/debug metadata. Public semantics should still treat
    // Buffer as pointer + size + kind.
    oo_group_t* group = nullptr;
    int owner_device = -1;

    // Valid only for OO_BUFFER_KIND_VMM.
    ooverlap::system::mapped_peer_buffer mapped{};
};
