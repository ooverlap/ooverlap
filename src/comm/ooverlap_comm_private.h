#pragma once

#include "comm/launch_config.h"
#include "comm/ooverlap_comm_internal.h"
#include "ooverlap/comm.h"

#include <cuda_runtime.h>

#include <cstddef>

namespace ooverlap {
namespace comm {
namespace api {

constexpr int kMaxPublicPeers = kOoMaxLocalDevices - 1;

struct CollectiveLaunchState {
    void* local_ptr = nullptr;

    /*
     * OOVERLAP_OUT_OF_PLACE_ALLREDUCE_REDUCE_FANOUT_PATCH
     *
     * Optional out-of-place binding. Existing in-place public APIs leave
     * out_of_place=false and use local_ptr/peer_ptrs for all logical roles.
     */
    bool out_of_place = false;
    void* local_input_ptr = nullptr;
    void* local_output_ptr = nullptr;

    void* peer_input_ptrs[kMaxPublicPeers] = {};
    void* peer_output_ptrs[kMaxPublicPeers] = {};

    void* peer_ptrs[kMaxPublicPeers] = {};
    int peer_ranks[kMaxPublicPeers] = {};
    int peer_devices[kMaxPublicPeers] = {};
    int peer_count = 0;

    /*
     * Legacy single-channel fields.  These mirror the DeviceMemory channel.
     */
    int* local_ready_signal = nullptr;
    const int* peer_ready_signals[kMaxPublicPeers] = {};

    /*
     * Channel-aware ready pointers/protocols.  The planner chooses a logical
     * ReadySignalChannel per edge; lowering resolves that channel through these
     * arrays.
     */
    int* local_ready_signal_by_channel[kOoReadySignalChannelCount] = {};
    const int* peer_ready_signals_by_channel
        [kMaxPublicPeers][kOoReadySignalChannelCount] = {};
    int ready_signal_protocol_by_channel[kOoReadySignalChannelCount] = {};
    int ready_signal_poll_sleep_cycles_by_channel[kOoReadySignalChannelCount] = {};

    /*
     * Group-owned staging slots, already converted to device-visible pointers.
     * These are optional until planners start emitting ShmStaging references.
     */
    void* staging_ptrs[kOoMaxStagingSlots] = {};
    size_t staging_bytes[kOoMaxStagingSlots] = {};
    int staging_numa_nodes[kOoMaxStagingSlots] = {};
    int staging_slot_count = 0;

    int rank = -1;
    int world_size = 0;
    int local_device = -1;
    int collective_epoch = 0;

    size_t dtype_size = 0;
    size_t bytes = 0;
};

oo_status_t exception_to_status();

oo_status_t cuda_to_status(cudaError_t error);

oo_status_t checked_element_bytes(
    size_t count,
    oo_dtype_t dtype,
    size_t* out_bytes);

oo_status_t checked_element_offset_bytes(
    size_t element_offset,
    oo_dtype_t dtype,
    size_t* out_offset_bytes);

oo_status_t fill_rank_partition(
    int rank,
    int world_size,
    size_t count,
    size_t* out_element_offset,
    size_t* out_count);

oo_status_t fill_tensor_slice(
    oo_buffer_t* local,
    int rank,
    int world_size,
    size_t base_element_offset,
    size_t count,
    oo_dtype_t dtype,
    oo_tensor_slice_t* out_slice);

LaunchConfig select_public_launch_config(
    CollectivePlanFor collective,
    size_t bytes,
    oo_tuning_mode_t tuning_mode);

/*
 * Optional explicit IPC buffer registration.
 * This fills the same import cache used by prepare_collective_launch().
 */
oo_status_t register_ipc_collective_buffers(
    oo_node_t* node,
    oo_buffer_t* local);

/*
 * Build a rank-local collective launch state from the group-owned current
 * collective buffer registry.
 *
 * Public collective APIs pass only this rank's local buffer.  This helper
 * refreshes group->collective_buffers[node->rank] with local, reads the other
 * rank buffers from group->collective_buffers[], and returns immediately.
 *
 * No host-side rank barrier happens here.  The GPU ready-signal protocol still
 * provides the kernel-side rendezvous.
 */
oo_status_t prepare_collective_launch(
    oo_node_t* node,
    oo_buffer_t* local,
    CollectivePlanFor collective,
    size_t element_offset,
    size_t count,
    oo_dtype_t dtype,
    CollectiveLaunchState* out);


/* OOVERLAP_ROUND_ROBIN_SLOT_POOL_PATCH_V1: skip IPC discovery and use an explicit rank-indexed buffer row. */
oo_status_t prepare_collective_launch_prebound(
    oo_node_t* node,
    oo_buffer_t* const* rank_buffers,
    int rank_buffer_count,
    CollectivePlanFor collective,
    size_t element_offset,
    size_t count,
    oo_dtype_t dtype,
    CollectiveLaunchState* out);

} // namespace api
} // namespace comm
} // namespace ooverlap
