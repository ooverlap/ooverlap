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

    void* peer_ptrs[kMaxPublicPeers] = {};
    int peer_count = 0;

    int* local_ready_signal = nullptr;
    const int* peer_ready_signals[kMaxPublicPeers] = {};

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
    size_t bytes,
    oo_tuning_mode_t tuning_mode);

oo_status_t prepare_collective_launch(
    oo_node_t* node,
    oo_buffer_t* local,
    oo_buffer_t* const* peers,
    int peer_count,
    size_t element_offset,
    size_t count,
    oo_dtype_t dtype,
    CollectiveLaunchState* out);

} // namespace api
} // namespace comm
} // namespace ooverlap
