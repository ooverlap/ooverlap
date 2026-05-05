#include "comm/ooverlap_comm_private.h"

#include "comm/tma_multi_gpu_all_gather_sm90.h"

namespace {

oo_status_t all_gather_impl(
    oo_node_t* node,
    oo_buffer_t* local,
    oo_buffer_t* const* peers,
    int peer_count,
    size_t element_offset,
    size_t count,
    oo_dtype_t dtype,
    oo_tuning_mode_t tuning_mode,
    cudaStream_t stream) {
    if (!oo_all_gather_supported(dtype)) {
        return OO_ERROR_UNSUPPORTED;
    }

    ooverlap::comm::api::CollectiveLaunchState launch{};

    oo_status_t status =
        ooverlap::comm::api::prepare_collective_launch(
            node,
            local,
            peers,
            peer_count,
            element_offset,
            count,
            dtype,
            &launch);

    if (status != OO_SUCCESS) {
        return status;
    }

    ooverlap::comm::LaunchConfig config =
        ooverlap::comm::api::select_public_launch_config(
            ooverlap::comm::CollectivePlanFor::AllGather,
            launch.bytes,
            tuning_mode);

    cudaError_t error =
        ooverlap::enqueue_tma_multi_gpu_all_gather_rank_sm90(
            launch.local_ptr,
            launch.local_ptr,
            launch.peer_ptrs,
            launch.peer_count,
            count,
            dtype,
            launch.rank,
            launch.world_size,
            launch.local_device,
            stream,
            launch.local_ready_signal,
            launch.peer_ready_signals,
            launch.collective_epoch,
            config);

    return ooverlap::comm::api::cuda_to_status(error);
}

} // namespace

extern "C" oo_status_t oo_all_gather_offset(
    oo_node_t* node,
    oo_buffer_t* local,
    oo_buffer_t* const* peers,
    int peer_count,
    size_t element_offset,
    size_t count,
    oo_dtype_t dtype,
    cudaStream_t stream) {
    return all_gather_impl(
        node,
        local,
        peers,
        peer_count,
        element_offset,
        count,
        dtype,
        OO_TUNING_BEST_PERFORMANCE,
        stream);
}

extern "C" oo_status_t oo_all_gather(
    oo_node_t* node,
    oo_buffer_t* local,
    oo_buffer_t* const* peers,
    int peer_count,
    size_t count,
    oo_dtype_t dtype,
    cudaStream_t stream) {
    return oo_all_gather_offset(
        node,
        local,
        peers,
        peer_count,
        0,
        count,
        dtype,
        stream);
}

extern "C" oo_status_t oo_all_gather_offset_tuned(
    oo_node_t* node,
    oo_buffer_t* local,
    oo_buffer_t* const* peers,
    int peer_count,
    size_t element_offset,
    size_t count,
    oo_dtype_t dtype,
    oo_tuning_mode_t tuning_mode,
    cudaStream_t stream) {
    return all_gather_impl(
        node,
        local,
        peers,
        peer_count,
        element_offset,
        count,
        dtype,
        tuning_mode,
        stream);
}

extern "C" oo_status_t oo_all_gather_tuned(
    oo_node_t* node,
    oo_buffer_t* local,
    oo_buffer_t* const* peers,
    int peer_count,
    size_t count,
    oo_dtype_t dtype,
    oo_tuning_mode_t tuning_mode,
    cudaStream_t stream) {
    return oo_all_gather_offset_tuned(
        node,
        local,
        peers,
        peer_count,
        0,
        count,
        dtype,
        tuning_mode,
        stream);
}
