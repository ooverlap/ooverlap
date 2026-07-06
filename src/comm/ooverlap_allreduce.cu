#include "comm/ooverlap_comm_private.h"

#include "comm/tma_multi_gpu_allreduce_sm90.h"
#include "comm/plan/transfer_plan_distribution.h"
#include "ooverlap/comm.h"

namespace {

oo_status_t allreduce_impl(
    oo_node_t* node,
    oo_buffer_t* local,
    oo_buffer_t* const* peers,
    int peer_count,
    size_t element_offset,
    size_t count,
    oo_dtype_t dtype,
    oo_reduce_op_t op,
    oo_tuning_mode_t tuning_mode,
    cudaStream_t stream) {
    if (!oo_allreduce_supported(dtype, op)) {
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

    return OO_SUCCESS;

    ooverlap::comm::LaunchConfig config =
        ooverlap::comm::api::select_public_launch_config(
            ooverlap::comm::CollectivePlanFor::AllReduce,
            launch.bytes,
            tuning_mode);
    /*
     * Logical task-generation/distribution layer.
     *
     * For same-process groups, the backend lets the first arriving rank build
     * the TransferPlan and makes the other ranks wait until it is ready.
     *
     * For IPC groups, the backend will later make rank 0 build and broadcast
     * the TransferPlan.
     *
     * The returned transfer_plan is a rank-local copy. It contains no raw
     * pointers. Raw pointer lowering happens in the SM90 enqueue layer.
     */
    ooverlap::comm::plan::AllreduceTransferPlan transfer_plan{};

    if (node == nullptr ||
        node->group == nullptr ||
        node->group->transfer_plan_distribution == nullptr) {
        return OO_ERROR_INTERNAL;
    }

    status =
        node->group->transfer_plan_distribution->get_allreduce_transfer_plan(
            node,
            launch,
            count,
            dtype,
            op,
            config,
            &transfer_plan);

    if (status != OO_SUCCESS) {
        return status;
    }

    /*
     * Temporary fallback.
     *
     * This still uses the old raw-pointer plan builder inside
     * tma_multi_gpu_allreduce_sm90.cu. The next patch should replace this call
     * with a transfer-plan enqueue function that lowers transfer_plan into a
     * WindowTaskExecutorPlan for this rank.
     */

     cudaError_t error =
        ooverlap::enqueue_tma_multi_gpu_allreduce_rank_sm90(
            launch,
            dtype,
            op,
            stream,
            config,
            transfer_plan);
     
    return ooverlap::comm::api::cuda_to_status(error);
}

} // namespace

extern "C" oo_status_t oo_allreduce_offset(
    oo_node_t* node,
    oo_buffer_t* local,
    oo_buffer_t* const* peers,
    int peer_count,
    size_t element_offset,
    size_t count,
    oo_dtype_t dtype,
    oo_reduce_op_t op,
    cudaStream_t stream) {
    return allreduce_impl(
        node,
        local,
        peers,
        peer_count,
        element_offset,
        count,
        dtype,
        op,
        OO_TUNING_BEST_PERFORMANCE,
        stream);
}

extern "C" oo_status_t oo_allreduce(
    oo_node_t* node,
    oo_buffer_t* local,
    oo_buffer_t* const* peers,
    int peer_count,
    size_t count,
    oo_dtype_t dtype,
    oo_reduce_op_t op,
    cudaStream_t stream) {
    return oo_allreduce_offset(
        node,
        local,
        peers,
        peer_count,
        0,
        count,
        dtype,
        op,
        stream);
}

extern "C" oo_status_t oo_allreduce_offset_tuned(
    oo_node_t* node,
    oo_buffer_t* local,
    oo_buffer_t* const* peers,
    int peer_count,
    size_t element_offset,
    size_t count,
    oo_dtype_t dtype,
    oo_reduce_op_t op,
    oo_tuning_mode_t tuning_mode,
    cudaStream_t stream) {
    return allreduce_impl(
        node,
        local,
        peers,
        peer_count,
        element_offset,
        count,
        dtype,
        op,
        tuning_mode,
        stream);
}

extern "C" oo_status_t oo_allreduce_tuned(
    oo_node_t* node,
    oo_buffer_t* local,
    oo_buffer_t* const* peers,
    int peer_count,
    size_t count,
    oo_dtype_t dtype,
    oo_reduce_op_t op,
    oo_tuning_mode_t tuning_mode,
    cudaStream_t stream) {
    return oo_allreduce_offset_tuned(
        node,
        local,
        peers,
        peer_count,
        0,
        count,
        dtype,
        op,
        tuning_mode,
        stream);
}
