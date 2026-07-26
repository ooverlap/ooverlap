#include "comm/ooverlap_comm_private.h"

#include "comm/plan/transfer_plan_distribution.h"
#include "comm/tuning/tuning_policy.h"
#include "comm/tma_multi_gpu_reduce_scatter_sm90.h"

namespace {

oo_status_t reduce_scatter_impl(
    oo_node_t* node,
    oo_buffer_t* local,
    size_t element_offset,
    size_t count,
    oo_dtype_t dtype,
    oo_reduce_op_t op,
    oo_tuning_mode_t tuning_mode,
    oo_tensor_slice_t* out_slice,
    cudaStream_t stream) {
    if (!oo_reduce_scatter_supported(dtype, op)) {
        return OO_ERROR_UNSUPPORTED;
    }

    if (node == nullptr ||
        node->group == nullptr ||
        node->group->transfer_plan_distribution == nullptr) {
        return OO_ERROR_INVALID_ARGUMENT;
    }

    ooverlap::comm::api::CollectiveLaunchState launch{};

    oo_status_t status =
        ooverlap::comm::api::prepare_collective_launch(
            node,
            local,
            ooverlap::comm::CollectivePlanFor::ReduceScatter,
            element_offset,
            count,
            dtype,
            &launch);

    if (status != OO_SUCCESS) {
        return status;
    }

    status =
        ooverlap::comm::api::fill_tensor_slice(
            local,
            launch.rank,
            launch.world_size,
            element_offset,
            count,
            dtype,
            out_slice);

    if (status != OO_SUCCESS) {
        return status;
    }

    ooverlap::comm::LaunchConfig config =
        ooverlap::comm::select_launch_config_for_collective(
            ooverlap::comm::CollectivePlanFor::ReduceScatter,
            launch.world_size,
            launch.bytes,
            dtype,
            ooverlap::comm::tuning_preference_from_public(
                tuning_mode));

    ooverlap::comm::plan::ReduceScatterTransferPlan* transfer_plan = nullptr;

    status =
        node->group->transfer_plan_distribution->get_reduce_scatter_transfer_plan(
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

    cudaError_t error =
        ooverlap::enqueue_tma_multi_gpu_reduce_scatter_rank_sm90(
            launch,
            dtype,
            op,
            stream,
            config,
            *transfer_plan);

    return ooverlap::comm::api::cuda_to_status(error);
}

} // namespace

extern "C" oo_status_t oo_reduce_scatter_offset(
    oo_node_t* node,
    oo_buffer_t* local,
    size_t element_offset,
    size_t count,
    oo_dtype_t dtype,
    oo_reduce_op_t op,
    oo_tensor_slice_t* out_slice,
    cudaStream_t stream) {
    return reduce_scatter_impl(
        node,
        local,
        element_offset,
        count,
        dtype,
        op,
        OO_TUNING_BEST_PERFORMANCE,
        out_slice,
        stream);
}

extern "C" oo_status_t oo_reduce_scatter(
    oo_node_t* node,
    oo_buffer_t* local,
    size_t count,
    oo_dtype_t dtype,
    oo_reduce_op_t op,
    oo_tensor_slice_t* out_slice,
    cudaStream_t stream) {
    return oo_reduce_scatter_offset(
        node,
        local,
        0,
        count,
        dtype,
        op,
        out_slice,
        stream);
}

extern "C" oo_status_t oo_reduce_scatter_offset_tuned(
    oo_node_t* node,
    oo_buffer_t* local,
    size_t element_offset,
    size_t count,
    oo_dtype_t dtype,
    oo_reduce_op_t op,
    oo_tuning_mode_t tuning_mode,
    oo_tensor_slice_t* out_slice,
    cudaStream_t stream) {
    return reduce_scatter_impl(
        node,
        local,
        element_offset,
        count,
        dtype,
        op,
        tuning_mode,
        out_slice,
        stream);
}

extern "C" oo_status_t oo_reduce_scatter_tuned(
    oo_node_t* node,
    oo_buffer_t* local,
    size_t count,
    oo_dtype_t dtype,
    oo_reduce_op_t op,
    oo_tuning_mode_t tuning_mode,
    oo_tensor_slice_t* out_slice,
    cudaStream_t stream) {
    return oo_reduce_scatter_offset_tuned(
        node,
        local,
        0,
        count,
        dtype,
        op,
        tuning_mode,
        out_slice,
        stream);
}
