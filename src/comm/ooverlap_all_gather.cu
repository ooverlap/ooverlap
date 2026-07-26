#include "comm/ooverlap_comm_private.h"

#include "comm/plan/transfer_plan_distribution.h"
#include "comm/tuning/tuning_policy.h"
#include "comm/tma_multi_gpu_all_gather_sm90.h"

namespace {

oo_status_t all_gather_impl(
    oo_node_t* node,
    oo_buffer_t* local,
    size_t element_offset,
    size_t count,
    oo_dtype_t dtype,
    oo_tuning_mode_t tuning_mode,
    cudaStream_t stream) {
    if (!oo_all_gather_supported(dtype)) {
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
            ooverlap::comm::CollectivePlanFor::AllGather,
            element_offset,
            count,
            dtype,
            &launch);

    if (status != OO_SUCCESS) {
        return status;
    }

    ooverlap::comm::LaunchConfig config =
        ooverlap::comm::select_launch_config_for_collective(
            ooverlap::comm::CollectivePlanFor::AllGather,
            launch.world_size,
            launch.bytes,
            ooverlap::comm::tuning_preference_from_public(
                tuning_mode));

    ooverlap::comm::plan::AllGatherTransferPlan* transfer_plan = nullptr;

    status =
        node->group->transfer_plan_distribution->get_all_gather_transfer_plan(
            node,
            launch,
            count,
            dtype,
            config,
            &transfer_plan);

    if (status != OO_SUCCESS) {
        return status;
    }

    cudaError_t error =
        ooverlap::enqueue_tma_multi_gpu_all_gather_rank_sm90(
            launch,
            dtype,
            stream,
            config,
            *transfer_plan);

    return ooverlap::comm::api::cuda_to_status(error);
}

} // namespace

extern "C" oo_status_t oo_all_gather_offset(
    oo_node_t* node,
    oo_buffer_t* local,
    size_t element_offset,
    size_t count,
    oo_dtype_t dtype,
    cudaStream_t stream) {
    return all_gather_impl(
        node,
        local,
        element_offset,
        count,
        dtype,
        OO_TUNING_BEST_PERFORMANCE,
        stream);
}

extern "C" oo_status_t oo_all_gather(
    oo_node_t* node,
    oo_buffer_t* local,
    size_t count,
    oo_dtype_t dtype,
    cudaStream_t stream) {
    return oo_all_gather_offset(
        node,
        local,
        0,
        count,
        dtype,
        stream);
}

extern "C" oo_status_t oo_all_gather_offset_tuned(
    oo_node_t* node,
    oo_buffer_t* local,
    size_t element_offset,
    size_t count,
    oo_dtype_t dtype,
    oo_tuning_mode_t tuning_mode,
    cudaStream_t stream) {
    return all_gather_impl(
        node,
        local,
        element_offset,
        count,
        dtype,
        tuning_mode,
        stream);
}

extern "C" oo_status_t oo_all_gather_tuned(
    oo_node_t* node,
    oo_buffer_t* local,
    size_t count,
    oo_dtype_t dtype,
    oo_tuning_mode_t tuning_mode,
    cudaStream_t stream) {
    return oo_all_gather_offset_tuned(
        node,
        local,
        0,
        count,
        dtype,
        tuning_mode,
        stream);
}
