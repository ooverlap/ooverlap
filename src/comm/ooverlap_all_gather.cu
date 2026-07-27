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
    cudaStream_t stream,
    oo_buffer_t* const* prebound_rank_buffers = nullptr,
    int prebound_rank_buffer_count = 0) {
    if (!oo_all_gather_supported(dtype)) {
        return OO_ERROR_UNSUPPORTED;
    }

    if (node == nullptr ||
        node->group == nullptr ||
        node->group->transfer_plan_distribution == nullptr) {
        return OO_ERROR_INVALID_ARGUMENT;
    }

    ooverlap::comm::api::CollectiveLaunchState launch{};

    oo_status_t status = OO_SUCCESS;

    if (prebound_rank_buffers != nullptr) {
        status =
            ooverlap::comm::api::prepare_collective_launch_prebound(
                node,
                prebound_rank_buffers,
                prebound_rank_buffer_count,
                ooverlap::comm::CollectivePlanFor::AllGather,
                element_offset,
                count,
                dtype,
                &launch);
    } else {
        status =
            ooverlap::comm::api::prepare_collective_launch(
                node,
                local,
                ooverlap::comm::CollectivePlanFor::AllGather,
                element_offset,
                count,
                dtype,
                &launch);
    }

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


extern "C" oo_status_t oo_all_gather_slot_tuned(
    oo_node_t* node,
    oo_ipc_slot_set_t* set,
    int slot_index,
    size_t count,
    oo_dtype_t dtype,
    oo_tuning_mode_t tuning_mode,
    cudaStream_t stream) {
    if (node == nullptr ||
        node->group == nullptr ||
        set == nullptr ||
        set->group != node->group ||
        set->world_size != node->group->num_devices ||
        slot_index < 0 ||
        slot_index >= set->slot_count ||
        node->rank < 0 ||
        node->rank >= set->world_size) {
        return OO_ERROR_INVALID_ARGUMENT;
    }

    oo_buffer_t* const* rank_buffers =
        set->rank_buffers.data() +
        static_cast<size_t>(slot_index) *
            static_cast<size_t>(set->world_size);

    oo_buffer_t* local = rank_buffers[node->rank];
    if (local == nullptr || local->ptr == nullptr) {
        return OO_ERROR_INVALID_ARGUMENT;
    }

    return all_gather_impl(
        node,
        local,
        0,
        count,
        dtype,
        tuning_mode,
        stream,
        rank_buffers,
        set->world_size);
}
