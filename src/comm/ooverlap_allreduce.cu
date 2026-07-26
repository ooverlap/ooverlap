#include "comm/ooverlap_comm_private.h"

#include "comm/plan/transfer_plan_distribution.h"
#include "comm/tuning/tuning_policy.h"
#include "comm/tma_multi_gpu_allreduce_sm90.h"
#include "ooverlap/comm.h"

namespace {

oo_status_t allreduce_impl(
    oo_node_t* node,
    oo_buffer_t* local,
    size_t element_offset,
    size_t count,
    oo_dtype_t dtype,
    oo_reduce_op_t op,
    oo_tuning_mode_t tuning_mode,
    cudaStream_t stream,
    oo_buffer_t* const* prebound_rank_buffers = nullptr,
    int prebound_rank_buffer_count = 0,
    int plan_scratch_index = 0) {
    /* OOVERLAP_ROUND_ROBIN_SLOT_POOL_PATCH_V1 */
    /* OOVERLAP_ROUND_ROBIN_PLAN_SCRATCH_RING_V1 */
    if (!oo_allreduce_supported(dtype, op)) {
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
                ooverlap::comm::CollectivePlanFor::AllReduce,
                element_offset,
                count,
                dtype,
                &launch);
    } else {
        status =
            ooverlap::comm::api::prepare_collective_launch(
                node,
                local,
                ooverlap::comm::CollectivePlanFor::AllReduce,
                element_offset,
                count,
                dtype,
                &launch);
    }

    if (status != OO_SUCCESS) {
        return status;
    }

    launch.plan_scratch_index = plan_scratch_index;

    ooverlap::comm::LaunchConfig config =
        ooverlap::comm::select_launch_config_for_collective(
            ooverlap::comm::CollectivePlanFor::AllReduce,
            launch.world_size,
            launch.bytes,
            dtype,
            ooverlap::comm::tuning_preference_from_public(
                tuning_mode));

    ooverlap::comm::plan::AllreduceTransferPlan* transfer_plan = nullptr;

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

    cudaError_t error =
        ooverlap::enqueue_tma_multi_gpu_allreduce_rank_sm90(
            launch,
            dtype,
            op,
            stream,
            config,
            *transfer_plan);

    return ooverlap::comm::api::cuda_to_status(error);
}

} // namespace

extern "C" oo_status_t oo_allreduce_offset(
    oo_node_t* node,
    oo_buffer_t* local,
    size_t element_offset,
    size_t count,
    oo_dtype_t dtype,
    oo_reduce_op_t op,
    cudaStream_t stream) {
    return allreduce_impl(
        node,
        local,
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
    size_t count,
    oo_dtype_t dtype,
    oo_reduce_op_t op,
    cudaStream_t stream) {
    return oo_allreduce_offset(
        node,
        local,
        0,
        count,
        dtype,
        op,
        stream);
}

extern "C" oo_status_t oo_allreduce_offset_tuned(
    oo_node_t* node,
    oo_buffer_t* local,
    size_t element_offset,
    size_t count,
    oo_dtype_t dtype,
    oo_reduce_op_t op,
    oo_tuning_mode_t tuning_mode,
    cudaStream_t stream) {
    return allreduce_impl(
        node,
        local,
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
    size_t count,
    oo_dtype_t dtype,
    oo_reduce_op_t op,
    oo_tuning_mode_t tuning_mode,
    cudaStream_t stream) {
    return oo_allreduce_offset_tuned(
        node,
        local,
        0,
        count,
        dtype,
        op,
        tuning_mode,
        stream);
}

extern "C" oo_status_t oo_allreduce_slot_tuned(
    oo_node_t* node,
    oo_ipc_slot_set_t* set,
    int slot_index,
    size_t count,
    oo_dtype_t dtype,
    oo_reduce_op_t op,
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

    return allreduce_impl(
        node,
        local,
        0,
        count,
        dtype,
        op,
        tuning_mode,
        stream,
        rank_buffers,
        set->world_size,
        slot_index);
}

