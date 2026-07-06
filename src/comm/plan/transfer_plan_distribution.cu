#include "comm/plan/transfer_plan_distribution.h"

#include "comm/plan/transfer_planner.h"

#include <condition_variable>
#include <mutex>

namespace ooverlap {
namespace comm {
namespace plan {
namespace {

bool same_launch_config(
    const LaunchConfig& a,
    const LaunchConfig& b) {
    if (a.threads != b.threads ||
        a.max_ctas != b.max_ctas ||
        a.window_chunks != b.window_chunks ||
        a.chunk_bytes != b.chunk_bytes ||
        a.stage_depth != b.stage_depth ||
        a.plan_for != b.plan_for) {
        return false;
    }

    switch (a.plan_for) {
        case CollectivePlanFor::AllReduce:
            return a.plan.allreduce == b.plan.allreduce;

        case CollectivePlanFor::ReduceScatter:
            return a.plan.reduce_scatter == b.plan.reduce_scatter;

        case CollectivePlanFor::AllGather:
            return a.plan.all_gather == b.plan.all_gather;

        default:
            return false;
    }
}

struct TransferPlanRequestKey {
    CollectivePlanFor collective = CollectivePlanFor::AllReduce;
    int epoch = 0;
    int world_size = 0;

    size_t count = 0;
    size_t dtype_size = 0;

    oo_dtype_t dtype = OO_DTYPE_FLOAT16;
    oo_reduce_op_t op = OO_REDUCE_ADD;

    LaunchConfig config{};
};

bool same_request_key(
    const TransferPlanRequestKey& a,
    const TransferPlanRequestKey& b) {
    return a.collective == b.collective &&
           a.epoch == b.epoch &&
           a.world_size == b.world_size &&
           a.count == b.count &&
           a.dtype_size == b.dtype_size &&
           a.dtype == b.dtype &&
           a.op == b.op &&
           same_launch_config(a.config, b.config);
}

TransferPlanRequestKey make_request_key(
    CollectivePlanFor collective,
    const ooverlap::comm::api::CollectiveLaunchState& launch,
    size_t count,
    oo_dtype_t dtype,
    oo_reduce_op_t op,
    const LaunchConfig& config) {
    TransferPlanRequestKey key{};
    key.collective = collective;
    key.epoch = launch.collective_epoch;
    key.world_size = launch.world_size;
    key.count = count;
    key.dtype_size = launch.dtype_size;
    key.dtype = dtype;
    key.op = op;
    key.config = config;
    return key;
}

TransferPlanBuildInput make_build_input(
    oo_group_t* group,
    const ooverlap::comm::api::CollectiveLaunchState& launch,
    size_t count,
    const LaunchConfig& config,
    CollectivePlanFor collective) {
    TransferPlanBuildInput input{};
    input.topo.topology =
        (group != nullptr && group->topology_valid) ? &group->topology : nullptr;
    input.topo.rank_devices = group != nullptr ? group->devices : nullptr;
    input.topo.world_size = group != nullptr ? group->num_devices : 0;

    input.collective = collective;
    input.launch_config = config;
    input.world_size = launch.world_size;
    input.count = count;
    input.dtype_size = launch.dtype_size;

    /*
     * Current public collectives are in-place over full logical buffers.
     * If/when compact input/output buffers are added, this is the bit that
     * should become collective-call metadata instead of a hardcoded false.
     */
    input.out_of_place = false;
    return input;
}

template <int MaxTransferTasks>
struct SameProcessPlanSlot {
    std::mutex mutex;
    std::condition_variable cv;

    bool active = false;
    bool ready = false;
    bool failed = false;

    int copied = 0;
    oo_status_t status = OO_SUCCESS;

    TransferPlanRequestKey key{};
    TransferPlan<MaxTransferTasks> plan{};
};

template <int MaxTransferTasks>
void reset_slot_locked(
    SameProcessPlanSlot<MaxTransferTasks>* slot) {
    if (slot == nullptr) {
        return;
    }

    slot->active = false;
    slot->ready = false;
    slot->failed = false;
    slot->copied = 0;
    slot->status = OO_SUCCESS;
    slot->key = TransferPlanRequestKey{};
    transfer_plan_clear(&slot->plan);
}

template <int MaxTransferTasks, typename Builder>
oo_status_t get_or_build_same_process_plan(
    SameProcessPlanSlot<MaxTransferTasks>* slot,
    oo_node_t* node,
    const ooverlap::comm::api::CollectiveLaunchState& launch,
    const TransferPlanRequestKey& key,
    TransferPlan<MaxTransferTasks>* out_plan,
    Builder&& builder) {
    if (slot == nullptr ||
        node == nullptr ||
        node->group == nullptr ||
        out_plan == nullptr ||
        launch.world_size <= 0) {
        return OO_ERROR_INVALID_ARGUMENT;
    }

    std::unique_lock<std::mutex> lock(slot->mutex);

    if (!slot->active) {
        slot->active = true;
        slot->ready = false;
        slot->failed = false;
        slot->copied = 0;
        slot->status = OO_SUCCESS;
        slot->key = key;

        const oo_status_t status =
            builder(&slot->plan);

        if (status != OO_SUCCESS) {
            slot->failed = true;
            slot->status = status;
        }

        slot->ready = true;
        slot->cv.notify_all();
    } else {
        if (!same_request_key(slot->key, key)) {
            return OO_ERROR_INVALID_ARGUMENT;
        }

        slot->cv.wait(lock, [slot]() {
            return slot->ready;
        });
    }

    if (slot->failed) {
        const oo_status_t status = slot->status;

        slot->copied += 1;

        if (slot->copied == slot->key.world_size) {
            reset_slot_locked(slot);
        }

        return status;
    }

    *out_plan = slot->plan;

    slot->copied += 1;

    if (slot->copied == slot->key.world_size) {
        reset_slot_locked(slot);
    }

    return OO_SUCCESS;
}

class SameProcessTransferPlanDistributionBackend final
    : public TransferPlanDistributionBackend {
public:
    oo_status_t get_allreduce_transfer_plan(
        oo_node_t* node,
        const ooverlap::comm::api::CollectiveLaunchState& launch,
        size_t count,
        oo_dtype_t dtype,
        oo_reduce_op_t op,
        const ooverlap::comm::LaunchConfig& config,
        AllreduceTransferPlan* out_plan) override {
        if (node == nullptr ||
            node->group == nullptr ||
            out_plan == nullptr ||
            launch.dtype_size == 0 ||
            config.plan_for != CollectivePlanFor::AllReduce) {
            return OO_ERROR_INVALID_ARGUMENT;
        }

        oo_group_t* group = node->group;

        const TransferPlanRequestKey key =
            make_request_key(
                CollectivePlanFor::AllReduce,
                launch,
                count,
                dtype,
                op,
                config);

        return get_or_build_same_process_plan(
            &allreduce_,
            node,
            launch,
            key,
            out_plan,
            [group, &launch, count, &config](AllreduceTransferPlan* plan) {
                if (group == nullptr ||
                    group->num_devices != launch.world_size) {
                    return OO_ERROR_INVALID_ARGUMENT;
                }

                const TransferPlanBuildInput input =
                    make_build_input(
                        group,
                        launch,
                        count,
                        config,
                        CollectivePlanFor::AllReduce);

                if (!build_allreduce_transfer_plan(plan, input)) {
                    return OO_ERROR_UNSUPPORTED;
                }

                return OO_SUCCESS;
            });
    }

    oo_status_t get_reduce_scatter_transfer_plan(
        oo_node_t* node,
        const ooverlap::comm::api::CollectiveLaunchState& launch,
        size_t count,
        oo_dtype_t dtype,
        oo_reduce_op_t op,
        const ooverlap::comm::LaunchConfig& config,
        ReduceScatterTransferPlan* out_plan) override {
        if (node == nullptr ||
            node->group == nullptr ||
            out_plan == nullptr ||
            launch.dtype_size == 0 ||
            config.plan_for != CollectivePlanFor::ReduceScatter) {
            return OO_ERROR_INVALID_ARGUMENT;
        }

        oo_group_t* group = node->group;

        const TransferPlanRequestKey key =
            make_request_key(
                CollectivePlanFor::ReduceScatter,
                launch,
                count,
                dtype,
                op,
                config);

        return get_or_build_same_process_plan(
            &reduce_scatter_,
            node,
            launch,
            key,
            out_plan,
            [group, &launch, count, &config](ReduceScatterTransferPlan* plan) {
                if (group == nullptr ||
                    group->num_devices != launch.world_size) {
                    return OO_ERROR_INVALID_ARGUMENT;
                }

                const TransferPlanBuildInput input =
                    make_build_input(
                        group,
                        launch,
                        count,
                        config,
                        CollectivePlanFor::ReduceScatter);

                if (!build_reduce_scatter_transfer_plan(plan, input)) {
                    return OO_ERROR_UNSUPPORTED;
                }

                return OO_SUCCESS;
            });
    }

    oo_status_t get_all_gather_transfer_plan(
        oo_node_t* node,
        const ooverlap::comm::api::CollectiveLaunchState& launch,
        size_t count,
        oo_dtype_t dtype,
        const ooverlap::comm::LaunchConfig& config,
        AllGatherTransferPlan* out_plan) override {
        if (node == nullptr ||
            node->group == nullptr ||
            out_plan == nullptr ||
            launch.dtype_size == 0 ||
            config.plan_for != CollectivePlanFor::AllGather) {
            return OO_ERROR_INVALID_ARGUMENT;
        }

        oo_group_t* group = node->group;

        const TransferPlanRequestKey key =
            make_request_key(
                CollectivePlanFor::AllGather,
                launch,
                count,
                dtype,
                OO_REDUCE_ADD,
                config);

        return get_or_build_same_process_plan(
            &all_gather_,
            node,
            launch,
            key,
            out_plan,
            [group, &launch, count, &config](AllGatherTransferPlan* plan) {
                if (group == nullptr ||
                    group->num_devices != launch.world_size) {
                    return OO_ERROR_INVALID_ARGUMENT;
                }

                const TransferPlanBuildInput input =
                    make_build_input(
                        group,
                        launch,
                        count,
                        config,
                        CollectivePlanFor::AllGather);

                if (!build_all_gather_transfer_plan(plan, input)) {
                    return OO_ERROR_UNSUPPORTED;
                }

                return OO_SUCCESS;
            });
    }

private:
    SameProcessPlanSlot<kTmaMultiGpuAllReduceMaxTransferTasks> allreduce_{};
    SameProcessPlanSlot<kTmaMultiGpuReduceScatterMaxTransferTasks> reduce_scatter_{};
    SameProcessPlanSlot<kTmaMultiGpuAllGatherMaxTransferTasks> all_gather_{};
};

} // namespace

std::unique_ptr<TransferPlanDistributionBackend>
make_same_process_transfer_plan_distribution_backend() {
    return std::unique_ptr<TransferPlanDistributionBackend>(
        new SameProcessTransferPlanDistributionBackend());
}

} // namespace plan
} // namespace comm
} // namespace ooverlap
