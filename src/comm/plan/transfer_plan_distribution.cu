#include "comm/plan/transfer_plan_distribution.h"

#include "comm/plan/transfer_planner.h"
#include "ooverlap/comm.h"

#include <mutex>
#include <utility>

namespace ooverlap {
namespace comm {
namespace plan {
namespace {

constexpr int kPlanCacheEntries = 16;

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
    input.out_of_place = false;
    return input;
}

template <int MaxTransferTasks>
struct CachedPlanEntry {
    bool valid = false;
    TransferPlanRequestKey key{};
    TransferPlan<MaxTransferTasks> plan{};
};

template <int MaxTransferTasks>
struct SameProcessPlanCache {
    std::mutex mutex;
    int next_victim = 0;
    CachedPlanEntry<MaxTransferTasks> entries[kPlanCacheEntries] = {};
};

template <int MaxTransferTasks, typename Builder>
oo_status_t get_or_build_cached_same_process_plan(
    SameProcessPlanCache<MaxTransferTasks>* cache,
    const TransferPlanRequestKey& key,
    TransferPlan<MaxTransferTasks>* out_plan,
    Builder&& builder) {
    if (cache == nullptr || key.world_size <= 0) {
        return OO_ERROR_INVALID_ARGUMENT;
    }

    std::lock_guard<std::mutex> lock(cache->mutex);

    for (int i = 0; i < kPlanCacheEntries; ++i) {
        CachedPlanEntry<MaxTransferTasks>& entry =
            cache->entries[i];

        if (entry.valid && same_request_key(entry.key, key)) {
            *out_plan = entry.plan;
            return OO_SUCCESS;
        }
    }

    int slot = -1;

    for (int i = 0; i < kPlanCacheEntries; ++i) {
        if (!cache->entries[i].valid) {
            slot = i;
            break;
        }
    }

    if (slot < 0) {
        slot = cache->next_victim;
        cache->next_victim =
            (cache->next_victim + 1) % kPlanCacheEntries;
    }

    CachedPlanEntry<MaxTransferTasks>& entry =
        cache->entries[slot];

    entry.valid = false;
    entry.key = key;
    transfer_plan_clear(&entry.plan);

    const oo_status_t status =
        builder(&entry.plan);

    if (status != OO_SUCCESS) {
        return status;
    }

    entry.valid = true;
    *out_plan = entry.plan;
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

        return get_or_build_cached_same_process_plan(
            &allreduce_,
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

                return build_allreduce_transfer_plan(plan, input)
                    ? OO_SUCCESS
                    : OO_ERROR_UNSUPPORTED;
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

        return get_or_build_cached_same_process_plan(
            &reduce_scatter_,
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

                return build_reduce_scatter_transfer_plan(plan, input)
                    ? OO_SUCCESS
                    : OO_ERROR_UNSUPPORTED;
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

        return get_or_build_cached_same_process_plan(
            &all_gather_,
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

                return build_all_gather_transfer_plan(plan, input)
                    ? OO_SUCCESS
                    : OO_ERROR_UNSUPPORTED;
            });
    }

private:
    SameProcessPlanCache<kTmaMultiGpuAllReduceMaxTransferTasks> allreduce_{};
    SameProcessPlanCache<kTmaMultiGpuReduceScatterMaxTransferTasks> reduce_scatter_{};
    SameProcessPlanCache<kTmaMultiGpuAllGatherMaxTransferTasks> all_gather_{};
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
