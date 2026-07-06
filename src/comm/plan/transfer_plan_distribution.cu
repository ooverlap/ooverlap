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
    return a.threads == b.threads &&
           a.max_ctas == b.max_ctas &&
           a.window_chunks == b.window_chunks &&
           a.chunk_bytes == b.chunk_bytes &&
           a.stage_depth == b.stage_depth &&
           a.plan_for == b.plan_for &&
           a.plan.allreduce == b.plan.allreduce;
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
            launch.world_size <= 0) {
            return OO_ERROR_INVALID_ARGUMENT;
        }

        oo_group_t* group = node->group;

        std::unique_lock<std::mutex> lock(mutex_);

        if (!active_) {
            active_ = true;
            ready_ = false;
            failed_ = false;
            copied_ = 0;

            epoch_ = launch.collective_epoch;
            world_size_ = launch.world_size;
            count_ = count;
            dtype_size_ = launch.dtype_size;
            dtype_ = dtype;
            op_ = op;
            config_ = config;

            const oo_status_t status =
                build_allreduce_plan_locked(
                    group,
                    launch,
                    count,
                    dtype,
                    op,
                    config);

            if (status != OO_SUCCESS) {
                failed_ = true;
                status_ = status;
                ready_ = true;
                cv_.notify_all();
                return status;
            }

            status_ = OO_SUCCESS;
            ready_ = true;
            cv_.notify_all();
        } else {
            if (epoch_ != launch.collective_epoch ||
                world_size_ != launch.world_size ||
                count_ != count ||
                dtype_size_ != launch.dtype_size ||
                dtype_ != dtype ||
                op_ != op ||
                !same_launch_config(config_, config)) {
                return OO_ERROR_INVALID_ARGUMENT;
            }

            cv_.wait(lock, [this]() {
                return ready_;
            });

            if (failed_) {
                return status_;
            }
        }

        *out_plan = plan_;

        copied_ += 1;

        if (copied_ == world_size_) {
            reset_locked();
        }

        return OO_SUCCESS;
    }

private:
    oo_status_t build_allreduce_plan_locked(
        oo_group_t* group,
        const ooverlap::comm::api::CollectiveLaunchState& launch,
        size_t count,
        oo_dtype_t dtype,
        oo_reduce_op_t,
        const ooverlap::comm::LaunchConfig& config) {
        if (group == nullptr ||
            group->num_devices != launch.world_size ||
            launch.dtype_size == 0) {
            return OO_ERROR_INVALID_ARGUMENT;
        }

        TransferPlanBuildInput input{};
        input.topo.topology =
            group->topology_valid ? &group->topology : nullptr;
        input.topo.rank_devices = group->devices;
        input.topo.world_size = group->num_devices;

        input.collective = CollectivePlanFor::AllReduce;
        input.launch_config = config;
        input.world_size = launch.world_size;
        input.count = count;
        input.dtype_size = launch.dtype_size;
        input.out_of_place = false;

        if (!build_allreduce_transfer_plan(&plan_, input)) {
            return OO_ERROR_UNSUPPORTED;
        }

        return OO_SUCCESS;
    }

    void reset_locked() {
        active_ = false;
        ready_ = false;
        failed_ = false;
        copied_ = 0;
        epoch_ = 0;
        world_size_ = 0;
        count_ = 0;
        dtype_size_ = 0;
        dtype_ = OO_DTYPE_FLOAT16;
        op_ = OO_REDUCE_ADD;
        config_ = LaunchConfig{};
        status_ = OO_SUCCESS;
        transfer_plan_clear(&plan_);
    }

    std::mutex mutex_;
    std::condition_variable cv_;

    bool active_ = false;
    bool ready_ = false;
    bool failed_ = false;

    int copied_ = 0;

    int epoch_ = 0;
    int world_size_ = 0;

    size_t count_ = 0;
    size_t dtype_size_ = 0;

    oo_dtype_t dtype_ = OO_DTYPE_FLOAT16;
    oo_reduce_op_t op_ = OO_REDUCE_ADD;
    LaunchConfig config_{};

    oo_status_t status_ = OO_SUCCESS;

    AllreduceTransferPlan plan_{};
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
