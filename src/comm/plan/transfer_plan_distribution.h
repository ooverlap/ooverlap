#pragma once

#include "comm/ooverlap_comm_private.h"
#include "comm/plan/plan_params.cuh"
#include "comm/plan/transfer_plan.h"

#include <memory>

namespace ooverlap {
namespace comm {
namespace plan {

constexpr int kTmaMultiGpuAllReduceMaxTransferTasks =
    (kTmaMultiGpuAllReduceMaxPeers + 1) *
    (2 * kTmaMultiGpuAllReduceMaxPeers + 1);

using AllreduceTransferPlan =
    TransferPlan<kTmaMultiGpuAllReduceMaxTransferTasks>;

class TransferPlanDistributionBackend {
public:
    virtual ~TransferPlanDistributionBackend() = default;

    virtual oo_status_t get_allreduce_transfer_plan(
        oo_node_t* node,
        const ooverlap::comm::api::CollectiveLaunchState& launch,
        size_t count,
        oo_dtype_t dtype,
        oo_reduce_op_t op,
        const ooverlap::comm::LaunchConfig& config,
        AllreduceTransferPlan* out_plan) = 0;
};

std::unique_ptr<TransferPlanDistributionBackend>
make_same_process_transfer_plan_distribution_backend();

} // namespace plan
} // namespace comm
} // namespace ooverlap
