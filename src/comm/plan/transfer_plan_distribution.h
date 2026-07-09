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

constexpr int kTmaMultiGpuReduceScatterMaxTransferTasks =
    (kTmaMultiGpuReduceScatterMaxPeers + 1) *
    (kTmaMultiGpuReduceScatterMaxPeers + 1);

constexpr int kTmaMultiGpuAllGatherMaxTransferTasks =
    (kTmaMultiGpuAllGatherMaxPeers + 1) *
    (kTmaMultiGpuAllGatherMaxPeers + 1);

using AllreduceTransferPlan =
    TransferPlan<kTmaMultiGpuAllReduceMaxTransferTasks>;

using ReduceScatterTransferPlan =
    TransferPlan<kTmaMultiGpuReduceScatterMaxTransferTasks>;

using AllGatherTransferPlan =
    TransferPlan<kTmaMultiGpuAllGatherMaxTransferTasks>;

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
        AllreduceTransferPlan** out_plan) = 0;

    virtual oo_status_t get_reduce_scatter_transfer_plan(
        oo_node_t* node,
        const ooverlap::comm::api::CollectiveLaunchState& launch,
        size_t count,
        oo_dtype_t dtype,
        oo_reduce_op_t op,
        const ooverlap::comm::LaunchConfig& config,
        ReduceScatterTransferPlan** out_plan) = 0;

    virtual oo_status_t get_all_gather_transfer_plan(
        oo_node_t* node,
        const ooverlap::comm::api::CollectiveLaunchState& launch,
        size_t count,
        oo_dtype_t dtype,
        const ooverlap::comm::LaunchConfig& config,
        AllGatherTransferPlan** out_plan) = 0;
};

std::unique_ptr<TransferPlanDistributionBackend>
make_same_process_transfer_plan_distribution_backend();

/*
 * OOVERLAP_IPC_SHARED_PLAN_BACKEND_DECL_PATCH:
 *
 * Multiprocess IPC backend.
 *
 * It maps a fixed-size POSIX shared-memory plan arena in every process.  On a
 * cache miss, rank 0 builds the pointer-free TransferPlan directly into the
 * shared arena and the other ranks read it from the same mapped memory after a
 * Broker barrier.  No chunk-by-chunk plan exchange is used.
 */
std::unique_ptr<TransferPlanDistributionBackend>
make_ipc_shared_plan_transfer_plan_distribution_backend(
    const char* broker_key,
    int local_rank,
    int world_size);

} // namespace plan
} // namespace comm
} // namespace ooverlap
