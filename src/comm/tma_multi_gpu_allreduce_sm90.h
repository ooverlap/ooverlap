#pragma once

#include "comm/ooverlap_comm_private.h"
#include "comm/plan/transfer_plan_distribution.h"

#include <cuda_runtime.h>

namespace ooverlap {

/*
 * Rank-local SM90 allreduce launch using a distributed logical TransferPlan.
 *
 * The public API prepares CollectiveLaunchState once.  This layer only binds
 * that launch state to the logical transfer plan, lowers it into a
 * WindowTaskExecutorPlan for the current rank, and launches the executor.
 */
cudaError_t enqueue_tma_multi_gpu_allreduce_rank_sm90(
    const comm::api::CollectiveLaunchState& launch,
    oo_dtype_t dtype,
    oo_reduce_op_t op,
    cudaStream_t stream,
    comm::LaunchConfig launch_config,
    const comm::plan::AllreduceTransferPlan& transfer_plan);

} // namespace ooverlap
