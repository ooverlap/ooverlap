#pragma once

#include "comm/ooverlap_comm_private.h"
#include "comm/plan/transfer_plan_distribution.h"

#include <cuda_runtime.h>

namespace ooverlap {

/*
 * Rank-local SM90 all-gather launch using a distributed logical TransferPlan.
 *
 * The public API prepares CollectiveLaunchState once.  This layer only binds
 * that launch state to the logical transfer plan, lowers it into a
 * WindowTaskExecutorPlan for the current rank, and launches the executor.
 */
cudaError_t enqueue_tma_multi_gpu_all_gather_rank_sm90(
    const comm::api::CollectiveLaunchState& launch,
    oo_dtype_t dtype,
    cudaStream_t stream,
    comm::LaunchConfig launch_config,
    const comm::plan::AllGatherTransferPlan& transfer_plan);

} // namespace ooverlap
