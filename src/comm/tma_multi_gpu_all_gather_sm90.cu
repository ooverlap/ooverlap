#include "comm/tma_multi_gpu_all_gather_sm90.h"

#include "ooverlap/system/runtime_utils.cuh"

#include "comm/kernels/multi_gpu_ready_signal.cuh"
#include "comm/kernels/multi_gpu_window_task_executor.cuh"
#include "comm/params.h"
#include "comm/pipeline/pipeline_tma_reduce.h"
#include "comm/plan/transfer_lowering.cuh"
#include "comm/plan/window_plan.cuh"
#include "comm/tma_variant_config.h"
#include "comm/utils/collective_utils.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <driver_types.h>
#include <new>
#include <stdexcept>

namespace ooverlap {
namespace {

template <int MaxPeers>
bool validate_all_gather_launch(
    const comm::api::CollectiveLaunchState& launch) {
    if (launch.local_ptr == nullptr ||
        launch.peer_count < 0 ||
        launch.peer_count > MaxPeers ||
        launch.world_size != launch.peer_count + 1 ||
        launch.rank < 0 ||
        launch.rank >= launch.world_size ||
        launch.local_device < 0 ||
        launch.world_size > kOoMaxLocalDevices) {
        return false;
    }

    bool seen[kOoMaxLocalDevices] = {};
    seen[launch.rank] = true;

    for (int peer_idx = 0; peer_idx < launch.peer_count; ++peer_idx) {
        const int peer_rank = launch.peer_ranks[peer_idx];

        if (launch.peer_ptrs[peer_idx] == nullptr ||
            peer_rank < 0 ||
            peer_rank >= launch.world_size ||
            peer_rank == launch.rank ||
            seen[peer_rank]) {
            return false;
        }

        seen[peer_rank] = true;
    }

    for (int rank = 0; rank < launch.world_size; ++rank) {
        if (!seen[rank]) {
            return false;
        }
    }

    return true;
}

template <
    typename DummyReduceOp,
    int ChunkBytes,
    int StageDepth,
    int FillDepth = StageDepth / 2,
    int LoadFillDepth = FillDepth>
cudaError_t launch_all_gather_rank_variant_sm90(
    const comm::api::CollectiveLaunchState& launch,
    comm::LaunchConfig launch_config,
    const comm::plan::AllGatherTransferPlan& transfer_plan,
    cudaStream_t stream) {
    using Variant = comm::TmaPipelineVariant<
        ChunkBytes,
        StageDepth,
        FillDepth,
        LoadFillDepth>;

    /*
     * The generic task executor template still needs a ReduceApply type.  The
     * all-gather transfer plan emits copy tasks only, so this reduce path is not
     * used.
     */
    using ReduceApply = comm::pipeline::PipelineTMAReduce<
        Variant::stage_depth,
        Variant::fill_depth,
        DummyReduceOp>;

    constexpr int MaxTasks =
        comm::plan::kTmaMultiGpuAllGatherMaxWindowTasks;
    constexpr int MaxPeers =
        comm::plan::kTmaMultiGpuAllGatherMaxPeers;
    constexpr int MaxRanks =
        kOoMaxLocalDevices;
    constexpr int MaxStagingSlots = 0;

    if (launch_config.plan_for != comm::CollectivePlanFor::AllGather ||
        !comm::launch_config_valid(launch_config)) {
        return cudaErrorInvalidValue;
    }

    if (launch_config.chunk_bytes != Variant::chunk_bytes ||
        launch_config.stage_depth != Variant::stage_depth) {
        return cudaErrorInvalidValue;
    }

    if (!validate_all_gather_launch<MaxPeers>(launch)) {
        return cudaErrorInvalidValue;
    }

    if (transfer_plan.world_size != launch.world_size ||
        transfer_plan.total_tasks < 0 ||
        transfer_plan.total_tasks >
            comm::plan::kTmaMultiGpuAllGatherMaxTransferTasks) {
        return cudaErrorInvalidValue;
    }

    /*
     * The public all-gather API is currently in-place over one full logical
     * buffer per rank.  The TransferPlan is pointer-free; this is where logical
     * rank buffers are bound to this rank's raw pointer view.
     */
    comm::plan::RankPointerBinding<MaxRanks, MaxStagingSlots> binding{};
    binding.current_rank = launch.rank;
    binding.world_size = launch.world_size;

    binding.rank_buffer[launch.rank] = launch.local_ptr;
    binding.rank_input[launch.rank] = launch.local_ptr;
    binding.rank_output[launch.rank] = launch.local_ptr;

    for (int peer_idx = 0; peer_idx < launch.peer_count; ++peer_idx) {
        const int peer_rank = launch.peer_ranks[peer_idx];

        binding.rank_buffer[peer_rank] = launch.peer_ptrs[peer_idx];
        binding.rank_input[peer_rank] = launch.peer_ptrs[peer_idx];
        binding.rank_output[peer_rank] = launch.peer_ptrs[peer_idx];
    }

    const auto ready_plan =
        comm::kernels::make_multi_gpu_ready_signal_plan<MaxPeers>(
            launch.peer_count,
            launch.peer_ready_signals);

    const bool use_ready_tasks =
        launch.local_ready_signal != nullptr &&
        launch.collective_epoch > 0 &&
        ready_plan.protocol !=
            comm::kernels::MultiGpuReadySignalProtocol::Disabled;

    const int ready_prefix_tasks_per_cta =
        use_ready_tasks ? (1 + ready_plan.peer_count) : 0;

    comm::plan::WindowTaskExecutorPlan<MaxTasks> window_plan{};
    int num_blocks = 0;

    const bool plan_ok =
        comm::plan::lower_transfer_plan_for_rank<
            comm::plan::kTmaMultiGpuAllGatherMaxTransferTasks,
            MaxTasks,
            MaxRanks,
            MaxStagingSlots>(
                transfer_plan,
                binding,
                launch_config,
                &window_plan,
                &num_blocks,
                ready_prefix_tasks_per_cta);

    if (!plan_ok) {
        return cudaErrorInvalidValue;
    }

    if (num_blocks <= 0 || window_plan.total_tasks <= 0) {
        return cudaSuccess;
    }

    if (use_ready_tasks) {
        if (!comm::plan::prepend_ready_tasks_to_each_cta(
                &window_plan,
                num_blocks,
                launch.local_ready_signal,
                launch.peer_ready_signals,
                ready_plan.peer_count,
                launch.collective_epoch,
                static_cast<int>(ready_plan.protocol),
                ready_plan.poll_sleep_cycles)) {
            return cudaErrorInvalidValue;
        }
    }

    system::runtime::set_device(launch.local_device);

    comm::kernels::configure_multi_gpu_window_task_executor_once<
        ReduceApply,
        ChunkBytes,
        StageDepth,
        MaxTasks,
        MaxPeers,
        Variant::fill_depth,
        Variant::load_fill_depth>(
            launch.local_device,
            "tma_multi_gpu_all_gather: requested shared memory exceeds opt-in limit");

    comm::kernels::multi_gpu_window_task_executor_kernel_sm90<
        ReduceApply,
        ChunkBytes,
        StageDepth,
        MaxTasks,
        MaxPeers,
        Variant::fill_depth,
        Variant::load_fill_depth><<<
            num_blocks,
            launch_config.threads,
            Variant::dynamic_shared_bytes,
            stream>>>(
                window_plan,
                launch.local_ready_signal,
                ready_plan,
                launch.collective_epoch);

    return cudaSuccess;
}

template <
    int ChunkBytes,
    int StageDepth,
    int FillDepth = StageDepth / 2,
    int LoadFillDepth = FillDepth>
cudaError_t dispatch_all_gather_dtype_sm90(
    const comm::api::CollectiveLaunchState& launch,
    oo_dtype_t dtype,
    cudaStream_t stream,
    comm::LaunchConfig launch_config,
    const comm::plan::AllGatherTransferPlan& transfer_plan) {
    if (dtype == OO_DTYPE_FLOAT16) {
        return launch_all_gather_rank_variant_sm90<
            comm::pipeline::PipelineReduceAddNoFtzF16,
            ChunkBytes,
            StageDepth,
            FillDepth,
            LoadFillDepth>(
                launch,
                launch_config,
                transfer_plan,
                stream);
    }

    if (dtype == OO_DTYPE_BFLOAT16) {
        return launch_all_gather_rank_variant_sm90<
            comm::pipeline::PipelineReduceAddBF16,
            ChunkBytes,
            StageDepth,
            FillDepth,
            LoadFillDepth>(
                launch,
                launch_config,
                transfer_plan,
                stream);
    }

    if (dtype == OO_DTYPE_FLOAT32) {
        return launch_all_gather_rank_variant_sm90<
            comm::pipeline::PipelineReduceAddF32,
            ChunkBytes,
            StageDepth,
            FillDepth,
            LoadFillDepth>(
                launch,
                launch_config,
                transfer_plan,
                stream);
    }

    return cudaErrorInvalidValue;
}

cudaError_t dispatch_all_gather_variant_sm90(
    const comm::api::CollectiveLaunchState& launch,
    oo_dtype_t dtype,
    cudaStream_t stream,
    comm::LaunchConfig launch_config,
    const comm::plan::AllGatherTransferPlan& transfer_plan) {
#define OO_TRY_VARIANT(CHUNK_BYTES_VALUE, STAGE_DEPTH_VALUE, FILL_DEPTH_VALUE, LOAD_FILL_DEPTH_VALUE) \
    if (launch_config.chunk_bytes == (CHUNK_BYTES_VALUE) &&                                      \
        launch_config.stage_depth == (STAGE_DEPTH_VALUE)) {                                      \
        return dispatch_all_gather_dtype_sm90<                                                    \
            (CHUNK_BYTES_VALUE),                                                                 \
            (STAGE_DEPTH_VALUE),                                                                 \
            (FILL_DEPTH_VALUE),                                                                  \
            (LOAD_FILL_DEPTH_VALUE)>(                                                           \
                launch,                                                                          \
                dtype,                                                                           \
                stream,                                                                          \
                launch_config,                                                                   \
                transfer_plan);                                                                  \
    }

    OOVERLAP_TMA_TWO_GPU_PEER_FOR_EACH_VARIANT_WITH_DEPTH(OO_TRY_VARIANT)

#undef OO_TRY_VARIANT

    return cudaErrorInvalidValue;
}

} // namespace

cudaError_t enqueue_tma_multi_gpu_all_gather_rank_sm90(
    const comm::api::CollectiveLaunchState& launch,
    oo_dtype_t dtype,
    cudaStream_t stream,
    comm::LaunchConfig launch_config,
    const comm::plan::AllGatherTransferPlan& transfer_plan) {
    if (launch.bytes == 0) {
        return cudaSuccess;
    }

    if (launch.local_ptr == nullptr) {
        return cudaErrorInvalidValue;
    }

    if (!comm::utils::dtype_supported_for_copy_collective(dtype)) {
        return cudaErrorInvalidValue;
    }

    if (launch_config.plan_for != comm::CollectivePlanFor::AllGather ||
        !comm::launch_config_valid(launch_config)) {
        return cudaErrorInvalidValue;
    }

    try {
        return dispatch_all_gather_variant_sm90(
            launch,
            dtype,
            stream,
            launch_config,
            transfer_plan);
    } catch (const std::bad_alloc&) {
        return cudaErrorMemoryAllocation;
    } catch (const std::exception&) {
        return cudaErrorUnknown;
    } catch (...) {
        return cudaErrorUnknown;
    }
}

} // namespace ooverlap
