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

    constexpr int MaxLoweringTasks =
        comm::plan::kTmaMultiGpuByValueMaxWindowTasks;
    constexpr int MaxPeers =
        comm::plan::kTmaMultiGpuAllGatherMaxPeers;

    constexpr int MaxRanks =
        kOoMaxLocalDevices;
    constexpr int MaxStagingSlots = kOoMaxStagingSlots;

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

    for (int slot = 0;
         slot < MaxStagingSlots && slot < launch.staging_slot_count;
         ++slot) {
        binding.shm_staging[slot] = launch.staging_ptrs[slot];
    }

    const int device_ready_channel =
        ::kOoReadySignalChannelDeviceMemory;

    const auto ready_plan =
        comm::kernels::make_multi_gpu_ready_signal_plan<MaxPeers>(
            launch.peer_count,
            launch.peer_ready_signals,
            static_cast<comm::kernels::MultiGpuReadySignalProtocol>(
                launch.ready_signal_protocol_by_channel[device_ready_channel]),
            launch.ready_signal_poll_sleep_cycles_by_channel[device_ready_channel]);

    comm::plan::ReadySignalBinding<MaxRanks> ready_binding{};
    ready_binding.epoch = launch.collective_epoch;

    bool has_ready_binding = false;

    for (int channel = 0;
         channel < comm::plan::kReadySignalChannelCount;
         ++channel) {
        ready_binding.local_ready_signal_by_channel[channel] =
            launch.local_ready_signal_by_channel[channel];
        ready_binding.protocol_by_channel[channel] =
            launch.ready_signal_protocol_by_channel[channel];

        if (launch.local_ready_signal_by_channel[channel] != nullptr) {
            has_ready_binding = true;
        }
    }

    ready_binding.local_ready_signal =
        launch.local_ready_signal;
    ready_binding.protocol =
        launch.ready_signal_protocol_by_channel
            [::kOoReadySignalChannelDeviceMemory];

    if (launch.rank >= 0 && launch.rank < MaxRanks) {
        ready_binding.ready_signal_by_rank[launch.rank] =
            launch.local_ready_signal;
        for (int channel = 0;
             channel < comm::plan::kReadySignalChannelCount;
             ++channel) {
            ready_binding.ready_signal_by_rank_channel[launch.rank][channel] =
                launch.local_ready_signal_by_channel[channel];
        }
    }

    for (int peer_idx = 0; peer_idx < ready_plan.peer_count; ++peer_idx) {
        const int peer_rank = launch.peer_ranks[peer_idx];

        if (peer_rank >= 0 && peer_rank < MaxRanks) {
            ready_binding.ready_signal_by_rank[peer_rank] =
                ready_plan.peer_ready_signals[peer_idx];
            for (int channel = 0;
                 channel < comm::plan::kReadySignalChannelCount;
                 ++channel) {
                ready_binding.ready_signal_by_rank_channel
                    [peer_rank][channel] =
                        launch.peer_ready_signals_by_channel[peer_idx][channel];
            }
        }
    }


    const bool use_ready_binding =
        has_ready_binding &&
        launch.collective_epoch > 0;

    const comm::plan::ReadySignalBinding<MaxRanks>* ready_binding_ptr =
        use_ready_binding ? &ready_binding : nullptr;

    /* OOVERLAP_ALL_COLLECTIVES_PLAN_BY_VALUE_V1 */
    static thread_local
        comm::plan::WindowTaskExecutorPlan<MaxLoweringTasks> window_plan;

    comm::kernels::CtaBarrierScratch cta_barrier_scratch{};

    const cudaError_t cta_barrier_scratch_error =
        comm::kernels::get_cached_cta_barrier_scratch(
            launch.local_device,
            launch.plan_scratch_index,
            &cta_barrier_scratch);

    if (cta_barrier_scratch_error != cudaSuccess ||
        cta_barrier_scratch.counter == nullptr ||
        cta_barrier_scratch.last_value == nullptr) {
        return cta_barrier_scratch_error != cudaSuccess
            ? cta_barrier_scratch_error
            : cudaErrorInvalidValue;
    }

    int num_blocks = 0;

    const unsigned int cta_barrier_start =
        *cta_barrier_scratch.last_value + 1u;

    comm::plan::lowering_detail::LoweringPassOptions lowering_options{};
    lowering_options.cta_barrier_start = cta_barrier_start;
    lowering_options.enable_final_peer_rendezvous =
        use_ready_binding && ready_plan.peer_count > 0;
    lowering_options.final_ready_phase =
        comm::plan::kReadySignalPhaseStride - 1;

    const bool plan_ok =
        comm::plan::lower_transfer_plan_for_rank<
            comm::plan::kTmaMultiGpuAllGatherMaxTransferTasks,
            MaxLoweringTasks,
            MaxRanks,
            MaxStagingSlots>(
                transfer_plan,
                binding,
                launch_config,
                &window_plan,
                &num_blocks,
                0,
                ready_binding_ptr,
                lowering_options);

    if (!plan_ok) {
        return cudaErrorInvalidValue;
    }

    if (num_blocks <= 0 || window_plan.total_tasks <= 0) {
        return cudaSuccess;
    }

    unsigned int cta_barrier_final_value = cta_barrier_start;
    const int first_stripe_tasks =
        window_plan.tasks_per_cta < window_plan.total_tasks
            ? window_plan.tasks_per_cta
            : window_plan.total_tasks;

    for (int i = 0; i < first_stripe_tasks; ++i) {
        if (window_plan.tasks[i].op ==
            comm::task::WindowTaskOp::Barrier) {
            cta_barrier_final_value =
                window_plan.tasks[i].payload.barrier_target;
        }
    }

    system::runtime::set_device(launch.local_device);

    const cudaError_t launch_error =
        comm::kernels::dispatch_multi_gpu_window_task_executor_by_value_sm90<
        ReduceApply,
        ChunkBytes,
        StageDepth,
        MaxLoweringTasks,
        MaxPeers,
        Variant::fill_depth,
        Variant::load_fill_depth>(
            window_plan,
            num_blocks,
            launch_config.threads,
            Variant::dynamic_shared_bytes,
            stream,
            launch.local_device,
            launch.local_ready_signal,
            ready_plan,
            launch.collective_epoch,
            "tma_multi_gpu_all_gather(by-value): requested shared memory exceeds opt-in limit",
            cta_barrier_scratch.counter,
            cta_barrier_start);

    if (launch_error != cudaSuccess) {
        return launch_error;
    }

    *cta_barrier_scratch.last_value =
        cta_barrier_final_value;

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
