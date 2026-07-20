#include "comm/tma_multi_gpu_reduce_scatter_sm90.h"

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
#include <cstdlib>
#include <driver_types.h>
#include <new>
#include <stdexcept>

namespace ooverlap {
namespace {

constexpr int kDefaultMaxCtasPerReduceTask = 8;

int max_ctas_per_reduce_task_from_env() {
    static const int value = [] {
        const char* text =
            std::getenv("OOVERLAP_MAX_CTAS_PER_REDUCE_TASK");

        if (text == nullptr || text[0] == '\0') {
            return kDefaultMaxCtasPerReduceTask;
        }

        char* end = nullptr;
        const long parsed = std::strtol(text, &end, 10);

        if (end == text || *end != '\0' || parsed <= 0) {
            return kDefaultMaxCtasPerReduceTask;
        }

        if (parsed > comm::task::kWindowTaskMaxCtas) {
            return comm::task::kWindowTaskMaxCtas;
        }

        return static_cast<int>(parsed);
    }();

    return value;
}

template <int MaxPeers>
bool validate_reduce_scatter_launch(
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
    typename ReduceOp,
    int ChunkBytes,
    int StageDepth,
    int FillDepth = StageDepth / 2,
    int LoadFillDepth = FillDepth>
cudaError_t launch_reduce_scatter_rank_variant_sm90(
    const comm::api::CollectiveLaunchState& launch,
    comm::LaunchConfig launch_config,
    const comm::plan::ReduceScatterTransferPlan& transfer_plan,
    cudaStream_t stream) {
    using Variant = comm::TmaPipelineVariant<
        ChunkBytes,
        StageDepth,
        FillDepth,
        LoadFillDepth>;

    using ReduceApply = comm::pipeline::PipelineTMAReduce<
        Variant::stage_depth,
        Variant::fill_depth,
        ReduceOp>;

    constexpr int MaxLoweringTasks =
        comm::plan::kTmaMultiGpuReduceScatterMaxWindowTasks;
    constexpr int MaxByValueTasks =
        comm::plan::kTmaMultiGpuByValueMaxWindowTasks;
    constexpr int MaxPeers =
        comm::plan::kTmaMultiGpuReduceScatterMaxPeers;

    using ByValuePlan =
        comm::plan::WindowTaskExecutorPlan<MaxByValueTasks>;

    static_assert(
        sizeof(ByValuePlan) <= 16 * 1024,
        "reduce-scatter by-value plan unexpectedly exceeds 16 KiB");

    constexpr int MaxRanks =
        kOoMaxLocalDevices;
    constexpr int MaxStagingSlots = kOoMaxStagingSlots;

    if (launch_config.plan_for != comm::CollectivePlanFor::ReduceScatter ||
        !comm::launch_config_valid(launch_config)) {
        return cudaErrorInvalidValue;
    }

    if (launch_config.chunk_bytes != Variant::chunk_bytes ||
        launch_config.stage_depth != Variant::stage_depth) {
        return cudaErrorInvalidValue;
    }

    if (!validate_reduce_scatter_launch<MaxPeers>(launch)) {
        return cudaErrorInvalidValue;
    }

    if (transfer_plan.world_size != launch.world_size ||
        transfer_plan.total_tasks < 0 ||
        transfer_plan.total_tasks >
            comm::plan::kTmaMultiGpuReduceScatterMaxTransferTasks) {
        return cudaErrorInvalidValue;
    }

    /*
     * The public reduce-scatter API is currently in-place over one full logical
     * buffer per rank.  The TransferPlan itself is pointer-free; this is the
     * only place where logical rank buffers are bound to the rank-local raw
     * pointer view.
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
        ready_binding.poll_sleep_cycles_by_channel[channel] =
            launch.ready_signal_poll_sleep_cycles_by_channel[channel];

        if (launch.local_ready_signal_by_channel[channel] != nullptr) {
            has_ready_binding = true;
        }
    }

    ready_binding.local_ready_signal =
        launch.local_ready_signal;
    ready_binding.protocol =
        launch.ready_signal_protocol_by_channel
            [::kOoReadySignalChannelDeviceMemory];
    ready_binding.poll_sleep_cycles =
        launch.ready_signal_poll_sleep_cycles_by_channel
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
    static thread_local ByValuePlan by_value_plan;

    int num_blocks = 0;

    comm::plan::lowering_detail::LoweringPassOptions lowering_options{};
    lowering_options.enable_reduce_cta_groups = true;
    lowering_options.max_ctas_per_reduce_task =
        max_ctas_per_reduce_task_from_env();

    const bool plan_ok =
        comm::plan::lower_transfer_plan_for_rank<
            comm::plan::kTmaMultiGpuReduceScatterMaxTransferTasks,
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

    if (!comm::plan::window_task_executor_plan_pack(
            window_plan,
            &by_value_plan)) {
        return cudaErrorInvalidConfiguration;
    }

    system::runtime::set_device(launch.local_device);

    comm::kernels::configure_multi_gpu_window_task_executor_once<
        ReduceApply,
        ChunkBytes,
        StageDepth,
        MaxByValueTasks,
        MaxPeers,
        Variant::fill_depth,
        Variant::load_fill_depth>(
            launch.local_device,
            "tma_multi_gpu_reduce_scatter(by-value): requested shared memory exceeds opt-in limit");

    return comm::kernels::launch_multi_gpu_window_task_executor_sm90<
        ReduceApply,
        ChunkBytes,
        StageDepth,
        MaxByValueTasks,
        MaxPeers,
        Variant::fill_depth,
        Variant::load_fill_depth>(
            by_value_plan,
            num_blocks,
            launch_config.threads,
            Variant::dynamic_shared_bytes,
            stream,
            launch.local_ready_signal,
            ready_plan,
            launch.collective_epoch);
}

template <
    int ChunkBytes,
    int StageDepth,
    int FillDepth = StageDepth / 2,
    int LoadFillDepth = FillDepth>
cudaError_t dispatch_reduce_scatter_dtype_op_sm90(
    const comm::api::CollectiveLaunchState& launch,
    oo_dtype_t dtype,
    oo_reduce_op_t op,
    cudaStream_t stream,
    comm::LaunchConfig launch_config,
    const comm::plan::ReduceScatterTransferPlan& transfer_plan) {
    if (dtype == OO_DTYPE_FLOAT16) {
        if (op == OO_REDUCE_ADD || op == OO_REDUCE_SUM) {
            return launch_reduce_scatter_rank_variant_sm90<
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

        if (op == OO_REDUCE_MIN) {
            return launch_reduce_scatter_rank_variant_sm90<
                comm::pipeline::PipelineReduceMinF16,
                ChunkBytes,
                StageDepth,
                FillDepth,
                LoadFillDepth>(
                    launch,
                    launch_config,
                    transfer_plan,
                    stream);
        }

        if (op == OO_REDUCE_MAX) {
            return launch_reduce_scatter_rank_variant_sm90<
                comm::pipeline::PipelineReduceMaxF16,
                ChunkBytes,
                StageDepth,
                FillDepth,
                LoadFillDepth>(
                    launch,
                    launch_config,
                    transfer_plan,
                    stream);
        }
    }

    if (dtype == OO_DTYPE_BFLOAT16) {
        if (op == OO_REDUCE_ADD || op == OO_REDUCE_SUM) {
            return launch_reduce_scatter_rank_variant_sm90<
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

        if (op == OO_REDUCE_MIN) {
            return launch_reduce_scatter_rank_variant_sm90<
                comm::pipeline::PipelineReduceMinBF16,
                ChunkBytes,
                StageDepth,
                FillDepth,
                LoadFillDepth>(
                    launch,
                    launch_config,
                    transfer_plan,
                    stream);
        }

        if (op == OO_REDUCE_MAX) {
            return launch_reduce_scatter_rank_variant_sm90<
                comm::pipeline::PipelineReduceMaxBF16,
                ChunkBytes,
                StageDepth,
                FillDepth,
                LoadFillDepth>(
                    launch,
                    launch_config,
                    transfer_plan,
                    stream);
        }
    }

    if (dtype == OO_DTYPE_FLOAT32 &&
        (op == OO_REDUCE_ADD || op == OO_REDUCE_SUM)) {
        return launch_reduce_scatter_rank_variant_sm90<
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

cudaError_t dispatch_reduce_scatter_variant_sm90(
    const comm::api::CollectiveLaunchState& launch,
    oo_dtype_t dtype,
    oo_reduce_op_t op,
    cudaStream_t stream,
    comm::LaunchConfig launch_config,
    const comm::plan::ReduceScatterTransferPlan& transfer_plan) {
#define OO_TRY_VARIANT(CHUNK_BYTES_VALUE, STAGE_DEPTH_VALUE, FILL_DEPTH_VALUE, LOAD_FILL_DEPTH_VALUE) \
    if (launch_config.chunk_bytes == (CHUNK_BYTES_VALUE) &&                                      \
        launch_config.stage_depth == (STAGE_DEPTH_VALUE)) {                                      \
        return dispatch_reduce_scatter_dtype_op_sm90<                                            \
            (CHUNK_BYTES_VALUE),                                                                 \
            (STAGE_DEPTH_VALUE),                                                                 \
            (FILL_DEPTH_VALUE),                                                                  \
            (LOAD_FILL_DEPTH_VALUE)>(                                                           \
                launch,                                                                          \
                dtype,                                                                           \
                op,                                                                              \
                stream,                                                                          \
                launch_config,                                                                   \
                transfer_plan);                                                                  \
    }

    OOVERLAP_TMA_TWO_GPU_PEER_FOR_EACH_VARIANT_WITH_DEPTH(OO_TRY_VARIANT)

#undef OO_TRY_VARIANT

    return cudaErrorInvalidValue;
}

} // namespace

cudaError_t enqueue_tma_multi_gpu_reduce_scatter_rank_sm90(
    const comm::api::CollectiveLaunchState& launch,
    oo_dtype_t dtype,
    oo_reduce_op_t op,
    cudaStream_t stream,
    comm::LaunchConfig launch_config,
    const comm::plan::ReduceScatterTransferPlan& transfer_plan) {
    if (launch.bytes == 0) {
        return cudaSuccess;
    }

    if (launch.local_ptr == nullptr) {
        return cudaErrorInvalidValue;
    }

    if (!comm::utils::reduce_op_supported_for_dtype(dtype, op)) {
        return cudaErrorInvalidValue;
    }

    if (launch_config.plan_for != comm::CollectivePlanFor::ReduceScatter ||
        !comm::launch_config_valid(launch_config)) {
        return cudaErrorInvalidValue;
    }

    try {
        return dispatch_reduce_scatter_variant_sm90(
            launch,
            dtype,
            op,
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
