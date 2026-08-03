#include "comm/tma_multi_gpu_allreduce_sm90.h"

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
bool validate_allreduce_launch(
    const comm::api::CollectiveLaunchState& launch) {
    if (((!launch.out_of_place && launch.local_ptr == nullptr) ||
         (launch.out_of_place &&
          (launch.local_input_ptr == nullptr ||
           launch.local_output_ptr == nullptr))) ||
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

        if (((!launch.out_of_place && launch.peer_ptrs[peer_idx] == nullptr) ||
             (launch.out_of_place &&
              (launch.peer_input_ptrs[peer_idx] == nullptr ||
               launch.peer_output_ptrs[peer_idx] == nullptr))) ||
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

#ifndef OOVERLAP_DEBUG_PRINT_WINDOW_TASKS
#define OOVERLAP_DEBUG_PRINT_WINDOW_TASKS 0
#endif

#if OOVERLAP_DEBUG_PRINT_WINDOW_TASKS
const char* debug_window_task_op_name(
    comm::task::WindowTaskOp op) {
    switch (op) {
        case comm::task::WindowTaskOp::None:
            return "None";
        case comm::task::WindowTaskOp::ReduceTMA:
            return "ReduceTMA";
        case comm::task::WindowTaskOp::CopyTMA:
            return "CopyTMA";
        case comm::task::WindowTaskOp::CopyFast:
            return "CopyFast";
        case comm::task::WindowTaskOp::ReadyPublish:
            return "ReadyPublish";
        case comm::task::WindowTaskOp::ReadyWait:
            return "ReadyWait";
        case comm::task::WindowTaskOp::ReadyPublishWait:
            return "ReadyPublishWait";
        case comm::task::WindowTaskOp::Barrier:
            return "Barrier";
        case comm::task::WindowTaskOp::FinalPeerRendezvous:
            return "FinalPeerRendezvous";
        default:
            return "Unknown";
    }
}

template <int MaxTasks>
void debug_print_window_task_plan(
    const char* tag,
    int rank,
    int num_blocks,
    const comm::plan::WindowTaskExecutorPlan<MaxTasks>& plan) {
    std::fprintf(
        stderr,
        "\n[%s rank=%d] WindowTaskExecutorPlan: total_tasks=%d tasks_per_cta=%d num_blocks=%d sizeof(WindowTask)=%zu sizeof(plan)=%zu\n",
        tag,
        rank,
        plan.total_tasks,
        plan.tasks_per_cta,
        num_blocks,
        sizeof(comm::task::WindowTask),
        sizeof(plan));

    for (int cta = 0; cta < num_blocks; ++cta) {
        std::fprintf(stderr, "  CTA %d:\n", cta);

        for (int local_task = 0;
             local_task < plan.tasks_per_cta;
             ++local_task) {
            const int task_idx =
                cta * plan.tasks_per_cta + local_task;

            if (task_idx >= plan.total_tasks) {
                break;
            }

            const comm::task::WindowTask& task =
                plan.tasks[task_idx];

            std::fprintf(
                stderr,
                "    task[%d] local=%d op=%s(%d) terminal=%d mask=%u",
                task_idx,
                local_task,
                debug_window_task_op_name(task.op),
                static_cast<int>(task.op),
                static_cast<int>(task.terminal),
                static_cast<unsigned int>(task.cta_mask));

            switch (task.op) {
                case comm::task::WindowTaskOp::ReduceTMA:
                case comm::task::WindowTaskOp::CopyTMA:
                case comm::task::WindowTaskOp::CopyFast:
                    std::fprintf(
                        stderr,
                        " src=%p dst=%p total_bytes=%u begin_window=%d end_window=%d window_chunks=%d",
                        task.payload.window.src,
                        task.payload.window.dst,
                        static_cast<unsigned int>(
                            task.payload.window.total_bytes),
                        task.payload.window.begin_window,
                        task.payload.window.end_window,
                        task.payload.window.window_chunks);
                    break;

                case comm::task::WindowTaskOp::CopyTMAFanout:
                case comm::task::WindowTaskOp::ReduceTMAFanout:
                    std::fprintf(
                        stderr,
                        " src=%p fanout_count=%u total_bytes=%u begin_window=%d end_window=%d window_chunks=%d",
                        task.payload.fanout.src,
                        static_cast<unsigned int>(
                            task.payload.fanout.fanout_dst_count),
                        static_cast<unsigned int>(
                            task.payload.fanout.total_bytes),
                        task.payload.fanout.begin_window,
                        task.payload.fanout.end_window,
                        task.payload.fanout.window_chunks);
                    break;

                case comm::task::WindowTaskOp::ReadyPublish:
                case comm::task::WindowTaskOp::ReadyWait:
                case comm::task::WindowTaskOp::ReadyPublishWait:
                    std::fprintf(
                        stderr,
                        " ready_signal=%p ready_wait_signal=%p ready_epoch=%d ready_protocol=%d ready_wait_epoch=%d ready_owner_cta=%d",
                        static_cast<void*>(
                            task.payload.ready.ready_signal),
                        static_cast<const void*>(
                            task.payload.ready.ready_wait_signal),
                        task.payload.ready.ready_epoch,
                        task.payload.ready.ready_protocol,
                        task.payload.ready.ready_wait_epoch,
                        task.payload.ready.ready_owner_cta);
                    break;

                case comm::task::WindowTaskOp::Barrier:
                    std::fprintf(
                        stderr,
                        " barrier_target=%u",
                        static_cast<unsigned int>(
                            task.payload.barrier_target));
                    break;

                case comm::task::WindowTaskOp::FinalPeerRendezvous:
                    std::fprintf(
                        stderr,
                        " ready_phase=%u",
                        static_cast<unsigned int>(
                            task.payload.ready_phase));
                    break;

                case comm::task::WindowTaskOp::None:
                default:
                    break;
            }

            std::fprintf(stderr, "\n");
        }
    }

    std::fprintf(stderr, "\n");
}
#endif

template <int MaxTasks>
unsigned int final_cta_barrier_counter_value(
    const comm::plan::WindowTaskExecutorPlan<MaxTasks>& plan,
    unsigned int entry_value) {
    unsigned int value = entry_value;
    const int first_stripe_tasks =
        plan.tasks_per_cta < plan.total_tasks
            ? plan.tasks_per_cta
            : plan.total_tasks;

    for (int i = 0; i < first_stripe_tasks; ++i) {
        if (plan.tasks[i].op == comm::task::WindowTaskOp::Barrier) {
            value = plan.tasks[i].payload.barrier_target;
        }
    }

    return value;
}

template <
    typename ReduceOp,
    int ChunkBytes,
    int StageDepth,
    int FillDepth = StageDepth / 2,
    int LoadFillDepth = FillDepth>
cudaError_t launch_allreduce_rank_variant_sm90(
    const comm::api::CollectiveLaunchState& launch,
    comm::LaunchConfig launch_config,
    const comm::plan::AllreduceTransferPlan& transfer_plan,
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
        comm::plan::kTmaMultiGpuAllReduceMaxWindowTasks;
    constexpr int MaxPeers =
        comm::plan::kTmaMultiGpuAllReduceMaxPeers;
    constexpr int MaxRanks =
        kOoMaxLocalDevices;
    constexpr int MaxStagingSlots = kOoMaxStagingSlots;

    if (launch_config.plan_for != comm::CollectivePlanFor::AllReduce ||
        !comm::launch_config_valid(launch_config)) {
        return cudaErrorInvalidValue;
    }

    if (launch_config.chunk_bytes != Variant::chunk_bytes ||
        launch_config.stage_depth != Variant::stage_depth) {
        return cudaErrorInvalidValue;
    }

    if (!validate_allreduce_launch<MaxPeers>(launch)) {
        return cudaErrorInvalidValue;
    }

    if (transfer_plan.world_size != launch.world_size ||
        transfer_plan.total_tasks < 0 ||
        transfer_plan.total_tasks >
            comm::plan::kTmaMultiGpuAllReduceMaxTransferTasks) {
        return cudaErrorInvalidValue;
    }

    /*
     * OOVERLAP_OUT_OF_PLACE_ALLREDUCE_REDUCE_FANOUT_PATCH
     *
     * Bind logical RankInput/RankOutput separately when an out-of-place API
     * supplies those pointers. In-place behavior remains unchanged.
     */
    comm::plan::RankPointerBinding<MaxRanks, MaxStagingSlots> binding{};
    binding.current_rank = launch.rank;
    binding.world_size = launch.world_size;

    void* local_input_ptr =
        launch.out_of_place ? launch.local_input_ptr : launch.local_ptr;
    void* local_output_ptr =
        launch.out_of_place ? launch.local_output_ptr : launch.local_ptr;

    binding.rank_buffer[launch.rank] = local_output_ptr;
    binding.rank_input[launch.rank] = local_input_ptr;
    binding.rank_output[launch.rank] = local_output_ptr;

    for (int peer_idx = 0; peer_idx < launch.peer_count; ++peer_idx) {
        const int peer_rank = launch.peer_ranks[peer_idx];

        void* peer_input_ptr =
            launch.out_of_place
                ? launch.peer_input_ptrs[peer_idx]
                : launch.peer_ptrs[peer_idx];
        void* peer_output_ptr =
            launch.out_of_place
                ? launch.peer_output_ptrs[peer_idx]
                : launch.peer_ptrs[peer_idx];

        binding.rank_buffer[peer_rank] = peer_output_ptr;
        binding.rank_input[peer_rank] = peer_input_ptr;
        binding.rank_output[peer_rank] = peer_output_ptr;
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

    /*
     * OOVERLAP_ALL_COLLECTIVES_PLAN_BY_VALUE_V1
     *
     * Lower into ordinary thread-local host memory. Keep only the persistent
     * CTA barrier counter and its host-side sequence value in a small cache.
     * There is no mapped plan and no CUDA event in this all-reduce path.
     */
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
    lowering_options.enable_reduce_cta_groups = true;
    lowering_options.max_ctas_per_reduce_task =
        launch_config.max_ctas_per_reduce_task;
    lowering_options.cta_barrier_start = cta_barrier_start;
    lowering_options.enable_final_peer_rendezvous =
        use_ready_binding && ready_plan.peer_count > 0;
    lowering_options.final_ready_phase =
        comm::plan::kReadySignalPhaseStride - 1;

    const bool plan_ok =
        comm::plan::lower_transfer_plan_for_rank<
            comm::plan::kTmaMultiGpuAllReduceMaxTransferTasks,
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

    const unsigned int cta_barrier_final_value =
        final_cta_barrier_counter_value(
            window_plan,
            cta_barrier_start);

    #if OOVERLAP_DEBUG_PRINT_WINDOW_TASKS
        debug_print_window_task_plan<MaxLoweringTasks>(
            "allreduce_by_value",
            launch.rank,
            num_blocks,
            window_plan);
    #endif

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
                "tma_multi_gpu_allreduce(by-value): requested shared memory exceeds opt-in limit",
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
cudaError_t dispatch_allreduce_dtype_op_sm90(
    const comm::api::CollectiveLaunchState& launch,
    oo_dtype_t dtype,
    oo_reduce_op_t op,
    cudaStream_t stream,
    comm::LaunchConfig launch_config,
    const comm::plan::AllreduceTransferPlan& transfer_plan) {
    if (dtype == OO_DTYPE_FLOAT16) {
        if (op == OO_REDUCE_ADD || op == OO_REDUCE_SUM) {
            return launch_allreduce_rank_variant_sm90<
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
            return launch_allreduce_rank_variant_sm90<
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
            return launch_allreduce_rank_variant_sm90<
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
            return launch_allreduce_rank_variant_sm90<
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
            return launch_allreduce_rank_variant_sm90<
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
            return launch_allreduce_rank_variant_sm90<
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
        return launch_allreduce_rank_variant_sm90<
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

cudaError_t dispatch_allreduce_variant_sm90(
    const comm::api::CollectiveLaunchState& launch,
    oo_dtype_t dtype,
    oo_reduce_op_t op,
    cudaStream_t stream,
    comm::LaunchConfig launch_config,
    const comm::plan::AllreduceTransferPlan& transfer_plan) {
#define OO_TRY_VARIANT(CHUNK_BYTES_VALUE, STAGE_DEPTH_VALUE, FILL_DEPTH_VALUE, LOAD_FILL_DEPTH_VALUE) \
    if (launch_config.chunk_bytes == (CHUNK_BYTES_VALUE) &&                                      \
        launch_config.stage_depth == (STAGE_DEPTH_VALUE)) {                                      \
        return dispatch_allreduce_dtype_op_sm90<                                                 \
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

cudaError_t enqueue_tma_multi_gpu_allreduce_rank_sm90(
    const comm::api::CollectiveLaunchState& launch,
    oo_dtype_t dtype,
    oo_reduce_op_t op,
    cudaStream_t stream,
    comm::LaunchConfig launch_config,
    const comm::plan::AllreduceTransferPlan& transfer_plan) {
    if (launch.bytes == 0) {
        return cudaSuccess;
    }

    if ((!launch.out_of_place && launch.local_ptr == nullptr) ||
        (launch.out_of_place &&
         (launch.local_input_ptr == nullptr ||
          launch.local_output_ptr == nullptr))) {
        return cudaErrorInvalidValue;
    }

    if (!comm::utils::reduce_op_supported_for_dtype(dtype, op)) {
        return cudaErrorInvalidValue;
    }

    if (launch_config.plan_for != comm::CollectivePlanFor::AllReduce ||
        !comm::launch_config_valid(launch_config)) {
        return cudaErrorInvalidValue;
    }

    try {
        return dispatch_allreduce_variant_sm90(
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
