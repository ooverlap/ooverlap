#include "comm/tma_two_gpu_peer_allreduce_sm90.h"

#include "ooverlap/system/runtime_utils.cuh"

#include "comm/launch_config.h"
#include "comm/params.h"
#include "comm/pipeline_tma_reduce.h"
#include "comm/tma_variant_config.h"
#include "comm/utils.h"
#include "comm/window_task.cuh"
#include "comm/window_task_executor_sm90.cuh"

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <stdexcept>
#include <unordered_map>

namespace ooverlap {
namespace {

constexpr int kSeqTasksPerCTA = 2;
constexpr int kOverlapTasksPerCTA = 1;

/*
 * Enough for:
 *
 *   TmaCopy:
 *     max_ctas <= 32, two tasks per CTA
 *
 *   SeqFastGmem:
 *     max_ctas <= 32, two tasks per CTA
 *
 *   OverlapFastGmem:
 *     max_ctas <= 64, one task per CTA
 *
 * The plan is passed as a kernel parameter, so the measured stream work does
 * not include a per-launch cudaMemcpyAsync task upload.
 */
constexpr int kMaxWindowTasks = 64;

struct SignalCacheEntry {
    int* ptr = nullptr;
    size_t capacity = 0;
};

__host__ __device__ __forceinline__ size_t dtype_size_bytes(
    oo_dtype_t dtype) {
    switch (dtype) {
        case OO_DTYPE_FLOAT16:
            return sizeof(half);
        case OO_DTYPE_BFLOAT16:
            return sizeof(__nv_bfloat16);
        case OO_DTYPE_FLOAT32:
            return sizeof(float);
        default:
            return 0;
    }
}

__host__ __device__ __forceinline__ int overlap_pair_count_for_windows(
    int owned_windows,
    int max_ctas) {
    if (owned_windows <= 0 ||
        max_ctas < TMA_TWO_GPU_PEER_OVERLAP_BLOCKS_PER_CTA) {
        return 0;
    }

    return comm::utils::min_int(
        owned_windows,
        max_ctas / TMA_TWO_GPU_PEER_OVERLAP_BLOCKS_PER_CTA);
}

SignalCacheEntry& signal_cache_for_device(int device) {
    static std::mutex mutex;
    static std::unordered_map<int, SignalCacheEntry> cache;

    std::lock_guard<std::mutex> lock(mutex);
    return cache[device];
}

int* ensure_signal_capacity(int device, size_t required_count) {
    if (required_count == 0) {
        return nullptr;
    }

    SignalCacheEntry& entry = signal_cache_for_device(device);

    if (entry.ptr != nullptr && entry.capacity >= required_count) {
        return entry.ptr;
    }

    system::runtime::set_device(device);

    if (entry.ptr != nullptr) {
        cudaFree(entry.ptr);
        entry.ptr = nullptr;
        entry.capacity = 0;
    }

    system::runtime::check_cuda(
        cudaMalloc(&entry.ptr, required_count * sizeof(int)),
        "cudaMalloc(overlap window ready flags)");

    entry.capacity = required_count;
    return entry.ptr;
}

template <int MaxTasks>
bool build_tma_copy_inplace_plan(
    comm::WindowTaskExecutorPlan<MaxTasks>* plan,
    const void* local_in,
    void* local_buf,
    void* peer_buf,
    size_t total_bytes,
    int rank,
    int num_windows,
    int ctas_per_rank,
    int window_chunks) {
    (void)local_in;

    if (plan == nullptr) {
        return false;
    }

    comm::window_task_executor_plan_clear(plan);

    plan->tasks_per_cta = kSeqTasksPerCTA;
    plan->total_tasks = ctas_per_rank * kSeqTasksPerCTA;

    if (plan->total_tasks > MaxTasks) {
        return false;
    }

    const comm::utils::WindowRange rank_range =
        comm::utils::rank_window_range(num_windows, rank);

    for (int cta_idx = 0; cta_idx < ctas_per_rank; ++cta_idx) {
        const comm::utils::WindowRange cta_range =
            comm::utils::cta_window_range(
                cta_idx,
                ctas_per_rank,
                rank_range);

        const int base = cta_idx * kSeqTasksPerCTA;

        /*
         * Normal TMA-copy in-place policy:
         *
         *   reduce peer -> local
         *   copy   local -> peer using TMA
         */
        plan->tasks[base + 0] =
            comm::make_reduce_tma_task(
                peer_buf,
                local_buf,
                total_bytes,
                cta_range.begin,
                cta_range.end,
                window_chunks,
                false);

        plan->tasks[base + 1] =
            comm::make_copy_tma_task(
                local_buf,
                peer_buf,
                total_bytes,
                cta_range.begin,
                cta_range.end,
                window_chunks,
                true);
    }

    return true;
}

template <int MaxTasks>
bool build_tma_copy_out_of_place_plan(
    comm::WindowTaskExecutorPlan<MaxTasks>* plan,
    const void* local_in,
    void* local_buf,
    void* peer_buf,
    size_t total_bytes,
    int num_windows,
    int ctas_per_rank,
    int window_chunks) {
    if (plan == nullptr) {
        return false;
    }

    comm::window_task_executor_plan_clear(plan);

    plan->tasks_per_cta = kSeqTasksPerCTA;
    plan->total_tasks = ctas_per_rank * kSeqTasksPerCTA;

    if (plan->total_tasks > MaxTasks) {
        return false;
    }

    comm::utils::WindowRange full_range{};
    full_range.begin = 0;
    full_range.end = num_windows;

    for (int cta_idx = 0; cta_idx < ctas_per_rank; ++cta_idx) {
        const comm::utils::WindowRange cta_range =
            comm::utils::cta_window_range(
                cta_idx,
                ctas_per_rank,
                full_range);

        const int base = cta_idx * kSeqTasksPerCTA;

        /*
         * Normal TMA-copy different-buffer policy:
         *
         *   copy   peer  -> local output using TMA
         *   reduce local -> local output using TMA reduce
         *
         * This path is selected only when local_in != local_buf and only for
         * AllreducePlanKind::TmaCopy.
         */
        plan->tasks[base + 0] =
            comm::make_copy_tma_task(
                peer_buf,
                local_buf,
                total_bytes,
                cta_range.begin,
                cta_range.end,
                window_chunks,
                false);

        plan->tasks[base + 1] =
            comm::make_reduce_tma_task(
                local_in,
                local_buf,
                total_bytes,
                cta_range.begin,
                cta_range.end,
                window_chunks,
                true);
    }

    return true;
}

template <int MaxTasks>
bool build_seq_fast_gmem_plan(
    comm::WindowTaskExecutorPlan<MaxTasks>* plan,
    const void* local_in,
    void* local_buf,
    void* peer_buf,
    size_t total_bytes,
    int rank,
    int num_windows,
    int ctas_per_rank,
    int window_chunks) {
    (void)local_in;

    if (plan == nullptr) {
        return false;
    }

    comm::window_task_executor_plan_clear(plan);

    plan->tasks_per_cta = kSeqTasksPerCTA;
    plan->total_tasks = ctas_per_rank * kSeqTasksPerCTA;

    if (plan->total_tasks > MaxTasks) {
        return false;
    }

    const comm::utils::WindowRange rank_range =
        comm::utils::rank_window_range(num_windows, rank);

    for (int cta_idx = 0; cta_idx < ctas_per_rank; ++cta_idx) {
        const comm::utils::WindowRange cta_range =
            comm::utils::cta_window_range(
                cta_idx,
                ctas_per_rank,
                rank_range);

        const int base = cta_idx * kSeqTasksPerCTA;

        /*
         * Sequential fast-gmem policy:
         *
         *   reduce peer -> local
         *   copy   local -> peer using fast global-memory vector copy
         */
        plan->tasks[base + 0] =
            comm::make_reduce_tma_task(
                peer_buf,
                local_buf,
                total_bytes,
                cta_range.begin,
                cta_range.end,
                window_chunks,
                false);

        plan->tasks[base + 1] =
            comm::make_copy_fast_task(
                local_buf,
                peer_buf,
                total_bytes,
                cta_range.begin,
                cta_range.end,
                window_chunks,
                true);
    }

    return true;
}

template <int MaxTasks>
bool build_overlap_fast_gmem_plan(
    comm::WindowTaskExecutorPlan<MaxTasks>* plan,
    const void* local_in,
    void* local_buf,
    void* peer_buf,
    size_t total_bytes,
    int rank,
    int num_windows,
    int pair_count,
    int window_chunks,
    int* window_ready_flags) {
    (void)local_in;

    if (plan == nullptr) {
        return false;
    }

    comm::window_task_executor_plan_clear(plan);

    plan->tasks_per_cta = kOverlapTasksPerCTA;
    plan->total_tasks =
        pair_count * TMA_TWO_GPU_PEER_OVERLAP_BLOCKS_PER_CTA;

    if (plan->total_tasks > MaxTasks) {
        return false;
    }

    const comm::utils::WindowRange rank_range =
        comm::utils::rank_window_range(num_windows, rank);

    for (int pair_idx = 0; pair_idx < pair_count; ++pair_idx) {
        const comm::utils::WindowRange pair_range =
            comm::utils::cta_window_range(
                pair_idx,
                pair_count,
                rank_range);

        const int producer_block =
            pair_idx * TMA_TWO_GPU_PEER_OVERLAP_BLOCKS_PER_CTA + 0;
        const int consumer_block =
            pair_idx * TMA_TWO_GPU_PEER_OVERLAP_BLOCKS_PER_CTA + 1;

        /*
         * Overlap fast-gmem policy:
         *
         *   producer CTA:
         *     reduce peer -> local and signal completed windows
         *
         *   consumer CTA:
         *     wait for signal, then copy local -> peer using fast gmem copy
         */
        plan->tasks[producer_block] =
            comm::make_reduce_tma_signal_task(
                peer_buf,
                local_buf,
                total_bytes,
                pair_range.begin,
                pair_range.end,
                window_chunks,
                window_ready_flags,
                rank_range.begin,
                true);

        plan->tasks[consumer_block] =
            comm::make_copy_fast_after_signal_task(
                local_buf,
                peer_buf,
                total_bytes,
                pair_range.begin,
                pair_range.end,
                window_chunks,
                window_ready_flags,
                rank_range.begin,
                true);
    }

    return true;
}

template <
    typename ReduceApply,
    int ElemBytes,
    int ChunkBytes,
    int StageDepth>
void configure_kernel_once_for(int device) {
    (void)ElemBytes;

    using Variant = comm::TmaPipelineVariant<ChunkBytes, StageDepth>;

    struct CacheEntry {
        bool configured = false;
        size_t dynamic_smem_bytes = 0;
    };

    static std::mutex mutex;
    static std::unordered_map<int, CacheEntry> cache;

    const size_t dynamic_smem_bytes = Variant::dynamic_shared_bytes;
    const size_t total_smem_bytes = Variant::total_shared_bytes;

    std::lock_guard<std::mutex> lock(mutex);

    auto it = cache.find(device);

    if (it != cache.end() &&
        it->second.configured &&
        it->second.dynamic_smem_bytes == dynamic_smem_bytes) {
        return;
    }

    system::runtime::set_device(device);

    cudaDeviceProp prop{};

    system::runtime::check_cuda(
        cudaGetDeviceProperties(&prop, device),
        "cudaGetDeviceProperties");

    if (total_smem_bytes >
        static_cast<size_t>(prop.sharedMemPerBlockOptin)) {
        throw std::runtime_error(
            "tma_two_gpu_peer_allreduce_configure_kernel_once: requested shared memory exceeds opt-in limit");
    }

    if (total_smem_bytes >
        static_cast<size_t>(prop.sharedMemPerBlock)) {
        system::runtime::check_cuda(
            cudaFuncSetAttribute(
                comm::window_task_executor_kernel_sm90<
                    ReduceApply,
                    ChunkBytes,
                    StageDepth,
                    kMaxWindowTasks>,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                static_cast<int>(dynamic_smem_bytes)),
            "cudaFuncSetAttribute(MaxDynamicSharedMemorySize)");
    }

    system::runtime::check_cuda(
        cudaFuncSetAttribute(
            comm::window_task_executor_kernel_sm90<
                ReduceApply,
                ChunkBytes,
                StageDepth,
                kMaxWindowTasks>,
            cudaFuncAttributePreferredSharedMemoryCarveout,
            100),
        "cudaFuncSetAttribute(PreferredSharedMemoryCarveout)");

    cache[device] = {true, dynamic_smem_bytes};
}

template <
    typename ReduceApply,
    int ElemBytes,
    int ChunkBytes,
    int StageDepth>
cudaError_t launch_rank_kernel_sm90(
    const void* local_in,
    void* local_buf,
    void* peer_buf,
    size_t count,
    int rank,
    int dev0,
    int dev1,
    cudaStream_t stream,
    int* local_ready_signal,
    const int* peer_ready_signal,
    int collective_epoch,
    comm::LaunchConfig launch_config) {
    using Variant = comm::TmaPipelineVariant<ChunkBytes, StageDepth>;

    if (!comm::launch_config_valid(launch_config)) {
        return cudaErrorInvalidValue;
    }

    if (launch_config.plan_kind ==
        comm::AllreducePlanKind::OverlapFastGmem) {
        if (!comm::launch_config_valid_for_overlap(launch_config)) {
            return cudaErrorInvalidValue;
        }
    }

    if (launch_config.chunk_bytes != Variant::chunk_bytes ||
        launch_config.stage_depth != Variant::stage_depth) {
        return cudaErrorInvalidValue;
    }

    const int device = (rank == 0) ? dev0 : dev1;

    const size_t total_bytes = count * static_cast<size_t>(ElemBytes);

    const int num_chunks =
        comm::utils::ceil_div_int64_to_int(
            total_bytes,
            Variant::chunk_bytes);

    const int num_windows =
        comm::utils::window_count_for_chunks(
            num_chunks,
            launch_config.window_chunks);

    const bool needs_rendezvous =
        local_ready_signal != nullptr &&
        peer_ready_signal != nullptr &&
        collective_epoch > 0;

    comm::WindowTaskExecutorPlan<kMaxWindowTasks> plan{};

    int num_blocks = 0;
    bool plan_ok = false;

    system::runtime::set_device(device);

    switch (launch_config.plan_kind) {
        case comm::AllreducePlanKind::TmaCopy: {
            const bool out_of_place = (local_in != local_buf);

            const int owned_windows =
                out_of_place
                    ? num_windows
                    : comm::utils::rank_window_count(num_windows, rank);

            const int ctas_per_rank =
                comm::utils::cta_count_for_windows(
                    owned_windows,
                    launch_config.max_ctas);

            num_blocks =
                needs_rendezvous
                    ? std::max(1, ctas_per_rank)
                    : ctas_per_rank;

            if (num_blocks <= 0) {
                return cudaSuccess;
            }

            if (ctas_per_rank * kSeqTasksPerCTA > kMaxWindowTasks) {
                return cudaErrorInvalidValue;
            }

            plan_ok =
                out_of_place
                    ? build_tma_copy_out_of_place_plan(
                          &plan,
                          local_in,
                          local_buf,
                          peer_buf,
                          total_bytes,
                          num_windows,
                          ctas_per_rank,
                          launch_config.window_chunks)
                    : build_tma_copy_inplace_plan(
                          &plan,
                          local_in,
                          local_buf,
                          peer_buf,
                          total_bytes,
                          rank,
                          num_windows,
                          ctas_per_rank,
                          launch_config.window_chunks);
            break;
        }

        case comm::AllreducePlanKind::SeqFastGmem: {
            const int owned_windows =
                comm::utils::rank_window_count(num_windows, rank);

            const int ctas_per_rank =
                comm::utils::cta_count_for_windows(
                    owned_windows,
                    launch_config.max_ctas);

            num_blocks =
                needs_rendezvous
                    ? std::max(1, ctas_per_rank)
                    : ctas_per_rank;

            if (num_blocks <= 0) {
                return cudaSuccess;
            }

            if (ctas_per_rank * kSeqTasksPerCTA > kMaxWindowTasks) {
                return cudaErrorInvalidValue;
            }

            plan_ok =
                build_seq_fast_gmem_plan(
                    &plan,
                    local_in,
                    local_buf,
                    peer_buf,
                    total_bytes,
                    rank,
                    num_windows,
                    ctas_per_rank,
                    launch_config.window_chunks);
            break;
        }

        case comm::AllreducePlanKind::OverlapFastGmem: {
            const int owned_windows =
                comm::utils::rank_window_count(num_windows, rank);

            const int pair_count =
                overlap_pair_count_for_windows(
                    owned_windows,
                    launch_config.max_ctas);

            num_blocks =
                pair_count * TMA_TWO_GPU_PEER_OVERLAP_BLOCKS_PER_CTA;

            if (needs_rendezvous) {
                num_blocks = std::max(1, num_blocks);
            }

            if (num_blocks <= 0) {
                return cudaSuccess;
            }

            if (pair_count *
                    TMA_TWO_GPU_PEER_OVERLAP_BLOCKS_PER_CTA *
                    kOverlapTasksPerCTA >
                kMaxWindowTasks) {
                return cudaErrorInvalidValue;
            }

            int* ready_flags = nullptr;

            if (owned_windows > 0) {
                ready_flags =
                    ensure_signal_capacity(
                        device,
                        static_cast<size_t>(owned_windows));

                system::runtime::check_cuda(
                    cudaMemsetAsync(
                        ready_flags,
                        0,
                        static_cast<size_t>(owned_windows) * sizeof(int),
                        stream),
                    "cudaMemsetAsync(overlap window ready flags)");
            }

            plan_ok =
                build_overlap_fast_gmem_plan(
                    &plan,
                    local_in,
                    local_buf,
                    peer_buf,
                    total_bytes,
                    rank,
                    num_windows,
                    pair_count,
                    launch_config.window_chunks,
                    ready_flags);
            break;
        }

        default:
            return cudaErrorInvalidValue;
    }

    if (!plan_ok) {
        return cudaErrorInvalidValue;
    }

    configure_kernel_once_for<
        ReduceApply,
        ElemBytes,
        ChunkBytes,
        StageDepth>(device);

    system::runtime::set_device(device);

    return comm::launch_window_task_executor_sm90<
        ReduceApply,
        ChunkBytes,
        StageDepth,
        kMaxWindowTasks>(
            plan,
            num_blocks,
            launch_config.threads,
            local_ready_signal,
            peer_ready_signal,
            collective_epoch,
            stream);
}

template <
    typename ReduceOp,
    int ElemBytes,
    int ChunkBytes,
    int StageDepth>
cudaError_t launch_reduce_op_sm90(
    const void* local_in,
    void* local_buf,
    void* peer_buf,
    size_t count,
    int rank,
    int dev0,
    int dev1,
    cudaStream_t stream,
    int* local_ready_signal,
    const int* peer_ready_signal,
    int collective_epoch,
    comm::LaunchConfig launch_config) {
    using Variant = comm::TmaPipelineVariant<ChunkBytes, StageDepth>;

    using ReduceApply = comm::PipelineTMAReduce<
        Variant::stage_depth,
        Variant::stage_gap,
        ReduceOp>;

    return launch_rank_kernel_sm90<
        ReduceApply,
        ElemBytes,
        ChunkBytes,
        StageDepth>(
            local_in,
            local_buf,
            peer_buf,
            count,
            rank,
            dev0,
            dev1,
            stream,
            local_ready_signal,
            peer_ready_signal,
            collective_epoch,
            launch_config);
}

template <
    typename ReduceOp,
    int ElemBytes,
    int ChunkBytes,
    int StageDepth>
void configure_reduce_op_sm90(int device) {
    using Variant = comm::TmaPipelineVariant<ChunkBytes, StageDepth>;

    using ReduceApply = comm::PipelineTMAReduce<
        Variant::stage_depth,
        Variant::stage_gap,
        ReduceOp>;

    configure_kernel_once_for<
        ReduceApply,
        ElemBytes,
        ChunkBytes,
        StageDepth>(device);
}

template <int ChunkBytes, int StageDepth>
cudaError_t dispatch_rank_kernel_variant_sm90(
    const void* local_in,
    void* local_buf,
    void* peer_buf,
    size_t count,
    oo_dtype_t dtype,
    oo_reduce_op_t op,
    int rank,
    int dev0,
    int dev1,
    cudaStream_t stream,
    int* local_ready_signal,
    const int* peer_ready_signal,
    int collective_epoch,
    comm::LaunchConfig launch_config) {
    if (dtype == OO_DTYPE_FLOAT16) {
        if (op == OO_REDUCE_ADD || op == OO_REDUCE_SUM) {
            return launch_reduce_op_sm90<
                comm::PipelineReduceAddNoFtzF16,
                static_cast<int>(sizeof(half)),
                ChunkBytes,
                StageDepth>(
                    local_in,
                    local_buf,
                    peer_buf,
                    count,
                    rank,
                    dev0,
                    dev1,
                    stream,
                    local_ready_signal,
                    peer_ready_signal,
                    collective_epoch,
                    launch_config);
        }

        if (op == OO_REDUCE_MIN) {
            return launch_reduce_op_sm90<
                comm::PipelineReduceMinF16,
                static_cast<int>(sizeof(half)),
                ChunkBytes,
                StageDepth>(
                    local_in,
                    local_buf,
                    peer_buf,
                    count,
                    rank,
                    dev0,
                    dev1,
                    stream,
                    local_ready_signal,
                    peer_ready_signal,
                    collective_epoch,
                    launch_config);
        }

        if (op == OO_REDUCE_MAX) {
            return launch_reduce_op_sm90<
                comm::PipelineReduceMaxF16,
                static_cast<int>(sizeof(half)),
                ChunkBytes,
                StageDepth>(
                    local_in,
                    local_buf,
                    peer_buf,
                    count,
                    rank,
                    dev0,
                    dev1,
                    stream,
                    local_ready_signal,
                    peer_ready_signal,
                    collective_epoch,
                    launch_config);
        }
    }

    if (dtype == OO_DTYPE_BFLOAT16) {
        if (op == OO_REDUCE_ADD || op == OO_REDUCE_SUM) {
            return launch_reduce_op_sm90<
                comm::PipelineReduceAddBF16,
                static_cast<int>(sizeof(__nv_bfloat16)),
                ChunkBytes,
                StageDepth>(
                    local_in,
                    local_buf,
                    peer_buf,
                    count,
                    rank,
                    dev0,
                    dev1,
                    stream,
                    local_ready_signal,
                    peer_ready_signal,
                    collective_epoch,
                    launch_config);
        }

        if (op == OO_REDUCE_MIN) {
            return launch_reduce_op_sm90<
                comm::PipelineReduceMinBF16,
                static_cast<int>(sizeof(__nv_bfloat16)),
                ChunkBytes,
                StageDepth>(
                    local_in,
                    local_buf,
                    peer_buf,
                    count,
                    rank,
                    dev0,
                    dev1,
                    stream,
                    local_ready_signal,
                    peer_ready_signal,
                    collective_epoch,
                    launch_config);
        }

        if (op == OO_REDUCE_MAX) {
            return launch_reduce_op_sm90<
                comm::PipelineReduceMaxBF16,
                static_cast<int>(sizeof(__nv_bfloat16)),
                ChunkBytes,
                StageDepth>(
                    local_in,
                    local_buf,
                    peer_buf,
                    count,
                    rank,
                    dev0,
                    dev1,
                    stream,
                    local_ready_signal,
                    peer_ready_signal,
                    collective_epoch,
                    launch_config);
        }
    }

    if (dtype == OO_DTYPE_FLOAT32) {
        if (op == OO_REDUCE_ADD || op == OO_REDUCE_SUM) {
            return launch_reduce_op_sm90<
                comm::PipelineReduceAddF32,
                static_cast<int>(sizeof(float)),
                ChunkBytes,
                StageDepth>(
                    local_in,
                    local_buf,
                    peer_buf,
                    count,
                    rank,
                    dev0,
                    dev1,
                    stream,
                    local_ready_signal,
                    peer_ready_signal,
                    collective_epoch,
                    launch_config);
        }
    }

    return cudaErrorInvalidValue;
}

cudaError_t dispatch_rank_kernel_sm90(
    const void* local_in,
    void* local_buf,
    void* peer_buf,
    size_t count,
    oo_dtype_t dtype,
    oo_reduce_op_t op,
    int rank,
    int dev0,
    int dev1,
    cudaStream_t stream,
    int* local_ready_signal,
    const int* peer_ready_signal,
    int collective_epoch,
    comm::LaunchConfig launch_config) {
#define OO_TRY_VARIANT(CHUNK_BYTES_VALUE, STAGE_DEPTH_VALUE)                 \
    if (launch_config.chunk_bytes == (CHUNK_BYTES_VALUE) &&                  \
        launch_config.stage_depth == (STAGE_DEPTH_VALUE)) {                  \
        return dispatch_rank_kernel_variant_sm90<                            \
            (CHUNK_BYTES_VALUE),                                             \
            (STAGE_DEPTH_VALUE)>(                                            \
                local_in,                                                    \
                local_buf,                                                   \
                peer_buf,                                                    \
                count,                                                       \
                dtype,                                                       \
                op,                                                          \
                rank,                                                        \
                dev0,                                                        \
                dev1,                                                        \
                stream,                                                      \
                local_ready_signal,                                          \
                peer_ready_signal,                                           \
                collective_epoch,                                            \
                launch_config);                                              \
    }

    OOVERLAP_TMA_TWO_GPU_PEER_FOR_EACH_VARIANT(OO_TRY_VARIANT)

#undef OO_TRY_VARIANT

    return cudaErrorInvalidValue;
}

template <int ChunkBytes, int StageDepth>
void configure_dispatch_variant_sm90(
    oo_dtype_t dtype,
    oo_reduce_op_t op,
    int device) {
    if (dtype == OO_DTYPE_FLOAT16) {
        if (op == OO_REDUCE_ADD || op == OO_REDUCE_SUM) {
            configure_reduce_op_sm90<
                comm::PipelineReduceAddNoFtzF16,
                static_cast<int>(sizeof(half)),
                ChunkBytes,
                StageDepth>(device);
            return;
        }

        if (op == OO_REDUCE_MIN) {
            configure_reduce_op_sm90<
                comm::PipelineReduceMinF16,
                static_cast<int>(sizeof(half)),
                ChunkBytes,
                StageDepth>(device);
            return;
        }

        if (op == OO_REDUCE_MAX) {
            configure_reduce_op_sm90<
                comm::PipelineReduceMaxF16,
                static_cast<int>(sizeof(half)),
                ChunkBytes,
                StageDepth>(device);
            return;
        }
    }

    if (dtype == OO_DTYPE_BFLOAT16) {
        if (op == OO_REDUCE_ADD || op == OO_REDUCE_SUM) {
            configure_reduce_op_sm90<
                comm::PipelineReduceAddBF16,
                static_cast<int>(sizeof(__nv_bfloat16)),
                ChunkBytes,
                StageDepth>(device);
            return;
        }

        if (op == OO_REDUCE_MIN) {
            configure_reduce_op_sm90<
                comm::PipelineReduceMinBF16,
                static_cast<int>(sizeof(__nv_bfloat16)),
                ChunkBytes,
                StageDepth>(device);
            return;
        }

        if (op == OO_REDUCE_MAX) {
            configure_reduce_op_sm90<
                comm::PipelineReduceMaxBF16,
                static_cast<int>(sizeof(__nv_bfloat16)),
                ChunkBytes,
                StageDepth>(device);
            return;
        }
    }

    if (dtype == OO_DTYPE_FLOAT32) {
        if (op == OO_REDUCE_ADD || op == OO_REDUCE_SUM) {
            configure_reduce_op_sm90<
                comm::PipelineReduceAddF32,
                static_cast<int>(sizeof(float)),
                ChunkBytes,
                StageDepth>(device);
            return;
        }
    }

    throw std::invalid_argument(
        "tma_two_gpu_peer_allreduce_configure_kernel_once: unsupported dtype/op");
}

void configure_dispatch_sm90(
    oo_dtype_t dtype,
    oo_reduce_op_t op,
    int device) {
    /*
     * Compatibility configure path: configure the default selected variant.
     * Runtime launches still configure the exact task-executor variant.
     */
    configure_dispatch_variant_sm90<
        TMA_TWO_GPU_PEER_DEFAULT_CHUNK_BYTES,
        TMA_TWO_GPU_PEER_DEFAULT_STAGE_DEPTH>(
            dtype,
            op,
            device);
}

} // namespace

void tma_two_gpu_peer_allreduce_configure_kernel_once(
    oo_dtype_t dtype,
    oo_reduce_op_t op,
    int device) {
    configure_dispatch_sm90(dtype, op, device);
}

cudaError_t enqueue_tma_two_gpu_peer_allreduce_rank_sm90(
    const void* local_in,
    void* local_buf,
    void* peer_buf,
    size_t count,
    oo_dtype_t dtype,
    oo_reduce_op_t op,
    int rank,
    int dev0,
    int dev1,
    cudaStream_t stream,
    int* local_ready_signal,
    const int* peer_ready_signal,
    int collective_epoch,
    comm::LaunchConfig launch_config) {
    if (local_in == nullptr || local_buf == nullptr || peer_buf == nullptr) {
        return cudaErrorInvalidDevicePointer;
    }
    if (count == 0) {
        return cudaErrorInvalidValue;
    }
    if (rank != 0 && rank != 1) {
        return cudaErrorInvalidValue;
    }
    if (dev0 == dev1) {
        return cudaErrorInvalidValue;
    }
    if (dtype_size_bytes(dtype) == 0) {
        return cudaErrorInvalidValue;
    }
    if (!comm::launch_config_valid(launch_config)) {
        return cudaErrorInvalidValue;
    }

    if (launch_config.plan_kind ==
        comm::AllreducePlanKind::OverlapFastGmem) {
        if (!comm::launch_config_valid_for_overlap(launch_config)) {
            return cudaErrorInvalidValue;
        }
    }

    return dispatch_rank_kernel_sm90(
        local_in,
        local_buf,
        peer_buf,
        count,
        dtype,
        op,
        rank,
        dev0,
        dev1,
        stream,
        local_ready_signal,
        peer_ready_signal,
        collective_epoch,
        launch_config);
}

} // namespace ooverlap
