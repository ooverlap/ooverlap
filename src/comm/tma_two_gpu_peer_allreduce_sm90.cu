#include "comm/tma_two_gpu_peer_allreduce_sm90.h"

#include "ooverlap/system/runtime_utils.cuh"

#include "comm/launch_config.h"
#include "comm/params.h"
#include "comm/pipeline_tma_reduce.h"
#include "comm/tma_variant_config.h"
#include "comm/utils.h"
#include "comm/window_pipeline_sm90.cuh"

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

template <
    typename ReduceApply,
    int ElemBytes,
    int ChunkBytes,
    int StageDepth>
__global__ void tma_two_gpu_allreduce_rank_kernel_sm90(
    const void* local_in,
    void* local_buf,
    void* peer_buf,
    size_t count,
    int rank,
    int* local_ready_signal,
    const int* peer_ready_signal,
    int collective_epoch,
    int ctas_per_rank,
    int window_chunks) {
    using Variant = comm::TmaPipelineVariant<ChunkBytes, StageDepth>;

    (void)local_in;

    comm::window_pipeline::wait_for_collective_ready(
        local_ready_signal,
        peer_ready_signal,
        collective_epoch);

    const int cta_idx = static_cast<int>(blockIdx.x);

    if (ctas_per_rank <= 0 || cta_idx >= ctas_per_rank) {
        return;
    }

    const size_t total_bytes = count * static_cast<size_t>(ElemBytes);

    const int num_chunks =
        comm::utils::ceil_div_int64_to_int(
            total_bytes,
            Variant::chunk_bytes);

    const int num_windows =
        comm::utils::window_count_for_chunks(
            num_chunks,
            window_chunks);

    const comm::utils::WindowRange rank_range =
        comm::utils::rank_window_range(num_windows, rank);

    const comm::utils::WindowRange cta_range =
        comm::utils::cta_window_range(
            cta_idx,
            ctas_per_rank,
            rank_range);

    if (cta_range.begin >= cta_range.end) {
        return;
    }

    extern __shared__ uint4 shared_storage_u4[];

    unsigned char* shared_raw =
        reinterpret_cast<unsigned char*>(shared_storage_u4);

    __shared__ sync::semaphore barriers[Variant::barrier_count];

    comm::window_pipeline::run_window_range<
        Variant::stage_depth,
        Variant::stage_gap,
        Variant::chunk_bytes,
        ReduceApply>(
            peer_buf,
            local_buf,
            total_bytes,
            cta_range.begin,
            cta_range.end,
            window_chunks,
            shared_raw,
            barriers);

    comm::window_pipeline::copy_window_range_tma<
        Variant::stage_depth,
        Variant::stage_gap,
        Variant::chunk_bytes>(
            local_buf,
            peer_buf,
            total_bytes,
            cta_range.begin,
            cta_range.end,
            window_chunks,
            shared_raw,
            barriers);
}

template <
    typename ReduceApply,
    int ElemBytes,
    int ChunkBytes,
    int StageDepth>
void configure_kernel_once_for(int device) {
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

    if (total_smem_bytes > static_cast<size_t>(prop.sharedMemPerBlockOptin)) {
        throw std::runtime_error(
            "tma_two_gpu_peer_allreduce_configure_kernel_once: requested shared memory exceeds opt-in limit");
    }

    if (total_smem_bytes > static_cast<size_t>(prop.sharedMemPerBlock)) {
        system::runtime::check_cuda(
            cudaFuncSetAttribute(
                tma_two_gpu_allreduce_rank_kernel_sm90<
                    ReduceApply,
                    ElemBytes,
                    ChunkBytes,
                    StageDepth>,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                static_cast<int>(dynamic_smem_bytes)),
            "cudaFuncSetAttribute(MaxDynamicSharedMemorySize)");

        system::runtime::check_cuda(
            cudaFuncSetAttribute(
                tma_two_gpu_allreduce_rank_kernel_sm90<
                    ReduceApply,
                    ElemBytes,
                    ChunkBytes,
                    StageDepth>,
                cudaFuncAttributePreferredSharedMemoryCarveout,
                100),
            "cudaFuncSetAttribute(PreferredSharedMemoryCarveout)");
    }

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

    const int owned_windows =
        comm::utils::rank_window_count(num_windows, rank);

    const int ctas_per_rank =
        comm::utils::cta_count_for_windows(
            owned_windows,
            launch_config.max_ctas);

    const bool needs_rendezvous =
        local_ready_signal != nullptr &&
        peer_ready_signal != nullptr &&
        collective_epoch > 0;

    const int num_blocks =
        needs_rendezvous ? std::max(1, ctas_per_rank) : ctas_per_rank;

    if (num_blocks <= 0) {
        return cudaSuccess;
    }

    configure_kernel_once_for<
        ReduceApply,
        ElemBytes,
        ChunkBytes,
        StageDepth>(device);

    system::runtime::set_device(device);

    tma_two_gpu_allreduce_rank_kernel_sm90<
        ReduceApply,
        ElemBytes,
        ChunkBytes,
        StageDepth><<<
            num_blocks,
            launch_config.threads,
            Variant::dynamic_shared_bytes,
            stream>>>(
                local_in,
                local_buf,
                peer_buf,
                count,
                rank,
                local_ready_signal,
                peer_ready_signal,
                collective_epoch,
                ctas_per_rank,
                launch_config.window_chunks);

    return cudaGetLastError();
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
        if (op == OO_REDUCE_ADD) {
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
        if (op == OO_REDUCE_ADD) {
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
        if (op == OO_REDUCE_ADD) {
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
        if (op == OO_REDUCE_ADD) {
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
        if (op == OO_REDUCE_ADD) {
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
        if (op == OO_REDUCE_ADD) {
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
     * This compatibility configure function only configures the default
     * pipeline variant. Launch-time dispatch still configures the exact
     * requested runtime-selected variant.
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
