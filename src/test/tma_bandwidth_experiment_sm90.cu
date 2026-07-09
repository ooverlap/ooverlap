#include "test/tma_bandwidth_experiment_sm90.h"

#include "comm/params.h"
#include "comm/pipeline/pipeline_stage.h"
#include "comm/pipeline/pipeline_tma_copy_batch.h"
#include "comm/pipeline/pipeline_tma_load.h"
#include "comm/pipeline/pipeline_tma_reduce_batch.h"

#include "ooverlap/system/peer_buffer.cuh"
#include "ooverlap/system/runtime_utils.cuh"
#include "ooverlap/testing/checks.cuh"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace ooverlap {
namespace {

constexpr int kTmaThreads = TMA_TWO_GPU_PEER_DEFAULT_THREADS;

constexpr int kChunkBytes = TMA_TWO_GPU_PEER_CHUNK_BYTES;
constexpr int kStageDepth = TMA_TWO_GPU_PEER_DEFAULT_STAGE_DEPTH;
constexpr int kFillDepth = TMA_TWO_GPU_PEER_DEFAULT_STAGE_GAP;
constexpr int kBarrierCount = TMA_TWO_GPU_PEER_BARRIER_COUNT;

constexpr size_t kTmaSmemBytes =
    static_cast<size_t>(kStageDepth) * static_cast<size_t>(kChunkBytes);

static_assert(kStageDepth > 0, "stage depth must be positive");
static_assert(kFillDepth > 0, "fill depth must be positive");
static_assert(kFillDepth <= kStageDepth, "fill depth must be <= stage depth");
static_assert((kChunkBytes % sizeof(uint4)) == 0, "chunk bytes must be 16B aligned");

enum BatchExperimentId {
    kBatchExperimentReduceTwoToOne = 0,
    kBatchExperimentCopyOneToTwo = 1,
    kBatchExperimentReduceOneToTwo = 2,
};

enum BatchMethodId {
    kBatchMethodReduceSequentialAllCtas = 0,
    kBatchMethodReduceSplitCtasGpuScope = 1,
    kBatchMethodCopySequentialAllCtas = 2,
    kBatchMethodCopyFanoutOneLoadTwoStores = 3,
    kBatchMethodReduceSplitCtasOppositeDirectionsGpuScope = 4,
    kBatchMethodReduceFanoutSequentialAllCtas = 5,
    kBatchMethodReduceFanoutOneLoadTwoReducesGpuScope = 6,
};

struct ChunkRange {
    int start_chunk = 0;
    int chunk_count = 0;
};

struct BatchExperimentBuffers {
    system::mapped_peer_buffer reduce_src0_peer{};
    system::mapped_peer_buffer reduce_src1_peer{};
    system::mapped_peer_buffer reduce_dst_local{};

    system::mapped_peer_buffer copy_src_local{};
    system::mapped_peer_buffer copy_dst0_peer{};
    system::mapped_peer_buffer copy_dst1_peer{};

    system::mapped_peer_buffer reduce_fanout_src_local{};
    system::mapped_peer_buffer reduce_fanout_dst0_peer{};
    system::mapped_peer_buffer reduce_fanout_dst1_peer{};
};

__host__ __device__ __forceinline__ int ceil_div_size_to_int(
    size_t x,
    size_t y) {
    return static_cast<int>((x + y - 1) / y);
}

__host__ __device__ __forceinline__ ChunkRange make_block_chunk_range(
    int block_idx,
    int num_blocks,
    int num_chunks) {
    const int chunks_per_block =
        (num_chunks + num_blocks - 1) / num_blocks;

    ChunkRange range{};
    range.start_chunk = block_idx * chunks_per_block;
    range.chunk_count = chunks_per_block;

    if (range.start_chunk >= num_chunks) {
        range.chunk_count = 0;
        return range;
    }

    const int remaining = num_chunks - range.start_chunk;

    if (range.chunk_count > remaining) {
        range.chunk_count = remaining;
    }

    return range;
}

__host__ __device__ __forceinline__ ChunkRange make_block_chunk_range_backward(
    int block_idx,
    int num_blocks,
    int num_chunks) {
    const int reverse_block_idx =
        num_blocks - 1 - block_idx;

    return make_block_chunk_range(
        reverse_block_idx,
        num_blocks,
        num_chunks);
}

__host__ __device__ __forceinline__ int chunk_for_iter(
    ChunkRange range,
    int iter,
    bool reverse_chunks) {
    return reverse_chunks
        ? range.start_chunk + (range.chunk_count - 1 - iter)
        : range.start_chunk + iter;
}

__host__ __forceinline__ bool is_aligned_16_host(
    const void* ptr) {
    return (
        (reinterpret_cast<uintptr_t>(ptr) &
         static_cast<uintptr_t>(sizeof(uint4) - 1)) == 0);
}

__host__ __forceinline__ bool is_aligned_16_size_host(
    size_t x) {
    return ((x & static_cast<size_t>(sizeof(uint4) - 1)) == 0);
}

__host__ __forceinline__ cudaError_t validate_ptr_and_size(
    const void* ptr,
    size_t bytes) {
    if (bytes == 0) {
        return cudaSuccess;
    }

    if (ptr == nullptr) {
        return cudaErrorInvalidValue;
    }

    if (!is_aligned_16_host(ptr) ||
        !is_aligned_16_size_host(bytes)) {
        return cudaErrorInvalidValue;
    }

    return cudaSuccess;
}

__host__ __forceinline__ cudaError_t validate_tma_two_ptrs(
    const void* a,
    const void* b,
    size_t bytes) {
    cudaError_t err = validate_ptr_and_size(a, bytes);
    if (err != cudaSuccess) {
        return err;
    }

    return validate_ptr_and_size(b, bytes);
}

__host__ __forceinline__ cudaError_t validate_tma_three_ptrs(
    const void* a,
    const void* b,
    const void* c,
    size_t bytes) {
    cudaError_t err = validate_ptr_and_size(a, bytes);
    if (err != cudaSuccess) {
        return err;
    }

    err = validate_ptr_and_size(b, bytes);
    if (err != cudaSuccess) {
        return err;
    }

    return validate_ptr_and_size(c, bytes);
}

template <size_t ChunkBytes>
__device__ __forceinline__ size_t chunk_offset_bytes(
    int chunk) {
    return static_cast<size_t>(chunk) * ChunkBytes;
}

template <size_t ChunkBytes>
__device__ __forceinline__ comm::pipeline::PipelineStage make_stage_for_abs_chunk(
    const unsigned char* src_bytes,
    unsigned char* dst_bytes,
    size_t total_bytes,
    int abs_chunk,
    int slot,
    unsigned char* shared_raw,
    sync::semaphore* barriers) {
    const size_t offset =
        chunk_offset_bytes<ChunkBytes>(abs_chunk);

    const size_t remaining =
        total_bytes > offset ? total_bytes - offset : 0;

    const size_t bytes =
        remaining < ChunkBytes ? remaining : ChunkBytes;

    return comm::pipeline::make_pipeline_stage(
        comm::pipeline::make_pipeline_chunk(
            src_bytes + offset,
            dst_bytes + offset,
            bytes),
        shared_raw + static_cast<size_t>(slot) * ChunkBytes,
        &barriers[slot]);
}

template <size_t ChunkBytes>
__device__ __forceinline__ unsigned char* pointer_for_abs_chunk(
    void* base,
    int abs_chunk) {
    return reinterpret_cast<unsigned char*>(base) +
        chunk_offset_bytes<ChunkBytes>(abs_chunk);
}

/*
 * One source buffer to one destination buffer by TMA reduce.
 *
 * This is the building block for:
 *   - sequential baseline: launch it twice with all CTAs
 *   - split-CTA experiment: one kernel maps half of the CTAs to src0 and half
 *     to src1 while reducing to the same destination address range.
 */
template <tma::TmaReduceScope Scope, bool ReverseChunks>
__device__ __forceinline__ void run_tma_reduce_one_range_thread0_only(
    const void* src_base,
    void* dst_base,
    size_t total_bytes,
    ChunkRange range,
    unsigned char* shared_raw,
    sync::semaphore* barriers) {
    if (threadIdx.x != 0) {
        return;
    }

    if (range.chunk_count <= 0 || total_bytes == 0) {
        return;
    }

    const unsigned char* src_bytes =
        reinterpret_cast<const unsigned char*>(src_base);

    unsigned char* dst_bytes =
        reinterpret_cast<unsigned char*>(dst_base);

    comm::pipeline::PipelineTMALoad load{};

    using ReduceBatch =
        comm::pipeline::PipelineTMAReduceBatch<
            kStageDepth,
            kFillDepth,
            comm::pipeline::PipelineReduceBatchAddNoFtzF16,
            Scope>;

    ReduceBatch reduce{};

    for (int warm = 0; warm < kFillDepth; ++warm) {
        if (warm >= range.chunk_count) {
            break;
        }

        const int abs_chunk =
            chunk_for_iter(range, warm, ReverseChunks);

        comm::pipeline::PipelineStage stage =
            make_stage_for_abs_chunk<kChunkBytes>(
                src_bytes,
                dst_bytes,
                total_bytes,
                abs_chunk,
                warm,
                shared_raw,
                barriers);

        load.issue(&stage);
    }

    for (int iter = 0; iter < range.chunk_count; ++iter) {
        const int abs_chunk =
            chunk_for_iter(range, iter, ReverseChunks);

        const int cur_slot =
            iter % kStageDepth;

        comm::pipeline::PipelineStage cur_stage =
            make_stage_for_abs_chunk<kChunkBytes>(
                src_bytes,
                dst_bytes,
                total_bytes,
                abs_chunk,
                cur_slot,
                shared_raw,
                barriers);

        load.wait_ready(&cur_stage);

        const int future_iter =
            iter + kFillDepth;

        if (future_iter < range.chunk_count) {
            const int future_abs_chunk =
                chunk_for_iter(range, future_iter, ReverseChunks);

            const int future_slot =
                future_iter % kStageDepth;

            comm::pipeline::PipelineStage future_stage =
                make_stage_for_abs_chunk<kChunkBytes>(
                    src_bytes,
                    dst_bytes,
                    total_bytes,
                    future_abs_chunk,
                    future_slot,
                    shared_raw,
                    barriers);

            if (iter >= kFillDepth) {
                reduce.wait_before_stage_reuse();
            }

            load.issue(&future_stage);
        }

        reduce.issue_bulk_commit(&cur_stage);
    }

    reduce.wait_complete();
    __threadfence_system();
}

/*
 * One source buffer to two destination buffers:
 *   load once into shared memory
 *   issue two shared->global TMA stores
 *   commit once
 */
__device__ __forceinline__ void run_tma_copy_fanout2_range_thread0_only(
    const void* src_base,
    void* dst0_base,
    void* dst1_base,
    size_t total_bytes,
    ChunkRange range,
    unsigned char* shared_raw,
    sync::semaphore* barriers) {
    if (threadIdx.x != 0) {
        return;
    }

    if (range.chunk_count <= 0 || total_bytes == 0) {
        return;
    }

    const unsigned char* src_bytes =
        reinterpret_cast<const unsigned char*>(src_base);

    unsigned char* dst0_bytes =
        reinterpret_cast<unsigned char*>(dst0_base);

    comm::pipeline::PipelineTMALoad load{};

    using CopyBatch =
        comm::pipeline::PipelineTMACopyBatch<
            kStageDepth,
            kFillDepth>;

    CopyBatch copy{};

    for (int warm = 0; warm < kFillDepth; ++warm) {
        if (warm >= range.chunk_count) {
            break;
        }

        const int abs_chunk =
            range.start_chunk + warm;

        comm::pipeline::PipelineStage stage =
            make_stage_for_abs_chunk<kChunkBytes>(
                src_bytes,
                dst0_bytes,
                total_bytes,
                abs_chunk,
                warm,
                shared_raw,
                barriers);

        load.issue(&stage);
    }

    for (int iter = 0; iter < range.chunk_count; ++iter) {
        const int abs_chunk =
            range.start_chunk + iter;

        const int cur_slot =
            iter % kStageDepth;

        comm::pipeline::PipelineStage cur_stage =
            make_stage_for_abs_chunk<kChunkBytes>(
                src_bytes,
                dst0_bytes,
                total_bytes,
                abs_chunk,
                cur_slot,
                shared_raw,
                barriers);

        load.wait_ready(&cur_stage);

        const int future_iter =
            iter + kFillDepth;

        if (future_iter < range.chunk_count) {
            const int future_abs_chunk =
                range.start_chunk + future_iter;

            const int future_slot =
                future_iter % kStageDepth;

            comm::pipeline::PipelineStage future_stage =
                make_stage_for_abs_chunk<kChunkBytes>(
                    src_bytes,
                    dst0_bytes,
                    total_bytes,
                    future_abs_chunk,
                    future_slot,
                    shared_raw,
                    barriers);

            if (iter >= kFillDepth) {
                copy.wait_before_stage_reuse();
            }

            load.issue(&future_stage);
        }

        void* dst0 =
            pointer_for_abs_chunk<kChunkBytes>(dst0_base, abs_chunk);

        void* dst1 =
            pointer_for_abs_chunk<kChunkBytes>(dst1_base, abs_chunk);

        copy.issue_fanout2_commit(&cur_stage, dst0, dst1);
    }

    copy.wait_complete();
    __threadfence_system();
}


/*
 * One source buffer to two destination buffers using TMA reductions:
 *   load once into shared memory
 *   issue two cp.reduce.async.bulk operations
 *   commit once
 */
template <tma::TmaReduceScope Scope>
__device__ __forceinline__ void run_tma_reduce_fanout2_range_thread0_only(
    const void* src_base,
    void* dst0_base,
    void* dst1_base,
    size_t total_bytes,
    ChunkRange range,
    unsigned char* shared_raw,
    sync::semaphore* barriers) {
    if (threadIdx.x != 0) {
        return;
    }

    if (range.chunk_count <= 0 || total_bytes == 0) {
        return;
    }

    const unsigned char* src_bytes =
        reinterpret_cast<const unsigned char*>(src_base);

    unsigned char* dst0_bytes =
        reinterpret_cast<unsigned char*>(dst0_base);

    comm::pipeline::PipelineTMALoad load{};

    using ReduceBatch =
        comm::pipeline::PipelineTMAReduceBatch<
            kStageDepth,
            kFillDepth,
            comm::pipeline::PipelineReduceBatchAddNoFtzF16,
            Scope>;

    ReduceBatch reduce{};

    for (int warm = 0; warm < kFillDepth; ++warm) {
        if (warm >= range.chunk_count) {
            break;
        }

        const int abs_chunk =
            range.start_chunk + warm;

        comm::pipeline::PipelineStage stage =
            make_stage_for_abs_chunk<kChunkBytes>(
                src_bytes,
                dst0_bytes,
                total_bytes,
                abs_chunk,
                warm,
                shared_raw,
                barriers);

        load.issue(&stage);
    }

    for (int iter = 0; iter < range.chunk_count; ++iter) {
        const int abs_chunk =
            range.start_chunk + iter;

        const int cur_slot =
            iter % kStageDepth;

        comm::pipeline::PipelineStage cur_stage =
            make_stage_for_abs_chunk<kChunkBytes>(
                src_bytes,
                dst0_bytes,
                total_bytes,
                abs_chunk,
                cur_slot,
                shared_raw,
                barriers);

        load.wait_ready(&cur_stage);

        const int future_iter =
            iter + kFillDepth;

        if (future_iter < range.chunk_count) {
            const int future_abs_chunk =
                range.start_chunk + future_iter;

            const int future_slot =
                future_iter % kStageDepth;

            comm::pipeline::PipelineStage future_stage =
                make_stage_for_abs_chunk<kChunkBytes>(
                    src_bytes,
                    dst0_bytes,
                    total_bytes,
                    future_abs_chunk,
                    future_slot,
                    shared_raw,
                    barriers);

            if (iter >= kFillDepth) {
                reduce.wait_before_stage_reuse();
            }

            load.issue(&future_stage);
        }

        void* dst0 =
            pointer_for_abs_chunk<kChunkBytes>(dst0_base, abs_chunk);

        void* dst1 =
            pointer_for_abs_chunk<kChunkBytes>(dst1_base, abs_chunk);

        reduce.issue_fence();
        reduce.issue_bulk_op_nofence(&cur_stage, dst0);
        reduce.issue_bulk_op_nofence(&cur_stage, dst1);
        reduce.commit();
    }

    reduce.wait_complete();
    __threadfence_system();
}

__global__ void tma_reduce_add_f16_gpu_scope_kernel(
    const void* src,
    void* dst,
    size_t total_bytes,
    int num_chunks) {
    const ChunkRange range =
        make_block_chunk_range(
            static_cast<int>(blockIdx.x),
            static_cast<int>(gridDim.x),
            num_chunks);

    if (range.chunk_count <= 0) {
        return;
    }

    extern __shared__ uint4 shared_storage_u4[];

    unsigned char* shared_raw =
        reinterpret_cast<unsigned char*>(shared_storage_u4);

    __shared__ sync::semaphore barriers[kBarrierCount];

    run_tma_reduce_one_range_thread0_only<tma::TmaReduceScope::Gpu, false>(
        src,
        dst,
        total_bytes,
        range,
        shared_raw,
        barriers);
}

template <bool OppositeDirections>
__global__ void tma_reduce_add_f16_split2_gpu_scope_kernel(
    const void* src0,
    const void* src1,
    void* dst,
    size_t total_bytes,
    int num_chunks,
    int ctas_for_src0) {
    const int total_ctas =
        static_cast<int>(gridDim.x);

    if (total_ctas < 2) {
        return;
    }

    if (ctas_for_src0 <= 0 || ctas_for_src0 >= total_ctas) {
        return;
    }

    const bool use_src0 =
        static_cast<int>(blockIdx.x) < ctas_for_src0;

    const int local_block =
        use_src0
            ? static_cast<int>(blockIdx.x)
            : static_cast<int>(blockIdx.x) - ctas_for_src0;

    const int local_blocks =
        use_src0
            ? ctas_for_src0
            : total_ctas - ctas_for_src0;

    const void* src =
        use_src0 ? src0 : src1;

    const ChunkRange range =
        (!use_src0 && OppositeDirections)
            ? make_block_chunk_range_backward(
                  local_block,
                  local_blocks,
                  num_chunks)
            : make_block_chunk_range(
                  local_block,
                  local_blocks,
                  num_chunks);

    if (range.chunk_count <= 0) {
        return;
    }

    extern __shared__ uint4 shared_storage_u4[];

    unsigned char* shared_raw =
        reinterpret_cast<unsigned char*>(shared_storage_u4);

    __shared__ sync::semaphore barriers[kBarrierCount];

    if (!use_src0 && OppositeDirections) {
        run_tma_reduce_one_range_thread0_only<tma::TmaReduceScope::Gpu, true>(
            src,
            dst,
            total_bytes,
            range,
            shared_raw,
            barriers);
    } else {
        run_tma_reduce_one_range_thread0_only<tma::TmaReduceScope::Gpu, false>(
            src,
            dst,
            total_bytes,
            range,
            shared_raw,
            barriers);
    }
}

__global__ void tma_copy_one_kernel(
    const void* src,
    void* dst,
    size_t total_bytes,
    int num_chunks) {
    const ChunkRange range =
        make_block_chunk_range(
            static_cast<int>(blockIdx.x),
            static_cast<int>(gridDim.x),
            num_chunks);

    if (range.chunk_count <= 0) {
        return;
    }

    extern __shared__ uint4 shared_storage_u4[];

    unsigned char* shared_raw =
        reinterpret_cast<unsigned char*>(shared_storage_u4);

    __shared__ sync::semaphore barriers[kBarrierCount];

    /*
     * The batched copy pipeline still exposes old behavior as issue_bulk_commit.
     * Use it here so the sequential baseline uses the same implementation family
     * as the fanout path.
     */
    if (threadIdx.x != 0 || total_bytes == 0) {
        return;
    }

    const unsigned char* src_bytes =
        reinterpret_cast<const unsigned char*>(src);

    unsigned char* dst_bytes =
        reinterpret_cast<unsigned char*>(dst);

    comm::pipeline::PipelineTMALoad load{};

    using CopyBatch =
        comm::pipeline::PipelineTMACopyBatch<
            kStageDepth,
            kFillDepth>;

    CopyBatch copy{};

    for (int warm = 0; warm < kFillDepth; ++warm) {
        if (warm >= range.chunk_count) {
            break;
        }

        const int abs_chunk =
            range.start_chunk + warm;

        comm::pipeline::PipelineStage stage =
            make_stage_for_abs_chunk<kChunkBytes>(
                src_bytes,
                dst_bytes,
                total_bytes,
                abs_chunk,
                warm,
                shared_raw,
                barriers);

        load.issue(&stage);
    }

    for (int iter = 0; iter < range.chunk_count; ++iter) {
        const int abs_chunk =
            range.start_chunk + iter;

        const int cur_slot =
            iter % kStageDepth;

        comm::pipeline::PipelineStage cur_stage =
            make_stage_for_abs_chunk<kChunkBytes>(
                src_bytes,
                dst_bytes,
                total_bytes,
                abs_chunk,
                cur_slot,
                shared_raw,
                barriers);

        load.wait_ready(&cur_stage);

        const int future_iter =
            iter + kFillDepth;

        if (future_iter < range.chunk_count) {
            const int future_abs_chunk =
                range.start_chunk + future_iter;

            const int future_slot =
                future_iter % kStageDepth;

            comm::pipeline::PipelineStage future_stage =
                make_stage_for_abs_chunk<kChunkBytes>(
                    src_bytes,
                    dst_bytes,
                    total_bytes,
                    future_abs_chunk,
                    future_slot,
                    shared_raw,
                    barriers);

            if (iter >= kFillDepth) {
                copy.wait_before_stage_reuse();
            }

            load.issue(&future_stage);
        }

        copy.issue_bulk_commit(&cur_stage);
    }

    copy.wait_complete();
    __threadfence_system();
}

__global__ void tma_copy_fanout2_kernel(
    const void* src,
    void* dst0,
    void* dst1,
    size_t total_bytes,
    int num_chunks) {
    const ChunkRange range =
        make_block_chunk_range(
            static_cast<int>(blockIdx.x),
            static_cast<int>(gridDim.x),
            num_chunks);

    if (range.chunk_count <= 0) {
        return;
    }

    extern __shared__ uint4 shared_storage_u4[];

    unsigned char* shared_raw =
        reinterpret_cast<unsigned char*>(shared_storage_u4);

    __shared__ sync::semaphore barriers[kBarrierCount];

    run_tma_copy_fanout2_range_thread0_only(
        src,
        dst0,
        dst1,
        total_bytes,
        range,
        shared_raw,
        barriers);
}


__global__ void tma_reduce_fanout2_gpu_scope_kernel(
    const void* src,
    void* dst0,
    void* dst1,
    size_t total_bytes,
    int num_chunks) {
    const ChunkRange range =
        make_block_chunk_range(
            static_cast<int>(blockIdx.x),
            static_cast<int>(gridDim.x),
            num_chunks);

    if (range.chunk_count <= 0) {
        return;
    }

    extern __shared__ uint4 shared_storage_u4[];

    unsigned char* shared_raw =
        reinterpret_cast<unsigned char*>(shared_storage_u4);

    __shared__ sync::semaphore barriers[kBarrierCount];

    run_tma_reduce_fanout2_range_thread0_only<tma::TmaReduceScope::Gpu>(
        src,
        dst0,
        dst1,
        total_bytes,
        range,
        shared_raw,
        barriers);
}

void configure_one_kernel(
    const void* kernel,
    size_t dynamic_smem_bytes,
    int device,
    const char* name) {
    system::runtime::set_device(device);

    cudaDeviceProp prop{};

    testing::check_cuda(
        cudaGetDeviceProperties(&prop, device),
        "cudaGetDeviceProperties");

    const size_t static_smem_bytes =
        static_cast<size_t>(kBarrierCount) * sizeof(sync::semaphore);

    const size_t total_smem_bytes =
        dynamic_smem_bytes + static_smem_bytes;

    if (total_smem_bytes >
        static_cast<size_t>(prop.sharedMemPerBlockOptin)) {
        throw std::runtime_error(
            std::string(name) +
            ": requested shared memory exceeds opt-in limit");
    }

    if (dynamic_smem_bytes >
        static_cast<size_t>(prop.sharedMemPerBlock)) {
        testing::check_cuda(
            cudaFuncSetAttribute(
                kernel,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                static_cast<int>(dynamic_smem_bytes)),
            "cudaFuncSetAttribute(MaxDynamicSharedMemorySize)");
    }

    testing::check_cuda(
        cudaFuncSetAttribute(
            kernel,
            cudaFuncAttributePreferredSharedMemoryCarveout,
            100),
        "cudaFuncSetAttribute(PreferredSharedMemoryCarveout)");
}

void configure_batch_kernels_once(int device) {
    static bool configured[32] = {};

    if (device >= 0 && device < 32 && configured[device]) {
        return;
    }

    configure_one_kernel(
        reinterpret_cast<const void*>(tma_reduce_add_f16_gpu_scope_kernel),
        kTmaSmemBytes,
        device,
        "tma_reduce_add_f16_gpu_scope_kernel");

    configure_one_kernel(
        reinterpret_cast<const void*>(
            tma_reduce_add_f16_split2_gpu_scope_kernel<false>),
        kTmaSmemBytes,
        device,
        "tma_reduce_add_f16_split2_gpu_scope_kernel<false>");

    configure_one_kernel(
        reinterpret_cast<const void*>(
            tma_reduce_add_f16_split2_gpu_scope_kernel<true>),
        kTmaSmemBytes,
        device,
        "tma_reduce_add_f16_split2_gpu_scope_kernel<true>");

    configure_one_kernel(
        reinterpret_cast<const void*>(tma_copy_one_kernel),
        kTmaSmemBytes,
        device,
        "tma_copy_one_kernel");

    configure_one_kernel(
        reinterpret_cast<const void*>(tma_copy_fanout2_kernel),
        kTmaSmemBytes,
        device,
        "tma_copy_fanout2_kernel");

    configure_one_kernel(
        reinterpret_cast<const void*>(tma_reduce_fanout2_gpu_scope_kernel),
        kTmaSmemBytes,
        device,
        "tma_reduce_fanout2_gpu_scope_kernel");

    if (device >= 0 && device < 32) {
        configured[device] = true;
    }
}

cudaError_t launch_reduce_one_gpu_scope(
    const void* src,
    void* dst,
    size_t bytes,
    int num_blocks,
    cudaStream_t stream) {
    cudaError_t valid =
        validate_tma_two_ptrs(src, dst, bytes);

    if (valid != cudaSuccess) {
        return valid;
    }

    if ((bytes % sizeof(half)) != 0) {
        return cudaErrorInvalidValue;
    }

    const int num_chunks =
        ceil_div_size_to_int(bytes, static_cast<size_t>(kChunkBytes));

    tma_reduce_add_f16_gpu_scope_kernel<<<
        num_blocks,
        kTmaThreads,
        kTmaSmemBytes,
        stream>>>(
            src,
            dst,
            bytes,
            num_chunks);

    return cudaGetLastError();
}

cudaError_t launch_reduce_seq2_gpu_scope(
    const void* src0,
    const void* src1,
    void* dst,
    size_t bytes,
    int num_blocks,
    cudaStream_t stream) {
    cudaError_t valid =
        validate_tma_three_ptrs(src0, src1, dst, bytes);

    if (valid != cudaSuccess) {
        return valid;
    }

    valid =
        launch_reduce_one_gpu_scope(
            src0,
            dst,
            bytes,
            num_blocks,
            stream);

    if (valid != cudaSuccess) {
        return valid;
    }

    return launch_reduce_one_gpu_scope(
        src1,
        dst,
        bytes,
        num_blocks,
        stream);
}

template <bool OppositeDirections>
cudaError_t launch_reduce_split2_gpu_scope(
    const void* src0,
    const void* src1,
    void* dst,
    size_t bytes,
    int num_blocks,
    cudaStream_t stream) {
    cudaError_t valid =
        validate_tma_three_ptrs(src0, src1, dst, bytes);

    if (valid != cudaSuccess) {
        return valid;
    }

    if ((bytes % sizeof(half)) != 0 || num_blocks < 2) {
        return cudaErrorInvalidValue;
    }

    const int num_chunks =
        ceil_div_size_to_int(bytes, static_cast<size_t>(kChunkBytes));

    const int ctas_for_src0 =
        num_blocks / 2;

    tma_reduce_add_f16_split2_gpu_scope_kernel<OppositeDirections><<<
        num_blocks,
        kTmaThreads,
        kTmaSmemBytes,
        stream>>>(
            src0,
            src1,
            dst,
            bytes,
            num_chunks,
            ctas_for_src0);

    return cudaGetLastError();
}

cudaError_t launch_copy_one(
    const void* src,
    void* dst,
    size_t bytes,
    int num_blocks,
    cudaStream_t stream) {
    cudaError_t valid =
        validate_tma_two_ptrs(src, dst, bytes);

    if (valid != cudaSuccess) {
        return valid;
    }

    const int num_chunks =
        ceil_div_size_to_int(bytes, static_cast<size_t>(kChunkBytes));

    tma_copy_one_kernel<<<
        num_blocks,
        kTmaThreads,
        kTmaSmemBytes,
        stream>>>(
            src,
            dst,
            bytes,
            num_chunks);

    return cudaGetLastError();
}

cudaError_t launch_copy_seq2(
    const void* src,
    void* dst0,
    void* dst1,
    size_t bytes,
    int num_blocks,
    cudaStream_t stream) {
    cudaError_t valid =
        validate_tma_three_ptrs(src, dst0, dst1, bytes);

    if (valid != cudaSuccess) {
        return valid;
    }

    valid =
        launch_copy_one(
            src,
            dst0,
            bytes,
            num_blocks,
            stream);

    if (valid != cudaSuccess) {
        return valid;
    }

    return launch_copy_one(
        src,
        dst1,
        bytes,
        num_blocks,
        stream);
}

cudaError_t launch_copy_fanout2(
    const void* src,
    void* dst0,
    void* dst1,
    size_t bytes,
    int num_blocks,
    cudaStream_t stream) {
    cudaError_t valid =
        validate_tma_three_ptrs(src, dst0, dst1, bytes);

    if (valid != cudaSuccess) {
        return valid;
    }

    const int num_chunks =
        ceil_div_size_to_int(bytes, static_cast<size_t>(kChunkBytes));

    tma_copy_fanout2_kernel<<<
        num_blocks,
        kTmaThreads,
        kTmaSmemBytes,
        stream>>>(
            src,
            dst0,
            dst1,
            bytes,
            num_chunks);

    return cudaGetLastError();
}


cudaError_t launch_reduce_fanout_seq2_gpu_scope(
    const void* src,
    void* dst0,
    void* dst1,
    size_t bytes,
    int num_blocks,
    cudaStream_t stream) {
    cudaError_t valid =
        validate_tma_three_ptrs(src, dst0, dst1, bytes);

    if (valid != cudaSuccess) {
        return valid;
    }

    valid =
        launch_reduce_one_gpu_scope(
            src,
            dst0,
            bytes,
            num_blocks,
            stream);

    if (valid != cudaSuccess) {
        return valid;
    }

    return launch_reduce_one_gpu_scope(
        src,
        dst1,
        bytes,
        num_blocks,
        stream);
}

cudaError_t launch_reduce_fanout2_gpu_scope(
    const void* src,
    void* dst0,
    void* dst1,
    size_t bytes,
    int num_blocks,
    cudaStream_t stream) {
    cudaError_t valid =
        validate_tma_three_ptrs(src, dst0, dst1, bytes);

    if (valid != cudaSuccess) {
        return valid;
    }

    if ((bytes % sizeof(half)) != 0) {
        return cudaErrorInvalidValue;
    }

    const int num_chunks =
        ceil_div_size_to_int(bytes, static_cast<size_t>(kChunkBytes));

    tma_reduce_fanout2_gpu_scope_kernel<<<
        num_blocks,
        kTmaThreads,
        kTmaSmemBytes,
        stream>>>(
            src,
            dst0,
            dst1,
            bytes,
            num_chunks);

    return cudaGetLastError();
}

template <typename Launch>
double benchmark_launch_ms(
    int kernel_device,
    int iters,
    int warmup,
    Launch launch) {
    system::runtime::set_device(kernel_device);

    cudaStream_t stream =
        system::runtime::create_stream_on_device(kernel_device);

    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;

    try {
        for (int i = 0; i < warmup; ++i) {
            testing::check_cuda(
                launch(stream),
                "launch warmup");
        }

        testing::check_cuda(
            cudaStreamSynchronize(stream),
            "cudaStreamSynchronize(warmup)");

        testing::check_cuda(
            cudaEventCreate(&start),
            "cudaEventCreate(start)");

        testing::check_cuda(
            cudaEventCreate(&stop),
            "cudaEventCreate(stop)");

        testing::check_cuda(
            cudaEventRecord(start, stream),
            "cudaEventRecord(start)");

        for (int i = 0; i < iters; ++i) {
            testing::check_cuda(
                launch(stream),
                "launch timed");
        }

        testing::check_cuda(
            cudaEventRecord(stop, stream),
            "cudaEventRecord(stop)");

        testing::check_cuda(
            cudaEventSynchronize(stop),
            "cudaEventSynchronize(stop)");

        float total_ms = 0.0f;

        testing::check_cuda(
            cudaEventElapsedTime(&total_ms, start, stop),
            "cudaEventElapsedTime");

        cudaEventDestroy(start);
        cudaEventDestroy(stop);

        system::runtime::destroy_stream_on_device(kernel_device, stream);

        return static_cast<double>(total_ms) / static_cast<double>(iters);
    } catch (...) {
        if (start != nullptr) {
            cudaEventDestroy(start);
        }

        if (stop != nullptr) {
            cudaEventDestroy(stop);
        }

        system::runtime::destroy_stream_on_device(kernel_device, stream);
        throw;
    }
}

void fill_buffer(
    int device,
    void* ptr,
    int byte_value,
    size_t bytes) {
    system::runtime::set_device(device);

    cudaStream_t stream =
        system::runtime::create_stream_on_device(device);

    try {
        testing::check_cuda(
            cudaMemsetAsync(ptr, byte_value, bytes, stream),
            "cudaMemsetAsync");

        system::runtime::sync_stream_on_device(
            device,
            stream,
            "sync memset");

        system::runtime::destroy_stream_on_device(device, stream);
    } catch (...) {
        system::runtime::destroy_stream_on_device(device, stream);
        throw;
    }
}

system::mapped_peer_buffer alloc_visible_buffer(
    size_t bytes,
    int owner_device,
    int dev0,
    int dev1) {
    std::vector<int> access_devices = {
        dev0,
        dev1,
    };

    return system::alloc_peer_visible_buffer(
        bytes,
        owner_device,
        access_devices);
}

BatchExperimentBuffers alloc_batch_buffers(
    size_t bytes,
    int local_device,
    int peer_device) {
    BatchExperimentBuffers bufs{};

    bufs.reduce_src0_peer =
        alloc_visible_buffer(
            bytes,
            peer_device,
            local_device,
            peer_device);

    bufs.reduce_src1_peer =
        alloc_visible_buffer(
            bytes,
            peer_device,
            local_device,
            peer_device);

    bufs.reduce_dst_local =
        alloc_visible_buffer(
            bytes,
            local_device,
            local_device,
            peer_device);

    bufs.copy_src_local =
        alloc_visible_buffer(
            bytes,
            local_device,
            local_device,
            peer_device);

    bufs.copy_dst0_peer =
        alloc_visible_buffer(
            bytes,
            peer_device,
            local_device,
            peer_device);

    bufs.copy_dst1_peer =
        alloc_visible_buffer(
            bytes,
            peer_device,
            local_device,
            peer_device);

    bufs.reduce_fanout_src_local =
        alloc_visible_buffer(
            bytes,
            local_device,
            local_device,
            peer_device);

    bufs.reduce_fanout_dst0_peer =
        alloc_visible_buffer(
            bytes,
            peer_device,
            local_device,
            peer_device);

    bufs.reduce_fanout_dst1_peer =
        alloc_visible_buffer(
            bytes,
            peer_device,
            local_device,
            peer_device);

    return bufs;
}

void free_batch_buffers(BatchExperimentBuffers& bufs) {
    system::free_peer_visible_buffer(bufs.reduce_src0_peer);
    system::free_peer_visible_buffer(bufs.reduce_src1_peer);
    system::free_peer_visible_buffer(bufs.reduce_dst_local);

    system::free_peer_visible_buffer(bufs.copy_src_local);
    system::free_peer_visible_buffer(bufs.copy_dst0_peer);
    system::free_peer_visible_buffer(bufs.copy_dst1_peer);

    system::free_peer_visible_buffer(bufs.reduce_fanout_src_local);
    system::free_peer_visible_buffer(bufs.reduce_fanout_dst0_peer);
    system::free_peer_visible_buffer(bufs.reduce_fanout_dst1_peer);
}

void initialize_batch_buffers(
    BatchExperimentBuffers& bufs,
    size_t bytes,
    int local_device,
    int peer_device) {
    fill_buffer(peer_device, bufs.reduce_src0_peer.ptr, 1, bytes);
    fill_buffer(peer_device, bufs.reduce_src1_peer.ptr, 2, bytes);
    fill_buffer(local_device, bufs.reduce_dst_local.ptr, 0, bytes);

    fill_buffer(local_device, bufs.copy_src_local.ptr, 7, bytes);
    fill_buffer(peer_device, bufs.copy_dst0_peer.ptr, 0, bytes);
    fill_buffer(peer_device, bufs.copy_dst1_peer.ptr, 0, bytes);

    fill_buffer(local_device, bufs.reduce_fanout_src_local.ptr, 3, bytes);
    fill_buffer(peer_device, bufs.reduce_fanout_dst0_peer.ptr, 0, bytes);
    fill_buffer(peer_device, bufs.reduce_fanout_dst1_peer.ptr, 0, bytes);
}

void add_batch_result(
    std::vector<std::map<std::string, double>>& results,
    int experiment,
    int method,
    int src_device,
    int dst_device,
    int kernel_device,
    size_t bytes,
    size_t payload_bytes,
    int num_blocks,
    int split0_ctas,
    int split1_ctas,
    double latency_ms) {
    const double seconds = latency_ms * 1.0e-3;

    const double gbps =
        seconds > 0.0
            ? static_cast<double>(payload_bytes) / seconds / 1.0e9
            : 0.0;

    results.push_back({
        {"experiment", static_cast<double>(experiment)},
        {"method", static_cast<double>(method)},
        {"src_device", static_cast<double>(src_device)},
        {"dst_device", static_cast<double>(dst_device)},
        {"kernel_device", static_cast<double>(kernel_device)},
        {"bytes", static_cast<double>(bytes)},
        {"payload_bytes", static_cast<double>(payload_bytes)},
        {"num_blocks", static_cast<double>(num_blocks)},
        {"split0_ctas", static_cast<double>(split0_ctas)},
        {"split1_ctas", static_cast<double>(split1_ctas)},
        {"latency_ms", latency_ms},
        {"gbps", gbps},
    });
}

void run_one_batch_size_and_block_count(
    std::vector<std::map<std::string, double>>& results,
    size_t bytes,
    int num_blocks,
    int iters,
    int warmup,
    int local_device,
    int peer_device) {
    if (num_blocks < 2) {
        throw std::invalid_argument(
            "batch experiment requires num_blocks >= 2");
    }

    configure_batch_kernels_once(local_device);

    BatchExperimentBuffers bufs =
        alloc_batch_buffers(
            bytes,
            local_device,
            peer_device);

    try {
        initialize_batch_buffers(
            bufs,
            bytes,
            local_device,
            peer_device);

        const int split0_ctas =
            num_blocks / 2;

        const int split1_ctas =
            num_blocks - split0_ctas;

        const size_t payload_bytes =
            static_cast<size_t>(2) * bytes;

        /*
         * Experiment A:
         *   two peer-owned source buffers -> one local destination buffer
         *
         * Baseline:
         *   launch reduce src0->dst with all CTAs, then src1->dst with all CTAs.
         */
        double ms =
            benchmark_launch_ms(
                local_device,
                iters,
                warmup,
                [&](cudaStream_t stream) {
                    return launch_reduce_seq2_gpu_scope(
                        bufs.reduce_src0_peer.ptr,
                        bufs.reduce_src1_peer.ptr,
                        bufs.reduce_dst_local.ptr,
                        bytes,
                        num_blocks,
                        stream);
                });

        add_batch_result(
            results,
            kBatchExperimentReduceTwoToOne,
            kBatchMethodReduceSequentialAllCtas,
            peer_device,
            local_device,
            local_device,
            bytes,
            payload_bytes,
            num_blocks,
            num_blocks,
            num_blocks,
            ms);

        /*
         * Concurrent split-CTA method:
         *   one launch, total CTAs fixed, src0 gets split0_ctas,
         *   src1 gets split1_ctas, both reduce into the same destination range
         *   using TmaReduceScope::Gpu.
         */
        ms =
            benchmark_launch_ms(
                local_device,
                iters,
                warmup,
                [&](cudaStream_t stream) {
                    return launch_reduce_split2_gpu_scope<false>(
                        bufs.reduce_src0_peer.ptr,
                        bufs.reduce_src1_peer.ptr,
                        bufs.reduce_dst_local.ptr,
                        bytes,
                        num_blocks,
                        stream);
                });

        add_batch_result(
            results,
            kBatchExperimentReduceTwoToOne,
            kBatchMethodReduceSplitCtasGpuScope,
            peer_device,
            local_device,
            local_device,
            bytes,
            payload_bytes,
            num_blocks,
            split0_ctas,
            split1_ctas,
            ms);

        ms =
            benchmark_launch_ms(
                local_device,
                iters,
                warmup,
                [&](cudaStream_t stream) {
                    return launch_reduce_split2_gpu_scope<true>(
                        bufs.reduce_src0_peer.ptr,
                        bufs.reduce_src1_peer.ptr,
                        bufs.reduce_dst_local.ptr,
                        bytes,
                        num_blocks,
                        stream);
                });

        add_batch_result(
            results,
            kBatchExperimentReduceTwoToOne,
            kBatchMethodReduceSplitCtasOppositeDirectionsGpuScope,
            peer_device,
            local_device,
            local_device,
            bytes,
            payload_bytes,
            num_blocks,
            split0_ctas,
            split1_ctas,
            ms);

        /*
         * Experiment B:
         *   one local source buffer -> two peer-owned destination buffers
         *
         * Baseline:
         *   launch copy src->dst0 with all CTAs, then src->dst1 with all CTAs.
         */
        ms =
            benchmark_launch_ms(
                local_device,
                iters,
                warmup,
                [&](cudaStream_t stream) {
                    return launch_copy_seq2(
                        bufs.copy_src_local.ptr,
                        bufs.copy_dst0_peer.ptr,
                        bufs.copy_dst1_peer.ptr,
                        bytes,
                        num_blocks,
                        stream);
                });

        add_batch_result(
            results,
            kBatchExperimentCopyOneToTwo,
            kBatchMethodCopySequentialAllCtas,
            local_device,
            peer_device,
            local_device,
            bytes,
            payload_bytes,
            num_blocks,
            num_blocks,
            num_blocks,
            ms);

        /*
         * Batched fan-out method:
         *   load src chunk once, issue two stores, commit once.
         */
        ms =
            benchmark_launch_ms(
                local_device,
                iters,
                warmup,
                [&](cudaStream_t stream) {
                    return launch_copy_fanout2(
                        bufs.copy_src_local.ptr,
                        bufs.copy_dst0_peer.ptr,
                        bufs.copy_dst1_peer.ptr,
                        bytes,
                        num_blocks,
                        stream);
                });

        add_batch_result(
            results,
            kBatchExperimentCopyOneToTwo,
            kBatchMethodCopyFanoutOneLoadTwoStores,
            local_device,
            peer_device,
            local_device,
            bytes,
            payload_bytes,
            num_blocks,
            num_blocks,
            num_blocks,
            ms);

        /*
         * Experiment C:
         *   local source buffer -> two peer-owned destination buffers by TMA reduce.
         */
        ms =
            benchmark_launch_ms(
                local_device,
                iters,
                warmup,
                [&](cudaStream_t stream) {
                    return launch_reduce_fanout_seq2_gpu_scope(
                        bufs.reduce_fanout_src_local.ptr,
                        bufs.reduce_fanout_dst0_peer.ptr,
                        bufs.reduce_fanout_dst1_peer.ptr,
                        bytes,
                        num_blocks,
                        stream);
                });

        add_batch_result(
            results,
            kBatchExperimentReduceOneToTwo,
            kBatchMethodReduceFanoutSequentialAllCtas,
            local_device,
            peer_device,
            local_device,
            bytes,
            payload_bytes,
            num_blocks,
            num_blocks,
            num_blocks,
            ms);

        ms =
            benchmark_launch_ms(
                local_device,
                iters,
                warmup,
                [&](cudaStream_t stream) {
                    return launch_reduce_fanout2_gpu_scope(
                        bufs.reduce_fanout_src_local.ptr,
                        bufs.reduce_fanout_dst0_peer.ptr,
                        bufs.reduce_fanout_dst1_peer.ptr,
                        bytes,
                        num_blocks,
                        stream);
                });

        add_batch_result(
            results,
            kBatchExperimentReduceOneToTwo,
            kBatchMethodReduceFanoutOneLoadTwoReducesGpuScope,
            local_device,
            peer_device,
            local_device,
            bytes,
            payload_bytes,
            num_blocks,
            num_blocks,
            num_blocks,
            ms);

        free_batch_buffers(bufs);
    } catch (...) {
        free_batch_buffers(bufs);
        throw;
    }
}

std::vector<int64_t> make_power_of_two_sizes(
    int64_t min_bytes,
    int64_t max_bytes) {
    std::vector<int64_t> sizes;

    for (int64_t b = min_bytes; b <= max_bytes;) {
        sizes.push_back(b);

        if (b > max_bytes / 2) {
            break;
        }

        b *= 2;
    }

    return sizes;
}

void validate_batch_sweep_args(
    const std::vector<int64_t>& sizes_bytes,
    const std::vector<int>& num_blocks_list,
    int iters,
    int warmup,
    int dev0,
    int dev1) {
    if (sizes_bytes.empty()) {
        throw std::invalid_argument("sizes_bytes must be non-empty");
    }

    if (num_blocks_list.empty()) {
        throw std::invalid_argument("num_blocks_list must be non-empty");
    }

    if (iters <= 0 || warmup < 0) {
        throw std::invalid_argument("invalid iteration counts");
    }

    if (dev0 < 0 || dev1 < 0 || dev0 == dev1) {
        throw std::invalid_argument("invalid devices");
    }

    for (int64_t bytes : sizes_bytes) {
        if (bytes <= 0) {
            throw std::invalid_argument("all sizes must be positive");
        }

        if (!is_aligned_16_size_host(static_cast<size_t>(bytes))) {
            throw std::invalid_argument("all sizes must be 16-byte aligned");
        }

        if ((static_cast<size_t>(bytes) % sizeof(half)) != 0) {
            throw std::invalid_argument(
                "reduce experiment requires sizes divisible by sizeof(half)");
        }
    }

    for (int blocks : num_blocks_list) {
        if (blocks < 2) {
            throw std::invalid_argument(
                "all num_blocks values must be >= 2 for split-CTA experiment");
        }
    }
}

} // namespace

std::vector<std::map<std::string, double>>
benchmark_tma_batch_experiment_sweep_sm90(
    const std::vector<int64_t>& sizes_bytes,
    const std::vector<int>& num_blocks_list,
    int iters,
    int warmup,
    int dev0,
    int dev1) {
    validate_batch_sweep_args(
        sizes_bytes,
        num_blocks_list,
        iters,
        warmup,
        dev0,
        dev1);

    /*
     * Kernel runs on dev0.  The experiment is intentionally asymmetric:
     *
     *   reduce: peer buffers on dev1 -> local buffer on dev0
     *   copy:   local buffer on dev0 -> peer buffers on dev1
     */
    system::runtime::ensure_context_on_device(dev0);
    system::runtime::ensure_context_on_device(dev1);

    std::vector<std::map<std::string, double>> results;

    for (int64_t bytes_i : sizes_bytes) {
        const size_t bytes =
            static_cast<size_t>(bytes_i);

        for (int num_blocks : num_blocks_list) {
            run_one_batch_size_and_block_count(
                results,
                bytes,
                num_blocks,
                iters,
                warmup,
                dev0,
                dev1);
        }
    }

    return results;
}

std::vector<std::map<std::string, double>>
benchmark_tma_batch_experiment_sm90(
    int64_t min_bytes,
    int64_t max_bytes,
    int iters,
    int warmup,
    int num_blocks,
    int dev0,
    int dev1) {
    if (min_bytes <= 0 || max_bytes <= 0 || min_bytes > max_bytes) {
        throw std::invalid_argument("invalid byte range");
    }

    std::vector<int64_t> sizes =
        make_power_of_two_sizes(
            min_bytes,
            max_bytes);

    std::vector<int> blocks = {
        num_blocks,
    };

    return benchmark_tma_batch_experiment_sweep_sm90(
        sizes,
        blocks,
        iters,
        warmup,
        dev0,
        dev1);
}

/*
 * Compatibility names for the previous tma_bandwidth_experiment API.
 *
 * These now run the new batch/fanout experiment so existing includes and
 * pybind-side names can be reused during quick iteration.
 */
std::vector<std::map<std::string, double>>
benchmark_tma_bandwidth_experiment_sweep_sm90(
    const std::vector<int64_t>& sizes_bytes,
    const std::vector<int>& num_blocks_list,
    int iters,
    int warmup,
    int dev0,
    int dev1,
    bool include_nccl) {
    (void)include_nccl;

    return benchmark_tma_batch_experiment_sweep_sm90(
        sizes_bytes,
        num_blocks_list,
        iters,
        warmup,
        dev0,
        dev1);
}

std::vector<std::map<std::string, double>>
benchmark_tma_bandwidth_experiment_sm90(
    int64_t min_bytes,
    int64_t max_bytes,
    int iters,
    int warmup,
    int num_blocks,
    int dev0,
    int dev1,
    bool include_mem_async,
    bool include_nccl) {
    (void)include_mem_async;
    (void)include_nccl;

    return benchmark_tma_batch_experiment_sm90(
        min_bytes,
        max_bytes,
        iters,
        warmup,
        num_blocks,
        dev0,
        dev1);
}

} // namespace ooverlap
