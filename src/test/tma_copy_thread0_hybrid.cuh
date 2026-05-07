#pragma once

#include "comm/pipeline/pipeline_stage.h"
#include "comm/pipeline/pipeline_tma_copy.h"
#include "comm/pipeline/pipeline_tma_load.h"
#include "ooverlap/mem/fast_copy.cuh"
#include "ooverlap/sync/sync.cuh"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ooverlap {
namespace testing {
namespace tma_hybrid_copy {

enum HybridCopyStatus {
    kHybridCopyOk = 0,
    kHybridCopyBadRange = 1,
    kHybridCopyBadAlignment = 2,
    kHybridCopyBadThreadConfig = 3,
};

__device__ __forceinline__ bool is_aligned_16(
    const void* ptr) {
    return (
        (reinterpret_cast<uintptr_t>(ptr) &
         static_cast<uintptr_t>(sizeof(uint4) - 1)) == 0);
}

__device__ __forceinline__ bool is_aligned_16_size(
    size_t x) {
    return ((x & static_cast<size_t>(sizeof(uint4) - 1)) == 0);
}

template <size_t ChunkBytes>
__device__ __forceinline__ size_t chunk_offset_bytes(
    int chunk) {
    return static_cast<size_t>(chunk) * ChunkBytes;
}

template <size_t ChunkBytes>
__device__ __forceinline__ bool range_is_16b_aligned(
    const void* src_base,
    const void* dst_base,
    size_t total_bytes,
    int begin_chunk,
    int end_chunk) {
    static_assert(ChunkBytes > 0, "ChunkBytes must be > 0");

    if (begin_chunk < 0 || end_chunk < begin_chunk) {
        return false;
    }

    const size_t begin_byte =
        chunk_offset_bytes<ChunkBytes>(begin_chunk);

    const size_t raw_end_byte =
        chunk_offset_bytes<ChunkBytes>(end_chunk);

    const size_t end_byte =
        raw_end_byte < total_bytes ? raw_end_byte : total_bytes;

    if (begin_byte > end_byte) {
        return false;
    }

    const size_t byte_count =
        end_byte - begin_byte;

    const uintptr_t src_addr =
        reinterpret_cast<uintptr_t>(src_base) + begin_byte;

    const uintptr_t dst_addr =
        reinterpret_cast<uintptr_t>(dst_base) + begin_byte;

    return
        is_aligned_16(reinterpret_cast<const void*>(src_addr)) &&
        is_aligned_16(reinterpret_cast<const void*>(dst_addr)) &&
        is_aligned_16_size(begin_byte) &&
        is_aligned_16_size(byte_count);
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

template <
    int StageDepth,
    int FillDepth,
    size_t ChunkBytes,
    typename Apply>
__device__ __forceinline__ HybridCopyStatus run_tma_copy_thread0_only_16b_aligned(
    const void* src_base,
    void* dst_base,
    size_t total_bytes,
    int begin_chunk,
    int end_chunk,
    unsigned char* shared_raw,
    sync::semaphore* barriers) {
    static_assert(StageDepth > 0, "StageDepth must be > 0");
    static_assert(FillDepth > 0, "FillDepth must be > 0");
    static_assert(FillDepth <= StageDepth, "FillDepth must be <= StageDepth");
    static_assert(ChunkBytes > 0, "ChunkBytes must be > 0");
    static_assert((ChunkBytes % sizeof(uint4)) == 0,
                  "ChunkBytes must be 16-byte aligned");

    if (threadIdx.x != 0) {
        return kHybridCopyOk;
    }

    if (begin_chunk >= end_chunk || total_bytes == 0) {
        return kHybridCopyOk;
    }

    if (begin_chunk < 0 || end_chunk < begin_chunk) {
        return kHybridCopyBadRange;
    }

    if (!range_is_16b_aligned<ChunkBytes>(
            src_base,
            dst_base,
            total_bytes,
            begin_chunk,
            end_chunk)) {
        return kHybridCopyBadAlignment;
    }

    const unsigned char* src_bytes =
        reinterpret_cast<const unsigned char*>(src_base);

    unsigned char* dst_bytes =
        reinterpret_cast<unsigned char*>(dst_base);

    comm::pipeline::PipelineTMALoad load{};
    Apply apply{};

    const int total_range_chunks =
        end_chunk - begin_chunk;

    for (int warm = 0; warm < FillDepth; ++warm) {
        if (warm >= total_range_chunks) {
            break;
        }

        const int abs_chunk =
            begin_chunk + warm;

        const int slot =
            warm;

        comm::pipeline::PipelineStage stage =
            make_stage_for_abs_chunk<ChunkBytes>(
                src_bytes,
                dst_bytes,
                total_bytes,
                abs_chunk,
                slot,
                shared_raw,
                barriers);

        load.issue(&stage);
    }

    for (int iter = 0; iter < total_range_chunks; ++iter) {
        const int abs_chunk =
            begin_chunk + iter;

        const int cur_slot =
            iter % StageDepth;

        comm::pipeline::PipelineStage cur_stage =
            make_stage_for_abs_chunk<ChunkBytes>(
                src_bytes,
                dst_bytes,
                total_bytes,
                abs_chunk,
                cur_slot,
                shared_raw,
                barriers);

        load.wait_ready(&cur_stage);

        const int future_iter =
            iter + FillDepth;

        if (future_iter < total_range_chunks) {
            const int future_abs_chunk =
                begin_chunk + future_iter;

            const int future_slot =
                future_iter % StageDepth;

            comm::pipeline::PipelineStage future_stage =
                make_stage_for_abs_chunk<ChunkBytes>(
                    src_bytes,
                    dst_bytes,
                    total_bytes,
                    future_abs_chunk,
                    future_slot,
                    shared_raw,
                    barriers);

            if (iter >= FillDepth) {
                apply.wait_before_stage_reuse();
            }

            load.issue(&future_stage);
        }

        apply.issue_bulk(&cur_stage);
    }

    apply.wait_complete();

    return kHybridCopyOk;
}

template <
    size_t ChunkBytes,
    typename VecT = uint4,
    int Unroll = 8>
__device__ __forceinline__ HybridCopyStatus run_fast_copy_workers_except_thread0_16b_aligned(
    const void* __restrict__ src_base,
    void* __restrict__ dst_base,
    size_t total_bytes,
    int begin_chunk,
    int end_chunk) {
    static_assert(ChunkBytes > 0, "ChunkBytes must be > 0");
    static_assert((ChunkBytes % sizeof(VecT)) == 0,
                  "ChunkBytes must be aligned to VecT");

    if (threadIdx.x == 0) {
        return kHybridCopyOk;
    }

    const int worker_count =
        static_cast<int>(blockDim.x) - 1;

    const int worker_lane =
        static_cast<int>(threadIdx.x) - 1;

    if (worker_count <= 0 || worker_lane < 0) {
        return kHybridCopyBadThreadConfig;
    }

    if (begin_chunk >= end_chunk || total_bytes == 0) {
        return kHybridCopyOk;
    }

    if (begin_chunk < 0 || end_chunk < begin_chunk) {
        return kHybridCopyBadRange;
    }

    if (!range_is_16b_aligned<ChunkBytes>(
            src_base,
            dst_base,
            total_bytes,
            begin_chunk,
            end_chunk)) {
        return kHybridCopyBadAlignment;
    }

    const size_t begin_byte =
        chunk_offset_bytes<ChunkBytes>(begin_chunk);

    const size_t raw_end_byte =
        chunk_offset_bytes<ChunkBytes>(end_chunk);

    const size_t end_byte =
        raw_end_byte < total_bytes ? raw_end_byte : total_bytes;

    const size_t byte_count =
        end_byte - begin_byte;

    comm::kernels::fast_copy::copy_byte_range<VecT, Unroll>(
        src_base,
        dst_base,
        begin_byte,
        byte_count,
        static_cast<size_t>(worker_lane),
        static_cast<size_t>(worker_count));

    return kHybridCopyOk;
}

__device__ __forceinline__ void record_hybrid_copy_error(
    int* error_code,
    HybridCopyStatus status) {
    if (error_code == nullptr || status == kHybridCopyOk) {
        return;
    }

    atomicCAS(
        error_code,
        static_cast<int>(kHybridCopyOk),
        static_cast<int>(status));
}

template <
    int StageDepth,
    int FillDepth,
    size_t ChunkBytes>
__device__ __forceinline__ void run_dual_dst_tma_plus_fast_copy_16b_aligned(
    const void* __restrict__ src_base,
    void* __restrict__ tma_dst_base,
    void* __restrict__ fast_dst_base,
    size_t total_bytes,
    int begin_chunk,
    int end_chunk,
    unsigned char* shared_raw,
    sync::semaphore* barriers,
    int* error_code = nullptr) {
    using CopyApply =
        comm::pipeline::PipelineTMACopy<StageDepth, FillDepth>;

    HybridCopyStatus status = kHybridCopyOk;

    if (threadIdx.x == 0) {
        status =
            run_tma_copy_thread0_only_16b_aligned<
                StageDepth,
                FillDepth,
                ChunkBytes,
                CopyApply>(
                    src_base,
                    tma_dst_base,
                    total_bytes,
                    begin_chunk,
                    end_chunk,
                    shared_raw,
                    barriers);
    } else {
        status =
            run_fast_copy_workers_except_thread0_16b_aligned<
                ChunkBytes,
                uint4,
                8>(
                    src_base,
                    fast_dst_base,
                    total_bytes,
                    begin_chunk,
                    end_chunk);
    }

    record_hybrid_copy_error(error_code, status);
}

} // namespace tma_hybrid_copy
} // namespace testing
} // namespace ooverlap
