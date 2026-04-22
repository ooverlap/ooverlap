#pragma once

#include "comm/exec/pipeline_stage.h"
#include "ooverlap/tma/tma.cuh"
#include "ooverlap/tma/tma_reduce.cuh"

#include <cuda_fp16.h>
#include <cstddef>
#include <cstdint>

namespace ooverlap {
namespace comm {
namespace exec {

struct PipelineTMAStepApplyNoFtzF16 {
    template <int StageDepth>
    __device__ __forceinline__ void wait_before_stage_reuse() const {
        // Both bulk store and bulk reduce use the same async bulk-group wait
        // machinery underneath. Keeping one wait-before-reuse point is enough.
        tma::store_async_read_wait<StageDepth - 1>();
    }

    __device__ __forceinline__ void issue_bulk(
        const PipelineStage* stage) const {
        if (stage == nullptr || !chunk_is_valid(&stage->chunk)) {
            return;
        }

        const size_t bulk_bytes = pipeline_stage_bulk_bytes(stage);
        if (bulk_bytes == 0) {
            return;
        }

        if (stage->chunk.op == ChunkOpKind::kReduceAddNoFtzF16) {
            tma::reduce_add_noftz_f16_async(
                stage->chunk.dst,
                stage->smem,
                static_cast<uint32_t>(bulk_bytes));
            return;
        }

        if (stage->chunk.op == ChunkOpKind::kCopy) {
            tma::store_async(
                stage->chunk.dst,
                stage->smem,
                static_cast<uint32_t>(bulk_bytes));
            return;
        }
    }

    __device__ __forceinline__ void finish_tail(
        const PipelineStage* stage) const {
        if (stage == nullptr || !chunk_is_valid(&stage->chunk)) {
            return;
        }

        const size_t bulk_bytes = pipeline_stage_bulk_bytes(stage);
        const size_t tail_bytes = pipeline_stage_tail_bytes(stage);

        if (tail_bytes == 0) {
            return;
        }

        const size_t bulk_elems = bulk_bytes / sizeof(half);
        const size_t tail_elems = tail_bytes / sizeof(half);

        if (stage->chunk.op == ChunkOpKind::kReduceAddNoFtzF16) {
            half* dst_half = reinterpret_cast<half*>(stage->chunk.dst);
            const half* src_half = reinterpret_cast<const half*>(stage->smem);

            for (size_t i = threadIdx.x; i < tail_elems; i += blockDim.x) {
                const size_t idx = bulk_elems + i;
                const float oldv = __half2float(dst_half[idx]);
                const float addv = __half2float(src_half[idx]);
                dst_half[idx] = __float2half_rn(oldv + addv);
            }
            return;
        }

        if (stage->chunk.op == ChunkOpKind::kCopy) {
            unsigned char* dst = stage->chunk.dst + bulk_bytes;
            const unsigned char* src = stage->smem + bulk_bytes;

            for (size_t i = threadIdx.x; i < tail_bytes; i += blockDim.x) {
                dst[i] = src[i];
            }
            return;
        }
    }

    __device__ __forceinline__ void wait_complete(
        const PipelineStage* stage) const {
        if (stage == nullptr || !chunk_is_valid(&stage->chunk)) {
            return;
        }

        const size_t bulk_bytes = pipeline_stage_bulk_bytes(stage);
        if (bulk_bytes == 0) {
            return;
        }

        if (stage->chunk.op == ChunkOpKind::kReduceAddNoFtzF16) {
            tma::reduce_async_wait<0>();
            return;
        }

        if (stage->chunk.op == ChunkOpKind::kCopy) {
            tma::store_async_wait<0>();
            return;
        }
    }
};

// Keep the old name alive so existing callsites do not need to change.
using PipelineTMAReduceAddNoFtzF16 = PipelineTMAStepApplyNoFtzF16;

} // namespace exec
} // namespace comm
} // namespace ooverlap
