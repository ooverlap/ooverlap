#pragma once

#include "comm/pipeline_stage.h"
#include "ooverlap/tma/tma_reduce.cuh"

#include <cuda_fp16.h>
#include <cstddef>
#include <cstdint>

namespace ooverlap {
namespace comm {

struct PipelineTMAReduceAddNoFtzF16 {
    template <int StageDepth>
    __device__ __forceinline__ void wait_before_stage_reuse() const {
        tma::reduce_async_read_wait<StageDepth - 1>();
    }

    __device__ __forceinline__ void issue_bulk(
        const PipelineStage* stage,
        unsigned char* dst_bytes) const {
        const size_t bulk_bytes = pipeline_stage_bulk_bytes(stage);
        if (bulk_bytes == 0) {
            return;
        }

        tma::reduce_add_noftz_f16_async(
            dst_bytes + stage->chunk_offset_bytes,
            stage->smem,
            static_cast<uint32_t>(bulk_bytes));
    }

    __device__ __forceinline__ void finish_tail(
        const PipelineStage* stage,
        half* dst_half) const {
        const size_t bulk_bytes = pipeline_stage_bulk_bytes(stage);
        const size_t tail_bytes = pipeline_stage_tail_bytes(stage);

        if (tail_bytes == 0) {
            return;
        }

        const size_t bulk_elems = bulk_bytes / sizeof(half);
        const size_t tail_elems = tail_bytes / sizeof(half);
        const size_t elem_offset = stage->chunk_offset_bytes / sizeof(half);

        const half* src_half = reinterpret_cast<const half*>(stage->smem);

        for (size_t i = threadIdx.x; i < tail_elems; i += blockDim.x) {
            const size_t idx = elem_offset + bulk_elems + i;
            const float oldv = __half2float(dst_half[idx]);
            const float addv = __half2float(src_half[bulk_elems + i]);
            dst_half[idx] = __float2half_rn(oldv + addv);
        }
    }
};

} // namespace comm
} // namespace ooverlap
