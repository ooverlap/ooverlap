#pragma once

#include "comm/pipeline/pipeline_stage.h"
#include "ooverlap/tma/tma.cuh"

#include <cstddef>
#include <cstdint>

namespace ooverlap {
namespace comm {
namespace pipeline {

template <int StageDepth, int FillDepth>
struct PipelineTMACopy {
    static_assert(StageDepth > 0, "StageDepth must be > 0");
    static_assert(FillDepth > 0, "FillDepth must be > 0");
    static_assert(FillDepth <= StageDepth, "FillDepth must be <= StageDepth");

    __device__ __forceinline__ void wait_before_stage_reuse() const {
        tma::store_async_read_wait<FillDepth - 1>();
    }

    __device__ __forceinline__ void issue_bulk(
        const PipelineStage* stage) const {
        const size_t bulk_bytes = pipeline_stage_bulk_bytes(stage);

        if (bulk_bytes == 0) {
            return;
        }

        tma::store_async(
            stage->chunk.dst,
            stage->smem,
            static_cast<uint32_t>(bulk_bytes));
    }

    __device__ __forceinline__ void finish_tail(
        const PipelineStage* stage) const {
        const size_t bulk_bytes = pipeline_stage_bulk_bytes(stage);
        const size_t tail_bytes = pipeline_stage_tail_bytes(stage);

        if (tail_bytes == 0) {
            return;
        }

        unsigned char* dst = stage->chunk.dst + bulk_bytes;
        const unsigned char* src = stage->smem + bulk_bytes;

        for (size_t i = threadIdx.x; i < tail_bytes; i += blockDim.x) {
            dst[i] = src[i];
        }
    }

    __device__ __forceinline__ void wait_complete() const {
        tma::store_async_wait<0>();
    }
};

} // namespace pipeline
} // namespace comm
} // namespace ooverlap
