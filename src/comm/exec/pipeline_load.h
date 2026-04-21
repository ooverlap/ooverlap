#pragma once

#include "comm/exec/pipeline_stage.h"
#include "ooverlap/tma/tma.cuh"

#include <cstddef>
#include <cstdint>

namespace ooverlap {
namespace comm {
namespace exec {
namespace {

__device__ __forceinline__ void pipeline_load_fail_invalid_layout() {
#if defined(__CUDA_ARCH__)
    asm volatile("trap;");
#endif
}

} // namespace

struct PipelineTMALoad {
    __device__ __forceinline__ void issue(
        const PipelineStage* stage) const {
        if (stage == nullptr || !chunk_is_valid(&stage->chunk)) {
            return;
        }

        if (!pipeline_stage_chunk_is_dense_packed(stage)) {
            pipeline_load_fail_invalid_layout();
            return;
        }

        sync::init_semaphore(*stage->load_barrier, 1);
        tma::expect_bytes(
            *stage->load_barrier,
            static_cast<uint32_t>(stage->chunk.bytes));

        for (int i = 0; i < stage->chunk.num_tile_spans; ++i) {
            const ChunkTileSpan& span = stage->chunk.tile_spans[i];
            if (!chunk_tile_span_is_valid(&span)) {
                pipeline_load_fail_invalid_layout();
                return;
            }

            tma::load_async(
                pipeline_stage_smem_ptr(stage, span.dst_offset_bytes),
                span.src,
                static_cast<uint32_t>(span.bytes),
                *stage->load_barrier);
        }
    }

    __device__ __forceinline__ void wait_ready(
        const PipelineStage* stage) const {
        sync::wait(*stage->load_barrier, 0);
    }
};

} // namespace exec
} // namespace comm
} // namespace ooverlap
