#pragma once

#include "comm/exec/pipeline_stage.h"
#include "ooverlap/tma/tma.cuh"

#include <cstddef>
#include <cstdint>

namespace ooverlap {
namespace comm {
namespace exec {

struct PipelineTMALoad {
    __device__ __forceinline__ void issue(
        const PipelineStage* stage) const {
        if (stage == nullptr || !chunk_is_valid(&stage->chunk)) {
            return;
        }

        sync::init_semaphore(*stage->load_barrier, 1);
        tma::expect_bytes(
            *stage->load_barrier,
            static_cast<uint32_t>(stage->chunk.bytes));

        tma::load_async(
            stage->smem,
            stage->chunk.src,
            static_cast<uint32_t>(stage->chunk.bytes),
            *stage->load_barrier);
    }

    __device__ __forceinline__ void wait_ready(
        const PipelineStage* stage) const {
        sync::wait(*stage->load_barrier, 0);
    }
};

} // namespace exec
} // namespace comm
} // namespace ooverlap
