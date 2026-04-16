#pragma once

#include "comm/pipeline_stage.h"
#include "ooverlap/tma/tma.cuh"

#include <cstdint>

namespace ooverlap {
namespace comm {

struct PipelineTMALoad {
    __device__ __forceinline__ void issue(
        const PipelineStage* stage,
        const unsigned char* src_bytes) const {
        sync::init_semaphore(*stage->load_barrier, 1);
        tma::expect_bytes(
            *stage->load_barrier,
            static_cast<uint32_t>(stage->chunk_bytes));
        tma::load_async(
            stage->smem,
            src_bytes + stage->chunk_offset_bytes,
            static_cast<uint32_t>(stage->chunk_bytes),
            *stage->load_barrier);
    }

    __device__ __forceinline__ void wait_ready(
        const PipelineStage* stage) const {
        sync::wait(*stage->load_barrier, 0);
    }
};

} // namespace comm
} // namespace ooverlap
