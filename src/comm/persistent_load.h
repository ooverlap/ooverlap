#pragma once

#include "comm/persistent_stage.h"
#include "ooverlap/tma/tma.cuh"

#include <cstdint>

namespace ooverlap {
namespace comm {

struct PersistentLinearTmaLoad {
    __device__ __forceinline__ void issue(
        const PersistentChunkStage* stage,
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
        const PersistentChunkStage* stage) const {
        sync::wait(*stage->load_barrier, 0);
    }
};

} // namespace comm
} // namespace ooverlap
