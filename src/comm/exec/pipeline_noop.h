#pragma once

#include "comm/exec/pipeline_stage.h"

namespace ooverlap {
namespace comm {
namespace exec {

struct PipelineNoOpLoad {
    __device__ __forceinline__ void issue(PipelineStage* stage) {
        (void)stage;
    }

    __device__ __forceinline__ void wait_ready(PipelineStage* stage) {
        (void)stage;
    }
};

struct PipelineNoOpApply {
    __device__ __forceinline__ void issue_bulk(PipelineStage* stage) {
        (void)stage;
    }

    __device__ __forceinline__ void finish_tail(PipelineStage* stage) {
        (void)stage;
    }

    __device__ __forceinline__ void wait_complete(PipelineStage* stage) {
        (void)stage;
    }
};

} // namespace exec
} // namespace comm
} // namespace ooverlap
