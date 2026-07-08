#pragma once

#include "comm/pipeline/pipeline_stage.h"
#include "ooverlap/tma/tma.cuh"

#include <cstddef>
#include <cstdint>

namespace ooverlap {
namespace comm {
namespace pipeline {

/*
 * Batched TMA store/copy pipeline.
 *
 * This is intentionally separate from PipelineTMACopy.  The old PipelineTMACopy
 * keeps one-stage -> one cp.async.bulk -> one commit_group behavior.
 *
 * This helper exposes the lower-level pieces needed for fan-out experiments:
 *
 *   load stage once
 *   wait_ready(stage)
 *   copy.issue_fence()
 *   copy.issue_bulk_op_nofence(stage, dst0)
 *   copy.issue_bulk_op_nofence(stage, dst1)
 *   copy.commit()
 *
 * The old full behavior is still available through issue_bulk_commit().
 */
template <int StageDepth, int FillDepth>
struct PipelineTMACopyBatch {
    static_assert(StageDepth > 0, "StageDepth must be > 0");
    static_assert(FillDepth > 0, "FillDepth must be > 0");
    static_assert(FillDepth <= StageDepth, "FillDepth must be <= StageDepth");

    __device__ __forceinline__ void wait_before_stage_reuse() const {
        tma::store_async_read_wait<FillDepth - 1>();
    }

    __device__ __forceinline__ void wait_complete() const {
        tma::store_async_wait<0>();
    }

    __device__ __forceinline__ void issue_fence() const {
        tma::store_fence_proxy_async_shared_cta();
    }

    __device__ __forceinline__ void commit() const {
        tma::store_commit_group();
    }

    __device__ __forceinline__ void issue_bulk_op_nofence(
        const PipelineStage* stage) const {
        issue_bulk_op_nofence(
            stage,
            stage != nullptr ? stage->chunk.dst : nullptr);
    }

    __device__ __forceinline__ void issue_bulk_op_nofence(
        const PipelineStage* stage,
        void* dst_override) const {
        if (stage == nullptr || dst_override == nullptr) {
            return;
        }

        const size_t bulk_bytes = pipeline_stage_bulk_bytes(stage);

        if (bulk_bytes == 0) {
            return;
        }

        tma::store_async_op_nofence(
            dst_override,
            stage->smem,
            static_cast<uint32_t>(bulk_bytes));
    }

    /*
     * Fence + op, but no commit.  Use this when each stage's smem producer may
     * need its own proxy fence, while still batching several stores into one
     * commit_group.
     */
    __device__ __forceinline__ void issue_bulk_op(
        const PipelineStage* stage) const {
        issue_bulk_op(
            stage,
            stage != nullptr ? stage->chunk.dst : nullptr);
    }

    __device__ __forceinline__ void issue_bulk_op(
        const PipelineStage* stage,
        void* dst_override) const {
        if (stage == nullptr || dst_override == nullptr) {
            return;
        }

        const size_t bulk_bytes = pipeline_stage_bulk_bytes(stage);

        if (bulk_bytes == 0) {
            return;
        }

        tma::store_async_op(
            dst_override,
            stage->smem,
            static_cast<uint32_t>(bulk_bytes));
    }

    /*
     * Old one-op-one-commit behavior, but with an explicit name and optional
     * destination override.
     */
    __device__ __forceinline__ void issue_bulk_commit(
        const PipelineStage* stage) const {
        issue_bulk_commit(
            stage,
            stage != nullptr ? stage->chunk.dst : nullptr);
    }

    __device__ __forceinline__ void issue_bulk_commit(
        const PipelineStage* stage,
        void* dst_override) const {
        if (stage == nullptr || dst_override == nullptr) {
            return;
        }

        const size_t bulk_bytes = pipeline_stage_bulk_bytes(stage);

        if (bulk_bytes == 0) {
            return;
        }

        tma::store_async_commit(
            dst_override,
            stage->smem,
            static_cast<uint32_t>(bulk_bytes));
    }

    /*
     * Compatibility with the old pipeline Apply concept.
     */
    __device__ __forceinline__ void issue_bulk(
        const PipelineStage* stage) const {
        issue_bulk_commit(stage);
    }

    /*
     * Common fan-out case: one loaded smem chunk, two global destinations,
     * one fence, one commit.
     */
    __device__ __forceinline__ void issue_fanout2_commit(
        const PipelineStage* stage,
        void* dst0,
        void* dst1) const {
        if (stage == nullptr || dst0 == nullptr || dst1 == nullptr) {
            return;
        }

        const size_t bulk_bytes = pipeline_stage_bulk_bytes(stage);

        if (bulk_bytes == 0) {
            return;
        }

        issue_fence();

        tma::store_async_op_nofence(
            dst0,
            stage->smem,
            static_cast<uint32_t>(bulk_bytes));

        tma::store_async_op_nofence(
            dst1,
            stage->smem,
            static_cast<uint32_t>(bulk_bytes));

        commit();
    }

    __device__ __forceinline__ void finish_tail(
        const PipelineStage* stage) const {
        finish_tail(
            stage,
            stage != nullptr ? stage->chunk.dst : nullptr);
    }

    __device__ __forceinline__ void finish_tail(
        const PipelineStage* stage,
        void* dst_override) const {
        if (stage == nullptr || dst_override == nullptr) {
            return;
        }

        const size_t bulk_bytes = pipeline_stage_bulk_bytes(stage);
        const size_t tail_bytes = pipeline_stage_tail_bytes(stage);

        if (tail_bytes == 0) {
            return;
        }

        unsigned char* dst =
            reinterpret_cast<unsigned char*>(dst_override) + bulk_bytes;

        const unsigned char* src =
            stage->smem + bulk_bytes;

        for (size_t i = threadIdx.x; i < tail_bytes; i += blockDim.x) {
            dst[i] = src[i];
        }
    }

    __device__ __forceinline__ void finish_tail_fanout2(
        const PipelineStage* stage,
        void* dst0,
        void* dst1) const {
        finish_tail(stage, dst0);
        finish_tail(stage, dst1);
    }
};

} // namespace pipeline
} // namespace comm
} // namespace ooverlap
