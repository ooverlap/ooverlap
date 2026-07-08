#pragma once

#include "comm/pipeline/pipeline_stage.h"
#include "ooverlap/tma/tma_reduce.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstddef>
#include <cstdint>

namespace ooverlap {
namespace comm {
namespace pipeline {

/*
 * Batched TMA reduce operation tags.
 *
 * These mirror the old PipelineReduce* tags, but expose split op/commit entry
 * points and explicit PTX 9.3 reduction scope.
 */

struct PipelineReduceBatchAddF16 {
    using scalar_t = half;

    template <tma::TmaReduceScope Scope>
    __device__ __forceinline__ static void issue_bulk_op_nofence(
        void* dst,
        void* smem,
        uint32_t bytes) {
        tma::reduce_add_f16_async_op_nofence<Scope>(dst, smem, bytes);
    }

    template <tma::TmaReduceScope Scope>
    __device__ __forceinline__ static void issue_bulk_op(
        void* dst,
        void* smem,
        uint32_t bytes) {
        tma::reduce_add_f16_async_op<Scope>(dst, smem, bytes);
    }

    template <tma::TmaReduceScope Scope>
    __device__ __forceinline__ static void issue_bulk_commit(
        void* dst,
        void* smem,
        uint32_t bytes) {
        tma::reduce_add_f16_async_commit<Scope>(dst, smem, bytes);
    }

    __device__ __forceinline__ static scalar_t apply_tail(
        scalar_t oldv,
        scalar_t newv) {
        return __float2half_rn(__half2float(oldv) + __half2float(newv));
    }
};

struct PipelineReduceBatchAddNoFtzF16 {
    using scalar_t = half;

    template <tma::TmaReduceScope Scope>
    __device__ __forceinline__ static void issue_bulk_op_nofence(
        void* dst,
        void* smem,
        uint32_t bytes) {
        tma::reduce_add_noftz_f16_async_op_nofence<Scope>(dst, smem, bytes);
    }

    template <tma::TmaReduceScope Scope>
    __device__ __forceinline__ static void issue_bulk_op(
        void* dst,
        void* smem,
        uint32_t bytes) {
        tma::reduce_add_noftz_f16_async_op<Scope>(dst, smem, bytes);
    }

    template <tma::TmaReduceScope Scope>
    __device__ __forceinline__ static void issue_bulk_commit(
        void* dst,
        void* smem,
        uint32_t bytes) {
        tma::reduce_add_noftz_f16_async_commit<Scope>(dst, smem, bytes);
    }

    __device__ __forceinline__ static scalar_t apply_tail(
        scalar_t oldv,
        scalar_t newv) {
        return __float2half_rn(__half2float(oldv) + __half2float(newv));
    }
};

struct PipelineReduceBatchAddBF16 {
    using scalar_t = __nv_bfloat16;

    template <tma::TmaReduceScope Scope>
    __device__ __forceinline__ static void issue_bulk_op_nofence(
        void* dst,
        void* smem,
        uint32_t bytes) {
        tma::reduce_add_bf16_async_op_nofence<Scope>(dst, smem, bytes);
    }

    template <tma::TmaReduceScope Scope>
    __device__ __forceinline__ static void issue_bulk_op(
        void* dst,
        void* smem,
        uint32_t bytes) {
        tma::reduce_add_bf16_async_op<Scope>(dst, smem, bytes);
    }

    template <tma::TmaReduceScope Scope>
    __device__ __forceinline__ static void issue_bulk_commit(
        void* dst,
        void* smem,
        uint32_t bytes) {
        tma::reduce_add_bf16_async_commit<Scope>(dst, smem, bytes);
    }

    __device__ __forceinline__ static scalar_t apply_tail(
        scalar_t oldv,
        scalar_t newv) {
        return __float2bfloat16_rn(
            __bfloat162float(oldv) + __bfloat162float(newv));
    }
};

struct PipelineReduceBatchAddF32 {
    using scalar_t = float;

    template <tma::TmaReduceScope Scope>
    __device__ __forceinline__ static void issue_bulk_op_nofence(
        void* dst,
        void* smem,
        uint32_t bytes) {
        tma::reduce_add_f32_async_op_nofence<Scope>(dst, smem, bytes);
    }

    template <tma::TmaReduceScope Scope>
    __device__ __forceinline__ static void issue_bulk_op(
        void* dst,
        void* smem,
        uint32_t bytes) {
        tma::reduce_add_f32_async_op<Scope>(dst, smem, bytes);
    }

    template <tma::TmaReduceScope Scope>
    __device__ __forceinline__ static void issue_bulk_commit(
        void* dst,
        void* smem,
        uint32_t bytes) {
        tma::reduce_add_f32_async_commit<Scope>(dst, smem, bytes);
    }

    __device__ __forceinline__ static scalar_t apply_tail(
        scalar_t oldv,
        scalar_t newv) {
        return oldv + newv;
    }
};

struct PipelineReduceBatchMinF16 {
    using scalar_t = half;

    template <tma::TmaReduceScope Scope>
    __device__ __forceinline__ static void issue_bulk_op_nofence(
        void* dst,
        void* smem,
        uint32_t bytes) {
        tma::reduce_min_f16_async_op_nofence<Scope>(dst, smem, bytes);
    }

    template <tma::TmaReduceScope Scope>
    __device__ __forceinline__ static void issue_bulk_op(
        void* dst,
        void* smem,
        uint32_t bytes) {
        tma::reduce_min_f16_async_op<Scope>(dst, smem, bytes);
    }

    template <tma::TmaReduceScope Scope>
    __device__ __forceinline__ static void issue_bulk_commit(
        void* dst,
        void* smem,
        uint32_t bytes) {
        tma::reduce_min_f16_async_commit<Scope>(dst, smem, bytes);
    }

    __device__ __forceinline__ static scalar_t apply_tail(
        scalar_t oldv,
        scalar_t newv) {
        const float a = __half2float(oldv);
        const float b = __half2float(newv);
        return __float2half_rn((a < b) ? a : b);
    }
};

struct PipelineReduceBatchMinBF16 {
    using scalar_t = __nv_bfloat16;

    template <tma::TmaReduceScope Scope>
    __device__ __forceinline__ static void issue_bulk_op_nofence(
        void* dst,
        void* smem,
        uint32_t bytes) {
        tma::reduce_min_bf16_async_op_nofence<Scope>(dst, smem, bytes);
    }

    template <tma::TmaReduceScope Scope>
    __device__ __forceinline__ static void issue_bulk_op(
        void* dst,
        void* smem,
        uint32_t bytes) {
        tma::reduce_min_bf16_async_op<Scope>(dst, smem, bytes);
    }

    template <tma::TmaReduceScope Scope>
    __device__ __forceinline__ static void issue_bulk_commit(
        void* dst,
        void* smem,
        uint32_t bytes) {
        tma::reduce_min_bf16_async_commit<Scope>(dst, smem, bytes);
    }

    __device__ __forceinline__ static scalar_t apply_tail(
        scalar_t oldv,
        scalar_t newv) {
        const float a = __bfloat162float(oldv);
        const float b = __bfloat162float(newv);
        return __float2bfloat16_rn((a < b) ? a : b);
    }
};

struct PipelineReduceBatchMaxF16 {
    using scalar_t = half;

    template <tma::TmaReduceScope Scope>
    __device__ __forceinline__ static void issue_bulk_op_nofence(
        void* dst,
        void* smem,
        uint32_t bytes) {
        tma::reduce_max_f16_async_op_nofence<Scope>(dst, smem, bytes);
    }

    template <tma::TmaReduceScope Scope>
    __device__ __forceinline__ static void issue_bulk_op(
        void* dst,
        void* smem,
        uint32_t bytes) {
        tma::reduce_max_f16_async_op<Scope>(dst, smem, bytes);
    }

    template <tma::TmaReduceScope Scope>
    __device__ __forceinline__ static void issue_bulk_commit(
        void* dst,
        void* smem,
        uint32_t bytes) {
        tma::reduce_max_f16_async_commit<Scope>(dst, smem, bytes);
    }

    __device__ __forceinline__ static scalar_t apply_tail(
        scalar_t oldv,
        scalar_t newv) {
        const float a = __half2float(oldv);
        const float b = __half2float(newv);
        return __float2half_rn((a > b) ? a : b);
    }
};

struct PipelineReduceBatchMaxBF16 {
    using scalar_t = __nv_bfloat16;

    template <tma::TmaReduceScope Scope>
    __device__ __forceinline__ static void issue_bulk_op_nofence(
        void* dst,
        void* smem,
        uint32_t bytes) {
        tma::reduce_max_bf16_async_op_nofence<Scope>(dst, smem, bytes);
    }

    template <tma::TmaReduceScope Scope>
    __device__ __forceinline__ static void issue_bulk_op(
        void* dst,
        void* smem,
        uint32_t bytes) {
        tma::reduce_max_bf16_async_op<Scope>(dst, smem, bytes);
    }

    template <tma::TmaReduceScope Scope>
    __device__ __forceinline__ static void issue_bulk_commit(
        void* dst,
        void* smem,
        uint32_t bytes) {
        tma::reduce_max_bf16_async_commit<Scope>(dst, smem, bytes);
    }

    __device__ __forceinline__ static scalar_t apply_tail(
        scalar_t oldv,
        scalar_t newv) {
        const float a = __bfloat162float(oldv);
        const float b = __bfloat162float(newv);
        return __float2bfloat16_rn((a > b) ? a : b);
    }
};

/*
 * Batched TMA reduce pipeline.
 *
 * Scope should be Gpu for the experiment where different CTAs on the same GPU
 * reduce different remote inputs into the same destination address range.
 *
 * Example:
 *   using Reduce =
 *       PipelineTMAReduceBatch<
 *           StageDepth,
 *           FillDepth,
 *           PipelineReduceBatchAddNoFtzF16,
 *           tma::TmaReduceScope::Gpu>;
 *
 *   reduce.issue_bulk_op_nofence(&stage, dst_override);
 *   reduce.commit();
 */
template <
    int StageDepth,
    int FillDepth,
    typename ReduceOp,
    tma::TmaReduceScope Scope =
        static_cast<tma::TmaReduceScope>(OOVERLAP_TMA_REDUCE_DEFAULT_SCOPE)>
struct PipelineTMAReduceBatch {
    static_assert(StageDepth > 0, "StageDepth must be > 0");
    static_assert(FillDepth > 0, "FillDepth must be > 0");
    static_assert(FillDepth <= StageDepth, "FillDepth must be <= StageDepth");

    using scalar_t = typename ReduceOp::scalar_t;

    __device__ __forceinline__ void wait_before_stage_reuse() const {
        tma::reduce_async_read_wait<FillDepth - 1>();
    }

    __device__ __forceinline__ void wait_complete() const {
        tma::reduce_async_wait<0>();
    }

    __device__ __forceinline__ void issue_fence() const {
        tma::reduce_fence_proxy_async_shared_cta();
    }

    __device__ __forceinline__ void commit() const {
        tma::reduce_commit_group();
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

        ReduceOp::template issue_bulk_op_nofence<Scope>(
            dst_override,
            stage->smem,
            static_cast<uint32_t>(bulk_bytes));
    }

    /*
     * Fence + reduce op, but no commit.
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

        ReduceOp::template issue_bulk_op<Scope>(
            dst_override,
            stage->smem,
            static_cast<uint32_t>(bulk_bytes));
    }

    /*
     * Fence + reduce op + commit.
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

        ReduceOp::template issue_bulk_commit<Scope>(
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
     * Common case for local batching: one fence, multiple reduce ops,
     * one commit.  This version handles two already-loaded stages.
     */
    __device__ __forceinline__ void issue_pair_commit(
        const PipelineStage* stage0,
        const PipelineStage* stage1,
        void* dst0,
        void* dst1) const {
        issue_fence();
        issue_bulk_op_nofence(stage0, dst0);
        issue_bulk_op_nofence(stage1, dst1);
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

        const size_t bulk_elems = bulk_bytes / sizeof(scalar_t);
        const size_t tail_elems = tail_bytes / sizeof(scalar_t);

        scalar_t* dst =
            reinterpret_cast<scalar_t*>(dst_override);

        const scalar_t* src =
            reinterpret_cast<const scalar_t*>(stage->smem);

        for (size_t i = threadIdx.x; i < tail_elems; i += blockDim.x) {
            const size_t idx = bulk_elems + i;
            dst[idx] = ReduceOp::apply_tail(dst[idx], src[idx]);
        }
    }
};

} // namespace pipeline
} // namespace comm
} // namespace ooverlap
