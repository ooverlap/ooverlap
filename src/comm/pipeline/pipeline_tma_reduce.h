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

#define OOVERLAP_PIPELINE_REDUCE_DEFINE_NOFENCE(TMA_BASE_NAME)               \
    template <tma::TmaReduceScope Scope>                                      \
    __device__ __forceinline__ static void issue_bulk_op_nofence(             \
        void* dst,                                                            \
        void* smem,                                                           \
        uint32_t bytes) {                                                     \
        tma::TMA_BASE_NAME##_op_nofence<Scope>(dst, smem, bytes);             \
    }

struct PipelineReduceAddF16 {
    using scalar_t = half;

    OOVERLAP_PIPELINE_REDUCE_DEFINE_NOFENCE(reduce_add_f16_async)

    __device__ __forceinline__ static void issue_bulk(
        void* dst,
        void* smem,
        uint32_t bytes) {
        tma::reduce_add_f16_async(dst, smem, bytes);
    }

    __device__ __forceinline__ static scalar_t apply_tail(
        scalar_t oldv,
        scalar_t newv) {
        return __float2half_rn(__half2float(oldv) + __half2float(newv));
    }
};

struct PipelineReduceAddNoFtzF16 {
    using scalar_t = half;

    OOVERLAP_PIPELINE_REDUCE_DEFINE_NOFENCE(reduce_add_noftz_f16_async)

    __device__ __forceinline__ static void issue_bulk(
        void* dst,
        void* smem,
        uint32_t bytes) {
        tma::reduce_add_noftz_f16_async(dst, smem, bytes);
    }

    __device__ __forceinline__ static scalar_t apply_tail(
        scalar_t oldv,
        scalar_t newv) {
        return __float2half_rn(__half2float(oldv) + __half2float(newv));
    }
};

struct PipelineReduceMinF16 {
    using scalar_t = half;

    OOVERLAP_PIPELINE_REDUCE_DEFINE_NOFENCE(reduce_min_f16_async)

    __device__ __forceinline__ static void issue_bulk(
        void* dst,
        void* smem,
        uint32_t bytes) {
        tma::reduce_min_f16_async(dst, smem, bytes);
    }

    __device__ __forceinline__ static scalar_t apply_tail(
        scalar_t oldv,
        scalar_t newv) {
        const float a = __half2float(oldv);
        const float b = __half2float(newv);
        return __float2half_rn((a < b) ? a : b);
    }
};

struct PipelineReduceMaxF16 {
    using scalar_t = half;

    OOVERLAP_PIPELINE_REDUCE_DEFINE_NOFENCE(reduce_max_f16_async)

    __device__ __forceinline__ static void issue_bulk(
        void* dst,
        void* smem,
        uint32_t bytes) {
        tma::reduce_max_f16_async(dst, smem, bytes);
    }

    __device__ __forceinline__ static scalar_t apply_tail(
        scalar_t oldv,
        scalar_t newv) {
        const float a = __half2float(oldv);
        const float b = __half2float(newv);
        return __float2half_rn((a > b) ? a : b);
    }
};

struct PipelineReduceAddBF16 {
    using scalar_t = __nv_bfloat16;

    OOVERLAP_PIPELINE_REDUCE_DEFINE_NOFENCE(reduce_add_noftz_bf16_async)

    __device__ __forceinline__ static void issue_bulk(
        void* dst,
        void* smem,
        uint32_t bytes) {
        tma::reduce_add_noftz_bf16_async(dst, smem, bytes);
    }

    __device__ __forceinline__ static scalar_t apply_tail(
        scalar_t oldv,
        scalar_t newv) {
        return __float2bfloat16_rn(
            __bfloat162float(oldv) + __bfloat162float(newv));
    }
};

struct PipelineReduceMinBF16 {
    using scalar_t = __nv_bfloat16;

    OOVERLAP_PIPELINE_REDUCE_DEFINE_NOFENCE(reduce_min_bf16_async)

    __device__ __forceinline__ static void issue_bulk(
        void* dst,
        void* smem,
        uint32_t bytes) {
        tma::reduce_min_bf16_async(dst, smem, bytes);
    }

    __device__ __forceinline__ static scalar_t apply_tail(
        scalar_t oldv,
        scalar_t newv) {
        const float a = __bfloat162float(oldv);
        const float b = __bfloat162float(newv);
        return __float2bfloat16_rn((a < b) ? a : b);
    }
};

struct PipelineReduceMaxBF16 {
    using scalar_t = __nv_bfloat16;

    OOVERLAP_PIPELINE_REDUCE_DEFINE_NOFENCE(reduce_max_bf16_async)

    __device__ __forceinline__ static void issue_bulk(
        void* dst,
        void* smem,
        uint32_t bytes) {
        tma::reduce_max_bf16_async(dst, smem, bytes);
    }

    __device__ __forceinline__ static scalar_t apply_tail(
        scalar_t oldv,
        scalar_t newv) {
        const float a = __bfloat162float(oldv);
        const float b = __bfloat162float(newv);
        return __float2bfloat16_rn((a > b) ? a : b);
    }
};

struct PipelineReduceAddF32 {
    using scalar_t = float;

    OOVERLAP_PIPELINE_REDUCE_DEFINE_NOFENCE(reduce_add_f32_async)

    __device__ __forceinline__ static void issue_bulk(
        void* dst,
        void* smem,
        uint32_t bytes) {
        tma::reduce_add_f32_async(dst, smem, bytes);
    }

    __device__ __forceinline__ static scalar_t apply_tail(
        scalar_t oldv,
        scalar_t newv) {
        return oldv + newv;
    }
};

#undef OOVERLAP_PIPELINE_REDUCE_DEFINE_NOFENCE

template <int StageDepth, int FillDepth, typename ReduceOp>
struct PipelineTMAReduce {
    static_assert(StageDepth > 0, "StageDepth must be > 0");
    static_assert(FillDepth > 0, "FillDepth must be > 0");
    static_assert(FillDepth <= StageDepth, "FillDepth must be <= StageDepth");

    using scalar_t = typename ReduceOp::scalar_t;

    __device__ __forceinline__ void wait_before_stage_reuse() const {
        tma::reduce_async_read_wait<FillDepth - 1>();
    }

    __device__ __forceinline__ void issue_fence() const {
        tma::reduce_fence_proxy_async_shared_cta();
    }

    __device__ __forceinline__ void commit() const {
        tma::reduce_commit_group();
    }

    template <tma::TmaReduceScope Scope>
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

    template <tma::TmaReduceScope Scope>
    __device__ __forceinline__ void issue_bulk_op_nofence_with_scope_fallback(
        const PipelineStage* stage,
        void* dst_override) const {
        if constexpr (
            Scope == tma::TmaReduceScope::Default ||
            OOVERLAP_TMA_REDUCE_HAS_PTX93_SCOPE != 0) {
            issue_bulk_op_nofence<Scope>(stage, dst_override);
        } else {
            issue_bulk_op_nofence<tma::TmaReduceScope::Default>(
                stage,
                dst_override);
        }
    }

    __device__ __forceinline__ void issue_bulk_op_nofence(
        const PipelineStage* stage,
        void* dst_override,
        tma::TmaReduceScope scope) const {
        switch (scope) {
            case tma::TmaReduceScope::Cta:
                issue_bulk_op_nofence_with_scope_fallback<
                    tma::TmaReduceScope::Cta>(stage, dst_override);
                return;
            case tma::TmaReduceScope::Cluster:
                issue_bulk_op_nofence_with_scope_fallback<
                    tma::TmaReduceScope::Cluster>(stage, dst_override);
                return;
            case tma::TmaReduceScope::Gpu:
                issue_bulk_op_nofence_with_scope_fallback<
                    tma::TmaReduceScope::Gpu>(stage, dst_override);
                return;
            case tma::TmaReduceScope::Sys:
                issue_bulk_op_nofence_with_scope_fallback<
                    tma::TmaReduceScope::Sys>(stage, dst_override);
                return;
            case tma::TmaReduceScope::Default:
            default:
                issue_bulk_op_nofence<tma::TmaReduceScope::Default>(
                    stage,
                    dst_override);
                return;
        }
    }

    __device__ __forceinline__ void issue_bulk(
        const PipelineStage* stage) const {
        const size_t bulk_bytes = pipeline_stage_bulk_bytes(stage);

        if (bulk_bytes == 0) {
            return;
        }

        ReduceOp::issue_bulk(
            stage->chunk.dst,
            stage->smem,
            static_cast<uint32_t>(bulk_bytes));
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

    __device__ __forceinline__ void wait_complete() const {
        tma::reduce_async_wait<0>();
    }
};

} // namespace pipeline
} // namespace comm
} // namespace ooverlap
