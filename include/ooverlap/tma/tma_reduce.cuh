#pragma once

#include <cuda_runtime.h>
#include <stdint.h>

#include "ooverlap/tma/tma.cuh"

namespace ooverlap {
namespace tma {

// -----------------------------------------------------------------------------
// SM90 TMA bulk reduction helpers
// -----------------------------------------------------------------------------
//
// This version keeps the old API intact, but adds two capabilities:
//
//   1. Optional PTX 9.3 explicit reduction scope:
//        TmaReduceScope::Cta
//        TmaReduceScope::Cluster
//        TmaReduceScope::Gpu
//        TmaReduceScope::Sys
//
//      The legacy/default path still emits the old instruction form:
//        cp.reduce.async.bulk.global.shared::cta.bulk_group...
//
//      On PTX 9.3, the old omitted-scope form is equivalent to
//      .relaxed.sys for the destination element-wise atomic reductions.
//
//   2. Split operation from commit:
//        *_op_nofence<Scope>(...)  emits only the cp.reduce instruction
//        *_op<Scope>(...)          emits fence + cp.reduce
//        *_commit<Scope>(...)      emits fence + cp.reduce + commit_group
//
//      The old functions keep old behavior:
//        reduce_add_f16_async(...) emits fence + cp.reduce + commit_group
//
// For quick compile-time testing of existing call sites, define:
//      OOVERLAP_TMA_REDUCE_DEFAULT_SCOPE=1  // Cta
//      OOVERLAP_TMA_REDUCE_DEFAULT_SCOPE=2  // Cluster
//      OOVERLAP_TMA_REDUCE_DEFAULT_SCOPE=3  // Gpu
//      OOVERLAP_TMA_REDUCE_DEFAULT_SCOPE=4  // Sys, explicit
//
// Leave it undefined or set to 0 for the exact old instruction spelling.
// -----------------------------------------------------------------------------

#ifndef OOVERLAP_TMA_REDUCE_HAS_PTX93_SCOPE
/*
 * Set this to 0 if building with a PTX version/toolchain that rejects
 * cp.reduce.async.bulk.relaxed.<scope>.
 *
 * Existing old call sites still default to the old no-.sem.scope instruction
 * spelling through OOVERLAP_TMA_REDUCE_DEFAULT_SCOPE=0.
 */
#define OOVERLAP_TMA_REDUCE_HAS_PTX93_SCOPE 1
#endif

#ifndef OOVERLAP_TMA_REDUCE_DEFAULT_SCOPE
/*
 * 0 = Default legacy spelling, no explicit .sem.scope.
 * 1 = .relaxed.cta
 * 2 = .relaxed.cluster
 * 3 = .relaxed.gpu
 * 4 = .relaxed.sys
 */
#define OOVERLAP_TMA_REDUCE_DEFAULT_SCOPE 0
#endif

enum class TmaReduceScope : int {
    Default = 0,
    Cta = 1,
    Cluster = 2,
    Gpu = 3,
    Sys = 4,
};

template <TmaReduceScope>
struct TmaReduceExplicitScopeRequiresPtx93 {
    static constexpr bool value = false;
};

__device__ __forceinline__ void reduce_fence_proxy_async_shared_cta() {
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
}

__device__ __forceinline__ void reduce_commit_group() {
    asm volatile("cp.async.bulk.commit_group;\n" ::: "memory");
}

template <int N = 0>
__device__ __forceinline__ void reduce_async_wait() {
    asm volatile(
        "cp.async.bulk.wait_group %0;\n"
        :
        : "n"(N)
        : "memory");
}

template <int N = 0>
__device__ __forceinline__ void reduce_async_read_wait() {
    asm volatile(
        "cp.async.bulk.wait_group.read %0;\n"
        :
        : "n"(N)
        : "memory");
}

#define OOVERLAP_TMA_REDUCE_DEFAULT_SCOPE_ENUM \
    static_cast<::ooverlap::tma::TmaReduceScope>(OOVERLAP_TMA_REDUCE_DEFAULT_SCOPE)

#define OOVERLAP_TMA_REDUCE_EMIT_LEGACY(PTX_SUFFIX)                         \
    asm volatile(                                                            \
        "cp.reduce.async.bulk.global.shared::cta.bulk_group." PTX_SUFFIX " " \
        "[%0], [%1], %2;\n"                                                  \
        :                                                                    \
        : "l"(cvta_to_global_u64(dst_gmem)),                                 \
          "r"(cvta_to_shared_u32(src_smem)),                                 \
          "r"(size_bytes)                                                    \
        : "memory")

#define OOVERLAP_TMA_REDUCE_EMIT_SCOPED(SCOPE_TOKEN, PTX_SUFFIX)                     \
    asm volatile(                                                                     \
        "cp.reduce.async.bulk.relaxed." SCOPE_TOKEN ".global.shared::cta.bulk_group." \
        PTX_SUFFIX " "                                                                \
        "[%0], [%1], %2;\n"                                                           \
        :                                                                             \
        : "l"(cvta_to_global_u64(dst_gmem)),                                          \
          "r"(cvta_to_shared_u32(src_smem)),                                          \
          "r"(size_bytes)                                                             \
        : "memory")

#if OOVERLAP_TMA_REDUCE_HAS_PTX93_SCOPE
#define OOVERLAP_TMA_REDUCE_EMIT_EXPLICIT_SCOPE(PTX_SUFFIX)                 \
    if constexpr (Scope == TmaReduceScope::Cta) {                           \
        OOVERLAP_TMA_REDUCE_EMIT_SCOPED("cta", PTX_SUFFIX);                 \
    } else if constexpr (Scope == TmaReduceScope::Cluster) {                \
        OOVERLAP_TMA_REDUCE_EMIT_SCOPED("cluster", PTX_SUFFIX);             \
    } else if constexpr (Scope == TmaReduceScope::Gpu) {                    \
        OOVERLAP_TMA_REDUCE_EMIT_SCOPED("gpu", PTX_SUFFIX);                 \
    } else if constexpr (Scope == TmaReduceScope::Sys) {                    \
        OOVERLAP_TMA_REDUCE_EMIT_SCOPED("sys", PTX_SUFFIX);                 \
    } else {                                                                \
        static_assert(                                                      \
            TmaReduceExplicitScopeRequiresPtx93<Scope>::value,              \
            "invalid TmaReduceScope");                                      \
    }
#else
#define OOVERLAP_TMA_REDUCE_EMIT_EXPLICIT_SCOPE(PTX_SUFFIX)                 \
    static_assert(                                                          \
        TmaReduceExplicitScopeRequiresPtx93<Scope>::value,                  \
        "explicit cp.reduce.async.bulk scope requires PTX 9.3; "             \
        "set OOVERLAP_TMA_REDUCE_HAS_PTX93_SCOPE=1 only when "              \
        "the toolchain accepts .relaxed.<scope>")
#endif

#define OOVERLAP_TMA_DEFINE_REDUCE_OP(BASE_NAME, PTX_SUFFIX)                 \
template <TmaReduceScope Scope>                                              \
__device__ __forceinline__ void BASE_NAME##_emit(                            \
    void* dst_gmem,                                                          \
    void* src_smem,                                                          \
    uint32_t size_bytes) {                                                   \
    if constexpr (Scope == TmaReduceScope::Default) {                        \
        OOVERLAP_TMA_REDUCE_EMIT_LEGACY(PTX_SUFFIX);                         \
    } else {                                                                 \
        OOVERLAP_TMA_REDUCE_EMIT_EXPLICIT_SCOPE(PTX_SUFFIX);                 \
    }                                                                        \
}                                                                            \
                                                                             \
template <TmaReduceScope Scope = OOVERLAP_TMA_REDUCE_DEFAULT_SCOPE_ENUM>     \
__device__ __forceinline__ void BASE_NAME##_op_nofence(                      \
    void* dst_gmem,                                                          \
    void* src_smem,                                                          \
    uint32_t size_bytes) {                                                   \
    if (size_bytes == 0) {                                                   \
        return;                                                              \
    }                                                                        \
    BASE_NAME##_emit<Scope>(dst_gmem, src_smem, size_bytes);                 \
}                                                                            \
                                                                             \
template <TmaReduceScope Scope = OOVERLAP_TMA_REDUCE_DEFAULT_SCOPE_ENUM>     \
__device__ __forceinline__ void BASE_NAME##_op(                              \
    void* dst_gmem,                                                          \
    void* src_smem,                                                          \
    uint32_t size_bytes) {                                                   \
    if (size_bytes == 0) {                                                   \
        return;                                                              \
    }                                                                        \
    reduce_fence_proxy_async_shared_cta();                                   \
    BASE_NAME##_emit<Scope>(dst_gmem, src_smem, size_bytes);                 \
}                                                                            \
                                                                             \
template <TmaReduceScope Scope = OOVERLAP_TMA_REDUCE_DEFAULT_SCOPE_ENUM>     \
__device__ __forceinline__ void BASE_NAME##_commit(                          \
    void* dst_gmem,                                                          \
    void* src_smem,                                                          \
    uint32_t size_bytes) {                                                   \
    BASE_NAME##_op<Scope>(dst_gmem, src_smem, size_bytes);                   \
    if (size_bytes != 0) {                                                   \
        reduce_commit_group();                                               \
    }                                                                        \
}                                                                            \
                                                                             \
__device__ __forceinline__ void BASE_NAME(                                   \
    void* dst_gmem,                                                          \
    void* src_smem,                                                          \
    uint32_t size_bytes) {                                                   \
    BASE_NAME##_commit<>(dst_gmem, src_smem, size_bytes);                    \
}

// -----------------------------------------------------------------------------
// add
// -----------------------------------------------------------------------------

OOVERLAP_TMA_DEFINE_REDUCE_OP(reduce_add_f16_async, "add.f16")
OOVERLAP_TMA_DEFINE_REDUCE_OP(reduce_add_noftz_f16_async, "add.noftz.f16")
OOVERLAP_TMA_DEFINE_REDUCE_OP(reduce_add_noftz_bf16_async, "add.noftz.bf16")
OOVERLAP_TMA_DEFINE_REDUCE_OP(reduce_add_f32_async, "add.f32")

// Compatibility name. BF16 add must be noftz for this instruction.
template <TmaReduceScope Scope = OOVERLAP_TMA_REDUCE_DEFAULT_SCOPE_ENUM>
__device__ __forceinline__ void reduce_add_bf16_async_op_nofence(
    void* dst_gmem,
    void* src_smem,
    uint32_t size_bytes) {
    reduce_add_noftz_bf16_async_op_nofence<Scope>(
        dst_gmem,
        src_smem,
        size_bytes);
}

template <TmaReduceScope Scope = OOVERLAP_TMA_REDUCE_DEFAULT_SCOPE_ENUM>
__device__ __forceinline__ void reduce_add_bf16_async_op(
    void* dst_gmem,
    void* src_smem,
    uint32_t size_bytes) {
    reduce_add_noftz_bf16_async_op<Scope>(
        dst_gmem,
        src_smem,
        size_bytes);
}

template <TmaReduceScope Scope = OOVERLAP_TMA_REDUCE_DEFAULT_SCOPE_ENUM>
__device__ __forceinline__ void reduce_add_bf16_async_commit(
    void* dst_gmem,
    void* src_smem,
    uint32_t size_bytes) {
    reduce_add_noftz_bf16_async_commit<Scope>(
        dst_gmem,
        src_smem,
        size_bytes);
}

__device__ __forceinline__ void reduce_add_bf16_async(
    void* dst_gmem,
    void* src_smem,
    uint32_t size_bytes) {
    reduce_add_bf16_async_commit<>(dst_gmem, src_smem, size_bytes);
}

// -----------------------------------------------------------------------------
// min
// -----------------------------------------------------------------------------

OOVERLAP_TMA_DEFINE_REDUCE_OP(reduce_min_f16_async, "min.f16")
OOVERLAP_TMA_DEFINE_REDUCE_OP(reduce_min_bf16_async, "min.bf16")

// -----------------------------------------------------------------------------
// max
// -----------------------------------------------------------------------------

OOVERLAP_TMA_DEFINE_REDUCE_OP(reduce_max_f16_async, "max.f16")
OOVERLAP_TMA_DEFINE_REDUCE_OP(reduce_max_bf16_async, "max.bf16")

#undef OOVERLAP_TMA_DEFINE_REDUCE_OP
#undef OOVERLAP_TMA_REDUCE_EMIT_EXPLICIT_SCOPE
#undef OOVERLAP_TMA_REDUCE_EMIT_SCOPED
#undef OOVERLAP_TMA_REDUCE_EMIT_LEGACY
#undef OOVERLAP_TMA_REDUCE_DEFAULT_SCOPE_ENUM

} // namespace tma
} // namespace ooverlap
