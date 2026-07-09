#pragma once

#include <cuda_runtime.h>
#include <stdint.h>

#include "ooverlap/sync/sync.cuh"

namespace ooverlap {
namespace tma {

__device__ __forceinline__ uint64_t cvta_to_global_u64(const void* ptr) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    uint64_t out;
    asm volatile(
        "cvta.to.global.u64 %0, %1;\n"
        : "=l"(out)
        : "l"(ptr));
    return out;
#else
    return reinterpret_cast<uint64_t>(ptr);
#endif
}

__device__ __forceinline__ uint32_t cvta_to_shared_u32(const void* ptr) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
}

__device__ __forceinline__ void expect_bytes(sync::semaphore& bar, uint32_t bytes) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    const uint32_t bar_ptr = cvta_to_shared_u32(&bar);
    asm volatile(
        "mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n"
        :
        : "r"(bar_ptr), "r"(bytes)
        : "memory");
#else
    (void)bar;
    (void)bytes;
#endif
}

__device__ __forceinline__ void load_async(
    void* dst_smem,
    const void* src_gmem,
    uint32_t size_bytes,
    sync::semaphore& bar) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    const uint32_t dst_smem_addr = cvta_to_shared_u32(dst_smem);
    const uint32_t bar_addr = cvta_to_shared_u32(&bar);
    const uint64_t src_gmem_addr = cvta_to_global_u64(src_gmem);

    asm volatile(
        "cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes "
        "[%0], [%1], %2, [%3];\n"
        :
        : "r"(dst_smem_addr),
          "l"(src_gmem_addr),
          "r"(size_bytes),
          "r"(bar_addr)
        : "memory");
#else
    (void)bar;

    char* d =
        reinterpret_cast<char*>(dst_smem);

    const char* s =
        reinterpret_cast<const char*>(src_gmem);

    for (uint32_t i = 0; i < size_bytes; ++i) {
        d[i] = s[i];
    }
#endif
}

// -----------------------------------------------------------------------------
// STORE-SIDE HELPERS (shared -> global)
// -----------------------------------------------------------------------------
//
// Existing API:
//   store_async(dst, src, bytes)
//
// still means:
//   fence.proxy.async.shared::cta;
//   cp.async.bulk.global.shared::cta.bulk_group;
//   cp.async.bulk.commit_group;
//
// New batching API:
//
//   store_fence_proxy_async_shared_cta();
//
//   store_async_op_nofence(dst0, src0, bytes0);
//   store_async_op_nofence(dst1, src1, bytes1);
//   store_async_op_nofence(dst2, src2, bytes2);
//
//   store_commit_group();
//
// or, when each store needs its own shared-proxy fence but not its own commit:
//
//   store_async_op(dst0, src0, bytes0);
//   store_async_op(dst1, src1, bytes1);
//   store_async_op(dst2, src2, bytes2);
//   store_commit_group();
//
// There is no `.relaxed.<scope>` here.  That scope qualifier belongs to the
// PTX 9.3 cp.reduce.async.bulk destination reductions, not this plain
// shared-to-global bulk store path.
// -----------------------------------------------------------------------------

__device__ __forceinline__ void store_fence_proxy_async_shared_cta() {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
#endif
}

__device__ __forceinline__ void store_commit_group() {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    asm volatile("cp.async.bulk.commit_group;\n" ::: "memory");
#endif
}

template <int N = 0>
__device__ __forceinline__ void store_async_wait() {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    asm volatile(
        "cp.async.bulk.wait_group %0;\n"
        :
        : "n"(N)
        : "memory");
#endif
}

template <int N = 0>
__device__ __forceinline__ void store_async_read_wait() {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    asm volatile(
        "cp.async.bulk.wait_group.read %0;\n"
        :
        : "n"(N)
        : "memory");
#endif
}

/*
 * Raw store operation, without fence and without commit_group.
 *
 * Use this after one explicit store_fence_proxy_async_shared_cta() when the
 * source shared-memory region has already been made visible to the async proxy.
 */
__device__ __forceinline__ void store_async_op_nofence(
    void* dst_gmem,
    void* src_smem,
    uint32_t size_bytes) {
    if (size_bytes == 0) {
        return;
    }

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    asm volatile(
        "cp.async.bulk.global.shared::cta.bulk_group "
        "[%0], [%1], %2;\n"
        :
        : "l"(cvta_to_global_u64(dst_gmem)),
          "r"(cvta_to_shared_u32(src_smem)),
          "r"(size_bytes)
        : "memory");
#else
    char* d = reinterpret_cast<char*>(dst_gmem);
    const char* s = reinterpret_cast<const char*>(src_smem);
    for (uint32_t i = 0; i < size_bytes; ++i) {
        d[i] = s[i];
    }
#endif
}

/*
 * Store operation with fence, but without commit_group.
 *
 * This is useful when each source region may have been produced independently,
 * but you still want to batch multiple cp.async.bulk operations into a single
 * commit group.
 */
__device__ __forceinline__ void store_async_op(
    void* dst_gmem,
    void* src_smem,
    uint32_t size_bytes) {
    if (size_bytes == 0) {
        return;
    }

    store_fence_proxy_async_shared_cta();
    store_async_op_nofence(dst_gmem, src_smem, size_bytes);
}

/*
 * Store operation with old one-op-one-commit behavior, but exposed under an
 * explicit name so call sites can choose between store_async_op[_nofence] and
 * store_commit_group().
 */
__device__ __forceinline__ void store_async_commit(
    void* dst_gmem,
    void* src_smem,
    uint32_t size_bytes) {
    store_async_op(dst_gmem, src_smem, size_bytes);

    if (size_bytes != 0) {
        store_commit_group();
    }
}

/*
 * Backward-compatible name.
 */
__device__ __forceinline__ void store_async(
    void* dst_gmem,
    void* src_smem,
    uint32_t size_bytes) {
    store_async_commit(dst_gmem, src_smem, size_bytes);
}


// -----------------------------------------------------------------------------
// FANOUT STORE UTILITIES
// -----------------------------------------------------------------------------
//
// OOVERLAP_TMA_STORE_FANOUT_UTIL_PATCH:
//
// These helpers keep the low-level TMA behavior explicit while making fanout
// call sites cleaner.  They do not add memory-scope qualifiers because plain
// cp.async.bulk.global.shared::cta.bulk_group does not have the PTX 9.3
// reduction `.relaxed.<scope>` qualifier.
//
// Argument order is source-smem first, then many global destinations:
//
//   store_async_fanout_commit(
//       smem,
//       bytes,
//       dst0,
//       dst1,
//       dst2);
//
// Split form:
//
//   store_fence_proxy_async_shared_cta();
//   store_async_fanout_op_nofence(smem, bytes, dst0, dst1, dst2);
//   store_commit_group();
// -----------------------------------------------------------------------------

template <typename... DstPtrs>
__device__ __forceinline__ void store_async_fanout_op_nofence(
    void* src_smem,
    uint32_t size_bytes,
    DstPtrs... dst_gmems) {
    static_assert(
        sizeof...(DstPtrs) > 0,
        "store_async_fanout_op_nofence requires at least one destination");

    if (size_bytes == 0) {
        return;
    }

    (store_async_op_nofence(
         dst_gmems,
         src_smem,
         size_bytes),
     ...);
}

template <typename... DstPtrs>
__device__ __forceinline__ void store_async_fanout_op(
    void* src_smem,
    uint32_t size_bytes,
    DstPtrs... dst_gmems) {
    static_assert(
        sizeof...(DstPtrs) > 0,
        "store_async_fanout_op requires at least one destination");

    if (size_bytes == 0) {
        return;
    }

    store_fence_proxy_async_shared_cta();

    store_async_fanout_op_nofence(
        src_smem,
        size_bytes,
        dst_gmems...);
}

template <typename... DstPtrs>
__device__ __forceinline__ void store_async_fanout_commit(
    void* src_smem,
    uint32_t size_bytes,
    DstPtrs... dst_gmems) {
    static_assert(
        sizeof...(DstPtrs) > 0,
        "store_async_fanout_commit requires at least one destination");

    store_async_fanout_op(
        src_smem,
        size_bytes,
        dst_gmems...);

    if (size_bytes != 0) {
        store_commit_group();
    }
}

} // namespace tma
} // namespace ooverlap
