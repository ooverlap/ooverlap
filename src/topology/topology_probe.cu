#include "topology/topology_probe.h"

#include "ooverlap/sync/sync.cuh"
#include "ooverlap/tma/tma.cuh"
#include "ooverlap/tma/tma_reduce.cuh"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdint>
#include <sstream>
#include <string>
#include <vector>
#include <cstdio>

#ifndef OOVERLAP_TOPOLOGY_TRACE
#define OOVERLAP_TOPOLOGY_TRACE 0
#endif

#if OOVERLAP_TOPOLOGY_TRACE
#define TOPO_PROBE_TRACE(fmt, ...)                                             \
    do {                                                                       \
        std::fprintf(                                                          \
            stderr,                                                            \
            "[topo-probe] %s:%d " fmt "\n",                                    \
            __func__,                                                          \
            __LINE__,                                                          \
            ##__VA_ARGS__);                                                    \
        std::fflush(stderr);                                                   \
    } while (0)
#else
#define TOPO_PROBE_TRACE(...)                                                  \
    do {                                                                       \
    } while (0)
#endif

namespace ooverlap {
namespace topology {
namespace detail {
namespace {

constexpr int kMagic = 0x123456;
constexpr int kHalfCount = 8;
constexpr size_t kHalfBytes = kHalfCount * sizeof(half);

ProbeResult pass_result() {
    ProbeResult r{};
    r.attempted = true;
    r.passed = true;
    r.error[0] = '\0';
    return r;
}

ProbeResult fail_result(const std::string& error) {
    ProbeResult r{};
    r.attempted = true;
    r.passed = false;
    std::snprintf(
        r.error,
        sizeof(r.error),
        "%s",
        error.c_str());
    return r;
}

ProbeResult cuda_fail(const char* what, cudaError_t err) {
    std::ostringstream oss;
    oss << what << " failed: " << cudaGetErrorString(err);
    return fail_result(oss.str());
}

bool check(cudaError_t err, const char* what, ProbeResult* out) {
    if (err == cudaSuccess) {
        return true;
    }

    TOPO_PROBE_TRACE("%s failed err=%d %s", what, int(err), cudaGetErrorString(err));

    if (out != nullptr) {
        *out = cuda_fail(what, err);
    }

    (void)cudaGetLastError();
    return false;
}

void fill_half_bits(std::vector<uint16_t>* v, uint16_t bits) {
    v->assign(kHalfCount, bits);
}

bool half_array_all_bits(
    const std::vector<uint16_t>& v,
    uint16_t bits) {
    if (v.size() != kHalfCount) {
        return false;
    }

    for (uint16_t x : v) {
        if (x != bits) {
            return false;
        }
    }

    return true;
}

__global__ void direct_load_store_kernel(
    const int* peer_src,
    int* peer_dst,
    int* local_result) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        const int v = peer_src[0];
        peer_dst[0] = v + 1;
        __threadfence_system();
        local_result[0] = (v == kMagic) ? 1 : -v;
    }
}

__global__ void direct_atomic_add_i32_kernel(
    int* peer_counter,
    int* local_old_value) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        const int old = atomicAdd(peer_counter, 1);
        __threadfence_system();
        local_old_value[0] = old;
    }
}

__global__ void tma_load_f16_kernel(
    const half* src_gmem,
    int* local_status) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    extern __shared__ unsigned char smem[];
    __shared__ sync::semaphore bar;

    if (threadIdx.x == 0) {
        sync::init_semaphore(bar, 1);
    }

    __syncthreads();

    if (threadIdx.x == 0) {
        tma::expect_bytes(bar, static_cast<uint32_t>(kHalfBytes));
        tma::load_async(
            smem,
            src_gmem,
            static_cast<uint32_t>(kHalfBytes),
            bar);
        sync::wait(bar, 0);

        const uint16_t* bits =
            reinterpret_cast<const uint16_t*>(smem);

        int ok = 1;
        #pragma unroll
        for (int i = 0; i < kHalfCount; ++i) {
            if (bits[i] != 0x4000u) {
                ok = 0;
            }
        }

        local_status[0] = ok ? 1 : -static_cast<int>(bits[0]);
    }
#else
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        local_status[0] = -999;
    }
#endif
}

__global__ void tma_store_f16_kernel(
    half* dst_gmem,
    int* local_status) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    extern __shared__ unsigned char smem[];

    if (threadIdx.x == 0) {
        uint16_t* bits =
            reinterpret_cast<uint16_t*>(smem);

        #pragma unroll
        for (int i = 0; i < kHalfCount; ++i) {
            bits[i] = 0x4400u;
        }

        tma::store_async(
            dst_gmem,
            smem,
            static_cast<uint32_t>(kHalfBytes));
        tma::store_async_wait<0>();

        local_status[0] = 1;
    }
#else
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        local_status[0] = -999;
    }
#endif
}

__global__ void tma_reduce_f16_kernel(
    const half* src_gmem,
    half* dst_gmem,
    int* local_status) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    extern __shared__ unsigned char smem[];
    __shared__ sync::semaphore bar;

    if (threadIdx.x == 0) {
        sync::init_semaphore(bar, 1);
    }

    __syncthreads();

    if (threadIdx.x == 0) {
        tma::expect_bytes(bar, static_cast<uint32_t>(kHalfBytes));
        tma::load_async(
            smem,
            src_gmem,
            static_cast<uint32_t>(kHalfBytes),
            bar);
        sync::wait(bar, 0);

        tma::reduce_add_noftz_f16_async(
            dst_gmem,
            smem,
            static_cast<uint32_t>(kHalfBytes));
        tma::reduce_async_wait<0>();

        local_status[0] = 1;
    }
#else
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        local_status[0] = -999;
    }
#endif
}

ProbeResult probe_direct_load_store_impl(
    int src_device,
    int dst_device) {
    TOPO_PROBE_TRACE("ENTER impl direct_load_store %d -> %d", src_device, dst_device);

    ProbeResult result{};

    int* peer_src = nullptr;
    int* peer_dst = nullptr;
    int* local_result = nullptr;

    TOPO_PROBE_TRACE("direct_load_store before cudaSetDevice(dst=%d)", dst_device);
    if (!check(cudaSetDevice(dst_device), "cudaSetDevice(dst)", &result)) {
        return result;
    }

    TOPO_PROBE_TRACE("direct_load_store before cudaMalloc(peer_src)");
    if (!check(cudaMalloc(&peer_src, sizeof(int)), "cudaMalloc(peer_src)", &result)) {
        return result;
    }
    TOPO_PROBE_TRACE("direct_load_store peer_src=%p", static_cast<void*>(peer_src));

    TOPO_PROBE_TRACE("direct_load_store before cudaMalloc(peer_dst)");
    if (!check(cudaMalloc(&peer_dst, sizeof(int)), "cudaMalloc(peer_dst)", &result)) {
        cudaFree(peer_src);
        return result;
    }
    TOPO_PROBE_TRACE("direct_load_store peer_dst=%p", static_cast<void*>(peer_dst));

    TOPO_PROBE_TRACE("direct_load_store before cudaMemset(peer_dst)");
    if (!check(cudaMemset(peer_dst, 0, sizeof(int)), "cudaMemset(peer_dst)", &result)) {
        cudaFree(peer_src);
        cudaFree(peer_dst);
        return result;
    }

    TOPO_PROBE_TRACE("direct_load_store before cudaMemcpy(peer_src <- magic)");
    if (!check(cudaMemcpy(peer_src, &kMagic, sizeof(int), cudaMemcpyHostToDevice),
               "cudaMemcpy(peer_src)", &result)) {
        cudaFree(peer_src);
        cudaFree(peer_dst);
        return result;
    }

    TOPO_PROBE_TRACE("direct_load_store before cudaSetDevice(src=%d)", src_device);
    if (!check(cudaSetDevice(src_device), "cudaSetDevice(src)", &result)) {
        cudaSetDevice(dst_device);
        cudaFree(peer_src);
        cudaFree(peer_dst);
        return result;
    }

    TOPO_PROBE_TRACE("direct_load_store before cudaMalloc(local_result)");
    if (!check(cudaMalloc(&local_result, sizeof(int)), "cudaMalloc(local_result)", &result)) {
        cudaSetDevice(dst_device);
        cudaFree(peer_src);
        cudaFree(peer_dst);
        return result;
    }
    TOPO_PROBE_TRACE("direct_load_store local_result=%p", static_cast<void*>(local_result));

    TOPO_PROBE_TRACE("direct_load_store before cudaMemset(local_result)");
    if (!check(cudaMemset(local_result, 0, sizeof(int)), "cudaMemset(local_result)", &result)) {
        cudaFree(local_result);
        cudaSetDevice(dst_device);
        cudaFree(peer_src);
        cudaFree(peer_dst);
        return result;
    }

    TOPO_PROBE_TRACE("direct_load_store before kernel launch");
    direct_load_store_kernel<<<1, 32>>>(peer_src, peer_dst, local_result);
    TOPO_PROBE_TRACE("direct_load_store after kernel launch before checks");

    if (!check(cudaGetLastError(), "direct_load_store_kernel launch", &result) ||
        !check(cudaDeviceSynchronize(), "direct_load_store_kernel sync", &result)) {
        TOPO_PROBE_TRACE("direct_load_store kernel failed, cleanup");
        cudaFree(local_result);
        cudaSetDevice(dst_device);
        cudaFree(peer_src);
        cudaFree(peer_dst);
        return result;
    }

    int got_local = 0;
    TOPO_PROBE_TRACE("direct_load_store before cudaMemcpy(local_result -> host)");
    if (!check(cudaMemcpy(&got_local, local_result, sizeof(int), cudaMemcpyDeviceToHost),
               "cudaMemcpy(local_result)", &result)) {
        cudaFree(local_result);
        cudaSetDevice(dst_device);
        cudaFree(peer_src);
        cudaFree(peer_dst);
        return result;
    }

    int got_peer = 0;
    TOPO_PROBE_TRACE("direct_load_store before cudaSetDevice(dst copyback=%d)", dst_device);
    if (!check(cudaSetDevice(dst_device), "cudaSetDevice(dst copyback)", &result) ||
        !check(cudaMemcpy(&got_peer, peer_dst, sizeof(int), cudaMemcpyDeviceToHost),
               "cudaMemcpy(peer_dst)", &result)) {
        cudaSetDevice(src_device);
        cudaFree(local_result);
        cudaSetDevice(dst_device);
        cudaFree(peer_src);
        cudaFree(peer_dst);
        return result;
    }

    TOPO_PROBE_TRACE("direct_load_store got_local=%d got_peer=%d", got_local, got_peer);

    TOPO_PROBE_TRACE("direct_load_store cleanup local_result on src=%d", src_device);
    cudaSetDevice(src_device);
    cudaFree(local_result);

    TOPO_PROBE_TRACE("direct_load_store cleanup peer buffers on dst=%d", dst_device);
    cudaSetDevice(dst_device);
    cudaFree(peer_src);
    cudaFree(peer_dst);

    if (got_local != 1 || got_peer != kMagic + 1) {
        std::ostringstream oss;
        oss << "direct load/store mismatch local=" << got_local
            << " peer=" << got_peer;
        TOPO_PROBE_TRACE("LEAVE impl direct_load_store fail: %s", oss.str().c_str());
        return fail_result(oss.str());
    }

    TOPO_PROBE_TRACE("LEAVE impl direct_load_store pass");
    return pass_result();
}

ProbeResult probe_direct_atomic_add_i32_impl(
    int src_device,
    int dst_device) {
    TOPO_PROBE_TRACE("ENTER impl direct_atomic_add_i32 %d -> %d", src_device, dst_device);

    ProbeResult result{};
    int* peer_counter = nullptr;
    int* local_old = nullptr;

    const int initial = 7;

    TOPO_PROBE_TRACE("direct_atomic before cudaSetDevice(dst=%d)", dst_device);
    if (!check(cudaSetDevice(dst_device), "cudaSetDevice(dst)", &result)) {
        return result;
    }

    TOPO_PROBE_TRACE("direct_atomic before cudaMalloc(peer_counter)");
    if (!check(cudaMalloc(&peer_counter, sizeof(int)), "cudaMalloc(peer_counter)", &result)) {
        return result;
    }
    TOPO_PROBE_TRACE("direct_atomic peer_counter=%p", static_cast<void*>(peer_counter));

    TOPO_PROBE_TRACE("direct_atomic before cudaMemcpy(init)");
    if (!check(cudaMemcpy(peer_counter, &initial, sizeof(int), cudaMemcpyHostToDevice),
               "cudaMemcpy(peer_counter)", &result)) {
        cudaFree(peer_counter);
        return result;
    }

    TOPO_PROBE_TRACE("direct_atomic before cudaSetDevice(src=%d)", src_device);
    if (!check(cudaSetDevice(src_device), "cudaSetDevice(src)", &result)) {
        cudaSetDevice(dst_device);
        cudaFree(peer_counter);
        return result;
    }

    TOPO_PROBE_TRACE("direct_atomic before cudaMalloc(local_old)");
    if (!check(cudaMalloc(&local_old, sizeof(int)), "cudaMalloc(local_old)", &result)) {
        cudaSetDevice(dst_device);
        cudaFree(peer_counter);
        return result;
    }
    TOPO_PROBE_TRACE("direct_atomic local_old=%p", static_cast<void*>(local_old));

    if (!check(cudaMemset(local_old, 0, sizeof(int)), "cudaMemset(local_old)", &result)) {
        cudaFree(local_old);
        cudaSetDevice(dst_device);
        cudaFree(peer_counter);
        return result;
    }

    TOPO_PROBE_TRACE("direct_atomic before kernel launch");
    direct_atomic_add_i32_kernel<<<1, 32>>>(peer_counter, local_old);
    TOPO_PROBE_TRACE("direct_atomic after kernel launch");

    if (!check(cudaGetLastError(), "direct_atomic_add_i32_kernel launch", &result) ||
        !check(cudaDeviceSynchronize(), "direct_atomic_add_i32_kernel sync", &result)) {
        cudaFree(local_old);
        cudaSetDevice(dst_device);
        cudaFree(peer_counter);
        return result;
    }

    int got_old = 0;
    if (!check(cudaMemcpy(&got_old, local_old, sizeof(int), cudaMemcpyDeviceToHost),
               "cudaMemcpy(local_old)", &result)) {
        cudaFree(local_old);
        cudaSetDevice(dst_device);
        cudaFree(peer_counter);
        return result;
    }

    int got_counter = 0;
    if (!check(cudaSetDevice(dst_device), "cudaSetDevice(dst copyback)", &result) ||
        !check(cudaMemcpy(&got_counter, peer_counter, sizeof(int), cudaMemcpyDeviceToHost),
               "cudaMemcpy(peer_counter)", &result)) {
        cudaSetDevice(src_device);
        cudaFree(local_old);
        cudaSetDevice(dst_device);
        cudaFree(peer_counter);
        return result;
    }

    TOPO_PROBE_TRACE("direct_atomic got_old=%d got_counter=%d", got_old, got_counter);

    cudaSetDevice(src_device);
    cudaFree(local_old);
    cudaSetDevice(dst_device);
    cudaFree(peer_counter);

    if (got_old != initial || got_counter != initial + 1) {
        std::ostringstream oss;
        oss << "direct atomic32 mismatch old=" << got_old
            << " counter=" << got_counter;
        TOPO_PROBE_TRACE("LEAVE impl direct_atomic fail: %s", oss.str().c_str());
        return fail_result(oss.str());
    }

    TOPO_PROBE_TRACE("LEAVE impl direct_atomic pass");
    return pass_result();
}

ProbeResult probe_tma_load_impl(
    int exec_device,
    const void* src_gmem) {
    TOPO_PROBE_TRACE("ENTER impl tma_load exec=%d src=%p", exec_device, src_gmem);

    ProbeResult result{};
    int* status = nullptr;

    if (!check(cudaSetDevice(exec_device), "cudaSetDevice(exec)", &result)) {
        return result;
    }
    if (!check(cudaMalloc(&status, sizeof(int)), "cudaMalloc(status)", &result)) {
        return result;
    }
    if (!check(cudaMemset(status, 0, sizeof(int)), "cudaMemset(status)", &result)) {
        cudaFree(status);
        return result;
    }

    TOPO_PROBE_TRACE("tma_load before kernel launch");
    tma_load_f16_kernel<<<1, 32, 64>>>(
        reinterpret_cast<const half*>(src_gmem),
        status);
    TOPO_PROBE_TRACE("tma_load after kernel launch");

    if (!check(cudaGetLastError(), "tma_load_f16_kernel launch", &result) ||
        !check(cudaDeviceSynchronize(), "tma_load_f16_kernel sync", &result)) {
        cudaFree(status);
        return result;
    }

    int got = 0;
    if (!check(cudaMemcpy(&got, status, sizeof(int), cudaMemcpyDeviceToHost),
               "cudaMemcpy(status)", &result)) {
        cudaFree(status);
        return result;
    }

    cudaFree(status);

    TOPO_PROBE_TRACE("tma_load status=%d", got);

    if (got != 1) {
        std::ostringstream oss;
        oss << "tma load mismatch status=" << got;
        return fail_result(oss.str());
    }

    return pass_result();
}

ProbeResult probe_tma_store_impl(
    int exec_device,
    void* dst_gmem,
    int dst_copyback_device,
    bool dst_is_host_mapped) {
    TOPO_PROBE_TRACE(
        "ENTER impl tma_store exec=%d dst=%p copyback=%d host_mapped=%d",
        exec_device,
        dst_gmem,
        dst_copyback_device,
        int(dst_is_host_mapped));

    ProbeResult result{};
    int* status = nullptr;

    if (!check(cudaSetDevice(exec_device), "cudaSetDevice(exec)", &result)) {
        return result;
    }
    if (!check(cudaMalloc(&status, sizeof(int)), "cudaMalloc(status)", &result)) {
        return result;
    }
    if (!check(cudaMemset(status, 0, sizeof(int)), "cudaMemset(status)", &result)) {
        cudaFree(status);
        return result;
    }

    TOPO_PROBE_TRACE("tma_store before kernel launch");
    tma_store_f16_kernel<<<1, 32, 64>>>(
        reinterpret_cast<half*>(dst_gmem),
        status);
    TOPO_PROBE_TRACE("tma_store after kernel launch");

    if (!check(cudaGetLastError(), "tma_store_f16_kernel launch", &result) ||
        !check(cudaDeviceSynchronize(), "tma_store_f16_kernel sync", &result)) {
        cudaFree(status);
        return result;
    }

    int got_status = 0;
    if (!check(cudaMemcpy(&got_status, status, sizeof(int), cudaMemcpyDeviceToHost),
               "cudaMemcpy(status)", &result)) {
        cudaFree(status);
        return result;
    }

    cudaFree(status);

    TOPO_PROBE_TRACE("tma_store status=%d", got_status);

    if (got_status != 1) {
        std::ostringstream oss;
        oss << "tma store status mismatch status=" << got_status;
        return fail_result(oss.str());
    }

    std::vector<uint16_t> got(kHalfCount);

    if (dst_is_host_mapped) {
        TOPO_PROBE_TRACE(
            "tma_store host-mapped copyback disabled in debug file; marking failed safely");
        return fail_result("tma store to mapped host memory is not safely copyback-readable in this debug probe");
    }

    if (!check(cudaSetDevice(dst_copyback_device),
               "cudaSetDevice(dst copyback)", &result) ||
        !check(cudaMemcpy(got.data(), dst_gmem, kHalfBytes, cudaMemcpyDeviceToHost),
               "cudaMemcpy(dst_gmem)", &result)) {
        return result;
    }

    if (!half_array_all_bits(got, 0x4400u)) {
        std::ostringstream oss;
        oss << "tma store mismatch first=0x"
            << std::hex << got[0];
        return fail_result(oss.str());
    }

    return pass_result();
}

ProbeResult probe_tma_reduce_impl(
    int exec_device,
    const void* src_gmem,
    half* local_dst_gmem) {
    TOPO_PROBE_TRACE(
        "ENTER impl tma_reduce exec=%d src=%p dst=%p",
        exec_device,
        src_gmem,
        static_cast<void*>(local_dst_gmem));

    ProbeResult result{};
    int* status = nullptr;

    std::vector<uint16_t> one;
    fill_half_bits(&one, 0x3c00u);

    if (!check(cudaSetDevice(exec_device), "cudaSetDevice(exec)", &result)) {
        return result;
    }
    if (!check(cudaMemcpy(local_dst_gmem, one.data(), kHalfBytes, cudaMemcpyHostToDevice),
               "cudaMemcpy(local_dst init)", &result)) {
        return result;
    }
    if (!check(cudaMalloc(&status, sizeof(int)), "cudaMalloc(status)", &result)) {
        return result;
    }
    if (!check(cudaMemset(status, 0, sizeof(int)), "cudaMemset(status)", &result)) {
        cudaFree(status);
        return result;
    }

    TOPO_PROBE_TRACE("tma_reduce before kernel launch");
    tma_reduce_f16_kernel<<<1, 32, 64>>>(
        reinterpret_cast<const half*>(src_gmem),
        local_dst_gmem,
        status);
    TOPO_PROBE_TRACE("tma_reduce after kernel launch");

    if (!check(cudaGetLastError(), "tma_reduce_f16_kernel launch", &result) ||
        !check(cudaDeviceSynchronize(), "tma_reduce_f16_kernel sync", &result)) {
        cudaFree(status);
        return result;
    }

    int got_status = 0;
    if (!check(cudaMemcpy(&got_status, status, sizeof(int), cudaMemcpyDeviceToHost),
               "cudaMemcpy(status)", &result)) {
        cudaFree(status);
        return result;
    }

    cudaFree(status);

    TOPO_PROBE_TRACE("tma_reduce status=%d", got_status);

    if (got_status != 1) {
        std::ostringstream oss;
        oss << "tma reduce status mismatch status=" << got_status;
        return fail_result(oss.str());
    }

    std::vector<uint16_t> got(kHalfCount);
    if (!check(cudaMemcpy(got.data(), local_dst_gmem, kHalfBytes, cudaMemcpyDeviceToHost),
               "cudaMemcpy(local_dst result)", &result)) {
        return result;
    }

    if (!half_array_all_bits(got, 0x4200u)) {
        std::ostringstream oss;
        oss << "tma reduce mismatch first=0x"
            << std::hex << got[0];
        return fail_result(oss.str());
    }

    return pass_result();
}

void* alloc_mapped_host_half_array(uint16_t fill_bits, ProbeResult* result) {
    TOPO_PROBE_TRACE("ENTER alloc_mapped_host_half_array bits=0x%x", unsigned(fill_bits));

    uint16_t* host = nullptr;

    if (!check(cudaHostAlloc(
                   reinterpret_cast<void**>(&host),
                   kHalfBytes,
                   cudaHostAllocMapped | cudaHostAllocPortable),
               "cudaHostAllocMapped",
               result)) {
        return nullptr;
    }

    TOPO_PROBE_TRACE("alloc_mapped_host_half_array host=%p", static_cast<void*>(host));

    for (int i = 0; i < kHalfCount; ++i) {
        host[i] = fill_bits;
    }

    return host;
}

bool mapped_host_device_pointer(
    int device,
    void* host_ptr,
    void** out_device_ptr,
    ProbeResult* result) {
    TOPO_PROBE_TRACE(
        "ENTER mapped_host_device_pointer device=%d host=%p",
        device,
        host_ptr);

    if (!check(cudaSetDevice(device), "cudaSetDevice(mapped host)", result)) {
        return false;
    }

    const bool ok =
        check(cudaHostGetDevicePointer(out_device_ptr, host_ptr, 0),
              "cudaHostGetDevicePointer",
              result);

    TOPO_PROBE_TRACE(
        "LEAVE mapped_host_device_pointer device=%d ok=%d dev_ptr=%p",
        device,
        int(ok),
        out_device_ptr != nullptr ? *out_device_ptr : nullptr);

    return ok;
}

void log_result(
    const char* name,
    int src_device,
    int dst_device,
    const ProbeResult& r) {
    TOPO_PROBE_TRACE(
        "LEAVE %s %d -> %d attempted=%d passed=%d err=%s",
        name,
        src_device,
        dst_device,
        int(r.attempted),
        int(r.passed),
        r.error);
}

} // namespace

ProbeResult probe_direct_load_store(
    int src_device,
    int dst_device) {
    TOPO_PROBE_TRACE(
        "ENTER probe_direct_load_store %d -> %d",
        src_device,
        dst_device);

    ProbeResult r =
        probe_direct_load_store_impl(
            src_device,
            dst_device);

    log_result("probe_direct_load_store", src_device, dst_device, r);
    return r;
}

ProbeResult probe_direct_atomic_add_i32(
    int src_device,
    int dst_device) {
    TOPO_PROBE_TRACE(
        "ENTER probe_direct_atomic_add_i32 %d -> %d",
        src_device,
        dst_device);

    ProbeResult r =
        probe_direct_atomic_add_i32_impl(
            src_device,
            dst_device);

    log_result("probe_direct_atomic_add_i32", src_device, dst_device, r);
    return r;
}

ProbeResult probe_direct_tma_load_f16(
    int src_device,
    int dst_device) {
    TOPO_PROBE_TRACE("ENTER probe_direct_tma_load_f16 %d -> %d", src_device, dst_device);

    ProbeResult result{};
    half* peer_src = nullptr;

    std::vector<uint16_t> two;
    fill_half_bits(&two, 0x4000u);

    if (!check(cudaSetDevice(dst_device), "cudaSetDevice(dst)", &result)) {
        log_result("probe_direct_tma_load_f16", src_device, dst_device, result);
        return result;
    }
    if (!check(cudaMalloc(&peer_src, kHalfBytes), "cudaMalloc(peer_src)", &result)) {
        log_result("probe_direct_tma_load_f16", src_device, dst_device, result);
        return result;
    }
    TOPO_PROBE_TRACE("direct_tma_load peer_src=%p", static_cast<void*>(peer_src));

    if (!check(cudaMemcpy(peer_src, two.data(), kHalfBytes, cudaMemcpyHostToDevice),
               "cudaMemcpy(peer_src)", &result)) {
        cudaFree(peer_src);
        log_result("probe_direct_tma_load_f16", src_device, dst_device, result);
        return result;
    }

    result = probe_tma_load_impl(src_device, peer_src);

    cudaSetDevice(dst_device);
    cudaFree(peer_src);

    log_result("probe_direct_tma_load_f16", src_device, dst_device, result);
    return result;
}

ProbeResult probe_direct_tma_store_f16(
    int src_device,
    int dst_device) {
    TOPO_PROBE_TRACE("ENTER probe_direct_tma_store_f16 %d -> %d", src_device, dst_device);

    ProbeResult result{};
    half* peer_dst = nullptr;

    std::vector<uint16_t> zero;
    fill_half_bits(&zero, 0x0000u);

    if (!check(cudaSetDevice(dst_device), "cudaSetDevice(dst)", &result)) {
        log_result("probe_direct_tma_store_f16", src_device, dst_device, result);
        return result;
    }
    if (!check(cudaMalloc(&peer_dst, kHalfBytes), "cudaMalloc(peer_dst)", &result)) {
        log_result("probe_direct_tma_store_f16", src_device, dst_device, result);
        return result;
    }
    TOPO_PROBE_TRACE("direct_tma_store peer_dst=%p", static_cast<void*>(peer_dst));

    if (!check(cudaMemcpy(peer_dst, zero.data(), kHalfBytes, cudaMemcpyHostToDevice),
               "cudaMemcpy(peer_dst)", &result)) {
        cudaFree(peer_dst);
        log_result("probe_direct_tma_store_f16", src_device, dst_device, result);
        return result;
    }

    result =
        probe_tma_store_impl(
            src_device,
            peer_dst,
            dst_device,
            false);

    cudaSetDevice(dst_device);
    cudaFree(peer_dst);

    log_result("probe_direct_tma_store_f16", src_device, dst_device, result);
    return result;
}

ProbeResult probe_direct_tma_reduce_f16(
    int src_device,
    int dst_device) {
    TOPO_PROBE_TRACE("ENTER probe_direct_tma_reduce_f16 %d -> %d", src_device, dst_device);

    ProbeResult result{};
    half* peer_src = nullptr;
    half* local_dst = nullptr;

    std::vector<uint16_t> two;
    fill_half_bits(&two, 0x4000u);

    if (!check(cudaSetDevice(dst_device), "cudaSetDevice(dst)", &result)) {
        log_result("probe_direct_tma_reduce_f16", src_device, dst_device, result);
        return result;
    }
    if (!check(cudaMalloc(&peer_src, kHalfBytes), "cudaMalloc(peer_src)", &result)) {
        log_result("probe_direct_tma_reduce_f16", src_device, dst_device, result);
        return result;
    }
    TOPO_PROBE_TRACE("direct_tma_reduce peer_src=%p", static_cast<void*>(peer_src));

    if (!check(cudaMemcpy(peer_src, two.data(), kHalfBytes, cudaMemcpyHostToDevice),
               "cudaMemcpy(peer_src)", &result)) {
        cudaFree(peer_src);
        log_result("probe_direct_tma_reduce_f16", src_device, dst_device, result);
        return result;
    }

    if (!check(cudaSetDevice(src_device), "cudaSetDevice(src)", &result)) {
        cudaSetDevice(dst_device);
        cudaFree(peer_src);
        log_result("probe_direct_tma_reduce_f16", src_device, dst_device, result);
        return result;
    }
    if (!check(cudaMalloc(&local_dst, kHalfBytes), "cudaMalloc(local_dst)", &result)) {
        cudaSetDevice(dst_device);
        cudaFree(peer_src);
        log_result("probe_direct_tma_reduce_f16", src_device, dst_device, result);
        return result;
    }
    TOPO_PROBE_TRACE("direct_tma_reduce local_dst=%p", static_cast<void*>(local_dst));

    result =
        probe_tma_reduce_impl(
            src_device,
            peer_src,
            local_dst);

    cudaSetDevice(src_device);
    cudaFree(local_dst);
    cudaSetDevice(dst_device);
    cudaFree(peer_src);

    log_result("probe_direct_tma_reduce_f16", src_device, dst_device, result);
    return result;
}

ProbeResult probe_shm_load_store(
    int src_device,
    int dst_device) {
    TOPO_PROBE_TRACE("ENTER probe_shm_load_store %d -> %d", src_device, dst_device);

    ProbeResult result{};
    int* host_int = nullptr;

    TOPO_PROBE_TRACE("shm_load_store before cudaHostAlloc");
    if (!check(cudaHostAlloc(
                   reinterpret_cast<void**>(&host_int),
                   3 * sizeof(int),
                   cudaHostAllocMapped | cudaHostAllocPortable),
               "cudaHostAllocMapped(int)",
               &result)) {
        log_result("probe_shm_load_store", src_device, dst_device, result);
        return result;
    }
    TOPO_PROBE_TRACE("shm_load_store host_int=%p", static_cast<void*>(host_int));

    host_int[0] = kMagic;
    host_int[1] = 0;
    host_int[2] = 0;

    void* src_dev_ptr = nullptr;
    void* dst_dev_ptr = nullptr;

    if (!mapped_host_device_pointer(src_device, host_int, &src_dev_ptr, &result) ||
        !mapped_host_device_pointer(dst_device, host_int, &dst_dev_ptr, &result)) {
        cudaFreeHost(host_int);
        log_result("probe_shm_load_store", src_device, dst_device, result);
        return result;
    }

    int* src_view = reinterpret_cast<int*>(src_dev_ptr);
    int* dst_view = reinterpret_cast<int*>(dst_dev_ptr);

    TOPO_PROBE_TRACE(
        "shm_load_store src_view=%p dst_view=%p",
        static_cast<void*>(src_view),
        static_cast<void*>(dst_view));

    if (!check(cudaSetDevice(src_device), "cudaSetDevice(src)", &result)) {
        cudaFreeHost(host_int);
        log_result("probe_shm_load_store", src_device, dst_device, result);
        return result;
    }

    TOPO_PROBE_TRACE("shm_load_store before src kernel");
    direct_load_store_kernel<<<1, 32>>>(
        src_view + 0,
        src_view + 1,
        src_view + 2);

    if (!check(cudaGetLastError(), "shm direct_load_store_kernel launch", &result) ||
        !check(cudaDeviceSynchronize(), "shm direct_load_store_kernel sync", &result)) {
        cudaFreeHost(host_int);
        log_result("probe_shm_load_store", src_device, dst_device, result);
        return result;
    }
    TOPO_PROBE_TRACE(
        "shm_load_store after src kernel h0=%d h1=%d h2=%d",
        host_int[0],
        host_int[1],
        host_int[2]);

    if (!check(cudaSetDevice(dst_device), "cudaSetDevice(dst)", &result)) {
        cudaFreeHost(host_int);
        log_result("probe_shm_load_store", src_device, dst_device, result);
        return result;
    }

    TOPO_PROBE_TRACE("shm_load_store before dst kernel");
    direct_load_store_kernel<<<1, 32>>>(
        dst_view + 1,
        dst_view + 0,
        dst_view + 2);

    if (!check(cudaGetLastError(), "shm dst load_store kernel launch", &result) ||
        !check(cudaDeviceSynchronize(), "shm dst load_store kernel sync", &result)) {
        cudaFreeHost(host_int);
        log_result("probe_shm_load_store", src_device, dst_device, result);
        return result;
    }

    TOPO_PROBE_TRACE(
        "shm_load_store after dst kernel h0=%d h1=%d h2=%d",
        host_int[0],
        host_int[1],
        host_int[2]);

    const bool ok =
        (host_int[0] == kMagic + 2) &&
        (host_int[1] == kMagic + 1);

    int h0 = host_int[0];
    int h1 = host_int[1];
    int h2 = host_int[2];

    TOPO_PROBE_TRACE("shm_load_store before cudaFreeHost host_int=%p", static_cast<void*>(host_int));
    cudaFreeHost(host_int);
    TOPO_PROBE_TRACE("shm_load_store after cudaFreeHost");

    if (!ok) {
        std::ostringstream oss;
        oss << "shm load/store mismatch h0=" << h0
            << " h1=" << h1
            << " h2=" << h2;
        result = fail_result(oss.str());
        log_result("probe_shm_load_store", src_device, dst_device, result);
        return result;
    }

    result = pass_result();
    log_result("probe_shm_load_store", src_device, dst_device, result);
    return result;
}

ProbeResult probe_shm_atomic_add_i32(
    int src_device,
    int dst_device) {
    TOPO_PROBE_TRACE("ENTER probe_shm_atomic_add_i32 %d -> %d", src_device, dst_device);
    (void)dst_device;

    ProbeResult result{};
    int* host_counter = nullptr;

    if (!check(cudaHostAlloc(
                   reinterpret_cast<void**>(&host_counter),
                   2 * sizeof(int),
                   cudaHostAllocMapped | cudaHostAllocPortable),
               "cudaHostAllocMapped(counter)",
               &result)) {
        log_result("probe_shm_atomic_add_i32", src_device, dst_device, result);
        return result;
    }

    TOPO_PROBE_TRACE("shm_atomic host_counter=%p", static_cast<void*>(host_counter));

    host_counter[0] = 7;
    host_counter[1] = 0;

    void* src_dev_ptr = nullptr;
    if (!mapped_host_device_pointer(src_device, host_counter, &src_dev_ptr, &result)) {
        cudaFreeHost(host_counter);
        log_result("probe_shm_atomic_add_i32", src_device, dst_device, result);
        return result;
    }

    int* src_view = reinterpret_cast<int*>(src_dev_ptr);

    if (!check(cudaSetDevice(src_device), "cudaSetDevice(src)", &result)) {
        cudaFreeHost(host_counter);
        log_result("probe_shm_atomic_add_i32", src_device, dst_device, result);
        return result;
    }

    TOPO_PROBE_TRACE("shm_atomic before kernel");
    direct_atomic_add_i32_kernel<<<1, 32>>>(
        src_view + 0,
        src_view + 1);

    if (!check(cudaGetLastError(), "shm atomic kernel launch", &result) ||
        !check(cudaDeviceSynchronize(), "shm atomic kernel sync", &result)) {
        cudaFreeHost(host_counter);
        log_result("probe_shm_atomic_add_i32", src_device, dst_device, result);
        return result;
    }

    const bool ok =
        host_counter[0] == 8 &&
        host_counter[1] == 7;

    int h0 = host_counter[0];
    int h1 = host_counter[1];

    TOPO_PROBE_TRACE("shm_atomic h0=%d h1=%d before free", h0, h1);
    cudaFreeHost(host_counter);

    if (!ok) {
        std::ostringstream oss;
        oss << "shm atomic32 mismatch counter=" << h0
            << " old=" << h1;
        result = fail_result(oss.str());
        log_result("probe_shm_atomic_add_i32", src_device, dst_device, result);
        return result;
    }

    result = pass_result();
    log_result("probe_shm_atomic_add_i32", src_device, dst_device, result);
    return result;
}

ProbeResult probe_shm_tma_load_f16(
    int src_device,
    int dst_device) {
    TOPO_PROBE_TRACE("ENTER probe_shm_tma_load_f16 %d -> %d", src_device, dst_device);
    (void)dst_device;

    ProbeResult result{};
    void* host = alloc_mapped_host_half_array(0x4000u, &result);
    if (host == nullptr) {
        log_result("probe_shm_tma_load_f16", src_device, dst_device, result);
        return result;
    }

    void* dev_ptr = nullptr;
    if (!mapped_host_device_pointer(src_device, host, &dev_ptr, &result)) {
        cudaFreeHost(host);
        log_result("probe_shm_tma_load_f16", src_device, dst_device, result);
        return result;
    }

    result = probe_tma_load_impl(src_device, dev_ptr);

    cudaFreeHost(host);
    log_result("probe_shm_tma_load_f16", src_device, dst_device, result);
    return result;
}

ProbeResult probe_shm_tma_store_f16(
    int src_device,
    int dst_device) {
    TOPO_PROBE_TRACE("ENTER probe_shm_tma_store_f16 %d -> %d", src_device, dst_device);
    (void)dst_device;

    ProbeResult result{};
    void* host = alloc_mapped_host_half_array(0x0000u, &result);
    if (host == nullptr) {
        log_result("probe_shm_tma_store_f16", src_device, dst_device, result);
        return result;
    }

    void* dev_ptr = nullptr;
    if (!mapped_host_device_pointer(src_device, host, &dev_ptr, &result)) {
        cudaFreeHost(host);
        log_result("probe_shm_tma_store_f16", src_device, dst_device, result);
        return result;
    }

    result =
        probe_tma_store_impl(
            src_device,
            dev_ptr,
            src_device,
            true);

    cudaFreeHost(host);
    log_result("probe_shm_tma_store_f16", src_device, dst_device, result);
    return result;
}

ProbeResult probe_shm_tma_reduce_f16(
    int src_device,
    int dst_device) {
    TOPO_PROBE_TRACE("ENTER probe_shm_tma_reduce_f16 %d -> %d", src_device, dst_device);
    (void)dst_device;

    ProbeResult result{};
    void* host_src = alloc_mapped_host_half_array(0x4000u, &result);
    if (host_src == nullptr) {
        log_result("probe_shm_tma_reduce_f16", src_device, dst_device, result);
        return result;
    }

    void* src_dev_ptr = nullptr;
    if (!mapped_host_device_pointer(src_device, host_src, &src_dev_ptr, &result)) {
        cudaFreeHost(host_src);
        log_result("probe_shm_tma_reduce_f16", src_device, dst_device, result);
        return result;
    }

    half* local_dst = nullptr;

    if (!check(cudaSetDevice(src_device), "cudaSetDevice(src)", &result)) {
        cudaFreeHost(host_src);
        log_result("probe_shm_tma_reduce_f16", src_device, dst_device, result);
        return result;
    }

    if (!check(cudaMalloc(&local_dst, kHalfBytes), "cudaMalloc(local_dst)", &result)) {
        cudaFreeHost(host_src);
        log_result("probe_shm_tma_reduce_f16", src_device, dst_device, result);
        return result;
    }

    result =
        probe_tma_reduce_impl(
            src_device,
            src_dev_ptr,
            local_dst);

    cudaFree(local_dst);
    cudaFreeHost(host_src);

    log_result("probe_shm_tma_reduce_f16", src_device, dst_device, result);
    return result;
}

} // namespace detail
} // namespace topology
} // namespace ooverlap
