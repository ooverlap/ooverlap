#include "topology/topology_probe.h"

#include "ooverlap/sync/sync.cuh"
#include "ooverlap/tma/tma.cuh"
#include "ooverlap/tma/tma_reduce.cuh"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <sstream>
#include <string>
#include <vector>

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
    return r;
}

ProbeResult fail_result(const std::string& error) {
    ProbeResult r{};
    r.attempted = true;
    r.passed = false;
    r.error = error;
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
            bits[i] = 0x4400u; // 4.0h
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
    ProbeResult result{};

    int* peer_src = nullptr;
    int* peer_dst = nullptr;
    int* local_result = nullptr;

    if (!check(cudaSetDevice(dst_device), "cudaSetDevice(dst)", &result)) {
        return result;
    }
    if (!check(cudaMalloc(&peer_src, sizeof(int)), "cudaMalloc(peer_src)", &result)) {
        return result;
    }
    if (!check(cudaMalloc(&peer_dst, sizeof(int)), "cudaMalloc(peer_dst)", &result)) {
        cudaFree(peer_src);
        return result;
    }
    if (!check(cudaMemset(peer_dst, 0, sizeof(int)), "cudaMemset(peer_dst)", &result)) {
        cudaFree(peer_src);
        cudaFree(peer_dst);
        return result;
    }
    if (!check(cudaMemcpy(peer_src, &kMagic, sizeof(int), cudaMemcpyHostToDevice),
               "cudaMemcpy(peer_src)", &result)) {
        cudaFree(peer_src);
        cudaFree(peer_dst);
        return result;
    }

    if (!check(cudaSetDevice(src_device), "cudaSetDevice(src)", &result)) {
        cudaSetDevice(dst_device);
        cudaFree(peer_src);
        cudaFree(peer_dst);
        return result;
    }
    if (!check(cudaMalloc(&local_result, sizeof(int)), "cudaMalloc(local_result)", &result)) {
        cudaSetDevice(dst_device);
        cudaFree(peer_src);
        cudaFree(peer_dst);
        return result;
    }
    if (!check(cudaMemset(local_result, 0, sizeof(int)), "cudaMemset(local_result)", &result)) {
        cudaFree(local_result);
        cudaSetDevice(dst_device);
        cudaFree(peer_src);
        cudaFree(peer_dst);
        return result;
    }

    direct_load_store_kernel<<<1, 32>>>(peer_src, peer_dst, local_result);

    if (!check(cudaGetLastError(), "direct_load_store_kernel launch", &result) ||
        !check(cudaDeviceSynchronize(), "direct_load_store_kernel sync", &result)) {
        cudaFree(local_result);
        cudaSetDevice(dst_device);
        cudaFree(peer_src);
        cudaFree(peer_dst);
        return result;
    }

    int got_local = 0;
    if (!check(cudaMemcpy(&got_local, local_result, sizeof(int), cudaMemcpyDeviceToHost),
               "cudaMemcpy(local_result)", &result)) {
        cudaFree(local_result);
        cudaSetDevice(dst_device);
        cudaFree(peer_src);
        cudaFree(peer_dst);
        return result;
    }

    int got_peer = 0;
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

    cudaSetDevice(src_device);
    cudaFree(local_result);
    cudaSetDevice(dst_device);
    cudaFree(peer_src);
    cudaFree(peer_dst);

    if (got_local != 1 || got_peer != kMagic + 1) {
        std::ostringstream oss;
        oss << "direct load/store mismatch local=" << got_local
            << " peer=" << got_peer;
        return fail_result(oss.str());
    }

    return pass_result();
}

ProbeResult probe_direct_atomic_add_i32_impl(
    int src_device,
    int dst_device) {
    ProbeResult result{};
    int* peer_counter = nullptr;
    int* local_old = nullptr;

    const int initial = 7;

    if (!check(cudaSetDevice(dst_device), "cudaSetDevice(dst)", &result)) {
        return result;
    }
    if (!check(cudaMalloc(&peer_counter, sizeof(int)), "cudaMalloc(peer_counter)", &result)) {
        return result;
    }
    if (!check(cudaMemcpy(peer_counter, &initial, sizeof(int), cudaMemcpyHostToDevice),
               "cudaMemcpy(peer_counter)", &result)) {
        cudaFree(peer_counter);
        return result;
    }

    if (!check(cudaSetDevice(src_device), "cudaSetDevice(src)", &result)) {
        cudaSetDevice(dst_device);
        cudaFree(peer_counter);
        return result;
    }
    if (!check(cudaMalloc(&local_old, sizeof(int)), "cudaMalloc(local_old)", &result)) {
        cudaSetDevice(dst_device);
        cudaFree(peer_counter);
        return result;
    }
    if (!check(cudaMemset(local_old, 0, sizeof(int)), "cudaMemset(local_old)", &result)) {
        cudaFree(local_old);
        cudaSetDevice(dst_device);
        cudaFree(peer_counter);
        return result;
    }

    direct_atomic_add_i32_kernel<<<1, 32>>>(peer_counter, local_old);

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

    cudaSetDevice(src_device);
    cudaFree(local_old);
    cudaSetDevice(dst_device);
    cudaFree(peer_counter);

    if (got_old != initial || got_counter != initial + 1) {
        std::ostringstream oss;
        oss << "direct atomic32 mismatch old=" << got_old
            << " counter=" << got_counter;
        return fail_result(oss.str());
    }

    return pass_result();
}

ProbeResult probe_tma_load_impl(
    int exec_device,
    const void* src_gmem) {
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

    tma_load_f16_kernel<<<1, 32, 64>>>(
        reinterpret_cast<const half*>(src_gmem),
        status);

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

    tma_store_f16_kernel<<<1, 32, 64>>>(
        reinterpret_cast<half*>(dst_gmem),
        status);

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

    if (got_status != 1) {
        std::ostringstream oss;
        oss << "tma store status mismatch status=" << got_status;
        return fail_result(oss.str());
    }

    std::vector<uint16_t> got(kHalfCount);

    if (dst_is_host_mapped) {
        const uint16_t* host_bits =
            reinterpret_cast<const uint16_t*>(dst_gmem);

        for (int i = 0; i < kHalfCount; ++i) {
            got[static_cast<size_t>(i)] = host_bits[i];
        }
    } else {
        if (!check(cudaSetDevice(dst_copyback_device),
                   "cudaSetDevice(dst copyback)", &result) ||
            !check(cudaMemcpy(got.data(), dst_gmem, kHalfBytes, cudaMemcpyDeviceToHost),
                   "cudaMemcpy(dst_gmem)", &result)) {
            return result;
        }
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

    tma_reduce_f16_kernel<<<1, 32, 64>>>(
        reinterpret_cast<const half*>(src_gmem),
        local_dst_gmem,
        status);

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
    uint16_t* host = nullptr;

    if (!check(cudaHostAlloc(
                   reinterpret_cast<void**>(&host),
                   kHalfBytes,
                   cudaHostAllocMapped | cudaHostAllocPortable),
               "cudaHostAllocMapped",
               result)) {
        return nullptr;
    }

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
    if (!check(cudaSetDevice(device), "cudaSetDevice(mapped host)", result)) {
        return false;
    }

    return check(cudaHostGetDevicePointer(out_device_ptr, host_ptr, 0),
                 "cudaHostGetDevicePointer",
                 result);
}

} // namespace

ProbeResult probe_direct_load_store(
    int src_device,
    int dst_device) {
    return probe_direct_load_store_impl(src_device, dst_device);
}

ProbeResult probe_direct_atomic_add_i32(
    int src_device,
    int dst_device) {
    return probe_direct_atomic_add_i32_impl(src_device, dst_device);
}

ProbeResult probe_direct_tma_load_f16(
    int src_device,
    int dst_device) {
    ProbeResult result{};
    half* peer_src = nullptr;

    std::vector<uint16_t> two;
    fill_half_bits(&two, 0x4000u);

    if (!check(cudaSetDevice(dst_device), "cudaSetDevice(dst)", &result)) {
        return result;
    }
    if (!check(cudaMalloc(&peer_src, kHalfBytes), "cudaMalloc(peer_src)", &result)) {
        return result;
    }
    if (!check(cudaMemcpy(peer_src, two.data(), kHalfBytes, cudaMemcpyHostToDevice),
               "cudaMemcpy(peer_src)", &result)) {
        cudaFree(peer_src);
        return result;
    }

    result = probe_tma_load_impl(src_device, peer_src);

    cudaSetDevice(dst_device);
    cudaFree(peer_src);

    return result;
}

ProbeResult probe_direct_tma_store_f16(
    int src_device,
    int dst_device) {
    ProbeResult result{};
    half* peer_dst = nullptr;

    std::vector<uint16_t> zero;
    fill_half_bits(&zero, 0x0000u);

    if (!check(cudaSetDevice(dst_device), "cudaSetDevice(dst)", &result)) {
        return result;
    }
    if (!check(cudaMalloc(&peer_dst, kHalfBytes), "cudaMalloc(peer_dst)", &result)) {
        return result;
    }
    if (!check(cudaMemcpy(peer_dst, zero.data(), kHalfBytes, cudaMemcpyHostToDevice),
               "cudaMemcpy(peer_dst)", &result)) {
        cudaFree(peer_dst);
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

    return result;
}

ProbeResult probe_direct_tma_reduce_f16(
    int src_device,
    int dst_device) {
    ProbeResult result{};
    half* peer_src = nullptr;
    half* local_dst = nullptr;

    std::vector<uint16_t> two;
    fill_half_bits(&two, 0x4000u);

    if (!check(cudaSetDevice(dst_device), "cudaSetDevice(dst)", &result)) {
        return result;
    }
    if (!check(cudaMalloc(&peer_src, kHalfBytes), "cudaMalloc(peer_src)", &result)) {
        return result;
    }
    if (!check(cudaMemcpy(peer_src, two.data(), kHalfBytes, cudaMemcpyHostToDevice),
               "cudaMemcpy(peer_src)", &result)) {
        cudaFree(peer_src);
        return result;
    }

    if (!check(cudaSetDevice(src_device), "cudaSetDevice(src)", &result)) {
        cudaSetDevice(dst_device);
        cudaFree(peer_src);
        return result;
    }
    if (!check(cudaMalloc(&local_dst, kHalfBytes), "cudaMalloc(local_dst)", &result)) {
        cudaSetDevice(dst_device);
        cudaFree(peer_src);
        return result;
    }

    result =
        probe_tma_reduce_impl(
            src_device,
            peer_src,
            local_dst);

    cudaSetDevice(src_device);
    cudaFree(local_dst);
    cudaSetDevice(dst_device);
    cudaFree(peer_src);

    return result;
}

ProbeResult probe_shm_load_store(
    int src_device,
    int dst_device) {
    ProbeResult result{};
    void* host = alloc_mapped_host_half_array(0x0000u, &result);
    if (host == nullptr) {
        return result;
    }

    int* host_int = nullptr;
    cudaFreeHost(host);

    if (!check(cudaHostAlloc(
                   reinterpret_cast<void**>(&host_int),
                   3 * sizeof(int),
                   cudaHostAllocMapped | cudaHostAllocPortable),
               "cudaHostAllocMapped(int)",
               &result)) {
        return result;
    }

    host_int[0] = kMagic;
    host_int[1] = 0;
    host_int[2] = 0;

    void* src_dev_ptr = nullptr;
    void* dst_dev_ptr = nullptr;

    if (!mapped_host_device_pointer(src_device, host_int, &src_dev_ptr, &result) ||
        !mapped_host_device_pointer(dst_device, host_int, &dst_dev_ptr, &result)) {
        cudaFreeHost(host_int);
        return result;
    }

    int* src_view = reinterpret_cast<int*>(src_dev_ptr);
    int* dst_view = reinterpret_cast<int*>(dst_dev_ptr);

    if (!check(cudaSetDevice(src_device), "cudaSetDevice(src)", &result)) {
        cudaFreeHost(host_int);
        return result;
    }

    direct_load_store_kernel<<<1, 32>>>(
        src_view + 0,
        src_view + 1,
        src_view + 2);

    if (!check(cudaGetLastError(), "shm direct_load_store_kernel launch", &result) ||
        !check(cudaDeviceSynchronize(), "shm direct_load_store_kernel sync", &result)) {
        cudaFreeHost(host_int);
        return result;
    }

    if (!check(cudaSetDevice(dst_device), "cudaSetDevice(dst)", &result)) {
        cudaFreeHost(host_int);
        return result;
    }

    direct_load_store_kernel<<<1, 32>>>(
        dst_view + 1,
        dst_view + 0,
        dst_view + 2);

    if (!check(cudaGetLastError(), "shm dst load_store kernel launch", &result) ||
        !check(cudaDeviceSynchronize(), "shm dst load_store kernel sync", &result)) {
        cudaFreeHost(host_int);
        return result;
    }

    const bool ok =
        (host_int[0] == kMagic + 2) &&
        (host_int[1] == kMagic + 1) &&
        (host_int[2] == 1);

    int h0 = host_int[0];
    int h1 = host_int[1];
    int h2 = host_int[2];

    cudaFreeHost(host_int);

    if (!ok) {
        std::ostringstream oss;
        oss << "shm load/store mismatch h0=" << h0
            << " h1=" << h1
            << " h2=" << h2;
        return fail_result(oss.str());
    }

    return pass_result();
}

ProbeResult probe_shm_atomic_add_i32(
    int src_device,
    int dst_device) {
    (void)dst_device;

    ProbeResult result{};
    int* host_counter = nullptr;

    if (!check(cudaHostAlloc(
                   reinterpret_cast<void**>(&host_counter),
                   2 * sizeof(int),
                   cudaHostAllocMapped | cudaHostAllocPortable),
               "cudaHostAllocMapped(counter)",
               &result)) {
        return result;
    }

    host_counter[0] = 7;
    host_counter[1] = 0;

    void* src_dev_ptr = nullptr;
    if (!mapped_host_device_pointer(src_device, host_counter, &src_dev_ptr, &result)) {
        cudaFreeHost(host_counter);
        return result;
    }

    int* src_view = reinterpret_cast<int*>(src_dev_ptr);

    if (!check(cudaSetDevice(src_device), "cudaSetDevice(src)", &result)) {
        cudaFreeHost(host_counter);
        return result;
    }

    direct_atomic_add_i32_kernel<<<1, 32>>>(
        src_view + 0,
        src_view + 1);

    if (!check(cudaGetLastError(), "shm atomic kernel launch", &result) ||
        !check(cudaDeviceSynchronize(), "shm atomic kernel sync", &result)) {
        cudaFreeHost(host_counter);
        return result;
    }

    const bool ok =
        host_counter[0] == 8 &&
        host_counter[1] == 7;

    int h0 = host_counter[0];
    int h1 = host_counter[1];

    cudaFreeHost(host_counter);

    if (!ok) {
        std::ostringstream oss;
        oss << "shm atomic32 mismatch counter=" << h0
            << " old=" << h1;
        return fail_result(oss.str());
    }

    return pass_result();
}

ProbeResult probe_shm_tma_load_f16(
    int src_device,
    int dst_device) {
    (void)dst_device;

    ProbeResult result{};
    void* host = alloc_mapped_host_half_array(0x4000u, &result);
    if (host == nullptr) {
        return result;
    }

    void* dev_ptr = nullptr;
    if (!mapped_host_device_pointer(src_device, host, &dev_ptr, &result)) {
        cudaFreeHost(host);
        return result;
    }

    result = probe_tma_load_impl(src_device, dev_ptr);

    cudaFreeHost(host);
    return result;
}

ProbeResult probe_shm_tma_store_f16(
    int src_device,
    int dst_device) {
    (void)dst_device;

    ProbeResult result{};
    void* host = alloc_mapped_host_half_array(0x0000u, &result);
    if (host == nullptr) {
        return result;
    }

    void* dev_ptr = nullptr;
    if (!mapped_host_device_pointer(src_device, host, &dev_ptr, &result)) {
        cudaFreeHost(host);
        return result;
    }

    result =
        probe_tma_store_impl(
            src_device,
            dev_ptr,
            src_device,
            true);

    cudaFreeHost(host);
    return result;
}

ProbeResult probe_shm_tma_reduce_f16(
    int src_device,
    int dst_device) {
    (void)dst_device;

    ProbeResult result{};
    void* host_src = alloc_mapped_host_half_array(0x4000u, &result);
    if (host_src == nullptr) {
        return result;
    }

    void* src_dev_ptr = nullptr;
    if (!mapped_host_device_pointer(src_device, host_src, &src_dev_ptr, &result)) {
        cudaFreeHost(host_src);
        return result;
    }

    half* local_dst = nullptr;

    if (!check(cudaSetDevice(src_device), "cudaSetDevice(src)", &result)) {
        cudaFreeHost(host_src);
        return result;
    }

    if (!check(cudaMalloc(&local_dst, kHalfBytes), "cudaMalloc(local_dst)", &result)) {
        cudaFreeHost(host_src);
        return result;
    }

    result =
        probe_tma_reduce_impl(
            src_device,
            src_dev_ptr,
            local_dst);

    cudaFree(local_dst);
    cudaFreeHost(host_src);

    return result;
}

} // namespace detail
} // namespace topology
} // namespace ooverlap
