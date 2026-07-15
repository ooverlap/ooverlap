#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <string>
#include <vector>

namespace {

constexpr std::size_t kDefaultElements = 1u << 20;       // 8 MiB per GPU.
constexpr std::uint64_t kDefaultAtomicOps = 1u << 18;    // 262,144 atomics per pair.
constexpr int kThreads = 256;
constexpr int kMaxBlocks = 256;

struct DeviceBuffers {
    std::uint64_t* data = nullptr;
    unsigned long long* mismatch_count = nullptr;
    unsigned long long* atomic_counter = nullptr;
};

__host__ __device__ __forceinline__ std::uint64_t pattern_value(
    int owner_device,
    std::size_t index) {
    std::uint64_t x =
        static_cast<std::uint64_t>(index) ^
        (0x9e3779b97f4a7c15ULL *
         static_cast<std::uint64_t>(owner_device + 1));
    x ^= x >> 30;
    x *= 0xbf58476d1ce4e5b9ULL;
    x ^= x >> 27;
    x *= 0x94d049bb133111ebULL;
    x ^= x >> 31;
    return x;
}

__global__ void initialize_buffer_kernel(
    std::uint64_t* data,
    std::size_t elements,
    int owner_device) {
    const std::size_t start =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t stride =
        static_cast<std::size_t>(gridDim.x) * blockDim.x;

    for (std::size_t i = start; i < elements; i += stride) {
        data[i] = pattern_value(owner_device, i);
    }
}

__global__ void validate_remote_read_kernel(
    const std::uint64_t* remote_data,
    std::size_t elements,
    int owner_device,
    unsigned long long* mismatch_count) {
    const std::size_t start =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t stride =
        static_cast<std::size_t>(gridDim.x) * blockDim.x;

    unsigned long long local_mismatches = 0;

    for (std::size_t i = start; i < elements; i += stride) {
        if (remote_data[i] != pattern_value(owner_device, i)) {
            ++local_mismatches;
        }
    }

    if (local_mismatches != 0) {
        atomicAdd(mismatch_count, local_mismatches);
    }
}

__global__ void remote_atomic_add_kernel(
    unsigned long long* remote_counter,
    unsigned long long operations) {
    const unsigned long long start =
        static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    const unsigned long long stride =
        static_cast<unsigned long long>(gridDim.x) * blockDim.x;

    for (unsigned long long i = start; i < operations; i += stride) {
        atomicAdd_system(remote_counter, 1ULL);
    }
}

bool check_cuda(cudaError_t error, const char* operation, int device = -1) {
    if (error == cudaSuccess) {
        return true;
    }

    if (device >= 0) {
        std::fprintf(
            stderr,
            "[CUDA error] GPU %d: %s: %s\n",
            device,
            operation,
            cudaGetErrorString(error));
    } else {
        std::fprintf(
            stderr,
            "[CUDA error] %s: %s\n",
            operation,
            cudaGetErrorString(error));
    }

    return false;
}

std::uint64_t parse_positive_u64(
    const char* text,
    std::uint64_t fallback,
    const char* name) {
    if (text == nullptr || *text == '\0') {
        return fallback;
    }

    char* end = nullptr;
    const unsigned long long value = std::strtoull(text, &end, 10);

    if (end == text || *end != '\0' || value == 0) {
        std::fprintf(
            stderr,
            "Invalid %s value '%s'; expected a positive integer.\n",
            name,
            text);
        std::exit(2);
    }

    return static_cast<std::uint64_t>(value);
}

int blocks_for_work(std::uint64_t work_items) {
    const std::uint64_t blocks =
        (work_items + static_cast<std::uint64_t>(kThreads) - 1) /
        static_cast<std::uint64_t>(kThreads);
    return static_cast<int>(
        std::max<std::uint64_t>(
            1,
            std::min<std::uint64_t>(blocks, kMaxBlocks)));
}

void free_buffers(std::vector<DeviceBuffers>* buffers) {
    if (buffers == nullptr) {
        return;
    }

    for (int device = 0;
         device < static_cast<int>(buffers->size());
         ++device) {
        cudaSetDevice(device);
        cudaFree((*buffers)[device].data);
        cudaFree((*buffers)[device].mismatch_count);
        cudaFree((*buffers)[device].atomic_counter);
    }
}

}  // namespace

int main(int argc, char** argv) {
    const std::uint64_t parsed_elements =
        parse_positive_u64(
            argc > 1 ? argv[1] : nullptr,
            kDefaultElements,
            "element count");
    const std::uint64_t atomic_operations =
        parse_positive_u64(
            argc > 2 ? argv[2] : nullptr,
            kDefaultAtomicOps,
            "atomic operation count");

    if (parsed_elements >
        static_cast<std::uint64_t>(
            std::numeric_limits<std::size_t>::max())) {
        std::fprintf(stderr, "Element count is too large for this platform.\n");
        return 2;
    }

    const std::size_t elements =
        static_cast<std::size_t>(parsed_elements);

    if (elements >
        std::numeric_limits<std::size_t>::max() /
            sizeof(std::uint64_t)) {
        std::fprintf(stderr, "Requested buffer size overflows size_t.\n");
        return 2;
    }

    const std::size_t bytes = elements * sizeof(std::uint64_t);

    int device_count = 0;
    if (!check_cuda(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount")) {
        return 1;
    }

    if (device_count < 2) {
        std::fprintf(
            stderr,
            "Need at least two visible CUDA GPUs; found %d.\n",
            device_count);
        return 2;
    }

    std::printf("Visible CUDA GPUs: %d\n", device_count);
    std::printf(
        "Read-test buffer per GPU: %zu bytes (%zu elements)\n",
        bytes,
        elements);
    std::printf(
        "Remote atomicAdd_system operations per directed pair: %llu\n\n",
        static_cast<unsigned long long>(atomic_operations));

    for (int device = 0; device < device_count; ++device) {
        cudaDeviceProp prop{};
        if (!check_cuda(
                cudaGetDeviceProperties(&prop, device),
                "cudaGetDeviceProperties",
                device)) {
            return 1;
        }

        std::printf(
            "GPU %d: %s, compute capability %d.%d, UVA=%s\n",
            device,
            prop.name,
            prop.major,
            prop.minor,
            prop.unifiedAddressing ? "yes" : "no");
    }

    std::printf("\nAllocating and initializing buffers...\n");

    std::vector<DeviceBuffers> buffers(
        static_cast<std::size_t>(device_count));

    for (int device = 0; device < device_count; ++device) {
        if (!check_cuda(cudaSetDevice(device), "cudaSetDevice", device) ||
            !check_cuda(
                cudaMalloc(
                    reinterpret_cast<void**>(&buffers[device].data),
                    bytes),
                "cudaMalloc(data)",
                device) ||
            !check_cuda(
                cudaMalloc(
                    reinterpret_cast<void**>(&buffers[device].mismatch_count),
                    sizeof(unsigned long long)),
                "cudaMalloc(mismatch_count)",
                device) ||
            !check_cuda(
                cudaMalloc(
                    reinterpret_cast<void**>(&buffers[device].atomic_counter),
                    sizeof(unsigned long long)),
                "cudaMalloc(atomic_counter)",
                device)) {
            free_buffers(&buffers);
            return 1;
        }

        initialize_buffer_kernel<<<blocks_for_work(elements), kThreads>>>(
            buffers[device].data,
            elements,
            device);

        if (!check_cuda(
                cudaGetLastError(),
                "initialize_buffer_kernel launch",
                device) ||
            !check_cuda(
                cudaDeviceSynchronize(),
                "initialize_buffer_kernel synchronize",
                device)) {
            free_buffers(&buffers);
            return 1;
        }
    }

    std::vector<std::vector<bool>> peer_enabled(
        static_cast<std::size_t>(device_count),
        std::vector<bool>(static_cast<std::size_t>(device_count), false));

    bool all_passed = true;

    std::printf("\n=== Peer access enablement ===\n");

    for (int accessor = 0; accessor < device_count; ++accessor) {
        if (!check_cuda(
                cudaSetDevice(accessor),
                "cudaSetDevice",
                accessor)) {
            all_passed = false;
            continue;
        }

        peer_enabled[accessor][accessor] = true;

        for (int owner = 0; owner < device_count; ++owner) {
            if (owner == accessor) {
                continue;
            }

            cudaError_t error = cudaDeviceEnablePeerAccess(owner, 0);

            if (error == cudaSuccess ||
                error == cudaErrorPeerAccessAlreadyEnabled) {
                if (error == cudaErrorPeerAccessAlreadyEnabled) {
                    cudaGetLastError();
                }
                peer_enabled[accessor][owner] = true;
                std::printf(
                    "GPU %d accessing GPU %d: ENABLED\n",
                    accessor,
                    owner);
            } else {
                std::printf(
                    "GPU %d accessing GPU %d: FAILED (%s)\n",
                    accessor,
                    owner,
                    cudaGetErrorString(error));
                cudaGetLastError();
                all_passed = false;
            }
        }
    }

    std::printf("\n=== Functional peer-read validation ===\n");

    for (int reader = 0; reader < device_count; ++reader) {
        for (int owner = 0; owner < device_count; ++owner) {
            if (reader == owner) {
                continue;
            }

            if (!peer_enabled[reader][owner]) {
                std::printf(
                    "reader GPU %d <- owner GPU %d: FAIL (peer access not enabled)\n",
                    reader,
                    owner);
                all_passed = false;
                continue;
            }

            bool pair_ok =
                check_cuda(cudaSetDevice(reader), "cudaSetDevice", reader) &&
                check_cuda(
                    cudaMemset(
                        buffers[reader].mismatch_count,
                        0,
                        sizeof(unsigned long long)),
                    "cudaMemset(mismatch_count)",
                    reader);

            if (pair_ok) {
                validate_remote_read_kernel<<<
                    blocks_for_work(elements),
                    kThreads>>>(
                        buffers[owner].data,
                        elements,
                        owner,
                        buffers[reader].mismatch_count);

                pair_ok =
                    check_cuda(
                        cudaGetLastError(),
                        "validate_remote_read_kernel launch",
                        reader) &&
                    check_cuda(
                        cudaDeviceSynchronize(),
                        "validate_remote_read_kernel synchronize",
                        reader);
            }

            unsigned long long mismatches = 0;

            if (pair_ok) {
                pair_ok = check_cuda(
                    cudaMemcpy(
                        &mismatches,
                        buffers[reader].mismatch_count,
                        sizeof(mismatches),
                        cudaMemcpyDeviceToHost),
                    "cudaMemcpy(mismatch_count)",
                    reader);
            }

            if (pair_ok && mismatches == 0) {
                std::printf(
                    "reader GPU %d <- owner GPU %d: PASS\n",
                    reader,
                    owner);
            } else {
                std::printf(
                    "reader GPU %d <- owner GPU %d: FAIL (mismatches=%llu)\n",
                    reader,
                    owner,
                    mismatches);
                all_passed = false;
            }
        }
    }

    std::printf("\n=== Functional remote atomic validation ===\n");

    for (int writer = 0; writer < device_count; ++writer) {
        for (int target = 0; target < device_count; ++target) {
            if (writer == target) {
                continue;
            }

            if (!peer_enabled[writer][target]) {
                std::printf(
                    "writer GPU %d -> target GPU %d: FAIL (peer access not enabled)\n",
                    writer,
                    target);
                all_passed = false;
                continue;
            }

            bool pair_ok =
                check_cuda(cudaSetDevice(target), "cudaSetDevice", target) &&
                check_cuda(
                    cudaMemset(
                        buffers[target].atomic_counter,
                        0,
                        sizeof(unsigned long long)),
                    "cudaMemset(atomic_counter)",
                    target) &&
                check_cuda(
                    cudaDeviceSynchronize(),
                    "atomic counter reset synchronize",
                    target) &&
                check_cuda(cudaSetDevice(writer), "cudaSetDevice", writer);

            if (pair_ok) {
                remote_atomic_add_kernel<<<
                    blocks_for_work(atomic_operations),
                    kThreads>>>(
                        buffers[target].atomic_counter,
                        atomic_operations);

                pair_ok =
                    check_cuda(
                        cudaGetLastError(),
                        "remote_atomic_add_kernel launch",
                        writer) &&
                    check_cuda(
                        cudaDeviceSynchronize(),
                        "remote_atomic_add_kernel synchronize",
                        writer);
            }

            unsigned long long observed = 0;

            if (pair_ok) {
                pair_ok =
                    check_cuda(cudaSetDevice(target), "cudaSetDevice", target) &&
                    check_cuda(
                        cudaMemcpy(
                            &observed,
                            buffers[target].atomic_counter,
                            sizeof(observed),
                            cudaMemcpyDeviceToHost),
                        "cudaMemcpy(atomic_counter)",
                        target);
            }

            if (pair_ok && observed == atomic_operations) {
                std::printf(
                    "writer GPU %d -> target GPU %d: PASS (%llu)\n",
                    writer,
                    target,
                    observed);
            } else {
                std::printf(
                    "writer GPU %d -> target GPU %d: FAIL "
                    "(expected=%llu observed=%llu)\n",
                    writer,
                    target,
                    static_cast<unsigned long long>(atomic_operations),
                    observed);
                all_passed = false;
            }
        }
    }

    free_buffers(&buffers);

    std::printf("\n=== Result ===\n");
    if (all_passed) {
        std::printf(
            "PASS: every directed GPU pair completed a validated remote read "
            "and remote atomicAdd_system.\n");
        return 0;
    }

    std::printf(
        "FAIL: one or more directed GPU pairs failed peer enablement, "
        "remote read validation, or remote atomic validation.\n");
    return 1;
}
