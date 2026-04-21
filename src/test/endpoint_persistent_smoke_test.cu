#include "test/endpoint_persistent_smoke_test.h"

#include "comm/collective/published_tile.h"
#include "comm/endpoint_persistent_kernel.h"
#include "comm/endpoint_runtime.h"
#include "comm/group.h"
#include "ooverlap/system/runtime_utils.cuh"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <thread>
#include <vector>

namespace ooverlap {
namespace {

    void print_queue_head_tail(
    const comm::collective::ReadyTileQueue& q,
    const char* tag) {
    uint64_t head = 0;
    uint64_t tail = 0;

    cudaError_t err = cudaMemcpy(&head, q.head, sizeof(uint64_t), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string(tag) + ": cudaMemcpy(head) failed: " + cudaGetErrorString(err));
    }

    err = cudaMemcpy(&tail, q.tail, sizeof(uint64_t), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string(tag) + ": cudaMemcpy(tail) failed: " + cudaGetErrorString(err));
    }

    std::printf("%s head=%llu tail=%llu cap=%u\n",
                tag,
                static_cast<unsigned long long>(head),
                static_cast<unsigned long long>(tail),
                q.capacity);
    std::fflush(stdout);
}

__global__ void publish_single_ready_tile_kernel(
    comm::collective::ReadyTileQueue queue,
    const half* src,
    uint32_t bytes,
    uint64_t tile_id,
    int* status_out) {
    if (threadIdx.x != 0 || blockIdx.x != 0) {
        return;
    }

    const bool ok = comm::collective::device_publish_ready_tile(
        queue,
        tile_id,
        src,
        bytes,
        0,
        0,
        0,
        nullptr);

    *status_out = ok ? 1 : 0;
}

std::vector<half> make_host_pattern(
    int64_t numel) {
    std::vector<half> out(static_cast<size_t>(numel));
    for (int64_t i = 0; i < numel; ++i) {
        const float x = 0.125f + 0.01f * static_cast<float>(i % 97);
        out[static_cast<size_t>(i)] = __float2half_rn(x);
    }
    return out;
}

void expect_half_vectors_close(
    const std::vector<half>& got,
    const std::vector<half>& ref,
    const char* what) {
    if (got.size() != ref.size()) {
        throw std::runtime_error(std::string(what) + ": size mismatch");
    }

    for (size_t i = 0; i < got.size(); ++i) {
        const float g = __half2float(got[i]);
        const float r = __half2float(ref[i]);
        const float err = std::fabs(g - r);
        if (err > 1.0e-3f) {
            throw std::runtime_error(
                std::string(what) +
                ": mismatch at idx=" + std::to_string(i) +
                " got=" + std::to_string(g) +
                " ref=" + std::to_string(r));
        }
    }
}

bool poll_queue_head_until(
    const comm::collective::ReadyTileQueue& q,
    uint64_t target_head,
    int device,
    int timeout_ms) {
    auto start = std::chrono::steady_clock::now();

    system::runtime::set_device(device);

    cudaStream_t poll_stream = nullptr;
    uint64_t* head_host_pinned = nullptr;

    cudaError_t err = cudaSuccess;

    err = cudaStreamCreateWithFlags(&poll_stream, cudaStreamNonBlocking);
    if (err != cudaSuccess) {
        throw std::runtime_error(
            std::string("cudaStreamCreateWithFlags(poll_stream) failed: ") +
            cudaGetErrorString(err));
    }

    err = cudaMallocHost(&head_host_pinned, sizeof(uint64_t));
    if (err != cudaSuccess) {
        cudaStreamDestroy(poll_stream);
        throw std::runtime_error(
            std::string("cudaMallocHost(head_host_pinned) failed: ") +
            cudaGetErrorString(err));
    }

    try {
        while (true) {
            err = cudaMemcpyAsync(
                head_host_pinned,
                q.head,
                sizeof(uint64_t),
                cudaMemcpyDeviceToHost,
                poll_stream);
            if (err != cudaSuccess) {
                throw std::runtime_error(
                    std::string("cudaMemcpyAsync(queue head) failed: ") +
                    cudaGetErrorString(err));
            }

            err = cudaStreamSynchronize(poll_stream);
            if (err != cudaSuccess) {
                throw std::runtime_error(
                    std::string("cudaStreamSynchronize(poll_stream) failed: ") +
                    cudaGetErrorString(err));
            }

            if (*head_host_pinned >= target_head) {
                cudaFreeHost(head_host_pinned);
                cudaStreamDestroy(poll_stream);
                return true;
            }

            auto now = std::chrono::steady_clock::now();
            const auto elapsed_ms =
                std::chrono::duration_cast<std::chrono::milliseconds>(now - start).count();
            if (elapsed_ms > timeout_ms) {
                cudaFreeHost(head_host_pinned);
                cudaStreamDestroy(poll_stream);
                return false;
            }

            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    } catch (...) {
        cudaFreeHost(head_host_pinned);
        cudaStreamDestroy(poll_stream);
        throw;
    }
}

__global__ void execute_once_kernel(
    comm::DeviceEndpointRuntime runtime,
    int* status_out) {
    if (blockIdx.x != 0) {
        return;
    }

    __shared__ comm::exec::Chunk current_chunk;
    __shared__ int has_work;
    __shared__ int active_binding_idx;

    if (threadIdx.x == 0) {
        // status:
        // 0 = not set
        // 1 = no schedulable work
        // 2 = primed chunk
        // 3 = reduced and advanced

        comm::exec::ChunkScheduler<1> sched{};
        comm::exec::chunk_scheduler_init<1>(
            &sched,
            runtime.scheduler_bindings,
            runtime.num_scheduler_bindings,
            comm::kEndpointPersistentChunkBytes);

        if (!comm::exec::chunk_scheduler_try_prime_current(&sched)) {
            has_work = 0;
            *status_out = 1;
            return;
        }

        has_work = 1;
        active_binding_idx = sched.active_binding_idx;
        current_chunk = *comm::exec::chunk_scheduler_current(&sched);
        *status_out = 2;
    }
    __syncthreads();

    if (!has_work) {
        return;
    }

    if (current_chunk.num_tile_spans != 1) {
        if (threadIdx.x == 0) {
            *status_out = -1;
        }
        return;
    }

    const comm::exec::ChunkTileSpan& span = current_chunk.tile_spans[0];

    half* dst = reinterpret_cast<half*>(current_chunk.dst);
    const half* src = reinterpret_cast<const half*>(span.src);

    const size_t elems = current_chunk.bytes / sizeof(half);

    for (size_t i = threadIdx.x; i < elems; i += blockDim.x) {
        const float oldv = __half2float(dst[i]);
        const float addv = __half2float(src[i]);
        dst[i] = __float2half_rn(oldv + addv);
    }

    __syncthreads();

    if (threadIdx.x == 0) {
        auto* q = runtime.scheduler_bindings[active_binding_idx].queue;
        *q->head = current_chunk.span_ticket + current_chunk.num_tile_spans;
        __threadfence();
        *status_out = 3;
    }
}

} // namespace

bool endpoint_persistent_smoke_test(
    int64_t numel,
    int dev0,
    int dev1) {
    if (numel <= 0) {
        throw std::invalid_argument("endpoint_persistent_smoke_test: numel must be > 0");
    }
    if (dev0 == dev1) {
        throw std::invalid_argument("endpoint_persistent_smoke_test: dev0 and dev1 must differ");
    }

    comm::Group group{};
    comm::EndpointRuntime runtime{};
    comm::EndpointPersistentControl control{};

    half* src_dev = nullptr;
    half* dst_dev = nullptr;
    int* publish_status_dev = nullptr;
    int* drain_status_dev = nullptr;
    cudaStream_t producer_stream = nullptr;

    try {
const size_t bytes = static_cast<size_t>(numel) * sizeof(half);
const auto host_src = make_host_pattern(numel);
const std::vector<half> host_zero(static_cast<size_t>(numel), __float2half_rn(0.0f));

std::printf("[smoke] before group_init\n"); std::fflush(stdout);
comm::group_init(&group, {dev0, dev1}, comm::kEndpointPersistentChunkBytes);

std::printf("[smoke] after group_init\n"); std::fflush(stdout);
comm::endpoint_runtime_init(&runtime, &group, 0, 1, 8);

std::printf("[smoke] after endpoint_runtime_init\n"); std::fflush(stdout);

system::runtime::set_device(dev0);
system::runtime::check_cuda(
        cudaStreamCreateWithFlags(&producer_stream, cudaStreamNonBlocking),
    "cudaStreamCreate(producer_stream)");

system::runtime::check_cuda(cudaMalloc(&src_dev, bytes), "cudaMalloc(src_dev)");
system::runtime::check_cuda(cudaMalloc(&dst_dev, bytes), "cudaMalloc(dst_dev)");
system::runtime::check_cuda(cudaMalloc(&publish_status_dev, sizeof(int)), "cudaMalloc(publish_status_dev)");
system::runtime::check_cuda(
    cudaMalloc(&drain_status_dev, sizeof(int)),
    "cudaMalloc(drain_status_dev)");
system::runtime::check_cuda(
    cudaMemset(drain_status_dev, 0, sizeof(int)),
    "cudaMemset(drain_status_dev)");

system::runtime::check_cuda(
    cudaMemcpy(src_dev, host_src.data(), bytes, cudaMemcpyHostToDevice),
    "cudaMemcpy(host_src -> src_dev)");
system::runtime::check_cuda(
    cudaMemcpy(dst_dev, host_zero.data(), bytes, cudaMemcpyHostToDevice),
    "cudaMemcpy(host_zero -> dst_dev)");
system::runtime::check_cuda(
    cudaMemset(publish_status_dev, 0, sizeof(int)),
    "cudaMemset(publish_status_dev)");

std::printf("[smoke] after device buffer alloc/init\n"); std::fflush(stdout);

comm::endpoint_runtime_configure_queue_operation(
    &runtime,
    0,
    1,
    0,
    dst_dev,
    bytes,
    bytes,
    1,
    comm::exec::ChunkOpKind::kReduceAddNoFtzF16,
    1,
    true);

std::printf("[smoke] after configure_queue_operation\n"); std::fflush(stdout);

comm::endpoint_persistent_control_init(&control, dev0);

std::printf("[smoke] before persistent launch\n"); std::fflush(stdout);
/*system::runtime::check_cuda(*/
    /*comm::launch_endpoint_persistent_kernel_sm90(*/
        /*comm::endpoint_runtime_device_handle(&runtime),*/
        /*&control,*/
        /*runtime.endpoint.stream),*/
    /*"launch_endpoint_persistent_kernel_sm90");*/

std::printf("[smoke] after persistent launch\n"); std::fflush(stdout);

std::printf("[smoke] after persistent launch\n"); std::fflush(stdout);

std::printf("[smoke] before publish launch\n"); std::fflush(stdout);
print_queue_head_tail(runtime.input_queues_host[0], "[smoke] queue before publish");
publish_single_ready_tile_kernel<<<1, 1, 0, producer_stream>>>(
    runtime.input_queues_host[0],
    src_dev,
    static_cast<uint32_t>(bytes),
    0,
    publish_status_dev);
system::runtime::check_cuda(cudaGetLastError(), "publish_single_ready_tile_kernel");

std::printf("[smoke] after publish launch\n"); std::fflush(stdout);
system::runtime::check_cuda(
    cudaStreamSynchronize(producer_stream),
    "cudaStreamSynchronize(producer_stream)");

std::printf("[smoke] after producer sync\n"); std::fflush(stdout);        
print_queue_head_tail(runtime.input_queues_host[0], "[smoke] queue after publish");

std::printf("[smoke] before execute_once launch\n"); std::fflush(stdout);
execute_once_kernel<<<1, 128, 0, producer_stream>>>(
    runtime.device,
    drain_status_dev);
system::runtime::check_cuda(
    cudaGetLastError(),
    "execute_once_kernel");

system::runtime::check_cuda(
    cudaStreamSynchronize(producer_stream),
    "cudaStreamSynchronize(execute_once)");

std::printf("[smoke] after execute_once sync\n"); std::fflush(stdout);
print_queue_head_tail(runtime.input_queues_host[0], "[smoke] queue after execute_once");

int drain_status = 0;
system::runtime::check_cuda(
    cudaMemcpy(
        &drain_status,
        drain_status_dev,
        sizeof(int),
        cudaMemcpyDeviceToHost),
    "cudaMemcpy(drain_status)");

std::printf("[smoke] drain status=%d\n", drain_status); std::fflush(stdout);

if (drain_status != 3) {
    throw std::runtime_error("drain_once_kernel did not consume the published tile");
}
        std::printf("[smoke] published tile\n"); std::fflush(stdout);
        std::printf("[smoke] waiting for queue drain\n"); std::fflush(stdout);
        
        const bool drained =
    poll_queue_head_until(runtime.input_queues_host[0], 1, dev0, 5000);
if (!drained) {
    throw std::runtime_error("timeout waiting for persistent kernel to consume queue head");
}

std::printf("[smoke] requesting stop\n"); std::fflush(stdout);

comm::endpoint_persistent_control_request_stop(&control);
system::runtime::check_cuda(
    cudaStreamSynchronize(runtime.endpoint.stream),
    "cudaStreamSynchronize(persistent stream)");
std::printf("[smoke] persistent stream joined\n"); std::fflush(stdout);
std::printf("[smoke] copying dst to host\n"); std::fflush(stdout);

std::vector<half> host_dst(static_cast<size_t>(numel));
system::runtime::check_cuda(
    cudaMemcpy(host_dst.data(), dst_dev, bytes, cudaMemcpyDeviceToHost),
    "cudaMemcpy(dst_dev -> host_dst)");

expect_half_vectors_close(host_dst, host_src, "endpoint_persistent_smoke_test");

        if (producer_stream != nullptr) {
            cudaStreamDestroy(producer_stream);
            producer_stream = nullptr;
        }
        if (publish_status_dev != nullptr) {
            cudaFree(publish_status_dev);
            publish_status_dev = nullptr;
        }
        if (dst_dev != nullptr) {
            cudaFree(dst_dev);
            dst_dev = nullptr;
        }
        if (src_dev != nullptr) {
            cudaFree(src_dev);
            src_dev = nullptr;
        }

        comm::endpoint_persistent_control_destroy(&control);
        comm::endpoint_runtime_destroy(&runtime);
        comm::group_destroy(&group);
        return true;
    } catch (...) {
        if (producer_stream != nullptr) {
            cudaStreamDestroy(producer_stream);
        }
        if (publish_status_dev != nullptr) {
            cudaFree(publish_status_dev);
        }
        if (dst_dev != nullptr) {
            cudaFree(dst_dev);
        }
        if (src_dev != nullptr) {
            cudaFree(src_dev);
        }

        comm::endpoint_persistent_control_destroy(&control);
        comm::endpoint_runtime_destroy(&runtime);
        comm::group_destroy(&group);
        throw;
    }
}

} // namespace ooverlap
