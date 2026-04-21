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

__global__ void publish_single_ready_tile_kernel(
    comm::collective::ReadyTileQueue queue,
    const half* src,
    uint32_t bytes,
    uint64_t tile_id,
    int* status_out) {
    if (blockIdx.x != 0 || threadIdx.x != 0) {
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

void copy_queue_state(
    const comm::collective::ReadyTileQueue& q,
    uint64_t* head_out,
    uint64_t* tail_out) {
    if (head_out == nullptr || tail_out == nullptr) {
        throw std::invalid_argument("copy_queue_state: null output");
    }

    cudaError_t err = cudaMemcpy(
        head_out,
        q.head,
        sizeof(uint64_t),
        cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        throw std::runtime_error(
            std::string("copy_queue_state(head) failed: ") +
            cudaGetErrorString(err));
    }

    err = cudaMemcpy(
        tail_out,
        q.tail,
        sizeof(uint64_t),
        cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        throw std::runtime_error(
            std::string("copy_queue_state(tail) failed: ") +
            cudaGetErrorString(err));
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
    cudaStream_t producer_stream = nullptr;

    bool control_initialized = false;
    bool persistent_launched = false;

    try {
        const size_t bytes = static_cast<size_t>(numel) * sizeof(half);
        const auto host_src = make_host_pattern(numel);
        const std::vector<half> host_zero(
            static_cast<size_t>(numel),
            __float2half_rn(0.0f));

        std::printf("[smoke] before group_init\n"); std::fflush(stdout);
        comm::group_init(&group, {dev0, dev1}, comm::kEndpointPersistentChunkBytes);

        std::printf("[smoke] after group_init\n"); std::fflush(stdout);
        comm::endpoint_runtime_init(&runtime, &group, 0, 1, 8);

        std::printf("[smoke] after endpoint_runtime_init\n"); std::fflush(stdout);

        system::runtime::set_device(dev0);
        system::runtime::check_cuda(
            cudaStreamCreateWithFlags(&producer_stream, cudaStreamNonBlocking),
            "cudaStreamCreateWithFlags(producer_stream)");

        system::runtime::check_cuda(
            cudaMalloc(&src_dev, bytes),
            "cudaMalloc(src_dev)");
        system::runtime::check_cuda(
            cudaMalloc(&dst_dev, bytes),
            "cudaMalloc(dst_dev)");
        system::runtime::check_cuda(
            cudaMalloc(&publish_status_dev, sizeof(int)),
            "cudaMalloc(publish_status_dev)");

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
        control_initialized = true;

        std::printf("[smoke] before publish launch\n"); std::fflush(stdout);
        publish_single_ready_tile_kernel<<<1, 1, 0, producer_stream>>>(
            runtime.input_queues_host[0],
            src_dev,
            static_cast<uint32_t>(bytes),
            0,
            publish_status_dev);
        system::runtime::check_cuda(
            cudaGetLastError(),
            "publish_single_ready_tile_kernel");

        std::printf("[smoke] after publish launch\n"); std::fflush(stdout);
        system::runtime::check_cuda(
            cudaStreamSynchronize(producer_stream),
            "cudaStreamSynchronize(producer_stream)");
        std::printf("[smoke] after producer sync\n"); std::fflush(stdout);

        int publish_status = 0;
        system::runtime::check_cuda(
            cudaMemcpy(
                &publish_status,
                publish_status_dev,
                sizeof(int),
                cudaMemcpyDeviceToHost),
            "cudaMemcpy(publish_status)");

        if (publish_status != 1) {
            throw std::runtime_error("publish_single_ready_tile_kernel failed to publish tile");
        }

        std::printf("[smoke] before persistent launch\n"); std::fflush(stdout);
        system::runtime::check_cuda(
            comm::launch_endpoint_persistent_kernel_sm90(
                comm::endpoint_runtime_device_handle(&runtime),
                &control,
                runtime.endpoint.stream),
            "launch_endpoint_persistent_kernel_sm90");
        persistent_launched = true;
        std::printf("[smoke] after persistent launch\n"); std::fflush(stdout);

        std::printf("[smoke] letting persistent kernel run\n"); std::fflush(stdout);
        std::this_thread::sleep_for(std::chrono::milliseconds(100));

        std::printf("[smoke] requesting stop\n"); std::fflush(stdout);
        comm::endpoint_persistent_control_request_stop(&control);

        system::runtime::check_cuda(
            cudaStreamSynchronize(runtime.endpoint.stream),
            "cudaStreamSynchronize(persistent stream)");
        std::printf("[smoke] persistent stream joined\n"); std::fflush(stdout);

        uint64_t head = 0;
        uint64_t tail = 0;
        copy_queue_state(runtime.input_queues_host[0], &head, &tail);

        std::printf(
            "[smoke] queue after persistent head=%llu tail=%llu\n",
            static_cast<unsigned long long>(head),
            static_cast<unsigned long long>(tail));
        std::fflush(stdout);

        if (head != 1 || tail != 1) {
            throw std::runtime_error(
                "persistent kernel did not consume exactly one published tile");
        }

        std::printf("[smoke] copying dst to host\n"); std::fflush(stdout);
        std::vector<half> host_dst(static_cast<size_t>(numel));
        system::runtime::check_cuda(
            cudaMemcpy(host_dst.data(), dst_dev, bytes, cudaMemcpyDeviceToHost),
            "cudaMemcpy(dst_dev -> host_dst)");

        expect_half_vectors_close(
            host_dst,
            host_src,
            "endpoint_persistent_smoke_test");

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

        if (control_initialized) {
            comm::endpoint_persistent_control_destroy(&control);
            control_initialized = false;
        }

        comm::endpoint_runtime_destroy(&runtime);
        comm::group_destroy(&group);
        return true;
    } catch (...) {
        try {
            if (persistent_launched && control_initialized) {
                comm::endpoint_persistent_control_request_stop(&control);
                cudaStreamSynchronize(runtime.endpoint.stream);
            }
        } catch (...) {
        }

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
        if (control_initialized) {
            comm::endpoint_persistent_control_destroy(&control);
        }

        comm::endpoint_runtime_destroy(&runtime);
        comm::group_destroy(&group);
        throw;
    }
}

} // namespace ooverlap
