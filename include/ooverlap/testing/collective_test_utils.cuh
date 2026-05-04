#pragma once

#include "ooverlap/testing/checks.cuh"
#include "ooverlap/testing/test_utils.cuh"
#include "ooverlap/system/runtime_utils.cuh"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <nccl.h>

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace ooverlap {
namespace testing {

enum class TestCollective {
    AllReduce = 0,
    ReduceScatter = 1,
    AllGather = 2,
};

inline TestCollective parse_collective(
    const std::string& value) {
    if (value == "allreduce" ||
        value == "all_reduce" ||
        value == "all-reduce" ||
        value == "ar") {
        return TestCollective::AllReduce;
    }

    if (value == "reduce_scatter" ||
        value == "reduce-scatter" ||
        value == "reducescatter" ||
        value == "rs") {
        return TestCollective::ReduceScatter;
    }

    if (value == "all_gather" ||
        value == "all-gather" ||
        value == "allgather" ||
        value == "ag") {
        return TestCollective::AllGather;
    }

    throw std::invalid_argument(
        "unknown collective '" + value +
        "'; expected allreduce, reduce_scatter, or all_gather");
}

inline const char* collective_name(
    TestCollective collective) {
    switch (collective) {
        case TestCollective::AllReduce:
            return "allreduce";
        case TestCollective::ReduceScatter:
            return "reduce_scatter";
        case TestCollective::AllGather:
            return "all_gather";
        default:
            return "unknown";
    }
}

inline double collective_code(
    TestCollective collective) {
    return static_cast<double>(static_cast<int>(collective));
}

inline void validate_numel_for_collective(
    TestCollective collective,
    int64_t numel,
    int world_size) {
    if (numel <= 0) {
        throw std::invalid_argument("numel must be > 0");
    }

    if ((collective == TestCollective::ReduceScatter ||
         collective == TestCollective::AllGather) &&
        (world_size > 0) &&
        ((numel % world_size) != 0)) {
        throw std::invalid_argument(
            std::string(collective_name(collective)) +
            " requires numel divisible by world_size for NCCL comparison");
    }
}

inline float rank_scale(int rank) {
    return rank == 0 ? 0.25f : 0.50f;
}

inline float rank_offset(int rank) {
    return rank == 0 ? 1.0f : 2.0f;
}

inline size_t rank_partition_begin(
    size_t count,
    int rank,
    int world_size) {
    const size_t world = static_cast<size_t>(world_size);
    const size_t r = static_cast<size_t>(rank);
    const size_t base = count / world;
    const size_t rem = count % world;

    return r * base + ((r < rem) ? r : rem);
}

inline size_t rank_partition_count(
    size_t count,
    int rank,
    int world_size) {
    const size_t world = static_cast<size_t>(world_size);
    const size_t r = static_cast<size_t>(rank);
    const size_t base = count / world;
    const size_t rem = count % world;

    return base + ((r < rem) ? 1 : 0);
}

inline std::vector<float> slice_vector(
    const std::vector<float>& values,
    size_t begin,
    size_t count) {
    if (begin > values.size() || count > values.size() - begin) {
        throw std::invalid_argument("slice_vector: invalid slice");
    }

    return std::vector<float>(
        values.begin() + static_cast<std::ptrdiff_t>(begin),
        values.begin() + static_cast<std::ptrdiff_t>(begin + count));
}

inline std::vector<float> reference_rank_fp16(
    int64_t numel,
    int rank) {
    return host_reference_pattern_fp16(
        numel,
        rank_scale(rank),
        rank_offset(rank));
}

inline std::vector<float> reference_sum_fp16(
    int64_t numel,
    int world_size) {
    std::vector<float> out(static_cast<size_t>(numel), 0.0f);

    for (int rank = 0; rank < world_size; ++rank) {
        const std::vector<float> ref =
            reference_rank_fp16(numel, rank);

        for (int64_t i = 0; i < numel; ++i) {
            float acc = out[static_cast<size_t>(i)] +
                        ref[static_cast<size_t>(i)];

            out[static_cast<size_t>(i)] = round_to_half(acc);
        }
    }

    return out;
}

inline std::vector<float> reference_all_gather_fp16(
    int64_t numel,
    int world_size) {
    std::vector<float> out(static_cast<size_t>(numel), 0.0f);

    for (int rank = 0; rank < world_size; ++rank) {
        const std::vector<float> ref =
            reference_rank_fp16(numel, rank);

        const size_t begin =
            rank_partition_begin(
                static_cast<size_t>(numel),
                rank,
                world_size);

        const size_t count =
            rank_partition_count(
                static_cast<size_t>(numel),
                rank,
                world_size);

        for (size_t i = 0; i < count; ++i) {
            const size_t idx = begin + i;
            out[idx] = ref[idx];
        }
    }

    return out;
}

inline void fill_rank_source_fp16(
    half* ptr,
    int64_t numel,
    int rank,
    int device,
    cudaStream_t stream) {
    system::runtime::set_device(device);

    fill_pattern(
        ptr,
        numel,
        rank_scale(rank),
        rank_offset(rank),
        stream);

    system::runtime::sync_stream_on_device(
        device,
        stream,
        "sync fill_rank_source_fp16");
}

inline void reset_work_buffer_async(
    half* work,
    const half* src,
    size_t bytes,
    int device,
    cudaStream_t stream) {
    system::runtime::set_device(device);

    check_cuda(
        cudaMemcpyAsync(
            work,
            src,
            bytes,
            cudaMemcpyDeviceToDevice,
            stream),
        "cudaMemcpyAsync(reset work buffer)");
}

inline void verify_allreduce_fp16(
    const char* label,
    half* work,
    int64_t numel,
    int rank,
    int world_size,
    int device) {
    const std::vector<float> got =
        copy_half_device_to_host_float(
            work,
            numel,
            device);

    const std::vector<float> ref =
        reference_sum_fp16(numel, world_size);

    expect_allclose(
        got,
        ref,
        (std::string(label) +
         " rank" +
         std::to_string(rank) +
         " allreduce").c_str());
}

inline void verify_reduce_scatter_fp16(
    const char* label,
    half* work,
    int64_t numel,
    int rank,
    int world_size,
    int device) {
    const std::vector<float> got =
        copy_half_device_to_host_float(
            work,
            numel,
            device);

    const std::vector<float> ref =
        reference_sum_fp16(numel, world_size);

    const size_t begin =
        rank_partition_begin(
            static_cast<size_t>(numel),
            rank,
            world_size);

    const size_t count =
        rank_partition_count(
            static_cast<size_t>(numel),
            rank,
            world_size);

    expect_allclose(
        slice_vector(got, begin, count),
        slice_vector(ref, begin, count),
        (std::string(label) +
         " rank" +
         std::to_string(rank) +
         " reduce_scatter").c_str());
}

inline void verify_all_gather_fp16(
    const char* label,
    half* work,
    int64_t numel,
    int rank,
    int world_size,
    int device) {
    const std::vector<float> got =
        copy_half_device_to_host_float(
            work,
            numel,
            device);

    const std::vector<float> ref =
        reference_all_gather_fp16(numel, world_size);

    expect_allclose(
        got,
        ref,
        (std::string(label) +
         " rank" +
         std::to_string(rank) +
         " all_gather").c_str());
}

inline void verify_collective_fp16(
    TestCollective collective,
    const char* label,
    half* work,
    int64_t numel,
    int rank,
    int world_size,
    int device) {
    if (collective == TestCollective::AllReduce) {
        verify_allreduce_fp16(
            label,
            work,
            numel,
            rank,
            world_size,
            device);
        return;
    }

    if (collective == TestCollective::ReduceScatter) {
        verify_reduce_scatter_fp16(
            label,
            work,
            numel,
            rank,
            world_size,
            device);
        return;
    }

    if (collective == TestCollective::AllGather) {
        verify_all_gather_fp16(
            label,
            work,
            numel,
            rank,
            world_size,
            device);
        return;
    }

    throw std::invalid_argument("verify_collective_fp16: unknown collective");
}

inline void launch_nccl_collective_fp16(
    TestCollective collective,
    ncclComm_t comm,
    half* work,
    size_t numel,
    int rank,
    int world_size,
    cudaStream_t stream) {
    if (collective == TestCollective::AllReduce) {
        OOVERLAP_TEST_NCCL_CHECK(
            ncclAllReduce(
                work,
                work,
                numel,
                ncclFloat16,
                ncclSum,
                comm,
                stream));
        return;
    }

    if (collective == TestCollective::ReduceScatter) {
        const size_t begin =
            rank_partition_begin(numel, rank, world_size);

        const size_t count =
            rank_partition_count(numel, rank, world_size);

        OOVERLAP_TEST_NCCL_CHECK(
            ncclReduceScatter(
                work,
                work + begin,
                count,
                ncclFloat16,
                ncclSum,
                comm,
                stream));
        return;
    }

    if (collective == TestCollective::AllGather) {
        const size_t begin =
            rank_partition_begin(numel, rank, world_size);

        const size_t count =
            rank_partition_count(numel, rank, world_size);

        OOVERLAP_TEST_NCCL_CHECK(
            ncclAllGather(
                work + begin,
                work,
                count,
                ncclFloat16,
                comm,
                stream));
        return;
    }

    throw std::invalid_argument("launch_nccl_collective_fp16: unknown collective");
}

} // namespace testing
} // namespace ooverlap
