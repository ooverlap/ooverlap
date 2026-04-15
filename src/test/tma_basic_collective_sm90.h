#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <cstddef>
#include <cstdint>
#include <vector>

#include "comm/communicator.h"

namespace ooverlap {

using BasicCollectiveState = Communicator;

bool init_basic_collective_same_process(
    BasicCollectiveState* st,
    const std::vector<int>& devices,
    size_t max_full_numel);

void destroy_basic_collective_same_process(BasicCollectiveState* st);

half* basic_collective_shard_output_ptr(BasicCollectiveState* st, int rank);
half* basic_collective_full_output_ptr(BasicCollectiveState* st, int rank);

cudaError_t enqueue_basic_reduce_scatter_tma_sm90(
    BasicCollectiveState* st,
    const std::vector<half*>& local_full_buffers,
    size_t full_numel);

cudaError_t enqueue_basic_all_gather_tma_sm90(
    BasicCollectiveState* st,
    const std::vector<half*>& local_shard_buffers,
    size_t shard_numel);

cudaError_t enqueue_basic_all_reduce_tma_sm90(
    BasicCollectiveState* st,
    const std::vector<half*>& local_full_buffers,
    size_t full_numel);

bool tma_basic_ngpu_reduce_scatter_smoke_test(
    int64_t full_numel,
    const std::vector<int64_t>& devices = {});

bool tma_basic_ngpu_all_gather_smoke_test(
    int64_t shard_numel,
    const std::vector<int64_t>& devices = {});

bool tma_basic_ngpu_all_reduce_smoke_test(
    int64_t full_numel,
    const std::vector<int64_t>& devices = {});

} // namespace ooverlap
