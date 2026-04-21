#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <cstddef>
#include <cstdint>
#include <map>
#include <string>

#include "comm/communicator.h"

namespace ooverlap {

cudaError_t enqueue_persistent_two_gpu_allreduce_sm90(
    comm::Communicator* comm,
    half* rank0_in,
    half* rank1_in,
    half* rank0_out,
    half* rank1_out,
    size_t numel);

bool tma_persistent_two_gpu_allreduce_smoke_test(
    int64_t numel,
    int dev0 = 0,
    int dev1 = 1);

std::map<std::string, double> benchmark_persistent_two_gpu_allreduce_sm90(
    int64_t numel,
    int iters,
    int warmup,
    int dev0 = 0,
    int dev1 = 1);

} // namespace ooverlap
