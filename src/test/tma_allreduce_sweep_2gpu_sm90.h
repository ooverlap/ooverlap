#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace ooverlap {

enum class SweepKernelKind {
    kTmaCopy = 0,
    kSeqFastGmem = 1,
    kOverlapFastGmem = 2,
};

struct SweepLaunchConfig {
    SweepKernelKind kernel = SweepKernelKind::kSeqFastGmem;
    int threads = 1024;
    int max_ctas = 16;
    int window_chunks = 32;
};

std::string benchmark_tma_two_gpu_allreduce_sweep_sm90(
    const std::vector<int64_t>& numels,
    const std::vector<std::string>& kernels,
    const std::vector<int>& threads,
    const std::vector<int>& max_ctas,
    const std::vector<int>& window_chunks,
    int iters,
    int warmup,
    int dev0 = 0,
    int dev1 = 1);

} // namespace ooverlap
