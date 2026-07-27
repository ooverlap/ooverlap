#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace ooverlap {

/*
 * Rank-generic, same-process SM90 CTA tuning sweep.
 *
 * The caller supplies one collective, a list of full logical fp16 element
 * counts, and the CUDA devices representing the logical ranks. All ranks are
 * created and launched inside this process with oo_group_create_p2p(); no CUDA
 * IPC or external broker is involved.
 *
 * The candidate configuration is read once per process from:
 *
 *   OOVERLAP_MAX_CTAS
 *   OOVERLAP_MAX_CTAS_PER_REDUCE_TASK
 *
 * Both variables must be positive integers. The Python tuner is expected to
 * spawn one fresh process for every candidate tuple, call this function with
 * all desired message sizes, then rank the returned avg_ms values.
 *
 * This benchmark deliberately bypasses the runtime tuning-policy selector and
 * dispatches the explicit TMA LaunchConfig represented by those environment
 * variables. dtype is intentionally fixed to fp16 because dtype is not part of
 * the simplified policy key.
 *
 * Returned numeric fields per size:
 *
 *   collective
 *   world_size
 *   numel
 *   bytes
 *   iters
 *   warmup
 *   max_ctas
 *   max_ctas_per_reduce_task
 *   total_ms
 *   avg_ms
 *   latency_us
 */
std::vector<std::map<std::string, double>>
benchmark_tma_collective_cta_sweep_sm90(
    const std::string& collective,
    const std::vector<int64_t>& numels,
    int iters,
    int warmup,
    const std::vector<int>& devices,
    bool verify = false);

} // namespace ooverlap
