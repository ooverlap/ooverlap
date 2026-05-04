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
    int chunk_bytes = 16 * 1024;
    int stage_depth = 8;
};

/*
 * New JSON-driven sweep executor.
 *
 * Input JSON shape:
 *
 * {
 *   "iters": 100,
 *   "warmup": 20,
 *   "dev0": 0,
 *   "dev1": 1,
 *   "scenarios": [
 *     {
 *       "id": "tma_128m",
 *       "backend": "ooverlap",
 *       "kernel": "tma_copy",
 *       "numel": 67108864,
 *       "threads": 1024,
 *       "window_chunks": 32,
 *       "chunk_bytes": 16384,
 *       "stage_depth": 8
 *     },
 *     {
 *       "id": "nccl_128m",
 *       "backend": "nccl",
 *       "numel": 67108864
 *     }
 *   ]
 * }
 *
 * Optional per-scenario fields override top-level fields:
 *
 *   iters
 *   warmup
 *   dev0
 *   dev1
 *
 * For ooverlap scenarios:
 *
 *   kernel: tma_copy | seq_fast_gmem | overlap_fast_gmem
 *   threads
 *   max_ctas        optional; if omitted, OOVERLAP_MAX_CTAS env is used if set
 *   window_chunks
 *   chunk_bytes
 *   stage_depth
 *
 * dtype/op are currently fixed:
 *
 *   dtype = fp16
 *   op    = sum
 *
 * Output JSON shape:
 *
 * {
 *   "ok": true,
 *   "results": [
 *     {
 *       "id": "...",
 *       "status": "ok",
 *       "backend": "ooverlap",
 *       "kernel": "tma_copy",
 *       "numel": ...,
 *       "bytes_per_rank": ...,
 *       "total_ms": ...,
 *       "avg_ms": ...,
 *       "effective_gbps_per_rank": ...,
 *       "effective_gbps_aggregate_2gpu": ...,
 *       ...
 *     }
 *   ],
 *   "errors": []
 * }
 */
std::string benchmark_tma_two_gpu_allreduce_sweep_json_sm90(
    const std::string& request_json);
} // namespace ooverlap
