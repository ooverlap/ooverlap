#pragma once

#include <string>

namespace ooverlap {

/*
 * JSON-driven SM90 two-GPU collective sweep executor.
 *
 * Python owns scenario generation. C++ only executes scenarios and returns
 * measured rows.
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
 *       "id": "ar_tma_128m",
 *       "backend": "ooverlap",
 *       "collective": "allreduce",
 *       "kernel": "tma_copy",
 *       "numel": 67108864,
 *       "threads": 1024,
 *       "max_ctas": 8,
 *       "window_chunks": 32,
 *       "chunk_bytes": 16384,
 *       "stage_depth": 8
 *     },
 *     {
 *       "id": "nccl_rs_128m",
 *       "backend": "nccl",
 *       "collective": "reduce_scatter",
 *       "numel": 67108864
 *     }
 *   ]
 * }
 *
 * Common fields:
 *
 *   id:
 *     Any JSON value. Copied back to the output row.
 *
 *   backend:
 *     "ooverlap" | "nccl"
 *
 *   collective:
 *     "allreduce" | "all_reduce" | "ar"
 *     "reduce_scatter" | "reduce-scatter" | "rs"
 *     "all_gather" | "all-gather" | "ag"
 *
 *   numel:
 *     Full logical tensor element count per rank. dtype is currently fp16.
 *
 *   bytes_per_rank:
 *     Alternative to numel. Must be divisible by sizeof(half).
 *
 * Optional top-level fields can be overridden per scenario:
 *
 *   iters
 *   warmup
 *   dev0
 *   dev1
 *
 * Ooverlap-only fields:
 *
 *   kernel:
 *     "tma_copy" | "seq_fast_gmem" | "overlap_fast_gmem"
 *
 *   threads
 *   max_ctas
 *   window_chunks
 *   chunk_bytes
 *   stage_depth
 *
 * If max_ctas is omitted, OOVERLAP_MAX_CTAS is used when set.
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
 *       "collective": "allreduce",
 *       "kernel": "tma_copy",
 *       "numel": ...,
 *       "bytes_per_rank": ...,
 *       "total_ms": ...,
 *       "avg_ms": ...,
 *       "effective_gbps_per_rank": ...,
 *       "effective_gbps_aggregate_2gpu": ...
 *     }
 *   ],
 *   "errors": []
 * }
 */
std::string benchmark_tma_two_gpu_collective_sweep_json(
    const std::string& request_json);

} // namespace ooverlap
