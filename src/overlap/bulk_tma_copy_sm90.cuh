#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstddef>

namespace ooverlap {

cudaError_t enqueue_bulk_tma_copy_sm90(
    const half* src,
    half* dst,
    size_t num_elements,
    cudaStream_t stream);

} // namespace ooverlap
