#include "comm/communicator.h"

#include "overlap/bulk_tma_copy_sm90.cuh"

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <cstddef>

namespace ooverlap {
namespace comm {

cudaError_t channel_send_bulk_tma(
    Communicator* comm,
    int src_rank,
    int dst_rank,
    int slot_idx,
    const half* src,
    size_t numel,
    cudaStream_t stream) {

    (void)comm;

    if (src == nullptr) {
        return cudaErrorInvalidDevicePointer;
    }

    CommBuffer* dst_buf = channel_get_slot_buffer(comm, src_rank, dst_rank, slot_idx);
    if (dst_buf == nullptr) {
        return cudaErrorInvalidValue;
    }

    const size_t bytes = numel * sizeof(half);
    if (bytes > dst_buf->bytes) {
        return cudaErrorInvalidValue;
    }

    return enqueue_bulk_tma_copy_sm90(src, buffer_as_half(dst_buf), numel, stream);
}

} // namespace comm
} // namespace ooverlap
