#pragma once

#include "ooverlap/system/peer_buffer.cuh"

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <cstddef>
#include <cstdint>

namespace ooverlap {

struct TmaTwoGpuPeerAllreduceState {
    int dev0 = -1;
    int dev1 = -1;
    int num_chunks = 0;

    // Each control buffer stores 2 * kMaxWindows ints:
    //   [0 .. kMaxWindows-1]                 = local-init-done flags
    //   [kMaxWindows .. 2*kMaxWindows - 1]   = peer-reduce-done flags
    system::mapped_peer_buffer progress0{};
    system::mapped_peer_buffer progress1{};
};

struct TmaTwoGpuPeerAllreduceOutputs {
    int dev0 = -1;
    int dev1 = -1;

    system::mapped_peer_buffer out0{};
    system::mapped_peer_buffer out1{};
};

int tma_two_gpu_peer_allreduce_compute_num_chunks(size_t numel);

void tma_two_gpu_peer_allreduce_configure_kernel_once(int device);

void tma_two_gpu_peer_allreduce_state_init(
    TmaTwoGpuPeerAllreduceState* st,
    int dev0,
    int dev1,
    size_t numel);

void tma_two_gpu_peer_allreduce_state_destroy(
    TmaTwoGpuPeerAllreduceState* st);

void tma_two_gpu_peer_allreduce_outputs_init(
    TmaTwoGpuPeerAllreduceOutputs* outs,
    int dev0,
    int dev1,
    size_t numel);

void tma_two_gpu_peer_allreduce_outputs_destroy(
    TmaTwoGpuPeerAllreduceOutputs* outs);

cudaError_t prime_tma_two_gpu_peer_allreduce_outputs_sm90(
    TmaTwoGpuPeerAllreduceState* st,
    const half* rank0_in,
    const half* rank1_in,
    half* rank0_out_peer,
    half* rank1_out_peer,
    size_t numel,
    cudaStream_t stream0,
    cudaStream_t stream1);

cudaError_t enqueue_tma_two_gpu_peer_allreduce_rank_sm90(
    const half* local_in,
    half* local_buf,
    half* peer_buf,
    size_t numel,
    int rank,
    int dev0,
    int dev1,
    cudaStream_t stream);

cudaError_t enqueue_tma_two_gpu_peer_allreduce_kernel_only_sm90(
    TmaTwoGpuPeerAllreduceState* st,
    const half* rank0_in,
    const half* rank1_in,
    half* rank0_out_peer,
    half* rank1_out_peer,
    size_t numel,
    cudaStream_t stream0,
    cudaStream_t stream1);

cudaError_t enqueue_tma_two_gpu_peer_allreduce_sm90(
    TmaTwoGpuPeerAllreduceState* st,
    const half* rank0_in,
    const half* rank1_in,
    half* rank0_out_peer,
    half* rank1_out_peer,
    size_t numel,
    cudaStream_t stream0,
    cudaStream_t stream1);

} // namespace ooverlap
