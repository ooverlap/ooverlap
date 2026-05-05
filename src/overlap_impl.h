#pragma once

#include <nccl.h>
#include <vector>
#include <string>
#include <cstddef>

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

#include "ooverlap/comm.h"

class OverlapImpl : public torch::CustomClassHolder {
public:
    OverlapImpl();
    ~OverlapImpl();

    void CutlassInit();
    void NcclInit(
        const int64_t tp_rank,
        const int64_t tp_size,
        const std::vector<int64_t> tp_id);

    void OoverlapIpcInit(
        const int64_t tp_rank,
        const int64_t tp_size,
        const std::vector<int64_t> devices,
        const std::string broker_key);

    void OoverlapRelease();
    void OverlapInit();

    // Plain SM90 CUTLASS GEMM.
    // Layout: A physical [M,K], B physical [N,K], C physical [N,M].
    void Gemm(
        at::Tensor A,
        at::Tensor B,
        at::Tensor C,
        int64_t Algo);

    // Plain SM90 CUTLASS GEMM followed by full-buffer NCCL allreduce.
    void GemmAllReduce(
        at::Tensor A,
        at::Tensor B,
        at::Tensor C,
        int64_t Algo);

    // Fused reorder+signal GEMM and segmented allreduce overlap.
    // Uses ooverlap segmented allreduce if OoverlapIpcInit() was called;
    // otherwise uses NCCL segmented allreduce.
    void GemmAllReduceOverlap(
        at::Tensor A,
        at::Tensor B,
        at::Tensor C,
        at::Tensor MM,
        at::Tensor RA,
        int64_t rLDN,
        at::Tensor cSEG_CPU,
        at::Tensor cSEG_GPU,
        int64_t Algo,
        int64_t active_sm_count,
        bool if_monitor);

    void GemmReduceScatterOverlap(
        at::Tensor A,
        at::Tensor B,
        at::Tensor C,
        at::Tensor D,
        at::Tensor MM,
        at::Tensor RA,
        at::Tensor RE,
        int64_t rLDN,
        at::Tensor cSEG_CPU,
        at::Tensor cSEG_GPU,
        int64_t Algo,
        bool if_monitor);

    void SegAllReduce(
        at::Tensor C,
        at::Tensor cSEG_CPU,
        int64_t SegNum);

    // Standalone full-buffer public ooverlap allreduce for bandwidth.py.
    void OoverlapAllReduce(at::Tensor C);

    void NcclAllReduce(at::Tensor C);
    void NcclReduceScatter(at::Tensor C, at::Tensor D);

private:
    void OoverlapUnregisterBuffer();
    void OoverlapEnsureBuffer(at::Tensor C);
    void OoverlapAllReduceSlice(
        size_t element_offset,
        size_t count,
        cudaStream_t stream);

    oo_group_t* oo_group_;
    oo_node_t* oo_node_;
    oo_buffer_t* oo_local_buf_;
    oo_buffer_t* oo_peer_buf_;

    void* oo_registered_ptr_;
    size_t oo_registered_bytes_;

    int64_t oo_rank_;
    int64_t oo_size_;
    int oo_devices_[2];
    bool oo_initialized_;

    cudaStream_t gemm_stream_;
    cudaEvent_t mm_ready_;
    cudaStream_t comm_stream_;
    cudaEvent_t gemm_finished_;

    ncclComm_t comm_;
    int64_t my_rank_;
    int64_t my_size_;
    bool overlap_init_done_;
};
