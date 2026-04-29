#pragma once

#include <nccl.h>
#include <vector>

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

class OverlapImpl : public torch::CustomClassHolder {
public:
    OverlapImpl();
    ~OverlapImpl();

    void CutlassInit();
    void NcclInit(const int64_t tp_rank, const int64_t tp_size, const std::vector<int64_t> tp_id);
    void OverlapInit();

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

    void NcclAllReduce(at::Tensor C);
    void NcclReduceScatter(at::Tensor C, at::Tensor D);

private:
    cudaStream_t gemm_stream_;
    cudaEvent_t mm_ready_;
    cudaStream_t comm_stream_;
    cudaEvent_t gemm_finished_;

    ncclComm_t comm_;
    int64_t my_rank_;
    int64_t my_size_;
    bool overlap_init_done_;
};
