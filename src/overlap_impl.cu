#include "overlap_impl.h"

#include "nccl_utils.h"
#include "wait.cuh"
#include "overlap/gemm_signal_sm90_dispatch.h"
#include "overlap/gemm_scatter_sm90_dispatch.h"
#include "overlap/scatter_row_remap_sm90.cuh"
#include "ooverlap/torch/torch_utils.h"

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstring>

namespace {
constexpr int kTileM = 128;
constexpr int kTileN = 128;
} // namespace

OverlapImpl::OverlapImpl()
    : gemm_stream_(nullptr),
      comm_stream_(nullptr),
      gemm_finished_(nullptr),
      mm_ready_(nullptr),
      comm_(nullptr),
      my_rank_(0),
      my_size_(1),
      overlap_init_done_(false) {}

OverlapImpl::~OverlapImpl() {
    if (gemm_finished_ != nullptr) {
        cudaEventDestroy(gemm_finished_);
        gemm_finished_ = nullptr;
    }
    if (comm_stream_ != nullptr) {
        cudaStreamDestroy(comm_stream_);
        comm_stream_ = nullptr;
    }
    if (comm_ != nullptr) {
        ncclCommDestroy(comm_);
        comm_ = nullptr;
    }
    if (mm_ready_ != nullptr) {
        cudaEventDestroy(mm_ready_);
        mm_ready_ = nullptr;
    }
}

void OverlapImpl::CutlassInit() {
    ooverlap::torch_utils::refresh_gemm_stream(gemm_stream_);
}

void OverlapImpl::NcclInit(const int64_t tp_rank, const int64_t tp_size, const std::vector<int64_t> tp_id) {
    my_rank_ = tp_rank;
    my_size_ = tp_size;

    TORCH_CHECK(
        static_cast<int64_t>(tp_id.size() * sizeof(int64_t)) == NCCL_UNIQUE_ID_BYTES,
        "tp_id must contain exactly NCCL_UNIQUE_ID_BYTES bytes; got ",
        tp_id.size(), " int64 values (", tp_id.size() * sizeof(int64_t), " bytes)"
    );

    ncclUniqueId uid;
    std::memcpy(uid.internal, tp_id.data(), NCCL_UNIQUE_ID_BYTES);

    if (my_size_ == 1) {
        comm_ = nullptr;
        return;
    }

    NCCL_CHECK(ncclCommInitRank(&comm_, my_size_, uid, my_rank_));
}

void OverlapImpl::OverlapInit() {
    if (!overlap_init_done_) {
        cudaError_t err = cudaStreamCreateWithPriority(&comm_stream_, cudaStreamNonBlocking, -5);
        TORCH_CHECK(err == cudaSuccess,
                    "cudaStreamCreateWithPriority failed: ", cudaGetErrorString(err));

        err = cudaEventCreateWithFlags(&gemm_finished_, cudaEventDisableTiming);
        TORCH_CHECK(err == cudaSuccess,
                    "cudaEventCreateWithFlags failed: ", cudaGetErrorString(err));

        err = cudaEventCreateWithFlags(&mm_ready_, cudaEventDisableTiming);
        TORCH_CHECK(err == cudaSuccess,
                    "cudaEventCreateWithFlags mm_ready_ failed: ",
                    cudaGetErrorString(err));

        overlap_init_done_ = true;
    }
}

void OverlapImpl::NcclAllReduce(at::Tensor C) {
    TORCH_CHECK(C.is_cuda(), "C must be CUDA");
    TORCH_CHECK(C.scalar_type() == torch::kFloat16, "C must be float16");
    TORCH_CHECK(C.is_contiguous(), "C must be contiguous");

    ooverlap::torch_utils::refresh_gemm_stream(gemm_stream_);

    if (my_size_ == 1 || comm_ == nullptr) {
        return;
    }

    half* c_ptr = reinterpret_cast<half*>(C.data_ptr<at::Half>());
    NCCL_CHECK(ncclAllReduce(
        (void*)c_ptr,
        (void*)c_ptr,
        static_cast<size_t>(C.numel()),
        ncclFloat16,
        ncclSum,
        comm_,
        gemm_stream_));
}

void OverlapImpl::NcclReduceScatter(at::Tensor C, at::Tensor D) {
    TORCH_CHECK(C.is_cuda() && D.is_cuda(), "C/D must be CUDA");
    TORCH_CHECK(C.scalar_type() == torch::kFloat16 && D.scalar_type() == torch::kFloat16,
                "C/D must be float16");
    TORCH_CHECK(C.is_contiguous() && D.is_contiguous(), "C/D must be contiguous");

    ooverlap::torch_utils::refresh_gemm_stream(gemm_stream_);

    half* c_ptr = reinterpret_cast<half*>(C.data_ptr<at::Half>());
    half* d_ptr = reinterpret_cast<half*>(D.data_ptr<at::Half>());

    if (my_size_ == 1 || comm_ == nullptr) {
        cudaError_t err = cudaMemcpyAsync(
            d_ptr, c_ptr, ooverlap::torch_utils::tensor_nbytes(D),
            cudaMemcpyDeviceToDevice, gemm_stream_);
        TORCH_CHECK(err == cudaSuccess, "cudaMemcpyAsync failed: ", cudaGetErrorString(err));
        return;
    }

    TORCH_CHECK(C.numel() % my_size_ == 0, "C.numel() must be divisible by world size");
    TORCH_CHECK(D.numel() == C.numel() / my_size_,
                "D.numel() must equal C.numel()/world_size");

    NCCL_CHECK(ncclReduceScatter(
        (void*)c_ptr,
        (void*)d_ptr,
        static_cast<size_t>(D.numel()),
        ncclFloat16,
        ncclSum,
        comm_,
        gemm_stream_));
}

void OverlapImpl::GemmAllReduceOverlap(
    at::Tensor A,
    at::Tensor B,
    at::Tensor C,
    at::Tensor MM,
    at::Tensor RA,
    int64_t rLDN,
    at::Tensor cSEG_CPU,
    at::Tensor cSEG_GPU,
    int64_t Algo,
    bool if_monitor) {

    ooverlap::torch_utils::check_common_gemm_inputs(A, B);
    TORCH_CHECK(C.is_cuda(), "C must be CUDA");
    TORCH_CHECK(C.scalar_type() == torch::kFloat16, "C must be float16");
    TORCH_CHECK(C.is_contiguous(), "C must be contiguous");
    TORCH_CHECK(MM.is_cuda() && RA.is_cuda() && cSEG_GPU.is_cuda(),
                "MM/RA/cSEG_GPU must be CUDA");
    TORCH_CHECK(cSEG_CPU.device().is_cpu(), "cSEG_CPU must be CPU");
    TORCH_CHECK(MM.scalar_type() == torch::kInt32, "MM must be int32");
    TORCH_CHECK(RA.scalar_type() == torch::kInt32, "RA must be int32");
    TORCH_CHECK(cSEG_CPU.scalar_type() == torch::kInt32 &&
                cSEG_GPU.scalar_type() == torch::kInt32,
                "cSEG tensors must be int32");
    TORCH_CHECK(rLDN > 0, "rLDN must be > 0");
    TORCH_CHECK(Algo >= 0 && Algo <= 4, "Unsupported algo=", Algo);

    ooverlap::torch_utils::ensure_streams_ready(gemm_stream_, comm_stream_, overlap_init_done_);

    if (gemm_finished_ == nullptr) {
        cudaError_t err = cudaEventCreateWithFlags(&gemm_finished_, cudaEventDisableTiming);
        TORCH_CHECK(err == cudaSuccess,
                    "cudaEventCreateWithFlags failed: ", cudaGetErrorString(err));
    }

    const int M = static_cast<int>(A.size(0));
    const int K = static_cast<int>(A.size(1));
    const int N = static_cast<int>(B.size(0));

    TORCH_CHECK(static_cast<int>(B.size(1)) == K, "B must have shape [N, K]");
    TORCH_CHECK(M % kTileM == 0 && N % kTileN == 0, "M/N must be multiples of 128");

    const int tile_rows = M / kTileM;
    const int tile_cols = N / kTileN;
    const int tile_num  = tile_rows * tile_cols;

    TORCH_CHECK(static_cast<int>(RA.numel()) == tile_num,
                "RA.numel() must equal tile count");
    TORCH_CHECK(C.numel() >= static_cast<int64_t>(M) * N,
                "C must hold at least M*N elements");

    const int seg_size = static_cast<int>(cSEG_GPU.numel());
    TORCH_CHECK(seg_size == static_cast<int>(cSEG_CPU.numel()),
                "cSEG_CPU/GPU size mismatch");

    auto* a_ptr = reinterpret_cast<half*>(A.data_ptr<at::Half>());
    auto* b_ptr = reinterpret_cast<half*>(B.data_ptr<at::Half>());
    auto* c_ptr = reinterpret_cast<half*>(C.data_ptr<at::Half>());
    auto* mm_ptr = MM.data_ptr<int>();
    auto* ra_ptr = RA.data_ptr<int>();
    auto* cseg_cpu_ptr = cSEG_CPU.data_ptr<int>();
    auto* cseg_gpu_ptr = cSEG_GPU.data_ptr<int>();

    cudaError_t err = cudaEventRecord(mm_ready_, gemm_stream_);
    TORCH_CHECK(err == cudaSuccess,
                "cudaEventRecord mm_ready_ failed: ",
                cudaGetErrorString(err));
    
    err = cudaStreamWaitEvent(comm_stream_, mm_ready_, 0);
    TORCH_CHECK(err == cudaSuccess,
                "cudaStreamWaitEvent mm_ready_ failed: ",
            cudaGetErrorString(err));

    bool ok = ooverlap::gemm_signal_sm90_dispatch(
        static_cast<int>(Algo),
        M, N, K,
        static_cast<int>(rLDN),
        seg_size,
        reinterpret_cast<int32_t*>(cseg_gpu_ptr),
        reinterpret_cast<void*>(a_ptr),
        reinterpret_cast<void*>(b_ptr),
        reinterpret_cast<void*>(c_ptr),
        reinterpret_cast<int32_t*>(mm_ptr),
        reinterpret_cast<int32_t*>(ra_ptr),
        if_monitor,
        gemm_stream_);

    TORCH_CHECK(ok, "Unsupported algo=", Algo);

    if (my_size_ == 1 || comm_ == nullptr) {
        return;
    }

    int acc_addr = 0;
    for (int iter = 0; iter < seg_size; ++iter) {
        const int this_seg = cseg_cpu_ptr[iter];
        const int comm_size = (M * N / tile_num) * this_seg;

        kernel_wait_flag<<<1, 1, 0, comm_stream_>>>(this_seg, (mm_ptr + iter));

        NCCL_CHECK(ncclAllReduce(
            (void*)(c_ptr + acc_addr),
            (void*)(c_ptr + acc_addr),
            static_cast<size_t>(comm_size),
            ncclFloat16,
            ncclSum,
            comm_,
            comm_stream_));

        acc_addr += comm_size;
    }

    cudaEventRecord(gemm_finished_, comm_stream_);
    cudaStreamWaitEvent(gemm_stream_, gemm_finished_, 0);
}

void OverlapImpl::GemmReduceScatterOverlap(
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
    bool if_monitor) {

    ooverlap::torch_utils::check_common_gemm_inputs(A, B);
    TORCH_CHECK(C.is_cuda() && D.is_cuda(), "C/D must be CUDA");
    TORCH_CHECK(C.scalar_type() == torch::kFloat16 &&
                D.scalar_type() == torch::kFloat16,
                "C/D must be float16");
    TORCH_CHECK(C.is_contiguous() && D.is_contiguous(), "C/D must be contiguous");
    TORCH_CHECK(MM.is_cuda() && RA.is_cuda() && RE.is_cuda() && cSEG_GPU.is_cuda(),
                "MM/RA/RE/cSEG_GPU must be CUDA");
    TORCH_CHECK(cSEG_CPU.device().is_cpu(), "cSEG_CPU must be CPU");
    TORCH_CHECK(MM.scalar_type() == torch::kInt32, "MM must be int32");
    TORCH_CHECK(RA.scalar_type() == torch::kInt32, "RA must be int32");
    TORCH_CHECK(RE.scalar_type() == torch::kInt32, "RE must be int32");
    TORCH_CHECK(cSEG_CPU.scalar_type() == torch::kInt32 &&
                cSEG_GPU.scalar_type() == torch::kInt32,
                "cSEG tensors must be int32");
    TORCH_CHECK(rLDN > 0, "rLDN must be > 0");
    TORCH_CHECK(Algo == 0, "Only algo=0 is currently supported for scatter overlap");

    ooverlap::torch_utils::ensure_streams_ready(gemm_stream_, comm_stream_, overlap_init_done_);

    if (gemm_finished_ == nullptr) {
        cudaError_t err = cudaEventCreateWithFlags(&gemm_finished_, cudaEventDisableTiming);
        TORCH_CHECK(err == cudaSuccess,
                    "cudaEventCreateWithFlags failed: ", cudaGetErrorString(err));
    }

    const int M = static_cast<int>(A.size(0));
    const int K = static_cast<int>(A.size(1));
    const int N = static_cast<int>(B.size(0));

    TORCH_CHECK(static_cast<int>(B.size(1)) == K, "B must have shape [N, K]");
    TORCH_CHECK(M % kTileM == 0 && N % kTileN == 0, "M/N must be multiples of 128");

    const int tile_rows = M / kTileM;
    const int tile_cols = N / kTileN;
    const int tile_num  = tile_rows * tile_cols;

    TORCH_CHECK(static_cast<int>(RA.numel()) == tile_num,
                "RA.numel() must equal tile count");
    TORCH_CHECK(static_cast<int>(RE.numel()) == (M * N) / kTileN,
                "RE.numel() must equal M*N/TileN");
    TORCH_CHECK(C.numel() >= static_cast<int64_t>(M) * N,
                "C must hold at least M*N elements");
    TORCH_CHECK(D.numel() * my_size_ == static_cast<int64_t>(M) * N || my_size_ == 1,
                "For RS, D.numel() must equal M*N/world_size");

    const int seg_size = static_cast<int>(cSEG_GPU.numel());
    TORCH_CHECK(seg_size == static_cast<int>(cSEG_CPU.numel()),
                "cSEG_CPU/GPU size mismatch");

    auto* a_ptr = reinterpret_cast<half*>(A.data_ptr<at::Half>());
    auto* b_ptr = reinterpret_cast<half*>(B.data_ptr<at::Half>());
    auto* c_ptr = reinterpret_cast<half*>(C.data_ptr<at::Half>());
    auto* d_ptr = reinterpret_cast<half*>(D.data_ptr<at::Half>());
    auto* mm_ptr = MM.data_ptr<int>();
    auto* ra_ptr = RA.data_ptr<int>();
    auto* re_ptr = RE.data_ptr<int>();
    auto* cseg_cpu_ptr = cSEG_CPU.data_ptr<int>();
    auto* cseg_gpu_ptr = cSEG_GPU.data_ptr<int>();

    bool ok = ooverlap::gemm_scatter_sm90_dispatch(
        static_cast<int>(Algo),
        M, N, K,
        static_cast<int>(rLDN),
        reinterpret_cast<int32_t*>(cseg_gpu_ptr),
        reinterpret_cast<void*>(a_ptr),
        reinterpret_cast<void*>(b_ptr),
        reinterpret_cast<void*>(c_ptr),
        reinterpret_cast<int32_t*>(mm_ptr),
        reinterpret_cast<int32_t*>(ra_ptr),
        reinterpret_cast<int32_t*>(re_ptr),
        if_monitor,
        gemm_stream_);

    TORCH_CHECK(ok, "Unsupported algo=", Algo);

    if (my_size_ == 1 || comm_ == nullptr) {
        cudaError_t err = cudaMemcpyAsync(
            d_ptr, c_ptr, ooverlap::torch_utils::tensor_nbytes(D),
            cudaMemcpyDeviceToDevice, gemm_stream_);
        TORCH_CHECK(err == cudaSuccess, "cudaMemcpyAsync failed: ", cudaGetErrorString(err));
        return;
    }

    at::Tensor TMP = at::empty_like(C);
    auto* tmp_ptr = reinterpret_cast<half*>(TMP.data_ptr<at::Half>());

    int acc_addr = 0;
    for (int iter = 0; iter < seg_size; ++iter) {
        const int this_seg = cseg_cpu_ptr[iter];
        const int comm_size = (M * N / tile_num) * this_seg;

        const int row_begin = acc_addr / kTileN;
        const int row_count = this_seg * kTileM;

        kernel_wait_flag<<<1, 1, 0, comm_stream_>>>(this_seg, (mm_ptr + iter));

        cudaError_t err = ooverlap::launch_scatter_row_remap_sm90(
            c_ptr + acc_addr,
            tmp_ptr + acc_addr,
            reinterpret_cast<const int32_t*>(re_ptr),
            row_begin,
            row_count,
            kTileN,
            comm_stream_);

        TORCH_CHECK(err == cudaSuccess,
                    "launch_scatter_row_remap_sm90 failed: ", cudaGetErrorString(err));

        NCCL_CHECK(ncclReduceScatter(
            (void*)(tmp_ptr + acc_addr),
            (void*)(d_ptr + acc_addr / my_size_),
            static_cast<size_t>(comm_size / my_size_),
            ncclFloat16,
            ncclSum,
            comm_,
            comm_stream_));

        acc_addr += comm_size;
    }

    cudaEventRecord(gemm_finished_, comm_stream_);
    cudaStreamWaitEvent(gemm_stream_, gemm_finished_, 0);
}
