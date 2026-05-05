#include "overlap_impl.h"

#include "nccl_utils.h"
#include "wait.cuh"

#include "gemm/gemm_signal_sm90_dispatch.h"
#include "gemm/gemm_scatter_sm90_dispatch.h"
#include "gemm/scatter_row_remap_sm90.cuh"

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <cstring>
#include <cstdint>

namespace {
constexpr int kTileM = 128;
constexpr int kTileN = 128;

const char* oo_status_to_string(oo_status_t status) {
    switch (status) {
        case OO_SUCCESS: return "OO_SUCCESS";
        case OO_ERROR_INVALID_ARGUMENT: return "OO_ERROR_INVALID_ARGUMENT";
        case OO_ERROR_INVALID_DEVICE: return "OO_ERROR_INVALID_DEVICE";
        case OO_ERROR_UNSUPPORTED: return "OO_ERROR_UNSUPPORTED";
        case OO_ERROR_CUDA: return "OO_ERROR_CUDA";
        case OO_ERROR_INTERNAL: return "OO_ERROR_INTERNAL";
        default: return "OO_ERROR_UNKNOWN";
    }
}

void OO_CHECK(oo_status_t status, const char* what) {
    TORCH_CHECK(status == OO_SUCCESS, what, " failed with ", oo_status_to_string(status));
}

void CUDA_CHECK(cudaError_t err, const char* what) {
    TORCH_CHECK(err == cudaSuccess, what, " failed with ", cudaGetErrorString(err));
}

void check_half_cuda(at::Tensor T, const char* name) {
    TORCH_CHECK(T.is_cuda(), name, " must be CUDA");
    TORCH_CHECK(T.scalar_type() == torch::kFloat16, name, " must be float16");
    TORCH_CHECK(T.is_contiguous(), name, " must be contiguous");
}

void check_i32_cuda(at::Tensor T, const char* name) {
    TORCH_CHECK(T.is_cuda(), name, " must be CUDA");
    TORCH_CHECK(T.scalar_type() == torch::kInt32, name, " must be int32");
    TORCH_CHECK(T.is_contiguous(), name, " must be contiguous");
}

void check_i32_cpu(at::Tensor T, const char* name) {
    TORCH_CHECK(T.device().is_cpu(), name, " must be CPU");
    TORCH_CHECK(T.scalar_type() == torch::kInt32, name, " must be int32");
    TORCH_CHECK(T.is_contiguous(), name, " must be contiguous");
}

size_t tensor_bytes(at::Tensor T) {
    return static_cast<size_t>(T.numel()) * static_cast<size_t>(T.element_size());
}

oo_tuning_mode_t default_oo_tuning_mode() {
    return OO_TUNING_BEST_PERFORMANCE;
}
} // namespace

OverlapImpl::OverlapImpl()
    : oo_group_(nullptr),
      oo_node_(nullptr),
      oo_local_buf_(nullptr),
      oo_peer_bufs_{nullptr},
      oo_peer_count_(0),
      oo_registered_ptr_(nullptr),
      oo_registered_bytes_(0),
      oo_rank_(-1),
      oo_size_(0),
      oo_devices_{-1, -1},
      oo_initialized_(false),
      gemm_stream_(nullptr),
      comm_stream_(nullptr),
      mm_ready_(nullptr),
      gemm_finished_(nullptr),
      comm_(nullptr),
      my_rank_(0),
      my_size_(1),
      overlap_init_done_(false) {}

OverlapImpl::~OverlapImpl() {
    OoverlapRelease();

    if (gemm_finished_ != nullptr) cudaEventDestroy(gemm_finished_);
    if (mm_ready_ != nullptr) cudaEventDestroy(mm_ready_);
    if (comm_stream_ != nullptr) cudaStreamDestroy(comm_stream_);
    if (comm_ != nullptr) ncclCommDestroy(comm_);
}

void OverlapImpl::CutlassInit() {
    gemm_stream_ = at::cuda::getCurrentCUDAStream().stream();
}

void OverlapImpl::NcclInit(
    const int64_t tp_rank,
    const int64_t tp_size,
    const std::vector<int64_t> tp_id) {

    my_rank_ = tp_rank;
    my_size_ = tp_size;

    if (comm_ != nullptr) {
        ncclCommDestroy(comm_);
        comm_ = nullptr;
    }

    if (my_size_ == 1) {
        return;
    }

    ncclUniqueId uid;
    std::memcpy(uid.internal, tp_id.data(), NCCL_UNIQUE_ID_BYTES);
    NCCL_CHECK(ncclCommInitRank(&comm_, my_size_, uid, my_rank_));
}

void OverlapImpl::OoverlapIpcInit(
    const int64_t tp_rank,
    const int64_t tp_size,
    const std::vector<int64_t> devices,
    const std::string broker_key) {

    TORCH_CHECK(tp_size == 2, "ooverlap IPC currently supports exactly 2 ranks");
    TORCH_CHECK(devices.size() == 2, "devices must contain exactly 2 CUDA device ids");

    OoverlapRelease();

    oo_rank_ = tp_rank;
    oo_size_ = tp_size;
    oo_devices_[0] = static_cast<int>(devices[0]);
    oo_devices_[1] = static_cast<int>(devices[1]);

    int devs[2] = {oo_devices_[0], oo_devices_[1]};
    CUDA_CHECK(cudaSetDevice(oo_devices_[oo_rank_]), "cudaSetDevice");

    OO_CHECK(
        oo_group_create_ipc(devs, 2, static_cast<int>(oo_rank_), broker_key.c_str(), &oo_group_),
        "oo_group_create_ipc");

    OO_CHECK(
        oo_node_create(oo_group_, static_cast<int>(oo_rank_), &oo_node_),
        "oo_node_create");

    oo_initialized_ = true;
}

void OverlapImpl::OoverlapUnregisterBuffer() {
    if (oo_group_ != nullptr) {
        oo_group_sync(oo_group_);
    }

    for (int i = 0; i < oo_peer_count_; ++i) {
        if (oo_peer_bufs_[i] != nullptr) {
            oo_buffer_destroy(oo_peer_bufs_[i]);
            oo_peer_bufs_[i] = nullptr;
        }
    }
    oo_peer_count_ = 0;

    if (oo_local_buf_ != nullptr) {
        oo_buffer_destroy(oo_local_buf_);
        oo_local_buf_ = nullptr;
    }

    oo_registered_ptr_ = nullptr;
    oo_registered_bytes_ = 0;

    if (oo_group_ != nullptr) {
        oo_group_sync(oo_group_);
    }
}

void OverlapImpl::OoverlapRelease() {
    OoverlapUnregisterBuffer();

    if (oo_node_ != nullptr) {
        oo_node_destroy(oo_node_);
        oo_node_ = nullptr;
    }

    if (oo_group_ != nullptr) {
        oo_group_destroy(oo_group_);
        oo_group_ = nullptr;
    }

    oo_rank_ = -1;
    oo_size_ = 0;
    oo_devices_[0] = -1;
    oo_devices_[1] = -1;
    oo_initialized_ = false;
}

void OverlapImpl::OoverlapEnsureBuffer(at::Tensor C) {
    TORCH_CHECK(oo_initialized_, "Call ooverlap_ipc_init first");
    check_half_cuda(C, "C");

    void* c_ptr = C.data_ptr<at::Half>();
    size_t bytes = tensor_bytes(C);

    if (oo_registered_ptr_ == c_ptr &&
        oo_registered_bytes_ == bytes &&
        oo_local_buf_ != nullptr &&
        oo_peer_count_ == 1 &&
        oo_peer_bufs_[0] != nullptr) {
        return;
    }

    OoverlapUnregisterBuffer();

    CUDA_CHECK(cudaSetDevice(oo_devices_[oo_rank_]), "cudaSetDevice");
    OO_CHECK(oo_buffer_wrap(oo_node_, c_ptr, bytes, &oo_local_buf_), "oo_buffer_wrap");

    OO_CHECK(
        oo_buffer_exchange_ipc_peers(oo_node_, oo_local_buf_, oo_peer_bufs_, &oo_peer_count_),
        "oo_buffer_exchange_ipc_peers");

    TORCH_CHECK(oo_peer_count_ == 1, "expected exactly one ooverlap peer");

    oo_registered_ptr_ = c_ptr;
    oo_registered_bytes_ = bytes;
}

void OverlapImpl::OoverlapAllReduceSlice(
    size_t element_offset,
    size_t count,
    cudaStream_t stream) {

    if (count == 0) {
        return;
    }

    TORCH_CHECK(oo_local_buf_ != nullptr && oo_peer_count_ == 1, "ooverlap buffer is not registered");

    OO_CHECK(
        oo_allreduce_offset_tuned(
            oo_node_,
            oo_local_buf_,
            oo_peer_bufs_,
            oo_peer_count_,
            element_offset,
            count,
            OO_DTYPE_FLOAT16,
            OO_REDUCE_SUM,
            default_oo_tuning_mode(),
            stream),
        "oo_allreduce_offset_tuned");
}

void OverlapImpl::OoverlapAllReduce(at::Tensor C) {
    check_half_cuda(C, "C");
    CutlassInit();
    OoverlapEnsureBuffer(C);

    OO_CHECK(
        oo_allreduce_tuned(
            oo_node_,
            oo_local_buf_,
            oo_peer_bufs_,
            oo_peer_count_,
            static_cast<size_t>(C.numel()),
            OO_DTYPE_FLOAT16,
            OO_REDUCE_SUM,
            default_oo_tuning_mode(),
            gemm_stream_),
        "oo_allreduce_tuned");
}

void OverlapImpl::OverlapInit() {
    if (overlap_init_done_) {
        return;
    }

    CUDA_CHECK(
        cudaStreamCreateWithPriority(&comm_stream_, cudaStreamNonBlocking, -5),
        "cudaStreamCreateWithPriority");

    CUDA_CHECK(
        cudaEventCreateWithFlags(&gemm_finished_, cudaEventDisableTiming),
        "cudaEventCreateWithFlags(gemm_finished)");

    CUDA_CHECK(
        cudaEventCreateWithFlags(&mm_ready_, cudaEventDisableTiming),
        "cudaEventCreateWithFlags(mm_ready)");

    overlap_init_done_ = true;
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
    int64_t active_sm_count,
    bool if_monitor) {

    check_half_cuda(A, "A");
    check_half_cuda(B, "B");
    check_half_cuda(C, "C");
    check_i32_cuda(MM, "MM");
    check_i32_cuda(RA, "RA");
    check_i32_cpu(cSEG_CPU, "cSEG_CPU");
    check_i32_cuda(cSEG_GPU, "cSEG_GPU");

    TORCH_CHECK(rLDN > 0, "rLDN must be > 0");

    ooverlap::GemmSignalSm90AlgoMeta meta{};
    TORCH_CHECK(
        ooverlap::gemm_signal_sm90_get_algo_meta(static_cast<int>(Algo), &meta),
        "Unsupported signal GEMM algo=", Algo);

    const int M = static_cast<int>(A.size(0));
    const int K = static_cast<int>(A.size(1));
    const int N = static_cast<int>(B.size(0));
    const int tile_m = meta.tile_m;
    const int tile_n = meta.tile_n;

    TORCH_CHECK(B.size(1) == K, "B must have shape [N, K]");
    TORCH_CHECK(M % tile_m == 0 && N % tile_n == 0, "M/N must be compatible with selected algo");

    const int tile_num = (M / tile_m) * (N / tile_n);
    const int seg_size = static_cast<int>(cSEG_GPU.numel());

    TORCH_CHECK(RA.numel() == tile_num, "RA.numel() must equal tile count");
    TORCH_CHECK(cSEG_CPU.numel() == seg_size && seg_size > 0, "bad cSEG size");
    TORCH_CHECK(MM.numel() >= static_cast<int64_t>(seg_size) + tile_num, "MM is too small");

    int* cseg_cpu_ptr = cSEG_CPU.data_ptr<int>();
    int seg_sum = 0;
    for (int i = 0; i < seg_size; ++i) {
        seg_sum += cseg_cpu_ptr[i];
    }
    TORCH_CHECK(seg_sum == tile_num, "sum(cSEG) must equal tile count");

    const int64_t packed_rows = (static_cast<int64_t>(tile_num) + rLDN - 1) / rLDN;
    const int64_t required_c =
        packed_rows * static_cast<int64_t>(tile_m) * rLDN * static_cast<int64_t>(tile_n);
    TORCH_CHECK(C.numel() >= required_c, "C is too small for packed GEMM output");

    OverlapInit();

    gemm_stream_ = at::cuda::getCurrentCUDAStream(A.get_device()).stream();

    if (oo_initialized_) {
        OoverlapEnsureBuffer(C);
    }

    half* a_ptr = reinterpret_cast<half*>(A.data_ptr<at::Half>());
    half* b_ptr = reinterpret_cast<half*>(B.data_ptr<at::Half>());
    half* c_ptr = reinterpret_cast<half*>(C.data_ptr<at::Half>());
    int* mm_ptr = MM.data_ptr<int>();
    int* ra_ptr = RA.data_ptr<int>();
    int* cseg_gpu_ptr = cSEG_GPU.data_ptr<int>();

    CUDA_CHECK(cudaEventRecord(mm_ready_, gemm_stream_), "cudaEventRecord(mm_ready)");
    CUDA_CHECK(cudaStreamWaitEvent(comm_stream_, mm_ready_, 0), "cudaStreamWaitEvent(mm_ready)");

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
        static_cast<int>(active_sm_count),
        if_monitor,
        gemm_stream_);

    TORCH_CHECK(ok, "gemm_signal_sm90_dispatch failed");

    if ((my_size_ == 1 || comm_ == nullptr) && !oo_initialized_) {
        return;
    }

    const int64_t elems_per_tile =
        static_cast<int64_t>(tile_m) * static_cast<int64_t>(tile_n);

    int64_t acc_addr = 0;

    for (int i = 0; i < seg_size; ++i) {
        const int this_seg = cseg_cpu_ptr[i];
        const int64_t comm_elems = elems_per_tile * static_cast<int64_t>(this_seg);

        kernel_wait_flag<<<1, 1, 0, comm_stream_>>>(this_seg, mm_ptr + i);
        CUDA_CHECK(cudaGetLastError(), "kernel_wait_flag");

        if (oo_initialized_) {
            OoverlapAllReduceSlice(
                static_cast<size_t>(acc_addr),
                static_cast<size_t>(comm_elems),
                comm_stream_);
        } else {
            NCCL_CHECK(ncclAllReduce(
                static_cast<void*>(c_ptr + acc_addr),
                static_cast<void*>(c_ptr + acc_addr),
                static_cast<size_t>(comm_elems),
                ncclFloat16,
                ncclSum,
                comm_,
                comm_stream_));
        }

        acc_addr += comm_elems;
    }

    CUDA_CHECK(cudaEventRecord(gemm_finished_, comm_stream_), "cudaEventRecord(gemm_finished)");
    CUDA_CHECK(cudaStreamWaitEvent(gemm_stream_, gemm_finished_, 0), "cudaStreamWaitEvent(gemm_finished)");
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

    check_half_cuda(A, "A");
    check_half_cuda(B, "B");
    check_half_cuda(C, "C");
    check_half_cuda(D, "D");
    check_i32_cuda(MM, "MM");
    check_i32_cuda(RA, "RA");
    check_i32_cuda(RE, "RE");
    check_i32_cpu(cSEG_CPU, "cSEG_CPU");
    check_i32_cuda(cSEG_GPU, "cSEG_GPU");

    TORCH_CHECK(rLDN > 0, "rLDN must be > 0");

    const int M = static_cast<int>(A.size(0));
    const int K = static_cast<int>(A.size(1));
    const int N = static_cast<int>(B.size(0));

    TORCH_CHECK(B.size(1) == K, "B must have shape [N, K]");
    TORCH_CHECK(M % kTileM == 0 && N % kTileN == 0, "M/N must be multiples of 128");

    const int tile_num = (M / kTileM) * (N / kTileN);
    const int seg_size = static_cast<int>(cSEG_GPU.numel());

    TORCH_CHECK(RA.numel() == tile_num, "RA.numel() must equal tile count");
    TORCH_CHECK(cSEG_CPU.numel() == seg_size && seg_size > 0, "bad cSEG size");
    TORCH_CHECK(MM.numel() >= static_cast<int64_t>(seg_size) + tile_num, "MM is too small");

    OverlapInit();
    gemm_stream_ = at::cuda::getCurrentCUDAStream(A.get_device()).stream();

    half* a_ptr = reinterpret_cast<half*>(A.data_ptr<at::Half>());
    half* b_ptr = reinterpret_cast<half*>(B.data_ptr<at::Half>());
    half* c_ptr = reinterpret_cast<half*>(C.data_ptr<at::Half>());
    half* d_ptr = reinterpret_cast<half*>(D.data_ptr<at::Half>());
    int* mm_ptr = MM.data_ptr<int>();
    int* ra_ptr = RA.data_ptr<int>();
    int* re_ptr = RE.data_ptr<int>();
    int* cseg_cpu_ptr = cSEG_CPU.data_ptr<int>();
    int* cseg_gpu_ptr = cSEG_GPU.data_ptr<int>();

    bool ok = ooverlap::gemm_scatter_sm90_dispatch(
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
        reinterpret_cast<int32_t*>(re_ptr),
        if_monitor,
        gemm_stream_);

    TORCH_CHECK(ok, "gemm_scatter_sm90_dispatch failed");

    if (my_size_ == 1 || comm_ == nullptr) {
        CUDA_CHECK(
            cudaMemcpyAsync(
                d_ptr,
                c_ptr,
                tensor_bytes(D),
                cudaMemcpyDeviceToDevice,
                gemm_stream_),
            "cudaMemcpyAsync");
        return;
    }

    at::Tensor TMP = at::empty_like(C);
    half* tmp_ptr = reinterpret_cast<half*>(TMP.data_ptr<at::Half>());

    int acc_addr = 0;

    for (int i = 0; i < seg_size; ++i) {
        const int this_seg = cseg_cpu_ptr[i];
        const int comm_size = (M * N / tile_num) * this_seg;
        const int row_begin = acc_addr / kTileN;
        const int row_count = this_seg * kTileM;

        kernel_wait_flag<<<1, 1, 0, comm_stream_>>>(this_seg, mm_ptr + i);
        CUDA_CHECK(cudaGetLastError(), "kernel_wait_flag");

        CUDA_CHECK(
            ooverlap::launch_scatter_row_remap_sm90(
                c_ptr + acc_addr,
                tmp_ptr + acc_addr,
                reinterpret_cast<const int32_t*>(re_ptr),
                row_begin,
                row_count,
                kTileN,
                comm_stream_),
            "launch_scatter_row_remap_sm90");

        NCCL_CHECK(ncclReduceScatter(
            static_cast<void*>(tmp_ptr + acc_addr),
            static_cast<void*>(d_ptr + acc_addr / my_size_),
            static_cast<size_t>(comm_size / my_size_),
            ncclFloat16,
            ncclSum,
            comm_,
            comm_stream_));

        acc_addr += comm_size;
    }

    CUDA_CHECK(cudaEventRecord(gemm_finished_, comm_stream_), "cudaEventRecord(gemm_finished)");
    CUDA_CHECK(cudaStreamWaitEvent(gemm_stream_, gemm_finished_, 0), "cudaStreamWaitEvent(gemm_finished)");
}
