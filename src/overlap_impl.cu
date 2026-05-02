#include "overlap_impl.h"

#include "nccl_utils.h"
#include "wait.cuh"
#include "overlap/gemm_signal_sm90_dispatch.h"
#include "overlap/gemm_scatter_sm90_dispatch.h"
#include "overlap/gemm_plain_sm90_dispatch.h"
#include "overlap/scatter_row_remap_sm90.cuh"
#include "ooverlap/torch/torch_utils.h"

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <cstring>
#include <cstdint>
#include <stdexcept>
#include <vector>

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

void OO_CHECK_OR_THROW(oo_status_t status, const char* what) {
    TORCH_CHECK(status == OO_SUCCESS, what, " failed with ", oo_status_to_string(status));
}

void CUDA_CHECK_OR_THROW(cudaError_t err, const char* what) {
    TORCH_CHECK(err == cudaSuccess, what, " failed with ", cudaGetErrorString(err));
}

void check_cuda_fp16_contiguous(at::Tensor T, const char* name) {
    TORCH_CHECK(T.is_cuda(), name, " must be CUDA");
    TORCH_CHECK(T.scalar_type() == torch::kFloat16, name, " must be float16");
    TORCH_CHECK(T.is_contiguous(), name, " must be contiguous");
}

void check_int32_cuda_contiguous(at::Tensor T, const char* name) {
    TORCH_CHECK(T.is_cuda(), name, " must be CUDA");
    TORCH_CHECK(T.scalar_type() == torch::kInt32, name, " must be int32");
    TORCH_CHECK(T.is_contiguous(), name, " must be contiguous");
}

void check_int32_cpu_contiguous(at::Tensor T, const char* name) {
    TORCH_CHECK(T.device().is_cpu(), name, " must be CPU");
    TORCH_CHECK(T.scalar_type() == torch::kInt32, name, " must be int32");
    TORCH_CHECK(T.is_contiguous(), name, " must be contiguous");
}

oo_tuning_mode_t default_oo_tuning_mode() {
    return OO_TUNING_BEST_PERFORMANCE;
}
} // namespace

OverlapImpl::OverlapImpl()
    : oo_group_(nullptr),
      oo_node_(nullptr),
      oo_local_buf_(nullptr),
      oo_peer_buf_(nullptr),
      oo_registered_ptr_(nullptr),
      oo_registered_bytes_(0),
      oo_rank_(-1),
      oo_size_(0),
      oo_devices_{-1, -1},
      oo_initialized_(false),
      gemm_stream_(nullptr),
      mm_ready_(nullptr),
      comm_stream_(nullptr),
      gemm_finished_(nullptr),
      comm_(nullptr),
      my_rank_(0),
      my_size_(1),
      overlap_init_done_(false) {}

OverlapImpl::~OverlapImpl() {
    OoverlapRelease();

    if (gemm_finished_ != nullptr) {
        cudaEventDestroy(gemm_finished_);
        gemm_finished_ = nullptr;
    }
    if (mm_ready_ != nullptr) {
        cudaEventDestroy(mm_ready_);
        mm_ready_ = nullptr;
    }
    if (comm_stream_ != nullptr) {
        cudaStreamDestroy(comm_stream_);
        comm_stream_ = nullptr;
    }
    if (comm_ != nullptr) {
        ncclCommDestroy(comm_);
        comm_ = nullptr;
    }
}

void OverlapImpl::CutlassInit() {
    ooverlap::torch_utils::refresh_gemm_stream(gemm_stream_);
}

void OverlapImpl::NcclInit(
    const int64_t tp_rank,
    const int64_t tp_size,
    const std::vector<int64_t> tp_id) {

    my_rank_ = tp_rank;
    my_size_ = tp_size;

    TORCH_CHECK(
        static_cast<int64_t>(tp_id.size() * sizeof(int64_t)) == NCCL_UNIQUE_ID_BYTES,
        "tp_id must contain exactly NCCL_UNIQUE_ID_BYTES bytes; got ",
        tp_id.size(), " int64 values (", tp_id.size() * sizeof(int64_t), " bytes)");

    if (comm_ != nullptr) {
        ncclCommDestroy(comm_);
        comm_ = nullptr;
    }

    ncclUniqueId uid;
    std::memcpy(uid.internal, tp_id.data(), NCCL_UNIQUE_ID_BYTES);

    if (my_size_ == 1) {
        comm_ = nullptr;
        return;
    }

    NCCL_CHECK(ncclCommInitRank(&comm_, my_size_, uid, my_rank_));
}

void OverlapImpl::OoverlapIpcInit(
    const int64_t tp_rank,
    const int64_t tp_size,
    const std::vector<int64_t> devices,
    const std::string broker_key) {

    TORCH_CHECK(tp_size == 2, "ooverlap IPC allreduce currently supports exactly 2 ranks");
    TORCH_CHECK(devices.size() == 2, "devices must contain exactly two CUDA device ids");
    TORCH_CHECK(tp_rank == 0 || tp_rank == 1, "tp_rank must be 0 or 1 for ooverlap IPC");
    TORCH_CHECK(!broker_key.empty(), "broker_key must be non-empty");

    OoverlapRelease();

    oo_rank_ = tp_rank;
    oo_size_ = tp_size;
    oo_devices_[0] = static_cast<int>(devices[0]);
    oo_devices_[1] = static_cast<int>(devices[1]);

    const int local_device = oo_devices_[oo_rank_];
    CUDA_CHECK_OR_THROW(cudaSetDevice(local_device), "cudaSetDevice(ooverlap local device)");

    int devs[2] = {oo_devices_[0], oo_devices_[1]};

    oo_group_t* new_group = nullptr;
    oo_node_t* new_node = nullptr;

    OO_CHECK_OR_THROW(
        oo_group_create_ipc(devs, 2, static_cast<int>(oo_rank_), broker_key.c_str(), &new_group),
        "oo_group_create_ipc");

    oo_status_t st = oo_node_create(new_group, static_cast<int>(oo_rank_), &new_node);
    if (st != OO_SUCCESS) {
        oo_group_destroy(new_group);
        OO_CHECK_OR_THROW(st, "oo_node_create");
    }

    oo_group_ = new_group;
    oo_node_ = new_node;
    oo_initialized_ = true;
}

void OverlapImpl::OoverlapUnregisterBuffer() {
    if (oo_group_ != nullptr) {
        try { oo_group_sync(oo_group_); } catch (...) {}
    }

    if (oo_peer_buf_ != nullptr) {
        oo_buffer_destroy(oo_peer_buf_);
        oo_peer_buf_ = nullptr;
    }

    if (oo_group_ != nullptr) {
        try { oo_group_sync(oo_group_); } catch (...) {}
    }

    if (oo_local_buf_ != nullptr) {
        oo_buffer_destroy(oo_local_buf_);
        oo_local_buf_ = nullptr;
    }

    if (oo_group_ != nullptr) {
        try { oo_group_sync(oo_group_); } catch (...) {}
    }

    oo_registered_ptr_ = nullptr;
    oo_registered_bytes_ = 0;
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
    TORCH_CHECK(oo_initialized_, "Call ooverlap_ipc_init before using ooverlap allreduce");
    TORCH_CHECK(oo_group_ != nullptr, "ooverlap group is null");
    TORCH_CHECK(oo_node_ != nullptr, "ooverlap node is null");

    check_cuda_fp16_contiguous(C, "C");

    const int expected_device = oo_devices_[oo_rank_];
    TORCH_CHECK(
        C.get_device() == expected_device,
        "C is on CUDA device ", C.get_device(),
        " but ooverlap local rank expects device ", expected_device);

    void* c_ptr = C.data_ptr<at::Half>();
    const size_t bytes = ooverlap::torch_utils::tensor_nbytes(C);
    TORCH_CHECK(bytes > 0, "C must be non-empty");

    if (oo_registered_ptr_ == c_ptr &&
        oo_registered_bytes_ == bytes &&
        oo_local_buf_ != nullptr &&
        oo_peer_buf_ != nullptr) {
        return;
    }

    OoverlapUnregisterBuffer();

    CUDA_CHECK_OR_THROW(cudaSetDevice(expected_device), "cudaSetDevice(before oo_buffer_wrap)");

    OO_CHECK_OR_THROW(
        oo_buffer_wrap(oo_node_, c_ptr, bytes, &oo_local_buf_),
        "oo_buffer_wrap(C)");

    OO_CHECK_OR_THROW(
        oo_buffer_exchange_ipc_peer(oo_node_, oo_local_buf_, &oo_peer_buf_),
        "oo_buffer_exchange_ipc_peer(C)");

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

    TORCH_CHECK(oo_local_buf_ != nullptr, "ooverlap local buffer is not registered");
    TORCH_CHECK(oo_peer_buf_ != nullptr, "ooverlap peer buffer is not registered");

    const size_t byte_offset = element_offset * sizeof(half);
    const size_t bytes = count * sizeof(half);

    TORCH_CHECK(
        byte_offset + bytes <= oo_registered_bytes_,
        "ooverlap allreduce slice is out of registered C buffer bounds");

    OO_CHECK_OR_THROW(
        oo_allreduce_offset_tuned(
            oo_node_,
            oo_local_buf_,
            oo_peer_buf_,
            element_offset,
            count,
            OO_DTYPE_FLOAT16,
            OO_REDUCE_SUM,
            default_oo_tuning_mode(),
            stream),
        "oo_allreduce_offset_tuned(slice)");
}

void OverlapImpl::OoverlapAllReduce(at::Tensor C) {
    check_cuda_fp16_contiguous(C, "C");
    ooverlap::torch_utils::refresh_gemm_stream(gemm_stream_);
    OoverlapEnsureBuffer(C);

    OO_CHECK_OR_THROW(
        oo_allreduce_tuned(
            oo_node_,
            oo_local_buf_,
            oo_peer_buf_,
            static_cast<size_t>(C.numel()),
            OO_DTYPE_FLOAT16,
            OO_REDUCE_SUM,
            default_oo_tuning_mode(),
            gemm_stream_),
        "oo_allreduce_tuned(C)");
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
                    "cudaEventCreateWithFlags mm_ready_ failed: ", cudaGetErrorString(err));

        overlap_init_done_ = true;
    }
}

void OverlapImpl::Gemm(
    at::Tensor A,
    at::Tensor B,
    at::Tensor C,
    int64_t Algo) {

    ooverlap::torch_utils::check_common_gemm_inputs(A, B);
    check_cuda_fp16_contiguous(C, "C");

    TORCH_CHECK(C.dim() == 2, "C must be 2D");

    const int M = static_cast<int>(A.size(0));
    const int K = static_cast<int>(A.size(1));
    const int N = static_cast<int>(B.size(0));

    TORCH_CHECK(static_cast<int>(B.size(1)) == K, "B must have shape [N, K]");
    TORCH_CHECK(static_cast<int>(C.size(0)) == N &&
                static_cast<int>(C.size(1)) == M,
                "C must have physical shape [N, M] for plain SM90 GEMM");

    ooverlap::GemmPlainSm90AlgoMeta meta{};
    bool meta_ok = ooverlap::gemm_plain_sm90_get_algo_meta(static_cast<int>(Algo), &meta);
    TORCH_CHECK(
        meta_ok,
        "Unsupported plain GEMM algo=", Algo,
        ". Generated SM90 plain algo count=", ooverlap::gemm_plain_sm90_algo_count());

    TORCH_CHECK(M % meta.tile_m == 0,
                "M=", M, " must be multiple of tile_m=", meta.tile_m,
                " for algo=", Algo);
    TORCH_CHECK(N % meta.tile_n == 0,
                "N=", N, " must be multiple of tile_n=", meta.tile_n,
                " for algo=", Algo);

    ooverlap::torch_utils::refresh_gemm_stream(gemm_stream_);

    bool ok = ooverlap::gemm_plain_sm90_dispatch(
        static_cast<int>(Algo),
        M, N, K,
        static_cast<void*>(A.data_ptr<at::Half>()),
        static_cast<void*>(B.data_ptr<at::Half>()),
        static_cast<void*>(C.data_ptr<at::Half>()),
        gemm_stream_);

    TORCH_CHECK(ok, "gemm_plain_sm90_dispatch failed for algo=", Algo);
}

void OverlapImpl::GemmAllReduce(
    at::Tensor A,
    at::Tensor B,
    at::Tensor C,
    int64_t Algo) {

    Gemm(A, B, C, Algo);

    if (my_size_ == 1 || comm_ == nullptr) {
        return;
    }

    half* c_ptr = reinterpret_cast<half*>(C.data_ptr<at::Half>());
    NCCL_CHECK(ncclAllReduce(
        static_cast<void*>(c_ptr),
        static_cast<void*>(c_ptr),
        static_cast<size_t>(C.numel()),
        ncclFloat16,
        ncclSum,
        comm_,
        gemm_stream_));
}

void OverlapImpl::NcclAllReduce(at::Tensor C) {
    check_cuda_fp16_contiguous(C, "C");

    if (my_size_ == 1 || comm_ == nullptr) {
        return;
    }

    ooverlap::torch_utils::refresh_gemm_stream(gemm_stream_);

    half* c_ptr = reinterpret_cast<half*>(C.data_ptr<at::Half>());
    NCCL_CHECK(ncclAllReduce(
        static_cast<void*>(c_ptr),
        static_cast<void*>(c_ptr),
        static_cast<size_t>(C.numel()),
        ncclFloat16,
        ncclSum,
        comm_,
        gemm_stream_));
}

void OverlapImpl::NcclReduceScatter(at::Tensor C, at::Tensor D) {
    check_cuda_fp16_contiguous(C, "C");
    check_cuda_fp16_contiguous(D, "D");

    ooverlap::torch_utils::refresh_gemm_stream(gemm_stream_);

    if (my_size_ == 1 || comm_ == nullptr) {
        TORCH_CHECK(D.numel() <= C.numel(), "D cannot be larger than C");
        cudaError_t err = cudaMemcpyAsync(
            D.data_ptr<at::Half>(),
            C.data_ptr<at::Half>(),
            ooverlap::torch_utils::tensor_nbytes(D),
            cudaMemcpyDeviceToDevice,
            gemm_stream_);
        TORCH_CHECK(err == cudaSuccess, "cudaMemcpyAsync failed: ", cudaGetErrorString(err));
        return;
    }

    TORCH_CHECK(C.numel() % my_size_ == 0,
                "C.numel() must be divisible by world size for reduce_scatter");
    TORCH_CHECK(D.numel() == C.numel() / my_size_,
                "D.numel() must be C.numel()/world_size");

    half* c_ptr = reinterpret_cast<half*>(C.data_ptr<at::Half>());
    half* d_ptr = reinterpret_cast<half*>(D.data_ptr<at::Half>());

    NCCL_CHECK(ncclReduceScatter(
        static_cast<void*>(c_ptr),
        static_cast<void*>(d_ptr),
        static_cast<size_t>(D.numel()),
        ncclFloat16,
        ncclSum,
        comm_,
        gemm_stream_));
}

void OverlapImpl::SegAllReduce(
    at::Tensor C,
    at::Tensor cSEG_CPU,
    int64_t SegNum) {

    check_cuda_fp16_contiguous(C, "C");
    check_int32_cpu_contiguous(cSEG_CPU, "cSEG_CPU");

    TORCH_CHECK(SegNum > 0, "SegNum must be > 0");
    TORCH_CHECK(cSEG_CPU.numel() >= SegNum,
                "cSEG_CPU must have at least SegNum elements");

    if (my_size_ == 1 || comm_ == nullptr) {
        return;
    }

    ooverlap::torch_utils::refresh_gemm_stream(gemm_stream_);

    half* c_ptr = reinterpret_cast<half*>(C.data_ptr<at::Half>());
    int* cseg_cpu_ptr = cSEG_CPU.data_ptr<int>();

    TORCH_CHECK(C.numel() % SegNum == 0,
                "C.numel() must be divisible by SegNum");

    const int64_t elems_per_segment_unit = C.numel() / SegNum;
    int64_t acc_addr = 0;
    int64_t total_units = 0;

    for (int64_t s = 0; s < SegNum; ++s) {
        const int units = cseg_cpu_ptr[s];
        TORCH_CHECK(units > 0, "cSEG_CPU[", s, "] must be > 0");

        total_units += units;
        TORCH_CHECK(total_units <= SegNum, "sum(cSEG_CPU) exceeds SegNum");

        const int64_t comm_elems = elems_per_segment_unit * units;

        NCCL_CHECK(ncclAllReduce(
            static_cast<void*>(c_ptr + acc_addr),
            static_cast<void*>(c_ptr + acc_addr),
            static_cast<size_t>(comm_elems),
            ncclFloat16,
            ncclSum,
            comm_,
            gemm_stream_));

        acc_addr += comm_elems;
    }

    TORCH_CHECK(total_units == SegNum, "sum(cSEG_CPU) must equal SegNum");
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

    ooverlap::torch_utils::check_common_gemm_inputs(A, B);
    check_cuda_fp16_contiguous(C, "C");
    check_int32_cuda_contiguous(MM, "MM");
    check_int32_cuda_contiguous(RA, "RA");
    check_int32_cpu_contiguous(cSEG_CPU, "cSEG_CPU");
    check_int32_cuda_contiguous(cSEG_GPU, "cSEG_GPU");

    TORCH_CHECK(rLDN > 0, "rLDN must be > 0");

    ooverlap::GemmSignalSm90AlgoMeta meta{};
    bool meta_ok = ooverlap::gemm_signal_sm90_get_algo_meta(static_cast<int>(Algo), &meta);
    TORCH_CHECK(
        meta_ok,
        "Unsupported algo=", Algo,
        ". Generated SM90 signal algo count=", ooverlap::gemm_signal_sm90_algo_count());

    const int tile_m = meta.tile_m;
    const int tile_n = meta.tile_n;

    TORCH_CHECK(tile_m > 0 && tile_n > 0,
                "Invalid algo metadata for algo=", Algo,
                ": tile_m=", tile_m, " tile_n=", tile_n);

    ooverlap::torch_utils::ensure_streams_ready(gemm_stream_, comm_stream_, overlap_init_done_);

    if (gemm_finished_ == nullptr) {
        cudaError_t err = cudaEventCreateWithFlags(&gemm_finished_, cudaEventDisableTiming);
        TORCH_CHECK(err == cudaSuccess,
                    "cudaEventCreateWithFlags failed: ", cudaGetErrorString(err));
    }

    if (mm_ready_ == nullptr) {
        cudaError_t err = cudaEventCreateWithFlags(&mm_ready_, cudaEventDisableTiming);
        TORCH_CHECK(err == cudaSuccess,
                    "cudaEventCreateWithFlags mm_ready_ failed: ", cudaGetErrorString(err));
    }

    const int M = static_cast<int>(A.size(0));
    const int K = static_cast<int>(A.size(1));
    const int N = static_cast<int>(B.size(0));

    TORCH_CHECK(static_cast<int>(B.size(1)) == K, "B must have shape [N, K]");
    TORCH_CHECK(M % tile_m == 0,
                "M=", M, " must be multiple of tile_m=", tile_m, " for algo=", Algo);
    TORCH_CHECK(N % tile_n == 0,
                "N=", N, " must be multiple of tile_n=", tile_n, " for algo=", Algo);

    const int tile_rows = M / tile_m;
    const int tile_cols = N / tile_n;
    const int tile_num = tile_rows * tile_cols;

    TORCH_CHECK(static_cast<int>(RA.numel()) == tile_num,
                "RA.numel() must equal tile count. RA.numel()=", RA.numel(),
                " tile_num=", tile_num,
                " tile_m=", tile_m,
                " tile_n=", tile_n,
                " tile_rows=", tile_rows,
                " tile_cols=", tile_cols,
                " algo=", Algo);

    const int seg_size = static_cast<int>(cSEG_GPU.numel());
    TORCH_CHECK(seg_size > 0, "cSEG must contain at least one segment");
    TORCH_CHECK(seg_size == static_cast<int>(cSEG_CPU.numel()),
                "cSEG_CPU/GPU size mismatch");

    TORCH_CHECK(MM.numel() >= static_cast<int64_t>(seg_size) + tile_num,
                "MM must have at least num_segments + num_tiles elements. MM.numel()=",
                MM.numel(), " num_segments=", seg_size, " num_tiles=", tile_num);

    if (if_monitor) {
        TORCH_CHECK(MM.numel() >= static_cast<int64_t>(seg_size) + tile_num + 1 + tile_num,
                    "When if_monitor=true, MM must have at least "
                    "num_segments + num_tiles + 1 + num_tiles elements. MM.numel()=",
                    MM.numel(),
                    " num_segments=", seg_size,
                    " num_tiles=", tile_num,
                    " required=", static_cast<int64_t>(seg_size) + tile_num + 1 + tile_num);
    }

    auto* cseg_cpu_ptr = cSEG_CPU.data_ptr<int>();

    int64_t total_segment_tiles = 0;
    for (int i = 0; i < seg_size; ++i) {
        const int this_seg = cseg_cpu_ptr[i];
        TORCH_CHECK(this_seg > 0, "cSEG[", i, "] must be > 0, got ", this_seg);

        total_segment_tiles += static_cast<int64_t>(this_seg);
        TORCH_CHECK(total_segment_tiles <= tile_num,
                    "Sum of cSEG exceeds tile_num. partial_sum=",
                    total_segment_tiles, " tile_num=", tile_num);
    }

    TORCH_CHECK(total_segment_tiles == tile_num,
                "Sum of cSEG must equal tile_num. sum=",
                total_segment_tiles, " tile_num=", tile_num);

    const int64_t packed_tile_cols = rLDN;
    const int64_t packed_tile_rows =
        (static_cast<int64_t>(tile_num) + packed_tile_cols - 1) / packed_tile_cols;

    const int64_t required_c_elems =
        packed_tile_rows *
        static_cast<int64_t>(tile_m) *
        packed_tile_cols *
        static_cast<int64_t>(tile_n);

    TORCH_CHECK(C.numel() >= required_c_elems,
                "C is too small for packed output. C.numel()=", C.numel(),
                " required=", required_c_elems,
                " packed_tile_rows=", packed_tile_rows,
                " packed_tile_cols=", packed_tile_cols,
                " tile_m=", tile_m,
                " tile_n=", tile_n,
                " algo=", Algo);

    auto* a_ptr = reinterpret_cast<half*>(A.data_ptr<at::Half>());
    auto* b_ptr = reinterpret_cast<half*>(B.data_ptr<at::Half>());
    auto* c_ptr = reinterpret_cast<half*>(C.data_ptr<at::Half>());
    auto* mm_ptr = MM.data_ptr<int>();
    auto* ra_ptr = RA.data_ptr<int>();
    auto* cseg_gpu_ptr = cSEG_GPU.data_ptr<int>();

    TORCH_CHECK(active_sm_count >= 0,
                "active_sm_count must be >= 0, got ", active_sm_count);

    const bool use_ooverlap = oo_initialized_;
    if (use_ooverlap) {
        OoverlapEnsureBuffer(C);
    }

    cudaError_t err = cudaEventRecord(mm_ready_, gemm_stream_);
    TORCH_CHECK(err == cudaSuccess,
                "cudaEventRecord mm_ready_ failed: ", cudaGetErrorString(err));

    err = cudaStreamWaitEvent(comm_stream_, mm_ready_, 0);
    TORCH_CHECK(err == cudaSuccess,
                "cudaStreamWaitEvent mm_ready_ failed: ", cudaGetErrorString(err));

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

    TORCH_CHECK(ok, "Unsupported algo=", Algo);

    if ((my_size_ == 1 || comm_ == nullptr) && !use_ooverlap) {
        return;
    }

    const int64_t elems_per_tile =
        static_cast<int64_t>(tile_m) * static_cast<int64_t>(tile_n);

    int64_t acc_addr = 0;

    for (int iter = 0; iter < seg_size; ++iter) {
        const int this_seg = cseg_cpu_ptr[iter];
        const int64_t comm_elems =
            elems_per_tile * static_cast<int64_t>(this_seg);

        kernel_wait_flag<<<1, 1, 0, comm_stream_>>>(
            this_seg,
            mm_ptr + iter);

        err = cudaGetLastError();
        TORCH_CHECK(err == cudaSuccess,
                    "kernel_wait_flag launch failed: ", cudaGetErrorString(err));

        if (use_ooverlap) {
            OoverlapAllReduceSlice(
                static_cast<size_t>(acc_addr),
                static_cast<size_t>(comm_elems),
                comm_stream_);
        } else {
            TORCH_CHECK(comm_ != nullptr,
                        "NCCL communicator is null; call nccl_init or ooverlap_ipc_init first");

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

    err = cudaEventRecord(gemm_finished_, comm_stream_);
    TORCH_CHECK(err == cudaSuccess,
                "cudaEventRecord gemm_finished_ failed: ", cudaGetErrorString(err));

    err = cudaStreamWaitEvent(gemm_stream_, gemm_finished_, 0);
    TORCH_CHECK(err == cudaSuccess,
                "cudaStreamWaitEvent gemm_finished_ failed: ", cudaGetErrorString(err));
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

    TORCH_CHECK(MM.numel() >= static_cast<int64_t>(seg_size) + tile_num,
                "MM must have at least num_segments + num_tiles elements. MM.numel()=",
                MM.numel(),
                " num_segments=", seg_size,
                " num_tiles=", tile_num);

    if (if_monitor) {
        TORCH_CHECK(MM.numel() >= static_cast<int64_t>(seg_size) + tile_num + 1 + tile_num,
                    "When if_monitor=true, MM must have at least "
                    "num_segments + num_tiles + 1 + num_tiles elements. MM.numel()=",
                    MM.numel(),
                    " required=", static_cast<int64_t>(seg_size) + tile_num + 1 + tile_num);
    }

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
