#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <cuda_runtime.h>

#include "ooverlap/comm.h"

#include <cstdint>
#include <memory>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace py = pybind11;

static cudaStream_t current_stream_for(const torch::Tensor& t) {
  TORCH_CHECK(t.is_cuda(), "tensor must be CUDA");
  const int dev = t.get_device();
  c10::cuda::CUDAGuard guard(t.device());
  return at::cuda::getCurrentCUDAStream(dev).stream();
}

static oo_dtype_t to_oo_dtype(torch::ScalarType t) {
  if (t == torch::kFloat16) return OO_DTYPE_FLOAT16;
  if (t == torch::kBFloat16) return OO_DTYPE_BFLOAT16;
  if (t == torch::kFloat32) return OO_DTYPE_FLOAT32;
  TORCH_CHECK(false, "unsupported dtype");
}

static void check_status(oo_status_t st, const char* what) {
  TORCH_CHECK(st == OO_SUCCESS, what, " failed: ", oo_status_string(st));
}

struct OoBufferDeleter {
  void operator()(oo_buffer_t* b) const {
    if (b != nullptr) {
      oo_buffer_destroy(b);
    }
  }
};

using OoBufferPtr = std::unique_ptr<oo_buffer_t, OoBufferDeleter>;

struct TensorIpcRange {
  void* logical_ptr = nullptr;
  size_t logical_bytes = 0;
  void* base_ptr = nullptr;
  size_t base_bytes = 0;
  size_t logical_offset_bytes = 0;
  oo_buffer_ipc_range_t range{};
};

static TensorIpcRange tensor_ipc_range(const torch::Tensor& t) {
  TORCH_CHECK(t.defined(), "tensor must be defined");
  TORCH_CHECK(t.is_cuda(), "tensor must be CUDA");
  TORCH_CHECK(t.is_contiguous(), "tensor must be contiguous");
  TORCH_CHECK(t.numel() > 0, "tensor must be non-empty");

  c10::cuda::CUDAGuard guard(t.device());

  TensorIpcRange out{};
  out.logical_ptr = const_cast<void*>(t.data_ptr());
  out.logical_bytes = t.nbytes();

  TORCH_CHECK(out.logical_ptr != nullptr, "tensor data_ptr is null");
  TORCH_CHECK(out.logical_bytes > 0, "tensor byte size must be non-zero");

  /*
   * Normal PyTorch CUDA tensors are allocated by CUDACachingAllocator.  The
   * data_ptr can be a suballocation inside a larger cached block, so use
   * getBaseAllocation() to recover the exportable CUDA allocation base and
   * byte size.  This is exactly the metadata oo_buffer_wrap_ipc_range needs.
   *
   * If the pointer is not owned by CUDACachingAllocator, for example a
   * torch::from_blob view over raw cudaMalloc memory, getBaseAllocation throws.
   * In that case, fall back to treating logical_ptr as the allocation base.
   */
  size_t base_bytes = 0;
  void* base_ptr = nullptr;

  try {
    base_ptr =
        c10::cuda::CUDACachingAllocator::getBaseAllocation(
            out.logical_ptr,
            &base_bytes);
  } catch (const c10::Error&) {
    base_ptr = nullptr;
    base_bytes = 0;
  } catch (const std::exception&) {
    base_ptr = nullptr;
    base_bytes = 0;
  }

  if (base_ptr == nullptr || base_bytes == 0) {
    base_ptr = out.logical_ptr;
    base_bytes = out.logical_bytes;
  }

  const auto logical_addr =
      reinterpret_cast<std::uintptr_t>(out.logical_ptr);
  const auto base_addr =
      reinterpret_cast<std::uintptr_t>(base_ptr);

  TORCH_CHECK(
      logical_addr >= base_addr,
      "tensor logical pointer is before allocation base");

  const size_t offset =
      static_cast<size_t>(logical_addr - base_addr);

  TORCH_CHECK(
      offset <= base_bytes,
      "tensor logical offset exceeds allocation size");

  TORCH_CHECK(
      out.logical_bytes <= base_bytes - offset,
      "tensor logical range exceeds allocation base: logical_bytes=",
      out.logical_bytes,
      " base_bytes=",
      base_bytes,
      " offset=",
      offset);

  out.base_ptr = base_ptr;
  out.base_bytes = base_bytes;
  out.logical_offset_bytes = offset;

  out.range.allocation_base_ptr = out.base_ptr;
  out.range.allocation_bytes = out.base_bytes;
  out.range.logical_offset_bytes = out.logical_offset_bytes;

  return out;
}

struct ScratchEntry {
  /*
   * OOVERLAP_TORCH_OFFSET_AWARE_IPC_RANGE_SCRATCH
   *
   * The scratch tensor is a normal PyTorch CUDA tensor.  We register it through
   * oo_buffer_wrap_ipc_range(), using CUDACachingAllocator::getBaseAllocation()
   * to provide allocation-base + logical-offset metadata to ooverlap.
   *
   * This exercises the same path we eventually want for vLLM/Torch allocations,
   * while keeping the scratch tensor alive for the communicator lifetime.
   */
  torch::Tensor tensor;
  oo_buffer_t* buffer = nullptr;
  int device = -1;
  bool registered = false;
  size_t bytes = 0;
};

class OoTorchCommunicator {
 public:
  OoTorchCommunicator(std::vector<int> devices,
                      int local_rank,
                      std::string broker_key)
      : devices_(std::move(devices)),
        local_rank_(local_rank),
        broker_key_(std::move(broker_key)) {
    TORCH_CHECK(!devices_.empty(), "devices cannot be empty");
    TORCH_CHECK(local_rank_ >= 0 && local_rank_ < (int)devices_.size(),
                "invalid local_rank");

    c10::cuda::CUDAGuard guard(
        torch::Device(torch::kCUDA, devices_[local_rank_]));

    check_status(
      oo_group_create_ipc(
        devices_.data(),
        (int)devices_.size(),
        local_rank_,
        broker_key_.c_str(),
        &group_),
      "oo_group_create_ipc");

    check_status(
      oo_node_create(group_, local_rank_, &node_),
      "oo_node_create");
  }

  ~OoTorchCommunicator() {
    destroy();
  }

  void destroy() {
    for (auto& kv : scratch_) {
      ScratchEntry& entry = kv.second;

      if (entry.buffer != nullptr) {
        oo_buffer_destroy(entry.buffer);
        entry.buffer = nullptr;
      }

      entry.tensor = torch::Tensor();
      entry.bytes = 0;
      entry.device = -1;
      entry.registered = false;
    }

    scratch_.clear();

    if (node_ != nullptr) {
      oo_node_destroy(node_);
      node_ = nullptr;
    }
    if (group_ != nullptr) {
      oo_group_destroy(group_);
      group_ = nullptr;
    }
  }

  torch::Tensor all_reduce(torch::Tensor input) {
    validate(input);

    c10::cuda::CUDAGuard guard(input.device());

    ScratchEntry& entry = get_scratch(input, "allreduce");

    entry.tensor.copy_(input);

    check_status(
      oo_allreduce_tuned(
        node_,
        entry.buffer,
        input.numel(),
        to_oo_dtype(input.scalar_type()),
        OO_REDUCE_SUM,
        OO_TUNING_BEST_PERFORMANCE,
        current_stream_for(input)),
      "oo_allreduce_tuned");

    return entry.tensor;
  }

  torch::Tensor all_reduce_out(torch::Tensor input, torch::Tensor output) {
    validate(input);
    validate(output);

    TORCH_CHECK(output.sizes() == input.sizes(), "output shape must match input shape");
    TORCH_CHECK(output.scalar_type() == input.scalar_type(), "output dtype must match input dtype");
    TORCH_CHECK(output.get_device() == input.get_device(), "output device must match input device");

    c10::cuda::CUDAGuard guard(input.device());

    output.copy_(input);

    OoBufferPtr buf(wrap_tensor(output, "oo_buffer_wrap_ipc_range(output)"));

    check_status(
      oo_buffer_register_ipc(node_, buf.get()),
      "oo_buffer_register_ipc(output)");

    check_status(
      oo_allreduce_tuned(
        node_,
        buf.get(),
        output.numel(),
        to_oo_dtype(output.scalar_type()),
        OO_REDUCE_SUM,
        OO_TUNING_BEST_PERFORMANCE,
        current_stream_for(output)),
      "oo_allreduce_tuned");

    return output;
  }

  torch::Tensor all_reduce_inplace(torch::Tensor tensor) {
    validate(tensor);

    c10::cuda::CUDAGuard guard(tensor.device());

    OoBufferPtr buf(wrap_tensor(tensor, "oo_buffer_wrap_ipc_range(inplace)"));

    check_status(
      oo_buffer_register_ipc(node_, buf.get()),
      "oo_buffer_register_ipc(inplace)");

    check_status(
      oo_allreduce_tuned(
        node_,
        buf.get(),
        tensor.numel(),
        to_oo_dtype(tensor.scalar_type()),
        OO_REDUCE_SUM,
        OO_TUNING_BEST_PERFORMANCE,
        current_stream_for(tensor)),
      "oo_allreduce_tuned");

    return tensor;
  }

 private:
  void validate(const torch::Tensor& t) {
    TORCH_CHECK(t.defined(), "tensor must be defined");
    TORCH_CHECK(t.is_cuda(), "tensor must be CUDA");
    TORCH_CHECK(t.is_contiguous(), "tensor must be contiguous");
    TORCH_CHECK(t.numel() > 0, "tensor must be non-empty");
    TORCH_CHECK(
        t.get_device() == devices_[local_rank_],
        "tensor device does not match communicator local rank device: tensor device=",
        t.get_device(), " expected=", devices_[local_rank_]);
    TORCH_CHECK(
        t.scalar_type() == torch::kFloat16 ||
        t.scalar_type() == torch::kBFloat16 ||
        t.scalar_type() == torch::kFloat32,
        "unsupported dtype");
  }

  std::string scratch_key(const torch::Tensor& t, const std::string& role) {
    std::ostringstream os;
    os << role
       << ":dtype=" << static_cast<int>(t.scalar_type())
       << ":device=" << t.get_device()
       << ":bytes=" << t.nbytes()
       << ":shape=";

    for (int64_t s : t.sizes()) {
      os << s << "x";
    }

    return os.str();
  }

  oo_buffer_t* wrap_tensor(const torch::Tensor& tensor, const char* what) {
    TensorIpcRange ipc = tensor_ipc_range(tensor);

    oo_buffer_t* raw = nullptr;

    check_status(
      oo_buffer_wrap_ipc_range(
        node_,
        ipc.logical_ptr,
        ipc.logical_bytes,
        &ipc.range,
        &raw),
      what);

    return raw;
  }

  ScratchEntry& get_scratch(torch::Tensor input, const std::string& role) {
    validate(input);

    c10::cuda::CUDAGuard guard(input.device());

    const std::string key = scratch_key(input, role);
    auto it = scratch_.find(key);
    if (it != scratch_.end()) {
      return it->second;
    }

    ScratchEntry entry;
    entry.device = input.get_device();
    entry.bytes = input.nbytes();

    TORCH_CHECK(entry.bytes > 0, "scratch byte size must be non-zero");

    entry.tensor = torch::empty(
        input.sizes(),
        input.options()
             .device(input.device())
             .dtype(input.scalar_type())
             .layout(torch::kStrided)
             .requires_grad(false),
        torch::MemoryFormat::Contiguous);

    TORCH_CHECK(entry.tensor.defined(), "scratch tensor is undefined");
    TORCH_CHECK(entry.tensor.is_cuda(), "scratch tensor must be CUDA");
    TORCH_CHECK(entry.tensor.is_contiguous(), "scratch tensor must be contiguous");
    TORCH_CHECK(entry.tensor.numel() == input.numel(), "scratch numel mismatch");
    TORCH_CHECK(entry.tensor.nbytes() == input.nbytes(), "scratch byte size mismatch");

    entry.buffer = wrap_tensor(entry.tensor, "oo_buffer_wrap_ipc_range(scratch)");

    check_status(
      oo_buffer_register_ipc(node_, entry.buffer),
      "oo_buffer_register_ipc(scratch)");

    entry.registered = true;

    auto inserted = scratch_.emplace(key, std::move(entry));
    return inserted.first->second;
  }

 private:
  std::vector<int> devices_;
  int local_rank_;
  std::string broker_key_;

  oo_group_t* group_ = nullptr;
  oo_node_t* node_ = nullptr;

  std::unordered_map<std::string, ScratchEntry> scratch_;
};

PYBIND11_MODULE(ooverlap_torch_ext, m) {
  py::class_<OoTorchCommunicator>(m, "Communicator")
      .def(py::init<std::vector<int>, int, std::string>(),
           py::arg("devices"),
           py::arg("local_rank"),
           py::arg("broker_key"))
      .def("all_reduce", &OoTorchCommunicator::all_reduce)
      .def("all_reduce_out", &OoTorchCommunicator::all_reduce_out)
      .def("all_reduce_inplace", &OoTorchCommunicator::all_reduce_inplace)
      .def("destroy", &OoTorchCommunicator::destroy);
}
