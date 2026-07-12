#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAGuard.h>

#include "ooverlap/comm.h"

#include <unordered_map>
#include <vector>
#include <string>

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

struct ScratchEntry {
  torch::Tensor tensor;
  oo_buffer_t* buffer = nullptr;
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

    c10::cuda::CUDAGuard guard(torch::Device(torch::kCUDA, devices_[local_rank_]));

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
      if (kv.second.buffer) {
        oo_buffer_destroy(kv.second.buffer);
        kv.second.buffer = nullptr;
      }
    }
    scratch_.clear();

    if (node_) {
      oo_node_destroy(node_);
      node_ = nullptr;
    }
    if (group_) {
      oo_group_destroy(group_);
      group_ = nullptr;
    }
  }

  torch::Tensor all_reduce(torch::Tensor input) {
    validate(input);

    c10::cuda::CUDAGuard guard(input.device());

    auto& entry = get_scratch(input, "allreduce");

    entry.tensor.copy_(input);

    const size_t count = input.numel();
    const oo_dtype_t dtype = to_oo_dtype(input.scalar_type());
    cudaStream_t stream = current_stream_for(input);

    check_status(
      oo_allreduce_tuned(
        node_,
        entry.buffer,
        count,
        dtype,
        OO_REDUCE_SUM,
        OO_TUNING_BEST_PERFORMANCE,
        stream),
      "oo_allreduce_tuned");

    return entry.tensor;
  }

  torch::Tensor all_reduce_inplace(torch::Tensor tensor) {
    validate(tensor);

    // Direct path: wrap/register this exact tensor.
    // Good for stable buffers, not for arbitrary temporary tensors.
    void* ptr = tensor.data_ptr();
    const size_t bytes = tensor.nbytes();

    oo_buffer_t* buf = nullptr;
    check_status(oo_buffer_wrap(node_, ptr, bytes, &buf), "oo_buffer_wrap");
    check_status(oo_buffer_register_ipc(node_, buf), "oo_buffer_register_ipc");

    check_status(
      oo_allreduce_tuned(
        node_,
        buf,
        tensor.numel(),
        to_oo_dtype(tensor.scalar_type()),
        OO_REDUCE_SUM,
        OO_TUNING_BEST_PERFORMANCE,
        current_stream_for(tensor)),
      "oo_allreduce_tuned");

    oo_buffer_destroy(buf);
    return tensor;
  }

 private:
 
  void validate(const torch::Tensor& t) {
    TORCH_CHECK(t.defined(), "tensor must be defined");
    TORCH_CHECK(t.is_cuda(), "tensor must be CUDA");
    TORCH_CHECK(t.is_contiguous(), "tensor must be contiguous");
    TORCH_CHECK(t.numel() > 0, "tensor must be non-empty");
    TORCH_CHECK(t.get_device() == devices_[local_rank_],
                "tensor device does not match communicator local rank device: tensor device=",
                t.get_device(), " expected=", devices_[local_rank_]);
    TORCH_CHECK(t.scalar_type() == torch::kFloat16 ||
                t.scalar_type() == torch::kBFloat16 ||
                t.scalar_type() == torch::kFloat32,
                "unsupported dtype");
  }

  std::string scratch_key(const torch::Tensor& t, const std::string& role) {
    return role + ":" +
           std::to_string((int)t.scalar_type()) + ":" +
           std::to_string(t.nbytes()) + ":" +
           std::to_string(t.get_device());
  }

  ScratchEntry& get_scratch(torch::Tensor input, const std::string& role) {
   validate(input);
 
   c10::cuda::CUDAGuard guard(input.device());
 
   const std::string key = scratch_key(input, role);
   auto it = scratch_.find(key);
   if (it != scratch_.end()) return it->second;
 
   ScratchEntry entry;
 
   entry.tensor = torch::empty(
       input.sizes(),
       input.options()
            .device(input.device())
            .layout(torch::kStrided)
            .requires_grad(false),
       torch::MemoryFormat::Contiguous);
 
   TORCH_CHECK(entry.tensor.defined(), "scratch tensor is undefined");
   TORCH_CHECK(entry.tensor.is_cuda(), "scratch tensor must be CUDA");
   TORCH_CHECK(entry.tensor.is_contiguous(), "scratch tensor must be contiguous");
   TORCH_CHECK(entry.tensor.numel() == input.numel(), "scratch numel mismatch");
 
   entry.bytes = entry.tensor.nbytes();
   TORCH_CHECK(entry.bytes == input.nbytes(), "scratch byte size mismatch");
 
   void* scratch_ptr = entry.tensor.data_ptr();
   TORCH_CHECK(scratch_ptr != nullptr, "scratch data_ptr is null");
 
   check_status(
     oo_buffer_wrap(node_, scratch_ptr, entry.bytes, &entry.buffer),
     "oo_buffer_wrap");
 
   check_status(
     oo_buffer_register_ipc(node_, entry.buffer),
     "oo_buffer_register_ipc");
 
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
      .def("all_reduce_inplace", &OoTorchCommunicator::all_reduce_inplace)
      .def("destroy", &OoTorchCommunicator::destroy);
}
