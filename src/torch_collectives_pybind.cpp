#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <cuda_runtime.h>

#include "ooverlap/comm.h"

#include <algorithm>
#include <atomic>
#include <cctype>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <deque>
#include <exception>
#include <limits>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
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


/* OOVERLAP_ROUND_ROBIN_SLOT_POOL_PATCH_V1 */
static torch::ScalarType parse_round_robin_dtype(std::string dtype) {
  std::transform(dtype.begin(), dtype.end(), dtype.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });
  if (dtype == "fp16" || dtype == "f16" || dtype == "float16" || dtype == "half") {
    return torch::kFloat16;
  }
  if (dtype == "bf16" || dtype == "bfloat16") {
    return torch::kBFloat16;
  }
  if (dtype == "fp32" || dtype == "f32" || dtype == "float32" || dtype == "float") {
    return torch::kFloat32;
  }
  TORCH_CHECK(false, "unsupported round-robin dtype: ", dtype);
}

static size_t round_robin_dtype_size(torch::ScalarType dtype) {
  if (dtype == torch::kFloat16 || dtype == torch::kBFloat16) return 2;
  if (dtype == torch::kFloat32) return 4;
  TORCH_CHECK(false, "unsupported round-robin dtype");
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

  const auto logical_addr = reinterpret_cast<std::uintptr_t>(out.logical_ptr);
  const auto base_addr = reinterpret_cast<std::uintptr_t>(base_ptr);

  TORCH_CHECK(logical_addr >= base_addr,
              "tensor logical pointer is before allocation base");

  const size_t offset = static_cast<size_t>(logical_addr - base_addr);

  TORCH_CHECK(offset <= base_bytes,
              "tensor logical offset exceeds allocation size");

  TORCH_CHECK(out.logical_bytes <= base_bytes - offset,
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

static std::string trim_copy(std::string s) {
  auto not_space = [](unsigned char c) { return !std::isspace(c); };
  s.erase(s.begin(), std::find_if(s.begin(), s.end(), not_space));
  s.erase(std::find_if(s.rbegin(), s.rend(), not_space).base(), s.end());
  return s;
}

static std::string lower_copy(std::string s) {
  std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });
  return s;
}

static torch::ScalarType parse_dtype_name(std::string dtype) {
  dtype = lower_copy(trim_copy(std::move(dtype)));
  if (dtype == "fp16" || dtype == "f16" || dtype == "float16" || dtype == "half") {
    return torch::kFloat16;
  }
  if (dtype == "bf16" || dtype == "bfloat16") {
    return torch::kBFloat16;
  }
  if (dtype == "fp32" || dtype == "f32" || dtype == "float32" || dtype == "float") {
    return torch::kFloat32;
  }
  TORCH_CHECK(false, "unsupported loaned slot dtype: ", dtype);
}

static size_t dtype_size(torch::ScalarType dtype) {
  if (dtype == torch::kFloat16 || dtype == torch::kBFloat16) return 2;
  if (dtype == torch::kFloat32) return 4;
  TORCH_CHECK(false, "unsupported dtype");
}

struct ScratchEntry {
  torch::Tensor tensor;
  oo_buffer_t* buffer = nullptr;
  int device = -1;
  bool registered = false;
  size_t bytes = 0;
};

struct RegisteredTensorEntry {
  torch::Tensor tensor;
  OoBufferPtr buffer;
  int device = -1;
  torch::ScalarType dtype = torch::kFloat32;
  size_t bytes = 0;
  int64_t numel = 0;
};


struct RoundRobinSlot {
  torch::Tensor tensor;
  OoBufferPtr buffer;
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
    std::lock_guard<std::mutex> oo_lock(oo_mutex_);

    if (round_robin_set_ != nullptr) {
      oo_ipc_slot_set_destroy(round_robin_set_);
      round_robin_set_ = nullptr;
    }
    round_robin_slots_.clear();
    round_robin_index_ = 0;
    round_robin_capacity_bytes_ = 0;

    if (all_gather_round_robin_set_ != nullptr) {
      oo_ipc_slot_set_destroy(all_gather_round_robin_set_);
      all_gather_round_robin_set_ = nullptr;
    }
    all_gather_round_robin_slots_.clear();
    all_gather_round_robin_index_ = 0;
    all_gather_round_robin_capacity_bytes_ = 0;

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
    registered_tensors_.clear();

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

    {
      std::lock_guard<std::mutex> lock(oo_mutex_);
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
    }

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

    OoBufferPtr buf;
    {
      std::lock_guard<std::mutex> lock(oo_mutex_);
      buf.reset(wrap_tensor(output, "oo_buffer_wrap_ipc_range(output)"));

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
    }

    return output;
  }

  torch::Tensor all_reduce_inplace(torch::Tensor tensor) {
    validate(tensor);

    c10::cuda::CUDAGuard guard(tensor.device());

    OoBufferPtr buf;
    {
      std::lock_guard<std::mutex> lock(oo_mutex_);
      buf.reset(wrap_tensor(tensor, "oo_buffer_wrap_ipc_range(inplace)"));

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
    }

    return tensor;
  }

  torch::Tensor all_reduce_inplace_cached(torch::Tensor tensor) {
    validate(tensor);

    c10::cuda::CUDAGuard guard(tensor.device());

    RegisteredTensorEntry& entry = get_registered_tensor_by_ptr(tensor, "inplace_cached");

    {
      std::lock_guard<std::mutex> lock(oo_mutex_);
      check_status(
        oo_allreduce_tuned(
          node_,
          entry.buffer.get(),
          tensor.numel(),
          to_oo_dtype(tensor.scalar_type()),
          OO_REDUCE_SUM,
          OO_TUNING_BEST_PERFORMANCE,
          current_stream_for(tensor)),
        "oo_allreduce_tuned");
    }

    return tensor;
  }



  void init_round_robin_slots(
      const std::string& dtype_name,
      int64_t capacity_bytes,
      int64_t slot_count) {
    TORCH_CHECK(round_robin_set_ == nullptr && round_robin_slots_.empty(),
                "round-robin pool is already initialized");
    TORCH_CHECK(capacity_bytes > 0, "capacity_bytes must be positive");
    TORCH_CHECK(slot_count > 0, "slot_count must be positive");

    const torch::ScalarType dtype = parse_round_robin_dtype(dtype_name);
    const size_t elem_size = round_robin_dtype_size(dtype);
    const size_t capacity = static_cast<size_t>(capacity_bytes);
    TORCH_CHECK(capacity % elem_size == 0,
                "capacity_bytes must be divisible by dtype size");

    const size_t capacity_elems = capacity / elem_size;
    TORCH_CHECK(capacity_elems > 0 &&
                    capacity_elems <=
                        static_cast<size_t>(std::numeric_limits<int64_t>::max()),
                "round-robin capacity is too large");

    c10::cuda::CUDAGuard guard(
        torch::Device(torch::kCUDA, devices_[local_rank_]));

    std::vector<RoundRobinSlot> slots;
    std::vector<oo_buffer_t*> local_buffers;
    slots.reserve(static_cast<size_t>(slot_count));
    local_buffers.reserve(static_cast<size_t>(slot_count));

    const auto options = torch::TensorOptions()
        .device(torch::Device(torch::kCUDA, devices_[local_rank_]))
        .dtype(dtype)
        .layout(torch::kStrided)
        .requires_grad(false);

    for (int64_t slot_index = 0; slot_index < slot_count; ++slot_index) {
      RoundRobinSlot slot;
      slot.tensor = torch::empty(
          {static_cast<int64_t>(capacity_elems)},
          options,
          torch::MemoryFormat::Contiguous);
      TORCH_CHECK(slot.tensor.defined() &&
                      slot.tensor.is_cuda() &&
                      slot.tensor.is_contiguous() &&
                      slot.tensor.nbytes() == capacity,
                  "failed to allocate round-robin slot");

      slot.buffer.reset(
          wrap_tensor(
              slot.tensor,
              "oo_buffer_wrap_ipc_range(round_robin_slot)"));
      local_buffers.push_back(slot.buffer.get());
      slots.push_back(std::move(slot));
    }

    oo_ipc_slot_set_t* set = nullptr;
    check_status(
        oo_ipc_slot_set_create(
            node_,
            local_buffers.data(),
            static_cast<int>(local_buffers.size()),
            &set),
        "oo_ipc_slot_set_create");

    round_robin_slots_ = std::move(slots);
    round_robin_set_ = set;
    round_robin_dtype_ = dtype;
    round_robin_capacity_bytes_ = capacity;
    round_robin_index_ = 0;
  }

  torch::Tensor all_reduce_round_robin(torch::Tensor input) {
    validate(input);
    TORCH_CHECK(round_robin_set_ != nullptr && !round_robin_slots_.empty(),
                "round-robin pool is not initialized");
    TORCH_CHECK(input.scalar_type() == round_robin_dtype_,
                "input dtype does not match round-robin pool dtype");
    TORCH_CHECK(input.nbytes() <= round_robin_capacity_bytes_,
                "input exceeds round-robin slot capacity: input_bytes=",
                input.nbytes(),
                " capacity_bytes=",
                round_robin_capacity_bytes_);

    c10::cuda::CUDAGuard guard(input.device());

    const size_t slot_index = round_robin_index_;
    RoundRobinSlot& slot = round_robin_slots_[slot_index];
    torch::Tensor view =
        slot.tensor
            .narrow(0, 0, input.numel())
            .view(input.sizes());

    view.copy_(input);

    check_status(
        oo_allreduce_slot_tuned(
            node_,
            round_robin_set_,
            static_cast<int>(slot_index),
            static_cast<size_t>(input.numel()),
            to_oo_dtype(input.scalar_type()),
            OO_REDUCE_SUM,
            OO_TUNING_BEST_PERFORMANCE,
            current_stream_for(view)),
        "oo_allreduce_slot_tuned");

    round_robin_index_ =
        (round_robin_index_ + 1) % round_robin_slots_.size();

    // Intentionally unsafe: this view aliases pool storage and is overwritten
    // when the round-robin index wraps after N later collective calls.
    return view;
  }

  size_t round_robin_slot_count() const {
    return round_robin_slots_.size();
  }

  size_t round_robin_next_slot() const {
    return round_robin_index_;
  }

  void init_all_gather_round_robin_slots(
      const std::string& dtype_name,
      int64_t capacity_bytes,
      int64_t slot_count) {
    TORCH_CHECK(
        all_gather_round_robin_set_ == nullptr &&
            all_gather_round_robin_slots_.empty(),
        "all-gather round-robin pool is already initialized");
    TORCH_CHECK(capacity_bytes > 0, "capacity_bytes must be positive");
    TORCH_CHECK(slot_count > 0, "slot_count must be positive");

    const torch::ScalarType dtype = parse_round_robin_dtype(dtype_name);
    const size_t elem_size = round_robin_dtype_size(dtype);
    const size_t capacity = static_cast<size_t>(capacity_bytes);
    TORCH_CHECK(capacity % elem_size == 0,
                "capacity_bytes must be divisible by dtype size");

    const size_t capacity_elems = capacity / elem_size;
    TORCH_CHECK(capacity_elems > 0 &&
                    capacity_elems <=
                        static_cast<size_t>(std::numeric_limits<int64_t>::max()),
                "all-gather round-robin capacity is too large");

    c10::cuda::CUDAGuard guard(
        torch::Device(torch::kCUDA, devices_[local_rank_]));

    std::vector<RoundRobinSlot> slots;
    std::vector<oo_buffer_t*> local_buffers;
    slots.reserve(static_cast<size_t>(slot_count));
    local_buffers.reserve(static_cast<size_t>(slot_count));

    const auto options = torch::TensorOptions()
        .device(torch::Device(torch::kCUDA, devices_[local_rank_]))
        .dtype(dtype)
        .layout(torch::kStrided)
        .requires_grad(false);

    for (int64_t slot_index = 0; slot_index < slot_count; ++slot_index) {
      RoundRobinSlot slot;
      slot.tensor = torch::empty(
          {static_cast<int64_t>(capacity_elems)},
          options,
          torch::MemoryFormat::Contiguous);
      TORCH_CHECK(slot.tensor.defined() &&
                      slot.tensor.is_cuda() &&
                      slot.tensor.is_contiguous() &&
                      slot.tensor.nbytes() == capacity,
                  "failed to allocate all-gather round-robin slot");

      slot.buffer.reset(
          wrap_tensor(
              slot.tensor,
              "oo_buffer_wrap_ipc_range(all_gather_round_robin_slot)"));
      local_buffers.push_back(slot.buffer.get());
      slots.push_back(std::move(slot));
    }

    oo_ipc_slot_set_t* set = nullptr;
    check_status(
        oo_ipc_slot_set_create(
            node_,
            local_buffers.data(),
            static_cast<int>(local_buffers.size()),
            &set),
        "oo_ipc_slot_set_create(all_gather_round_robin)");

    all_gather_round_robin_slots_ = std::move(slots);
    all_gather_round_robin_set_ = set;
    all_gather_round_robin_dtype_ = dtype;
    all_gather_round_robin_capacity_bytes_ = capacity;
    all_gather_round_robin_index_ = 0;
  }

  torch::Tensor all_gather_round_robin(torch::Tensor input) {
    validate(input);
    TORCH_CHECK(input.dim() > 0,
                "all_gather_round_robin requires at least one dimension");
    TORCH_CHECK(
        all_gather_round_robin_set_ != nullptr &&
            !all_gather_round_robin_slots_.empty(),
        "all-gather round-robin pool is not initialized");
    TORCH_CHECK(input.scalar_type() == all_gather_round_robin_dtype_,
                "input dtype does not match all-gather round-robin pool dtype");

    const size_t world_size = devices_.size();
    const size_t input_numel = static_cast<size_t>(input.numel());
    TORCH_CHECK(
        input_numel <= std::numeric_limits<size_t>::max() / world_size,
        "all-gather element count overflow");
    TORCH_CHECK(
        input.nbytes() <= std::numeric_limits<size_t>::max() / world_size,
        "all-gather byte count overflow");

    const size_t output_numel = input_numel * world_size;
    const size_t output_bytes = input.nbytes() * world_size;

    TORCH_CHECK(
        output_numel <=
            static_cast<size_t>(std::numeric_limits<int64_t>::max()),
        "all-gather output element count is too large");
    TORCH_CHECK(
        output_bytes <= all_gather_round_robin_capacity_bytes_,
        "all-gather output exceeds round-robin slot capacity: output_bytes=",
        output_bytes,
        " capacity_bytes=",
        all_gather_round_robin_capacity_bytes_);
    TORCH_CHECK(
        input.size(0) <=
            std::numeric_limits<int64_t>::max() /
                static_cast<int64_t>(world_size),
        "all-gather output dimension overflow");

    size_t local_offset = 0;
    size_t local_count = 0;
    check_status(
        oo_rank_partition(
            local_rank_,
            static_cast<int>(world_size),
            output_numel,
            &local_offset,
            &local_count),
        "oo_rank_partition(all_gather_round_robin)");
    TORCH_CHECK(local_count == input_numel,
                "all-gather local partition does not match input size");

    c10::cuda::CUDAGuard guard(input.device());

    const size_t slot_index = all_gather_round_robin_index_;
    RoundRobinSlot& slot = all_gather_round_robin_slots_[slot_index];

    std::vector<int64_t> output_sizes = input.sizes().vec();
    output_sizes[0] *= static_cast<int64_t>(world_size);

    torch::Tensor output =
        slot.tensor
            .narrow(
                0,
                0,
                static_cast<int64_t>(output_numel))
            .view(output_sizes);

    torch::Tensor local_view =
        slot.tensor
            .narrow(
                0,
                static_cast<int64_t>(local_offset),
                static_cast<int64_t>(local_count))
            .view(input.sizes());

    local_view.copy_(input);

    check_status(
        oo_all_gather_slot_tuned(
            node_,
            all_gather_round_robin_set_,
            static_cast<int>(slot_index),
            output_numel,
            to_oo_dtype(input.scalar_type()),
            OO_TUNING_BEST_PERFORMANCE,
            current_stream_for(output)),
        "oo_all_gather_slot_tuned");

    all_gather_round_robin_index_ =
        (all_gather_round_robin_index_ + 1) %
        all_gather_round_robin_slots_.size();

    // Intentionally unsafe: this view aliases pool storage and is overwritten
    // when the all-gather round-robin index wraps after N later calls.
    return output;
  }

  size_t all_gather_round_robin_slot_count() const {
    return all_gather_round_robin_slots_.size();
  }

  size_t all_gather_round_robin_next_slot() const {
    return all_gather_round_robin_index_;
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

  std::string registered_tensor_key(const torch::Tensor& t, const std::string& role) {
    std::ostringstream os;
    os << role
       << ":dtype=" << static_cast<int>(t.scalar_type())
       << ":device=" << t.get_device()
       << ":bytes=" << t.nbytes()
       << ":numel=" << t.numel()
       << ":ptr=" << reinterpret_cast<std::uintptr_t>(t.data_ptr())
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

    {
      std::lock_guard<std::mutex> lock(oo_mutex_);
      entry.buffer = wrap_tensor(entry.tensor, "oo_buffer_wrap_ipc_range(scratch)");

      check_status(
        oo_buffer_register_ipc(node_, entry.buffer),
        "oo_buffer_register_ipc(scratch)");
    }

    entry.registered = true;

    auto inserted = scratch_.emplace(key, std::move(entry));
    return inserted.first->second;
  }

  RegisteredTensorEntry& get_registered_tensor_by_ptr(torch::Tensor tensor, const std::string& role) {
    validate(tensor);
    c10::cuda::CUDAGuard guard(tensor.device());

    const std::string key = registered_tensor_key(tensor, role);
    auto it = registered_tensors_.find(key);
    if (it != registered_tensors_.end()) {
      return it->second;
    }

    RegisteredTensorEntry entry;
    entry.tensor = tensor;
    entry.device = tensor.get_device();
    entry.dtype = tensor.scalar_type();
    entry.bytes = tensor.nbytes();
    entry.numel = tensor.numel();

    {
      std::lock_guard<std::mutex> lock(oo_mutex_);
      entry.buffer.reset(wrap_tensor(tensor, "oo_buffer_wrap_ipc_range(registered_tensor)"));

      check_status(
        oo_buffer_register_ipc(node_, entry.buffer.get()),
        "oo_buffer_register_ipc(registered_tensor)");
    }

    auto inserted = registered_tensors_.emplace(key, std::move(entry));
    return inserted.first->second;
  }


 private:
  std::vector<int> devices_;
  int local_rank_;
  std::string broker_key_;

  oo_group_t* group_ = nullptr;
  oo_node_t* node_ = nullptr;

  std::unordered_map<std::string, ScratchEntry> scratch_;
  std::unordered_map<std::string, RegisteredTensorEntry> registered_tensors_;

  mutable std::mutex oo_mutex_;

  oo_ipc_slot_set_t* round_robin_set_ = nullptr;
  std::vector<RoundRobinSlot> round_robin_slots_;
  torch::ScalarType round_robin_dtype_ = torch::kBFloat16;
  size_t round_robin_capacity_bytes_ = 0;
  size_t round_robin_index_ = 0;

  oo_ipc_slot_set_t* all_gather_round_robin_set_ = nullptr;
  std::vector<RoundRobinSlot> all_gather_round_robin_slots_;
  torch::ScalarType all_gather_round_robin_dtype_ = torch::kBFloat16;
  size_t all_gather_round_robin_capacity_bytes_ = 0;
  size_t all_gather_round_robin_index_ = 0;
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
      .def("all_reduce_inplace_cached", &OoTorchCommunicator::all_reduce_inplace_cached)
      .def("init_round_robin_slots", &OoTorchCommunicator::init_round_robin_slots,
           py::arg("dtype"), py::arg("capacity_bytes"), py::arg("slot_count"))
      .def("all_reduce_round_robin", &OoTorchCommunicator::all_reduce_round_robin,
           py::arg("input"))
      .def("round_robin_slot_count", &OoTorchCommunicator::round_robin_slot_count)
      .def("round_robin_next_slot", &OoTorchCommunicator::round_robin_next_slot)
      .def("init_all_gather_round_robin_slots",
           &OoTorchCommunicator::init_all_gather_round_robin_slots,
           py::arg("dtype"), py::arg("capacity_bytes"), py::arg("slot_count"))
      .def("all_gather_round_robin",
           &OoTorchCommunicator::all_gather_round_robin,
           py::arg("input"))
      .def("all_gather_round_robin_slot_count",
           &OoTorchCommunicator::all_gather_round_robin_slot_count)
      .def("all_gather_round_robin_next_slot",
           &OoTorchCommunicator::all_gather_round_robin_next_slot)
      .def("destroy", &OoTorchCommunicator::destroy);
}
