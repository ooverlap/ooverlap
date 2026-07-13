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

struct LoanedSlotSpec {
  torch::ScalarType dtype = torch::kBFloat16;
  size_t capacity_bytes = 0;
  size_t count = 0;
};

struct LoanedSlot {
  torch::Tensor tensor;
  OoBufferPtr buffer;
  int device = -1;
  torch::ScalarType dtype = torch::kFloat32;
  size_t capacity_bytes = 0;
  size_t capacity_elems = 0;
};

struct LoanedReplenishRequest {
  torch::ScalarType dtype = torch::kBFloat16;
  size_t capacity_bytes = 0;
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
    stop_loaned_replenisher();

    std::lock_guard<std::mutex> oo_lock(oo_mutex_);
    std::lock_guard<std::mutex> loaned_lock(loaned_mutex_);

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
    loaned_ready_.clear();
    loaned_retired_buffers_.clear();
    loaned_specs_.clear();
    loaned_replenish_queue_.clear();
    loaned_replenisher_error_ = nullptr;

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

  void init_loaned_slots(const std::string& spec) {
    check_replenisher_error();
    std::vector<LoanedSlotSpec> specs = parse_loaned_slot_spec(spec);
    {
      std::lock_guard<std::mutex> lock(loaned_mutex_);
      loaned_specs_.insert(loaned_specs_.end(), specs.begin(), specs.end());
    }

    for (const LoanedSlotSpec& s : specs) {
      for (size_t i = 0; i < s.count; ++i) {
        LoanedSlot slot = make_loaned_slot(s.dtype, s.capacity_bytes);
        std::lock_guard<std::mutex> lock(loaned_mutex_);
        loaned_ready_.push_back(std::move(slot));
      }
    }
  }

  void init_loaned_slots_from_env() {
    const char* env = std::getenv("OOVERLAP_LOANED_SLOTS");
    TORCH_CHECK(env != nullptr && std::string(env).size() > 0,
                "OOVERLAP_LOANED_SLOTS is not set");
    init_loaned_slots(std::string(env));
  }

  void start_loaned_replenisher() {
    std::lock_guard<std::mutex> lock(loaned_mutex_);
    if (loaned_replenisher_started_) {
      loaned_replenish_enabled_ = true;
      return;
    }
    loaned_replenisher_stop_ = false;
    loaned_replenish_enabled_ = true;
    loaned_replenisher_error_ = nullptr;
    loaned_replenisher_started_ = true;
    loaned_replenisher_running_.store(true, std::memory_order_release);
    loaned_replenisher_ = std::thread(&OoTorchCommunicator::loaned_replenisher_loop, this);
  }

  void stop_loaned_replenisher() {
    bool should_join = false;
    {
      std::lock_guard<std::mutex> lock(loaned_mutex_);
      if (loaned_replenisher_started_) {
        loaned_replenisher_stop_ = true;
        loaned_replenish_enabled_ = false;
        should_join = true;
      }
    }
    loaned_cv_.notify_all();

    if (should_join && loaned_replenisher_.joinable()) {
      loaned_replenisher_.join();
    }

    {
      std::lock_guard<std::mutex> lock(loaned_mutex_);
      loaned_replenisher_started_ = false;
      loaned_replenisher_stop_ = false;
      loaned_replenisher_running_.store(false, std::memory_order_release);
    }
  }

  void set_loaned_replenish_enabled(bool enabled) {
    std::lock_guard<std::mutex> lock(loaned_mutex_);
    loaned_replenish_enabled_ = enabled;
    if (enabled) {
      loaned_cv_.notify_all();
    }
  }

  bool loaned_replenisher_running() const {
    return loaned_replenisher_running_.load(std::memory_order_acquire);
  }

  size_t loaned_replenish_queue_count() const {
    std::lock_guard<std::mutex> lock(loaned_mutex_);
    return loaned_replenish_queue_.size();
  }

  void replenish_loaned_slots_sync(int64_t max_items = -1) {
    check_replenisher_error();
    int64_t done = 0;
    while (max_items < 0 || done < max_items) {
      LoanedReplenishRequest req;
      {
        std::lock_guard<std::mutex> lock(loaned_mutex_);
        if (loaned_replenish_queue_.empty()) {
          break;
        }
        req = loaned_replenish_queue_.front();
        loaned_replenish_queue_.pop_front();
      }

      LoanedSlot slot = make_loaned_slot(req.dtype, req.capacity_bytes);
      {
        std::lock_guard<std::mutex> lock(loaned_mutex_);
        loaned_ready_.push_back(std::move(slot));
      }
      ++done;
    }
  }

  torch::Tensor all_reduce_loaned(torch::Tensor input, bool fallback_to_out = true) {
    check_replenisher_error();
    validate(input);
    c10::cuda::CUDAGuard guard(input.device());

    const int device = input.get_device();
    const torch::ScalarType dtype = input.scalar_type();
    const size_t bytes = input.nbytes();

    int best_idx = -1;
    size_t best_capacity = std::numeric_limits<size_t>::max();
    size_t replenish_capacity = 0;

    {
      std::lock_guard<std::mutex> lock(loaned_mutex_);
      replenish_capacity = choose_loaned_capacity_locked(dtype, bytes);

      for (size_t i = 0; i < loaned_ready_.size(); ++i) {
        const LoanedSlot& slot = loaned_ready_[i];
        if (!slot.tensor.defined() || slot.buffer == nullptr) {
          continue;
        }
        if (slot.device != device || slot.dtype != dtype) {
          continue;
        }
        if (slot.capacity_bytes < bytes) {
          continue;
        }
        if (slot.capacity_bytes < best_capacity) {
          best_capacity = slot.capacity_bytes;
          best_idx = static_cast<int>(i);
        }
      }
    }

    if (best_idx < 0) {
      torch::Tensor out;
      if (!fallback_to_out) {
        TORCH_CHECK(false,
                    "no loaned slot available for tensor: bytes=", bytes,
                    " dtype=", static_cast<int>(dtype),
                    " device=", device);
      }
      out = torch::empty_like(input);
      torch::Tensor result = all_reduce_out(input, out);
      if (replenish_capacity != 0) {
        enqueue_loaned_replenish(dtype, replenish_capacity);
      }
      return result;
    }

    LoanedSlot slot;
    {
      std::lock_guard<std::mutex> lock(loaned_mutex_);
      TORCH_CHECK(static_cast<size_t>(best_idx) < loaned_ready_.size(),
                  "loaned slot index changed unexpectedly");
      slot = std::move(loaned_ready_[static_cast<size_t>(best_idx)]);
      loaned_ready_.erase(loaned_ready_.begin() + best_idx);
    }

    TORCH_CHECK(slot.tensor.defined(), "loaned slot tensor is undefined");
    TORCH_CHECK(slot.buffer != nullptr, "loaned slot buffer is null");
    TORCH_CHECK(slot.capacity_elems >= static_cast<size_t>(input.numel()),
                "loaned slot capacity elements too small");

    torch::Tensor view =
        slot.tensor
            .narrow(0, 0, input.numel())
            .view(input.sizes());

    TORCH_CHECK(view.is_contiguous(), "loaned slot view must be contiguous");
    TORCH_CHECK(view.sizes() == input.sizes(), "loaned slot view shape mismatch");
    TORCH_CHECK(view.scalar_type() == input.scalar_type(), "loaned slot view dtype mismatch");
    TORCH_CHECK(view.get_device() == input.get_device(), "loaned slot view device mismatch");

    view.copy_(input);

    {
      std::lock_guard<std::mutex> lock(oo_mutex_);
      check_status(
        oo_allreduce_tuned(
          node_,
          slot.buffer.get(),
          input.numel(),
          to_oo_dtype(input.scalar_type()),
          OO_REDUCE_SUM,
          OO_TUNING_BEST_PERFORMANCE,
          current_stream_for(view)),
        "oo_allreduce_tuned(loaned)");
    }

    torch::Tensor result = view;
    const torch::ScalarType used_dtype = slot.dtype;
    const size_t used_capacity = slot.capacity_bytes;

    {
      std::lock_guard<std::mutex> lock(loaned_mutex_);
      loaned_retired_buffers_.push_back(std::move(slot.buffer));
    }
    slot.tensor = torch::Tensor();

    enqueue_loaned_replenish(used_dtype, used_capacity);

    return result;
  }

  size_t loaned_ready_count() const {
    std::lock_guard<std::mutex> lock(loaned_mutex_);
    return loaned_ready_.size();
  }

  size_t loaned_retired_count() const {
    std::lock_guard<std::mutex> lock(loaned_mutex_);
    return loaned_retired_buffers_.size();
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

  void check_replenisher_error() {
    std::exception_ptr err;
    {
      std::lock_guard<std::mutex> lock(loaned_mutex_);
      err = loaned_replenisher_error_;
    }
    if (err) {
      std::rethrow_exception(err);
    }
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

  std::vector<LoanedSlotSpec> parse_loaned_slot_spec(const std::string& spec) {
    std::vector<LoanedSlotSpec> out;
    std::stringstream ss(spec);
    std::string item;

    while (std::getline(ss, item, ',')) {
      item = trim_copy(item);
      if (item.empty()) {
        continue;
      }

      std::stringstream is(item);
      std::string dtype_s;
      std::string bytes_s;
      std::string count_s;

      TORCH_CHECK(std::getline(is, dtype_s, ':'), "bad loaned slot item: ", item);
      TORCH_CHECK(std::getline(is, bytes_s, ':'), "bad loaned slot item: ", item);
      TORCH_CHECK(std::getline(is, count_s, ':'), "bad loaned slot item: ", item);

      dtype_s = trim_copy(dtype_s);
      bytes_s = trim_copy(bytes_s);
      count_s = trim_copy(count_s);

      LoanedSlotSpec slot;
      slot.dtype = parse_dtype_name(dtype_s);

      try {
        slot.capacity_bytes = static_cast<size_t>(std::stoull(bytes_s));
        slot.count = static_cast<size_t>(std::stoull(count_s));
      } catch (const std::exception& e) {
        TORCH_CHECK(false, "bad loaned slot numeric field in item '", item, "': ", e.what());
      }

      const size_t elem_size = dtype_size(slot.dtype);
      TORCH_CHECK(slot.capacity_bytes > 0, "loaned slot capacity must be > 0: ", item);
      TORCH_CHECK(slot.capacity_bytes % elem_size == 0,
                  "loaned slot capacity bytes must be divisible by dtype size: ", item);
      TORCH_CHECK(slot.count > 0, "loaned slot count must be > 0: ", item);

      out.push_back(slot);
    }

    TORCH_CHECK(!out.empty(), "loaned slot spec is empty");
    return out;
  }

  size_t choose_loaned_capacity_locked(torch::ScalarType dtype, size_t bytes) const {
    size_t best = std::numeric_limits<size_t>::max();
    for (const LoanedSlotSpec& spec : loaned_specs_) {
      if (spec.dtype != dtype) {
        continue;
      }
      if (spec.capacity_bytes < bytes) {
        continue;
      }
      if (spec.capacity_bytes < best) {
        best = spec.capacity_bytes;
      }
    }
    return best == std::numeric_limits<size_t>::max() ? 0 : best;
  }

  void enqueue_loaned_replenish(torch::ScalarType dtype, size_t capacity_bytes) {
    std::lock_guard<std::mutex> lock(loaned_mutex_);
    if (!loaned_replenish_enabled_ || capacity_bytes == 0) {
      return;
    }
    loaned_replenish_queue_.push_back(LoanedReplenishRequest{dtype, capacity_bytes});
    loaned_cv_.notify_one();
  }

  LoanedSlot make_loaned_slot(torch::ScalarType dtype, size_t capacity_bytes) {
    const size_t elem_size = dtype_size(dtype);
    TORCH_CHECK(capacity_bytes % elem_size == 0,
                "loaned slot capacity bytes must be divisible by dtype size");
    const size_t capacity_elems = capacity_bytes / elem_size;
    TORCH_CHECK(capacity_elems > 0, "loaned slot capacity elements must be > 0");
    TORCH_CHECK(capacity_elems <= static_cast<size_t>(std::numeric_limits<int64_t>::max()),
                "loaned slot capacity too large");

    c10::cuda::CUDAGuard guard(
        torch::Device(torch::kCUDA, devices_[local_rank_]));

    LoanedSlot slot;
    slot.device = devices_[local_rank_];
    slot.dtype = dtype;
    slot.capacity_bytes = capacity_bytes;
    slot.capacity_elems = capacity_elems;

    auto options = torch::TensorOptions()
        .device(torch::Device(torch::kCUDA, slot.device))
        .dtype(dtype)
        .layout(torch::kStrided)
        .requires_grad(false);

    slot.tensor = torch::empty(
        {static_cast<int64_t>(capacity_elems)},
        options,
        torch::MemoryFormat::Contiguous);

    TORCH_CHECK(slot.tensor.defined(), "loaned slot tensor is undefined");
    TORCH_CHECK(slot.tensor.is_cuda(), "loaned slot tensor must be CUDA");
    TORCH_CHECK(slot.tensor.is_contiguous(), "loaned slot tensor must be contiguous");
    TORCH_CHECK(slot.tensor.nbytes() == capacity_bytes,
                "loaned slot tensor byte size mismatch");

    {
      std::lock_guard<std::mutex> lock(oo_mutex_);
      slot.buffer.reset(wrap_tensor(slot.tensor, "oo_buffer_wrap_ipc_range(loaned_slot)"));

      check_status(
        oo_buffer_register_ipc(node_, slot.buffer.get()),
        "oo_buffer_register_ipc(loaned_slot)");
    }

    return slot;
  }

  void loaned_replenisher_loop() {
    try {
      while (true) {
        LoanedReplenishRequest req;
        {
          std::unique_lock<std::mutex> lock(loaned_mutex_);
          loaned_cv_.wait(lock, [&] {
            return loaned_replenisher_stop_ || !loaned_replenish_queue_.empty();
          });

          if (loaned_replenisher_stop_ && loaned_replenish_queue_.empty()) {
            break;
          }

          req = loaned_replenish_queue_.front();
          loaned_replenish_queue_.pop_front();
        }

        LoanedSlot slot = make_loaned_slot(req.dtype, req.capacity_bytes);

        {
          std::lock_guard<std::mutex> lock(loaned_mutex_);
          loaned_ready_.push_back(std::move(slot));
        }
      }
    } catch (...) {
      std::lock_guard<std::mutex> lock(loaned_mutex_);
      loaned_replenisher_error_ = std::current_exception();
      loaned_replenisher_stop_ = true;
      loaned_replenish_enabled_ = false;
    }

    loaned_replenisher_running_.store(false, std::memory_order_release);
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
  mutable std::mutex loaned_mutex_;
  std::condition_variable loaned_cv_;
  std::vector<LoanedSlot> loaned_ready_;
  std::vector<OoBufferPtr> loaned_retired_buffers_;
  std::vector<LoanedSlotSpec> loaned_specs_;
  std::deque<LoanedReplenishRequest> loaned_replenish_queue_;
  std::thread loaned_replenisher_;
  bool loaned_replenisher_started_ = false;
  bool loaned_replenisher_stop_ = false;
  bool loaned_replenish_enabled_ = false;
  std::exception_ptr loaned_replenisher_error_ = nullptr;
  std::atomic<bool> loaned_replenisher_running_{false};
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
      .def("init_loaned_slots", &OoTorchCommunicator::init_loaned_slots,
           py::arg("spec"))
      .def("init_loaned_slots_from_env", &OoTorchCommunicator::init_loaned_slots_from_env)
      .def("start_loaned_replenisher", &OoTorchCommunicator::start_loaned_replenisher)
      .def("stop_loaned_replenisher", &OoTorchCommunicator::stop_loaned_replenisher)
      .def("set_loaned_replenish_enabled", &OoTorchCommunicator::set_loaned_replenish_enabled,
           py::arg("enabled"))
      .def("loaned_replenisher_running", &OoTorchCommunicator::loaned_replenisher_running)
      .def("loaned_replenish_queue_count", &OoTorchCommunicator::loaned_replenish_queue_count)
      .def("replenish_loaned_slots_sync", &OoTorchCommunicator::replenish_loaned_slots_sync,
           py::arg("max_items") = -1)
      .def("all_reduce_loaned", &OoTorchCommunicator::all_reduce_loaned,
           py::arg("input"), py::arg("fallback_to_out") = true)
      .def("loaned_ready_count", &OoTorchCommunicator::loaned_ready_count)
      .def("loaned_retired_count", &OoTorchCommunicator::loaned_retired_count)
      .def("destroy", &OoTorchCommunicator::destroy);
}
