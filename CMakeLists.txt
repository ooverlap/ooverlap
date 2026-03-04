cmake_minimum_required(VERSION 3.20)
project(ooverlap LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# --- Torch ---
find_package(Torch REQUIRED)
# Helps match PyTorch ABI flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# --- CUDA ---
find_package(CUDAToolkit REQUIRED)

# --- NCCL (simple finder) ---
# Prefer: export NCCL_ROOT=/path/to/nccl
find_path(NCCL_INCLUDE_DIR nccl.h
  HINTS ENV NCCL_ROOT
  PATH_SUFFIXES include)
find_library(NCCL_LIBRARY nccl
  HINTS ENV NCCL_ROOT
  PATH_SUFFIXES lib lib64)

if(NOT NCCL_INCLUDE_DIR OR NOT NCCL_LIBRARY)
  message(FATAL_ERROR "NCCL not found. Set NCCL_ROOT to your NCCL install.")
endif()

add_library(ooverlap_pybinding SHARED
  src/pybind.cpp
  src/baseline_impl.cu
  src/nccl_utils.cu
)

target_include_directories(ooverlap_pybinding PRIVATE
  ${NCCL_INCLUDE_DIR}
)

target_link_libraries(ooverlap_pybinding PRIVATE
  ${TORCH_LIBRARIES}
  CUDA::cudart
  CUDA::cublas
  ${NCCL_LIBRARY}
)

# Set your arch list (example: A100=80, H100=90)
# You can also pass -DCMAKE_CUDA_ARCHITECTURES=80 at configure time.
set_property(TARGET ooverlap_pybinding PROPERTY CUDA_ARCHITECTURES 80)

set_property(TARGET ooverlap_pybinding PROPERTY CUDA_SEPARABLE_COMPILATION ON)
