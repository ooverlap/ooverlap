cmake_minimum_required(VERSION 3.20)
project(ooverlap LANGUAGES CXX CUDA)

set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_INCLUDES 0)
set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_LIBRARIES 0)
set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_OBJECTS 0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# --- CUTLASS ---
if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/src/third-party/cutlass/include/cutlass/cutlass.h")
  message(FATAL_ERROR
    "CUTLASS submodule not found. Run:\n"
    "  git submodule update --init --recursive")
endif()
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/third_party/cutlass.cmake)

# --- Torch ---
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# --- CUDA ---
find_package(CUDAToolkit REQUIRED)

# --- NCCL (simple finder) ---
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

find_package(Python3 REQUIRED COMPONENTS Development)
target_include_directories(ooverlap_pybinding PRIVATE ${Python3_INCLUDE_DIRS})
target_link_libraries(ooverlap_pybinding PRIVATE Python3::Python)

target_link_libraries(ooverlap_pybinding PRIVATE
  cutlass
  ${TORCH_LIBRARIES}
  CUDA::cudart
  CUDA::cublas
  ${NCCL_LIBRARY}
)

# For Hopper/TMA later use 90 (or 80;90 for fatbin)
set_property(TARGET ooverlap_pybinding PROPERTY CUDA_ARCHITECTURES 80)
set_property(TARGET ooverlap_pybinding PROPERTY CUDA_SEPARABLE_COMPILATION ON)
