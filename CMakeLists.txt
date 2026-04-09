cmake_minimum_required(VERSION 3.20)
project(ooverlap LANGUAGES CXX CUDA)

unset(CMAKE_CUDA_ARCHITECTURES CACHE)
set(CMAKE_CUDA_ARCHITECTURES OFF CACHE STRING "Disable default CUDA arch flags" FORCE)

# If CUDAARCHS was exported in the shell, stop it from seeding CMake.
unset(ENV{CUDAARCHS})

string(REGEX REPLACE " ?-gencode arch=compute_89,code=sm_89" "" CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}")
string(REGEX REPLACE " ?--generate-code=arch=compute_89,code=sm_89" "" CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}")

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
set(ENV{TORCH_CUDA_ARCH_LIST} "9.0")
find_package(Torch REQUIRED)
string(REPLACE "-gencode;arch=compute_89,code=sm_89" "" CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}")
string(REPLACE "-gencode arch=compute_89,code=sm_89" "" CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}")
string(REPLACE "--generate-code=arch=compute_89,code=sm_89" "" CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}")

string(REPLACE "-gencode;arch=compute_89,code=sm_89" "" TORCH_CXX_FLAGS "${TORCH_CXX_FLAGS}")
string(REPLACE "-gencode arch=compute_89,code=sm_89" "" TORCH_CXX_FLAGS "${TORCH_CXX_FLAGS}")
string(REPLACE "--generate-code=arch=compute_89,code=sm_89" "" TORCH_CXX_FLAGS "${TORCH_CXX_FLAGS}")

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

add_subdirectory(src)
