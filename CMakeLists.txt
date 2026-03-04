cmake_minimum_required(VERSION 3.20)
project(MyProject LANGUAGES CXX CUDA)  # enable CUDA if you compile .cu files

# (optional but common)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Make sure submodules are present (CMake-side check)
if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/src/third-party/cutlass/include/cutlass/cutlass.h")
  message(FATAL_ERROR
    "CUTLASS submodule not found. Run:\n"
    "  git submodule update --init --recursive"
  )
endif()

include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/third_party/cutlass.cmake)
