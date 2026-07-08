cmake_minimum_required(VERSION 3.20)
project(ooverlap LANGUAGES CXX CUDA)


# OOVERLAP_DEFAULT_RELEASE_PATCH:
# Default to Release for single-config generators unless the user explicitly
# selected another build type.
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(OOVERLAP_LOG_LEVEL "2" CACHE STRING
    "ooverlap compile-time log level: 0 none, 1 fatal, 2 error, 3 warn, 4 info, 5 debug, 6 trace")

add_compile_definitions(OOVERLAP_LOG_LEVEL=${OOVERLAP_LOG_LEVEL})

if(CMAKE_SOURCE_DIR STREQUAL PROJECT_SOURCE_DIR)
  set(OOVERLAP_TOP_LEVEL ON)
else()
  set(OOVERLAP_TOP_LEVEL OFF)
endif()

option(OOVERLAP_BUILD_CORE "Build the public ooverlap communication library" ON)

if(OOVERLAP_TOP_LEVEL)
  set(OOVERLAP_BUILD_PYTHON_DEFAULT ON)
  set(OOVERLAP_BUILD_TESTS_DEFAULT ON)
else()
  set(OOVERLAP_BUILD_PYTHON_DEFAULT OFF)
  set(OOVERLAP_BUILD_TESTS_DEFAULT OFF)
endif()

option(OOVERLAP_BUILD_PYTHON "Build the Python extension module" ${OOVERLAP_BUILD_PYTHON_DEFAULT})
option(OOVERLAP_BUILD_TESTS "Build benchmark/test helper translation units" ${OOVERLAP_BUILD_TESTS_DEFAULT})
option(OOVERLAP_BUILD_GEMM "Build GEMM/CUTLASS helper libraries" ${OOVERLAP_BUILD_PYTHON})
option(OOVERLAP_BUILD_LEGACY_OVERLAP "Build legacy overlap/NCCL/Torch integration" ${OOVERLAP_BUILD_PYTHON})
option(OOVERLAP_BUILD_RMSNORM "Build RMSNorm helpers" ${OOVERLAP_BUILD_PYTHON})
option(OOVERLAP_USE_BUNDLED_NCCL "Build and link NCCL from src/third-party/nccl" ON)

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

find_package(CUDAToolkit REQUIRED)

if(OOVERLAP_BUILD_GEMM OR OOVERLAP_BUILD_PYTHON OR OOVERLAP_BUILD_LEGACY_OVERLAP)
  if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/src/third-party/cutlass/include/cutlass/cutlass.h")
    message(FATAL_ERROR
      "CUTLASS submodule not found. Run:\n"
      "  git submodule update --init --recursive")
  endif()
  include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/third_party/cutlass.cmake)
endif()

if(OOVERLAP_BUILD_PYTHON OR OOVERLAP_BUILD_LEGACY_OVERLAP)
  set(ENV{TORCH_CUDA_ARCH_LIST} "9.0")
  find_package(Torch REQUIRED)

  string(REPLACE "-gencode;arch=compute_89,code=sm_89" "" CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}")
  string(REPLACE "-gencode arch=compute_89,code=sm_89" "" CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}")
  string(REPLACE "--generate-code=arch=compute_89,code=sm_89" "" CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}")

  string(REPLACE "-gencode;arch=compute_89,code=sm_89" "" TORCH_CXX_FLAGS "${TORCH_CXX_FLAGS}")
  string(REPLACE "-gencode arch=compute_89,code=sm_89" "" TORCH_CXX_FLAGS "${TORCH_CXX_FLAGS}")
  string(REPLACE "--generate-code=arch=compute_89,code=sm_89" "" TORCH_CXX_FLAGS "${TORCH_CXX_FLAGS}")

  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
endif()

if(OOVERLAP_BUILD_PYTHON OR OOVERLAP_BUILD_TESTS OR OOVERLAP_BUILD_LEGACY_OVERLAP)
  if(OOVERLAP_USE_BUNDLED_NCCL)
    include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/third_party/nccl.cmake)
  else()
    find_path(NCCL_INCLUDE_DIR nccl.h
      HINTS ENV NCCL_ROOT
      PATH_SUFFIXES include)
    find_library(NCCL_LIBRARY nccl
      HINTS ENV NCCL_ROOT
      PATH_SUFFIXES lib lib64)

    if(NOT NCCL_INCLUDE_DIR OR NOT NCCL_LIBRARY)
      message(FATAL_ERROR "NCCL not found. Set NCCL_ROOT to your NCCL install.")
    endif()

    add_library(ooverlap_nccl SHARED IMPORTED GLOBAL)
    add_library(NCCL::NCCL ALIAS ooverlap_nccl)
    set_target_properties(ooverlap_nccl PROPERTIES
      IMPORTED_LOCATION ${NCCL_LIBRARY}
      INTERFACE_INCLUDE_DIRECTORIES ${NCCL_INCLUDE_DIR}
    )
  endif()
endif()

if(OOVERLAP_BUILD_PYTHON OR OOVERLAP_BUILD_TESTS)
  if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/src/third-party/nlohmann_json/include/nlohmann/json.hpp")
    message(FATAL_ERROR
      "nlohmann/json submodule not found. Run:\n"
      "  git submodule update --init --recursive")
  endif()

  add_library(nlohmann_json INTERFACE)
  add_library(nlohmann_json::nlohmann_json ALIAS nlohmann_json)

  target_include_directories(nlohmann_json INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/src/third-party/nlohmann_json/include
  )
endif()

add_subdirectory(src)
