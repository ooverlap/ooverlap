cmake_minimum_required(VERSION 3.20)

if(NOT TARGET CUDA::cudart)
  find_package(CUDAToolkit REQUIRED)
endif()

find_package(Threads REQUIRED)

set(OOVERLAP_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR})

set(OOVERLAP_SM90A_GENCODE "--generate-code=arch=compute_90a,code=sm_90a")
set(OOVERLAP_SM90A_PTX     "--generate-code=arch=compute_90a,code=compute_90a")

function(ooverlap_force_sm90a target)
  if(TARGET ${target})
    set_target_properties(${target} PROPERTIES CUDA_ARCHITECTURES OFF)
    target_compile_options(${target} PRIVATE
      $<$<COMPILE_LANGUAGE:CUDA>:${OOVERLAP_SM90A_GENCODE}>
      $<$<COMPILE_LANGUAGE:CUDA>:${OOVERLAP_SM90A_PTX}>
    )
  endif()
endfunction()

# Optional cudadevrt lookup.
set(OOVERLAP_CUDADEVRT_LIBRARY "")

if(EXISTS "/apps/Common/software/CUDA/12.4.0/targets/x86_64-linux/lib/libcudadevrt.a")
  set(OOVERLAP_CUDADEVRT_LIBRARY
      "/apps/Common/software/CUDA/12.4.0/targets/x86_64-linux/lib/libcudadevrt.a")
endif()

if(NOT OOVERLAP_CUDADEVRT_LIBRARY)
  find_library(OOVERLAP_CUDADEVRT_LIBRARY
    NAMES cudadevrt libcudadevrt.a
    HINTS
      ${CUDAToolkit_LIBRARY_DIR}
      ${CUDAToolkit_LIBRARY_ROOT}
      ${CUDAToolkit_LIBRARY_ROOT}/lib64
      ${CUDAToolkit_LIBRARY_ROOT}/lib
      ${CUDAToolkit_ROOT_DIR}/targets/x86_64-linux/lib
      ${CUDAToolkit_ROOT_DIR}/lib64
      ${CUDAToolkit_ROOT_DIR}/lib
      /usr/local/cuda/targets/x86_64-linux/lib
      /usr/local/cuda/lib64
      /usr/local/cuda/lib
  )
endif()

function(ooverlap_link_cudadevrt target visibility)
  if(OOVERLAP_CUDADEVRT_LIBRARY)
    target_link_libraries(${target} ${visibility} ${OOVERLAP_CUDADEVRT_LIBRARY})
  endif()
endfunction()

# -------------------------
# System utilities
# -------------------------
add_library(ooverlap_system STATIC
  ${OOVERLAP_SRC_DIR}/system/broker.cpp
)

add_library(ooverlap::system ALIAS ooverlap_system)

set_property(TARGET ooverlap_system PROPERTY POSITION_INDEPENDENT_CODE ON)

target_include_directories(ooverlap_system
  PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
  PRIVATE
    ${OOVERLAP_SRC_DIR}
)

target_link_libraries(ooverlap_system
  PUBLIC
    Threads::Threads
)

# -------------------------
# Core public communication library
# -------------------------
if(OOVERLAP_BUILD_CORE)
  add_library(ooverlap_comm STATIC
    ${OOVERLAP_SRC_DIR}/comm/ooverlap_comm.cu
    ${OOVERLAP_SRC_DIR}/comm/tma_two_gpu_peer_allreduce_sm90.cu
  )

  add_library(ooverlap::comm ALIAS ooverlap_comm)

  ooverlap_force_sm90a(ooverlap_comm)

  set_property(TARGET ooverlap_comm PROPERTY POSITION_INDEPENDENT_CODE ON)
  set_property(TARGET ooverlap_comm PROPERTY CUDA_SEPARABLE_COMPILATION ON)
  set_property(TARGET ooverlap_comm PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)

  target_include_directories(ooverlap_comm
    PUBLIC
      $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
      $<INSTALL_INTERFACE:include>
    PRIVATE
      ${OOVERLAP_SRC_DIR}
  )

  target_link_libraries(ooverlap_comm
    PUBLIC
      ooverlap_system
      CUDA::cudart
    PRIVATE
      CUDA::cuda_driver
  )

  ooverlap_link_cudadevrt(ooverlap_comm PRIVATE)
endif()

# Everything below is optional development/Python/benchmark infrastructure.
if(NOT OOVERLAP_BUILD_PYTHON AND NOT OOVERLAP_BUILD_TESTS AND NOT OOVERLAP_BUILD_LEGACY_OVERLAP)
  return()
endif()

# -------------------------
# Dependency helpers for Python/Torch builds
# -------------------------
if(OOVERLAP_BUILD_PYTHON OR OOVERLAP_BUILD_LEGACY_OVERLAP)
  if(NOT DEFINED TORCH_LIBRARIES)
    find_package(Torch REQUIRED)
  endif()

  if(NOT TARGET Python3::Python)
    find_package(Python3 REQUIRED COMPONENTS Development)
  endif()

  set(TORCH_PY_INCLUDE_DIRS "")
  if(Python3_EXECUTABLE)
    execute_process(
      COMMAND ${Python3_EXECUTABLE} -c
        "import torch, torch.utils.cpp_extension as e; print(';'.join(e.include_paths()))"
      OUTPUT_VARIABLE TORCH_PY_INCLUDE_DIRS
      ERROR_QUIET
      OUTPUT_STRIP_TRAILING_WHITESPACE
    )
  endif()

  set(TORCH_PY_LIBDIR "")
  set(TORCH_PYTHON_SO "")
  if(Python3_EXECUTABLE)
    execute_process(
      COMMAND ${Python3_EXECUTABLE} -c
        "import torch, os; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))"
      OUTPUT_VARIABLE TORCH_PY_LIBDIR
      OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    execute_process(
      COMMAND ${Python3_EXECUTABLE} -c
        "import torch, os; print(os.path.join(os.path.dirname(torch.__file__), 'lib', 'libtorch_python.so'))"
      OUTPUT_VARIABLE TORCH_PYTHON_SO
      OUTPUT_STRIP_TRAILING_WHITESPACE
    )
  endif()

  function(ooverlap_apply_torch target)
    if(DEFINED TORCH_INCLUDE_DIRS)
      target_include_directories(${target} PRIVATE ${TORCH_INCLUDE_DIRS})
    endif()

    if(TORCH_PY_INCLUDE_DIRS)
      target_include_directories(${target} PRIVATE ${TORCH_PY_INCLUDE_DIRS})
    endif()

    if(TARGET Python3::Python)
      target_link_libraries(${target} PRIVATE Python3::Python)
    elseif(DEFINED Python3_INCLUDE_DIRS)
      target_include_directories(${target} PRIVATE ${Python3_INCLUDE_DIRS})
    endif()

    if(DEFINED TORCH_CXX_FLAGS)
      separate_arguments(_TORCH_CXX_FLAGS_LIST NATIVE_COMMAND "${TORCH_CXX_FLAGS}")
      target_compile_options(${target} PRIVATE ${_TORCH_CXX_FLAGS_LIST})
    endif()

    if(DEFINED TORCH_LIBRARIES)
      target_link_libraries(${target} PRIVATE ${TORCH_LIBRARIES})
    endif()
  endfunction()
endif()

if((OOVERLAP_BUILD_PYTHON OR OOVERLAP_BUILD_TESTS OR OOVERLAP_BUILD_LEGACY_OVERLAP)
   AND (NOT DEFINED NCCL_INCLUDE_DIR OR NOT DEFINED NCCL_LIBRARY))
  find_path(NCCL_INCLUDE_DIR nccl.h
    HINTS ENV NCCL_ROOT
    PATH_SUFFIXES include)
  find_library(NCCL_LIBRARY nccl
    HINTS ENV NCCL_ROOT
    PATH_SUFFIXES lib lib64)

  if(NOT NCCL_INCLUDE_DIR OR NOT NCCL_LIBRARY)
    message(FATAL_ERROR "NCCL not found. Set NCCL_ROOT to your NCCL install.")
  endif()
endif()

if((OOVERLAP_BUILD_GEMM OR OOVERLAP_BUILD_PYTHON OR OOVERLAP_BUILD_LEGACY_OVERLAP)
   AND NOT TARGET cutlass)
  add_library(cutlass INTERFACE)
  target_include_directories(cutlass INTERFACE
    ${PROJECT_SOURCE_DIR}/src/third-party/cutlass/include
    ${PROJECT_SOURCE_DIR}/src/third-party/cutlass/tools/util/include
  )
endif()

# -------------------------
# Benchmark/test helper library
# -------------------------
if(OOVERLAP_BUILD_TESTS AND EXISTS ${OOVERLAP_SRC_DIR}/test/persistent_allreduce_2gpu_sm90.cu)
  add_library(ooverlap_tests STATIC
    ${OOVERLAP_SRC_DIR}/test/persistent_allreduce_2gpu_sm90.cu
    ${OOVERLAP_SRC_DIR}/test/ipc_allreduce_2gpu_sm90.cu
    ${OOVERLAP_SRC_DIR}/test/ipc_allreduce_benchmark_2gpu_sm90.cu
    ${OOVERLAP_SRC_DIR}/test/tma_bandwidth_experiment_sm90.cu
    ${OOVERLAP_SRC_DIR}/test/tma_allreduce_sweep_2gpu_sm90.cu
    ${OOVERLAP_SRC_DIR}/test/public_allreduce_benchmark_2gpu_sm90.cu
  )

  ooverlap_force_sm90a(ooverlap_tests)

  set_property(TARGET ooverlap_tests PROPERTY POSITION_INDEPENDENT_CODE ON)
  set_property(TARGET ooverlap_tests PROPERTY CUDA_SEPARABLE_COMPILATION ON)
  set_property(TARGET ooverlap_tests PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)

  target_include_directories(ooverlap_tests PUBLIC
    ${PROJECT_SOURCE_DIR}/include
    ${OOVERLAP_SRC_DIR}
    ${NCCL_INCLUDE_DIR}
  )

  target_link_libraries(ooverlap_tests PUBLIC
    ooverlap_comm
    CUDA::cudart
    CUDA::cuda_driver
    ${NCCL_LIBRARY}
  )

  ooverlap_link_cudadevrt(ooverlap_tests PUBLIC)
endif()

# -------------------------
# GEMM library
# -------------------------
if(OOVERLAP_BUILD_GEMM)
  file(GLOB gemm_files CONFIGURE_DEPENDS
    ${OOVERLAP_SRC_DIR}/gemm/*.cu
    ${OOVERLAP_SRC_DIR}/gemm/*.cpp
    ${OOVERLAP_SRC_DIR}/gemm/*.cc
  )

  if(gemm_files)
    add_library(ooverlap_gemm STATIC ${gemm_files})
    ooverlap_force_sm90a(ooverlap_gemm)

    set_property(TARGET ooverlap_gemm PROPERTY POSITION_INDEPENDENT_CODE ON)
    set_property(TARGET ooverlap_gemm PROPERTY CUDA_SEPARABLE_COMPILATION ON)

    target_include_directories(ooverlap_gemm PUBLIC
      ${PROJECT_SOURCE_DIR}/include
      ${OOVERLAP_SRC_DIR}
      ${PROJECT_SOURCE_DIR}/src/third-party/cutlass/include
      ${PROJECT_SOURCE_DIR}/src/third-party/cutlass/tools/util/include
    )

    target_link_libraries(ooverlap_gemm PUBLIC
      cutlass
      CUDA::cudart
      CUDA::cuda_driver
    )
  endif()
endif()

# -------------------------
# Legacy overlap library
# -------------------------
if(OOVERLAP_BUILD_LEGACY_OVERLAP)
  file(GLOB overlap_files CONFIGURE_DEPENDS
    ${OOVERLAP_SRC_DIR}/overlap/*.cu
    ${OOVERLAP_SRC_DIR}/overlap/*.cpp
    ${OOVERLAP_SRC_DIR}/sm90/*.cu
    ${OOVERLAP_SRC_DIR}/sm90/*.cpp
  )

  if(overlap_files)
    add_library(ooverlap_overlap STATIC ${overlap_files})
    ooverlap_force_sm90a(ooverlap_overlap)

    set_property(TARGET ooverlap_overlap PROPERTY POSITION_INDEPENDENT_CODE ON)
    set_property(TARGET ooverlap_overlap PROPERTY CUDA_SEPARABLE_COMPILATION ON)
    set_property(TARGET ooverlap_overlap PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)

    target_include_directories(ooverlap_overlap PUBLIC
      ${PROJECT_SOURCE_DIR}/include
      ${OOVERLAP_SRC_DIR}
      ${PROJECT_SOURCE_DIR}/src/third-party/cutlass/include
      ${PROJECT_SOURCE_DIR}/src/third-party/cutlass/tools/util/include
    )

    ooverlap_apply_torch(ooverlap_overlap)

    target_link_libraries(ooverlap_overlap PUBLIC
      cutlass
      CUDA::cudart
      CUDA::cuda_driver
    )

    ooverlap_link_cudadevrt(ooverlap_overlap PUBLIC)
  endif()
endif()

# -------------------------
# RMSNorm library
# -------------------------
if(OOVERLAP_BUILD_RMSNORM)
  file(GLOB norm_files CONFIGURE_DEPENDS
    ${OOVERLAP_SRC_DIR}/rmsnorm/*.cu
    ${OOVERLAP_SRC_DIR}/rmsnorm/*.cpp
    ${OOVERLAP_SRC_DIR}/rmsnorm/*.cc
  )

  if(norm_files)
    add_library(ooverlap_norm STATIC ${norm_files})
    ooverlap_force_sm90a(ooverlap_norm)

    set_property(TARGET ooverlap_norm PROPERTY POSITION_INDEPENDENT_CODE ON)
    set_property(TARGET ooverlap_norm PROPERTY CUDA_SEPARABLE_COMPILATION ON)

    target_include_directories(ooverlap_norm PUBLIC
      ${OOVERLAP_SRC_DIR}
      ${NCCL_INCLUDE_DIR}
      ${PROJECT_SOURCE_DIR}/include
    )

    ooverlap_apply_torch(ooverlap_norm)

    target_link_libraries(ooverlap_norm PUBLIC
      CUDA::cudart
      CUDA::cuda_driver
    )
  endif()
endif()

# -------------------------
# baseline_impl
# -------------------------
if(OOVERLAP_BUILD_PYTHON AND EXISTS ${OOVERLAP_SRC_DIR}/baseline_impl.cu)
  add_library(ooverlap_baseline_impl STATIC ${OOVERLAP_SRC_DIR}/baseline_impl.cu)
  set_source_files_properties(${OOVERLAP_SRC_DIR}/baseline_impl.cu PROPERTIES LANGUAGE CXX)

  set_property(TARGET ooverlap_baseline_impl PROPERTY POSITION_INDEPENDENT_CODE ON)

  target_include_directories(ooverlap_baseline_impl PUBLIC
    ${OOVERLAP_SRC_DIR}
    ${NCCL_INCLUDE_DIR}
    ${PROJECT_SOURCE_DIR}/include
  )

  ooverlap_apply_torch(ooverlap_baseline_impl)

  if(TARGET ooverlap_gemm)
    target_link_libraries(ooverlap_baseline_impl PUBLIC ooverlap_gemm)
  endif()
  if(TARGET ooverlap_overlap)
    target_link_libraries(ooverlap_baseline_impl PUBLIC ooverlap_overlap)
  endif()
  if(TARGET ooverlap_norm)
    target_link_libraries(ooverlap_baseline_impl PUBLIC ooverlap_norm)
  endif()

  target_link_libraries(ooverlap_baseline_impl PUBLIC
    CUDA::cudart
    CUDA::cuda_driver
    CUDA::cublas
    ${NCCL_LIBRARY}
  )
endif()

# -------------------------
# overlap_impl
# -------------------------
if(OOVERLAP_BUILD_PYTHON AND EXISTS ${OOVERLAP_SRC_DIR}/overlap_impl.cu)
  add_library(ooverlap_overlap_impl STATIC ${OOVERLAP_SRC_DIR}/overlap_impl.cu)
  ooverlap_force_sm90a(ooverlap_overlap_impl)

  set_property(TARGET ooverlap_overlap_impl PROPERTY POSITION_INDEPENDENT_CODE ON)
  set_property(TARGET ooverlap_overlap_impl PROPERTY CUDA_SEPARABLE_COMPILATION ON)
  set_property(TARGET ooverlap_overlap_impl PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)

  target_include_directories(ooverlap_overlap_impl PUBLIC
    ${OOVERLAP_SRC_DIR}
    ${NCCL_INCLUDE_DIR}
    ${PROJECT_SOURCE_DIR}/include
  )

  ooverlap_apply_torch(ooverlap_overlap_impl)

  if(TARGET ooverlap_overlap)
    target_link_libraries(ooverlap_overlap_impl PUBLIC ooverlap_overlap)
  endif()
  if(TARGET ooverlap_norm)
    target_link_libraries(ooverlap_overlap_impl PUBLIC ooverlap_norm)
  endif()

  target_link_libraries(ooverlap_overlap_impl PUBLIC
    CUDA::cudart
    CUDA::cuda_driver
    ${NCCL_LIBRARY}
  )

  ooverlap_link_cudadevrt(ooverlap_overlap_impl PUBLIC)
endif()

# -------------------------
# nccl_tool
# -------------------------
if(OOVERLAP_BUILD_PYTHON AND EXISTS ${OOVERLAP_SRC_DIR}/nccl_utils.cu)
  add_library(ooverlap_nccl_tool STATIC ${OOVERLAP_SRC_DIR}/nccl_utils.cu)
  set_source_files_properties(${OOVERLAP_SRC_DIR}/nccl_utils.cu PROPERTIES LANGUAGE CXX)

  set_property(TARGET ooverlap_nccl_tool PROPERTY POSITION_INDEPENDENT_CODE ON)

  target_include_directories(ooverlap_nccl_tool PUBLIC
    ${OOVERLAP_SRC_DIR}
    ${NCCL_INCLUDE_DIR}
    ${PROJECT_SOURCE_DIR}/include
  )

  ooverlap_apply_torch(ooverlap_nccl_tool)

  target_link_libraries(ooverlap_nccl_tool PUBLIC
    CUDA::cudart
    CUDA::cuda_driver
    ${NCCL_LIBRARY}
  )
endif()

# -------------------------
# Python extension module
# -------------------------
if(OOVERLAP_BUILD_PYTHON)
  if(NOT EXISTS ${OOVERLAP_SRC_DIR}/pybind.cpp)
    message(FATAL_ERROR "src/pybind.cpp not found")
  endif()

  set(OOVERLAP_EXT_SOURCES
    ${OOVERLAP_SRC_DIR}/pybind.cpp
  )

  # pybind.cpp binds the persistent allreduce benchmark/smoke functions
  # directly, so compile their implementation directly into the extension.
  # Do not rely on the optional ooverlap_tests static library for these symbols.
  if(EXISTS ${OOVERLAP_SRC_DIR}/test/persistent_allreduce_2gpu_sm90.cu)
    list(APPEND OOVERLAP_EXT_SOURCES
      ${OOVERLAP_SRC_DIR}/test/persistent_allreduce_2gpu_sm90.cu
    )
  else()
    message(FATAL_ERROR
      "pybind.cpp requires src/test/persistent_allreduce_2gpu_sm90.cu, but it was not found")
  endif()
  
  if(EXISTS ${OOVERLAP_SRC_DIR}/test/ipc_allreduce_2gpu_sm90.cu)
    list(APPEND OOVERLAP_EXT_SOURCES
      ${OOVERLAP_SRC_DIR}/test/ipc_allreduce_2gpu_sm90.cu
    )
  else()
    message(FATAL_ERROR
      "pybind.cpp requires src/test/ipc_allreduce_2gpu_sm90.cu, but it was not found")
  endif()

  if(EXISTS ${OOVERLAP_SRC_DIR}/test/ipc_allreduce_benchmark_2gpu_sm90.cu)
    list(APPEND OOVERLAP_EXT_SOURCES
      ${OOVERLAP_SRC_DIR}/test/ipc_allreduce_benchmark_2gpu_sm90.cu
    )
  else()
    message(FATAL_ERROR
      "pybind.cpp requires src/test/ipc_allreduce_benchmark_2gpu_sm90.cu, but it was not found")
  endif()
  
  if(EXISTS ${OOVERLAP_SRC_DIR}/test/tma_bandwidth_experiment_sm90.cu)
    list(APPEND OOVERLAP_EXT_SOURCES
      ${OOVERLAP_SRC_DIR}/test/tma_bandwidth_experiment_sm90.cu
    )
  else()
    message(FATAL_ERROR
      "pybind.cpp requires src/test/tma_bandwidth_experiment_sm90.cu, but it was not found")
  endif()

  if(EXISTS ${OOVERLAP_SRC_DIR}/test/tma_allreduce_sweep_2gpu_sm90.cu)
    list(APPEND OOVERLAP_EXT_SOURCES
      ${OOVERLAP_SRC_DIR}/test/tma_allreduce_sweep_2gpu_sm90.cu
    )
  else()
    message(FATAL_ERROR
      "pybind.cpp requires src/test/tma_allreduce_sweep_2gpu_sm90.cu, but it was not found")
  endif()
  
  if(EXISTS ${OOVERLAP_SRC_DIR}/test/public_allreduce_benchmark_2gpu_sm90.cu)
    list(APPEND OOVERLAP_EXT_SOURCES
      ${OOVERLAP_SRC_DIR}/test/public_allreduce_benchmark_2gpu_sm90.cu
    )
  else()
    message(FATAL_ERROR
      "pybind.cpp requires src/test/public_allreduce_benchmark_2gpu_sm90.cu, but it was not found")
  endif()

  add_library(ooverlap_ext MODULE ${OOVERLAP_EXT_SOURCES})
  target_link_libraries(ooverlap_ext PRIVATE ooverlap_system)

  ooverlap_force_sm90a(ooverlap_ext)

  set_target_properties(ooverlap_ext PROPERTIES
    PREFIX ""
    LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib
    POSITION_INDEPENDENT_CODE ON
  )

  set_property(TARGET ooverlap_ext PROPERTY CUDA_SEPARABLE_COMPILATION ON)
  set_property(TARGET ooverlap_ext PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)

  target_include_directories(ooverlap_ext PRIVATE
    ${OOVERLAP_SRC_DIR}
    ${NCCL_INCLUDE_DIR}
    ${Python3_INCLUDE_DIRS}
    ${PROJECT_SOURCE_DIR}/include
  )

  if(TARGET ooverlap_comm)
    target_link_libraries(ooverlap_ext PRIVATE ooverlap_comm)
  endif()
  if(TARGET ooverlap_overlap)
    target_link_libraries(ooverlap_ext PRIVATE ooverlap_overlap)
  endif()
  if(TARGET ooverlap_baseline_impl)
    target_link_libraries(ooverlap_ext PRIVATE ooverlap_baseline_impl)
  endif()
  if(TARGET ooverlap_nccl_tool)
    target_link_libraries(ooverlap_ext PRIVATE ooverlap_nccl_tool)
  endif()
  if(TARGET ooverlap_norm)
    target_link_libraries(ooverlap_ext PRIVATE ooverlap_norm)
  endif()
  if(TARGET ooverlap_overlap_impl)
    target_link_libraries(ooverlap_ext PRIVATE ooverlap_overlap_impl)
  endif()

  ooverlap_apply_torch(ooverlap_ext)
  target_link_libraries(ooverlap_ext PRIVATE Python3::Python)

  if(EXISTS "${TORCH_PYTHON_SO}")
    message(STATUS "Linking libtorch_python: ${TORCH_PYTHON_SO}")
    target_link_options(ooverlap_ext PRIVATE "-Wl,--no-as-needed")
    target_link_libraries(ooverlap_ext PRIVATE "${TORCH_PYTHON_SO}")
  else()
    message(FATAL_ERROR "libtorch_python.so not found at: ${TORCH_PYTHON_SO}")
  endif()

  target_link_libraries(ooverlap_ext PRIVATE
    ${TORCH_LIBRARIES}
    CUDA::cudart
    CUDA::cuda_driver
    CUDA::cublas
    ${NCCL_LIBRARY}
  )

  if(TARGET cutlass)
    target_link_libraries(ooverlap_ext PRIVATE cutlass)
  endif()

  ooverlap_link_cudadevrt(ooverlap_ext PRIVATE)

  if(TORCH_PY_LIBDIR)
    message(STATUS "Torch libdir (RPATH): ${TORCH_PY_LIBDIR}")
    set_target_properties(ooverlap_ext PROPERTIES
      BUILD_RPATH "${TORCH_PY_LIBDIR}"
      INSTALL_RPATH "${TORCH_PY_LIBDIR};\$ORIGIN"
    )
    target_link_options(ooverlap_ext PRIVATE "-Wl,-rpath,${TORCH_PY_LIBDIR}")
  endif()
endif()
