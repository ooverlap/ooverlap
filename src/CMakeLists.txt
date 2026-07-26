cmake_minimum_required(VERSION 3.20)

find_package(CUDAToolkit REQUIRED)
find_package(Threads REQUIRED)

set(OOVERLAP_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR})

set(OOVERLAP_SM90A_GENCODE "--generate-code=arch=compute_90a,code=sm_90a")
set(OOVERLAP_SM90A_PTX     "--generate-code=arch=compute_90a,code=compute_90a")

# -------------------------
# Small build helpers
# -------------------------

function(ooverlap_force_sm90a target)
  if(TARGET ${target})
    set_target_properties(${target} PROPERTIES CUDA_ARCHITECTURES OFF)
    target_compile_options(${target} PRIVATE
      $<$<COMPILE_LANGUAGE:CUDA>:${OOVERLAP_SM90A_GENCODE}>
      $<$<COMPILE_LANGUAGE:CUDA>:${OOVERLAP_SM90A_PTX}>
    )
  endif()
endfunction()

function(ooverlap_enable_pic target)
  set_property(TARGET ${target} PROPERTY POSITION_INDEPENDENT_CODE ON)
endfunction()



# OOVERLAP_TARGET_RELEASE_OPTS_PATCH:
# Force optimized compilation for hot ooverlap targets even when the caller
# forgets -DCMAKE_BUILD_TYPE=Release.
function(ooverlap_enable_release_opts target)
  if(TARGET ${target})
    target_compile_definitions(${target} PRIVATE NDEBUG)
    target_compile_options(${target} PRIVATE
      $<$<COMPILE_LANGUAGE:CXX>:-O3>
      $<$<COMPILE_LANGUAGE:CUDA>:-O3>
      # OOVERLAP_CUDA_MAX_PERF_FLAGS_V3
      #$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-O3>
      #$<$<COMPILE_LANGUAGE:CUDA>:--extra-device-vectorization>
      #$<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
    )
  endif()
endfunction()
function(ooverlap_public_include_dirs target)
  target_include_directories(${target} PUBLIC
    ${PROJECT_SOURCE_DIR}/include
    ${OOVERLAP_SRC_DIR}
  )
endfunction()

# Optional cudadevrt lookup.
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
    /apps/Common/software/CUDA/12.4.0/targets/x86_64-linux/lib
    /usr/local/cuda/targets/x86_64-linux/lib
    /usr/local/cuda/lib64
    /usr/local/cuda/lib
)

function(ooverlap_link_cudadevrt target visibility)
  if(OOVERLAP_CUDADEVRT_LIBRARY)
    target_link_libraries(${target} ${visibility} ${OOVERLAP_CUDADEVRT_LIBRARY})
  endif()
endfunction()

# -------------------------
# Dependency decisions
# -------------------------

set(OOVERLAP_NEED_CORE OFF)
if(OOVERLAP_BUILD_CORE OR OOVERLAP_BUILD_PYTHON OR OOVERLAP_BUILD_TESTS OR OOVERLAP_BUILD_TORCH_COLLECTIVES)
  set(OOVERLAP_NEED_CORE ON)
endif()

set(OOVERLAP_NEED_TORCH OFF)
if(OOVERLAP_BUILD_PYTHON OR OOVERLAP_BUILD_RMSNORM OR OOVERLAP_BUILD_TORCH_COLLECTIVES)
  set(OOVERLAP_NEED_TORCH ON)
endif()

set(OOVERLAP_NEED_NCCL OFF)
if(OOVERLAP_BUILD_PYTHON OR OOVERLAP_BUILD_TESTS OR OOVERLAP_BUILD_RMSNORM)
  set(OOVERLAP_NEED_NCCL ON)
endif()

set(OOVERLAP_NEED_CUTLASS OFF)
if(OOVERLAP_BUILD_GEMM OR OOVERLAP_BUILD_PYTHON OR OOVERLAP_BUILD_LEGACY_OVERLAP)
  set(OOVERLAP_NEED_CUTLASS ON)
endif()

if(OOVERLAP_NEED_TORCH)
  if(NOT DEFINED TORCH_LIBRARIES)
    find_package(Torch REQUIRED)
  endif()

  find_package(Python3 REQUIRED COMPONENTS Interpreter Development)

  execute_process(
    COMMAND ${Python3_EXECUTABLE} -c
      "import torch, torch.utils.cpp_extension as e; print(';'.join(e.include_paths()))"
    OUTPUT_VARIABLE TORCH_PY_INCLUDE_DIRS
    ERROR_QUIET
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )

  execute_process(
    COMMAND ${Python3_EXECUTABLE} -c
      "import torch, os; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))"
    OUTPUT_VARIABLE TORCH_PY_LIBDIR
    ERROR_QUIET
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )

  execute_process(
    COMMAND ${Python3_EXECUTABLE} -c
      "import torch, os; print(os.path.join(os.path.dirname(torch.__file__), 'lib', 'libtorch_python.so'))"
    OUTPUT_VARIABLE TORCH_PYTHON_SO
    ERROR_QUIET
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )

  function(ooverlap_apply_torch target)
    if(DEFINED TORCH_INCLUDE_DIRS)
      target_include_directories(${target} PRIVATE ${TORCH_INCLUDE_DIRS})
    endif()

    if(TORCH_PY_INCLUDE_DIRS)
      target_include_directories(${target} PRIVATE ${TORCH_PY_INCLUDE_DIRS})
    endif()

    if(TARGET Python3::Python)
      target_link_libraries(${target} PRIVATE Python3::Python)
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

if(OOVERLAP_NEED_CUTLASS AND NOT TARGET cutlass)
  add_library(cutlass INTERFACE)
  target_include_directories(cutlass INTERFACE
    ${PROJECT_SOURCE_DIR}/src/third-party/cutlass/include
    ${PROJECT_SOURCE_DIR}/src/third-party/cutlass/tools/util/include
  )
endif()

# -------------------------
# System utilities
# -------------------------

add_library(ooverlap_topology STATIC
  ${OOVERLAP_SRC_DIR}/topology/topology.cpp
  ${OOVERLAP_SRC_DIR}/topology/topology_probe.cu
)

add_library(ooverlap::topology ALIAS ooverlap_topology)

ooverlap_force_sm90a(ooverlap_topology)
ooverlap_enable_pic(ooverlap_topology)

ooverlap_enable_release_opts(ooverlap_topology)
set_property(TARGET ooverlap_topology PROPERTY CUDA_SEPARABLE_COMPILATION ON)
set_property(TARGET ooverlap_topology PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)

target_include_directories(ooverlap_topology
  PUBLIC
    $<BUILD_INTERFACE:${OOVERLAP_SRC_DIR}>
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(ooverlap_topology
  PUBLIC
    CUDA::cudart
)

ooverlap_link_cudadevrt(ooverlap_topology PRIVATE)


add_library(ooverlap_system STATIC
  ${OOVERLAP_SRC_DIR}/system/broker.cpp
)

add_library(ooverlap::system ALIAS ooverlap_system)

ooverlap_enable_pic(ooverlap_system)

ooverlap_enable_release_opts(ooverlap_system)
target_include_directories(ooverlap_system
  PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
  PRIVATE
    ${OOVERLAP_SRC_DIR}
)

target_link_libraries(ooverlap_system PUBLIC Threads::Threads)

# -------------------------
# Core communication library
# -------------------------

if(OOVERLAP_NEED_CORE)
  set_source_files_properties(
    ${OOVERLAP_SRC_DIR}/comm/tuning/tuning_policy.cpp
    PROPERTIES LANGUAGE CUDA
  )

  add_library(ooverlap_comm STATIC
    ${OOVERLAP_SRC_DIR}/comm/ooverlap_comm_public.cu
    ${OOVERLAP_SRC_DIR}/comm/ooverlap_comm_private.cu
    ${OOVERLAP_SRC_DIR}/comm/ooverlap_allreduce.cu
    ${OOVERLAP_SRC_DIR}/comm/ooverlap_reduce_scatter.cu
    ${OOVERLAP_SRC_DIR}/comm/ooverlap_all_gather.cu
    ${OOVERLAP_SRC_DIR}/comm/tma_multi_gpu_allreduce_sm90.cu
    ${OOVERLAP_SRC_DIR}/comm/tma_multi_gpu_reduce_scatter_sm90.cu
    ${OOVERLAP_SRC_DIR}/comm/tma_multi_gpu_all_gather_sm90.cu
    ${OOVERLAP_SRC_DIR}/comm/tuning/tuning_policy.cpp

    ${OOVERLAP_SRC_DIR}/comm/plan/transfer_plan_distribution.cu
  )

  add_library(ooverlap::comm ALIAS ooverlap_comm)

  ooverlap_force_sm90a(ooverlap_comm)
  ooverlap_enable_pic(ooverlap_comm)

  ooverlap_enable_release_opts(ooverlap_comm)
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
      ooverlap_topology
      CUDA::cudart
    PRIVATE
      CUDA::cuda_driver
  )

  if(TARGET nlohmann_json::nlohmann_json)
    target_link_libraries(ooverlap_comm PRIVATE nlohmann_json::nlohmann_json)
  endif()

  ooverlap_link_cudadevrt(ooverlap_comm PRIVATE)
endif()

# -------------------------
# GEMM library
#
# The old source folder was src/overlap. The code now lives in src/gemm.
# Keep the compatibility target name ooverlap_overlap as an alias.
# -------------------------

if(OOVERLAP_NEED_CUTLASS)
  file(GLOB_RECURSE OOVERLAP_GEMM_SOURCES CONFIGURE_DEPENDS
    ${OOVERLAP_SRC_DIR}/gemm/*.cu
    ${OOVERLAP_SRC_DIR}/gemm/*.cpp
    ${OOVERLAP_SRC_DIR}/gemm/*.cc
    ${OOVERLAP_SRC_DIR}/sm90/*.cu
    ${OOVERLAP_SRC_DIR}/sm90/*.cpp
    ${OOVERLAP_SRC_DIR}/sm90/*.cc
  )

  if(NOT OOVERLAP_GEMM_SOURCES)
    message(FATAL_ERROR
      "GEMM sources were requested, but no sources were found under src/gemm or src/sm90.")
  endif()

  add_library(ooverlap_gemm STATIC ${OOVERLAP_GEMM_SOURCES})
  add_library(ooverlap::gemm ALIAS ooverlap_gemm)

  if(NOT TARGET ooverlap_overlap)
    add_library(ooverlap_overlap ALIAS ooverlap_gemm)
  endif()

  ooverlap_force_sm90a(ooverlap_gemm)
  ooverlap_enable_pic(ooverlap_gemm)

  ooverlap_enable_release_opts(ooverlap_gemm)
  # CUTLASS SM90 WGMMA kernels should not be compiled with RDC.
  set_property(TARGET ooverlap_gemm PROPERTY CUDA_SEPARABLE_COMPILATION OFF)
  set_property(TARGET ooverlap_gemm PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS OFF)

  ooverlap_public_include_dirs(ooverlap_gemm)

  target_include_directories(ooverlap_gemm PUBLIC
    ${PROJECT_SOURCE_DIR}/src/third-party/cutlass/include
    ${PROJECT_SOURCE_DIR}/src/third-party/cutlass/tools/util/include
  )

  if(OOVERLAP_NEED_TORCH)
    ooverlap_apply_torch(ooverlap_gemm)
  endif()

  target_compile_definitions(ooverlap_gemm PRIVATE NDEBUG)
  target_compile_options(ooverlap_gemm PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3>
  )

  target_link_libraries(ooverlap_gemm PUBLIC
    cutlass
    CUDA::cudart
    CUDA::cuda_driver
  )
endif()

# -------------------------
# Test / benchmark helper library
# -------------------------

if(OOVERLAP_BUILD_TESTS OR OOVERLAP_BUILD_PYTHON)
  set(OOVERLAP_TEST_SOURCES
    ${OOVERLAP_SRC_DIR}/test/persistent_external_p2p_collective_2gpu_sm90.cu
    ${OOVERLAP_SRC_DIR}/test/external_p2p_collective_sweep_sm90.cu
    ${OOVERLAP_SRC_DIR}/test/host_mapped_ready_microtest.cu
    ${OOVERLAP_SRC_DIR}/test/ipc_collective_sm90.cu
    ${OOVERLAP_SRC_DIR}/test/tma_bandwidth_experiment_sm90.cu
    ${OOVERLAP_SRC_DIR}/test/ipc_external_p2p_collective_2gpu_sm90.cu
    ${OOVERLAP_SRC_DIR}/test/tma_collective_sweep_2gpu.cu
  )

  foreach(test_src IN LISTS OOVERLAP_TEST_SOURCES)
    if(NOT EXISTS ${test_src})
      message(FATAL_ERROR "Required test/benchmark source not found: ${test_src}")
    endif()
  endforeach()

  add_library(ooverlap_tests STATIC ${OOVERLAP_TEST_SOURCES})
  add_library(ooverlap::tests ALIAS ooverlap_tests)

  ooverlap_force_sm90a(ooverlap_tests)
  ooverlap_enable_pic(ooverlap_tests)

  ooverlap_enable_release_opts(ooverlap_tests)
  set_property(TARGET ooverlap_tests PROPERTY CUDA_SEPARABLE_COMPILATION ON)
  set_property(TARGET ooverlap_tests PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)

  ooverlap_public_include_dirs(ooverlap_tests)
  target_include_directories(ooverlap_tests BEFORE PUBLIC ${NCCL_INCLUDE_DIR})

  target_link_libraries(ooverlap_tests PUBLIC
    ooverlap_comm
    CUDA::cudart
    CUDA::cuda_driver
    NCCL::NCCL
  )

  if(TARGET nlohmann_json::nlohmann_json)
    target_link_libraries(ooverlap_tests PUBLIC nlohmann_json::nlohmann_json)
  endif()

  ooverlap_link_cudadevrt(ooverlap_tests PUBLIC)
endif()

# -------------------------
# RMSNorm library
# -------------------------

if(OOVERLAP_BUILD_RMSNORM)
  file(GLOB_RECURSE OOVERLAP_NORM_SOURCES CONFIGURE_DEPENDS
    ${OOVERLAP_SRC_DIR}/rmsnorm/*.cu
    ${OOVERLAP_SRC_DIR}/rmsnorm/*.cpp
    ${OOVERLAP_SRC_DIR}/rmsnorm/*.cc
  )

  if(OOVERLAP_NORM_SOURCES)
    add_library(ooverlap_norm STATIC ${OOVERLAP_NORM_SOURCES})
    add_library(ooverlap::norm ALIAS ooverlap_norm)

    ooverlap_force_sm90a(ooverlap_norm)
    ooverlap_enable_pic(ooverlap_norm)

    ooverlap_enable_release_opts(ooverlap_norm)
    set_property(TARGET ooverlap_norm PROPERTY CUDA_SEPARABLE_COMPILATION ON)

    ooverlap_public_include_dirs(ooverlap_norm)
    target_include_directories(ooverlap_norm PUBLIC ${NCCL_INCLUDE_DIR})

    ooverlap_apply_torch(ooverlap_norm)

    target_link_libraries(ooverlap_norm PUBLIC
      CUDA::cudart
      CUDA::cuda_driver
    )
  endif()
endif()

# -------------------------
# Python-only support libraries
# -------------------------

if(OOVERLAP_BUILD_PYTHON)
  if(NOT EXISTS ${OOVERLAP_SRC_DIR}/baseline_impl.cu)
    message(FATAL_ERROR "src/baseline_impl.cu not found")
  endif()

  add_library(ooverlap_baseline_impl STATIC ${OOVERLAP_SRC_DIR}/baseline_impl.cu)
  set_source_files_properties(${OOVERLAP_SRC_DIR}/baseline_impl.cu PROPERTIES LANGUAGE CXX)

  ooverlap_enable_pic(ooverlap_baseline_impl)
  ooverlap_enable_release_opts(ooverlap_baseline_impl)
  ooverlap_public_include_dirs(ooverlap_baseline_impl)

  target_include_directories(ooverlap_baseline_impl PUBLIC ${NCCL_INCLUDE_DIR})
  ooverlap_apply_torch(ooverlap_baseline_impl)

  target_link_libraries(ooverlap_baseline_impl PUBLIC
    ooverlap_gemm
    CUDA::cudart
    CUDA::cuda_driver
    CUDA::cublas
    NCCL::NCCL
  )

  if(NOT EXISTS ${OOVERLAP_SRC_DIR}/overlap_impl.cu)
    message(FATAL_ERROR "src/overlap_impl.cu not found")
  endif()

  add_library(ooverlap_overlap_impl STATIC ${OOVERLAP_SRC_DIR}/overlap_impl.cu)

  ooverlap_force_sm90a(ooverlap_overlap_impl)
  ooverlap_enable_pic(ooverlap_overlap_impl)

  ooverlap_enable_release_opts(ooverlap_overlap_impl)
  set_property(TARGET ooverlap_overlap_impl PROPERTY CUDA_SEPARABLE_COMPILATION ON)
  set_property(TARGET ooverlap_overlap_impl PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)

  ooverlap_public_include_dirs(ooverlap_overlap_impl)
  target_include_directories(ooverlap_overlap_impl PUBLIC ${NCCL_INCLUDE_DIR})
  ooverlap_apply_torch(ooverlap_overlap_impl)

  target_link_libraries(ooverlap_overlap_impl PUBLIC
    ooverlap_comm
    ooverlap_gemm
    CUDA::cudart
    CUDA::cuda_driver
    NCCL::NCCL
  )

  ooverlap_link_cudadevrt(ooverlap_overlap_impl PUBLIC)

  if(NOT EXISTS ${OOVERLAP_SRC_DIR}/nccl_utils.cu)
    message(FATAL_ERROR "src/nccl_utils.cu not found")
  endif()

  add_library(ooverlap_nccl_tool STATIC ${OOVERLAP_SRC_DIR}/nccl_utils.cu)
  set_source_files_properties(${OOVERLAP_SRC_DIR}/nccl_utils.cu PROPERTIES LANGUAGE CXX)

  ooverlap_enable_pic(ooverlap_nccl_tool)
  ooverlap_enable_release_opts(ooverlap_nccl_tool)
  ooverlap_public_include_dirs(ooverlap_nccl_tool)

  target_include_directories(ooverlap_nccl_tool PUBLIC ${NCCL_INCLUDE_DIR})
  ooverlap_apply_torch(ooverlap_nccl_tool)

  target_link_libraries(ooverlap_nccl_tool PUBLIC
    CUDA::cudart
    CUDA::cuda_driver
    NCCL::NCCL
  )
endif()

# -------------------------
# Cli for topology
# -------------------------
add_executable(ooverlap_topology_cli
  ${OOVERLAP_SRC_DIR}/topology/topology_cli.cpp
)


ooverlap_enable_release_opts(ooverlap_topology_cli)
target_include_directories(ooverlap_topology_cli PRIVATE
  ${OOVERLAP_SRC_DIR}
  ${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(ooverlap_topology_cli PRIVATE
  ooverlap_topology
  CUDA::cudart
)

# -------------------------
# Minimal PyTorch collective wrapper
# -------------------------
#
# This target is intentionally separate from ooverlap_ext.
# ooverlap_ext is the legacy/demo Python extension and pulls in benchmark,
# GEMM, NCCL-helper, and test sources.  ooverlap_torch_ext is the small
# PyTorch-facing wrapper around the public include/ooverlap/comm.h API.
#
if(OOVERLAP_BUILD_TORCH_COLLECTIVES)
  if(NOT EXISTS ${OOVERLAP_SRC_DIR}/torch_collectives_pybind.cpp)
    message(FATAL_ERROR "src/torch_collectives_pybind.cpp not found")
  endif()

  add_library(ooverlap_torch_ext MODULE
    ${OOVERLAP_SRC_DIR}/torch_collectives_pybind.cpp)

  ooverlap_force_sm90a(ooverlap_torch_ext)
  ooverlap_enable_pic(ooverlap_torch_ext)
  ooverlap_enable_release_opts(ooverlap_torch_ext)

  set_target_properties(ooverlap_torch_ext PROPERTIES
    PREFIX ""
    LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib
    LINKER_LANGUAGE CUDA
  )

  set_property(TARGET ooverlap_torch_ext PROPERTY CUDA_SEPARABLE_COMPILATION ON)
  set_property(TARGET ooverlap_torch_ext PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)

  target_include_directories(ooverlap_torch_ext PRIVATE
    ${PROJECT_SOURCE_DIR}/include
    ${OOVERLAP_SRC_DIR}
    ${Python3_INCLUDE_DIRS}
  )

  target_link_libraries(ooverlap_torch_ext PRIVATE
    ooverlap_comm
    CUDA::cudart
    CUDA::cuda_driver
  )

  ooverlap_apply_torch(ooverlap_torch_ext)

  if(EXISTS "${TORCH_PYTHON_SO}")
    message(STATUS "Linking libtorch_python for ooverlap_torch_ext: ${TORCH_PYTHON_SO}")
    target_link_options(ooverlap_torch_ext PRIVATE "-Wl,--no-as-needed")
    target_link_libraries(ooverlap_torch_ext PRIVATE "${TORCH_PYTHON_SO}")
  else()
    message(FATAL_ERROR "libtorch_python.so not found at: ${TORCH_PYTHON_SO}")
  endif()

  ooverlap_link_cudadevrt(ooverlap_torch_ext PRIVATE)

  if(TORCH_PY_LIBDIR)
    message(STATUS "Torch libdir for ooverlap_torch_ext RPATH: ${TORCH_PY_LIBDIR}")
    set_target_properties(ooverlap_torch_ext PROPERTIES
      BUILD_RPATH "${TORCH_PY_LIBDIR}"
      INSTALL_RPATH "${TORCH_PY_LIBDIR};\$ORIGIN"
    )
    target_link_options(ooverlap_torch_ext PRIVATE "-Wl,-rpath,${TORCH_PY_LIBDIR}")
  endif()
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
    ${OOVERLAP_SRC_DIR}/test/persistent_external_p2p_collective_2gpu_sm90.cu
    ${OOVERLAP_SRC_DIR}/test/external_p2p_collective_sweep_sm90.cu
    ${OOVERLAP_SRC_DIR}/test/host_mapped_ready_microtest.cu
    ${OOVERLAP_SRC_DIR}/test/ipc_collective_sm90.cu
    ${OOVERLAP_SRC_DIR}/test/tma_bandwidth_experiment_sm90.cu
    ${OOVERLAP_SRC_DIR}/test/ipc_external_p2p_collective_2gpu_sm90.cu
    ${OOVERLAP_SRC_DIR}/test/tma_collective_sweep_2gpu.cu
  )
  
  foreach(ext_src IN LISTS OOVERLAP_EXT_SOURCES)
    if(NOT EXISTS ${ext_src})
      message(FATAL_ERROR "Required Python extension source not found: ${ext_src}")
    endif()
  endforeach()
  
  add_library(ooverlap_ext MODULE ${OOVERLAP_EXT_SOURCES})

  ooverlap_force_sm90a(ooverlap_ext)
  ooverlap_enable_pic(ooverlap_ext)

  ooverlap_enable_release_opts(ooverlap_ext)
  set_target_properties(ooverlap_ext PROPERTIES
    PREFIX ""
    LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib
    LINKER_LANGUAGE CUDA
  )

  set_property(TARGET ooverlap_ext PROPERTY CUDA_SEPARABLE_COMPILATION ON)
  set_property(TARGET ooverlap_ext PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)

  target_include_directories(ooverlap_ext PRIVATE
    ${PROJECT_SOURCE_DIR}/include
    ${OOVERLAP_SRC_DIR}
    ${NCCL_INCLUDE_DIR}
    ${Python3_INCLUDE_DIRS}
  )

  target_link_libraries(ooverlap_ext PRIVATE
    ooverlap_system
    ooverlap_comm
    ooverlap_gemm
    ooverlap_baseline_impl
    ooverlap_overlap_impl
    ooverlap_nccl_tool
    CUDA::cudart
    CUDA::cuda_driver
    CUDA::cublas
    NCCL::NCCL
  )

  if(TARGET ooverlap_norm)
    target_link_libraries(ooverlap_ext PRIVATE ooverlap_norm)
  endif()

  if(TARGET nlohmann_json::nlohmann_json)
    target_link_libraries(ooverlap_ext PRIVATE nlohmann_json::nlohmann_json)
  endif()

  ooverlap_apply_torch(ooverlap_ext)

  if(EXISTS "${TORCH_PYTHON_SO}")
    message(STATUS "Linking libtorch_python: ${TORCH_PYTHON_SO}")
    target_link_options(ooverlap_ext PRIVATE "-Wl,--no-as-needed")
    target_link_libraries(ooverlap_ext PRIVATE "${TORCH_PYTHON_SO}")
  else()
    message(FATAL_ERROR "libtorch_python.so not found at: ${TORCH_PYTHON_SO}")
  endif()

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
