cmake_minimum_required(VERSION 3.20)

# -------------------------
# Safety / dependency guards
# -------------------------
if(NOT TARGET CUDA::cudart)
  find_package(CUDAToolkit REQUIRED)
endif()

if(NOT DEFINED TORCH_LIBRARIES)
  find_package(Torch REQUIRED)
endif()

if(NOT TARGET Python3::Python)
  find_package(Python3 REQUIRED COMPONENTS Development)
endif()

# Optional: get Python torch extension include paths
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

# Torch lib dir + libtorch_python path (needed for at::Tensor pybind caster)
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

  # Python.h for torch headers that touch Python bits (safe to add always)
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

if(NOT DEFINED NCCL_INCLUDE_DIR OR NOT DEFINED NCCL_LIBRARY)
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

# CUTLASS interface target fallback
if(NOT TARGET cutlass)
  add_library(cutlass INTERFACE)
  target_include_directories(cutlass INTERFACE
    ${PROJECT_SOURCE_DIR}/src/third-party/cutlass/include
    ${PROJECT_SOURCE_DIR}/src/third-party/cutlass/tools/util/include
  )
endif()

set(OOVERLAP_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR})

set(OOVERLAP_SM90A_GENCODE "--generate-code=arch=compute_90a,code=sm_90a")
set(OOVERLAP_SM90A_PTX     "--generate-code=arch=compute_90a,code=compute_90a")

function(ooverlap_force_sm90a target)
  if(TARGET ${target})
    # Important: disable any default CMake/NVCC architecture selection
    # (for example sm_89 coming from cache/toolchain/default GPU detection).
    set_target_properties(${target} PROPERTIES CUDA_ARCHITECTURES OFF)

    target_compile_options(${target} PRIVATE
      $<$<COMPILE_LANGUAGE:CUDA>:${OOVERLAP_SM90A_GENCODE}>
      $<$<COMPILE_LANGUAGE:CUDA>:${OOVERLAP_SM90A_PTX}>
    )

    target_link_options(${target} PRIVATE
      $<$<LINK_LANGUAGE:CUDA>:${OOVERLAP_SM90A_GENCODE}>
      $<$<LINK_LANGUAGE:CUDA>:${OOVERLAP_SM90A_PTX}>
    )
  endif()
endfunction()

# -------------------------
# gemm library
# -------------------------
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

  target_link_libraries(ooverlap_gemm PUBLIC cutlass CUDA::cudart)
endif()

# -------------------------
# overlap library (signal/reorder kernels, sm80/sm90, etc.)
# -------------------------
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

  target_include_directories(ooverlap_overlap PUBLIC
    ${PROJECT_SOURCE_DIR}/include
    ${OOVERLAP_SRC_DIR}
    ${PROJECT_SOURCE_DIR}/src/third-party/cutlass/include
    ${PROJECT_SOURCE_DIR}/src/third-party/cutlass/tools/util/include
  )

  target_link_libraries(ooverlap_overlap PUBLIC
    cutlass
    CUDA::cudart
    CUDA::cuda_driver
  )
endif()

# -------------------------
# rmsnorm library (optional)
# -------------------------
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
  )
endif()

# -------------------------
# baseline_impl (compile as CXX to avoid NVCC+ATen issues)
# -------------------------
if(EXISTS ${OOVERLAP_SRC_DIR}/baseline_impl.cu)
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

  target_link_libraries(ooverlap_baseline_impl PUBLIC CUDA::cudart CUDA::cublas ${NCCL_LIBRARY})
endif()

# -------------------------
# overlap_impl (needs CUDA because it launches wait/remap kernels)
# -------------------------
if(EXISTS ${OOVERLAP_SRC_DIR}/overlap_impl.cu)
  add_library(ooverlap_overlap_impl STATIC ${OOVERLAP_SRC_DIR}/overlap_impl.cu)
  ooverlap_force_sm90a(ooverlap_overlap_impl)

  set_property(TARGET ooverlap_overlap_impl PROPERTY POSITION_INDEPENDENT_CODE ON)
  set_property(TARGET ooverlap_overlap_impl PROPERTY CUDA_SEPARABLE_COMPILATION ON)

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
endif()

# -------------------------
# nccl_tool (compile as CXX to avoid NVCC+ATen issues)
# -------------------------
if(EXISTS ${OOVERLAP_SRC_DIR}/nccl_utils.cu)
  add_library(ooverlap_nccl_tool STATIC ${OOVERLAP_SRC_DIR}/nccl_utils.cu)
  set_source_files_properties(${OOVERLAP_SRC_DIR}/nccl_utils.cu PROPERTIES LANGUAGE CXX)

  set_property(TARGET ooverlap_nccl_tool PROPERTY POSITION_INDEPENDENT_CODE ON)

  target_include_directories(ooverlap_nccl_tool PUBLIC
    ${OOVERLAP_SRC_DIR}
    ${NCCL_INCLUDE_DIR}
    ${PROJECT_SOURCE_DIR}/include
  )

  ooverlap_apply_torch(ooverlap_nccl_tool)
  target_link_libraries(ooverlap_nccl_tool PUBLIC CUDA::cudart ${NCCL_LIBRARY})
endif()

# -------------------------
# Python extension module: ooverlap_ext.so
#   (this is what your test imports)
# -------------------------
if(NOT EXISTS ${OOVERLAP_SRC_DIR}/pybind.cpp)
  message(FATAL_ERROR "src/pybind.cpp not found")
endif()

add_library(ooverlap_ext MODULE ${OOVERLAP_SRC_DIR}/pybind.cpp)

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

# Link internal libs
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

# Torch + Python includes/flags/libs
ooverlap_apply_torch(ooverlap_ext)
target_link_libraries(ooverlap_ext PRIVATE Python3::Python)

# >>> KEY FIX: link libtorch_python so at::Tensor pybind caster is available
if(EXISTS "${TORCH_PYTHON_SO}")
  message(STATUS "Linking libtorch_python: ${TORCH_PYTHON_SO}")
  target_link_options(ooverlap_ext PRIVATE "-Wl,--no-as-needed")
  target_link_libraries(ooverlap_ext PRIVATE "${TORCH_PYTHON_SO}")
else()
  message(FATAL_ERROR "libtorch_python.so not found at: ${TORCH_PYTHON_SO}")
endif()

# External deps
target_link_libraries(ooverlap_ext PRIVATE
  cutlass
  ${TORCH_LIBRARIES}
  CUDA::cudart
  CUDA::cuda_driver
  CUDA::cublas
  ${NCCL_LIBRARY}
)

# RPATH to torch/lib for runtime
if(TORCH_PY_LIBDIR)
  message(STATUS "Torch libdir (RPATH): ${TORCH_PY_LIBDIR}")
  set_target_properties(ooverlap_ext PROPERTIES
    BUILD_RPATH "${TORCH_PY_LIBDIR}"
    INSTALL_RPATH "${TORCH_PY_LIBDIR};\$ORIGIN"
  )
  target_link_options(ooverlap_ext PRIVATE "-Wl,-rpath,${TORCH_PY_LIBDIR}")
endif()
