include(ExternalProject)

set(OOVERLAP_NCCL_SOURCE_DIR
  ${PROJECT_SOURCE_DIR}/src/third-party/nccl
)

set(OOVERLAP_NCCL_BUILD_DIR
  ${CMAKE_BINARY_DIR}/third_party/nccl
)

set(OOVERLAP_NCCL_INCLUDE_DIR
  ${OOVERLAP_NCCL_BUILD_DIR}/include
)

set(OOVERLAP_NCCL_LIBRARY
  ${OOVERLAP_NCCL_BUILD_DIR}/lib/libnccl.so
)

if(NOT EXISTS "${OOVERLAP_NCCL_SOURCE_DIR}/src/nccl.h.in")
  message(FATAL_ERROR
    "Bundled NCCL submodule not found. Run:\n"
    "  git submodule update --init --recursive src/third-party/nccl")
endif()

file(MAKE_DIRECTORY
  ${OOVERLAP_NCCL_INCLUDE_DIR}
  ${OOVERLAP_NCCL_BUILD_DIR}/lib
)

if(CUDAToolkit_ROOT_DIR)
  set(_OOVERLAP_CUDA_HOME ${CUDAToolkit_ROOT_DIR})
elseif(CUDAToolkit_BIN_DIR)
  get_filename_component(_OOVERLAP_CUDA_HOME ${CUDAToolkit_BIN_DIR} DIRECTORY)
else()
  set(_OOVERLAP_CUDA_HOME /usr/local/cuda)
endif()

set(OOVERLAP_NCCL_GENCODE
  "-gencode=arch=compute_90,code=sm_90"
  CACHE STRING
  "NVCC_GENCODE used when building bundled NCCL"
)

ExternalProject_Add(ooverlap_nccl_external
  SOURCE_DIR ${OOVERLAP_NCCL_SOURCE_DIR}
  CONFIGURE_COMMAND ""
  BUILD_COMMAND
    ${CMAKE_MAKE_PROGRAM}
      -C ${OOVERLAP_NCCL_SOURCE_DIR}
      -j
      src.build
      CUDA_HOME=${_OOVERLAP_CUDA_HOME}
      BUILDDIR=${OOVERLAP_NCCL_BUILD_DIR}
      NVCC_GENCODE=${OOVERLAP_NCCL_GENCODE}
  INSTALL_COMMAND ""
  BUILD_BYPRODUCTS ${OOVERLAP_NCCL_LIBRARY}
)

add_library(ooverlap_nccl SHARED IMPORTED GLOBAL)
add_library(NCCL::NCCL ALIAS ooverlap_nccl)

set_target_properties(ooverlap_nccl PROPERTIES
  IMPORTED_LOCATION ${OOVERLAP_NCCL_LIBRARY}
  INTERFACE_INCLUDE_DIRECTORIES ${OOVERLAP_NCCL_INCLUDE_DIR}
)

add_dependencies(ooverlap_nccl ooverlap_nccl_external)

set(NCCL_INCLUDE_DIR ${OOVERLAP_NCCL_INCLUDE_DIR})
set(NCCL_LIBRARY ${OOVERLAP_NCCL_LIBRARY})
