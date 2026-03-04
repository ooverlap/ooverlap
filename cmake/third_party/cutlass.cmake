# cmake/third_party/cutlass.cmake

add_library(cutlass INTERFACE)

# CUTLASS headers live here
target_include_directories(cutlass INTERFACE
  ${CMAKE_CURRENT_LIST_DIR}/../../src/third-party/cutlass/include
  ${CMAKE_CURRENT_LIST_DIR}/../../src/third-party/cutlass/tools/util/include
)

# CUTLASS typically wants C++17
target_compile_features(cutlass INTERFACE cxx_std_17)

# Helpful CUDA flags for modern template-heavy CUDA code (CUTLASS often needs these)
# These only apply when compiling CUDA sources.
target_compile_options(cutlass INTERFACE
  $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr --expt-extended-lambda>
)
