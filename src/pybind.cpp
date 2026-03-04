#include "baseline_impl.h"
#include "nccl_utils.h"

#include <torch/extension.h>
#include <torch/script.h>

// Wrappers to match torch custom class calling convention (intrusive_ptr)
template<typename T>
void NcclInitWrapper(const c10::intrusive_ptr<T>& self,
                     const int64_t tp_rank,
                     const int64_t tp_size,
                     const std::vector<int64_t>& tp_id) {
  self->NcclInit(tp_rank, tp_size, tp_id);
}

template<typename T>
void CublasInitWrapper(const c10::intrusive_ptr<T>& self) {
  self->CublasInit();
}

template<typename T>
void GemmAllReduceWrapper(const c10::intrusive_ptr<T>& self,
                          at::Tensor A, at::Tensor B, at::Tensor C) {
  self->GemmAllReduce(A, B, C);
}

template<typename T>
void GemmReduceScatterWrapper(const c10::intrusive_ptr<T>& self,
                              at::Tensor A, at::Tensor B, at::Tensor C, at::Tensor D) {
  self->GemmReduceScatter(A, B, C, D);
}

template<typename T>
void GemmWrapper(const c10::intrusive_ptr<T>& self,
                 at::Tensor A, at::Tensor B, at::Tensor C) {
  self->Gemm(A, B, C);
}

template<typename T>
void NcclAllReduceWrapper(const c10::intrusive_ptr<T>& self, at::Tensor C) {
  self->NcclAllReduce(C);
}

template<typename T>
void NcclReduceScatterWrapper(const c10::intrusive_ptr<T>& self, at::Tensor C) {
  self->NcclReduceScatter(C);
}

TORCH_LIBRARY(ooverlap_class, m) {
  m.class_<BaselineImpl>("BaselineImpl")
    .def(torch::init())
    .def("nccl_init", &NcclInitWrapper<BaselineImpl>)
    .def("cublas_init", &CublasInitWrapper<BaselineImpl>)
    .def("gemm", &GemmWrapper<BaselineImpl>)
    .def("gemm_allreduce", &GemmAllReduceWrapper<BaselineImpl>)
    .def("gemm_reducescatter", &GemmReduceScatterWrapper<BaselineImpl>)
    .def("nccl_allreduce", &NcclAllReduceWrapper<BaselineImpl>)
    .def("nccl_reducescatter", &NcclReduceScatterWrapper<BaselineImpl>);
}

TORCH_LIBRARY(ooverlap_op, m) {
  m.def("generate_nccl_id", &generate_nccl_id);
}
