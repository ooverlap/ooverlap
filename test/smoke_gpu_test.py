# smoke_single_gpu.py
import glob, torch

so = glob.glob("../build/*.so")[0]
torch.ops.load_library(so)

torch.cuda.set_device(0)

base = torch.classes.ooverlap_class.BaselineImpl()
base.cublas_init()

M, N, K = 1024, 2048, 1024
A = torch.randn((M, K), device="cuda", dtype=torch.float16)
B = torch.randn((N, K), device="cuda", dtype=torch.float16)  # IMPORTANT: B is [N,K]
C = torch.empty((M, N), device="cuda", dtype=torch.float16)

base.gemm(A, B, C)

# reference: A @ B^T
Cref = A @ B.t()

err = (C - Cref).abs().max().item()
print("max_abs_err:", err)
