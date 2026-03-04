# correctness_allreduce.py
import argparse, glob, torch
import torch.multiprocessing as mp

def worker(rank, world, so_path, nccl_id, M, N, K):
    torch.cuda.set_device(rank)
    torch.ops.load_library(so_path)

    base = torch.classes.ooverlap_class.BaselineImpl()
    base.nccl_init(rank, world, nccl_id)
    base.cublas_init()

    #A = torch.randn((M, K), device="cuda", dtype=torch.float16)
    #B = torch.randn((N, K), device="cuda", dtype=torch.float16)
    A = (0.01 * torch.randn((M,K), device="cuda", dtype=torch.float16))
    B = (0.01 * torch.randn((N,K), device="cuda", dtype=torch.float16))
    C = torch.empty((M, N), device="cuda", dtype=torch.float16)

    # Baseline op
    base.gemm_allreduce(A, B, C)

    # Reference using SAME NCCL communicator (no torch.distributed needed)
    Cref = (A @ B.t()).contiguous()
    base.nccl_allreduce(Cref)

    max_err = (C - Cref).abs().max().item()
    if rank == 0:
        print(f"[allreduce correctness] max_abs_err={max_err}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--so", default=None)
    ap.add_argument("--gpus", type=int, default=2)
    ap.add_argument("--m", type=int, default=2048)
    ap.add_argument("--n", type=int, default=8192)
    ap.add_argument("--k", type=int, default=4096)
    args = ap.parse_args()

    so = args.so or glob.glob("../build/*.so")[0]
    torch.ops.load_library(so)
    nccl_id = torch.ops.ooverlap_op.generate_nccl_id()

    mp.spawn(worker, args=(args.gpus, so, nccl_id, args.m, args.n, args.k),
             nprocs=args.gpus, join=True)

if __name__ == "__main__":
    main()
