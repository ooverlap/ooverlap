# bench_baseline.py
import argparse, glob, statistics, torch
import torch.multiprocessing as mp

def time_op(fn, warmup=20, iters=200):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    s = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    e = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        s[i].record()
        fn()
        e[i].record()
    torch.cuda.synchronize()
    return [s[i].elapsed_time(e[i]) for i in range(iters)]

def worker(rank, world, so_path, nccl_id, op, M, N, K, warmup, iters, out):
    torch.cuda.set_device(rank)
    torch.ops.load_library(so_path)

    base = torch.classes.ooverlap_class.BaselineImpl()
    base.nccl_init(rank, world, nccl_id)
    base.cublas_init()

    A = torch.randn((M, K), device="cuda", dtype=torch.float16)
    B = torch.randn((N, K), device="cuda", dtype=torch.float16)
    C = torch.empty((M, N), device="cuda", dtype=torch.float16)

    if op == "gemm":
        fn = lambda: base.gemm(A, B, C)
    elif op == "allreduce":
        fn = lambda: base.gemm_allreduce(A, B, C)
    elif op == "reducescatter":
        assert M % world == 0, "For this baseline, choose M divisible by world_size"
        D = torch.empty((M // world, N), device="cuda", dtype=torch.float16)
        fn = lambda: base.gemm_reducescatter(A, B, C, D)
    else:
        raise ValueError(op)

    out[rank] = time_op(fn, warmup=warmup, iters=iters)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--so", default=None)
    ap.add_argument("--gpus", type=int, default=2)
    ap.add_argument("--op", choices=["gemm", "allreduce", "reducescatter"], default="allreduce")
    ap.add_argument("--m", type=int, default=2048)
    ap.add_argument("--n", type=int, default=8192)
    ap.add_argument("--k", type=int, default=4096)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--iters", type=int, default=200)
    args = ap.parse_args()

    so = args.so or glob.glob("../build/*.so")[0]
    torch.ops.load_library(so)
    nccl_id = torch.ops.ooverlap_op.generate_nccl_id()

    mgr = mp.Manager()
    out = mgr.dict()

    mp.spawn(worker, args=(args.gpus, so, nccl_id, args.op,
                           args.m, args.n, args.k, args.warmup, args.iters, out),
             nprocs=args.gpus, join=True)

    # max over ranks per iter
    worst = [max(out[r][i] for r in range(args.gpus)) for i in range(args.iters)]
    mean_ms = statistics.mean(worst)
    p50 = statistics.median(worst)
    p90 = statistics.quantiles(worst, n=10)[8]

    t_s = mean_ms / 1e3
    tflops = (2 * args.m * args.n * args.k) / (t_s * 1e12)

    print(f"[bench] op={args.op} world={args.gpus} M={args.m} N={args.n} K={args.k}")
    print(f"[bench] mean={mean_ms:.3f} ms  p50={p50:.3f} ms  p90={p90:.3f} ms")
    print(f"[bench] (2*M*N*K)/time = {tflops:.2f} TFLOPs  (includes comm time if op!=gemm)")

if __name__ == "__main__":
    main()
