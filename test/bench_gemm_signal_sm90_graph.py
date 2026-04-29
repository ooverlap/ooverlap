import argparse
import importlib.util
from pathlib import Path

import torch


def load_ooverlap_ext():
    root = Path(__file__).resolve().parents[1]
    so = root / "build" / "lib" / "ooverlap_ext.so"
    if not so.exists():
        raise FileNotFoundError(f"Could not find {so}. Build first.")
    spec = importlib.util.spec_from_file_location("ooverlap_ext", str(so))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def make_identity_ra(M, N, tile_m=128, tile_n=128, device="cuda"):
    tile_rows = M // tile_m
    tile_cols = N // tile_n
    num_tiles = tile_rows * tile_cols
    return torch.arange(num_tiles, device=device, dtype=torch.int32)


def time_cuda(fn, warmup, iters):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(iters):
        fn()
    end.record()

    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--m", type=int, default=1024)
    ap.add_argument("--n", type=int, default=1024)
    ap.add_argument("--k", type=int, default=4096)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--iters", type=int, default=1000)
    ap.add_argument("--algo", type=int, default=0)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    torch.cuda.set_device(args.device)
    torch.manual_seed(1234)

    ext = load_ooverlap_ext()

    M = args.m
    N = args.n
    K = args.k

    tile_m = 128
    tile_n = 128

    assert M % tile_m == 0
    assert N % tile_n == 0

    tile_rows = M // tile_m
    tile_cols = N // tile_n
    num_tiles = tile_rows * tile_cols

    # Normal output layout:
    #
    #   ReLDN = tile_cols
    #   ldD = ReLDN * 128 = N
    #
    # So C_ours is normal [M, N].
    reldn = tile_cols

    A = torch.randn((M, K), device="cuda", dtype=torch.float16)
    B_ref = torch.randn((K, N), device="cuda", dtype=torch.float16)
    B_packed = B_ref.t().contiguous()

    C_torch = torch.empty((M, N), device="cuda", dtype=torch.float16)
    C_ours = torch.empty((M, N), device="cuda", dtype=torch.float16)

    RA = make_identity_ra(M, N, tile_m, tile_n, device="cuda")
    CommThr = torch.tensor([num_tiles], device="cuda", dtype=torch.int32)
    MM = torch.zeros((1 + num_tiles,), device="cuda", dtype=torch.int32)

    monitor = False

    def run_torch_eager():
        torch.matmul(A, B_ref, out=C_torch)

    def run_ours_eager():
        ext.gemm_signal_sm90(
            A,
            B_packed,
            C_ours,
            MM,
            RA,
            CommThr,
            int(reldn),
            int(args.algo),
            monitor,
        )

    # Correctness check.
    run_torch_eager()
    run_ours_eager()
    torch.cuda.synchronize()

    max_err = (C_ours - C_torch).abs().max().item()

    # Eager timings.
    torch_eager_ms = time_cuda(run_torch_eager, args.warmup, args.iters)
    ours_eager_ms = time_cuda(run_ours_eager, args.warmup, args.iters)

    # Capture torch GEMM.
    torch.cuda.synchronize()
    g_torch = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g_torch):
        torch.matmul(A, B_ref, out=C_torch)

    torch_graph_ms = time_cuda(g_torch.replay, args.warmup, args.iters)

    # Capture our GEMM.
    #
    # If this fails, it means the current pybind/CUTLASS path does something
    # during invocation that is not graph-capture-safe. Then we need a cached
    # C++ object path.
    our_graph_capture_ok = True
    our_graph_ms = None
    try:
        torch.cuda.synchronize()
        g_ours = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g_ours):
            ext.gemm_signal_sm90(
                A,
                B_packed,
                C_ours,
                MM,
                RA,
                CommThr,
                int(reldn),
                int(args.algo),
                monitor,
            )

        our_graph_ms = time_cuda(g_ours.replay, args.warmup, args.iters)

    except Exception as e:
        our_graph_capture_ok = False
        our_graph_error = repr(e)

    flops = 2.0 * M * N * K

    def tflops(ms):
        return flops / (ms * 1.0e-3) / 1.0e12

    print("========================================")
    print("GEMM-only eager vs CUDA Graph benchmark")
    print(f"M={M} N={N} K={K}")
    print(f"tile_rows={tile_rows} tile_cols={tile_cols} num_tiles={num_tiles}")
    print(f"reldn={reldn} normal_output_shape=({M}, {N})")
    print("")
    print(f"max_abs_err:            {max_err}")
    print("")
    print(f"torch eager latency:    {torch_eager_ms:.6f} ms")
    print(f"torch graph latency:    {torch_graph_ms:.6f} ms")
    print(f"our eager latency:      {ours_eager_ms:.6f} ms")

    if our_graph_capture_ok:
        print(f"our graph latency:      {our_graph_ms:.6f} ms")
    else:
        print(f"our graph latency:      CAPTURE FAILED")
        print(f"our graph error:        {our_graph_error}")

    print("")
    print(f"torch eager TFLOP/s:    {tflops(torch_eager_ms):.2f}")
    print(f"torch graph TFLOP/s:    {tflops(torch_graph_ms):.2f}")
    print(f"our eager TFLOP/s:      {tflops(ours_eager_ms):.2f}")

    if our_graph_capture_ok:
        print(f"our graph TFLOP/s:      {tflops(our_graph_ms):.2f}")
        print("")
        print(f"our eager / graph slow: {ours_eager_ms / our_graph_ms:.4f}x")
        print(f"torch graph / our graph speed: {torch_graph_ms / our_graph_ms:.4f}x")
    print("========================================")

    if args.check:
        assert max_err < 0.75, f"max_abs_err too large: {max_err}"


if __name__ == "__main__":
    main()
