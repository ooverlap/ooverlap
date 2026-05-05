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


def make_identity_ra(M, N, tile_m=128, tile_n=128, device="cuda"):
    assert M % tile_m == 0
    assert N % tile_n == 0
    tile_rows = M // tile_m
    tile_cols = N // tile_n
    num_tiles = tile_rows * tile_cols
    return torch.arange(num_tiles, device=device, dtype=torch.int32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--m", type=int, default=1024)
    ap.add_argument("--n", type=int, default=1024)
    ap.add_argument("--k", type=int, default=4096)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--iters", type=int, default=500)
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

    assert M % tile_m == 0, "M must be multiple of 128"
    assert N % tile_n == 0, "N must be multiple of 128"

    tile_rows = M // tile_m
    tile_cols = N // tile_n
    num_tiles = tile_rows * tile_cols

    # IMPORTANT:
    # For this diagnostic, use normal output layout.
    #
    # ReLDN = tile_cols
    #
    # Since:
    #   ldD = ReLDN * TileN = tile_cols * 128 = N
    #
    # So D_ours shape [M, N] is a normal row-major output.
    reldn = tile_cols

    A = torch.randn((M, K), device="cuda", dtype=torch.float16)

    # Logical B for torch:
    #   B_ref shape = [K, N]
    #
    # Our CUTLASS wrapper expects:
    #   B_packed shape = [N, K]
    #
    # because the kernel treats B as column-major KxN.
    B_ref = torch.randn((K, N), device="cuda", dtype=torch.float16)
    B_packed = B_ref.t().contiguous()

    C_torch = torch.empty((M, N), device="cuda", dtype=torch.float16)
    C_ours = torch.empty((M, N), device="cuda", dtype=torch.float16)

    RA = make_identity_ra(M, N, tile_m, tile_n, device="cuda")

    # If signaling code is enabled, CommThr=[num_tiles] means one segment.
    # If signaling code is commented out, these are harmless.
    CommThr = torch.tensor([num_tiles], device="cuda", dtype=torch.int32)

    # Layout expected by original signaling path:
    #   MM[0]                  = segment counter
    #   MM[1 : 1 + num_tiles]  = per-tile counters
    MM = torch.zeros((1 + num_tiles,), device="cuda", dtype=torch.int32)

    monitor = False

    def run_torch_gemm():
        torch.matmul(A, B_ref, out=C_torch)

    def run_our_gemm():
        # Clear MM only if signaling code is enabled.
        # If store_tail is commented out, this is just a tiny extra kernel.
        # For the purest measurement, set --no-mm-zero by editing this out.
        MM.zero_()

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

    # Correctness check before timing.
    run_torch_gemm()
    run_our_gemm()
    torch.cuda.synchronize()

    max_err = (C_ours - C_torch).abs().max().item()

    torch_ms = time_cuda(run_torch_gemm, args.warmup, args.iters)
    ours_ms = time_cuda(run_our_gemm, args.warmup, args.iters)

    # GEMM FLOPs: 2*M*N*K
    flops = 2.0 * M * N * K
    torch_tflops = flops / (torch_ms * 1.0e-3) / 1.0e12
    ours_tflops = flops / (ours_ms * 1.0e-3) / 1.0e12

    print("========================================")
    print("GEMM-only benchmark")
    print(f"M={M} N={N} K={K}")
    print(f"tile_rows={tile_rows} tile_cols={tile_cols} num_tiles={num_tiles}")
    print(f"reldn={reldn} normal_output_shape=({M}, {N})")
    print("")
    print(f"max_abs_err:        {max_err}")
    print("")
    print(f"torch GEMM latency: {torch_ms:.6f} ms")
    print(f"our GEMM latency:   {ours_ms:.6f} ms")
    print("")
    print(f"torch GEMM TFLOP/s: {torch_tflops:.2f}")
    print(f"our GEMM TFLOP/s:   {ours_tflops:.2f}")
    print("")
    if ours_ms > 0:
        print(f"torch / ours speed: {torch_ms / ours_ms:.4f}x")
        print(f"ours / torch slow:  {ours_ms / torch_ms:.4f}x")
    print("========================================")

    if args.check:
        assert max_err < 0.75, f"max_abs_err too large: {max_err}"


if __name__ == "__main__":
    main()
