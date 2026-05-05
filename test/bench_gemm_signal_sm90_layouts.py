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


def algo_tile_shape(algo):
    if algo in (0, 1, 2, 3, 4):
        return 128, 128
    raise ValueError(f"Unsupported algo={algo}")


def make_identity_ra(M, N, tile_m, tile_n, device="cuda"):
    tile_rows = M // tile_m
    tile_cols = N // tile_n
    num_tiles = tile_rows * tile_cols
    return torch.arange(num_tiles, device=device, dtype=torch.int32)


def make_column_major_ra(M, N, tile_m, tile_n, device="cuda"):
    tile_rows = M // tile_m
    tile_cols = N // tile_n
    num_tiles = tile_rows * tile_cols

    ra = torch.empty((num_tiles,), device=device, dtype=torch.int32)

    packed_pos = 0
    for tc in range(tile_cols):
        for tr in range(tile_rows):
            logical_tile = tr * tile_cols + tc
            ra[logical_tile] = packed_pos
            packed_pos += 1

    return ra


def unpack_packed_to_normal(D_packed, RA, M, N, reldn, tile_m, tile_n):
    tile_rows = M // tile_m
    tile_cols = N // tile_n
    num_tiles = tile_rows * tile_cols

    D_normal = torch.empty((M, N), device=D_packed.device, dtype=D_packed.dtype)
    ra_cpu = RA.detach().cpu().tolist()

    for logical_tile in range(num_tiles):
        packed_tile = int(ra_cpu[logical_tile])

        logical_tile_row = logical_tile // tile_cols
        logical_tile_col = logical_tile % tile_cols

        packed_tile_row = packed_tile // reldn
        packed_tile_col = packed_tile % reldn

        src_r0 = packed_tile_row * tile_m
        src_c0 = packed_tile_col * tile_n

        dst_r0 = logical_tile_row * tile_m
        dst_c0 = logical_tile_col * tile_n

        D_normal[
            dst_r0 : dst_r0 + tile_m,
            dst_c0 : dst_c0 + tile_n,
        ].copy_(
            D_packed[
                src_r0 : src_r0 + tile_m,
                src_c0 : src_c0 + tile_n,
            ]
        )

    return D_normal


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
    ap.add_argument("--algo", type=int, default=1)
    ap.add_argument("--layout", choices=["normal", "packed"], default="normal")
    ap.add_argument("--ra", choices=["identity", "column_major"], default="identity")
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--iters", type=int, default=1000)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    torch.cuda.set_device(args.device)
    torch.manual_seed(1234)

    ext = load_ooverlap_ext()

    M, N, K = args.m, args.n, args.k
    tile_m, tile_n = algo_tile_shape(args.algo)

    assert M % tile_m == 0
    assert N % tile_n == 0

    tile_rows = M // tile_m
    tile_cols = N // tile_n
    num_tiles = tile_rows * tile_cols

    A = torch.randn((M, K), device="cuda", dtype=torch.float16)
    B_ref = torch.randn((K, N), device="cuda", dtype=torch.float16)
    B_packed = B_ref.t().contiguous()

    C_ref = torch.empty((M, N), device="cuda", dtype=torch.float16)

    if args.ra == "identity":
        RA = make_identity_ra(M, N, tile_m, tile_n, device="cuda")
    else:
        RA = make_column_major_ra(M, N, tile_m, tile_n, device="cuda")

    if args.layout == "normal":
        # Normal row-major layout:
        #   ReLDN = tile_cols
        #   ldD = ReLDN * tile_n = N
        reldn = tile_cols
        C_ours = torch.empty((M, N), device="cuda", dtype=torch.float16)

    else:
        # Packed tile layout:
        #   ReLDN = 1
        #   physical shape = [num_tiles * tile_m, tile_n]
        reldn = 1
        packed_tile_cols = reldn
        packed_tile_rows = (num_tiles + packed_tile_cols - 1) // packed_tile_cols
        packed_M = packed_tile_rows * tile_m
        packed_N = packed_tile_cols * tile_n
        C_ours = torch.empty((packed_M, packed_N), device="cuda", dtype=torch.float16)

    # One segment, enough for GEMM-only signaling metadata.
    CommThr = torch.tensor([num_tiles], device="cuda", dtype=torch.int32)
    MM = torch.zeros((1 + num_tiles,), device="cuda", dtype=torch.int32)

    monitor = False

    def run_ref():
        torch.matmul(A, B_ref, out=C_ref)

    def run_ours():
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

    # Correctness.
    run_ref()
    run_ours()
    torch.cuda.synchronize()

    if args.layout == "normal":
        C_check = C_ours
    else:
        C_check = unpack_packed_to_normal(
            C_ours,
            RA,
            M,
            N,
            reldn,
            tile_m,
            tile_n,
        )
        torch.cuda.synchronize()

    max_err = (C_check - C_ref).abs().max().item()

    # Graph capture for our GEMM.
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
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

    ours_ms = time_cuda(g.replay, args.warmup, args.iters)

    # Graph capture for torch.
    torch.cuda.synchronize()
    g_ref = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g_ref):
        torch.matmul(A, B_ref, out=C_ref)

    torch_ms = time_cuda(g_ref.replay, args.warmup, args.iters)

    flops = 2.0 * M * N * K

    print("========================================")
    print("GEMM layout benchmark")
    print(f"algo={args.algo}")
    print(f"layout={args.layout}")
    print(f"ra={args.ra}")
    print(f"M={M} N={N} K={K}")
    print(f"tile_m={tile_m} tile_n={tile_n}")
    print(f"tile_rows={tile_rows} tile_cols={tile_cols} num_tiles={num_tiles}")
    print(f"reldn={reldn}")
    print("")
    print(f"max_abs_err:       {max_err}")
    print("")
    print(f"torch graph:       {torch_ms:.6f} ms")
    print(f"ours graph:        {ours_ms:.6f} ms")
    print(f"torch TFLOP/s:     {flops / (torch_ms * 1e-3) / 1e12:.2f}")
    print(f"ours TFLOP/s:      {flops / (ours_ms * 1e-3) / 1e12:.2f}")
    print(f"torch/ours speed:  {torch_ms / ours_ms:.4f}x")
    print("========================================")

    if args.check:
        assert max_err < 0.75, f"max_abs_err too large: {max_err}"


if __name__ == "__main__":
    main()
