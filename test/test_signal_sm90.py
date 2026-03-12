import torch
import importlib.util
from pathlib import Path


def load_ooverlap_ext():
    root = Path(__file__).resolve().parents[1]
    so = root / "build" / "lib" / "ooverlap_ext.so"
    if not so.exists():
        raise FileNotFoundError(f"Could not find {so}. Build first.")
    spec = importlib.util.spec_from_file_location("ooverlap_ext", str(so))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    assert torch.cuda.is_available(), "torch.cuda.is_available() is False"
    torch.manual_seed(0)

    ext = load_ooverlap_ext()

    # algo=0 corresponds to TB 128x128, so pick multiples of 128
    M, N, K = 256, 256, 128
    TileM, TileN = 128, 128
    tile_rows = M // TileM
    tile_cols = N // TileN
    num_tiles = tile_rows * tile_cols

    # Identity reorder + no reshape change
    ReLDN = tile_cols

    # A is (M,K) row-major
    A = torch.randn((M, K), device="cuda", dtype=torch.float16)

    # Logical B_ref is (K,N)
    B_ref = torch.randn((K, N), device="cuda", dtype=torch.float16)

    # Pass B as (N,K) row-major, which matches column-major (K,N) storage expected by kernel
    B_packed = B_ref.t().contiguous()

    # Output buffer D: for identity reorder with ReLDN=tile_cols => ldD == N
    D = torch.empty((M, N), device="cuda", dtype=torch.float16)

    # Monitored counters: one segment
    MM = torch.zeros((1,), device="cuda", dtype=torch.int32)

    # Reorder array length = num_tiles (identity)
    RA = torch.arange(num_tiles, device="cuda", dtype=torch.int32)

    # One communication segment containing all tiles
    CommThr = torch.tensor([num_tiles], device="cuda", dtype=torch.int32)

    # Run op (algo=0, monitor=False)
    ext.gemm_signal_sm90(A, B_packed, D, MM, RA, CommThr, int(ReLDN), 0, False)

    torch.cuda.synchronize()

    # Check correctness
    ref = (A.float() @ B_ref.float()).half()
    max_abs = (D - ref).abs().max().item()
    print("max_abs_error:", max_abs)

    # fp16 GEMM tolerance (bring-up)
    assert max_abs < 0.5, f"Too much error: {max_abs}"

    # Check signaling count
    mm0 = int(MM.cpu().item())
    print("MM[0]:", mm0, "expected:", num_tiles)
    assert mm0 == num_tiles, f"Signal count mismatch: {mm0} vs {num_tiles}"

    print("PASS ✅  1-GPU GEMM+signal test")


if __name__ == "__main__":
    main()
