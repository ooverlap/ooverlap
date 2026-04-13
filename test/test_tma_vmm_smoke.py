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


def main():
    parser = argparse.ArgumentParser("2-GPU VMM + bulk-TMA smoke test")
    parser.add_argument("--num-elements", type=int, default=1 << 20,
                        help="number of fp16 elements to copy")
    parser.add_argument("--src-device", type=int, default=0,
                        help="source GPU id")
    parser.add_argument("--dst-device", type=int, default=1,
                        help="destination GPU id")
    args = parser.parse_args()

    assert torch.cuda.is_available(), "torch.cuda.is_available() is False"
    ndev = torch.cuda.device_count()
    assert ndev >= 2, f"Need at least 2 GPUs, found {ndev}"

    print(f"[info] torch {torch.__version__}")
    print(f"[info] cuda available: {torch.cuda.is_available()}")
    print(f"[info] cuda device count: {ndev}")
    print(f"[info] running VMM+bulk-TMA test: src={args.src_device}, dst={args.dst_device}, "
          f"num_elements={args.num_elements}")

    ext = load_ooverlap_ext()

    # Optional sanity prints
    print("[info] loaded extension:", ext)
    print("[info] has tma_vmm_smoke_test:", hasattr(ext, "tma_vmm_smoke_test"))

    ok = ext.tma_vmm_smoke_test(
        int(args.num_elements),
        int(args.src_device),
        int(args.dst_device),
    )

    torch.cuda.synchronize(args.src_device)
    torch.cuda.synchronize(args.dst_device)

    print("[result] tma_vmm_smoke_test returned:", ok)
    assert ok is True, "Smoke test returned False"

    print("PASS ✅  2-GPU VMM + bulk-TMA smoke test")


if __name__ == "__main__":
    main()
