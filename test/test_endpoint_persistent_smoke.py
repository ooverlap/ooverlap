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
    parser = argparse.ArgumentParser("Endpoint persistent smoke test")
    parser.add_argument("--numel", type=int, default=1024)
    parser.add_argument("--dev0", type=int, default=0)
    parser.add_argument("--dev1", type=int, default=1)
    args = parser.parse_args()

    assert torch.cuda.is_available(), "torch.cuda.is_available() is False"
    assert torch.cuda.device_count() >= 2, "Need at least 2 GPUs"

    ext = load_ooverlap_ext()
    print("loaded extension")
    print("calling smoke test")
    ok = ext.endpoint_persistent_smoke_test(
        int(args.numel),
        int(args.dev0),
        int(args.dev1),
    )
    print(f"[result] endpoint persistent smoke: {ok}")
    assert ok is True

    torch.cuda.synchronize(args.dev0)
    torch.cuda.synchronize(args.dev1)
    print("PASS ✅ endpoint persistent smoke test")


if __name__ == "__main__":
    main()
