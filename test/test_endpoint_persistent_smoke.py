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
    parser.add_argument("--devices", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--timeout-ms", type=int, default=5000)
    args = parser.parse_args()

    assert torch.cuda.is_available(), "torch.cuda.is_available() is False"
    assert len(args.devices) >= 2, "Need at least 2 devices"

    device_count = torch.cuda.device_count()
    for d in args.devices:
        assert 0 <= d < device_count, f"Invalid CUDA device id: {d}"

    ext = load_ooverlap_ext()
    print("loaded extension")
    print("calling smoke test")
    ok = ext.endpoint_persistent_smoke_test(
        int(args.numel),
        [int(d) for d in args.devices],
        int(args.timeout_ms),
    )
    print(f"[result] endpoint persistent smoke: {ok}")
    assert ok is True

    for d in args.devices:
        torch.cuda.synchronize(d)

    print("PASS ✅ endpoint persistent smoke test")


if __name__ == "__main__":
    main()
