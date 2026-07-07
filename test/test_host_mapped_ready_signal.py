import argparse
import importlib.util
from pathlib import Path


def load_ooverlap_ext():
    root = Path(__file__).resolve().parents[1]
    so = root / "build" / "lib" / "ooverlap_ext.so"
    if not so.exists():
        raise FileNotFoundError(f"Could not find {so}. Build first.")

    spec = importlib.util.spec_from_file_location("ooverlap_ext", str(so))
    mod = importlib.util.module_from_spec(spec)

    if spec.loader is None:
        raise RuntimeError(f"Could not load extension spec for {so}")

    spec.loader.exec_module(mod)
    return mod


def main():
    parser = argparse.ArgumentParser("HostMapped ready signal roundtrip test")
    parser.add_argument("--dev-publish", type=int, required=True)
    parser.add_argument("--dev-wait", type=int, required=True)
    parser.add_argument("--value", type=int, default=7)
    parser.add_argument("--max-iters", type=int, default=100_000_000)
    parser.add_argument("--both-directions", action="store_true")
    args = parser.parse_args()

    ext = load_ooverlap_ext()

    pairs = [(args.dev_publish, args.dev_wait)]

    if args.both_directions:
        pairs.append((args.dev_wait, args.dev_publish))

    for pub, wait in pairs:
        result = ext.host_mapped_ready_signal_roundtrip(
            int(pub),
            int(wait),
            int(args.value),
            int(args.max_iters),
        )

        print(f"\n[HostMapped ready] publish={pub} wait={wait}")
        for key, value in result.items():
            print(f"  {key}: {value}")

        if not result["ok"]:
            raise SystemExit(
                f"HostMapped ready failed for publish={pub}, wait={wait}"
            )

    print("\nPASS host mapped ready signal roundtrip")


if __name__ == "__main__":
    main()
