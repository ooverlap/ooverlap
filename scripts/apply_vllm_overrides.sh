#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${1:-${PYTHON_BIN:-python3}}"

"$PYTHON_BIN" - "$ROOT_DIR" <<'PY'
from __future__ import annotations

import hashlib
import py_compile
import shutil
import sys
from pathlib import Path

import vllm

EXPECTED_VERSION = "0.25.0"

if vllm.__version__ != EXPECTED_VERSION:
    raise SystemExit(
        f"error: expected vLLM {EXPECTED_VERSION}, "
        f"found {vllm.__version__}"
    )

repo_root = Path(sys.argv[1]).resolve()
source_root = (
    repo_root
    / "patches"
    / "vllm"
    / "v0.25.0"
    / "vllm"
)
vllm_root = Path(vllm.__file__).resolve().parent

files = [
    Path("distributed/device_communicators/cuda_communicator.py"),
    Path("distributed/device_communicators/ooverlap_all_reduce.py"),
    Path("model_executor/models/qwen2.py"),
]

# Official upstream vLLM v0.25.0 Git blob IDs.
# These checks prevent accidentally overwriting an unexpected vLLM version.
upstream_blobs = {
    Path("distributed/device_communicators/cuda_communicator.py"):
        "555fd0ec9489c2486e280da629503006e899a443",
    Path("model_executor/models/qwen2.py"):
        "182b9758308dbfe82e7bfa0fe161e40a365bf14a",
}


def git_blob_sha(path: Path) -> str:
    data = path.read_bytes()
    header = f"blob {len(data)}\0".encode()
    return hashlib.sha1(header + data).hexdigest()


for rel in files:
    src = source_root / rel
    dst = vllm_root / rel

    if not src.is_file():
        raise SystemExit(f"error: vendored vLLM override is missing: {src}")

    if not dst.parent.is_dir():
        raise SystemExit(f"error: unexpected vLLM layout: {dst.parent}")

    # Already at exactly the desired override.
    if dst.is_file() and dst.read_bytes() == src.read_bytes():
        print(f"[ok] vLLM override already installed: {rel}")
        py_compile.compile(str(dst), doraise=True)
        continue

    expected_base = upstream_blobs.get(rel)
    if expected_base is not None and dst.is_file():
        actual_base = git_blob_sha(dst)

        # Accept either pristine upstream v0.25.0 or a previous OOVERLAP
        # override from an earlier setup run.
        if actual_base != expected_base:
            current = dst.read_bytes()
            if b"OOVERLAP_" not in current:
                raise SystemExit(
                    "error: refusing to overwrite unexpected vLLM file:\n"
                    f"  file: {dst}\n"
                    f"  expected upstream blob: {expected_base}\n"
                    f"  actual blob:            {actual_base}"
                )

    shutil.copyfile(src, dst)
    dst.chmod(0o644)
    py_compile.compile(str(dst), doraise=True)

    print(f"[patch] {rel}")

print(f"[info] patched vLLM {vllm.__version__} at {vllm_root}")
PY
