#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

CUTLASS_DIR="$ROOT_DIR/src/third-party/cutlass"
PATCH_FILE="$ROOT_DIR/patches/cutlass/v3.9.0-tma-signalling.patch"
CUTLASS_BASE="e94e888df3551224738bfa505787b515eae8352f"

[[ -d "$CUTLASS_DIR" ]] || {
  echo "error: CUTLASS submodule is missing" >&2
  exit 1
}

[[ -f "$PATCH_FILE" ]] || {
  echo "error: CUTLASS patch is missing: $PATCH_FILE" >&2
  exit 1
}

actual_commit="$(git -C "$CUTLASS_DIR" rev-parse HEAD)"

[[ "$actual_commit" == "$CUTLASS_BASE" ]] || {
  echo "error: unexpected CUTLASS revision: $actual_commit" >&2
  echo "expected: $CUTLASS_BASE (v3.9.0)" >&2
  exit 1
}

# Already applied.
if git -C "$CUTLASS_DIR" apply \
     --reverse --check "$PATCH_FILE" >/dev/null 2>&1; then
  echo "CUTLASS patch already applied"
  exit 0
fi

git -C "$CUTLASS_DIR" apply --check "$PATCH_FILE"
git -C "$CUTLASS_DIR" apply "$PATCH_FILE"

echo "Applied CUTLASS v3.9.0 local patch"
