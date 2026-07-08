#!/usr/bin/env python3
from __future__ import annotations

import argparse
import site
import sys
from pathlib import Path

MARKER = "OOVERLAP_GCC_TYPENAME_WORKAROUND_PATCH"
BACKUP_SUFFIX = ".gcc_typename_workaround.bak"

NLOHMANN_OLD = "static_cast<typename decltype(number_buffer)::difference_type>(n_chars)"
NLOHMANN_NEW = "static_cast<std::ptrdiff_t>(n_chars) /* OOVERLAP_GCC_TYPENAME_WORKAROUND_PATCH */"

TORCH_OLD = "static_cast<typename decltype(impl_->list)::difference_type>(pos)"
TORCH_NEW = "static_cast<std::ptrdiff_t>(pos) /* OOVERLAP_GCC_TYPENAME_WORKAROUND_PATCH */"


def repo_root_from(start: Path) -> Path:
    cur = start.resolve()
    if cur.is_file():
        cur = cur.parent

    for path in [cur, *cur.parents]:
        if (path / ".git").exists() or (path / "src").exists():
            return path

    return cur


def backup_path(path: Path) -> Path:
    return path.with_name(path.name + BACKUP_SUFFIX)


def ensure_cstddef_for_std_ptrdiff_t(text: str) -> str:
    if "#include <cstddef>" in text:
        return text

    if "#pragma once" in text:
        return text.replace("#pragma once", "#pragma once\n\n#include <cstddef>", 1)

    lines = text.splitlines(keepends=True)
    last_include = -1
    for i, line in enumerate(lines[:80]):
        if line.lstrip().startswith("#include"):
            last_include = i

    if last_include >= 0:
        lines.insert(last_include + 1, "#include <cstddef>\n")
        return "".join(lines)

    return "#include <cstddef>\n" + text


def patch_file(path: Path, replacements: list[tuple[str, str]], dry_run: bool) -> bool:
    if not path.exists():
        print(f"[skip] missing: {path}")
        return False

    text = path.read_text()
    original = text
    changed = False

    if MARKER in text:
        print(f"[ok] already patched: {path}")
        return True

    for old, new in replacements:
        if old in text:
            text = text.replace(old, new)
            changed = True

    if not changed:
        print(f"[skip] pattern not found: {path}")
        return False

    text = ensure_cstddef_for_std_ptrdiff_t(text)

    print(f"[patch] {path}")
    if dry_run:
        return True

    bak = backup_path(path)
    if not bak.exists():
        bak.write_text(original)

    path.write_text(text)
    print(f"       backup: {bak}")
    return True


def revert_file(path: Path) -> bool:
    bak = backup_path(path)
    if not bak.exists():
        print(f"[skip] backup not found: {bak}")
        return False

    path.write_text(bak.read_text())
    print(f"[revert] {path} <- {bak}")
    return True


def candidate_nlohmann_paths(root: Path) -> list[Path]:
    candidates = [
        root / "src/third-party/nlohmann_json/include/nlohmann/detail/output/serializer.hpp",
        root / "src/third_party/nlohmann_json/include/nlohmann/detail/output/serializer.hpp",
        root / "third-party/nlohmann_json/include/nlohmann/detail/output/serializer.hpp",
        root / "third_party/nlohmann_json/include/nlohmann/detail/output/serializer.hpp",
    ]

    found: list[Path] = []
    for path in candidates:
        if path.exists() and path not in found:
            found.append(path)

    if found:
        return found

    search_roots = [root / "src", root / "third-party", root / "third_party"]
    for search_root in search_roots:
        if not search_root.exists():
            continue

        for path in search_root.rglob("serializer.hpp"):
            normalized = str(path).replace("\\", "/")
            if "nlohmann" in normalized and "/detail/output/serializer.hpp" in normalized:
                if path not in found:
                    found.append(path)

    return found


def candidate_torch_list_inl_paths(extra_path: str | None) -> list[Path]:
    found: list[Path] = []

    def add(path: Path) -> None:
        if path.exists() and path not in found:
            found.append(path)

    if extra_path:
        add(Path(extra_path).expanduser().resolve())

    try:
        import torch  # type: ignore

        add(
            Path(torch.__file__).resolve().parent
            / "include"
            / "ATen"
            / "core"
            / "List_inl.h"
        )
    except Exception as exc:
        print(f"[warn] could not import torch to locate List_inl.h: {exc}")

    prefixes: list[Path] = []
    raw_prefixes = [sys.prefix, sys.exec_prefix]
    try:
        raw_prefixes.extend(site.getsitepackages())
    except Exception:
        pass
    try:
        raw_prefixes.append(site.getusersitepackages())
    except Exception:
        pass

    for raw in raw_prefixes:
        if raw:
            prefixes.append(Path(raw))

    for prefix in prefixes:
        add(prefix / "lib/python3.13/site-packages/torch/include/ATen/core/List_inl.h")
        add(prefix / "site-packages/torch/include/ATen/core/List_inl.h")

    for entry in sys.path:
        if not entry:
            continue
        add(Path(entry) / "torch/include/ATen/core/List_inl.h")

    return found


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Patch nlohmann_json and Torch headers for GCC/NVCC builds that "
            "reject static_cast<typename decltype(... )::difference_type>(...)."
        )
    )
    parser.add_argument(
        "--repo-root",
        default=None,
        help="Path to repo root. Default: auto-detect from current directory.",
    )
    parser.add_argument(
        "--torch-list-inl",
        default=None,
        help="Explicit path to torch/include/ATen/core/List_inl.h.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--revert", action="store_true")
    args = parser.parse_args()

    root = repo_root_from(Path(args.repo_root or "."))
    print(f"[info] repo root: {root}")

    nlohmann_paths = candidate_nlohmann_paths(root)
    torch_paths = candidate_torch_list_inl_paths(args.torch_list_inl)

    if not nlohmann_paths:
        print("[warn] did not find nlohmann serializer.hpp under repo")
    if not torch_paths:
        print("[warn] did not find Torch ATen/core/List_inl.h")

    ok = True

    if args.revert:
        for path in nlohmann_paths + torch_paths:
            ok = revert_file(path) and ok
        return 0 if ok else 1

    nlohmann_ok = False
    for path in nlohmann_paths:
        nlohmann_ok = patch_file(
            path,
            [(NLOHMANN_OLD, NLOHMANN_NEW)],
            args.dry_run,
        ) or nlohmann_ok

    torch_ok = False
    for path in torch_paths:
        torch_ok = patch_file(
            path,
            [(TORCH_OLD, TORCH_NEW)],
            args.dry_run,
        ) or torch_ok

    if not nlohmann_ok:
        print("[error] nlohmann serializer.hpp was not patched")
        ok = False
    if not torch_ok:
        print("[error] Torch List_inl.h was not patched")
        ok = False

    if ok:
        print("[done] patched headers for GCC typename workaround")
        print("[note] rebuild from a clean enough state, e.g. rerun cmake build or remove stale objects")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
