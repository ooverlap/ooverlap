#!/usr/bin/env python3
"""
Automate SM90 ooverlap evaluation.

Flow per shape:
  1. profile configs from CUTLASS CSV with tool/gen_config_sm90.py
  2. for each comm_sm_slack:
       - run tool/search.py with retry list of min-group trials
       - run test/test.py without CTA caps
       - run test/test.py with CTA caps for both NCCL and ooverlap
       - save one scenario JSON
  3. save one merged evaluation JSON

Important resume behavior:
  - Existing results/eval_sm90/scenarios/*.json files are loaded at startup.
  - evaluation_result.json is rebuilt from all loaded scenario JSONs plus new runs.
  - Existing successful scenarios are skipped by default unless
    --rerun-existing-scenarios is passed.

Edit the global arrays below.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# =============================================================================
# Edit these.
# =============================================================================

@dataclass(frozen=True)
class Shape:
    m: int
    n: int
    k: int
    csv: str = ""


@dataclass(frozen=True)
class MinGroupTrial:
    min_group_size: int
    min_effective_group_size: int

    # If not None, skip this trial when M*N is larger than max_mn.
    # This avoids very expensive search cases from small min_group_size.
    max_mn: Optional[int] = None

SHAPES = [
#     Shape(16384, 2048, 1024, "~/csv_out_h100/m16384n2048k1024.gemm.csv"),
    # Shape(16384, 2048, 2048, "~/csv_out_h100/m16384n2048k2048.gemm.csv"),
    # Shape(16384, 2048, 4096, "~/csv_out_h100/m16384n2048k4096.gemm.csv"),
    # Shape(16384, 2048, 8192, "~/csv_out_h100/m16384n2048k8192.gemm.csv"),
    # Shape(16384, 4096, 1024, "~/csv_out_h100/m16384n4096k1024.gemm.csv"),
    # Shape(16384, 4096, 2048, "~/csv_out_h100/m16384n4096k2048.gemm.csv"),
    # Shape(16384, 4096, 4096, "~/csv_out_h100/m16384n4096k4096.gemm.csv"),
    # Shape(16384, 4096, 8192, "~/csv_out_h100/m16384n4096k8192.gemm.csv"),
    # Shape(16384, 8192, 1024, "~/csv_out_h100/m16384n8192k1024.gemm.csv"),
    Shape(16384, 8192, 2048, "~/csv_out_h100/m16384n8192k2048.gemm.csv"),
    Shape(16384, 8192, 4096, "~/csv_out_h100/m16384n8192k4096.gemm.csv"),
    Shape(16384, 8192, 8192, "~/csv_out_h100/m16384n8192k8192.gemm.csv"),

  #   Shape(32768, 2048, 1024, "~/csv_out_h100/m32768n2048k1024.gemm.csv"),
    # Shape(32768, 2048, 2048, "~/csv_out_h100/m32768n2048k2048.gemm.csv"),
    # Shape(32768, 2048, 4096, "~/csv_out_h100/m32768n2048k4096.gemm.csv"),
    # Shape(32768, 2048, 8192, "~/csv_out_h100/m32768n2048k8192.gemm.csv"),
    # Shape(32768, 4096, 1024, "~/csv_out_h100/m32768n4096k1024.gemm.csv"),
    # Shape(32768, 4096, 2048, "~/csv_out_h100/m32768n4096k2048.gemm.csv"),
    # Shape(32768, 4096, 4096, "~/csv_out_h100/m32768n4096k4096.gemm.csv"),
    # Shape(32768, 4096, 8192, "~/csv_out_h100/m32768n4096k8192.gemm.csv"),
    # Shape(32768, 8192, 1024, "~/csv_out_h100/m32768n8192k1024.gemm.csv"),
    Shape(32768, 8192, 2048, "~/csv_out_h100/m32768n8192k2048.gemm.csv"),
    Shape(32768, 8192, 4096, "~/csv_out_h100/m32768n8192k4096.gemm.csv"),
    Shape(32768, 8192, 8192, "~/csv_out_h100/m32768n8192k8192.gemm.csv"),
    
    # Shape(49152, 2048, 2048, "~/csv_out_h100/m49152n2048k2048.gemm.csv"),
    # Shape(49152, 2048, 4096, "~/csv_out_h100/m49152n2048k4096.gemm.csv"),
    # Shape(49152, 2048, 8192, "~/csv_out_h100/m49152n2048k8192.gemm.csv"),
    # Shape(49152, 4096, 1024, "~/csv_out_h100/m49152n4096k1024.gemm.csv"),
    # Shape(49152, 4096, 2048, "~/csv_out_h100/m49152n4096k2048.gemm.csv"),
    # Shape(49152, 4096, 4096, "~/csv_out_h100/m49152n4096k4096.gemm.csv"),
    # Shape(49152, 4096, 8192, "~/csv_out_h100/m49152n4096k8192.gemm.csv"),
    # Shape(49152, 8192, 1024, "~/csv_out_h100/m49152n8192k1024.gemm.csv"),
    # Shape(49152, 8192, 2048, "~/csv_out_h100/m49152n8192k2048.gemm.csv"),
    # Shape(49152, 8192, 4096, "~/csv_out_h100/m49152n8192k4096.gemm.csv"),
    # Shape(49152, 8192, 8192, "~/csv_out_h100/m49152n8192k8192.gemm.csv"),

    # Shape(8192, 2048, 1024, "~/csv_out_h100/m8192n2048k1024.gemm.csv"),
    # Shape(8192, 2048, 2048, "~/csv_out_h100/m8192n2048k2048.gemm.csv"),
    # Shape(8192, 2048, 4096, "~/csv_out_h100/m8192n2048k4096.gemm.csv"),
    # Shape(8192, 2048, 8192, "~/csv_out_h100/m8192n2048k8192.gemm.csv"),
    # Shape(8192, 4096, 1024, "~/csv_out_h100/m8192n4096k1024.gemm.csv"),
    # Shape(8192, 4096, 2048, "~/csv_out_h100/m8192n4096k2048.gemm.csv"),
    # Shape(8192, 4096, 4096, "~/csv_out_h100/m8192n4096k4096.gemm.csv"),
    # Shape(8192, 4096, 8192, "~/csv_out_h100/m8192n4096k8192.gemm.csv"),
    # Shape(8192, 8192, 1024, "~/csv_out_h100/m8192n8192k1024.gemm.csv"),
    # Shape(8192, 8192, 2048, "~/csv_out_h100/m8192n8192k2048.gemm.csv"),
    # Shape(8192, 8192, 4096, "~/csv_out_h100/m8192n8192k4096.gemm.csv"),
    # Shape(8192, 8192, 8192, "~/csv_out_h100/m8192n8192k8192.gemm.csv"),
]

# Used only when Shape.csv is empty.
CSV_DIR = "~/csv_out_h100"
CSV_NAME_TEMPLATE = "m{m}n{n}k{k}.gemm.csv"

COMM_OP = "all_reduce"
COMM_BACKEND = "both"

COMM_SM_SLACKS = [
    4,
    8,
    16,
]

# search.py retries these in order.
# max_mn=None means always allowed.
MIN_GROUP_TRIALS: List[MinGroupTrial] = [
    MinGroupTrial(4, 2, max_mn=8192 * 4096),
    MinGroupTrial(8, 6),
    MinGroupTrial(10, 8),
    MinGroupTrial(12, 10),
    MinGroupTrial(16, 12),
    MinGroupTrial(20, 16),
    MinGroupTrial(24, 20),
]

PROFILE_SCRIPT = "tool/gen_config_sm90.py"
SEARCH_SCRIPT = "tool/search.py"
TEST_SCRIPT = "test/test.py"

PROFILE_TOP_CSV = 100
PROFILE_TOP_SAVE = 30
PROFILE_WARMUP = 50
PROFILE_ITERS = 500

PROFILE_EXTRA_ARGS = [
    "--csv-accum-dtype", "f16",
    "--layout", "packed",
    "--csv-c-layout", "any",
    "--csv-d-layout", "any",
]

SEARCH_EXTRA_ARGS = [
    "--predictive_search",
]

TEST_EXTRA_ARGS = [
    # Add extra test.py args here if needed.
]

# If non-empty, this is injected into test.py env.
# Set to "" if you do not want it.
OOVERLAP_TUNING_POLICY = ""

# If non-empty, overrides CUDA_VISIBLE_DEVICES for all child commands.
# Otherwise the parent environment is preserved.
CUDA_VISIBLE_DEVICES = ""

OUT_DIR = Path("results/eval_sm90")

# Store full stdout in scenario JSON. Usually false because logs are saved separately.
STORE_FULL_STDOUT_IN_JSON = False

# Resume behavior. Existing ok scenario JSONs are skipped by default.
SKIP_EXISTING_OK_SCENARIOS = True


# =============================================================================
# Helpers.
# =============================================================================

def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def now_stamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def shape_id(shape: Shape) -> str:
    return f"m{shape.m}n{shape.n}k{shape.k}"


def scenario_id(shape: Shape, slack: int) -> str:
    return f"{shape_id(shape)}__slack{int(slack)}"


def scenario_json_path(run_dir: Path, sid: str) -> Path:
    return run_dir / "scenarios" / f"{sid}.json"


def load_json_if_exists(path: Path) -> Optional[Any]:
    if not path.exists() or path.stat().st_size == 0:
        return None
    return json.loads(path.read_text())


def write_json(path: Path, data: Any) -> None:
    ensure_dir(path.parent)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def save_scenario(run_dir: Path, scenario: Dict[str, Any]) -> None:
    write_json(
        scenario_json_path(run_dir, str(scenario["scenario_id"])),
        scenario,
    )


def load_existing_scenarios(run_dir: Path) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    scen_dir = run_dir / "scenarios"

    if not scen_dir.exists():
        return out

    for path in sorted(scen_dir.glob("*.json")):
        data = load_json_if_exists(path)
        if not isinstance(data, dict):
            continue

        sid = data.get("scenario_id")
        if not sid:
            sid = path.stem
            data["scenario_id"] = sid

        data["_loaded_from_existing_scenario_json"] = str(path)
        out[str(sid)] = data

    return out


def scenario_sort_key(row: Dict[str, Any]) -> Tuple[int, int, int, int, str]:
    shape = row.get("shape", {})
    return (
        int(shape.get("m", 0) or 0),
        int(shape.get("n", 0) or 0),
        int(shape.get("k", 0) or 0),
        int(row.get("comm_sm_slack", 0) or 0),
        str(row.get("scenario_id", "")),
    )


def sorted_scenarios(scenarios_by_id: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted(scenarios_by_id.values(), key=scenario_sort_key)


def command_to_string(cmd: List[str]) -> str:
    return " ".join(cmd)


def make_env(for_test: bool = False) -> Dict[str, str]:
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    if CUDA_VISIBLE_DEVICES:
        env["CUDA_VISIBLE_DEVICES"] = CUDA_VISIBLE_DEVICES

    if for_test and OOVERLAP_TUNING_POLICY:
        env["OOVERLAP_TUNING_POLICY"] = OOVERLAP_TUNING_POLICY

    return env


def run_cmd(
    cmd: List[str],
    log_path: Path,
    env: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    ensure_dir(log_path.parent)

    print("")
    print("========================================")
    print(command_to_string(cmd))
    print(f"log: {log_path}")
    print("========================================")

    t0 = time.time()

    proc = subprocess.run(
        cmd,
        cwd=str(repo_root()),
        env=env or os.environ.copy(),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    dt = time.time() - t0
    log_path.write_text(proc.stdout)

    print(f"returncode={proc.returncode} elapsed_sec={dt:.2f}")

    return {
        "cmd": cmd,
        "cmd_str": command_to_string(cmd),
        "returncode": int(proc.returncode),
        "elapsed_sec": float(dt),
        "log_path": str(log_path),
        "stdout": proc.stdout if STORE_FULL_STDOUT_IN_JSON else "",
        "stdout_tail": proc.stdout[-8000:],
    }


def get_gpu_name_slug() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(torch.cuda.current_device())
            return props.name.lower().replace(" ", "_")
    except Exception:
        pass

    return ""


def shape_config_path(shape: Shape) -> Path:
    root = repo_root()
    gpu = get_gpu_name_slug()
    config_dir = root / "configs"

    if gpu:
        exact = config_dir / f"{shape_id(shape)}_{gpu}_packed_sm90.json"
        if exact.exists():
            return exact
        return exact

    matches = sorted(
        config_dir.glob(f"{shape_id(shape)}_*_packed_sm90.json"),
        key=lambda p: p.stat().st_mtime,
    )
    if matches:
        return matches[-1]

    return config_dir / f"{shape_id(shape)}_unknown_packed_sm90.json"


def solution_path(shape: Shape, backend: str) -> Path:
    root = repo_root()
    gpu = get_gpu_name_slug()
    config_dir = root / "configs"

    if gpu:
        exact = config_dir / f"solution_{backend}_{shape_id(shape)}_{gpu}_packed_sm90.json"
        if exact.exists():
            return exact
        return exact

    matches = sorted(
        config_dir.glob(f"solution_{backend}_{shape_id(shape)}_*_packed_sm90.json"),
        key=lambda p: p.stat().st_mtime,
    )
    if matches:
        return matches[-1]

    return config_dir / f"solution_{backend}_{shape_id(shape)}_unknown_packed_sm90.json"


def csv_path_for_shape(shape: Shape) -> Path:
    if shape.csv:
        return Path(shape.csv).expanduser().resolve()

    name = CSV_NAME_TEMPLATE.format(m=shape.m, n=shape.n, k=shape.k)
    return Path(CSV_DIR).expanduser().resolve() / name


def config_has_algos(path: Path) -> bool:
    data = load_json_if_exists(path)
    if not isinstance(data, dict):
        return False

    algos = data.get("Algo", [])
    return isinstance(algos, list) and len(algos) > 0


def load_solution_summary(shape: Shape, backend: str) -> Dict[str, Any]:
    path = solution_path(shape, backend)
    data = load_json_if_exists(path)

    if not isinstance(data, dict):
        return {
            "ok": False,
            "path": str(path),
            "error": "solution json missing or invalid",
        }

    return {
        "ok": True,
        "path": str(path),
        "M": data.get("M"),
        "N": data.get("N"),
        "K": data.get("K"),
        "comm_backend": data.get("comm_backend", backend),
        "comm_op": data.get("comm_op"),
        "Algo": data.get("Algo"),
        "BM": data.get("BM"),
        "BN": data.get("BN"),
        "dur": data.get("dur"),
        "cSeg": data.get("cSeg", []),
        "hint_stable_count": data.get("hint_stable_count", len(data.get("hint", []))),
        "unstable_tail_count": data.get("unstable_tail_count"),
        "selected_predicted_latency_ms": data.get("selected_predicted_latency_ms"),
        "searched_latency_ms": data.get("searched_latency_ms"),
        "sm_count": data.get("sm_count"),
        "comm_sm_slack": data.get("comm_sm_slack"),
        "compute_sms": data.get("compute_sms"),
        "bandwidth_path": data.get("bandwidth_path"),
    }


def min_group_trial_allowed(shape: Shape, trial: MinGroupTrial) -> Tuple[bool, str]:
    mn = int(shape.m) * int(shape.n)

    if trial.max_mn is None:
        return True, ""

    if mn > int(trial.max_mn):
        return False, f"M*N={mn} > max_mn={trial.max_mn}"

    return True, ""


def min_group_trial_to_dict(trial: MinGroupTrial) -> Dict[str, Any]:
    return {
        "min_group_size": int(trial.min_group_size),
        "min_effective_group_size": int(trial.min_effective_group_size),
        "max_mn": None if trial.max_mn is None else int(trial.max_mn),
    }


def existing_scenario_is_ok(
    scenarios_by_id: Dict[str, Dict[str, Any]],
    sid: str,
) -> bool:
    old = scenarios_by_id.get(sid)
    return isinstance(old, dict) and old.get("status") == "ok"


# =============================================================================
# Parsing test.py output.
# =============================================================================

_FLOAT_KEYS = {
    "comm_dur_ms",
    "overlap_dur_ms",
    "cublas_baseline_ms",
    "speedup_vs_cublas",
    "plain_baseline_ms",
    "speedup_vs_plain",
}

_STR_KEYS = {
    "backend",
    "comm_op",
}

_TEST_KEYS = _FLOAT_KEYS | _STR_KEYS


def parse_test_stdout(stdout: str) -> Dict[str, Any]:
    """
    Parse per-backend Item/Value sections from test.py output.

    Expected lines:
      backend                                      nccl
      comm_dur_ms                                2.5475
      overlap_dur_ms                             7.4508
      cublas_baseline_ms                         7.0190
      speedup_vs_cublas                          0.9420
      plain_baseline_ms                          7.8297
      speedup_vs_plain                           1.0508
    """

    by_backend: Dict[str, Dict[str, Any]] = {}
    current_backend: Optional[str] = None

    run_re = re.compile(r"#\s*Running\s+backend=(nccl|ooverlap)")
    line_re = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s+(.+?)\s*$")

    for raw in stdout.splitlines():
        line = raw.rstrip()

        m = run_re.search(line)
        if m:
            current_backend = m.group(1)
            by_backend.setdefault(current_backend, {})
            continue

        m = line_re.match(line)
        if not m:
            continue

        key, val = m.group(1), m.group(2).strip()
        if key not in _TEST_KEYS:
            continue

        if key == "backend":
            backend = val.split()[-1]
            if backend in ("nccl", "ooverlap"):
                current_backend = backend
                by_backend.setdefault(current_backend, {})
                by_backend[current_backend]["backend"] = backend
            continue

        if current_backend is None:
            continue

        by_backend.setdefault(current_backend, {})

        if key in _FLOAT_KEYS:
            try:
                by_backend[current_backend][key] = float(val.split()[-1])
            except Exception:
                by_backend[current_backend][key] = None
        else:
            by_backend[current_backend][key] = val.split()[-1]

    cross: Dict[str, Optional[float]] = {}

    nccl = by_backend.get("nccl", {})
    oo = by_backend.get("ooverlap", {})

    def ratio(num: Any, den: Any) -> Optional[float]:
        try:
            den_f = float(den)
            if den_f == 0.0:
                return None
            return float(num) / den_f
        except Exception:
            return None

    if nccl and oo:
        cross = {
            "nccl_overlap_vs_ooverlap_cublas_baseline": ratio(
                oo.get("cublas_baseline_ms"),
                nccl.get("overlap_dur_ms"),
            ),
            "ooverlap_overlap_vs_nccl_cublas_baseline": ratio(
                nccl.get("cublas_baseline_ms"),
                oo.get("overlap_dur_ms"),
            ),
            "nccl_overlap_vs_ooverlap_plain_baseline": ratio(
                oo.get("plain_baseline_ms"),
                nccl.get("overlap_dur_ms"),
            ),
            "ooverlap_overlap_vs_nccl_plain_baseline": ratio(
                nccl.get("plain_baseline_ms"),
                oo.get("overlap_dur_ms"),
            ),
            "ooverlap_overlap_vs_nccl_overlap": ratio(
                nccl.get("overlap_dur_ms"),
                oo.get("overlap_dur_ms"),
            ),
        }

    return {
        "by_backend": by_backend,
        "cross": cross,
    }


# =============================================================================
# Pipeline stages.
# =============================================================================

def run_profile(
    shape: Shape,
    run_dir: Path,
    skip_existing_profile: bool,
) -> Dict[str, Any]:
    cfg_path = shape_config_path(shape)

    if skip_existing_profile and config_has_algos(cfg_path):
        print(f"[profile] skip existing config: {cfg_path}")
        return {
            "skipped": True,
            "ok": True,
            "config_path": str(cfg_path),
            "reason": "existing config has Algo entries",
        }

    csv_path = csv_path_for_shape(shape)
    if not csv_path.exists():
        return {
            "skipped": False,
            "ok": False,
            "config_path": str(cfg_path),
            "csv": str(csv_path),
            "error": "csv file does not exist",
        }

    cmd = [
        sys.executable,
        PROFILE_SCRIPT,
        "--m", str(shape.m),
        "--n", str(shape.n),
        "--k", str(shape.k),
        "--csv", str(csv_path),
        "--top-csv", str(PROFILE_TOP_CSV),
        "--top-save", str(PROFILE_TOP_SAVE),
        "--warmup", str(PROFILE_WARMUP),
        "--iters", str(PROFILE_ITERS),
        *PROFILE_EXTRA_ARGS,
    ]

    res = run_cmd(
        cmd,
        run_dir / "logs" / f"{shape_id(shape)}__profile.log",
        env=make_env(for_test=False),
    )

    cfg_path = shape_config_path(shape)
    ok = res["returncode"] == 0 and config_has_algos(cfg_path)

    return {
        "skipped": False,
        "ok": bool(ok),
        "config_path": str(cfg_path),
        "csv": str(csv_path),
        "run": res,
        "error": "" if ok else "profile command failed or config has no Algo entries",
    }


def run_search_trial(
    shape: Shape,
    slack: int,
    trial_cfg: MinGroupTrial,
    run_dir: Path,
) -> Dict[str, Any]:
    trial_name = (
        f"{shape_id(shape)}__slack{slack}"
        f"__mg{trial_cfg.min_group_size}"
        f"__meg{trial_cfg.min_effective_group_size}"
    )

    cmd = [
        sys.executable,
        SEARCH_SCRIPT,
        "--m", str(shape.m),
        "--n", str(shape.n),
        "--k", str(shape.k),
        "--comm_op", COMM_OP,
        "--comm_backend", COMM_BACKEND,
        "--comm_sm_slack", str(slack),
        "--min_group_size", str(trial_cfg.min_group_size),
        "--min_effective_group_size", str(trial_cfg.min_effective_group_size),
        *SEARCH_EXTRA_ARGS,
    ]

    res = run_cmd(
        cmd,
        run_dir / "logs" / f"{trial_name}__search.log",
        env=make_env(for_test=False),
    )

    solutions = {
        "nccl": load_solution_summary(shape, "nccl"),
        "ooverlap": load_solution_summary(shape, "ooverlap"),
    }

    solution_ok = (
        solutions["nccl"].get("ok")
        and solutions["ooverlap"].get("ok")
        and int(solutions["nccl"].get("comm_sm_slack", -1)) == int(slack)
        and int(solutions["ooverlap"].get("comm_sm_slack", -1)) == int(slack)
    )

    ok = res["returncode"] == 0 and solution_ok

    return {
        "ok": bool(ok),
        "skipped": False,
        **min_group_trial_to_dict(trial_cfg),
        "run": res,
        "solutions": solutions,
        "error": "" if ok else "search failed or backend solution JSONs are missing/stale",
    }


def run_search_with_retries(shape: Shape, slack: int, run_dir: Path) -> Dict[str, Any]:
    trials = []

    for trial_cfg in MIN_GROUP_TRIALS:
        allowed, reason = min_group_trial_allowed(shape, trial_cfg)

        if not allowed:
            trial = {
                "ok": False,
                "skipped": True,
                "skip_reason": reason,
                **min_group_trial_to_dict(trial_cfg),
            }
            trials.append(trial)

            print(
                f"[search] skip min_group={trial_cfg.min_group_size} "
                f"min_effective={trial_cfg.min_effective_group_size}: {reason}"
            )
            continue

        trial = run_search_trial(shape, slack, trial_cfg, run_dir)
        trials.append(trial)

        if trial["ok"]:
            return {
                "ok": True,
                "selected": min_group_trial_to_dict(trial_cfg),
                "trials": trials,
                "solutions": trial["solutions"],
            }

    last_solutions = {}
    for trial in reversed(trials):
        if isinstance(trial, dict) and "solutions" in trial:
            last_solutions = trial["solutions"]
            break

    return {
        "ok": False,
        "selected": None,
        "trials": trials,
        "solutions": last_solutions,
        "error": "all allowed min_group trials failed or were skipped",
    }


def run_test(
    shape: Shape,
    slack: int,
    capped: bool,
    run_dir: Path,
) -> Dict[str, Any]:
    label = "capped" if capped else "uncapped"

    cmd = [
        sys.executable,
        TEST_SCRIPT,
        "--m", str(shape.m),
        "--n", str(shape.n),
        "--k", str(shape.k),
        "--comm_op", COMM_OP,
        "--comm_backend", COMM_BACKEND,
        *TEST_EXTRA_ARGS,
    ]

    if capped:
        cmd.extend([
            "--set_nccl_comm_ctas_to_comm_sms",
            "--set_ooverlap_comm_ctas_to_comm_sms",
        ])

    res = run_cmd(
        cmd,
        run_dir / "logs" / f"{shape_id(shape)}__slack{slack}__test_{label}.log",
        env=make_env(for_test=True),
    )

    stdout = res["stdout"]
    if not stdout:
        stdout = Path(res["log_path"]).read_text()

    parsed = parse_test_stdout(stdout)

    return {
        "ok": res["returncode"] == 0,
        "capped": bool(capped),
        "run": res,
        "parsed": parsed,
        "error": "" if res["returncode"] == 0 else "test.py failed",
    }


def run_scenario(shape: Shape, slack: int, run_dir: Path) -> Dict[str, Any]:
    sid = scenario_id(shape, slack)

    scenario: Dict[str, Any] = {
        "scenario_id": sid,
        "shape": {
            "m": int(shape.m),
            "n": int(shape.n),
            "k": int(shape.k),
        },
        "comm_op": COMM_OP,
        "comm_backend": COMM_BACKEND,
        "comm_sm_slack": int(slack),
        "status": "running",
        "started_at": now_stamp(),
    }

    save_scenario(run_dir, scenario)

    search = run_search_with_retries(shape, slack, run_dir)
    scenario["search"] = search

    if not search["ok"]:
        scenario["status"] = "failed"
        scenario["failed_stage"] = "search"
        scenario["finished_at"] = now_stamp()
        save_scenario(run_dir, scenario)
        return scenario

    scenario["solutions"] = {
        "nccl": load_solution_summary(shape, "nccl"),
        "ooverlap": load_solution_summary(shape, "ooverlap"),
    }
    save_scenario(run_dir, scenario)

    tests: Dict[str, Any] = {}

    tests["uncapped"] = run_test(shape, slack, capped=False, run_dir=run_dir)
    scenario["tests"] = tests
    save_scenario(run_dir, scenario)

    tests["capped"] = run_test(shape, slack, capped=True, run_dir=run_dir)
    scenario["tests"] = tests

    if not tests["uncapped"]["ok"]:
        scenario["status"] = "failed"
        scenario["failed_stage"] = "test_uncapped"
    elif not tests["capped"]["ok"]:
        scenario["status"] = "failed"
        scenario["failed_stage"] = "test_capped"
    else:
        scenario["status"] = "ok"

    scenario["finished_at"] = now_stamp()
    save_scenario(run_dir, scenario)
    return scenario


# =============================================================================
# Summary / main.
# =============================================================================

def summarize(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    ok = [r for r in results if r.get("status") == "ok"]
    failed = [r for r in results if r.get("status") != "ok"]

    return {
        "scenario_count": len(results),
        "ok_count": len(ok),
        "failed_count": len(failed),
        "failed": [
            {
                "scenario_id": r.get("scenario_id"),
                "failed_stage": r.get("failed_stage"),
                "shape": r.get("shape"),
                "comm_sm_slack": r.get("comm_sm_slack"),
            }
            for r in failed
        ],
    }


def parse_args():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--skip-existing-profile",
        action="store_true",
        help=(
            "Skip tool/gen_config_sm90.py when configs/m..._packed_sm90.json "
            "already exists and has Algo entries. Default is to profile again."
        ),
    )

    ap.add_argument(
        "--rerun-existing-scenarios",
        action="store_true",
        help=(
            "Rerun scenarios even if results/eval_sm90/scenarios/<scenario>.json "
            "already exists with status=ok. Default is to skip existing ok scenarios."
        ),
    )

    ap.add_argument(
        "--out-dir",
        type=str,
        default=str(OUT_DIR),
        help="Directory for logs and JSON outputs.",
    )

    ap.add_argument(
        "--only-shape",
        type=str,
        default="",
        help="Optional filter like m32768n8192k4096.",
    )

    return ap.parse_args()


def main() -> int:
    args = parse_args()

    if not SHAPES:
        raise SystemExit("SHAPES is empty. Edit SHAPES near the top of this script first.")

    run_dir = Path(args.out_dir).resolve()
    ensure_dir(run_dir)
    ensure_dir(run_dir / "logs")
    ensure_dir(run_dir / "scenarios")

    scenarios_by_id = load_existing_scenarios(run_dir)
    print(f"[resume] loaded existing scenarios: {len(scenarios_by_id)}")

    selected_shapes = SHAPES
    if args.only_shape:
        selected_shapes = [s for s in SHAPES if shape_id(s) == args.only_shape]
        if not selected_shapes:
            raise SystemExit(f"No shape matched --only-shape {args.only_shape}")

    meta = {
        "started_at": now_stamp(),
        "repo_root": str(repo_root()),
        "profile_script": PROFILE_SCRIPT,
        "search_script": SEARCH_SCRIPT,
        "test_script": TEST_SCRIPT,
        "comm_op": COMM_OP,
        "comm_backend": COMM_BACKEND,
        "comm_sm_slacks": COMM_SM_SLACKS,
        "min_group_trials": [
            min_group_trial_to_dict(x)
            for x in MIN_GROUP_TRIALS
        ],
        "skip_existing_profile": bool(args.skip_existing_profile),
        "skip_existing_ok_scenarios": bool(SKIP_EXISTING_OK_SCENARIOS and not args.rerun_existing_scenarios),
        "ooverlap_tuning_policy": OOVERLAP_TUNING_POLICY,
        "cuda_visible_devices": CUDA_VISIBLE_DEVICES or os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "out_dir": str(run_dir),
    }

    def write_partial() -> None:
        all_results = sorted_scenarios(scenarios_by_id)
        merged = {
            "meta": meta,
            "summary": summarize(all_results),
            "scenarios": all_results,
            "updated_at": now_stamp(),
        }
        write_json(run_dir / "evaluation_result.partial.json", merged)

    for shape in selected_shapes:
        print("")
        print("################################################################################")
        print(f"# Shape {shape_id(shape)}")
        print("################################################################################")

        shape_sids = [scenario_id(shape, slack) for slack in COMM_SM_SLACKS]
        skip_existing_scenarios = SKIP_EXISTING_OK_SCENARIOS and not args.rerun_existing_scenarios

        if skip_existing_scenarios and all(
            existing_scenario_is_ok(scenarios_by_id, sid)
            for sid in shape_sids
        ):
            print(f"[resume] skip shape {shape_id(shape)}: all slack scenarios already ok")
            write_partial()
            continue

        profile = run_profile(shape, run_dir, args.skip_existing_profile)

        if not profile["ok"]:
            for slack in COMM_SM_SLACKS:
                sid = scenario_id(shape, slack)

                if skip_existing_scenarios and existing_scenario_is_ok(scenarios_by_id, sid):
                    print(f"[resume] keep existing ok scenario: {sid}")
                    continue

                scenario = {
                    "scenario_id": sid,
                    "shape": {"m": shape.m, "n": shape.n, "k": shape.k},
                    "comm_op": COMM_OP,
                    "comm_backend": COMM_BACKEND,
                    "comm_sm_slack": int(slack),
                    "status": "failed",
                    "failed_stage": "profile",
                    "profile": profile,
                    "started_at": now_stamp(),
                    "finished_at": now_stamp(),
                }

                save_scenario(run_dir, scenario)
                scenarios_by_id[sid] = scenario

            write_partial()
            continue

        for slack in COMM_SM_SLACKS:
            sid = scenario_id(shape, slack)

            if skip_existing_scenarios and existing_scenario_is_ok(scenarios_by_id, sid):
                print(f"[resume] skip existing ok scenario: {sid}")
                write_partial()
                continue

            print("")
            print("--------------------------------------------------------------------------------")
            print(f"# Scenario {sid}")
            print("--------------------------------------------------------------------------------")

            scenario = run_scenario(shape, slack, run_dir)
            scenario["profile"] = profile

            save_scenario(run_dir, scenario)
            scenarios_by_id[sid] = scenario

            write_partial()

    all_results = sorted_scenarios(scenarios_by_id)

    final = {
        "meta": meta,
        "summary": summarize(all_results),
        "scenarios": all_results,
        "finished_at": now_stamp(),
    }

    write_json(run_dir / "evaluation_result.json", final)

    print("")
    print("################################################################################")
    print("# DONE")
    print("################################################################################")
    print(json.dumps(final["summary"], indent=2))
    print(f"wrote: {run_dir / 'evaluation_result.json'}")

    return 0 if final["summary"]["failed_count"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
