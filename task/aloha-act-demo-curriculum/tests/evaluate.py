#!/usr/bin/env python3
"""Judge for rsi/aloha-act-demo-curriculum.

Run by tests/test.sh as ``python -I /tests/evaluate.py`` in a fresh Judge container whose
/workspace is a read-only snapshot of Work. Steps:

1. preflight + fixed-asset integrity (hashes and package versions from the build manifest)
2. stage the candidate (policy.json, optional scores.csv) with stable single reads
3. validate and compile the candidate into the fixed-budget exposure schedule
4. train ACT from scratch with the fixed recipe (subprocess, cuda:0)
5. evaluate the checkpoint on the hidden seed block in gym-aloha (subprocess)
6. write /logs/verifier/reward.json  (reward = hidden success rate in [0, 1])

Exit codes: 0 scored; 2 candidate invalid (no reward, agent feedback explains why);
1 infrastructure/run error (no reward). Everything printed to stdout is agent feedback;
hidden seeds and per-episode seeds are never printed.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path

sys.path.insert(0, "/tests")
import bootstrap  # noqa: E402

LOGS = Path("/logs/verifier")
HIDDEN = Path("/tests/hidden_seeds.json")
TRAIN_TIMEOUT_S = 6 * 3600
EVAL_TIMEOUT_S = 2 * 3600


def emit(obj: dict) -> None:
    print(json.dumps(obj, sort_keys=True), flush=True)


def _tail(path: Path, lines: int = 40) -> str:
    try:
        data = path.read_text(errors="replace").splitlines()
    except OSError:
        return ""
    return "\n".join(data[-lines:])


def run_subprocess(name: str, args: list[str], log_path: Path, timeout: int) -> int:
    """Run a task-owned CLI in an isolated interpreter; stream output to a log file only."""
    cmd = [sys.executable, "-I", "-c", f"from aloha_curriculum.cli import {name}; raise SystemExit({name}())", *args]
    with open(log_path, "w", encoding="utf-8") as log:
        try:
            proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, cwd="/", timeout=timeout, check=False)
        except subprocess.TimeoutExpired:
            return -1
    return proc.returncode


def main() -> int:
    started = time.monotonic()
    LOGS.mkdir(parents=True, exist_ok=True)
    home = Path(os.environ.get("HOME", "/run/aloha-judge-home"))
    home.mkdir(parents=True, exist_ok=True)
    try:
        bootstrap.preflight()
        manifest = bootstrap.verify_manifest()
    except bootstrap.InfrastructureError as exc:
        emit({"event": "infrastructure_error", "stage": "preflight", "detail": str(exc)})
        return 1
    protocol = json.loads(bootstrap.PROTOCOL.read_text())
    hidden = json.loads(HIDDEN.read_text())
    emit({"event": "judge_started", "task": "aloha-act-demo-curriculum", "protocol_version": protocol["protocol_version"],
          "steps": protocol["training"]["steps"], "batch_size": protocol["training"]["batch_size"],
          "hidden_episodes": hidden["n_episodes"], "lerobot": manifest["packages"].get("lerobot")})

    work = Path(tempfile.mkdtemp(prefix="aloha-judge-"))
    staging = work / "candidate"
    try:
        staged = bootstrap.stage_candidate(staging)
    except bootstrap.CandidateError as exc:
        emit({"event": "candidate_invalid", "stage": "workspace", "detail": str(exc)})
        return 2
    emit({"event": "candidate_staged", "files": staged["files"], "sizes": staged["sizes"],
          "policy_sha256": staged["policy_sha256"], "scores_sha256": staged["scores_sha256"]})

    # Validate + compile in-process (pure numpy/pandas on task-owned inputs).
    from aloha_curriculum import compile as compile_mod
    from aloha_curriculum.cli import load_candidate
    from aloha_curriculum.errors import CandidateInvalid

    try:
        policy, features, scores, meta = load_candidate(staging, bootstrap.FEATURES, protocol)
        proto = compile_mod.protocol_from_dict(protocol)
        schedule, summary = compile_mod.compile_schedule(policy, features, scores, proto)
    except CandidateInvalid as exc:
        emit({"event": "candidate_invalid", "stage": "validate_compile", "detail": str(exc)})
        return 2
    except Exception as exc:  # noqa: BLE001
        emit({"event": "infrastructure_error", "stage": "validate_compile", "detail": f"{type(exc).__name__}: {exc}"})
        traceback.print_exc()
        return 1
    if schedule.size != proto.exposures:
        emit({"event": "infrastructure_error", "stage": "validate_compile", "detail": "schedule length mismatch"})
        return 1
    sched_dir = work / "schedule"
    compile_mod.save_schedule(schedule, summary, sched_dir)
    emit({"event": "schedule_compiled", **{k: summary[k] for k in (
        "policy_name", "exposures", "steps", "batch_size", "schedule_sha256", "distinct_frames", "distinct_frames_fraction",
        "max_frame_exposures", "episodes_used", "episode_exposures", "t_norm_decile_exposures", "chunk_pad_mean",
        "chunk_pad_fraction_padded", "entropy_min_nats", "entropy_mean_nats", "entropy_uniform_nats", "support_min_frames", "stage_blocks")}})

    # Train (fixed recipe, official budget, fixed seed).
    train_dir = work / "train"
    t0 = time.monotonic()
    rc = run_subprocess("train_main", [
        "--schedule", str(sched_dir / "schedule.npy"), "--protocol", str(bootstrap.PROTOCOL),
        "--dataset-root", str(bootstrap.DATASET_ROOT), "--output", str(train_dir), "--device", "cuda",
    ], LOGS / "train.log", TRAIN_TIMEOUT_S)
    train_s = time.monotonic() - t0
    if rc != 0:
        emit({"event": "run_error", "stage": "train", "returncode": rc, "elapsed_s": round(train_s, 1), "log_tail": _tail(LOGS / "train.log")})
        return 1
    info = json.loads((train_dir / "train_info.json").read_text())
    tr = protocol["training"]
    if not (info["official_budget"] and info["steps"] == tr["steps"] and info["batch_size"] == tr["batch_size"]
            and info["exposures"] == tr["steps"] * tr["batch_size"] and info["seed"] == tr["seed"]):
        emit({"event": "infrastructure_error", "stage": "train", "detail": "training accounting does not match the fixed protocol"})
        return 1
    history = info.get("history", [])
    stride = max(1, len(history) // 100)
    emit({"event": "train_complete", "steps": info["steps"], "exposures": info["exposures"], "seed": info["seed"],
          "train_s": round(info["train_s"], 1), "steps_per_s": round(info["steps_per_s"], 3), "peak_gpu_mem_gb": round(info["peak_gpu_mem_gb"], 2),
          "final_loss_mean_last_1000": round(info["final_loss_mean_last_1000"], 5),
          "final_l1_mean_last_1000": round(info["final_l1_mean_last_1000"], 5),
          "final_kl_mean_last_1000": round(info["final_kl_mean_last_1000"], 5),
          "loss_history": [{"step": h["step"], "loss": round(h["loss"], 5), "l1": round(h["l1_loss"], 5), "kl": round(h["kl_loss"], 5)} for h in history[::stride]]})

    # Evaluate on the hidden seed block.
    eval_dir = work / "eval"
    t0 = time.monotonic()
    rc = run_subprocess("eval_main", [
        "--checkpoint", str(train_dir / "pretrained_model"), "--protocol", str(bootstrap.PROTOCOL), "--output", str(eval_dir),
        "--start-seed", str(int(hidden["start_seed"])), "--n-episodes", str(int(hidden["n_episodes"])),
        "--n-envs", str(int(hidden["n_envs"])), "--device", "cuda",
    ], LOGS / "eval.log", EVAL_TIMEOUT_S)
    eval_s = time.monotonic() - t0
    if rc != 0:
        emit({"event": "run_error", "stage": "eval", "returncode": rc, "elapsed_s": round(eval_s, 1), "log_tail": _tail(LOGS / "eval.log")})
        return 1
    result = json.loads((eval_dir / "eval_result.json").read_text())
    if result["n_episodes"] != hidden["n_episodes"] or result["start_seed"] != hidden["start_seed"]:
        emit({"event": "infrastructure_error", "stage": "eval", "detail": "evaluation accounting does not match the hidden protocol"})
        return 1
    successes = int(result["successes"])
    n = int(result["n_episodes"])
    reward = successes / n
    if abs(reward - float(result["success_rate"])) > 1e-12:
        emit({"event": "infrastructure_error", "stage": "eval", "detail": "success rate inconsistent"})
        return 1
    # Operator-only artefacts (never printed): full eval result with per-episode seeds.
    (LOGS / "eval_result.json").write_text(json.dumps(result, indent=2) + "\n")
    (LOGS / "schedule_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    (LOGS / "train_info.json").write_text(json.dumps(info, indent=2) + "\n")

    payload = {
        "reward": reward,
        "successes": successes,
        "n_episodes": n,
        "success_rate_se": result["success_rate_se"],
        "avg_max_reward": result["avg_max_reward"],
        "max_reward_histogram": result["max_reward_histogram"],
        "train_s": round(info["train_s"], 1),
        "eval_s": round(result["eval_s"], 1),
        "gl_renderer": result.get("gl_renderer", "unknown"),
        "schedule_sha256": summary["schedule_sha256"],
        "policy_sha256": staged["policy_sha256"],
        "elapsed_s": round(time.monotonic() - started, 1),
    }
    emit({"event": "complete", **payload})
    with open(LOGS / "reward.json", "x", encoding="utf-8") as stream:
        json.dump({"reward": reward, "successes": successes, "n_episodes": n, "train_s": payload["train_s"], "eval_s": payload["eval_s"]}, stream)
    return 0


if __name__ == "__main__":
    try:
        code = main()
    except Exception as exc:  # noqa: BLE001
        emit({"event": "infrastructure_error", "stage": "unhandled", "detail": f"{type(exc).__name__}: {exc}"})
        traceback.print_exc()
        code = 1
    raise SystemExit(code)
