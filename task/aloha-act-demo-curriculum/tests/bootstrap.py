"""Judge bootstrap: environment scrubbing, fixed-asset integrity and candidate staging.

Imported by evaluate.py (both run with ``python -I`` from a clean Judge container).
Everything here is task-owned; nothing under /workspace is ever imported or executed.
"""

from __future__ import annotations

import hashlib
import importlib.metadata as md
import json
import os
import shutil
import sys
from pathlib import Path

PUBLIC = Path("/opt/aloha_public")
ASSETS = Path("/opt/aloha_assets")
MANIFEST = PUBLIC / "manifest.json"
PROTOCOL = PUBLIC / "fixed_train.json"
FEATURES = PUBLIC / "features.parquet"
DATASET_ROOT = ASSETS / "aloha_sim_insertion_human"
WORKSPACE = Path("/workspace")
CANDIDATE_DIR = WORKSPACE / "curriculum"
ALLOWED_WORKSPACE_ENTRIES = {"curriculum", "research"}
ALLOWED_CANDIDATE_FILES = {"policy.json", "scores.csv"}
MAX_POLICY_BYTES = 65_536
MAX_SCORES_BYTES = 8 * 1024 * 1024


class InfrastructureError(RuntimeError):
    """The Judge environment is not in its expected state; no score is produced."""


class CandidateError(RuntimeError):
    """The submission violates the task contract; it is not scoreable."""


def preflight() -> None:
    """Make sure nothing from the workspace can shadow task-owned code."""
    os.environ.pop("PYTHONPATH", None)
    ws = os.path.realpath(WORKSPACE)
    keep = []
    for entry in sys.path:
        real = os.path.realpath(entry) if entry else os.getcwd()
        if real == ws or real.startswith(ws + os.sep):
            continue
        keep.append(entry)
    sys.path[:] = keep
    if not PUBLIC.is_dir() or not ASSETS.is_dir():
        raise InfrastructureError("fixed asset directories are missing")
    for var in ("HF_HUB_OFFLINE", "MUJOCO_GL"):
        if not os.environ.get(var):
            raise InfrastructureError(f"{var} must be set by test.sh")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def verify_manifest() -> dict:
    """Recompute every fixed-asset hash and check package versions against the build manifest."""
    if not MANIFEST.is_file():
        raise InfrastructureError("manifest.json missing")
    manifest = json.loads(MANIFEST.read_text())
    bad: list[str] = []
    for rel, digest in manifest["files"].items():
        path = Path(rel)
        if not path.is_file():
            bad.append(f"missing {rel}")
            continue
        if sha256_file(path) != digest:
            bad.append(f"hash mismatch {rel}")
    for name, version in manifest["packages"].items():
        try:
            found = md.version(name)
        except md.PackageNotFoundError:
            found = "missing"
        if found != version:
            bad.append(f"package {name}: expected {version}, found {found}")
    if bad:
        raise InfrastructureError("fixed assets changed: " + "; ".join(bad[:20]))
    return manifest


def _stable_copy(src: Path, dst: Path, limit: int) -> int:
    """Copy a regular file by reading it once; reject anything that is not a plain file."""
    if src.is_symlink() or not src.is_file():
        raise CandidateError(f"{src} must be a regular file")
    size = src.stat().st_size
    if size > limit:
        raise CandidateError(f"{src.name} is {size} bytes, limit {limit}")
    data = src.read_bytes()
    if len(data) != size:
        raise CandidateError(f"{src.name} changed size while being read")
    dst.write_bytes(data)
    return size


def stage_candidate(staging: Path) -> dict:
    """Validate the workspace layout and copy the candidate into a private directory."""
    if not WORKSPACE.is_dir():
        raise CandidateError("/workspace is missing")
    entries = sorted(p.name for p in WORKSPACE.iterdir())
    extra = [e for e in entries if e not in ALLOWED_WORKSPACE_ENTRIES]
    if extra:
        raise CandidateError(f"/workspace may only contain {sorted(ALLOWED_WORKSPACE_ENTRIES)}; found {extra}")
    if CANDIDATE_DIR.is_symlink() or not CANDIDATE_DIR.is_dir():
        raise CandidateError("/workspace/curriculum must be a directory")
    files = sorted(p.name for p in CANDIDATE_DIR.iterdir())
    extra = [f for f in files if f not in ALLOWED_CANDIDATE_FILES]
    if extra:
        raise CandidateError(f"/workspace/curriculum may only contain {sorted(ALLOWED_CANDIDATE_FILES)}; found {extra}")
    if "policy.json" not in files:
        raise CandidateError("/workspace/curriculum/policy.json is missing")
    staging.mkdir(parents=True, exist_ok=True)
    sizes = {"policy.json": _stable_copy(CANDIDATE_DIR / "policy.json", staging / "policy.json", MAX_POLICY_BYTES)}
    if "scores.csv" in files:
        sizes["scores.csv"] = _stable_copy(CANDIDATE_DIR / "scores.csv", staging / "scores.csv", MAX_SCORES_BYTES)
    return {
        "files": files,
        "sizes": sizes,
        "policy_sha256": sha256_file(staging / "policy.json"),
        "scores_sha256": sha256_file(staging / "scores.csv") if "scores.csv" in files else None,
    }


def clean_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
