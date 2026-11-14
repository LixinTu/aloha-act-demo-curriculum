"""Tests for the Judge's workspace/candidate staging rules (tests/bootstrap.py)."""

import importlib.util
import os
from pathlib import Path

import pytest

TESTS_DIR = Path(__file__).resolve().parents[2] / "tests"


@pytest.fixture()
def bootstrap(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("judge_bootstrap", TESTS_DIR / "bootstrap.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    ws = tmp_path / "workspace"
    ws.mkdir()
    monkeypatch.setattr(mod, "WORKSPACE", ws)
    monkeypatch.setattr(mod, "CANDIDATE_DIR", ws / "curriculum")
    return mod


def make_ws(bootstrap, policy=b'{"schema_version": 1}', scores=None, extra_root=(), extra_cand=()):
    ws = bootstrap.WORKSPACE
    (ws / "curriculum").mkdir(exist_ok=True)
    (ws / "research").mkdir(exist_ok=True)
    if policy is not None:
        (ws / "curriculum" / "policy.json").write_bytes(policy)
    if scores is not None:
        (ws / "curriculum" / "scores.csv").write_bytes(scores)
    for name in extra_root:
        (ws / name).write_text("x")
    for name in extra_cand:
        (ws / "curriculum" / name).write_text("x")
    return ws


def test_stage_ok(bootstrap, tmp_path):
    make_ws(bootstrap, scores=b"episode_index,frame_index,s\n0,0,1\n")
    staged = bootstrap.stage_candidate(tmp_path / "staging")
    assert staged["files"] == ["policy.json", "scores.csv"]
    assert (tmp_path / "staging" / "policy.json").read_bytes() == b'{"schema_version": 1}'
    assert staged["scores_sha256"] is not None


def test_rejects_extra_root_entry(bootstrap, tmp_path):
    make_ws(bootstrap, extra_root=["notes.txt"])
    with pytest.raises(bootstrap.CandidateError):
        bootstrap.stage_candidate(tmp_path / "staging")


def test_rejects_extra_candidate_file(bootstrap, tmp_path):
    make_ws(bootstrap, extra_cand=["helper.py"])
    with pytest.raises(bootstrap.CandidateError):
        bootstrap.stage_candidate(tmp_path / "staging")


def test_rejects_missing_policy(bootstrap, tmp_path):
    make_ws(bootstrap, policy=None)
    with pytest.raises(bootstrap.CandidateError):
        bootstrap.stage_candidate(tmp_path / "staging")


def test_rejects_symlink_policy(bootstrap, tmp_path):
    ws = make_ws(bootstrap, policy=None)
    target = tmp_path / "elsewhere.json"
    target.write_text("{}")
    os.symlink(target, ws / "curriculum" / "policy.json")
    with pytest.raises(bootstrap.CandidateError):
        bootstrap.stage_candidate(tmp_path / "staging")


def test_rejects_oversized_policy(bootstrap, tmp_path):
    make_ws(bootstrap, policy=b"x" * (bootstrap.MAX_POLICY_BYTES + 1))
    with pytest.raises(bootstrap.CandidateError):
        bootstrap.stage_candidate(tmp_path / "staging")


def test_preflight_scrubs_workspace_from_sys_path(bootstrap, monkeypatch):
    import sys

    ws = bootstrap.WORKSPACE
    monkeypatch.setattr(bootstrap, "PUBLIC", ws)  # any existing dir
    monkeypatch.setattr(bootstrap, "ASSETS", ws)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("MUJOCO_GL", "egl")
    monkeypatch.setenv("PYTHONPATH", str(ws))
    monkeypatch.setattr(sys, "path", [str(ws), str(ws / "sub"), "/usr/lib/python3"])
    bootstrap.preflight()
    assert sys.path == ["/usr/lib/python3"]
    assert "PYTHONPATH" not in os.environ
