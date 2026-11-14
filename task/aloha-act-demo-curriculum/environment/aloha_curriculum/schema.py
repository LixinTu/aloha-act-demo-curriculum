"""Validation of agent-authored candidates: ``policy.json`` and optional ``scores.csv``.

Candidates are data. This module checks structure, bounds and referential integrity
against the public feature table, and returns typed objects the compiler consumes.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import dsl
from .errors import CandidateInvalid
from .features import IDENTITY_COLUMNS

SCHEMA_VERSION = 1
MAX_POLICY_BYTES = 65_536
MAX_SCORES_BYTES = 8 * 1024 * 1024
MAX_SCORE_COLUMNS = 16
MAX_STAGES = 16
MAX_NAME_LEN = 64
MAX_NOTES_LEN = 2000
SCORE_NAME_RE = re.compile(r"^[a-z][a-z0-9_]{0,31}$")
RESERVED_NAMES = set(IDENTITY_COLUMNS) | set(dsl.VARS)




@dataclass(frozen=True)
class Stage:
    until: float
    weight: Any
    mask: Any | None


@dataclass(frozen=True)
class Policy:
    name: str
    scores_file: str | None
    include_episodes: list[int] | None  # None == all
    exclude_episodes: list[int]
    min_frame: int
    max_frame: int
    max_chunk_pad: int
    stages: list[Stage]
    raw: dict = field(repr=False)


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise CandidateInvalid(msg)


def _int_list(x: Any, what: str, *, lo: int, hi: int) -> list[int]:
    _require(isinstance(x, list), f"{what} must be a list of integers")
    out: list[int] = []
    for v in x:
        _require(isinstance(v, int) and not isinstance(v, bool), f"{what} must contain integers only")
        _require(lo <= v <= hi, f"{what} entries must lie in [{lo}, {hi}]")
        out.append(v)
    _require(len(set(out)) == len(out), f"{what} must not contain duplicates")
    return out


def load_policy_text(path: Path) -> dict:
    size = path.stat().st_size
    _require(size <= MAX_POLICY_BYTES, f"policy.json is {size} bytes, limit {MAX_POLICY_BYTES}")
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as e:
        raise CandidateInvalid(f"policy.json is not UTF-8: {e}") from e
    try:
        obj = json.loads(text)
    except json.JSONDecodeError as e:
        raise CandidateInvalid(f"policy.json is not valid JSON: {e}") from e
    _require(isinstance(obj, dict), "policy.json must contain a JSON object")
    return obj


def parse_policy(
    obj: dict,
    *,
    feature_names: set[str],
    score_names: set[str],
    num_episodes: int,
    max_frame_index: int,
    chunk_size: int,
) -> Policy:
    allowed = {"schema_version", "name", "notes", "scores", "episodes", "anchors", "stages"}
    extra = set(obj) - allowed
    _require(not extra, f"unknown top-level keys {sorted(extra)}")
    _require(obj.get("schema_version") == SCHEMA_VERSION, f"schema_version must be {SCHEMA_VERSION}")
    name = obj.get("name", "unnamed")
    _require(isinstance(name, str) and 1 <= len(name) <= MAX_NAME_LEN, "name must be a short string")
    notes = obj.get("notes", "")
    _require(isinstance(notes, str) and len(notes) <= MAX_NOTES_LEN, "notes must be a string <= 2000 chars")

    scores_file = obj.get("scores")
    _require(scores_file is None or scores_file == "scores.csv", "scores must be null or 'scores.csv'")
    if scores_file is None:
        _require(not score_names, "scores.csv present but policy does not declare 'scores': 'scores.csv'")

    episodes = obj.get("episodes", {"include": "all", "exclude": []})
    _require(isinstance(episodes, dict) and set(episodes) <= {"include", "exclude"}, "episodes must be {include, exclude}")
    inc = episodes.get("include", "all")
    include: list[int] | None
    if inc == "all":
        include = None
    else:
        include = _int_list(inc, "episodes.include", lo=0, hi=num_episodes - 1)
        _require(len(include) >= 1, "episodes.include must keep at least one episode")
    exclude = _int_list(episodes.get("exclude", []), "episodes.exclude", lo=0, hi=num_episodes - 1)

    anchors = obj.get("anchors", {})
    _require(isinstance(anchors, dict) and set(anchors) <= {"min_frame", "max_frame", "max_chunk_pad"}, "anchors keys")
    min_frame = anchors.get("min_frame", 0)
    max_frame = anchors.get("max_frame", max_frame_index)
    max_chunk_pad = anchors.get("max_chunk_pad", chunk_size - 1)
    for v, what in ((min_frame, "min_frame"), (max_frame, "max_frame"), (max_chunk_pad, "max_chunk_pad")):
        _require(isinstance(v, int) and not isinstance(v, bool), f"anchors.{what} must be an integer")
    _require(0 <= min_frame <= max_frame <= max_frame_index, "anchors must satisfy 0 <= min_frame <= max_frame <= last frame")
    _require(0 <= max_chunk_pad <= chunk_size - 1, f"anchors.max_chunk_pad must lie in [0, {chunk_size - 1}]")

    stages_raw = obj.get("stages")
    _require(isinstance(stages_raw, list) and 1 <= len(stages_raw) <= MAX_STAGES, f"stages must be a list of 1..{MAX_STAGES}")
    stages: list[Stage] = []
    prev = 0.0
    for i, st in enumerate(stages_raw):
        _require(isinstance(st, dict) and set(st) <= {"until", "weight", "mask"}, f"stage {i} keys must be until/weight/mask")
        until = st.get("until")
        _require(isinstance(until, (int, float)) and not isinstance(until, bool) and math.isfinite(until), f"stage {i}.until must be a number")
        until = float(until)
        _require(prev < until <= 1.0, f"stage {i}.until must be strictly increasing and <= 1.0")
        _require("weight" in st, f"stage {i} needs a weight expression")
        dsl.check_bytes(st["weight"], what=f"stage {i}.weight")
        dsl.validate_expression(st["weight"], feature_names=feature_names, score_names=score_names)
        mask = st.get("mask")
        if mask is not None:
            dsl.check_bytes(mask, what=f"stage {i}.mask")
            dsl.validate_expression(mask, feature_names=feature_names, score_names=score_names)
        stages.append(Stage(until=until, weight=st["weight"], mask=mask))
        prev = until
    _require(stages[-1].until == 1.0, "the last stage must end at until = 1.0")

    return Policy(
        name=name,
        scores_file=scores_file,
        include_episodes=include,
        exclude_episodes=exclude,
        min_frame=min_frame,
        max_frame=max_frame,
        max_chunk_pad=max_chunk_pad,
        stages=stages,
        raw=obj,
    )


def load_scores(path: Path, *, features: pd.DataFrame, feature_names: set[str]) -> pd.DataFrame:
    """Load and validate scores.csv; returns a frame aligned to ``features`` row order."""
    size = path.stat().st_size
    _require(size <= MAX_SCORES_BYTES, f"scores.csv is {size} bytes, limit {MAX_SCORES_BYTES}")
    try:
        df = pd.read_csv(path, encoding="utf-8", engine="c", float_precision="round_trip")
    except Exception as e:  # noqa: BLE001 - any parse failure is a candidate error
        raise CandidateInvalid(f"scores.csv could not be parsed: {e}") from e
    cols = list(df.columns)
    _require(cols[:2] == ["episode_index", "frame_index"], "scores.csv must start with columns episode_index, frame_index")
    score_cols = cols[2:]
    _require(1 <= len(score_cols) <= MAX_SCORE_COLUMNS, f"scores.csv needs 1..{MAX_SCORE_COLUMNS} score columns")
    for c in score_cols:
        _require(bool(SCORE_NAME_RE.match(c)), f"score column {c!r} must match {SCORE_NAME_RE.pattern}")
        _require(c not in feature_names and c not in RESERVED_NAMES, f"score column {c!r} collides with a public feature or reserved name")
    _require(len(set(score_cols)) == len(score_cols), "duplicate score column names")
    n = len(features)
    _require(len(df) == n, f"scores.csv has {len(df)} rows, expected {n}")
    for c in ("episode_index", "frame_index"):
        _require(pd.api.types.is_integer_dtype(df[c]), f"scores.csv column {c} must be integer")
    key_df = pd.DataFrame({"episode_index": df["episode_index"].to_numpy(), "frame_index": df["frame_index"].to_numpy()})
    _require(not key_df.duplicated().any(), "scores.csv has duplicate (episode_index, frame_index) rows")
    merged = features[["episode_index", "frame_index"]].merge(
        df, on=["episode_index", "frame_index"], how="left", validate="one_to_one"
    )
    _require(len(merged) == n, "scores.csv rows do not align with the dataset")
    out = pd.DataFrame(index=np.arange(n))
    for c in score_cols:
        vals = pd.to_numeric(merged[c], errors="coerce").to_numpy(dtype=np.float64)
        _require(bool(np.all(np.isfinite(vals))), f"score column {c!r} has missing or non-finite values")
        out[c] = vals
    return out


def policy_sha256(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()
