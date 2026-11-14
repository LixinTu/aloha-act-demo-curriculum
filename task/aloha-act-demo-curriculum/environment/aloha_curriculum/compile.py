"""Deterministic compilation of a validated curriculum policy into an exposure schedule.

The schedule is an int64 array of exactly ``steps * batch_size`` dataset frame indices.
Training step ``s`` consumes entries ``[s*batch_size, (s+1)*batch_size)`` in order. The
Judge recompiles the schedule itself from the candidate and the fixed protocol, so the
agent cannot alter the budget; the sha256 of the schedule is reported as feedback.

Sampling: steps are grouped into blocks of ``block_steps``. For each block the stage that
contains the block's midpoint provides the weight (and optional mask) expression, which is
evaluated with ``t_train`` and ``t_stage`` set to the midpoint progress. Weights must be
finite and non-negative; the block's distribution is weight x masks normalised to one.

Each block picks its frames with the protocol's ``schedule_draw`` rule, using
``numpy.random.default_rng(schedule_seed)``:

- ``balanced`` (protocol version 3): a frame falls due once per unit of its accumulated target, the
  running sum of ``block_draws x probability``, at a random point of that unit. Each block takes the
  frames that are due and shuffles them. Exposures therefore stay within about one of their targets,
  and a block is a mixture that follows its distribution. Uniform weights give LeRobot's shuffled
  epochs: every frame once, in random order, before any frame twice.
- ``iid``: independent draws with replacement (protocol versions 1 and 2). Exposures per frame are
  then Poisson distributed; on the 800,000-exposure uniform schedule they range from 11 to 55.
"""

from __future__ import annotations

import hashlib
import heapq
import json
import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from . import dsl
from .features import IDENTITY_COLUMNS
from .errors import CandidateInvalid
from .schema import Policy


DRAW_RULES = ("balanced", "iid")


@dataclass(frozen=True)
class ScheduleProtocol:
    steps: int
    batch_size: int
    block_steps: int
    schedule_seed: int
    chunk_size: int
    draw: str

    @property
    def exposures(self) -> int:
        return self.steps * self.batch_size


def _uses_vars(expr: Any) -> bool:
    if isinstance(expr, dict):
        if "var" in expr:
            return True
        return any(_uses_vars(v) for v in expr.values())
    if isinstance(expr, list):
        return any(_uses_vars(v) for v in expr)
    return False


def build_context(features: pd.DataFrame, scores: pd.DataFrame | None, variables: dict[str, float]) -> dsl.Context:
    ep = features["episode_index"].to_numpy().astype(np.int64)
    n = len(features)
    starts = np.flatnonzero(np.r_[True, ep[1:] != ep[:-1]]).astype(np.int64)
    lengths = np.diff(np.r_[starts, n]).astype(np.int64)
    feats = {c: features[c].to_numpy().astype(np.float64) for c in features.columns}
    sc = {c: scores[c].to_numpy().astype(np.float64) for c in scores.columns} if scores is not None else {}
    return dsl.Context(n=n, episode_starts=starts, episode_lengths=lengths, features=feats, scores=sc, variables=dict(variables))


def global_mask(policy: Policy, features: pd.DataFrame) -> np.ndarray:
    ep = features["episode_index"].to_numpy()
    fr = features["frame_index"].to_numpy()
    pad = features["chunk_pad"].to_numpy()
    mask = np.ones(len(features), dtype=bool)
    if policy.include_episodes is not None:
        mask &= np.isin(ep, np.asarray(policy.include_episodes))
    if policy.exclude_episodes:
        mask &= ~np.isin(ep, np.asarray(policy.exclude_episodes))
    mask &= (fr >= policy.min_frame) & (fr <= policy.max_frame)
    mask &= pad <= policy.max_chunk_pad
    if not mask.any():
        raise CandidateInvalid("episode/anchor constraints leave no frame eligible for sampling")
    return mask


def _stage_for(policy: Policy, t: float) -> tuple[int, float]:
    prev = 0.0
    for i, st in enumerate(policy.stages):
        if t < st.until or i == len(policy.stages) - 1:
            span = st.until - prev
            return i, (t - prev) / span if span > 0 else 0.0
        prev = st.until
    raise AssertionError("unreachable")


def _draw_balanced(
    p: np.ndarray, draws: int, targets: np.ndarray, due_at: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Pick ``draws`` frames for one block: the frames most overdue on their accumulated targets.

    A frame falls due when its target passes ``due_at``: a random point of the current unit of
    target, redrawn for the next unit after each pick. The random point keeps frames with similar
    weights out of step; ranking frames by how far behind they are would send them in waves of one
    kind of frame. A frame gets several copies in a block only if its share needs them. Only frames
    with positive probability in this block can be picked. Updates ``due_at`` in place.
    """
    priority = targets - due_at
    priority[p <= 0] = -np.inf
    # A frame's later copies rank below its first, so one outside the top ``draws`` gets no pick.
    k = min(draws, p.size)
    top = np.argpartition(-priority, k - 1)[:k]
    heap = [(-priority[i], int(i)) for i in top if np.isfinite(priority[i])]
    heapq.heapify(heap)
    picks = np.empty(draws, dtype=np.int64)
    offsets = rng.random(draws)
    for j in range(draws):
        negative_priority, frame = heap[0]
        picks[j] = frame
        next_due = math.floor(due_at[frame]) + 1.0 + offsets[j]
        heapq.heapreplace(heap, (negative_priority + next_due - due_at[frame], frame))
        due_at[frame] = next_due
    rng.shuffle(picks)
    return picks


def compile_schedule(
    policy: Policy,
    features: pd.DataFrame,
    scores: pd.DataFrame | None,
    protocol: ScheduleProtocol,
) -> tuple[np.ndarray, dict]:
    if protocol.draw not in DRAW_RULES:
        raise ValueError(f"unknown schedule_draw {protocol.draw!r}, expected one of {DRAW_RULES}")
    n = len(features)
    gmask = global_mask(policy, features)
    rng = np.random.default_rng([int(protocol.schedule_seed), 0])
    schedule = np.empty(protocol.exposures, dtype=np.int64)
    targets = np.zeros(n, dtype=np.float64)
    due_at = rng.random(n) if protocol.draw == "balanced" else None
    num_blocks = math.ceil(protocol.steps / protocol.block_steps)

    cache: dict[int, tuple[np.ndarray, np.ndarray | None]] = {}
    stage_dynamic = [_uses_vars(st.weight) or (st.mask is not None and _uses_vars(st.mask)) for st in policy.stages]

    block_entropy: list[float] = []
    block_support: list[int] = []
    stage_of_block: list[int] = []
    pos = 0
    for b in range(num_blocks):
        step_lo = b * protocol.block_steps
        step_hi = min(step_lo + protocol.block_steps, protocol.steps)
        n_steps = step_hi - step_lo
        t_train = (step_lo + 0.5 * n_steps) / protocol.steps
        s_idx, t_stage = _stage_for(policy, t_train)
        stage = policy.stages[s_idx]
        if stage_dynamic[s_idx] or s_idx not in cache:
            ctx = build_context(features, scores, {"t_train": t_train, "t_stage": t_stage})
            w = dsl.evaluate(stage.weight, ctx)
            if np.any(w < 0):
                raise CandidateInvalid(f"stage {s_idx} weight is negative for {int(np.sum(w < 0))} frames; weights must be >= 0")
            m = dsl.evaluate(stage.mask, ctx) if stage.mask is not None else None
            if not stage_dynamic[s_idx]:
                cache[s_idx] = (w, m)
        else:
            w, m = cache[s_idx]
        p = w * gmask
        if m is not None:
            p = p * (m > 0.5)
        total = float(p.sum())
        if not (math.isfinite(total) and total > 0.0):
            raise CandidateInvalid(f"stage {s_idx} assigns zero total weight to eligible frames at t_train={t_train:.4f}")
        p = p / total
        draws = n_steps * protocol.batch_size
        targets += draws * p
        if protocol.draw == "balanced":
            schedule[pos : pos + draws] = _draw_balanced(p, draws, targets, due_at, rng)
        else:
            schedule[pos : pos + draws] = rng.choice(n, size=draws, replace=True, p=p)
        pos += draws
        nz = p[p > 0]
        block_entropy.append(float(-(nz * np.log(nz)).sum()))
        block_support.append(int(nz.size))
        stage_of_block.append(s_idx)
    if pos != protocol.exposures:
        raise AssertionError("schedule length mismatch")

    summary = summarize(schedule, features, policy, protocol, block_entropy, block_support, stage_of_block, targets)
    return schedule, summary


def schedule_sha256(schedule: np.ndarray) -> str:
    arr = np.ascontiguousarray(schedule, dtype="<i8")
    return hashlib.sha256(arr.tobytes()).hexdigest()


def summarize(
    schedule: np.ndarray,
    features: pd.DataFrame,
    policy: Policy,
    protocol: ScheduleProtocol,
    block_entropy: list[float],
    block_support: list[int],
    stage_of_block: list[int],
    targets: np.ndarray,
) -> dict:
    n = len(features)
    counts = np.bincount(schedule, minlength=n)
    ep = features["episode_index"].to_numpy()
    num_ep = int(ep.max()) + 1
    ep_counts = np.bincount(ep[schedule], minlength=num_ep)
    t_norm = features["t_norm"].to_numpy()
    deciles = np.bincount(np.minimum((t_norm[schedule] * 10).astype(int), 9), minlength=10)
    pad = features["chunk_pad"].to_numpy()[schedule]
    uniform_entropy = math.log(n)
    return {
        "policy_name": policy.name,
        "exposures": int(schedule.size),
        "steps": protocol.steps,
        "batch_size": protocol.batch_size,
        "block_steps": protocol.block_steps,
        "schedule_seed": protocol.schedule_seed,
        "draw": protocol.draw,
        "schedule_sha256": schedule_sha256(schedule),
        "distinct_frames": int((counts > 0).sum()),
        "distinct_frames_fraction": float((counts > 0).mean()),
        "max_frame_exposures": int(counts.max()),
        "min_frame_exposures": int(counts.min()),
        "max_abs_exposure_deviation": float(np.abs(counts - targets).max()),
        "episode_exposures": ep_counts.astype(int).tolist(),
        "episodes_used": int((ep_counts > 0).sum()),
        "t_norm_decile_exposures": deciles.astype(int).tolist(),
        "chunk_pad_mean": float(pad.mean()),
        "chunk_pad_fraction_padded": float((pad > 0).mean()),
        "entropy_min_nats": float(min(block_entropy)),
        "entropy_mean_nats": float(sum(block_entropy) / len(block_entropy)),
        "entropy_uniform_nats": float(uniform_entropy),
        "support_min_frames": int(min(block_support)),
        "stage_blocks": [int(stage_of_block.count(i)) for i in range(len(policy.stages))],
    }


def protocol_from_dict(d: dict) -> ScheduleProtocol:
    tr = d["training"]
    return ScheduleProtocol(
        steps=int(tr["steps"]),
        batch_size=int(tr["batch_size"]),
        block_steps=int(tr["block_steps"]),
        schedule_seed=int(tr["schedule_seed"]),
        chunk_size=int(d["policy"]["chunk_size"]),
        draw=tr["schedule_draw"],
    )


def save_schedule(schedule: np.ndarray, summary: dict, out_dir) -> None:
    from pathlib import Path

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "schedule.npy", np.ascontiguousarray(schedule, dtype=np.int64))
    (out / "schedule_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")


__all__ = [
    "ScheduleProtocol",
    "compile_schedule",
    "schedule_sha256",
    "protocol_from_dict",
    "save_schedule",
    "global_mask",
    "build_context",
    "IDENTITY_COLUMNS",
]
