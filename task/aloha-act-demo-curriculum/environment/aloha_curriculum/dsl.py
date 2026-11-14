"""Bounded expression DSL for demonstration-curriculum weights.

An expression is a JSON tree evaluated to one float64 value per dataset frame.

Leaves
------
``{"const": <finite number>}``              constant broadcast to every frame
``{"feature": "<public feature column>"}``  task-owned per-frame feature (see features.py)
``{"score": "<agent score column>"}``       agent-authored per-frame score (scores.csv)
``{"var": "t_train"}``                       global training progress of the block, in [0, 1)
``{"var": "t_stage"}``                       progress inside the current stage, in [0, 1)

Internal nodes
--------------
``{"op": "<name>", "args": [<expr>, ...]}`` with the closed operator set in ``OPS``.
Boolean operators return 0.0/1.0 vectors. Integer parameters (window sizes, shifts)
must be ``{"const": k}`` leaves within the documented bounds.

The evaluator never calls ``eval``/``exec``; unknown keys, unknown operators, wrong
arity, out-of-bound constants, oversized trees and non-finite intermediate results all
raise ``DSLError``.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from .errors import CandidateInvalid

MAX_BYTES = 65_536
MAX_NODES = 512
MAX_DEPTH = 24
MAX_ABS_CONST = 1.0e6
MAX_WINDOW = 250
MAX_SHIFT = 500
MAX_POW = 8.0
EXP_CLIP = 50.0
EPS = 1.0e-9
VARS = ("t_train", "t_stage")


class DSLError(CandidateInvalid):
    """Raised for any structural or numerical violation of the DSL contract (a candidate error)."""


@dataclass(frozen=True)
class Context:
    """Evaluation context: column lookups plus episode structure.

    ``episode_starts`` holds the first row of each episode; rows of one episode are
    contiguous and ordered by frame. ``features`` and ``scores`` map column names to
    float64 vectors of length ``n``.
    """

    n: int
    episode_starts: np.ndarray  # int64, shape (num_episodes,)
    episode_lengths: np.ndarray  # int64, shape (num_episodes,)
    features: dict[str, np.ndarray]
    scores: dict[str, np.ndarray]
    variables: dict[str, float]

    def episode_slices(self) -> list[slice]:
        return [
            slice(int(s), int(s + l))
            for s, l in zip(self.episode_starts.tolist(), self.episode_lengths.tolist(), strict=True)
        ]


# --------------------------------------------------------------------------------------
# Structural validation
# --------------------------------------------------------------------------------------


def _is_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _const_value(node: Any, *, what: str) -> float:
    if not (isinstance(node, dict) and set(node) == {"const"}):
        raise DSLError(f"{what} must be a {{'const': k}} leaf")
    v = node["const"]
    if not _is_number(v) or not math.isfinite(v):
        raise DSLError(f"{what} must be a finite number")
    return float(v)


def _int_const(node: Any, *, lo: int, hi: int, what: str) -> int:
    v = _const_value(node, what=what)
    if v != int(v) or not (lo <= int(v) <= hi):
        raise DSLError(f"{what} must be an integer in [{lo}, {hi}], got {v!r}")
    return int(v)


# op name -> (arity, kind). kind documents which args must be const ints.
OPS: dict[str, tuple[int, str]] = {
    # arithmetic
    "add": (2, "num"),
    "sub": (2, "num"),
    "mul": (2, "num"),
    "div": (2, "num"),
    "neg": (1, "num"),
    "abs": (1, "num"),
    "min": (2, "num"),
    "max": (2, "num"),
    "clip": (3, "num"),  # clip(x, lo, hi) with lo/hi expressions
    "exp": (1, "num"),
    "log1p": (1, "num"),
    "sqrt": (1, "num"),
    "square": (1, "num"),
    "pow": (2, "pow"),  # pow(x, p) with p a const in [-MAX_POW, MAX_POW]
    "sigmoid": (1, "num"),
    "tanh": (1, "num"),
    # comparison / logic (0.0 or 1.0)
    "lt": (2, "num"),
    "le": (2, "num"),
    "gt": (2, "num"),
    "ge": (2, "num"),
    "eq": (2, "num"),
    "and": (2, "num"),
    "or": (2, "num"),
    "not": (1, "num"),
    "where": (3, "num"),  # where(cond, a, b)
    # dataset-wide
    "rank": (1, "num"),  # ordinal percentile rank over all frames, in [0, 1]
    "zscore": (1, "num"),
    # per-episode aggregates broadcast to frames
    "ep_mean": (1, "num"),
    "ep_min": (1, "num"),
    "ep_max": (1, "num"),
    "ep_sum": (1, "num"),
    "ep_std": (1, "num"),
    "ep_rank": (1, "num"),  # percentile rank of the per-episode mean among episodes
    "ep_cumsum": (1, "num"),
    "ep_cummax": (1, "num"),
    "ep_first_where": (1, "num"),  # first frame_index with cond > 0.5, else -1
    "ep_last_where": (1, "num"),  # last frame_index with cond > 0.5, else -1
    # temporal (within episode, edge-padded)
    "shift": (2, "shift"),  # shift(x, k): value k frames earlier (k>0) or later (k<0)
    "window_mean": (2, "window"),  # centered window [i-k, i+k]
    "window_max": (2, "window"),
    "window_min": (2, "window"),
    "future_mean": (2, "window"),  # [i, i+k]
    "future_max": (2, "window"),
    "future_sum": (2, "window"),
    "past_mean": (2, "window"),  # [i-k, i]
}


def validate_expression(
    expr: Any,
    *,
    feature_names: set[str],
    score_names: set[str],
) -> int:
    """Validate structure and bounds; return the node count."""
    counter = {"nodes": 0}

    def visit(node: Any, depth: int) -> None:
        if depth > MAX_DEPTH:
            raise DSLError(f"expression depth exceeds {MAX_DEPTH}")
        counter["nodes"] += 1
        if counter["nodes"] > MAX_NODES:
            raise DSLError(f"expression has more than {MAX_NODES} nodes")
        if not isinstance(node, dict):
            raise DSLError("every expression node must be a JSON object")
        keys = set(node)
        if keys == {"const"}:
            v = node["const"]
            if not _is_number(v) or not math.isfinite(v) or abs(v) > MAX_ABS_CONST:
                raise DSLError(f"const must be a finite number with |value| <= {MAX_ABS_CONST}")
            return
        if keys == {"feature"}:
            name = node["feature"]
            if not isinstance(name, str) or name not in feature_names:
                raise DSLError(f"unknown feature {name!r}")
            return
        if keys == {"score"}:
            name = node["score"]
            if not isinstance(name, str) or name not in score_names:
                raise DSLError(f"unknown score column {name!r} (declare it in scores.csv)")
            return
        if keys == {"var"}:
            name = node["var"]
            if name not in VARS:
                raise DSLError(f"unknown variable {name!r}; allowed: {VARS}")
            return
        if keys == {"op", "args"}:
            op = node["op"]
            if not isinstance(op, str) or op not in OPS:
                raise DSLError(f"unknown operator {op!r}")
            arity, kind = OPS[op]
            args = node["args"]
            if not isinstance(args, list) or len(args) != arity:
                raise DSLError(f"operator {op!r} expects {arity} args")
            if kind == "pow":
                visit(args[0], depth + 1)
                counter["nodes"] += 1
                p = _const_value(args[1], what="pow exponent")
                if abs(p) > MAX_POW:
                    raise DSLError(f"pow exponent must satisfy |p| <= {MAX_POW}")
                return
            if kind == "shift":
                visit(args[0], depth + 1)
                counter["nodes"] += 1
                _int_const(args[1], lo=-MAX_SHIFT, hi=MAX_SHIFT, what="shift amount")
                return
            if kind == "window":
                visit(args[0], depth + 1)
                counter["nodes"] += 1
                _int_const(args[1], lo=0, hi=MAX_WINDOW, what="window size")
                return
            for a in args:
                visit(a, depth + 1)
            return
        raise DSLError(f"malformed node with keys {sorted(keys)}")

    visit(expr, 1)
    return counter["nodes"]


def check_bytes(obj: Any, *, what: str = "expression") -> None:
    size = len(json.dumps(obj, separators=(",", ":")).encode("utf-8"))
    if size > MAX_BYTES:
        raise DSLError(f"{what} serialises to {size} bytes, limit {MAX_BYTES}")


# --------------------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------------------


def _finite(x: np.ndarray, op: str) -> np.ndarray:
    if not np.all(np.isfinite(x)):
        raise DSLError(f"operator {op!r} produced a non-finite value")
    return x


def _safe_div(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    denom = np.where(np.abs(b) < EPS, np.where(b < 0, -EPS, EPS), b)
    return a / denom


def _rank(x: np.ndarray) -> np.ndarray:
    n = x.shape[0]
    if n == 1:
        return np.zeros_like(x)
    order = np.argsort(x, kind="stable")
    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = np.arange(n, dtype=np.float64)
    return ranks / float(n - 1)


def _per_episode(ctx: Context, x: np.ndarray, fn: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
    out = np.empty_like(x)
    for sl in ctx.episode_slices():
        out[sl] = fn(x[sl])
    return out


def _episode_scalar(ctx: Context, x: np.ndarray, fn: Callable[[np.ndarray], float]) -> np.ndarray:
    out = np.empty_like(x)
    for sl in ctx.episode_slices():
        out[sl] = fn(x[sl])
    return out


def _window(x: np.ndarray, lo: int, hi: int, reducer: str) -> np.ndarray:
    """Reduce over [i+lo, i+hi] (clamped to the episode) for every i."""
    n = x.shape[0]
    if n == 0:
        return x
    idx = np.arange(n)
    starts = np.clip(idx + lo, 0, n - 1)
    ends = np.clip(idx + hi, 0, n - 1)
    if reducer in ("mean", "sum"):
        cs = np.concatenate([[0.0], np.cumsum(x)])
        sums = cs[ends + 1] - cs[starts]
        if reducer == "sum":
            return sums
        return sums / (ends - starts + 1)
    # max / min via padded sliding window (n <= a few thousand, width <= 2*MAX_WINDOW+1)
    width = hi - lo + 1
    pad_lo, pad_hi = max(0, -lo), max(0, hi)
    padded = np.pad(x, (pad_lo, pad_hi), mode="edge")
    view = np.lib.stride_tricks.sliding_window_view(padded, width)
    # view[i] covers padded[i : i+width] == x[i - pad_lo : i - pad_lo + width]; we want x[i+lo : i+hi]
    out = view.max(axis=1) if reducer == "max" else view.min(axis=1)
    return out[: n]


def _first_where(c: np.ndarray) -> float:
    hits = np.flatnonzero(c > 0.5)
    return float(hits[0]) if hits.size else -1.0


def _last_where(c: np.ndarray) -> float:
    hits = np.flatnonzero(c > 0.5)
    return float(hits[-1]) if hits.size else -1.0


def evaluate(expr: Any, ctx: Context) -> np.ndarray:
    """Evaluate a validated expression to a float64 vector of length ``ctx.n``."""

    def ev(node: dict) -> np.ndarray:
        if "const" in node:
            return np.full(ctx.n, float(node["const"]), dtype=np.float64)
        if "feature" in node:
            return ctx.features[node["feature"]].astype(np.float64, copy=False)
        if "score" in node:
            return ctx.scores[node["score"]].astype(np.float64, copy=False)
        if "var" in node:
            return np.full(ctx.n, float(ctx.variables[node["var"]]), dtype=np.float64)
        op, args = node["op"], node["args"]
        if op == "add":
            return _finite(ev(args[0]) + ev(args[1]), op)
        if op == "sub":
            return _finite(ev(args[0]) - ev(args[1]), op)
        if op == "mul":
            return _finite(ev(args[0]) * ev(args[1]), op)
        if op == "div":
            return _finite(_safe_div(ev(args[0]), ev(args[1])), op)
        if op == "neg":
            return -ev(args[0])
        if op == "abs":
            return np.abs(ev(args[0]))
        if op == "min":
            return np.minimum(ev(args[0]), ev(args[1]))
        if op == "max":
            return np.maximum(ev(args[0]), ev(args[1]))
        if op == "clip":
            x, lo, hi = ev(args[0]), ev(args[1]), ev(args[2])
            return np.minimum(np.maximum(x, lo), np.maximum(lo, hi))
        if op == "exp":
            return np.exp(np.clip(ev(args[0]), -EXP_CLIP, EXP_CLIP))
        if op == "log1p":
            return np.log1p(np.maximum(ev(args[0]), 0.0))
        if op == "sqrt":
            return np.sqrt(np.maximum(ev(args[0]), 0.0))
        if op == "square":
            return _finite(np.square(ev(args[0])), op)
        if op == "pow":
            x = np.maximum(ev(args[0]), 0.0)
            p = float(args[1]["const"])
            if p < 0:
                x = np.maximum(x, EPS)
            return _finite(np.power(x, p), op)
        if op == "sigmoid":
            z = np.clip(ev(args[0]), -EXP_CLIP, EXP_CLIP)
            return 1.0 / (1.0 + np.exp(-z))
        if op == "tanh":
            return np.tanh(ev(args[0]))
        if op == "lt":
            return (ev(args[0]) < ev(args[1])).astype(np.float64)
        if op == "le":
            return (ev(args[0]) <= ev(args[1])).astype(np.float64)
        if op == "gt":
            return (ev(args[0]) > ev(args[1])).astype(np.float64)
        if op == "ge":
            return (ev(args[0]) >= ev(args[1])).astype(np.float64)
        if op == "eq":
            return np.isclose(ev(args[0]), ev(args[1]), rtol=0.0, atol=1e-9).astype(np.float64)
        if op == "and":
            return ((ev(args[0]) > 0.5) & (ev(args[1]) > 0.5)).astype(np.float64)
        if op == "or":
            return ((ev(args[0]) > 0.5) | (ev(args[1]) > 0.5)).astype(np.float64)
        if op == "not":
            return (ev(args[0]) <= 0.5).astype(np.float64)
        if op == "where":
            return np.where(ev(args[0]) > 0.5, ev(args[1]), ev(args[2]))
        if op == "rank":
            return _rank(ev(args[0]))
        if op == "zscore":
            x = ev(args[0])
            sd = float(np.std(x))
            return (x - float(np.mean(x))) / sd if sd > EPS else np.zeros_like(x)
        if op == "ep_mean":
            return _episode_scalar(ctx, ev(args[0]), lambda v: float(np.mean(v)))
        if op == "ep_min":
            return _episode_scalar(ctx, ev(args[0]), lambda v: float(np.min(v)))
        if op == "ep_max":
            return _episode_scalar(ctx, ev(args[0]), lambda v: float(np.max(v)))
        if op == "ep_sum":
            return _finite(_episode_scalar(ctx, ev(args[0]), lambda v: float(np.sum(v))), op)
        if op == "ep_std":
            return _episode_scalar(ctx, ev(args[0]), lambda v: float(np.std(v)))
        if op == "ep_rank":
            means = np.array([float(np.mean(ev(args[0])[sl])) for sl in ctx.episode_slices()])
            ranks = _rank(means)
            out = np.empty(ctx.n, dtype=np.float64)
            for r, sl in zip(ranks, ctx.episode_slices(), strict=True):
                out[sl] = r
            return out
        if op == "ep_cumsum":
            return _finite(_per_episode(ctx, ev(args[0]), np.cumsum), op)
        if op == "ep_cummax":
            return _per_episode(ctx, ev(args[0]), np.maximum.accumulate)
        if op == "ep_first_where":
            return _episode_scalar(ctx, ev(args[0]), _first_where)
        if op == "ep_last_where":
            return _episode_scalar(ctx, ev(args[0]), _last_where)
        if op == "shift":
            k = int(args[1]["const"])

            def do_shift(v: np.ndarray) -> np.ndarray:
                idx = np.clip(np.arange(v.shape[0]) - k, 0, v.shape[0] - 1)
                return v[idx]

            return _per_episode(ctx, ev(args[0]), do_shift)
        if op in ("window_mean", "window_max", "window_min", "future_mean", "future_max", "future_sum", "past_mean"):
            k = int(args[1]["const"])
            x = ev(args[0])
            if op == "window_mean":
                return _finite(_per_episode(ctx, x, lambda v: _window(v, -k, k, "mean")), op)
            if op == "window_max":
                return _per_episode(ctx, x, lambda v: _window(v, -k, k, "max"))
            if op == "window_min":
                return _per_episode(ctx, x, lambda v: _window(v, -k, k, "min"))
            if op == "future_mean":
                return _finite(_per_episode(ctx, x, lambda v: _window(v, 0, k, "mean")), op)
            if op == "future_max":
                return _per_episode(ctx, x, lambda v: _window(v, 0, k, "max"))
            if op == "future_sum":
                return _finite(_per_episode(ctx, x, lambda v: _window(v, 0, k, "sum")), op)
            return _finite(_per_episode(ctx, x, lambda v: _window(v, -k, 0, "mean")), op)
        raise DSLError(f"unhandled operator {op!r}")  # pragma: no cover

    out = ev(expr)
    if out.shape != (ctx.n,):
        raise DSLError("expression did not evaluate to one value per frame")
    return _finite(out.astype(np.float64, copy=False), "root")
