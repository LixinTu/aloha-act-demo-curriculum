"""Command-line entry points (installed as aloha-features / aloha-validate / aloha-compile /
aloha-train / aloha-eval). Exit codes: 0 ok, 2 candidate invalid, 1 infrastructure error."""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import sys
from pathlib import Path

import numpy as np

from . import compile as compile_mod
from . import features as features_mod
from . import schema

EXIT_OK = 0
EXIT_INFRA = 1
EXIT_CANDIDATE = 2


def _setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", stream=sys.stderr)


# Version 2 (2026-11-08) normalises camera images with ImageNet statistics, as LeRobot's training
# does by default; version 1 used the dataset's own pixel statistics.
# Version 3 (2026-11-08) replaces independent draws with replacement by balanced exposures (see
# compile.py) and scales images as lerobot-train does. Older versions are no longer accepted.
PROTOCOL_VERSION = 3


def load_protocol(path: Path) -> dict:
    protocol = json.loads(path.read_text())
    if protocol.get("protocol_version") != PROTOCOL_VERSION:
        raise RuntimeError(f"unsupported protocol_version {protocol.get('protocol_version')!r}, expected {PROTOCOL_VERSION}")
    return protocol


def load_candidate(candidate_dir: Path, features_path: Path, protocol: dict):
    """Validate policy.json (+ optional scores.csv) against the feature table.

    Returns (policy, features_df, scores_df_or_None, meta dict).
    """
    features = features_mod.read_features(features_path)
    feature_names = set(features_mod.feature_names(features))
    policy_path = candidate_dir / "policy.json"
    scores_path = candidate_dir / "scores.csv"
    if not policy_path.is_file():
        raise schema.CandidateInvalid(f"missing {policy_path}")
    entries = sorted(p.name for p in candidate_dir.iterdir())
    allowed = {"policy.json", "scores.csv"}
    extra = [e for e in entries if e not in allowed]
    if extra:
        raise schema.CandidateInvalid(f"candidate directory may only contain policy.json and scores.csv; found {extra}")
    raw = schema.load_policy_text(policy_path)
    scores = None
    score_names: set[str] = set()
    if scores_path.exists():
        if not scores_path.is_file():
            raise schema.CandidateInvalid("scores.csv must be a regular file")
        scores = schema.load_scores(scores_path, features=features, feature_names=feature_names)
        score_names = set(scores.columns)
    if raw.get("scores") == "scores.csv" and scores is None:
        raise schema.CandidateInvalid("policy declares scores.csv but the file is missing")
    policy = schema.parse_policy(
        raw,
        feature_names=feature_names,
        score_names=score_names,
        num_episodes=int(protocol["dataset"]["total_episodes"]),
        max_frame_index=int(features["frame_index"].max()),
        chunk_size=int(protocol["policy"]["chunk_size"]),
    )
    meta = {
        "policy_sha256": schema.policy_sha256(policy_path),
        "scores_sha256": schema.policy_sha256(scores_path) if scores is not None else None,
        "score_columns": sorted(score_names),
        "policy_name": policy.name,
        "stages": len(policy.stages),
    }
    return policy, features, scores, meta


def features_main(argv: list[str] | None = None) -> int:
    _setup_logging()
    return features_mod.main(argv)


def _candidate_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--candidate-dir", required=True, help="directory holding policy.json (+ scores.csv)")
    p.add_argument("--features", required=True, help="public feature table (parquet)")
    p.add_argument("--protocol", required=True, help="fixed_train.json")


def validate_main(argv: list[str] | None = None) -> int:
    _setup_logging()
    p = argparse.ArgumentParser(description="Validate a curriculum candidate and dry-run its compilation.")
    _candidate_args(p)
    p.add_argument("--steps", type=int, default=None, help="override steps for a quick dry run (not the official budget)")
    a = p.parse_args(argv)
    try:
        protocol = load_protocol(Path(a.protocol))
        policy, features, scores, meta = load_candidate(Path(a.candidate_dir), Path(a.features), protocol)
        proto = compile_mod.protocol_from_dict(protocol)
        if a.steps is not None:
            proto = dataclasses.replace(proto, steps=int(a.steps))
        _, summary = compile_mod.compile_schedule(policy, features, scores, proto)
    except schema.CandidateInvalid as e:
        print(json.dumps({"valid": False, "error": str(e)}, indent=2))
        return EXIT_CANDIDATE
    except Exception as e:  # noqa: BLE001
        print(json.dumps({"valid": False, "infrastructure_error": f"{type(e).__name__}: {e}"}, indent=2))
        return EXIT_INFRA
    print(json.dumps({"valid": True, **meta, "schedule": summary}, indent=2))
    return EXIT_OK


def compile_main(argv: list[str] | None = None) -> int:
    _setup_logging()
    p = argparse.ArgumentParser(description="Compile a curriculum candidate into an exposure schedule.")
    _candidate_args(p)
    p.add_argument("--output", required=True, help="directory for schedule.npy and schedule_summary.json")
    p.add_argument("--steps", type=int, default=None, help="override steps (diagnostic runs only)")
    a = p.parse_args(argv)
    try:
        protocol = load_protocol(Path(a.protocol))
        policy, features, scores, meta = load_candidate(Path(a.candidate_dir), Path(a.features), protocol)
        proto = compile_mod.protocol_from_dict(protocol)
        if a.steps is not None:
            proto = dataclasses.replace(proto, steps=int(a.steps))
        schedule, summary = compile_mod.compile_schedule(policy, features, scores, proto)
        summary.update(meta)
        compile_mod.save_schedule(schedule, summary, a.output)
    except schema.CandidateInvalid as e:
        print(json.dumps({"valid": False, "error": str(e)}, indent=2))
        return EXIT_CANDIDATE
    print(json.dumps(summary, indent=2))
    return EXIT_OK


def train_main(argv: list[str] | None = None) -> int:
    _setup_logging()
    from . import train as train_mod

    p = argparse.ArgumentParser(description="Train ACT with the fixed recipe over a compiled schedule.")
    p.add_argument("--schedule", required=True, help="schedule.npy from aloha-compile")
    p.add_argument("--protocol", required=True)
    p.add_argument("--dataset-root", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--steps", type=int, default=None, help="diagnostic override; the schedule must match")
    p.add_argument("--seed", type=int, default=None, help="diagnostic override of the training seed")
    p.add_argument("--video-backend", default=None)
    p.add_argument("--log-every", type=int, default=None)
    a = p.parse_args(argv)
    protocol = load_protocol(Path(a.protocol))
    schedule = np.load(a.schedule)
    info = train_mod.train(
        protocol=protocol,
        schedule=schedule,
        dataset_root=Path(a.dataset_root),
        output_dir=Path(a.output),
        device=a.device,
        steps_override=a.steps,
        seed_override=a.seed,
        video_backend=a.video_backend,
        log_every=a.log_every,
    )
    print(json.dumps({k: v for k, v in info.items() if k != "history"}, indent=2))
    return EXIT_OK


def eval_main(argv: list[str] | None = None) -> int:
    _setup_logging()
    from . import evaluate as eval_mod

    p = argparse.ArgumentParser(description="Evaluate an ACT checkpoint in gym-aloha AlohaInsertion-v0.")
    p.add_argument("--checkpoint", required=True, help="pretrained_model directory")
    p.add_argument("--protocol", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--start-seed", type=int, default=None, help="default: the public proxy seed block")
    p.add_argument("--n-episodes", type=int, default=None, help="default: the public proxy episode count")
    p.add_argument("--n-envs", type=int, default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument("--async-envs", action="store_true")
    a = p.parse_args(argv)
    protocol = load_protocol(Path(a.protocol))
    proxy = protocol["evaluation"]["proxy"]
    result = eval_mod.evaluate(
        protocol=protocol,
        checkpoint=Path(a.checkpoint),
        start_seed=int(a.start_seed if a.start_seed is not None else proxy["start_seed"]),
        n_episodes=int(a.n_episodes if a.n_episodes is not None else proxy["n_episodes"]),
        output_dir=Path(a.output),
        n_envs=a.n_envs,
        device=a.device,
        use_async_envs=a.async_envs,
    )
    print(json.dumps({k: v for k, v in result.items() if k != "per_episode"}, indent=2))
    return EXIT_OK
