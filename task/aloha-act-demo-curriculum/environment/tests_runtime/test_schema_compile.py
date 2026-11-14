import copy
import json

import numpy as np
import pandas as pd
import pytest

from aloha_curriculum import compile as compile_mod
from aloha_curriculum import schema
from aloha_curriculum.features import feature_names

CHUNK = 10
PROTO = compile_mod.ScheduleProtocol(steps=40, batch_size=4, block_steps=5, schedule_seed=123, chunk_size=CHUNK, draw="balanced")


def uniform_policy():
    return {
        "schema_version": 1,
        "name": "uniform",
        "scores": None,
        "episodes": {"include": "all", "exclude": []},
        "anchors": {"min_frame": 0, "max_frame": 49, "max_chunk_pad": CHUNK - 1},
        "stages": [{"until": 1.0, "weight": {"const": 1.0}, "mask": None}],
    }


def parse(obj, toy_features, scores=None):
    return schema.parse_policy(obj, feature_names=set(feature_names(toy_features)), score_names=set(scores.columns) if scores is not None else set(), num_episodes=4, max_frame_index=49, chunk_size=CHUNK)


def test_uniform_compiles_deterministically(toy_features):
    pol = parse(uniform_policy(), toy_features)
    s1, sum1 = compile_mod.compile_schedule(pol, toy_features, None, PROTO)
    s2, sum2 = compile_mod.compile_schedule(pol, toy_features, None, PROTO)
    assert s1.size == PROTO.exposures and np.array_equal(s1, s2)
    assert sum1["schedule_sha256"] == sum2["schedule_sha256"]
    assert sum1["episodes_used"] == 4 and sum1["support_min_frames"] == 200


def test_seed_changes_schedule(toy_features):
    pol = parse(uniform_policy(), toy_features)
    s1, _ = compile_mod.compile_schedule(pol, toy_features, None, PROTO)
    other = compile_mod.ScheduleProtocol(steps=40, batch_size=4, block_steps=5, schedule_seed=124, chunk_size=CHUNK, draw="balanced")
    s2, _ = compile_mod.compile_schedule(pol, toy_features, None, other)
    assert not np.array_equal(s1, s2)


def test_episode_and_anchor_constraints(toy_features):
    obj = uniform_policy()
    obj["episodes"] = {"include": [1, 2], "exclude": [2]}
    obj["anchors"] = {"min_frame": 10, "max_frame": 20, "max_chunk_pad": 0}
    pol = parse(obj, toy_features)
    s, summary = compile_mod.compile_schedule(pol, toy_features, None, PROTO)
    ep = toy_features["episode_index"].to_numpy()[s]
    fr = toy_features["frame_index"].to_numpy()[s]
    assert set(ep.tolist()) == {1} and fr.min() >= 10 and fr.max() <= 20
    assert summary["episode_exposures"][1] == PROTO.exposures


def test_stages_and_vars(toy_features):
    obj = uniform_policy()
    obj["stages"] = [
        {"until": 0.5, "weight": {"op": "lt", "args": [{"feature": "t_norm"}, {"const": 0.5}]}, "mask": None},
        {"until": 1.0, "weight": {"op": "mul", "args": [{"var": "t_stage"}, {"op": "ge", "args": [{"feature": "t_norm"}, {"const": 0.5}]}]}, "mask": None},
    ]
    pol = parse(obj, toy_features)
    s, summary = compile_mod.compile_schedule(pol, toy_features, None, PROTO)
    t = toy_features["t_norm"].to_numpy()[s]
    half = PROTO.exposures // 2
    assert np.all(t[:half] < 0.5) and np.all(t[half:] >= 0.5)
    assert summary["stage_blocks"] == [4, 4]


def test_mask_and_zero_mass(toy_features):
    obj = uniform_policy()
    obj["stages"] = [{"until": 1.0, "weight": {"const": 1.0}, "mask": {"op": "gt", "args": [{"feature": "t_norm"}, {"const": 2.0}]}}]
    pol = parse(obj, toy_features)
    with pytest.raises(schema.CandidateInvalid):
        compile_mod.compile_schedule(pol, toy_features, None, PROTO)
    obj["stages"] = [{"until": 1.0, "weight": {"op": "sub", "args": [{"const": 0.0}, {"feature": "t_norm"}]}, "mask": None}]
    pol = parse(obj, toy_features)
    with pytest.raises(schema.CandidateInvalid):
        compile_mod.compile_schedule(pol, toy_features, None, PROTO)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda o: o.update({"schema_version": 2}),
        lambda o: o.update({"bogus": 1}),
        lambda o: o.update({"stages": []}),
        lambda o: o.update({"stages": [{"until": 0.5, "weight": {"const": 1}}]}),
        lambda o: o.update({"stages": [{"until": 0.5, "weight": {"const": 1}}, {"until": 0.4, "weight": {"const": 1}}]}),
        lambda o: o.update({"episodes": {"include": []}}),
        lambda o: o.update({"episodes": {"include": [0, 0]}}),
        lambda o: o.update({"episodes": {"include": [99]}}),
        lambda o: o.update({"anchors": {"min_frame": 30, "max_frame": 20}}),
        lambda o: o.update({"anchors": {"max_chunk_pad": CHUNK}}),
        lambda o: o.update({"scores": "other.csv"}),
        lambda o: o.update({"stages": [{"until": 1.0, "weight": {"score": "s"}}]}),
    ],
)
def test_policy_rejections(toy_features, mutate):
    obj = uniform_policy()
    mutate(obj)
    with pytest.raises(schema.CandidateInvalid):
        parse(obj, toy_features)


def test_scores_roundtrip(tmp_path, toy_features):
    df = toy_features[["episode_index", "frame_index"]].copy()
    df["difficulty"] = np.linspace(0, 1, len(df))
    path = tmp_path / "scores.csv"
    df.sample(frac=1.0, random_state=1).to_csv(path, index=False)  # shuffled rows must still align
    scores = schema.load_scores(path, features=toy_features, feature_names=set(feature_names(toy_features)))
    assert list(scores.columns) == ["difficulty"]
    assert np.allclose(scores["difficulty"].to_numpy(), np.linspace(0, 1, len(df)))
    obj = uniform_policy()
    obj["scores"] = "scores.csv"
    obj["stages"] = [{"until": 1.0, "weight": {"op": "add", "args": [{"score": "difficulty"}, {"const": 0.01}]}, "mask": None}]
    pol = parse(obj, toy_features, scores)
    s, _ = compile_mod.compile_schedule(pol, toy_features, scores, PROTO)
    assert s.size == PROTO.exposures


@pytest.mark.parametrize(
    "spoil",
    [
        lambda df: df.iloc[:-1],
        lambda df: pd.concat([df, df.iloc[:1]]),
        lambda df: df.assign(difficulty=[np.nan] + [0.0] * (len(df) - 1)),
        lambda df: df.rename(columns={"difficulty": "T_norm"}),
        lambda df: df.rename(columns={"difficulty": "t_norm"}),
        lambda df: df.rename(columns={"difficulty": "t_train"}),
        lambda df: df[["frame_index", "episode_index", "difficulty"]],
    ],
)
def test_scores_rejections(tmp_path, toy_features, spoil):
    df = toy_features[["episode_index", "frame_index"]].copy()
    df["difficulty"] = 0.5
    path = tmp_path / "scores.csv"
    spoil(df).to_csv(path, index=False)
    with pytest.raises(schema.CandidateInvalid):
        schema.load_scores(path, features=toy_features, feature_names=set(feature_names(toy_features)))


def test_policy_bytes_limit(tmp_path):
    big = {"schema_version": 1, "notes": "x" * (schema.MAX_POLICY_BYTES + 10)}
    path = tmp_path / "policy.json"
    path.write_text(json.dumps(big))
    with pytest.raises(schema.CandidateInvalid):
        schema.load_policy_text(path)
