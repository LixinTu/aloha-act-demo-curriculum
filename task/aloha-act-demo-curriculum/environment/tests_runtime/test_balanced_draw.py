"""The balanced schedule draw of protocol version 3.

Independent draws with replacement give each frame a Poisson number of exposures; a full-budget
ACT trained that way scored clearly below lerobot-train with shuffled epochs. These tests pin what
replaced them: exposures follow accumulated targets, uniform weights give shuffled epochs, and
blocks stay mixed under other weights. A first version ranked frames by how far behind they were,
which met the first two and sent high and low weights in alternating waves.
"""

import copy
import dataclasses

import numpy as np
import pytest

from aloha_curriculum import compile as compile_mod
from test_schema_compile import CHUNK, parse, uniform_policy

# 200 toy frames, 20 draws per block, 400 draws in all: two whole epochs.
TWO_EPOCHS = compile_mod.ScheduleProtocol(steps=100, batch_size=4, block_steps=5, schedule_seed=7, chunk_size=CHUNK, draw="balanced")
BLOCK_DRAWS = TWO_EPOCHS.block_steps * TWO_EPOCHS.batch_size


def compile_uniform(toy_features, proto=TWO_EPOCHS):
    return compile_mod.compile_schedule(parse(uniform_policy(), toy_features), toy_features, None, proto)


def weighted_policy(weight, stages=None):
    obj = uniform_policy()
    obj["stages"] = stages or [{"until": 1.0, "weight": weight, "mask": None}]
    return obj


def test_uniform_exposes_every_frame_equally(toy_features):
    s, summary = compile_uniform(toy_features)
    assert np.all(np.bincount(s, minlength=200) == 2)
    assert summary["draw"] == "balanced"
    assert summary["min_frame_exposures"] == summary["max_frame_exposures"] == 2
    assert summary["max_abs_exposure_deviation"] < 1e-9


def test_uniform_is_a_permutation_per_epoch_at_every_block(toy_features):
    s, _ = compile_uniform(toy_features)
    assert np.array_equal(np.sort(s[:200]), np.arange(200))
    for end in range(BLOCK_DRAWS, s.size + 1, BLOCK_DRAWS):
        counts = np.bincount(s[:end], minlength=200)
        assert counts.max() - counts.min() <= 1


def test_epochs_are_shuffled_differently(toy_features):
    s, _ = compile_uniform(toy_features)
    assert not np.array_equal(s[:200], s[200:400])
    assert not np.array_equal(s[:200], np.sort(s[:200]))
    # A batch mixes episodes: the mean number of distinct episodes in a batch of 4 from 4 equal
    # episodes is 2.73 for independent draws, so sorted or clumped picks would fall well below it.
    episodes = toy_features["episode_index"].to_numpy()[s].reshape(-1, 4)
    assert np.mean([len(set(batch)) for batch in episodes.tolist()]) > 2.4


def test_exposures_follow_weights(toy_features):
    obj = weighted_policy({"op": "add", "args": [{"const": 1.0}, {"op": "mul", "args": [{"const": 2.0}, {"op": "ge", "args": [{"feature": "t_norm"}, {"const": 0.5}]}]}]})
    s, summary = compile_mod.compile_schedule(parse(obj, toy_features), toy_features, None, TWO_EPOCHS)
    counts = np.bincount(s, minlength=200)
    late = toy_features["t_norm"].to_numpy() >= 0.5
    assert set(counts[late].tolist()) == {3} and set(counts[~late].tolist()) == {1}
    assert summary["max_abs_exposure_deviation"] < 1e-9


THREE_TO_ONE = {"op": "add", "args": [{"const": 1.0}, {"op": "mul", "args": [{"const": 2.0}, {"op": "ge", "args": [{"feature": "t_norm"}, {"const": 0.5}]}]}]}
SKEWED = {"op": "exp", "args": [{"op": "mul", "args": [{"const": 4.0}, {"feature": "t_norm"}]}]}
# 400 blocks of 20 draws over the 200 toy frames: enough blocks to measure their composition.
LONG = dataclasses.replace(TWO_EPOCHS, steps=2000)


def test_skewed_weights_stay_close_to_targets(toy_features):
    _, summary = compile_mod.compile_schedule(parse(weighted_policy(SKEWED), toy_features), toy_features, None, LONG)
    assert summary["max_abs_exposure_deviation"] <= 1.5


@pytest.mark.parametrize("weight, late_share", [(THREE_TO_ONE, 0.75), (SKEWED, None)])
def test_blocks_are_mixed_under_unequal_weights(toy_features, weight, late_share):
    s, _ = compile_mod.compile_schedule(parse(weighted_policy(weight), toy_features), toy_features, None, LONG)
    late = toy_features["t_norm"].to_numpy()[s].reshape(-1, BLOCK_DRAWS) >= 0.5
    share = late.mean(axis=1)
    if late_share is not None:
        assert abs(share.mean() - late_share) < 0.01
    # Independent draws give sd(share) = sqrt(m (1 - m) / 20); waves of one kind of frame give about 0.4.
    assert share.std() < 1.5 * np.sqrt(share.mean() * (1 - share.mean()) / BLOCK_DRAWS)
    assert abs(np.corrcoef(share[:-1], share[1:])[0, 1]) < 0.2


def test_uniform_schedule_is_pinned(toy_features):
    """The measured baseline used this exact rule; a change here means re-measuring it."""
    _, summary = compile_uniform(toy_features)
    assert summary["schedule_sha256"] == "c1ff5d6a6a5dc2ea910e2cc7ba2bc183275dcbbe3a751b52b5da70c79945f4ac"


def test_small_support_takes_several_copies_per_block(toy_features):
    obj = uniform_policy()
    obj["anchors"] = {"min_frame": 10, "max_frame": 14, "max_chunk_pad": 0}
    obj["episodes"] = {"include": [1], "exclude": []}
    s, summary = compile_mod.compile_schedule(parse(obj, toy_features), toy_features, None, TWO_EPOCHS)
    assert s.size == TWO_EPOCHS.exposures
    counts = np.bincount(s, minlength=200)
    assert np.count_nonzero(counts) == 5 and counts.max() == counts[counts > 0].min() == 80
    for block in s.reshape(-1, BLOCK_DRAWS):
        assert np.all(np.bincount(block)[np.unique(block)] == 4)


def test_excluded_frames_get_nothing_after_a_stage_change(toy_features):
    obj = weighted_policy(None, stages=[
        {"until": 0.5, "weight": {"const": 1.0}, "mask": None},
        {"until": 1.0, "weight": {"const": 1.0}, "mask": {"op": "lt", "args": [{"feature": "t_norm"}, {"const": 0.5}]}},
    ])
    s, _ = compile_mod.compile_schedule(parse(obj, toy_features), toy_features, None, TWO_EPOCHS)
    t = toy_features["t_norm"].to_numpy()[s]
    assert np.all(t[s.size // 2:] < 0.5)
    assert np.any(t[: s.size // 2] >= 0.5)


def test_balanced_draw_is_deterministic(toy_features):
    s1, _ = compile_uniform(toy_features)
    s2, _ = compile_uniform(toy_features)
    assert np.array_equal(s1, s2)


def test_iid_rule_still_reproduces_independent_draws(toy_features):
    proto = dataclasses.replace(TWO_EPOCHS, draw="iid")
    s, summary = compile_uniform(toy_features, proto)
    rng = np.random.default_rng([proto.schedule_seed, 0])
    p = np.full(200, 1 / 200)
    expected = np.concatenate([rng.choice(200, size=BLOCK_DRAWS, replace=True, p=p) for _ in range(20)])
    assert np.array_equal(s, expected)
    assert summary["draw"] == "iid"


def test_unknown_draw_rule_is_rejected(toy_features):
    with pytest.raises(ValueError, match="schedule_draw"):
        compile_uniform(toy_features, dataclasses.replace(TWO_EPOCHS, draw="stratified"))


def test_fixed_protocol_states_the_rule(protocol):
    assert protocol["training"]["schedule_draw"] == "balanced"
    assert compile_mod.protocol_from_dict(protocol).draw == "balanced"
    missing = copy.deepcopy(protocol)
    del missing["training"]["schedule_draw"]
    with pytest.raises(KeyError):
        compile_mod.protocol_from_dict(missing)
