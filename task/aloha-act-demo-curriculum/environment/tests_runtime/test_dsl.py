import numpy as np
import pytest

from aloha_curriculum import dsl
from aloha_curriculum.compile import build_context


def ctx_from(df, variables=None):
    return build_context(df, None, variables or {"t_train": 0.25, "t_stage": 0.5})


def test_const_and_arith(toy_features):
    ctx = ctx_from(toy_features)
    e = {"op": "add", "args": [{"const": 2}, {"op": "mul", "args": [{"feature": "t_norm"}, {"const": 3}]}]}
    dsl.validate_expression(e, feature_names=set(toy_features.columns), score_names=set())
    out = dsl.evaluate(e, ctx)
    assert np.allclose(out, 2 + 3 * toy_features["t_norm"].to_numpy())


def test_vars(toy_features):
    ctx = ctx_from(toy_features, {"t_train": 0.4, "t_stage": 0.1})
    assert np.allclose(dsl.evaluate({"var": "t_train"}, ctx), 0.4)
    assert np.allclose(dsl.evaluate({"var": "t_stage"}, ctx), 0.1)


@pytest.mark.parametrize(
    "bad",
    [
        {"op": "eval", "args": [{"const": 1}]},
        {"const": float("inf")},
        {"const": 1e9},
        {"feature": "nope"},
        {"score": "nope"},
        {"var": "t_nope"},
        {"op": "add", "args": [{"const": 1}]},
        {"op": "pow", "args": [{"const": 2}, {"const": 9}]},
        {"op": "shift", "args": [{"const": 2}, {"feature": "t_norm"}]},
        {"op": "window_mean", "args": [{"const": 2}, {"const": 251}]},
        {"op": "add", "args": [{"const": 1}, {"const": 2}], "extra": 1},
        "not a node",
        {"op": "add", "args": [{"const": 1}, "x"]},
    ],
)
def test_structural_rejections(toy_features, bad):
    with pytest.raises(dsl.DSLError):
        dsl.validate_expression(bad, feature_names=set(toy_features.columns), score_names=set())


def test_depth_and_node_limits(toy_features):
    e = {"const": 1.0}
    for _ in range(dsl.MAX_DEPTH + 1):
        e = {"op": "neg", "args": [e]}
    with pytest.raises(dsl.DSLError):
        dsl.validate_expression(e, feature_names=set(toy_features.columns), score_names=set())
    wide = {"const": 0.0}
    for _ in range(dsl.MAX_NODES):
        wide = {"op": "add", "args": [wide, {"const": 1.0}]}
    with pytest.raises(dsl.DSLError):
        dsl.validate_expression(wide, feature_names=set(toy_features.columns), score_names=set())


def test_safe_numerics(toy_features):
    ctx = ctx_from(toy_features)
    assert np.all(np.isfinite(dsl.evaluate({"op": "div", "args": [{"const": 1}, {"const": 0}]}, ctx)))
    assert np.all(np.isfinite(dsl.evaluate({"op": "exp", "args": [{"const": 1000}]}, ctx)))
    assert np.all(np.isfinite(dsl.evaluate({"op": "log1p", "args": [{"const": -5}]}, ctx)))
    assert np.all(np.isfinite(dsl.evaluate({"op": "sqrt", "args": [{"const": -5}]}, ctx)))
    assert np.all(np.isfinite(dsl.evaluate({"op": "pow", "args": [{"const": 0}, {"const": -2}]}, ctx)))


def test_episode_ops(toy_features):
    ctx = ctx_from(toy_features)
    fi = toy_features["frame_index"].to_numpy().astype(float)
    assert np.allclose(dsl.evaluate({"op": "ep_mean", "args": [{"feature": "frame_index"}]}, ctx), 24.5)
    assert np.allclose(dsl.evaluate({"op": "ep_max", "args": [{"feature": "frame_index"}]}, ctx), 49)
    cond = {"op": "ge", "args": [{"feature": "frame_index"}, {"const": 30}]}
    assert np.allclose(dsl.evaluate({"op": "ep_first_where", "args": [cond]}, ctx), 30)
    assert np.allclose(dsl.evaluate({"op": "ep_last_where", "args": [cond]}, ctx), 49)
    never = {"op": "ge", "args": [{"feature": "frame_index"}, {"const": 1000}]}
    assert np.allclose(dsl.evaluate({"op": "ep_first_where", "args": [never]}, ctx), -1)
    cs = dsl.evaluate({"op": "ep_cumsum", "args": [{"const": 1}]}, ctx)
    assert np.allclose(cs, fi + 1)
    # shift by +1: value of the previous frame, edge-padded
    sh = dsl.evaluate({"op": "shift", "args": [{"feature": "frame_index"}, {"const": 1}]}, ctx)
    assert np.allclose(sh, np.maximum(fi - 1, 0))
    fm = dsl.evaluate({"op": "future_mean", "args": [{"feature": "frame_index"}, {"const": 2}]}, ctx)
    assert np.isclose(fm[0], 1.0) and np.isclose(fm[49], 49.0)
    wm = dsl.evaluate({"op": "window_max", "args": [{"feature": "frame_index"}, {"const": 2}]}, ctx)
    assert np.isclose(wm[0], 2.0) and np.isclose(wm[49], 49.0) and np.isclose(wm[10], 12.0)
    wmin = dsl.evaluate({"op": "window_min", "args": [{"feature": "frame_index"}, {"const": 2}]}, ctx)
    assert np.isclose(wmin[0], 0.0) and np.isclose(wmin[10], 8.0)
    r = dsl.evaluate({"op": "rank", "args": [{"feature": "global_index"}]}, ctx)
    assert np.isclose(r.min(), 0.0) and np.isclose(r.max(), 1.0)
    er = dsl.evaluate({"op": "ep_rank", "args": [{"feature": "episode_index"}]}, ctx)
    assert set(np.unique(er).round(6)) == {0.0, 1 / 3, 2 / 3, 1.0} or np.allclose(sorted(np.unique(er)), [0, 1 / 3, 2 / 3, 1])
