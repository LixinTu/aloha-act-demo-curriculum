import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from aloha_curriculum import features as features_mod

ENV_DIR = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def protocol() -> dict:
    return json.loads((ENV_DIR / "fixed_train.json").read_text())


@pytest.fixture(scope="session")
def toy_features() -> pd.DataFrame:
    """A small synthetic dataset: 4 episodes of 50 frames, chunk_size 10, same columns as the real table."""
    rng = np.random.default_rng(0)
    rows = []
    for e in range(4):
        L = 50
        for f in range(L):
            rows.append({"episode_index": e, "frame_index": f, "global_index": e * L + f, "t_norm": f / (L - 1), "frames_to_end": L - 1 - f, "chunk_pad": max(0, f + 10 - L)})
    df = pd.DataFrame(rows)
    for m in features_mod.MOTORS:
        df[f"state_{m}"] = rng.normal(size=len(df))
        df[f"action_{m}"] = rng.normal(size=len(df))
    for c in ["cmd_err_l2", "cmd_err_left", "cmd_err_right", "speed_l2", "speed_left", "speed_right", "accel_l2", "jerk_l2", "action_speed_l2", "action_speed_left", "action_speed_right", "gripper_left_cmd", "gripper_right_cmd", "gripper_left_vel", "gripper_right_vel", "chunk_path_l2", "chunk_speed_mean", "chunk_gripper_left_delta", "chunk_gripper_right_delta", "ep_length", "ep_path_l2", "ep_speed_mean", "ep_speed_max", "ep_jerk_mean", "ep_gripper_left_min", "ep_gripper_right_min", "ep_cmd_err_mean"]:
        df[c] = rng.uniform(size=len(df))
    df["ep_length"] = 50.0
    return df
