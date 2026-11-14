"""Task-owned per-frame feature table for lerobot/aloha_sim_insertion_human.

The table is computed from the parquet data only (joint states and commanded actions);
no video is decoded. Rows are ordered by the dataset's global ``index`` and episodes are
contiguous, which the DSL's per-episode operators rely on.

Column groups (all float64 unless noted):
  identity   : episode_index, frame_index, global_index (int64)
  time       : t_norm, frames_to_end, chunk_pad
  raw        : state_<motor> x14, action_<motor> x14
  tracking   : cmd_err_l2, cmd_err_left, cmd_err_right
  kinematics : speed_l2, speed_left, speed_right, accel_l2, jerk_l2,
               action_speed_l2, action_speed_left, action_speed_right
  gripper    : gripper_left_cmd, gripper_right_cmd, gripper_left_vel, gripper_right_vel
  chunk      : chunk_path_l2, chunk_speed_mean, chunk_gripper_left_delta, chunk_gripper_right_delta
  episode    : ep_length, ep_path_l2, ep_speed_mean, ep_speed_max, ep_jerk_mean,
               ep_gripper_left_min, ep_gripper_right_min, ep_cmd_err_mean
"""

from __future__ import annotations

import glob
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

MOTORS = [
    "left_waist", "left_shoulder", "left_elbow", "left_forearm_roll", "left_wrist_angle",
    "left_wrist_rotate", "left_gripper", "right_waist", "right_shoulder", "right_elbow",
    "right_forearm_roll", "right_wrist_angle", "right_wrist_rotate", "right_gripper",
]
LEFT = slice(0, 7)
RIGHT = slice(7, 14)
LEFT_GRIPPER = 6
RIGHT_GRIPPER = 13
IDENTITY_COLUMNS = ("episode_index", "frame_index", "global_index")


class FeatureError(RuntimeError):
    pass


def _finite_diff(x: np.ndarray, fps: float) -> np.ndarray:
    """Forward difference scaled by fps; the last row reuses the backward difference."""
    d = np.empty_like(x)
    if x.shape[0] == 1:
        d[:] = 0.0
        return d
    d[:-1] = (x[1:] - x[:-1]) * fps
    d[-1] = d[-2]
    return d


def load_frames(dataset_root: Path) -> tuple[pd.DataFrame, dict]:
    info = json.loads((dataset_root / "meta" / "info.json").read_text())
    files = sorted(glob.glob(str(dataset_root / "data" / "chunk-*" / "file-*.parquet")))
    if not files:
        raise FeatureError(f"no data parquet files under {dataset_root}")
    frames = pd.concat([pq.read_table(f).to_pandas() for f in files], ignore_index=True)
    frames = frames.sort_values("index", kind="stable").reset_index(drop=True)
    n = len(frames)
    if n != int(info["total_frames"]):
        raise FeatureError(f"row count {n} != info.total_frames {info['total_frames']}")
    if not np.array_equal(frames["index"].to_numpy(), np.arange(n)):
        raise FeatureError("global index is not 0..N-1 after sorting")
    ep = frames["episode_index"].to_numpy()
    fr = frames["frame_index"].to_numpy()
    starts = np.flatnonzero(np.r_[True, ep[1:] != ep[:-1]])
    lengths = np.diff(np.r_[starts, n])
    for s, l in zip(starts, lengths, strict=True):
        if not np.array_equal(fr[s : s + l], np.arange(l)):
            raise FeatureError(f"episode {ep[s]} frames are not contiguous 0..{l - 1}")
    if not np.array_equal(ep[starts], np.arange(len(starts))):
        raise FeatureError("episodes are not ordered 0..E-1")
    return frames, info


def build_features(dataset_root: Path, chunk_size: int = 100) -> pd.DataFrame:
    frames, info = load_frames(dataset_root)
    fps = float(info["fps"])
    motors = info["features"]["observation.state"]["names"]["motors"]
    if list(motors) != MOTORS:
        raise FeatureError(f"unexpected motor names {motors}")
    state = np.stack(frames["observation.state"].to_numpy()).astype(np.float64)
    action = np.stack(frames["action"].to_numpy()).astype(np.float64)
    n = state.shape[0]
    ep = frames["episode_index"].to_numpy().astype(np.int64)
    fr = frames["frame_index"].to_numpy().astype(np.int64)
    starts = np.flatnonzero(np.r_[True, ep[1:] != ep[:-1]])
    lengths = np.diff(np.r_[starts, n])

    out: dict[str, np.ndarray] = {
        "episode_index": ep,
        "frame_index": fr,
        "global_index": np.arange(n, dtype=np.int64),
    }
    ep_len = np.repeat(lengths, lengths).astype(np.float64)
    out["t_norm"] = fr / np.maximum(ep_len - 1.0, 1.0)
    out["frames_to_end"] = ep_len - 1.0 - fr
    out["chunk_pad"] = np.maximum(0.0, fr + chunk_size - ep_len)
    for j, m in enumerate(MOTORS):
        out[f"state_{m}"] = state[:, j]
    for j, m in enumerate(MOTORS):
        out[f"action_{m}"] = action[:, j]
    err = action - state
    out["cmd_err_l2"] = np.linalg.norm(err, axis=1)
    out["cmd_err_left"] = np.linalg.norm(err[:, LEFT], axis=1)
    out["cmd_err_right"] = np.linalg.norm(err[:, RIGHT], axis=1)

    speed = np.zeros_like(state)
    accel = np.zeros_like(state)
    jerk = np.zeros_like(state)
    a_speed = np.zeros_like(action)
    chunk_path = np.zeros(n)
    chunk_speed_mean = np.zeros(n)
    chunk_gl = np.zeros(n)
    chunk_gr = np.zeros(n)
    for s, l in zip(starts, lengths, strict=True):
        sl = slice(int(s), int(s + l))
        v = _finite_diff(state[sl], fps)
        a = _finite_diff(v, fps)
        jk = _finite_diff(a, fps)
        speed[sl], accel[sl], jerk[sl] = v, a, jk
        av = _finite_diff(action[sl], fps)
        a_speed[sl] = av
        step_norm = np.linalg.norm(action[sl][1:] - action[sl][:-1], axis=1)  # length l-1
        cs = np.concatenate([[0.0], np.cumsum(step_norm)])
        idx = np.arange(l)
        end = np.minimum(idx + chunk_size - 1, l - 1)  # last frame of the chunk
        chunk_path[sl] = cs[end] - cs[idx]
        chunk_speed_mean[sl] = chunk_path[sl] * fps / np.maximum(end - idx, 1)
        chunk_gl[sl] = action[sl][end, LEFT_GRIPPER] - action[sl][idx, LEFT_GRIPPER]
        chunk_gr[sl] = action[sl][end, RIGHT_GRIPPER] - action[sl][idx, RIGHT_GRIPPER]

    out["speed_l2"] = np.linalg.norm(speed, axis=1)
    out["speed_left"] = np.linalg.norm(speed[:, LEFT], axis=1)
    out["speed_right"] = np.linalg.norm(speed[:, RIGHT], axis=1)
    out["accel_l2"] = np.linalg.norm(accel, axis=1)
    out["jerk_l2"] = np.linalg.norm(jerk, axis=1)
    out["action_speed_l2"] = np.linalg.norm(a_speed, axis=1)
    out["action_speed_left"] = np.linalg.norm(a_speed[:, LEFT], axis=1)
    out["action_speed_right"] = np.linalg.norm(a_speed[:, RIGHT], axis=1)
    out["gripper_left_cmd"] = action[:, LEFT_GRIPPER]
    out["gripper_right_cmd"] = action[:, RIGHT_GRIPPER]
    out["gripper_left_vel"] = a_speed[:, LEFT_GRIPPER]
    out["gripper_right_vel"] = a_speed[:, RIGHT_GRIPPER]
    out["chunk_path_l2"] = chunk_path
    out["chunk_speed_mean"] = chunk_speed_mean
    out["chunk_gripper_left_delta"] = chunk_gl
    out["chunk_gripper_right_delta"] = chunk_gr

    def broadcast(fn) -> np.ndarray:
        vals = np.array([fn(slice(int(s), int(s + l))) for s, l in zip(starts, lengths, strict=True)])
        return np.repeat(vals, lengths)

    out["ep_length"] = ep_len
    out["ep_path_l2"] = broadcast(lambda sl: float(np.sum(np.linalg.norm(action[sl][1:] - action[sl][:-1], axis=1))))
    out["ep_speed_mean"] = broadcast(lambda sl: float(np.mean(out["speed_l2"][sl])))
    out["ep_speed_max"] = broadcast(lambda sl: float(np.max(out["speed_l2"][sl])))
    out["ep_jerk_mean"] = broadcast(lambda sl: float(np.mean(out["jerk_l2"][sl])))
    out["ep_gripper_left_min"] = broadcast(lambda sl: float(np.min(action[sl][:, LEFT_GRIPPER])))
    out["ep_gripper_right_min"] = broadcast(lambda sl: float(np.min(action[sl][:, RIGHT_GRIPPER])))
    out["ep_cmd_err_mean"] = broadcast(lambda sl: float(np.mean(out["cmd_err_l2"][sl])))

    df = pd.DataFrame(out)
    for c in df.columns:
        if c in IDENTITY_COLUMNS:
            continue
        if not np.all(np.isfinite(df[c].to_numpy())):
            raise FeatureError(f"feature {c} contains non-finite values")
    return df


def feature_names(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in IDENTITY_COLUMNS] + list(IDENTITY_COLUMNS)


def write_features(df: pd.DataFrame, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False, engine="pyarrow", compression="zstd")
    return sha256_file(path)


def read_features(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path, engine="pyarrow")
    for c in IDENTITY_COLUMNS:
        if c not in df.columns:
            raise FeatureError(f"feature table lacks {c}")
    return df


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main(argv: list[str] | None = None) -> int:
    import argparse

    p = argparse.ArgumentParser(description="Build the public per-frame feature table.")
    p.add_argument("--dataset-root", required=True)
    p.add_argument("--output", required=True, help="parquet path")
    p.add_argument("--csv", default=None, help="optional CSV copy for agents")
    p.add_argument("--chunk-size", type=int, default=100)
    a = p.parse_args(argv)
    df = build_features(Path(a.dataset_root), a.chunk_size)
    digest = write_features(df, Path(a.output))
    if a.csv:
        Path(a.csv).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(a.csv, index=False, float_format="%.9g")
    print(json.dumps({"rows": len(df), "columns": len(df.columns), "sha256": digest, "output": os.fspath(a.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
