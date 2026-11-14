#!/usr/bin/env python3
"""Is the simulator side of the evaluation bit-reproducible across processes?

No policy and no torch: each child process resets gym-aloha AlohaInsertion-v0 with the same seed and
applies the same pseudo-random action sequence, then returns the joint-state trajectory and the
rendered camera frames. The parent compares the children pairwise and writes a JSON summary.
On 2026-11-12 (RTX 4070 Laptop GPU, driver 595.99.02, MUJOCO_GL=egl) the physics was identical in
every process, while 15-18% of the frames differed by one intensity level in 1-5 of 307,200 pixels,
which is why two evaluations of one checkpoint do not repeat episode by episode.

Usage: MUJOCO_GL=egl .venv/bin/python scripts/render_determinism.py [--runs 3] [--steps 120] [--output PATH]
Exit 0 whatever the outcome (it is a measurement); the summary says what was found.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

SEED = 100000  # first seed of the public proxy block


def child(out: Path, steps: int) -> int:
    import gymnasium as gym
    import gym_aloha  # noqa: F401 - registers the environments

    env = gym.make("gym_aloha/AlohaInsertion-v0", obs_type="pixels_agent_pos", render_mode="rgb_array")
    obs, _ = env.reset(seed=SEED)
    rng = np.random.default_rng(0)
    low, high = env.action_space.low, env.action_space.high
    base = np.asarray(obs["agent_pos"], dtype=np.float32)
    states, frames = [], []
    for _ in range(steps):
        action = np.clip(base + rng.normal(0.0, 0.15, size=base.shape).astype(np.float32), low, high)
        obs, _, terminated, truncated, _ = env.step(action)
        states.append(np.array(obs["agent_pos"]))
        frames.append(np.array(obs["pixels"]["top"]))
        if terminated or truncated:
            break
    env.close()
    np.savez(out, states=np.stack(states), frames=np.stack(frames))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--steps", type=int, default=120)
    ap.add_argument("--output", type=Path)
    ap.add_argument("--child", type=Path, help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.child:
        return child(args.child, args.steps)

    with tempfile.TemporaryDirectory() as tmp:
        runs = []
        for i in range(args.runs):
            path = Path(tmp) / f"run{i}.npz"
            subprocess.run([sys.executable, "-I", __file__, "--child", str(path), "--steps", str(args.steps)],
                           check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            data = np.load(path)
            runs.append((data["states"], data["frames"].astype(np.int16)))
    pairs = []
    for a in range(args.runs):
        for b in range(a + 1, args.runs):
            diff = np.abs(runs[a][1] - runs[b][1])
            pixels = (diff.max(axis=3) > 0).reshape(diff.shape[0], -1).sum(axis=1)
            differing = pixels > 0
            pairs.append({
                "runs": [a, b],
                "states_identical": bool(np.array_equal(runs[a][0], runs[b][0])),
                "frames": int(diff.shape[0]),
                "frames_differing": int(differing.sum()),
                "max_abs_pixel_difference": int(diff.max()),
                "pixels_differing_per_differing_frame_median": float(np.median(pixels[differing])) if differing.any() else 0.0,
                "pixels_differing_per_differing_frame_max": int(pixels.max()),
            })
    frame_shape = list(runs[0][1].shape[1:])
    out = {
        "gym_id": "gym_aloha/AlohaInsertion-v0",
        "seed": SEED,
        "steps": int(runs[0][0].shape[0]),
        "processes": args.runs,
        "mujoco_gl": os.environ.get("MUJOCO_GL", ""),
        "frame_shape": frame_shape,
        "pixels_per_frame": frame_shape[0] * frame_shape[1],
        "state_sha256": [hashlib.sha256(np.ascontiguousarray(s).tobytes()).hexdigest() for s, _ in runs],
        "frames_sha256": [hashlib.sha256(np.ascontiguousarray(f.astype(np.uint8)).tobytes()).hexdigest() for _, f in runs],
        "pairs": pairs,
        "physics_identical": all(p["states_identical"] for p in pairs),
        "rendering_identical": all(p["frames_differing"] == 0 for p in pairs),
    }
    text = json.dumps(out, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    print(text, end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
