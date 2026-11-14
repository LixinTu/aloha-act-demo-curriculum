"""Closed-loop evaluation of an ACT checkpoint in gym-aloha AlohaInsertion-v0.

Uses ``lerobot.scripts.lerobot_eval.eval_policy`` so the rollout semantics (seeding,
success = reward 4, episode_length 400) are exactly those of ``lerobot-eval``.
"""

from __future__ import annotations

import json
import logging
import os
import platform
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch

log = logging.getLogger("aloha_curriculum.evaluate")


def probe_gl_renderer() -> str:
    """Name of the OpenGL renderer MuJoCo's EGL context resolves to (GPU vs software)."""
    try:
        import mujoco
        from OpenGL import GL

        ctx = mujoco.GLContext(16, 16)
        try:
            ctx.make_current()
            renderer = GL.glGetString(GL.GL_RENDERER)
            return renderer.decode() if renderer else "unknown"
        finally:
            try:
                ctx.free()
            except Exception:  # noqa: BLE001 - glvnd teardown noise
                pass
    except Exception as exc:  # noqa: BLE001
        return f"unavailable ({type(exc).__name__})"


def configure_cudnn(ev: dict) -> bool:
    """Set the cuDNN mode for evaluation from the protocol's ``evaluation`` block.

    Returns True when ``cudnn_deterministic`` is set: the convolution algorithm choice is then fixed.
    Without it the evaluator behaves like ``lerobot-eval`` (benchmark mode): each process picks
    algorithms by timing, so two evaluations of one checkpoint differ in the last bits of the policy
    output and, through the contact-rich simulation, in episode outcomes (57 and 60 successes of 500
    for the same weights on 2026-11-12).
    """
    deterministic = bool(ev.get("cudnn_deterministic", False))
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cuda.matmul.allow_tf32 = True
    return deterministic


def load_policy_and_processors(checkpoint: Path, device: str):
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.envs.configs import AlohaEnv
    from lerobot.policies.factory import make_policy, make_pre_post_processors

    cfg = PreTrainedConfig.from_pretrained(str(checkpoint))
    cfg.pretrained_path = str(checkpoint)
    cfg.device = device
    env_cfg = AlohaEnv()
    policy = make_policy(cfg=cfg, env_cfg=env_cfg)
    policy.eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg,
        pretrained_path=str(checkpoint),
        preprocessor_overrides={"device_processor": {"device": str(policy.config.device)}},
    )
    return cfg, policy, preprocessor, postprocessor


def evaluate(
    *,
    protocol: dict,
    checkpoint: Path,
    start_seed: int,
    n_episodes: int,
    output_dir: Path,
    n_envs: int | None = None,
    device: str = "cuda",
    use_async_envs: bool = False,
    seed_everything: int = 0,
) -> dict:
    from lerobot.envs.configs import AlohaEnv
    from lerobot.envs.factory import make_env, make_env_pre_post_processors
    from lerobot.scripts.lerobot_eval import eval_policy
    from lerobot.utils.random_utils import set_seed

    ev = protocol["evaluation"]
    gym_id = ev["gym_id"]
    env_cfg = AlohaEnv(
        task=gym_id.split("/", 1)[1],
        episode_length=int(ev["episode_length"]),
        obs_type=ev["obs_type"],
    )
    if env_cfg.gym_id != gym_id:
        raise RuntimeError(f"env id mismatch: {env_cfg.gym_id} != {gym_id}")
    n_envs = int(n_envs or ev["n_envs"])
    n_envs = max(1, min(n_envs, n_episodes))

    output_dir.mkdir(parents=True, exist_ok=True)
    gl_renderer = probe_gl_renderer()
    log.info("OpenGL renderer: %s", gl_renderer)
    cudnn_deterministic = configure_cudnn(ev)
    set_seed(seed_everything)

    cfg, policy, preprocessor, postprocessor = load_policy_and_processors(checkpoint, device)
    env_pre, env_post = make_env_pre_post_processors(env_cfg=env_cfg, policy_cfg=cfg)
    envs = make_env(env_cfg, n_envs=n_envs, use_async_envs=use_async_envs)
    vec_env = envs[env_cfg.type][0]
    t0 = time.perf_counter()
    try:
        with torch.no_grad(), (torch.autocast(device_type="cuda") if cfg.use_amp else nullcontext()):
            info = eval_policy(
                vec_env,
                policy,
                env_pre,
                env_post,
                preprocessor,
                postprocessor,
                n_episodes=n_episodes,
                max_episodes_rendered=0,
                videos_dir=None,
                return_episode_data=False,
                start_seed=start_seed,
            )
    finally:
        vec_env.close()
    wall = time.perf_counter() - t0

    per = info["per_episode"]
    successes = np.array([bool(e["success"]) for e in per])
    max_rewards = np.array([float(e["max_reward"]) for e in per])
    seeds = [int(e["seed"]) for e in per]
    if len(per) != n_episodes:
        raise RuntimeError(f"evaluated {len(per)} episodes, expected {n_episodes}")
    expected = list(range(start_seed, start_seed + n_episodes))
    if seeds != expected:
        raise RuntimeError("episode seeds are not the expected contiguous block")
    hist = {str(k): int((max_rewards == k).sum()) for k in range(5)}
    p = float(successes.mean())
    result: dict[str, Any] = {
        "checkpoint": os.fspath(checkpoint),
        "gym_id": gym_id,
        "episode_length": int(ev["episode_length"]),
        "n_episodes": n_episodes,
        "n_envs": n_envs,
        "start_seed": start_seed,
        "successes": int(successes.sum()),
        "success_rate": p,
        "success_rate_se": float(np.sqrt(p * (1 - p) / n_episodes)) if n_episodes > 1 else None,
        "avg_max_reward": float(max_rewards.mean()),
        "avg_sum_reward": float(info["aggregated"]["avg_sum_reward"]),
        "max_reward_histogram": hist,
        "eval_s": wall,
        "eval_ep_s": wall / n_episodes,
        "device": device,
        "cudnn_deterministic": cudnn_deterministic,
        "mujoco_gl": os.environ.get("MUJOCO_GL", ""),
        "gl_renderer": gl_renderer,
        "platform": platform.platform(),
        "per_episode": [{"seed": s, "success": bool(e["success"]), "max_reward": float(e["max_reward"]), "sum_reward": float(e["sum_reward"])} for s, e in zip(seeds, per, strict=True)],
    }
    (output_dir / "eval_result.json").write_text(json.dumps(result, indent=2) + "\n")
    log.info("success %d/%d = %.4f (SE %.4f) in %.0fs", result["successes"], n_episodes, p, result["success_rate_se"] or 0.0, wall)
    return result
