"""Fixed ACT trainer driven by a compiled exposure schedule.

This mirrors ``lerobot.scripts.lerobot_train`` (v0.6.1) for a single process without
mixed precision: same policy factory, same processors, same optimizer preset, same
forward/backward/clip/step sequence as ``update_policy``. The only intentional
difference is the sampler: instead of ``EpisodeAwareSampler`` (uniform shuffled epochs)
the DataLoader follows the compiled schedule exactly, so the sequence of training
batches is a pure function of (candidate, fixed protocol). Given the same frame order, the
weights are bit-identical to lerobot-train's in deterministic mode; scripts/check_trainer_parity.sh
checks this. Independent draws with replacement scored clearly below lerobot-train, so the
order is not neutral: see compile.py for how the schedule is drawn.

Camera images are normalised with ImageNet statistics, as ``make_dataset`` does when
``DatasetConfig.use_imagenet_stats`` is true (LeRobot's default and the setting of the released
checkpoint). Protocol version 1 built the dataset directly and so normalised images with the
dataset's own pixel statistics; ``check_image_normalization`` now refuses to train if the
preprocessor disagrees with the protocol.
"""

from __future__ import annotations

import json
import logging
import math
import os
import platform
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import torch

log = logging.getLogger("aloha_curriculum.train")


class ScheduleSampler(torch.utils.data.Sampler[int]):
    """Yield dataset indices in the exact order of the compiled schedule."""

    def __init__(self, schedule: np.ndarray):
        self._schedule = np.ascontiguousarray(schedule, dtype=np.int64)

    def __iter__(self) -> Iterator[int]:
        return iter(self._schedule.tolist())

    def __len__(self) -> int:
        return int(self._schedule.size)


def build_policy_config(protocol: dict, device: str):
    from lerobot.configs.types import NormalizationMode
    from lerobot.policies.act.configuration_act import ACTConfig

    p = dict(protocol["policy"])
    if p.pop("type") != "act":
        raise ValueError("fixed protocol must describe an ACT policy")
    p["normalization_mapping"] = {k: NormalizationMode(v) for k, v in p["normalization_mapping"].items()}
    cfg = ACTConfig(**p)
    cfg.device = device
    cfg.use_amp = bool(protocol["training"].get("use_amp", False))
    return cfg


def load_dataset(protocol: dict, dataset_root: Path, policy_cfg, video_backend: str | None):
    from lerobot.datasets.factory import resolve_delta_timestamps
    from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

    ds = protocol["dataset"]
    meta = LeRobotDatasetMetadata(ds["repo_id"], root=dataset_root, revision=ds["revision"])
    if meta.total_frames != ds["total_frames"] or meta.total_episodes != ds["total_episodes"]:
        raise RuntimeError("dataset metadata does not match the fixed protocol")
    if int(meta.fps) != int(ds["fps"]):
        raise RuntimeError("dataset fps does not match the fixed protocol")
    delta_timestamps = resolve_delta_timestamps(policy_cfg, meta)
    dataset = LeRobotDataset(
        ds["repo_id"],
        root=dataset_root,
        revision=ds["revision"],
        delta_timestamps=delta_timestamps,
        video_backend=video_backend,
        return_uint8=True,
    )
    if len(dataset) != ds["total_frames"]:
        raise RuntimeError(f"dataset has {len(dataset)} frames, protocol expects {ds['total_frames']}")
    if not isinstance(ds.get("use_imagenet_stats"), bool):
        raise RuntimeError("fixed protocol must set dataset.use_imagenet_stats to true or false")
    if ds["use_imagenet_stats"]:
        apply_imagenet_stats(dataset.meta)
    return dataset


def scale_images(batch: dict, camera_keys, device: str) -> dict:
    """Turn uint8 camera frames into float32 in [0, 1] on the training device, as lerobot-train does.

    CUDA computes ``x / 255.0`` as ``x * (1 / 255)``, which rounds 126 of the 256 pixel values
    differently from the CPU, by at most 6e-8. It changes nothing systematic, but scaling anywhere
    else would rule out bit-identical weights.
    """
    for key in camera_keys:
        if key in batch and batch[key].dtype == torch.uint8:
            batch[key] = batch[key].to(device).to(dtype=torch.float32) / 255.0
    return batch


def apply_imagenet_stats(meta) -> list[str]:
    """Replace the mean/std of every (non-depth) camera key with ImageNet statistics.

    Same rule as ``lerobot.datasets.factory.make_dataset`` (v0.6.1, lines 137-142); the
    ImageNet-pretrained ResNet18 backbone expects these statistics. Other statistics (min, max,
    count) and non-camera keys are left unchanged. Returns the camera keys that were changed.
    """
    from lerobot.utils.constants import IMAGENET_STATS

    changed = []
    for key in meta.camera_keys:
        if key in meta.depth_keys:
            continue
        for stats_type, stats in IMAGENET_STATS.items():
            meta.stats[key][stats_type] = torch.tensor(stats, dtype=torch.float32)
        changed.append(key)
    return changed


def check_image_normalization(preprocessor, camera_keys, use_imagenet_stats: bool) -> str:
    """Fail before training unless the preprocessor normalises images as the protocol states."""
    from lerobot.utils.constants import IMAGENET_STATS

    norm = [s for s in preprocessor.steps if type(s).__name__ == "NormalizerProcessorStep"]
    if len(norm) != 1:
        raise RuntimeError(f"expected one normalizer step in the preprocessor, found {len(norm)}")
    if not camera_keys:
        raise RuntimeError("dataset has no camera keys")
    for key in camera_keys:
        got = norm[0].stats[key]
        is_imagenet = all(
            np.allclose(torch.as_tensor(got[t]).detach().cpu().numpy(), np.asarray(v, dtype=np.float32))
            for t, v in IMAGENET_STATS.items()
        )
        if is_imagenet != use_imagenet_stats:
            raise RuntimeError(
                f"preprocessor normalises {key} with {'ImageNet' if is_imagenet else 'dataset'} statistics, "
                f"but the protocol sets use_imagenet_stats={use_imagenet_stats}"
            )
    return "imagenet" if use_imagenet_stats else "dataset"


def _optimizer_from_protocol(protocol: dict, policy):
    from lerobot.optim.optimizers import AdamWConfig

    o = protocol["optimizer"]
    if o["type"] != "adamw":
        raise ValueError("fixed protocol optimizer must be adamw")
    preset = policy.config.get_optimizer_preset()
    # The ACT preset carries lr / weight_decay / grad_clip from the policy config; assert the
    # protocol agrees so the two sources cannot drift apart silently.
    if not isinstance(preset, AdamWConfig):
        raise RuntimeError("ACT optimizer preset is not AdamW")
    if not math.isclose(preset.lr, o["lr"]) or not math.isclose(preset.weight_decay, o["weight_decay"]):
        raise RuntimeError("protocol optimizer fields disagree with the ACT optimizer preset")
    if list(preset.betas) != list(o["betas"]) or not math.isclose(preset.eps, o["eps"]):
        raise RuntimeError("protocol betas/eps disagree with the ACT optimizer preset")
    if not math.isclose(preset.grad_clip_norm, protocol["training"]["grad_clip_norm"]):
        raise RuntimeError("protocol grad_clip_norm disagrees with the ACT optimizer preset")
    params = policy.get_optim_params()
    return preset.build(params), preset.grad_clip_norm


def train(
    *,
    protocol: dict,
    schedule: np.ndarray,
    dataset_root: Path,
    output_dir: Path,
    device: str = "cuda",
    steps_override: int | None = None,
    seed_override: int | None = None,
    video_backend: str | None = None,
    log_every: int | None = None,
) -> dict:
    """Train ACT for exactly ``steps`` steps over ``schedule`` and save a checkpoint.

    Returns a JSON-serialisable summary that is also written to ``output_dir/train_info.json``.
    """
    from lerobot.policies.factory import make_policy, make_pre_post_processors
    from lerobot.utils.random_utils import set_seed

    tr = protocol["training"]
    steps = int(steps_override) if steps_override is not None else int(tr["steps"])
    batch_size = int(tr["batch_size"])
    seed = int(seed_override) if seed_override is not None else int(tr["seed"])
    official = steps_override is None and seed_override is None
    if schedule.size != steps * batch_size:
        raise RuntimeError(f"schedule has {schedule.size} exposures, expected steps*batch = {steps * batch_size}")

    output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(seed)
    if bool(tr.get("cudnn_deterministic", False)):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    policy_cfg = build_policy_config(protocol, device)
    dataset = load_dataset(protocol, dataset_root, policy_cfg, video_backend)
    if int(schedule.min()) < 0 or int(schedule.max()) >= len(dataset):
        raise RuntimeError("schedule references frames outside the dataset")

    policy = make_policy(cfg=policy_cfg, ds_meta=dataset.meta)
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_cfg, pretrained_path=None, dataset_stats=dataset.meta.stats
    )
    image_normalization = check_image_normalization(
        preprocessor, dataset.meta.camera_keys, protocol["dataset"]["use_imagenet_stats"]
    )
    optimizer, grad_clip_norm = _optimizer_from_protocol(protocol, policy)

    num_workers = int(tr["num_workers"])
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=ScheduleSampler(schedule),
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.startswith("cuda"),
        drop_last=False,
        prefetch_factor=2 if num_workers > 0 else None,
        persistent_workers=False,
    )
    if len(loader) != steps:
        raise RuntimeError(f"dataloader yields {len(loader)} batches, expected {steps}")

    log_every = int(log_every or tr.get("log_every", 200))
    policy.train()
    losses: list[float] = []
    l1s: list[float] = []
    kls: list[float] = []
    grad_norms: list[float] = []
    window: dict[str, list[float]] = {"loss": [], "l1": [], "kl": [], "gn": [], "data_s": [], "step_s": []}
    history: list[dict[str, float]] = []
    t_start = time.perf_counter()
    t_fetch = time.perf_counter()
    step = 0
    peak_mem = 0.0
    for batch in loader:
        data_s = time.perf_counter() - t_fetch
        t_step = time.perf_counter()
        batch = scale_images(batch, dataset.meta.camera_keys, device)
        batch = preprocessor(batch)
        loss, output = policy.forward(batch)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), grad_clip_norm, error_if_nonfinite=False)
        optimizer.step()
        optimizer.zero_grad()
        if hasattr(policy, "update") and callable(policy.update):
            policy.update()
        step += 1
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        step_s = time.perf_counter() - t_step
        lv = float(loss.item())
        if not math.isfinite(lv):
            raise RuntimeError(f"non-finite loss at step {step}")
        gn = float(grad_norm.item())
        l1 = float(output.get("l1_loss", float("nan"))) if output else float("nan")
        kl = float(output.get("kld_loss", output.get("kl_loss", float("nan")))) if output else float("nan")
        losses.append(lv)
        l1s.append(l1)
        kls.append(kl)
        grad_norms.append(gn)
        for k, v in (("loss", lv), ("l1", l1), ("kl", kl), ("gn", gn), ("data_s", data_s), ("step_s", step_s)):
            window[k].append(v)
        if step % log_every == 0 or step == steps:
            if device.startswith("cuda"):
                peak_mem = max(peak_mem, torch.cuda.max_memory_allocated() / 1e9)
            rec = {
                "step": step,
                "loss": float(np.mean(window["loss"])),
                "l1_loss": float(np.nanmean(window["l1"])),
                "kl_loss": float(np.nanmean(window["kl"])),
                "grad_norm": float(np.mean(window["gn"])),
                "data_s": float(np.mean(window["data_s"])),
                "step_s": float(np.mean(window["step_s"])),
                "elapsed_s": time.perf_counter() - t_start,
            }
            history.append(rec)
            log.info(
                "step %d/%d loss %.4f l1 %.4f kl %.4f gn %.2f data %.3fs step %.3fs elapsed %.0fs",
                step, steps, rec["loss"], rec["l1_loss"], rec["kl_loss"], rec["grad_norm"], rec["data_s"], rec["step_s"], rec["elapsed_s"],
            )
            for v in window.values():
                v.clear()
        t_fetch = time.perf_counter()
    if step != steps:
        raise RuntimeError(f"trained {step} steps, expected {steps}")
    train_s = time.perf_counter() - t_start

    ckpt = output_dir / "pretrained_model"
    policy.save_pretrained(ckpt)
    preprocessor.save_pretrained(ckpt)
    postprocessor.save_pretrained(ckpt)
    np.save(output_dir / "loss_curve.npy", np.stack([np.asarray(losses), np.asarray(l1s), np.asarray(kls), np.asarray(grad_norms)], axis=1))

    tail = max(1, min(len(losses), 1000))
    info: dict[str, Any] = {
        "official_budget": official,
        "steps": steps,
        "batch_size": batch_size,
        "exposures": int(schedule.size),
        "seed": seed,
        "device": device,
        "device_name": torch.cuda.get_device_name(0) if device.startswith("cuda") and torch.cuda.is_available() else platform.processor(),
        "cudnn_deterministic": bool(tr.get("cudnn_deterministic", False)),
        "protocol_version": protocol["protocol_version"],
        "image_normalization": image_normalization,
        "train_s": train_s,
        "steps_per_s": steps / train_s if train_s > 0 else None,
        "peak_gpu_mem_gb": peak_mem,
        "final_loss_mean_last_1000": float(np.mean(losses[-tail:])),
        "final_l1_mean_last_1000": float(np.nanmean(l1s[-tail:])),
        "final_kl_mean_last_1000": float(np.nanmean(kls[-tail:])),
        "history": history,
        "versions": _versions(),
        "checkpoint": os.fspath(ckpt),
    }
    (output_dir / "train_info.json").write_text(json.dumps(info, indent=2) + "\n")
    return info


def _versions() -> dict[str, str]:
    import importlib.metadata as md

    out = {"python": platform.python_version(), "torch": torch.__version__}
    for name in ("lerobot", "gym-aloha", "mujoco", "dm-control", "torchvision", "torchcodec", "av", "numpy", "gymnasium"):
        try:
            out[name] = md.version(name)
        except md.PackageNotFoundError:
            out[name] = "missing"
    return out
