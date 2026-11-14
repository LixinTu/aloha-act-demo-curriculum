"""Image normalisation of the fixed trainer (protocol version 2).

Protocol version 1 normalised camera images with the dataset's own pixel statistics (mean about
0.16), while LeRobot's training, and the released checkpoint, use ImageNet statistics. These tests
pin the corrected behaviour.
"""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from aloha_curriculum.train import apply_imagenet_stats, check_image_normalization, scale_images

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
CAM = "observation.images.top"


def _meta():
    return SimpleNamespace(
        camera_keys=[CAM],
        depth_keys=[],
        stats={
            CAM: {"mean": torch.full((3, 1, 1), 0.16), "std": torch.full((3, 1, 1), 0.2), "min": torch.zeros(3, 1, 1)},
            "observation.state": {"mean": torch.zeros(14), "std": torch.ones(14)},
        },
    )


class NormalizerProcessorStep:  # same class name as LeRobot's step; only .stats is read
    def __init__(self, stats):
        self.stats = stats


def test_fixed_protocol_uses_imagenet_stats(protocol):
    assert protocol["protocol_version"] == 3
    assert protocol["dataset"]["use_imagenet_stats"] is True


def test_apply_imagenet_stats_changes_only_camera_mean_and_std():
    meta = _meta()
    assert apply_imagenet_stats(meta) == [CAM]
    assert np.allclose(meta.stats[CAM]["mean"].flatten().numpy(), IMAGENET_MEAN)
    assert np.allclose(meta.stats[CAM]["std"].flatten().numpy(), IMAGENET_STD)
    assert meta.stats[CAM]["mean"].shape == (3, 1, 1)
    assert torch.equal(meta.stats[CAM]["min"], torch.zeros(3, 1, 1))
    assert torch.equal(meta.stats["observation.state"]["mean"], torch.zeros(14))


def test_apply_imagenet_stats_skips_depth_cameras():
    meta = _meta()
    meta.depth_keys = [CAM]
    assert apply_imagenet_stats(meta) == []
    assert np.allclose(meta.stats[CAM]["mean"].flatten().numpy(), 0.16)


def test_check_image_normalization_accepts_imagenet_and_rejects_dataset_stats():
    meta = _meta()
    pre = SimpleNamespace(steps=[NormalizerProcessorStep(meta.stats)])
    with pytest.raises(RuntimeError, match="dataset statistics"):
        check_image_normalization(pre, [CAM], use_imagenet_stats=True)
    assert check_image_normalization(pre, [CAM], use_imagenet_stats=False) == "dataset"
    apply_imagenet_stats(meta)
    assert check_image_normalization(pre, [CAM], use_imagenet_stats=True) == "imagenet"
    with pytest.raises(RuntimeError, match="ImageNet statistics"):
        check_image_normalization(pre, [CAM], use_imagenet_stats=False)


def test_scale_images_matches_lerobot_train():
    frames = torch.arange(256, dtype=torch.uint8).reshape(1, 1, 1, 256)
    state = torch.ones(2)
    batch = scale_images({CAM: frames.clone(), "observation.state": state}, [CAM], "cpu")
    assert batch[CAM].dtype == torch.float32
    assert torch.equal(batch[CAM], frames.to(dtype=torch.float32) / 255.0)
    assert batch["observation.state"] is state


def test_scale_images_leaves_float_frames_alone():
    frames = torch.full((1, 1, 1, 2), 0.5)
    assert scale_images({CAM: frames}, [CAM], "cpu")[CAM] is frames
