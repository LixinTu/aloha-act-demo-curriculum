"""The evaluator's cuDNN mode follows the protocol's ``evaluation.cudnn_deterministic`` switch.

Background (2026-11-12): the evaluator always ran in benchmark mode, like ``lerobot-eval``, and two
evaluations of one bit-identical checkpoint gave 57 and 60 successes of 500.
"""

import pytest
import torch

from aloha_curriculum.evaluate import configure_cudnn


@pytest.fixture(autouse=True)
def restore_backend_flags():
    saved = (torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark, torch.backends.cuda.matmul.allow_tf32)
    yield
    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark, torch.backends.cuda.matmul.allow_tf32 = saved


def test_default_is_the_benchmark_mode_of_lerobot_eval():
    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False
    assert configure_cudnn({"n_envs": 10}) is False
    assert torch.backends.cudnn.benchmark is True
    assert torch.backends.cudnn.deterministic is False
    assert torch.backends.cuda.matmul.allow_tf32 is True


def test_switch_fixes_the_algorithm_choice():
    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = False, True
    assert configure_cudnn({"cudnn_deterministic": True}) is True
    assert torch.backends.cudnn.deterministic is True
    assert torch.backends.cudnn.benchmark is False
    assert torch.backends.cuda.matmul.allow_tf32 is True


def test_explicit_false_equals_the_default():
    assert configure_cudnn({"cudnn_deterministic": False}) is False
    assert torch.backends.cudnn.benchmark is True
    assert torch.backends.cudnn.deterministic is False
