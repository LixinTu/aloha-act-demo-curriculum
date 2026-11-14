FROM nvidia/cuda:12.8.1-base-ubuntu24.04

ARG DEBIAN_FRONTEND=noninteractive
ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MUJOCO_GL=egl \
    PYOPENGL_PLATFORM=egl \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics \
    HF_HUB_OFFLINE=1 \
    HF_DATASETS_OFFLINE=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    TORCH_HOME=/opt/aloha_assets/torch \
    PATH=/opt/aloha_venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

# System libraries: Python 3.12, EGL/GL for MuJoCo (NVIDIA EGL via the container toolkit's
# graphics capability, Mesa EGL/llvmpipe as a software fallback), ffmpeg for AV1 decoding.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates curl git build-essential \
        python3.12 python3.12-venv python3.12-dev \
        libgl1 libegl1 libglvnd0 libgles2 libegl-mesa0 libgl1-mesa-dri libglib2.0-0 \
        ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# glvnd vendor registration for NVIDIA EGL. The NVIDIA Container Toolkit (graphics capability)
# mounts libEGL_nvidia.so.0 but not this ICD file, so without it the EGL loader silently
# falls back to Mesa llvmpipe (software rendering, ~11x slower at 480x640). When no NVIDIA
# library is mounted, glvnd skips this vendor and Mesa is used.
RUN mkdir -p /usr/share/glvnd/egl_vendor.d \
    && printf '{\n    "file_format_version" : "1.0.0",\n    "ICD" : {\n        "library_path" : "libEGL_nvidia.so.0"\n    }\n}\n' > /usr/share/glvnd/egl_vendor.d/10_nvidia.json

RUN python3.12 -m venv /opt/aloha_venv \
    && /opt/aloha_venv/bin/python -m pip install --no-cache-dir "pip==25.3"

# Exact Python environment: the freeze of the development venv. torch/torchvision come from
# the PyTorch cu128 index only; every other wheel (including the CPU torchcodec build that
# decodes through the system FFmpeg) comes from PyPI only, so no package can be served by
# the wrong index. lerobot is pinned to the v0.6.1 commit.
COPY requirements-torch.lock requirements.lock /opt/aloha_build/
RUN /opt/aloha_venv/bin/python -m pip install --no-cache-dir --no-deps \
        --index-url=https://download.pytorch.org/whl/cu128 \
        -r /opt/aloha_build/requirements-torch.lock \
    && /opt/aloha_venv/bin/python -m pip install --no-cache-dir --no-deps \
        --index-url=https://pypi.org/simple \
        -r /opt/aloha_build/requirements.lock \
    && /opt/aloha_venv/bin/python -c "import lerobot, gym_aloha, mujoco, torch, torchvision, torchcodec; assert lerobot.__version__ == '0.6.1', lerobot.__version__; assert torch.__version__ == '2.11.0+cu128', torch.__version__; assert torchcodec.__version__ == '0.11.1+cpu', torchcodec.__version__; assert torch.cuda.is_available() or True"

# Task-owned runtime package (installed non-editable; sources also kept readable for agents).
COPY pyproject.toml /opt/aloha_public/pyproject.toml
COPY aloha_curriculum /opt/aloha_public/aloha_curriculum
RUN /opt/aloha_venv/bin/python -m pip install --no-cache-dir --no-deps /opt/aloha_public \
    && find /opt/aloha_public -name __pycache__ -type d -prune -exec rm -rf {} +

# Fixed protocol, baseline policy, pinned asset digests, build scripts.
COPY fixed_train.json default_policy.json pins.json /opt/aloha_public/
COPY assets.sha256 prepare_assets.py build_manifest.py /opt/aloha_build/

# Build-time network: dataset revision, reference checkpoint revision, ResNet18 weights.
RUN HF_HUB_OFFLINE=0 /opt/aloha_venv/bin/python /opt/aloha_build/prepare_assets.py \
        --protocol /opt/aloha_public/fixed_train.json \
        --versions /opt/aloha_public/pins.json \
        --assets-sha256 /opt/aloha_build/assets.sha256 \
        --assets-dir /opt/aloha_assets \
        --public-dir /opt/aloha_public \
        --torch-home /opt/aloha_assets/torch \
    && /opt/aloha_venv/bin/python /opt/aloha_build/build_manifest.py \
        --roots /opt/aloha_public /opt/aloha_assets \
        --output /opt/aloha_public/manifest.json \
        --lerobot-commit 7e241bd630a3719a56157a497ce5d08f244784f1 \
    && chmod -R a-w /opt/aloha_public /opt/aloha_assets \
    && rm -rf /root/.cache/huggingface /root/.cache/pip

# Agent workspace: only the candidate directory (seeded with the uniform baseline) and a
# free research directory. The Judge rejects any other top-level entry.
RUN mkdir -p /workspace/curriculum /workspace/research \
    && install -m 0644 /opt/aloha_public/default_policy.json /workspace/curriculum/policy.json \
    && test "$(find /workspace -mindepth 1 -maxdepth 1 -printf '%f\n' | sort | tr '\n' ' ')" = "curriculum research "

WORKDIR /workspace
