#!/bin/bash
set -euo pipefail
cd /workspace
exec env -i \
  HOME=/run/aloha-judge-home \
  PATH=/opt/aloha_venv/bin:/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
  LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-/usr/local/nvidia/lib:/usr/local/nvidia/lib64}" \
  NVIDIA_VISIBLE_DEVICES=all \
  NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics \
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
  LANG=C.UTF-8 \
  MUJOCO_GL=egl \
  PYOPENGL_PLATFORM=egl \
  HF_HUB_OFFLINE=1 \
  HF_DATASETS_OFFLINE=1 \
  HF_HOME=/run/aloha-judge-hf \
  HF_LEROBOT_HOME=/run/aloha-judge-hf/lerobot \
  TORCH_HOME=/opt/aloha_assets/torch \
  OMP_NUM_THREADS=8 \
  PYTHONHASHSEED=0 \
  PYTHONDONTWRITEBYTECODE=1 \
  PYTHONNOUSERSITE=1 \
  PYTHONUNBUFFERED=1 \
  /opt/aloha_venv/bin/python -I /tests/evaluate.py
