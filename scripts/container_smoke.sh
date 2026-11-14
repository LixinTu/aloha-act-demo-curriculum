#!/usr/bin/env bash
# Smoke-test the built task image with the GPU: public tools, short train/eval, Judge preflight.
# Usage: scripts/container_smoke.sh [image-tag]   (default aloha-act-demo-curriculum:dev)
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$root"
image="${1:-aloha-act-demo-curriculum:dev}"
tests="$root/task/aloha-act-demo-curriculum/tests"
sg docker -c "docker run --rm --gpus all --shm-size=8g -e SMOKE_SKIP_TRAIN='${SMOKE_SKIP_TRAIN:-0}' -v '$tests:/tests:ro' '$image' bash -c '
set -euo pipefail
echo \"== image facts ==\"; python -c \"import torch, torchcodec, lerobot, mujoco; print(torch.__version__, torchcodec.__version__, lerobot.__version__, mujoco.__version__, \\\"cuda\\\", torch.cuda.is_available(), torch.cuda.get_device_name(0))\"
ls /workspace; ls /workspace/curriculum; test -f /opt/aloha_public/manifest.json && echo manifest ok
echo \"== validate baseline candidate ==\"
aloha-validate --candidate-dir /workspace/curriculum --features /opt/aloha_public/features.parquet --protocol /opt/aloha_public/fixed_train.json | head -12
echo \"== compile 100-step diagnostic schedule ==\"
aloha-compile --candidate-dir /workspace/curriculum --features /opt/aloha_public/features.parquet --protocol /opt/aloha_public/fixed_train.json --output /tmp/smoke/schedule --steps 100 | grep -E \"schedule_sha256|exposures\"
if [ \"\${SMOKE_SKIP_TRAIN:-0}\" = \"1\" ]; then echo \"== training step skipped (SMOKE_SKIP_TRAIN=1) ==\"; else
echo \"== 100-step training ==\"
aloha-train --schedule /tmp/smoke/schedule/schedule.npy --protocol /opt/aloha_public/fixed_train.json --dataset-root /opt/aloha_assets/aloha_sim_insertion_human --output /tmp/smoke/train --steps 100 --log-every 50 | grep -E \"steps_per_s|peak_gpu_mem_gb|official_budget\"
echo \"== 5-episode proxy evaluation (EGL in container) ==\"
aloha-eval --checkpoint /tmp/smoke/train/pretrained_model --protocol /opt/aloha_public/fixed_train.json --output /tmp/smoke/eval --n-episodes 5 --n-envs 5 | grep -E \"success_rate\\\"|eval_s|mujoco_gl|n_episodes\"
fi
echo \"== reference policy loads and evaluates (2 episodes) ==\"
aloha-eval --checkpoint /opt/aloha_assets/reference_policy --protocol /opt/aloha_public/fixed_train.json --output /tmp/smoke/eval_ref --n-episodes 2 --n-envs 2 | grep -E \"successes|eval_s\"
echo \"== Judge preflight + manifest verification (as test.sh would run it) ==\"
env -i HOME=/tmp/judge-home PATH=/opt/aloha_venv/bin:/usr/bin:/bin MUJOCO_GL=egl HF_HUB_OFFLINE=1 TORCH_HOME=/opt/aloha_assets/torch /opt/aloha_venv/bin/python -I -c \"import sys; sys.path.insert(0, \\\"/tests\\\"); import bootstrap; bootstrap.preflight(); m = bootstrap.verify_manifest(); print(\\\"manifest verified:\\\", len(m[\\\"files\\\"]), \\\"files\\\", m[\\\"packages\\\"][\\\"lerobot\\\"], m[\\\"packages\\\"][\\\"torch\\\"])\"
echo \"== workspace write check (Work user is root in the harness) ==\"
touch /workspace/research/.probe && rm /workspace/research/.probe && echo research writable
echo SMOKE OK
'"
