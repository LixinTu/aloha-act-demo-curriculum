#!/usr/bin/env bash
# Control run: the uniform baseline trained by LeRobot v0.6.1's own lerobot-train instead of aloha-train.
# lerobot-train samples with EpisodeAwareSampler (shuffled epochs, no replacement) and its own training
# loop; everything else matches the fixed protocol: dataset revision, ACT defaults (identical to the
# protocol's policy section in v0.6.1), seed, batch 8, 100,000 steps, ImageNet image statistics. The
# checkpoint is scored by the task's evaluator on the Judge's hidden block and the proxy block.
# It separates "aloha-train deviates from upstream" from "the released checkpoint's older LeRobot
# (3c0a209) and dataset format train differently".
# Training runs in .venv-control: the pinned lock files plus accelerate (lerobot-train needs it; the
# main venv does not have it). Its package list must equal pins/control-venv.freeze.txt.
# Usage: scripts/run_lerobot_train_control.sh [seed] [out_root]
#   defaults: seed 1000, out_root experiments/control_lerobot_train. Refuses a non-empty output directory.
#   After an interruption, RESUME=1 with the same arguments continues from the last saved checkpoint.
#   Checkpoints every 20,000 steps, as the released run did; saving does not change training.
#   DETERMINISTIC=1 adds --cudnn_deterministic=true (cudnn.deterministic on, cudnn.benchmark off, TF32
#   unchanged), the mode of scripts/check_trainer_parity.sh; the checkpoint's train_config.json records
#   it. Default: the control's normal mode. Use a different out_root, e.g. experiments/control_lerobot_train_det.
set -euo pipefail
seed="${1:-1000}"; out_root="${2:-experiments/control_lerobot_train}"
mode=()
if [ "${DETERMINISTIC:-0}" = 1 ]; then mode=(--cudnn_deterministic=true); fi
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"
E=task/aloha-act-demo-curriculum/environment
out="$out_root/seed$seed"
last="$out/train/checkpoints/last"
if [ "${RESUME:-0}" = 1 ]; then
  [ -f "$last/pretrained_model/train_config.json" ] || { echo "nothing to resume in $out" >&2; exit 2; }
elif [ -e "$out" ] && [ -n "$(ls -A "$out")" ]; then
  echo "refusing to overwrite non-empty $out" >&2; exit 2
fi
source scripts/require_free_gpu.sh  # exit 4 while another process computes on the GPU
mkdir -p "$out"
export MUJOCO_GL=egl HF_HUB_OFFLINE=1 PYTHONHASHSEED=0
if ! diff <(uv pip freeze --python .venv-control/bin/python | sort) pins/control-venv.freeze.txt >/dev/null; then
  echo ".venv-control does not match pins/control-venv.freeze.txt" >&2; exit 3
fi
cp pins/control-venv.freeze.txt "$out/control-venv.freeze.txt"
if [ "${RESUME:-0}" = 1 ]; then
  .venv-control/bin/lerobot-train --config_path="$last/pretrained_model/train_config.json" --resume=true
else
  rev=$(.venv/bin/python -I -c "import json,sys; print(json.load(open(sys.argv[1]))['dataset']['revision'])" "$E/fixed_train.json")
  .venv-control/bin/lerobot-train \
    --policy.type=act --policy.device=cuda --policy.push_to_hub=false \
    --dataset.repo_id=lerobot/aloha_sim_insertion_human --dataset.root=data/aloha_sim_insertion_human \
    --dataset.revision="$rev" ${mode[@]+"${mode[@]}"} \
    --seed="$seed" --batch_size=8 --steps=100000 --num_workers=4 --log_freq=200 \
    --save_freq=20000 --wandb.enable=false \
    --output_dir="$out/train"
fi
step=$(.venv/bin/python -I -c "import json,sys; print(json.load(open(sys.argv[1]))['step'])" "$last/training_state/training_step.json")
[ "$step" = 100000 ] || { echo "last checkpoint is step $step, not 100000" >&2; exit 1; }
ckpt="$last/pretrained_model"
hidden_start=$(.venv/bin/python -I -c "import json,sys; print(json.load(open(sys.argv[1]))['start_seed'])" \
  task/aloha-act-demo-curriculum/tests/hidden_seeds.json)
.venv/bin/aloha-eval --checkpoint "$ckpt" --protocol $E/fixed_train.json \
  --output "$out/eval_hidden" --start-seed "$hidden_start" --n-episodes 500 --n-envs 10 > "$out/eval_hidden.json"
.venv/bin/aloha-eval --checkpoint "$ckpt" --protocol $E/fixed_train.json \
  --output "$out/eval_proxy" > "$out/eval_proxy.json"
echo "lerobot-train control seed $seed done"
