#!/usr/bin/env bash
# Guard: given the same frame order, aloha-train must produce the same weights as LeRobot v0.6.1's
# own lerobot-train, bit for bit, in deterministic mode. Run it after any change to train.py, the
# fixed protocol or the pinned packages. Differences that once slipped through: image statistics
# (protocol v1) and scaling images in the data workers.
# Usage: scripts/check_trainer_parity.sh [steps] [out_root]
#   steps defaults to 300 (about 2 minutes); 3125 is one whole epoch, every frame once (about 17 minutes).
#   out_root defaults to experiments/trainer_parity; output goes to <out_root>/steps<steps>/.
#   SAVE_FREQ=<n> makes lerobot-train also save a checkpoint every n steps, as the control run did (every
#   20,000); saving reads the random state without changing it. Default: save only at the end.
# Exit 0: every tensor identical. Exit 1: they differ, or a step failed. Exit 2: output not empty.
# Exit 3: .venv-control does not match its pinned package list.
set -euo pipefail
steps="${1:-300}"; out_root="${2:-experiments/trainer_parity}"
save_freq="${SAVE_FREQ:-$steps}"
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"
E=task/aloha-act-demo-curriculum/environment
out="$out_root/steps$steps"
if [ -e "$out" ] && [ -n "$(ls -A "$out")" ]; then echo "refusing to overwrite non-empty $out" >&2; exit 2; fi
mkdir -p "$out"
export MUJOCO_GL=egl HF_HUB_OFFLINE=1 PYTHONHASHSEED=0
if ! diff <(uv pip freeze --python .venv-control/bin/python | sort) pins/control-venv.freeze.txt >/dev/null; then
  echo ".venv-control does not match pins/control-venv.freeze.txt" >&2; exit 3
fi
.venv/bin/python scripts/trainer_parity.py order --protocol "$E/fixed_train.json" \
  --dataset-root data/aloha_sim_insertion_human --steps "$steps" \
  --output-order "$out/order.npy" --output-protocol "$out/protocol_deterministic.json"
read -r rev seed batch < <(.venv/bin/python -I -c "import json,sys; p=json.load(open(sys.argv[1])); print(p['dataset']['revision'], p['training']['seed'], p['training']['batch_size'])" "$E/fixed_train.json")
.venv/bin/aloha-train --schedule "$out/order.npy" --protocol "$out/protocol_deterministic.json" \
  --dataset-root data/aloha_sim_insertion_human --output "$out/aloha_train" --steps "$steps" > "$out/aloha_train.json"
.venv-control/bin/lerobot-train \
  --policy.type=act --policy.device=cuda --policy.push_to_hub=false \
  --dataset.repo_id=lerobot/aloha_sim_insertion_human --dataset.root=data/aloha_sim_insertion_human \
  --dataset.revision="$rev" --cudnn_deterministic=true \
  --seed="$seed" --batch_size="$batch" --steps="$steps" --num_workers=4 --log_freq=100 \
  --save_freq="$save_freq" --wandb.enable=false --output_dir="$out/lerobot_train" > "$out/lerobot_train.log" 2>&1
.venv/bin/python scripts/trainer_parity.py compare "$out/aloha_train/pretrained_model/model.safetensors" \
  "$out/lerobot_train/checkpoints/last/pretrained_model/model.safetensors" --steps "$steps" \
  --save-freq "$save_freq" --output "$out/result.json"
