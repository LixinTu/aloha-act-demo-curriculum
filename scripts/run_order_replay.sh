#!/usr/bin/env bash
# Replay: aloha-train over the frame order lerobot-train uses at seed 1000, in normal mode like the control.
# Same seed, same order, same code path as the control (scripts/check_trainer_parity.sh shows the trainers
# bit-identical in deterministic mode), so what differs from the control is numerical noise, plus anything
# that only appears after the first epoch. The order is checked against lerobot-train's real loader chain
# before any GPU time is spent. Rule for reading the result: docs/plan.md, step 1b.
# Usage: scripts/run_order_replay.sh [out_root]   (default experiments/order_replay; output in <out_root>/seed1000)
# Refuses a non-empty output directory. aloha-train has no mid-run checkpoints: an interruption means restarting.
set -euo pipefail
out_root="${1:-experiments/order_replay}"
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"
E=task/aloha-act-demo-curriculum/environment
out="$out_root/seed1000"
if [ -e "$out" ] && [ -n "$(ls -A "$out")" ]; then echo "refusing to overwrite non-empty $out" >&2; exit 2; fi
mkdir -p "$out"
export MUJOCO_GL=egl HF_HUB_OFFLINE=1 PYTHONHASHSEED=0
.venv/bin/python scripts/trainer_parity.py order --protocol "$E/fixed_train.json" \
  --dataset-root data/aloha_sim_insertion_human --steps 100000 --output-order "$out/order.npy"
CUDA_VISIBLE_DEVICES="" .venv-control/bin/python scripts/check_loader_order.py --order "$out/order.npy" --protocol "$E/fixed_train.json"
.venv/bin/aloha-train --schedule "$out/order.npy" --protocol "$E/fixed_train.json" \
  --dataset-root data/aloha_sim_insertion_human --output "$out/train" > "$out/train.json"
hidden_start=$(.venv/bin/python -I -c "import json,sys; print(json.load(open(sys.argv[1]))['start_seed'])" \
  task/aloha-act-demo-curriculum/tests/hidden_seeds.json)
.venv/bin/aloha-eval --checkpoint "$out/train/pretrained_model" --protocol "$E/fixed_train.json" \
  --output "$out/eval_hidden" --start-seed "$hidden_start" --n-episodes 500 --n-envs 10 > "$out/eval_hidden.json"
.venv/bin/aloha-eval --checkpoint "$out/train/pretrained_model" --protocol "$E/fixed_train.json" \
  --output "$out/eval_proxy" > "$out/eval_proxy.json"
echo "order replay done"
