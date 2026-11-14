#!/usr/bin/env bash
# Evaluate the released checkpoint (lerobot/act_aloha_sim_insertion_human, migrated to the current
# processor format in data/reference_policy) under the Judge's evaluation protocol: the 500 hidden
# seeds, then the 100 public proxy seeds, with the same evaluator settings as run_candidate.sh.
#
# This is an author measurement, not a baseline. The checkpoint was trained with the same recipe
# and budget (100,000 steps, batch 8, seed 1000) but with an older LeRobot (3c0a209), the v2
# dataset format and epoch shuffling instead of the task's compiled schedule.
#
# Usage: scripts/eval_reference.sh [out_dir]   (default experiments/reference)
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$root"
out="${1:-experiments/reference}"
E=task/aloha-act-demo-curriculum/environment
ckpt=data/reference_policy
export MUJOCO_GL=egl HF_HUB_OFFLINE=1 PYTHONHASHSEED=0
mkdir -p "$out"
sha256sum "$ckpt/model.safetensors" > "$out/checkpoint.sha256"
hidden_start=$(python3 -c "import json; print(json.load(open('task/aloha-act-demo-curriculum/tests/hidden_seeds.json'))['start_seed'])")
.venv/bin/aloha-eval --checkpoint "$ckpt" --protocol $E/fixed_train.json \
  --output "$out/eval_hidden" --start-seed "$hidden_start" --n-episodes 500 --n-envs 10 > "$out/eval_hidden.json"
.venv/bin/aloha-eval --checkpoint "$ckpt" --protocol $E/fixed_train.json \
  --output "$out/eval_proxy" > "$out/eval_proxy.json"
echo "reference eval done: $out"
