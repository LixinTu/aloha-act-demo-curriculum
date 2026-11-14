#!/usr/bin/env bash
# Hidden + proxy evaluation of an existing baseline checkpoint. Usage: scripts/eval_seed.sh <seed>
set -euo pipefail
seed="$1"; root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$root"
E=task/aloha-act-demo-curriculum/environment; out=experiments/baseline/seed$seed
export MUJOCO_GL=egl HF_HUB_OFFLINE=1 PYTHONHASHSEED=0
hidden_start=$(python3 -c "import json; print(json.load(open('task/aloha-act-demo-curriculum/tests/hidden_seeds.json'))['start_seed'])")
.venv/bin/aloha-eval --checkpoint "$out/train/pretrained_model" --protocol $E/fixed_train.json --output "$out/eval_hidden" --start-seed "$hidden_start" --n-episodes 500 --n-envs 10 > "$out/eval_hidden.json"
.venv/bin/aloha-eval --checkpoint "$out/train/pretrained_model" --protocol $E/fixed_train.json --output "$out/eval_proxy" > "$out/eval_proxy.json"
echo "eval seed $seed done"
