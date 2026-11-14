#!/usr/bin/env bash
# Full-length parity: aloha-train and LeRobot v0.6.1's lerobot-train over lerobot-train's own seed-1000 frame
# order for all 100,000 steps in deterministic mode, lerobot-train saving every 20,000 steps like the control;
# then the Judge's hidden 500 episodes and the proxy 100 for the resulting model. Bit-identical weights mean
# one model, scored once (lerobot-train's copy); if they differ, both are scored (descriptive only) and the
# divergence has to be located. Rule for reading the result: docs/plan.md, step 1b ("整程确定性比对").
# Usage: scripts/run_bg.sh full-parity scripts/run_full_parity.sh
#   Output: experiments/trainer_parity/steps100000 (about 9 h of training, then about 30 min of scoring).
# Exit 4: another process computes on the GPU (it would slow both runs; the user's own GPU work comes first).
# Exit 0: identical and scored. Exit 1: weights differ (both scored), or a step failed (no result.json).
# Exit 2/3: as check_trainer_parity.sh (output not empty / .venv-control off its pins), nothing scored.
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"
busy="$(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader)"
if [ -n "$busy" ]; then
  echo "refusing to start: the GPU already runs compute processes:" >&2
  echo "$busy" >&2
  exit 4
fi
E=task/aloha-act-demo-curriculum/environment
out=experiments/trainer_parity/steps100000
set +e
SAVE_FREQ=20000 scripts/check_trainer_parity.sh 100000
parity=$?
set -e
echo "check_trainer_parity.sh exit=$parity"
if [ "$parity" -ne 0 ] && [ "$parity" -ne 1 ]; then exit "$parity"; fi
if [ ! -f "$out/result.json" ]; then
  echo "no $out/result.json: a step failed before the comparison" >&2
  exit 1
fi
export MUJOCO_GL=egl HF_HUB_OFFLINE=1 PYTHONHASHSEED=0
hidden_start=$(.venv/bin/python -I -c "import json,sys; print(json.load(open(sys.argv[1]))['start_seed'])" \
  task/aloha-act-demo-curriculum/tests/hidden_seeds.json)
score() {  # <checkpoint dir> <output dir>
  mkdir -p "$2"
  .venv/bin/aloha-eval --checkpoint "$1" --protocol "$E/fixed_train.json" \
    --output "$2/eval_hidden" --start-seed "$hidden_start" --n-episodes 500 --n-envs 10 > "$2/eval_hidden.json"
  .venv/bin/aloha-eval --checkpoint "$1" --protocol "$E/fixed_train.json" \
    --output "$2/eval_proxy" > "$2/eval_proxy.json"
}
score "$out/lerobot_train/checkpoints/last/pretrained_model" "$out/scores/lerobot_train"
if [ "$parity" -eq 1 ]; then
  score "$out/aloha_train/pretrained_model" "$out/scores/aloha_train"
fi
echo "full parity done (weights $([ "$parity" -eq 0 ] && echo identical || echo differ))"
exit "$parity"
