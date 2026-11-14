#!/usr/bin/env bash
# Score earlier checkpoints of the lerobot-train control on the hidden block, to see how much one run's
# score moves between checkpoints. Descriptive only: no decision rule depends on these numbers.
# Usage: scripts/eval_control_checkpoints.sh <step> [<step>...]   e.g. 80000 60000
# Output: <run>/checkpoint_evals/step<step>/, run defaulting to experiments/control_lerobot_train/seed1000;
#   RUN=<dir> scores another lerobot-train run laid out the same way (e.g. the unchanged re-run).
#   Refuses to overwrite.
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"
E=task/aloha-act-demo-curriculum/environment
run="${RUN:-experiments/control_lerobot_train/seed1000}"
export MUJOCO_GL=egl HF_HUB_OFFLINE=1 PYTHONHASHSEED=0
hidden_start=$(.venv/bin/python -I -c "import json,sys; print(json.load(open(sys.argv[1]))['start_seed'])" \
  task/aloha-act-demo-curriculum/tests/hidden_seeds.json)
for step in "$@"; do
  ckpt="$run/train/checkpoints/$(printf '%06d' "$step")/pretrained_model"
  out="$run/checkpoint_evals/step$(printf '%06d' "$step")"
  [ -d "$ckpt" ] || { echo "no checkpoint at $ckpt" >&2; exit 2; }
  if [ -e "$out" ] && [ -n "$(ls -A "$out")" ]; then echo "refusing to overwrite non-empty $out" >&2; exit 2; fi
  mkdir -p "$out"
  .venv/bin/aloha-eval --checkpoint "$ckpt" --protocol "$E/fixed_train.json" \
    --output "$out" --start-seed "$hidden_start" --n-episodes 500 --n-envs 10 > "$out/stdout.json"
done
echo "control checkpoint evaluations done"
