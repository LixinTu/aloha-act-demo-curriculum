#!/usr/bin/env bash
# Compare async vs sync vector envs and check eval repeatability on the reference checkpoint.
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$root"
E=task/aloha-act-demo-curriculum/environment
export MUJOCO_GL=egl HF_HUB_OFFLINE=1
.venv/bin/aloha-eval --checkpoint data/reference_policy --protocol $E/fixed_train.json --output experiments/smoke_eval_reference_async --start-seed 100000 --n-episodes 20 --n-envs 20 --async-envs > experiments/smoke_eval_reference_async.json
.venv/bin/aloha-eval --checkpoint data/reference_policy --protocol $E/fixed_train.json --output experiments/smoke_eval_reference_rerun --start-seed 100000 --n-episodes 20 --n-envs 20 > experiments/smoke_eval_reference_rerun.json
echo "variants done"
