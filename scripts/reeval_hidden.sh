#!/usr/bin/env bash
# Evaluate an existing run's checkpoint once more on the hidden block (500 episodes), with the run's
# own protocol.json. Added after the 2026-11-12 crash to tell a disturbed evaluation from an
# evaluation that is not bit-deterministic (docs/plan.md, "2026-11-12 crash").
# Usage: scripts/reeval_hidden.sh <run_dir> <out_dir>
# Writes <out_dir>/{checkpoint.sha256,eval_hidden/,eval_hidden.json,kernel_health.txt}. Refuses to
# overwrite a non-empty output directory. Exit 4: GPU busy. Exit 5 (3): this boot's kernel log shows
# faults (cannot be read). Exit 6: finished, but the kernel reported faults while it ran.
set -euo pipefail
run="$1"; out="$2"
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$root"
weights="$run/train/pretrained_model/model.safetensors"
if [ ! -f "$weights" ] || [ ! -f "$run/protocol.json" ]; then echo "no checkpoint or protocol.json under $run" >&2; exit 2; fi
if [ -e "$out" ] && [ -n "$(ls -A "$out")" ]; then echo "refusing to overwrite non-empty $out" >&2; exit 2; fi
source scripts/require_free_gpu.sh  # exit 4 while another process computes on the GPU
source scripts/run_guards.sh        # exit 5 if this boot's kernel log already shows faults (3 if unreadable)
mkdir -p "$out"
export MUJOCO_GL=egl HF_HUB_OFFLINE=1 PYTHONHASHSEED=0
sha256sum "$weights" > "$out/checkpoint.sha256"
hidden_start=$(python3 -c "import json; print(json.load(open('task/aloha-act-demo-curriculum/tests/hidden_seeds.json'))['start_seed'])")
.venv/bin/aloha-eval --checkpoint "$run/train/pretrained_model" --protocol "$run/protocol.json" \
  --output "$out/eval_hidden" --start-seed "$hidden_start" --n-episodes 500 --n-envs 10 > "$out/eval_hidden.json"
kernel_health_after "$out"  # exit 6 if the kernel reported faults while this ran: the result is suspect
echo "re-evaluation done: $out"
