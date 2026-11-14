#!/usr/bin/env bash
# Full-budget run of any curriculum candidate under the Judge protocol (host venv):
# compile -> 100k-step train (seed 1000 unless given) -> hidden 500 + proxy 100 evaluation.
# Usage: scripts/run_candidate.sh <out_dir> <candidate_dir> [train_seed]
# Exit 4: GPU busy. Exit 5 (3): this boot's kernel log shows faults (cannot be read). Exit 6: the run
# finished, but the kernel reported faults while it ran (scripts/run_guards.sh).
set -euo pipefail
out="$1"; cand="$2"; seed="${3:-1000}"
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$root"
if [ -e "$out" ] && [ -n "$(ls -A "$out")" ]; then echo "refusing to overwrite non-empty $out" >&2; exit 2; fi
E=task/aloha-act-demo-curriculum/environment
source scripts/require_free_gpu.sh  # exit 4 while another process computes on the GPU
source scripts/run_guards.sh        # exit 5 if this boot's kernel log already shows faults (3 if unreadable)
mkdir -p "$out"
export MUJOCO_GL=egl HF_HUB_OFFLINE=1 PYTHONHASHSEED=0
.venv/bin/aloha-compile --candidate-dir "$cand" --features experiments/features/features.parquet \
  --protocol $E/fixed_train.json --output "$out/schedule" > "$out/compile.json"
seed_arg=(); if [ "$seed" != "1000" ]; then seed_arg=(--seed "$seed"); fi
.venv/bin/aloha-train --schedule "$out/schedule/schedule.npy" --protocol $E/fixed_train.json \
  --dataset-root data/aloha_sim_insertion_human --output "$out/train" "${seed_arg[@]}" > "$out/train.json"
record_checkpoint_sha256 "$out"  # <out>/checkpoint.sha256: lets a deterministic re-run be compared bit for bit
hidden_start=$(python3 -c "import json; print(json.load(open('task/aloha-act-demo-curriculum/tests/hidden_seeds.json'))['start_seed'])")
.venv/bin/aloha-eval --checkpoint "$out/train/pretrained_model" --protocol $E/fixed_train.json \
  --output "$out/eval_hidden" --start-seed "$hidden_start" --n-episodes 500 --n-envs 10 > "$out/eval_hidden.json"
.venv/bin/aloha-eval --checkpoint "$out/train/pretrained_model" --protocol $E/fixed_train.json \
  --output "$out/eval_proxy" > "$out/eval_proxy.json"
kernel_health_after "$out"  # exit 6 if the kernel reported faults while this ran: the result is suspect
echo "candidate run done: $out"
