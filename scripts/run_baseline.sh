#!/usr/bin/env bash
# Full-budget uniform-sampling baseline for one training seed.
# Usage: [PROTOCOL=<file>] scripts/run_baseline.sh <seed> [out_root]   (seed 1000 == the Judge's fixed protocol seed)
# Produces <out_root>/seed<seed>/{protocol.json,schedule,train,checkpoint.sha256,eval_hidden,eval_proxy,
# kernel_health.txt}; out_root defaults to experiments/baseline (protocol v1 results). Refuses to overwrite a
# non-empty output directory. Exit 4: GPU busy. Exit 5 (3): this boot's kernel log shows faults (cannot be
# read). Exit 6: the run finished, but the kernel reported faults while it ran (scripts/run_guards.sh).
# PROTOCOL defaults to the Judge's fixed protocol. Whichever file is given is copied to <out>/protocol.json
# first, and compile, training and evaluation all read that copy, so the run records exactly what it used
# even if the original changes meanwhile.
set -euo pipefail
seed="$1"; out_root="${2:-experiments/baseline}"
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"
E=task/aloha-act-demo-curriculum/environment
protocol="${PROTOCOL:-$E/fixed_train.json}"
if [ ! -f "$protocol" ]; then echo "no protocol file $protocol" >&2; exit 2; fi
out="$out_root/seed$seed"
if [ -e "$out" ] && [ -n "$(ls -A "$out")" ]; then echo "refusing to overwrite non-empty $out" >&2; exit 2; fi
source scripts/require_free_gpu.sh  # exit 4 while another process computes on the GPU
source scripts/run_guards.sh        # exit 5 if this boot's kernel log already shows faults (3 if unreadable)
mkdir -p "$out"
cp "$protocol" "$out/protocol.json"
p="$out/protocol.json"
export MUJOCO_GL=egl HF_HUB_OFFLINE=1 PYTHONHASHSEED=0
.venv/bin/aloha-compile --candidate-dir experiments/uniform_candidate --features experiments/features/features.parquet \
  --protocol "$p" --output "$out/schedule" > "$out/compile.json"
seed_arg=()
if [ "$seed" != "1000" ]; then seed_arg=(--seed "$seed"); fi
.venv/bin/aloha-train --schedule "$out/schedule/schedule.npy" --protocol "$p" \
  --dataset-root data/aloha_sim_insertion_human --output "$out/train" "${seed_arg[@]}" > "$out/train.json"
record_checkpoint_sha256 "$out"  # <out>/checkpoint.sha256: lets a deterministic re-run be compared bit for bit
hidden_start=$(python3 -c "import json; print(json.load(open('task/aloha-act-demo-curriculum/tests/hidden_seeds.json'))['start_seed'])")
.venv/bin/aloha-eval --checkpoint "$out/train/pretrained_model" --protocol "$p" \
  --output "$out/eval_hidden" --start-seed "$hidden_start" --n-episodes 500 --n-envs 10 > "$out/eval_hidden.json"
.venv/bin/aloha-eval --checkpoint "$out/train/pretrained_model" --protocol "$p" \
  --output "$out/eval_proxy" > "$out/eval_proxy.json"
kernel_health_after "$out"  # exit 6 if the kernel reported faults while this ran: the result is suspect
echo "baseline seed $seed done"
