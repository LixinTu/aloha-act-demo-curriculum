#!/usr/bin/env bash
# Evaluate an existing run's checkpoint on the hidden and/or the proxy seed block with a given protocol
# file (for example one whose evaluation block sets cudnn_deterministic).
# Usage: PROTOCOL=<file> scripts/eval_blocks.sh <run_dir> <out_dir> <hidden|proxy|both>
# The protocol is copied to <out_dir>/protocol.json and the evaluation reads that copy. <out_dir> may be
# <run_dir> itself. Writes <out_dir>/{protocol.json,checkpoint.sha256,eval_hidden*,eval_proxy*,
# kernel_health.txt}. Exit 2: bad arguments, or a requested block already has output, or <out_dir>
# holds another protocol.json. Exit 4: GPU busy. Exit 5 (3): this boot's kernel log shows faults (cannot
# be read). Exit 6: finished, but the kernel reported faults while it ran.
set -euo pipefail
if [ "$#" -ne 3 ]; then echo "usage: PROTOCOL=<file> $0 <run_dir> <out_dir> <hidden|proxy|both>" >&2; exit 2; fi
run="$1"; out="$2"; which="$3"
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$root"
protocol="${PROTOCOL:-}"
if [ -z "$protocol" ] || [ ! -f "$protocol" ]; then echo "set PROTOCOL to the protocol file to evaluate with" >&2; exit 2; fi
case "$which" in
  hidden) blocks=(hidden) ;;
  proxy) blocks=(proxy) ;;
  both) blocks=(hidden proxy) ;;
  *) echo "third argument must be hidden, proxy or both, got '$which'" >&2; exit 2 ;;
esac
weights="$run/train/pretrained_model/model.safetensors"
if [ ! -f "$weights" ]; then echo "no checkpoint at $weights" >&2; exit 2; fi
for b in "${blocks[@]}"; do
  if [ -e "$out/eval_$b" ] || [ -e "$out/eval_$b.json" ]; then echo "refusing: $out already has an eval_$b" >&2; exit 2; fi
done
if [ -e "$out/protocol.json" ] && ! cmp -s "$protocol" "$out/protocol.json"; then
  echo "refusing: $out/protocol.json differs from $protocol" >&2; exit 2
fi
source scripts/require_free_gpu.sh  # exit 4 while another process computes on the GPU
source scripts/run_guards.sh        # exit 5 if this boot's kernel log already shows faults (3 if unreadable)
mkdir -p "$out"
if [ ! -e "$out/protocol.json" ]; then cp "$protocol" "$out/protocol.json"; fi
p="$out/protocol.json"
sha256sum "$weights" > "$out/checkpoint.sha256"
export MUJOCO_GL=egl HF_HUB_OFFLINE=1 PYTHONHASHSEED=0
for b in "${blocks[@]}"; do
  if [ "$b" = hidden ]; then
    hidden_start=$(python3 -c "import json; print(json.load(open('task/aloha-act-demo-curriculum/tests/hidden_seeds.json'))['start_seed'])")
    .venv/bin/aloha-eval --checkpoint "$run/train/pretrained_model" --protocol "$p" \
      --output "$out/eval_hidden" --start-seed "$hidden_start" --n-episodes 500 --n-envs 10 > "$out/eval_hidden.json"
  else
    .venv/bin/aloha-eval --checkpoint "$run/train/pretrained_model" --protocol "$p" \
      --output "$out/eval_proxy" > "$out/eval_proxy.json"
  fi
done
kernel_health_after "$out"  # exit 6 if the kernel reported faults while this ran: the result is suspect
echo "evaluation done: $out ($which)"
