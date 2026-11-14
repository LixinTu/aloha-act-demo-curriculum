#!/usr/bin/env bash
# Build a baseline seed directory from an existing run's training outputs plus a new evaluation. Used
# when only the evaluator changed: retraining in deterministic mode would reproduce the same weights
# bit for bit, so the stored ones are reused. Copies schedule/, train/, compile.json and train.json from
# <src_seed_dir>, checks the copied weights against <src_seed_dir>/checkpoint.sha256, then evaluates the
# hidden and proxy blocks with PROTOCOL (scripts/eval_blocks.sh, whose exit codes it passes on).
# Usage: PROTOCOL=<file> scripts/reeval_baseline_seed.sh <src_seed_dir> <dst_seed_dir>
# Exit 2: bad arguments, or <dst_seed_dir> already has evaluations or other content. Exit 7: the copied
# weights do not match the registered hash.
set -euo pipefail
if [ "$#" -ne 2 ]; then echo "usage: PROTOCOL=<file> $0 <src_seed_dir> <dst_seed_dir>" >&2; exit 2; fi
src="$1"; dst="$2"
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$root"
if [ ! -f "$src/checkpoint.sha256" ] || [ ! -f "$src/train/pretrained_model/model.safetensors" ]; then
  echo "no registered checkpoint under $src" >&2; exit 2
fi
want="$(cut -d' ' -f1 "$src/checkpoint.sha256")"
if [ "${#want}" -ne 64 ]; then echo "$src/checkpoint.sha256 holds no SHA-256" >&2; exit 2; fi
for f in eval_hidden eval_hidden.json eval_proxy eval_proxy.json; do
  if [ -e "$dst/$f" ]; then echo "refusing: $dst already has $f" >&2; exit 2; fi
done
if [ ! -f "$dst/train/pretrained_model/model.safetensors" ]; then
  # First attempt. A second attempt after "GPU busy" (exit 4) finds the copy in place and keeps it.
  if [ -e "$dst" ] && [ -n "$(ls -A "$dst")" ]; then echo "refusing: $dst exists without a checkpoint" >&2; exit 2; fi
  mkdir -p "$dst"
  cp -r "$src/schedule" "$src/train" "$dst/"
  cp "$src/compile.json" "$src/train.json" "$dst/"
fi
got="$(sha256sum "$dst/train/pretrained_model/model.safetensors" | cut -d' ' -f1)"
if [ "$got" != "$want" ]; then
  echo "the weights in $dst ($got) differ from the registered hash of $src ($want)" >&2; exit 7
fi
exec scripts/eval_blocks.sh "$dst" "$dst" both
