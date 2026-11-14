#!/usr/bin/env bash
# Evaluator determinism check and, only if it passes, the deterministic re-evaluation of the three
# final-baseline checkpoints. Reading: docs/plan.md, "评测确定性的检验和读法", committed before this ran.
#   1. det-proxy-run1, det-proxy-run2: the seed-1002 checkpoint on the public proxy block, twice, with
#      experiments/eval_determinism/protocol_eval_det.json (the fixed protocol plus
#      evaluation.cudnn_deterministic). scripts/compare_evals.py must find every episode identical;
#      otherwise the chain stops with exit 8 and the protocol stays as it is.
#   2. final-seed<seed> for 1000, 1001, 1002: scripts/reeval_baseline_seed.sh reuses the stored weights
#      (checked against the registered hash) and evaluates the hidden and proxy blocks into
#      experiments/baseline_final/seed<seed>.
#   3. det-hidden-run2: the seed-1002 checkpoint on the hidden block once more; it must be identical to
#      step 2's evaluation, otherwise exit 9 (the change is dropped).
# Stage logs and exit codes: logs/<stage>.{log,exit}; comparisons: experiments/eval_determinism/*.json.
# Each stage waits for IDLE_S seconds (default 120) without GPU compute processes.
# Usage: setsid scripts/run_bg.sh eval-det-chain scripts/eval_determinism_chain.sh > /dev/null 2>&1 < /dev/null &
set -uo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$root"
idle="${IDLE_S:-120}"
source scripts/stage_lib.sh  # stage <name> <command...>
D=experiments/eval_determinism
P="$D/protocol_eval_det.json"
src=experiments/baseline_v3_det
if [ ! -f "$P" ]; then echo "no protocol file $P; chain not started"; exit 1; fi

# --- 1: is a deterministic evaluation reproducible? (public proxy block only)
stage det-proxy-run1 env PROTOCOL="$P" scripts/eval_blocks.sh "$src/seed1002" "$D/proxy_run1" proxy || exit 1
stage det-proxy-run2 env PROTOCOL="$P" scripts/eval_blocks.sh "$src/seed1002" "$D/proxy_run2" proxy || exit 1
if ! .venv/bin/python scripts/compare_evals.py "$D/proxy_run1/eval_proxy/eval_result.json" \
    "$D/proxy_run2/eval_proxy/eval_result.json" --output "$D/proxy_run1_vs_run2.json"; then
  echo "two deterministic evaluations of one checkpoint differ on the proxy block: protocol unchanged; chain stopped"
  exit 8
fi

# --- 2: the three final-baseline checkpoints under the deterministic evaluator
for seed in 1000 1001 1002; do
  stage "final-seed$seed" env PROTOCOL="$P" scripts/reeval_baseline_seed.sh "$src/seed$seed" "experiments/baseline_final/seed$seed" || exit 1
done

# --- 3: the hidden block must reproduce as well
stage det-hidden-run2 env PROTOCOL="$P" scripts/eval_blocks.sh "$src/seed1002" "$D/hidden_seed1002_run2" hidden || exit 1
if ! .venv/bin/python scripts/compare_evals.py experiments/baseline_final/seed1002/eval_hidden/eval_result.json \
    "$D/hidden_seed1002_run2/eval_hidden/eval_result.json" --output "$D/hidden_seed1002_run1_vs_run2.json"; then
  echo "two deterministic evaluations of one checkpoint differ on the hidden block: the change is dropped; chain stopped"
  exit 9
fi
echo "evaluator determinism chain done at $(date -Is)"
