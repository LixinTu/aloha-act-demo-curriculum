#!/usr/bin/env bash
# Final-protocol production chain on the shared GPU (2026-11-12; extended after that morning's crash):
#   1. verify-seed1002: re-run the final-protocol baseline seed 1002 into
#      experiments/baseline_v3_det_verify and compare the SHA-256 of its weights with the committed
#      experiments/baseline_v3_det/seed1002/checkpoint.sha256 (the original run ended while the kernel
#      was already reporting memory corruption);
#   2. only if they differ: verify2-seed1002, a third run into experiments/baseline_v3_det_verify2;
#      the chain goes on only if that run reproduces one of the first two bit for bit, and stops with
#      exit 7 if not (deterministic mode does not reproduce on this machine);
#   3. oracle-run-final: the RSI-Harness oracle run;
#   4. probe-final-<name>: the two author probes, then candidate_stats.py against the final baseline.
# Each stage starts only after the GPU has had no compute process for IDLE_S seconds (default 300):
# the user's Isaac Sim jobs come first, and a stage that finds the GPU busy at its start exits 4, in
# which case it is queued again (up to 5 tries). Stage logs and exit codes: logs/<stage>.{log,exit}.
# A stage whose .exit file says 0 is skipped, so the chain can be started again after an interruption;
# any other value stops the chain until someone has looked at it and moved the file away.
# Reading rules: docs/plan.md ("2026-11-12 crash" and steps 3-4), committed before the chain started.
# Usage: setsid scripts/run_bg.sh final-chain scripts/final_protocol_chain.sh > /dev/null 2>&1 < /dev/null &
set -uo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$root"
idle="${IDLE_S:-300}"
protocol=experiments/trainer_parity/steps100000/protocol_deterministic.json

source scripts/stage_lib.sh  # stage <name> <command...>

weights_sha() {  # <run_dir>: SHA-256 of the trained weights
  local f="$1/train/pretrained_model/model.safetensors" line
  if ! line="$(sha256sum "$f")"; then return 1; fi
  printf '%s\n' "${line%% *}"
}

# --- 1/2: is the seed-1002 baseline run reproduced bit for bit?
if ! want="$(cut -d' ' -f1 experiments/baseline_v3_det/seed1002/checkpoint.sha256)" || [ "${#want}" -ne 64 ]; then
  echo "no committed weights hash in experiments/baseline_v3_det/seed1002/checkpoint.sha256; chain not started"
  exit 1
fi
stage verify-seed1002 env PROTOCOL="$protocol" scripts/run_baseline.sh 1002 experiments/baseline_v3_det_verify \
  || { echo "verification run failed; nothing else started"; exit 1; }
if ! got="$(weights_sha experiments/baseline_v3_det_verify/seed1002)"; then
  echo "cannot hash the verification run's weights; chain stopped"; exit 1
fi
echo "# seed 1002 weights: original $want"
echo "# seed 1002 weights: verification $got"
if [ "$got" = "$want" ]; then
  echo "# verification: bit-identical, the original seed-1002 training is reproduced"
else
  echo "# verification: the weights differ; third run"
  stage verify2-seed1002 env PROTOCOL="$protocol" scripts/run_baseline.sh 1002 experiments/baseline_v3_det_verify2 \
    || { echo "third run failed; nothing else started"; exit 1; }
  if ! got2="$(weights_sha experiments/baseline_v3_det_verify2/seed1002)"; then
    echo "cannot hash the third run's weights; chain stopped"; exit 1
  fi
  echo "# seed 1002 weights: third run $got2"
  if [ "$got2" = "$want" ]; then
    echo "# the third run reproduces the original run; the first verification run is the odd one"
  elif [ "$got2" = "$got" ]; then
    echo "# the third run reproduces the verification run; the original run is the odd one"
  else
    echo "three runs of seed 1002 gave three different weights: deterministic mode does not reproduce on this machine; chain stopped"
    exit 7
  fi
fi

# --- 3: RSI-Harness oracle run under the final protocol
stage oracle-run-final scripts/oracle_run.sh \
  || { echo "oracle stage failed; probes not started (GPU left free for a rerun)"; exit 1; }

# --- 4: author probes under the final protocol
for p in drop_idle insertion_focus; do
  out="experiments/probes_final/$p/run_seed1000"
  stage "probe-final-$p" scripts/run_candidate.sh "$out" "experiments/probes/$p/candidate" 1000 || exit 1
  if ! .venv/bin/python scripts/candidate_stats.py "$out" --baseline-dir experiments/baseline_v3_det \
      > "logs/probe-final-$p.stats.log" 2>&1; then
    echo "candidate_stats failed for $p; see logs/probe-final-$p.stats.log"; exit 1
  fi
done
echo "final-protocol chain done at $(date -Is)"
