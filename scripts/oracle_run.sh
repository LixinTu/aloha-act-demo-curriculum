#!/usr/bin/env bash
# Full official-path validation: RSI-Harness builds the Base/Work images from the task,
# runs solution/solve.sh as the oracle agent, submits once, and the hidden Judge scores it.
# One Judge round is a full 100k-step training plus a 500-episode evaluation, so it refuses to start
# (exit 4) while another process computes on the GPU, and (exit 5, or 3 if the log cannot be read)
# when this boot's kernel log already shows faults. Exit 6: the run finished, but the kernel reported
# faults while it ran, so its score is suspect (scripts/run_guards.sh; 2026-11-12 crash).
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$root"
task="$root/task/aloha-act-demo-curriculum"
source scripts/require_free_gpu.sh
source scripts/run_guards.sh
mkdir -p "$root/harness/data" "$root/harness/logs"
harness_rc=0
sg docker -c "rsi-harness run '$task' --agent oracle --gpus 0 --max-submissions 1 --timeout 30600 \
  --data-root '$root/harness/data' --logs-root '$root/harness/logs' --verbose" || harness_rc=$?
if [ "$harness_rc" -ne 0 ]; then
  echo "rsi-harness exit=$harness_rc" >&2
  exit "$harness_rc"
fi
# The harness names the run directory itself, so the kernel report goes to this script's output.
scripts/kernel_health.sh --since "$run_started" || {
  echo "the oracle run finished, but the kernel check since $run_started failed: treat its score as suspect" >&2
  exit 6
}
