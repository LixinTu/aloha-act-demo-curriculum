#!/usr/bin/env bash
# Run baseline seeds 1001 and 1002 back to back (each: compile, 100k-step train, hidden + proxy eval).
set -uo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$root"
for seed in 1001 1002; do
  scripts/run_baseline.sh "$seed" > "logs/baseline-seed$seed.log" 2>&1
  rc=$?; echo "$rc" > "logs/baseline-seed$seed.exit"
  echo "seed $seed exit=$rc at $(date -Is)"
  if [ "$rc" != "0" ]; then echo "stopping queue after failure"; exit "$rc"; fi
done
echo "all seeds done"
