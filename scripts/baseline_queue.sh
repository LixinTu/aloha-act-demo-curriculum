#!/usr/bin/env bash
# Wait for the seed-1000 baseline to finish successfully, then run seeds 1001 and 1002 back to back.
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$root"
while [ ! -f logs/baseline-seed1000.exit ]; do sleep 60; done
if [ "$(cat logs/baseline-seed1000.exit)" != "0" ]; then
  echo "seed 1000 failed (exit $(cat logs/baseline-seed1000.exit)); queue aborted"; exit 1
fi
scripts/run_baseline.sh 1001
scripts/run_baseline.sh 1002
echo "queue done"
