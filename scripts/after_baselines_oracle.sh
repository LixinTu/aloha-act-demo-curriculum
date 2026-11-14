#!/usr/bin/env bash
# Wait for the seed 1001/1002 baseline queue to finish successfully, then run the full container
# smoke test (with training) and the RSI-Harness oracle validation on the now-free GPU.
set -uo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$root"
while [ ! -f logs/baseline-rest.exit ]; do sleep 120; done
rc=$(cat logs/baseline-rest.exit)
if [ "$rc" != "0" ]; then echo "baseline queue exit=$rc; not starting oracle"; exit 1; fi
echo "# baselines done at $(date -Is); full smoke test"
SMOKE_SKIP_TRAIN=0 scripts/container_smoke.sh > logs/container-smoke-full.log 2>&1
rc=$?; echo "$rc" > logs/container-smoke-full.exit; echo "# smoke exit=$rc"
if [ "$rc" != "0" ]; then echo "full smoke test failed; not starting oracle"; exit 1; fi
echo "# oracle run starting at $(date -Is)"
scripts/oracle_run.sh > logs/oracle-run.log 2>&1
rc=$?; echo "$rc" > logs/oracle-run.exit; echo "# oracle exit=$rc at $(date -Is)"
exit "$rc"
