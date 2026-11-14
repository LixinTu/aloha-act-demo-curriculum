#!/usr/bin/env bash
# After a successful oracle run, run the author probes back to back on the free GPU.
set -uo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$root"
while [ ! -f logs/oracle-run.exit ]; do sleep 120; done
rc=$(cat logs/oracle-run.exit)
if [ "$rc" != "0" ]; then echo "oracle exit=$rc; probes not started (GPU left free for an oracle rerun)"; exit 1; fi
for p in drop_idle insertion_focus; do
  echo "# probe $p starting at $(date -Is)"
  scripts/run_candidate.sh "experiments/probes/$p/run_seed1000" "experiments/probes/$p/candidate" 1000 > "logs/probe-$p.log" 2>&1
  rc=$?; echo "$rc" > "logs/probe-$p.exit"; echo "# probe $p exit=$rc at $(date -Is)"
  if [ "$rc" != "0" ]; then exit "$rc"; fi
done
echo "all probes done"
