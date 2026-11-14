#!/usr/bin/env bash
# Wait until no process has computed on the GPU for <idle-seconds> in a row, then run <command...>.
# The user's own GPU work comes first. Their batches start one process per item with short gaps in
# between, so a single empty poll is not enough; and a GPU shared with another job already ran out of
# memory once (docs/plan.md, first hidden evaluation of seed 1000). Polls nvidia-smi every POLL_S seconds
# (default 60); any busy poll restarts the count, and so does a gap between polls longer than 3 x POLL_S
# (a suspend or a stall, during which nobody watched the GPU).
# Usage: scripts/run_bg.sh full-parity scripts/wait_gpu_idle.sh 3600 scripts/run_full_parity.sh
# Exit 2: bad arguments. Exit 3: nvidia-smi failed (a failed query is not an idle GPU).
# Otherwise the command's own exit code (it replaces this script).
set -euo pipefail
if [ "$#" -lt 2 ] || ! [[ "$1" =~ ^[1-9][0-9]*$ ]]; then
  echo "usage: $0 <idle-seconds> <command...>" >&2
  exit 2
fi
need="$1"; shift
poll="${POLL_S:-60}"
if ! [[ "$poll" =~ ^[1-9][0-9]*$ ]]; then
  echo "POLL_S must be a positive integer, got '$poll'" >&2
  exit 2
fi
since=""; last=""; state=""
echo "$(date -Is) waiting for ${need} s without GPU compute processes (poll every ${poll} s), then: $*"
while :; do
  if ! busy="$(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader)"; then
    echo "$(date -Is) nvidia-smi failed; a failed query is not an idle GPU" >&2
    exit 3
  fi
  now="$(date +%s)"
  if [ -n "$last" ] && [ $((now - last)) -gt $((3 * poll)) ] && [ -n "$since" ]; then
    echo "$(date -Is) $((now - last)) s since the previous poll (suspend or stall); the idle count restarts"
    since=""
  fi
  last="$now"
  if [ -n "$busy" ]; then
    [ "$state" = busy ] || echo "$(date -Is) GPU busy: $(printf '%s' "$busy" | tr '\n' ';')"
    state=busy; since=""
  else
    [ "$state" = idle ] || echo "$(date -Is) GPU idle; counting"
    state=idle; since="${since:-$now}"
    if [ $((now - since)) -ge "$need" ]; then break; fi
  fi
  sleep "$poll"
done
echo "$(date -Is) GPU idle for $((now - since)) s; starting: $*"
exec "$@"
