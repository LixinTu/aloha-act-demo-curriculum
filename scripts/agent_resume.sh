#!/usr/bin/env bash
# Continue an RSI-Harness agent run as its next attempt (`rsi-harness resume`; RSI-Harness README,
# "Resume": Work is rebuilt from the run's retained image, files are kept, the Agent continues its own
# session and is told what changed). For a run whose attempt an infrastructure failure ended or crippled
# (2026-11-13: a systemd reload removed the Work container's GPU access; docs/plan.md step 5).
#
#   RESUME_DECIDED_BY="<who decided, and why>" scripts/run_bg.sh <name> scripts/agent_resume.sh <run_id>
#
# The run keeps the Agent time and submissions it has left. Extra time (ADD_TIME seconds) goes beyond
# the budget the operator approved, so it also needs ADD_TIME_APPROVED=yes, set only on the operator's
# word. RESUME_DECIDED_BY goes into the log so the record says who started the attempt.
# Automatic updates must be held for UPDATE_HOLD_MIN_S (default 43200) seconds (scripts/update_hold_check.sh:
# a systemd reload or re-exec takes the GPU away from the containers; it cost rounds 1 and 4 of the agent
# run). Only the operator can waive that, by setting UPDATE_HOLD_WAIVED_BY='<who, when, why>'.
# Exit codes of its own checks: 2 usage, 4 GPU busy, 6 automatic updates are not held, 7 this boot's kernel
# log is not clean, 8 the kernel reported faults during the attempt (its Judge scores are suspect), 9 the
# dry run refused to resume. Otherwise the harness's exit code.
set -euo pipefail
if [ "$#" -ne 1 ] || ! [[ "$1" =~ ^[0-9a-f]{32}$ ]]; then
  echo "usage: RESUME_DECIDED_BY='...' $0 <run_id: 32 hex digits>" >&2
  exit 2
fi
run_id="$1"
if [ -z "${RESUME_DECIDED_BY:-}" ]; then
  echo "refusing to resume: set RESUME_DECIDED_BY to who decided the resume and why" >&2
  exit 2
fi
add_time="${ADD_TIME:-0}"
if ! [[ "$add_time" =~ ^[0-9]+$ ]]; then
  echo "ADD_TIME must be whole seconds, got '$add_time'" >&2
  exit 2
fi
if [ "$add_time" -gt 0 ] && [ "${ADD_TIME_APPROVED:-}" != "yes" ]; then
  echo "refusing to add ${add_time}s: extra Agent time needs the operator's approval (ADD_TIME_APPROVED=yes)" >&2
  exit 2
fi
hold_min="${UPDATE_HOLD_MIN_S:-43200}"
if ! [[ "$hold_min" =~ ^[0-9]+$ ]]; then
  echo "UPDATE_HOLD_MIN_S must be whole seconds, got '$hold_min'" >&2
  exit 2
fi
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$root"

busy="$(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader)"
if [ -n "$busy" ]; then
  echo "refusing to resume: the GPU already runs compute processes:" >&2
  echo "$busy" >&2
  exit 4
fi
if ! scripts/kernel_health.sh >&2; then
  echo "refusing to resume: the kernel log of this boot is not clean; reboot and check the hardware first" >&2
  exit 7
fi
if [ -n "${UPDATE_HOLD_WAIVED_BY:-}" ]; then
  echo "automatic-update hold not checked: waived by $UPDATE_HOLD_WAIVED_BY" >&2
elif ! scripts/update_hold_check.sh "$hold_min" >&2; then
  echo "refusing to resume: automatic updates are not held for ${hold_min}s; a systemd reload or re-exec would take the GPU away from the containers" >&2
  exit 6
fi
attempt_started="$(date '+%Y-%m-%d %H:%M:%S')"

echo "resume run=$run_id add_time=${add_time}s decided_by: $RESUME_DECIDED_BY"
roots="--data-root '$root/harness/data' --logs-root '$root/harness/logs'"
add_args=""
if [ "$add_time" -gt 0 ]; then add_args="--add-time $add_time"; fi
if ! sg docker -c "rsi-harness resume '$run_id' $add_args $roots --dry-run"; then
  echo "the dry run refused to resume $run_id; nothing was changed" >&2
  exit 9
fi
harness_rc=0
sg docker -c "rsi-harness resume '$run_id' $add_args $roots --verbose" || harness_rc=$?
echo "rsi-harness exit=$harness_rc"
# The kernel check runs however the attempt ended: faults make any score of this attempt suspect.
if ! scripts/kernel_health.sh --since "$attempt_started"; then
  echo "the kernel check since $attempt_started failed: treat this attempt's Judge scores as suspect" >&2
  exit 8
fi
exit "$harness_rc"
