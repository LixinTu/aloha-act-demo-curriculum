#!/usr/bin/env bash
# Stop an RSI-Harness agent run between two Judge rounds and resume it at once: option "续跑" of
# docs/plan.md step 5, for an attempt that a host fault crippled (2026-11-13: a systemd reload removed
# the Work container's GPU access; the next attempt rebuilds Work, which restores it).
#
#   RESUME_DECIDED_BY='<who decided, when, why>' scripts/run_bg.sh agent-resume \
#     scripts/stop_then_resume.sh <run_id> <harness_pid>
#
# STOP_LOG_NAME (default agent-run) names the scripts/run_bg.sh job whose harness is stopped: its console
# is logs/<name>.log and its exit code logs/<name>.exit. The first attempt runs under agent-run
# (`rsi-harness run`); a resumed one under the name that started it, e.g. agent-resume
# (`rsi-harness resume <run_id>`). Give the new job another name, e.g. agent-resume2.
# REQUIRE_GPU_FAULT_ROUND (e.g. agent-4, optional) makes the stop conditional: when no round is in
# flight, that round must have ended without "completed" and its verifier eval.log must show
# "No CUDA GPUs are available" (the error a systemd reload left behind on 2026-11-13); otherwise nothing
# is stopped or resumed (exit 13).
#
# 1. Waits until no Judge round is in flight, that is every "Judge agent-N: snapshotting Work..." line
#    of logs/<STOP_LOG_NAME>.log has its "Judge agent-N finished:" line, and until that has held for
#    SETTLE_S seconds (default 30), so the Agent has received the last round's feedback.
# 2. Sends one SIGINT to the harness, which then ends the attempt itself: acceptance closes, a round
#    accepted in the meantime is judged to completion and recorded, the Agent stops and Work is kept as
#    the run's retained image (RSI-Harness runtime/submissions.py close(), runtime/coordinator.py
#    cleanup_resources). The harness ignores later signals during that cleanup, so none are sent.
# 3. Once the attempt has ended as cancelled (exit 1 and "Status: cancelled"), hands over to
#    scripts/agent_resume.sh behind scripts/wait_gpu_idle.sh 120.
#
# Exit codes of its own: 2 usage; 3 <harness_pid> is not this repo's running `rsi-harness run` or
# `rsi-harness resume <run_id>`; 10 the attempt did not end as cancelled (nothing was resumed); 11 a wait
# timed out (MAX_WAIT_S before the stop, MAX_STOP_WAIT_S after it, or no exit file EXIT_GRACE_S after the
# harness ended); 12 the run ended by itself before the stop (nothing was stopped or resumed);
# 13 REQUIRE_GPU_FAULT_ROUND was not met (nothing was stopped or resumed).
# Otherwise the exit code of wait_gpu_idle.sh and agent_resume.sh.
set -euo pipefail
usage() {
  echo "usage: RESUME_DECIDED_BY='...' $0 <run_id: 32 hex digits> <harness_pid>" >&2
  exit 2
}
[ "$#" -eq 2 ] || usage
[[ "$1" =~ ^[0-9a-f]{32}$ ]] || usage
[[ "$2" =~ ^[1-9][0-9]*$ ]] || usage
run_id="$1"; pid="$2"
if [ -z "${RESUME_DECIDED_BY:-}" ]; then
  echo "refusing: set RESUME_DECIDED_BY to who decided the resume and why" >&2
  exit 2
fi
poll="${STOP_POLL_S:-5}"; settle="${SETTLE_S:-30}"
max_wait="${MAX_WAIT_S:-36000}"; max_stop_wait="${MAX_STOP_WAIT_S:-32400}"; grace="${EXIT_GRACE_S:-120}"
if ! [[ "$poll" =~ ^[1-9][0-9]*$ ]]; then
  echo "STOP_POLL_S must be a positive whole number of seconds, got '$poll'" >&2
  exit 2
fi
for value in "$settle" "$max_wait" "$max_stop_wait" "$grace"; do
  if ! [[ "$value" =~ ^[0-9]+$ ]]; then
    echo "SETTLE_S, MAX_WAIT_S, MAX_STOP_WAIT_S and EXIT_GRACE_S must be whole seconds, got '$value'" >&2
    exit 2
  fi
done
log_name="${STOP_LOG_NAME:-agent-run}"
if ! [[ "$log_name" =~ ^[a-z0-9][a-z0-9-]*$ ]]; then
  echo "STOP_LOG_NAME must be a scripts/run_bg.sh job name like agent-resume, got '$log_name'" >&2
  exit 2
fi
gpu_fault_round="${REQUIRE_GPU_FAULT_ROUND:-}"
if [ -n "$gpu_fault_round" ] && ! [[ "$gpu_fault_round" =~ ^agent-[1-9][0-9]*$ ]]; then
  echo "REQUIRE_GPU_FAULT_ROUND must name a round like agent-4, got '$gpu_fault_round'" >&2
  exit 2
fi
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$root"
log="logs/$log_name.log"; exitf="logs/$log_name.exit"
if [ ! -f "$log" ]; then
  echo "no $log: this script stops the run that 'scripts/run_bg.sh $log_name' started" >&2
  exit 2
fi

is_harness() {  # the pid is alive and runs `rsi-harness run` on this repo's task or resumes this run
  local cmd
  [ -r "/proc/$pid/cmdline" ] || return 1
  cmd="$(tr '\0' ' ' < "/proc/$pid/cmdline")" || return 1
  [[ "$cmd" == *"/rsi-harness run $root/task/"* ]] && return 0
  [[ "$cmd" == *"/rsi-harness resume $run_id "* && "$cmd" == *" --data-root $root/harness/data "* ]]
}
gpu_fault_shown() {  # REQUIRE_GPU_FAULT_ROUND ended without a score because its evaluation found no GPU
  local status evals
  status="$(awk -v r="$gpu_fault_round" '$1 == "Judge" && $2 == r && $3 == "finished:" { s = $4 } END { print s }' "$log")"
  if [ -z "$status" ]; then
    echo "$(date -Is) $log has no 'Judge $gpu_fault_round finished:' line" >&2
    return 1
  fi
  if [ "$status" = completed ]; then
    echo "$(date -Is) round $gpu_fault_round ended as completed: there is no GPU fault to recover from" >&2
    return 1
  fi
  shopt -s nullglob
  evals=(harness/logs/runs/"$run_id"/*/verifier/"$gpu_fault_round"/eval.log)
  shopt -u nullglob
  if [ "${#evals[@]}" -ne 1 ]; then
    echo "$(date -Is) round $gpu_fault_round ended as $status; expected one verifier eval.log, found ${#evals[@]}" >&2
    return 1
  fi
  if ! grep -q 'No CUDA GPUs are available' "${evals[0]}"; then
    echo "$(date -Is) round $gpu_fault_round ended as $status, but ${evals[0]} does not show 'No CUDA GPUs are available'" >&2
    return 1
  fi
  echo "$(date -Is) round $gpu_fault_round ended as $status and its evaluation found no GPU (${evals[0]})"
}
in_flight() {  # Judge rounds started but not finished, from the harness's own console lines
  awk '/^Judge agent-[0-9]+: snapshotting Work/ { sub(/:$/, "", $2); started[$2] = 1 }
       /^Judge agent-[0-9]+ finished:/ { finished[$2] = 1 }
       END { n = 0; for (r in started) if (!(r in finished)) n++; print n }' "$log"
}
run_ended() {  # scripts/run_bg.sh has written the run's exit code
  [ -s "$exitf" ]
}

echo "$(date -Is) stop-then-resume run=$run_id harness_pid=$pid log=$log settle=${settle}s" \
  "require_gpu_fault_round=${gpu_fault_round:-none} decided_by: $RESUME_DECIDED_BY"
if run_ended; then
  echo "$(date -Is) the run already ended (exit $(cat "$exitf")); nothing was stopped or resumed" >&2
  exit 12
fi
if ! is_harness; then
  echo "pid $pid is not this repo's running 'rsi-harness run' or 'rsi-harness resume $run_id'; nothing was stopped" >&2
  exit 3
fi

started="$(date +%s)"; quiet_since=""; gone_since=""; last=""
while :; do
  now="$(date +%s)"
  if run_ended; then
    echo "$(date -Is) the run ended by itself (exit $(cat "$exitf")); nothing was stopped or resumed" >&2
    exit 12
  fi
  if ! is_harness; then
    # run_bg.sh writes the exit file right after the harness ends.
    gone_since="${gone_since:-$now}"
    if [ $((now - gone_since)) -ge "$grace" ]; then
      echo "$(date -Is) pid $pid is gone and $exitf did not appear within ${grace}s; nothing was stopped or resumed" >&2
      exit 11
    fi
    sleep "$poll"
    continue
  fi
  n="$(in_flight)"
  if [ "$n" != "$last" ]; then
    echo "$(date -Is) Judge rounds in flight: $n"
    last="$n"
  fi
  if [ "$n" -eq 0 ]; then
    quiet_since="${quiet_since:-$now}"
    if [ $((now - quiet_since)) -ge "$settle" ]; then break; fi
  else
    quiet_since=""
  fi
  if [ $((now - started)) -ge "$max_wait" ]; then
    echo "$(date -Is) a Judge round was still in flight after ${max_wait}s; nothing was stopped or resumed" >&2
    exit 11
  fi
  sleep "$poll"
done

if [ -n "$gpu_fault_round" ] && ! gpu_fault_shown; then
  echo "$(date -Is) REQUIRE_GPU_FAULT_ROUND=$gpu_fault_round is not met; nothing was stopped or resumed" >&2
  exit 13
fi
offset="$(stat -c %s "$log")"
if ! kill -INT "$pid"; then
  echo "$(date -Is) SIGINT to pid $pid failed; nothing was stopped or resumed" >&2
  exit 3
fi
echo "$(date -Is) sent SIGINT to the harness (pid $pid): it closes acceptance, finishes an accepted round, stops the Agent and keeps Work"
stop_sent="$(date +%s)"; gone_since=""
until run_ended; do
  now="$(date +%s)"
  if ! is_harness; then
    gone_since="${gone_since:-$now}"
    if [ $((now - gone_since)) -ge "$grace" ]; then
      echo "$(date -Is) the harness ended but $exitf did not appear within ${grace}s; nothing was resumed" >&2
      exit 11
    fi
  elif [ $((now - stop_sent)) -ge "$max_stop_wait" ]; then
    echo "$(date -Is) the harness still runs ${max_stop_wait}s after SIGINT; no other signal was sent and nothing was resumed" >&2
    exit 11
  fi
  sleep "$poll"
done
rc="$(cat "$exitf")"
after_stop="$(tail -c +"$((offset + 1))" "$log")"
echo "$(date -Is) the attempt ended: exit=$rc"
if [ "$rc" != 1 ] || ! grep -qE '^[[:space:]]*Status:[[:space:]]+cancelled[[:space:]]*$' <<< "$after_stop"; then
  echo "$(date -Is) the attempt did not end as cancelled (exit $rc; harness output above); nothing was resumed" >&2
  exit 10
fi
echo "$(date -Is) the attempt ended as cancelled; resuming run $run_id once the GPU has been idle for 120 s"
export RESUME_DECIDED_BY
exec scripts/wait_gpu_idle.sh 120 scripts/agent_resume.sh "$run_id"
