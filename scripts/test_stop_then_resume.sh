#!/usr/bin/env bash
# Tests scripts/stop_then_resume.sh with a fake `rsi-harness run` or `rsi-harness resume` (a Python process
# that waits for SIGINT and then ends like the real harness under run_bg.sh), a fake wait_gpu_idle.sh and a
# fake agent_resume.sh.
# Prints one line per check and "failures: N"; exit 0 only when every check passes. Takes about a minute.
set -uo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
tmp="$(mktemp -d)"
repo="$tmp/repo"; bin="$tmp/bin"; calls="$tmp/calls"
mkdir -p "$repo/scripts" "$repo/logs" "$repo/task/aloha-act-demo-curriculum" "$bin"
cp "$root/scripts/stop_then_resume.sh" "$repo/scripts/"
run_id="26d53b84a84540999c969b3e556392ae"
log="$repo/logs/agent-run.log"; exitf="$repo/logs/agent-run.exit"
pids=()
cleanup() {
  local p
  for p in "${pids[@]}"; do
    if kill -0 "$p" 2>/dev/null; then kill -TERM "$p"; fi
  done
  wait
  rm -rf "$tmp"
}
trap cleanup EXIT

cat > "$repo/scripts/wait_gpu_idle.sh" <<'EOF'
#!/usr/bin/env bash
echo "wait_gpu_idle $1" >> "$CALLS"
shift
exec "$@"
EOF
cat > "$repo/scripts/agent_resume.sh" <<'EOF'
#!/usr/bin/env bash
echo "agent_resume $* decided_by=${RESUME_DECIDED_BY:-} add_time=${ADD_TIME:-}" >> "$CALLS"
exit "${RESUME_RC:-0}"
EOF
cat > "$bin/rsi-harness" <<'EOF'
#!/usr/bin/env python3
"""Fake `rsi-harness run`: waits for SIGINT, then ends the attempt like the real one under run_bg.sh."""
import os, re, signal, sys, time

log, exitf, calls = os.environ["FAKE_LOG"], os.environ["FAKE_EXIT"], os.environ["CALLS"]


def note(line):
    with open(calls, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def in_flight():
    with open(log, encoding="utf-8") as f:
        text = f.read()
    started = set(re.findall(r"^Judge (agent-\d+): snapshotting Work", text, re.M))
    finished = set(re.findall(r"^Judge (agent-\d+) finished:", text, re.M))
    return len(started - finished)


def on_int(signum, frame):
    note(f"SIGINT in_flight={in_flight()} at={time.time():.3f}")
    if os.environ.get("FAKE_IGNORE_INT"):
        return
    rc = int(os.environ.get("FAKE_RC", "1"))
    with open(log, "a", encoding="utf-8") as f:
        f.write("\nRun completed in 1.0s\n  Run ID:           x\n"
                f"  Status:           {os.environ.get('FAKE_STATUS', 'cancelled')}\n")
    if not os.environ.get("FAKE_NO_EXIT_FILE"):
        with open(exitf, "w", encoding="utf-8") as f:
            f.write(f"{rc}\n")
    sys.exit(rc)


signal.signal(signal.SIGINT, on_int)
signal.signal(signal.SIGTERM, lambda signum, frame: sys.exit(99))
open(os.environ["FAKE_READY"], "w").close()
deadline = time.time() + float(os.environ.get("FAKE_LIFETIME_S", "120"))
while time.time() < deadline:
    time.sleep(0.05)
if os.environ.get("FAKE_SELF_EXIT_FILE"):  # the run ended by itself; run_bg.sh wrote its exit code
    with open(exitf, "w", encoding="utf-8") as f:
        f.write("0\n")
sys.exit(0)
EOF
chmod +x "$repo/scripts/"*.sh "$bin/rsi-harness"

failures=0
hpid=""
start_fake() {  # <task dir> | resume <run id> <data root>, then [VAR=value...]: starts the fake harness as
  # `rsi-harness run <task dir>` or `rsi-harness resume <run id> --data-root <data root>`; sets hpid once
  # its handlers are in place
  local args
  if [ "$1" = resume ]; then
    args=(resume "$2" --data-root "$3" --logs-root "$repo/harness/logs" --verbose); shift 3
  else
    args=(run "$1" --agent claude-code); shift
  fi
  rm -f "$tmp/ready"
  env -i PATH="/usr/bin:/bin" CALLS="$calls" FAKE_LOG="$log" FAKE_EXIT="$exitf" FAKE_READY="$tmp/ready" "$@" \
    python3 -I "$bin/rsi-harness" "${args[@]}" &
  hpid=$!
  pids+=("$hpid")
  local i
  for i in $(seq 100); do
    if [ -e "$tmp/ready" ]; then return 0; fi
    sleep 0.05
  done
  echo "FAIL the fake harness did not start"; failures=$((failures + 1))
}
stop_fake() {
  if kill -0 "$hpid" 2>/dev/null; then kill -TERM "$hpid"; fi
  wait "$hpid" 2>/dev/null
}
reset() {  # <run log contents>
  : > "$calls"; rm -f "$exitf"; printf '%s' "$1" > "$log"
}
run() {  # <description> <expected exit> [VAR=value...] -- [args...]
  local desc="$1" want="$2"; shift 2
  local envs=()
  while [ "$#" -gt 0 ] && [ "$1" != "--" ]; do envs+=("$1"); shift; done
  shift
  env -i PATH="/usr/bin:/bin" CALLS="$calls" HOME="$tmp" "${envs[@]}" \
    bash "$repo/scripts/stop_then_resume.sh" "$@" > "$tmp/out" 2>&1
  local rc=$?
  if [ "$rc" -eq "$want" ]; then echo "ok   $desc (exit $rc)"
  else echo "FAIL $desc: exit $rc, want $want"; sed 's/^/     /' "$tmp/out"; failures=$((failures + 1)); fi
}
expect() {  # <description> <command...>: the command must succeed
  local desc="$1"; shift
  if "$@"; then echo "ok   $desc"
  else
    echo "FAIL $desc"
    sed 's/^/     calls: /' "$calls"; sed 's/^/     out: /' "$tmp/out"
    failures=$((failures + 1))
  fi
}
no_calls() { [ ! -s "$calls" ]; }
no_sigint() { ! grep -q '^SIGINT' "$calls"; }
no_resume() { ! grep -q '^agent_resume\|^wait_gpu_idle' "$calls"; }
no_sigint_no_resume() { no_sigint && no_resume; }
one_quiet_sigint() {  # exactly one SIGINT, and no round was in flight when it came
  [ "$(grep -c '^SIGINT' "$calls")" -eq 1 ] && grep -q '^SIGINT in_flight=0 ' "$calls"
}
sigint_after() {  # <marker> <seconds>: the SIGINT came at least <seconds> after the marker line
  local m s
  m="$(sed -n "s/^$1 at=//p" "$calls" | head -n 1)"
  s="$(sed -n 's/^SIGINT in_flight=[0-9]* at=//p' "$calls" | head -n 1)"
  [ -n "$m" ] && [ -n "$s" ] && awk -v m="$m" -v s="$s" -v d="$2" 'BEGIN { exit !(s - m >= d) }'
}
resumed() {  # agent_resume.sh ran behind wait_gpu_idle.sh 120 with the decision and no extra time
  grep -qx 'wait_gpu_idle 120' "$calls" && grep -qxF "agent_resume $run_id decided_by=test add_time=" "$calls"
}

D="RESUME_DECIDED_BY=test"
FAST=(STOP_POLL_S=1 SETTLE_S=2 EXIT_GRACE_S=3)
TASK="$repo/task/aloha-act-demo-curriculum"
R1=$'Judge agent-1: snapshotting Work...\nJudge agent-1: running /tests/test.sh...\nJudge agent-1 finished: verifier_error score=- runtime=16233.16s\n'
R2_OPEN=$'Judge agent-2: snapshotting Work...\nJudge agent-2: running /tests/test.sh...\n'
R2_DONE=$'Judge agent-2 finished: completed score=0.17 runtime=17000.00s\n'

reset "$R1"
run "no arguments" 2 "$D" --
run "run id not 32 hex digits" 2 "$D" -- 26d53b84 123
run "pid not a number" 2 "$D" -- "$run_id" abc
run "RESUME_DECIDED_BY missing" 2 -- "$run_id" 123
run "SETTLE_S not a number" 2 "$D" SETTLE_S=1m -- "$run_id" 123
run "STOP_POLL_S zero" 2 "$D" STOP_POLL_S=0 -- "$run_id" 123
rm -f "$log"
run "no logs/agent-run.log" 2 "$D" -- "$run_id" 123
expect "  nothing signalled or resumed" no_calls

reset "$R1$R2_OPEN"
start_fake "$tmp/other/task/aloha-act-demo-curriculum"
run "pid runs rsi-harness for another repo" 3 "$D" "${FAST[@]}" -- "$run_id" "$hpid"
expect "  that process got no SIGINT, nothing resumed" no_sigint_no_resume
stop_fake

reset "$R1$R2_OPEN"
start_fake "$TASK"
( sleep 3; printf '%s' "$R2_DONE" >> "$log"; echo "round2_finished at=$(date +%s.%N)" >> "$calls" ) &
appender=$!
run "round in flight, then finished: stop, then resume" 0 "$D" "${FAST[@]}" -- "$run_id" "$hpid"
wait "$appender"
expect "  exactly one SIGINT, with no round in flight" one_quiet_sigint
expect "  the SIGINT came at least 1 s after the round finished" sigint_after round2_finished 1
expect "  agent_resume.sh behind wait_gpu_idle.sh 120, decision passed, no extra time" resumed
stop_fake

reset "$R1$R2_OPEN$R2_DONE"
start_fake "$TASK"
( sleep 1; printf 'Judge agent-3: snapshotting Work...\n' >> "$log"
  sleep 4; printf 'Judge agent-3 finished: completed score=0.15 runtime=4.00s\n' >> "$log"
  echo "round3_finished at=$(date +%s.%N)" >> "$calls" ) &
appender=$!
run "a round starts during the settle time: wait for it as well" 0 "$D" STOP_POLL_S=1 SETTLE_S=4 EXIT_GRACE_S=3 -- "$run_id" "$hpid"
wait "$appender"
expect "  exactly one SIGINT, with no round in flight" one_quiet_sigint
expect "  the SIGINT came after round 3 finished" sigint_after round3_finished 1
expect "  resumed" resumed
stop_fake

reset "$R1$R2_OPEN"
start_fake "$TASK" FAKE_LIFETIME_S=2 FAKE_SELF_EXIT_FILE=1
run "the run ends by itself while a round is in flight" 12 "$D" "${FAST[@]}" -- "$run_id" "$hpid"
expect "  no SIGINT, no resume" no_sigint_no_resume
stop_fake

reset "$R1"; echo 0 > "$exitf"
start_fake "$TASK"
run "the run had already ended" 12 "$D" "${FAST[@]}" -- "$run_id" "$hpid"
expect "  no SIGINT, no resume" no_sigint_no_resume
stop_fake

reset "$R1$R2_OPEN"
start_fake "$TASK"
run "a round is still in flight after MAX_WAIT_S" 11 "$D" "${FAST[@]}" MAX_WAIT_S=2 -- "$run_id" "$hpid"
expect "  no SIGINT, no resume" no_sigint_no_resume
stop_fake

reset "$R1$R2_OPEN$R2_DONE"
start_fake "$TASK" FAKE_STATUS=failed
run "the attempt ends as failed, not cancelled" 10 "$D" "${FAST[@]}" -- "$run_id" "$hpid"
expect "  one SIGINT, no resume" one_quiet_sigint
expect "  no resume" no_resume
stop_fake

reset "$R1$R2_OPEN$R2_DONE"
start_fake "$TASK" FAKE_RC=0
run "the harness exits 0 after SIGINT" 10 "$D" "${FAST[@]}" -- "$run_id" "$hpid"
expect "  no resume" no_resume
stop_fake

reset "$R1$R2_OPEN$R2_DONE"$'\nRun completed in 1.0s\n  Status:           cancelled\n'
start_fake "$TASK" FAKE_STATUS=failed
run "a 'Status: cancelled' line from before the stop does not count" 10 "$D" "${FAST[@]}" -- "$run_id" "$hpid"
expect "  no resume" no_resume
stop_fake

reset "$R1$R2_OPEN$R2_DONE"
start_fake "$TASK" FAKE_NO_EXIT_FILE=1
run "the harness ends but no exit file appears" 11 "$D" "${FAST[@]}" -- "$run_id" "$hpid"
expect "  one SIGINT, no resume" one_quiet_sigint
expect "  no resume" no_resume
stop_fake

reset "$R1$R2_OPEN$R2_DONE"
start_fake "$TASK" FAKE_IGNORE_INT=1
run "the harness still runs MAX_STOP_WAIT_S after SIGINT" 11 "$D" "${FAST[@]}" MAX_STOP_WAIT_S=2 -- "$run_id" "$hpid"
expect "  exactly one SIGINT, no second signal, no resume" one_quiet_sigint
expect "  no resume" no_resume
stop_fake

reset "$R1$R2_OPEN$R2_DONE"
start_fake "$TASK"
run "agent_resume.sh's exit code becomes the script's" 9 "$D" "${FAST[@]}" RESUME_RC=9 -- "$run_id" "$hpid"
expect "  resumed" resumed
stop_fake

# A resumed attempt: STOP_LOG_NAME picks its console log and exit file, and its harness runs
# `rsi-harness resume <run_id>`. REQUIRE_GPU_FAULT_ROUND makes the stop conditional.
log="$repo/logs/agent-resume.log"; exitf="$repo/logs/agent-resume.exit"
R="STOP_LOG_NAME=agent-resume"; G="REQUIRE_GPU_FAULT_ROUND=agent-4"
DATA="$repo/harness/data"
EVAL_DIR="$repo/harness/logs/runs/$run_id/aloha-act-demo-curriculum/verifier/agent-4"
mkdir -p "$EVAL_DIR"
R3=$'Judge agent-3: snapshotting Work...\nJudge agent-3: running /tests/test.sh...\nJudge agent-3 finished: completed score=0.146 runtime=17887.67s\n'
R4_OPEN=$'Judge agent-4: snapshotting Work...\nJudge agent-4: running /tests/test.sh...\n'
R4_FAULT=$'Judge agent-4 finished: verifier_error score=- runtime=16300.00s\n'
R4_DONE=$'Judge agent-4 finished: completed score=0.18 runtime=18000.00s\n'
NO_GPU=$'Traceback (most recent call last):\nRuntimeError: No CUDA GPUs are available\n'

rm -f "$log" "$exitf"; : > "$calls"
run "STOP_LOG_NAME is not a job name" 2 "$D" STOP_LOG_NAME=../x -- "$run_id" 123
run "REQUIRE_GPU_FAULT_ROUND is not a round name" 2 "$D" "$R" REQUIRE_GPU_FAULT_ROUND=4 -- "$run_id" 123
run "no logs/agent-resume.log" 2 "$D" "$R" -- "$run_id" 123
expect "  nothing signalled or resumed" no_calls

reset "$R3$R4_OPEN"
start_fake resume "$run_id" "$tmp/other/harness/data"
run "pid resumes this run id for another repo" 3 "$D" "$R" "${FAST[@]}" -- "$run_id" "$hpid"
expect "  that process got no SIGINT, nothing resumed" no_sigint_no_resume
stop_fake

reset "$R3$R4_OPEN"
start_fake resume 0123456789abcdef0123456789abcdef "$DATA"
run "pid resumes another run" 3 "$D" "$R" "${FAST[@]}" -- "$run_id" "$hpid"
expect "  that process got no SIGINT, nothing resumed" no_sigint_no_resume
stop_fake

reset "$R3$R4_OPEN"
start_fake resume "$run_id" "$DATA"
( sleep 3; printf '%s' "$R4_DONE" >> "$log"; echo "round4_finished at=$(date +%s.%N)" >> "$calls" ) &
appender=$!
run "resumed attempt, round in flight, then finished: stop, then resume" 0 "$D" "$R" "${FAST[@]}" -- "$run_id" "$hpid"
wait "$appender"
expect "  exactly one SIGINT, with no round in flight" one_quiet_sigint
expect "  the SIGINT came at least 1 s after the round finished" sigint_after round4_finished 1
expect "  resumed" resumed
stop_fake

reset "$R3$R4_OPEN$R4_DONE"; printf '%s' "$NO_GPU" > "$EVAL_DIR/eval.log"
start_fake resume "$run_id" "$DATA"
run "GPU-fault condition: the round completed" 13 "$D" "$R" "$G" "${FAST[@]}" -- "$run_id" "$hpid"
expect "  no SIGINT, no resume" no_sigint_no_resume
stop_fake

reset "$R3$R4_OPEN$R4_FAULT"; printf 'Traceback (most recent call last):\nValueError: bad policy\n' > "$EVAL_DIR/eval.log"
start_fake resume "$run_id" "$DATA"
run "GPU-fault condition: the round failed for another reason" 13 "$D" "$R" "$G" "${FAST[@]}" -- "$run_id" "$hpid"
expect "  no SIGINT, no resume" no_sigint_no_resume
stop_fake

reset "$R3$R4_OPEN$R4_FAULT"; rm -f "$EVAL_DIR/eval.log"
start_fake resume "$run_id" "$DATA"
run "GPU-fault condition: the round has no eval.log" 13 "$D" "$R" "$G" "${FAST[@]}" -- "$run_id" "$hpid"
expect "  no SIGINT, no resume" no_sigint_no_resume
stop_fake

reset "$R3"; printf '%s' "$NO_GPU" > "$EVAL_DIR/eval.log"
start_fake resume "$run_id" "$DATA"
run "GPU-fault condition: the round never ran in this attempt" 13 "$D" "$R" "$G" "${FAST[@]}" -- "$run_id" "$hpid"
expect "  no SIGINT, no resume" no_sigint_no_resume
stop_fake

reset "$R3$R4_OPEN"; printf '%s' "$NO_GPU" > "$EVAL_DIR/eval.log"
start_fake resume "$run_id" "$DATA"
( sleep 3; printf '%s' "$R4_FAULT" >> "$log"; echo "round4_finished at=$(date +%s.%N)" >> "$calls" ) &
appender=$!
run "GPU-fault condition met: stop, then resume" 0 "$D" "$R" "$G" "${FAST[@]}" -- "$run_id" "$hpid"
wait "$appender"
expect "  exactly one SIGINT, with no round in flight" one_quiet_sigint
expect "  the SIGINT came at least 1 s after the round finished" sigint_after round4_finished 1
expect "  resumed" resumed
stop_fake

reset "$R3$R4_OPEN$R4_FAULT"; printf '%s' "$NO_GPU" > "$EVAL_DIR/eval.log"
start_fake resume "$run_id" "$DATA" FAKE_LIFETIME_S=5 FAKE_SELF_EXIT_FILE=1
( sleep 0.5; printf 'Judge agent-5: snapshotting Work...\n' >> "$log" ) &
appender=$!
run "the Agent submits again within the settle time, then the run ends by itself" 12 "$D" "$R" "$G" STOP_POLL_S=1 SETTLE_S=3 EXIT_GRACE_S=3 -- "$run_id" "$hpid"
wait "$appender"
expect "  no SIGINT, no resume" no_sigint_no_resume
stop_fake

echo "failures: $failures"
[ "$failures" -eq 0 ]
