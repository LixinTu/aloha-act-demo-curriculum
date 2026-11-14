#!/usr/bin/env bash
# Tests scripts/agent_resume.sh with fake nvidia-smi, sg, rsi-harness, kernel_health.sh and
# update_hold_check.sh.
# Prints one line per check and "failures: N"; exit 0 only when every check passes.
set -uo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
tmp="$(mktemp -d)"; trap 'rm -rf "$tmp"' EXIT
repo="$tmp/repo"; bin="$tmp/bin"
mkdir -p "$repo/scripts" "$bin"
cp "$root/scripts/agent_resume.sh" "$repo/scripts/"
calls="$tmp/calls"
run_id="26d53b84a84540999c969b3e556392ae"

cat > "$repo/scripts/kernel_health.sh" <<'EOF'
#!/usr/bin/env bash
echo "kernel_health $*" >> "$CALLS"
if [ "${1:-}" = "--since" ]; then exit "${KH_AFTER:-0}"; fi
exit "${KH_BEFORE:-0}"
EOF
cat > "$repo/scripts/update_hold_check.sh" <<'EOF'
#!/usr/bin/env bash
echo "update_hold_check $*" >> "$CALLS"
exit "${HOLD_RC:-0}"
EOF
cat > "$bin/nvidia-smi" <<'EOF'
#!/usr/bin/env bash
if [ -n "${GPU_BUSY:-}" ]; then echo "123, python"; fi
EOF
cat > "$bin/sg" <<'EOF'
#!/usr/bin/env bash
[ "$1" = docker ] && [ "$2" = -c ] || { echo "fake sg: unexpected args $*" >&2; exit 99; }
exec bash -c "$3"
EOF
cat > "$bin/rsi-harness" <<'EOF'
#!/usr/bin/env bash
echo "rsi-harness $*" >> "$CALLS"
case " $* " in
  *" --dry-run "*) exit "${DRY_RC:-0}" ;;
  *) exit "${REAL_RC:-0}" ;;
esac
EOF
chmod +x "$repo/scripts/"*.sh "$bin/"*

failures=0
check() {  # <description> <expected exit> <expected calls (grep -c patterns separated by |)> -- env... -- args...
  local desc="$1" want="$2"; shift 2
  local envs=() args=()
  while [ "$#" -gt 0 ] && [ "$1" != "--" ]; do envs+=("$1"); shift; done
  shift
  args=("$@")
  : > "$calls"
  env -i PATH="$bin:/usr/bin:/bin" CALLS="$calls" HOME="$tmp" "${envs[@]}" \
    bash "$repo/scripts/agent_resume.sh" "${args[@]}" > "$tmp/out" 2>&1
  local rc=$?
  if [ "$rc" -eq "$want" ]; then echo "ok   $desc (exit $rc)"
  else echo "FAIL $desc: exit $rc, want $want"; sed 's/^/     /' "$tmp/out"; failures=$((failures + 1)); fi
}
expect_calls() {  # <description> <expected number of lines in calls> [pattern ...]: each pattern must occur
  local desc="$1" n="$2"; shift 2
  local got; got="$(wc -l < "$calls")"
  local bad=0
  [ "$got" -eq "$n" ] || bad=1
  for p in "$@"; do grep -qF -- "$p" "$calls" || bad=1; done
  if [ "$bad" -eq 0 ]; then echo "ok   $desc"
  else echo "FAIL $desc: calls were"; sed 's/^/     /' "$calls"; failures=$((failures + 1)); fi
}
D="RESUME_DECIDED_BY=test"

check "no arguments" 2 "$D" --
expect_calls "  nothing ran" 0
check "run id not 32 hex digits" 2 "$D" -- 26d53b84
check "two arguments" 2 "$D" -- "$run_id" extra
check "RESUME_DECIDED_BY missing" 2 -- "$run_id"
expect_calls "  nothing ran" 0
check "ADD_TIME not a number" 2 "$D" ADD_TIME=1h -- "$run_id"
check "ADD_TIME without approval" 2 "$D" ADD_TIME=16233 -- "$run_id"
expect_calls "  nothing ran" 0
check "GPU busy" 4 "$D" GPU_BUSY=1 -- "$run_id"
expect_calls "  no kernel check, no harness" 0
check "kernel log not clean before" 7 "$D" KH_BEFORE=5 -- "$run_id"
expect_calls "  only the kernel check ran" 1 "kernel_health "
check "UPDATE_HOLD_MIN_S not a number" 2 "$D" UPDATE_HOLD_MIN_S=12h -- "$run_id"
expect_calls "  nothing ran" 0
check "automatic updates not held" 6 "$D" HOLD_RC=1 -- "$run_id"
expect_calls "  kernel and hold checks only, no harness" 2 "kernel_health " "update_hold_check 43200"
check "hold state cannot be read" 6 "$D" HOLD_RC=2 -- "$run_id"
expect_calls "  no harness" 2 "update_hold_check 43200"
check "not held, but the operator waived the check" 0 "$D" HOLD_RC=1 UPDATE_HOLD_WAIVED_BY=test-waiver -- "$run_id"
expect_calls "  hold check skipped; dry run, resume, kernel --since ran" 4 "--dry-run" "--verbose" "kernel_health --since"
grep -q "waived by test-waiver" "$tmp/out" && echo "ok   the log names who waived the hold" || { echo "FAIL the waiver is not logged"; failures=$((failures + 1)); }
check "UPDATE_HOLD_MIN_S is passed on" 0 "$D" UPDATE_HOLD_MIN_S=7200 -- "$run_id"
expect_calls "  hold check asked for 7200 s" 5 "update_hold_check 7200"
check "dry run refuses" 9 "$D" DRY_RC=1 -- "$run_id"
expect_calls "  dry run only, no real resume" 3 "update_hold_check 43200" "--dry-run"
check "happy path" 0 "$D" -- "$run_id"
expect_calls "  kernel, hold, dry run, resume, kernel --since" 5 \
  "update_hold_check 43200" \
  "rsi-harness resume $run_id --data-root $repo/harness/data --logs-root $repo/harness/logs --dry-run" \
  "rsi-harness resume $run_id --data-root $repo/harness/data --logs-root $repo/harness/logs --verbose" \
  "kernel_health --since"
if grep -q -- "--add-time" "$calls"; then echo "FAIL happy path passed --add-time"; failures=$((failures + 1)); else echo "ok   happy path grants no time"; fi
grep -q "decided_by: test" "$tmp/out" && echo "ok   the log names who decided" || { echo "FAIL decided_by not logged"; failures=$((failures + 1)); }
check "approved extra time" 0 "$D" ADD_TIME=16233 ADD_TIME_APPROVED=yes -- "$run_id"
expect_calls "  both harness calls carry --add-time 16233" 5 \
  "rsi-harness resume $run_id --add-time 16233 --data-root $repo/harness/data --logs-root $repo/harness/logs --dry-run" \
  "rsi-harness resume $run_id --add-time 16233 --data-root $repo/harness/data --logs-root $repo/harness/logs --verbose"
check "harness fails, kernel clean" 1 "$D" REAL_RC=1 -- "$run_id"
expect_calls "  kernel checked after a failed attempt too" 5 "kernel_health --since"
check "harness ok, kernel faults during the attempt" 8 "$D" KH_AFTER=5 -- "$run_id"
check "harness fails and kernel faults" 8 "$D" REAL_RC=1 KH_AFTER=5 -- "$run_id"

echo "failures: $failures"
[ "$failures" -eq 0 ]
