#!/usr/bin/env bash
# Tests scripts/update_hold_check.sh with a fake snap and a fake systemctl.
# Prints one line per check and "failures: N"; exit 0 only when every check passes.
set -uo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
tmp="$(mktemp -d)"; trap 'rm -rf "$tmp"' EXIT
bin="$tmp/bin"; mkdir -p "$bin"
cat > "$bin/snap" <<'EOF'
#!/usr/bin/env bash
[ "$*" = "refresh --time --abs-time" ] || { echo "fake snap: unexpected args $*" >&2; exit 99; }
if [ -n "${FAKE_SNAP_RC:-}" ]; then echo "error: cannot communicate with server" >&2; exit "$FAKE_SNAP_RC"; fi
echo "timer: 00:00~24:00/4"
echo "last: 2026-11-14T01:26:00-08:00"
case "${FAKE_HOLD:-none}" in
  none) ;;
  forever) echo "hold: forever" ;;
  junk) echo "hold: sometime soon" ;;
  *) echo "hold: $(date -d "+${FAKE_HOLD} seconds" --iso-8601=seconds)" ;;
esac
echo "next: 2026-11-14T10:40:00-08:00"
EOF
cat > "$bin/systemctl" <<'EOF'
#!/usr/bin/env bash
[ "$1" = is-active ] && [ "$#" -eq 2 ] || { echo "fake systemctl: unexpected args $*" >&2; exit 99; }
case "$2" in
  apt-daily.timer) state="${FAKE_APT_DAILY:-inactive}" ;;
  apt-daily-upgrade.timer) state="${FAKE_APT_UPGRADE:-inactive}" ;;
  *) echo "fake systemctl: unexpected unit $2" >&2; exit 99 ;;
esac
echo "$state"
[ "$state" = active ]
EOF
chmod +x "$bin/"*

failures=0
check() {  # <description> <expected exit> [VAR=value...] -- [args...]
  local desc="$1" want="$2"; shift 2
  local envs=()
  while [ "$#" -gt 0 ] && [ "$1" != "--" ]; do envs+=("$1"); shift; done
  shift
  env -i PATH="$bin:/usr/bin:/bin" "${envs[@]}" bash "$root/scripts/update_hold_check.sh" "$@" > "$tmp/out" 2>&1
  local rc=$?
  if [ "$rc" -eq "$want" ]; then echo "ok   $desc (exit $rc)"
  else echo "FAIL $desc: exit $rc, want $want"; sed 's/^/     /' "$tmp/out"; failures=$((failures + 1)); fi
}
says() {  # <description> <fixed string>: the last check's output contains it
  if grep -qF -- "$2" "$tmp/out"; then echo "ok   $1"
  else echo "FAIL $1: output was"; sed 's/^/     /' "$tmp/out"; failures=$((failures + 1)); fi
}

check "min_seconds not a number" 2 FAKE_HOLD=172800 -- 12h
check "two arguments" 2 FAKE_HOLD=172800 -- 1 2
check "held 48 h, apt timers stopped" 0 FAKE_HOLD=172800 --
says "  names the hold" "automatic updates held: snap hold until 20"
check "held forever, apt timers stopped" 0 FAKE_HOLD=forever --
check "no snap hold" 1 --
says "  says snap is not held" "snap auto-refresh is not held"
says "  prints the sudo command" "sudo snap refresh --hold=48h && sudo systemctl stop apt-daily.timer apt-daily-upgrade.timer"
check "snap hold ends in 1 h, 12 h needed" 1 FAKE_HOLD=3600 --
says "  says the hold ends too soon" "sooner than 43200s from now"
check "snap hold ends in 1 h, 30 min needed" 0 FAKE_HOLD=3600 -- 1800
check "apt-daily-upgrade.timer still active" 1 FAKE_HOLD=172800 FAKE_APT_UPGRADE=active --
says "  names the timer" "apt-daily-upgrade.timer is active"
check "apt-daily.timer still active" 1 FAKE_HOLD=172800 FAKE_APT_DAILY=active --
check "nothing held: every problem is listed" 1 FAKE_APT_DAILY=active FAKE_APT_UPGRADE=active --
if [ "$(grep -c '^automatic updates are not held: ' "$tmp/out")" -eq 3 ]; then echo "ok   three problems listed"
else echo "FAIL three problems listed:"; sed 's/^/     /' "$tmp/out"; failures=$((failures + 1)); fi
check "a failed apt timer counts as stopped" 0 FAKE_HOLD=172800 FAKE_APT_DAILY=failed --
check "snap cannot be read" 2 FAKE_SNAP_RC=1 --
check "hold time cannot be read" 2 FAKE_HOLD=junk --

echo "failures: $failures"
[ "$failures" -eq 0 ]
