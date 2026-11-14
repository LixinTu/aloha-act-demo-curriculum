#!/usr/bin/env bash
# Tests for the two GPU guards against a fake nvidia-smi (about 40 s, no GPU needed):
# scripts/require_free_gpu.sh (sourced by the launch scripts: exit 4 before any output when the GPU is
# busy or cannot be checked) and scripts/wait_gpu_idle.sh (POLL_S=1).
# Usage: scripts/test_gpu_guards.sh   (exit 0 = all passed, otherwise the number of failed checks)
# The launch scripts run under a 10 s timeout with a fake sg on PATH, so a broken guard fails the test
# instead of starting a real run.
set -uo pipefail
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fake="$(mktemp -d)"
trap 'rm -rf "$fake"' EXIT
# The fake answers call N with line N of $fake/outcomes (the last line repeats).
cat > "$fake/nvidia-smi" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
d="$(cd "$(dirname "$0")" && pwd)"
n=$(( $(cat "$d/count") + 1 )); echo "$n" > "$d/count"
total=$(wc -l < "$d/outcomes"); i=$(( n < total ? n : total ))
case "$(sed -n "${i}p" "$d/outcomes")" in
  busy) echo "123, /fake/python" ;;
  idle) ;;
  stall) sleep 4 ;;
  fail) exit 9 ;;
esac
EOF
cat > "$fake/sg" <<'EOF'
#!/usr/bin/env bash
echo "FAKE SG CALLED: $*"
exit 99
EOF
chmod +x "$fake/nvidia-smi" "$fake/sg"
fails=0
check() {  # <name> <want exit> <got exit> <output> <must match> <must not match>
  local name=$1 want=$2 rc=$3 out=$4 must=$5 mustnot=$6 bad=0
  echo "== $name: exit $rc (want $want), nvidia-smi calls $(cat "$fake/count")"
  if [ "$rc" -ne "$want" ]; then echo "   FAIL: exit code"; bad=1; fi
  if ! printf '%s\n' "$out" | grep -qE "$must"; then echo "   FAIL: missing /$must/"; bad=1; fi
  if [ -n "$mustnot" ] && printf '%s\n' "$out" | grep -qE "$mustnot"; then echo "   FAIL: found /$mustnot/"; bad=1; fi
  if [ "$bad" -ne 0 ]; then fails=$((fails + 1)); printf '%s\n' "$out" | sed 's/^/   | /'; fi
}
set_outcomes() { printf '%s\n' $1 > "$fake/outcomes"; echo 0 > "$fake/count"; }

# --- require_free_gpu.sh on its own
for case in "idle 0 ^STARTED$ refusing" "busy 4 123,./fake/python ^STARTED$" "fail 4 nvidia-smi.failed ^STARTED$"; do
  read -r outcome want must mustnot <<< "$case"
  set_outcomes "$outcome"
  out="$(cd "$repo" && PATH="$fake:$PATH" bash -c 'set -euo pipefail; source scripts/require_free_gpu.sh; echo STARTED' 2>&1)"
  check "require_free_gpu $outcome" "$want" $? "$out" "$must" "$mustnot"
done
if [ "$fails" -gt 0 ]; then echo "the guard itself failed; not running the launch scripts"; exit "$fails"; fi

# --- each launch script stops with exit 4 before creating its output
tmp="$fake/out"
launch() {  # <name> <path that must not appear> <command...>
  local name=$1 created=$2; shift 2
  set_outcomes busy
  local out rc
  out="$(cd "$repo" && PATH="$fake:$PATH" timeout 10 "$@" 2>&1)"; rc=$?
  check "$name refuses on a busy GPU" 4 "$rc" "$out" "refusing to start: the GPU already runs" "FAKE SG CALLED"
  if [ -e "$created" ]; then echo "   FAIL: created $created"; fails=$((fails + 1)); fi
}
launch run_baseline.sh "$tmp/baseline/seed1001" scripts/run_baseline.sh 1001 "$tmp/baseline"
launch run_candidate.sh "$tmp/candidate" scripts/run_candidate.sh "$tmp/candidate" experiments/uniform_candidate
launch run_lerobot_train_control.sh "$tmp/control/seed1000" scripts/run_lerobot_train_control.sh 1000 "$tmp/control"
launch oracle_run.sh "$fake/none" scripts/oracle_run.sh

# --- wait_gpu_idle.sh
w() {  # <name> <want exit> <timeout s> <outcomes> <must match> <must not match> <idle s> [command...]
  local name=$1 want=$2 to=$3 outs=$4 must=$5 mustnot=$6 need=$7; shift 7
  set_outcomes "$outs"
  local out rc
  out="$(PATH="$fake:$PATH" POLL_S=1 timeout "$to" "$repo/scripts/wait_gpu_idle.sh" "$need" "$@" 2>&1)"; rc=$?
  check "wait_gpu_idle $name" "$want" "$rc" "$out" "$must" "$mustnot"
}
w busy-forever     124 4  "busy"            "GPU busy: 123, /fake/python" "^RAN$|starting:" 2 echo RAN
w idle-busy-idle   0   15 "idle busy idle"  "GPU idle for 2 s; starting: echo RAN" "restarts" 2 echo RAN
w stall-restarts   0   20 "idle stall idle" "since the previous poll \(suspend or stall\)" "" 3 echo RAN
w query-fails      3   5  "fail"            "nvidia-smi failed" "^RAN$" 2 echo RAN
w passes-exit-code 7   5  "idle"            "starting: bash -c exit 7" "" 1 bash -c 'exit 7'
w bad-idle-arg     2   5  "idle"            "usage:" "^RAN$" x echo RAN
w no-command       2   5  "idle"            "usage:" "" 5
echo "failures: $fails"
exit "$fails"
