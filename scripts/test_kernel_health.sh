#!/usr/bin/env bash
# Tests for scripts/kernel_health.sh and scripts/run_guards.sh (a few seconds, no GPU needed).
# Fixtures: scripts/testdata/kernel_fault_2026-11-12.log (real lines from the boot that hung on
# 2026-11-12) and scripts/testdata/kernel_clean.log.
# Usage: scripts/test_kernel_health.sh   (exit 0 = all passed, otherwise the number of failed checks)
# The launch scripts run under a 10 s timeout with a fake nvidia-smi (idle GPU) and a fake sg on PATH,
# so a broken guard fails the test instead of starting a real run. scripts/agent_run.sh is not executed
# here (it needs the operator's confirmation); it gets a syntax check only.
set -uo pipefail
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fault="$repo/scripts/testdata/kernel_fault_2026-11-12.log"
clean="$repo/scripts/testdata/kernel_clean.log"
fake="$(mktemp -d)"
trap 'rm -rf "$fake"' EXIT
printf '#!/usr/bin/env bash\nexit 0\n' > "$fake/nvidia-smi"   # no compute process: the GPU is idle
printf '#!/usr/bin/env bash\necho "FAKE SG CALLED: $*"\nexit 99\n' > "$fake/sg"
chmod +x "$fake/nvidia-smi" "$fake/sg"
fails=0
check() {  # <name> <want exit> <got exit> <output> <must match> <must not match>
  local name=$1 want=$2 rc=$3 out=$4 must=$5 mustnot=$6 bad=0
  echo "== $name: exit $rc (want $want)"
  if [ "$rc" -ne "$want" ]; then echo "   FAIL: exit code"; bad=1; fi
  if ! printf '%s\n' "$out" | grep -qE "$must"; then echo "   FAIL: missing /$must/"; bad=1; fi
  if [ -n "$mustnot" ] && printf '%s\n' "$out" | grep -qE "$mustnot"; then echo "   FAIL: found /$mustnot/"; bad=1; fi
  if [ "$bad" -ne 0 ]; then fails=$((fails + 1)); printf '%s\n' "$out" | sed 's/^/   | /'; fi
}
expect_file() {  # <name> <path> <must match>
  if [ -s "$2" ] && grep -qE "$3" "$2"; then echo "== $1: ok"; else echo "== $1: FAIL ($2 missing or without /$3/)"; fails=$((fails + 1)); fi
}

# --- kernel_health.sh on its own
out="$(KERNEL_HEALTH_LOG="$fault" "$repo/scripts/kernel_health.sh" 2>&1)"
check "fault fixture" 5 $? "$out" "kernel fault lines in .*: 8;" "kernel log clean"
out="$(KERNEL_HEALTH_LOG="$clean" "$repo/scripts/kernel_health.sh" 2>&1)"
check "clean fixture" 0 $? "$out" "^kernel log clean" "fault lines"
out="$(KERNEL_HEALTH_LOG="$fake/missing.log" "$repo/scripts/kernel_health.sh" 2>&1)"
check "unreadable log file" 3 $? "$out" "cannot read the kernel log file" "kernel log clean"
out="$("$repo/scripts/kernel_health.sh" --since 2>&1)"
check "--since without a time" 2 $? "$out" "usage:" "kernel log clean"
out="$("$repo/scripts/kernel_health.sh" --bogus now 2>&1)"
check "unknown option" 2 $? "$out" "usage:" "kernel log clean"
# every fault line of the fixture is recognised on its own, and no benign line is
n=0
while IFS= read -r line; do
  n=$((n + 1))
  printf '%s\n' "$line" > "$fake/one.log"
  out="$(KERNEL_HEALTH_LOG="$fake/one.log" "$repo/scripts/kernel_health.sh" 2>&1)"; rc=$?
  case "$line" in
    *"BUG: Bad "*|*"BUG: unable to handle page fault"*|*"Oops: "*|*"soft lockup"*) want=5 ;;
    *) want=0 ;;
  esac
  if [ "$rc" -ne "$want" ]; then echo "== fixture line $n: FAIL exit $rc (want $want): $line"; fails=$((fails + 1)); fi
done < "$fault"
echo "== fixture lines checked one by one: $n"
# the live kernel log must be readable (0 = clean, 5 = faults in this boot; 3 would mean unreadable)
out="$("$repo/scripts/kernel_health.sh" 2>&1)"; rc=$?
if [ "$rc" -eq 0 ] || [ "$rc" -eq 5 ]; then echo "== live kernel log readable: exit $rc ($(printf '%s\n' "$out" | head -n 1))"
else echo "== live kernel log: FAIL exit $rc"; printf '%s\n' "$out" | sed 's/^/   | /'; fails=$((fails + 1)); fi

# --- run_guards.sh
out="$(cd "$repo" && KERNEL_HEALTH_LOG="$fault" bash -c 'set -euo pipefail; source scripts/run_guards.sh; echo STARTED' 2>&1)"
check "run_guards refuses on a faulty kernel" 5 $? "$out" "refusing to start: the kernel log of this boot is not clean" "^STARTED$"
out="$(cd "$repo" && KERNEL_HEALTH_LOG="$fake/missing.log" bash -c 'set -euo pipefail; source scripts/run_guards.sh; echo STARTED' 2>&1)"
check "run_guards refuses on an unreadable log" 3 $? "$out" "refusing to start" "^STARTED$"
out="$(cd "$repo" && KERNEL_HEALTH_LOG="$clean" bash -c 'set -euo pipefail; source scripts/run_guards.sh; echo STARTED' 2>&1)"
check "run_guards passes on a clean kernel" 0 $? "$out" "^STARTED$" "refusing"
run="$fake/run"; mkdir -p "$run"
out="$(cd "$repo" && KERNEL_HEALTH_LOG="$clean" bash -c 'set -euo pipefail; source scripts/run_guards.sh; export KERNEL_HEALTH_LOG="$1"; kernel_health_after "$2"; echo FINISHED' _ "$fault" "$run" 2>&1)"
check "kernel_health_after flags faults during the run" 6 $? "$out" "treat the result in .* as suspect" "^FINISHED$"
expect_file "kernel_health.txt records the faults" "$run/kernel_health.txt" "kernel fault lines"
out="$(cd "$repo" && KERNEL_HEALTH_LOG="$clean" bash -c 'set -euo pipefail; source scripts/run_guards.sh; kernel_health_after "$1"; echo FINISHED' _ "$run" 2>&1)"
check "kernel_health_after passes a clean run" 0 $? "$out" "^FINISHED$" "suspect"
expect_file "kernel_health.txt records the clean check" "$run/kernel_health.txt" "^kernel log clean"
out="$(cd "$repo" && KERNEL_HEALTH_LOG="$clean" bash -c 'set -euo pipefail; source scripts/run_guards.sh; record_checkpoint_sha256 "$1"; echo RECORDED' _ "$run" 2>&1)"
check "record_checkpoint_sha256 fails without weights" 1 $? "$out" "no weights at" "^RECORDED$"
mkdir -p "$run/train/pretrained_model"; printf 'weights' > "$run/train/pretrained_model/model.safetensors"
out="$(cd "$repo" && KERNEL_HEALTH_LOG="$clean" bash -c 'set -euo pipefail; source scripts/run_guards.sh; record_checkpoint_sha256 "$1"; sha256sum -c "$1/checkpoint.sha256"' _ "$run" 2>&1)"
check "record_checkpoint_sha256 writes a checkable hash" 0 $? "$out" "model.safetensors: (OK|成功)" "FAILED|失败"

# --- each launch script stops with exit 5 before creating its output or calling the harness
tmp="$fake/out"
launch() {  # <name> <path that must not appear> <command...>
  local name=$1 created=$2; shift 2
  local out rc
  out="$(cd "$repo" && PATH="$fake:$PATH" KERNEL_HEALTH_LOG="$fault" timeout 10 "$@" 2>&1)"; rc=$?
  check "$name refuses on a faulty kernel" 5 "$rc" "$out" "refusing to start: the kernel log of this boot is not clean" "FAKE SG CALLED"
  if [ -e "$created" ]; then echo "   FAIL: created $created"; fails=$((fails + 1)); fi
}
launch run_baseline.sh "$tmp/baseline/seed1001" scripts/run_baseline.sh 1001 "$tmp/baseline"
launch run_candidate.sh "$tmp/candidate" scripts/run_candidate.sh "$tmp/candidate" experiments/uniform_candidate
launch oracle_run.sh "$fake/none" scripts/oracle_run.sh
for s in agent_run.sh final_protocol_chain.sh run_baseline.sh run_candidate.sh oracle_run.sh kernel_health.sh run_guards.sh; do
  if bash -n "$repo/scripts/$s"; then echo "== syntax $s: ok"; else echo "== syntax $s: FAIL"; fails=$((fails + 1)); fi
done
echo "failures: $fails"
exit "$fails"
