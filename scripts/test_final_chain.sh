#!/usr/bin/env bash
# Control-flow tests for scripts/final_protocol_chain.sh against fake stage commands (a few seconds,
# no GPU): the weights comparison of the seed-1002 verification, the third run, the stop on three
# different weights, skipping finished stages on a restart, and stopping on a failed stage.
# Usage: scripts/test_final_chain.sh   (exit 0 = all passed, otherwise the number of failed checks)
set -uo pipefail
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
top="$(mktemp -d)"
trap 'rm -rf "$top"' EXIT
fails=0

make_root() {  # <dir>: a fake repository root holding the real chain script and fake stage commands
  local t="$1"
  mkdir -p "$t/scripts" "$t/logs" "$t/.venv/bin" "$t/experiments/baseline_v3_det/seed1002" \
    "$t/experiments/trainer_parity/steps100000"
  cp "$repo/scripts/final_protocol_chain.sh" "$repo/scripts/stage_lib.sh" "$t/scripts/"
  : > "$t/experiments/trainer_parity/steps100000/protocol_deterministic.json"
  : > "$t/calls"
  printf '%s  experiments/baseline_v3_det/seed1002/train/pretrained_model/model.safetensors\n' \
    "$(printf 'original' | sha256sum | cut -d' ' -f1)" > "$t/experiments/baseline_v3_det/seed1002/checkpoint.sha256"
  cat > "$t/scripts/wait_gpu_idle.sh" <<'EOF'
#!/usr/bin/env bash
shift
exec "$@"
EOF
  cat > "$t/scripts/run_baseline.sh" <<'EOF'
#!/usr/bin/env bash
# fake: writes the weights named by the next word of FAKE_WEIGHTS; exits FAKE_BASELINE_RC if set
root="$(cd "$(dirname "$0")/.." && pwd)"
echo "run_baseline $*" >> "$root/calls"
if [ -n "${FAKE_BASELINE_RC:-}" ]; then exit "$FAKE_BASELINE_RC"; fi
n="$(grep -c '^run_baseline' "$root/calls")"
content="$(printf '%s\n' ${FAKE_WEIGHTS:-original} | sed -n "${n}p")"
out="$root/$2/seed$1/train/pretrained_model"
mkdir -p "$out"
printf '%s' "$content" > "$out/model.safetensors"
EOF
  cat > "$t/scripts/oracle_run.sh" <<'EOF'
#!/usr/bin/env bash
echo "oracle_run" >> "$(cd "$(dirname "$0")/.." && pwd)/calls"
exit "${FAKE_ORACLE_RC:-0}"
EOF
  cat > "$t/scripts/run_candidate.sh" <<'EOF'
#!/usr/bin/env bash
echo "run_candidate $1" >> "$(cd "$(dirname "$0")/.." && pwd)/calls"
EOF
  cat > "$t/.venv/bin/python" <<'EOF'
#!/usr/bin/env bash
echo "candidate_stats $2" >> "$(cd "$(dirname "$0")/../.." && pwd)/calls"
EOF
  chmod +x "$t/scripts/"*.sh "$t/.venv/bin/python"
}

run_case() {  # <name> <want exit> <must match in output> <expected calls, comma separated> [VAR=value...]
  local name=$1 want=$2 must=$3 calls=$4; shift 4
  local t="$top/$name"
  if [ ! -d "$t" ]; then make_root "$t"; fi
  local out rc got bad=0
  out="$(env "$@" IDLE_S=1 "$t/scripts/final_protocol_chain.sh" 2>&1)"; rc=$?
  got="$(cut -d' ' -f1 "$t/calls" | paste -sd, -)"
  echo "== $name: exit $rc (want $want); calls: $got"
  if [ "$rc" -ne "$want" ]; then echo "   FAIL: exit code"; bad=1; fi
  if ! printf '%s\n' "$out" | grep -qE "$must"; then echo "   FAIL: missing /$must/"; bad=1; fi
  if [ "$got" != "$calls" ]; then echo "   FAIL: calls, want $calls"; bad=1; fi
  if [ "$bad" -ne 0 ]; then fails=$((fails + 1)); printf '%s\n' "$out" | sed 's/^/   | /'; fi
}

all="run_baseline,oracle_run,run_candidate,candidate_stats,run_candidate,candidate_stats"
run_case identical 0 "bit-identical, the original seed-1002 training is reproduced" "$all" FAKE_WEIGHTS="original"
# restart: every stage is skipped; only the cheap statistics are computed again
run_case identical 0 "probe-final-insertion_focus already done" "$all,candidate_stats,candidate_stats" FAKE_WEIGHTS="changed"
run_case third-matches-original 0 "the first verification run is the odd one" "run_baseline,$all" FAKE_WEIGHTS="other original"
run_case third-matches-verification 0 "the original run is the odd one" "run_baseline,$all" FAKE_WEIGHTS="other other"
run_case three-different 7 "three different weights" "run_baseline,run_baseline" FAKE_WEIGHTS="other another"
run_case verification-fails 1 "verification run failed; nothing else started" "run_baseline" FAKE_BASELINE_RC=6
run_case verification-fails 1 "logs/verify-seed1002.exit says '6'" "run_baseline" FAKE_WEIGHTS="original"   # restart: stays stopped
run_case oracle-fails 1 "oracle stage failed; probes not started" "run_baseline,oracle_run" FAKE_ORACLE_RC=6
t="$top/no-hash"; make_root "$t"; : > "$t/experiments/baseline_v3_det/seed1002/checkpoint.sha256"
run_case no-hash 1 "no committed weights hash" ""
echo "failures: $fails"
exit "$fails"
