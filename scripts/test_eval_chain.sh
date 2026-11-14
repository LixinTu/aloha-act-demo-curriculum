#!/usr/bin/env bash
# Tests without GPU work (a few seconds):
#   1. control flow of scripts/eval_determinism_chain.sh against fake stage commands: stop with exit 8
#      when the proxy comparison differs, exit 9 when the hidden comparison differs, exit 1 on a failed
#      stage, skip finished stages on a restart;
#   2. the refusals of scripts/eval_blocks.sh and scripts/reeval_baseline_seed.sh (bad arguments,
#      existing output, busy GPU, faulty kernel log, wrong weights hash). Every case ends before an
#      evaluation would start; a 10 s timeout and a fake nvidia-smi guard against a broken refusal.
# Usage: scripts/test_eval_chain.sh   (exit 0 = all passed, otherwise the number of failed checks)
set -uo pipefail
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
top="$(mktemp -d)"
trap 'rm -rf "$top"' EXIT
fails=0

# --- 1: chain control flow
make_root() {  # <dir>: a fake repository root with the real chain and stage library, fake commands
  local t="$1"
  mkdir -p "$t/scripts" "$t/logs" "$t/.venv/bin" "$t/experiments/eval_determinism"
  cp "$repo/scripts/eval_determinism_chain.sh" "$repo/scripts/stage_lib.sh" "$t/scripts/"
  : > "$t/experiments/eval_determinism/protocol_eval_det.json"
  : > "$t/calls"
  printf '#!/usr/bin/env bash\nshift\nexec "$@"\n' > "$t/scripts/wait_gpu_idle.sh"
  cat > "$t/scripts/eval_blocks.sh" <<'EOF'
#!/usr/bin/env bash
echo "eval_blocks $3" >> "$(cd "$(dirname "$0")/.." && pwd)/calls"
exit "${FAKE_EVAL_RC:-0}"
EOF
  cat > "$t/scripts/reeval_baseline_seed.sh" <<'EOF'
#!/usr/bin/env bash
echo "reeval_baseline_seed $(basename "$1")" >> "$(cd "$(dirname "$0")/.." && pwd)/calls"
EOF
  cat > "$t/.venv/bin/python" <<'EOF'
#!/usr/bin/env bash
# fake compare_evals.py: call N exits with word N of FAKE_COMPARE (default 0)
root="$(cd "$(dirname "$0")/../.." && pwd)"
echo "compare" >> "$root/calls"
n="$(grep -c '^compare' "$root/calls")"
rc="$(printf '%s\n' ${FAKE_COMPARE:-0} | sed -n "${n}p")"
exit "${rc:-0}"
EOF
  chmod +x "$t/scripts/"*.sh "$t/.venv/bin/python"
}
run_case() {  # <name> <want exit> <must match in output> <expected calls> [VAR=value...]
  local name=$1 want=$2 must=$3 calls=$4; shift 4
  local t="$top/$name"
  if [ ! -d "$t" ]; then make_root "$t"; fi
  local out rc got bad=0
  out="$(env "$@" IDLE_S=1 "$t/scripts/eval_determinism_chain.sh" 2>&1)"; rc=$?
  got="$(tr ' ' ':' < "$t/calls" | paste -sd, -)"
  echo "== chain $name: exit $rc (want $want); calls: $got"
  if [ "$rc" -ne "$want" ]; then echo "   FAIL: exit code"; bad=1; fi
  if ! printf '%s\n' "$out" | grep -qE "$must"; then echo "   FAIL: missing /$must/"; bad=1; fi
  if [ "$got" != "$calls" ]; then echo "   FAIL: calls, want $calls"; bad=1; fi
  if [ "$bad" -ne 0 ]; then fails=$((fails + 1)); printf '%s\n' "$out" | sed 's/^/   | /'; fi
}
proxy="eval_blocks:proxy,eval_blocks:proxy,compare"
full="$proxy,reeval_baseline_seed:seed1000,reeval_baseline_seed:seed1001,reeval_baseline_seed:seed1002,eval_blocks:hidden,compare"
run_case pass 0 "evaluator determinism chain done" "$full" FAKE_COMPARE="0 0"
run_case pass 0 "det-hidden-run2 already done" "$full,compare,compare" FAKE_COMPARE="0 0 0 0"   # restart: stages skipped
run_case proxy-differs 8 "differ on the proxy block: protocol unchanged" "$proxy" FAKE_COMPARE="1"
run_case hidden-differs 9 "differ on the hidden block: the change is dropped" "$full" FAKE_COMPARE="0 1"
run_case stage-fails 1 "det-proxy-run1 exit=6" "eval_blocks:proxy" FAKE_EVAL_RC=6
t="$top/no-protocol"; make_root "$t"; rm "$t/experiments/eval_determinism/protocol_eval_det.json"
run_case no-protocol 1 "no protocol file" ""

# --- 2: refusals of the real evaluation scripts
fake="$top/bin"; mkdir -p "$fake"
cat > "$fake/nvidia-smi" <<'EOF'
#!/usr/bin/env bash
if [ -e "$(dirname "$0")/busy" ]; then echo "123, /fake/python"; fi
EOF
chmod +x "$fake/nvidia-smi"
fault="$repo/scripts/testdata/kernel_fault_2026-11-12.log"
clean="$repo/scripts/testdata/kernel_clean.log"
proto="$repo/task/aloha-act-demo-curriculum/environment/fixed_train.json"
run="$top/run"; mkdir -p "$run/train/pretrained_model" "$run/schedule"
printf 'weights' > "$run/train/pretrained_model/model.safetensors"
: > "$run/compile.json"; : > "$run/train.json"
sha256sum "$run/train/pretrained_model/model.safetensors" > "$run/checkpoint.sha256"
check() {  # <name> <want exit> <must match> <path that must not exist or ""> <command...>
  local name=$1 want=$2 must=$3 absent=$4; shift 4
  local out rc bad=0
  out="$(cd "$repo" && PATH="$fake:$PATH" timeout 10 "$@" 2>&1)"; rc=$?
  echo "== $name: exit $rc (want $want)"
  if [ "$rc" -ne "$want" ]; then echo "   FAIL: exit code"; bad=1; fi
  if ! printf '%s\n' "$out" | grep -qE "$must"; then echo "   FAIL: missing /$must/"; bad=1; fi
  if [ -n "$absent" ] && [ -e "$absent" ]; then echo "   FAIL: created $absent"; bad=1; fi
  if [ "$bad" -ne 0 ]; then fails=$((fails + 1)); printf '%s\n' "$out" | sed 's/^/   | /'; fi
}
o="$top/out"
check "eval_blocks without PROTOCOL" 2 "set PROTOCOL" "$o" env -u PROTOCOL scripts/eval_blocks.sh "$run" "$o" proxy
check "eval_blocks bad block name" 2 "must be hidden, proxy or both" "$o" env PROTOCOL="$proto" scripts/eval_blocks.sh "$run" "$o" all
check "eval_blocks without a checkpoint" 2 "no checkpoint at" "$o" env PROTOCOL="$proto" scripts/eval_blocks.sh "$top/none" "$o" proxy
touch "$fake/busy"
check "eval_blocks on a busy GPU" 4 "refusing to start: the GPU already runs" "$o" env PROTOCOL="$proto" KERNEL_HEALTH_LOG="$clean" scripts/eval_blocks.sh "$run" "$o" both
rm "$fake/busy"
check "eval_blocks on a faulty kernel" 5 "kernel log of this boot is not clean" "$o" env PROTOCOL="$proto" KERNEL_HEALTH_LOG="$fault" scripts/eval_blocks.sh "$run" "$o" both
e="$top/has-eval"; mkdir -p "$e/eval_proxy"
check "eval_blocks with existing output" 2 "already has an eval_proxy" "" env PROTOCOL="$proto" KERNEL_HEALTH_LOG="$fault" scripts/eval_blocks.sh "$run" "$e" proxy
q="$top/other-protocol"; mkdir -p "$q"; echo '{}' > "$q/protocol.json"
check "eval_blocks with another protocol.json" 2 "protocol.json differs" "" env PROTOCOL="$proto" KERNEL_HEALTH_LOG="$fault" scripts/eval_blocks.sh "$run" "$q" proxy
d="$top/dst"
check "reeval_baseline_seed without a registered hash" 2 "no registered checkpoint" "$d" env PROTOCOL="$proto" scripts/reeval_baseline_seed.sh "$top/none" "$d"
check "reeval_baseline_seed with existing evaluations" 2 "already has eval_proxy" "" env PROTOCOL="$proto" scripts/reeval_baseline_seed.sh "$run" "$e"
check "reeval_baseline_seed into a foreign directory" 2 "exists without a checkpoint" "" env PROTOCOL="$proto" scripts/reeval_baseline_seed.sh "$run" "$q"
touch "$fake/busy"
check "reeval_baseline_seed copies, then stops on a busy GPU" 4 "refusing to start: the GPU already runs" "$d/eval_hidden" env PROTOCOL="$proto" KERNEL_HEALTH_LOG="$clean" scripts/reeval_baseline_seed.sh "$run" "$d"
if cmp -s "$run/train/pretrained_model/model.safetensors" "$d/train/pretrained_model/model.safetensors" && [ -e "$d/schedule" ] && [ -e "$d/train.json" ]; then
  echo "== reeval_baseline_seed copied schedule and training outputs: ok"
else echo "== reeval_baseline_seed copy: FAIL"; fails=$((fails + 1)); fi
check "reeval_baseline_seed second attempt keeps the copy" 4 "refusing to start: the GPU already runs" "$d/eval_hidden" env PROTOCOL="$proto" KERNEL_HEALTH_LOG="$clean" scripts/reeval_baseline_seed.sh "$run" "$d"
printf 'changed' > "$d/train/pretrained_model/model.safetensors"
check "reeval_baseline_seed with changed weights" 7 "differ from the registered hash" "$d/eval_hidden" env PROTOCOL="$proto" KERNEL_HEALTH_LOG="$clean" scripts/reeval_baseline_seed.sh "$run" "$d"
rm "$fake/busy"
echo "failures: $fails"
exit "$fails"
