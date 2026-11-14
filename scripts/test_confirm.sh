#!/usr/bin/env bash
# Tests for the confirmation step of an agent's best candidate (a few seconds, no GPU):
#   1. scripts/agent_round_info.py on a real scored Judge log (the final-protocol oracle round) and on
#      synthetic invalid, failed, truncated and inconsistent logs;
#   2. scripts/confirm_stats.py on synthetic retrains: the verdict boundaries of the rule written down
#      before the agent run (168 and 121 of 1000) and every refusal;
#   3. scripts/confirm_candidate.sh in a scratch copy of the repository with fake training and waiting
#      commands: the hash check against the Judge round, the copy, the two seeds, a restart, refusals.
# Usage: scripts/test_confirm.sh   (exit 0 = all passed, otherwise the number of failed checks)
set -uo pipefail
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
py="$repo/.venv/bin/python"
top="$(mktemp -d)"
trap 'rm -rf "$top"' EXIT
fails=0
check() {  # <name> <want exit> <got exit> <output> <must match> [must not match]
  local name=$1 want=$2 rc=$3 out=$4 must=$5 mustnot=${6:-} bad=0
  echo "== $name: exit $rc (want $want)"
  if [ "$rc" -ne "$want" ]; then echo "   FAIL: exit code"; bad=1; fi
  if ! printf '%s\n' "$out" | grep -qE -- "$must"; then echo "   FAIL: missing /$must/"; bad=1; fi
  if [ -n "$mustnot" ] && printf '%s\n' "$out" | grep -qE -- "$mustnot"; then echo "   FAIL: found /$mustnot/"; bad=1; fi
  if [ "$bad" -ne 0 ]; then fails=$((fails + 1)); printf '%s\n' "$out" | sed 's/^/   | /'; fi
}

# --- 1: agent_round_info.py
out="$("$py" "$repo/scripts/agent_round_info.py" "$repo/experiments/oracle_final/judge/judge_feedback_agent-1.log" --output "$top/r.json" 2>&1)"
check "round info: real scored round" 0 $? "$out" "round scored: 81/500"
P=7c3b41dd3c3c77247d58db304721d80e52a4cced58201c0cf347b0c08816cab3
S=c6e00c8736a118c6b77ab9a9edba77b06db9c4a3a6463328061ceb4de0bd400c
staged="{\"event\": \"candidate_staged\", \"files\": [\"policy.json\"], \"policy_sha256\": \"$P\", \"scores_sha256\": null, \"sizes\": {}}"
compiled="{\"event\": \"schedule_compiled\", \"policy_name\": \"x\", \"schedule_sha256\": \"$S\"}"
printf '%s\n' '{"event": "judge_started"}' "$staged" '{"event": "candidate_invalid", "stage": "validate_compile", "detail": "negative weight"}' > "$top/invalid.log"
out="$("$py" "$repo/scripts/agent_round_info.py" "$top/invalid.log" 2>&1)"
check "round info: invalid candidate" 2 $? "$out" "not scored: candidate_invalid \(negative weight\)"
printf '%s\n' "$staged" "$compiled" '{"event": "run_error", "stage": "train", "returncode": -1, "log_tail": "timeout"}' > "$top/error.log"
out="$("$py" "$repo/scripts/agent_round_info.py" "$top/error.log" 2>&1)"
check "round info: run error" 2 $? "$out" "not scored: run_error"
printf '%s\n' "$staged" "$compiled" 'Traceback (most recent call last):' > "$top/short.log"
out="$("$py" "$repo/scripts/agent_round_info.py" "$top/short.log" 2>&1)"
check "round info: log ends early" 2 $? "$out" "not scored: incomplete"
printf '%s\n' "$staged" "$compiled" "{\"event\": \"complete\", \"policy_sha256\": \"other\", \"schedule_sha256\": \"$S\", \"reward\": 0.2, \"successes\": 100, \"n_episodes\": 500}" > "$top/odd.log"
out="$("$py" "$repo/scripts/agent_round_info.py" "$top/odd.log" 2>&1)"
check "round info: complete names another policy" 2 $? "$out" "not scored: inconsistent"
out="$("$py" "$repo/scripts/agent_round_info.py" "$top/none.log" 2>&1)"
check "round info: missing log" 2 $? "$out" "no feedback log"

# --- 2: confirm_stats.py on synthetic retrains
make_case() {  # <dir> <k1001> <k1002> [key=value overrides: seed1001=, sched=, policy=, cudnn=, status=]
  "$py" -I - "$repo" "$@" <<'EOF'
import json, sys
from pathlib import Path
repo, d, k1, k2 = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
opt = dict(a.split("=", 1) for a in sys.argv[5:])
hidden = json.loads((repo / "task/aloha-act-demo-curriculum/tests/hidden_seeds.json").read_text())
start, n = int(hidden["start_seed"]), int(hidden["n_episodes"])
def write(p, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj))
def ev(k):
    return {"start_seed": start, "n_episodes": n, "successes": k, "success_rate": k / n,
            "per_episode": [{"seed": start + i, "success": i < k, "max_reward": 4.0 if i < k else 1.0} for i in range(n)]}
P, S = "a" * 64, "b" * 64
base = d / "baseline"
write(base / "summary.json", {"protocol": {"protocol_version": 3, "cudnn_deterministic": True},
                              "seeds": [{"seed": 1000, "hidden_successes": 80}, {"seed": 1001, "hidden_successes": 64},
                                        {"seed": 1002, "hidden_successes": 57}],
                              "single_run_sd_estimate": 0.02357965224510319})
write(base / "seed1001/eval_hidden/eval_result.json", ev(64))
write(base / "seed1002/eval_hidden/eval_result.json", ev(57))
c = d / "confirm"
write(c / "judge_round.json", {"status": opt.get("status", "scored"), "policy_name": "cand", "policy_sha256": P,
                               "scores_sha256": None, "schedule_sha256": S, "successes": 95, "n_episodes": n, "reward": 95 / n})
for seed, k in ((1001, k1), (1002, k2)):
    r = c / f"seed{seed}"
    write(r / "train/train_info.json", {"seed": int(opt.get(f"seed{seed}", seed)), "steps": 100000, "exposures": 800000,
                                        "protocol_version": 3, "cudnn_deterministic": opt.get("cudnn", "1") == "1", "train_s": 16000.0})
    write(r / "schedule/schedule_summary.json", {"schedule_sha256": opt.get("sched", S) if seed == 1002 else S,
                                                 "policy_sha256": opt.get("policy", P), "scores_sha256": None})
    write(r / "eval_hidden/eval_result.json", ev(k))
EOF
}
verdict() {  # <name> <want exit> <must match> <k1001> <k1002> [overrides...] [-- extra confirm_stats args]
  local name=$1 want=$2 must=$3; shift 3
  local d="$top/stats-$name" extra=()
  local args=()
  while [ "$#" -gt 0 ]; do
    if [ "$1" = "--" ]; then shift; extra=("$@"); break; fi
    args+=("$1"); shift
  done
  make_case "$d" "${args[@]}"
  out="$("$py" "$repo/scripts/confirm_stats.py" "$d/confirm" --baseline-dir "$d/baseline" "${extra[@]}" 2>&1)"
  check "confirm_stats: $name" "$want" $? "$out" "$must"
}
verdict confirmed-at-168 0 "168/1000 against the baseline's 121/1000 .*needs >= 168/1000; verdict: confirmed$" 84 84
verdict positive-at-167 0 "verdict: positive, not confirmed$" 84 83
verdict positive-at-122 0 "verdict: positive, not confirmed$" 61 61
verdict not-reproduced-at-121 0 "verdict: not reproduced$" 64 57
verdict schedule-differs 2 "seed1002: schedule_sha256 .* differs from the Judge round's" 90 90 sched="$(printf 'c%.0s' {1..64})"
verdict policy-differs 2 "policy_sha256 .* differs from the Judge round's" 90 90 policy="$(printf 'd%.0s' {1..64})"
verdict wrong-seed 2 "seed1001: trained at seed 1000" 90 90 seed1001=1000
verdict other-protocol 2 "protocol .* differs from the baseline's" 90 90 cudnn=0
verdict unscored-round 2 "the Judge round was not scored: candidate_invalid" 90 90 status=candidate_invalid
verdict need-changed 3 "differs from the value written down before the run \(167\)" 90 90 -- --expected-need 167
if [ -f "$top/stats-confirmed-at-168/confirm/confirm.json" ] && grep -q '"verdict": "confirmed"' "$top/stats-confirmed-at-168/confirm/confirm.json"; then
  echo "== confirm.json written with the verdict: ok"
else echo "== confirm.json: FAIL"; fails=$((fails + 1)); fi
if grep -q "$("$py" -c "import json; print(json.load(open('$repo/task/aloha-act-demo-curriculum/tests/hidden_seeds.json'))['start_seed'])")" "$top/stats-confirmed-at-168/confirm/confirm.json"; then
  echo "== confirm.json lists a hidden seed: FAIL"; fails=$((fails + 1))
else echo "== confirm.json lists no hidden seed: ok"; fi

# --- 3: confirm_candidate.sh in a scratch repository with fake training
fake="$top/repo"
mkdir -p "$fake/scripts" "$fake/logs" "$fake/.venv/bin" "$fake/cand"
cp "$repo/scripts/confirm_candidate.sh" "$repo/scripts/stage_lib.sh" "$repo/scripts/agent_round_info.py" "$fake/scripts/"
ln -s "$py" "$fake/.venv/bin/python"
printf '#!/usr/bin/env bash\nshift\nexec "$@"\n' > "$fake/scripts/wait_gpu_idle.sh"
cat > "$fake/scripts/run_candidate.sh" <<'EOF'
#!/usr/bin/env bash
root="$(cd "$(dirname "$0")/.." && pwd)"
echo "run_candidate $1 $3" >> "$root/calls"
mkdir -p "$root/$1"
EOF
cat > "$fake/scripts/confirm_stats.py" <<'EOF'
import sys
print("fake confirm_stats", sys.argv[1])
EOF
chmod +x "$fake/scripts/"*.sh
printf '{"schema_version": 1, "name": "cand"}\n' > "$fake/cand/policy.json"
pol="$(sha256sum "$fake/cand/policy.json" | cut -d' ' -f1)"
printf '%s\n' "{\"event\": \"candidate_staged\", \"files\": [\"policy.json\"], \"policy_sha256\": \"$pol\", \"scores_sha256\": null}" \
  "{\"event\": \"schedule_compiled\", \"policy_name\": \"cand\", \"schedule_sha256\": \"$S\"}" \
  "{\"event\": \"complete\", \"policy_sha256\": \"$pol\", \"schedule_sha256\": \"$S\", \"reward\": 0.19, \"successes\": 95, \"n_episodes\": 500}" > "$fake/round.log"
run_confirm() {  # <name> <want exit> <must match> <expected calls> <args...>
  local name=$1 want=$2 must=$3 calls=$4; shift 4
  : > "$fake/calls"
  out="$(cd "$fake" && IDLE_S=1 scripts/confirm_candidate.sh "$@" 2>&1)"; rc=$?
  local got; got="$(paste -sd, "$fake/calls")"
  check "confirm_candidate: $name" "$want" "$rc" "$out" "$must"
  if [ "$got" != "$calls" ]; then echo "   FAIL: calls '$got', want '$calls'"; fails=$((fails + 1)); fi
}
seeds="run_candidate experiments/agent_confirm/c1/seed1001 1001,run_candidate experiments/agent_confirm/c1/seed1002 1002"
run_confirm "matching candidate" 0 "fake confirm_stats experiments/agent_confirm/c1" "$seeds" c1 cand round.log
if cmp -s "$fake/cand/policy.json" "$fake/experiments/agent_confirm/c1/candidate/policy.json"; then echo "== candidate copied: ok"
else echo "== candidate copied: FAIL"; fails=$((fails + 1)); fi
run_confirm "restart skips both stages" 0 "confirm-c1-seed1002 already done" "" c1 cand round.log
printf 'changed' > "$fake/experiments/agent_confirm/c1/candidate/policy.json"
run_confirm "copied candidate changed" 2 "does not match the Judge round" "" c1 cand round.log
mkdir -p "$fake/cand2"; printf '{"schema_version": 1, "name": "other"}\n' > "$fake/cand2/policy.json"
run_confirm "candidate differs from the staged one" 2 "does not match what the Judge staged" "" c2 cand2 round.log
cp "$fake/cand/policy.json" "$fake/cand2/policy.json"; printf 'x' > "$fake/cand2/notes.txt"
run_confirm "unexpected extra file" 2 "unexpected files" "" c3 cand2 round.log
printf '%s\n' '{"event": "candidate_invalid", "stage": "workspace", "detail": "bad layout"}' > "$fake/bad.log"
run_confirm "unscored round" 2 "was not scored; nothing to confirm" "" c4 cand bad.log
run_confirm "bad name" 2 "name must match" "" "C 5" cand round.log
run_confirm "wrong argument count" 2 "usage:" "" c6 cand
for s in confirm_candidate.sh test_confirm.sh; do
  if bash -n "$repo/scripts/$s"; then echo "== syntax $s: ok"; else echo "== syntax $s: FAIL"; fails=$((fails + 1)); fi
done
echo "failures: $fails"
exit "$fails"
