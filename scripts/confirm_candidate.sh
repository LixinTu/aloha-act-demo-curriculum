#!/usr/bin/env bash
# Confirmation retrains of an agent's best candidate (docs/plan.md step 5; reading committed in 9d7a839
# before the agent run). Start it only after the agent run has ended: it trains on the shared GPU.
#   1. the round's Judge feedback log -> <out>/judge_round.json (scripts/agent_round_info.py); the round
#      must have been scored;
#   2. the candidate files must hash to what the Judge staged in that round; they are copied to
#      <out>/candidate/ once, and both retrains read that copy;
#   3. stages confirm-<name>-seed1001 and confirm-<name>-seed1002: scripts/run_candidate.sh at training
#      seeds 1001 and 1002 into <out>/seed1001 and <out>/seed1002 (each waits for IDLE_S seconds, default
#      900, without GPU compute processes; scripts/stage_lib.sh);
#   4. scripts/confirm_stats.py <out> -> <out>/confirm.json and the verdict.
# <out> is experiments/agent_confirm/<name>. A restart skips finished stages and re-checks the copied
# candidate against the Judge's hashes.
# Usage: setsid scripts/run_bg.sh confirm-<name> scripts/confirm_candidate.sh <name> <candidate_dir> <feedback_log> > /dev/null 2>&1 < /dev/null &
# Exit 2: bad arguments, an unscored round or a hash mismatch; otherwise the exit code of the failing step.
set -uo pipefail
if [ "$#" -ne 3 ]; then
  echo "usage: $0 <name> <candidate_dir> <round_feedback_log>" >&2
  exit 2
fi
name="$1"; cand="$2"; feedback="$3"
if ! [[ "$name" =~ ^[a-z0-9][a-z0-9_-]{0,63}$ ]]; then echo "name must match [a-z0-9][a-z0-9_-]*, got '$name'" >&2; exit 2; fi
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$root"
idle="${IDLE_S:-900}"
source scripts/stage_lib.sh  # stage <name> <command...>
out="experiments/agent_confirm/$name"
mkdir -p "$out"

if ! .venv/bin/python scripts/agent_round_info.py "$feedback" --output "$out/judge_round.json"; then
  echo "the round in $feedback was not scored; nothing to confirm" >&2
  exit 2
fi

check_hashes() {  # <dir>: its policy.json (and scores.csv) must hash to the Judge round's values
  .venv/bin/python -I - "$1" "$out/judge_round.json" <<'EOF'
import hashlib, json, sys
from pathlib import Path
d, judge = Path(sys.argv[1]), json.load(open(sys.argv[2]))
def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else None
bad = []
if sha(d / "policy.json") != judge["policy_sha256"]:
    bad.append(f"policy.json {sha(d / 'policy.json')} != staged {judge['policy_sha256']}")
if sha(d / "scores.csv") != judge["scores_sha256"]:
    bad.append(f"scores.csv {sha(d / 'scores.csv')} != staged {judge['scores_sha256']}")
extra = sorted(p.name for p in d.iterdir() if p.name not in ("policy.json", "scores.csv"))
if extra:
    bad.append(f"unexpected files: {extra}")
for b in bad:
    print(b, file=sys.stderr)
sys.exit(1 if bad else 0)
EOF
}

if [ -d "$out/candidate" ]; then
  check_hashes "$out/candidate" || { echo "$out/candidate does not match the Judge round" >&2; exit 2; }
else
  check_hashes "$cand" || { echo "$cand does not match what the Judge staged in that round" >&2; exit 2; }
  mkdir -p "$out/candidate"
  cp "$cand/policy.json" "$out/candidate/"
  if [ -f "$cand/scores.csv" ]; then cp "$cand/scores.csv" "$out/candidate/"; fi
  check_hashes "$out/candidate" || { echo "the copy in $out/candidate does not match" >&2; exit 2; }
fi
echo "# candidate $out/candidate matches the Judge round ($(cut -c1-80 <<< "$feedback"))"

for seed in 1001 1002; do
  stage "confirm-$name-seed$seed" scripts/run_candidate.sh "$out/seed$seed" "$out/candidate" "$seed" || exit $?
done
.venv/bin/python scripts/confirm_stats.py "$out"
