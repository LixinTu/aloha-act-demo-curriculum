#!/usr/bin/env python3
"""Summarise one Judge round from its feedback log (the JSON lines tests/evaluate.py prints).

The log of round <id> of an RSI-Harness run is harness/logs/runs/<run>/<task>/feedback/<id>.log. The
summary names the candidate the Judge staged (policy and scores hashes), the schedule it compiled and
the score, so a later step can check that the files it uses are the ones that were scored.
Hidden seeds never appear in the log, so the summary carries none.

Usage: .venv/bin/python scripts/agent_round_info.py <feedback_log> [--output PATH]
Exit 0: the round was scored. Exit 2: it was not (invalid candidate, run error, infrastructure error,
or the log ends early); the summary still records what is known.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def summarise(path: Path) -> dict:
    events: dict[str, dict] = {}
    with path.open(errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(ev, dict) and "event" in ev:
                events.setdefault(ev["event"], ev)
    staged = events.get("candidate_staged", {})
    compiled = events.get("schedule_compiled", {})
    trained = events.get("train_complete", {})
    done = events.get("complete")
    out = {
        "feedback_log": str(path),
        "events": sorted(events),
        "policy_sha256": staged.get("policy_sha256"),
        "scores_sha256": staged.get("scores_sha256"),
        "files": staged.get("files"),
        "policy_name": compiled.get("policy_name"),
        "schedule_sha256": compiled.get("schedule_sha256"),
        "train_seed": trained.get("seed"),
        "train_s": trained.get("train_s"),
    }
    failure = next((events[e] for e in ("candidate_invalid", "run_error", "infrastructure_error") if e in events), None)
    if done is not None and failure is None:
        out.update({
            "status": "scored",
            "reward": done.get("reward"),
            "successes": done.get("successes"),
            "n_episodes": done.get("n_episodes"),
            "max_reward_histogram": done.get("max_reward_histogram"),
            "eval_s": done.get("eval_s"),
        })
        if done.get("policy_sha256") != out["policy_sha256"] or done.get("schedule_sha256") != out["schedule_sha256"]:
            out["status"] = "inconsistent"
            out["detail"] = "the complete event names another policy or schedule than the staged and compiled ones"
    elif failure is not None:
        out["status"] = failure["event"]
        out["stage"] = failure.get("stage")
        out["detail"] = failure.get("detail") or failure.get("log_tail", "")[-400:]
    else:
        out["status"] = "incomplete"
        out["detail"] = "the log has no complete event"
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("feedback_log", type=Path)
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()
    if not args.feedback_log.is_file():
        print(f"no feedback log at {args.feedback_log}")
        return 2
    out = summarise(args.feedback_log)
    text = json.dumps(out, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    if out["status"] == "scored":
        print(f"round scored: {out['successes']}/{out['n_episodes']} (reward {out['reward']}), policy "
              f"{out['policy_name']} {str(out['policy_sha256'])[:12]}, schedule {str(out['schedule_sha256'])[:12]}")
        return 0
    print(f"round not scored: {out['status']} ({out.get('detail', '')})")
    return 2


if __name__ == "__main__":
    sys.exit(main())
