#!/usr/bin/env python3
"""Confirmation of an agent's best candidate by host retraining (docs/plan.md, step 5; the rule was
committed in 9d7a839, before the agent run started).

Reads <confirm_dir>/judge_round.json (scripts/agent_round_info.py: the Judge's own record of the round
that selected the candidate), the two host retrains <confirm_dir>/seed1001 and <confirm_dir>/seed1002
(scripts/run_candidate.sh) and the final baseline's summary.json. The Judge round itself, at training
seed 1000, was selected as the best of several rounds and stays out of the test.

Rule: k = the two retrains' hidden successes summed; b = the baseline's hidden successes at training
seeds 1001 and 1002 summed (64 + 57 = 121 of 1000). Each arm is the mean of two runs, so with the
single-run s.d. s estimated by the baseline (2.36 pp) the difference of the two means has standard
error s; need = ceil((b / 1000 + 1.96 s) x 1000) = 168. Verdict: k >= need "confirmed"; b < k < need
"positive, not confirmed"; k <= b "not reproduced".

Checks before any verdict (exit 2): the Judge round was scored; both retrains used the full budget, the
Judge's hidden seed block and the baseline's protocol; they trained at seeds 1001 and 1002; their
schedules and candidate hashes equal each other and the Judge round's.

Usage: .venv/bin/python scripts/confirm_stats.py <confirm_dir> [--baseline-dir DIR] [--expected-need N] [--output PATH]
Exit 0 with a verdict; 2 when a check fails; 3 when the computed requirement differs from
--expected-need (default 168, the value written down before the run).
Writes <confirm_dir>/confirm.json by default; the report never lists seeds of the hidden block.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from candidate_stats import ROOT, Z, episode_key, load, protocol_of, uses_hidden_block, wilson  # noqa: E402

SEEDS = (1001, 1002)
VERDICTS = ("confirmed", "positive, not confirmed", "not reproduced")


def paired(run: dict, base: dict) -> dict:
    pr = {episode_key(e): bool(e["success"]) for e in run["per_episode"]}
    pb = {episode_key(e): bool(e["success"]) for e in base["per_episode"]}
    if set(pr) != set(pb):
        raise SystemExit("the retrain and the baseline did not evaluate the same episodes")
    only_run = sum(1 for s in pr if pr[s] and not pb[s])
    only_base = sum(1 for s in pr if pb[s] and not pr[s])
    disc = only_run + only_base
    return {"both": sum(1 for s in pr if pr[s] and pb[s]), "candidate_only": only_run, "baseline_only": only_base,
            "mcnemar_exact_p": stats.binomtest(min(only_run, only_base), disc, 0.5).pvalue if disc else 1.0}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("confirm_dir", type=Path)
    ap.add_argument("--baseline-dir", type=Path, default=ROOT / "experiments/baseline_v3_det")
    ap.add_argument("--expected-need", type=int, default=168)
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()
    cdir, bdir = args.confirm_dir, args.baseline_dir

    problems: list[str] = []
    judge = load(cdir / "judge_round.json")
    if judge.get("status") != "scored":
        problems.append(f"the Judge round was not scored: {judge.get('status')}")
    summary = load(bdir / "summary.json")
    base_protocol = summary.get("protocol")
    base_seeds = {int(r["seed"]): int(r["hidden_successes"]) for r in summary["seeds"]}

    runs = {}
    for seed in SEEDS:
        d = cdir / f"seed{seed}"
        train = load(d / "train/train_info.json")
        sched = load(d / "schedule/schedule_summary.json")
        hidden = load(d / "eval_hidden/eval_result.json")
        runs[seed] = {"train": train, "sched": sched, "hidden": hidden}
        if int(train["seed"]) != seed:
            problems.append(f"seed{seed}: trained at seed {train['seed']}")
        if train["steps"] != 100_000 or train["exposures"] != 800_000:
            problems.append(f"seed{seed}: not the full budget ({train['steps']} steps, {train['exposures']} exposures)")
        if not uses_hidden_block(hidden) or len(hidden.get("per_episode", [])) != hidden["n_episodes"]:
            problems.append(f"seed{seed}: the hidden evaluation did not use the Judge's seed block")
        if protocol_of(train, hidden) != base_protocol:
            problems.append(f"seed{seed}: protocol {protocol_of(train, hidden)} differs from the baseline's {base_protocol}")
        for key in ("schedule_sha256", "policy_sha256", "scores_sha256"):
            if sched.get(key) != judge.get(key):
                problems.append(f"seed{seed}: {key} {str(sched.get(key))[:12]} differs from the Judge round's {str(judge.get(key))[:12]}")
        if seed not in base_seeds:
            problems.append(f"the baseline has no seed {seed}")
    if problems:
        for p in problems:
            print(p)
        return 2

    k = sum(int(runs[s]["hidden"]["successes"]) for s in SEEDS)
    n = sum(int(runs[s]["hidden"]["n_episodes"]) for s in SEEDS)
    b = sum(base_seeds[s] for s in SEEDS)
    sd = float(summary["single_run_sd_estimate"])
    margin = Z * sd
    need = math.ceil((b / n + margin) * n - 1e-9)
    if need != args.expected_need:
        print(f"the computed requirement {need}/{n} differs from the value written down before the run "
              f"({args.expected_need}); the baseline summary changed? No verdict.")
        return 3
    verdict = VERDICTS[0] if k >= need else (VERDICTS[1] if k > b else VERDICTS[2])
    out = {
        "confirm_dir": str(cdir),
        "baseline_dir": str(bdir),
        "rule": "docs/plan.md step 5 (committed in 9d7a839 before the agent run)",
        "judge_round": {key: judge.get(key) for key in ("feedback_log", "policy_name", "policy_sha256", "scores_sha256",
                                                          "schedule_sha256", "successes", "n_episodes", "reward")},
        "retrains": {str(s): {"hidden_successes": int(runs[s]["hidden"]["successes"]),
                              "wilson95": wilson(int(runs[s]["hidden"]["successes"]), int(runs[s]["hidden"]["n_episodes"])),
                              "baseline_same_seed": base_seeds[s],
                              "paired_vs_baseline_same_seed": paired(runs[s]["hidden"], load(bdir / f"seed{s}/eval_hidden/eval_result.json")),
                              "train_s": runs[s]["train"].get("train_s")} for s in SEEDS},
        "candidate_successes": k,
        "baseline_successes": b,
        "episodes_per_arm": n,
        "single_run_sd": sd,
        "margin": margin,
        "required_successes": need,
        "difference_pp": (k - b) / n * 100,
        "verdict": verdict,
    }
    dest = args.output or cdir / "confirm.json"
    dest.write_text(json.dumps(out, indent=2) + "\n")
    print(f"{judge.get('policy_name')}: retrains at seeds 1001/1002 {k}/{n} against the baseline's {b}/{n} "
          f"({out['difference_pp']:+.2f} pp); confirmation needs >= {need}/{n}; verdict: {verdict}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
