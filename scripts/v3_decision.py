"""Thresholds and verdict for the protocol v3 uniform baseline, pooled over training seeds.

docs/plan.md step 1b sends a v3 seed-1000 score of 77-87 to more seeds; this fixes what the pooled
score must reach, before those seeds run. Two comparisons on the Judge's hidden block:
- above v1: the test behind the >= 88/500 candidate threshold (scripts/baseline_stats.py), with the
  single-run variance divided by the number of seeds;
- below the control: two-proportion z test against lerobot-train's seed-1000 score.
With two or more seeds it also reports their spread, and the test against the control with the observed
spread in place of binomial noise. That is a sensitivity check; the rule uses the thresholds. It also
prints the spread of all uniform aloha-train runs, because a pair of seeds says little on its own.
Only runs that finished both evaluations count; any other seed directory is reported and skipped.

Usage: .venv/bin/python scripts/v3_decision.py [--output PATH]
"""

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from candidate_stats import BASELINE, ROOT, Z, load, uses_hidden_block, wilson  # noqa: E402

V3 = ROOT / "experiments/baseline_v3"
CONTROL = ROOT / "experiments/control_lerobot_train/seed1000"
EPISODES = 500


def min_above_v1(seeds: int) -> int:
    """Smallest pooled success count significantly above the v1 pooled baseline."""
    base = load(BASELINE / "summary.json")
    pooled = base["pooled_successes"] / base["pooled_episodes"]
    pooled_var = pooled * (1 - pooled) / base["pooled_episodes"]
    se = math.sqrt(base["single_run_sd_estimate"] ** 2 / seeds + pooled_var)
    return math.ceil((pooled + Z * se) * EPISODES * seeds + 1e-9)


def max_below_control(seeds: int, control: int) -> int:
    """Largest pooled success count significantly below the control."""
    n = EPISODES * seeds
    for s in range(control * seeds, -1, -1):
        p = (s + control) / (n + EPISODES)
        z = (control / EPISODES - s / n) / math.sqrt(p * (1 - p) * (1 / n + 1 / EPISODES))
        if z >= Z:
            return s
    raise ValueError("no success count is significantly below the control")


def finished_runs() -> list[dict]:
    runs, sha = [], None
    for d in sorted(V3.glob("seed*")):
        if not ((d / "eval_hidden/eval_result.json").exists() and (d / "eval_proxy/eval_result.json").exists()):
            print(f"{d.relative_to(ROOT)}: not finished, not counted")
            continue
        train = load(d / "train.json")
        hidden = load(d / "eval_hidden/eval_result.json")
        proxy = load(d / "eval_proxy/eval_result.json")
        schedule = load(d / "schedule/schedule_summary.json")["schedule_sha256"]
        sha = sha or schedule
        problems = []
        if train["protocol_version"] != 3 or train["steps"] != 100_000 or train["exposures"] != 800_000:
            problems.append("not a full-budget protocol v3 run")
        if not uses_hidden_block(hidden):
            problems.append("hidden evaluation did not use the Judge's seed block")
        if schedule != sha:
            problems.append(f"schedule {schedule} differs from {sha}")
        if problems:
            raise SystemExit(f"{d.relative_to(ROOT)}: " + "; ".join(problems))
        runs.append({"seed": train["seed"], "hidden_successes": hidden["successes"],
                     "proxy_successes": proxy["successes"],
                     "levels": [hidden["max_reward_histogram"][str(k)] for k in range(5)]})
    return runs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()

    base = load(BASELINE / "summary.json")
    if min_above_v1(1) != base["min_successes_out_of_500_to_beat_baseline"]:
        raise SystemExit("min_above_v1(1) does not reproduce the threshold in the baseline summary")
    control = load(CONTROL / "eval_hidden/eval_result.json")
    if not uses_hidden_block(control):
        raise SystemExit("the control's hidden evaluation did not use the Judge's seed block")

    thresholds = {}
    for m in (1, 2, 3):
        thresholds[m] = {"above_v1_at_least": min_above_v1(m),
                         "below_control_at_most": max_below_control(m, control["successes"])}
        print(f"{m} seed(s), out of {EPISODES * m}: above v1 >= {thresholds[m]['above_v1_at_least']}, "
              f"below control <= {thresholds[m]['below_control_at_most']}")

    runs = finished_runs()
    m = len(runs)
    total = sum(r["hidden_successes"] for r in runs)
    verdict = None
    spread = None
    if m:
        t = thresholds[m]
        verdict = ("above_v1" if total >= t["above_v1_at_least"] else
                   "below_control" if total <= t["below_control_at_most"] else "neither")
        lo, hi = wilson(total, EPISODES * m)
        print(f"seeds {[r['seed'] for r in runs]}, hidden {[r['hidden_successes'] for r in runs]}: "
              f"pooled {total}/{EPISODES * m} = {total / (EPISODES * m) * 100:.2f}% "
              f"(Wilson {lo * 100:.1f}-{hi * 100:.1f}%); verdict {verdict}")
    if m >= 2:
        k_control = control["successes"]
        n = EPISODES * m
        rate, control_rate = total / n, k_control / EPISODES
        p = (total + k_control) / (n + EPISODES)
        z_binomial = (control_rate - rate) / math.sqrt(p * (1 - p) * (1 / n + 1 / EPISODES))
        sd = statistics.stdev(r["hidden_successes"] / EPISODES for r in runs)
        binomial = math.sqrt(rate * (1 - rate) / EPISODES)
        z_observed = (control_rate - rate) / (sd * math.sqrt(1 / m + 1))
        chi2, chi2_p, dof, _ = stats.chi2_contingency(
            [[r["hidden_successes"], EPISODES - r["hidden_successes"]] for r in runs], correction=False)
        spread = {"sd_across_seeds": sd, "binomial_sd_single_run": binomial, "homogeneity_chi2": chi2,
                  "homogeneity_dof": dof, "homogeneity_p": chi2_p,
                  "control_minus_pooled_pp": (control_rate - rate) * 100,
                  "z_vs_control_binomial": z_binomial, "p_vs_control_binomial": math.erfc(z_binomial / math.sqrt(2)),
                  "z_vs_control_observed_spread": z_observed,
                  "p_vs_control_observed_spread": math.erfc(z_observed / math.sqrt(2))}
        print(f"seed spread: s.d. {sd * 100:.2f} pp (binomial {binomial * 100:.2f} pp), homogeneity chi2 {chi2:.2f}, "
              f"p = {chi2_p:.3f}")
        # Context, not part of the rule: a pair of seeds says little about spread on its own.
        from uniform_runs_spread import aloha_train_dispersion
        everything = aloha_train_dispersion()
        spread["all_uniform_aloha_train_runs"] = everything
        print(f"all {everything['runs']} uniform aloha-train runs: homogeneity p = {everything['p']:.3f}, dispersion "
              f"{everything['phi']:.2f} (95% {everything['phi95'][0]:.2f}-{everything['phi95'][1]:.2f}); "
              f"see scripts/uniform_runs_spread.py")
        print(f"control minus pooled: {spread['control_minus_pooled_pp']:.2f} pp; binomial noise z = {z_binomial:.2f}, "
              f"p = {spread['p_vs_control_binomial']:.4f}; observed spread z = {z_observed:.2f}, "
              f"p = {spread['p_vs_control_observed_spread']:.3f}")
    if args.output:
        args.output.write_text(json.dumps({
            "control_successes": control["successes"],
            "v1_pooled": f"{base['pooled_successes']}/{base['pooled_episodes']}",
            "thresholds_by_seed_count": thresholds,
            "runs": runs,
            "pooled_successes": total,
            "pooled_episodes": EPISODES * m,
            "verdict": verdict,
            "spread": spread,
        }, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
