"""Run-to-run spread of the uniform baseline, and the gap between LeRobot-trained and aloha-train models.

Every run here trained uniform sampling at the full budget and was scored on the Judge's 500 hidden
seeds. The aloha-train runs span protocols v1-v3 and the order replay (protocol v3 trained on
lerobot-train's own frame order); pooling them assumes they share one success rate, which the data so
far do not contradict but do not establish.

Reports:
- homogeneity of the aloha-train runs (Pearson chi-square) and the dispersion factor phi = chi2 / df,
  the ratio of the observed spread to binomial noise, with its 95% interval;
- LeRobot-trained runs against the aloha-train runs, once with binomial noise only, once with the
  observed run-to-run spread (pooled-variance t test on run rates), and once with phi at its upper bound;
- for one run's earlier checkpoints, how far scores move within a run (descriptive, not independent).

Usage: .venv/bin/python scripts/uniform_runs_spread.py [--output PATH]
"""

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from candidate_stats import ROOT, episode_key, load, uses_hidden_block  # noqa: E402

EPISODES = 500
ALOHA_TRAIN = {
    "v1 seed 1000": "experiments/baseline/seed1000/eval_hidden/eval_result.json",
    "v1 seed 1000, RSI-Harness re-run": "experiments/oracle/judge/eval_result.json",
    "v1 seed 1001": "experiments/baseline/seed1001/eval_hidden/eval_result.json",
    "v1 seed 1002": "experiments/baseline/seed1002/eval_hidden/eval_result.json",
    "v2 seed 1000": "experiments/baseline_v2/seed1000/eval_hidden/eval_result.json",
}
# Every finished protocol v3 seed and order replay counts, so a later run cannot be left out.
ALOHA_TRAIN.update({
    f"v3 {path.parts[-3].replace('seed', 'seed ')}": str(path.relative_to(ROOT))
    for path in sorted((ROOT / "experiments/baseline_v3").glob("seed*/eval_hidden/eval_result.json"))
})
# Every order replay counts (experiments/order_replay, order_replay2, ...): protocol v3 trained on
# lerobot-train's own frame order in normal mode, so a repeat cannot be left out either.
ALOHA_TRAIN.update({
    "order replay " + path.parts[-3].replace("seed", "seed ")
    + ("" if path.parts[-4] == "order_replay" else f", {path.parts[-4].replace('order_replay', 'repeat ')}"):
        str(path.relative_to(ROOT))
    for path in sorted(ROOT.glob("experiments/order_replay*/seed*/eval_hidden/eval_result.json"))
})
# Protocol v3 seeds trained in deterministic cuDNN mode (same schedule.npy as the v3 seeds above).
ALOHA_TRAIN.update({
    f"v3 deterministic mode {path.parts[-3].replace('seed', 'seed ')}": str(path.relative_to(ROOT))
    for path in sorted((ROOT / "experiments/baseline_v3_det").glob("seed*/eval_hidden/eval_result.json"))
})
LEROBOT = {
    "lerobot-train v0.6.1 seed 1000": "experiments/control_lerobot_train/seed1000/eval_hidden/eval_result.json",
    "released checkpoint (LeRobot 3c0a209)": "experiments/reference/eval_hidden/eval_result.json",
}
# Every finished lerobot-train run counts: the unchanged re-run of the control (normal mode) and the
# deterministic-mode seeds. The seed-1000 entries are one seed trained several times, so the t test
# below over-counts independent seeds on the LeRobot side; it stays descriptive.
LEROBOT.update({
    f"lerobot-train v0.6.1 {path.parts[-3].replace('seed', 'seed ')}, re-run": str(path.relative_to(ROOT))
    for path in sorted((ROOT / "experiments/control_lerobot_train_rerun").glob("seed*/eval_hidden/eval_result.json"))
})
LEROBOT.update({
    f"lerobot-train v0.6.1 deterministic mode {path.parts[-3].replace('seed', 'seed ')}": str(path.relative_to(ROOT))
    for path in sorted((ROOT / "experiments/control_lerobot_train_det").glob("seed*/eval_hidden/eval_result.json"))
})
CONTROL_CHECKPOINTS = {
    "step 60000": "experiments/control_lerobot_train/seed1000/checkpoint_evals/step060000/eval_result.json",
    "step 80000": "experiments/control_lerobot_train/seed1000/checkpoint_evals/step080000/eval_result.json",
    "step 100000": "experiments/control_lerobot_train/seed1000/eval_hidden/eval_result.json",
}


def read(runs: dict) -> dict:
    out = {}
    for label, path in runs.items():
        result = load(ROOT / path)
        if not uses_hidden_block(result) or result["n_episodes"] != EPISODES:
            raise SystemExit(f"{path}: not the Judge's {EPISODES}-episode hidden block")
        out[label] = result
    return out


def homogeneity(counts: list[int]) -> dict:
    table = [[k, EPISODES - k] for k in counts]
    chi2, p, dof, _ = stats.chi2_contingency(table, correction=False)
    return {"chi2": chi2, "dof": dof, "p": p, "phi": chi2 / dof,
            "phi95": [chi2 / stats.chi2.ppf(0.975, dof), chi2 / stats.chi2.ppf(0.025, dof)]}


def aloha_train_dispersion() -> dict:
    """Homogeneity of all uniform aloha-train runs, for scripts that look at a subset of them."""
    counts = [r["successes"] for r in read(ALOHA_TRAIN).values()]
    return {"runs": len(counts), "successes": sum(counts), **homogeneity(counts)}


def mcnemar(a: dict, b: dict) -> dict:
    pa = {episode_key(e): bool(e["success"]) for e in a["per_episode"]}
    pb = {episode_key(e): bool(e["success"]) for e in b["per_episode"]}
    if set(pa) != set(pb):
        raise SystemExit("runs were not evaluated on the same seeds")
    a_only = sum(pa[s] and not pb[s] for s in pa)
    b_only = sum(pb[s] and not pa[s] for s in pa)
    both = sum(pa[s] and pb[s] for s in pa)
    disc = a_only + b_only
    return {"both": both, "first_only": a_only, "second_only": b_only,
            "independence_both": sum(pa.values()) * sum(pb.values()) / len(pa),
            "mcnemar_exact_p": stats.binomtest(min(a_only, b_only), disc, 0.5).pvalue if disc else 1.0}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()

    ours, lerobot = read(ALOHA_TRAIN), read(LEROBOT)
    k_ours = [r["successes"] for r in ours.values()]
    k_lerobot = [r["successes"] for r in lerobot.values()]
    v3 = [ours[label]["successes"] for label in ours if label.startswith("v3")]

    report = {"aloha_train": dict(zip(ours, k_ours)), "lerobot": dict(zip(lerobot, k_lerobot))}
    print("aloha-train runs:", ", ".join(f"{label} {k}" for label, k in zip(ours, k_ours)))
    print("LeRobot-trained runs:", ", ".join(f"{label} {k}" for label, k in zip(lerobot, k_lerobot)))

    for name, counts in (("all aloha-train runs", k_ours), ("v3 runs only", v3)):
        h = homogeneity(counts)
        rate = sum(counts) / (EPISODES * len(counts))
        sd = statistics.stdev(k / EPISODES for k in counts)
        binomial = math.sqrt(rate * (1 - rate) / EPISODES)
        h.update({"runs": len(counts), "pooled_rate": rate, "sd_runs": sd, "binomial_sd": binomial})
        report[name] = h
        print(f"{name} ({len(counts)}): pooled {sum(counts)}/{EPISODES * len(counts)} = {rate * 100:.2f}%, "
              f"run s.d. {sd * 100:.2f} pp vs binomial {binomial * 100:.2f} pp; chi2 {h['chi2']:.2f}, df {h['dof']}, "
              f"p = {h['p']:.3f}; phi {h['phi']:.2f} (95% {h['phi95'][0]:.2f}-{h['phi95'][1]:.2f})")

    # LeRobot-trained against aloha-train: three noise models.
    m, n = len(k_ours), len(k_lerobot)
    rate_ours = sum(k_ours) / (EPISODES * m)
    rate_lerobot = sum(k_lerobot) / (EPISODES * n)
    diff = rate_lerobot - rate_ours
    p_all = (sum(k_ours) + sum(k_lerobot)) / (EPISODES * (m + n))
    z_binomial = diff / math.sqrt(p_all * (1 - p_all) * (1 / (EPISODES * m) + 1 / (EPISODES * n)))
    t = stats.ttest_ind([k / EPISODES for k in k_lerobot], [k / EPISODES for k in k_ours], equal_var=True)
    phi_upper = report["all aloha-train runs"]["phi95"][1]
    sd_upper = math.sqrt(phi_upper * rate_ours * (1 - rate_ours) / EPISODES)
    z_upper = diff / (sd_upper * math.sqrt(1 / m + 1 / n))
    report["lerobot_vs_aloha_train"] = {
        "diff_pp": diff * 100,
        "binomial": {"z": z_binomial, "p": math.erfc(abs(z_binomial) / math.sqrt(2))},
        "observed_spread_t": {"t": float(t.statistic), "df": m + n - 2, "p": float(t.pvalue)},
        "phi_upper": {"phi": phi_upper, "z": z_upper, "p": math.erfc(abs(z_upper) / math.sqrt(2))},
    }
    print(f"LeRobot-trained {sum(k_lerobot)}/{EPISODES * n} = {rate_lerobot * 100:.2f}% vs aloha-train "
          f"{rate_ours * 100:.2f}%: {diff * 100:+.2f} pp")
    print(f"  binomial noise only: z = {z_binomial:.2f}, p = {math.erfc(abs(z_binomial) / math.sqrt(2)):.2g}")
    print(f"  observed run spread (t test on run rates, df {m + n - 2}): t = {t.statistic:.2f}, p = {t.pvalue:.2g}")
    print(f"  phi at its upper 95% bound ({phi_upper:.2f}): z = {z_upper:.2f}, "
          f"p = {math.erfc(abs(z_upper) / math.sqrt(2)):.2g}")

    checkpoints = read(CONTROL_CHECKPOINTS)
    labels = list(checkpoints)
    report["control_checkpoints"] = {label: checkpoints[label]["successes"] for label in labels}
    print("control checkpoints (one run, not independent):",
          ", ".join(f"{label} {checkpoints[label]['successes']}" for label in labels))
    pairs = {}
    for first, second in zip(labels, labels[1:]):
        pair = mcnemar(checkpoints[second], checkpoints[first])
        pairs[f"{second} vs {first}"] = pair
        print(f"  {second} vs {first}: both {pair['both']} (independence {pair['independence_both']:.1f}), "
              f"only {second} {pair['first_only']}, only {first} {pair['second_only']}, "
              f"McNemar p = {pair['mcnemar_exact_p']:.3g}")
    report["control_checkpoint_pairs"] = pairs
    if args.output:
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
