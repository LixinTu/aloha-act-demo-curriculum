#!/usr/bin/env python3
"""Summarise the uniform-sampling baseline over training seeds.

Reads <baseline dir>/seed*/{eval_hidden/eval_result.json, eval_proxy/eval_result.json,
train/train_info.json} and writes <baseline dir>/summary.json (or --output) plus a markdown table
on stdout. The directory defaults to experiments/baseline (protocol v1); the final baseline is
summarised with --baseline-dir. All numbers come from those files; nothing is estimated except the
derived statistics documented below. Every seed must be a full-budget run evaluated on the Judge's
500 hidden seeds, and all seeds must share one protocol (version, cuDNN mode of training, cuDNN mode
of the hidden evaluation). The summary records that protocol, so scripts/candidate_stats.py can
refuse a candidate trained or evaluated under another.

Usage: .venv/bin/python scripts/baseline_stats.py [--baseline-dir DIR] [--output PATH]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from candidate_stats import BASELINE, ROOT, protocol_of, uses_hidden_block  # noqa: E402
from scipy import stats  # noqa: E402

Z = 1.959963984540054


def wilson(k: int, n: int) -> tuple[float, float]:
    p = k / n
    denom = 1 + Z * Z / n
    centre = (p + Z * Z / (2 * n)) / denom
    half = Z * math.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n)) / denom
    return centre - half, centre + half


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline-dir", type=Path, default=BASELINE)
    ap.add_argument("--output", type=Path, help="default: <baseline dir>/summary.json")
    args = ap.parse_args()
    base = args.baseline_dir.resolve()

    rows, protocols = [], {}
    for d in sorted(base.glob("seed*")):
        hidden = json.loads((d / "eval_hidden" / "eval_result.json").read_text())
        proxy = json.loads((d / "eval_proxy" / "eval_result.json").read_text())
        train = json.loads((d / "train" / "train_info.json").read_text())
        if not uses_hidden_block(hidden):
            raise SystemExit(f"{d.name}: the hidden evaluation did not use the Judge's seed block")
        protocols[d.name] = protocol_of(train, hidden)
        rows.append({
            "seed": int(train["seed"]),
            "protocol_seed": bool(train["official_budget"]),
            "steps": int(train["steps"]),
            "exposures": int(train["exposures"]),
            "hidden_successes": int(hidden["successes"]),
            "hidden_episodes": int(hidden["n_episodes"]),
            "hidden_rate": float(hidden["success_rate"]),
            "hidden_wilson95": wilson(int(hidden["successes"]), int(hidden["n_episodes"])),
            "hidden_max_reward_histogram": hidden["max_reward_histogram"],
            "proxy_successes": int(proxy["successes"]),
            "proxy_episodes": int(proxy["n_episodes"]),
            "train_s": float(train["train_s"]),
            "steps_per_s": float(train["steps_per_s"]),
            "hidden_eval_s": float(hidden["eval_s"]),
            "final_l1_last_1000": float(train["final_l1_mean_last_1000"]),
        })
    if len(rows) < 2:
        raise SystemExit(f"need at least two seeds in {base}")
    for r in rows:
        if r["steps"] != 100_000 or r["exposures"] != 800_000 or r["hidden_episodes"] != 500:
            raise SystemExit(f"seed {r['seed']} is not a full-budget 500-episode run")
    distinct = {json.dumps(p, sort_keys=True) for p in protocols.values()}
    if len(distinct) != 1:
        raise SystemExit(f"the seeds mix protocols: {protocols}")
    protocol = next(iter(protocols.values()))

    rates = [r["hidden_rate"] for r in rows]
    m = len(rates)
    mean = sum(rates) / m
    sd = math.sqrt(sum((x - mean) ** 2 for x in rates) / (m - 1))
    k = sum(r["hidden_successes"] for r in rows)
    n = sum(r["hidden_episodes"] for r in rows)
    pooled = k / n
    binom_se_single = math.sqrt(pooled * (1 - pooled) / 500)
    # Homogeneity of the seeds' success rates (2 x m contingency table, Pearson chi-square, m - 1 df).
    chi2 = 0.0
    for r in rows:
        e_s = r["hidden_episodes"] * pooled
        e_f = r["hidden_episodes"] * (1 - pooled)
        chi2 += (r["hidden_successes"] - e_s) ** 2 / e_s + ((r["hidden_episodes"] - r["hidden_successes"]) - e_f) ** 2 / e_f
    df = m - 1
    p_hom = float(stats.chi2.sf(chi2, df))
    seed_var = max(0.0, sd * sd - binom_se_single ** 2)
    single_run_sd = math.sqrt(seed_var + binom_se_single ** 2)
    pooled_se = math.sqrt(pooled * (1 - pooled) / n)
    diff_se = math.sqrt(single_run_sd ** 2 + pooled_se ** 2)
    threshold = Z * diff_se
    summary = {
        "candidate": "uniform (default_policy.json)",
        "baseline_dir": str(base.relative_to(ROOT)),
        "protocol": protocol,
        "seeds": rows,
        "mean_rate": mean,
        "sd_rate_across_seeds": sd,
        "pooled_successes": k,
        "pooled_episodes": n,
        "pooled_rate": pooled,
        "pooled_wilson95": wilson(k, n),
        "binomial_se_single_500": binom_se_single,
        "homogeneity_chi2": chi2,
        "homogeneity_df": df,
        "homogeneity_p": p_hom,
        "training_seed_variance_estimate": seed_var,
        "single_run_sd_estimate": single_run_sd,
        "two_sided_95_threshold_vs_pooled_baseline": threshold,
        "min_successes_out_of_500_to_beat_baseline": math.ceil((pooled + threshold) * 500 + 1e-9),
        "reported_not_reproduced": {"source": "lerobot/act_aloha_sim_insertion_human model card", "success_rate": 0.206, "note": "different lerobot version and protocol; the repo's eval_info.json is the transfer-cube file"},
    }
    dest = args.output or base / "summary.json"
    dest.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"baseline {summary['baseline_dir']}, protocol v{protocol['protocol_version']}, "
          f"cudnn_deterministic {protocol['cudnn_deterministic']}, deterministic evaluation "
          f"{protocol.get('eval_cudnn_deterministic', False)}; written to {dest}")
    print("| seed | hidden successes / 500 | rate | Wilson 95% CI | proxy / 100 | train h | eval min |")
    print("| --- | --- | --- | --- | --- | --- | --- |")
    for r in rows:
        lo, hi = r["hidden_wilson95"]
        tag = " (Judge seed)" if r["protocol_seed"] else ""
        print(f"| {r['seed']}{tag} | {r['hidden_successes']} | {r['hidden_rate']*100:.1f}% | {lo*100:.1f}-{hi*100:.1f}% | {r['proxy_successes']} | {r['train_s']/3600:.2f} | {r['hidden_eval_s']/60:.0f} |")
    lo, hi = summary["pooled_wilson95"]
    print(f"\npooled {k}/{n} = {pooled*100:.2f}% (Wilson 95% {lo*100:.1f}-{hi*100:.1f}%); mean of seeds {mean*100:.2f}% +- {sd*100:.2f} pp (sd)")
    print(f"single-run binomial SE {binom_se_single*100:.2f} pp; homogeneity chi2 {chi2:.3f}, df {df}, p = {p_hom:.3f}; training-seed variance estimate {seed_var:.2e}")
    print(f"two-sided 95% threshold vs pooled baseline: {threshold*100:.2f} pp -> a single Judge run needs >= {summary['min_successes_out_of_500_to_beat_baseline']}/500")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
