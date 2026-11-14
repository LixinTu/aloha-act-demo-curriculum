"""Score one full-budget candidate run against the uniform baseline.

Reads <run_dir>/{schedule/schedule_summary.json, train/train_info.json,
eval_hidden/eval_result.json, eval_proxy/eval_result.json} as written by scripts/run_candidate.sh,
plus <baseline dir>/summary.json (scripts/baseline_stats.py) and the baseline's seed-1000 run. The
baseline directory defaults to experiments/baseline (protocol v1); the final protocol's baseline is
experiments/baseline_v3_det (deterministic mode, seeds 1000/1001/1002). The significance threshold is
the one stored in the baseline summary (v1: >= 88/500; final: >= 92/500); it is not recomputed here. A candidate trained
under another protocol (version or cuDNN mode) than the baseline is refused unless
--allow-other-protocol is given, and then the output says so. score_hidden() is shared with
scripts/reference_stats.py and scripts/paired_runs.py.

Usage: .venv/bin/python scripts/candidate_stats.py <run_dir> [--baseline-dir DIR]
           [--allow-other-protocol] [--output PATH]
Writes <run_dir>/stats.json by default and prints a one-line summary.
"""

import argparse
import json
import math
import sys
from pathlib import Path

from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "experiments/baseline"
HIDDEN_SEEDS = ROOT / "task/aloha-act-demo-curriculum/tests/hidden_seeds.json"
# The public copy of this repository (scripts/make_public_copy.py) has no hidden_seeds.json: its
# hidden-block results carry start_seed WITHHELD and each episode's index in the block in place of
# its seed.
WITHHELD = "withheld"
LEVELS = [str(k) for k in range(5)]
Z = 1.959963984540054


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def wilson(k: int, n: int) -> list[float]:
    p = k / n
    denom = 1 + Z * Z / n
    centre = (p + Z * Z / (2 * n)) / denom
    half = Z * math.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n)) / denom
    return [centre - half, centre + half]


def uses_hidden_block(hidden: dict) -> bool:
    if not HIDDEN_SEEDS.exists():
        # Public copy: only results marked as the withheld hidden block qualify. With the seeds file
        # present (this repository) a WITHHELD result never matches the real start seed.
        return hidden.get("start_seed") == WITHHELD
    seeds = load(HIDDEN_SEEDS)
    return hidden["n_episodes"] == seeds["n_episodes"] and hidden["start_seed"] == seeds["start_seed"]


def episode_key(e: dict) -> int:
    """Key that pairs one episode across two evaluations of the same seed block: its seed, or in the
    public copy its index in the withheld hidden block. A record with neither raises KeyError."""
    return int(e["seed"]) if "seed" in e else int(e["episode"])


def protocol_of(train: dict, hidden: dict) -> dict:
    """What a baseline and a candidate must share: the protocol version (protocol v1 runs carry
    none), the cuDNN mode of training, and the cuDNN mode of the hidden evaluation. The evaluation
    key is present only for a deterministic evaluation, so summaries written before the evaluator
    could run in that mode (always the benchmark mode of lerobot-eval) keep comparing equal."""
    protocol = {"protocol_version": int(train.get("protocol_version", 1)),
                "cudnn_deterministic": bool(train["cudnn_deterministic"])}
    if hidden.get("cudnn_deterministic", False):
        protocol["eval_cudnn_deterministic"] = True
    return protocol


def score_hidden(hidden: dict, pair_with_host: bool, baseline: Path = BASELINE) -> dict:
    """Hidden-block result against the pooled uniform baseline and, if asked, paired per episode
    against the baseline's uniform run at training seed 1000. The caller checks uses_hidden_block()."""
    base = load(baseline / "summary.json")
    host = load(baseline / "seed1000/eval_hidden/eval_result.json")

    k, n = hidden["successes"], hidden["n_episodes"]
    rate = k / n
    pooled_k, pooled_n = base["pooled_successes"], base["pooled_episodes"]
    pooled = pooled_k / pooled_n
    se = math.sqrt(base["single_run_sd_estimate"] ** 2 + pooled * (1 - pooled) / pooled_n)
    z = (rate - pooled) / se
    p_two_sided = math.erfc(abs(z) / math.sqrt(2))
    need = base["min_successes_out_of_500_to_beat_baseline"]

    base_levels = [sum(s["hidden_max_reward_histogram"][lv] for s in base["seeds"]) for lv in LEVELS]
    cand_levels = [hidden["max_reward_histogram"][lv] for lv in LEVELS]
    chi2, chi2_p, dof, _ = stats.chi2_contingency([base_levels, cand_levels])

    paired = None
    if pair_with_host:
        ph = {episode_key(e): e["success"] for e in host["per_episode"]}
        pc = {episode_key(e): e["success"] for e in hidden["per_episode"]}
        if set(ph) == set(pc):
            host_only = sum(ph[s] and not pc[s] for s in ph)
            cand_only = sum(pc[s] and not ph[s] for s in ph)
            both = sum(ph[s] and pc[s] for s in ph)
            disc = host_only + cand_only
            paired = {
                "against": f"uniform run {(baseline / 'seed1000').relative_to(ROOT)}, same hidden seeds",
                "both": both, "baseline_only": host_only, "candidate_only": cand_only,
                "neither": len(ph) - both - disc,
                "mcnemar_exact_p": stats.binomtest(min(host_only, cand_only), disc, 0.5).pvalue if disc else 1.0,
            }

    return {
        "hidden": {
            "successes": k, "n_episodes": n, "rate": rate, "wilson95": wilson(k, n),
            "max_reward_histogram": cand_levels, "avg_max_reward": hidden.get("avg_max_reward"),
            "eval_s": hidden["eval_s"], "gl_renderer": hidden.get("gl_renderer"),
        },
        "vs_pooled_baseline": {
            "baseline": f"{pooled_k}/{pooled_n}",
            "diff_pp": (rate - pooled) * 100,
            "z": z,
            "p_two_sided": p_two_sided,
            "required_successes": need,
            "exceeds_threshold": k >= need,
            "levels_baseline_pooled": base_levels,
            "levels_chi2": chi2, "levels_p": chi2_p, "levels_dof": dof,
        },
        "paired_vs_host_seed1000": paired,
    }


def summary_line(label: str, scored: dict, proxy: dict, baseline: Path = BASELINE) -> str:
    base = load(baseline / "summary.json")
    pooled = base["pooled_successes"] / base["pooled_episodes"]
    h, v = scored["hidden"], scored["vs_pooled_baseline"]
    verdict = "exceeds" if v["exceeds_threshold"] else "does not exceed"
    return (f"{label}: {h['successes']}/{h['n_episodes']} = {h['rate']*100:.1f}% (Wilson {h['wilson95'][0]*100:.1f}-"
            f"{h['wilson95'][1]*100:.1f}%), {v['diff_pp']:+.2f} pp vs pooled "
            f"{pooled*100:.2f}%, z = {v['z']:.2f}, p = {v['p_two_sided']:.3f}; {verdict} the >= "
            f"{v['required_successes']}/500 threshold; levels {h['max_reward_histogram']} vs pooled "
            f"{v['levels_baseline_pooled']} (p = {v['levels_p']:.3f}); proxy {proxy['successes']}/{proxy['n_episodes']}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--baseline-dir", type=Path, default=BASELINE)
    ap.add_argument("--allow-other-protocol", action="store_true",
                    help="score against a baseline of another protocol (descriptive only; recorded in the output)")
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()
    run = args.run_dir.resolve()
    baseline = args.baseline_dir.resolve()

    sched = load(run / "schedule/schedule_summary.json")
    train = load(run / "train/train_info.json")
    hidden = load(run / "eval_hidden/eval_result.json")
    proxy = load(run / "eval_proxy/eval_result.json")
    base = load(baseline / "summary.json")

    problems = []
    if train["steps"] != 100_000 or train["exposures"] != 800_000:
        problems.append(f"not the full budget: steps {train['steps']}, exposures {train['exposures']}")
    if not uses_hidden_block(hidden):
        problems.append("hidden evaluation did not use the Judge's seed block")
    candidate_protocol = protocol_of(train, hidden)
    baseline_protocol = base.get("protocol")
    if baseline_protocol is None:
        problems.append(f"{baseline / 'summary.json'} records no protocol; regenerate it with "
                        f"scripts/baseline_stats.py --baseline-dir {baseline.relative_to(ROOT)}")
    elif candidate_protocol != baseline_protocol and not args.allow_other_protocol:
        problems.append(f"candidate protocol {candidate_protocol} differs from the baseline's {baseline_protocol}; "
                        "pass the matching --baseline-dir (or --allow-other-protocol for a descriptive comparison)")
    if problems:
        for p in problems:
            print(p)
        return 1

    scored = score_hidden(hidden, pair_with_host=train["seed"] == 1000, baseline=baseline)
    out = {
        "run_dir": str(run.relative_to(ROOT)),
        "baseline_dir": str(baseline.relative_to(ROOT)),
        "candidate_protocol": candidate_protocol,
        "baseline_protocol": baseline_protocol,
        "protocol_matches": candidate_protocol == baseline_protocol,
        "policy_name": sched.get("policy_name"),
        "schedule_sha256": sched["schedule_sha256"],
        "schedule": {key: sched.get(key) for key in (
            "distinct_frames_fraction", "entropy_mean_nats", "entropy_uniform_nats",
            "chunk_pad_fraction_padded", "stage_blocks", "episodes_used")},
        "train": {key: train.get(key) for key in (
            "seed", "steps", "exposures", "train_s", "steps_per_s", "final_l1_mean_last_1000",
            "final_kl_mean_last_1000", "peak_gpu_mem_gb")},
        "hidden": scored["hidden"],
        "proxy": {"successes": proxy["successes"], "n_episodes": proxy["n_episodes"]},
        "vs_pooled_baseline": scored["vs_pooled_baseline"],
        "paired_vs_host_seed1000": scored["paired_vs_host_seed1000"],
    }
    dest = args.output or run / "stats.json"
    dest.write_text(json.dumps(out, indent=2) + "\n")
    print(summary_line(out["run_dir"], scored, proxy, baseline=baseline))
    if not out["protocol_matches"]:
        print(f"note: baseline {out['baseline_dir']} is protocol {baseline_protocol}, the candidate "
              f"{candidate_protocol}; descriptive only")
    return 0


if __name__ == "__main__":
    sys.exit(main())
