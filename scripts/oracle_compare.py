"""Compare an RSI-Harness oracle Judge run with the host baseline run at seed 1000.

Both runs use the uniform default policy, training seed 1000 and the same 500 hidden seeds, so
the comparison is paired per episode. Reads only evidence files and writes the comparison as JSON.
Besides the paired outcome it reports whether the two trainings logged identical losses (in
deterministic mode an identical software stack gives bit-identical training) and whether every
episode has the same outcome, which separates a difference in training from one in evaluation.

Usage: .venv/bin/python scripts/oracle_compare.py [--host-dir DIR] [--judge-dir DIR]
           [--baseline-summary FILE] [--run-id ID] [--output FILE]
The defaults are the protocol v1 run of 2026-11-07 (experiments/oracle).
"""

import argparse
import json
import math
import sys
from pathlib import Path

from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from candidate_stats import episode_key  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
LEVELS = [str(k) for k in range(5)]


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def shown(path: Path) -> str:
    """Path relative to the repository when it lies inside it."""
    resolved = path.resolve()
    return str(resolved.relative_to(ROOT)) if resolved.is_relative_to(ROOT) else str(resolved)


def loss_history(train: dict) -> list:
    return [(h["step"], h["loss"], h["l1_loss"], h["kl_loss"]) for h in train.get("history", [])]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host-dir", type=Path, default=ROOT / "experiments/baseline/seed1000")
    ap.add_argument("--judge-dir", type=Path, default=ROOT / "experiments/oracle/judge")
    ap.add_argument("--baseline-summary", type=Path, default=ROOT / "experiments/baseline/summary.json")
    ap.add_argument("--run-id", default="c504ab36acb7416eb92a37b8c1735ed1")
    ap.add_argument("--output", type=Path, default=ROOT / "experiments/oracle/oracle_vs_host_seed1000.json")
    args = ap.parse_args()
    host_dir, judge_dir = args.host_dir, args.judge_dir

    host = load(host_dir / "eval_hidden/eval_result.json")
    judge = load(judge_dir / "eval_result.json")
    reward = load(judge_dir / "reward.json")
    final = load(judge_dir / "final_result.json")
    summary = load(args.baseline_summary)
    host_train = load(host_dir / "train/train_info.json")
    judge_train = load(judge_dir / "train_info.json")

    if final["status"] != "completed" or final["agent_exit_code"] != 0:
        print(f"oracle run not completed: {final['status']} exit {final['agent_exit_code']}")
        return 1
    if reward["successes"] != judge["successes"] or reward["n_episodes"] != judge["n_episodes"]:
        print("reward.json and eval_result.json disagree")
        return 1
    if host["start_seed"] != judge["start_seed"] or host["n_episodes"] != judge["n_episodes"]:
        print("the two runs did not evaluate the same seed block")
        return 1

    ph = {episode_key(e): e for e in host["per_episode"]}
    pj = {episode_key(e): e for e in judge["per_episode"]}
    if set(ph) != set(pj) or len(ph) != host["n_episodes"]:
        print("per-episode seeds differ between the runs")
        return 1
    n = len(ph)

    both = sum(ph[s]["success"] and pj[s]["success"] for s in ph)
    host_only = sum(ph[s]["success"] and not pj[s]["success"] for s in ph)
    judge_only = sum(pj[s]["success"] and not ph[s]["success"] for s in ph)
    neither = n - both - host_only - judge_only
    discordant = host_only + judge_only
    mcnemar_p = stats.binomtest(min(host_only, judge_only), discordant, 0.5).pvalue if discordant else 1.0

    table = [[host["max_reward_histogram"][k] for k in LEVELS],
             [judge["max_reward_histogram"][k] for k in LEVELS]]
    same_level = sum(ph[s]["max_reward"] == pj[s]["max_reward"] for s in ph)
    if table[0] == table[1]:
        chi2, chi2_p, dof = 0.0, 1.0, len(LEVELS) - 1
    else:
        keep = [i for i in range(len(LEVELS)) if table[0][i] + table[1][i] > 0]
        chi2, chi2_p, dof, _ = stats.chi2_contingency([[row[i] for i in keep] for row in table])
    levels_h, levels_j = [ph[s]["max_reward"] for s in ph], [pj[s]["max_reward"] for s in ph]
    pearson_r = 1.0 if levels_h == levels_j else float(stats.pearsonr(levels_h, levels_j).statistic)
    with_sums = all("sum_reward" in e for e in ph.values()) and all("sum_reward" in e for e in pj.values())
    different_sum = sum(ph[s]["sum_reward"] != pj[s]["sum_reward"] for s in ph) if with_sums else None

    pooled_k, pooled_n = summary["pooled_successes"], summary["pooled_episodes"]
    p_pool = pooled_k / pooled_n
    z = (judge["successes"] / n - p_pool) / math.sqrt(p_pool * (1 - p_pool) * (1 / n + 1 / pooled_n))

    hist_h, hist_j = loss_history(host_train), loss_history(judge_train)
    out = {
        "description": "RSI-Harness oracle run (uniform default policy) vs host baseline seed 1000; "
                       "same schedule, training seed and hidden seed block",
        "oracle_run_id": args.run_id,
        "host_dir": shown(host_dir),
        "judge_dir": shown(judge_dir),
        "schedule_sha256_host": load(host_dir / "schedule/schedule_summary.json")["schedule_sha256"],
        "schedule_sha256_judge": load(judge_dir / "schedule_summary.json")["schedule_sha256"],
        "host_successes": host["successes"],
        "judge_successes": judge["successes"],
        "n_episodes": n,
        "paired_2x2": {"both": both, "host_only": host_only, "judge_only": judge_only, "neither": neither},
        "mcnemar_exact_p": mcnemar_p,
        "same_max_reward_level_episodes": same_level,
        "different_sum_reward_episodes": different_sum,
        "identical_per_episode": discordant == 0 and same_level == n and different_sum in (0, None),
        "max_reward_pearson_r": pearson_r,
        "level_histogram_host": table[0],
        "level_histogram_judge": table[1],
        "level_histogram_chi2": chi2,
        "level_histogram_p": chi2_p,
        "level_histogram_dof": dof,
        "judge_vs_pooled_baseline_z": z,
        "train_cudnn_deterministic": [host_train.get("cudnn_deterministic"), judge_train.get("cudnn_deterministic")],
        "eval_cudnn_deterministic": [host.get("cudnn_deterministic", False), judge.get("cudnn_deterministic", False)],
        "train_loss_history_points": [len(hist_h), len(hist_j)],
        "train_loss_history_identical": bool(hist_h) and hist_h == hist_j,
        "train_final_loss_identical": host_train["final_loss_mean_last_1000"] == judge_train["final_loss_mean_last_1000"],
        "train_versions_host": host_train.get("versions"),
        "train_versions_judge": judge_train.get("versions"),
        "judge_gl_renderer": judge["gl_renderer"],
        "host_gl_renderer": host.get("gl_renderer"),
        "host_train_s": host_train["train_s"],
        "judge_train_s": judge_train["train_s"],
        "host_eval_s": host["eval_s"],
        "judge_eval_s": judge["eval_s"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
