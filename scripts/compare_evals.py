#!/usr/bin/env python3
"""Compare two evaluations of one seed block episode by episode.

Usage: .venv/bin/python scripts/compare_evals.py <eval_result.json A> <eval_result.json B> [--output PATH]

Exit 0: every episode has the same success, max_reward and sum_reward in both files.
Exit 1: some episodes differ (the counts are printed and written to --output).
Exit 2: the two files are not evaluations of the same seed block.
The report never lists seeds, so it can be shown where hidden seeds must not appear.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from candidate_stats import episode_key  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("a", type=Path)
    ap.add_argument("b", type=Path)
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()
    a = json.loads(args.a.read_text())
    b = json.loads(args.b.read_text())
    pa = {episode_key(e): e for e in a["per_episode"]}
    pb = {episode_key(e): e for e in b["per_episode"]}
    if a["start_seed"] != b["start_seed"] or a["n_episodes"] != b["n_episodes"] or set(pa) != set(pb) or len(pa) != a["n_episodes"]:
        print("the two files are not evaluations of the same seed block")
        return 2
    only_a = sum(1 for s in pa if pa[s]["success"] and not pb[s]["success"])
    only_b = sum(1 for s in pa if pb[s]["success"] and not pa[s]["success"])
    discordant = only_a + only_b
    out = {
        "a": str(args.a),
        "b": str(args.b),
        "n_episodes": len(pa),
        "cudnn_deterministic": [a.get("cudnn_deterministic", False), b.get("cudnn_deterministic", False)],
        "successes": [a["successes"], b["successes"]],
        "different_success": discordant,
        "only_a_success": only_a,
        "only_b_success": only_b,
        "mcnemar_exact_p": stats.binomtest(min(only_a, only_b), discordant, 0.5).pvalue if discordant else 1.0,
        "different_max_reward": sum(1 for s in pa if pa[s]["max_reward"] != pb[s]["max_reward"]),
        "different_sum_reward": sum(1 for s in pa if pa[s]["sum_reward"] != pb[s]["sum_reward"]),
        "max_reward_histograms": [a["max_reward_histogram"], b["max_reward_histogram"]],
        "eval_s": [a["eval_s"], b["eval_s"]],
    }
    out["identical"] = not (out["different_success"] or out["different_max_reward"] or out["different_sum_reward"])
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(out, indent=2) + "\n")
    print(f"{out['n_episodes']} episodes: successes {a['successes']} vs {b['successes']}; different success "
          f"{discordant} (only A {only_a}, only B {only_b}, McNemar p = {out['mcnemar_exact_p']:.3g}); different "
          f"max_reward {out['different_max_reward']}; different sum_reward {out['different_sum_reward']}; "
          f"{'IDENTICAL' if out['identical'] else 'DIFFERENT'}")
    return 0 if out["identical"] else 1


if __name__ == "__main__":
    sys.exit(main())
