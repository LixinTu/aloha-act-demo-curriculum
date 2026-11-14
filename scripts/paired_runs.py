"""Compare one run's hidden-block result, episode by episode, with other runs on the same seeds.

For runs that candidate_stats.py cannot read (no compiled schedule), such as the lerobot-train
control, and for comparisons with runs other than the v1 host baseline. Episodes are paired by
environment seed; the test is McNemar's exact test on the discordant pairs.

Usage: .venv/bin/python scripts/paired_runs.py <run_dir> <other_run_dir>... [--output PATH]
Each directory holds eval_hidden/eval_result.json. Writes JSON to --output if given and prints a table.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from candidate_stats import ROOT, episode_key, load, score_hidden, summary_line, uses_hidden_block  # noqa: E402
from scipy import stats  # noqa: E402


def paired(a: dict, b: dict) -> dict:
    pa = {episode_key(e): bool(e["success"]) for e in a["per_episode"]}
    pb = {episode_key(e): bool(e["success"]) for e in b["per_episode"]}
    if set(pa) != set(pb):
        raise ValueError("runs were not evaluated on the same seeds")
    both = sum(pa[s] and pb[s] for s in pa)
    a_only = sum(pa[s] and not pb[s] for s in pa)
    b_only = sum(pb[s] and not pa[s] for s in pa)
    disc = a_only + b_only
    return {
        "other_successes": sum(pb.values()),
        "both": both,
        "this_only": a_only,
        "other_only": b_only,
        "neither": len(pa) - both - disc,
        "mcnemar_exact_p": stats.binomtest(min(a_only, b_only), disc, 0.5).pvalue if disc else 1.0,
        "other_max_reward_histogram": b["max_reward_histogram"],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("others", type=Path, nargs="+")
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()

    hidden = load(args.run_dir / "eval_hidden/eval_result.json")
    proxy = load(args.run_dir / "eval_proxy/eval_result.json")
    if not uses_hidden_block(hidden):
        print("hidden evaluation did not use the Judge's seed block")
        return 1
    scored = score_hidden(hidden, pair_with_host=False)
    label = str(args.run_dir.resolve().relative_to(ROOT))
    pairs = {}
    for other in args.others:
        result = load(other / "eval_hidden/eval_result.json")
        if not uses_hidden_block(result):
            print(f"{other}: hidden evaluation did not use the Judge's seed block")
            return 1
        pairs[str(other.resolve().relative_to(ROOT))] = paired(hidden, result)

    print(summary_line(label, scored, proxy))
    for name, p in pairs.items():
        print(f"  vs {name}: {p['other_successes']}/500; both {p['both']}, this only {p['this_only']}, "
              f"other only {p['other_only']}, neither {p['neither']}; McNemar p = {p['mcnemar_exact_p']:.3g}")
    if args.output:
        args.output.write_text(json.dumps({
            "run_dir": label,
            "hidden": scored["hidden"],
            "proxy": {"successes": proxy["successes"], "n_episodes": proxy["n_episodes"]},
            "vs_pooled_v1_baseline": scored["vs_pooled_baseline"],
            "paired": pairs,
        }, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
