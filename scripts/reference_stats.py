"""Score the released checkpoint's evaluation (scripts/eval_reference.sh) against the uniform baseline.

The released checkpoint lerobot/act_aloha_sim_insertion_human was trained with the task's recipe
and budget (100,000 steps, batch 8, seed 1000) but with an older LeRobot (3c0a209), the v2 dataset
format and epoch shuffling. Its score is an author measurement, not the matched baseline. The
comparison is the same as in scripts/candidate_stats.py (shared score_hidden()).

Usage: .venv/bin/python scripts/reference_stats.py [run_dir] [--output PATH]
Default run_dir experiments/reference; writes <run_dir>/stats.json and prints a one-line summary.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from candidate_stats import ROOT, load, score_hidden, summary_line, uses_hidden_block  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path, nargs="?", default=ROOT / "experiments/reference")
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()
    run = args.run_dir.resolve()

    hidden = load(run / "eval_hidden/eval_result.json")
    proxy = load(run / "eval_proxy/eval_result.json")
    if not uses_hidden_block(hidden):
        print("hidden evaluation did not use the Judge's seed block")
        return 1
    train_config = load(ROOT / "data/reference_policy/train_config.json")

    scored = score_hidden(hidden, pair_with_host=True)
    out = {
        "run_dir": str(run.relative_to(ROOT)),
        "checkpoint": hidden.get("checkpoint"),
        "checkpoint_sha256": (run / "checkpoint.sha256").read_text().split()[0],
        "note": "released checkpoint: same recipe and budget, older LeRobot, v2 dataset format, "
                "epoch shuffling; an author measurement, not the matched baseline",
        "train_config": {key: train_config.get(key) for key in ("steps", "batch_size", "seed")},
        "hidden": scored["hidden"],
        "proxy": {"successes": proxy["successes"], "n_episodes": proxy["n_episodes"]},
        "vs_pooled_baseline": scored["vs_pooled_baseline"],
        "paired_vs_host_seed1000": scored["paired_vs_host_seed1000"],
    }
    dest = args.output or run / "stats.json"
    dest.write_text(json.dumps(out, indent=2) + "\n")
    print(summary_line(out["run_dir"], scored, proxy))
    return 0


if __name__ == "__main__":
    sys.exit(main())
