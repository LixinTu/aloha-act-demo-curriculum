# Agent-designed demonstration curricula for fixed-budget ACT on ALOHA insertion

Reference implementation of a proposed [OpenRSI-Index](https://github.com/OpenRSI-Foundation/OpenRSI-Index)
AutoResearch task, built on [LeRobot](https://github.com/huggingface/lerobot) v0.6.1
(commit `7e241bd630a3719a56157a497ce5d08f244784f1`).

**Question.** With the ACT policy, the 50 human teleoperation demonstrations of
`lerobot/aloha_sim_insertion_human`, their action labels, the optimizer, the 100,000-step x batch-8
training budget and the seeds all fixed, can a research agent raise closed-loop insertion success in
the ALOHA simulation by designing which demonstration frames serve as action-chunk anchors, with what
weight, and in which training stage?

**Status (2026-11-19).** The proposal was accepted into the OpenRSI-Index contribution pipeline
([Discussion #350](https://github.com/OpenRSI-Foundation/OpenRSI-Index/discussions/350)). The
official task generated from this implementation was run once on 2026-11-15 to 11-18 and is under
OpenRSI review; its result is summarized in [Official OpenRSI run](#official-openrsi-run). The rest
of this repository is unchanged: the contributor's reference implementation and pre-proposal
measurements, which are not an official OpenRSI-Index task. Every number below comes from a real run.
Result files are in `experiments/` and run logs in `logs/`, except for the contributor-side agent
run, whose RSI-Harness records list the hidden seeds and are withheld; `docs/research_report.md`
(section 8) describes it.

## Official OpenRSI run

The OpenRSI pipeline generated the official task from this implementation. The contributor validated
it and ran it once with RSI-Harness on an RTX 4070 Laptop GPU from 2026-11-15 to 2026-11-18. The
task, its hidden evaluation, the run logs and the review reports are in the private OpenRSI task
repository and are not published here.

The official protocol has its own uniform baselines, measured at three (training seed, schedule seed)
pairs, and its own hidden block of 500 seeds. Its numbers are therefore not comparable with the
pre-proposal table under [Results](#results).

| Item | Value |
| --- | --- |
| Agent | Claude Code, `claude-opus-5-5` at maximum reasoning. Budget: 55 h of wall-clock time including Judge time, on one GPU shared by Work and Judge |
| Rounds | 11, all scored. Every round retrains ACT from scratch, about 4.8 h per round |
| Uniform baseline, one run per pair | 67, 74 and 67 of 500 at pairs 0, 1 and 2 |
| Best round, at pair 0 (the Judge's pair) | 99 of 500 (round 9). 5 of the 11 rounds reached the pre-declared single-round requirement of 87 |
| Pre-declared confirmation: the best round's exact candidate retrained at the two private pairs | 86 + 86 = 172 of 1,000, against 141 for uniform sampling. Requirement: 172. **Confirmed, with zero margin** |

- **What the selected curriculum does.**
  - It drops, as action-chunk anchors, the frames after each demonstration's estimated insertion.
  - It weights the remaining frames by how reliably each demonstration's recorded actions still
    succeed when they are replayed in the simulator with the objects slightly displaced.
- **Size of the gain.** At the confirmation pairs the gain is +3.1 percentage points (17.2% against
  14.1%), with a 95% interval of -0.1 to +6.3 points. The best round's 99 of 500 overstates it: the
  same candidate scored 86 and 86 when it was retrained at the other two pairs.
- **Limits of the result.**
  - It meets the pre-declared rule exactly at the threshold, and that rule assumes no run-to-run
    variance beyond binomial.
  - It rests on one simulated hidden block of 500 episodes and one machine.
  - It does not show which of the two ingredients helps, and it says nothing about other tasks or
    real robots.
- **Checks so far.**
  - An independent scientific re-verification, in a separate Claude context whose blinding to the
    per-round scores was incomplete, supported the candidate's validity and the classification.
  - The trajectory audit is pending.
  - OpenRSI's own review and reproduction are outstanding.

## Contents

| Path | Contents |
| --- | --- |
| `task/aloha-act-demo-curriculum/` | The Harbor task: instruction, Docker environment with the task's tools (`aloha-validate`, `aloha-compile`, `aloha-train`, `aloha-eval`, `aloha-features`) and curriculum DSL, solution, Judge (`tests/`) and runtime unit tests. Its README gives the protocol in full. |
| `scripts/` | The run, statistics and safety scripts behind every measurement. |
| `experiments/` | Result files of every run: training summaries, compiled schedules, evaluation results, paired comparisons, the author probes' curricula. |
| `logs/` | Run logs, with local paths shortened. |
| `pins/` | Exact package versions and asset hashes. |
| `docs/research_report.md` | Methods, all results and the decision rules, with dates. |
| `docs/related_work.md` | Checked references. |

## Results

Pre-proposal measurements under this repository's protocol: 500 hidden episodes per run, RTX 4070
Laptop GPU, 2026-11-03 to 2026-11-14.

| Run | Successes of 500 |
| --- | --- |
| Source launch path: unmodified `lerobot-train` v0.6.1, seed 1000 | 100 (re-run: 102) |
| Released checkpoint `lerobot/act_aloha_sim_insertion_human` | 101 |
| Unmodified `lerobot-train` in deterministic mode, seed 1001 | 70 |
| Uniform baseline, final protocol (task trainer, deterministic mode), seeds 1000 / 1001 / 1002 | 80 / 64 / 57 (pooled 13.40%) |
| RSI-Harness oracle run, final protocol, seed 1000 | 81 |
| Author probes (hand-designed curricula), final protocol: drop-idle-tail / insertion-focus | 84 / 88 |
| Contributor-side agent run (Claude Code, `claude-opus-5-5`), the two rounds that scored | 83 / 73 |

- A single Judge run needs at least 92/500 to exceed the pooled uniform baseline at two-sided 95%
  (`experiments/baseline_v3_det/summary.json`). No run has reached it so far.
- In deterministic mode the task's trainer and unmodified `lerobot-train` produce bit-identical
  weights for the same frame order and seed. The source recipe's 20% is the score of seed 1000; at
  seed 1001 the same unmodified script scores 70/500.
- Training is bit-reproducible. Evaluation is not: one checkpoint scored 57, 60 and 62 in three
  evaluations, because GPU rendering differs by a few pixels between processes.
- `docs/research_report.md` has the full record, including the two agent rounds that were lost to
  host faults.

## Withheld from this copy

- **The hidden evaluation seeds.** `task/aloha-act-demo-curriculum/tests/hidden_seeds.json` is not
  included. In the hidden-block results under `experiments/`, `start_seed` reads `"withheld"` and
  each episode carries its index in the block (`episode`) instead of its seed. The statistics scripts
  (`paired_runs.py`, `compare_evals.py`, `uniform_runs_spread.py`, `baseline_stats.py`,
  `candidate_stats.py`) pair episodes by that index, so the published statistics can be recomputed
  from these files. To run the Judge, create the seeds file as described in
  `task/aloha-act-demo-curriculum/tests/HIDDEN_SEEDS.md`; results on your own block will differ from
  the ones here.
- **Run records that list the seeds.** RSI-Harness records of the oracle and agent runs are not
  included, apart from the Judge outputs in `experiments/oracle*/judge/`.
- **Model weights.** Their SHA-256 hashes are in `experiments/**/checkpoint.sha256`.
- **The contributor's working notes.** The documents and script comments cite them as
  `docs/plan.md`, together with commit hashes of the contributor's working repository. This copy
  starts a new history.

## Running

The task requires Docker with the NVIDIA Container Toolkit and one CUDA GPU. Build and run it as
described in `task/aloha-act-demo-curriculum/README.md`. The build downloads the dataset and the
released checkpoint from Hugging Face at the revisions pinned in `pins/versions.json`; they are not
redistributed here. `experiments/features/` holds per-frame features computed from that dataset.

The host scripts in `scripts/` call a Python 3.12 environment at `.venv`. It holds the packages
pinned in `pins/requirements.freeze.txt` (torch from the PyTorch cu128 index) and the task package
from `task/aloha-act-demo-curriculum/environment`. The `scripts/test_*.sh` files are self-contained
tests; `test_confirm.sh` also needs a `tests/hidden_seeds.json` of your own.

## License and credits

Code in this repository is released under the Apache License 2.0 (`LICENSE`). It builds on LeRobot
(Apache-2.0) and gym-aloha (Apache-2.0). The `lerobot/aloha_sim_insertion_human` dataset is MIT
licensed. ACT and ALOHA are from Zhao, Kumar, Levine and Finn, *Learning Fine-Grained Bimanual
Manipulation with Low-Cost Hardware*, RSS 2023.

Contact: Lixin Tu, ltu35@gatech.edu
