# Agent-Designed Demonstration Curricula for Fixed-Budget ACT on ALOHA Insertion

*Research report and OpenRSI-Index task design. Status: 2026-11-13. The final protocol (protocol v3 with deterministic training) is fixed and measured: baseline, threshold, oracle validation and author probes are summarised in the Abstract and detailed in Sections 5.5 and 7. The first agent run is in progress (Section 8). Sections 5.2-5.4 and the first part of Section 7 report protocol v1 and are kept as history; Section 5.5 records in order how the gap to LeRobot's own training script was traced, each step with its reading committed before the run.*

## Abstract

We build an OpenRSI-Index research task in which an AI research agent improves the closed-loop
success of an Action Chunking Transformer (ACT) on the simulated ALOHA peg-insertion task by
designing only the *sampling distribution* of 50 human teleoperation demonstrations during
training. The model, data, action labels, optimizer and the 800,000-exposure budget are fixed;
the agent submits a declarative curriculum that the evaluator compiles, retrains and scores on
500 hidden-seed episodes. Dataset analysis shows that roughly one third of all demonstration
frames are post-task holding, so uniform sampling spends a third of the budget on chunks that
teach the robot to stay still.

**Final protocol and baseline.** The Judge trains with protocol v3 (balanced exposure draws,
LeRobot's image path and ImageNet statistics) in deterministic cuDNN mode, where our trainer and
LeRobot v0.6.1's own `lerobot-train` produce bit-identical weights over the full budget. Uniform
sampling scores 80, 64 and 57 of 500 at training seeds 1000, 1001 and 1002, pooled **13.40%
(201/1500)**, with a seed-to-seed s.d. of 2.36 pp against 1.52 pp of binomial noise. By the formula
committed before these runs, a single Judge round needs **>= 92/500** to exceed the baseline at
two-sided 95%. Training is bit-reproducible: a repeat of seed 1002, made because that run ended while
the machine's kernel was logging memory corruption, gave the same weights hash, and the RSI-Harness
oracle run in a clean container reproduced the host's seed-1000 training to the last logged loss
(81/500 against the host's 80/500, McNemar p = 1.0). Evaluation is not: one checkpoint scored 57, 60
and 62 of 500 in three evaluations, because GPU rendering differs in a few pixels between processes.
That re-evaluation noise (about 0.6 pp) lies inside the threshold's binomial term.

**Author probes.** Two hand-designed curricula, committed before they ran, test whether the action
space moves the score. drop-idle-tail moves the 28% of exposures that fall on post-task holding to
the active part of each demonstration; insertion-focus also triples the weight of the bimanual,
contact-rich phase late in training. Under the final protocol they score 84/500 (+3.4 pp, p = 0.18)
and 88/500 (+4.2 pp, p = 0.095), both below the threshold (74 and 78 under protocol v1). The obvious
mechanism-level moves do not reach a gain one round can detect; whether the action space has
measurable headroom remains open, and the first agent run (Section 8) is under way.

**How the baseline was settled.** The released LeRobot checkpoint, trained with the same recipe and
budget, scores **101/500 (20.2%)** on this evaluator, and `lerobot-train` v0.6.1 at the same training
seed scores 100/500 (102/500 when re-run), while the twelve uniform runs of our trainer pool to
13.60%. Each step of the investigation had its reading committed before it ran (Section 5.5). An
image-normalisation deviation was real but not the cause (protocol v2: 60/500); neither the frame
draw (protocol v3: 82 and 59) nor the exact frame order (two replays of `lerobot-train`'s order: 66
and 70) explained the gap; in deterministic mode the two trainers are bit-identical over 100,000
steps, and that model scores 100/500. The decisive run was `lerobot-train` itself in deterministic
mode at training seed 1001: 70/500, indistinguishable from our trainer's 64 at the same seed
(p = 0.56) and well below its own 100 at seed 1000 (p = 0.004). Seed 1000 was a high draw of the
recipe's own seed-to-seed spread, so the gap is seed variance, not a trainer defect. One difference
stays unexplained: in normal cuDNN mode, on the same input, our trainer gave 66 and 70 where
`lerobot-train` gave 100 and 102, and a kernel-level comparison found the autotuned algorithms
unstable even within one trainer. The Judge therefore trains in deterministic mode, where the
question does not arise.

## 1. Research question

Every ACT training sample is a frame together with the next 100 actions. Which frames serve as
chunk anchors, with what weight, and at which stage of training, is therefore a decision about the
*temporal structure* of the demonstrations (approach, bimanual grasp, transport, contact-rich
insertion, post-task holding), not a scalar hyperparameter.

**Question.** With architecture, data, labels, optimizer and budget fixed, can an agent-designed
curriculum over (episode, chunk anchor, training stage) raise closed-loop insertion success?

**Falsifiable criterion.** A curriculum whose single fixed-seed evaluator run achieves >= 92/500
successes exceeds the final-protocol uniform baseline at two-sided 95% (Section 5.5; under protocol
v1 the threshold was 88/500, Section 5.3).

## 2. Related work

Demonstration curation for imitation learning so far operates on whole demonstrations, decided
once before training: Demo-SCORE filters demonstrations with a classifier trained on policy
rollouts (Chen et al., RSS 2025, arXiv:2503.03707); CUPID ranks demonstrations by influence on
closed-loop return (Agia et al., CoRL 2025, arXiv:2506.19121); DemInf scores quality with mutual
information over state-action chunks (arXiv:2502.08623); RINSE and a PSD spectral metric filter by
smoothness and argue that action chunking amplifies the effect of data quality (arXiv:2604.23000,
arXiv:2605.01544). robomimic established that offline imitation is highly sensitive to
demonstration quality (Mandlekar et al., CoRL 2021, arXiv:2108.03298). ACT and the ALOHA
simulation data come from Zhao et al. (RSS 2023, arXiv:2304.13705); Zhang et al. (ICLR 2026,
arXiv:2507.09061) analyse why action chunking limits compounding error.

## 3. Novelty

- **Selection unit.** Prior work keeps or drops whole demonstrations. Here the unit is the chunk
  anchor inside a trajectory, and selection can change over training time.
- **Agent research loop.** The task is posed as an iterative hypothesis-test loop for an AI
  research agent, scored by retraining and closed-loop evaluation, not as a one-shot filter.
- **Within OpenRSI-Index.** Existing data-curriculum tasks (`reasonir-difficulty-curriculum`,
  `learnability-cot`) target language-model retrieval and SFT; the only robotics task
  (`isaaclab-peginsert-reward-search`) fixes data and searches RL rewards; Task Ideas #134
  generates new demonstrations with MimicGen rather than re-weighting human ones.

## 4. Task design

**Action space (data only).** `policy.json` (<= 64 KiB) declares episode inclusion/exclusion,
anchor eligibility (frame range, maximum padded steps in the chunk) and 1-16 training stages, each
with a non-negative per-frame weight expression and an optional mask. Expressions use a closed DSL:
arithmetic, comparisons, dataset-wide `rank`/`zscore`, per-episode aggregates (`ep_mean`,
`ep_cummax`, `ep_first_where`, ...), temporal windows (`window_mean`, `future_max`, `shift`, ...)
and training-progress variables `t_train`, `t_stage`, over 61 public per-frame features (joint
states and actions, velocities, jerk, gripper commands, chunk path length, episode aggregates). An
optional `scores.csv` adds up to 16 agent-computed per-frame columns (for example, per-frame
prediction error of the released checkpoint).

**Compiler.** 100,000 steps are split into 2,000 blocks of 50 steps; each block picks 400 anchors so
that each frame's exposures follow its accumulated share of `weight x mask x eligibility`. A frame
falls due once per unit of that share, at a random point of the unit drawn with a fixed RNG, and each
block takes the most overdue frames and shuffles them, giving exactly 800,000 exposures. Uniform
weights give shuffled epochs with exactly 32 exposures per frame. This is protocol v3; versions 1 and
2 drew with replacement (Section 5.5). The schedule SHA-256 is reported.

**Fixed trainer.** LeRobot v0.6.1's ACT, processors and optimizer preset with the
forward/backward/clip/step order of `lerobot_train.update_policy`; the only change is a sampler
that follows the schedule. Given the same frame order its weights are bit-identical to
`lerobot-train`'s in deterministic mode (`scripts/check_trainer_parity.sh`).

**Evaluator (Judge).** Verifies all fixed assets against a build-time SHA-256 manifest,
copies the two candidate files once into a private directory, recompiles the schedule, trains from
scratch with seed 1000, checks step/exposure/seed accounting, evaluates 500 hidden consecutive seeds
with LeRobot's `eval_policy`, and writes `reward = successes / 500`. Invalid candidates and
failed runs are unscored. Agent code and checkpoints are never scoring inputs; this is why the task
retrains at evaluation time (a submitted checkpoint cannot prove its schedule or budget).

## 5. Experimental setup and baseline

### 5.1 Fixed protocol

| Item | Value |
| --- | --- |
| Code | lerobot v0.6.1 (`7e241bd6`), gym-aloha 0.1.5 (`6203fac5`), MuJoCo 3.8.1, dm-control 1.0.41, torch 2.11.0+cu128 |
| Data | `lerobot/aloha_sim_insertion_human` @ `cc571a3c` (50 episodes x 500 frames, 50 fps, 480x640 top camera, 14-D state and action) |
| Model | ACT, ResNet18 ImageNet init, chunk 100, VAE latent 32, KL weight 10, dropout 0.1 |
| Training | 100,000 steps x batch 8, AdamW lr 1e-5, wd 1e-4, grad clip 10, seed 1000, no augmentation; final protocol (v3): balanced exposure draws, LeRobot's image path and ImageNet statistics, deterministic cuDNN mode |
| Evaluation | `AlohaInsertion-v0`, 400 steps, success = reward level 4, 500 hidden seeds, 10 synchronous envs; public proxy seeds 100000-100099; cuDNN benchmark mode as in `lerobot-eval` |
| Hardware | NVIDIA RTX 4070 Laptop GPU (8 GB), Intel i9-13900HX, 62 GB RAM |

### 5.2 Uniform baseline (protocol v1)

| Training seed | Hidden successes / 500 | Rate | Wilson 95% | Levels 0/1/2/3/4 | Proxy / 100 | Train | Eval |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1000 (evaluator seed) | 68 | 13.6% | 10.9-16.9% | 85/137/35/175/68 | 10 | 4.81 h | 35 min |
| 1001 | 76 | 15.2% | 12.3-18.6% | 81/96/28/219/76 | 17 | 4.49 h | 31 min |
| 1002 | 65 | 13.0% | 10.3-16.2% | 96/94/37/208/65 | 15 | 4.66 h | 83 min |
| **Pooled** | **209 / 1500** | **13.93%** | **12.3-15.8%** | | | | |

Levels: 0 nothing, 1 both grippers touch, 2 both objects lifted, 3 peg touches socket, 4 pin
inside. About 40% of episodes reach contact (level 3) but only 14% complete the insertion.

The model card of the released checkpoint reports 20.6%; it was produced with an older LeRobot
(`3c0a209`) and dataset format, and that repository's `eval_info.json` is the transfer-cube file
(83.0%), so the figure is reported, not reproduced, and is not used as a baseline. Our own
evaluation of that checkpoint on the hidden block gives 20.2% (Section 5.5). It remains a reference
point rather than the matched baseline, because it was trained with older code.

### 5.3 Noise (protocol v1)

A single 500-episode run has binomial standard error 1.55 pp. The three seeds are homogeneous
(Pearson chi-square 1.08, df 2, p = 0.58); the training-seed variance estimate is zero. Two v3 seeds
later differed by more, but all eight uniform runs of our trainer together spread as binomial noise
(dispersion 1.02, 95% interval 0.44-4.21; Section 5.5), so the binomial threshold below stands. The
standard error of a single candidate run against the pooled baseline is 1.79 pp, so the two-sided
95% threshold is 3.50 pp: a single run needs >= 88/500 (17.6%).

**Selection across rounds.** The threshold holds for one pre-specified run, but the harness keeps
the best score of an agent's rounds. Suppose every submission performed exactly like the uniform
baseline (binomial, p = 209/1500). The expected best of N rounds would then be 74.0/500 for N = 2,
77.7 for N = 4 and 81.8 for N = 10, that is +0.9, +1.6 and +2.4 pp. The chance that at least one
round reaches 88/500 would be 2.5%, 4.9% and 11.8%. A best-round score is therefore not evidence of
improvement by itself. Before any claim, the best candidate of each agent run is re-trained with
independent training seeds and re-evaluated (Section 8).

### 5.4 Production-path replicate (protocol v1)

The uniform policy was also scored end to end by RSI-Harness with its oracle agent (run
`c504ab36acb7416eb92a37b8c1735ed1`, 2026-11-07, one GPU shared by Work and Judge in `release-all`
mode). The Judge ran in a clean container with no network, re-verified the asset manifest,
recompiled the schedule, retrained and evaluated the 500 hidden seeds.

| Check | Host run, seed 1000 | RSI-Harness Judge |
| --- | --- | --- |
| Schedule SHA-256 | `5d45f8e8...4751461` | identical |
| Successes / 500 | 68 | 69 |
| Levels 0/1/2/3/4 | 85/137/35/175/68 | 74/132/28/197/69 |
| EGL renderer | NVIDIA | NVIDIA GeForce RTX 4070 Laptop GPU |
| Train / hidden eval | 17,327 s / 2,125 s | 16,043 s / 1,377 s |

The two runs agree (McNemar exact p = 1.0; reward-level chi-square 2.94, df 4, p = 0.57; the Judge
run against the pooled baseline z = -0.07). Episode-level outcomes repeat only partly: 31
successes are shared, against about 9 expected for independent runs and 68 for identical ones,
and the highest reward level per episode correlates at r = 0.62 (328 of 500 episodes reach the
same level). Retraining with the same seed is not bit-identical (cuDNN autotuning, rendering
jitter). The initial state explains part of each outcome, but most run-to-run variation comes
from training and simulation noise, so candidates are compared by counts, not by episodes.
Because runs on the shared hidden block are positively correlated, the independent-binomial
threshold of Section 5.3 is conservative. The replicate is not added to the pooled baseline,
which stays the three pre-declared training seeds. Evidence: `experiments/oracle/judge/` (copied from the harness logs, with
`SHA256SUMS`) and `experiments/oracle/oracle_vs_host_seed1000.json`, regenerated by
`scripts/oracle_compare.py`.

### 5.5 Released checkpoint and the protocol v1 deviation

The released checkpoint `lerobot/act_aloha_sim_insertion_human` (revision `33259aa`) was migrated
to the current processor format; its `model.safetensors` SHA-256 after migration is
`5e0e5e5c...4c95dc5b`. It was trained with the task's recipe and budget: 100,000 steps, batch 8,
seed 1000, the same ACT configuration and optimizer, no augmentation. It differs in the LeRobot
version (`3c0a209`), the v2 dataset format and epoch shuffling. `scripts/eval_reference.sh`
evaluated it with the Judge's evaluator on the 500 hidden seeds and the 100 proxy seeds;
`scripts/reference_stats.py` scores it.

| Measure | Released checkpoint | Uniform baseline, protocol v1 |
| --- | --- | --- |
| Hidden successes | 101/500 (20.2%, Wilson 16.9-23.9%) | 209/1500 pooled (13.93%) |
| Levels 0/1/2/3/4 | 72/121/16/190/101 | 262/327/100/602/209 (three seeds) |
| Proxy / 100 | 17 | 10, 17, 15 |

Against the pooled v1 baseline the difference is +6.27 pp (z = 3.50, p = 0.0005; 95% interval
+2.8 to +9.8 pp), above the 88/500 threshold. The level histogram also differs (chi-square 20.8,
df 4, p = 0.0004). Paired with the host seed-1000 run, 30 successes are shared, 38 occur only under
the v1 baseline and 71 only under the released checkpoint (McNemar exact p = 0.002). The
evaluation reproduces the model card's 20.6% within noise, although that repository's own
evaluation file belongs to another task (Section 5.2).

A model trained with the same recipe and budget that scores 6 pp above our baseline points to a
training difference. The saved configurations agree except for the device field, and the state
and action normalisation statistics agree to 1e-6. The image statistics differ. The released
checkpoint normalises camera images with ImageNet statistics (mean 0.485/0.456/0.406, std
0.229/0.224/0.225); ours used the dataset's pixel statistics (mean about 0.16, std about 0.20).
LeRobot's `make_dataset` replaces camera statistics with ImageNet's when
`DatasetConfig.use_imagenet_stats` is true, which is its default (`lerobot/datasets/factory.py`
lines 137-142, `lerobot/configs/default.py` line 41). The v1 trainer built `LeRobotDataset`
directly and skipped that step. The ImageNet-pretrained ResNet18 therefore saw an average pixel
shifted by 1.1-1.4 standard deviations from the normalisation it was pretrained with.

Protocol v2 (commit `87103b0`) fixes this.
- It sets `dataset.use_imagenet_stats: true` and applies LeRobot's own `IMAGENET_STATS`.
- `check_image_normalization` refuses to train when the preprocessor disagrees with the protocol.
- Four unit tests cover the change (55 pass), the uniform schedule SHA-256 is unchanged, and a
  200-step smoke run saves ImageNet statistics.

The v2 uniform baseline at training seed 1000 (2026-11-08, 08:50-13:45) scored 60/500 (12.0%,
Wilson 9.4-15.1%) on the hidden block and 14/100 on the proxy block. That is -1.93 pp against the
pooled v1 baseline (z = -1.08, p = 0.28). Paired with the v1 run at the same seed on the same
episodes, 19 successes are shared, 49 occur only under v1 and 41 only under v2 (McNemar exact
p = 0.46). The deviation was real, but it does not explain the gap. Under the rule committed before
the result (commit `4eb9fab`: at most 76/500 means investigate first), seeds 1001 and 1002 were not
run.

Four quick checks then ruled out other explanations.
- All three models start from ImageNet weights: their conv1 weights correlate 0.9999 with
  torchvision's ResNet18 weights, against 0.002 for a random initialisation.
- The uniform schedule is well shuffled: a batch of 8 holds 7.46 distinct episodes on average, the
  same as independent draws.
- The released training configuration matches the fixed protocol in every training field, and the
  ACT defaults of LeRobot v0.6.1 equal the protocol's policy section.
- The model card states that the released model is the 100,000-step checkpoint, not one selected
  from the intermediate saves.

A control run then trained the same baseline with LeRobot v0.6.1's unmodified `lerobot-train`
(shuffled epochs without replacement, its own training loop) and scored it with the task
evaluator (`scripts/run_lerobot_train_control.sh`; environment: the lock files plus `accelerate`
1.15.0, pinned in `pins/control-venv.freeze.txt`). Its decision rule was committed before the run
(docs/plan.md, step 1a). At least 88/500 would mean our trainer still deviates from upstream. At
most 76/500 would place the gap in the released checkpoint's older LeRobot and dataset format.

The control scored 100/500 (20.0%, Wilson 16.7-23.7%) on the hidden block and 12/100 on the proxy
block, with levels 77/125/28/170/100 (2026-11-08; training 16,175 s). Paired with other runs on the
same episodes (`experiments/control_lerobot_train/seed1000/stats.json`):

| Run | Hidden /500 | Successes only under the control / only under the other run | McNemar exact p |
| --- | --- | --- | --- |
| Released checkpoint | 101 | 57 / 58 | 1.0 |
| `aloha-train` v2, seed 1000 | 60 | 71 / 31 | 9e-5 |
| `aloha-train` v1, seed 1000 | 68 | 68 / 36 | 0.002 |
| `aloha-train` v1, seed 1001 | 76 | 69 / 45 | 0.03 |
| `aloha-train` v1, seed 1002 | 65 | 72 / 37 | 0.001 |

The current software stack therefore reproduces the released checkpoint's score, and the older stack
is not the explanation. Our trainer was the outlier, so we compared the two trainers directly.
- The checkpoints agree: `config.json` differs only in `push_to_hub`, the processor files are the
  same and all 90 normalisation statistics are equal value by value.
- The training curves do not separate them: loss and L1 nearly coincide throughout (final L1 0.043
  against 0.0435). Fit on the training frames cannot see this gap.
- We ran `aloha-train` on the first 300 steps of `lerobot-train`'s frame order. In deterministic mode
  repeated runs of one trainer are bit-identical. The two trainers differed by at most 1.2e-3, the
  size of run-to-run noise in normal mode (9.8e-4). The source is where images are scaled:
  `lerobot-train` computes x/255 on the GPU, in fact x * (1/255), while `aloha-train` computed it on
  the CPU in the data workers, and 126 of the 256 pixel values round differently, by at most 6e-8.
  With the scaling in the same place, all 234 tensors are bit-identical. The check script repeats
  this over a whole epoch, 3,125 steps with every frame once, and all 234 are again bit-identical
  (`experiments/trainer_parity/steps3125/result.json`).

Initialisation, random numbers, optimizer, training loop and data path are therefore the same, and
what remains is the frame order. Protocols v1 and v2 drew frames independently with replacement. Over
a run each frame received 11 to 55 exposures (s.d. 5.66), and each window of 25,000 draws missed 36.8%
of the frames. `lerobot-train` and the released run shuffle epochs: exactly 32 exposures per frame.
The shuffling check above looked only at how episodes mix within a batch and could not see this, and
we had listed the difference as a harmless limitation.

Protocol v3 replaces the draw. A frame falls due once per unit of its accumulated target, at a random
point of the unit, and each block takes the most overdue frames. Uniform weights give shuffled epochs;
other weights are followed with low variance, and blocks stay as mixed as independent draws (3:1
weights: per-block share s.d. 0.024 against 0.022). Candidates and the baseline therefore share the
rule. A first version ranked frames by how far behind they were; it sent high and low weights in
alternating waves (per-block share s.d. 0.429) and was replaced before any non-uniform policy ran
under it. `aloha-train` now also scales images as `lerobot-train` does, and
`scripts/check_trainer_parity.sh` checks the bit-identity.

That the draw explains the gap was a hypothesis, with a rule committed before the run (docs/plan.md,
step 1b): at least 88/500 for the v3 baseline at seed 1000 means the draw explains the gap, 77-87 means
a second seed first, and at most 76/500 means it does not.

The v3 baseline at seed 1000 scored 82/500 (16.4%, Wilson 13.4-19.9%) on the hidden block and 24/100 on
the proxy block, with levels 92/114/16/196/82 (2026-11-09; training 15,701 s). That is +2.47 pp against
the v1 pooled baseline (z = 1.38, p = 0.17). Paired on the same episodes
(`experiments/baseline_v3/seed1000/paired_vs_others.json`):

| Run | Hidden /500 | Successes only under v3 / only under the other run | McNemar exact p |
| --- | --- | --- | --- |
| `lerobot-train` control, seed 1000 | 100 | 46 / 64 | 0.105 |
| Released checkpoint | 101 | 42 / 61 | 0.076 |
| `aloha-train` v2, seed 1000 | 60 | 54 / 32 | 0.023 |
| `aloha-train` v1, seed 1000 | 68 | 48 / 34 | 0.15 |
| `aloha-train` v1, seed 1001 | 76 | 53 / 47 | 0.62 |
| `aloha-train` v1, seed 1002 | 65 | 57 / 40 | 0.10 |

One seed separates v3 from neither side, so the rule asks for a second seed. The thresholds for pooled
seeds were committed before that seed started (`scripts/v3_decision.py`): out of 1000, at least 168 is
significantly above v1 and at most 159 significantly below the control; out of 1500, at least 247 and
at most 242. All v3 seeds share one frame order, so pooling reduces initialisation and evaluation
noise but not the chance in that order. The control is a single seed, so even three v3 seeds would
detect only gaps above about 3.9 pp.

The hidden and proxy blocks rank v3 seed 1000 and the control in opposite orders (82 against 100, 24
against 12). At these rates 100 episodes have a standard error of about 4 pp, so the proxy cannot rank
runs this close.

The second seed scored 59/500 (11.8%, Wilson 9.3-14.9%) on the hidden block and 11/100 on the proxy
block, with levels 121/102/21/197/59 (2026-11-09; training 15,562 s). Pooled: 141/1000 = 14.10% (Wilson
12.1-16.4%), below the 159 threshold. Under the committed rule the draw therefore does not suffice to
explain the gap on this evidence, and the next step is to replay the control's frame order, not to run a
third seed (`experiments/baseline_v3/decision_after_seed1001.json`).

Three further observations do not change that decision.
- The two v3 seeds share a frame order and differ only in the training seed, yet scored 82 and 59
  (two-proportion p = 0.037, McNemar p = 0.019, s.d. 3.25 pp against a binomial 1.56 pp). The v1 seeds
  were tight, and the two groups together give homogeneity p = 0.14, so variance above binomial noise
  is a sign, not a finding. If it is real, the test against the control overstates significance: with
  the observed spread the 5.90 pp gap gives z = 1.48 (p = 0.14), not z = 2.93 (p = 0.003).
- The v3 pool, 14.10%, is almost the same as the pool of the v1 and v2 runs with independent draws,
  269/2000 = 13.45%.
- The v3 schedule and `lerobot-train`'s order show the same structure: 32 exposures per frame, the same
  distribution of draws between exposures of a frame (mean 25,000, s.d. 10,203), 7.46 distinct episodes
  per batch, and the same time statistics per batch and per block
  (`experiments/order_replay/order_structure.txt`). Independent draws differ on the exposure measures
  (11-55 exposures, gap s.d. 24,189).

With the trainers bit-identical over one epoch and the draws alike in distribution, we know of no
systematic difference left between v3 and the control. Two explanations remain: the variance of a
single training run, or a difference that one epoch of parity cannot reveal. All runs taken together
before the replay favour the second (`scripts/uniform_runs_spread.py`; the numbers with the replay
follow below).
- The seven uniform runs of `aloha-train` (three v1 seeds, the RSI-Harness re-run of v1 seed 1000, v2
  and two v3 seeds) pool to 479/3500 = 13.69%. Their spread, 1.66 pp, is close to the binomial 1.54 pp
  (homogeneity p = 0.32; dispersion factor 1.17, 95% interval 0.48-5.66), and 82 against 59 is the most
  extreme pair among them. Pooling assumes the three protocols share one success rate, which the data
  do not contradict.
- The two LeRobot-trained models, the control (100) and the released checkpoint (101, older software
  stack), pool to 20.10%, 6.41 pp higher. Against the observed run-to-run spread t = 5.20 (p = 0.0013);
  with the dispersion at its upper bound, p = 0.029.
- A lucky control would need both LeRobot runs to be lucky. The control's earlier checkpoints also
  score high and rise steadily: 84, 94 and 100 of 500 at 60,000, 80,000 and 100,000 steps
  (`experiments/control_lerobot_train/seed1000/checkpoint_evals/`).

The one-epoch parity did not exercise two things the control did: crossing an epoch boundary (31 times)
and saving a checkpoint mid-run (every 20,000 steps). A two-epoch parity check with a save at the
boundary, its reading committed before it started (docs/plan.md, step 1b), was again bit-identical:
234 of 234 tensors after 6,250 steps (`experiments/trainer_parity/steps6250/result.json`). Neither the
boundary nor the save changes training; a difference that appears later is not yet excluded.

The replay trains `aloha-train` on the control's exact frame order at seed 1000 in normal mode
(`scripts/run_order_replay.sh`). Before training it checks the rebuilt order against `lerobot-train`'s
real loader chain over three epochs. The reading was committed before it ran (docs/plan.md, step 1b): at
least 88/500 suggests that this frame order matters, and the same order is then replayed with another
training seed; at most 87/500 means the control's 100/500 did not clearly reproduce from the same
inputs. In that case the full-length parity check comes first (added before the result, given the
analysis above). One run cannot settle this either way.

The replay scored 66/500 (13.2%, Wilson 10.5-16.4%) on the hidden block and 14/100 on the proxy block,
with levels 74/148/40/172/66 (2026-11-09; training 18,477 s, about 73 minutes of it alongside the
two-epoch parity check).

| Run | Hidden /500 | Successes only in the replay / only in the other run | McNemar exact p |
| --- | --- | --- | --- |
| `lerobot-train` control, seed 1000 (same order and seed) | 100 | 25 / 59 | 0.0003 |
| Released checkpoint | 101 | 28 / 63 | 0.0003 |
| `aloha-train` v3, seed 1000 | 82 | 35 / 51 | 0.11 |
| `aloha-train` v3, seed 1001 | 59 | 47 / 40 | 0.52 |
| `aloha-train` v2, seed 1000 | 60 | 42 / 36 | 0.57 |
| `aloha-train` v1, seeds 1000, 1001, 1002 | 68, 76, 65 | 39/41, 44/54, 44/43 | 0.91, 0.36, 1 |

Under the committed rule (at most 76) the control did not reproduce from the same inputs, and the frame
order is excluded: our trainer, given the control's own order, scores at its usual level
(`experiments/order_replay/seed1000/paired_vs_others.json`). Three further checks point the same way.
- With the replay, the eight uniform runs of `aloha-train` pool to 545/4000 = 13.63%. Their spread,
  1.55 pp, equals the binomial 1.53 pp (homogeneity p = 0.42; dispersion 1.02, 95% interval
  0.44-4.21). The two LeRobot-trained models are 6.48 pp higher: t = 5.66 against the observed spread
  (p = 0.0005), p = 0.009 with the dispersion at its upper bound. The replay meets the inclusion rule
  stated in the script before its result (uniform sampling, full budget, hidden block). Run-to-run
  variance above binomial noise is thus ruled out as the explanation for our trainer; for
  `lerobot-train`, which has run once, it is unmeasured.
- The saved inference files match again: `config.json` differs only in `push_to_hub`, the processor
  files are identical, and both normaliser tensors files have the same SHA-256 in the replay, the
  control and both models of the two-epoch check. The evaluator reads only these files, not the
  `train_config.json` that `lerobot-train` also writes.
- With the same frame order, each 200-step log window covers the same batches in both runs. In every
  10,000-step block the replay's training loss is within 0.0-0.3% of the control's and its gradient
  norm within 0.0-0.7% (`scripts/paired_loss_curves.py`,
  `experiments/order_replay/seed1000/paired_loss_vs_control.json`; on the bit-identical 6,250-step pair
  every difference is within the log's rounding). Both trainers fit the training data equally well;
  the gap does not show in the loss. This is descriptive and enters no decision.

Two explanations remain: a difference that appears only after 6,250 steps or only in normal mode
(all parity checks ran in deterministic mode, all scored runs in normal mode), or a lucky control. The
full-length deterministic parity check (`scripts/run_full_parity.sh`, 100,000 steps with saves every
20,000 like the control, then scoring) tests the first. Its reading was committed before it runs
(docs/plan.md, step 1b): bit-identical weights scoring at least 88/500 point to normal mode (or chance),
and a second deterministic seed follows; if that is also high, running the evaluator in deterministic
mode (as fast on this GPU) would make its uniform baseline the official launch path itself.
Bit-identical weights scoring at most 87/500 point to a lucky control or an effect only of
`lerobot-train`'s normal mode, and `lerobot-train` is re-run unchanged. Weights that differ mean a
later divergence, which is located before anything else runs. A third v3 seed is deferred until the
protocol is settled.

Result (2026-11-09 22:48 to 11-10 08:14, exit 0): after all 100,000 steps the weights are
bit-identical (234 of 234 tensors, `experiments/trainer_parity/steps100000/result.json`). The one
model scores 100/500 on the hidden block (20.0%, Wilson 16.7-23.7%) and 13/100 on the proxy block.
Paired by episode it is indistinguishable from the control (43 against 43 discordant episodes,
McNemar p = 1) and from the released checkpoint (53 against 54, p = 1). It is above the normal-mode
replay of the same order and seed (52 against 18, p = 6e-5) and above every other run of our trainer
(p <= 0.031) except v3 seed 1000 (55 against 37, p = 0.076)
(`experiments/trainer_parity/steps100000/paired_runs.json`). It shares only 57 successful episodes
with the control: two different models of equal skill. By the committed reading the gap appears only
in normal mode, or by chance. Its cause is unknown; both trainers set the same normal-mode flags
(cuDNN autotuning, TF32 matrix products).

The second deterministic seed was written down before it started (docs/plan.md, step 1b). Our trainer
runs on the Judge's own uniform schedule at training seed 1001, so it differs from the normal-mode v3
seed 1001 run (59/500) only in the cuDNN mode (same protocol file as the parity check, same
`schedule.npy`). At least 88/500: discuss with the user making the Judge train in deterministic mode
(protocol v4), whose uniform baseline is then the official launch path with
`--cudnn_deterministic=true`, bit for bit. At most 87/500: re-run `lerobot-train` unchanged to measure
its own spread. The final significance threshold was fixed at the same time: the current
`scripts/baseline_stats.py` formula on the final protocol's three uniform seeds only.

Result (2026-11-10 10:44 to 15:40, exit 0; `experiments/baseline_v3_det/seed1001`): 64/500 on the
hidden block (12.8%, Wilson 10.2-16.0%), 16/100 on the proxy block; `train_info.json` records
`cudnn_deterministic: true`, protocol 3, seed 1001, 16,032 s of training (6.24 steps/s). Paired by
episode it is indistinguishable from the normal-mode v3 seed 1001 run (36 against 31 discordant
episodes, McNemar p = 0.63) and from the other runs of our trainer (p >= 0.063), and below the parity
model, the control and the released checkpoint (33 against 69, p = 0.0005; 35 against 72,
p = 0.0004) (`experiments/baseline_v3_det/seed1001/paired_runs.json`). By the committed reading
(64 <= 87) deterministic mode does not lift the uniform baseline. The nine uniform runs of our
trainer pool to 609/4500 = 13.53%, with a run-to-run s.d. of 1.47 pp against 1.53 pp of binomial
noise (dispersion 0.93, 95% 0.42-3.40, homogeneity p = 0.49); the two LeRobot-trained models are
6.57 pp higher (t test on run rates, p = 0.0002) (`experiments/baseline_v3_det/uniform_runs_spread.json`). This
run came from a trainer that is bit-identical to `lerobot-train` in this mode; only the training
seed and one uniform shuffled frame order differ from the parity run. Every model near 20% is a
seed-1000 trajectory: the control, the parity model, and the released checkpoint, whose
`train_config.json` also records seed 1000. The official recipe's 20% is therefore so far the score
of one seed, and its seed-to-seed spread has never been measured. Three explanations remain: seed
1000 happens to be a good trajectory, a difference between the trainers that appears only in
normal mode, or a difference between the Judge's compiled uniform order and `lerobot-train`'s
loader order that the order statistics did not show.

The next run, `lerobot-train` re-run unchanged (seed 1000, normal mode, the control's exact command
and environment: `scripts/run_lerobot_train_control.sh 1000 experiments/control_lerobot_train_rerun`),
was written down before it started (docs/plan.md, step 1b). R is its hidden successes; the threshold
stays 88. Its final weights are first compared bit for bit with the control's and with the parity
model's (`scripts/trainer_parity.py compare`). Bit-identical to the control: `lerobot-train`'s normal
mode is reproducible on this GPU and R is the control's 100 up to evaluation noise, so our trainer's
normal-mode replay (66) differs from it on identical inputs, and that difference is located (weights
compared in normal mode after 300 and 6,250 steps) before anything else runs. Different weights and
R >= 88: the seed-1000 trajectory stays high under cuDNN perturbation while our replay scored 66
(p = 0.0003 against the control), so a normal-mode trainer difference remains the leading
explanation and is located first; `lerobot-train` in deterministic mode at seed 1001 then separates
a lucky seed from a 20% recipe. Different weights and R <= 87: the official recipe's own normal-mode
runs at one seed and order span 100 and at most 87, a trainer difference is not established, the
Judge keeps its trainer, and `lerobot-train` in deterministic mode at seed 1001 measures the recipe's
seed-to-seed spread before the protocol is settled with the user. Descriptive only: R's pairings with
the control, the parity model, the replay and the deterministic seed 1001 run, and the scores of its
60,000- and 80,000-step checkpoints (the control's were 84 and 94).

Result (2026-11-10 15:59 to 20:55, exit 0; `experiments/control_lerobot_train_rerun/seed1000`):
102/500 on the hidden block (20.4%, Wilson 17.1-24.2%), 15/100 on the proxy block; 16,085 s of
training (6.22 steps/s), `cudnn_deterministic: False` as in the control. Its weights differ from the
control's at every saved checkpoint (153 of 234 tensors at 20,000 steps already; the 81 identical
tensors are untrained buffers): the relative Frobenius distance grows from 1.0% at 20,000 steps to
3.0% at 100,000, the same distance as from either run to the deterministic parity model (3.0% and
3.1%) (`experiments/control_lerobot_train_rerun/seed1000/weights/`). Normal mode therefore perturbs
`lerobot-train` from the start, and after 100,000 steps two of its own runs are as far apart as either
is from the deterministic model, so a weight comparison in normal mode can show that two runs differ
but not which trainer deviates. Paired by episode the re-run is indistinguishable from the control
(43 against 41 discordant episodes, McNemar p = 0.91), the parity model (44 against 42, p = 0.91) and
the released checkpoint (64 against 63, p = 1), and above the normal-mode replay (59 against 23,
p = 9e-5), the deterministic seed 1001 run (76 against 38, p = 0.0005) and every other run of our
trainer (p <= 0.014) except v3 seed 1000 (62 against 42, p = 0.062) (`paired_runs.json`). By the
committed reading (different weights, R = 102 >= 88): the seed-1000 trajectory of `lerobot-train` is
high under normal-mode perturbation (100, 102), while our trainer's normal-mode replay of the same seed
and order (66) stays unexplained. A code audit finds no difference in the normal-mode switches:
`train.py` lines 198-203 of our trainer and `lerobot_train.py` lines 287-292 of LeRobot both enable
`cudnn.benchmark` in normal mode, `cudnn.deterministic` without benchmarking in deterministic mode,
and TF32 in both. The three LeRobot-trained models pool to 303/1500 = 20.20%, 6.67 pp above the nine
uniform runs of our trainer (t test on run rates, df 10, p = 2e-5; p = 0.0004 at the dispersion's
upper bound) (`experiments/control_lerobot_train_rerun/uniform_runs_spread.json`), but they are three
models of one seed, not three seeds.

The localisation the rule asks for was adapted before it ran (docs/plan.md, step 1b): because two
normal-mode runs of one trainer already diverge at the first checkpoint, "the first step at which the
trainers diverge" does not exist in normal mode, and only the size of the divergence can be compared.
`scripts/check_normal_mode_noise.sh 300` trains the same seed-1000 order for 300 steps twice with each
trainer in normal mode and compares all six pairs of final weights (`scripts/trainer_parity.py
compare` now also reports the root-mean-square difference and the relative Frobenius distance).
Reading, with W the larger within-trainer distance and X each cross-trainer distance: all six pairs
bit-identical, repeat at 6,250 steps; within-trainer pairs identical but a cross-trainer pair
different, a systematic normal-mode difference exists and is located (cuDNN algorithm logs) before any
baseline runs; within-trainer pairs different and every X within a factor of 10 of W over the same
tensors, the trainers share the same step-by-step noise, the 6,250-step comparison is dropped as
uninformative, and the normal-mode hypothesis can only be tested by outcome, with a second normal-mode
replay queued after the deterministic seed-1001 run; within-trainer pairs different but some X more
than 10x W, a systematic difference is located first. This check changes no decision above.

The decisive run, written down before it starts (docs/plan.md, step 1b): `lerobot-train` in
deterministic mode at seed 1001 (`DETERMINISTIC=1 scripts/run_lerobot_train_control.sh 1001
experiments/control_lerobot_train_det`), the official launch path with `--cudnn_deterministic=true`,
the mode in which it is bit-identical to our trainer, its own seed-1001 loader order, saves every
20,000 steps, then the hidden and proxy blocks. D3 is its hidden successes; the threshold stays 88. It
differs from the deterministic seed-1001 run of our trainer (64/500) only in the frame order: same
trainer code, same seed and initialisation, same mode. D3 >= 88: the recipe is 20% at a second seed, a
lucky seed is excluded, and the gap lies in the frame order itself (the Judge's compiled uniform order
against `lerobot-train`'s loader order); the compiled order is then examined against
`EpisodeAwareSampler` (whole epochs without replacement, epoch boundaries, dropped frames) with a rule
committed first, and the baseline protocol adopts the loader order or a corrected compiler before the
protocol is settled with the user and three seeds run. D3 <= 87: the recipe scores like our runs at
seed 1001, seed 1000 was a high draw of the recipe's own seed-to-seed spread (100 against D3), the gap
is seed variance rather than a trainer defect, the Judge's uniform baseline is representative, and the
replay (66) is recorded as one normal-mode draw; the protocol (deterministic Judge or not, which three
seeds) is then settled with the user and the seed-to-seed spread goes into the proposal. Descriptive
only: D3's pairings with the 64, the control, the re-run, the parity model and the released
checkpoint; the scores of its 60,000- and 80,000-step checkpoints; and the re-run's own 60,000- and
80,000-step checkpoints (`RUN=experiments/control_lerobot_train_rerun/seed1000
scripts/eval_control_checkpoints.sh 60000 80000`), scored after it.

Result (2026-11-11, exit 0; logs/lerobot-det-seed1001.log; experiments/control_lerobot_train_det/seed1001,
whose train_config.json records `cudnn_deterministic: true` and seed 1001; 16,266 s of training at
6.15 steps/s): D3 = 70/500 = 14.0% (Wilson 11.2-17.3%), proxy 14/100, reward-level histogram
85/122/36/187/70. Paired by episode it is indistinguishable from our trainer's deterministic seed-1001
run on the Judge's compiled uniform order (64: 39 against 33 discordant, McNemar p = 0.56), from the
normal-mode replay (66, p = 0.75) and from every uniform run of our trainer (p = 0.25-0.91), and it is
below the control (p = 0.0042), the re-run (p = 0.0048), the parity model (p = 0.0039) and the released
checkpoint (p = 0.001). Against the pooled v1 baseline it is +0.07 pp (p = 0.97), below the 88
threshold. Under the rule above (D3 <= 87): the official launch path scores like our trainer at seed
1001, so seed 1000 was a high draw of the recipe's own seed-to-seed spread (100 and 102 at seed 1000
against 70 at seed 1001; 100 against 70 in the same deterministic mode), the gap of about 6.5 pp
between the LeRobot-trained seed-1000 models and our uniform runs is seed variance rather than a
trainer defect, and the frame order is not the cause (the Judge's compiled uniform order and
`lerobot-train`'s loader order give 64 and 70 at the same seed and mode). The Judge's uniform baseline
(nine runs, 609/4500 = 13.53%) is therefore representative of the recipe at seeds other than 1000; the
replay (66) is recorded as one normal-mode draw, and the second normal-mode replay runs as queued with
its rule unchanged. Descriptively, the four LeRobot-trained models pool to 373/2000 = 18.65%, +5.12 pp
over the nine uniform runs (t p = 0.0016), but three of the four are seed 1000
(experiments/control_lerobot_train_det/uniform_runs_spread.json). The protocol is settled with the user
next: a deterministic Judge (same speed here, bit-reproducible scoring, supported by either branch of
the second replay) and three training seeds, proposed 1000/1001/1002 (the deterministic v3 seed 1001
exists; two more cost about 10 h), after which the threshold is computed from those three seeds only
and the seed-to-seed spread goes into the proposal. The re-run's 60,000- and 80,000-step checkpoints
score 90 and 93 (control 84 and 94; both runs 100-102 at 100,000 steps): both seed-1000 runs are near
18-20% from 60,000 steps on (descriptive; logs/rerun-ckpt-evals.log). D3's 60,000- and 80,000-step
checkpoints, scored on 2026-11-11 (exit 0; logs/det-ckpt-evals.log;
experiments/control_lerobot_train_det/seed1001/checkpoint_evals/), give 65 and 75 (final model 70): the
seed-1001 trajectory sits at 13-15% from 60,000 steps on while the two seed-1000 trajectories sit at
17-20%, so the seed difference is present through the second half of training rather than appearing at
the end (descriptive; enters no decision).

Second normal-mode replay (2026-11-11, exit 0; logs/order-replay2.log; experiments/order_replay2/seed1000;
17,773 s from launch to the end of both evaluations; rule committed before it ran, above): R2 = 70/500 =
14.0% (Wilson 11.2-17.3%), proxy 14/100, reward-level histogram 81/129/34/186/70. Paired by episode it is
indistinguishable from the first replay (66: 45 against 41 discordant, McNemar p = 0.75), from D3 (70: 47
against 47, p = 1), from the deterministic seed-1001 run (64, p = 0.60) and from every other uniform run
(p = 0.23-0.92), and below the control (p = 0.0021), the re-run (p = 0.0007), the parity model
(p = 0.0012) and the released checkpoint (p = 0.0053); +0.07 pp against the pooled v1 baseline
(p = 0.97), below the 88 threshold. Under the committed rule (R2 <= 87): the four normal-mode draws of
the same input split by trainer, our trainer 66 and 70 against `lerobot-train` 100 and 102, so in normal
mode the two trainers differ in a way the 300-step weight comparison cannot see (candidate: the cuDNN
benchmark choosing different algorithms or precisions in the two processes), and the Judge's baseline
protocol must use deterministic mode, where the trainers are bit-identical, until this is resolved. This
does not alter the reading of D3, which rests on deterministic mode (100 at seed 1000 against 70 at seed
1001 with bit-identical trainers); it says that our trainer's normal mode does not reproduce
`lerobot-train`'s normal mode on the seed-1000 trajectory, a difference that deterministic mode removes.
Caveat: with four draws, a split by trainer has probability 1/6 under exchangeability, so the split alone
is weak evidence; the two tight pairs (66/70 and 100/102) more than 30 episodes apart make a systematic
difference the more economical reading. With this run the ten uniform runs of our trainer pool to
679/5000 = 13.58% (run s.d. 1.40 pp against binomial 1.53 pp, homogeneity p = 0.59, dispersion 0.83)
against 373/2000 = 18.65% for the four LeRobot-trained models (+5.07 pp, t p = 0.0009; three of the four
seed 1000; descriptive; experiments/order_replay2/uniform_runs_spread.json).

Algorithm comparison, reading committed before the run (2026-11-11 morning;
`scripts/check_normal_mode_algos.sh`, `scripts/normal_mode_algos.py`): the four 300-step runs of the
noise check again (two per trainer, one order, seed 1000, normal mode), each launched unchanged inside a
wrapper whose only hook counts backward calls; after step 200 it records two complete steps of GPU
kernel names and counts with torch.profiler (the name carries the cuDNN or cuBLAS algorithm and its TF32
or FP32 variant), every cuBLAS and cuBLASLt matmul call through the libraries' own runtime log switches
(shape, algorithm id, compute type), and the effective torch switches. cuDNN's own API log is not used:
formatting it inside the calls that cudnn.benchmark times would perturb the choice it is meant to
observe, and on cuDNN 9.19 it cannot be switched on at runtime (cudnnSetCallback does not reach the graph
API, with or without a user callback; tried the same morning). The benchmark choice is made in step 1,
before anything is recorded. Reading: (i) any switch differs between the processes (TF32 for matmul or
cuDNN, benchmark, deterministic, matmul precision, NVIDIA_TF32_OVERRIDE): the difference is
configuration, named; a confirming 5-h replay with that switch matched (at least 88 confirms, at most 87
refutes) is the user's call, since the final protocol uses deterministic mode anyway. (ii) Switches
identical, each trainer's two runs identical in their conv-type and matmul-type kernel multisets, the
trainers different: the two trainers stably choose different algorithms or precisions in normal mode,
the most likely source of the normal-mode-only difference; the differing kernels are recorded, the
confirming replay is the user's call, the Judge uses deterministic mode. (iii) Switches identical but a
trainer's two runs differ from each other: the benchmark choice is unstable within a trainer and this
comparison cannot separate the trainers; recorded; the Judge uses deterministic mode. (iv) Everything
identical (switches, conv and matmul kernels, cuBLAS algorithms and precisions): no systematic
normal-mode difference is visible in two steps, the low/high split of the four draws stays unexplained
(a 1/6 coincidence, or a difference that appears after 300 steps), the Judge uses deterministic mode and
no further 5-h run is spent on it. Descriptive only: differences among the other kernels (elementwise,
optimizer, copies), total launches, the profiler's time table.

Result (2026-11-11, exit 0; logs/normal-algos.log; experiments/trainer_parity/normal_algos300/summary.json;
the four runs use the noise check's order byte for byte): the switches are identical in all four
processes (benchmark on, deterministic off, TF32 on for cuDNN and matmul, matmul precision "high", no
overriding environment variables); the matmul-type kernels (520 launches per two steps), the cuBLAS
calls (44 distinct) and the cuBLASLt calls (58 distinct) are identical in all four; the conv-type
kernels (386-398 launches) differ in every pair: within a trainer aloha_a against aloha_b in 11 entries
(summed count difference 56) and lerobot_a against lerobot_b in 17 (78), across trainers in 6, 15, 11
and 17 entries (22, 68, 58, 92), and the most similar pair is a cross-trainer one (aloha_a against
lerobot_a). The choices differ in precision as well as tiling: every run launches 24-32 FP32 CUDA-core
`scudnn` kernels (Winograd among them) next to 78-86 TF32 tensor-core kernels, and which convolutions
run in FP32 and which in TF32 changes from run to run (lerobot_b alone uses `scudnn_128x128_relu_small`
and more split-K). Under the committed reading, branch (iii): the benchmark choice is unstable within a
trainer, so this comparison cannot separate the trainers; recorded, and the Judge uses deterministic
mode. The remaining kernels differ only in copies and one scalar multiply (host-to-device copies 18
against 20, memsets 128/140, a scalar multiply 6 against 10 per two steps, our trainer against
`lerobot-train`), consistent with bit-identical deterministic parity and unrelated to algorithms. So
normal mode is a different numerical path in every run, not merely a different accumulation order,
which matches the noise check (within-trainer weight distance equal to cross-trainer). The 66/70
against 100/102 split stays unexplained (a 1/6 coincidence, or a difference outside these two steps or
invisible in kernel names); per the rule no further 5-h run is spent on it.

**Final protocol.** Fixed on 2026-11-11 12:33 with the user, before the remaining runs started, and
recorded in docs/plan.md: the Judge trains in deterministic mode (`protocol_deterministic.json`, the v3
`fixed_train.json` with only `cudnn_deterministic: true` changed, byte-identical to the file the
deterministic seed-1001 run used), the uniform baseline is seeds 1000, 1001 and 1002, and the
significance threshold comes from those three seeds only through the committed `baseline_stats.py`
formula; the reading was written before the runs started and has no decision branch, so no score
triggers a re-run or a change of seed. Seed 1000 (2026-11-11 12:44 to 17:50, exit 0;
logs/det-seed1000.log; `experiments/baseline_v3_det/seed1000`): 80/500 = 16.0% (Wilson 13.0-19.5%),
proxy 14/100, reward-level histogram 96/109/19/196/80; 16,190 s of training (6.18 steps/s), 1,743 s
hidden and 415 s proxy evaluation; `train.json` records `cudnn_deterministic: true`, protocol 3,
seed 1000, and `schedule.npy` is byte-identical to the normal-mode v3 seed-1000 run's, so the two
differ only in the cuDNN mode. Paired by episode (`paired_runs.json`): against that normal-mode run
(82) 36 against 38 discordant episodes, McNemar p = 0.91; against the deterministic seed-1001 run (64)
49 against 33, p = 0.097; against `lerobot-train` deterministic seed 1001 (70) p = 0.34; against the
control, its re-run, the parity model and the released checkpoint (100, 102, 100, 101) p = 0.052,
0.035, 0.045 and 0.050. The eleven uniform runs of our trainer now pool to 759/5500 = 13.80% (run s.d.
1.51 pp against 1.54 pp binomial, dispersion 0.96, 95% 0.47-2.96, homogeneity p = 0.48); the four
LeRobot-trained models (373/2000 = 18.65%, three of them seed 1000) sit 4.85 pp higher (t test on run
rates, p = 0.0011) (`experiments/baseline_v3_det/uniform_runs_spread.json`). Seed 1001 under the same
protocol is the 64/500 above. Seed 1002 (2026-11-11 20:56 to 2026-11-12 02:30, exit 0;
logs/det-seed1002.log; `experiments/baseline_v3_det/seed1002`): 57/500 = 11.4% (Wilson 8.9-14.5%),
proxy 16/100, histogram 97/113/29/204/57; 17,556 s of training (5.70 steps/s, slower than the other
two seeds, not investigated; in deterministic mode the score depends only on the seed and the data),
2,087 s hidden and 378 s proxy evaluation; `schedule.npy` is identical to the other two seeds' (the
Judge's compiled uniform order does not depend on the training seed, so the three runs differ only in
the training seed). Paired by episode: against seed 1000 (80) 35 against 58 discordant, McNemar
p = 0.022; against seed 1001 (64) p = 0.50; against the control and the released checkpoint
p = 5e-5 and 2e-5 (`paired_runs.json`).

**Final baseline and threshold** (`scripts/baseline_stats.py --baseline-dir experiments/baseline_v3_det`,
exit 0, 2026-11-12; `experiments/baseline_v3_det/summary.json`; the formula is the one committed in
98ff41f, applied to these three seeds only): 80, 64 and 57 of 500, pooled 201/1500 = **13.40%**
(Wilson 11.8-15.2%); mean of seeds 13.40% with s.d. 2.36 pp against a single-run binomial s.e. of
1.52 pp; homogeneity chi-square 4.79, df 2, p = 0.091; training-seed variance estimate 3.24e-4; the
two-sided 95% threshold against the pooled baseline is 4.93 pp, so a single Judge run needs
**>= 92/500**. Under the committed reading there is no decision branch: this is the task's baseline,
and the seed-to-seed spread (23 episodes between the best and the worst seed) is reported as
measured. All twelve uniform runs of our trainer across protocols pool to 816/6000 = 13.60% (run
s.d. 1.60 pp against 1.53 pp binomial, dispersion 1.09, 95% 0.55-3.14, homogeneity p = 0.37); the
five protocol-v3 runs alone (normal mode 82/59, deterministic 80/64/57) show dispersion 2.36 (95%
0.85-19.5, p = 0.051), descriptive; the four LeRobot-trained models (373/2000 = 18.65%, three of them
seed 1000) sit 5.05 pp above the twelve (t test on run rates, p = 0.0007).

**Machine fault on 2026-11-12 and the check of seed 1002.** The development machine's kernel logged
memory corruption from 01:34 on ("Bad rss-counter state" twice, "Bad page state" once) and hung at
04:09 on a kernel pointer that differed from a valid one in a single bit, with this project's run and
another GPU job sharing the machine after 3.4 days of continuous load; no earlier boot since
2026-10-21 shows such errors. The seed-1002 run trained until 01:49 and was evaluated until 02:30,
so its last 15 minutes of training and its whole evaluation fall into the faulty period; seeds 1000
and 1001 finished 8 hours or more before the first fault. The first final-protocol oracle run was
lost in the hang, 62 minutes into the Judge's training. After the reboot the repository, the
committed result files (recomputed and compared key by key), the dataset hashes and the software
versions all checked out. The seed-1002 run is repeated with the same command; deterministic mode
is bit-reproducible across processes on this machine (full-length parity above), so the rule,
committed before the repeat (docs/plan.md, "2026-11-12 crash"), is: identical weights (SHA-256) and
identical per-episode outcomes confirm 57/500; different weights trigger a third run, and the
result reproduced by two runs is the seed-1002 result, with the baseline and threshold recomputed if
it is not the original; three different weights stop all GPU work until the hardware is checked. The
launch scripts now refuse to start when the current boot's kernel log shows such faults and flag a
run during which they appear (`scripts/kernel_health.sh`, `scripts/run_guards.sh`), and every run
records the SHA-256 of its weights.

Result of the repeat (2026-11-12 10:40 to 15:33, exit 0, kernel log clean before and after;
`experiments/baseline_v3_det_verify/seed1002`): the weights have the same SHA-256 (`f594f15f…`) and
all 500 logged losses are equal, so the original training was not affected and stands. The
evaluation of those same weights did not repeat: 60/500 against the original 57/500, with 25 episodes
of different outcome and 396 of different reward sum. Under the committed rule a third evaluation
followed (62/500, exit 0); no two of the three are identical per episode, so the evaluation is not
bit-deterministic and seed 1002 keeps its first value, 57. Over the three evaluations 428 episodes
never succeed, 11 succeed once, 15 twice and 46 always, which puts the re-evaluation s.d. of the
count at about 2.9 episodes (0.59 pp), well inside the 7.6 episodes of the binomial term that the
threshold uses.

**Why an evaluation does not repeat.** The evaluator ran cuDNN in benchmark mode, as `lerobot-eval`
does (`lerobot_eval.py`, line 746), so each process picks convolution algorithms by timing. A switch
for a deterministic choice was added (`evaluation.cudnn_deterministic`, off by default) and tested
under a rule committed before the test, on the public proxy block only: two deterministic evaluations
of one checkpoint both scored 16/100 with no episode of different outcome, but 24 episodes differed
in reward sum and one in reward level (`experiments/eval_determinism/proxy_run1_vs_run2.json`). By
that rule the protocol stayed as it was. The remaining source is the renderer: with no policy at all,
three processes given the same seed and the same fixed actions produce bit-identical joint-state
trajectories, while 12-20 of 120 camera frames differ between any two of them, by one intensity level
in 1-5 of 307,200 pixels (`scripts/render_determinism.py`,
`experiments/eval_determinism/render_determinism.json`; NVIDIA EGL, driver 595.99.02). The policy's
output then differs in its last bits and the contact-rich simulation amplifies that. Software
rendering is about ten times slower and would push a Judge round past its time limit, and changing
the render settings would move the evaluation images away from the demonstrations', so this source is
left as it is. Consequences: the Judge's training is bit-reproducible and its evaluation carries a
re-evaluation noise of about 0.6 pp; the threshold is unchanged because its binomial term bounds the
variance of a count over fixed seeds with per-episode success probabilities; an identical candidate
submitted twice scores within that noise, which is one more reason to confirm a best candidate with
independent training seeds; and the oracle check compares logged losses for the training and uses a
paired test for the outcomes.

**Production path under the final protocol.** The RSI-Harness oracle run was repeated on 2026-11-12
(run `8f721dab03844104a15b295fbfd30535`, 16:19 to 21:22, `rsi-harness` exit 0, kernel log clean before
and after; evidence in `experiments/oracle_final/`). The harness rebuilt the task images from
scratch, the oracle agent submitted the default uniform policy once, and the Judge trained and
evaluated it in a clean container: status `completed`, reward 0.162 = 81/500, one Judge round
17,495 s (training 16,134 s, evaluation 1,346 s). The reading had been corrected before the result,
once it was known that evaluations do not repeat: identical logged losses mean the container
reproduces the host's training, and the outcomes are then compared with a paired test. Against the
host baseline run at seed 1000 (80/500) the schedule hash is the same and all 500 logged losses and
the final loss mean are equal to the last digit, with the container differing from the host only in
the Python patch version (3.12.3 against 3.12.15). The outcomes share 78 successes with 2 host-only
and 3 Judge-only (McNemar exact p = 1.0); 490 episodes reach the same reward level and 83 differ in
reward sum, the same order as two evaluations of one checkpoint on the host
(`experiments/oracle_final/oracle_vs_host_seed1000.json`). So the official path reproduces the host
training bit for bit and its score within re-evaluation noise; the oracle score enters neither the
baseline nor the threshold.

The v1 numbers in Sections 5-7 stay as recorded until then. The task image must be rebuilt and the
oracle re-run under the final protocol before any harness run.

## 6. Temporal structure of the demonstrations

Computed from joint states and gripper commands over all 50 episodes:

| Event | Mean frame (min-max) |
| --- | --- |
| Left gripper grasps the socket | 125 (95-180) |
| Both grippers holding | 219 (171-293) |
| End of task motion (25-frame mean joint speed < 0.5 rad/s afterwards) | 334 (238-494) |

92% of frames 400-499 and on average the last ~166 frames of each demonstration are post-task
holding. Under uniform sampling about a third of all exposures fall on these anchors.

## 7. Author probes

Two hand-designed curricula test whether the action space moves the score. They are task-author
probes, not agent results. Both first ran under protocol v1 and were re-run unchanged under the
final protocol on 2026-11-12/13 (results at the end of this section). Both were committed before either ran (`eef0ed6`, 2026-11-07 17:19,
`experiments/probes/*/candidate/`). Each runs the Judge's protocol on the host
(`scripts/run_candidate.sh`: training seed 1000, the 500 hidden seeds, then the 100 public proxy
seeds) and is scored by `scripts/candidate_stats.py` against the pre-declared threshold of
Section 5.3.

| Probe | Hypothesis | Eligible frames | Hidden successes / 500 | Wilson 95% | vs pooled 13.93% | Proxy / 100 |
| --- | --- | --- | --- | --- | --- | --- |
| drop-idle-tail | Reallocating budget away from post-task holding raises success | 71.8% | 74 (14.8%) | 12.0-18.2% | +0.87 pp, z = 0.48, p = 0.63 | 21 |
| insertion-focus | Same eligibility; after 30% of training, 3x weight on the bimanual phase emphasises contact-rich insertion | 71.8% | 78 (15.6%) | 12.7-19.0% | +1.67 pp, z = 0.93, p = 0.35 | 15 |

**drop-idle-tail.** The mask keeps anchors up to 25 frames after the end of task motion (the last
frame whose centred 25-frame mean joint speed exceeds 0.5 rad/s). The compiled schedule (SHA-256
`6957178b...46bc7789`) moves the 28% of exposures that uniform sampling places on later frames to
earlier ones. Each of the first five deciles of normalised episode time gets about 111,000
exposures instead of 80,000, and the last decile gets 6,854. The share of anchors whose 100-step
chunk runs past the episode end falls from 19.8% to 3.4%. Training used the full budget (100,000
steps, 800,000 exposures, 4.44 h), and the hidden evaluation used the Judge's seed block.

The probe scored 74/500, which is below the 88/500 threshold, so H1 is not supported at this
sample size. Against the pooled baseline the difference is +0.87 pp, with a 95% interval of -2.6
to +4.4 pp (normal approximation, s.e. 1.79 pp). Paired against the host uniform run at the same
training seed, 28 successes are shared, 40 occur only under uniform sampling and 46 only under
drop-idle-tail (McNemar exact p = 0.59). The reward-level histogram 67/133/29/197/74 differs from
the pooled baseline at chi-square 8.21, df 4, p = 0.084. It is, however, close to the uniform
Judge replicate's 74/132/28/197/69, which itself gives p = 0.22 against the pooled histogram, so it
is no evidence of a shift between phases either.

The public proxy gave 21/100, above all three baselines (10, 17 and 15). A 100-episode proxy
therefore suggested a gain that the hidden block does not confirm, which is the noise an agent
has to reason about when it chooses what to submit. Evidence: `experiments/probes/drop_idle/`
(`run_seed1000/stats.json`, the schedule, training and evaluation records; the checkpoint is not
committed) and `logs/probe-drop_idle.log`.

**insertion-focus.** The probe uses the same eligibility mask. For the first 600 blocks (30% of
steps) the weights are uniform. For the remaining 1,400 blocks, frames where both grippers are
closed after having opened get weight 3 (schedule SHA-256 `3ceea10b...97964103`). Evaluating the
policy's own expressions with the task's DSL over the compiled schedule shows that the mechanism
took effect: the bimanual phase receives 65.4% of the exposures in the last 70% of training,
against 38.7% in the first stage and under drop-idle-tail. Uniform sampling gives 56.0%, but much
of that is post-task holding with both grippers closed. Training used the full budget (4.44 h).

The probe scored 78/500, below the 88/500 threshold, so H2 is not supported either. Against the
pooled baseline the difference is +1.67 pp, with a 95% interval of -1.8 to +5.2 pp. Paired against
the host uniform run, 35 successes are shared, 33 occur only under uniform sampling and 43 only
under insertion-focus (McNemar exact p = 0.30). The reward-level histogram 85/104/34/199/78
matches the pooled baseline (chi-square 0.97, df 4, p = 0.91). Moving about a quarter of the
late-training budget onto the contact-rich phase therefore did not detectably convert contact
(level 3) into insertion (level 4). The proxy gave 15/100. Evidence:
`experiments/probes/insertion_focus/` and `logs/probe-insertion_focus.log`.

**Both probes.** Neither hand-designed curriculum exceeds the threshold. Both point estimates lie
above the pooled baseline (+0.9 and +1.7 pp), and both lie inside the range that evaluator noise
alone produces. The proxy and the hidden block even order the two probes differently (21 against
15 on the proxy, 74 against 78 on the hidden block), although neither difference is significant.
The obvious mechanism-level moves therefore do not reach a gain that one 500-episode round can
detect. This does not show that no curriculum can, and whether the action space has measurable
headroom remains open.

**Under the final protocol.** The same two candidate files were re-run after the final-protocol
oracle (deterministic training, balanced exposure draws, training seed 1000; both exit 0, kernel
log clean before and after; `experiments/probes_final/`), and scored against the final pooled
baseline of 13.40% and its threshold of 92/500, with a reading committed before they ran: descriptive
headroom evidence only, never a change to the baseline, the threshold or the task.

| Probe | Schedule SHA-256 | Hidden successes / 500 | Wilson 95% | vs pooled 13.40% | Paired with uniform seed 1000 (80) | Proxy / 100 |
| --- | --- | --- | --- | --- | --- | --- |
| drop-idle-tail | `e7c5fe8d…` | 84 (16.8%) | 13.8-20.3% | +3.40 pp (95% -1.5 to +8.3), z = 1.35, p = 0.18 | 39 shared, 41 / 45 discordant, McNemar p = 0.75 | 14 |
| insertion-focus | `71ab4441…` | 88 (17.6%) | 14.5-21.2% | +4.20 pp (95% -0.7 to +9.1), z = 1.67, p = 0.095 | 37 shared, 43 / 51 discordant, McNemar p = 0.47 | 13 |

Neither reaches 92/500, so by the committed reading the conclusion is the same as under v1: the two
obvious mechanism-level moves do not reach a gain one round can detect, and headroom remains open.
Descriptively both point estimates are positive and larger than under v1 (+0.87 and +1.67 pp), and
both reward-level histograms (95/111/17/193/84 and 88/111/30/183/88) are within noise of the pooled
one (p = 0.12 each). The paired comparison with the uniform run at the same seed shares only 39 and
37 successes: a different schedule is a different training trajectory, so the pairing says little
about the mechanism. Each run took about 4.9 h (training 16,199 s and 16,013 s, hidden evaluation
1,402 s and 1,404 s).

## 8. Agent research runs (first run, stopped early)

The first run stopped on 2026-11-14 at 07:56 with about 3 h of agent time left (see below). Whether
to resume it is the contributor's decision, and no conclusion is drawn before that is settled. It
started on 2026-11-13 at 10:51 on the
contributor's explicit go (run `26d53b84a84540999c969b3e556392ae`; the agent began working at
10:53; the kernel log was clean at the start). Settings: Claude Code with
`claude-opus-5-5` at maximum reasoning (the harness README's own example), the local login, one GPU
shared by Work and Judge, a 24 h agent budget and no limit on submissions; one Judge round takes
about 4.9 h, so at most about four rounds fit. Since the oracle run only the instruction's noise
section (now the measured final-protocol figures) and the author metadata changed; `environment/`,
`tests/` and `solution/` are byte-identical.

Round 1 produced no score, because of the host rather than the candidate. The agent submitted its
first candidate at 11:26. The Judge trained it to the full budget (800,000 exposures), but the
evaluation process, started at 15:56:55, stopped after 4.2 s with "No CUDA GPUs are available", and
the harness recorded a verifier error after 16,233 s. At 12:46 a snap refresh had made systemd
reload its configuration twice. With Docker's systemd cgroup driver and `--gpus` device injection,
such a reload removes running containers' access to the GPU devices, and a running container cannot
regain it (NVIDIA Container Toolkit troubleshooting guide). The training process had opened the GPU
before the reload and finished; the new evaluation process could not open it. The Work container,
created before the reload, lost GPU access the same way, while each Judge round starts in a new
container. Under the committed reading, resuming the run (`rsi-harness resume`) was the
contributor's decision. At about 19:01 the contributor chose to resume without extra agent time.
Round 2 was then being judged (since 16:55, with Work paused). The harness ends an attempt on its
first SIGINT: it closes acceptance, lets a round it has accepted finish and records it, then stops
the agent and keeps Work as an image. A script (`scripts/stop_then_resume.sh`) sends that signal
once no round has been in flight for 30 s, so that the agent first receives the feedback of round 2,
and then resumes the run as attempt 2 in a newly built Work container. Time between attempts does
not count against the agent's 24 h.

Round 2 was judged to completion at 22:14 and scored 83/500 (16.6%; Judge training 16,216 s,
evaluation 2,891 s). Its candidate is the round-1 candidate unchanged (`c1-trim-post-insertion-focus`,
policy.json SHA-256 `09c6ea30…e9ed`, the same scores.csv): after the verifier error the agent checked
that its curriculum matched that candidate and submitted it again. 83 is below the single-round
threshold of 92. For description only: the baseline at the same training seed 1000 scored 80/500 on
the host and 81/500 in the harness oracle, and three evaluations of one set of weights gave 57, 60
and 62. No systemd reload was requested after 12:47. The script sent its single SIGINT at 22:14:51,
30 s after the round finished, and the agent had read the round's feedback by then. The attempt
ended at 22:15:11 as cancelled, with both rounds recorded and Work kept as an image. At 22:17 the
run resumed as attempt 2, in a new Work container on the same GPU and in the agent's own session,
with 40,913 s of the 86,400 s used and 45,487 s left.

Round 3 was submitted seven minutes after the resume and scored 73/500 (14.6%), also below the
threshold. Its candidate, `c5-robustness-weighted-demos` (policy.json SHA-256 `e4f3f80f…b87a`),
weights each demonstration, separately for the grasp and the insertion phase, by how robust its
recorded actions proved when replayed with perturbations in the stock evaluation environment
(according to the agent's own description and scripts). No systemd reload was requested during the
round.

Round 4 produced no score, again because of the host. The agent submitted
`c6-fast-demo-emphasis` at 03:26, three minutes after round 3 finished, and called it its final
round; according to its own description and scripts, it weights each demonstration by how early both
grippers reach the insertion pose. At 06:33 an unattended upgrade installed a glibc security update,
whose package script re-executes systemd, and systemd re-executed at 06:33:50 and 06:34:11. The
Judge's training, which had the GPU open, finished; its evaluation, started at 07:56, stopped at once
with "No CUDA GPUs are available", as in round 1 (verifier error after 16,212 s). This was the first
time on this machine that a re-execution, and not only a reload, was seen to remove a running
container's GPU access. When Work was unpaused after the round, the agent's first model call failed
with "OAuth session expired and could not be refreshed" (HTTP 401), and the agent exited (exit code 1,
not timed out) with 75,628 s of its 86,400 s used. The harness gives each attempt a copy of the
host's Claude Code login, and Work then refreshes that copy on its own. The host's login file was
rewritten at 05:15, during the round's pause, so the most likely cause is that the host's refresh
invalidated the copy's refresh token. This is an inference: the credentials were not read. The
harness recorded the run as completed after four rounds. Two were valid (83 and 73/500, both below
92), and the agent's last candidate was never scored. Under the committed reading, the completed
rounds stand and resuming is the contributor's decision.

The reading is committed before any run (docs/plan.md, step 5). Every round is reported with its
candidate, including invalid and failed ones, and the agent's trajectory is kept unedited. A round
of at least 92/500 is "above the single-round threshold", not a confirmed improvement: if every
round were another draw of the baseline (single-run s.d. 2.36 pp), one of four rounds would reach
92/500 about 7% of the time and the best of four would sit about 2.4 pp above the baseline (3.6 pp
for ten rounds; the 1.6 and 2.4 pp quoted for protocol v1 counted binomial noise only). The best
valid candidate, if it clears the threshold, is retrained on the host at training seeds 1001 and
1002 and compared with the baseline's 64 + 57 = 121/1000 at those seeds; the Judge round at seed
1000 was selected as the maximum and stays out of that test. It counts as confirmed only with at
least 168/1000 over the two retrainings (difference of two two-run means, standard error 2.36 pp,
95% margin 4.62 pp), as positive but unconfirmed with 122-167, and as not reproduced with 121 or
fewer. Without a round above the threshold the run is reported as no detectable improvement.
Ablations that remove one mechanism at a time follow only for a confirmed candidate. Such a run
uses the local task implementation and supports the proposal; the trajectory that counts in the
official flow is run later on the task repository the OpenRSI bot generates.

## 9. Reproducibility

- All versions, revisions and file hashes: `pins/versions.json`, `pins/assets.sha256`,
  `pins/requirements.freeze.txt`; image lock files `environment/requirements*.lock`.
- 71 unit tests for the DSL, schema, compiler including the balanced draw, Judge staging and the
  trainer's image path: `pytest tests_runtime`.
- Trainer parity with LeRobot (same frame order, bit-identical weights):
  `scripts/check_trainer_parity.sh [steps]`; passed at 300, 3,125 and 6,250 steps, the last with
  `SAVE_FREQ=3125` so that `lerobot-train` also saves at the epoch boundary. Full length with scoring:
  `scripts/run_full_parity.sh`; passed 2026-11-10 (bit-identical after 100,000 steps, 9.4 h including
  scoring, `experiments/trainer_parity/steps100000/`).
- GPU guards: the launch scripts refuse to start (exit 4) while another process computes on the GPU
  (`scripts/require_free_gpu.sh`); `scripts/test_gpu_guards.sh` tests both guards against a fake
  `nvidia-smi`.
- Frame-order replay: `scripts/run_order_replay.sh`; spread of all uniform runs:
  `scripts/uniform_runs_spread.py`; training loss of two runs on the same batches:
  `scripts/paired_loss_curves.py`.
- Container smoke test (GPU, EGL, manifest): `scripts/container_smoke.sh`, passed 2026-11-07.
- Baselines: `[PROTOCOL=<file>] scripts/run_baseline.sh <seed> [out_root]` (the protocol file used is
  kept in the run as `protocol.json`); statistics:
  `scripts/baseline_stats.py [--baseline-dir DIR]` (one protocol per baseline, recorded in its
  `summary.json`); oracle vs host comparison: `scripts/oracle_compare.py`.
- Released checkpoint: `scripts/eval_reference.sh`; statistics: `scripts/reference_stats.py`.
- LeRobot control: `scripts/run_lerobot_train_control.sh`; paired comparison of any runs on the
  hidden block: `scripts/paired_runs.py`.
- Author probes: `scripts/run_candidate.sh <run_dir> <candidate_dir> 1000`; statistics:
  `scripts/candidate_stats.py <run_dir> [--baseline-dir DIR]` (refuses runs that are not the full
  budget, not the hidden seed block, or trained under another protocol than the baseline;
  `--allow-other-protocol` gives a comparison marked descriptive).
- Official-path validation: `scripts/oracle_run.sh` (RSI-Harness oracle; passed under the final
  protocol on 2026-11-12, reward 0.162, exit 0, run `8f721dab03844104a15b295fbfd30535`, evidence in
  `experiments/oracle_final/`, comparison with `scripts/oracle_compare.py`; under protocol v1 on
  2026-11-07, reward 0.138, run `c504ab36acb7416eb92a37b8c1735ed1`, evidence in `experiments/oracle/`).
- After a machine fault: `scripts/kernel_health.sh` (fault lines in this boot's kernel log; the
  launch scripts call it before and after a run), `checkpoint.sha256` in every run directory, and
  `scripts/compare_evals.py` for two evaluations of one seed block.

## 10. Compute

| Item | GPU wall time |
| --- | --- |
| Three baselines: training | 13.97 h |
| Three baselines: hidden evaluation | 2.49 h |
| Three baselines: proxy evaluation | 0.44 h |
| RSI-Harness oracle: one Judge round | 4.84 h |
| Author probe drop-idle-tail: training, hidden and proxy evaluation | 4.92 h |
| Author probe insertion-focus: training, hidden and proxy evaluation | 4.92 h |
| Released checkpoint: hidden and proxy evaluation | 0.48 h |
| Uniform baseline, protocol v2, seed 1000: training, hidden and proxy evaluation | 4.91 h |
| `lerobot-train` control, seed 1000: training, hidden and proxy evaluation | 4.98 h |
| `lerobot-train` control, first attempt, cut off by a reboot at 28% | 1.26 h |
| Uniform baseline, protocol v3, seed 1000: training, hidden and proxy evaluation | 4.76 h |
| Same run, first attempt, stopped at step 2,400 to replace the draw rule | 0.13 h |
| Uniform baseline, protocol v3, seed 1001: training, hidden and proxy evaluation | 4.72 h |
| `lerobot-train` control: hidden evaluation of the 60,000- and 80,000-step checkpoints, alongside the replay | 1.00 h |
| Frame-order replay, seed 1000: training, hidden and proxy evaluation (partly alongside the two-epoch check) | 5.60 h |
| Full-length deterministic parity: both trainers for 100,000 steps, then hidden and proxy evaluation | 9.42 h |
| Second deterministic seed: our trainer on the Judge's uniform schedule, seed 1001, then hidden and proxy evaluation | 4.94 h |
| `lerobot-train` re-run unchanged, seed 1000: training, hidden and proxy evaluation | 4.95 h |
| `lerobot-train` deterministic mode, seed 1001: training, hidden and proxy evaluation | 5.00 h |
| `lerobot-train` re-run: 60,000- and 80,000-step checkpoint evaluations | 0.80 h |
| Second normal-mode frame-order replay, seed 1000: training, hidden and proxy evaluation | 4.94 h |
| Kernel-level comparison: four 300-step normal-mode runs with recording | 0.07 h |
| `lerobot-train` deterministic mode, seed 1001: 60,000- and 80,000-step checkpoint evaluations | 0.97 h |
| Final protocol (deterministic Judge), seed 1000: training, hidden and proxy evaluation | 5.10 h |
| Final protocol (deterministic Judge), seed 1002: training, hidden and proxy evaluation | 5.57 h |
| Final-protocol oracle run, first attempt, lost to a machine hang 62 min into the Judge's training | 1.03 h |
| Repeat of final-protocol seed 1002 after the machine fault: training, hidden and proxy evaluation | 4.88 h |
| Third hidden evaluation of the seed-1002 weights | 0.40 h |
| Evaluator determinism check: two deterministic proxy evaluations | 0.16 h |
| RSI-Harness oracle under the final protocol: image rebuild and one Judge round | 5.04 h |
| Author probe drop-idle-tail under the final protocol: training, hidden and proxy evaluation | 4.97 h |
| Author probe insertion-focus under the final protocol: training, hidden and proxy evaluation | 4.92 h |
| **Total so far** | **122.57 h** on one RTX 4070 Laptop GPU |

The totals exclude smoke tests, image builds and the shorter trainer parity checks (the two-epoch
check took 1.2 h alongside the replay and slowed its training by about 0.8 h against the two v3 seeds).
One evaluator round on this laptop took 4.86 h through RSI-Harness under the final protocol (4.84 h
under protocol v1). Host runs of training plus
hidden evaluation took 4.65-6.04 h (six baselines and two probes). So 48 h hold at most 7.9-10.3 rounds,
even with no agent time between them. With any agent time that is below OpenRSI's 10-loop
guideline, a non-blocking flag. H100 round time is not measured.

## 11. Limitations

- 50 demonstrations: excluding one episode is a large intervention, and conclusions may not
  transfer to larger datasets.
- Simulation with a single camera; no real-robot validation.
- The v3 uniform schedule has the distribution of LeRobot's epoch shuffling but not its exact order,
  and the schedule seed is fixed, so all training seeds share one frame order. Replaying LeRobot's
  exact order scored 66/500, so this is not what separates our trainer from LeRobot's.
- The 20% that LeRobot's own script and the released checkpoint reach is a seed-1000 result: at
  seed 1001 the script scores 70/500 in deterministic mode, like our trainer. Its seed-to-seed spread
  is measured at two seeds only, and in normal cuDNN mode, on the same input, our trainer stays below
  the script (66 and 70 against 100 and 102) for a reason not found. The Judge avoids the question by
  training in deterministic mode, where the two are bit-identical.
- The Judge trains each candidate at one training seed (1000), and the seed-to-seed s.d. of the
  uniform baseline (2.36 pp over three seeds) is larger than the binomial s.e. (1.52 pp). The
  threshold includes that term, but three seeds estimate it poorly; a best candidate is confirmed by
  retraining at other seeds (Section 8).
- Training is bit-reproducible on this machine; across GPUs, drivers or library builds it has not
  been tested. Evaluation is not bit-reproducible: GPU rendering differs in a few pixels between
  processes and the evaluator lets cuDNN pick algorithms by timing, so one checkpoint's score moves by
  about 0.6 pp between evaluations.
- The development machine had a hardware-type fault (kernel memory corruption, 2026-11-12). The run
  it touched was repeated and reproduced bit for bit, and the launch scripts now refuse to start, or
  flag the result, when the kernel log shows such faults; earlier runs predate any logged fault.
- Evaluation is CPU-bound and its wall time varies 2.6x on the laptop.
