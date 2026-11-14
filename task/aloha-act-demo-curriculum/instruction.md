# Agent-designed demonstration curriculum for ACT on ALOHA insertion

## Goal

Raise the closed-loop success rate of an **ACT** (Action Chunking Transformer) policy on
`gym_aloha/AlohaInsertion-v0` by designing **how the 50 human teleoperation demonstrations are
sampled during training**: which episodes are used, which frames may serve as the start of a
100-step action chunk, how strongly each frame is weighted, and how all of that changes over the
course of training. The model, the raw data, the optimiser and the training budget are fixed.
Your candidate is data, not code.

Primary reward: hidden-seed success rate in `[0, 1]` (higher is better). The Judge retrains ACT
from scratch under the fixed recipe for every submission, so the only thing that can move the
score is the exposure schedule your policy compiles to.

## What is fixed (verified by the Judge before every run)

| Component | Value |
| --- | --- |
| Dataset | `lerobot/aloha_sim_insertion_human`, revision `cc571a3c661df81b566dbfde3d5c1e85fcdf7884` (v3.0): 50 episodes x 500 frames at 50 fps, one top camera (480x640), 14-D joint state and 14-D joint-position action |
| Model | ACT exactly as in `lerobot/act_aloha_sim_insertion_human/config.json`: ResNet18 (ImageNet init), `chunk_size = n_action_steps = 100`, VAE latent 32, `kl_weight = 10`, dropout 0.1 |
| Training | 100,000 steps x batch 8 = **800,000 frame exposures**; AdamW lr 1e-5, weight decay 1e-4, grad clip 10; training seed 1000, deterministic cuDNN mode; no image augmentation |
| Evaluation | `AlohaInsertion-v0`, `episode_length = 400`, success = reward level 4 (peg pin inside the socket); 500 hidden seeds, 10 parallel envs |
| Software | lerobot v0.6.1 (`7e241bd6`), gym-aloha 0.1.5, mujoco 3.8.1, torch 2.11.0+cu128 (see `/opt/aloha_public/manifest.json`) |

The full protocol is `/opt/aloha_public/fixed_train.json`. The baseline candidate is uniform
sampling (`/opt/aloha_public/default_policy.json`), which is what `/workspace/curriculum/policy.json`
contains when you start.

## What you control

`/workspace/curriculum/policy.json` (<= 64 KiB) and optionally
`/workspace/curriculum/scores.csv` (<= 8 MiB). Nothing else under `/workspace/curriculum` is
allowed. `/workspace/research/` is yours for notes, scripts, diagnostics and checkpoints; the
Judge ignores it. `/workspace` must contain only those two directories.

### policy.json

```json
{
  "schema_version": 1,
  "name": "late-insertion-focus",
  "notes": "optional, <= 2000 chars",
  "scores": null,
  "episodes": {"include": "all", "exclude": [7, 23]},
  "anchors": {"min_frame": 0, "max_frame": 499, "max_chunk_pad": 99},
  "stages": [
    {"until": 0.4, "weight": {"const": 1.0}, "mask": null},
    {"until": 1.0,
     "weight": {"op": "add", "args": [{"const": 0.2},
                                      {"op": "mul", "args": [{"var": "t_stage"},
                                                             {"op": "ge", "args": [{"feature": "t_norm"}, {"const": 0.5}]}]}]},
     "mask": null}
  ]
}
```

- `episodes`: `include` is `"all"` or a list of episode indices (0..49); `exclude` removes episodes.
- `anchors`: global frame-level eligibility. `chunk_pad` is the number of padded steps in the
  100-step action chunk that starts at a frame (frames 401..499 have pads; the loss ignores padded
  steps through `action_is_pad`). `max_chunk_pad = 0` keeps only fully observed chunks.
- `stages`: 1..16 consecutive training stages, `until` is the fraction of the 100,000 steps at which
  the stage ends (strictly increasing, last one 1.0). Each stage has a `weight` expression
  (non-negative, one value per frame) and an optional boolean `mask` expression (frames with mask
  <= 0.5 are excluded). Within a stage the sampling distribution is
  `weight x mask x eligibility`, normalised.
- Steps are grouped into blocks of 50; each block's distribution is evaluated with `t_train`
  (global progress, 0..1) and `t_stage` (progress inside the stage) at the block midpoint and
  `50 x 8 = 400` anchors are picked so that each frame's exposures follow its accumulated share of
  the distribution: a frame falls due once per unit of that share, at a random point of the unit,
  and each block takes the most overdue frames and shuffles them. Equal weights mean equal exposures,
  a block's mix follows the distribution, and uniform weights give LeRobot's shuffled epochs (every
  frame once before any frame twice). A fixed RNG seed makes the compiled schedule a deterministic
  function of your candidate. The same compiler runs in the Judge.

### Expression DSL (closed operator set, no code)

Leaves: `{"const": x}`, `{"feature": "<column>"}`, `{"score": "<your column>"}`,
`{"var": "t_train" | "t_stage"}`. Nodes: `{"op": "<name>", "args": [...]}`.

| Group | Operators |
| --- | --- |
| arithmetic | `add sub mul div neg abs min max clip(x,lo,hi) exp log1p sqrt square pow(x, const p) sigmoid tanh` |
| logic (0/1) | `lt le gt ge eq and or not where(cond,a,b)` |
| dataset-wide | `rank(x)` percentile rank over all frames, `zscore(x)` |
| per episode (broadcast) | `ep_mean ep_min ep_max ep_sum ep_std ep_rank ep_cumsum ep_cummax ep_first_where(cond) ep_last_where(cond)` |
| temporal (within episode, edge padded) | `shift(x, const k)`, `window_mean/window_max/window_min(x, const k)` over `[i-k, i+k]`, `future_mean/future_max/future_sum(x, const k)` over `[i, i+k]`, `past_mean(x, const k)` over `[i-k, i]` |

Limits: 512 nodes and depth 24 per expression, `|const| <= 1e6`, window `k <= 250`, shift
`|k| <= 500`, `|p| <= 8`. Division, `exp`, `log1p`, `sqrt` and `pow` are clamped to stay finite.
Negative weights, non-finite values, zero total mass in any block, unknown names and any
structural violation make the candidate **invalid**: the Judge reports the reason and produces no
score for that submission.

### Public per-frame features (`/opt/aloha_public/features.parquet`, also `.csv`)

25,000 rows (one per frame) with these columns, all usable in expressions by name:
`episode_index frame_index global_index t_norm frames_to_end chunk_pad`,
`state_<motor>` and `action_<motor>` for the 14 motors
(`left/right_{waist,shoulder,elbow,forearm_roll,wrist_angle,wrist_rotate,gripper}`),
`cmd_err_l2 cmd_err_left cmd_err_right` (commanded-minus-measured joint gap),
`speed_l2 speed_left speed_right accel_l2 jerk_l2` (joint-space kinematics from the measured state),
`action_speed_l2 action_speed_left action_speed_right`,
`gripper_left_cmd gripper_right_cmd gripper_left_vel gripper_right_vel`,
`chunk_path_l2 chunk_speed_mean chunk_gripper_left_delta chunk_gripper_right_delta`
(properties of the 100-step chunk that starts at the frame),
`ep_length ep_path_l2 ep_speed_mean ep_speed_max ep_jerk_mean ep_gripper_left_min ep_gripper_right_min ep_cmd_err_mean`
(episode aggregates broadcast to every frame). `/opt/aloha_public/features_schema.json` lists them.

### scores.csv (optional)

Your own per-frame columns, computed any way you like in Work: columns
`episode_index,frame_index,<name>...` with 1..16 extra float columns named `[a-z][a-z0-9_]{0,31}`
(not colliding with public names), exactly one row per frame, all values finite. Set
`"scores": "scores.csv"` in the policy and reference columns as `{"score": "<name>"}`. Typical
uses: phase labels from the video, per-frame prediction error of the reference policy, learned
difficulty, contact proxies.

## Tools in Work (all on PATH; fixed copies under `/opt/aloha_public`)

| Command | Purpose |
| --- | --- |
| `aloha-validate --candidate-dir /workspace/curriculum --features /opt/aloha_public/features.parquet --protocol /opt/aloha_public/fixed_train.json` | Validate and dry-compile the candidate; prints the schedule summary (entropy, episode coverage, decile exposure, chunk padding) or the exact rejection reason. Exit 2 = invalid. |
| `aloha-compile ... --output DIR [--steps N]` | Write `schedule.npy` + `schedule_summary.json`. `--steps` only for diagnostic runs. |
| `aloha-train --schedule DIR/schedule.npy --protocol ... --dataset-root /opt/aloha_assets/aloha_sim_insertion_human --output DIR [--steps N] [--seed S]` | The Judge's fixed trainer. A full run is 100,000 steps; `--steps` gives short diagnostics (compile the schedule with the same `--steps`). |
| `aloha-eval --checkpoint DIR/pretrained_model --protocol ... --output DIR [--n-episodes K] [--start-seed S] [--n-envs E]` | Closed-loop evaluation. Defaults to the **public proxy block** (seeds 100000..100099, 100 episodes), which is disjoint from the hidden Judge seeds. |
| `aloha-features --dataset-root ... --output ...` | Regenerate the public feature table. |

Also available offline: the dataset at `/opt/aloha_assets/aloha_sim_insertion_human` (LeRobot
v3.0 format, loadable with `LeRobotDataset(repo_id, root=...)`), the official reference checkpoint
migrated to the current format at `/opt/aloha_assets/reference_policy` (the published
`lerobot/act_aloha_sim_insertion_human`, usable for per-frame prediction-error scores or as a
sanity check), the torchvision ResNet18 weights in `TORCH_HOME`, and the task-owned sources under
`/opt/aloha_public/aloha_curriculum` (read-only). There is no network.

Compute: a full training run takes roughly 100,000 / (steps per second of your GPU); the Judge logs
its own throughput in every round. A 500-episode evaluation is CPU-bound (MuJoCo physics plus
EGL rendering in 10 sequential environments) and takes about 30-80 minutes on a laptop-class CPU.
Short diagnostic trainings (e.g. 5,000-20,000 steps) and proxy evaluations are the cheap way to
test hypotheses before spending a submission.

## Submitting

Stop any of your own processes that use the GPU first: on a single-GPU host the Judge runs on the
same device as Work, and the Harness refuses a submission while a Work process still holds it.
Then run `rsi-submit`. A fresh Judge container then:

1. verifies every fixed asset and package version against the build manifest;
2. checks `/workspace` layout and copies `policy.json` (+ `scores.csv`) with single stable reads;
3. validates and compiles the candidate itself (`candidate_invalid` events explain any rejection);
4. trains ACT from scratch: 100,000 steps, batch 8, seed 1000, on `cuda:0`;
5. evaluates 500 hidden-seed episodes;
6. writes the reward.

Feedback (JSON lines on stdout): `candidate_staged` (hashes), `schedule_compiled` (schedule
sha256, distinct frames, episode exposures, decile histogram, chunk padding, entropy),
`train_complete` (throughput, final loss, a 100-point loss history), and `complete` with
`reward`, `successes`, `n_episodes`, `success_rate_se`, `avg_max_reward` and a histogram of the
best reward level reached per episode (0..4: nothing / both grippers touching / both objects
lifted / peg touching socket / pin inside). Hidden seeds are never revealed. Invalid candidates,
training failures and timeouts produce **no score** for that round; the best valid reward is kept.

## Research directions worth testing

- Temporal structure: ACT learns from (frame, next-100-actions) pairs. Anchors near the end of
  an episode see padded chunks; anchors just before grasp or insertion see the contact-rich
  transitions. Does over- or under-sampling particular phases help?
- Phase-aware curricula: stage schedules that emphasise approach/grasp first and insertion later
  (or the reverse), expressed with `t_train`, `ep_first_where` on gripper events, or your own
  `scores.csv` phase labels.
- Demonstration quality: `ep_jerk_mean`, `ep_path_l2`, `ep_cmd_err_mean` and `ep_rank` rank
  smooth vs hesitant demonstrations; excluding or down-weighting the noisiest episodes is a
  classic hypothesis (Demo-SCORE / CUPID style), and the 50-episode pool makes each exclusion a
  real decision.
- Difficulty: run the reference policy over the dataset, score each frame by its chunk
  prediction error, and build easy-to-hard or hard-first schedules.

## How large a difference is real

Measured for this protocol on the task author's host (one RTX 4070 Laptop GPU, November 2026):

- Uniform sampling, the default policy, scores 80/500 at the Judge's training seed 1000. Over
  training seeds 1000, 1001 and 1002 it scores 80, 64 and 57 of 500 (pooled 13.4%). The spread
  between training seeds (s.d. about 2.4 percentage points) is larger than the binomial standard
  error of one 500-episode evaluation (about 1.5 points).
- The Judge trains in deterministic mode with a fixed seed, so an identical candidate retrains to
  bit-identical weights. Its score then moves only with the evaluation, which does not repeat
  exactly (three evaluations of one checkpoint gave 57, 60 and 62 of 500). Resubmitting the same
  candidate tells you nothing about training noise.
- A different compiled schedule gives a different training trajectory even at the same seed: the
  same uniform exposures in two different orders scored 80 and 100 of 500. Treat any change to the
  schedule like a change of training seed. Two candidates need to differ by about 6.5 points
  (33 episodes) before one round each tells them apart at the 95% level.
- One round needs at least 92/500 (18.4%) to exceed the uniform baseline at the two-sided 95%
  level. The best of several rounds overstates a candidate: if every round were just another draw
  of the baseline, the best of 4 would sit about 2.4 points above it and the best of 10 about 3.6,
  and one of 4 such rounds would reach 92/500 about 7% of the time.

Favour hypotheses that should move the score by clearly more than this, use `--seed` in your own
diagnostic runs to see the seed-to-seed spread, and say in your notes which conclusions rest on a
single run.
