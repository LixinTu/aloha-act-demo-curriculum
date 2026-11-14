# ALOHA insertion: agent-designed demonstration curriculum for ACT

## Overview

This Harbor task studies how a research agent can raise the closed-loop success of a fixed-budget **ACT** (Action Chunking Transformer) policy on `gym_aloha/AlohaInsertion-v0` by designing only the **sampling distribution of human teleoperation demonstrations during training**. The candidate is `/workspace/curriculum/policy.json`, a declarative curriculum (episode selection, chunk-anchor eligibility, and 1-16 training stages whose per-frame weights and masks are expressions in a closed, bounded DSL), plus an optional agent-computed `/workspace/curriculum/scores.csv` of per-frame scores. The Judge validates the candidate, recompiles it into the exact 800,000-exposure schedule itself, trains ACT from scratch with the fixed recipe and seed, evaluates 500 hidden-seed episodes, and reports the success rate. Candidate code and checkpoints are never scoring inputs.

The research question is how the temporal structure of teleoperated trajectories (approach, bimanual grasp, lift, contact-rich insertion) interacts with action chunking: every ACT sample is a frame plus the next 100 actions, so which frames serve as chunk anchors, in which training stage, and with what weight is a mechanism-level choice rather than a hyperparameter.

## Baseline and evidence

> **Status 2026-11-12.** `environment/fixed_train.json` is the final protocol: protocol v3 (balanced exposure draws, LeRobot's image path and ImageNet statistics) with `cudnn_deterministic: true`, under which the fixed trainer and LeRobot v0.6.1's own `lerobot-train` produce bit-identical weights over the full 100,000-step budget (`scripts/check_trainer_parity.sh`; full-length check 2026-11-10). The file is byte-identical to the `protocol.json` of the three baseline runs. History: `lerobot-train` scored 100/500 at seed 1000 against 60-82/500 for the fixed trainer's earlier runs; the gap was traced, with a rule committed before each run, to the recipe's own seed-to-seed spread (`lerobot-train` in deterministic mode scores 100 at seed 1000 and 70 at seed 1001), not to the trainer or the frame order; in normal cuDNN mode the two trainers still differ in a way that kernel-level comparison could not localise, which is why the Judge trains in deterministic mode. The oracle run under the final protocol passed on 2026-11-12 (below).

| Item | Value |
| --- | --- |
| Policy source | `huggingface/lerobot` v0.6.1, commit `7e241bd630a3719a56157a497ce5d08f244784f1` (ACT implementation, processors, `eval_policy`) |
| Simulator | `gym-aloha` 0.1.5 (commit `6203fac50be47065056b69094e44308972202258`), MuJoCo 3.8.1, dm-control 1.0.41 |
| Data | `lerobot/aloha_sim_insertion_human` revision `cc571a3c661df81b566dbfde3d5c1e85fcdf7884` (LeRobot v3.0 format, 50 episodes x 500 frames, 50 fps, MIT) |
| Matched baseline (final protocol) | Uniform sampling (`environment/default_policy.json`) under the final protocol (deterministic mode): **16.0% (80/500)** at the Judge's training seed 1000; over three full-budget training seeds 1000/1001/1002: 80, 64, 57 successes of 500, pooled **13.40% (201/1500, Wilson 95% 11.8-15.2%)**, seed s.d. 2.36 pp (2026-11-10 to 2026-11-12, RTX 4070 Laptop GPU; `experiments/baseline_v3_det/summary.json`). The three seeds share one compiled uniform frame order and differ only in the training seed. The seed-1002 run ended while the development machine's kernel was logging memory corruption (it hung on 2026-11-12 04:09); a repeat with the same command reproduced its weights bit for bit (`experiments/baseline_v3_det/seed1002/checkpoint.sha256`), so the training stands. Protocol v1 (2026-11-06) gave 68/76/65, pooled 13.93%. |
| Noise (final protocol) | A single 500-episode run has binomial s.e. 1.52 pp. The three seeds spread more than that (Pearson chi-square 4.79, df 2, p = 0.091; training-seed variance estimate 3.2e-4), so the threshold includes the training-seed term: a single Judge run needs **>= 92/500 (18.4%)** to exceed the pooled baseline at two-sided 95% (formula committed before the runs; under v1 it was 88/500). Across all twelve uniform runs of the fixed trainer the dispersion is 1.09 (95% 0.55-3.14). Selecting the best of N Judge rounds inflates the best score under the null by about 2.4 pp at N = 4 and 3.6 pp at N = 10 (single-run s.d. 2.36 pp; 1.6 and 2.4 pp from binomial noise alone). The Judge's training is bit-reproducible on one machine; its evaluation is not: one checkpoint scored 57, 60 and 62 of 500 in three evaluations (re-evaluation s.d. about 0.6 pp, inside the binomial term), because the GPU-rendered camera image differs by one intensity level in a few pixels between processes while the physics is identical (`scripts/render_determinism.py`), and the evaluator lets cuDNN pick algorithms by timing as `lerobot-eval` does (`evaluation.cudnn_deterministic` exists and is off; it did not make two evaluations identical). |
| Production-path check (final protocol) | RSI-Harness oracle run `8f721dab03844104a15b295fbfd30535` (2026-11-12, images rebuilt from scratch, one GPU, Judge GPU mode `release-all`): `solve.sh` exit 0, one submission, Judge status `completed`, **reward 0.162 (81/500)**; one Judge round took 17,495 s (training 16,134 s, evaluation 1,346 s). The Judge's training is bit-identical to the host baseline run at seed 1000: same schedule hash and all 500 logged losses equal to the last digit, with the container differing from the host only in the Python patch version. Its EGL renderer is the NVIDIA GPU, and its outcome matches the host's 80/500 within re-evaluation noise (78 shared successes, 5 discordant episodes, McNemar exact p = 1.0). Evidence: `experiments/oracle_final/`. |
| Production-path check (v1, history) | RSI-Harness oracle run `c504ab36acb7416eb92a37b8c1735ed1` (2026-11-07, normal cuDNN mode): **reward 0.138 (69/500)** against the host seed-1000 run's 68/500 (McNemar exact p = 1.0). Only 31 successes were shared, because retraining in normal mode gives a different model each time; candidates are compared by counts, not by episodes. |
| Released checkpoint | The model card of `lerobot/act_aloha_sim_insertion_human` reports 20.6%. It was produced with an older LeRobot (commit `3c0a209`, 2025-01-31) and dataset format, and that repository's `eval_info.json` is byte-identical in structure to the transfer-cube file (83.0%, video paths under `act_aloha_transfer`), so the 20.6% has no per-episode evidence. Evaluated with this task's evaluator on the 500 hidden seeds, the checkpoint scores 101/500 = 20.2% (Wilson 16.9-23.9%; 2026-11-08), consistent with the card. It was trained with older code and data format, so it is a reference point, not the matched baseline. |
| Released reference | The official checkpoint (revision `33259aa86eb45fdf85350280044a33d9d50e40c3`, `model.safetensors` SHA-256 `ad37bb2e9e341338a47cc31da43e7805998f8a1168f1399296be28a830a8a9d2`) is migrated at build time to the current processor format (`/opt/aloha_assets/reference_policy`) and offered to agents as an analysis tool (e.g. per-frame chunk prediction error). It is never a scoring input. |

Per-seed evidence (v1): hidden-block success histogram of the highest reward level reached per episode (0 nothing, 1 both grippers touching, 2 both objects lifted, 3 peg touching socket, 4 pin inside = success):

| Training seed | 0 | 1 | 2 | 3 | 4 | Proxy (100 public seeds) | Train time | Eval time |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1000 (Judge seed) | 85 | 137 | 35 | 175 | 68 | 10 | 4.81 h | 35 min |
| 1000, RSI-Harness Judge (oracle) | 74 | 132 | 28 | 197 | 69 | not run | 4.46 h | 23 min |
| 1001 | 81 | 96 | 28 | 219 | 76 | 17 | 4.49 h | 31 min |
| 1002 | 96 | 94 | 37 | 208 | 65 | 15 | 4.66 h | 83 min |

Roughly 40% of episodes bring the peg into contact with the socket (level 3) while only ~14% complete the insertion, so most headroom is in the contact-rich final phase.

The build installs the exact CPython 3.12 closure: `environment/requirements-torch.lock` from the PyTorch cu128 index only (torch 2.11.0+cu128, torchvision 0.26.0+cu128), then `environment/requirements.lock` from PyPI only, with dependency resolution disabled (`--no-deps`). `environment/prepare_assets.py` downloads the pinned dataset and checkpoint revisions, verifies all 18 Hub files against `environment/assets.sha256`, downloads and verifies the torchvision ResNet18 ImageNet weights (SHA-256 `f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec`), builds the public 25,000 x 61 per-frame feature table, and migrates the reference checkpoint. `environment/build_manifest.py` then records the SHA-256 of every file under `/opt/aloha_public` and `/opt/aloha_assets` and the versions of 13 packages; both trees are made read-only.

## Environment and evaluation

| Item | Value |
| --- | --- |
| Workdir | `/workspace` containing only `curriculum/` (initialised with the uniform policy) and `research/` (free, ignored by the Judge) |
| Fixed protocol | `environment/fixed_train.json`: ACT exactly as the official config, camera images normalised with ImageNet statistics (`dataset.use_imagenet_stats`) (ResNet18, `chunk_size = n_action_steps = 100`, VAE latent 32, KL weight 10); 100,000 steps x batch 8 = 800,000 exposures; AdamW lr 1e-5, wd 1e-4, grad clip 10; training seed 1000; no augmentation; schedule seed fixed; balanced schedule draw (`training.schedule_draw`, protocol v3) |
| Candidate | `policy.json` <= 64 KiB, optional `scores.csv` <= 8 MiB (1-16 finite float columns, one row per frame); data only, parsed by `aloha_curriculum/schema.py` and `dsl.py` (closed operator set, 512 nodes, depth 24, bounded constants and windows) |
| Schedule | `aloha_curriculum/compile.py`: 2,000 blocks of 50 steps; each block picks 400 anchors that follow the accumulated share of `weight x mask x eligibility` (a frame falls due once per unit of its share, at a random point; uniform weights give shuffled epochs), using a fixed RNG; length exactly 800,000; SHA-256 reported |
| Trainer | `aloha_curriculum/train.py`: LeRobot's ACT, processors and optimizer preset with the forward/backward/clip/step order of `lerobot_train.update_policy`; the only change is a sampler that follows the compiled schedule |
| Evaluation | `aloha_curriculum/evaluate.py` calls LeRobot's `eval_policy`: `AlohaInsertion-v0`, `episode_length = 400`, 10 synchronous envs, success = reward level 4; 500 consecutive hidden seeds in `tests/hidden_seeds.json`; public proxy block 100000-100099 |
| Work/Judge | One GPU each (a single-GPU host runs them on the same device: the Harness releases Work's GPU to the Judge); 16 CPUs; 65,536 MiB RAM; `shm_size: 8g` (`environment/docker-compose.yaml`); MuJoCo EGL on the NVIDIA ICD with Mesa as fallback |
| Network | Build fetches pinned public inputs; Work and Judge are `no-network` |
| Public tools | `aloha-validate`, `aloha-compile`, `aloha-train` (diagnostic `--steps`), `aloha-eval` (proxy block by default), `aloha-features`; sources readable under `/opt/aloha_public/aloha_curriculum` |
| Judge | `tests/evaluate.py`: preflight (`python -I`, `env -i`, workspace removed from `sys.path`), manifest re-verification, workspace layout and single stable reads of the candidate, validation and recompilation, fresh training in a subprocess with accounting checks (steps, exposures, seed), hidden evaluation in a subprocess with seed-block checks, `reward = successes / 500` |
| Failure semantics | Invalid candidate (exit 2) and run/infrastructure errors (exit 1) write no reward; the round is unscored |
| Timeouts | Build 7,200 s; Agent 86,400 s; Verifier 28,800 s (on an RTX 4070 Laptop GPU one Judge round took 17,435 s through RSI-Harness, and host runs of train plus hidden evaluation took 17,384-21,746 s; datacenter GPUs not yet measured) |

The Judge never imports or executes anything from `/workspace`; the candidate files are copied with size limits into a private directory before parsing, and symlinks or extra files are rejected. The schedule is recompiled by the Judge from the candidate and the fixed protocol, so the training budget cannot be changed by the candidate. Hidden seeds and per-episode results are written only to operator logs (`/logs/verifier`), never to Agent feedback.

## Run and validate

```bash
export RSI_MODEL='operator-selected-model'
export RSI_REASONING_EFFORT='operator-selected-effort'
rsi-harness run "$PWD/rsi-tasks/aloha-act-demo-curriculum" \
  --agent codex \
  --model "$RSI_MODEL" \
  --reasoning-effort "$RSI_REASONING_EFFORT" \
  --gpus 0,1 \
  --primary-reward reward \
  --score-direction maximize \
  --verbose
```

Oracle validation (uniform baseline through the production path):

```bash
rsi-harness run "$PWD/rsi-tasks/aloha-act-demo-curriculum" --agent oracle --gpus 0 \
  --max-submissions 1 --timeout 30600 --verbose
```

Task-owned runtime unit tests (CPU only, no environment build):

```bash
cd rsi-tasks/aloha-act-demo-curriculum/environment && python -m pytest tests_runtime -q
```
