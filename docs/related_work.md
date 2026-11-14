# Related work (verified references, 2026-11-02)

Each entry was checked against its arXiv/proceedings page on 2026-11-02. Claims about the papers
are limited to what the abstracts state.

## Demonstration curation for imitation learning
- **Demo-SCORE** — Chen, Lessing, Liu, Finn. *Curating Demonstrations using Online Experience.* RSS 2025.
  arXiv:2503.03707. Trains a policy on all demonstrations, learns a classifier from online rollouts that
  separates successful from unsuccessful behaviour, discards demonstrations with low estimated success
  probability, retrains. Reports 15-35% absolute success-rate gains over training on all demonstrations.
- **CUPID** — Agia et al. *Curating Data your Robot Loves with Influence Functions.* CoRL 2025 (PMLR 305).
  arXiv:2506.19121. Influence functions estimate each demonstration's effect on expected closed-loop return;
  training on <33% of curated data matches state-of-the-art diffusion policies on RoboMimic. Needs 25-100
  evaluation rollouts for stable influence estimates.
- **DemInf** — *Robot Data Curation with Mutual Information Estimators.* arXiv:2502.08623. Scores demonstration
  quality by mutual information over state/action chunks; subsampling with DemInf improved policies on RoboMimic
  and real ALOHA tasks.
- **RINSE** — *Learning from the Best: Smoothness-Driven Metrics for Data Quality in Imitation Learning.*
  arXiv:2604.23000 (April 2026). Smoothness-based demonstration filtering; argues that filtering lowers the
  conditional action variance of the retained data and that the downstream effect is amplified by action
  chunking and compounding error.
- **PSD spectral metric** — arXiv:2605.01544 (May 2026). Training-free frequency-domain ranking of demonstrations;
  compared against DemInf and CUPID.
- **robomimic** — Mandlekar et al. *What Matters in Learning from Offline Human Demonstrations for Robot
  Manipulation.* CoRL 2021 (PMLR 164). arXiv:2108.03298. Six offline algorithms, proficient-human vs
  mixed-human vs machine-generated datasets; performance depends strongly on demonstration quality.

## Action chunking
- **ACT / ALOHA** — Zhao, Kumar, Levine, Finn. *Learning Fine-Grained Bimanual Manipulation with Low-Cost
  Hardware.* RSS 2023. arXiv:2304.13705. The policy and the `aloha_sim_insertion_human` data used here.
- **Zhang et al.** — *Action chunking and exploratory data collection.* ICLR 2026. arXiv:2507.09061. Theory on
  why chunking and exploratory data circumvent exponential compounding errors in different regimes.

## Position of this task
All curation methods above operate at the **episode** level and decide once, before training, which
demonstrations to keep. This task lets a research agent act at the **(episode, chunk-anchor frame,
training stage)** level inside a fixed-budget ACT recipe: sampling weights, anchor eligibility and
stage schedules are declarative, compiled by task-owned code, and scored by retraining and closed-loop
evaluation. No method in the list ties data selection to the action-chunk anchor or to training time,
and none is evaluated as an agent-driven research loop. The OpenRSI tasks `reasonir-difficulty-curriculum`
and `learnability-cot` are data-curriculum tasks for language models with retrieval/SFT objectives; the
`isaaclab-peginsert-reward-search` task fixes data and searches reward graphs for RL. Task Ideas discussion
#134 (adaptive MimicGen) generates new demonstrations in robosuite rather than re-weighting human ones.
