#!/usr/bin/env bash
# Normal-mode algorithm comparison: the same four 300-step runs as scripts/check_normal_mode_noise.sh
# (aloha-train twice, LeRobot v0.6.1's lerobot-train twice, one seed-1000 frame order, normal mode),
# launched through scripts/normal_mode_algos.py, which from step 200 on records two complete steps of
# GPU kernel names (the cuDNN/cuBLAS algorithm and its TF32/FP32 variant), the cuBLAS and cuBLASLt
# matmul calls, and the effective torch switches, then compares the four records pairwise.
# cudnn.benchmark's choice is made in step 1 and is not observed while it is made, so it is not perturbed.
# Rule for reading the result: docs/plan.md, step 1b ("cuDNN/cuBLAS 算法比对").
# Usage: scripts/check_normal_mode_algos.sh [steps] [start_step] [out_root]
#   steps defaults to 300, start_step to 200 (two steps are recorded from there), out_root to
#   experiments/trainer_parity; output goes to <out_root>/normal_algos<steps>/ (runs aloha_a, aloha_b,
#   lerobot_a, lerobot_b, each with record/{switches,kernels,cublas,cublaslt,window}.json; summary.json).
# Exit 0: all four runs recorded and compared (differences are a result, not a failure).
# Exit 1: a run or the comparison failed. Exit 2: output not empty. Exit 3: .venv-control off its pins.
# Exit 4: another process computes on the GPU.
set -euo pipefail
steps="${1:-300}"; start="${2:-200}"; out_root="${3:-experiments/trainer_parity}"
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"
E=task/aloha-act-demo-curriculum/environment
out="$out_root/normal_algos$steps"
if [ -e "$out" ] && [ -n "$(ls -A "$out")" ]; then echo "refusing to overwrite non-empty $out" >&2; exit 2; fi
source scripts/require_free_gpu.sh  # exit 4 while another process computes on the GPU
mkdir -p "$out"
export MUJOCO_GL=egl HF_HUB_OFFLINE=1 PYTHONHASHSEED=0
if ! diff <(uv pip freeze --python .venv-control/bin/python | sort) pins/control-venv.freeze.txt >/dev/null; then
  echo ".venv-control does not match pins/control-venv.freeze.txt" >&2; exit 3
fi
echo "$(date -Is) normal-mode algorithm comparison, $steps steps, recording 2 steps from step $start, output $out"
.venv/bin/python scripts/trainer_parity.py order --protocol "$E/fixed_train.json" \
  --dataset-root data/aloha_sim_insertion_human --steps "$steps" --output-order "$out/order.npy"
if [ -f experiments/trainer_parity/normal_steps300/order.npy ] && cmp -s "$out/order.npy" experiments/trainer_parity/normal_steps300/order.npy; then
  echo "order.npy is identical to the noise check's order"
fi
read -r rev seed batch < <(.venv/bin/python -I -c "import json,sys; p=json.load(open(sys.argv[1])); print(p['dataset']['revision'], p['training']['seed'], p['training']['batch_size'])" "$E/fixed_train.json")
for run in aloha_a aloha_b; do
  echo "$(date -Is) aloha-train $run"
  .venv/bin/python scripts/normal_mode_algos.py run --trainer aloha --record-dir "$out/$run/record" \
    --start-step "$start" --steps 2 -- \
    --schedule "$out/order.npy" --protocol "$E/fixed_train.json" \
    --dataset-root data/aloha_sim_insertion_human --output "$out/$run" --steps "$steps" > "$out/$run.json"
done
for run in lerobot_a lerobot_b; do
  echo "$(date -Is) lerobot-train $run"
  .venv-control/bin/python scripts/normal_mode_algos.py run --trainer lerobot --record-dir "$out/$run/record" \
    --start-step "$start" --steps 2 -- \
    --policy.type=act --policy.device=cuda --policy.push_to_hub=false \
    --dataset.repo_id=lerobot/aloha_sim_insertion_human --dataset.root=data/aloha_sim_insertion_human \
    --dataset.revision="$rev" \
    --seed="$seed" --batch_size="$batch" --steps="$steps" --num_workers=4 --log_freq=100 \
    --save_freq="$steps" --wandb.enable=false --output_dir="$out/$run" > "$out/$run.log" 2>&1
done
echo "$(date -Is) comparing"
.venv/bin/python scripts/normal_mode_algos.py compare "$out"/aloha_a/record "$out"/aloha_b/record \
  "$out"/lerobot_a/record "$out"/lerobot_b/record --output "$out/summary.json"
echo "$(date -Is) normal-mode algorithm comparison done"
