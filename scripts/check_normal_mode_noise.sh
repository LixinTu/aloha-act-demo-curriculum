#!/usr/bin/env bash
# Normal-mode noise check: the same seed-1000 frame order through aloha-train twice and through LeRobot
# v0.6.1's lerobot-train twice, all in normal mode (cudnn.benchmark on, as the control run and the order
# replay trained), then every pair of final weights compared. The two within-trainer pairs measure how far
# one trainer's own runs drift apart in normal mode; the four cross-trainer pairs are read against that.
# In deterministic mode the trainers are bit-identical (scripts/check_trainer_parity.sh), so any
# normal-mode-only difference between them must show here, or not be found by comparing weights at all.
# Rule for reading the result: docs/plan.md, step 1b ("普通模式噪声比对").
# Usage: scripts/check_normal_mode_noise.sh [steps] [out_root]
#   steps defaults to 300 (about 10 minutes for the four runs); out_root to experiments/trainer_parity;
#   output goes to <out_root>/normal_steps<steps>/ (runs aloha_a, aloha_b, lerobot_a, lerobot_b;
#   result_<x>_vs_<y>.json per pair; summary.json).
# Exit 0: all four runs trained and all six pairs compared (differences are expected, not a failure).
# Exit 1: a run or a comparison failed. Exit 2: output not empty. Exit 3: .venv-control off its pins.
# Exit 4: another process computes on the GPU.
set -euo pipefail
steps="${1:-300}"; out_root="${2:-experiments/trainer_parity}"
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"
E=task/aloha-act-demo-curriculum/environment
out="$out_root/normal_steps$steps"
if [ -e "$out" ] && [ -n "$(ls -A "$out")" ]; then echo "refusing to overwrite non-empty $out" >&2; exit 2; fi
source scripts/require_free_gpu.sh  # exit 4 while another process computes on the GPU
mkdir -p "$out"
export MUJOCO_GL=egl HF_HUB_OFFLINE=1 PYTHONHASHSEED=0
if ! diff <(uv pip freeze --python .venv-control/bin/python | sort) pins/control-venv.freeze.txt >/dev/null; then
  echo ".venv-control does not match pins/control-venv.freeze.txt" >&2; exit 3
fi
echo "$(date -Is) normal-mode noise check, $steps steps, output $out"
.venv/bin/python scripts/trainer_parity.py order --protocol "$E/fixed_train.json" \
  --dataset-root data/aloha_sim_insertion_human --steps "$steps" --output-order "$out/order.npy"
read -r rev seed batch < <(.venv/bin/python -I -c "import json,sys; p=json.load(open(sys.argv[1])); print(p['dataset']['revision'], p['training']['seed'], p['training']['batch_size'])" "$E/fixed_train.json")
for run in aloha_a aloha_b; do
  echo "$(date -Is) aloha-train $run"
  .venv/bin/aloha-train --schedule "$out/order.npy" --protocol "$E/fixed_train.json" \
    --dataset-root data/aloha_sim_insertion_human --output "$out/$run" --steps "$steps" > "$out/$run.json"
done
for run in lerobot_a lerobot_b; do
  echo "$(date -Is) lerobot-train $run"
  .venv-control/bin/lerobot-train \
    --policy.type=act --policy.device=cuda --policy.push_to_hub=false \
    --dataset.repo_id=lerobot/aloha_sim_insertion_human --dataset.root=data/aloha_sim_insertion_human \
    --dataset.revision="$rev" \
    --seed="$seed" --batch_size="$batch" --steps="$steps" --num_workers=4 --log_freq=100 \
    --save_freq="$steps" --wandb.enable=false --output_dir="$out/$run" > "$out/$run.log" 2>&1
done
weights() {  # <run> -> path of its final weights
  case "$1" in
    aloha_*) echo "$out/$1/pretrained_model/model.safetensors" ;;
    lerobot_*) echo "$out/$1/checkpoints/last/pretrained_model/model.safetensors" ;;
  esac
}
failed=0
for pair in "aloha_a aloha_b" "lerobot_a lerobot_b" "aloha_a lerobot_a" "aloha_a lerobot_b" "aloha_b lerobot_a" "aloha_b lerobot_b"; do
  read -r x y <<< "$pair"
  printf '%s vs %s: ' "$x" "$y"
  set +e
  .venv/bin/python scripts/trainer_parity.py compare "$(weights "$x")" "$(weights "$y")" \
    --steps "$steps" --save-freq "$steps" --output "$out/result_${x}_vs_${y}.json"
  rc=$?
  set -e
  if [ "$rc" -ne 0 ] && [ "$rc" -ne 1 ]; then echo "comparison $x vs $y failed (exit $rc)" >&2; failed=1; fi
done
[ "$failed" -eq 0 ] || exit 1
.venv/bin/python -I - "$out" "$steps" <<'EOF'
import json, sys
from pathlib import Path
out, steps = Path(sys.argv[1]), int(sys.argv[2])
pairs = {p.stem[len("result_"):]: json.loads(p.read_text()) for p in sorted(out.glob("result_*_vs_*.json"))}
within = {k: v for k, v in pairs.items() if k.split("_vs_")[0][:5] == k.split("_vs_")[1][:5]}
cross = {k: v for k, v in pairs.items() if k not in within}
summary = {"steps": steps, "pairs": pairs,
           "within_trainer": sorted(within), "cross_trainer": sorted(cross),
           "max_within": {m: max(v[m] for v in within.values()) for m in ("worst_abs_diff", "rms_diff", "rel_fro", "differing")},
           "min_cross": {m: min(v[m] for v in cross.values()) for m in ("worst_abs_diff", "rms_diff", "rel_fro", "differing")},
           "max_cross": {m: max(v[m] for v in cross.values()) for m in ("worst_abs_diff", "rms_diff", "rel_fro", "differing")}}
(out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(f"{'pair':24s} {'differing':>9s} {'worst_abs':>10s} {'rms':>10s} {'rel_fro':>10s}")
for k, v in pairs.items():
    print(f"{k:24s} {v['differing']:9d} {v['worst_abs_diff']:10.3e} {v['rms_diff']:10.3e} {v['rel_fro']:10.3e}")
EOF
echo "$(date -Is) normal-mode noise check done"
