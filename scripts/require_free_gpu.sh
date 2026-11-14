# Sourced by the GPU launch scripts before they create any output: exit 4 if a process already
# computes on the GPU, or if nvidia-smi cannot tell. The user's own GPU work comes first, and a GPU
# shared with another job already ran out of memory once (docs/plan.md, first hidden evaluation of
# seed 1000). To start as soon as the GPU is free instead of now:
#   scripts/run_bg.sh <name> scripts/wait_gpu_idle.sh 3600 <command...>
if ! gpu_busy="$(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader)"; then
  echo "refusing to start: nvidia-smi failed, so the GPU cannot be checked" >&2
  exit 4
fi
if [ -n "$gpu_busy" ]; then
  echo "refusing to start: the GPU already runs compute processes:" >&2
  echo "$gpu_busy" >&2
  exit 4
fi
unset gpu_busy
