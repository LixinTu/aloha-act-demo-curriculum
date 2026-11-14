#!/usr/bin/env bash
# Formal RSI-Harness research run with a real Agent on this host (one GPU, release-all mode).
#
# This spends model usage (the local Claude Code login or an API key) and holds the GPU for up to
# the Agent timeout, so it refuses to start unless the operator confirmed the run explicitly:
#
#   AGENT_RUN_CONFIRMED=yes scripts/run_bg.sh agent-run scripts/agent_run.sh
#
# Settings (environment variables, defaults in brackets):
#   RSI_AGENT          Agent adapter [claude-code]
#   RSI_MODEL          model [claude-opus-5-5]
#   RSI_EFFORT         reasoning effort [max]
#   RSI_AGENT_AUTH     "local" reuses the local CLI login; "key" uses RSI_AGENT_API_KEY [local]
#   RSI_TIMEOUT        Agent time limit in seconds [86400, the task's agent.timeout_sec]
#   MIN_FREE_GB        minimum free disk for the Work image build [25]
#
# Exit codes of its own checks: 3 not confirmed, 4 GPU busy, 5 low disk, 6 key missing, 7 this boot's
# kernel log shows faults or cannot be read, 8 the run finished but the kernel reported faults while it
# ran (scripts/kernel_health.sh). Otherwise the harness's exit code.
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$root"
task="$root/task/aloha-act-demo-curriculum"

agent="${RSI_AGENT:-claude-code}"
model="${RSI_MODEL:-claude-opus-5-5}"
effort="${RSI_EFFORT:-max}"
auth="${RSI_AGENT_AUTH:-local}"
timeout="${RSI_TIMEOUT:-86400}"
min_free_gb="${MIN_FREE_GB:-25}"

if [ "${AGENT_RUN_CONFIRMED:-}" != "yes" ]; then
  echo "refusing to start: set AGENT_RUN_CONFIRMED=yes after the operator approved model usage and GPU time" >&2
  exit 3
fi

# In release-all mode the Judge takes the whole GPU; any other GPU job (an author probe, Ollama)
# slows the Judge past its 8 h cap or runs it out of memory.
busy="$(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader)"
if [ -n "$busy" ]; then
  echo "refusing to start: the GPU already runs compute processes:" >&2
  echo "$busy" >&2
  exit 4
fi

# A kernel that already reported memory corruption or faults in this boot must not produce Judge
# scores (2026-11-12 crash, docs/plan.md).
if ! scripts/kernel_health.sh >&2; then
  echo "refusing to start: the kernel log of this boot is not clean; reboot and check the hardware first" >&2
  exit 7
fi
run_started="$(date '+%Y-%m-%d %H:%M:%S')"

free_gb="$(df --output=avail -BG "$root" | tail -n 1 | tr -dc '0-9')"
if [ "$free_gb" -lt "$min_free_gb" ]; then
  echo "refusing to start: ${free_gb} GB free under $root, need ${min_free_gb} GB for the Work image" >&2
  exit 5
fi

case "$auth" in
  local) auth_args="--agent-auth local" ;;
  key)
    if [ -z "${RSI_AGENT_API_KEY:-}" ]; then
      echo "refusing to start: RSI_AGENT_AUTH=key but RSI_AGENT_API_KEY is empty" >&2
      exit 6
    fi
    auth_args="" ;;
  *) echo "RSI_AGENT_AUTH must be local or key, got '$auth'" >&2; exit 2 ;;
esac

echo "agent=$agent model=$model effort=$effort auth=$auth timeout=${timeout}s free_disk=${free_gb}GB"
mkdir -p "$root/harness/data" "$root/harness/logs"
harness_rc=0
sg docker -c "rsi-harness run '$task' --agent '$agent' --model '$model' --reasoning-effort '$effort' \
  $auth_args --gpus 0 --timeout '$timeout' --primary-reward reward --score-direction maximize \
  --data-root '$root/harness/data' --logs-root '$root/harness/logs' --verbose" || harness_rc=$?
if [ "$harness_rc" -ne 0 ]; then
  echo "rsi-harness exit=$harness_rc" >&2
  exit "$harness_rc"
fi
# Exit 8: the run finished, but the kernel reported faults while it ran; its Judge scores are suspect.
scripts/kernel_health.sh --since "$run_started" || {
  echo "the agent run finished, but the kernel check since $run_started failed: treat its Judge scores as suspect" >&2
  exit 8
}
