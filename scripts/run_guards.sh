# Sourced by the result-producing launch scripts (after scripts/require_free_gpu.sh, before they
# create any output). Three guards against accepting a result from a faulty machine, added after the
# 2026-11-12 crash (docs/plan.md, "2026-11-12 crash"):
#   1. on sourcing: exit 5 if this boot's kernel log already shows fault lines, exit 3 if the log
#      cannot be read (scripts/kernel_health.sh);
#   2. record_checkpoint_sha256 <run_dir>: writes <run_dir>/checkpoint.sha256 for the trained weights,
#      so a deterministic re-run can be compared bit for bit (check with `sha256sum -c` from the
#      repository root);
#   3. kernel_health_after <run_dir>: writes <run_dir>/kernel_health.txt and exits 6 if the kernel
#      reported faults between sourcing and now: the run finished, but its result is suspect.
if ! guards_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; then
  echo "run_guards.sh: cannot resolve its own directory" >&2
  exit 3
fi
run_started="$(date '+%Y-%m-%d %H:%M:%S')"
"$guards_dir/kernel_health.sh" >&2 || {
  guard_rc=$?
  echo "refusing to start: the kernel log of this boot is not clean (exit $guard_rc); reboot and check the hardware first" >&2
  exit "$guard_rc"
}

record_checkpoint_sha256() {  # <run_dir>
  local weights="$1/train/pretrained_model/model.safetensors"
  if [ ! -f "$weights" ]; then
    echo "record_checkpoint_sha256: no weights at $weights" >&2
    return 1
  fi
  sha256sum "$weights" > "$1/checkpoint.sha256"
}

kernel_health_after() {  # <run_dir>
  local rc=0
  "$guards_dir/kernel_health.sh" --since "$run_started" > "$1/kernel_health.txt" 2>&1 || rc=$?
  if [ "$rc" -ne 0 ]; then
    cat "$1/kernel_health.txt" >&2
    echo "the run finished, but the kernel check since $run_started failed (exit $rc): treat the result in $1 as suspect" >&2
    exit 6
  fi
}
