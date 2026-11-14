#!/usr/bin/env bash
# Exit 0 only when the host's automatic updates are held for at least <min_seconds> (default 43200).
#
#   scripts/update_hold_check.sh [min_seconds]
#
# Why: on this machine a systemd reload or re-exec takes the GPU away from running containers (Docker's
# systemd cgroup driver + cgroup v2 + --gpus; NVIDIA Container Toolkit troubleshooting guide). During the
# agent run it cost two Judge rounds: snapd's refresh reloaded systemd on 2026-11-13 12:46, and the
# unattended glibc upgrade re-executed it on 2026-11-14 06:33 (docs/plan.md step 5).
# Held means both of:
#   - `snap refresh --time --abs-time` has a "hold:" line that is "forever" or ends >= min_seconds from now;
#   - apt-daily.timer and apt-daily-upgrade.timer are not active.
# Only the user can set the hold (sudo):
#   sudo snap refresh --hold=48h && sudo systemctl stop apt-daily.timer apt-daily-upgrade.timer
# Exit codes: 0 held; 1 not held (what is missing is printed); 2 usage, or the state could not be read.
set -uo pipefail
min="${1:-43200}"
if [ "$#" -gt 1 ] || ! [[ "$min" =~ ^[0-9]+$ ]]; then
  echo "usage: $0 [min_seconds]" >&2
  exit 2
fi
problems=()
times="$(snap refresh --time --abs-time 2>&1)"; rc=$?
if [ "$rc" -ne 0 ]; then
  echo "cannot read the snap refresh schedule (snap exit $rc): $times" >&2
  exit 2
fi
hold="$(sed -n 's/^hold:[[:space:]]*//p' <<< "$times" | head -n 1)"
if [ -z "$hold" ]; then
  problems+=("snap auto-refresh is not held")
elif [ "$hold" != forever ]; then
  if ! until_s="$(date -d "$hold" +%s 2>/dev/null)"; then
    echo "cannot read the snap hold time '$hold'" >&2
    exit 2
  fi
  left=$((until_s - $(date +%s)))
  if [ "$left" -lt "$min" ]; then
    problems+=("the snap hold ends in ${left}s ($hold), sooner than ${min}s from now")
  fi
fi
for timer in apt-daily.timer apt-daily-upgrade.timer; do
  state="$(systemctl is-active "$timer" 2>&1)"
  case "$state" in
    inactive|failed) ;;
    *) problems+=("$timer is $state") ;;
  esac
done
if [ "${#problems[@]}" -gt 0 ]; then
  printf 'automatic updates are not held: %s\n' "${problems[@]}" >&2
  echo "the user's sudo hold: sudo snap refresh --hold=48h && sudo systemctl stop apt-daily.timer apt-daily-upgrade.timer" >&2
  exit 1
fi
echo "automatic updates held: snap hold until $hold; apt-daily.timer and apt-daily-upgrade.timer not active"
