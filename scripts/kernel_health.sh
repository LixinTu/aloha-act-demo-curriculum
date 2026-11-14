#!/usr/bin/env bash
# Kernel fault check for result-producing runs.
# On 2026-11-12 the kernel logged "Bad rss-counter state" and "Bad page state" from 01:34 on and hung
# at 04:09 with a page-fault Oops on a list pointer that had one flipped bit (docs/plan.md, "2026-11-12
# crash"). A baseline run that ended at 02:30 was accepted, and an oracle run was started at 03:04,
# before anyone read the kernel log. The launch scripts therefore call this before they start (any
# fault line in the current boot) and after they finish (fault lines since the run started), through
# scripts/run_guards.sh.
# Usage: scripts/kernel_health.sh [--since "<YYYY-MM-DD HH:MM:SS>"]
# Exit 0: no fault line. Exit 5: fault lines found (their count and the first ones are printed).
# Exit 3: the kernel log cannot be read (an unreadable log is not a healthy kernel). Exit 2: usage.
# KERNEL_HEALTH_LOG=<file> reads that file instead of journalctl and ignores --since (tests).
set -uo pipefail
since=""
if [ "$#" -gt 0 ]; then
  if [ "$#" -ne 2 ] || [ "$1" != "--since" ] || [ -z "$2" ]; then
    echo "usage: $0 [--since \"<YYYY-MM-DD HH:MM:SS>\"]" >&2
    exit 2
  fi
  since="$2"
fi
# Memory corruption, kernel faults and hangs, hardware errors, filesystem errors. GPU "Xid" lines are
# left out on purpose: another program's rendering bug raises them without touching this run.
pattern='BUG: Bad page state|BUG: Bad page map|BUG: Bad rss-counter state|BUG: unable to handle page fault|BUG: kernel NULL pointer dereference|kernel BUG at|general protection fault|Oops: |watchdog: BUG: soft lockup|Corrupted page table|Machine check|\[Hardware Error\]|EXT4-fs error'
what="this boot"
if [ -n "${KERNEL_HEALTH_LOG:-}" ]; then
  what="$KERNEL_HEALTH_LOG"
  if ! log="$(cat -- "$KERNEL_HEALTH_LOG" 2>&1)"; then
    echo "cannot read the kernel log file $KERNEL_HEALTH_LOG: $log" >&2
    exit 3
  fi
else
  # A user outside the systemd-journal/adm groups gets an empty kernel log, not an error.
  if ! probe="$(journalctl -k -b 0 --no-pager -q -n 1 2>&1)" || [ -z "$probe" ]; then
    echo "cannot read this boot's kernel log with journalctl: ${probe:-empty output}" >&2
    exit 3
  fi
  args=(-k -b 0 --no-pager -q -o short-iso)
  if [ -n "$since" ]; then args+=(--since "$since"); what="this boot since $since"; fi
  if ! log="$(journalctl "${args[@]}" 2>&1)"; then
    echo "journalctl failed: $log" >&2
    exit 3
  fi
fi
hits="$(printf '%s\n' "$log" | sed -n -E "/$pattern/p")"
if [ -n "$hits" ]; then
  n="$(printf '%s\n' "$hits" | wc -l)"
  echo "kernel fault lines in $what: $n; the first ones:"
  printf '%s\n' "$hits" | head -n 6 | cut -c1-200
  exit 5
fi
echo "kernel log clean ($what)"
