#!/usr/bin/env bash
# Run a long command in the background with a log and an exit-code file.
# Usage: scripts/run_bg.sh <run-name> <command...>
# Writes logs/<run-name>.log and logs/<run-name>.exit (always, even on failure); prints the PID.
# Holds a systemd inhibitor lock while it runs, so sleep, lid close and ordinary reboots wait for it
# (a reboot killed a run at 28% on 2026-11-08). To reboot anyway: systemctl reboot -i
set -uo pipefail
name="$1"; shift
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$root/logs"
log="$root/logs/$name.log"; exitf="$root/logs/$name.exit"
rm -f "$exitf"
(
  cd "$root"
  echo "# started $(date -Is) : $*" >> "$log"
  systemd-inhibit --what=shutdown:sleep:idle:handle-lid-switch --mode=block \
    --who=aloha-demo-curriculum --why="GPU run $name, see logs/$name.log" "$@" >> "$log" 2>&1
  rc=$?
  echo "$rc" > "$exitf"
  echo "# finished $(date -Is) exit=$rc" >> "$log"
) &
echo "pid=$! log=$log exit_file=$exitf"
