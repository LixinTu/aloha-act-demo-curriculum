# Sourced by the chain scripts (final_protocol_chain.sh, eval_determinism_chain.sh), which run from the
# repository root and set $idle (seconds without GPU compute processes before a stage may start).
# stage <name> <command...>: wait for an idle GPU, run the command, keep its output in logs/<name>.log
# and its exit code in logs/<name>.exit. A stage that finds the GPU busy at its start exits 4 and is
# queued again (up to 5 tries). A stage whose .exit file says 0 is skipped, so a chain can be started
# again after an interruption; any other value stops the chain until someone has looked at
# logs/<name>.log and moved both files away.
stage() {
  local name="$1"; shift
  if [ -e "logs/$name.exit" ]; then
    local prev; prev="$(cat "logs/$name.exit")"
    if [ "$prev" = 0 ]; then
      echo "# $name already done (logs/$name.exit says 0); skipped"
      return 0
    fi
    echo "logs/$name.exit says '$prev' from an earlier attempt; look at logs/$name.log, then move both away to run it again" >&2
    return 2
  fi
  local rc=4 tries=0
  while [ "$rc" = 4 ] && [ "$tries" -lt 5 ]; do
    tries=$((tries + 1))
    echo "# $name queued at $(date -Is) (try $tries): $*"
    scripts/wait_gpu_idle.sh "$idle" "$@" >> "logs/$name.log" 2>&1
    rc=$?
    if [ "$rc" = 4 ]; then echo "# $name found the GPU busy at its start (exit 4); queued again"; fi
  done
  echo "$rc" > "logs/$name.exit"
  echo "# $name exit=$rc at $(date -Is)"
  return "$rc"
}
