#!/bin/bash
# Follow-up 15 (label rcx): the LAUNCHER of every follow-up 15 wrapper (cloud/af3_run_many.sh runs it as
# "bash $LAUNCHER" with the part's settings in the environment), as scripts/monitor/fu14_guarded_launch.sh was for
# the A100 of follow-up 14. Before each part it applies the stop rule of the specification row of 2026-11-14
# (item g): scripts/monitor/rcx_budget.py guard $PLAN. GO: it becomes cloud/af3_run.sh (exec, same PID, so the
# wrapper follows it as usual). NO GO: no VM is created; it appends the reason to results/rcx_cap_stop.txt (the
# marker that turns every later guarded launch into NO GO until Lorenzo decides, in a section 10 row), touches STOP
# in the wrapper's run folder so that it starts no further part (running parts go on), and exits 3. A part A3 part
# refused only because part A4 has not ended writes no marker (the gate of item g has not opened yet). Each decision
# is appended to results/rcx_guard.tsv. Changes nothing in cloud/.
set -u
cd "$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
: "${PLAN:?PLAN is set by cloud/af3_run_many.sh}"
MARKER="${RCX_MARKER:-results/rcx_cap_stop.txt}"
LOG="${RCX_GUARD_LOG:-results/rcx_guard.tsv}"
OUT=$(RCX_MARKER="$MARKER" python3 scripts/monitor/rcx_budget.py guard "$PLAN" 2>&1)
RC=$?
echo "$OUT" | sed 's/^/[rcx guard] /'
LAST=$(echo "$OUT" | tail -1)
printf '%s\t%s\t%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$PLAN" "$LAST" >> "$LOG"
if [ "$RC" = 0 ]; then
  exec bash "${RCX_LAUNCHER:-cloud/af3_run.sh}"
fi
case "$LAST" in
  *"projected spend"* | *"launch after"* | *"is not a follow-up 15 part"*)
    printf '%s\t%s\t%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$PLAN" "$LAST" >> "$MARKER" ;;
esac
# The wrapper's run folder: the newest results/af3/.many/<RUN_ID>/ whose part folder for this plan is running.
NAME=$(basename "$PLAN" .yaml | tr 'A-Z' 'a-z' | tr -c 'a-z0-9_\n-' '-')
sleep 2
for d in $(ls -1d results/af3/.many/*/ 2> /dev/null | sort -r); do
  if [ "$(cat "${d}parts/${PLATFORM:-a100}_${NAME}/status" 2> /dev/null)" = running ]; then
    touch "${d}STOP"
    echo "[rcx guard] NO GO: no VM created; ${d}STOP written (no further part starts)"
    exit 3
  fi
done
echo "[rcx guard] NO GO: no VM created; the wrapper's run folder was not found (stop it by hand)"
exit 3
