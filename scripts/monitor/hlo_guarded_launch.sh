#!/bin/bash
# Follow-up 16 (label hlo): the LAUNCHER of the follow-up 16 wrapper (cloud/af3_run_many.sh runs it as "bash $LAUNCHER"
# with the part's settings in the environment), as scripts/monitor/rcx_guarded_launch.sh was for follow-up 15. Before
# each part it applies the stop rule of the specification row of 2026-11-16 (item g): scripts/monitor/hlo_budget.py
# guard $PLAN. GO: it becomes cloud/af3_run.sh (exec, same PID, so the wrapper follows it as usual). NO GO: no VM is
# created; it appends the reason to results/hlo_cap_stop.txt (the marker that turns every later guarded launch into
# NO GO until Lorenzo decides, in a section 10 row), touches STOP in the wrapper's run folder so that it starts no
# further part (a running part goes on), and exits 3. Each decision is appended to results/hlo_guard.tsv. Changes
# nothing in cloud/.
set -u
cd "$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
: "${PLAN:?PLAN is set by cloud/af3_run_many.sh}"
MARKER="${HLO_MARKER:-results/hlo_cap_stop.txt}"
LOG="${HLO_GUARD_LOG:-results/hlo_guard.tsv}"
OUT=$(HLO_MARKER="$MARKER" python3 scripts/monitor/hlo_budget.py guard "$PLAN" 2>&1)
RC=$?
echo "$OUT" | sed 's/^/[hlo guard] /'
LAST=$(echo "$OUT" | tail -1)
printf '%s\t%s\t%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$PLAN" "$LAST" >> "$LOG"
if [ "$RC" = 0 ]; then
  exec bash "${HLO_LAUNCHER:-cloud/af3_run.sh}"
fi
case "$LAST" in
  *"projected spend"* | *"launch after"* | *"is not a follow-up 16 part"*)
    printf '%s\t%s\t%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$PLAN" "$LAST" >> "$MARKER" ;;
esac
# The wrapper's run folder: the newest results/af3/.many/<RUN_ID>/ whose part folder for this plan is running.
NAME=$(basename "$PLAN" .yaml | tr 'A-Z' 'a-z' | tr -c 'a-z0-9_\n-' '-')
sleep 2
for d in $(ls -1d results/af3/.many/*/ 2> /dev/null | sort -r); do
  if [ "$(cat "${d}parts/${PLATFORM:-v5e}_${NAME}/status" 2> /dev/null)" = running ]; then
    touch "${d}STOP"
    echo "[hlo guard] NO GO: no VM created; ${d}STOP written (no further part starts)"
    exit 3
  fi
done
echo "[hlo guard] NO GO: no VM created; the wrapper's run folder was not found (stop it by hand)"
exit 3
