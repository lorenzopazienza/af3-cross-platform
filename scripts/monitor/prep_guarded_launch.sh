#!/bin/bash
# Follow-ups 17 to 21 (labels dtp, tri, rcf, aut, xvm): the LAUNCHER of every wrapper (cloud/af3_run_many.sh runs it as
# "bash $LAUNCHER" with the part's settings in the environment), as scripts/monitor/rcx_guarded_launch.sh was for
# follow-up 15. Before each part it applies the stop rule of the specification row (item g):
# scripts/monitor/prep_budget.py guard $PLAN (the label is the folder of the plan, harness/plans/<label>/). GO: it
# becomes cloud/af3_run.sh (exec, same PID, so the wrapper follows it as usual). NO GO: no VM is created; it appends
# the reason to results/<label>_cap_stop.txt (the marker that turns every later guarded launch of that follow-up into
# NO GO until Lorenzo decides, in a section 10 row), touches STOP in the wrapper's run folder so that it starts no
# further part (running parts go on), and exits 3. A part B part refused only because part A has not ended writes no
# marker. Each decision is appended to results/<label>_guard.tsv. Changes nothing in cloud/.
set -u
cd "$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
: "${PLAN:?PLAN is set by cloud/af3_run_many.sh}"
LABEL=$(basename "$(dirname "$PLAN")")
MARKER="${PREP_MARKER:-results/${LABEL}_cap_stop.txt}"
LOG="${PREP_GUARD_LOG:-results/${LABEL}_guard.tsv}"
OUT=$(PREP_MARKER="$MARKER" python3 scripts/monitor/prep_budget.py guard "$PLAN" 2>&1)
RC=$?
echo "$OUT" | sed "s/^/[$LABEL guard] /"
LAST=$(echo "$OUT" | tail -1)
printf '%s\t%s\t%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$PLAN" "$LAST" >> "$LOG"
if [ "$RC" = 0 ]; then
  exec bash "${PREP_LAUNCHER:-cloud/af3_run.sh}"
fi
case "$LAST" in
  *"projected spend"* | *"launch after"* | *"is not a part of follow-ups"*)
    printf '%s\t%s\t%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$PLAN" "$LAST" >> "$MARKER" ;;
esac
# The wrapper's run folder: the newest results/af3/.many/<RUN_ID>/ whose part folder for this plan is running.
NAME=$(basename "$PLAN" .yaml | tr 'A-Z' 'a-z' | tr -c 'a-z0-9_\n-' '-')
sleep 2
for d in $(ls -1d results/af3/.many/*/ 2> /dev/null | sort -r); do
  if [ "$(cat "${d}parts/${PLATFORM:-l4}_${NAME}/status" 2> /dev/null)" = running ]; then
    touch "${d}STOP"
    echo "[$LABEL guard] NO GO: no VM created; ${d}STOP written (no further part starts)"
    exit 3
  fi
done
echo "[$LABEL guard] NO GO: no VM created; the wrapper's run folder was not found (stop it by hand)"
exit 3
