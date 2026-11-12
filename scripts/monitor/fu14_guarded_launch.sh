#!/bin/bash
# Follow-up 14 (label f32): the A100 wrapper's LAUNCHER (cloud/af3_run_many.sh runs it as "bash $LAUNCHER" with the
# part's settings in the environment). Before each part it applies the stop rule of the specification row of
# 2026-11-12 (item g): scripts/monitor/fu14_budget.py guard $PLAN. GO: it becomes cloud/af3_run.sh (exec, same PID,
# so the wrapper follows it as usual). NO GO: no VM is created; it appends the reason to results/f32_cap_stop.txt
# (the marker that turns every later guarded launch into NO GO until Lorenzo decides, in a section 10 row), touches
# STOP in the wrapper's run folder so that it starts no further part (running parts go on), and exits 3. The wrapper
# records the part as "no VM created"; retry_arm_par.sh would retry it in its next round and the marker refuses it
# again, so Lorenzo stops that wrapper with Ctrl-C when the marker appears. Each decision is appended to
# results/f32_guard.tsv. Changes nothing in cloud/.
set -u
cd "$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
: "${PLAN:?PLAN is set by cloud/af3_run_many.sh}"
MARKER="${F32_MARKER:-results/f32_cap_stop.txt}"
LOG="${F32_GUARD_LOG:-results/f32_guard.tsv}"
OUT=$(F32_MARKER="$MARKER" python3 scripts/monitor/fu14_budget.py guard "$PLAN" 2>&1)
RC=$?
echo "$OUT" | sed 's/^/[fu14 guard] /'
printf '%s\t%s\t%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$PLAN" "$(echo "$OUT" | tail -1)" >> "$LOG"
if [ "$RC" = 0 ]; then
  exec bash "${F32_LAUNCHER:-cloud/af3_run.sh}"
fi
printf '%s\t%s\t%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$PLAN" "$(echo "$OUT" | tail -1)" >> "$MARKER"
# The wrapper's run folder: the newest results/af3/.many/<RUN_ID>/ whose part folder for this plan is running.
NAME=$(basename "$PLAN" .yaml | tr 'A-Z' 'a-z' | tr -c 'a-z0-9_\n-' '-')
sleep 2
for d in $(ls -1d results/af3/.many/*/ 2> /dev/null | sort -r); do
  if [ "$(cat "${d}parts/${PLATFORM:-a100}_${NAME}/status" 2> /dev/null)" = running ]; then
    touch "${d}STOP"
    echo "[fu14 guard] NO GO: no VM created; ${d}STOP written (no further part starts); marker $MARKER"
    exit 3
  fi
done
echo "[fu14 guard] NO GO: no VM created; the wrapper's run folder was not found (stop it by hand); marker $MARKER"
exit 3
