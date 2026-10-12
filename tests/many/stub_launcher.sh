#!/bin/bash
# Stand-in for cloud/af3_run.sh in the af3_run_many.sh tests: no gcloud, no VM. Behaves like the launcher
# as af3_run_many.sh sees it: prints the session line, writes results/af3/.launcher/<session>/launch_record.txt
# (CREATED=1 once a VM "exists"), exits with the launcher's codes. Per plan name, STUB_MODES="name:mode ..."
#   ok    VM created, job exit 0, session fetched, exit 0 (default)
#   novm  no zone had capacity: no VM, exit 1
#   fail  VM created, job failed, VM deleted, exit 1
#   kept  VM created, job not finished, VM kept (resume command printed), exit 1
# STUB_SLEEP seconds of "running"; events (start/end, epoch, platform, plan) appended to $STUB_EVENTS.
set -u
name=$(basename "$PLAN" .yaml | tr 'A-Z' 'a-z' | tr -c 'a-z0-9_\n-' '-')
mode=ok
for m in ${STUB_MODES:-}; do [ "${m%%:*}" = "$name" ] && mode=${m#*:}; done
stamp=$(date -u +%Y%m%dT%H%M%SZ)
session="${stamp}_${PLATFORM}_${name}"
echo "   session: results/af3/$session/   VM: af3-run-$PLATFORM-x"
mkdir -p "results/af3/.launcher/$session"
echo "SESSION=$session" > "results/af3/.launcher/$session/launch_record.txt"
echo "start $(python3 -c "import time; print(time.time())") $PLATFORM $name $$ YES=${YES:-} ZONE=${ZONE:-} EXTRA=${SPOT:-} ZONES=${ZONES:-} PROV=${PROVISIONING:-}" >> "$STUB_EVENTS"
if [ "$mode" = novm ]; then
  echo "!! No zone had capacity for g2-standard-8 (standard). Tried, in order: a b"
  echo "end $(python3 -c "import time; print(time.time())") $PLATFORM $name $$" >> "$STUB_EVENTS"
  exit 1
fi
echo "CREATED=1" >> "results/af3/.launcher/$session/launch_record.txt"
trap 'echo "end $(python3 -c "import time; print(time.time())") $PLATFORM $name $$ TERM" >> "$STUB_EVENTS"; echo ">> Deleting VM (interrupted)"; exit 130' TERM
sleep "${STUB_SLEEP:-2}" &
wait $!
echo "end $(python3 -c "import time; print(time.time())") $PLATFORM $name $$" >> "$STUB_EVENTS"
case "$mode" in
  ok) mkdir -p "results/af3/$session"; echo ">> exit code 0"; exit 0 ;;
  fail) echo ">> exit code 1"; exit 1 ;;
  kept) echo "!! af3-run-$PLATFORM-x is NOT deleted: the API reports it RUNNING and its job has not finished."
        echo "!!   RESUME_VM=af3-run-$PLATFORM-x SESSION=$session bash cloud/af3_run.sh"; exit 1 ;;
esac
