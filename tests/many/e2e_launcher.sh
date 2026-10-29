#!/bin/bash
# The real cloud/af3_run.sh (as cloud/af3_run.testkit.sh in the scratch repository) against the fake gcloud (tests/cloud), one fake state folder per launch, so
# several fake VMs can run at once. E2E_ZONES="planname:FAKE_ZONES spec" gives a part its zone behaviours
# (for example e2e_l4_nocap=europe-west4-a:sync+us-east4-a:sync; "+" separates zones). Sourced kit: $KIT (tests/cloud of the scratch copy).
. "$KIT/common.sh"
name=$(basename "$PLAN" .yaml)
zones=""
for z in ${E2E_ZONES:-}; do [ "${z%%=*}" = "$name" ] && zones=$(echo "${z#*=}" | tr "+" " "); done
D="$WORK/many_state/${name}_$(date +%s)_$$"
mkdir -p "$D/home"
if [ -n "$zones" ]; then
  fake_env "$D" FAKE_SCENARIO=ok FAKE_PLATFORM="$PLATFORM" FAKE_ZONES="$zones" /bin/bash cloud/af3_run.testkit.sh
else
  fake_env "$D" FAKE_SCENARIO=ok FAKE_PLATFORM="$PLATFORM" /bin/bash cloud/af3_run.testkit.sh
fi
rc=$?
echo "$D" >> "$WORK/many_state/dirs"
exit $rc
