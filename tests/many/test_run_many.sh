#!/bin/bash
# Tests of cloud/af3_run_many.sh (draft), on a scratch copy of the repository, against the fake gcloud of
# tests/cloud (same guard: refuses to run unless the fake is the gcloud on PATH). Creates no cloud resources.
#   bash tests/many/test_run_many.sh            # from the scratch copy's root; CLOUD_TEST_WORK sets WORK
# Part 1 uses a stub launcher (tests/many/stub_launcher.sh) for the wrapper's own logic: caps per platform,
# staggering, one launcher per part (duplicates, earlier sessions, locks), retries only without a VM, kept
# VMs, the STOP file, Ctrl-C, the prompt, refusals and the summary. Part 2 runs the real cloud/af3_run.sh
# against the fake gcloud, one fake state per launch (tests/many/e2e_launcher.sh). Part 3 feeds the
# generated main-study parts file to the wrapper with the stub launcher.
MANY=$(cd "$(dirname "$0")" && pwd)
. "$MANY/../cloud/common.sh"
export KIT WORK
PASS=0 FAIL=0
check() { if eval "$2"; then PASS=$((PASS + 1)); echo "  ok   $1"; else FAIL=$((FAIL + 1)); echo "  FAIL $1"; fi; }

bash "$KIT/mkrepo.sh" > /dev/null
R="$WORK/repo"
cp "$REPO/cloud/af3_run_many.sh" "$R/cloud/"
mkplan() {   # NAME PLATFORM CONFIG TARGETS SEEDS REPS WARM DEADLINE
  mkdir -p "$R/harness/plans/t"
  cat > "$R/harness/plans/t/$1.yaml" <<PLAN
description: af3_run_many test part $1
label: main
weights: gcs
configs:
  $2: $3
targets:
  default: [$4]
seeds: [$5]
fresh_reps: $6
warm_rerun_seed: $7
num_recycles: default
num_diffusion_samples: default
run_timeout_min: 30
deadline_min:
  $2: $8
PLAN
}
for i in 01 02 03 04 05 06; do mkplan t_l4_$i l4 l4_xla 7U3J 1 1 none 30; done
for i in 01 02 03; do mkplan t_v5_$i v5e tpu_xla 7U3J 1 1 none 30; done
mkplan t_cpu_01 cpu cpu_xla 7U3J 1 1 none 30
for i in 01 02 03 04 05 06; do mkplan t_a100_$i a100 a100_xla 7U3J 1 1 none 30; done
mkplan t_cpuonly cpu cpu_xla 7U3J 1 1 none 30

many() {   # NAME PARTS_FILE [VAR=value ...]: the wrapper with the stub launcher; output, rc and events in state_many_NAME
  local n=$1 parts=$2; shift 2
  S="$WORK/state_many_$n"; rm -rf "$S"; mkdir -p "$S"
  (cd "$R" && fake_env "$S" PARTS="$parts" ALLOW_OTHER_LAUNCHERS=1 STAGGER_S=1 POLL_S=1 \
    LAUNCHER="$MANY/stub_launcher.sh" STUB_EVENTS="$S/events" "$@" /bin/bash cloud/af3_run_many.sh > "$S/out" 2>&1)
  echo $? > "$S/rc"; touch "$S/events"
}
rc() { cat "$WORK/state_many_$1/rc"; }
starts() { grep -c '^start' "$WORK/state_many_$1/events"; }
starts_of() { grep -c "^start [0-9.]* [a-z0-9]* $2 " "$WORK/state_many_$1/events"; }
maxconc() {   # NAME PLATFORM: largest number of stub launchers of PLATFORM running at once
  python3 - "$WORK/state_many_$1/events" "$2" <<'PY'
import sys
ev = [l.split() for l in open(sys.argv[1])]
pts = sorted((float(e[1]), 1 if e[0] == 'start' else -1) for e in ev if e[2] == sys.argv[2])
cur = best = 0
for _, d in sorted(pts, key=lambda x: (x[0], x[1])):
  cur += d; best = max(best, cur)
print(best)
PY
}
mingap() { python3 -c "
import sys
s = sorted(float(l.split()[1]) for l in open('$WORK/state_many_$1/events') if l.startswith('start'))
print(min((b - a for a, b in zip(s, s[1:])), default=99))"; }
oks() { grep -cE "^(l4|a100|v5e|v6e|cpu) +harness/[^ ]+ +ok: launcher exit 0" "$WORK/state_many_$1/out"; }   # summary lines only
fresh() { rm -rf "$R/results"; }

echo "======== part 1: wrapper logic (stub launcher)"
P="$WORK/parts_caps.txt"
{ for i in 01 02 03 04 05 06; do echo "l4 harness/plans/t/t_l4_$i.yaml"; done
  echo "# a comment"; echo
  for i in 01 02 03; do echo "v5e harness/plans/t/t_v5_$i.yaml   # trailing comment"; done
  echo "cpu harness/plans/t/t_cpu_01.yaml"; } > "$P"
fresh; many caps "$P" YES=1 MAX_L4=2 MAX_V5E=1 MAX_CPU=1 STUB_SLEEP=3
check "caps: exit 0" '[ "$(rc caps)" = 0 ]'
check "caps: every part launched once (10)" '[ "$(starts caps)" = 10 ] && [ "$(grep -c "^start" "$S/events" | tr -d " ")" = 10 ]'
check "caps: at most 2 L4 at once, and 2 reached" '[ "$(maxconc caps l4)" = 2 ]'
check "caps: at most 1 v5e at once" '[ "$(maxconc caps v5e)" = 1 ]'
check "caps: launches at least STAGGER_S apart" 'python3 -c "import sys; sys.exit(0 if $(mingap caps) >= 0.95 else 1)"'
check "caps: every launcher got YES=1" '[ "$(grep -c "YES=1 " "$S/events")" = 10 ]'
check "caps: one log per part" '[ "$(ls "$R"/results/af3/.many/*/t_*.log | wc -l | tr -d " ")" = 10 ]'
check "caps: summary lists 10 ok parts" '[ "$(oks caps)" = 10 ] && grep -q "parts not started: 0" "$S/out"'
check "caps: locks released" '[ -z "$(ls "$R/results/af3/.many/locks" 2> /dev/null)" ]'

fresh; printf 'l4 harness/plans/t/t_l4_01.yaml\nl4 harness/plans/t/t_l4_01.yaml\n' > "$WORK/parts_dup.txt"
many dup "$WORK/parts_dup.txt" YES=1
check "duplicate part: refused (exit 2), nothing launched" '[ "$(rc dup)" = 2 ] && [ "$(starts dup)" = 0 ] && grep -q "listed twice" "$S/out"'
for bad in "gpu harness/plans/t/t_l4_01.yaml" "l4 harness/plans/t/missing.yaml" "l4 harness/plans/t/t_cpuonly.yaml" \
           "l4 harness/plans/t/t_l4_01.yaml PLAN=x" "l4 harness/plans/t/t_l4_01.yaml FOO=1"; do
  fresh; { echo "l4 harness/plans/t/t_l4_02.yaml"; echo "$bad"; } > "$WORK/parts_bad.txt"
  many bad "$WORK/parts_bad.txt" YES=1
  check "refused, nothing launched: '$bad'" '[ "$(rc bad)" = 2 ] && [ "$(starts bad)" = 0 ]'
done
fresh; many env "$WORK/parts_dup.txt" YES=1 AF2_COMMIT=x
check "cloud/env.sh sourced: refused" '[ "$(rc env)" = 2 ] && grep -q "env.sh is sourced" "$S/out"'
# A stand-in for another running launcher. It carries the test kit's launcher name (cloud/af3_run.testkit.sh), never
# cloud/af3_run.sh, so it cannot block a real wrapper started while the tests run; the wrapper under test scans for
# it because fake_env sets AF3_TEST_LAUNCHER_NAME. '; :' keeps bash from exec'ing sleep, which would drop the name.
fresh; (bash -c 'sleep 15; :' cloud/af3_run.testkit.sh &) ; sleep 0.5
printf 'l4 harness/plans/t/t_l4_01.yaml\n' > "$WORK/parts_one.txt"
S="$WORK/state_many_others"; rm -rf "$S"; mkdir -p "$S"
(cd "$R" && fake_env "$S" PARTS="$WORK/parts_one.txt" YES=1 LAUNCHER="$MANY/stub_launcher.sh" STUB_EVENTS="$S/events" \
  /bin/bash cloud/af3_run_many.sh > "$S/out" 2>&1); echo $? > "$S/rc"; touch "$S/events"
check "another af3_run.sh running: refused without ALLOW_OTHER_LAUNCHERS=1" '[ "$(rc others)" = 2 ] && grep -q "already running" "$S/out"'
# The same stand-in seen by a wrapper run as a real one (AF3_TEST_LAUNCHER_NAME unset): not counted, so the prompt is
# reached (answered no: nothing launched). A test process never blocks a real launch.
S="$WORK/state_many_real_scan"; rm -rf "$S"; mkdir -p "$S"
(cd "$R" && echo n | fake_env "$S" PARTS="$WORK/parts_one.txt" LAUNCHER="$MANY/stub_launcher.sh" STUB_EVENTS="$S/events" \
  env -u AF3_TEST_LAUNCHER_NAME -u YES /bin/bash cloud/af3_run_many.sh > "$S/out" 2>&1); echo $? > "$S/rc"; touch "$S/events"
check "a test launcher running: a real wrapper does not count it" \
  '[ "$(rc real_scan)" = 2 ] && grep -q "Not started" "$S/out" && ! grep -q "already running" "$S/out"'
pkill -f "sleep 15" 2> /dev/null
# The real wrapper's pattern (launcher_pattern of cloud/af3_run_many.sh, checked as strings: no test ever starts a
# process named cloud/af3_run.sh): any checkout, relative or absolute path; never the test kit's name or a longer one.
eval "$(sed -n '/^launcher_pattern() {/,/^}/p' "$REPO/cloud/af3_run_many.sh")"
PAT=$(launcher_pattern af3_run.sh)
hit=0 miss=0
for c in "bash cloud/af3_run.sh" "/bin/bash /Users/x/Desktop/alphafold-on-tpu/cloud/af3_run.sh" \
         "/bin/bash /Users/x/Desktop/af3-fu7-10/cloud/af3_run.sh" "bash -c sleep 15; : cloud/af3_run.sh"; do
  echo "$c" | grep -Eq "$PAT" && hit=$((hit + 1))
done
for c in "bash cloud/af3_run.testkit.sh" "bash cloud/af3_run_many.sh" "bash cloud/af3_run.shx" "bash cloud/af3_runXsh" \
         "bash xcloud/af3_run.sh"; do
  echo "$c" | grep -Eq "$PAT" && miss=$((miss + 1))
done
check "real wrapper pattern: matches the launcher of any checkout (4/4), no test or other name (0/5)" '[ "$hit" = 4 ] && [ "$miss" = 0 ]'
check "scratch repository: no cloud/af3_run.sh, only cloud/af3_run.testkit.sh" \
  '[ ! -e "$R/cloud/af3_run.sh" ] && [ -f "$R/cloud/af3_run.testkit.sh" ]'

fresh; many prompt_no "$WORK/parts_one.txt" < /dev/null
check "prompt answered no: nothing launched" '[ "$(rc prompt_no)" = 2 ] && [ "$(starts prompt_no)" = 0 ] && grep -q "Not started" "$S/out"'
fresh; S="$WORK/state_many_prompt_yes"; rm -rf "$S"; mkdir -p "$S"
(cd "$R" && echo y | fake_env "$S" PARTS="$WORK/parts_one.txt" ALLOW_OTHER_LAUNCHERS=1 POLL_S=1 \
  LAUNCHER="$MANY/stub_launcher.sh" STUB_EVENTS="$S/events" /bin/bash cloud/af3_run_many.sh > "$S/out" 2>&1); echo $? > "$S/rc"
check "prompt answered y: launched, exit 0" '[ "$(rc prompt_yes)" = 0 ] && [ "$(starts prompt_yes)" = 1 ]'

# One launcher per part: earlier sessions and locks.
fresh; mkdir -p "$R/results/af3/.launcher/20261020T000000Z_l4_t_l4_01" "$R/results/af3/.launcher/20261020T000001Z_l4_t_l4_02"
printf 'SESSION=x\nCREATED=1\n' > "$R/results/af3/.launcher/20261020T000000Z_l4_t_l4_01/launch_record.txt"
printf 'SESSION=y\n' > "$R/results/af3/.launcher/20261020T000001Z_l4_t_l4_02/launch_record.txt"
printf 'l4 harness/plans/t/t_l4_01.yaml\nl4 harness/plans/t/t_l4_02.yaml\n' > "$WORK/parts_two.txt"
many earlier "$WORK/parts_two.txt" YES=1
check "part with a VM-creating earlier session: skipped, not launched" '[ "$(starts_of earlier t_l4_01)" = 0 ] && grep -q "t_l4_01.yaml *skipped: already launched" "$S/out"'
check "part whose earlier launch created no VM: launched" '[ "$(starts_of earlier t_l4_02)" = 1 ]'
check "a skipped part does not fail the run (exit 0)" '[ "$(rc earlier)" = 0 ]'
many relaunch "$WORK/parts_two.txt" YES=1 ALLOW_RELAUNCH="t_l4_01 t_l4_02"
check "ALLOW_RELAUNCH launches parts with earlier sessions" '[ "$(starts relaunch)" = 2 ] && [ "$(rc relaunch)" = 0 ]'
fresh; mkdir -p "$R/results/af3/.many/locks/l4_t_l4_01" "$R/results/af3/.many/locks/l4_t_l4_02"
sleep 30 & LIVE=$!
echo "$LIVE other" > "$R/results/af3/.many/locks/l4_t_l4_01/owner"
echo "999999 old" > "$R/results/af3/.many/locks/l4_t_l4_02/owner"
many locks "$WORK/parts_two.txt" YES=1
check "lock held by a live launcher: part skipped" '[ "$(starts_of locks t_l4_01)" = 0 ] && grep -q "another launcher holds" "$S/out"'
check "stale lock (process gone, no session): removed, part launched" '[ "$(starts_of locks t_l4_02)" = 1 ]'
kill $LIVE 2> /dev/null

# Retries only without a VM; failed, kept.
fresh; printf 'l4 harness/plans/t/t_l4_01.yaml\nl4 harness/plans/t/t_l4_02.yaml\nl4 harness/plans/t/t_l4_03.yaml\nl4 harness/plans/t/t_l4_04.yaml\n' > "$WORK/parts_four.txt"
many retry "$WORK/parts_four.txt" YES=1 MAX_ATTEMPTS=3 RETRY_WAIT_S=1 STUB_SLEEP=1 \
  STUB_MODES="t_l4_01:novm t_l4_02:fail t_l4_03:kept"
check "no VM created: launched again, up to MAX_ATTEMPTS (3)" '[ "$(starts_of retry t_l4_01)" = 3 ] && grep -q "t_l4_01.yaml *FAILED, no VM created.*after 3 attempt" "$S/out"'
check "one log per attempt" 'M="$(ls -d "$R"/results/af3/.many/2*/ | tail -1)"; [ -f "${M}t_l4_01.log" ] && [ -f "${M}t_l4_01.attempt2.log" ] && [ -f "${M}t_l4_01.attempt3.log" ]'
check "failure after a VM was created: never relaunched" '[ "$(starts_of retry t_l4_02)" = 1 ] && grep -q "t_l4_02.yaml *FAILED after a VM was created" "$S/out"'
check "kept VM: reported with its resume command, not relaunched" '[ "$(starts_of retry t_l4_03)" = 1 ] && grep -q "t_l4_03.yaml *VM KEPT.*RESUME_VM=af3-run-l4-x" "$S/out"'
check "exit 1 when a part failed; the ok part is ok" '[ "$(rc retry)" = 1 ] && grep -q "t_l4_04.yaml *ok:" "$S/out" && grep -q "parts not started: 3" "$S/out"'
check "failed parts listed with their sessions in relaunch_parts.txt and an unchanged relaunch command; never relaunched" \
  'M="$(ls -d "$R"/results/af3/.many/2*/ | tail -1)"; [ "$(grep -c "^l4 " "${M}relaunch_parts.txt")" = 2 ] && grep -q "t_l4_01.yaml .*FAILED, no VM" "${M}relaunch_parts.txt" && grep -q "t_l4_02.yaml .*FAILED after a VM was created.*session" "${M}relaunch_parts.txt" && grep -q "ALLOW_RELAUNCH=\"t_l4_01 t_l4_02\" .*bash cloud/af3_run_many.sh" "$S/out" && [ "$(starts_of retry t_l4_02)" = 1 ]'
check "kept VM: listed for resume, not in the relaunch file" \
  'M="$(ls -d "$R"/results/af3/.many/2*/ | tail -1)"; ! grep -q t_l4_03 "${M}relaunch_parts.txt" && grep -A1 "resume, do not relaunch" "$S/out" | grep -q "t_l4_03.yaml: RESUME_VM=af3-run-l4-x"'
fresh; many noretry "$WORK/parts_one.txt" YES=1 STUB_MODES="t_l4_01:novm"
check "MAX_ATTEMPTS=1 (default): no second launch" '[ "$(starts noretry)" = 1 ] && [ "$(rc noretry)" = 1 ]'

# Per-part settings reach that launcher only.
fresh; printf 'l4 harness/plans/t/t_l4_01.yaml ZONE=us-east4-a SPOT=1\nl4 harness/plans/t/t_l4_02.yaml ZONES=us-east4-a,us-east4-c\n' > "$WORK/parts_vars.txt"
many vars "$WORK/parts_vars.txt" YES=1
check "per-part ZONE and SPOT passed to that part only" 'grep "t_l4_01" "$S/events" | grep -q "ZONE=us-east4-a EXTRA=1" && grep "^start.* t_l4_02 " "$S/events" | grep -q "ZONE= EXTRA= ZONES=us-east4-a us-east4-c PROV= MT="'

# v6e parts run on demand (section 10, main study execution row): PROVISIONING=standard unless the line sets it.
mkplan t_v6_01 v6e tpu_xla 7U3J 1 1 none 30
mkplan t_v6_02 v6e tpu_xla 7U3J 1 1 none 30
fresh; printf 'v6e harness/plans/t/t_v6_01.yaml\nv6e harness/plans/t/t_v6_02.yaml PROVISIONING=flex_start\nl4 harness/plans/t/t_l4_01.yaml\n' > "$WORK/parts_v6.txt"
many v6prov "$WORK/parts_v6.txt" YES=1
check "v6e part gets PROVISIONING=standard; an explicit PROVISIONING on its line wins; other platforms get none" \
  'grep "^start.* t_v6_01 " "$S/events" | grep -q "PROV=standard MT=" && grep "^start.* t_v6_02 " "$S/events" | grep -q "PROV=flex_start MT=" && grep "^start.* t_l4_01 " "$S/events" | grep -q "PROV= MT="'

# STOP file and Ctrl-C.
fresh; printf 'l4 harness/plans/t/t_l4_01.yaml\nl4 harness/plans/t/t_l4_02.yaml\nl4 harness/plans/t/t_l4_03.yaml\n' > "$WORK/parts_three.txt"
S="$WORK/state_many_stop"; rm -rf "$S"; mkdir -p "$S"
(cd "$R" && fake_env "$S" PARTS="$WORK/parts_three.txt" YES=1 ALLOW_OTHER_LAUNCHERS=1 MAX_L4=1 POLL_S=1 STUB_SLEEP=4 \
  LAUNCHER="$MANY/stub_launcher.sh" STUB_EVENTS="$S/events" /bin/bash cloud/af3_run_many.sh > "$S/out" 2>&1; echo $? > "$S/rc") &
for i in $(seq 1 50); do [ -s "$S/events" ] && break; sleep 0.2; done
touch "$(ls -d "$R"/results/af3/.many/2*/ | tail -1)STOP"; wait
check "STOP file: no further part starts, the running one finishes" '[ "$(starts stop)" = 1 ] && [ "$(grep -c "^end" "$S/events")" = 1 ] && [ "$(grep -cE "^l4 +harness/[^ ]+ +not started \(stopped\)$" "$S/out")" = 2 ] && [ "$(rc stop)" = 1 ]'
fresh; S="$WORK/state_many_ctrlc"; rm -rf "$S"; mkdir -p "$S"
python3 "$MANY/many_ctrlc.py" "$S" "$S/events" 1 1 PARTS="$WORK/parts_three.txt" YES=1 ALLOW_OTHER_LAUNCHERS=1 \
  MAX_L4=1 POLL_S=1 STUB_SLEEP=4 LAUNCHER="$MANY/stub_launcher.sh" STUB_EVENTS="$S/events" > "$S/rc"
check "Ctrl-C: no further part starts; the running launcher ignores it and finishes; summary; exit 1" \
  '[ "$(starts ctrlc)" = 1 ] && grep -q "^end [0-9.]* l4 t_l4_01 [0-9]*$" "$S/events" && grep -q "interrupt: no further part" "$S/out" && grep -q "not started (stopped)" "$S/out" && [ "$(cat "$S/rc")" = 1 ]'
fresh; S="$WORK/state_many_ctrlc2"; rm -rf "$S"; mkdir -p "$S"
python3 "$MANY/many_ctrlc.py" "$S" "$S/events" 1 2 PARTS="$WORK/parts_three.txt" YES=1 ALLOW_OTHER_LAUNCHERS=1 \
  MAX_L4=1 POLL_S=1 STUB_SLEEP=5 LAUNCHER="$MANY/stub_launcher.sh" STUB_EVENTS="$S/events" > "$S/rc"
sleep 6
check "second Ctrl-C: the wrapper leaves at once, listing the PID; the launcher goes on to its end" \
  '[ "$(cat "$S/rc")" = 1 ] && grep -q "second interrupt" "$S/out" && grep -q "PID [0-9]* l4 harness/plans/t/t_l4_01.yaml" "$S/out" && grep -q "^end [0-9.]* l4 t_l4_01 [0-9]*$" "$S/events"'

echo "======== part 1b: A100 parts (follow-up 3): one machine type, at most 2 in us-central1 and 1 in europe-west4"
regconc() {   # NAME REGION: largest number of A100 stub launchers running at once in REGION (field ZR=)
  python3 - "$WORK/state_many_$1/events" "$2" <<'PY'
import sys
starts, ev = {}, []
for l in open(sys.argv[1]):
  f = l.split()
  if f[0] == 'start' and f[2] == 'a100':
    zr = [x for x in f if x.startswith('ZR=')][0][3:]
    starts[f[4]] = zr
    if zr == sys.argv[2]: ev.append((float(f[1]), 1))
  elif f[0] == 'end' and f[2] == 'a100' and starts.get(f[4]) == sys.argv[2]:
    ev.append((float(f[1]), -1))
cur = best = 0
for _, d in sorted(ev, key=lambda x: (x[0], x[1])):
  cur += d; best = max(best, cur)
print(best)
PY
}
P="$WORK/parts_a100.txt"
{ for i in 01 02 03 04; do echo "a100 harness/plans/t/t_a100_$i.yaml SPOT=0"; done
  echo "a100 harness/plans/t/t_a100_05.yaml SPOT=0 ZONES=europe-west4-a"; } > "$P"
fresh; many a100 "$P" YES=1 MAX_A100=3 A100_MACHINE_TYPE=a2-ultragpu-1g STUB_SLEEP=3
check "a100: exit 0, every part launched once (5)" '[ "$(rc a100)" = 0 ] && [ "$(grep -c "^start [0-9.]* a100 " "$WORK/state_many_a100/events")" = 5 ]'
check "a100: every launcher got MACHINE_TYPE=a2-ultragpu-1g" \
  '[ "$(grep "^start" "$WORK/state_many_a100/events" | grep -c " MT=a2-ultragpu-1g ")" = 5 ]'
check "a100: at most MAX_A100=3 at a time, and 3 reached" '[ "$(maxconc a100 a100)" = 3 ]'
check "a100: at most 2 at a time in us-central1, and 2 reached" '[ "$(regconc a100 us-central1)" = 2 ]'
check "a100: at most 1 at a time in europe-west4" '[ "$(regconc a100 europe-west4)" = 1 ]'
check "a100: unpinned parts get one region's zones (us-central1, us-east5, europe-west4 or us-east4)" \
  '! grep "^start.* a100 t_a100_0[1-4] " "$WORK/state_many_a100/events" | grep -v -e "ZONES=us-central1-a us-central1-c PROV" -e "ZONES=us-east5-a us-east5-b PROV" -e "ZONES=europe-west4-a PROV" -e "ZONES=us-east4-c PROV" | grep -q .'
check "a100: the pinned part kept its ZONES=europe-west4-a" \
  'grep "^start.* a100 t_a100_05 " "$WORK/state_many_a100/events" | grep -q "ZONES=europe-west4-a PROV.* ZR=europe-west4"'
P5="$WORK/parts_a100_five.txt"
for i in 01 02 03 04 05 06; do echo "a100 harness/plans/t/t_a100_$i.yaml SPOT=0"; done > "$P5"
fresh; many a100_five "$P5" YES=1 MAX_A100=5 A100_MACHINE_TYPE=a2-ultragpu-1g STUB_SLEEP=12
check "a100 x6, MAX_A100=5: exit 0, every part once, 5 at a time" \
  '[ "$(rc a100_five)" = 0 ] && [ "$(grep -c "^start [0-9.]* a100 " "$WORK/state_many_a100_five/events")" = 6 ] && [ "$(maxconc a100_five a100)" = 5 ]'
check "a100 x6, MAX_A100=5: regional caps reached and kept (us-central1 2, us-east5 1, europe-west4 1, us-east4 1)" \
  '[ "$(regconc a100_five us-central1)" = 2 ] && [ "$(regconc a100_five us-east5)" = 1 ] && [ "$(regconc a100_five europe-west4)" = 1 ] && [ "$(regconc a100_five us-east4)" = 1 ]'
fresh; many a100_six "$P5" YES=1 MAX_A100=6 A100_MACHINE_TYPE=a2-ultragpu-1g
check "a100: MAX_A100=6 is refused (the regional caps add up to 5)" \
  '[ "$(rc a100_six)" = 2 ] && [ "$(starts a100_six)" = 0 ] && grep -q "above the regional A100 caps" "$WORK/state_many_a100_six/out"'
fresh; many a100_nomt "$P" YES=1 MAX_A100=3
check "a100: refused without A100_MACHINE_TYPE, nothing started" \
  '[ "$(rc a100_nomt)" = 2 ] && [ "$(starts a100_nomt)" = 0 ] && grep -q "A100 parts need A100_MACHINE_TYPE" "$WORK/state_many_a100_nomt/out"'
fresh; many a100_globalmt "$P" YES=1 MAX_A100=3 A100_MACHINE_TYPE=a2-ultragpu-1g MACHINE_TYPE=a2-highgpu-1g
check "a100: a MACHINE_TYPE in the shell is refused (it would reach every launcher)" \
  '[ "$(rc a100_globalmt)" = 2 ] && [ "$(starts a100_globalmt)" = 0 ] && grep -q "MACHINE_TYPE is set in this shell" "$WORK/state_many_a100_globalmt/out"'
P2="$WORK/parts_a100_mixed.txt"
{ echo "a100 harness/plans/t/t_a100_01.yaml MACHINE_TYPE=a2-highgpu-1g"; echo "a100 harness/plans/t/t_a100_02.yaml"; } > "$P2"
fresh; many a100_mixed "$P2" YES=1 A100_MACHINE_TYPE=a2-ultragpu-1g
check "a100: a per-part MACHINE_TYPE is refused (one wrapper run never mixes machine types)" \
  '[ "$(rc a100_mixed)" = 2 ] && [ "$(starts a100_mixed)" = 0 ] && grep -q "not a per-part setting" "$WORK/state_many_a100_mixed/out"'
P3="$WORK/parts_a100_span.txt"
echo "a100 harness/plans/t/t_a100_01.yaml ZONES=us-central1-a,europe-west4-a" > "$P3"
fresh; many a100_span "$P3" YES=1 A100_MACHINE_TYPE=a2-ultragpu-1g
check "a100: a part whose zones span two regions is refused" \
  '[ "$(rc a100_span)" = 2 ] && grep -q "span regions" "$WORK/state_many_a100_span/out"'

echo "======== part 2: the real cloud/af3_run.sh against the fake gcloud"
mkplan e2e_l4_a l4 l4_xla 7U3J 1 2 1 30
mkplan e2e_l4_b l4 l4_xla 7U3J 1 1 none 30
mkplan e2e_v5 v5e tpu_xla 7U3J 1 1 none 30
mkplan e2e_l4_nocap l4 l4_xla 7U3J 1 1 none 30
mkplan e2e_v6 v6e tpu_xla 7U3J 1 1 none 30
fresh; rm -rf "$WORK/many_state"; mkdir -p "$WORK/many_state"
printf 'l4 harness/plans/t/e2e_l4_a.yaml\nl4 harness/plans/t/e2e_l4_b.yaml\nv5e harness/plans/t/e2e_v5.yaml\nv6e harness/plans/t/e2e_v6.yaml\nl4 harness/plans/t/e2e_l4_nocap.yaml ZONES=europe-west4-a,us-east4-a\n' > "$WORK/parts_e2e.txt"
S="$WORK/state_many_e2e"; rm -rf "$S"; mkdir -p "$S"
(cd "$R" && fake_env "$S" PARTS="$WORK/parts_e2e.txt" YES=1 ALLOW_OTHER_LAUNCHERS=1 MAX_L4=2 MAX_ATTEMPTS=2 RETRY_WAIT_S=1 \
  STAGGER_S=2 POLL_S=1 LAUNCHER="$MANY/e2e_launcher.sh" E2E_ZONES="e2e_l4_nocap=europe-west4-a:sync+us-east4-a:sync" \
  WEIGHTS_GCS_URI=gs://private-weights/af3.bin.zst RESULTS_GCS_URI=gs://pilot-results/study FETCH=full SPOT=0 \
  /bin/bash cloud/af3_run_many.sh > "$S/out" 2>&1); echo $? > "$S/rc"
cat "$S/out" | sed -n '/^af3_run_many/,$p' | sed 's/^/   | /'
check "e2e: four parts ran and were fetched (exit 0 each)" '[ "$(oks e2e)" = 4 ] && [ "$(ls -d "$R"/results/af3/2*_l4_e2e_l4_a "$R"/results/af3/2*_l4_e2e_l4_b "$R"/results/af3/2*_v5e_e2e_v5 "$R"/results/af3/2*_v6e_e2e_v6 2> /dev/null | wc -l | tr -d " ")" = 4 ]'
check "e2e: the v6e part was created on demand (--provisioning-model=STANDARD), recorded as standard" \
  'V=$(ls -d "$WORK"/many_state/e2e_v6_* | head -1); grep -q -- "--provisioning-model=STANDARD" "$V/create_args" && ! grep -q -- "FLEX_START" "$V/create_args" && grep -q "^PROVISIONING=standard$" "$R"/results/af3/.launcher/2*_v6e_e2e_v6/launch_record.txt'
check "e2e: warm rerun ran in its part (plan runs fetched)" 'grep -q "seed1_warm" "$R"/results/af3/2*_l4_e2e_l4_a/plan_runs.jsonl'
check "e2e: stockout part: both comma-listed zones reached the launcher and were tried" 'L="$(ls -d "$R"/results/af3/.many/2*/ | tail -1)e2e_l4_nocap.log"; grep "No zone had capacity" "$L" | grep -q "europe-west4-a" && grep "No zone had capacity" "$L" | grep -q "us-east4-a"'
check "e2e: stockout part tried twice, never got a VM, reported" 'grep -q "e2e_l4_nocap.yaml *FAILED, no VM created.*after 2 attempt" "$S/out" && [ "$(grep -c e2e_l4_nocap "$WORK/many_state/dirs")" = 2 ]'
check "e2e: no fake VM left in any launch (all deleted or never created)" '! ls "$WORK"/many_state/*/created > /dev/null 2>&1'
check "e2e: wrapper exit 1 (one part failed)" '[ "$(rc e2e)" = 1 ]'
check "e2e: weights URI never printed in any log" '! grep -rq private-weights "$(ls -d "$R"/results/af3/.many/2*/ | tail -1)"'

echo "======== part 2b: the three A100 smoke tests (results/parts_a100_smoke.txt), real launcher, MAX_A100=5"
if [ -f "$REPO/results/parts_a100_smoke.txt" ]; then
  fresh; rm -rf "$WORK/many_state"; mkdir -p "$WORK/many_state"
  S="$WORK/state_many_smoke"; rm -rf "$S"; mkdir -p "$S"
  (cd "$R" && fake_env "$S" PARTS="$REPO/results/parts_a100_smoke.txt" YES=1 ALLOW_OTHER_LAUNCHERS=1 MAX_A100=5 \
    A100_MACHINE_TYPE=a2-ultragpu-1g STAGGER_S=2 POLL_S=1 FAKE_RUN_S=15 LAUNCHER="$MANY/e2e_launcher.sh" \
    WEIGHTS_GCS_URI=gs://private-weights/af3.bin.zst RESULTS_GCS_URI=gs://followup-results/study \
    /bin/bash cloud/af3_run_many.sh > "$S/out" 2>&1); echo $? > "$S/rc"
  sed -n '/^af3_run_many/,$p' "$S/out" | sed 's/^/   | /'
  check "smoke: wrapper exit 0, three sessions fetched" \
    '[ "$(rc smoke)" = 0 ] && [ "$(ls -d "$R"/results/af3/2*_a100_a100_smoke* 2> /dev/null | wc -l | tr -d " ")" = 3 ]'
  check "smoke: one session per configuration (a100_xla, a100_xla_highest, a100_triton), label smoke" \
    '[ "$(for f in "$R"/results/af3/2*_a100_a100_smoke*/*/session.json; do python3 -c "import json,sys; d=json.load(open(sys.argv[1])); print(d[\"label\"], d[\"config_name\"])" "$f"; done | sort | tr "\n" " ")" = "smoke a100_triton smoke a100_xla smoke a100_xla_highest " ]'
  check "smoke: every VM a2-ultragpu-1g on demand with --maintenance-policy=TERMINATE" \
    '[ "$(cat "$WORK"/many_state/a100_smoke*/create_args | tr " " "\n" | grep -c -e "^--machine-type=a2-ultragpu-1g$")" = 3 ] && [ "$(cat "$WORK"/many_state/a100_smoke*/create_args | tr " " "\n" | grep -c -e "^--provisioning-model=STANDARD$")" = 3 ] && [ "$(cat "$WORK"/many_state/a100_smoke*/create_args | tr " " "\n" | grep -c -e "^--maintenance-policy=TERMINATE$")" = 3 ]'
  check "smoke: three at once within the regional caps (2 in us-central1, then 1 in us-east5, the next cheapest)" \
    '[ "$(cat "$WORK"/many_state/a100_smoke*/created_zone | grep -c "^us-central1-")" = 2 ] && [ "$(cat "$WORK"/many_state/a100_smoke*/created_zone | grep -c "^us-east5-")" = 1 ]'
  check "smoke: no fake VM left" '! ls "$WORK"/many_state/*/created > /dev/null 2>&1'
else
  echo "  (no results/parts_a100_smoke.txt: part 2b skipped)"
fi

echo "======== part 3: the generated main-study parts (stub launcher)"
if [ -d "$REPO/harness/plans/main" ]; then
  fresh; cp -R "$REPO/harness/plans/main" "$R/harness/plans/"
  many main "$R/harness/plans/main/parts_all.txt" YES=1 STAGGER_S=0 STUB_SLEEP=12 MAX_L4=8 MAX_V5E=1 MAX_V6E=1 MAX_CPU=4
  check "main: every part validated and launched once ($(grep -vc '^#' "$R/harness/plans/main/parts_all.txt"))" \
    '[ "$(rc main)" = 0 ] && [ "$(starts main)" = "$(grep -vc "^#" "$R/harness/plans/main/parts_all.txt")" ]'
  check "main: caps held (L4 <= 8, v5e <= 1, v6e <= 1, cpu <= 4)" '[ "$(maxconc main l4)" -le 8 ] && [ "$(maxconc main v5e)" = 1 ] && [ "$(maxconc main v6e)" = 1 ] && [ "$(maxconc main cpu)" -le 4 ]'
  check "main: 8 L4 reached" '[ "$(maxconc main l4)" = 8 ]'
else
  echo "  (no harness/plans/main in the copy: part 3 skipped)"
fi
rm -rf "$R/results"
echo "======== $PASS passed, $FAIL failed"
[ "$FAIL" = 0 ]
