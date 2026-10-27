#!/usr/bin/env bash
# Runs a plan (harness/plans/*.yaml) on one cloud platform, with the same
# frozen inputs everywhere. Creates one single-device VM, uploads only what
# the plan needs, runs cloud/vm_af3_job.sh there detached (setup, then the
# plan), follows it with short SSH polls, fetches results/af3/<session>/ and
# deletes the VM, also on error, deadline or Ctrl-C.
#
# Run from the repo root, in a shell where cloud/env.sh is NOT sourced:
#   PLATFORM=v5e bash cloud/af3_run.sh                  # the probe (default plan)
#   PLATFORM=v6e PLAN=harness/plans/pilot.yaml WEIGHTS_GCS_URI=gs://BUCKET/af3.bin.zst \
#     bash cloud/af3_run.sh
#
# PLATFORM  API              default                       provisioning
# cpu       Compute Engine   n2-highmem-16 (Ice Lake)      Spot (SPOT=0: on-demand)
# l4        Compute Engine   g2-standard-8 (1 x L4)        Spot (SPOT=0: on-demand,
#                                                          PROVISIONING=flex_start: Flex-start)
# a100      Compute Engine   none: MACHINE_TYPE must be    as l4 (follow-up 3 runs on demand,
#                            a2-ultragpu-1g (1 x A100      SPOT=0; Spot A100 80 GB quota is 0)
#                            80 GB)
# h100      Compute Engine   a3-highgpu-1g (1 x H100       as a100 (follow-up 9; Spot or on demand; the
#                            80 GB), the only type taken   on-demand H100 quota of the project is 0)
# v5e       legacy TPU API   v5litepod-1                   Spot (SPOT=0: on-demand)
# v6e       Compute Engine   ct6e-standard-1t              Flex-start (PROVISIONING=standard:
#                                                          on-demand)
#
# Zones: ZONES="z1 z2 ..." (Compute Engine platforms: cpu, l4, a100, v6e) tries each
# zone in order and keeps the first VM that is created. A stockout or capacity
# error (seen when the create call fails at once, or later in the insert
# operation, also when the VM goes STAGING, STOPPING, then disappears) moves on
# to the next zone; any other error stops. If Google's error names zones with
# capacity (errorDetails[].errorInfo.metadatas.zonesAvailable), those are tried
# first, once each. Every attempt is printed; a failed zone is checked to hold
# no VM before the next one. PLATFORM=l4 without ZONE or ZONES uses
# L4_DEFAULT_ZONES below (every Europe and US zone with nvidia-l4 per
# `gcloud compute accelerator-types list --filter="name=nvidia-l4"`, 2026-10-09).
# PLATFORM=a100 without ZONE or ZONES uses A100_DEFAULT_ZONES below (the zones of the four regions with
# NVIDIA_A100_80GB_GPUS quota that offer a2-ultragpu-1g, cheapest region first; cloud/af3_run_many.sh gives each
# A100 part the zones of one region, to keep within the regional quota). ZONE=z alone means that one zone.
#
# Overrides: PROJECT ZONE ZONES MACHINE_TYPE SPOT PROVISIONING (spot, standard, or
# flex_start for l4 and a100; flex_start or standard for v6e; wins over SPOT) IMAGE_PROJECT IMAGE_FAMILY RUNTIME
# (v5e) BOOT_DISK_SIZE MIN_CPU_PLATFORM (cpu) REQUEST_VALID_FOR (Flex-start
# queue, 90s to 2h, default 2h) DETACHED_DEADLINE_MIN MAX_RUN_DURATION WEIGHTS
# (random or gcs; default: the plan's) WEIGHTS_GCS_URI (gcs only; never
# printed) VM_NAME YES=1 (no prompt).
#
# Featurise-only plans (follow-up 5, notes/analysis_plan.md section 10, row of
# 2026-10-18; plan key featurise_only, harness/run_plan.py): PLATFORM=cpu,
# WEIGHTS=none (the only value they take: no weights and no weights URI reach
# the VM), and the plan's host (machine type, min CPU platform, image) is used;
# a MACHINE_TYPE, MIN_CPU_PLATFORM, IMAGE_PROJECT or IMAGE_FAMILY that
# disagrees with it is refused. A plan with ccd_sha256 needs CCD_FILE, a local
# components.cif with that SHA-256: it is uploaded (zstd) and build_data uses it
# instead of the copy the AlphaFold3 build downloads (cloud/vm_af3_setup.sh).
#
# Alternative software stack (follow-up 7, plan key stack, harness/run_plan.py):
# the plan's pinned jax, jaxlib and libtpu (TPU) or jax, jaxlib and the
# jax-cuda12 plugin and pjrt (GPU) are passed to the VM (STACK), which installs
# them after its normal setup (cloud/vm_af3_setup.sh); on TPU the stack's libtpu
# is the LIBTPU_VERSION. A STACK in this shell, or a plan with both stack and a
# libtpu list, is refused; without the key nothing changes.
#
# Eigen from our bucket (follow-ups 7 to 9, plan key eigen_source: archive, harness/run_plan.py): the VM
# takes the Eigen source of libcifpp's build from the pinned archive (cloud/eigen_pin.sh, cloud/vm_eigen_source.sh)
# instead of cloning gitlab.com (EIGEN_SOURCE=archive on the VM, recorded in the launch record). An EIGEN_SOURCE in
# this shell is refused; without the key nothing changes.
#
# Plan dates (follow-up 6 stop rules, plan keys launch_by and finish_by): both are
# printed; a new launch after launch_by (UTC date) is refused before anything is
# created; a part not finished by finish_by is reported as incomplete.
#
# Flex-start (v6e, and l4 or a100 with PROVISIONING=flex_start): the create request
# queues up to REQUEST_VALID_FOR and is not billed while queued; the VM gets
# --max-run-duration and --instance-termination-action=DELETE like every
# Compute Engine VM here. Flags as documented in `gcloud compute instances
# create --help` and https://cloud.google.com/compute/docs/instances/create-flex-start-vm
#
# Results bucket (optional): RESULTS_GCS_URI=gs://BUCKET[/PREFIX], a private
# bucket separate from the weights bucket. The VM uploads each target's
# folder as soon as it finishes, so a preempted or unreachable VM loses at
# most the target in progress; objects go to RESULTS_GCS_URI/af3/<session>/.
# If the final SSH fetch fails, the laptop fetches from the bucket instead.
# FETCH=light (default) brings records, logs, *_summary_confidences.json,
# *_ranking_scores.csv and the mmCIFs to the laptop and leaves *_data.json
# and the full *_confidences.json in the bucket; FETCH=full brings
# everything. Without RESULTS_GCS_URI, or if the VM's final upload failed,
# the fetch is full, since the VM and its copy are deleted afterwards.
#
# Preemption: when an SSH poll fails, the launcher asks the API for the VM's
# state; a preempted, stopped or deleted VM ends the run at once and goes
# straight to cleanup (bucket fetch if configured, delete, leftover check).
#
# TPU stack (v5e, v6e): jax/jaxlib 0.10.2 with libtpu LIBTPU_VERSION (default
# 0.0.43.2; LIBTPU_VERSION=0.0.42.1 for the version jax[tpu]==0.10.2 pins;
# cloud/vm_tpu_stack.sh explains the choice). A plan with a "libtpu" list sets
# it instead (its first version; the others follow on the same VM); a
# LIBTPU_VERSION that disagrees with that list is refused. Ignored on cpu, l4 and a100.
#
# Comparison: if the plan names a compare_reference and that folder exists on
# the laptop, the fetched session is compared with it at the end, bit for bit
# and by RMSD (harness/compare_runs.py; compare_runs.txt in the session).
#
# Price: the worst case uses cloud/prices.csv for the zone's region; if the
# region (or machine type) has no row there, it says which list price it shows
# instead. Without a published price for the provisioning model (G2
# Flex-start), it uses the on-demand price as the upper bound and says so.
#
# Connection loss (cloud/lib_detached.sh): a failed poll is classified with
# the API. No network on this laptop (or the API unreachable): the launcher
# waits with backoff until the deadline and touches nothing. The VM up but
# SSH failing: it goes on following until the deadline and reads the job's
# end from the results bucket if one is set. The VM preempted, stopped or
# gone: it stops at once. One line is printed per change, not per retry.
#
# Deleting: the EXIT trap deletes the VM (results first) when the job has
# finished (EXIT_CODE), at the follow deadline, on Ctrl-C, when the VM is
# already gone or the job no longer runs, or before the job was started. A
# VM the API reports running whose job has not finished is never deleted
# otherwise: the launcher keeps it, says so, and prints the resume command.
# If the API is unreachable when the VM may be deleted, it waits up to
# CLEANUP_WAIT_MIN (30) minutes for it. Independently, Google deletes every
# VM at the end of its lifetime: --max-run-duration with
# --instance-termination-action=DELETE on every Compute Engine VM (Spot and
# on-demand too), the systemd watchdog on the legacy TPU VM. The script never
# adopts or deletes a VM it did not create, and lists leftovers at the end.
#
# Session records: results/af3/.launcher/<session>/ holds launch_record.txt
# (VM, zone, plan, bucket, deadline; read by resume mode), uploaded_files.txt
# (git blob hash of every file uploaded to the VM, with the commit and the
# number of uncommitted files at launch) and launcher_events.txt (one line
# per event: VM created, connection lost and back with durations, job
# finished, deadline, every delete-or-keep decision). They are copied into
# the fetched session and uploaded to the results bucket at the end. On the
# VM, the job wrapper writes EXIT_CODE and final_status.json and copies both
# to RESULTS_GCS_URI/af3/<session>/ (cloud/vm_job_finish.sh).
#
# Resume mode: re-attach to the VM of an earlier launch of this script (for
# example after the laptop was closed or the launcher was killed), follow
# it, fetch and delete it as usual:
#   RESUME_VM=af3-run-v5e-20261009t154308z SESSION=20261009T154308Z_v5e_pilot bash cloud/af3_run.sh
# The settings come from the session's launch_record.txt; RESUME_VM must be
# the VM recorded there, and the VM's labels (purpose, platform, plan,
# session) and its session folder must match, or nothing is touched. The
# follow deadline is what was left of the original one (at least 1 minute);
# DETACHED_DEADLINE_MIN overrides it. A session launched before launch
# records existed is accepted only if RESUME_VM is the name that launcher
# derived from the session (af3-run-<platform>-<stamp>), the labels match,
# and ZONE is given; PLAN, RESULTS_GCS_URI and FETCH as at launch.
# macOS bash 3.2: no arrays.
set -euo pipefail

die() { echo "!! $*" >&2; exit 1; }

# cloud/env.sh exports ZONE, TPU_NAME and other settings for the multi-chip AF2
# sessions; inheriting them here would create the wrong VM.
[ -z "${AF2_COMMIT:-}" ] || die "cloud/env.sh is sourced in this shell. Open a fresh shell; this script has its own defaults."

# Resume mode (header): the settings come from the session's launch record.
RESUME=0 RESUME_LEGACY=0 RESUME_NOTE=""
if [ -n "${RESUME_VM:-}" ]; then
  RESUME=1
  [ -n "${SESSION:-}" ] || die "RESUME_VM needs SESSION=<session folder name, as under results/af3/>."
  echo "$SESSION" | grep -Eq '^[0-9]{8}T[0-9]{6}Z_(cpu|l4|a100|h100|v5e|v6e)_[a-z0-9_-]+$' || die "Not a session name: SESSION='$SESSION'."
  REC="$(git rev-parse --show-toplevel)/results/af3/.launcher/$SESSION/launch_record.txt"
  rec() { sed -n "s/^$1=//p" "$REC" | tail -1; }
  if [ -f "$REC" ]; then
    [ "$(rec SESSION)" = "$SESSION" ] || die "$REC is not the record of $SESSION."
    [ "$(rec VM_NAME)" = "$RESUME_VM" ] \
      || die "$RESUME_VM is not in the records of $SESSION (its VM is $(rec VM_NAME)); nothing was touched."
    [ "$(rec CREATED)" = "1" ] || die "The record of $SESSION shows no created VM; nothing to resume."
    PLATFORM=$(rec PLATFORM) PLAN=$(rec PLAN) ZONE=$(rec ZONE) PROJECT=$(rec PROJECT)
    MACHINE_TYPE=$(rec MACHINE_TYPE) PROVISIONING=$(rec PROVISIONING) WEIGHTS=$(rec WEIGHTS)
    RESULTS_GCS_URI=$(rec RESULTS_GCS_URI) FETCH="${FETCH:-$(rec FETCH)}" LIBTPU_VERSION=$(rec LIBTPU_VERSION)
    MAX_RUN_DURATION=$(rec MAX_RUN_DURATION)
    if [ -z "${DETACHED_DEADLINE_MIN:-}" ]; then
      UNTIL=$(rec FOLLOW_UNTIL)
      [ -n "$UNTIL" ] || UNTIL=$(( $(date +%s) + $(rec DEADLINE_MIN) * 60 ))
      DETACHED_DEADLINE_MIN=$(( (UNTIL - $(date +%s) + 59) / 60 ))
      if [ "$DETACHED_DEADLINE_MIN" -lt 1 ]; then
        DETACHED_DEADLINE_MIN=1
        RESUME_NOTE="the original follow deadline has passed: following for 1 min (DETACHED_DEADLINE_MIN sets more)"
      fi
    fi
  else
    # Launched before launch records existed: only the VM that launcher named.
    RESUME_LEGACY=1
    PLATFORM=$(echo "$SESSION" | cut -d_ -f2)
    LEGACY_PLAN_NAME=$(echo "$SESSION" | cut -d_ -f3-)
    [ "$RESUME_VM" = "af3-run-$PLATFORM-$(echo "${SESSION%%_*}" | tr 'A-Z' 'a-z')" ] \
      || die "$SESSION has no launch record, and $RESUME_VM is not the VM name its launcher derived; nothing was touched."
    [ -n "${ZONE:-}" ] || die "$SESSION has no launch record: give ZONE=<the zone the launcher reported> as well."
    PLAN="${PLAN:-harness/plans/$LEGACY_PLAN_NAME.yaml}"
  fi
  unset ZONES
  VM_NAME=$RESUME_VM
fi
: "${PLATFORM:?set PLATFORM to cpu, l4, a100, h100, v5e or v6e}"
[ -z "${STACK:-}" ] || die "STACK is set in this shell; an alternative stack comes only from a plan's stack key."
[ -z "${EIGEN_SOURCE:-}" ] || die "EIGEN_SOURCE is set in this shell; the Eigen archive comes only from a plan's eigen_source key."
command -v gcloud > /dev/null || die "gcloud not found."
command -v python3 > /dev/null || die "python3 not found."

PLAN="${PLAN:-harness/plans/probe.yaml}"
# Read before cloud/lib_detached.sh is sourced: it sets its own default of 60.
DEADLINE_OVERRIDE="${DETACHED_DEADLINE_MIN:-}"
PROJECT="${PROJECT:-af2-tpu-benchmark}"
SPOT="${SPOT:-1}"
BOOT_DISK_SIZE="${BOOT_DISK_SIZE:-100GB}"
BOOT_DISK_TYPE=""
RUNTIME=""
MIN_CPU_PLATFORM_DEFAULT=""
# A featurise-only plan fixes its host (header); the launch settings must agree with it.
PLAN_HOST=$(python3 "$(git rev-parse --show-toplevel)/harness/run_plan.py" --plan "$PLAN" --platform "$PLATFORM" --describe 2> /dev/null \
  | python3 -c "import json,sys; h=json.load(sys.stdin).get('host') or {}; print('\n'.join(k + '=' + v for k, v in h.items()))" 2> /dev/null || true)
if [ -n "$PLAN_HOST" ]; then
  [ "$PLATFORM" = "cpu" ] || die "The plan $PLAN fixes a CPU-only host: PLATFORM=cpu, not $PLATFORM."
  for kv in MACHINE_TYPE:machine_type MIN_CPU_PLATFORM:min_cpu_platform IMAGE_PROJECT:image_project IMAGE_FAMILY:image_family; do
    var=${kv%%:*}
    want=$(echo "$PLAN_HOST" | sed -n "s/^${kv#*:}=//p")
    eval "have=\${$var:-}"
    [ -z "$have" ] || [ "$have" = "$want" ] || die "$var=$have disagrees with the plan's host ($want, $PLAN)."
    eval "$var=\$want"
  done
fi
case "$PLATFORM" in
  cpu)
    VM_API=gce
    MACHINE_TYPE="${MACHINE_TYPE:-n2-highmem-16}"
    ZONE="${ZONE:-europe-west4-a}"
    IMAGE_PROJECT="${IMAGE_PROJECT:-ubuntu-os-cloud}"
    IMAGE_FAMILY="${IMAGE_FAMILY:-ubuntu-2204-lts}"
    BOOT_DISK_TYPE=pd-balanced
    case "$MACHINE_TYPE" in n2-*) MIN_CPU_PLATFORM_DEFAULT="Intel Ice Lake" ;; esac
    case "$MACHINE_TYPE" in
      a2-* | a3-* | a4-* | g2-* | g4-* | ct* | *tpu*) die "PLATFORM=cpu takes a CPU-only machine type, not $MACHINE_TYPE." ;;
    esac ;;
  l4)
    VM_API=gce
    MACHINE_TYPE="${MACHINE_TYPE:-g2-standard-8}"
    # Europe first (europe-west4 is where the other platforms run), then the US
    # zones, us-east4 first (it had L4 capacity on 2026-10-09).
    L4_DEFAULT_ZONES="europe-west4-a europe-west4-b europe-west4-c europe-west1-b europe-west1-c \
europe-west3-a europe-west3-b europe-west2-a europe-west2-b europe-west6-b europe-west6-c \
us-east4-a us-east4-c us-central1-a us-central1-b us-central1-c us-east1-b us-east1-c us-east1-d \
us-west1-a us-west1-b us-west1-c us-west4-a us-west4-c"
    if [ -z "${ZONES:-}" ] && [ -z "${ZONE:-}" ]; then ZONES=$L4_DEFAULT_ZONES; fi
    ZONE="${ZONE:-europe-west4-a}"
    IMAGE_PROJECT="${IMAGE_PROJECT:-ubuntu-os-accelerator-images}"
    IMAGE_FAMILY="${IMAGE_FAMILY:-ubuntu-accelerator-2204-amd64-with-nvidia-580}"
    BOOT_DISK_TYPE=pd-balanced
    case "$MACHINE_TYPE" in
      g2-standard-4 | g2-standard-8 | g2-standard-12 | g2-standard-16 | g2-standard-32) ;;
      *) die "PLATFORM=l4 takes a single-L4 type (g2-standard-4/8/12/16/32), not $MACHINE_TYPE." ;;
    esac ;;
  a100)
    VM_API=gce
    # Follow-up 3 (notes/analysis_plan.md section 10, rows of 2026-10-16): one machine type for every A100 run,
    # a2-ultragpu-1g (one A100 80 GB); a2-highgpu-1g (A100 40 GB) is not used. No default: the launch names it.
    [ -n "${MACHINE_TYPE:-}" ] || die "PLATFORM=a100 needs MACHINE_TYPE=a2-ultragpu-1g (the follow-up 3 machine type; no default)."
    [ "$MACHINE_TYPE" = "a2-ultragpu-1g" ] \
      || die "PLATFORM=a100 takes a2-ultragpu-1g only (one A100 80 GB, section 10, 2026-10-16), not $MACHINE_TYPE."
    # The zones that offer a2-ultragpu-1g in the regions with NVIDIA_A100_80GB_GPUS quota (us-central1 2,
    # us-east5 1, europe-west4 1, us-east4 1; the last two regions approved later on 2026-10-16), per
    # `gcloud compute machine-types list --filter="name=a2-ultragpu-1g"` and `gcloud compute accelerator-types list
    # --filter="name=nvidia-a100-80gb"`, 2026-10-16; regions in order of on-demand price (cloud/prices.csv).
    A100_DEFAULT_ZONES="us-central1-a us-central1-c us-east5-a us-east5-b europe-west4-a us-east4-c"
    if [ -z "${ZONES:-}" ] && [ -z "${ZONE:-}" ]; then ZONES=$A100_DEFAULT_ZONES; fi
    ZONE="${ZONE:-us-central1-a}"
    for z in $(echo ${ZONES:-$ZONE}); do
      case " $A100_DEFAULT_ZONES " in *" $z "*) ;; *) die "PLATFORM=a100: $z is not one of the zones with a2-ultragpu-1g quota ($A100_DEFAULT_ZONES)." ;; esac
    done
    IMAGE_PROJECT="${IMAGE_PROJECT:-ubuntu-os-accelerator-images}"
    IMAGE_FAMILY="${IMAGE_FAMILY:-ubuntu-accelerator-2204-amd64-with-nvidia-580}"
    BOOT_DISK_TYPE=pd-balanced ;;
  h100)
    VM_API=gce
    # Follow-up 9 (label h100): one H100 80 GB, a3-highgpu-1g (26 vCPU, 234 GiB), the only type taken. Zones: those
    # with a3-highgpu-1g in the three US regions where the project ran A100s (`gcloud compute machine-types list
    # --filter="name=a3-highgpu-1g"`, 2026-10-27; europe-west4, dearer, is left out), regions in order of on-demand
    # price (cloud/prices.csv). Quotas read 2026-10-27 (Cloud Quotas, read-only): Spot H100 64 per region,
    # GPUS_ALL_REGIONS 8; on-demand H100 0.
    MACHINE_TYPE="${MACHINE_TYPE:-a3-highgpu-1g}"
    [ "$MACHINE_TYPE" = "a3-highgpu-1g" ] || die "PLATFORM=h100 takes a3-highgpu-1g only (one H100 80 GB), not $MACHINE_TYPE."
    H100_DEFAULT_ZONES="us-central1-a us-central1-b us-central1-c us-east4-a us-east4-b us-east4-c us-east5-a"
    if [ -z "${ZONES:-}" ] && [ -z "${ZONE:-}" ]; then ZONES=$H100_DEFAULT_ZONES; fi
    ZONE="${ZONE:-us-central1-a}"
    for z in $(echo ${ZONES:-$ZONE}); do
      case " $H100_DEFAULT_ZONES " in *" $z "*) ;; *) die "PLATFORM=h100: $z is not one of the zones with a3-highgpu-1g ($H100_DEFAULT_ZONES)." ;; esac
    done
    IMAGE_PROJECT="${IMAGE_PROJECT:-ubuntu-os-accelerator-images}"
    IMAGE_FAMILY="${IMAGE_FAMILY:-ubuntu-accelerator-2204-amd64-with-nvidia-580}"
    BOOT_DISK_TYPE=pd-balanced ;;
  v5e)
    VM_API=tpu
    MACHINE_TYPE="${MACHINE_TYPE:-v5litepod-1}"
    RUNTIME="${RUNTIME:-v2-alpha-tpuv5-lite}"
    ZONE="${ZONE:-europe-west4-b}"
    [ "$MACHINE_TYPE" = "v5litepod-1" ] || die "PLATFORM=v5e takes v5litepod-1 only (single chip), not $MACHINE_TYPE." ;;
  v6e)
    VM_API=gce
    MACHINE_TYPE="${MACHINE_TYPE:-ct6e-standard-1t}"
    ZONE="${ZONE:-europe-west4-a}"
    IMAGE_PROJECT="${IMAGE_PROJECT:-ubuntu-os-accelerator-images}"
    IMAGE_FAMILY="${IMAGE_FAMILY:-ubuntu-accel-2204-amd64-tpu-v5e-v5p-v6e}"
    [ "$MACHINE_TYPE" = "ct6e-standard-1t" ] || die "PLATFORM=v6e takes ct6e-standard-1t only (single chip), not $MACHINE_TYPE." ;;
  *) die "PLATFORM must be cpu, l4, a100, h100, v5e or v6e." ;;
esac
MIN_CPU_PLATFORM="${MIN_CPU_PLATFORM-$MIN_CPU_PLATFORM_DEFAULT}"
case "${PROVISIONING:-}" in
  "")
    case "$PLATFORM" in
      v6e) PROVISIONING=flex_start ;;
      *) if [ "$SPOT" = "1" ]; then PROVISIONING=spot; else PROVISIONING=standard; fi ;;
    esac ;;
  flex_start)
    case "$PLATFORM" in
      l4 | a100 | h100 | v6e) ;;
      *) die "PROVISIONING=flex_start is supported here for PLATFORM=l4, a100, h100 and v6e only." ;;
    esac ;;
  standard) ;;
  spot)
    # 2026-10-12: v6e also runs on demand (no Flex-start capacity since
    # 2026-10-10); Spot stays refused, since a preempted v6e run is lost.
    [ "$PLATFORM" != "v6e" ] || die "PLATFORM=v6e runs with Flex-start or on demand (PROVISIONING=flex_start or standard), not Spot." ;;
  *) die "PROVISIONING must be spot, standard or flex_start." ;;
esac
# The zones to try, in order (one zone unless ZONES is set).
ZONES=$(echo ${ZONES:-$ZONE})
for z in $ZONES; do
  echo "$z" | grep -Eq '^[a-z]+-[a-z]+[0-9]+-[a-z]$' || die "Not a zone name in ZONES: '$z'."
done
set -- $ZONES
ZONE_COUNT=$#
ZONE=$1
if [ "$VM_API" = "tpu" ] && [ "$ZONE_COUNT" -gt 1 ]; then
  die "ZONES with more than one zone is supported for the Compute Engine platforms (cpu, l4, a100, h100, v6e), not v5e."
fi
REGION="${ZONE%-*}"
REGIONS=$(for z in $ZONES; do echo "${z%-*}"; done | awk '!seen[$0]++' | tr '\n' ' ')

# "1d2h3m4s" -> seconds.
to_seconds() {
  echo "$1" | grep -Eq '^([0-9]+d)?([0-9]+h)?([0-9]+m)?([0-9]+s)?$' && [ -n "$1" ] \
    || die "Not a duration like 2h or 90m: '$1'"
  echo $(( $(echo "$1" | sed -E 's/([0-9]+)d/\1*86400+/; s/([0-9]+)h/\1*3600+/; s/([0-9]+)m/\1*60+/; s/([0-9]+)s/\1+/; s/\+$//') ))
}

cd "$(git rev-parse --show-toplevel)"
source cloud/lib_vm.sh
source cloud/lib_detached.sh

# The plan, expanded for this platform (targets, runs, deadline, weights).
DESC=$(python3 harness/run_plan.py --plan "$PLAN" --platform "$PLATFORM" --describe) \
  || die "Could not read the plan $PLAN."
jget() { echo "$DESC" | python3 -c "import json,sys; v=json.load(sys.stdin)$1; print(v)"; }
TARGETS=$(jget "['targets']" | python3 -c "import ast,sys; print(','.join(ast.literal_eval(sys.stdin.read())))")
[ -n "$TARGETS" ] || die "The plan has no targets for $PLATFORM."
PLAN_NAME=$(basename "$PLAN" .yaml | tr 'A-Z' 'a-z' | tr -c 'a-z0-9_\n-' '-')
PLAN_WEIGHTS=$(jget "['weights']")
# The frozen-input manifest of the plan (plan key manifest; inputs/manifest.csv by default) and its targets' inputs.
MANIFEST=$(jget ".get('manifest', 'inputs/manifest.csv')")
INPUT_FILES=$(python3 -c "import csv,sys; m={r['pdb_id']: r['input_path'] for r in csv.DictReader(open(sys.argv[1]))}; print(' '.join(m[t] for t in sys.argv[2].split(',') if t in m))" "$MANIFEST" "$TARGETS")
PLAN_DEADLINE=$(jget "['deadline_min']")
WEIGHTS="${WEIGHTS:-$PLAN_WEIGHTS}"
PLAN_FEATURISE=$(jget ".get('featurise_only', False)")
[ "$PLAN_FEATURISE" != "True" ] || [ "$WEIGHTS" = "none" ] \
  || die "The plan $PLAN is featurise-only: WEIGHTS=none (no weights reach the VM), not $WEIGHTS."
case "$WEIGHTS" in
  none)
    [ "$PLAN_FEATURISE" = "True" ] || die "WEIGHTS=none is for featurise-only plans; $PLAN runs the model."
    [ -z "${WEIGHTS_GCS_URI:-}" ] || echo "   note: WEIGHTS=none, so WEIGHTS_GCS_URI is ignored (never uploaded)" ;;
  random) [ -z "${WEIGHTS_GCS_URI:-}" ] || echo "   note: WEIGHTS=random, so WEIGHTS_GCS_URI is ignored" ;;
  gcs)
    # Resume mode starts nothing, so it needs no weights URI.
    [ "$RESUME" = "1" ] || case "${WEIGHTS_GCS_URI:-}" in gs://?*/?*) ;; *) die "WEIGHTS=gcs needs WEIGHTS_GCS_URI=gs://BUCKET/OBJECT (a private bucket in $PROJECT)." ;; esac ;;
  *) die "WEIGHTS must be random or gcs (none for featurise-only plans)." ;;
esac
[ "$WEIGHTS" = "$PLAN_WEIGHTS" ] || echo "   note: the plan asks for weights=$PLAN_WEIGHTS; running with WEIGHTS=$WEIGHTS"
PLAN_LAUNCH_BY=$(jget ".get('launch_by') or ''")
PLAN_FINISH_BY=$(jget ".get('finish_by') or ''")
if [ -n "$PLAN_LAUNCH_BY" ] && [ "$RESUME" != "1" ] && [ "$(date -u +%Y-%m-%d)" \> "$PLAN_LAUNCH_BY" ]; then
  die "The plan $PLAN may be launched only by $PLAN_LAUNCH_BY (today, UTC: $(date -u +%Y-%m-%d)); nothing was created."
fi
# The CCD a plan fixes (ccd_sha256): a local components.cif with that SHA-256, uploaded to the VM.
PLAN_CCD=$(jget ".get('ccd_sha256') or ''")
if [ -n "$PLAN_CCD" ] && [ "$RESUME" != "1" ]; then
  [ -n "${CCD_FILE:-}" ] && [ -f "$CCD_FILE" ] \
    || die "The plan fixes the CCD (components.cif, SHA-256 $(echo "$PLAN_CCD" | cut -c1-12)...): set CCD_FILE to a local copy."
  [ "$( (sha256sum "$CCD_FILE" 2> /dev/null || shasum -a 256 "$CCD_FILE") | cut -d' ' -f1)" = "$PLAN_CCD" ] \
    || die "CCD_FILE ($CCD_FILE) is not the plan's components.cif (SHA-256 differs from $(echo "$PLAN_CCD" | cut -c1-12)...)."
  command -v zstd > /dev/null || die "zstd not found (needed to upload CCD_FILE)."
fi

# TPU stack: libtpu version(s) and the comparison reference from the plan.
PLAN_LIBTPU=$(jget "['libtpu']" | python3 -c "import ast,sys; print(' '.join(ast.literal_eval(sys.stdin.read())))")
COMPARE_REF=$(jget "['compare_reference']")
[ "$COMPARE_REF" != "None" ] || COMPARE_REF=""
# The plan's alternative stack (header): package==version pairs, comma-separated for the VM.
STACK_SPEC=$(jget ".get('stack') or {}" | python3 -c "import ast,sys; d=ast.literal_eval(sys.stdin.read()); print(','.join(k + '==' + v for k, v in d.items()))")
STACK_LIBTPU=$(echo "$STACK_SPEC" | tr ',' '\n' | sed -n 's/^libtpu==//p')
STACK_JAX=$(echo "$STACK_SPEC" | tr ',' '\n' | sed -n 's/^jax==//p')
EIGEN_SOURCE=$(jget ".get('eigen_source') or ''")   # plan key eigen_source (cloud/vm_eigen_source.sh); empty: unchanged
[ -z "$STACK_SPEC" ] || [ -z "$PLAN_LIBTPU" ] || die "The plan has both a stack and a libtpu list; a stack plan pins its own libtpu."
case "$PLATFORM" in
  v5e | v6e)
    if [ -n "$STACK_LIBTPU" ]; then
      [ -z "${LIBTPU_VERSION:-}" ] || [ "$LIBTPU_VERSION" = "$STACK_LIBTPU" ] \
        || die "LIBTPU_VERSION=$LIBTPU_VERSION disagrees with the plan's stack (libtpu $STACK_LIBTPU); unset it."
      LIBTPU_VERSION=$STACK_LIBTPU
    elif [ -n "$PLAN_LIBTPU" ]; then
      set -- $PLAN_LIBTPU
      [ -z "${LIBTPU_VERSION:-}" ] || [ "$LIBTPU_VERSION" = "$1" ] \
        || die "LIBTPU_VERSION=$LIBTPU_VERSION disagrees with the plan's libtpu list ($PLAN_LIBTPU); unset it."
      LIBTPU_VERSION=$1
    else
      LIBTPU_VERSION="${LIBTPU_VERSION:-0.0.43.2}"
    fi
    echo "$LIBTPU_VERSION" | grep -Eq '^[0-9]+(\.[0-9]+)+$' || die "LIBTPU_VERSION must be a version like 0.0.43.2."
    ;;
  *)
    [ -z "${LIBTPU_VERSION:-}" ] || echo "   note: LIBTPU_VERSION is for TPU platforms; ignored on $PLATFORM"
    LIBTPU_VERSION="" ;;
esac

# Results bucket and fetch mode.
RESULTS_GCS_URI="${RESULTS_GCS_URI:-}"
FETCH="${FETCH:-light}"
case "$FETCH" in light | full) ;; *) die "FETCH must be light or full." ;; esac
if [ -n "$RESULTS_GCS_URI" ]; then
  RESULTS_GCS_URI="${RESULTS_GCS_URI%/}"
  echo "$RESULTS_GCS_URI" | grep -Eq '^gs://[a-z0-9][a-z0-9._-]*[a-z0-9](/[A-Za-z0-9._-]+)*$' \
    || die "RESULTS_GCS_URI must be gs://BUCKET or gs://BUCKET/PREFIX (letters, digits, . _ - /)."
  RESULTS_BUCKET=$(echo "$RESULTS_GCS_URI" | cut -d/ -f3)
  if [ "$WEIGHTS" = "gcs" ] && [ "$RESUME" != "1" ] && [ "$RESULTS_BUCKET" = "$(echo "$WEIGHTS_GCS_URI" | cut -d/ -f3)" ]; then
    die "RESULTS_GCS_URI must be a different bucket from the weights bucket."
  fi
  FETCH_EFFECTIVE=$FETCH
else
  FETCH_EFFECTIVE=full
fi

DETACHED_DEADLINE_MIN="${DEADLINE_OVERRIDE:-$PLAN_DEADLINE}"
[ "$DETACHED_DEADLINE_MIN" -gt 0 ] 2> /dev/null \
  || die "The plan's deadline for $PLATFORM is not set (0). Size it from the probe, or set DETACHED_DEADLINE_MIN."
MAX_RUN_DURATION="${MAX_RUN_DURATION:-$(( DETACHED_DEADLINE_MIN + 30 ))m}"
MAX_RUN_S=$(to_seconds "$MAX_RUN_DURATION")
[ "$MAX_RUN_S" -ge $(( DETACHED_DEADLINE_MIN * 60 + 600 )) ] \
  || die "MAX_RUN_DURATION ($MAX_RUN_DURATION) must exceed the deadline ($DETACHED_DEADLINE_MIN min) by 10 min or more."
WATCHDOG_HOURS=$(( (MAX_RUN_S + 3599) / 3600 ))
if [ "$PROVISIONING" = "flex_start" ]; then
  # Google accepts 90 s to 2 h for a zonal Flex-start request.
  REQUEST_VALID_FOR="${REQUEST_VALID_FOR:-2h}"
  REQUEST_VALID_S=$(to_seconds "$REQUEST_VALID_FOR")
  [ "$REQUEST_VALID_S" -ge 90 ] && [ "$REQUEST_VALID_S" -le 7200 ] \
    || die "REQUEST_VALID_FOR ($REQUEST_VALID_FOR) must be between 90s and 2h for a Flex-start request in one zone."
  WAIT_S=$(( REQUEST_VALID_S + 600 ))
else
  WAIT_S=900
fi

if [ "$RESUME" != "1" ]; then
  # Frozen inputs must match the manifest before anything is created.
  echo ">> Checking the plan's frozen inputs against $MANIFEST"
  python3 harness/verify_inputs.py --targets "$TARGETS" --manifest "$MANIFEST" \
    || die "Frozen inputs do not match $MANIFEST; nothing was created."
  STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
  SESSION="${STAMP}_${PLATFORM}_${PLAN_NAME}"
  [ ! -e "results/af3/$SESSION" ] || die "results/af3/$SESSION exists; sessions are never overwritten."
  [ ! -e "results/af3/.launcher/$SESSION" ] || die "results/af3/.launcher/$SESSION exists; sessions are never overwritten."
  VM_NAME="${VM_NAME:-af3-run-$PLATFORM-$(echo "$STAMP" | tr 'A-Z' 'a-z')}"
fi
LAUNCH_DIR="results/af3/.launcher/$SESSION"
DETACHED_EVENTS="$LAUNCH_DIR/launcher_events.txt"
OUT_REL="alphafold-on-tpu/results/af3/$SESSION"
BENCH_COMMIT=$(git rev-parse HEAD)
EXPIRES=$(( $(date +%s) + MAX_RUN_S + WAIT_S ))
RESULTS_SESSION_URI="${RESULTS_GCS_URI:+$RESULTS_GCS_URI/af3/$SESSION}"
vm_init

# What will run and what it can cost, before anything is created. The price
# is the row for this platform, provisioning, machine type and the zone's
# region; any fallback is printed (PRICE_NOTES).
PRICE_OUT=$(python3 - "$PLATFORM" "$PROVISIONING" "$MACHINE_TYPE" "$REGIONS" <<'EOF'
import csv, sys
platform, prov, machine, regions = sys.argv[1:5]
regions = regions.split()
rows = [r for r in csv.DictReader(open('cloud/prices.csv')) if r['platform'] == platform]

def pick(provisioning):
  cands = [r for r in rows if r['provisioning'] == provisioning]
  for ok in (lambda r: r['machine_type'] == machine and r['region'] == region,
             lambda r: r['region'] == region,
             lambda r: r['machine_type'] == machine and r['region'] == 'europe-west4',
             lambda r: r['region'] == 'europe-west4',
             lambda r: True):
    for r in cands:
      if ok(r):
        return r
  return None

notes, best, fallback = [], None, []
for region in regions:
  row = pick(prov)
  if row is None:
    sys.exit(0)
  if not row['usd_per_hour'].strip():
    std = pick('standard')
    if std is None or not std['usd_per_hour'].strip():
      sys.exit(0)
    notes.append(f"no published {prov} price for {row['machine_type']} in {row['region']} "
                 f"({row['source_url']}, read {row['read_on']}); the worst case uses the "
                 f"on-demand price as an upper bound")
    row = std
  if row['region'] != region:
    fallback.append((region, row['region']))
  if row['machine_type'] != machine:
    notes.append(f"no price for {machine} in cloud/prices.csv: the price shown is for {row['machine_type']}")
  if best is None or float(row['usd_per_hour']) > float(best['usd_per_hour']):
    best = row
for used in dict.fromkeys(u for _, u in fallback):
  missing = [r for r, u in fallback if u == used]
  notes.append(f"no {platform} {prov} price in cloud/prices.csv for {', '.join(missing)}: "
               f"the {used} list price is used for {'it' if len(missing) == 1 else 'them'}")
if len(regions) > 1:
  notes.append(f"the worst case uses the highest price among the zones' regions: {best['region']}")
print(best['usd_per_hour'], best['read_on'], best['source_url'])
for n in dict.fromkeys(notes):
  print(n)
EOF
)
PRICE_LINE=$(echo "$PRICE_OUT" | head -1)
PRICE_NOTES=$(echo "$PRICE_OUT" | tail -n +2)
[ -n "$PRICE_LINE" ] || die "No $PLATFORM $PROVISIONING price in cloud/prices.csv."
set -- $PRICE_LINE
PRICE=$1 PRICE_DATE=$2 PRICE_URL=$3
if [ "$VM_API" = "tpu" ]; then LIFETIME_H=$WATCHDOG_HOURS; else LIFETIME_H=$(python3 -c "print(round($MAX_RUN_S/3600, 2))"); fi
WORST=$(python3 -c "print(f'{$PRICE * $LIFETIME_H:.2f}')")

echo
echo "$DESC" | python3 -c "
import json, sys
d = json.load(sys.stdin)
print(f\">> Plan {d['plan']} ({d['label']}): {len(d['runs'])} harness runs, {d['processes']} AlphaFold3 processes\")
print('   targets:', ', '.join(f'{t} ({d[\"tokens\"][t]} tokens)' for t in d['targets']))
print(f\"   seeds {d['seeds']}, fresh-cache reps {d['fresh_reps']}, warm rerun of seed: {d['warm_rerun_seed'] or 'none'}\")
print(f\"   recycles {d['num_recycles'] or 'AF3 default'}, diffusion samples {d['num_diffusion_samples'] or 'AF3 default'}\")
print('   runs:', ', '.join(r['run'] for r in d['runs']))"
if [ "$ZONE_COUNT" -gt 1 ]; then
  echo ">> Platform $PLATFORM: $MACHINE_TYPE, provisioning $PROVISIONING, $ZONE_COUNT zones tried in order:"
  echo "$ZONES" | fold -s -w 96 | sed 's/^/     /'
else
  echo ">> Platform $PLATFORM: $MACHINE_TYPE${RUNTIME:+ (runtime $RUNTIME)}, zone $ZONE, provisioning $PROVISIONING"
fi
[ "$VM_API" = "tpu" ] || echo "   image $IMAGE_PROJECT/$IMAGE_FAMILY${MIN_CPU_PLATFORM:+, min CPU platform $MIN_CPU_PLATFORM}"
case "$WEIGHTS" in
  gcs) echo "   weights: gcs (copied on the VM from your bucket; URI not printed)" ;;
  none) echo "   weights: none (featurisation only; no weights reach the VM)" ;;
  *) echo "   weights: random (generated on the VM)" ;;
esac
[ -z "$PLAN_CCD" ] || echo "   CCD: components.cif $(echo "$PLAN_CCD" | cut -c1-12)... from CCD_FILE, uploaded and given to build_data"
[ -z "$STACK_SPEC" ] || [ -n "$LIBTPU_VERSION" ] || echo "   GPU stack: the plan's alternative stack, $(echo "$STACK_SPEC" | tr ',' ' ')"
if [ "$EIGEN_SOURCE" = "archive" ]; then
  ( source cloud/eigen_pin.sh; echo "   Eigen (libcifpp build): the plan's archive $EIGEN_ARCHIVE_URI, sha256 $(echo "$EIGEN_ARCHIVE_SHA256" | cut -c1-12)..., commit $(echo "$EIGEN_COMMIT" | cut -c1-12) (tag $EIGEN_TAG), not gitlab.com" )
fi
[ -z "$PLAN_LAUNCH_BY$PLAN_FINISH_BY" ] || echo "   dates: launch by ${PLAN_LAUNCH_BY:-any date}; a part not finished by ${PLAN_FINISH_BY:-any date} is reported as incomplete"
if [ -n "$LIBTPU_VERSION" ]; then
  STACK_NOTE=""
  [ "$LIBTPU_VERSION" = "0.0.42.1" ] || STACK_NOTE=" (outside the 0.0.42.* that jax[tpu]==0.10.2 pins)"
  if [ -n "$STACK_SPEC" ]; then echo "   TPU stack: the plan's alternative stack, jax/jaxlib $STACK_JAX with libtpu $LIBTPU_VERSION"
  else echo "   TPU stack: jax/jaxlib 0.10.2 with libtpu $LIBTPU_VERSION$STACK_NOTE"; fi
  N_LIBTPU=$(echo $PLAN_LIBTPU | wc -w | tr -d ' ')
  [ "$N_LIBTPU" -le 1 ] || echo "     then on the same VM, in order: $(echo $PLAN_LIBTPU | cut -d' ' -f2-)"
fi
if [ -n "$COMPARE_REF" ]; then
  if [ -d "$COMPARE_REF" ]; then echo "   compared at the end with: $COMPARE_REF"
  else echo "   compare_reference $COMPARE_REF is not on this machine: no comparison at the end"; fi
fi
echo "   follow deadline: $DETACHED_DEADLINE_MIN min"
if [ "$VM_API" = "tpu" ]; then
  echo "   VM lifetime limit: watchdog deletes it after $WATCHDOG_HOURS h"
else
  echo "   VM lifetime limit: --max-run-duration=$MAX_RUN_DURATION, then Google deletes it"
  if [ "$PROVISIONING" = "flex_start" ]; then
    echo "   Flex-start queue: up to $REQUEST_VALID_FOR (not billed while queued)"
    [ "$ZONE_COUNT" -le 1 ] || echo "     per zone: up to $ZONE_COUNT x $REQUEST_VALID_FOR of waiting if every zone queues and fails"
  fi
fi
echo "   worst-case cost: \$$WORST = \$$PRICE/h x $LIFETIME_H h, list price read $PRICE_DATE"
echo "     ($PRICE_URL; excludes boot disk and network, a few cents)"
if [ -n "$PRICE_NOTES" ]; then echo "$PRICE_NOTES" | sed 's/^/   !! price note: /'; fi
if [ -n "$RESULTS_GCS_URI" ]; then
  echo "   results bucket: $RESULTS_SESSION_URI/ (each target uploaded when it finishes); fetch: $FETCH_EFFECTIVE"
else
  echo "   results bucket: none (RESULTS_GCS_URI not set); fetch: full over SSH (FETCH=light needs the bucket)"
fi
echo "   session: results/af3/$SESSION/   VM: $VM_NAME"
if [ "$RESUME" = "1" ]; then
  echo "   RESUME: re-attach to $VM_NAME in $ZONE (no VM is created, nothing is uploaded or started)$([ "$RESUME_LEGACY" = 1 ] && echo "; session without a launch record")"
  [ -z "$RESUME_NOTE" ] || echo "   !! $RESUME_NOTE"
fi
if [ "${YES:-0}" != "1" ]; then
  printf 'Proceed? [y/N] '
  read -r ANSWER || ANSWER=""
  case "$ANSWER" in y | Y | yes | YES) ;; *) echo ">> Not started; nothing was created."; exit 0 ;; esac
fi

# Never adopt (and later delete) a VM this run did not create. (Compute
# Engine: checked again in every zone tried.)
if [ "$RESUME" != "1" ] && vm_exists; then die "$VM_NAME already exists in $ZONE. Delete it or set VM_NAME."; fi
mkdir -p "$LAUNCH_DIR"
CLEANUP_WAIT_MIN="${CLEANUP_WAIT_MIN:-30}"

# record KEY VALUE: appends to the session's launch record (the last value of
# a key wins; resume mode reads it). Never holds the weights URI.
record() { echo "$1=$2" >> "$LAUNCH_DIR/launch_record.txt"; }

# The job's exit code from the results bucket (written there by
# cloud/vm_job_finish.sh), or nothing; cloud/lib_detached.sh asks for it when
# SSH fails while the VM is up.
detached_exit_elsewhere() {
  [ -n "$RESULTS_SESSION_URI" ] || return 0
  gcloud storage cat "$RESULTS_SESSION_URI/EXIT_CODE" 2> /dev/null | head -1 | tr -d ' \r\n'
}

# delete_reason: why the VM may be deleted now (see the header), or nothing.
delete_reason() {
  local s
  if [ "$STARTED" != "1" ]; then echo "the job was not started"; return 0; fi
  if [ -n "$DETACHED_RC_VIA" ]; then echo "the job finished with exit code $DETACHED_RC"; return 0; fi
  case "$DETACHED_END" in
    deadline) echo "the follow deadline was reached"; return 0 ;;
    vm_gone) echo "the VM is $DETACHED_VM_STATE"; return 0 ;;
    job_gone) echo "the job no longer runs and left no EXIT_CODE"; return 0 ;;
  esac
  if [ "$INTERRUPTED" = "1" ]; then echo "interrupted (Ctrl-C or TERM)"; return 0; fi
  s=$(vm_state)
  if vm_state_is_gone "$s"; then echo "the VM is $s"; fi
  return 0
}

# wait_for_api MAX_S: the VM's state as soon as the API answers (backoff),
# UNKNOWN if it did not within MAX_S seconds.
wait_for_api() {
  local s w="$DETACHED_POLL_S" t0
  t0=$(date +%s)
  s=$(vm_state)
  while [ "$s" = "UNKNOWN" ] && [ $(( $(date +%s) - t0 )) -lt "$1" ]; do
    sleep "$w"
    w=$(( w * 2 )); [ "$w" -le "$DETACHED_MAX_BACKOFF_S" ] || w=$DETACHED_MAX_BACKOFF_S
    s=$(vm_state)
  done
  echo "$s"
}

# The launcher's records go into the fetched session and the results bucket.
launcher_files_out() {
  local f files=""
  for f in launch_record.txt uploaded_files.txt launcher_events.txt; do
    [ ! -f "$LAUNCH_DIR/$f" ] || files="$files $LAUNCH_DIR/$f"
  done
  [ -n "$files" ] || return 0
  [ ! -d "results/af3/$SESSION" ] || cp $files "results/af3/$SESSION/"
  if [ -n "$RESULTS_SESSION_URI" ]; then
    if gcloud storage cp $files "$RESULTS_SESSION_URI/" > /dev/null 2>&1; then
      echo ">> Launcher records uploaded to $RESULTS_SESSION_URI/"
    else
      echo "!! Could not upload the launcher records; they are in $LAUNCH_DIR/"
    fi
  fi
}

CREATE_ISSUED=0 STARTED=0 RUN_DONE=0 INTERRUPTED=0
cleanup() {
  local rc=$? op why vstate
  [ "$rc" -ne 0 ] || [ "$RUN_DONE" = "1" ] || rc=1
  trap - EXIT INT TERM
  set +e
  if [ "$CREATE_ISSUED" = "1" ]; then
    why=$(delete_reason)
    vstate=""
    [ "$STARTED" != "1" ] && [ -n "$why" ] || vstate=$(vm_state)
    if [ "$vstate" = "UNKNOWN" ] && [ -n "$why" ] && [ "$INTERRUPTED" != "1" ]; then
      echo "!! The API cannot be reached from this laptop: waiting up to $CLEANUP_WAIT_MIN min for it, to fetch and delete"
      detached_event "cleanup: API unreachable; waiting up to $CLEANUP_WAIT_MIN min"
      vstate=$(wait_for_api $(( CLEANUP_WAIT_MIN * 60 )))
      detached_event "cleanup: after waiting, VM state $vstate"
    fi
    if [ "$STARTED" = "1" ]; then
      ! vm_state_is_gone "$vstate" || DETACHED_VM_GONE=1
      if results_fetch "$OUT_REL" results/af3 "$FETCH_EFFECTIVE" "$RESULTS_SESSION_URI"; then
        detached_event "fetch: done ($FETCH_EFFECTIVE)"
        if [ -n "$COMPARE_REF" ] && [ -d "$COMPARE_REF" ] && [ -d "results/af3/$SESSION" ]; then
          echo ">> Comparing with $COMPARE_REF (results/af3/$SESSION/compare_runs.txt)"
          python3 harness/compare_runs.py --session "results/af3/$SESSION" --reference "$COMPARE_REF" | sed 's/^/   /'
          [ "$FETCH_EFFECTIVE" = "full" ] || echo "   (FETCH=light: the full confidences stayed in the bucket and show as missing)"
        fi
      else
        detached_event "fetch: failed (SSH and bucket)"
      fi
    fi
    if [ -n "$why" ]; then
      echo ">> Deleting $VM_NAME ($why)"
      detached_event "delete: requested ($why)"
      if vm_delete; then
        detached_event "delete: done"
      else
        op=$(vm_state)
        if [ "$op" = "NOT_FOUND" ]; then detached_event "delete: the VM was already gone"; else detached_event "delete: failed (VM state $op)"; fi
        if [ "$VM_API" = "gce" ]; then
          op=$(vm_insert_op status)
          if [ "$op" = "PENDING" ] || [ "$op" = "RUNNING" ]; then
            echo "!! The create request for $VM_NAME is still queued and could not be cancelled."
            echo "!! If it is granted, Google deletes the VM after $MAX_RUN_DURATION. To delete it sooner:"
            echo "!!   gcloud compute instances delete $VM_NAME --project=$PROJECT --zone=$ZONE --quiet"
          fi
        fi
      fi
    else
      echo "!! $VM_NAME is NOT deleted: the API reports it $vstate and its job has not finished."
      echo "!! It goes on running; Google deletes it at the end of its lifetime" \
           "($([ "$VM_API" = tpu ] && echo "watchdog, $WATCHDOG_HOURS h" || echo "--max-run-duration=$MAX_RUN_DURATION") after creation)."
      echo "!! To follow it again, fetch and delete it:"
      echo "!!   RESUME_VM=$VM_NAME SESSION=$SESSION$([ "$RESUME_LEGACY" = 1 ] && echo " ZONE=$ZONE") bash cloud/af3_run.sh"
      detached_event "kept: VM $vstate and job not finished; not deleted (resume: RESUME_VM=$VM_NAME SESSION=$SESSION)"
      [ "$rc" -ne 0 ] || rc=1
    fi
  fi
  vm_leftovers
  detached_event "launcher: exit code $rc"
  launcher_files_out
  echo ">> exit code $rc"
  [ ! -d "results/af3/$SESSION" ] || echo ">> results: results/af3/$SESSION/"
  exit "$rc"
}
trap cleanup EXIT
trap 'INTERRUPTED=1; exit 130' INT TERM

# follow_job: follows the started job to its end (or the deadline).
follow_job() {
  record FOLLOW_UNTIL $(( $(date +%s) + DETACHED_DEADLINE_MIN * 60 ))
  if ! detached_wait "$OUT_REL"; then
    [ "$DETACHED_VM_GONE" != "1" ] || die "$VM_NAME was preempted or deleted during the run; cleaning up."
    [ "$DETACHED_END" != "deadline" ] || die "Follow deadline reached; cleaning up (results first)."
    die "No result from the job."
  fi
  if [ "$DETACHED_RC" != "0" ]; then
    echo "!! The job exited with $DETACHED_RC (setup: 3 inputs, 4 weights, 5 device or TPU stack;" \
         "plan: 1 some runs failed, all recorded, 5 a libtpu switch failed); results are fetched below"
    exit "$DETACHED_RC"
  fi
  echo ">> Plan finished: every harness run exited 0"
  RUN_DONE=1
}

# resume_attach: checks that VM_NAME is this session's VM (labels, session
# folder); dies before anything is touched otherwise.
label() { gcloud $VM_RES describe "$VM_NAME" $GC --format="value(labels.$1)" 2> /dev/null | tail -1; }
resume_attach() {
  local s want reply
  s=$(vm_state)
  case "$s" in
    UNKNOWN) die "The API cannot be reached to check $VM_NAME; nothing was touched. Try again when the network is back." ;;
    NOT_FOUND) die "$VM_NAME does not exist in $ZONE: nothing to follow.$([ -z "$RESULTS_SESSION_URI" ] || echo " What the job uploaded is in $RESULTS_SESSION_URI/.")" ;;
  esac
  [ "$RESUME_LEGACY" != "1" ] || [ "$PLAN_NAME" = "$LEGACY_PLAN_NAME" ] \
    || die "PLAN $PLAN does not match the session's plan $LEGACY_PLAN_NAME; nothing was touched."
  want=$(echo "$SESSION" | tr 'A-Z' 'a-z' | cut -c1-63)
  [ "$(label purpose)" = "af3-run" ] && [ "$(label platform)" = "$PLATFORM" ] && [ "$(label plan)" = "$PLAN_NAME" ] \
    && { [ "$RESUME_LEGACY" = "1" ] || [ "$(label session)" = "$want" ]; } \
    || die "$VM_NAME does not carry this session's labels (purpose af3-run, platform $PLATFORM, plan $PLAN_NAME$([ "$RESUME_LEGACY" = 1 ] || echo ", session $want")); nothing was touched."
  reply=$(vm_ssh "if [ -d \$HOME/$OUT_REL ]; then echo '@@ HAS'; else echo '@@ MISSING'; fi" 2> /dev/null | grep '^@@ ' | head -1 || true)
  case "$reply" in
    "@@ MISSING") die "$VM_NAME has no folder ~/$OUT_REL: not this session's VM; nothing was touched." ;;
    "@@ HAS") ;;
    *) echo "   (SSH not reachable now: the record and the labels match; following anyway)" ;;
  esac
  if [ "$RESUME_LEGACY" = "1" ]; then
    record SESSION "$SESSION"; record VM_NAME "$VM_NAME"; record VM_API "$VM_API"; record PLATFORM "$PLATFORM"
    record PROJECT "$PROJECT"; record ZONE "$ZONE"; record PLAN "$PLAN"; record MACHINE_TYPE "$MACHINE_TYPE"
    record PROVISIONING "$PROVISIONING"; record WEIGHTS "$WEIGHTS"; record RESULTS_GCS_URI "$RESULTS_GCS_URI"
    record FETCH "$FETCH"; record LIBTPU_VERSION "$LIBTPU_VERSION"; record DEADLINE_MIN "$DETACHED_DEADLINE_MIN"
    record MAX_RUN_DURATION "$MAX_RUN_DURATION"; record CREATED 1; record LEGACY 1
  fi
  record RESUMED_UTC "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  detached_event "resume: re-attached to $VM_NAME (state $s)$([ "$RESUME_LEGACY" != 1 ] || echo ", session without a launch record"); deadline $DETACHED_DEADLINE_MIN min"
  if vm_state_is_gone "$s"; then
    CREATE_ISSUED=1 STARTED=1 DETACHED_VM_GONE=1 DETACHED_END=vm_gone DETACHED_VM_STATE=$s
    die "$VM_NAME is $s: nothing to follow; fetching what the bucket has, then deleting it."
  fi
}

if [ "$RESUME" = "1" ]; then
  resume_attach
  CREATE_ISSUED=1 STARTED=1
  DETACHED_SCRIPT=cloud/vm_af3_job.sh
  DETACHED_LOG=job.log
  follow_job
  exit 0
fi

# A new launch: the record first, so a launcher killed from here on can be resumed.
BENCH_DIRTY=$(git status --porcelain -- af3_tpu harness targets inputs cloud | wc -l | tr -d ' ')
record SESSION "$SESSION"; record VM_NAME "$VM_NAME"; record VM_API "$VM_API"; record PLATFORM "$PLATFORM"
record PROJECT "$PROJECT"; record ZONE "$ZONE"; record PLAN "$PLAN"; record MACHINE_TYPE "$MACHINE_TYPE"
record PROVISIONING "$PROVISIONING"; record WEIGHTS "$WEIGHTS"; record RESULTS_GCS_URI "$RESULTS_GCS_URI"
record FETCH "$FETCH"; record LIBTPU_VERSION "$LIBTPU_VERSION"; record DEADLINE_MIN "$DETACHED_DEADLINE_MIN"
record MAX_RUN_DURATION "$MAX_RUN_DURATION"; record COMMIT "$BENCH_COMMIT"; record DIRTY_FILES "$BENCH_DIRTY"
[ -z "$STACK_SPEC" ] || record STACK "$STACK_SPEC"
[ -z "$EIGEN_SOURCE" ] || record EIGEN_SOURCE "$EIGEN_SOURCE"
record LAUNCHED_UTC "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
detached_event "launch: session $SESSION, VM $VM_NAME ($MACHINE_TYPE, $PROVISIONING), plan $PLAN, commit $BENCH_COMMIT, $BENCH_DIRTY uncommitted file(s) under af3_tpu harness targets inputs cloud"

LABELS="purpose=af3-run,platform=$PLATFORM,plan=$PLAN_NAME,session=$(echo "$SESSION" | tr 'A-Z' 'a-z' | cut -c1-63),owner=lorenzo,expires=$EXPIRES"
CREATE_ISSUED=1
if [ "$VM_API" = "tpu" ]; then
  echo ">> Creating $VM_NAME ($MACHINE_TYPE, $RUNTIME) in $ZONE, $PROVISIONING"
  SPOT_FLAG=""
  [ "$PROVISIONING" != "spot" ] || SPOT_FLAG="--spot"
  vm_create_tpu "--accelerator-type=$MACHINE_TYPE --version=$RUNTIME \
--scopes=https://www.googleapis.com/auth/cloud-platform --labels=$LABELS $SPOT_FLAG"
  vm_wait_running "$WAIT_S" || exit 1
  echo ">> Arming watchdog: $VM_NAME deletes itself in $WATCHDOG_HOURS h"
  vm_arm_watchdog "$WATCHDOG_HOURS"
else
  FLAGS="--machine-type=$MACHINE_TYPE --image-family=$IMAGE_FAMILY --image-project=$IMAGE_PROJECT \
--boot-disk-size=$BOOT_DISK_SIZE${BOOT_DISK_TYPE:+ --boot-disk-type=$BOOT_DISK_TYPE} \
--provisioning-model=$(echo "$PROVISIONING" | tr 'a-z' 'A-Z') --max-run-duration=$MAX_RUN_DURATION \
--instance-termination-action=DELETE --maintenance-policy=TERMINATE --reservation-affinity=none \
--scopes=cloud-platform --labels=$LABELS"
  [ "$PROVISIONING" != "flex_start" ] || FLAGS="$FLAGS --request-valid-for-duration=$REQUEST_VALID_FOR"
  # One attempt per zone; Google's suggested zones go to the front, once each.
  ZONES_LEFT=$ZONES TRIED="" ATTEMPT=0 CREATED=0
  while [ -n "$ZONES_LEFT" ]; do
    set -- $ZONES_LEFT
    ZONE=$1
    shift
    ZONES_LEFT="$*"
    case " $TRIED " in *" $ZONE "*) continue ;; esac
    TRIED="$TRIED $ZONE"
    ATTEMPT=$((ATTEMPT + 1))
    vm_init
    echo ">> Zone attempt $ATTEMPT: requesting $VM_NAME ($MACHINE_TYPE) in $ZONE, $PROVISIONING"
    if vm_exists; then die "$VM_NAME already exists in $ZONE. Delete it or set VM_NAME."; fi
    CREATE_ISSUED=1
    if CREATE_OUT=$(vm_create_gce "$FLAGS" 2>&1); then
      echo "$CREATE_OUT" | grep -v '^ *$' | sed 's/^/   /' | head -5
      if vm_wait_running "$WAIT_S"; then CREATED=1; break; fi
      FAILURE=$(vm_insert_failure)
    else
      echo "$CREATE_OUT" | grep -v '^ *$' | sed 's/^/   /' | tail -4
      FAILURE=$(echo "$CREATE_OUT" | vm_classify_failure text)
    fi
    vm_clear_zone || exit 1
    case "$FAILURE" in
      CAPACITY*)
        # Nothing left in this zone and no request pending: nothing to clean up.
        CREATE_ISSUED=0
        SUGGESTED=$(echo ${FAILURE#CAPACITY})
        echo "   zone $ZONE: no capacity${SUGGESTED:+; Google suggests: $SUGGESTED}"
        [ -z "$SUGGESTED" ] || ZONES_LEFT="$SUGGESTED $ZONES_LEFT" ;;
      NONE)
        # For example a Flex-start request still queued: cleanup reports it.
        echo "!! zone $ZONE: the VM did not start and the insert operation recorded no error; stopping"
        exit 1 ;;
      *)
        echo "!! zone $ZONE: ${FAILURE#OTHER }"
        echo "!! Not a capacity error; stopping (no further zones tried)"
        exit 1 ;;
    esac
  done
  if [ "$CREATED" != "1" ]; then
    echo "!! No zone had capacity for $MACHINE_TYPE ($PROVISIONING). Tried, in order:$TRIED"
    exit 1
  fi
  REGION="${ZONE%-*}"
  echo ">> $VM_NAME is running in $ZONE (zone attempt $ATTEMPT)"
fi
record CREATED 1; record ZONE "$ZONE"; record CREATED_UTC "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
detached_event "vm: $VM_NAME running in $ZONE"
echo ">> Waiting for SSH"
vm_wait_ssh || exit 1

echo ">> Uploading the plan's files (commit $BENCH_COMMIT)"
TGZ=$(mktemp -t af3-run.XXXXXX)
FILES="af3_tpu harness targets inputs/manifest.csv cloud/vm_af3_setup.sh cloud/vm_af3_weights.sh cloud/vm_device_check.sh cloud/vm_tpu_stack.sh cloud/vm_af3_job.sh cloud/vm_job_finish.sh"
# The Eigen archive step (plan key eigen_source) needs its script and pin; without the key the upload is unchanged.
[ "$EIGEN_SOURCE" != "archive" ] || FILES="$FILES cloud/vm_eigen_source.sh cloud/eigen_pin.sh"
[ "$MANIFEST" = "inputs/manifest.csv" ] || FILES="$FILES $MANIFEST"
FILES="$FILES $INPUT_FILES"
# Provenance: the git blob hash of every uploaded file (comparable with
# `git ls-tree -r <commit>`), with the commit and the uncommitted-file count.
{
  echo "# commit $BENCH_COMMIT; uncommitted files under af3_tpu harness targets inputs cloud: $BENCH_DIRTY"
  find $FILES -type f ! -path '*/__pycache__/*' | sort | while read -r f; do echo "$(git hash-object "$f")  $f"; done
} > "$LAUNCH_DIR/uploaded_files.txt"
tar czf "$TGZ" --exclude=__pycache__ $FILES
vm_upload "$TGZ" af3_run.tgz
rm -f "$TGZ"
vm_ssh "mkdir -p ~/alphafold-on-tpu && tar xzf ~/af3_run.tgz -C ~/alphafold-on-tpu && rm -f ~/af3_run.tgz"
if [ -n "$PLAN_CCD" ]; then
  echo ">> Uploading the plan's CCD (components.cif $(echo "$PLAN_CCD" | cut -c1-12)..., zstd)"
  CCD_ZST=$(mktemp -t af3-ccd.XXXXXX)
  zstd -q -f -T0 "$CCD_FILE" -o "$CCD_ZST"
  vm_upload "$CCD_ZST" components.cif.zst
  rm -f "$CCD_ZST"
fi
if [ "$WEIGHTS" = "gcs" ]; then
  URI_FILE=$(mktemp -t af3-uri.XXXXXX)
  chmod 600 "$URI_FILE"
  printf '%s' "$WEIGHTS_GCS_URI" > "$URI_FILE"
  vm_upload "$URI_FILE" .af3_weights_uri
  rm -f "$URI_FILE"
  vm_ssh "chmod 600 ~/.af3_weights_uri"
fi

echo ">> Starting cloud/vm_af3_job.sh detached on the VM (setup, then the plan)"
DETACHED_SCRIPT=cloud/vm_af3_job.sh
DETACHED_LOG=job.log
STARTED=1
detached_start "$OUT_REL" \
  "PLATFORM=$PLATFORM PLAN=$PLAN WEIGHTS=$WEIGHTS TARGETS=$TARGETS SESSION=$SESSION RESULTS_URI=$RESULTS_SESSION_URI LIBTPU_VERSION=$LIBTPU_VERSION${PLAN_CCD:+ CCD_SHA256=$PLAN_CCD}${STACK_SPEC:+ STACK=$STACK_SPEC}${EIGEN_SOURCE:+ EIGEN_SOURCE=$EIGEN_SOURCE}" \
  || die "Could not start the job on the VM."
record STARTED_UTC "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
detached_event "job: started on the VM"
follow_job
