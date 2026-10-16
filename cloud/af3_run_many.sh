#!/usr/bin/env bash
# Draft (2026-10-12, not in the repository): launches a list of plan parts with cloud/af3_run.sh in
# parallel, at most MAX_<PLATFORM> launchers per platform at a time, one log per part, never more than one
# launcher per part, and a summary at the end.
#
# Run from the repo root, in a shell where cloud/env.sh is NOT sourced:
#   PARTS=harness/plans/main/parts_all.txt RESULTS_GCS_URI=gs://BUCKET WEIGHTS_GCS_URI=gs://BUCKET/af3.bin.zst \
#     SPOT=0 bash cloud/af3_run_many.sh
#
# PARTS: one part per line, "PLATFORM PLAN [VAR=value ...]" (cpu, l4, a100, v5e or v6e; PLAN a plan file under
# harness/plans/); blank lines and # comments are ignored. Optional per-part variables, passed to that
# launcher only: ZONE ZONES SPOT PROVISIONING REQUEST_VALID_FOR FETCH LIBTPU_VERSION (ZONES with commas,
# ZONES=us-east4-a,us-east4-c, given to the launcher as its space-separated list). Every other setting
# (RESULTS_GCS_URI, WEIGHTS_GCS_URI, FETCH, SPOT, ...) comes from this shell and reaches every launcher.
# Parts are started in file order, as soon as their platform has a free slot.
#
# v6e parts get PROVISIONING=standard (on demand; notes/analysis_plan.md section 10, main study execution row
# of 2026-10-12) unless their line sets PROVISIONING; V6E_PROVISIONING changes that default.
#
# A100 parts (follow-up 3, notes/analysis_plan.md section 10, rows of 2026-10-16): every one gets
# MACHINE_TYPE=$A100_MACHINE_TYPE (required when the file has A100 parts; the launcher takes a2-ultragpu-1g only),
# so one run of this script never mixes A100 machine types; a per-part MACHINE_TYPE is refused, and so is a
# MACHINE_TYPE in this shell (it would reach every launcher). Besides MAX_A100, at most A100_REGION_CAPS run per
# region ("region:N ...", default "us-central1:2 europe-west4:1", the regional NVIDIA_A100_80GB_GPUS quota of
# 2026-10-16): a part is started only when a region has a free slot, and gets that region's zones
# (A100_REGION_ZONES, default "us-central1:us-central1-a,us-central1-c europe-west4:europe-west4-a", the zones
# that offer a2-ultragpu-1g) unless its line sets ZONE or ZONES of one listed region (then that region's cap).
#
# Settings: MAX_L4 (8: GPUS_ALL_REGIONS), MAX_A100 (1), MAX_V5E (1), MAX_V6E (1), MAX_CPU (4) launchers at a time;
# STAGGER_S (5) seconds between two launches (session and VM names carry the launch second);
# POLL_S (15); MAX_ATTEMPTS (1) launches per part, a further attempt only when the previous one created no
# VM (no capacity in any zone), after RETRY_WAIT_S (600); ALLOW_RELAUNCH ("" or a space-separated list of
# plan names) to launch a part again that has an earlier session; LAUNCHER (cloud/af3_run.sh);
# ALLOW_OTHER_LAUNCHERS=1 to start while af3_run.sh processes this script did not start are running
# (they hold quota and VMs this script cannot count); YES=1 (no prompt; the launchers always get YES=1).
#
# One launcher per part: a part is skipped if a session of it already created a VM (results/af3/.launcher/
# <stamp>_<platform>_<plan>/launch_record.txt with CREATED=1, or results/af3/<that session>/ exists),
# unless its plan name is in ALLOW_RELAUNCH; a lock (results/af3/.many/locks/<platform>_<plan>, created
# atomically, holding the launcher's PID) keeps two copies of this script from launching it together; a
# stale lock (its process gone) is removed only if no session of the part created a VM.
#
# State and logs: results/af3/.many/<RUN_ID>/ (RUN_ID = UTC stamp): <plan>.log per attempt
# (<plan>.attempt2.log, ...), summary.txt. Touch results/af3/.many/<RUN_ID>/STOP to start no further
# parts (running launchers go on). Ctrl-C does the same; a second Ctrl-C leaves at once. The launchers run
# in the background and ignore Ctrl-C (bash: background jobs of a non-interactive shell ignore SIGINT), so
# they follow, fetch and delete their VMs as usual; to stop one, kill -TERM its PID (listed), and it
# fetches and deletes its VM as on Ctrl-C.
#
# A part whose launcher failed is never launched again automatically (a further attempt happens only when
# no VM was created, and only with MAX_ATTEMPTS above 1). The summary lists each failed or unstarted part with
# its sessions, and writes them to relaunch_parts.txt with the command that relaunches them unchanged
# (section 10, main study execution row (c)); a part whose VM was kept is listed with its resume command
# instead, since relaunching it would run it twice.
#
# Exit status: 0 if every part's launcher exited 0, 1 otherwise, 2 if nothing was started (refused).
# macOS bash 3.2: no arrays.
set -uo pipefail

die() { echo "!! $*" >&2; exit 2; }
say() { echo "[$(date -u +%H:%M:%S)] $*"; }

[ -z "${AF2_COMMIT:-}" ] || die "cloud/env.sh is sourced in this shell. Open a fresh shell."
ROOT=$(git rev-parse --show-toplevel 2> /dev/null) || die "Run from inside the repository."
cd "$ROOT" || die "cannot enter $ROOT"
: "${PARTS:?PARTS=<file of parts> is required}"
[ -f "$PARTS" ] || die "No parts file: $PARTS"
LAUNCHER="${LAUNCHER:-cloud/af3_run.sh}"
[ -f "$LAUNCHER" ] || die "No launcher: $LAUNCHER"
MAX_L4="${MAX_L4:-8}" MAX_A100="${MAX_A100:-1}" MAX_V5E="${MAX_V5E:-1}" MAX_V6E="${MAX_V6E:-1}" MAX_CPU="${MAX_CPU:-4}"
A100_MACHINE_TYPE="${A100_MACHINE_TYPE:-}"
A100_REGION_CAPS="${A100_REGION_CAPS:-us-central1:2 europe-west4:1}"
A100_REGION_ZONES="${A100_REGION_ZONES:-us-central1:us-central1-a,us-central1-c europe-west4:europe-west4-a}"
[ -z "${MACHINE_TYPE:-}" ] || die "MACHINE_TYPE is set in this shell and would reach every launcher; unset it (A100 parts take A100_MACHINE_TYPE)."
STAGGER_S="${STAGGER_S:-5}" POLL_S="${POLL_S:-15}" MAX_ATTEMPTS="${MAX_ATTEMPTS:-1}" RETRY_WAIT_S="${RETRY_WAIT_S:-600}"
ALLOW_RELAUNCH="${ALLOW_RELAUNCH:-}"
for v in MAX_L4 MAX_A100 MAX_V5E MAX_V6E MAX_CPU STAGGER_S POLL_S MAX_ATTEMPTS RETRY_WAIT_S; do
  eval "x=\$$v"
  echo "$x" | grep -Eq '^[0-9]+$' || die "$v must be a whole number (got '$x')."
done
[ "$MAX_ATTEMPTS" -ge 1 ] || die "MAX_ATTEMPTS must be 1 or more."
for rc in $A100_REGION_CAPS; do
  echo "$rc" | grep -Eq '^[a-z]+-[a-z]+[0-9]+:[0-9]+$' || die "A100_REGION_CAPS: '$rc' is not region:N."
done
# A100 regions: the cap of one, the zones of one (comma list), the region of a zone.
a100_cap() { local rc; for rc in $A100_REGION_CAPS; do [ "${rc%%:*}" = "$1" ] && { echo "${rc#*:}"; return 0; }; done; return 1; }
a100_zones() { local rz; for rz in $A100_REGION_ZONES; do [ "${rz%%:*}" = "$1" ] && { echo "${rz#*:}"; return 0; }; done; return 1; }
zone_region() { echo "${1%-*}"; }

cap() { case "$1" in l4) echo "$MAX_L4" ;; a100) echo "$MAX_A100" ;; v5e) echo "$MAX_V5E" ;; v6e) echo "$MAX_V6E" ;; cpu) echo "$MAX_CPU" ;; esac; }
# The launcher's own transformation of the plan file name (cloud/af3_run.sh, PLAN_NAME).
plan_name() { basename "$1" .yaml | tr 'A-Z' 'a-z' | tr -c 'a-z0-9_\n-' '-'; }
# Sessions of a part that created a VM (or were fetched): one per line.
earlier_sessions() {
  local p=$1 n=$2 r s
  for r in results/af3/.launcher/*_"${p}"_"${n}"/launch_record.txt; do
    [ -f "$r" ] || continue
    s=$(basename "$(dirname "$r")")
    if grep -q '^CREATED=1$' "$r"; then echo "$s"; fi
  done
  for s in results/af3/*_"${p}"_"${n}"; do
    [ -d "$s" ] && echo "$(basename "$s")"
  done
}
relaunch_allowed() { case " $ALLOW_RELAUNCH " in *" $1 "*) return 0 ;; esac; return 1; }

# Other af3_run.sh processes (not started by this script) hold slots this script cannot count.
OTHERS=$(pgrep -f "cloud/af3_run.sh" 2> /dev/null | tr '\n' ' ')
if [ -n "$OTHERS" ] && [ "${ALLOW_OTHER_LAUNCHERS:-0}" != "1" ]; then
  die "af3_run.sh is already running (PIDs $OTHERS). Wait for it, or set ALLOW_OTHER_LAUNCHERS=1 and lower MAX_<PLATFORM> to leave room for it."
fi

# ---- Read and check every part before anything starts.
RUN_ID=$(date -u +%Y%m%dT%H%M%SZ)
RUN_DIR="results/af3/.many/$RUN_ID"
LOCKS="results/af3/.many/locks"
[ ! -e "$RUN_DIR" ] || die "$RUN_DIR exists."
mkdir -p "$RUN_DIR/parts" "$LOCKS" || die "cannot create $RUN_DIR"
N=0 BAD=0
while IFS= read -r line || [ -n "$line" ]; do
  line=$(echo "$line" | sed 's/#.*//; s/^[[:space:]]*//; s/[[:space:]]*$//')
  [ -n "$line" ] || continue
  set -- $line
  platform=$1 plan=${2:-}
  shift 2 2> /dev/null || shift $#
  extra=""
  for kv in "$@"; do
    case "$kv" in
      ZONE=* | ZONES=* | SPOT=* | PROVISIONING=* | REQUEST_VALID_FOR=* | FETCH=* | LIBTPU_VERSION=*) extra="$extra $kv" ;;
      *) echo "!! $plan: '$kv' is not a per-part setting this script passes on"; BAD=1 ;;
    esac
  done
  case "$platform" in cpu | l4 | a100 | v5e | v6e) ;; *) echo "!! '$platform' is not a platform (cpu, l4, a100, v5e, v6e): $line"; BAD=1; continue ;; esac
  pinned=""
  if [ "$platform" = a100 ]; then
    [ -n "$A100_MACHINE_TYPE" ] || { echo "!! $plan: A100 parts need A100_MACHINE_TYPE (a2-ultragpu-1g) in this shell"; BAD=1; continue; }
    for kv in $extra; do
      case "$kv" in
        ZONE=* | ZONES=*)
          for z in $(echo "${kv#*=}" | tr ',' ' '); do
            r=$(zone_region "$z")
            [ -z "$pinned" ] || [ "$pinned" = "$r" ] || { echo "!! $plan: its zones span regions $pinned and $r (one region per A100 part)"; BAD=1; }
            pinned=$r
          done ;;
      esac
    done
    if [ -n "$pinned" ] && ! a100_cap "$pinned" > /dev/null; then
      echo "!! $plan: region $pinned has no A100 slot in A100_REGION_CAPS ($A100_REGION_CAPS)"; BAD=1; continue
    fi
  fi
  [ -f "$plan" ] || { echo "!! no plan file: $plan"; BAD=1; continue; }
  name=$(plan_name "$plan")
  key="${platform}_${name}"
  if [ -d "$RUN_DIR/parts/$key" ]; then echo "!! $platform $plan is listed twice"; BAD=1; continue; fi
  deadline=$(python3 harness/run_plan.py --plan "$plan" --platform "$platform" --describe 2> /dev/null \
    | python3 -c "import json,sys; print(json.load(sys.stdin)['deadline_min'])" 2> /dev/null) || deadline=""
  if [ -z "$deadline" ] || [ "$deadline" -le 0 ] 2> /dev/null; then
    echo "!! $plan does not expand for $platform with a deadline (harness/run_plan.py --describe)"; BAD=1; continue
  fi
  N=$((N + 1))
  d="$RUN_DIR/parts/$key"
  mkdir -p "$d"
  printf '%s\n' "$N" > "$d/order"
  printf '%s %s%s\n' "$platform" "$plan" "$extra" > "$d/line"
  [ "$platform" != a100 ] || echo "$pinned" > "$d/pinned_region"
  echo 0 > "$d/attempts"
  echo 0 > "$d/not_before"
  prev=$(earlier_sessions "$platform" "$name" | sort -u | tr '\n' ' ')
  if [ -n "$prev" ] && ! relaunch_allowed "$name"; then
    echo "skipped: already launched (sessions: $prev); ALLOW_RELAUNCH=\"$name\" launches it again" > "$d/result"
    echo done > "$d/status"
  else
    echo pending > "$d/status"
  fi
done < "$PARTS"
[ "$BAD" = 0 ] || die "The parts file has errors (above); nothing was started."
[ "$N" -gt 0 ] || die "No parts in $PARTS."

parts_in_order() { for d in "$RUN_DIR"/parts/*/; do echo "$(cat "$d/order") $(basename "$d")"; done | sort -n | cut -d' ' -f2; }
count_status() { local k c=0; for k in $(parts_in_order); do [ "$(cat "$RUN_DIR/parts/$k/status")" = "$1" ] && c=$((c + 1)); done; echo $c; }

echo ">> Parts from $PARTS ($N): at most l4 $MAX_L4, a100 $MAX_A100, v5e $MAX_V5E, v6e $MAX_V6E, cpu $MAX_CPU at a time; up to $MAX_ATTEMPTS launch(es) per part"
if ls -d "$RUN_DIR"/parts/a100_* > /dev/null 2>&1; then
  echo "   a100: machine type $A100_MACHINE_TYPE for every part; per region at most $A100_REGION_CAPS; zones $A100_REGION_ZONES"
fi
for k in $(parts_in_order); do
  d="$RUN_DIR/parts/$k"
  if [ "$(cat "$d/status")" = done ]; then echo "   $(cat "$d/line")   [$(cat "$d/result")]"; else echo "   $(cat "$d/line")"; fi
done
echo "   logs and summary: $RUN_DIR/"
echo "   each launcher prints its own worst-case cost in its log (YES=1 is passed to every launcher)"
if [ "$(count_status pending)" = 0 ]; then
  echo ">> Nothing to launch."
  exit 2
fi
if [ "${YES:-0}" != "1" ]; then
  printf 'Launch these parts? [y/N] '
  read -r ANSWER || ANSWER=""
  case "$ANSWER" in y | Y | yes | YES) ;; *) echo ">> Not started; nothing was launched."; rm -rf "$RUN_DIR"; exit 2 ;; esac
fi

# ---- Locks: one launcher per part, also across copies of this script.
lock_part() {   # KEY PLATFORM NAME: 0 if this script now holds the part's lock
  local k=$1 p=$2 n=$3 l="$LOCKS/$1" pid
  if mkdir "$l" 2> /dev/null; then echo "$$ $RUN_ID" > "$l/owner"; return 0; fi
  pid=$(cut -d' ' -f1 "$l/owner" 2> /dev/null)
  [ -f "$l/launcher_pid" ] && pid=$(cat "$l/launcher_pid")
  if [ -n "$pid" ] && kill -0 "$pid" 2> /dev/null; then return 1; fi
  if [ -n "$(earlier_sessions "$p" "$n")" ] && ! relaunch_allowed "$n"; then return 1; fi
  rm -rf "$l" && mkdir "$l" 2> /dev/null && echo "$$ $RUN_ID" > "$l/owner"
}
unlock_part() { rm -rf "$LOCKS/$1"; }

launch() {   # KEY
  local k=$1 d="$RUN_DIR/parts/$1" platform plan extra n a log
  read -r platform plan extra < "$d/line"
  n=$(plan_name "$plan")
  if [ -n "$(earlier_sessions "$platform" "$n")" ] && ! relaunch_allowed "$n" && [ "$(cat "$d/attempts")" = 0 ]; then
    echo "skipped: launched meanwhile by another process (sessions: $(earlier_sessions "$platform" "$n" | sort -u | tr '\n' ' '))" > "$d/result"
    echo done > "$d/status"; return
  fi
  if ! lock_part "$k" "$platform" "$n"; then
    echo "skipped: another launcher holds this part ($LOCKS/$k)" > "$d/result"
    echo done > "$d/status"; return
  fi
  a=$(( $(cat "$d/attempts") + 1 )); echo $a > "$d/attempts"
  log="$RUN_DIR/$n.log"; [ "$a" = 1 ] || log="$RUN_DIR/$n.attempt$a.log"
  echo "$log" > "$d/log"
  rm -f "$d/rc" "$d/launcher_pid"
  (
    set -- YES=1 PLATFORM="$platform" PLAN="$plan"
    case " $extra " in
      *" PROVISIONING="*) ;;
      *) [ "$platform" != v6e ] || set -- "$@" PROVISIONING="${V6E_PROVISIONING:-standard}" ;;
    esac
    if [ "$platform" = a100 ]; then
      set -- "$@" MACHINE_TYPE="$A100_MACHINE_TYPE"
      [ -n "$(cat "$d/pinned_region")" ] || set -- "$@" ZONES="$(a100_zones "$(cat "$d/region")" | tr ',' ' ')"
    fi
    for kv in $extra; do
      case "$kv" in ZONES=*) kv=$(echo "$kv" | tr ',' ' ') ;; esac
      set -- "$@" "$kv"
    done
    env "$@" bash "$LAUNCHER" < /dev/null > "$log" 2>&1 &
    lp=$!
    echo $lp > "$d/launcher_pid"; echo $lp > "$LOCKS/$k/launcher_pid"
    wait $lp
    echo $? > "$d/rc"
  ) &
  echo running > "$d/status"
  LAST_LAUNCH=$(date +%s)
  sleep 1
  say "started $platform $plan (attempt $a, launcher PID $(cat "$d/launcher_pid" 2> /dev/null || echo '?'), log $log)$([ "$platform" != a100 ] || echo ", region $(cat "$d/region")")"
}

finish() {   # KEY: the launcher has exited
  local k=$1 d="$RUN_DIR/parts/$1" rc platform plan extra n log session rec created
  rc=$(cat "$d/rc"); log=$(cat "$d/log")
  read -r platform plan extra < "$d/line"
  n=$(plan_name "$plan")
  session=$(sed -n 's#^   session: results/af3/\([^/]*\)/.*#\1#p' "$log" | tail -1)
  rec="results/af3/.launcher/$session/launch_record.txt"
  created=0
  [ -n "$session" ] && [ -f "$rec" ] && grep -q '^CREATED=1$' "$rec" && created=1
  echo "$session" >> "$d/sessions"
  if [ "$rc" = 0 ]; then
    echo "ok: launcher exit 0, session $session" > "$d/result"; echo done > "$d/status"
  elif grep -q 'is NOT deleted' "$log"; then
    echo "VM KEPT (job not finished): launcher exit $rc, session $session; $(grep -m1 'RESUME_VM=' "$log" | sed 's/^!! *//')" > "$d/result"
    echo done > "$d/status"
  elif [ "$created" = 1 ]; then
    echo "FAILED after a VM was created: launcher exit $rc, session $session (not relaunched)" > "$d/result"
    echo done > "$d/status"
  elif [ "$(cat "$d/attempts")" -lt "$MAX_ATTEMPTS" ] && [ "$STOPPING" = 0 ]; then
    echo "no VM created: launcher exit $rc${session:+, session $session}; next attempt after $RETRY_WAIT_S s" > "$d/result"
    echo $(( $(date +%s) + RETRY_WAIT_S )) > "$d/not_before"
    echo pending > "$d/status"
  else
    echo "FAILED, no VM created: launcher exit $rc${session:+, session $session} after $(cat "$d/attempts") attempt(s)" > "$d/result"
    echo done > "$d/status"
  fi
  unlock_part "$k"
  say "$platform $plan: $(cat "$d/result")"
}

STOPPING=0 LAST_LAUNCH=0
on_signal() {
  if [ "$STOPPING" = 1 ]; then
    echo; say "second interrupt: leaving now. Running launchers go on and delete their VMs:"
    for k in $(parts_in_order); do
      [ "$(cat "$RUN_DIR/parts/$k/status")" = running ] && echo "   PID $(cat "$RUN_DIR/parts/$k/launcher_pid" 2> /dev/null) $(cat "$RUN_DIR/parts/$k/line")"
    done
    exit 1
  fi
  STOPPING=1
  echo; say "interrupt: no further part starts; waiting for the running launchers (Ctrl-C again to leave; kill -TERM PID stops one)"
}
trap on_signal INT TERM

running_on() { local k c=0; for k in $(parts_in_order); do
  [ "$(cat "$RUN_DIR/parts/$k/status")" = running ] && [ "$(cut -d' ' -f1 "$RUN_DIR/parts/$k/line")" = "$1" ] && c=$((c + 1)); done; echo $c; }
# A100 parts running in a region (the region each was started in).
a100_running_in() { local k c=0; for k in $(parts_in_order); do
  [ "$(cat "$RUN_DIR/parts/$k/status")" = running ] && [ -f "$RUN_DIR/parts/$k/region" ] \
    && [ "$(cat "$RUN_DIR/parts/$k/region")" = "$1" ] && c=$((c + 1)); done; echo $c; }
# KEY: a region with a free A100 slot for this part (its own pinned region, else the first of A100_REGION_CAPS).
a100_free_region() { local pin rc r; pin=$(cat "$RUN_DIR/parts/$1/pinned_region")
  for rc in $A100_REGION_CAPS; do r=${rc%%:*}
    [ -z "$pin" ] || [ "$pin" = "$r" ] || continue
    [ "$(a100_running_in "$r")" -lt "${rc#*:}" ] && { echo "$r"; return 0; }
  done; return 1; }

while :; do
  for k in $(parts_in_order); do
    [ "$(cat "$RUN_DIR/parts/$k/status")" = running ] && [ -f "$RUN_DIR/parts/$k/rc" ] && finish "$k"
  done
  if [ -f "$RUN_DIR/STOP" ] && [ "$STOPPING" = 0 ]; then STOPPING=1; say "STOP file found: no further part starts"; fi
  if [ "$STOPPING" = 0 ]; then
    for k in $(parts_in_order); do
      d="$RUN_DIR/parts/$k"
      [ "$(cat "$d/status")" = pending ] || continue
      [ "$(date +%s)" -ge "$(cat "$d/not_before")" ] || continue
      p=$(cut -d' ' -f1 "$d/line")
      [ "$(running_on "$p")" -lt "$(cap "$p")" ] || continue
      if [ "$p" = a100 ]; then reg=$(a100_free_region "$k") || continue; echo "$reg" > "$d/region"; fi
      wait_s=$(( LAST_LAUNCH + STAGGER_S - $(date +%s) ))
      [ "$wait_s" -le 0 ] || sleep "$wait_s"
      [ "$STOPPING" = 0 ] && [ ! -f "$RUN_DIR/STOP" ] || break
      launch "$k"
    done
  fi
  [ "$(count_status running)" = 0 ] && { [ "$STOPPING" = 1 ] || [ "$(count_status pending)" = 0 ]; } && break
  sleep "$POLL_S"
done

# ---- Summary.
FAILS=0
S="$RUN_DIR/summary.txt"
echo "af3_run_many $RUN_ID: $N part(s) from $PARTS" > "$S"
for k in $(parts_in_order); do
  d="$RUN_DIR/parts/$k"
  if [ "$(cat "$d/status")" = pending ]; then echo "not started (stopped)" > "$d/result"; fi
  r=$(cat "$d/result" 2> /dev/null || echo '?')
  case "$r" in ok:* | skipped:*) ;; *) FAILS=$((FAILS + 1)) ;; esac
  printf '%-4s %-44s %s\n' "$(cut -d' ' -f1 "$d/line")" "$(cut -d' ' -f2 "$d/line")" "$r" >> "$S"
done
echo "launchers that did not exit 0, or parts not started: $FAILS" >> "$S"
# Failed and unstarted parts, for an unchanged relaunch decided by the operator; kept VMs, for a resume.
REL="$RUN_DIR/relaunch_parts.txt" RELNAMES="" KEPT=""
: > "$REL"
for k in $(parts_in_order); do
  d="$RUN_DIR/parts/$k"
  r=$(cat "$d/result" 2> /dev/null || echo '?')
  case "$r" in
    FAILED* | "not started"*)
      read -r platform plan extra < "$d/line"
      echo "$(cat "$d/line")   # $(echo "$r" | cut -c1-120)" >> "$REL"
      [ ! -s "$d/sessions" ] || RELNAMES="$RELNAMES $(plan_name "$plan")" ;;
    "VM KEPT"*) KEPT="$KEPT
   $(cut -d' ' -f2 "$d/line"): $(echo "$r" | sed -n 's/.*\(RESUME_VM=.*\)$/\1/p')" ;;
  esac
done
if [ -s "$REL" ]; then
  {
    echo "Failed or unstarted parts (not relaunched automatically), in $REL:"
    sed 's/^/   /' "$REL"
    echo "To relaunch them unchanged (record the relaunch in section 10), with the same settings as this run:"
    echo "   PARTS=$REL${RELNAMES:+ ALLOW_RELAUNCH=\"${RELNAMES# }\"} RESULTS_GCS_URI=... WEIGHTS_GCS_URI=... bash cloud/af3_run_many.sh"
  } >> "$S"
else
  rm -f "$REL"
fi
[ -z "$KEPT" ] || echo "VMs kept by their launcher (job not finished): resume, do not relaunch:$KEPT" >> "$S"
cat "$S"
[ "$FAILS" = 0 ]
