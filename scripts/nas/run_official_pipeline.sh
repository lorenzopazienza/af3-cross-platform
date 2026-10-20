#!/bin/bash
# Follow-up 6 (label msa): AlphaFold3's own data pipeline on the official-pipeline inputs.
# notes/analysis_plan.md section 10, row "Follow-up experiment 6, AlphaFold3's own data pipeline" (2026-10-19).
#
# Runs inside an af3pipeline VM (naspaz8-af3 or naspaz6-af3) as lollo, one target at a time, in tmux.
# Settings are those of the row and of /data/tools/smoke_2pv7.sh: --norun_inference, every other flag at its
# default, databases mounted read-only, --jackhmmer_n_cpu=3 on naspaz8-af3 and 4 on naspaz6-af3.
#
#   bash run_official_pipeline.sh run1 all     # naspaz8-af3: the 60 targets in nas_order (goes to inference)
#   bash run_official_pipeline.sh run2 pilot   # naspaz8-af3: second run of the 10 pilot targets
#   bash run_official_pipeline.sh run1 pilot   # naspaz6-af3: the 10 pilot targets
#
# LIMIT=3 runs only the next 3 targets and stops (first timing and memory check).
# Re-running the same command resumes: targets already completed are skipped, a failed attempt is kept in failed/.
# The script stops at the first failure, or if the VM memory peak of a target goes above 85% of the VM's memory.
set -uo pipefail

RUN=${1:?"usage: run_official_pipeline.sh run1|run2 all|pilot"}
SET=${2:?"usage: run_official_pipeline.sh run1|run2 all|pilot"}
LIMIT=${LIMIT:-0}

TARBALL=/data/inputs_official.tar.gz
TARBALL_SHA=4579cd6f676bb1219b5bce36fe9140bebc8e884a544deb3f9d16da487b71e5b3
INPUTS=/data/inputs_official
MANIFEST_SHA=f047e3a64ac943f5437f72c8012f295543c5d2a6822dfc07034dfd048da9e934
IMAGE=alphafold3:v3.0.4
IMAGE_ID=sha256:01f1b25a5ccfd5dd90240880b62b6dc1ded9f8895d352a2c6435ac293cfde38a
DB_DIR=/data/public_databases
DB_FINGERPRINT=63e88cfe6c2527cb16cf1b57d1aedb380859b9df32007ef766d47edbe791fad2

die() { echo "STOP: $*" >&2; exit 1; }
now() { date -u +%Y-%m-%dT%H:%M:%SZ; }

HOST=$(hostname)
case "$HOST" in
  naspaz8-af3) NCPU=3 ;;
  naspaz6-af3) NCPU=4 ;;
  *) die "host $HOST is not naspaz8-af3 or naspaz6-af3" ;;
esac
case "$HOST/$RUN/$SET" in
  naspaz8-af3/run1/all|naspaz8-af3/run2/pilot|naspaz6-af3/run1/pilot) ;;
  *) die "$HOST $RUN $SET is not one of the runs of the follow-up 6 row" ;;
esac

OUT=/data/runs/${HOST}_${RUN}
mkdir -p "$OUT/output" "$OUT/logs" "$OUT/failed"
exec 9>"$OUT/.lock"
flock -n 9 || die "another run is already using $OUT"
CSV="$OUT/results.csv"
[ -f "$CSV" ] || echo "pdb_id,host,run,jackhmmer_n_cpu,start_utc,end_utc,seconds,vm_mem_baseline_mib,vm_mem_peak_mib,exit_code,data_json,data_json_sha256,unpaired_seqs,paired_seqs,templates" > "$CSV"

# 1. Inputs: tarball, manifest and every JSON against the manifest
[ -f "$TARBALL" ] || die "$TARBALL not found (copy it from the Mac with scp)"
[ "$(sha256sum "$TARBALL" | cut -d' ' -f1)" = "$TARBALL_SHA" ] || die "$TARBALL SHA-256 mismatch"
[ -d "$INPUTS" ] || tar -xzf "$TARBALL" -C /data
[ "$(sha256sum "$INPUTS/manifest.csv" | cut -d' ' -f1)" = "$MANIFEST_SHA" ] || die "manifest.csv SHA-256 mismatch"
python3 - "$INPUTS" "$SET" "$OUT/targets.txt" <<'PY' || die "input check failed"
import csv, hashlib, os, sys
inputs, which, out = sys.argv[1:]
rows = sorted(csv.DictReader(open(os.path.join(inputs, "manifest.csv"))), key=lambda r: int(r["nas_order"]))
assert len(rows) == 60, len(rows)
for r in rows:
    h = hashlib.sha256(open(os.path.join(inputs, r["pdb_id"] + ".json"), "rb").read()).hexdigest()
    if h != r["official_input_sha256"]:
        sys.exit(f"{r['pdb_id']}.json SHA-256 mismatch")
sel = [r["pdb_id"] for r in rows if which == "all" or r["pilot"] == "yes"]
open(out, "w").write("\n".join(sel) + "\n")
print(f"inputs OK: 60 JSON match the manifest; {len(sel)} targets selected ({which})")
PY

# 2. Image, databases, host state
[ "$(docker image inspect "$IMAGE" --format '{{.Id}}' 2>/dev/null)" = "$IMAGE_ID" ] || die "image $IMAGE is not $IMAGE_ID"
FOUND=""
for f in /data/provenance/db_sha256_*.txt; do
  [ "$(sha256sum "$f" | cut -d' ' -f1)" = "$DB_FINGERPRINT" ] && FOUND="$f"
done
[ -n "$FOUND" ] || die "no database manifest with fingerprint $DB_FINGERPRINT in /data/provenance"
[ -d "$DB_DIR" ] || die "$DB_DIR not found"
for t in apt-daily.timer apt-daily-upgrade.timer; do
  [ "$(systemctl is-enabled "$t" 2>/dev/null)" = "disabled" ] || echo "WARNING: $t is not disabled"
done
MEM_TOTAL=$(free -m | awk '/^Mem:/{print $2}')

{
  echo "time_utc=$(now)"
  echo "host=$HOST kernel=$(uname -r) os=$(lsb_release -ds 2>/dev/null)"
  echo "vcpu=$(nproc) mem_total_mib=$MEM_TOTAL"
  echo "cpu=$(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2- | sed 's/^ //')"
  echo "docker=$(docker version --format '{{.Server.Version}}')"
  echo "image=$IMAGE image_id=$IMAGE_ID"
  echo "db_dir=$DB_DIR db_manifest=$FOUND db_fingerprint=$DB_FINGERPRINT"
  echo "inputs=$TARBALL sha256=$TARBALL_SHA manifest_sha256=$MANIFEST_SHA"
  echo "run=$RUN set=$SET jackhmmer_n_cpu=$NCPU"
  echo "flags=--norun_inference --jackhmmer_n_cpu=$NCPU (all others default)"
  echo "script_sha256=$(sha256sum "$0" | cut -d' ' -f1)"
} >> "$OUT/provenance.txt"

# 3. Targets, one at a time
DONE_NOW=0
TOTAL=$(wc -l < "$OUT/targets.txt")
i=0
while read -r PDB; do
  i=$((i + 1))
  if awk -F, -v p="$PDB" '$1 == p && $10 == "0"' "$CSV" | grep -q .; then
    echo "[$i/$TOTAL] $PDB already done, skipped"; continue
  fi
  if [ "$LIMIT" -gt 0 ] && [ "$DONE_NOW" -ge "$LIMIT" ]; then
    echo "LIMIT=$LIMIT reached, stopping (re-run the same command to continue)"; break
  fi
  T="$OUT/output/$PDB"
  if [ -e "$T" ]; then mv "$T" "$OUT/failed/${PDB}_$(date -u +%Y%m%dT%H%M%SZ)"; fi
  mkdir -p "$T"
  LOG="$OUT/logs/$PDB.log"; MEM="$OUT/logs/$PDB.mem"
  BASE=$(free -m | awk '/^Mem:/{print $3}')
  ( while sleep 5; do free -m | awk '/^Mem:/{print $3}'; done ) > "$MEM" &
  SAMPLER=$!
  START=$(now); S=$(date +%s)
  echo "[$i/$TOTAL] $PDB started $START"
  docker run --rm --name "af3_${RUN}_${PDB}" \
    --volume "$INPUTS":/root/af_input:ro \
    --volume "$T":/root/af_output \
    --volume "$DB_DIR":/root/public_databases:ro \
    "$IMAGE" \
    python run_alphafold.py \
      --json_path="/root/af_input/$PDB.json" \
      --output_dir=/root/af_output \
      --db_dir=/root/public_databases \
      --norun_inference \
      --jackhmmer_n_cpu="$NCPU" \
    > "$LOG" 2>&1
  RC=$?
  E=$(date +%s); END=$(now)
  kill "$SAMPLER" 2>/dev/null; wait "$SAMPLER" 2>/dev/null
  PEAK=$( (cat "$MEM"; echo "$BASE") | sort -n | tail -n 1)
  DJ=$(find "$T" -name '*_data.json' | head -n 1)
  STATS=",,,,"
  if [ "$RC" = "0" ] && [ -n "$DJ" ]; then
    STATS=$(python3 - "$DJ" <<'PY'
import hashlib, json, sys
f = sys.argv[1]
d = json.load(open(f))
n = lambda m: sum(1 for l in (m or "").splitlines() if l.startswith(">"))
p = [s["protein"] for s in d["sequences"] if "protein" in s][0]
print(",".join([f, hashlib.sha256(open(f, "rb").read()).hexdigest(),
                str(n(p.get("unpairedMsa"))), str(n(p.get("pairedMsa"))), str(len(p.get("templates") or []))]))
PY
)
  else
    [ "$RC" = "0" ] && RC=99
  fi
  echo "$PDB,$HOST,$RUN,$NCPU,$START,$END,$((E - S)),$BASE,$PEAK,$RC,$STATS" >> "$CSV"
  echo "[$i/$TOTAL] $PDB exit $RC, $(( (E - S) / 60 )) min $(( (E - S) % 60 )) s, VM memory peak $PEAK of $MEM_TOTAL MiB"
  if [ "$RC" != "0" ]; then
    tail -n 20 "$LOG"; die "$PDB failed (exit $RC), log $LOG"
  fi
  [ $((PEAK * 100)) -le $((MEM_TOTAL * 85)) ] || die "$PDB used more than 85% of the VM memory ($PEAK of $MEM_TOTAL MiB)"
  DONE_NOW=$((DONE_NOW + 1))
done < "$OUT/targets.txt"

# 4. Summary and output manifest when every target is done
NDONE=$(awk -F, 'NR > 1 && $10 == "0"' "$CSV" | cut -d, -f1 | sort -u | wc -l)
echo "--- $OUT: $NDONE of $TOTAL targets done"
if [ "$NDONE" = "$TOTAL" ]; then
  (cd "$OUT" && find output -type f | LC_ALL=C sort | xargs -d '\n' sha256sum) > "$OUT/outputs_sha256.txt"
  echo "all targets done: $(now); output manifest $OUT/outputs_sha256.txt, SHA-256 $(sha256sum "$OUT/outputs_sha256.txt" | cut -d' ' -f1)" | tee -a "$OUT/provenance.txt"
fi
