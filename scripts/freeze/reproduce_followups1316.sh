#!/bin/bash
# Data freeze extended to follow-ups 13 to 16, the memory limit on A and H and the warm cost on GT and H
# (notes/analysis_plan.md section 10, the freeze row drafted in drafts/section10_freeze_fu13_16.md): the sessions of
# freeze 68923012 (results/freeze/followups612_20261108/, unchanged) and the 144 sessions of
# scripts/analysis_runs/{fu13,fu14,rcx,hlo,memlimit_ah,warm_th}/sessions.txt, in a folder of its own. The local session
# folders of those labels that no list holds are listed in its summary.md with their section 10 rows, and not frozen.
# The follow-up 6 NAS deliveries keep their own freezes and are listed in its summary.md. Read-only bucket listing;
# appended logs of the follow-up sessions are downloaded to a temporary file for the prefix check. Then
# results/freeze/INDEX.md. Run from the repository root:
#   DATE=YYYYMMDD bash scripts/freeze/reproduce_followups1316.sh
set -eu
set -o pipefail
O=results/freeze/followups1316_${DATE:?set DATE=YYYYMMDD, the freeze date (the output folder is followups1316_DATE)}
[ ! -e "$O" ] || { echo "!! $O exists: a freeze is never overwritten" >&2; exit 2; }
mkdir -p "$O"
analysis/.venv/bin/python scripts/freeze/make_freeze_manifest.py --bucket gs://af2-tpu-benchmark-af3-results \
  --followups1316 --out "$O" 2>&1 | tee "$O/run.log"
analysis/.venv/bin/python scripts/freeze/freeze_index.py
