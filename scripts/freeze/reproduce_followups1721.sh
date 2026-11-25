#!/bin/bash
# Data freeze extended to follow-ups 17 to 21 (notes/analysis_plan.md section 10, the freeze row drafted on 2026-11-25):
# the 439 sessions of freeze 4ec185c5 (results/freeze/followups1316_20261117/, unchanged) and the 168 sessions of
# scripts/analysis_runs/{dtp,tri,rcf,aut,xvm}/sessions.txt, in a folder of its own. The local session folders of those
# labels that no list holds are listed in its summary.md with their section 10 rows, and not frozen. The follow-up 6 NAS
# deliveries keep their own freezes and are listed in its summary.md. Read-only bucket listing; appended logs of the
# follow-up sessions are downloaded to a temporary file for the prefix check. Then results/freeze/INDEX.md. Run from the
# repository root:
#   DATE=YYYYMMDD bash scripts/freeze/reproduce_followups1721.sh
set -eu
set -o pipefail
O=results/freeze/followups1721_${DATE:?set DATE=YYYYMMDD, the freeze date (the output folder is followups1721_DATE)}
[ ! -e "$O" ] || { echo "!! $O exists: a freeze is never overwritten" >&2; exit 2; }
mkdir -p "$O"
analysis/.venv/bin/python scripts/freeze/make_freeze_manifest.py --bucket gs://af2-tpu-benchmark-af3-results \
  --followups1721 --out "$O" 2>&1 | tee "$O/run.log"
analysis/.venv/bin/python scripts/freeze/freeze_index.py
