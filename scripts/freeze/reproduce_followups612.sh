#!/bin/bash
# Data freeze extended to follow-ups 6 to 12 (notes/analysis_plan.md section 10, the freeze row drafted in
# drafts/section10_freeze_fu6_12.md): the sessions of freeze a4fb532a (results/freeze/feat_20261018/, unchanged) and the
# 135 sessions of scripts/analysis_runs/{msa,sw,seeds,h100,repl,prec5,large}/sessions.txt, in a folder of its own. The
# follow-up 6 NAS deliveries keep their own freezes and are listed in its summary.md. Read-only bucket listing; appended
# logs of the follow-up sessions are downloaded to a temporary file for the prefix check. Then results/freeze/INDEX.md.
# Run from the repository root:
#   DATE=YYYYMMDD bash scripts/freeze/reproduce_followups612.sh
set -eu
set -o pipefail
O=results/freeze/followups612_${DATE:?set DATE=YYYYMMDD, the freeze date (the output folder is followups612_DATE)}
[ ! -e "$O" ] || { echo "!! $O exists: a freeze is never overwritten" >&2; exit 2; }
mkdir -p "$O"
analysis/.venv/bin/python scripts/freeze/make_freeze_manifest.py --bucket gs://af2-tpu-benchmark-af3-results \
  --followups612 --out "$O" 2>&1 | tee "$O/run.log"
analysis/.venv/bin/python scripts/freeze/freeze_index.py
