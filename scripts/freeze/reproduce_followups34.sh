#!/bin/bash
# Data freeze extended to follow-ups 3 and 4 (notes/analysis_plan.md section 10, rows of 2026-10-17 and 2026-10-18):
# the sessions of freeze 14e1fc4c and the 66 cpu60 and a100 sessions, in a folder of its own next to results/freeze/
# (whose manifest and identifier stay unchanged). Read-only bucket listing; appended logs of the new sessions are
# downloaded to a temporary file for the prefix check. Run from the repository root:
#   bash scripts/freeze/reproduce_followups34.sh
set -eu
O=results/freeze/followups34_20261018
mkdir -p "$O"
analysis/.venv/bin/python scripts/freeze/make_freeze_manifest.py --bucket gs://af2-tpu-benchmark-af3-results \
  --followups34 --out "$O" 2>&1 | tee "$O/run.log"
