#!/bin/bash
# Data freeze extended to follow-up 5 (notes/analysis_plan.md section 10, row "Follow-up 5, launch" of 2026-10-18):
# the sessions of freeze fc297328 (results/freeze/followups34_20261018/, unchanged) and the 4 feat sessions, in a
# folder of its own. Read-only bucket listing; appended logs of the follow-up sessions are downloaded to a temporary
# file for the prefix check. Run from the repository root:
#   bash scripts/freeze/reproduce_feat.sh
set -eu
O=results/freeze/feat_20261018
mkdir -p "$O"
analysis/.venv/bin/python scripts/freeze/make_freeze_manifest.py --bucket gs://af2-tpu-benchmark-af3-results \
  --feat --out "$O" 2>&1 | tee "$O/run.log"
