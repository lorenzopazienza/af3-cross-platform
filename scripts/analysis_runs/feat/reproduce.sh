#!/bin/bash
# Follow-up 5, host-side featurisation (notes/analysis_plan.md section 10, row of 2026-10-18; label feat): the
# feature hashes of the four host families compared (analysis/feat.py: checks first, then every (target, seed,
# feature), then the row's decision rule). scripts/analysis_runs/feat/sessions.txt names the four sessions the
# section 10 rows list, one per line, written once every feat part has finished. Run from the repository root:
#   DATE=YYYYMMDD bash scripts/analysis_runs/feat/reproduce.sh
set -eu
O=results/analysis/feat_${DATE:?set DATE=YYYYMMDD, the analysis date (the output folder is feat_DATE)}
S=scripts/analysis_runs/feat/sessions.txt
[ -f "$S" ] || { echo "!! $S does not exist: list the four feat sessions of the section 10 rows there first" >&2; exit 2; }
mkdir -p "$O"
analysis/.venv/bin/python analysis/feat.py --sessions "$S" --out "$O" 2>&1 | tee "$O/run.log"
