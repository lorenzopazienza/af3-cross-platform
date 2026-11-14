#!/bin/bash
# Rounded ranking score against the full-precision score in the main study (label rounded; analysis/rounded_score.py),
# section 10 row of drafts/section10_rounded_score_20261114.md: a post hoc, descriptive count of frozen outputs. Reads
# the main study's per-sample table (no session is collected); runs the synthetic tests first, then the count. Run from
# the repository root, after the row is committed:
#   DATE=YYYYMMDD bash scripts/analysis_runs/rounded/reproduce.sh
set -eu
set -o pipefail
O=results/analysis/rounded_${DATE:?set DATE=YYYYMMDD, the analysis date (the output folder is rounded_DATE)}
if [ -e "$O/rounded.csv" ]; then echo "$O/rounded.csv exists; the output folder must be new" >&2; exit 1; fi
mkdir -p "$O"
git rev-parse HEAD > "$O/analysis_commit.txt"
{
analysis/.venv/bin/python -m pytest -q -p no:cacheprovider analysis/tests/test_rounded_score.py
analysis/.venv/bin/python analysis/rounded_score.py --main . --out "$O"
} 2>&1 | tee "$O/run.log"
