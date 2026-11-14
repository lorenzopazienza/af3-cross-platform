#!/bin/bash
# Returned structures across platforms (label returned; analysis/returned_structures.py), section 10 row of
# drafts/section10_returned_spec_20261114.md: a post hoc analysis of frozen outputs asked by the reviewers, descriptive.
# Reads the per-sample tables and seed pairs of the main study and follow-ups 4, 13 and 14 (no session is collected);
# runs the synthetic tests first, then the analysis. Run from the repository root, after the row is committed:
#   DATE=YYYYMMDD bash scripts/analysis_runs/returned/reproduce.sh
set -eu
set -o pipefail
O=results/analysis/returned_${DATE:?set DATE=YYYYMMDD, the analysis date (the output folder is returned_DATE)}
if [ -e "$O/quality.csv" ]; then echo "$O/quality.csv exists; the output folder must be new" >&2; exit 1; fi
mkdir -p "$O"
git rev-parse HEAD > "$O/analysis_commit.txt"
{
analysis/.venv/bin/python -m pytest -q -p no:cacheprovider analysis/tests/test_returned_structures.py
analysis/.venv/bin/python analysis/returned_structures.py --main . --out "$O" --replicates 10000 --seed 2026
} 2>&1 | tee "$O/run.log"
