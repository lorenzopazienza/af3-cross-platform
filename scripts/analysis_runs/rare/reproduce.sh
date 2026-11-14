#!/bin/bash
# Rare-event intervals (label rare; analysis/rare_intervals.py), section 10 row of
# drafts/section10_rare_spec_20261114.md: a post hoc sensitivity analysis of registered readings asked by the
# reviewers; the registered readings stay. Reads the seed pairs and the registered interval tables of the main study
# and follow-ups 3, 4, 9, 11, 12 and 14 (no session is collected); runs the synthetic tests first, then the analysis.
# Run from the repository root, after the row is committed:
#   DATE=YYYYMMDD bash scripts/analysis_runs/rare/reproduce.sh
set -eu
set -o pipefail
O=results/analysis/rare_${DATE:?set DATE=YYYYMMDD, the analysis date (the output folder is rare_DATE)}
if [ -e "$O/intervals.csv" ]; then echo "$O/intervals.csv exists; the output folder must be new" >&2; exit 1; fi
mkdir -p "$O"
git rev-parse HEAD > "$O/analysis_commit.txt"
{
analysis/.venv/bin/python -m pytest -q -p no:cacheprovider analysis/tests/test_rare_intervals.py
analysis/.venv/bin/python analysis/rare_intervals.py --main . --out "$O" --replicates 10000 --seed 2026
} 2>&1 | tee "$O/run.log"
