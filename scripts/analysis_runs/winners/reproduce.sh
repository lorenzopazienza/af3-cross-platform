#!/bin/bash
# Returned poses compared directly across platforms (label winners; analysis/winners.py), section 10 row of
# drafts/section10_winners_spec_20261114.md: a post hoc analysis of frozen outputs asked by the reviewers, descriptive.
# Reads the per-sample tables and seed pairs of the main study and follow-ups 3 and 9, and the sample mmCIF files under
# results/af3/ (all local; no session is collected); runs the synthetic tests first, then the analysis. Run from the
# repository root, after the row is committed:
#   DATE=YYYYMMDD bash scripts/analysis_runs/winners/reproduce.sh
set -eu
set -o pipefail
O=results/analysis/winners_${DATE:?set DATE=YYYYMMDD, the analysis date (the output folder is winners_DATE)}
if [ -e "$O/summary.csv" ]; then echo "$O/summary.csv exists; the output folder must be new" >&2; exit 1; fi
mkdir -p "$O"
git rev-parse HEAD > "$O/analysis_commit.txt"
{
analysis/.venv/bin/python -m pytest -q -p no:cacheprovider analysis/tests/test_winners.py
analysis/.venv/bin/python analysis/winners.py --main . --out "$O" --replicates 10000 --seed 2026 --jobs 8
} 2>&1 | tee "$O/run.log"
