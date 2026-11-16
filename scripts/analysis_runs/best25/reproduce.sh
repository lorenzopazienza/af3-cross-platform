#!/bin/bash
# Success of the best of all samples per platform (label best25; analysis/best25.py), section 10 row of
# drafts/section10_best25_spec_20261116.md: a post hoc analysis of frozen outputs, descriptive. Reads the per-sample
# tables, seed pairs and Q3 tables of the main study and follow-ups 3, 4 and 9 (all local; no structure file is read and
# no session is collected); runs the synthetic tests first, then the analysis. Run from the repository root, after the
# row is committed:
#   DATE=YYYYMMDD bash scripts/analysis_runs/best25/reproduce.sh
set -eu
set -o pipefail
O=results/analysis/best25_${DATE:?set DATE=YYYYMMDD, the analysis date (the output folder is best25_DATE)}
if [ -e "$O/differences.csv" ]; then echo "$O/differences.csv exists; the output folder must be new" >&2; exit 1; fi
mkdir -p "$O"
git rev-parse HEAD > "$O/analysis_commit.txt"
{
analysis/.venv/bin/python -m pytest -q -p no:cacheprovider analysis/tests/test_best25.py
analysis/.venv/bin/python analysis/best25.py --main . --out "$O" --replicates 10000 --seed 2026
} 2>&1 | tee "$O/run.log"
