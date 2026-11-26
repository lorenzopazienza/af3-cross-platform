#!/bin/bash
# Follow-up 3, the reading of section 11 for PR(A), HWh(V6,A) and HWh(A,C) (analysis/fu03_a6.py; point A6 of the
# review of 2026-11-26), computed from frozen outputs: the registered seed pairs and rows of follow-up 3 and of the
# main study, the per-sample tables of follow-ups 3 and 4 and of the main study, and the supplement table T22b (all
# local; no structure file is read and no session is collected). Runs the synthetic tests first, then the analysis.
# Run from the repository root (or from a worktree whose results/ links to the main checkout's):
#   DATE=YYYYMMDD bash scripts/analysis_runs/fu03_a6/reproduce.sh
set -eu
set -o pipefail
O=results/analysis/fu03_a6_${DATE:?set DATE=YYYYMMDD, the analysis date (the output folder is fu03_a6_DATE)}
if [ -e "$O/results.csv" ]; then echo "$O/results.csv exists; the output folder must be new" >&2; exit 1; fi
mkdir -p "$O"
git rev-parse HEAD > "$O/analysis_commit.txt"
{
analysis/.venv/bin/python -m pytest -q -p no:cacheprovider analysis/tests/test_fu03_a6.py
analysis/.venv/bin/python analysis/fu03_a6.py --main . --out "$O" --replicates 10000 --seed 2026
} 2>&1 | tee "$O/run.log"
