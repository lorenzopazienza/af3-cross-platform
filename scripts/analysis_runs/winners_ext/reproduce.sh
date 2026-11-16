#!/bin/bash
# Returned poses compared directly, further contrasts (label winners_ext; analysis/winners_ext.py, which reuses
# analysis/winners.py unchanged), section 10 row of drafts/section10_winners_ext_spec_20261116.md: a post hoc analysis
# of frozen outputs asked by the reviewers, descriptive. HW(G,C) of the main study and of follow-up 4, and DU (G against
# GT), with RC(GT) of follow-up 15 as its reference. Reads the per-sample tables and seed pairs of the main study and
# follow-ups 4 and 15, and the sample mmCIF files under
# results/af3/ (all local; no session is collected); runs the synthetic tests first, then the analysis. Run from the
# repository root, after the row is committed:
#   DATE=YYYYMMDD bash scripts/analysis_runs/winners_ext/reproduce.sh
set -eu
set -o pipefail
O=results/analysis/winners_ext_${DATE:?set DATE=YYYYMMDD, the analysis date (the output folder is winners_ext_DATE)}
if [ -e "$O/summary.csv" ]; then echo "$O/summary.csv exists; the output folder must be new" >&2; exit 1; fi
mkdir -p "$O"
git rev-parse HEAD > "$O/analysis_commit.txt"
{
analysis/.venv/bin/python -m pytest -q -p no:cacheprovider analysis/tests/test_winners_ext.py analysis/tests/test_winners.py
analysis/.venv/bin/python analysis/winners_ext.py --main . --out "$O" --replicates 10000 --seed 2026 --jobs 8
} 2>&1 | tee "$O/run.log"
