#!/bin/bash
# Ranking rules and their cost (label ranking; analysis/ranking_rules.py), section 10 row C1: a follow-up analysis of
# frozen outputs, outside the pre-registered study, descriptive; part (iv) exploratory. Reads the per-sample tables of
# the main study, follow-ups 8, 9 and 3 and the main study's cost summary (no session is collected); runs the
# synthetic tests first, then the analysis. Run from the repository root:
#   DATE=YYYYMMDD bash scripts/analysis_runs/ranking/reproduce.sh
set -eu
O=results/analysis/ranking_${DATE:?set DATE=YYYYMMDD, the analysis date (the output folder is ranking_DATE)}
mkdir -p "$O"
{
analysis/.venv/bin/python -m pytest -q analysis/tests/test_ranking_rules.py
analysis/.venv/bin/python analysis/ranking_rules.py --main . --out "$O" --replicates 10000 --seed 2026
} 2>&1 | tee "$O/run.log"
