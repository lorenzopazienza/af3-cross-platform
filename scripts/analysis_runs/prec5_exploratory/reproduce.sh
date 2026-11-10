#!/bin/bash
# Exploratory analysis (c) on follow-up 12's five seeds (label prec5_exploratory; analysis/prec5_exploratory.py),
# notes/analysis_plan.md section 10, row of 2026-11-10 "Two computations on frozen outputs for the paper", part (1):
# analysis/exploratory.py's function perturbation, unchanged, on PR(V6), PR(G) and HWh(V6,G), seeds 1 to 5. Reads
# results/analysis/prec5_20261107/pairs_seed.csv and pairs_sample.csv and writes nothing there; no session is
# collected. The synthetic tests run first. Run from the repository root:
#   DATE=YYYYMMDD bash scripts/analysis_runs/prec5_exploratory/reproduce.sh
set -eu
O=results/analysis/prec5_exploratory_${DATE:?set DATE=YYYYMMDD, the analysis date (the output folder is prec5_exploratory_DATE)}
if [ -e "$O/exploratory_c.csv" ]; then echo "$O/exploratory_c.csv exists; the output folder must be new" >&2; exit 1; fi
mkdir -p "$O"
{
analysis/.venv/bin/python -m pytest -q analysis/tests/test_prec5_exploratory.py
analysis/.venv/bin/python analysis/prec5_exploratory.py --main . --out "$O" --replicates 10000 --seed 2026
} 2>&1 | tee "$O/run.log"
