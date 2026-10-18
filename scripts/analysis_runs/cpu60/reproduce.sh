#!/bin/bash
# Follow-up 4, the CPU arm on every target (notes/analysis_plan.md section 10, row of 2026-10-16; label cpu60): the
# per-sample table of the cpu60 sessions listed in sessions.txt (preempted sessions are not) with M1 (cached by model
# SHA-256), then the analysis on the 60 targets.
# Run from the repository root, once every cpu60 part has finished, with the analysis date:
#   DATE=YYYYMMDD bash scripts/analysis_runs/cpu60/reproduce.sh
set -eu
O=results/analysis/cpu60_${DATE:?set DATE=YYYYMMDD, the analysis date (the output folder is cpu60_DATE)}
mkdir -p "$O"
{
analysis/.venv/bin/python analysis/collect.py --labels cpu60 --sessions scripts/analysis_runs/cpu60/sessions.txt --out "$O" --jobs 8
analysis/.venv/bin/python analysis/cpu60.py --samples "$O/samples.csv" --out "$O" \
  --main results/analysis/main_20261014/primary_v2 --replicates 10000 --seed 2026
} 2>&1 | tee "$O/run.log"
