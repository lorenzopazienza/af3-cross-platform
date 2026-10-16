#!/bin/bash
# Follow-up 3, the A100 arms (notes/analysis_plan.md section 10, rows of 2026-10-16; label a100): the per-sample
# table of the a100 sessions with M1 (cached by model SHA-256), then the analysis: checks 1 and 2 first (it stops if
# one fails), every contrast of the row, Q3, M6 at both regions' a2-ultragpu-1g prices, M7. With CPU60_DATE, C also
# on the follow-up 4 targets (HW(A,C) and HWh(A,C) on 60 targets). Run from the repository root:
#   DATE=YYYYMMDD [CPU60_DATE=YYYYMMDD] bash scripts/analysis_runs/a100/reproduce.sh
set -eu
O=results/analysis/a100_${DATE:?set DATE=YYYYMMDD, the analysis date (the output folder is a100_DATE)}
CPU60=""
[ -z "${CPU60_DATE:-}" ] || CPU60="--cpu60 results/analysis/cpu60_$CPU60_DATE/samples.csv"
mkdir -p "$O"
{
analysis/.venv/bin/python analysis/collect.py --labels a100 --out "$O" --jobs 8
analysis/.venv/bin/python analysis/a100.py --samples "$O/samples.csv" --out "$O" $CPU60 \
  --main results/analysis/main_20261014/primary_v2 --replicates 10000 --seed 2026
} 2>&1 | tee "$O/run.log"
