#!/bin/bash
# Exploratory analysis of why top-1 flips occur (notes/analysis_plan.md section 10, row of 2026-10-14), on the
# primary analysis after the M4 correction (../primary_v2). Run from the repository root:
#   bash scripts/analysis_runs/main_20261014/exploratory/reproduce.sh
set -eu
O=results/analysis/main_20261014/exploratory
analysis/.venv/bin/python analysis/exploratory.py --main results/analysis/main_20261014 --out "$O" \
  --replicates 10000 --seed 2026 2>&1 | tee "$O/run.log"
