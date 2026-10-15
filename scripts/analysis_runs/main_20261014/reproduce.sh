#!/bin/bash
# Main study, pre-specified analysis (notes/analysis_plan.md sections 4, 5, 7, 11 and the section 10 rows),
# after checks 1 and 2 passed on all 53 parts (results/analysis/main_monitor_20261013/check_final.txt,
# strict_summary_all.txt). Run from the repository root: bash scripts/analysis_runs/main_20261014/reproduce.sh
set -eu
O=results/analysis/main_20261014
S=scripts/analysis_runs/main_20261014   # the scripts; every output goes under $O
ROOTS=""
while read -r s; do ROOTS="$ROOTS --root results/af3/$s"; done < "$O/sessions_main.txt"
analysis/.venv/bin/python analysis/collect.py $ROOTS --labels main --out "$O" --jobs 8          # M1 (cached by model SHA-256)
analysis/.venv/bin/python analysis/contrasts.py --samples "$O/samples.csv" --labels main --out "$O/contrasts" \
  --replicates 10000 --seed 2026
analysis/.venv/bin/python analysis/cost.py --samples "$O/samples.csv" --labels main --out "$O/cost"
analysis/.venv/bin/python "$S/decision_inputs.py" "$O"
