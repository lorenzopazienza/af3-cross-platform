#!/bin/bash
# Main study, primary analysis rerun after the section 10 row of 2026-10-14 "Correction to the implementation of M4"
# (analysis/score_ligand.pair_divergence computes the C-alpha RMSD independently of the ligand scoring). Same steps,
# options and seeds as ../reproduce.sh; outputs only in this folder. m1_cache.jsonl is a copy of ../m1_cache.jsonl
# (M1 by model SHA-256; M1's code is unchanged), so collect.py reuses it. Run from the repository root:
#   bash scripts/analysis_runs/main_20261014/primary_v2/reproduce.sh
set -eu
M=results/analysis/main_20261014
S=scripts/analysis_runs/main_20261014   # the scripts; every output goes under $M
O=$M/primary_v2
ROOTS=""
while read -r s; do ROOTS="$ROOTS --root results/af3/$s"; done < "$M/sessions_main.txt"
{
analysis/.venv/bin/python analysis/collect.py $ROOTS --labels main --out "$O" --jobs 8
analysis/.venv/bin/python analysis/contrasts.py --samples "$O/samples.csv" --labels main --out "$O/contrasts" \
  --replicates 10000 --seed 2026
analysis/.venv/bin/python analysis/cost.py --samples "$O/samples.csv" --labels main --out "$O/cost"
analysis/.venv/bin/python "$S/decision_inputs.py" "$O"
} 2>&1 | tee "$O/run.log"
analysis/.venv/bin/python "$S/primary_v2/compare_with_primary.py" | tee "$O/compare.txt"
