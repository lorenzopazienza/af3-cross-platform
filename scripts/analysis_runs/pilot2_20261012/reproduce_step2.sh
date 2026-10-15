#!/bin/bash
# Pilot2, Phase 2 Step 2 (2026-10-12), after every Step 1 check passed (reproduce_step1.sh).
# Run from the repository root: bash scripts/analysis_runs/pilot2_20261012/reproduce_step2.sh
set -eu
O=results/analysis/pilot2_20261012
S=scripts/analysis_runs/pilot2_20261012   # the scripts; every output goes under $O
ROOTS=""
while read -r s; do ROOTS="$ROOTS --root results/af3/$s"; done < "$O/sessions_pilot2.txt"
analysis/.venv/bin/python analysis/collect.py $ROOTS --labels pilot2 --out "$O/step2" --jobs 6     # M1 (cached by model SHA-256)
analysis/.venv/bin/python analysis/contrasts.py --samples "$O/step2/samples.csv" --labels pilot2 --out "$O/step2/contrasts" \
  --replicates 10000 --seed 2026
analysis/.venv/bin/python analysis/cost.py --samples "$O/step2/samples.csv" --labels pilot2 --out "$O/step2/cost"
analysis/.venv/bin/python "$S/decision_inputs.py" "$O/step2"
