#!/bin/bash
# Pilot2, Phase 2 Step 1 (all arms, 2026-10-12): checks 1 and 2 (no M1), then check 3 (M1 on V6 r1 only).
# Run from the repository root: bash scripts/analysis_runs/pilot2_20261012/reproduce_step1.sh
set -eu
O=results/analysis/pilot2_20261012
S=scripts/analysis_runs/pilot2_20261012   # the scripts; every output goes under $O
ROOTS=""
while read -r s; do ROOTS="$ROOTS --root results/af3/$s"; done < "$O/sessions_pilot2.txt"
analysis/.venv/bin/python analysis/collect.py $ROOTS --labels pilot2 --out "$O/step1" --no_m1
analysis/.venv/bin/python "$S/step1_checks.py" "$O/step1"
# bash "$S/fetch_check2_files.sh"        # read-only bucket copies (full confidences, data.json) of r1 seed 1 and w
analysis/.venv/bin/python "$S/step1_checks.py" "$O/step1" --bit_identity
bash "$S/crosscheck_compare_runs.sh" "${TMPDIR:-/tmp}/pilot2_crosscheck"
# Check 3: M1 for the V6 session only, then the overall top-1 success on V6 r1.
analysis/.venv/bin/python analysis/collect.py --root results/af3/20261012T115650Z_v6e_pilot --labels pilot2 --out "$O/step1_v6_m1"
analysis/.venv/bin/python "$S/check3.py" "$O/step1_v6_m1"
