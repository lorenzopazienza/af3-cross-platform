#!/bin/bash
# Pilot2, Phase 2 Step 1, part a (2026-10-12): run-record table of the completed arms, without M1.
# Run from the repository root: bash scripts/analysis_runs/pilot2_20261012/reproduce_step1a.sh
set -eu
O=results/analysis/pilot2_20261012
S=scripts/analysis_runs/pilot2_20261012   # the scripts; every output goes under $O
ROOTS=""
while read -r s; do ROOTS="$ROOTS --root results/af3/$s"; done < "$O/sessions_step1a.txt"
analysis/.venv/bin/python analysis/collect.py $ROOTS --labels pilot2 --out "$O/step1a" --no_m1
# Check 1, 2a and 2b from the run records; bit-identity only after they pass (needs fetch_check2_files.sh first).
analysis/.venv/bin/python "$S/step1_checks.py" "$O/step1a"
# bash "$S/fetch_check2_files.sh"        # read-only bucket copies, already made on 2026-10-12
analysis/.venv/bin/python "$S/step1_checks.py" "$O/step1a" --bit_identity
bash "$S/crosscheck_compare_runs.sh" "${TMPDIR:-/tmp}/pilot2_crosscheck"
