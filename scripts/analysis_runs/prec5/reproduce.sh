#!/bin/bash
# Follow-up 12, the precision-matched arms to five seeds (V6h and Gh, seeds 3 to 5, label prec5; analysis/prec5.py)
# (notes/analysis_plan.md section 10, row of 2026-11-03): the per-sample table of the sessions listed in
# scripts/analysis_runs/prec5/sessions.txt (written once the arms have finished, from the section 10 launch rows, and
# committed before any analysis run), with M1 (cached by model SHA-256), then the analysis. Run from the repository root:
#   DATE=YYYYMMDD bash scripts/analysis_runs/prec5/reproduce.sh
set -eu
O=results/analysis/prec5_${DATE:?set DATE=YYYYMMDD, the analysis date (the output folder is prec5_DATE)}
S=scripts/analysis_runs/prec5/sessions.txt
[ -f "$S" ] || { echo "!! $S does not exist: list the prec5 sessions of the section 10 rows there first" >&2; exit 2; }
grep -qv '^[[:space:]]*\(#\|$\)' "$S" || { echo "!! $S lists no session yet: list the prec5 sessions of the section 10 rows there first" >&2; exit 2; }
mkdir -p "$O"
{
analysis/.venv/bin/python analysis/collect.py --labels prec5 --sessions "$S" --out "$O" --jobs 8
analysis/.venv/bin/python analysis/prec5.py --samples "$O/samples.csv" --out "$O" --replicates 10000 --seed 2026
} 2>&1 | tee "$O/run.log"
