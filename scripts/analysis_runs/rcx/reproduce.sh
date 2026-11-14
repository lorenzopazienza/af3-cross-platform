#!/bin/bash
# Follow-up 15, recompilation of Af, Hf (part A4) and GT (part A3) (label rcx; analysis/rcx.py)
# (notes/analysis_plan.md section 10, row of 2026-11-14): the per-sample table of the sessions listed in
# scripts/analysis_runs/rcx/sessions.txt (written once the parts have finished, from the section 10 launch rows, and
# committed before any analysis run), with M1 (cached by model SHA-256), then the analysis. The r1 sides and the
# references are read from their analysis folders (results/analysis/fu14_20261113b, main_20261014/primary_v2,
# h100_20261104). Before it: scripts/analysis_runs/rcx/fetch_full_outputs.py (read-only copies from the results bucket
# of the full outputs of Af r1 and Hf r1 that the strict check compares byte for byte); this script stops if any is
# missing. The tests run first. Run from the repository root:
#   DATE=YYYYMMDD bash scripts/analysis_runs/rcx/reproduce.sh
set -eu
O=results/analysis/rcx_${DATE:?set DATE=YYYYMMDD, the analysis date (the output folder is rcx_DATE)}
S=scripts/analysis_runs/rcx/sessions.txt
[ -f "$S" ] || { echo "!! $S does not exist: list the rcx sessions of the section 10 rows there first" >&2; exit 2; }
grep -qv '^[[:space:]]*\(#\|$\)' "$S" || { echo "!! $S lists no session yet: list the rcx sessions of the section 10 rows there first" >&2; exit 2; }
if [ -e "$O/samples.csv" ]; then echo "$O/samples.csv exists; the output folder must be new" >&2; exit 1; fi
python3 scripts/analysis_runs/rcx/fetch_full_outputs.py --check || exit 2
mkdir -p "$O"
{
analysis/.venv/bin/python -m pytest -q -p no:cacheprovider analysis/tests/test_rcx.py
analysis/.venv/bin/python analysis/collect.py --labels rcx --sessions "$S" --out "$O" --jobs 8
analysis/.venv/bin/python analysis/rcx.py --samples "$O/samples.csv" --out "$O" --replicates 10000 --seed 2026
} 2>&1 | tee "$O/run.log"
