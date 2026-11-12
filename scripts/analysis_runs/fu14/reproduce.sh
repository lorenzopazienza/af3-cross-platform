#!/bin/bash
# Follow-up 14, the float32 trunk on every accelerator (label f32; analysis/fu14.py)
# (notes/analysis_plan.md section 10, row of 2026-11-12): the per-sample table of the sessions listed in
# scripts/analysis_runs/fu14/sessions.txt (written once the arms have finished, from the section 10 launch rows, and
# committed before any analysis run), with M1 (cached by model SHA-256), then the analysis. Follow-up 13's V6f and Gf
# and the bfloat16 references are read from their analysis folders (results/analysis/fu13_20261112, main_20261014,
# a100_20261018, h100_20261104, prec5_20261107). Before it: scripts/analysis_runs/fu14/fetch_full_outputs.py
# (read-only copies from the results bucket of the full outputs check 5 and item b2 compare byte for byte); this
# script stops if any is missing. The tests run first. Run from the repository root:
#   DATE=YYYYMMDD bash scripts/analysis_runs/fu14/reproduce.sh
set -eu
O=results/analysis/fu14_${DATE:?set DATE=YYYYMMDD, the analysis date (the output folder is fu14_DATE)}
S=scripts/analysis_runs/fu14/sessions.txt
[ -f "$S" ] || { echo "!! $S does not exist: list the f32 sessions of the section 10 rows there first" >&2; exit 2; }
grep -qv '^[[:space:]]*\(#\|$\)' "$S" || { echo "!! $S lists no session yet: list the f32 sessions of the section 10 rows there first" >&2; exit 2; }
if [ -e "$O/samples.csv" ]; then echo "$O/samples.csv exists; the output folder must be new" >&2; exit 1; fi
python3 scripts/analysis_runs/fu14/fetch_full_outputs.py --check || exit 2
mkdir -p "$O"
{
analysis/.venv/bin/python -m pytest -q -p no:cacheprovider analysis/tests/test_fu14.py
analysis/.venv/bin/python analysis/collect.py --labels f32 --sessions "$S" --out "$O" --jobs 8
analysis/.venv/bin/python analysis/fu14.py --samples "$O/samples.csv" --out "$O" --replicates 10000 --seed 2026
} 2>&1 | tee "$O/run.log"
