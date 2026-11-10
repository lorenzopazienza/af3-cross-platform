#!/bin/bash
# Follow-up 13, why the top-ranked sample flips (label why; analysis/fu13.py, analysis/fu13_figures.py)
# (notes/analysis_plan.md section 10, row of 2026-11-10): the per-sample table of the sessions listed in
# scripts/analysis_runs/fu13/sessions.txt (written once the arms have finished, from the section 10 launch rows, and
# committed before any analysis run), with M1 (cached by model SHA-256), then the analysis and its two figures. The
# capture files of L are read from the session folders (results/af3/<session>/<run>/<target>/fu13_capture_seed-1.npz).
# Run from the repository root:
#   DATE=YYYYMMDD bash scripts/analysis_runs/fu13/reproduce.sh
set -eu
O=results/analysis/fu13_${DATE:?set DATE=YYYYMMDD, the analysis date (the output folder is fu13_DATE)}
S=scripts/analysis_runs/fu13/sessions.txt
[ -f "$S" ] || { echo "!! $S does not exist: list the why sessions of the section 10 rows there first" >&2; exit 2; }
grep -qv '^[[:space:]]*\(#\|$\)' "$S" || { echo "!! $S lists no session yet: list the why sessions of the section 10 rows there first" >&2; exit 2; }
mkdir -p "$O"
{
analysis/.venv/bin/python analysis/collect.py --labels why --sessions "$S" --out "$O" --jobs 8
analysis/.venv/bin/python analysis/fu13.py --samples "$O/samples.csv" --out "$O" --replicates 10000 --seed 2026
analysis/figures/.venv/bin/python analysis/fu13_figures.py --dir "$O"
} 2>&1 | tee "$O/run.log"
