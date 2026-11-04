#!/bin/bash
# Follow-up 11, replication on new targets (V5, V6, G with label repl; analysis/repl.py)
# (notes/analysis_plan.md section 10, row of 2026-11-03): the per-sample table of the sessions listed in
# scripts/analysis_runs/repl/sessions.txt (written once the arms have finished, from the section 10 launch rows, and
# committed before any analysis run), with M1 (cached by model SHA-256), then the analysis. Run from the repository root:
#   DATE=YYYYMMDD bash scripts/analysis_runs/repl/reproduce.sh
set -eu
O=results/analysis/repl_${DATE:?set DATE=YYYYMMDD, the analysis date (the output folder is repl_DATE)}
S=scripts/analysis_runs/repl/sessions.txt
[ -f "$S" ] || { echo "!! $S does not exist: list the repl sessions of the section 10 rows there first" >&2; exit 2; }
grep -qv '^[[:space:]]*\(#\|$\)' "$S" || { echo "!! $S lists no session yet: list the repl sessions of the section 10 rows there first" >&2; exit 2; }
mkdir -p "$O"
{
analysis/.venv/bin/python analysis/collect.py --labels repl --sessions "$S" --out "$O" --jobs 8
analysis/.venv/bin/python analysis/repl.py --samples "$O/samples.csv" --out "$O" --replicates 10000 --seed 2026
} 2>&1 | tee "$O/run.log"
