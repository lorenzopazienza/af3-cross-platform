#!/bin/bash
# Follow-up 20 (the L4 without autotuning; label aut; analysis/aut.py) (notes/analysis_plan.md section 10, row
# "Follow-up experiment 20"): the per-sample table of the sessions listed in scripts/analysis_runs/aut/sessions.txt
# (written once the parts have finished, by check_sessions.py --write, from the section 10 launch rows, and committed
# before any analysis run), with M1 (cached by model SHA-256), then the analysis. The references are read from their analysis folder
# (results/analysis/main_20261014/primary_v2: G r1 and r2, V6 r1, RC(G), HW(V6,G), HW(V6,Gd)). Every
# file compared is in the light fetch: no full outputs are fetched. The tests run first. Run from the repository root:
#   DATE=YYYYMMDD bash scripts/analysis_runs/aut/reproduce.sh
set -eu
O=results/analysis/aut_${DATE:?set DATE=YYYYMMDD, the analysis date (the output folder is aut_DATE)}
S=scripts/analysis_runs/aut/sessions.txt
[ -f "$S" ] || { echo "!! $S does not exist: list the aut sessions of the section 10 rows there first" >&2; exit 2; }
grep -qv '^[[:space:]]*\(#\|$\)' "$S" || { echo "!! $S lists no session yet: list the aut sessions of the section 10 rows there first" >&2; exit 2; }
if [ -e "$O/samples.csv" ]; then echo "$O/samples.csv exists; the output folder must be new" >&2; exit 1; fi
mkdir -p "$O"
{
analysis/.venv/bin/python -m pytest -q -p no:cacheprovider analysis/tests/test_aut.py
analysis/.venv/bin/python analysis/collect.py --labels aut --sessions "$S" --out "$O" --jobs 8
analysis/.venv/bin/python analysis/aut.py --samples "$O/samples.csv" --out "$O" --replicates 10000 --seed 2026
} 2>&1 | tee "$O/run.log"
exit "${PIPESTATUS[0]}"
