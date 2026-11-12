#!/bin/bash
# Memory limit on A and H (label memlimit_ah; analysis/memlimit_ah.py, analysis/memlimit.py's logic unchanged),
# notes/analysis_plan.md section 10, the specification row of 2026-11-10 and its analysis row: the complete session of
# each part, listed in scripts/analysis_runs/memlimit_ah/sessions.txt (written from the launch rows once both parts
# have finished, and committed before any analysis run). Reads only those sessions' session.json and run.json files,
# follow-up 3's and 9's samples.csv (the 1,024-token reference process) and cloud/prices.csv; the predicted structures
# are not read. The synthetic tests run first. Run from the repository root:
#   DATE=YYYYMMDD bash scripts/analysis_runs/memlimit_ah/reproduce.sh
set -eu
O=results/analysis/memlimit_ah_${DATE:?set DATE=YYYYMMDD, the analysis date (the output folder is memlimit_ah_DATE)}
S=scripts/analysis_runs/memlimit_ah/sessions.txt
grep -qv '^[[:space:]]*\(#\|$\)' "$S" || { echo "!! $S lists no session yet: list the complete memlimit_ah sessions of the section 10 rows there first" >&2; exit 2; }
if [ -e "$O/summary.csv" ]; then echo "$O/summary.csv exists; the output folder must be new" >&2; exit 1; fi
mkdir -p "$O"
{
analysis/.venv/bin/python -m pytest -q -p no:cacheprovider analysis/tests/test_memlimit_ah.py
analysis/.venv/bin/python analysis/memlimit_ah.py --sessions "$S" --out "$O"
} 2>&1 | tee "$O/run.log"
