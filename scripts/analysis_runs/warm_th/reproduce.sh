#!/bin/bash
# Warm cost on GT and H (label warm_th; analysis/warm_th.py), notes/analysis_plan.md section 10, the specification row
# of 2026-11-11: the complete session of each part, listed in scripts/analysis_runs/warm_th/sessions.txt (written from
# the launch rows once both parts have finished, and committed before any analysis run). analysis/collect.py builds
# the per-sample table of those sessions only, without M1 (no prediction enters any finding); analysis/warm_th.py
# runs the checks (sessions, completeness, exit codes, 2a, 2b, bit-identity of every warm output against its fresh
# run) and, only if all pass, the cost (analysis/cost.py, cloud/prices.csv). The synthetic tests run first. Run from
# the repository root:
#   DATE=YYYYMMDD bash scripts/analysis_runs/warm_th/reproduce.sh
set -eu
O=results/analysis/warm_th_${DATE:?set DATE=YYYYMMDD, the analysis date (the output folder is warm_th_DATE)}
S=scripts/analysis_runs/warm_th/sessions.txt
grep -qv '^[[:space:]]*\(#\|$\)' "$S" || { echo "!! $S lists no session yet: list the complete warm_th sessions of the section 10 rows there first" >&2; exit 2; }
if [ -e "$O/summary.csv" ]; then echo "$O/summary.csv exists; the output folder must be new" >&2; exit 1; fi
mkdir -p "$O"
set -o pipefail
{
analysis/.venv/bin/python -m pytest -q -p no:cacheprovider analysis/tests/test_warm_th.py
analysis/.venv/bin/python analysis/collect.py --labels warm_th --sessions "$S" --out "$O" --no_m1
analysis/.venv/bin/python analysis/warm_th.py --samples "$O/samples.csv" --sessions "$S" --out "$O"
} 2>&1 | tee "$O/run.log"
