#!/bin/bash
# Follow-up 7, the software stack on the same chip (SW(V6), SW(G)) (drafts/section10_fu7-10.md; label sw): the per-sample table of the sessions listed in
# scripts/analysis_runs/sw/sessions.txt (written once the arm has finished, from the section 10 launch rows), with
# M1 (cached by model SHA-256), then analysis/sw.py. Run from the repository root:
#   DATE=YYYYMMDD bash scripts/analysis_runs/sw/reproduce.sh
set -eu
O=results/analysis/sw_${DATE:?set DATE=YYYYMMDD, the analysis date (the output folder is sw_DATE)}
S=scripts/analysis_runs/sw/sessions.txt
[ -f "$S" ] || { echo "!! $S does not exist: list the sw sessions of the section 10 rows there first" >&2; exit 2; }
mkdir -p "$O"
{
analysis/.venv/bin/python analysis/collect.py --labels sw --sessions "$S" --out "$O" --jobs 8
analysis/.venv/bin/python analysis/sw.py --samples "$O/samples.csv" --out "$O" --replicates 10000 --seed 2026
} 2>&1 | tee "$O/run.log"
