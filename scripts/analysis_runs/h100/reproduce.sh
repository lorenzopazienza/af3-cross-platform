#!/bin/bash
# Follow-up 9, the H100 (HW(H,A), HW(H,G), HW(H,V6), HW(H,V5), SD(H), RC(H) if r2 ran) (drafts/section10_fu7-10.md; label h100): the per-sample table of the sessions listed in
# scripts/analysis_runs/h100/sessions.txt (written once the arm has finished, from the section 10 launch rows), with
# M1 (cached by model SHA-256), then analysis/h100.py. Run from the repository root:
#   DATE=YYYYMMDD bash scripts/analysis_runs/h100/reproduce.sh
set -eu
O=results/analysis/h100_${DATE:?set DATE=YYYYMMDD, the analysis date (the output folder is h100_DATE)}
S=scripts/analysis_runs/h100/sessions.txt
[ -f "$S" ] || { echo "!! $S does not exist: list the h100 sessions of the section 10 rows there first" >&2; exit 2; }
mkdir -p "$O"
{
analysis/.venv/bin/python analysis/collect.py --labels h100 --sessions "$S" --out "$O" --jobs 8
analysis/.venv/bin/python analysis/h100.py --samples "$O/samples.csv" --out "$O" --replicates 10000 --seed 2026
} 2>&1 | tee "$O/run.log"
