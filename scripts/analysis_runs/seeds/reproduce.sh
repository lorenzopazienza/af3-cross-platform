#!/bin/bash
# Follow-up 8, Q3 to 20 seeds on V5 and G (drafts/section10_fu7-10.md; label seeds): the per-sample table of the sessions listed in
# scripts/analysis_runs/seeds/sessions.txt (written once the arm has finished, from the section 10 launch rows), with
# M1 (cached by model SHA-256), then analysis/seeds.py. Run from the repository root:
#   DATE=YYYYMMDD bash scripts/analysis_runs/seeds/reproduce.sh
set -eu
O=results/analysis/seeds_${DATE:?set DATE=YYYYMMDD, the analysis date (the output folder is seeds_DATE)}
S=scripts/analysis_runs/seeds/sessions.txt
[ -f "$S" ] || { echo "!! $S does not exist: list the seeds sessions of the section 10 rows there first" >&2; exit 2; }
mkdir -p "$O"
{
analysis/.venv/bin/python analysis/collect.py --labels seeds --sessions "$S" --out "$O" --jobs 8
analysis/.venv/bin/python analysis/seeds.py --samples "$O/samples.csv" --out "$O" --replicates 10000 --seed 2026
} 2>&1 | tee "$O/run.log"
