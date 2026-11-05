#!/bin/bash
# Follow-up 10, larger inputs on the whole PDB (V5, V6, G with label large; analysis/large.py)
# (notes/analysis_plan.md section 10, row of 2026-11-04, item d; OSF addendum https://osf.io/nsbgp): the per-sample
# table of the sessions listed in scripts/analysis_runs/large/sessions.txt (written once the arms have finished, from
# the section 10 launch rows, and committed before any analysis run), with M1 (cached by model SHA-256; reference: the
# deposited model, data/large_refs/, from targets/large/build_large_subset.py), then the analysis. Run from the
# repository root:
#   DATE=YYYYMMDD bash scripts/analysis_runs/large/reproduce.sh
set -eu
O=results/analysis/large_${DATE:?set DATE=YYYYMMDD, the analysis date (the output folder is large_DATE)}
S=scripts/analysis_runs/large/sessions.txt
[ -f "$S" ] || { echo "!! $S does not exist: list the large sessions of the section 10 rows there first" >&2; exit 2; }
grep -qv '^[[:space:]]*\(#\|$\)' "$S" || { echo "!! $S lists no session yet: list the large sessions of the section 10 rows there first" >&2; exit 2; }
mkdir -p "$O"
{
analysis/.venv/bin/python analysis/collect.py --labels large --sessions "$S" --out "$O" --jobs 8
analysis/.venv/bin/python analysis/large.py --samples "$O/samples.csv" --out "$O" --replicates 10000 --seed 2026
} 2>&1 | tee "$O/run.log"
