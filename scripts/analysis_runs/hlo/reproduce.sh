#!/bin/bash
# Follow-up 16, the HLO of the model on v5e and v6e (label hlo; analysis/hlo.py) (notes/analysis_plan.md section 10,
# row of 2026-11-16): checks 1 and 2 (dumps present; outputs bit-identical with the main study's V5 and V6 r1, seed 1),
# then the comparison of the HLO, for the sessions listed in scripts/analysis_runs/hlo/sessions.txt (written once both
# parts have finished and committed before any analysis run). Before it: scripts/analysis_runs/hlo/fetch_dumps.py
# --check (every process's dumps and full outputs local); this script stops if any is missing. The tests run first.
# Run from the repository root:
#   DATE=YYYYMMDD bash scripts/analysis_runs/hlo/reproduce.sh
set -eu
O=results/analysis/hlo_${DATE:?set DATE=YYYYMMDD, the analysis date (the output folder is hlo_DATE)}
S=scripts/analysis_runs/hlo/sessions.txt
[ -f "$S" ] || { echo "!! $S does not exist: list the hlo sessions of the section 10 rows there first" >&2; exit 2; }
grep -qv '^[[:space:]]*\(#\|$\)' "$S" || { echo "!! $S lists no session yet: list the hlo sessions of the section 10 rows there first" >&2; exit 2; }
if [ -e "$O/summary.md" ]; then echo "$O/summary.md exists; the output folder must be new" >&2; exit 1; fi
python3 scripts/analysis_runs/hlo/fetch_dumps.py --check || exit 2
mkdir -p "$O"
{
analysis/.venv/bin/python -m pytest -q -p no:cacheprovider analysis/tests/test_hlo.py
analysis/.venv/bin/python analysis/hlo.py --sessions "$S" --out "$O"
} 2>&1 | tee "$O/run.log"
