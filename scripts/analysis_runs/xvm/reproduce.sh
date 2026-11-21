#!/bin/bash
# Follow-up 21 (recompilation on a new VM of the same GPU type; label xvm; analysis/xvm.py) (notes/analysis_plan.md section 10, row
# "Follow-up experiment 21"): the per-sample table of the sessions listed in scripts/analysis_runs/xvm/sessions.txt
# (written once the parts have finished, by check_sessions.py --write, from the section 10 launch rows, and committed
# before any analysis run), with M1 (cached by model SHA-256), then the analysis. The r1 and r2 sides are read from their analysis folders
# (results/analysis/main_20261014/primary_v2 for G, a100_20261018 for A). Every
# file compared is in the light fetch: no full outputs are fetched. The tests run first. Run from the repository root:
#   DATE=YYYYMMDD bash scripts/analysis_runs/xvm/reproduce.sh
set -eu
O=results/analysis/xvm_${DATE:?set DATE=YYYYMMDD, the analysis date (the output folder is xvm_DATE)}
S=scripts/analysis_runs/xvm/sessions.txt
[ -f "$S" ] || { echo "!! $S does not exist: list the xvm sessions of the section 10 rows there first" >&2; exit 2; }
grep -qv '^[[:space:]]*\(#\|$\)' "$S" || { echo "!! $S lists no session yet: list the xvm sessions of the section 10 rows there first" >&2; exit 2; }
if [ -e "$O/samples.csv" ]; then echo "$O/samples.csv exists; the output folder must be new" >&2; exit 1; fi
mkdir -p "$O"
{
analysis/.venv/bin/python -m pytest -q -p no:cacheprovider analysis/tests/test_xvm.py
analysis/.venv/bin/python analysis/collect.py --labels xvm --sessions "$S" --out "$O" --jobs 8
analysis/.venv/bin/python analysis/xvm.py --samples "$O/samples.csv" --out "$O" --replicates 10000 --seed 2026
} 2>&1 | tee "$O/run.log"
exit "${PIPESTATUS[0]}"
