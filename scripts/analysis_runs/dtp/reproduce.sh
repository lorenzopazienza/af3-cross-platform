#!/bin/bash
# Follow-up 17, the float32 trunk at the default matmul precision (label dtp; analysis/dtp.py)
# (notes/analysis_plan.md section 10, row "Follow-up experiment 17"): the per-sample table of the sessions listed in
# scripts/analysis_runs/dtp/sessions.txt (written once the parts have finished, from the section 10 launch rows, and
# committed before any analysis run), with M1 (cached by model SHA-256), then the analysis. The references are read
# from their analysis folders (results/analysis/main_20261014/primary_v2, prec5_20261107, fu13_20261112,
# fu14_20261113b, a100_20261018, h100_20261104). Before it: scripts/analysis_runs/dtp/fetch_full_outputs.py
# (read-only copies from the results bucket of the full outputs of the V6fd r1 pilot processes that check 5 compares
# byte for byte); this script stops if any is missing. The tests run first. Run from the repository root:
#   DATE=YYYYMMDD bash scripts/analysis_runs/dtp/reproduce.sh
set -eu
O=results/analysis/dtp_${DATE:?set DATE=YYYYMMDD, the analysis date (the output folder is dtp_DATE)}
S=scripts/analysis_runs/dtp/sessions.txt
[ -f "$S" ] || { echo "!! $S does not exist: list the dtp sessions of the section 10 rows there first" >&2; exit 2; }
grep -qv '^[[:space:]]*\(#\|$\)' "$S" || { echo "!! $S lists no session yet: list the dtp sessions of the section 10 rows there first" >&2; exit 2; }
if [ -e "$O/samples.csv" ]; then echo "$O/samples.csv exists; the output folder must be new" >&2; exit 1; fi
python3 scripts/analysis_runs/dtp/fetch_full_outputs.py --check || exit 2
mkdir -p "$O"
{
analysis/.venv/bin/python -m pytest -q -p no:cacheprovider analysis/tests/test_dtp.py analysis/tests/test_prep_budget.py
analysis/.venv/bin/python analysis/collect.py --labels dtp --sessions "$S" --out "$O" --jobs 8
analysis/.venv/bin/python analysis/dtp.py --samples "$O/samples.csv" --out "$O" --replicates 10000 --seed 2026
} 2>&1 | tee "$O/run.log"
