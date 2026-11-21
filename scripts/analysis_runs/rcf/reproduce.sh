#!/bin/bash
# Follow-up 19 (recompilation of Af and Hf on the other 50 targets; label rcf; analysis/rcf.py) (notes/analysis_plan.md section 10, row
# "Follow-up experiment 19"): the per-sample table of the sessions listed in scripts/analysis_runs/rcf/sessions.txt
# (written once the parts have finished, by check_sessions.py --write, from the section 10 launch rows, and committed
# before any analysis run), with M1 (cached by model SHA-256), then the analysis. The r1 sides, the r2 of the 10 pilot targets and the references are
# read from their analysis folders (results/analysis/fu14_20261113b, rcx_20261115, h100_20261104). The 50 new targets
# are compared on the light-fetch files; the 10 pilot targets on every output file, with the full outputs fetched for
# follow-up 15 (scripts/analysis_runs/rcx/fetch_full_outputs.py --check must pass). The tests run first. Run from the repository root:
#   DATE=YYYYMMDD bash scripts/analysis_runs/rcf/reproduce.sh
set -eu
O=results/analysis/rcf_${DATE:?set DATE=YYYYMMDD, the analysis date (the output folder is rcf_DATE)}
S=scripts/analysis_runs/rcf/sessions.txt
[ -f "$S" ] || { echo "!! $S does not exist: list the rcf sessions of the section 10 rows there first" >&2; exit 2; }
grep -qv '^[[:space:]]*\(#\|$\)' "$S" || { echo "!! $S lists no session yet: list the rcf sessions of the section 10 rows there first" >&2; exit 2; }
if [ -e "$O/samples.csv" ]; then echo "$O/samples.csv exists; the output folder must be new" >&2; exit 1; fi
python3 scripts/analysis_runs/rcx/fetch_full_outputs.py --check || exit 2
mkdir -p "$O"
{
analysis/.venv/bin/python -m pytest -q -p no:cacheprovider analysis/tests/test_rcf.py
analysis/.venv/bin/python analysis/collect.py --labels rcf --sessions "$S" --out "$O" --jobs 8
analysis/.venv/bin/python analysis/rcf.py --samples "$O/samples.csv" --out "$O" --replicates 10000 --seed 2026
} 2>&1 | tee "$O/run.log"
exit "${PIPESTATUS[0]}"
