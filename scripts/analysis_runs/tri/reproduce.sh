#!/bin/bash
# Follow-up 18, AlphaFold 3's default Triton attention on the H100 and the A100 (label tri; analysis/tri.py)
# (notes/analysis_plan.md section 10, row "Follow-up experiment 18"): the per-sample table of the sessions listed in
# scripts/analysis_runs/tri/sessions.txt (written once the parts have finished, from the section 10 launch rows, and
# committed before any analysis run), with M1 (cached by model SHA-256), then the analysis. AT r1, A, H, G, GT and the
# registered seed pairs are read from their analysis folders (results/analysis/a100_20261018, h100_20261104,
# main_20261014/primary_v2, rcx_20261115); HW(GT,AT) is compared with the post hoc analysis gtat (GTAT, default the
# newest results/analysis/gtat_*). Every arm was fetched light: no full output is needed. The M1 rerun with
# PoseBusters' full report for the main outputs whose success differs (posebusters 0.6.5, as pb_flips) runs by default;
# PB_FULL=0 leaves it out (the PoseBusters checks' excess is then missing). The tests run first. Run from the repository root:
#   DATE=YYYYMMDD [GTAT=results/analysis/gtat_YYYYMMDD] [PB_FULL=0] bash scripts/analysis_runs/tri/reproduce.sh
set -eu
O=results/analysis/tri_${DATE:?set DATE=YYYYMMDD, the analysis date (the output folder is tri_DATE)}
S=scripts/analysis_runs/tri/sessions.txt
[ -f "$S" ] || { echo "!! $S does not exist: list the tri sessions of the section 10 rows there first" >&2; exit 2; }
grep -qv '^[[:space:]]*\(#\|$\)' "$S" || { echo "!! $S lists no session yet: list the tri sessions of the section 10 rows there first" >&2; exit 2; }
if [ -e "$O/samples.csv" ]; then echo "$O/samples.csv exists; the output folder must be new" >&2; exit 1; fi
EXTRA=""
[ -z "${GTAT:-}" ] || EXTRA="--gtat $GTAT"
[ "${PB_FULL:-1}" != 1 ] || EXTRA="$EXTRA --pb_full --jobs 8"
mkdir -p "$O"
{
analysis/.venv/bin/python -m pytest -q -p no:cacheprovider analysis/tests/test_tri.py
analysis/.venv/bin/python analysis/collect.py --labels tri --sessions "$S" --out "$O" --jobs 8
analysis/.venv/bin/python analysis/tri.py --samples "$O/samples.csv" --out "$O" --replicates 10000 --seed 2026 $EXTRA
} 2>&1 | tee "$O/run.log"
