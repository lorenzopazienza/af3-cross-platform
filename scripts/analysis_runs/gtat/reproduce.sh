#!/bin/bash
# The L4 against the A100, both with AlphaFold 3's default Triton attention (label gtat; analysis/gtat.py), section 10
# row of drafts/section10_gtat_spec_20261121.md: a post hoc analysis of frozen outputs. Reads the per-sample tables of
# GT (main study) and AT (follow-up 3), the registered seed pairs and rows of the main study, follow-up 3 and
# follow-up 15, the follow-up 3 session list and the freeze session list (all local; no structure file is read and no
# session is collected); runs the synthetic tests first, then the analysis. Run from the repository root (or from a
# worktree whose results/ links to the main checkout's), after the row is committed:
#   DATE=YYYYMMDD bash scripts/analysis_runs/gtat/reproduce.sh
set -eu
set -o pipefail
O=results/analysis/gtat_${DATE:?set DATE=YYYYMMDD, the analysis date (the output folder is gtat_DATE)}
if [ -e "$O/hw_minus_rc.csv" ]; then echo "$O/hw_minus_rc.csv exists; the output folder must be new" >&2; exit 1; fi
mkdir -p "$O"
git rev-parse HEAD > "$O/analysis_commit.txt"
{
analysis/.venv/bin/python -m pytest -q -p no:cacheprovider analysis/tests/test_gtat.py
analysis/.venv/bin/python analysis/gtat.py --main . --out "$O" --replicates 10000 --seed 2026
} 2>&1 | tee "$O/run.log"
