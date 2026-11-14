#!/bin/bash
# Matched recompilation reference for the CPU pair (label cpu_matched; analysis/cpu_matched.py), section 10 row of
# drafts/section10_cpu_matched_spec_20261114.md: a post hoc sensitivity analysis of a registered reading asked by the
# reviewers; the registered reading stays. Reads the seed pairs and registered tables of the main study and follow-up 4
# (no session is collected); runs the synthetic tests first, then the analysis. Run from the repository root, after
# the row is committed:
#   DATE=YYYYMMDD bash scripts/analysis_runs/cpu_matched/reproduce.sh
set -eu
set -o pipefail
O=results/analysis/cpu_matched_${DATE:?set DATE=YYYYMMDD, the analysis date (the output folder is cpu_matched_DATE)}
if [ -e "$O/matched.csv" ]; then echo "$O/matched.csv exists; the output folder must be new" >&2; exit 1; fi
mkdir -p "$O"
git rev-parse HEAD > "$O/analysis_commit.txt"
{
analysis/.venv/bin/python -m pytest -q -p no:cacheprovider analysis/tests/test_cpu_matched.py
analysis/.venv/bin/python analysis/cpu_matched.py --main . --out "$O" --replicates 10000 --seed 2026
} 2>&1 | tee "$O/run.log"
