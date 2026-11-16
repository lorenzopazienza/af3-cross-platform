#!/bin/bash
# Which success criterion changes in a success flip, and by how much (label pb_flips; analysis/pb_flips.py), section 10
# row of drafts/section10_pb_flips_spec_20261116.md: a post hoc analysis of frozen outputs asked by the reviewers,
# descriptive. Reads the per-sample tables and seed pairs of the main study and follow-ups 3 and 9, and reruns M1
# (analysis/score_ligand.py, posebusters 0.6.5, configuration dock) on the top-1 mmCIF files of both sides of every
# success flip under results/af3/ (all local; no session is collected); the analysis stops unless the rerun reproduces
# the saved M1 exactly. Runs the synthetic tests first, then the analysis. Run from the repository root, after the row
# is committed:
#   DATE=YYYYMMDD bash scripts/analysis_runs/pb_flips/reproduce.sh
set -eu
set -o pipefail
O=results/analysis/pb_flips_${DATE:?set DATE=YYYYMMDD, the analysis date (the output folder is pb_flips_DATE)}
if [ -e "$O/summary.csv" ]; then echo "$O/summary.csv exists; the output folder must be new" >&2; exit 1; fi
mkdir -p "$O"
git rev-parse HEAD > "$O/analysis_commit.txt"
{
analysis/.venv/bin/python -c 'import importlib.metadata as m; print(" ".join(p + "==" + m.version(p) for p in ("posebusters", "rdkit", "gemmi", "numpy", "pandas")))'
analysis/.venv/bin/python -m pytest -q -p no:cacheprovider analysis/tests/test_pb_flips.py
analysis/.venv/bin/python analysis/pb_flips.py --main . --out "$O" --replicates 10000 --seed 2026 --jobs 8
} 2>&1 | tee "$O/run.log"
