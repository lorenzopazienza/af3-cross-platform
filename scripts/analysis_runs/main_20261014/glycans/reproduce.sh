#!/bin/bash
# M1 for three glycan ligands (notes/analysis_plan.md section 10, row of 2026-10-14). Run from the repository root:
#   bash scripts/analysis_runs/main_20261014/glycans/reproduce.sh
# Reads results/analysis/main_20261014/{samples.csv,contrasts/,cost/} and writes only into this folder.
set -eu
O=results/analysis/main_20261014/glycans
S=scripts/analysis_runs/main_20261014/glycans   # the scripts; they write into $O
# (2) 57 targets and (3) post-hoc rescoring, side by side with (1) the primary outputs
analysis/.venv/bin/python analysis/glycans.py --main results/analysis/main_20261014 --jobs 8 --replicates 10000 \
  --seed 2026 2>&1 | tee "$O/run.log"
# (4) descriptive: AlphaFold3's glycan O1 rule on PoseBusters V1 and V2 (CCD of 2026-10-08), and the local
# featurisation test (AlphaFold3 install on this machine, no weights)
analysis/.venv/bin/python "$S/audit_posebusters_glycans.py" > /dev/null
(cd third_party/alphafold3 && .venv/bin/python "../../$S/af3_local_glycan_test.py" > /dev/null 2>&1)
