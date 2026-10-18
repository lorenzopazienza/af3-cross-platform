#!/bin/bash
# The paper assets with the follow-up 3 and 4 tables and figure (T21, T22, F14) from the analyses of 2026-10-18, in a
# folder of their own: results/paper_assets/ stays the switch-off output (scripts/analysis_runs/paper_assets/
# reproduce.sh, checked byte for byte by check_byte_identity.sh). Run from the repository root:
#   bash scripts/analysis_runs/paper_assets/reproduce_followups34.sh
set -eu
analysis/figures/.venv/bin/python analysis/paper_assets.py --out results/paper_assets_followups34 --followups34 \
  --cpu60_dir results/analysis/cpu60_20261018 --a100_dir results/analysis/a100_20261018
