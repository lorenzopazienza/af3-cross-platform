#!/bin/bash
# Every table and figure for the paper, from the frozen analysis outputs (analysis/paper_assets.py). Environment:
#   uv venv --python 3.12 analysis/figures/.venv && \
#     uv pip install --python analysis/figures/.venv/bin/python -r analysis/requirements-figures.txt
# Run from the repository root: bash scripts/analysis_runs/paper_assets/reproduce.sh
set -eu
analysis/figures/.venv/bin/python analysis/paper_assets.py --out results/paper_assets
