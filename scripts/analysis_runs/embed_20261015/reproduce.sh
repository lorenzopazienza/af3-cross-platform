#!/bin/bash
# Follow-up experiment 2, where the divergence arises (notes/analysis_plan.md section 10, rows of 2026-10-14 and
# 2026-10-15): trunk embeddings and structures of the embed sessions, per target and per pair, and bit-identity with
# the main study. Run from the repository root:
#   bash scripts/analysis_runs/embed_20261015/reproduce.sh
set -eu
O=results/analysis/embed_20261015
analysis/.venv/bin/python analysis/embed.py --out "$O" > "$O/run.log" 2>&1
