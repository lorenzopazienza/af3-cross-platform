#!/bin/bash
# Follow-up experiment 1, protein-protein complexes (notes/analysis_plan.md section 10, rows of 2026-10-14 and
# 2026-10-15): the per-sample table, DockQ 2.1.3 of every model (cached by model SHA-256 in dockq_cache.jsonl), and
# the full analysis. Environments: analysis/.venv (analysis/requirements.txt) and analysis/ppi/.venv
# (analysis/requirements-ppi.txt). Run from the repository root:
#   bash scripts/analysis_runs/ppi_20261015/reproduce.sh
set -eu
O=results/analysis/ppi_20261015
analysis/.venv/bin/python analysis/collect.py --labels ppi --no_m1 --out "$O"
analysis/ppi/.venv/bin/python analysis/score_dockq.py --collect --out "$O" > "$O/dockq_run.log" 2>&1
analysis/.venv/bin/python analysis/ppi.py --dir "$O" --replicates 10000 --seed 2026 > "$O/ppi_run.log" 2>&1
