#!/bin/bash
# Campaign simulation (label campaign; analysis/campaign.py) (notes/analysis_plan.md section 10, row A2): a follow-up
# analysis of frozen outputs, outside the pre-registered study, descriptive. No session is collected: the analysis
# reads pairs_seed.csv and samples.csv of the main study and of follow-ups 3, 4, 9 and 11 under results/analysis/.
# The synthetic tests run first. Run from the repository root:
#   DATE=YYYYMMDD bash scripts/analysis_runs/campaign/reproduce.sh
set -eu
O=results/analysis/campaign_${DATE:?set DATE=YYYYMMDD, the analysis date (the output folder is campaign_DATE)}
mkdir -p "$O"
{
analysis/.venv/bin/python -m pytest -q analysis/tests/test_campaign.py
analysis/.venv/bin/python analysis/campaign.py --main . --out "$O" --replicates 10000 --seed 2026
} 2>&1 | tee "$O/run.log"
