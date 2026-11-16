#!/bin/bash
# Follow-up 16 (label hlo) budget rule before the wrapper (notes/analysis_plan.md section 10, row of 2026-11-16
# "Follow-up experiment 16", items f to h): see scripts/monitor/hlo_budget.py for the rule.
# Run from the repository root, with the Google TPU Builders credit read from the console (40.47 USD on 2026-11-16;
# the Free Trial credit is not counted):
#   bash scripts/monitor/credit_check_hlo.sh BALANCE_USD
# Prints GO or NO GO with the reasons; exit 0 for GO, 1 for NO GO, 2 for a usage error. Reads only local files.
set -u
cd "$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
echo "${1:-}" | grep -Eq '^[0-9]+(\.[0-9]+)?$' || { echo "usage: credit_check_hlo.sh BALANCE_USD (USD, read from the console)" >&2; exit 2; }
python3 scripts/monitor/hlo_budget.py check "$1"
