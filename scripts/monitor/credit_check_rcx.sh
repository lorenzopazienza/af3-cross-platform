#!/bin/bash
# Follow-up 15 (label rcx) budget rule before each wrapper (notes/analysis_plan.md section 10, row of 2026-11-14
# "Follow-up experiment 15", items f to h): see scripts/monitor/rcx_budget.py for the rule.
# Run from the repository root, with the Google TPU Builders credit read from the console (87.00 USD on 2026-11-14;
# the Free Trial credit is not counted):
#   bash scripts/monitor/credit_check_rcx.sh WRAPPER BALANCE_USD
#   WRAPPER: a100 (Af r2) or h100 (Hf r2), part A4; l4 (GT r2, part A3, only after part A4 has ended)
# Prints GO or NO GO with the reasons; exit 0 for GO, 1 for NO GO, 2 for a usage error. Reads only local files.
set -u
cd "$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
case "${1:-}" in a100 | h100 | l4) ;;
  *) echo "usage: credit_check_rcx.sh a100|h100|l4 BALANCE_USD" >&2; exit 2 ;; esac
echo "${2:-}" | grep -Eq '^[0-9]+(\.[0-9]+)?$' || { echo "BALANCE_USD must be a number (USD), read from the console" >&2; exit 2; }
python3 scripts/monitor/rcx_budget.py check "$1" "$2"
