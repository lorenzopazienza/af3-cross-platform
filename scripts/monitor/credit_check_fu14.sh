#!/bin/bash
# Follow-up 14 (label f32) budget rule before each wrapper (notes/analysis_plan.md section 10, row of 2026-11-12
# "Follow-up experiment 14", items f to h): see scripts/monitor/fu14_budget.py for the rule.
# Run from the repository root, with the Google TPU Builders credit read from the console (the Free Trial credit is
# not counted):
#   bash scripts/monitor/credit_check_fu14.sh WRAPPER BALANCE_USD
#   WRAPPER: v5e (V5f and V5f r2), a100 (Af), h100 (Hf), det (item b2: Ad and Hd, only after the float32 arms)
# Prints GO or NO GO with the reasons; exit 0 for GO, 1 for NO GO, 2 for a usage error. Reads only local files.
set -u
cd "$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
case "${1:-}" in v5e | a100 | h100 | det) ;;
  *) echo "usage: credit_check_fu14.sh v5e|a100|h100|det BALANCE_USD" >&2; exit 2 ;; esac
echo "${2:-}" | grep -Eq '^[0-9]+(\.[0-9]+)?$' || { echo "BALANCE_USD must be a number (USD), read from the console" >&2; exit 2; }
python3 scripts/monitor/fu14_budget.py check "$1" "$2"
