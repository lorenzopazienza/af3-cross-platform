#!/bin/bash
# Follow-ups 17 to 21 (labels dtp, tri, rcf, aut, xvm): budget rule before each wrapper (notes/analysis_plan.md section 10,
# items f to h of each specification row): see scripts/monitor/prep_budget.py for the rule.
# Run from the repository root, with the Google TPU Builders credit read from the console (the Free Trial credit is
# not counted):
#   bash scripts/monitor/credit_check_prep.sh LABEL WRAPPER BALANCE_USD
#   dtp: v6e, l4 (part A); a100, h100 (part B, only after part A has ended)
#   tri: h100 (part A); a100 (part B, only after part A has ended)
#   rcf: a100, h100;  aut: l4;  xvm: l4 (part A); a100 (part B, only after part A has ended)
# Prints GO or NO GO with the reasons; exit 0 for GO, 1 for NO GO, 2 for a usage error. Reads only local files.
set -u
cd "$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
case "${1:-}" in dtp | tri | rcf | aut | xvm) ;;
  *) echo "usage: credit_check_prep.sh dtp|tri|rcf|aut|xvm WRAPPER BALANCE_USD" >&2; exit 2 ;; esac
echo "${3:-}" | grep -Eq '^[0-9]+(\.[0-9]+)?$' || { echo "BALANCE_USD must be a number (USD), read from the console" >&2; exit 2; }
python3 scripts/monitor/prep_budget.py check "$1" "${2:-}" "$3"
