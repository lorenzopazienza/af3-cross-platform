#!/bin/bash
# Follow-up 6 budget rule (notes/analysis_plan.md section 10, row of 2026-10-19): an arm is launched only if the credit
# balance read before its launch, minus its estimated cost, stays at or above 100 USD; the inference arms have a cost
# limit of 140 USD. Run from the repository root before each arm, with the balance read from the console:
#   bash scripts/monitor/credit_check.sh ARM BALANCE_USD        # ARM: go_r1, v6o, v5o, go_r2 (or smoke)
#
# The balance cannot be read from the command line: the Cloud Billing API and gcloud expose billing accounts,
# budgets and exported costs, not the remaining promotional credits. Manual check: Google Cloud console, Billing,
# the project's billing account, Credits; the remaining value of every active credit, summed (in the follow-up 3
# reading of 2026-10-17, "559 USD used of 1,257 USD of credits", the balance was 698 USD). Billing lags up to a day,
# so a run that ended in the last 24 h may not be in it yet: subtract it by hand (the script prints the follow-up 6
# spend it estimates from the session records).
#
# Estimated cost of the arm: the expected USD of its parts (harness/plans/msa/parts.csv). Spend so far: every
# follow-up 6 session (labels msa and msa_smoke; the smoke tests are reported apart, outside the 140 USD limit) in results/af3/, VM minutes from launcher_events.txt (VM running
# to delete done, or to now while it runs) x the on-demand price of its platform in parts.csv.
# Prints GO or NO GO with the reason; exit 0 for GO, 1 for NO GO, 2 for a usage error. Reads only local files.
set -u
ARM=${1:-} BAL=${2:-}
case "$ARM" in go_r1 | v6o | v5o | go_r2 | smoke) ;; *) echo "usage: credit_check.sh go_r1|v6o|v5o|go_r2|smoke BALANCE_USD" >&2; exit 2 ;; esac
echo "$BAL" | grep -Eq '^[0-9]+(\.[0-9]+)?$' || { echo "BALANCE_USD must be a number (USD), read from the console" >&2; exit 2; }
cd "$(git rev-parse --show-toplevel)"
python3 - "$ARM" "$BAL" <<'PY'
import csv, datetime as dt, pathlib, re, sys
arm, bal = sys.argv[1], float(sys.argv[2])
parts = list(csv.DictReader(open('harness/plans/msa/parts.csv')))
price = {p['platform']: float(p['usd_per_h']) for p in parts}
SMOKE = {'v5e': 1.56 * 0.33, 'v6e': 2.97 * 0.29, 'l4': 0.8972 * 0.41}   # expected USD of the three smoke tests
est = sum(SMOKE.values()) if arm == 'smoke' else sum(float(p['expected_usd']) for p in parts if p['arm'].lower() == arm)
worst = None if arm == 'smoke' else sum(float(p['worst_usd']) for p in parts if p['arm'].lower() == arm)
now = dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)
spent, smoke_spent, n = 0.0, 0.0, 0
for s in sorted(pathlib.Path('results/af3').glob('*_msa_*')) + sorted(pathlib.Path('results/af3/.launcher').glob('*_msa_*')):
  f = s / 'launcher_events.txt'
  m = re.match(r'\d{8}T\d{6}Z_(v5e|v6e|l4)_msa_', s.name)
  if not m or not f.is_file():
    continue
  ev = {}
  for line in f.read_text().splitlines():
    mm = re.match(r'(\S+)\s+(\w+): (\w+)', line)
    if mm:
      ev.setdefault(f'{mm.group(2)}:{mm.group(3)}', dt.datetime.strptime(mm.group(1), '%Y-%m-%dT%H:%M:%SZ'))
  start = next((v for k, v in ev.items() if k.startswith('vm:')), None)
  if start is None:
    continue
  end = ev.get('delete:done', now)
  usd = (end - start).total_seconds() / 3600 * price.get(m.group(1), max(price.values()))
  if '_msa_smoke' in s.name:   # engineering checks: not an inference arm, outside the 140 USD limit
    smoke_spent += usd
  else:
    spent += usd
  n += 1
left = bal - est
print(f'arm {arm}: estimated cost {est:.2f} USD' + (f' (worst case {worst:.2f})' if worst is not None else '')
      + f'; balance read {bal:.2f} USD; balance minus estimate {left:.2f} USD (rule: at least 100)')
print(f'follow-up 6 spend so far, from {n} session record(s): inference arms {spent:.2f} USD, smoke tests '
      f'{smoke_spent:.2f} USD; inference arms with this arm {spent + (0 if arm == "smoke" else est):.2f} USD '
      '(cost limit of the inference arms: 140)')
reasons = []
if left < 100:
  reasons.append(f'balance minus estimate {left:.2f} < 100 USD')
if arm != 'smoke' and spent + est > 140:
  reasons.append(f'spend with this arm {spent + est:.2f} > 140 USD')
print('GO' if not reasons else 'NO GO: ' + '; '.join(reasons))
sys.exit(0 if not reasons else 1)
PY
