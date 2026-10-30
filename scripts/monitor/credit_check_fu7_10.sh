#!/bin/bash
# Follow-ups 7 to 9 budget rule (drafts/section10_fu7-10.md): an arm is launched only if the credit balance read before
# its launch, minus the arm's expected cost, stays at or above the floor of 25 USD, and the follow-up's spend with this
# arm stays within its cap (seeds 95 since 2026-10-30, was 75; sw 55, h100 85 USD, about 1.2 times each follow-up's expected cost; the smoke
# tests are reported apart, outside the caps). Credit: the TPU Builders credit only (the Free Trial does not cover TPUs
# or GPUs, Cloud Billing support, 2026-10-27). Run from the repository root before each arm, with the balance read
# from the console:
#   bash scripts/monitor/credit_check_fu7_10.sh ARM BALANCE_USD
#   ARM: v5x, gx (seeds); v6s, gs (sw); h, h_r2 (h100); sw_smoke, h100_smoke
# Expected cost: the planned column of harness/plans/fu7_10_parts.csv, as decided on 2026-10-27: Spot on v5e, L4 and
# H100 (V5x, Gx, Gs, H, H r2 and their smoke tests), on demand on v6e (V6s and its smoke test).
#
# The balance cannot be read from the command line (see scripts/monitor/credit_check.sh): Google Cloud console,
# Billing, Credits, the remaining value of every active credit, summed. Billing lags up to a day: subtract by hand a run
# that ended in the last 24 h (the script prints the spend it estimates from the session records).
#
# Spend so far: every session of the follow-up (labels seeds, sw, h100; sw_smoke and h100_smoke apart) in results/af3/
# and results/af3/.launcher/ (each session once), VM minutes from launcher_events.txt (VM running to delete done, or to
# now while it runs) x the price of its platform in fu7_10_parts.csv for the provisioning in its launch_record.txt
# (spot: the Spot price; standard, anything else or no record: on demand).
# Prints GO or NO GO with the reason; exit 0 for GO, 1 for NO GO, 2 for a usage error. Reads only local files.
set -u
ARM=${1:-} BAL=${2:-}
case "$ARM" in v5x | gx | v6s | gs | h | h_r2 | sw_smoke | h100_smoke) ;;
  *) echo "usage: credit_check_fu7_10.sh v5x|gx|v6s|gs|h|h_r2|sw_smoke|h100_smoke BALANCE_USD" >&2; exit 2 ;; esac
echo "$BAL" | grep -Eq '^[0-9]+(\.[0-9]+)?$' || { echo "BALANCE_USD must be a number (USD), read from the console" >&2; exit 2; }
cd "$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
python3 - "$ARM" "$BAL" <<'PY'
import csv, datetime as dt, pathlib, re, sys
arm, bal = sys.argv[1], float(sys.argv[2])
FLOOR = 25.0
CAP = {'seeds': 95.0, 'sw': 55.0, 'h100': 85.0}
parts = list(csv.DictReader(open('harness/plans/fu7_10_parts.csv')))
od = {p['platform']: float(p['usd_per_h_on_demand']) for p in parts}
planned = {p['platform']: (p['provisioning'], float(p['usd_per_h_planned'])) for p in parts}
spot = {pl: usd for pl, (prov, usd) in planned.items() if prov == 'spot'}
# Smoke tests: one process of 2PV7 with provisioning, setup and teardown (the main-study medians, H from the A100
# sessions), VM hours v6e 0.29, l4 0.41, h100 0.42, at each platform's planned price.
SMOKE = {'sw_smoke': (('v6e', 0.29), ('l4', 0.41)), 'h100_smoke': (('h100', 0.42),)}
if arm in SMOKE:
  label = arm
  est = sum(planned[pl][1] * h for pl, h in SMOKE[arm])
  worst = None
else:
  rows = [p for p in parts if p['arm'].lower().replace(' ', '_') == arm]
  label = rows[0]['label']
  est = sum(float(p['expected_usd_planned']) for p in rows)
  worst = sum(float(p['worst_usd_planned']) for p in rows)
follow = label.replace('_smoke', '')
now = dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)
spent, smoke_spent, n = 0.0, 0.0, 0
seen = set()
for s in sorted(pathlib.Path('results/af3').glob('*')) + sorted(pathlib.Path('results/af3/.launcher').glob('*')):
  m = re.match(r'\d{8}T\d{6}Z_(v5e|v6e|l4|h100)_(sw_smoke|h100_smoke|seeds|sw|h100)_', s.name)
  f = s / 'launcher_events.txt'
  if not m or m.group(2).replace('_smoke', '') != follow or not f.is_file() or s.name in seen:
    continue
  seen.add(s.name)
  ev = {}
  for line in f.read_text().splitlines():
    mm = re.match(r'(\S+)\s+(\w+): (\w+)', line)
    if mm:
      ev.setdefault(f'{mm.group(2)}:{mm.group(3)}', dt.datetime.strptime(mm.group(1), '%Y-%m-%dT%H:%M:%SZ'))
  start = next((v for k, v in ev.items() if k.startswith('vm:')), None)
  if start is None:
    continue
  rec = s / 'launch_record.txt'
  prov = 'standard'
  if rec.is_file():
    for line in rec.read_text().splitlines():
      if line.startswith('PROVISIONING='):
        prov = line.split('=', 1)[1]
  pl = m.group(1)
  rate = spot[pl] if prov == 'spot' and pl in spot else od.get(pl, max(od.values()))
  usd = (ev.get('delete:done', now) - start).total_seconds() / 3600 * rate
  if m.group(2).endswith('_smoke'):
    smoke_spent += usd
  else:
    spent += usd
  n += 1
left = bal - est
how = ', '.join(sorted({f'{pl} {planned[pl][0]}' for pl, _ in SMOKE[arm]} if arm in SMOKE else
                       {f'{p["platform"]} {p["provisioning"]}' for p in rows}))
print(f'arm {arm} ({how}): expected cost {est:.2f} USD' + (f' (worst case {worst:.2f})' if worst is not None else '')
      + f'; balance read {bal:.2f} USD; balance minus expected {left:.2f} USD (floor: {FLOOR:.0f})')
print(f'follow-up {follow} spend so far, from {n} session record(s): arms {spent:.2f} USD, smoke tests {smoke_spent:.2f} '
      f'USD; arms with this arm {spent + (0 if arm in SMOKE else est):.2f} USD (cap of {follow}: {CAP[follow]:.0f})')
reasons = []
if left < FLOOR:
  reasons.append(f'balance minus expected {left:.2f} < {FLOOR:.0f} USD')
if arm not in SMOKE and spent + est > CAP[follow]:
  reasons.append(f'spend of {follow} with this arm {spent + est:.2f} > {CAP[follow]:.0f} USD')
print('GO' if not reasons else 'NO GO: ' + '; '.join(reasons))
sys.exit(0 if not reasons else 1)
PY
