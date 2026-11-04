#!/bin/bash
# Follow-ups 11 (repl) and 12 (prec5) budget rule (notes/analysis_plan.md section 10, rows of 2026-11-03: "budget rule
# as follow-ups 7 to 9"): an arm is launched only if the credit balance read before its launch, minus the expected cost
# of every other arm still running and minus this arm's expected cost, stays at or above the floor of 25 USD, and the
# follow-up's spend with this arm stays within its cap (about 1.2 times the follow-up's expected cost: repl 101, prec5
# 24 USD). Two arms at once (the parallel-arms row): an arm counts as running while one of its sessions has a
# launcher_events.txt without a "launcher: exit" line; its whole expected cost is subtracted, since the console
# balance lags the spend. Run from the repository root before each arm, with the balance read from the console:
#   bash scripts/monitor/credit_check_fu11_12.sh ARM BALANCE_USD
#   ARM: v5r, v6r, gr (repl); v6h5, gh5 (prec5)
# Expected cost: the planned column of harness/plans/fu11_12_parts.csv (on demand on v5e, Flex-start on v6e, Spot on
# L4, as proposed in the implementation row); of a running arm of follow-ups 7 to 9: harness/plans/fu7_10_parts.csv.
# Follow-up 12 is launched only if every run can finish by 2026-11-14 (its row): GO also needs the arm's expected
# duration from now to end by 2026-11-14 24:00 UTC (v6e parts one after another, the launcher's one TPU at a time; L4
# parts in waves of 5, the L4 launchers left while the H100 arms run). For follow-up 11 the expected end is printed
# (an arm not finished by 2026-11-14 is reported as incomplete; not a launch condition).
#
# The balance cannot be read from the command line (see scripts/monitor/credit_check.sh): Google Cloud console,
# Billing, Credits, the remaining value of every active credit, summed. Billing lags up to a day: subtract by hand a
# run that ended in the last 24 h (the script prints the spend it estimates from the session records).
#
# Spend so far: every session of the follow-up (labels repl, prec5) in results/af3/ and results/af3/.launcher/ (each
# session once), VM minutes from launcher_events.txt (VM running to delete done, or the VM already gone, or the
# launcher's exit, or now while it runs) x the price of its platform for the provisioning in its launch_record.txt
# (spot: the Spot price of fu11_12_parts.csv; flex_start: the Flex-start price of cloud/prices.csv; anything else:
# on demand). NOW=YYYY-MM-DDTHH:MM:SSZ replaces the clock (tests).
# Prints GO or NO GO with the reason; exit 0 for GO, 1 for NO GO, 2 for a usage error. Reads only local files.
set -u
ARM=${1:-} BAL=${2:-}
case "$ARM" in v5r | v6r | gr | v6h5 | gh5) ;;
  *) echo "usage: credit_check_fu11_12.sh v5r|v6r|gr|v6h5|gh5 BALANCE_USD" >&2; exit 2 ;; esac
echo "$BAL" | grep -Eq '^[0-9]+(\.[0-9]+)?$' || { echo "BALANCE_USD must be a number (USD), read from the console" >&2; exit 2; }
cd "$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
python3 - "$ARM" "$BAL" "${NOW:-}" <<'PY'
import csv, datetime as dt, math, pathlib, re, sys
arm, bal, now_arg = sys.argv[1], float(sys.argv[2]), sys.argv[3]
FLOOR = 25.0
CAP = {'repl': 101.0, 'prec5': 24.0}
FINISH_BY = dt.datetime(2026, 11, 15)          # the end of 2026-11-14, UTC
L4_SLOTS = 5
now = dt.datetime.strptime(now_arg, '%Y-%m-%dT%H:%M:%SZ') if now_arg else \
    dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)
parts = list(csv.DictReader(open('harness/plans/fu11_12_parts.csv')))
old = list(csv.DictReader(open('harness/plans/fu7_10_parts.csv'))) if pathlib.Path('harness/plans/fu7_10_parts.csv').is_file() else []
key = lambda p: p['arm'].lower().replace(' ', '_')  # noqa: E731
expected = {}
for p in old + parts:
  expected[key(p)] = expected.get(key(p), 0.0) + float(p['expected_usd_planned'])
od = {p['platform']: float(p['usd_per_h_on_demand']) for p in parts}
spot = {p['platform']: float(p['usd_per_h_planned']) for p in parts if p['provisioning'] == 'spot'}
flex = {}
for r in csv.DictReader(open('cloud/prices.csv')):
  if r['provisioning'] == 'flex_start' and r['usd_per_hour']:
    flex[r['platform']] = max(flex.get(r['platform'], 0.0), float(r['usd_per_hour']))
rows = [p for p in parts if key(p) == arm]
label = rows[0]['label']
est = sum(float(p['expected_usd_planned']) for p in rows)
worst = sum(float(p['worst_usd_planned']) for p in rows)


def events(f):
  ev = {}
  for line in f.read_text().splitlines():
    m = re.match(r'(\S+)\s+(\w+): (\w+)', line)
    if m:
      ev.setdefault(f'{m.group(2)}:{m.group(3)}', dt.datetime.strptime(m.group(1), '%Y-%m-%dT%H:%M:%SZ'))
  return ev


spent, n, running, seen = 0.0, 0, {}, set()
for s in sorted(pathlib.Path('results/af3').glob('*')) + sorted(pathlib.Path('results/af3/.launcher').glob('*')):
  m = re.match(r'\d{8}T\d{6}Z_(v5e|v6e|l4|h100)_(repl|prec5|h100|seeds|sw)_(.+)_\d\d$', s.name)
  f = s / 'launcher_events.txt'
  if not m or not f.is_file() or s.name in seen:
    continue
  seen.add(s.name)
  ev = events(f)
  if 'launcher:exit' not in ev:
    running.setdefault(m.group(3), []).append(s.name)
  if m.group(2) != label:
    continue
  start = next((v for k, v in ev.items() if k.startswith('vm:')), None)
  if start is None:
    continue
  prov = 'standard'
  rec = s / 'launch_record.txt'
  if rec.is_file():
    for line in rec.read_text().splitlines():
      if line.startswith('PROVISIONING='):
        prov = line.split('=', 1)[1]
  pl = m.group(1)
  rate = spot[pl] if prov == 'spot' and pl in spot else flex[pl] if prov == 'flex_start' and pl in flex else \
      od.get(pl, max(od.values()))
  end = ev.get('delete:done') or ev.get('delete:the') or ev.get('launcher:exit') or now
  spent += (end - start).total_seconds() / 3600 * rate
  n += 1
others = {a: expected.get(a, 0.0) for a in running if a != arm}
left = bal - sum(others.values()) - est
print(f'arm {arm} ({rows[0]["platform"]} {rows[0]["provisioning"]}): expected cost {est:.2f} USD (worst case {worst:.2f}); '
      f'balance read {bal:.2f} USD')
for a, usd in sorted(others.items()):
  print(f'running arm {a} ({len(running[a])} session(s) whose launcher has not exited): expected {usd:.2f} USD subtracted')
print(f'balance minus running arms minus expected {left:.2f} USD (floor: {FLOOR:.0f})')
print(f'follow-up {label} spend so far, from {n} session record(s): {spent:.2f} USD; with this arm {spent + est:.2f} USD '
      f'(cap of {label}: {CAP[label]:.0f})')
vm = [float(p['expected_vm_h']) for p in rows]
hours = sum(vm) if rows[0]['platform'] in ('v5e', 'v6e') else math.ceil(len(vm) / L4_SLOTS) * max(vm)
end = now + dt.timedelta(hours=hours)
print(f'expected duration {hours:.1f} h ({len(vm)} part(s)): from {now:%Y-%m-%d %H:%M} UTC to about {end:%Y-%m-%d %H:%M} UTC '
      f'(finish by 2026-11-14 24:00 UTC)')
reasons = []
if left < FLOOR:
  reasons.append(f'balance minus running arms minus expected {left:.2f} < {FLOOR:.0f} USD')
if spent + est > CAP[label]:
  reasons.append(f'spend of {label} with this arm {spent + est:.2f} > {CAP[label]:.0f} USD')
if label == 'prec5' and end > FINISH_BY:
  reasons.append('its runs cannot finish by 2026-11-14 (follow-up 12 is then reported as not run)')
print('GO' if not reasons else 'NO GO: ' + '; '.join(reasons))
sys.exit(0 if not reasons else 1)
PY
