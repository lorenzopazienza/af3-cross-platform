#!/bin/bash
# Follow-up 13 (label why) budget rule (notes/analysis_plan.md section 10, row of 2026-11-10 "Follow-up experiment 13",
# items g and h): an arm is launched only if the Google TPU Builders credit balance read before its launch, minus the
# expected cost of every other arm still running (follow-ups 7 to 13), minus this arm's expected cost, stays at or above
# the floor of 25 USD, and follow-up 13's spend with this arm stays within its cap of 417 USD (the worst case of item g).
# Item h: GO also needs the launch on or before 2026-11-25 (UTC) and the arm's expected end by 2026-11-30 24:00 UTC
# ("every run finished by 2026-11-30"); after either date the answer is NO GO.
# Run from the repository root before each arm, with the balance read from the console (or the chain's stand-in):
#   bash scripts/monitor/credit_check_fu13.sh ARM BALANCE_USD
#   ARM: l5, l6, lg, v6f, p, gf
# BALANCE_USD: the Google TPU Builders Program credit only (574.88 USD on 2026-11-10); the Free Trial upgrade credit
# is not counted (item g).
# Expected cost: the expected_usd_planned column of harness/plans/fu13_parts.csv (on demand on v5e and L4, Flex-start
# on v6e); of a running arm of another follow-up: harness/plans/fu10_parts.csv, fu11_12_parts.csv or fu7_10_parts.csv.
# An arm counts as running while one of its sessions has a launcher_events.txt without a "launcher: exit" line; its
# whole expected cost is subtracted, since the console balance lags the spend.
# Expected duration from now: v5e and v6e parts one after another (one TPU of a platform at a time); L4 parts on 8
# launchers (MAX_L4=8), longest first, each to the launcher free first.
#
# The balance cannot be read from the command line (see scripts/monitor/credit_check.sh): Google Cloud console,
# Billing, Credits, the remaining value of the TPU Builders credit. Billing lags up to a day.
#
# Spend so far: every session of label why in results/af3/ and results/af3/.launcher/ (each session once), VM minutes
# from launcher_events.txt (VM running to delete done, or the VM already gone, or the launcher's exit, or now while it
# runs) x the price of its platform for the provisioning in its launch_record.txt (flex_start: the Flex-start price of
# cloud/prices.csv; anything else: the on-demand price of fu13_parts.csv).
# NOW=YYYY-MM-DDTHH:MM:SSZ replaces the clock (tests). Prints GO or NO GO with the reasons; exit 0 for GO, 1 for NO GO,
# 2 for a usage error. Reads only local files.
set -u
ARM=${1:-} BAL=${2:-}
case "$ARM" in l5 | l6 | lg | v6f | p | gf) ;;
  *) echo "usage: credit_check_fu13.sh l5|l6|lg|v6f|p|gf BALANCE_USD" >&2; exit 2 ;; esac
echo "$BAL" | grep -Eq '^[0-9]+(\.[0-9]+)?$' || { echo "BALANCE_USD must be a number (USD), read from the console" >&2; exit 2; }
cd "$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
python3 - "$ARM" "$BAL" "${NOW:-}" <<'PY'
import csv, datetime as dt, heapq, pathlib, re, sys
arm, bal, now_arg = sys.argv[1], float(sys.argv[2]), sys.argv[3]
FLOOR, CAP, LABEL = 25.0, 417.0, 'why'
LAUNCH_BY = dt.datetime(2026, 11, 26)                                        # launch on or before 2026-11-25, UTC
FINISH_BY = dt.datetime(2026, 12, 1)                                         # the end of 2026-11-30, UTC
L4_SLOTS = 8
now = dt.datetime.strptime(now_arg, '%Y-%m-%dT%H:%M:%SZ') if now_arg else \
    dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)
parts = list(csv.DictReader(open('harness/plans/fu13_parts.csv')))
others_csv = []
for f in ('harness/plans/fu7_10_parts.csv', 'harness/plans/fu11_12_parts.csv', 'harness/plans/fu10_parts.csv'):
  if pathlib.Path(f).is_file():
    others_csv += list(csv.DictReader(open(f)))
key = lambda p: p['arm'].lower().replace(' ', '_')  # noqa: E731
expected = {}
for p in others_csv + parts:
  expected[key(p)] = expected.get(key(p), 0.0) + float(p['expected_usd_planned'])
od = {p['platform']: float(p['usd_per_h_on_demand']) for p in parts}
flex = {}
for r in csv.DictReader(open('cloud/prices.csv')):
  if r['provisioning'] == 'flex_start' and r['usd_per_hour']:
    flex[r['platform']] = max(flex.get(r['platform'], 0.0), float(r['usd_per_hour']))
rows = [p for p in parts if key(p) == arm]
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
  m = re.match(r'\d{8}T\d{6}Z_(v5e|v6e|l4|h100|a100)_(why|large|repl|prec5|h100|seeds|sw)_(.+)_\d\d$', s.name)
  f = s / 'launcher_events.txt'
  if not m or not f.is_file() or s.name in seen:
    continue
  seen.add(s.name)
  ev = events(f)
  if 'launcher:exit' not in ev:
    running.setdefault(m.group(3), []).append(s.name)
  if m.group(2) != LABEL:
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
  rate = flex[pl] if prov == 'flex_start' and pl in flex else od.get(pl, max(od.values()))
  end = ev.get('delete:done') or ev.get('delete:the') or ev.get('launcher:exit') or now
  spent += (end - start).total_seconds() / 3600 * rate
  n += 1
others = {a: expected.get(a, 0.0) for a in running if a != arm}
left = bal - sum(others.values()) - est
print(f'arm {arm} ({rows[0]["platform"]} {rows[0]["provisioning"]}): expected cost {est:.2f} USD (worst case {worst:.2f}); '
      f'TPU Builders balance read {bal:.2f} USD')
for a, usd in sorted(others.items()):
  print(f'running arm {a} ({len(running[a])} session(s) whose launcher has not exited): expected {usd:.2f} USD subtracted')
print(f'balance minus running arms minus expected {left:.2f} USD (floor: {FLOOR:.0f})')
print(f'follow-up 13 spend so far, from {n} session record(s): {spent:.2f} USD; with this arm {spent + est:.2f} USD '
      f'(cap: {CAP:.0f})')
vm = sorted((float(p['expected_vm_h']) for p in rows), reverse=True)
if rows[0]['platform'] in ('v5e', 'v6e'):
  hours = sum(vm)
else:
  slots = [0.0] * L4_SLOTS
  for h in vm:
    heapq.heapreplace(slots, slots[0] + h)
  hours = max(slots)
end = now + dt.timedelta(hours=hours)
print(f'expected duration {hours:.1f} h ({len(vm)} part(s)): from {now:%Y-%m-%d %H:%M} UTC to about {end:%Y-%m-%d %H:%M} UTC '
      f'(finish by 2026-11-30 24:00 UTC; launch by 2026-11-25)')
reasons = []
if left < FLOOR:
  reasons.append(f'balance minus running arms minus expected {left:.2f} < {FLOOR:.0f} USD')
if spent + est > CAP:
  reasons.append(f'spend of follow-up 13 with this arm {spent + est:.2f} > {CAP:.0f} USD')
if now >= LAUNCH_BY:
  reasons.append('launch after 2026-11-25 (row item h)')
if end > FINISH_BY:
  reasons.append('its runs cannot finish by 2026-11-30 (row item h)')
print('GO' if not reasons else 'NO GO: ' + '; '.join(reasons))
sys.exit(0 if not reasons else 1)
PY
