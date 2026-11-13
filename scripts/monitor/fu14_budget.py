"""Follow-up 14 (label f32) budget rules (notes/analysis_plan.md section 10, row of 2026-11-12, items f and g), read-only.

    python3 scripts/monitor/fu14_budget.py check WRAPPER BALANCE_USD     # WRAPPER: v5e, a100, h100, det
    python3 scripts/monitor/fu14_budget.py guard PLAN                    # before each part of the A100 wrapper

check (scripts/monitor/credit_check_fu14.sh, before each wrapper):
  float32 wrappers (v5e: V5f and V5f r2; a100: Af; h100: Hf): GO if the TPU Builders balance read from the console,
  minus the expected cost still to come of every other float32 wrapper with a running session, minus this wrapper's
  expected cost, stays at or above the floor of 25 USD; and the projected spend of the float32 arms (below) with this
  wrapper's expected cost stays within the cap of 300 USD; and the launch is on or before 2026-11-18 (UTC) and the
  wrapper's expected end is by 2026-11-20 24:00 UTC.
  det (item b2, Ad and Hd): GO only if no float32 session is still running (the float32 arms have ended), the balance
  is at least the worst case of item b2 plus 25 USD, item b2's spend with its worst case stays within its own cap of
  62 USD, and the dates as above.
guard (scripts/monitor/fu14_guarded_launch.sh, the A100 wrapper's LAUNCHER, before each part): NO GO if the stop
  marker results/f32_cap_stop.txt exists, or if the projected spend of the float32 arms plus this part's expected cost
  (at the on-demand price when the part is launched on demand, the fallback of item f) exceeds the cap of 300 USD, or
  after the launch date; GO otherwise.
Projected spend of the float32 arms: the spend so far of every f32 session of a float32 part (VM minutes from
launcher_events.txt, VM running to delete done, or the VM already gone, or the launcher's exit, or now while it runs,
times the price of its platform for the provisioning in its launch_record.txt: Spot or on demand, the
usd_per_h_planned and usd_per_h_on_demand columns of harness/plans/fu14_parts.csv), plus what each float32 part still
needs: 0 if a session of it ended with launcher exit 0; for a session still running, its expected cost minus what it
has spent (at least 0); for a part not started (or whose sessions failed, so it will be relaunched), its expected cost
if it is on v5e or the H100 (those wrappers launch every part) and 0 on the A100 (the guard decides each A100 part as
it comes). Expected costs: the expected_usd_planned column of harness/plans/fu14_parts.csv.
NOW=YYYY-MM-DDTHH:MM:SSZ replaces the clock and RESULTS_AF3 the results folder (tests). Prints GO or NO GO with the
reasons; exit 0 for GO, 1 for NO GO, 2 for a usage error. Standard library only; reads only local files.
"""
import csv
import datetime as dt
import heapq
import os
import pathlib
import re
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]
AF3 = pathlib.Path(os.environ.get('RESULTS_AF3') or REPO / 'results' / 'af3')
MARKER = pathlib.Path(os.environ.get('F32_MARKER') or REPO / 'results' / 'f32_cap_stop.txt')
PARTS = REPO / 'harness' / 'plans' / 'fu14_parts.csv'
FLOOR, CAP_F32, CAP_DET = 25.0, 300.0, 62.0
LAUNCH_BY = dt.datetime(2026, 11, 19)          # launch on or before 2026-11-18, UTC
FINISH_BY = dt.datetime(2026, 11, 21)          # the end of 2026-11-20, UTC
FLOAT32 = ('V5f', 'V5f_r2', 'Af', 'Hf')
DET = ('Ad', 'Hd')
WRAPPERS = {'v5e': ('V5f', 'V5f_r2'), 'a100': ('Af',), 'h100': ('Hf',), 'det': DET}
SLOTS = {'v5e': 4, 'a100': 5, 'h100': 5}
SESSION = re.compile(r'^\d{8}T\d{6}Z_(v5e|a100|h100)_(f32_[a-z0-9_]+_\d\d)$')


def now():
  n = os.environ.get('NOW')
  return dt.datetime.strptime(n, '%Y-%m-%dT%H:%M:%SZ') if n else dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)


def parts():
  return {p['part']: p for p in csv.DictReader(open(PARTS))}


def events(f):
  ev, code = {}, None
  for line in f.read_text().splitlines():
    m = re.match(r'(\S+)\s+(\w+): (\w+)(.*)', line)
    if m:
      ev.setdefault(f'{m.group(2)}:{m.group(3)}', dt.datetime.strptime(m.group(1), '%Y-%m-%dT%H:%M:%SZ'))
      if m.group(2) == 'launcher' and m.group(3) == 'exit':
        code = (m.group(4).split() or [None])[-1]
  return ev, code


def sessions(ps, t):
  """[{name, part, spent, running, ok}] of every f32 session that belongs to a part of fu14_parts.csv."""
  out, seen = [], set()
  for s in sorted(AF3.glob('*')) + sorted((AF3 / '.launcher').glob('*')):
    m = SESSION.match(s.name)
    f = s / 'launcher_events.txt'
    if not m or m.group(2) not in ps or not f.is_file() or s.name in seen:
      continue
    seen.add(s.name)
    p = ps[m.group(2)]
    ev, code = events(f)
    prov = 'standard'
    for d in (s, AF3 / '.launcher' / s.name):
      rec = d / 'launch_record.txt'
      if rec.is_file():
        for line in rec.read_text().splitlines():
          if line.startswith('PROVISIONING='):
            prov = line.split('=', 1)[1]
    rate = float(p['usd_per_h_planned']) if prov == p['provisioning'] else float(p['usd_per_h_on_demand'])
    start = next((v for k, v in ev.items() if k.startswith('vm:')), None)
    end = ev.get('delete:done') or ev.get('delete:the') or ev.get('launcher:exit') or t
    spent = (end - start).total_seconds() / 3600 * rate if start else 0.0
    out.append({'name': s.name, 'part': m.group(2), 'spent': spent, 'running': 'launcher:exit' not in ev,
                'ok': code == '0'})
  return out


def still_needed(p, ss, guard_platform=None):
  """USD a float32 part still needs (see the module text)."""
  mine = [s for s in ss if s['part'] == p['part']]
  if any(s['ok'] for s in mine):
    return 0.0
  run = [s for s in mine if s['running']]
  if run:
    return max(0.0, float(p['expected_usd_planned']) - sum(s['spent'] for s in run))
  if p['platform'] == guard_platform:
    return 0.0
  return float(p['expected_usd_planned'])


def wrapper_hours(rows):
  pl = rows[0]['platform']
  hs = sorted((float(p['expected_vm_h']) for p in rows), reverse=True)
  if len({p['platform'] for p in rows}) > 1:      # det: the A100 and the H100 at once
    return max(wrapper_hours([p for p in rows if p['platform'] == q]) for q in {p['platform'] for p in rows})
  slots = [0.0] * SLOTS[pl]
  for h in hs:
    heapq.heapreplace(slots, slots[0] + h)
  return max(slots)


def check(wrapper, bal):
  ps, t = parts(), now()
  ss = sessions(ps, t)
  rows = [p for p in ps.values() if p['arm'] in WRAPPERS[wrapper]]
  est = sum(float(p['expected_usd_planned']) for p in rows)
  worst = sum(float(p['worst_usd_planned']) for p in rows)
  reasons = []
  print(f'wrapper {wrapper} ({", ".join(WRAPPERS[wrapper])}; {len(rows)} parts, {rows[0]["provisioning"]}): expected '
        f'{est:.2f} USD (worst case {worst:.2f}); TPU Builders balance read {bal:.2f} USD')
  running = {s['part'] for s in ss if s['running']}
  if wrapper == 'det':
    f32_running = sorted(s['name'] for s in ss if s['running'] and ps[s['part']]['arm'] in FLOAT32)
    spent = sum(s['spent'] for s in ss if ps[s['part']]['arm'] in DET)
    print(f'float32 sessions still running: {len(f32_running)}; item b2 spend so far {spent:.2f} USD, with its worst '
          f'case {spent + worst:.2f} USD (cap {CAP_DET:.0f}); balance needed: worst case + {FLOOR:.0f} = {worst + FLOOR:.2f} USD')
    if f32_running:
      reasons.append(f'the float32 arms have not ended ({len(f32_running)} session(s) running)')
    if bal < worst + FLOOR:
      reasons.append(f'balance {bal:.2f} < worst case of item b2 + {FLOOR:.0f} = {worst + FLOOR:.2f} USD')
    if spent + worst > CAP_DET:
      reasons.append(f'item b2 spend with its worst case {spent + worst:.2f} > {CAP_DET:.0f} USD')
  else:
    f32 = [p for p in ps.values() if p['arm'] in FLOAT32]
    spent = sum(s['spent'] for s in ss if ps[s['part']]['arm'] in FLOAT32)
    others = {}
    for p in f32:
      w = next(k for k, arms in WRAPPERS.items() if p['arm'] in arms)
      if w != wrapper and any(q['platform'] == p['platform'] and q['part'] in running for q in f32):
        others[w] = others.get(w, 0.0) + still_needed(p, ss)
    left = bal - sum(others.values()) - est
    proj = spent + sum(others.values()) + est
    for w, usd in sorted(others.items()):
      print(f'running wrapper {w}: still needs about {usd:.2f} USD, subtracted')
    print(f'balance minus running wrappers minus expected {left:.2f} USD (floor {FLOOR:.0f}); float32 spend so far '
          f'{spent:.2f} USD, projected with this wrapper {proj:.2f} USD (cap {CAP_F32:.0f})')
    if left < FLOOR:
      reasons.append(f'balance minus running wrappers minus expected {left:.2f} < {FLOOR:.0f} USD')
    if proj > CAP_F32:
      reasons.append(f'projected float32 spend {proj:.2f} > {CAP_F32:.0f} USD')
    if MARKER.is_file():
      reasons.append(f'the cap stop marker {MARKER.name} exists (Lorenzo decides)')
  hours = wrapper_hours(rows)
  end = t + dt.timedelta(hours=hours)
  print(f'expected duration {hours:.1f} h: from {t:%Y-%m-%d %H:%M} UTC to about {end:%Y-%m-%d %H:%M} UTC '
        f'(launch by 2026-11-18, finish by 2026-11-20 24:00 UTC)')
  if t >= LAUNCH_BY:
    reasons.append('launch after 2026-11-18 (row item h)')
  if end > FINISH_BY:
    reasons.append('its runs cannot finish by 2026-11-20 (row item h)')
  print('GO' if not reasons else 'NO GO: ' + '; '.join(reasons))
  return 0 if not reasons else 1


def guard(plan):
  ps, t = parts(), now()
  name = pathlib.Path(plan).stem.lower()
  if name not in ps:
    print(f'guard: {plan} is not a follow-up 14 part; NO GO')
    return 1
  me = ps[name]
  if me['arm'] in DET:
    print(f'guard: {name} is an item b2 part; the float32 cap does not apply (credit_check_fu14.sh det gates it); GO')
    return 0
  ss = sessions(ps, t)
  f32 = [p for p in ps.values() if p['arm'] in FLOAT32]
  spent = sum(s['spent'] for s in ss if ps[s['part']]['arm'] in FLOAT32)
  need = sum(still_needed(p, ss, guard_platform=me['platform']) for p in f32 if p['part'] != name)
  # this part's provisioning as cloud/af3_run.sh will take it: PROVISIONING if set, else SPOT (its default 1)
  prov = os.environ.get('PROVISIONING') or ('spot' if os.environ.get('SPOT', '1') == '1' else 'standard')
  mine = float(me['expected_usd_planned'] if prov == me['provisioning'] else me['expected_usd_on_demand'])
  proj = spent + need + mine
  print(f'guard {name} ({prov}): float32 spend so far {spent:.2f} USD, still needed by the other parts {need:.2f}, '
        f'this part {mine:.2f}: projected {proj:.2f} USD (cap {CAP_F32:.0f})')
  reasons = []
  if MARKER.is_file():
    reasons.append(f'the cap stop marker {MARKER} exists (Lorenzo decides)')
  if proj > CAP_F32:
    reasons.append(f'projected float32 spend {proj:.2f} > {CAP_F32:.0f} USD')
  if t >= LAUNCH_BY:
    reasons.append('launch after 2026-11-18 (row item h)')
  print('GO' if not reasons else 'NO GO: ' + '; '.join(reasons))
  return 0 if not reasons else 1


def main(argv):
  if len(argv) == 3 and argv[0] == 'check' and argv[1] in WRAPPERS and re.fullmatch(r'[0-9]+(\.[0-9]+)?', argv[2]):
    return check(argv[1], float(argv[2]))
  if len(argv) == 2 and argv[0] == 'guard':
    return guard(argv[1])
  print('usage: fu14_budget.py check v5e|a100|h100|det BALANCE_USD | guard PLAN', file=sys.stderr)
  return 2


if __name__ == '__main__':
  sys.exit(main(sys.argv[1:]))
