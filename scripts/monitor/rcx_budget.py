"""Follow-up 15 (label rcx) budget rules (notes/analysis_plan.md section 10, row of 2026-11-14, items f and g), read-only.
Sibling of scripts/monitor/fu14_budget.py (its clock and event reading, imported unchanged).

    python3 scripts/monitor/rcx_budget.py check WRAPPER BALANCE_USD     # WRAPPER: a100, h100 (part A4), l4 (part A3)
    python3 scripts/monitor/rcx_budget.py guard PLAN                    # before each part of every wrapper
    python3 scripts/monitor/rcx_budget.py spent                         # spend so far of follow-up 15 (USD)

Cap of follow-up 15: 62 USD, the TPU Builders balance read on 2026-11-14 (87 USD) minus the floor of 25 USD.
check (scripts/monitor/credit_check_rcx.sh, before each wrapper):
  a100, h100 (part A4, Af r2 and Hf r2): GO if the balance read, minus the spend of follow-up 15 still to come of the
  other part A4 wrapper when it is running (its worst case less what it has spent), minus this wrapper's worst case,
  stays at or above 25 USD; and the spend so far plus that and this worst case stays within 62 USD; and the dates.
  l4 (part A3, GT r2): GO only if part A4 has ended (every Af r2 and Hf r2 part has a session whose launcher exited 0
  and no follow-up 15 session is running); and the balance minus the worst case of part A3 stays at or above 25 USD;
  and the spend so far of follow-up 15 plus the worst case of part A3 stays within 62 USD; and the dates. Part A4 is
  never held back for part A3: part A3 starts only from what part A4 has left.
guard (scripts/monitor/rcx_guarded_launch.sh, the LAUNCHER of every wrapper, before each part): NO GO if the stop
  marker results/rcx_cap_stop.txt exists, or if the projected spend of follow-up 15 (the spend so far, plus what every
  other part still needs, plus this part's expected cost) exceeds 62 USD, or after the launch date, or for a part A3
  part while a part A4 part has no complete session; GO otherwise.
What a part still needs: 0 if a session of it ended with launcher exit 0; for a session still running, its expected
cost minus what it has spent (at least 0); for a part not started or whose sessions failed (so it will be
relaunched), its expected cost if its wrapper launches it on its own (a part A4 part of the other GPU, or a part A3
part once part A3 has a session), and 0 on the platform of the guarded part (the guard decides each of them as it
comes). Spend of a session: VM running to deletion (or the VM already gone, or the launcher's exit, or now while it
runs) at the price of its provisioning (harness/plans/rcx_parts.csv, usd_per_h_planned for Spot,
usd_per_h_on_demand otherwise), as fu14_budget.sessions reads it.
NOW=YYYY-MM-DDTHH:MM:SSZ replaces the clock and RESULTS_AF3 the results folder (tests). Prints GO or NO GO with the
reasons; exit 0 for GO, 1 for NO GO, 2 for a usage error. Standard library only; reads only local files.
"""
import csv
import datetime as dt
import os
import pathlib
import re
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'scripts' / 'monitor'))
import fu14_budget as fb  # noqa: E402

MARKER = pathlib.Path(os.environ.get('RCX_MARKER') or REPO / 'results' / 'rcx_cap_stop.txt')
PARTS = REPO / 'harness' / 'plans' / 'rcx_parts.csv'
BALANCE_READ = 87.00                           # TPU Builders credit, read by Lorenzo on 2026-11-14
FLOOR, CAP = 25.0, 62.0
LAUNCH_BY = dt.datetime(2026, 11, 18)          # launch on or before 2026-11-17, UTC
FINISH_BY = dt.datetime(2026, 11, 19)          # the end of 2026-11-18, UTC
A4, A3 = ('Af_r2', 'Hf_r2'), ('GT_r2',)
WRAPPERS = {'a100': ('Af_r2',), 'h100': ('Hf_r2',), 'l4': A3}
SLOTS = {'a100': 4, 'h100': 3, 'l4': 8}
SESSION = re.compile(r'^\d{8}T\d{6}Z_(a100|h100|l4)_(rcx_[a-z0-9_]+_\d\d)$')


def parts():
  return {p['part']: p for p in csv.DictReader(open(PARTS))}


def sessions(ps, t):
  """[{name, part, spent, running, ok}] of every rcx session of a part of rcx_parts.csv (fu14_budget.sessions, with
  this follow-up's session names; fu14_budget.events reads the launcher's events)."""
  af3 = pathlib.Path(os.environ.get('RESULTS_AF3') or REPO / 'results' / 'af3')
  out, seen = [], set()
  for s in sorted(af3.glob('*')) + sorted((af3 / '.launcher').glob('*')):
    m = SESSION.match(s.name)
    f = s / 'launcher_events.txt'
    if not m or m.group(2) not in ps or not f.is_file() or s.name in seen:
      continue
    seen.add(s.name)
    p = ps[m.group(2)]
    ev, code = fb.events(f)
    prov = 'standard'
    for d in (s, af3 / '.launcher' / s.name):
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


def complete(p, ss):
  return any(s['ok'] for s in ss if s['part'] == p['part'])


def still_needed(p, ss, guard_platform=None, a3_started=False):
  mine = [s for s in ss if s['part'] == p['part']]
  if any(s['ok'] for s in mine):
    return 0.0
  run = [s for s in mine if s['running']]
  if run:
    return max(0.0, float(p['expected_usd_planned']) - sum(s['spent'] for s in run))
  if p['platform'] == guard_platform:
    return 0.0
  if p['arm'] in A3 and not a3_started:
    return 0.0
  return float(p['expected_usd_planned'])


def hours(rows):
  import heapq
  out = 0.0
  for pl in {p['platform'] for p in rows}:
    hs = sorted((float(p['expected_vm_h']) for p in rows if p['platform'] == pl), reverse=True)
    slots = [0.0] * SLOTS[pl]
    for h in hs:
      heapq.heapreplace(slots, slots[0] + h)
    out = max(out, max(slots))
  return out


def spent_total(ps, ss):
  return sum(s['spent'] for s in ss if s['part'] in ps)


def check(wrapper, bal):
  ps, t = parts(), fb.now()
  ss = sessions(ps, t)
  rows = [p for p in ps.values() if p['arm'] in WRAPPERS[wrapper]]
  worst = sum(float(p['worst_usd_planned']) for p in rows)
  est = sum(float(p['expected_usd_planned']) for p in rows)
  spent = spent_total(ps, ss)
  reasons = []
  print(f'wrapper {wrapper} ({", ".join(WRAPPERS[wrapper])}; {len(rows)} parts, {rows[0]["provisioning"]}): expected '
        f'{est:.2f} USD, worst case {worst:.2f}; TPU Builders balance {bal:.2f} USD; follow-up 15 spend so far {spent:.2f}')
  if wrapper == 'l4':
    a4 = [p for p in ps.values() if p['arm'] in A4]
    missing = sorted(p['part'] for p in a4 if not complete(p, ss))
    running = sorted(s['name'] for s in ss if s['running'])
    print(f'part A4: {len(a4) - len(missing)} of {len(a4)} parts complete; follow-up 15 sessions running: {len(running)}')
    if missing:
      reasons.append(f'part A4 has not ended ({len(missing)} part(s) without a complete session: {" ".join(missing)})')
    if running:
      reasons.append(f'{len(running)} follow-up 15 session(s) still running')
    left, proj = bal - worst, spent + worst
  else:
    other = [w for w in ('a100', 'h100') if w != wrapper][0]
    orows = [p for p in ps.values() if p['arm'] in WRAPPERS[other]]
    orun = any(s['running'] for s in ss if ps[s['part']]['arm'] in WRAPPERS[other])
    oneed = 0.0
    if orun:
      ospent = sum(s['spent'] for s in ss if ps[s['part']]['arm'] in WRAPPERS[other])
      oneed = max(0.0, sum(float(p['worst_usd_planned']) for p in orows) - ospent)
      print(f'running wrapper {other}: still needs up to {oneed:.2f} USD (its worst case less its spend), subtracted')
    left, proj = bal - oneed - worst, spent + oneed + worst
  print(f'balance minus what is still to come {left:.2f} USD (floor {FLOOR:.0f}); follow-up 15 spend projected with '
        f'this wrapper\'s worst case {proj:.2f} USD (cap {CAP:.0f})')
  if left < FLOOR:
    reasons.append(f'balance minus what is still to come {left:.2f} < {FLOOR:.0f} USD')
  if proj > CAP:
    reasons.append(f'projected spend {proj:.2f} > {CAP:.0f} USD')
  if MARKER.is_file():
    reasons.append(f'the cap stop marker {MARKER.name} exists (Lorenzo decides)')
  h = hours(rows)
  end = t + dt.timedelta(hours=h)
  print(f'expected duration {h:.1f} h: from {t:%Y-%m-%d %H:%M} UTC to about {end:%Y-%m-%d %H:%M} UTC '
        f'(launch by 2026-11-17, finish by 2026-11-18 24:00 UTC)')
  if t >= LAUNCH_BY:
    reasons.append('launch after 2026-11-17 (row item h)')
  if end > FINISH_BY:
    reasons.append('its runs cannot finish by 2026-11-18 (row item h)')
  print('GO' if not reasons else 'NO GO: ' + '; '.join(reasons))
  return 0 if not reasons else 1


def guard(plan):
  ps, t = parts(), fb.now()
  name = pathlib.Path(plan).stem.lower()
  if name not in ps:
    print(f'guard: {plan} is not a follow-up 15 part; NO GO')
    return 1
  me = ps[name]
  ss = sessions(ps, t)
  a3_started = any(ps[s['part']]['arm'] in A3 for s in ss)
  spent = spent_total(ps, ss)
  need = sum(still_needed(p, ss, me['platform'], a3_started) for p in ps.values() if p['part'] != name)
  prov = os.environ.get('PROVISIONING') or ('spot' if os.environ.get('SPOT', '1') == '1' else 'standard')
  mine = float(me['expected_usd_planned'] if prov == me['provisioning'] else me['expected_usd_on_demand'])
  proj = spent + need + mine
  print(f'guard {name} ({prov}): follow-up 15 spend so far {spent:.2f} USD, still needed by the other parts '
        f'{need:.2f}, this part {mine:.2f}: projected {proj:.2f} USD (cap {CAP:.0f})')
  reasons = []
  if MARKER.is_file():
    reasons.append(f'the cap stop marker {MARKER} exists (Lorenzo decides)')
  if proj > CAP:
    reasons.append(f'projected spend {proj:.2f} > {CAP:.0f} USD')
  if me['arm'] in A3:
    missing = [p['part'] for p in ps.values() if p['arm'] in A4 and not complete(p, ss)]
    if missing:
      reasons.append(f'part A4 has not ended ({len(missing)} part(s) without a complete session)')
  if t >= LAUNCH_BY:
    reasons.append('launch after 2026-11-17 (row item h)')
  print('GO' if not reasons else 'NO GO: ' + '; '.join(reasons))
  return 0 if not reasons else 1


def main(argv):
  if len(argv) == 3 and argv[0] == 'check' and argv[1] in WRAPPERS and re.fullmatch(r'[0-9]+(\.[0-9]+)?', argv[2]):
    return check(argv[1], float(argv[2]))
  if len(argv) == 2 and argv[0] == 'guard':
    return guard(argv[1])
  if argv == ['spent']:
    ps = parts()
    print(f'{spent_total(ps, sessions(ps, fb.now())):.2f}')
    return 0
  print('usage: rcx_budget.py check a100|h100|l4 BALANCE_USD | guard PLAN | spent', file=sys.stderr)
  return 2


if __name__ == '__main__':
  sys.exit(main(sys.argv[1:]))
