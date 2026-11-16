"""Follow-up 16 (label hlo) budget rules (notes/analysis_plan.md section 10, row of 2026-11-16, items f and g), read-only.
Sibling of scripts/monitor/rcx_budget.py (fu14_budget's clock and event reading, imported unchanged).

    python3 scripts/monitor/hlo_budget.py check BALANCE_USD     # before the wrapper (both parts)
    python3 scripts/monitor/hlo_budget.py guard PLAN            # before each part (scripts/monitor/hlo_guarded_launch.sh)
    python3 scripts/monitor/hlo_budget.py spent                 # spend so far of follow-up 16 (USD)

Cap of follow-up 16: 15 USD, within the TPU Builders balance read on 2026-11-16 (40.47 USD) less the floor of 25 USD.
check (scripts/monitor/credit_check_hlo.sh): GO if the balance read, minus the lifetime bound of every part not yet
complete (the most a part can cost before its launcher deletes the VM: deadline + 30 min, v5e rounded up to whole
hours, at the on-demand price), stays at or above 25 USD; and the spend so far plus that bound stays within 15 USD;
and the dates (launch by 2026-11-17, expected end by 2026-11-18, UTC).
guard (the LAUNCHER of the wrapper, before each part): NO GO if the stop marker results/hlo_cap_stop.txt exists, or
if the projected spend of follow-up 16 (the spend so far, plus what the other part still needs, plus this part's worst
case) exceeds 15 USD, or after the launch date; GO otherwise. What the other part still needs: 0 if a session of it
ended with launcher exit 0; for a session still running, its worst case minus what it has spent (at least 0); else its
worst case. Spend of a session: VM running to deletion (or the VM already gone, or the launcher's exit, or now while
it runs) at the on-demand price of harness/plans/hlo_parts.csv.
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

MARKER = pathlib.Path(os.environ.get('HLO_MARKER') or REPO / 'results' / 'hlo_cap_stop.txt')
PARTS = REPO / 'harness' / 'plans' / 'hlo_parts.csv'
BALANCE_READ = 40.47                           # TPU Builders credit, read by Lorenzo on 2026-11-16
FLOOR, CAP = 25.0, 15.0
LAUNCH_BY = dt.datetime(2026, 11, 18)          # launch on or before 2026-11-17, UTC
FINISH_BY = dt.datetime(2026, 11, 19)          # the end of 2026-11-18, UTC
SESSION = re.compile(r'^\d{8}T\d{6}Z_(v5e|v6e)_(hlo_[a-z0-9_]+_\d\d)$')


def parts():
  return {p['part']: p for p in csv.DictReader(open(PARTS))}


def sessions(ps, t):
  """[{name, part, spent, running, ok}] of every hlo session of a part of hlo_parts.csv."""
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
    start = next((v for k, v in ev.items() if k.startswith('vm:')), None)
    end = ev.get('delete:done') or ev.get('delete:the') or ev.get('launcher:exit') or t
    spent = (end - start).total_seconds() / 3600 * float(p['usd_per_h_on_demand']) if start else 0.0
    out.append({'name': s.name, 'part': m.group(2), 'spent': spent, 'running': 'launcher:exit' not in ev,
                'ok': code == '0'})
  return out


def complete(p, ss):
  return any(s['ok'] for s in ss if s['part'] == p['part'])


def still_needed(p, ss):
  mine = [s for s in ss if s['part'] == p['part']]
  if any(s['ok'] for s in mine):
    return 0.0
  run = [s for s in mine if s['running']]
  if run:
    return max(0.0, float(p['worst_usd_planned']) - sum(s['spent'] for s in run))
  return float(p['worst_usd_planned'])


def spent_total(ps, ss):
  return sum(s['spent'] for s in ss if s['part'] in ps)


def check(bal):
  ps, t = parts(), fb.now()
  ss = sessions(ps, t)
  todo = [p for p in ps.values() if not complete(p, ss)]
  bound = sum(float(p['lifetime_usd_planned']) for p in todo)
  worst = sum(float(p['worst_usd_planned']) for p in todo)
  spent = spent_total(ps, ss)
  print(f'follow-up 16: {len(todo)} of {len(ps)} parts to run, on demand: worst case {worst:.2f} USD, lifetime bound '
        f'{bound:.2f}; TPU Builders balance {bal:.2f} USD; follow-up 16 spend so far {spent:.2f}')
  left, proj = bal - bound, spent + bound
  print(f'balance minus the lifetime bound {left:.2f} USD (floor {FLOOR:.0f}); spend projected with the lifetime bound '
        f'{proj:.2f} USD (cap {CAP:.0f})')
  reasons = []
  if not todo:
    reasons.append('every part already has a complete session')
  if left < FLOOR:
    reasons.append(f'balance minus the lifetime bound {left:.2f} < {FLOOR:.0f} USD')
  if proj > CAP:
    reasons.append(f'projected spend {proj:.2f} > {CAP:.0f} USD')
  if MARKER.is_file():
    reasons.append(f'the cap stop marker {MARKER.name} exists (Lorenzo decides)')
  h = max([float(p['worst_vm_h']) for p in todo] or [0.0])
  end = t + dt.timedelta(hours=h)
  print(f'worst duration {h:.2f} h (both parts at once): from {t:%Y-%m-%d %H:%M} UTC to about {end:%Y-%m-%d %H:%M} UTC '
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
    print(f'guard: {plan} is not a follow-up 16 part; NO GO')
    return 1
  me = ps[name]
  ss = sessions(ps, t)
  spent = spent_total(ps, ss)
  need = sum(still_needed(p, ss) for p in ps.values() if p['part'] != name)
  mine = float(me['worst_usd_planned'])
  proj = spent + need + mine
  print(f'guard {name} (on demand): follow-up 16 spend so far {spent:.2f} USD, still needed by the other part '
        f'{need:.2f}, this part (worst) {mine:.2f}: projected {proj:.2f} USD (cap {CAP:.0f})')
  reasons = []
  if MARKER.is_file():
    reasons.append(f'the cap stop marker {MARKER} exists (Lorenzo decides)')
  if proj > CAP:
    reasons.append(f'projected spend {proj:.2f} > {CAP:.0f} USD')
  if t >= LAUNCH_BY:
    reasons.append('launch after 2026-11-17 (row item h)')
  print('GO' if not reasons else 'NO GO: ' + '; '.join(reasons))
  return 0 if not reasons else 1


def main(argv):
  if len(argv) == 2 and argv[0] == 'check' and re.fullmatch(r'[0-9]+(\.[0-9]+)?', argv[1]):
    return check(float(argv[1]))
  if len(argv) == 2 and argv[0] == 'guard':
    return guard(argv[1])
  if argv == ['spent']:
    ps = parts()
    print(f'{spent_total(ps, sessions(ps, fb.now())):.2f}')
    return 0
  print('usage: hlo_budget.py check BALANCE_USD | guard PLAN | spent', file=sys.stderr)
  return 2


if __name__ == '__main__':
  sys.exit(main(sys.argv[1:]))
