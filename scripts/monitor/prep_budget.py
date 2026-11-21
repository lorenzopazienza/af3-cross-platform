"""Budget rules of follow-ups 17 to 21 (labels dtp, tri, rcf, aut, xvm; notes/analysis_plan.md section 10, items f and g
of each specification row), read-only. Generalisation of scripts/monitor/rcx_budget.py to the five labels (its clock
and event reading come from scripts/monitor/fu14_budget.py, imported unchanged).

    python3 scripts/monitor/prep_budget.py check LABEL WRAPPER BALANCE_USD   # before each wrapper
    python3 scripts/monitor/prep_budget.py guard PLAN                        # before each part (the guarded launcher)
    python3 scripts/monitor/prep_budget.py spent LABEL                       # spend so far of the follow-up (USD)
    python3 scripts/monitor/prep_budget.py spent all                         # of the five together
    python3 scripts/monitor/prep_budget.py costs LABEL|all                   # expected, realistic, worst, cap

Fallback on demand after one preemption (Lorenzo's decision of 2026-11-21: speed first): a part whose Spot VM is
preempted or lost once is relaunched on demand (PROVISIONING=standard on its line, written by the retry wrapper
prep_retry_arm.sh and by the supervisor, outside the repository) where the project has an on-demand quota (L4, A100,
v6e); the project has no on-demand H100 quota (console, 2026-11-21), so an H100 part is relaunched on Spot again. A v6e
part that obtains no Flex-start VM in its zones (each request valid REQUEST_VALID_FOR on its line) is relaunched on
demand as well. The worst case of a part therefore includes a fallback allowance: worst with fallback = its planned
worst case (the VM lost at the end of its worst job) plus LOSS_WORST[platform] times its worst case at the fallback's
price (on demand; Spot for the H100), LOSS_WORST the share of parts assumed to fall back at worst (L4 0.75, A100 0.5,
H100 0.25, v6e 1.0; the shares of parts that lost a Spot VM in the run records up to 2026-11-15 were about 0.5, 0.25
and 0.1, and the v6e lacked Flex-start capacity for hours in follow-up 13). LOSS_REAL (0.5, 0.25, 0.1, 0.5) gives the
realistic estimate: the VM lost halfway through the expected job, the part rerun at the fallback's expected cost.
Caps (fixed in each row, independent of the balance, which is read at launch): the worst case with fallback of every
part of the follow-up, times 1.15, rounded up to 5 USD (CAP). Floor of the balance: 25 USD.
check (scripts/monitor/credit_check_prep.sh, before each wrapper): GO if (1) for a wrapper of part B, every part A
part of the same follow-up has a complete session and no session of the follow-up is running; (2) the balance read,
minus what every other running wrapper of the five follow-ups may still spend (its worst case with fallback less its
spend), minus this wrapper's worst case with fallback, stays at or above 25 USD; (3) the follow-up's spend so far, plus
what its other running wrappers may still spend, plus this wrapper's worst case with fallback, stays within the
follow-up's cap; (4) no stop marker of the
follow-up exists; (5) the dates: launch on or before LAUNCH_BY, expected end on or before FINISH_BY.
guard (scripts/monitor/prep_guarded_launch.sh, the LAUNCHER of every wrapper, before each part): NO GO if the stop
marker results/<label>_cap_stop.txt exists, or if the projected spend of the follow-up (the spend so far, plus what
every other part still needs, plus this part's expected cost) exceeds its cap, or after LAUNCH_BY, or for a part of
part B while a part A part of the same follow-up has no complete session; GO otherwise.
What a part still needs: 0 if a session of it ended with launcher exit 0; for a session still running, its expected
cost minus what it has spent (at least 0); for a part not started or whose sessions failed, its expected cost if its
wrapper has a session (it will be launched or relaunched), and 0 on the platform of the guarded part (the guard
decides each of them as it comes). Spend of a session: VM running to deletion (or the VM already gone, or the
launcher's exit, or now while it runs) at the price of its provisioning (harness/plans/<label>_parts.csv,
usd_per_h_planned for the planned provisioning, usd_per_h_on_demand otherwise); the guarded part is priced on demand
when its line sets PROVISIONING=standard (a fallback).
NOW=YYYY-MM-DDTHH:MM:SSZ replaces the clock and RESULTS_AF3 the results folder (tests). Prints GO or NO GO with the
reasons; exit 0 for GO, 1 for NO GO, 2 for a usage error. Standard library only; reads only local files.
"""
import csv
import datetime as dt
import heapq
import math
import os
import pathlib
import re
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'scripts' / 'monitor'))
import fu14_budget as fb  # noqa: E402

FLOOR = 25.0
LAUNCH_BY = dt.datetime(2026, 12, 7)          # launch on or before 2026-12-06, UTC
FINISH_BY = dt.datetime(2026, 12, 9)          # the end of 2026-12-08, UTC
# Share of parts assumed to lose a Spot (or Flex-start) VM once and to be rerun on demand: at worst (budget rules and
# caps) and realistic (the estimate of COSTI_E_CALENDARIO.md); see the docstring.
LOSS_WORST = {'l4': 0.75, 'a100': 0.5, 'h100': 0.25, 'v6e': 1.0}
LOSS_REAL = {'l4': 0.5, 'a100': 0.25, 'h100': 0.1, 'v6e': 0.5}
OD_FALLBACK = ('l4', 'a100', 'v6e')        # platforms with an on-demand quota; the H100 falls back to Spot again
# label: follow-up, cap (USD: ceil5(1.15 x the worst case with fallback of all its parts)), arms of part A and part B,
# wrappers (name: arms), slots per platform (the MAX_<PLATFORM> of the launch order with GPUS_ALL_REGIONS 32, 2026-11-21)
LABELS = {
    'dtp': {'fu': 17, 'cap': 590.0, 'A': ('V6fd', 'V6fd_r2', 'Gfd'), 'B': ('Afd', 'Hfd'),
            'wrappers': {'v6e': ('V6fd', 'V6fd_r2'), 'l4': ('Gfd',), 'a100': ('Afd',), 'h100': ('Hfd',)},
            'slots': {'v6e': 4, 'l4': 20, 'a100': 12, 'h100': 8}},
    'tri': {'fu': 18, 'cap': 340.0, 'A': ('HT', 'HT_r2'), 'B': ('AT_r2',),
            'wrappers': {'h100': ('HT', 'HT_r2'), 'a100': ('AT_r2',)}, 'slots': {'a100': 12, 'h100': 12}},
    'rcf': {'fu': 19, 'cap': 275.0, 'A': ('Af_r2b', 'Hf_r2b'), 'B': (),
            'wrappers': {'a100': ('Af_r2b',), 'h100': ('Hf_r2b',)}, 'slots': {'a100': 12, 'h100': 10}},
    'aut': {'fu': 20, 'cap': 115.0, 'A': ('Gna',), 'B': (), 'wrappers': {'l4': ('Gna',)}, 'slots': {'l4': 24}},
    'xvm': {'fu': 21, 'cap': 170.0, 'A': ('G_r3',), 'B': ('A_r3',),
            'wrappers': {'l4': ('G_r3',), 'a100': ('A_r3',)}, 'slots': {'l4': 12, 'a100': 12}},
}
SESSION = re.compile(r'^\d{8}T\d{6}Z_(v6e|l4|a100|h100)_((dtp|tri|rcf|aut|xvm)_[a-z0-9_]+_\d\d)$')


def marker(label):
  return pathlib.Path(os.environ.get('PREP_MARKER') or REPO / 'results' / f'{label}_cap_stop.txt')


def parts(label):
  return {p['part']: p for p in csv.DictReader(open(REPO / 'harness' / 'plans' / f'{label}_parts.csv'))}


def all_parts():
  out = {}
  for label in LABELS:
    if (REPO / 'harness' / 'plans' / f'{label}_parts.csv').is_file():
      out.update(parts(label))
  return out


def sessions(ps, t):
  """[{name, part, spent, running, ok}] of every session of a part in ps (fu14_budget.events reads the events)."""
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


def fallback_col(p, kind):
  """The part's cost at the fallback's price: on demand where OD_FALLBACK, else its planned provisioning."""
  return float(p[f'{kind}_usd_on_demand' if p['platform'] in OD_FALLBACK else f'{kind}_usd_planned'])


def worst_fb(p):
  """Worst case of a part with the fallback allowance (docstring)."""
  return float(p['worst_usd_planned']) + LOSS_WORST[p['platform']] * fallback_col(p, 'worst')


def realistic_fb(p):
  """Realistic cost of a part: a share LOSS_REAL loses its VM halfway through the expected job and reruns on demand."""
  r = LOSS_REAL[p['platform']]
  return float(p['expected_usd_planned']) * (1 - 0.5 * r) + r * fallback_col(p, 'expected')


def cap_of(rows):
  """ceil5(1.15 x the worst case with fallback of the parts), the rule of the caps."""
  return 5.0 * math.ceil(round(1.15 * sum(worst_fb(p) for p in rows), 6) / 5)


def complete(part, ss):
  return any(s['ok'] for s in ss if s['part'] == part)


def wrapper_of(p):
  cfg = LABELS[p['label']]
  return next(w for w, arms in cfg['wrappers'].items() if p['arm'] in arms)


def still_needed(p, ss, guard_platform, started_wrappers):
  mine = [s for s in ss if s['part'] == p['part']]
  if any(s['ok'] for s in mine):
    return 0.0
  run = [s for s in mine if s['running']]
  if run:
    return max(0.0, float(p['expected_usd_planned']) - sum(s['spent'] for s in run))
  if p['platform'] == guard_platform:
    return 0.0
  if (p['label'], wrapper_of(p)) not in started_wrappers:
    return 0.0
  return float(p['expected_usd_planned'])


def hours(rows, slots):
  out = 0.0
  for pl in {p['platform'] for p in rows}:
    hs = sorted((float(p['expected_vm_h']) for p in rows if p['platform'] == pl), reverse=True)
    sl = [0.0] * slots[pl]
    for h in hs:
      heapq.heapreplace(sl, sl[0] + h)
    out = max(out, max(sl))
  return out


def running_wrappers(aps, ss):
  """{(label, wrapper): worst case less spend} of every wrapper with a running session."""
  out = {}
  for s in ss:
    if s['running']:
      p = aps[s['part']]
      out[(p['label'], wrapper_of(p))] = None
  for key in out:
    label, w = key
    rows = [p for p in aps.values() if p['label'] == label and p['arm'] in LABELS[label]['wrappers'][w]]
    spent = sum(s['spent'] for s in ss if s['part'] in {p['part'] for p in rows})
    out[key] = max(0.0, sum(worst_fb(p) for p in rows) - spent)
  return out


def check(label, wrapper, bal):
  cfg = LABELS[label]
  aps, t = all_parts(), fb.now()
  ss = sessions(aps, t)
  rows = [p for p in aps.values() if p['label'] == label and p['arm'] in cfg['wrappers'][wrapper]]
  worst = sum(worst_fb(p) for p in rows)
  est = sum(float(p['expected_usd_planned']) for p in rows)
  spent = sum(s['spent'] for s in ss if aps[s['part']]['label'] == label)
  reasons = []
  print(f'follow-up {cfg["fu"]} ({label}), wrapper {wrapper} ({", ".join(cfg["wrappers"][wrapper])}; {len(rows)} parts, '
        f'{rows[0]["provisioning"]}): expected {est:.2f} USD, worst case with fallback {worst:.2f}; TPU Builders balance {bal:.2f} USD; '
        f'follow-up {cfg["fu"]} spend so far {spent:.2f}')
  if any(p['arm'] in cfg['B'] for p in rows):
    a = [p for p in aps.values() if p['label'] == label and p['arm'] in cfg['A']]
    missing = sorted(p['part'] for p in a if not complete(p['part'], ss))
    running = sorted(s['name'] for s in ss if s['running'] and aps[s['part']]['label'] == label)
    print(f'part A: {len(a) - len(missing)} of {len(a)} parts complete; follow-up {cfg["fu"]} sessions running: '
          f'{len(running)}')
    if missing:
      reasons.append(f'part A has not ended ({len(missing)} part(s) without a complete session)')
    if running:
      reasons.append(f'{len(running)} follow-up {cfg["fu"]} session(s) still running')
  others = {k: v for k, v in running_wrappers(aps, ss).items() if k != (label, wrapper)}
  for (ol, ow), need in sorted(others.items()):
    print(f'running wrapper {ol} {ow}: may still spend up to {need:.2f} USD (its worst case with fallback less its '
          f'spend), subtracted')
  left = bal - sum(others.values()) - worst
  proj = spent + sum(v for (ol, _), v in others.items() if ol == label) + worst
  print(f'balance minus what is still to come {left:.2f} USD (floor {FLOOR:.0f}); follow-up {cfg["fu"]} spend '
        f'projected with this wrapper\'s worst case with fallback {proj:.2f} USD (cap {cfg["cap"]:.0f})')
  if left < FLOOR:
    reasons.append(f'balance minus what is still to come {left:.2f} < {FLOOR:.0f} USD')
  if proj > cfg['cap']:
    reasons.append(f'projected spend {proj:.2f} > {cfg["cap"]:.0f} USD')
  if marker(label).is_file():
    reasons.append(f'the cap stop marker {marker(label).name} exists (Lorenzo decides)')
  h = hours(rows, cfg['slots'])
  end = t + dt.timedelta(hours=h)
  print(f'expected duration {h:.1f} h: from {t:%Y-%m-%d %H:%M} UTC to about {end:%Y-%m-%d %H:%M} UTC '
        f'(launch by {LAUNCH_BY - dt.timedelta(days=1):%Y-%m-%d}, finish by {FINISH_BY - dt.timedelta(days=1):%Y-%m-%d} '
        f'24:00 UTC)')
  if t >= LAUNCH_BY:
    reasons.append(f'launch after {LAUNCH_BY - dt.timedelta(days=1):%Y-%m-%d} (row item h)')
  if end > FINISH_BY:
    reasons.append(f'its runs cannot finish by {FINISH_BY - dt.timedelta(days=1):%Y-%m-%d} (row item h)')
  print('GO' if not reasons else 'NO GO: ' + '; '.join(reasons))
  return 0 if not reasons else 1


def guard(plan):
  path = pathlib.Path(plan)
  label, name = path.parent.name, path.stem.lower()
  if label not in LABELS or name not in parts(label):
    print(f'guard: {plan} is not a part of follow-ups 17 to 21; NO GO')
    return 1
  cfg = LABELS[label]
  aps, t = all_parts(), fb.now()
  me = aps[name]
  ss = sessions(aps, t)
  ss_l = [s for s in ss if aps[s['part']]['label'] == label]
  started = {(aps[s['part']]['label'], wrapper_of(aps[s['part']])) for s in ss}
  spent = sum(s['spent'] for s in ss_l)
  need = sum(still_needed(p, ss_l, me['platform'], started) for p in aps.values()
             if p['label'] == label and p['part'] != name)
  prov = os.environ.get('PROVISIONING') or ('spot' if os.environ.get('SPOT', '1') == '1' else 'standard')
  if os.environ.get('V6E_PROVISIONING') and me['platform'] == 'v6e':
    prov = os.environ['V6E_PROVISIONING']
  mine = float(me['expected_usd_planned'] if prov == me['provisioning'] else me['expected_usd_on_demand'])
  proj = spent + need + mine
  print(f'guard {name} ({prov}): follow-up {cfg["fu"]} spend so far {spent:.2f} USD, still needed by the other parts '
        f'{need:.2f}, this part {mine:.2f}: projected {proj:.2f} USD (cap {cfg["cap"]:.0f})')
  reasons = []
  if marker(label).is_file():
    reasons.append(f'the cap stop marker {marker(label)} exists (Lorenzo decides)')
  if proj > cfg['cap']:
    reasons.append(f'projected spend {proj:.2f} > {cfg["cap"]:.0f} USD')
  if me['arm'] in cfg['B']:
    missing = [p['part'] for p in aps.values() if p['label'] == label and p['arm'] in cfg['A']
               and not complete(p['part'], ss_l)]
    if missing:
      reasons.append(f'part A has not ended ({len(missing)} part(s) without a complete session)')
  if t >= LAUNCH_BY:
    reasons.append(f'launch after {LAUNCH_BY - dt.timedelta(days=1):%Y-%m-%d} (row item h)')
  print('GO' if not reasons else 'NO GO: ' + '; '.join(reasons))
  return 0 if not reasons else 1


def spent_of(label):
  aps = all_parts()
  ss = sessions(aps, fb.now())
  return sum(s['spent'] for s in ss if label == 'all' or aps[s['part']]['label'] == label)


def costs(which):
  """Per follow-up and part of the row: expected (no preemption), realistic with fallback, worst without preemption,
  worst with fallback, every part lost once at the end of its worst job and rerun on demand at worst, and the cap."""
  print('| FU | label | part | parts | expected | realistic with fallback | worst, no preemption | worst with fallback '
        '| every part lost once | cap |')
  print('|---|---|---|---|---|---|---|---|---|---|')
  tot = [0.0] * 5
  for label, cfg in LABELS.items():
    if which not in ('all', label):
      continue
    ps = list(parts(label).values())
    for rp in sorted({p['row_part'] for p in ps}):
      q = [p for p in ps if p['row_part'] == rp]
      v = [sum(float(p['expected_usd_planned']) for p in q), sum(realistic_fb(p) for p in q),
           sum(float(p['worst_usd_planned']) for p in q), sum(worst_fb(p) for p in q),
           sum(float(p['worst_usd_planned']) + fallback_col(p, 'worst') for p in q)]
      tot = [a + b for a, b in zip(tot, v)]
      print(f'| {cfg["fu"]} | {label} | {rp} | {len(q)} | ' + ' | '.join(f'{x:.2f}' for x in v)
            + f' | {cfg["cap"]:.0f} (rule {cap_of(ps):.0f}) |')
  print('| all | | | | ' + ' | '.join(f'{x:.2f}' for x in tot) + ' | |')
  return 0


def main(argv):
  if (len(argv) == 4 and argv[0] == 'check' and argv[1] in LABELS and argv[2] in LABELS[argv[1]]['wrappers']
          and re.fullmatch(r'[0-9]+(\.[0-9]+)?', argv[3])):
    return check(argv[1], argv[2], float(argv[3]))
  if len(argv) == 2 and argv[0] == 'guard':
    return guard(argv[1])
  if len(argv) == 2 and argv[0] == 'spent' and (argv[1] in LABELS or argv[1] == 'all'):
    print(f'{spent_of(argv[1]):.2f}')
    return 0
  if len(argv) == 2 and argv[0] == 'costs' and (argv[1] in LABELS or argv[1] == 'all'):
    return costs(argv[1])
  print('usage: prep_budget.py check dtp|tri|rcf|aut|xvm WRAPPER BALANCE_USD | guard PLAN | spent LABEL|all | '
        'costs LABEL|all',
        file=sys.stderr)
  return 2


if __name__ == '__main__':
  sys.exit(main(sys.argv[1:]))
