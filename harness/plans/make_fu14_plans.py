"""Plan parts of follow-up 14 (label f32; notes/analysis_plan.md section 10, row of 2026-11-12 "Follow-up experiment 14"),
sized from the run records of the bfloat16 arms on the same hardware, for the shortest time to the last result.
Sibling of harness/plans/make_fu13_plans.py: the same timings, packing helpers and prices (make_fu7_10_plans).

    python3 harness/plans/make_fu14_plans.py [--check]

Arms (item b of the row; every arm keeps the settings of its bfloat16 source arm, only its configuration's follow-up
14 settings change):
  V5f     tpu_xla_fu13_f32 on v5e   (source V5, main study)   the 60 targets, seeds 1 to 5, r1
  V5f_r2  tpu_xla_fu13_f32 on v5e                              the 10 pilot targets, seed 1, r2 (check 5)
  Af      a100_xla_fu14_f32 on A100 (source A, follow-up 3)    the 60 targets, seeds 1 to 5, r1
  Hf      h100_xla_fu14_f32 on H100 (source H, follow-up 9)    the 60 targets, seeds 1 to 5, r1
  Ad      a100_xla_det on A100      (item b2)                  the 10 pilot targets, seeds 1 to 5, r1 and r2
  Hd      h100_xla_det on H100      (item b2)                  the same
Sizing (item g): per token bucket, the median wall of the fresh processes of the source arm (V5: main study; A:
follow-up 3; H: follow-up 9, its own sessions) times a factor, expected and worst: float32 1.58 on every platform
(follow-up 13's ratio of median wall time), worst 3.0 on v5e, 3.8 on the A100 and 3.6 on the H100 (item g: the
matmul share of the bfloat16 time from follow-up 13's L4 ratio, 1 + p (r - 1) with r the float32 to bfloat16 matmul
time ratio, 16 on the A100 and 15 on the H100); deterministic 1.12 and 1.5 (Gd against G, main study). Per-part
overheads: the source arm's median provisioning, setup and teardown. Parts hold one target's runs (r1 and r2 of the
deterministic arms on one VM); each arm is cut into the smallest number of parts that is a multiple of its
platform's slots (SLOTS: v5e 4, A100 5, H100 5; V5f r2 one part) and keeps every part at most 8 h (TPU) or 4 h (GPU)
of job at the worst factor; targets are dealt longest first, each to the lightest part. Deadline ceil(worst job min
x 1.25 + 20). Prices (cloud/prices.csv, make_fu7_10_plans.price, the highest row among the launcher's regions): v5e
on demand; A100 and H100 Spot (item f; the on-demand price is listed beside them).
Writes harness/plans/f32/*.yaml, harness/plans/fu14_parts.csv, harness/plans/fu14_table.md and the parts files
results/parts_f32_v5e.txt, parts_f32_a100.txt, parts_f32_h100.txt, parts_f32_det.txt and the reserve files
parts_f32_v5e_ew4a.txt (the v5e parts with ZONE=europe-west4-a) and parts_f32_a100_ondemand.txt (the Af parts with
PROVISIONING=standard), all outside git. --check writes nothing and exits 1 if a tracked file differs.
"""
import argparse
import csv
import heapq
import io
import json
import math
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]
PLANS = REPO / 'harness' / 'plans'
sys.path.insert(0, str(PLANS))
import make_fu7_10_plans as m7  # noqa: E402

LABEL = 'f32'
LAUNCH_BY, FINISH_BY = '2026-11-18', '2026-11-20'
SPEC = 'notes/analysis_plan.md section 10, row of 2026-11-12 (follow-up experiment 14)'
PROVISIONING = {'v5e': 'standard', 'a100': 'spot', 'h100': 'spot'}
SLOTS = {'v5e': 4, 'a100': 5, 'h100': 5}
F32_FACTORS = {'v5e': (1.58, 3.0), 'a100': (1.58, 3.8), 'h100': (1.58, 3.6)}
DET_FACTORS = (1.12, 1.5)
FLOAT32_ARMS = ('V5f', 'V5f_r2', 'Af', 'Hf')
DET_ARMS = ('Ad', 'Hd')
# arm: (source arm, platform, config, targets ('all' or 'pilot'), seeds, fresh_reps, rep_start, what)
ARMS = {
    'V5f': ('V5', 'v5e', 'tpu_xla_fu13_f32', 'all', [1, 2, 3, 4, 5], 1, 1, 'V5f r1'),
    'V5f_r2': ('V5', 'v5e', 'tpu_xla_fu13_f32', 'pilot', [1], 1, 2, 'V5f r2 on the pilot targets (check 5)'),
    'Af': ('A', 'a100', 'a100_xla_fu14_f32', 'all', [1, 2, 3, 4, 5], 1, 1, 'Af r1'),
    'Hf': ('H', 'h100', 'h100_xla_fu14_f32', 'all', [1, 2, 3, 4, 5], 1, 1, 'Hf r1'),
    'Ad': ('A', 'a100', 'a100_xla_det', 'pilot', [1, 2, 3, 4, 5], 2, 1, 'Ad r1 and r2 (item b2)'),
    'Hd': ('H', 'h100', 'h100_xla_det', 'pilot', [1, 2, 3, 4, 5], 2, 1, 'Hd r1 and r2 (item b2)'),
}
ORDER = list(ARMS)
SOURCE_TEXT = {'V5': 'V5 (main study)', 'A': 'A (follow-up 3)', 'H': 'H (follow-up 9)'}


def h_sessions():
  return [l.split('#')[0].strip() for l in (REPO / 'scripts' / 'analysis_runs' / 'h100' / 'sessions.txt').read_text().splitlines()
          if l.split('#')[0].strip()]


def timings(source, manifest):
  """make_fu7_10_plans.timings; for H from follow-up 9's own sessions (H and H r2), not the A-based estimate."""
  if source != 'H':
    return m7.timings(source, manifest)
  orig = m7.sessions
  m7.sessions = lambda _source: h_sessions()
  try:
    return m7.timings('H_from_follow_up_9', manifest)
  finally:
    m7.sessions = orig


def factors(arm):
  return DET_FACTORS if arm in DET_ARMS else F32_FACTORS[ARMS[arm][1]]


def deal(items, k):
  """Targets longest first, each to the lightest of k parts."""
  bins = [{'items': [], 'load': 0.0} for _ in range(k)]
  for name, sec in sorted(items, key=lambda x: (-x[1], x[0])):
    b = min(bins, key=lambda b: (b['load'], len(b['items'])))
    b['items'].append(name)
    b['load'] += sec
  return bins


def split(arm, items, cap):
  step = 1 if arm == 'V5f_r2' else SLOTS[ARMS[arm][1]]
  k = step
  while True:
    bins = deal(items, k)
    if max(b['load'] for b in bins) <= cap or k >= len(items):
      return [b for b in bins if b['items']]
    k += step


def plan_text(arm, i, n, ts, deadline, job_e_h, job_w_h):
  source, platform, config, _, seeds, reps, rep_start, what = ARMS[arm]
  fe, fw = factors(arm)
  reps_line = f'fresh_reps: {reps}\n' + (f'rep_start: {rep_start}\n' if rep_start != 1 else '')
  return (f'# Follow-up 14 ({SPEC}; label {LABEL}), arm {arm.replace("_", " ")}, part {i} of {n}: {what}.\n'
          f'# The settings of {SOURCE_TEXT[source]} with config {config} on {platform} (harness/configs.yaml: its follow-up 14\n'
          f'# settings are the only change).\n'
          f'# Generated by harness/plans/make_fu14_plans.py: {len(ts)} target(s), seeds {seeds[0]}..{seeds[-1]}, '
          f'{reps} fresh repetition(s) from r{rep_start}; expected job {job_e_h:.2f} h (worst {job_w_h:.2f} h: '
          f'{source} medians x {fe:g}, worst x {fw:g}); deadline ceil(worst job min x 1.25 + 20).\n\n'
          f'description: follow-up {LABEL}, {arm.replace("_", " ")} part {i} of {n} ({what})\nlabel: {LABEL}\nweights: gcs\n'
          f'launch_by: {LAUNCH_BY}\nfinish_by: {FINISH_BY}\n' + m7.EIGEN
          + f'\nconfigs:\n  {platform}: {config}\n\ntargets:\n  default: [{", ".join(ts)}]\n\n'
          f'seeds: [{", ".join(map(str, seeds))}]\n{reps_line}warm_rerun_seed: none\nnum_recycles: default\n'
          f'num_diffusion_samples: default\nrun_timeout_min: 180\n\ndeadline_min:\n  {platform}: {deadline}\n')


def wrapper_hours(parts, platform, arms, key):
  hs = sorted((p[key] for p in parts if p['platform'] == platform and p['arm'] in arms), reverse=True)
  slots = [0.0] * SLOTS[platform]
  for h in hs:
    heapq.heapreplace(slots, slots[0] + h)
  return max(slots) if hs else 0.0


def build():
  manifest = {r['pdb_id']: r for r in csv.DictReader(open(REPO / 'inputs' / 'manifest.csv'))}
  subset = [r['pdb_id'] for r in csv.DictReader(open(REPO / 'targets' / 'posebusters_subset.csv'))]
  pilot = [t for t in subset if manifest[t]['pilot'] == 'yes']
  frozen_seeds = set(json.loads((REPO / 'inputs' / 'manifest.json').read_text())['seeds'])
  files, parts, cache = {}, [], {}
  for arm in ORDER:
    source, platform, config, which, seeds, reps, rep_start, what = ARMS[arm]
    if not set(seeds) <= frozen_seeds:
      raise SystemExit(f'{arm}: seeds {seeds} are not all in the frozen inputs')
    fe, fw = factors(arm)
    if source not in cache:
      cache[source] = timings(source, manifest)
    walls, o = cache[source]
    cap = m7.MAX_H['l4' if platform in ('a100', 'h100') else platform] * 3600 - o['setup'] * 60
    ts_all = subset if which == 'all' else pilot
    runs = len(seeds) * reps
    bins = split(arm, [(t, runs * walls[int(manifest[t]['bucket'])] * fw) for t in ts_all], cap)
    for i, b in enumerate(bins, 1):
      name = f'{LABEL}_{arm.lower()}_{i:02d}'
      ts = [t for t in ts_all if t in b['items']]
      load_e = sum(runs * walls[int(manifest[t]['bucket'])] * fe for t in ts)
      job_e, job_w = o['setup'] + load_e / 60, o['setup'] + b['load'] / 60          # minutes
      deadline = math.ceil(job_w * 1.25 + 20)
      life_h = math.ceil((deadline + 30) / 60) if platform == 'v5e' else (deadline + 30) / 60
      vm_e = (o['provision'] + job_e + o['teardown']) / 60
      vm_w = (o['provision'] + job_w + o['teardown']) / 60
      files[f'harness/plans/{LABEL}/{name}.yaml'] = plan_text(arm, i, len(bins), ts, deadline, job_e / 60, job_w / 60)
      prov = PROVISIONING[platform]
      od, pr = m7.price(platform, 'standard'), m7.price(platform, prov)
      parts.append({'part': name, 'arm': arm, 'label': LABEL, 'platform': platform, 'config': config, 'what': what,
                    'targets': len(ts), 'seeds': ' '.join(map(str, seeds)),
                    'repetitions': ' '.join(f'r{rep_start + k}' for k in range(reps)), 'processes': len(ts) * runs,
                    'factor_expected': fe, 'factor_worst': fw, 'expected_vm_h': round(vm_e, 2),
                    'worst_vm_h': round(vm_w, 2), 'deadline_min': deadline, 'lifetime_h': round(life_h, 2),
                    'usd_per_h_on_demand': od, 'usd_per_h_planned': pr, 'provisioning': prov,
                    'expected_usd_planned': round(pr * vm_e, 2), 'worst_usd_planned': round(pr * vm_w, 2),
                    'lifetime_usd_planned': round(pr * life_h, 2),
                    'expected_usd_on_demand': round(od * vm_e, 2), 'worst_usd_on_demand': round(od * vm_w, 2)})
  buf = io.StringIO()
  w = csv.DictWriter(buf, fieldnames=list(parts[0]), lineterminator='\n')
  w.writeheader()
  w.writerows(parts)
  files['harness/plans/fu14_parts.csv'] = buf.getvalue()

  def lines(ps, extra=lambda p: ''):
    ps = sorted(ps, key=lambda p: (ORDER.index(p['arm']), -p['worst_vm_h'], p['part']))   # float32 first, longest first
    return ''.join(f'{p["platform"]} harness/plans/{LABEL}/{p["part"]}.yaml{extra(p)}\n' for p in ps)
  full = lambda p: ' FETCH=full' if p['arm'] in ('V5f_r2',) + DET_ARMS else ''  # noqa: E731
  head = '# follow-up 14 (label f32), generated by harness/plans/make_fu14_plans.py: '
  files['results/parts_f32_v5e.txt'] = head + 'V5f and V5f r2 on v5e, on demand, europe-west4-b\n' + lines(
      [p for p in parts if p['platform'] == 'v5e'], full)
  files['results/parts_f32_v5e_ew4a.txt'] = head + ('reserve: the v5e parts in europe-west4-a (v5e quota 512; '
                                                     'v5litepod-1 offered there, read 2026-11-12)\n') + lines(
      [p for p in parts if p['platform'] == 'v5e'], lambda p: full(p) + ' ZONE=europe-west4-a')
  files['results/parts_f32_a100.txt'] = head + 'Af on A100, Spot (SPOT=1 in the wrapper)\n' + lines(
      [p for p in parts if p['arm'] == 'Af'])
  files['results/parts_f32_a100_ondemand.txt'] = head + ('fallback: the Af parts on demand, for parts that found no '
                                                         'Spot capacity (item f; a row first)\n') + lines(
      [p for p in parts if p['arm'] == 'Af'], lambda p: ' PROVISIONING=standard')
  files['results/parts_f32_h100.txt'] = head + 'Hf on H100, Spot (SPOT=1 in the wrapper)\n' + lines(
      [p for p in parts if p['arm'] == 'Hf'])
  files['results/parts_f32_det.txt'] = head + ('item b2, Ad and Hd, Spot, launched after the float32 arms '
                                               '(credit_check_fu14.sh det)\n') + lines(
      [p for p in parts if p['arm'] in DET_ARMS], full)
  files['harness/plans/fu14_table.md'] = table(parts, cache)
  return files, parts


def table(parts, cache):
  s = lambda ps, k: sum(p[k] for p in ps)  # noqa: E731
  L = ['# Follow-up 14: parts and cost (generated by harness/plans/make_fu14_plans.py)', '',
       f'Label {LABEL}; {SPEC}. Provisioning: v5e on demand, A100 and H100 Spot (on-demand price beside them). '
       'Expected and worst: the bfloat16 medians of the source arm times the factors of item g (float32 1.58; worst '
       '3.0 on v5e, 3.8 on the A100, 3.6 on the H100; deterministic 1.12 and 1.5). Lifetime: the bound set by the part '
       'deadlines (deadline + 30 min).', '',
       '| arm | platform | parts | processes | VM h expected | VM h worst | USD expected | USD worst | USD lifetime bound | USD expected on demand | USD worst on demand |',
       '|---|---|---|---|---|---|---|---|---|---|---|']
  for arm in ORDER:
    ps = [p for p in parts if p['arm'] == arm]
    L.append(f'| {arm.replace("_", " ")} | {ps[0]["platform"]} | {len(ps)} | {s(ps, "processes")} | '
             f'{s(ps, "expected_vm_h"):.1f} | {s(ps, "worst_vm_h"):.1f} | {s(ps, "expected_usd_planned"):.2f} | '
             f'{s(ps, "worst_usd_planned"):.2f} | {s(ps, "lifetime_usd_planned"):.2f} | '
             f'{s(ps, "expected_usd_on_demand"):.2f} | {s(ps, "worst_usd_on_demand"):.2f} |')
  for grp, arms in (('float32 arms', FLOAT32_ARMS), ('item b2', DET_ARMS)):
    ps = [p for p in parts if p['arm'] in arms]
    L.append(f'| {grp} | | {len(ps)} | {s(ps, "processes")} | {s(ps, "expected_vm_h"):.1f} | {s(ps, "worst_vm_h"):.1f} | '
             f'{s(ps, "expected_usd_planned"):.2f} | {s(ps, "worst_usd_planned"):.2f} | {s(ps, "lifetime_usd_planned"):.2f} | '
             f'{s(ps, "expected_usd_on_demand"):.2f} | {s(ps, "worst_usd_on_demand"):.2f} |')
  dur = []
  for pl, arms in (('v5e', FLOAT32_ARMS), ('a100', FLOAT32_ARMS), ('h100', FLOAT32_ARMS), ('a100', DET_ARMS),
                   ('h100', DET_ARMS)):
    what = 'float32' if arms == FLOAT32_ARMS else 'item b2'
    dur.append(f'{pl} {what} {wrapper_hours(parts, pl, arms, "expected_vm_h"):.1f} h expected, '
               f'{wrapper_hours(parts, pl, arms, "worst_vm_h"):.1f} h worst ({SLOTS[pl]} at a time)')
  L += ['', 'Duration of each wrapper from its first VM, without capacity waits (parts longest first, each to the '
        'slot free first): ' + '; '.join(dur) + '. The item b2 wrapper starts after the float32 arms have ended.', '',
        'Process wall per bucket (median s, fresh): ' + '; '.join(
            f'{src}: ' + ', '.join(f'{b} {cache[src][0][b]:.0f}' for b in sorted(cache[src][0])) for src in cache) + '.',
        'Overheads (median min): ' + '; '.join(f'{src}: provisioning {cache[src][1]["provision"]:.1f}, setup '
                                               f'{cache[src][1]["setup"]:.1f}, teardown {cache[src][1]["teardown"]:.1f}'
                                               for src in cache) + '.']
  return '\n'.join(L) + '\n'


def main():
  ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
  ap.add_argument('--check', action='store_true')
  args = ap.parse_args()
  files, _ = build()
  if args.check:
    absent = [p for p in files if p.startswith('results/') and not (REPO / p).is_file()]
    bad = [p for p, t in files.items() if p not in absent and (not (REPO / p).is_file() or (REPO / p).read_text() != t)]
    stale = sorted(str(p.relative_to(REPO)) for p in (PLANS / LABEL).glob('*.yaml')
                   if str(p.relative_to(REPO)) not in files)
    print('up to date' if not bad and not stale else 'DIFFERS: ' + ', '.join(bad + stale))
    return 1 if bad or stale else 0
  for old in (PLANS / LABEL).glob('*.yaml'):
    old.unlink()
  for p, t in files.items():
    (REPO / p).parent.mkdir(parents=True, exist_ok=True)
    (REPO / p).write_text(t)
  print(files['harness/plans/fu14_table.md'])
  return 0


if __name__ == '__main__':
  sys.exit(main())
