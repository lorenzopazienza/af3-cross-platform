"""Plan parts of follow-up 15 (label rcx; notes/analysis_plan.md section 10, row of 2026-11-14 "Follow-up experiment
15"), the recompilation of the float32 trunk on the A100 and the H100 (part A4) and of the L4 Triton arm (part A3),
sized from the run records of the r1 sessions they repeat. Sibling of harness/plans/make_fu14_plans.py.

    python3 harness/plans/make_rcx_plans.py [--check]

Arms (item b of the row; each is r2 of an existing arm, with that arm's configuration unchanged):
  Af_r2  a100_xla_fu14_f32 on A100 (r1: Af of follow-up 14)   the 10 pilot targets, seeds 1 to 5, r2   part A4
  Hf_r2  h100_xla_fu14_f32 on H100 (r1: Hf of follow-up 14)   the 10 pilot targets, seeds 1 to 5, r2   part A4
  GT_r2  l4_triton on L4           (r1: GT of the main study) the 60 targets, seeds 1 to 5, r2         part A3
Sizing (item g): per token bucket, the median wall of the fresh processes of the r1 sessions of the same
configuration (Af and Hf: scripts/analysis_runs/fu14/sessions.txt; GT: the main study's sessions_main.txt), the same
values as results/analysis/fu14_20261113b/cost.csv for Af and Hf; factor 1.0 expected and 1.25 worst (the largest
ratio of a session's process time to these medians is 1.01 for Af, 1.01 for Hf and 1.02 for GT); per-part overheads,
the median provisioning, setup and teardown of those sessions. Parts hold whole targets; each arm is cut into the
number of parts of its platform's slots (SLOTS: A100 4, without us-east4; H100 3; L4 8), more only if a part would
exceed 4 h of job at the worst factor; targets are dealt longest first, each to the lightest part. Deadline
ceil(worst job min x 1.25 + 20). Prices (cloud/prices.csv): the highest Spot row among the regions each wrapper uses
(A100: us-central1, us-east5, europe-west4; H100: us-central1, us-east5; L4: every region of the launcher's L4
zones except us-east4 that has a row), the on-demand price beside it. Worst with preemption: every part preempted once
at the end of its worst job and relaunched whole (twice the worst VM hours), an upper bound for one preemption per part.
Writes harness/plans/rcx/*.yaml, harness/plans/rcx_parts.csv, harness/plans/rcx_table.md and, outside git, the parts
files results/parts_rcx_a100.txt, parts_rcx_h100.txt (part A4) and parts_rcx_l4.txt (part A3). --check writes
nothing and exits 1 if a tracked file differs.
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

LABEL = 'rcx'
LAUNCH_BY, FINISH_BY = '2026-11-17', '2026-11-18'
SPEC = 'notes/analysis_plan.md section 10, row of 2026-11-14 (follow-up experiment 15)'
PROVISIONING = {'a100': 'spot', 'h100': 'spot', 'l4': 'spot'}
SLOTS = {'a100': 4, 'h100': 3, 'l4': 8}
FACTORS = (1.0, 1.25)
MAX_JOB_H = 4.0
REGIONS = {'a100': ('us-central1', 'us-east5', 'europe-west4'), 'h100': ('us-central1', 'us-east5'),
           'l4': ('europe-west4', 'europe-west1', 'europe-west3', 'europe-west2', 'europe-west6', 'us-central1',
                  'us-east1', 'us-west1', 'us-west4')}
H100_ZONES = 'us-central1-a,us-central1-b,us-central1-c,us-east5-a'
L4_ZONES = ('europe-west4-a,europe-west4-b,europe-west4-c,europe-west1-b,europe-west1-c,europe-west3-a,europe-west3-b,'
            'europe-west2-a,europe-west2-b,europe-west6-b,europe-west6-c,us-central1-a,us-central1-b,us-central1-c,'
            'us-east1-b,us-east1-c,us-east1-d,us-west1-a,us-west1-b,us-west1-c,us-west4-a,us-west4-c')
# arm: (r1 condition, platform, config, targets ('all' or 'pilot'), seeds, part of the row, what)
ARMS = {
    'Af_r2': ('Af', 'a100', 'a100_xla_fu14_f32', 'pilot', [1, 2, 3, 4, 5], 'A4', 'Af r2, recompilation of Af (follow-up 14)'),
    'Hf_r2': ('Hf', 'h100', 'h100_xla_fu14_f32', 'pilot', [1, 2, 3, 4, 5], 'A4', 'Hf r2, recompilation of Hf (follow-up 14)'),
    'GT_r2': ('GT', 'l4', 'l4_triton', 'all', [1, 2, 3, 4, 5], 'A3', 'GT r2, recompilation of GT (main study)'),
}
ORDER = list(ARMS)
A4, A3 = ('Af_r2', 'Hf_r2'), ('GT_r2',)
SOURCE_TEXT = {'Af': 'Af r1 (follow-up 14)', 'Hf': 'Hf r1 (follow-up 14)', 'GT': 'GT r1 (main study)'}


def source_sessions(source):
  if source == 'GT':
    return [s for s in (REPO / 'results' / 'analysis' / 'main_20261014' / 'sessions_main.txt').read_text().split()
            if '_l4_main_gt_' in s]
  pat = {'Af': '_a100_f32_af_', 'Hf': '_h100_f32_hf_'}[source]
  return [l.split('#')[0].strip() for l in (REPO / 'scripts' / 'analysis_runs' / 'fu14' / 'sessions.txt').read_text()
          .splitlines() if pat in l.split('#')[0]]


def timings(source, manifest):
  """make_fu7_10_plans.timings on the r1 sessions of the same configuration."""
  orig = m7.sessions
  m7.sessions = lambda _source: source_sessions(source)
  try:
    return m7.timings(f'{source}_r1', manifest)
  finally:
    m7.sessions = orig


def price(platform, prov):
  rows = [r for r in csv.DictReader(open(REPO / 'cloud' / 'prices.csv')) if r['platform'] == platform
          and r['usd_per_hour'] and r['region'] in REGIONS[platform]]
  return max(float(r['usd_per_hour']) for r in rows if r['provisioning'] == prov)


def deal(items, k):
  """Targets longest first, each to the lightest of k parts."""
  bins = [{'items': [], 'load': 0.0} for _ in range(k)]
  for name, sec in sorted(items, key=lambda x: (-x[1], x[0])):
    b = min(bins, key=lambda b: (b['load'], len(b['items'])))
    b['items'].append(name)
    b['load'] += sec
  return bins


def split(platform, items, cap):
  k = SLOTS[platform]
  while True:
    bins = deal(items, k)
    if max(b['load'] for b in bins) <= cap or k >= len(items):
      return [b for b in bins if b['items']]
    k += SLOTS[platform]


def plan_text(arm, i, n, ts, deadline, job_e_h, job_w_h):
  source, platform, config, _, seeds, part, what = ARMS[arm]
  fe, fw = FACTORS
  return (f'# Follow-up 15 ({SPEC}; label {LABEL}), part {part}, arm {arm.replace("_", " ")}, part {i} of {n}: {what}.\n'
          f'# The settings of {SOURCE_TEXT[source]}, config {config} on {platform} unchanged; only the repetition (r2: a\n'
          f'# new process on a new VM) differs.\n'
          f'# Generated by harness/plans/make_rcx_plans.py: {len(ts)} target(s), seeds {seeds[0]}..{seeds[-1]}, r2; '
          f'expected job {job_e_h:.2f} h (worst {job_w_h:.2f} h: {source} r1 medians x {fe:g}, worst x {fw:g}); '
          f'deadline ceil(worst job min x 1.25 + 20).\n\n'
          f'description: follow-up {LABEL}, {arm.replace("_", " ")} part {i} of {n} ({what})\nlabel: {LABEL}\n'
          f'weights: gcs\nlaunch_by: {LAUNCH_BY}\nfinish_by: {FINISH_BY}\n' + m7.EIGEN
          + f'\nconfigs:\n  {platform}: {config}\n\ntargets:\n  default: [{", ".join(ts)}]\n\n'
          f'seeds: [{", ".join(map(str, seeds))}]\nfresh_reps: 1\nrep_start: 2\nwarm_rerun_seed: none\n'
          f'num_recycles: default\nnum_diffusion_samples: default\nrun_timeout_min: 180\n\n'
          f'deadline_min:\n  {platform}: {deadline}\n')


def wrapper_hours(parts, arms, key):
  ps = [p for p in parts if p['arm'] in arms]
  if not ps:
    return 0.0
  out = 0.0
  for pl in sorted({p['platform'] for p in ps}):
    hs = sorted((p[key] for p in ps if p['platform'] == pl), reverse=True)
    slots = [0.0] * SLOTS[pl]
    for h in hs:
      heapq.heapreplace(slots, slots[0] + h)
    out = max(out, max(slots))
  return out


def build():
  manifest = {r['pdb_id']: r for r in csv.DictReader(open(REPO / 'inputs' / 'manifest.csv'))}
  subset = [r['pdb_id'] for r in csv.DictReader(open(REPO / 'targets' / 'posebusters_subset.csv'))]
  pilot = [t for t in subset if manifest[t]['pilot'] == 'yes']
  frozen_seeds = set(json.loads((REPO / 'inputs' / 'manifest.json').read_text())['seeds'])
  files, parts, cache = {}, [], {}
  fe, fw = FACTORS
  for arm in ORDER:
    source, platform, config, which, seeds, part, what = ARMS[arm]
    if not set(seeds) <= frozen_seeds:
      raise SystemExit(f'{arm}: seeds {seeds} are not all in the frozen inputs')
    if source not in cache:
      cache[source] = timings(source, manifest)
    walls, o = cache[source]
    cap = MAX_JOB_H * 3600 - o['setup'] * 60
    ts_all = subset if which == 'all' else pilot
    bins = split(platform, [(t, len(seeds) * walls[int(manifest[t]['bucket'])] * fw) for t in ts_all], cap)
    od, pr = price(platform, 'standard'), price(platform, PROVISIONING[platform])
    for i, b in enumerate(bins, 1):
      name = f'{LABEL}_{arm.lower()}_{i:02d}'
      ts = [t for t in ts_all if t in b['items']]
      load_e = sum(len(seeds) * walls[int(manifest[t]['bucket'])] * fe for t in ts)
      job_e, job_w = o['setup'] + load_e / 60, o['setup'] + b['load'] / 60          # minutes
      deadline = math.ceil(job_w * 1.25 + 20)
      life_h = (deadline + 30) / 60
      vm_e = (o['provision'] + job_e + o['teardown']) / 60
      vm_w = (o['provision'] + job_w + o['teardown']) / 60
      files[f'harness/plans/{LABEL}/{name}.yaml'] = plan_text(arm, i, len(bins), ts, deadline, job_e / 60, job_w / 60)
      parts.append({'part': name, 'arm': arm, 'row_part': part, 'label': LABEL, 'platform': platform, 'config': config,
                    'what': what, 'targets': len(ts), 'seeds': ' '.join(map(str, seeds)), 'repetitions': 'r2',
                    'processes': len(ts) * len(seeds), 'factor_expected': fe, 'factor_worst': fw,
                    'expected_vm_h': round(vm_e, 2), 'worst_vm_h': round(vm_w, 2), 'deadline_min': deadline,
                    'lifetime_h': round(life_h, 2), 'usd_per_h_on_demand': od, 'usd_per_h_planned': pr,
                    'provisioning': PROVISIONING[platform], 'expected_usd_planned': round(pr * vm_e, 2),
                    'worst_usd_planned': round(pr * vm_w, 2),
                    'worst_preempted_usd_planned': round(2 * pr * vm_w, 2),
                    'lifetime_usd_planned': round(pr * life_h, 2),
                    'expected_usd_on_demand': round(od * vm_e, 2), 'worst_usd_on_demand': round(od * vm_w, 2)})
  buf = io.StringIO()
  w = csv.DictWriter(buf, fieldnames=list(parts[0]), lineterminator='\n')
  w.writeheader()
  w.writerows(parts)
  files['harness/plans/rcx_parts.csv'] = buf.getvalue()

  def lines(ps, extra=''):
    ps = sorted(ps, key=lambda p: (-p['worst_vm_h'], p['part']))      # longest first
    return ''.join(f'{p["platform"]} harness/plans/{LABEL}/{p["part"]}.yaml{extra}\n' for p in ps)
  head = '# follow-up 15 (label rcx), generated by harness/plans/make_rcx_plans.py: '
  files['results/parts_rcx_a100.txt'] = head + ('part A4, Af r2 on A100, Spot (SPOT=1 and the region caps without '
                                                'us-east4 in the wrapper; FETCH=full)\n') + lines(
      [p for p in parts if p['arm'] == 'Af_r2'], ' FETCH=full')
  files['results/parts_rcx_h100.txt'] = head + ('part A4, Hf r2 on H100, Spot (SPOT=1 in the wrapper; zones without '
                                                'us-east4; FETCH=full)\n') + lines(
      [p for p in parts if p['arm'] == 'Hf_r2'], f' FETCH=full ZONES={H100_ZONES}')
  files['results/parts_rcx_l4.txt'] = head + ('part A3, GT r2 on L4, Spot (SPOT=1 in the wrapper; zones without '
                                              'us-east4; light fetch as GT r1), only after the gate of item g\n') + lines(
      [p for p in parts if p['arm'] == 'GT_r2'], f' ZONES={L4_ZONES}')
  files['harness/plans/rcx_table.md'] = table(parts, cache)
  return files, parts


def table(parts, cache):
  s = lambda ps, k: sum(p[k] for p in ps)  # noqa: E731
  L = ['# Follow-up 15: parts and cost (generated by harness/plans/make_rcx_plans.py)', '',
       f'Label {LABEL}; {SPEC}. Provisioning: Spot on every platform (on-demand price beside it). Expected and worst: '
       'the medians per bucket of the r1 processes of the same configuration times 1.0 and 1.25, plus the median '
       'per-part overheads of those sessions. Worst with preemption: every part preempted once at the end of its '
       'worst job and relaunched whole. Lifetime: the bound set by the part deadlines (deadline + 30 min).', '',
       '| arm | part of the row | platform | parts | processes | VM h expected | VM h worst | USD expected | USD worst | '
       'USD worst with preemption | USD lifetime bound | USD expected on demand | USD worst on demand |',
       '|---|---|---|---|---|---|---|---|---|---|---|---|---|']
  for arm in ORDER:
    ps = [p for p in parts if p['arm'] == arm]
    L.append(f'| {arm.replace("_", " ")} | {ps[0]["row_part"]} | {ps[0]["platform"]} | {len(ps)} | {s(ps, "processes")} | '
             f'{s(ps, "expected_vm_h"):.2f} | {s(ps, "worst_vm_h"):.2f} | {s(ps, "expected_usd_planned"):.2f} | '
             f'{s(ps, "worst_usd_planned"):.2f} | {s(ps, "worst_preempted_usd_planned"):.2f} | '
             f'{s(ps, "lifetime_usd_planned"):.2f} | {s(ps, "expected_usd_on_demand"):.2f} | '
             f'{s(ps, "worst_usd_on_demand"):.2f} |')
  for grp, arms in (('A4', A4), ('A3', A3), ('A4 and A3', A4 + A3)):
    ps = [p for p in parts if p['arm'] in arms]
    L.append(f'| {grp} | | | {len(ps)} | {s(ps, "processes")} | {s(ps, "expected_vm_h"):.2f} | '
             f'{s(ps, "worst_vm_h"):.2f} | {s(ps, "expected_usd_planned"):.2f} | {s(ps, "worst_usd_planned"):.2f} | '
             f'{s(ps, "worst_preempted_usd_planned"):.2f} | {s(ps, "lifetime_usd_planned"):.2f} | '
             f'{s(ps, "expected_usd_on_demand"):.2f} | {s(ps, "worst_usd_on_demand"):.2f} |')
  dur = []
  for what, arms in (('A100 (Af r2)', ('Af_r2',)), ('H100 (Hf r2)', ('Hf_r2',)), ('L4 (GT r2)', A3)):
    dur.append(f'{what} {wrapper_hours(parts, arms, "expected_vm_h"):.1f} h expected, '
               f'{wrapper_hours(parts, arms, "worst_vm_h"):.1f} h worst')
  L += ['', 'Duration of each wrapper from its first VM, without capacity waits or preemptions (parts longest first, '
        f'each to the slot free first; slots A100 {SLOTS["a100"]}, H100 {SLOTS["h100"]}, L4 {SLOTS["l4"]}): '
        + '; '.join(dur) + '. The A100 and H100 wrappers start together; the L4 wrapper starts after part A4 has '
        'ended and its gate (item g) answers GO.', '',
        'Process wall per bucket (median s, fresh r1 processes): ' + '; '.join(
            f'{src}: ' + ', '.join(f'{b} {cache[src][0][b]:.1f}' for b in sorted(cache[src][0])) for src in cache) + '.',
        'Overheads (median min): ' + '; '.join(f'{src}: provisioning {cache[src][1]["provision"]:.1f}, setup '
                                               f'{cache[src][1]["setup"]:.1f}, teardown {cache[src][1]["teardown"]:.1f}'
                                               for src in cache) + '.',
        'Prices (USD per hour, Spot, highest among the regions used): ' + '; '.join(
            f'{pl} {price(pl, "spot"):.4f} (on demand {price(pl, "standard"):.4f})' for pl in ('a100', 'h100', 'l4')) + '.']
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
  print(files['harness/plans/rcx_table.md'])
  return 0


if __name__ == '__main__':
  sys.exit(main())
