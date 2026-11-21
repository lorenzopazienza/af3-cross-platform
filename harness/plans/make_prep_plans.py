"""Plan parts of follow-ups 17 to 21 (notes/analysis_plan.md section 10, specification rows of the follow-ups prepared
on 2026-11-21 for the new credit), sized from the run records of the arms they repeat or extend. Sibling of
harness/plans/make_rcx_plans.py: the same timings (make_fu7_10_plans.timings on named sessions), dealing of targets,
deadlines and prices; one generator for the four labels.

    python3 harness/plans/make_prep_plans.py --label dtp|tri|rcf|aut|xvm [--check]

Labels and arms (item b of each row; "part" is the part of the row, A mandatory, B conditional on its gate):
  dtp  follow-up 17, the float32 trunk at the default matmul precision (the missing cell of data type x precision)
       V6fd     tpu_xla_dtp_f32def on v6e    the 60 targets, seeds 1 to 5, r1                         part A
       V6fd_r2  tpu_xla_dtp_f32def on v6e    the 10 pilot targets, seed 1, r2 (bit-identity check)    part A
       Gfd      l4_xla_dtp_f32def on L4      the 60 targets, seeds 1 to 5, r1 and r2 on one VM        part A
       Afd      a100_xla_dtp_f32def on A100  the 60 targets, seeds 1 to 5, r1                         part B
       Hfd      h100_xla_dtp_f32def on H100  the 60 targets, seeds 1 to 5, r1                         part B
  tri  follow-up 18, AlphaFold 3's default Triton attention on the H100 and the A100
       HT       h100_triton on H100          the 60 targets, seeds 1 to 5, r1                         part A
       HT_r2    h100_triton on H100          the 60 targets, seeds 1 to 5, r2 (new VMs, as H r2)      part A
       AT_r2    a100_triton on A100          the 60 targets, seeds 1 to 5, r2 (r1: AT, follow-up 3)   part B
  rcf  follow-up 19, recompilation of the float32 A100 and H100 arms on the other 50 targets
       Af_r2b   a100_xla_fu14_f32 on A100    the 50 non-pilot targets, seeds 1 to 5, r2               part A
       Hf_r2b   h100_xla_fu14_f32 on H100    the 50 non-pilot targets, seeds 1 to 5, r2               part A
  aut  follow-up 20, the L4 without autotuning
       Gna      l4_xla_noautotune on L4      the 60 targets, seeds 1 to 5, r1 and r2 on one VM        part A
  xvm  follow-up 21, recompilation on new VMs of the same GPU type
       G_r3     l4_xla on L4                 the 60 targets, seeds 1 to 5, r3 (new VMs; r1, r2: main) part A
       A_r3     a100_xla on A100             the 60 targets, seeds 1 to 5, r3 (new VMs; r1, r2: FU3)  part B
Sizing (item g of each row): per token bucket, the median wall of the fresh processes of the named source sessions
(V6f and Gf: follow-up 13; Af and Hf: follow-up 14; H: follow-up 9; A and AT: follow-up 3; G: the main study), times a
factor, expected and worst (FACTORS); per-part overheads, the median provisioning, setup and teardown of those
sessions. Parts hold whole targets (r1 and r2 together where fresh_reps is 2); each arm is cut into the number of
parts of its platform's slots (SLOTS), more only if a part would exceed 8 h (TPU) or 4 h (GPU) of job at the worst
factor; targets are dealt longest first, each to the lightest part. Deadline ceil(worst job min x 1.25 + 20). Prices
(cloud/prices.csv): the highest row of the planned provisioning among the regions each wrapper uses, the on-demand
price beside it. Worst with preemption: every part preempted once at the end of its worst job and relaunched whole.
Writes harness/plans/<label>/*.yaml, harness/plans/<label>_parts.csv, harness/plans/<label>_table.md and, outside
git, the parts files results/parts_<label>_<platform>.txt (one per wrapper). --check writes nothing and exits 1 if a
tracked file differs.
"""
import argparse
import csv
import heapq
import io
import itertools
import json
import math
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]
PLANS = REPO / 'harness' / 'plans'
sys.path.insert(0, str(PLANS))
import make_fu7_10_plans as m7  # noqa: E402

# Dates of every label: launch on or before LAUNCH_BY, every run finished by FINISH_BY (plan keys launch_by and
# finish_by, checked by cloud/af3_run.sh), fixed in the specification rows; see COSTI_E_CALENDARIO.md for the reason.
LAUNCH_BY, FINISH_BY = '2026-12-06', '2026-12-08'
MAX_JOB_H = {'v6e': 8.0, 'v5e': 8.0, 'l4': 4.0, 'a100': 4.0, 'h100': 4.0}
# v6e: cloud/prices.csv has no v6e row for us-east5; the launcher prices us-east5 at europe-west4's (its own note).
# A100: the three regions of the quotas approved on 2026-11-21 (A100 80 GB on demand and preemptible: us-central1 8,
# us-east5 4, us-east4 4); the wrappers' A100_REGION_CAPS is A100_CAPS below.
REGIONS = {'v6e': ('europe-west4',), 'a100': ('us-central1', 'us-east5', 'us-east4'),
           'h100': ('us-central1', 'us-east5', 'us-east4'),
           'l4': ('europe-west4', 'europe-west1', 'europe-west3', 'europe-west2', 'europe-west6', 'us-central1',
                  'us-east1', 'us-west1', 'us-west4', 'us-east4')}
# H100 and A100 over the three regions of the quotas approved on 2026-11-21 (us-central1, us-east5, us-east4; us-east4
# last: setup failures of 2026-11-12, now relaunched unchanged by the retry wrapper); the L4 adds us-east4 at the end.
H100_ZONES = 'us-central1-a,us-central1-b,us-central1-c,us-east5-a,us-east4-a,us-east4-b,us-east4-c'
# v6e (Lorenzo's reading of the quotas, 2026-11-21: on-demand and preemptible v6e quota in every zone below; the limit
# in follow-up 13 was capacity, not quota): the zones tried in order, each with a Flex-start request valid for
# V6E_FLEX_WAIT; a part that obtains no Flex-start VM in any of them is relaunched on demand over the same zones by the
# retry wrapper (prep_retry_arm.sh, outside the repository), and so is a part whose VM is lost once.
V6E_ZONES = 'europe-west4-a,europe-west4-b,europe-west4-c,us-east5-a,us-east5-b'
V6E_FLEX_WAIT = '15m'
A100_CAPS = 'us-east5:8 us-central1:8 us-east4:8'    # cheapest Spot first; us-east4 last (setup failures 2026-11-12)
COST_SLACK = 0.10     # part size: at most 10% above the expected cost of the fewest parts the job cap allows
# GPUS_ALL_REGIONS 32 (approved 2026-11-21): the MAX_<PLATFORM> of each wrapper in the launch order of
# COSTI_E_CALENDARIO.md (key "max" of each label): it sizes the parts (build) and the durations of the tables.
L4_ZONES = ('europe-west4-a,europe-west4-b,europe-west4-c,europe-west1-b,europe-west1-c,europe-west3-a,europe-west3-b,'
            'europe-west2-a,europe-west2-b,europe-west6-b,europe-west6-c,us-central1-a,us-central1-b,us-central1-c,'
            'us-east1-b,us-east1-c,us-east1-d,us-west1-a,us-west1-b,us-west1-c,us-west4-a,us-west4-c,us-east4-a,us-east4-c')
# source: (sessions file, pattern in the session name), the run records whose fresh processes size an arm
SOURCES = {
    'V6f': ('scripts/analysis_runs/fu13/sessions.txt', '_v6e_why_v6f_'),
    'Gf': ('scripts/analysis_runs/fu13/sessions.txt', '_l4_why_gf_'),
    'Af': ('scripts/analysis_runs/fu14/sessions.txt', '_a100_f32_af_'),
    'Hf': ('scripts/analysis_runs/fu14/sessions.txt', '_h100_f32_hf_'),
    'H': ('scripts/analysis_runs/h100/sessions.txt', '_h100_h100_h_'),
    'AT': ('scripts/analysis_runs/a100/sessions.txt', '_a100_a100_at_'),
    'G': ('results/analysis/main_20261014/sessions_main.txt', '_l4_main_g_'),
    'A': ('scripts/analysis_runs/a100/sessions.txt', '_a100_a100_a_'),
}
# arm: (source, platform, config, targets ('all', 'pilot' or 'nonpilot'), seeds, fresh_reps, rep_start, part of the
#       row, fetch ('full' or 'light'), (factor expected, factor worst), what)
S5 = [1, 2, 3, 4, 5]
LABELS = {
    'dtp': {
        'fu': 17, 'spec': 'notes/analysis_plan.md section 10, row "Follow-up experiment 17" (label dtp)',
        'provisioning': {'v6e': 'flex_start', 'l4': 'spot', 'a100': 'spot', 'h100': 'spot'},
        'max': {'v6e': 4, 'l4': 20, 'a100': 12, 'h100': 8},
        'arms': {
            'V6fd': ('V6f', 'v6e', 'tpu_xla_dtp_f32def', 'all', S5, 1, 1, 'A', 'light', (1.0, 1.25),
                     'V6fd r1, float32 trunk at the default matmul precision on v6e'),
            'V6fd_r2': ('V6f', 'v6e', 'tpu_xla_dtp_f32def', 'pilot', [1], 1, 2, 'A', 'full', (1.0, 1.25),
                        'V6fd r2 on the pilot targets, seed 1 (bit-identity check)'),
            'Gfd': ('Gf', 'l4', 'l4_xla_dtp_f32def', 'all', S5, 2, 1, 'A', 'light', (1.0, 1.25),
                    'Gfd r1 and r2, float32 trunk at the default matmul precision on the L4'),
            'Afd': ('Af', 'a100', 'a100_xla_dtp_f32def', 'all', S5, 1, 1, 'B', 'light', (1.0, 1.25),
                    'Afd r1, float32 trunk at the default matmul precision on the A100'),
            'Hfd': ('Hf', 'h100', 'h100_xla_dtp_f32def', 'all', S5, 1, 1, 'B', 'light', (1.0, 1.25),
                    'Hfd r1, float32 trunk at the default matmul precision on the H100'),
        },
        # wrapper: (platform, arms); the order is the launch order (part B only after its gate)
        'wrappers': {'v6e': ('v6e', ('V6fd', 'V6fd_r2')), 'l4': ('l4', ('Gfd',)), 'a100': ('a100', ('Afd',)),
                     'h100': ('h100', ('Hfd',))},
    },
    'tri': {
        'fu': 18, 'spec': 'notes/analysis_plan.md section 10, row "Follow-up experiment 18" (label tri)',
        'provisioning': {'a100': 'spot', 'h100': 'spot'},
        'max': {'a100': 12, 'h100': 12},
        'arms': {
            'HT': ('H', 'h100', 'h100_triton', 'all', S5, 1, 1, 'A', 'light', (1.0, 1.5),
                   'HT r1, Triton attention on the H100'),
            'HT_r2': ('H', 'h100', 'h100_triton', 'all', S5, 1, 2, 'A', 'light', (1.0, 1.5),
                      'HT r2, recompilation of HT on new VMs (as H r2 of follow-up 9)'),
            'AT_r2': ('AT', 'a100', 'a100_triton', 'all', S5, 1, 2, 'B', 'light', (1.0, 1.25),
                      'AT r2, recompilation of AT (follow-up 3) on new VMs'),
        },
        'wrappers': {'h100': ('h100', ('HT', 'HT_r2')), 'a100': ('a100', ('AT_r2',))},
    },
    'rcf': {
        'fu': 19, 'spec': 'notes/analysis_plan.md section 10, row "Follow-up experiment 19" (label rcf)',
        'provisioning': {'a100': 'spot', 'h100': 'spot'},
        'max': {'a100': 12, 'h100': 10},
        'arms': {
            'Af_r2b': ('Af', 'a100', 'a100_xla_fu14_f32', 'nonpilot', S5, 1, 2, 'A', 'light', (1.0, 1.25),
                       'Af r2 on the 50 non-pilot targets (follow-up 15 ran the 10 pilot targets)'),
            'Hf_r2b': ('Hf', 'h100', 'h100_xla_fu14_f32', 'nonpilot', S5, 1, 2, 'A', 'light', (1.0, 1.25),
                       'Hf r2 on the 50 non-pilot targets (follow-up 15 ran the 10 pilot targets)'),
        },
        'wrappers': {'a100': ('a100', ('Af_r2b',)), 'h100': ('h100', ('Hf_r2b',))},
    },
    'aut': {
        'fu': 20, 'spec': 'notes/analysis_plan.md section 10, row "Follow-up experiment 20" (label aut)',
        'provisioning': {'l4': 'spot'},
        'max': {'l4': 24},
        'arms': {
            'Gna': ('G', 'l4', 'l4_xla_noautotune', 'all', S5, 2, 1, 'A', 'light', (1.0, 1.5),
                    'Gna r1 and r2, the L4 with --xla_gpu_autotune_level=0'),
        },
        'wrappers': {'l4': ('l4', ('Gna',))},
    },
    'xvm': {
        'fu': 21, 'spec': 'notes/analysis_plan.md section 10, row "Follow-up experiment 21" (label xvm)',
        'provisioning': {'l4': 'spot', 'a100': 'spot'},
        'max': {'l4': 12, 'a100': 12},
        'arms': {
            'G_r3': ('G', 'l4', 'l4_xla', 'all', S5, 1, 3, 'A', 'light', (1.0, 1.25),
                     'G r3, a third repetition of G (main study) on new VMs'),
            'A_r3': ('A', 'a100', 'a100_xla', 'all', S5, 1, 3, 'B', 'light', (1.0, 1.25),
                     'A r3, a third repetition of A (follow-up 3) on new VMs'),
        },
        'wrappers': {'l4': ('l4', ('G_r3',)), 'a100': ('a100', ('A_r3',))},
    },
}


def source_sessions(source):
  path, pat = SOURCES[source]
  return [l.split('#')[0].strip() for l in (REPO / path).read_text().splitlines()
          if l.split('#')[0].strip() and pat in l.split('#')[0]]


def timings(source, manifest):
  """make_fu7_10_plans.timings on the named sessions (a name that is not 'H', so no A-to-H scaling is applied)."""
  orig = m7.sessions
  m7.sessions = lambda _source: source_sessions(source)
  try:
    return m7.timings(f'{source}_src', manifest)
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


def split(k0, items, cap):
  k = k0
  while True:
    bins = deal(items, k)
    if max(b['load'] for b in bins) <= cap or k >= len(items):
      return [b for b in bins if b['items']]
    k += k0


def plan_text(cfg, label, arm, i, n, ts, deadline, job_e_h, job_w_h):
  source, platform, config, _, seeds, reps, rep_start, part, _, (fe, fw), what = cfg['arms'][arm]
  rep_text = 'r1 and r2' if reps == 2 else f'r{rep_start}'
  return (f'# Follow-up {cfg["fu"]} ({cfg["spec"]}), part {part}, arm {arm.replace("_", " ")}, part {i} of {n}: '
          f'{what}.\n'
          f'# Generated by harness/plans/make_prep_plans.py --label {label}: {len(ts)} target(s), seeds {seeds[0]}..'
          f'{seeds[-1]}, {rep_text}; expected job {job_e_h:.2f} h (worst {job_w_h:.2f} h: {source} medians x {fe:g}, '
          f'worst x {fw:g}); deadline ceil(worst job min x 1.25 + 20).\n\n'
          f'description: follow-up {label}, {arm.replace("_", " ")} part {i} of {n} ({what})\nlabel: {label}\n'
          f'weights: gcs\nlaunch_by: {LAUNCH_BY}\nfinish_by: {FINISH_BY}\n' + m7.EIGEN
          + f'\nconfigs:\n  {platform}: {config}\n\ntargets:\n  default: [{", ".join(ts)}]\n\n'
          f'seeds: [{", ".join(map(str, seeds))}]\nfresh_reps: {reps}\nrep_start: {rep_start}\nwarm_rerun_seed: none\n'
          f'num_recycles: default\nnum_diffusion_samples: default\nrun_timeout_min: 180\n\n'
          f'deadline_min:\n  {platform}: {deadline}\n')


def lpt(hours, slots):
  """Duration of parts run longest first, each to the slot free first, on the given number of slots."""
  sl = [0.0] * slots
  for h in sorted(hours, reverse=True):
    heapq.heapreplace(sl, sl[0] + h)
  return max(sl)


def wrapper_hours(parts, slots, arms, key):
  ps = [p for p in parts if p['arm'] in arms]
  if not ps:
    return 0.0
  out = 0.0
  for pl in sorted({p['platform'] for p in ps}):
    hs = sorted((p[key] for p in ps if p['platform'] == pl), reverse=True)
    sl = [0.0] * slots[pl]
    for h in hs:
      heapq.heapreplace(sl, sl[0] + h)
    out = max(out, max(sl))
  return out


def build(label):
  cfg = LABELS[label]
  manifest = {r['pdb_id']: r for r in csv.DictReader(open(REPO / 'inputs' / 'manifest.csv'))}
  subset = [r['pdb_id'] for r in csv.DictReader(open(REPO / 'targets' / 'posebusters_subset.csv'))]
  pilot = [t for t in subset if manifest[t]['pilot'] == 'yes']
  sets = {'all': subset, 'pilot': pilot, 'nonpilot': [t for t in subset if t not in pilot]}
  frozen_seeds = set(json.loads((REPO / 'inputs' / 'manifest.json').read_text())['seeds'])
  files, parts, cache, cands = {}, [], {}, {}
  for arm, a in cfg['arms'].items():
    source, platform, config, which, seeds, reps, rep_start, part, fetch, (fe, fw), what = a
    if not set(seeds) <= frozen_seeds:
      raise SystemExit(f'{arm}: seeds {seeds} are not all in the frozen inputs')
    if source not in cache:
      cache[source] = timings(source, manifest)
    walls, o = cache[source]
    per = len(seeds) * reps
    cap = MAX_JOB_H[platform] * 3600 - o['setup'] * 60
    ts_all = sets[which]
    items = [(t, per * walls[int(manifest[t]['bucket'])] * fw) for t in ts_all]
    prov = cfg['provisioning'][platform]
    od, pr = price(platform, 'standard'), price(platform, prov)
    ks = [1] if (which == 'pilot' and len(seeds) == 1) else range(1, len(ts_all) + 1)
    opts = {}
    for k in ks:
      bins = [b for b in deal(items, k) if b['items']]
      if max(b['load'] for b in bins) > cap and k < len(ts_all):
        continue
      rows = []
      for b in bins:
        ts = [x for x in ts_all if x in b['items']]
        load_e = sum(per * walls[int(manifest[x]['bucket'])] * fe for x in ts)
        job_e, job_w = o['setup'] + load_e / 60, o['setup'] + b['load'] / 60          # minutes
        rows.append((ts, job_e, job_w, (o['provision'] + job_e + o['teardown']) / 60,
                     (o['provision'] + job_w + o['teardown']) / 60))
      opts[k] = {'rows': rows, 'usd': sum(pr * r[3] for r in rows)}
    k_min = min(opts)
    # part size (2026-11-21): the number of parts that minimises the wrapper's expected duration on its MAX slots,
    # with the arm's expected cost at most COST_SLACK above that of the fewest parts the job cap allows
    opts = {k: v for k, v in opts.items() if v['usd'] <= (1 + COST_SLACK) * opts[k_min]['usd'] + 1e-9}
    cands[arm] = {'opts': opts, 'a': a, 'o': o, 'od': od, 'pr': pr, 'prov': prov, 'per': per, 'platform': platform}
  chosen = {}
  for wname, (platform, arms) in cfg['wrappers'].items():
    slots = cfg['max'][platform]
    best = None
    for combo in itertools.product(*[sorted(cands[x]['opts']) for x in arms]):
      hs = [r[3] for x, k in zip(arms, combo) for r in cands[x]['opts'][k]['rows']]
      span = lpt(hs, slots)
      usd = sum(cands[x]['opts'][k]['usd'] for x, k in zip(arms, combo))
      key = (round(span, 6), round(usd, 6), sum(combo))
      if best is None or key < best[0]:
        best = (key, combo)
    chosen.update(dict(zip(arms, best[1])))
  for arm in cfg['arms']:
    c = cands[arm]
    source, platform, config, which, seeds, reps, rep_start, part, fetch, (fe, fw), what = c['a']
    rows = c['opts'][chosen[arm]]['rows']
    pr, od, prov, per = c['pr'], c['od'], c['prov'], c['per']
    for i, (ts, job_e, job_w, vm_e, vm_w) in enumerate(rows, 1):
      name = f'{label}_{arm.lower()}_{i:02d}'
      deadline = math.ceil(job_w * 1.25 + 20)
      life_h = (deadline + 30) / 60
      files[f'harness/plans/{label}/{name}.yaml'] = plan_text(cfg, label, arm, i, len(rows), ts, deadline,
                                                              job_e / 60, job_w / 60)
      parts.append({'part': name, 'arm': arm, 'row_part': part, 'label': label, 'platform': platform,
                    'config': config, 'what': what, 'targets': len(ts), 'seeds': ' '.join(map(str, seeds)),
                    'repetitions': 'r1 r2' if reps == 2 else f'r{rep_start}', 'processes': len(ts) * per,
                    'fetch': fetch, 'factor_expected': fe, 'factor_worst': fw,
                    'expected_vm_h': round(vm_e, 2), 'worst_vm_h': round(vm_w, 2), 'deadline_min': deadline,
                    'lifetime_h': round(life_h, 2), 'usd_per_h_on_demand': od, 'usd_per_h_planned': pr,
                    'provisioning': prov, 'expected_usd_planned': round(pr * vm_e, 2),
                    'worst_usd_planned': round(pr * vm_w, 2),
                    'worst_preempted_usd_planned': round(2 * pr * vm_w, 2),
                    'lifetime_usd_planned': round(pr * life_h, 2),
                    'expected_usd_on_demand': round(od * vm_e, 2), 'worst_usd_on_demand': round(od * vm_w, 2)})
  buf = io.StringIO()
  w = csv.DictWriter(buf, fieldnames=list(parts[0]), lineterminator='\n')
  w.writeheader()
  w.writerows(parts)
  files[f'harness/plans/{label}_parts.csv'] = buf.getvalue()
  for wname, (platform, arms) in cfg['wrappers'].items():
    ps = sorted([p for p in parts if p['arm'] in arms], key=lambda p: (-p['worst_vm_h'], p['part']))  # longest first
    extra = {'l4': f' ZONES={L4_ZONES}', 'h100': f' ZONES={H100_ZONES}',
             'v6e': f' ZONES={V6E_ZONES} REQUEST_VALID_FOR={V6E_FLEX_WAIT}'}.get(platform, '')
    where = {'l4': ' (zones without us-east4)', 'h100': ' (zones without us-east4)',
             'v6e': f' (zones in order, Flex-start request valid {V6E_FLEX_WAIT} in each; then on demand)'}
    head = (f'# follow-up {cfg["fu"]} (label {label}), generated by harness/plans/make_prep_plans.py: arms '
            f'{", ".join(a.replace("_", " ") for a in arms)} on {platform}, {cfg["provisioning"][platform]}'
            f'{where.get(platform, "")}\n')
    files[f'results/parts_{label}_{wname}.txt'] = head + ''.join(
        f'{p["platform"]} harness/plans/{label}/{p["part"]}.yaml{" FETCH=full" if p["fetch"] == "full" else ""}'
        f'{extra}\n' for p in ps)
  files[f'harness/plans/{label}_table.md'] = table(label, parts, cache)
  return files, parts


def table(label, parts, cache):
  cfg = LABELS[label]
  s = lambda ps, k: sum(p[k] for p in ps)  # noqa: E731
  L = [f'# Follow-up {cfg["fu"]}: parts and cost (generated by harness/plans/make_prep_plans.py --label {label})', '',
       f'Label {label}; {cfg["spec"]}. Provisioning: ' + ', '.join(f'{k} {v}' for k, v in cfg['provisioning'].items())
       + ' (on-demand price beside it). Expected and worst: the medians per bucket of the fresh processes of the source '
       'sessions times the factors of each arm, plus the median per-part overheads of those sessions. Worst with '
       'preemption: every part preempted once at the end of its worst job and relaunched whole. Lifetime: the bound '
       'set by the part deadlines (deadline + 30 min).', '',
       '| arm | part of the row | platform | parts | processes | factors | VM h expected | VM h worst | USD expected | '
       'USD worst | USD worst with preemption | USD lifetime bound | USD expected on demand | USD worst on demand |',
       '|---|---|---|---|---|---|---|---|---|---|---|---|---|---|']
  for arm in cfg['arms']:
    ps = [p for p in parts if p['arm'] == arm]
    L.append(f'| {arm.replace("_", " ")} | {ps[0]["row_part"]} | {ps[0]["platform"]} | {len(ps)} | '
             f'{s(ps, "processes")} | {ps[0]["factor_expected"]:g} / {ps[0]["factor_worst"]:g} | '
             f'{s(ps, "expected_vm_h"):.2f} | {s(ps, "worst_vm_h"):.2f} | {s(ps, "expected_usd_planned"):.2f} | '
             f'{s(ps, "worst_usd_planned"):.2f} | {s(ps, "worst_preempted_usd_planned"):.2f} | '
             f'{s(ps, "lifetime_usd_planned"):.2f} | {s(ps, "expected_usd_on_demand"):.2f} | '
             f'{s(ps, "worst_usd_on_demand"):.2f} |')
  groups = sorted({p['row_part'] for p in parts})
  for grp in groups + (['all'] if len(groups) > 1 else []):
    ps = [p for p in parts if grp == 'all' or p['row_part'] == grp]
    L.append(f'| part {grp} | | | {len(ps)} | {s(ps, "processes")} | | {s(ps, "expected_vm_h"):.2f} | '
             f'{s(ps, "worst_vm_h"):.2f} | {s(ps, "expected_usd_planned"):.2f} | {s(ps, "worst_usd_planned"):.2f} | '
             f'{s(ps, "worst_preempted_usd_planned"):.2f} | {s(ps, "lifetime_usd_planned"):.2f} | '
             f'{s(ps, "expected_usd_on_demand"):.2f} | {s(ps, "worst_usd_on_demand"):.2f} |')
  dur = []
  for wname, (platform, arms) in cfg['wrappers'].items():
    dur.append(f'{wname} ({", ".join(a.replace("_", " ") for a in arms)}) '
               f'{wrapper_hours(parts, cfg["max"], arms, "expected_vm_h"):.1f} h expected, '
               f'{wrapper_hours(parts, cfg["max"], arms, "worst_vm_h"):.1f} h worst')
  L += ['', 'Duration of each wrapper from its first VM, without capacity waits or preemptions (parts longest first, '
        'each to the slot free first; slots ' + ', '.join(f'{k} {v}' for k, v in cfg['max'].items())
        + '): '
        + '; '.join(dur) + '.', '',
        'Process wall per bucket (median s, fresh processes of the source sessions): ' + '; '.join(
            f'{src}: ' + ', '.join(f'{b} {cache[src][0][b]:.1f}' for b in sorted(cache[src][0])) for src in cache) + '.',
        'Overheads (median min): ' + '; '.join(f'{src}: provisioning {cache[src][1]["provision"]:.1f}, setup '
                                               f'{cache[src][1]["setup"]:.1f}, teardown {cache[src][1]["teardown"]:.1f}'
                                               for src in cache) + '.',
        'Prices (USD per hour, highest among the regions used): ' + '; '.join(
            f'{pl} {price(pl, pv):.4f} {pv} (on demand {price(pl, "standard"):.4f})'
            for pl, pv in cfg['provisioning'].items()) + '.']
  return '\n'.join(L) + '\n'


def main():
  ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
  ap.add_argument('--label', required=True, choices=sorted(LABELS))
  ap.add_argument('--check', action='store_true')
  args = ap.parse_args()
  files, _ = build(args.label)
  if args.check:
    absent = [p for p in files if p.startswith('results/') and not (REPO / p).is_file()]
    bad = [p for p, t in files.items() if p not in absent and (not (REPO / p).is_file() or (REPO / p).read_text() != t)]
    stale = sorted(str(p.relative_to(REPO)) for p in (PLANS / args.label).glob('*.yaml')
                   if str(p.relative_to(REPO)) not in files)
    print('up to date' if not bad and not stale else 'DIFFERS: ' + ', '.join(bad + stale))
    return 1 if bad or stale else 0
  for old in (PLANS / args.label).glob('*.yaml'):
    old.unlink()
  for p, t in files.items():
    (REPO / p).parent.mkdir(parents=True, exist_ok=True)
    (REPO / p).write_text(t)
  print(files[f'harness/plans/{args.label}_table.md'])
  return 0


if __name__ == '__main__':
  sys.exit(main())
