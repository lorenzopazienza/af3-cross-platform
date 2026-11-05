"""Plan parts of follow-up 10, larger inputs on the whole PDB (label large; notes/analysis_plan.md section 10, row of
2026-11-04 "Follow-up experiment 10, larger inputs on the whole PDB", items c, e and f; OSF addendum
https://osf.io/nsbgp). Sibling of harness/plans/make_fu11_12_plans.py: the sizing, deadlines, lifetimes and prices of
harness/plans/make_fu7_10_plans.py (its functions, imported).

    python3 harness/plans/make_fu10_plans.py [--check]

Arms (the main-study settings of the source arm; only the targets and the repetitions change):
  V5l  tpu_xla on v5e (source V5), on demand;
  V6l  tpu_xla on v6e (source V6), Flex-start;
  Gl   l4_xla on L4 (source G), Spot.
Targets: the 68 of follow-up 10 (inputs/large/manifest.csv, plan key manifest; until the inputs are built, sized from
targets/large/large_subset.csv, whose token counts the inputs must equal). Seeds 1 to 5. Repetitions (row item c):
r1 on every target on every arm; r2 on Gl for every target; r2 on V5l and V6l only for the 20 targets of the row's
subset (large_subset.csv column tpu_r2). A plan has one fresh_reps value, so on V5l and V6l the subset's targets and
the others go to separate parts (fresh_reps 2 and 1); a target's runs are always in one part (r1 and r2 on one VM, as
RC needs). No warm reruns.
Sizing (row item f): parts packed first-fit decreasing up to 8 h of job on TPU and 4 h on GPU (make_fu7_10_plans.pack),
deadline ceil(job min x 1.25 + 20) as follow-up 11; process wall per token bucket: the main-study medians of the
source arm up to 1,024 tokens (make_fu7_10_plans.timings) and the memory-limit runs of 2026-10-14 for 1,280 and
1,536 (results/analysis/memlimit_20261014/processes.csv, source memlimit, exit 0, one process each); overheads: the
source arm's main-study medians. A target whose runs alone exceed the part size gets a part of its own (an L4 target
of 1,536 tokens with r1 and r2: 10 x 1,751 s, 4.9 h).
Checks before anything is written: every bucket is one the source arm's chip ran in the memory-limit measurement
(v5e up to 1,536, its largest, 2,048 out of memory; v6e up to 2,048; L4 up to 2,560); the longest process wall stays
within run_timeout_min (180, per AlphaFold3 process, harness/run_af3.py --timeout_min).
Every plan: label large, weights gcs, finish_by 2026-11-14 (row item e), eigen_source archive, manifest
inputs/large/manifest.csv, run_timeout_min 180, no launch_by (the launch dates of item e are checked by
scripts/monitor/credit_check_fu10.sh).
Prices: cloud/prices.csv, on demand and as the row sets the provisioning (PROPOSED).
Writes harness/plans/large/*.yaml, harness/plans/fu10_parts.csv, harness/plans/fu10_table.md and the parts files
results/parts_{v5l,v6l,gl}.txt (outside git). --check writes nothing and exits 1 if a tracked file differs.
"""
import argparse
import csv
import io
import math
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]
PLANS = REPO / 'harness' / 'plans'
sys.path.insert(0, str(PLANS))
from make_fu7_10_plans import EIGEN, MAX_H, pack, price, timings  # noqa: E402

LABEL = 'large'
FINISH_BY = '2026-11-14'
SPEC = 'notes/analysis_plan.md section 10, row of 2026-11-04 (follow-up experiment 10, larger inputs on the whole PDB)'
MANIFEST = 'inputs/large/manifest.csv'
SUBSET = 'targets/large/large_subset.csv'
MEMLIMIT = 'results/analysis/memlimit_20261014/processes.csv'
RUN_TIMEOUT_MIN = 180
# arm: (source arm, platform, config, r2 on every target)
ARMS = {'V5l': ('V5', 'v5e', 'tpu_xla', False), 'V6l': ('V6', 'v6e', 'tpu_xla', False), 'Gl': ('G', 'l4', 'l4_xla', True)}
ORDER = ['V5l', 'V6l', 'Gl']
PROPOSED = {'v5e': 'standard', 'v6e': 'flex_start', 'l4': 'spot'}
SEEDS = range(1, 6)
# The largest bucket each chip ran in the memory-limit measurement (section 10, rows of 2026-10-14).
LARGEST_BUCKET = {'v5e': 1536, 'v6e': 2048, 'l4': 2560}


def large_targets():
  """{pdb_id: row} of the follow-up 10 targets (the manifest once built, else the targets CSV), the r2 subset, and
  which file the sizes come from."""
  sub = {r['pdb_id']: r for r in csv.DictReader(open(REPO / SUBSET))}
  r2 = {t for t, r in sub.items() if r['tpu_r2'] == 'yes'}
  path = REPO / MANIFEST
  if not path.is_file():
    return sub, r2, SUBSET
  rows = {r['pdb_id']: r for r in csv.DictReader(open(path))}
  bad = [t for t, r in rows.items() if t not in sub or r['num_tokens'] != sub[t]['num_tokens']]
  if bad:
    raise SystemExit(f'{MANIFEST}: targets not in {SUBSET} or with other token counts: {bad}')
  return rows, r2 & set(rows), MANIFEST


def memlimit_walls():
  """{(source arm, bucket): wall s} of the memory-limit runs for 1,280 and 1,536 tokens."""
  out = {}
  for r in csv.DictReader(open(REPO / MEMLIMIT)):
    if r['source'] == 'memlimit' and r['exit_code'] == '0' and int(r['size']) in (1280, 1536):
      out[(r['arm'], int(r['size']))] = float(r['wall_s'])
  return out


def plan_text(arm, i, n, ts, reps, deadline, job_h):
  source, platform, config, _ = ARMS[arm]
  r2 = ('r1 and r2 (every target)' if ARMS[arm][3] else
        'r1 and r2 (targets of the r2 subset)' if reps == 2 else 'r1 (targets outside the r2 subset)')
  return (f'# Follow-up 10 ({SPEC}; label {LABEL}), arm {arm}, part {i} of {n}: {r2}.\n'
          f'# The main-study settings of {source}, config {config} on {platform}; only the targets (follow-up 10: '
          f'34 large and 34 control complexes from the PDB) and the repetitions change.\n'
          f'# Generated by harness/plans/make_fu10_plans.py: {len(ts)} target(s), seeds 1..5, {reps} fresh '
          f'repetition(s); expected job {job_h:.2f} h (main-study medians of {source} up to 1,024 tokens, the '
          f'memory-limit runs for 1,280 and 1,536); deadline ceil(job min x 1.25 + 20).\n\n'
          f'description: follow-up {LABEL}, {arm} part {i} of {n}\nlabel: {LABEL}\nweights: gcs\n'
          f'finish_by: {FINISH_BY}\n' + EIGEN + f'manifest: {MANIFEST}\n'
          f'\nconfigs:\n  {platform}: {config}\n\ntargets:\n  default: [{", ".join(ts)}]\n\n'
          f'seeds: [1, 2, 3, 4, 5]\nfresh_reps: {reps}\nwarm_rerun_seed: none\nnum_recycles: default\n'
          f'num_diffusion_samples: default\nrun_timeout_min: {RUN_TIMEOUT_MIN}\n\ndeadline_min:\n  {platform}: {deadline}\n')


def build():
  main_manifest = {r['pdb_id']: r for r in csv.DictReader(open(REPO / 'inputs' / 'manifest.csv'))}
  targets, r2, sized_from = large_targets()
  big = memlimit_walls()
  files, parts, cache = {}, [], {}
  for arm in ORDER:
    source, platform, config, r2_all = ARMS[arm]
    walls, o = cache.setdefault(source, timings(source, main_manifest))
    buckets = {int(r['bucket']) for r in targets.values()}
    if max(buckets) > LARGEST_BUCKET[platform]:
      raise SystemExit(f'{arm}: bucket {max(buckets)} is above the largest that ran on {platform}')

    def wall(t):
      b = int(targets[t]['bucket'])
      return walls[b] if b <= 1024 else big[(source, b)]
    longest = max(wall(t) for t in targets)
    if longest > RUN_TIMEOUT_MIN * 60:
      raise SystemExit(f'{arm}: a process of {longest:.0f} s exceeds run_timeout_min {RUN_TIMEOUT_MIN}')
    groups = [(2, sorted(targets))] if r2_all else [(2, sorted(t for t in targets if t in r2)),
                                                    (1, sorted(t for t in targets if t not in r2))]
    bins = []
    for reps, ts in groups:
      for b in pack([(t, len(SEEDS) * reps * wall(t)) for t in ts], MAX_H[platform] * 3600 - o['setup'] * 60):
        bins.append((reps, b))
    for i, (reps, b) in enumerate(bins, 1):
      name = f'{LABEL}_{arm.lower()}_{i:02d}'
      job_min = o['setup'] + b['load'] / 60
      deadline = math.ceil(job_min * 1.25 + 20)
      life_h = math.ceil((deadline + 30) / 60) if platform == 'v5e' else (deadline + 30) / 60
      vm_h = (o['provision'] + job_min + o['teardown']) / 60
      ts = sorted(b['items'])
      files[f'harness/plans/{LABEL}/{name}.yaml'] = plan_text(arm, i, len(bins), ts, reps, deadline, job_min / 60)
      od, pr = price(platform, 'standard'), price(platform, PROPOSED[platform])
      parts.append({'part': name, 'arm': arm, 'label': LABEL, 'platform': platform, 'config': config,
                    'fresh_reps': reps, 'targets': len(ts), 'processes': len(ts) * len(SEEDS) * reps,
                    'longest_process_s': round(max(wall(t) for t in ts)), 'expected_vm_h': round(vm_h, 2),
                    'deadline_min': deadline, 'lifetime_h': round(life_h, 2), 'usd_per_h_on_demand': od,
                    'usd_per_h_planned': pr, 'provisioning': PROPOSED[platform],
                    'expected_usd_on_demand': round(od * vm_h, 2), 'worst_usd_on_demand': round(od * life_h, 2),
                    'expected_usd_planned': round(pr * vm_h, 2), 'worst_usd_planned': round(pr * life_h, 2)})
  buf = io.StringIO()
  w = csv.DictWriter(buf, fieldnames=list(parts[0]), lineterminator='\n')
  w.writeheader()
  w.writerows(parts)
  files['harness/plans/fu10_parts.csv'] = buf.getvalue()
  for arm in ORDER:
    files[f'results/parts_{arm.lower()}.txt'] = (f'# arm {arm} (generated by harness/plans/make_fu10_plans.py)\n'
                                                 + ''.join(f'{p["platform"]} harness/plans/{LABEL}/{p["part"]}.yaml\n'
                                                           for p in parts if p['arm'] == arm))
  L = ['# Follow-up 10: parts and cost (generated by harness/plans/make_fu10_plans.py)', '',
       f'Targets: {len(targets)}, sized from {sized_from}; r2 on V5l and V6l: {len(r2)} targets (the row\'s subset).',
       '', '| arm | platform | parts | processes | VM h | on demand: expected, worst USD | '
       'proposed: expected, worst USD |', '|---|---|---|---|---|---|---|']
  tot = [0.0] * 4
  for arm in ORDER:
    ps = [p for p in parts if p['arm'] == arm]
    s = lambda k: sum(p[k] for p in ps)  # noqa: E731
    for j, k in enumerate(('expected_usd_on_demand', 'worst_usd_on_demand', 'expected_usd_planned', 'worst_usd_planned')):
      tot[j] += s(k)
    L.append(f'| {arm} | {ps[0]["platform"]} | {len(ps)} | {s("processes")} | {s("expected_vm_h"):.1f} | '
             f'{s("expected_usd_on_demand"):.2f}, {s("worst_usd_on_demand"):.2f} | '
             f'{s("expected_usd_planned"):.2f}, {s("worst_usd_planned"):.2f} ({ps[0]["provisioning"]}) |')
  L.append(f'| all | | {len(parts)} | {sum(p["processes"] for p in parts)} | '
           f'{sum(p["expected_vm_h"] for p in parts):.1f} | {tot[0]:.2f}, {tot[1]:.2f} | {tot[2]:.2f}, {tot[3]:.2f} |')
  L += ['', 'Process wall per bucket (s, fresh): ' + '; '.join(
      f'{src}: ' + ', '.join(f'{b} {cache[src][0][b]:.0f}' for b in sorted(cache[src][0])) +
      f', 1280 {big[(src, 1280)]:.0f}, 1536 {big[(src, 1536)]:.0f}' for src in cache) +
      ' (up to 1,024: main-study medians; 1,280 and 1,536: memory-limit runs, one process each).',
      'Overheads (median min): ' + '; '.join(f'{src}: provisioning {cache[src][1]["provision"]:.1f}, setup '
                                             f'{cache[src][1]["setup"]:.1f}, teardown {cache[src][1]["teardown"]:.1f}'
                                             for src in cache) + '.',
      f'Memory: every bucket is at most the largest that ran on its chip in the memory-limit measurement '
      f'(v5e {LARGEST_BUCKET["v5e"]}, v6e {LARGEST_BUCKET["v6e"]}, L4 {LARGEST_BUCKET["l4"]}). Longest process: '
      + ', '.join(f'{arm} {max(p["longest_process_s"] for p in parts if p["arm"] == arm)} s' for arm in ORDER)
      + f' (run_timeout_min {RUN_TIMEOUT_MIN} per process).']
  files['harness/plans/fu10_table.md'] = '\n'.join(L) + '\n'
  return files


def main():
  ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
  ap.add_argument('--check', action='store_true')
  args = ap.parse_args()
  files = build()
  if args.check:
    absent = [p for p in files if p.startswith('results/') and not (REPO / p).is_file()]
    bad = [p for p, t in files.items() if p not in absent and (not (REPO / p).is_file() or (REPO / p).read_text() != t)]
    stale = [str(p.relative_to(REPO)) for p in (PLANS / LABEL).glob('*.yaml') if str(p.relative_to(REPO)) not in files]
    print('up to date' if not bad and not stale else 'DIFFERS: ' + ', '.join(bad + stale))
    return 1 if bad or stale else 0
  for old in (PLANS / LABEL).glob('*.yaml'):
    old.unlink()
  for p, t in files.items():
    (REPO / p).parent.mkdir(parents=True, exist_ok=True)
    (REPO / p).write_text(t)
  print(files['harness/plans/fu10_table.md'])
  return 0


if __name__ == '__main__':
  sys.exit(main())
