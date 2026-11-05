"""Follow-up 10 (large) on the whole PDB: expected VM hours and cost per arm of the draft design, from the targets of
targets/large/pdb_targets.csv (or --targets). A planning estimate for the specification row; no plan file is written.

    python3 drafts/fu10_cost.py [--targets targets/large/pdb_targets_amended.csv] [--rc_subset 10]

Design: V5, V6 and G with the main-study settings, seeds 1 to 5; r1 on every target; r2 on G for every target; r2 on V5
and V6 for a seeded subset of rc_subset large and rc_subset control targets (numpy.random.default_rng(2026), drawn
here only to size the parts; the row fixes the draw). A target's runs share one part (r1 and r2 on one VM, as RC
needs), parts packed first-fit decreasing up to 8 h of job on TPU and 4 h on GPU (harness/plans/make_fu7_10_plans.py,
its functions). Process wall per token bucket: the main-study medians for buckets up to 1,024
(make_fu7_10_plans.timings) and the memory-limit runs for 1,280 and 1,536 (results/analysis/memlimit_20261014/
processes.csv, one process each); overheads per part: the main-study medians. Prices: cloud/prices.csv, the highest
row among the launcher's regions; on demand, Spot (v5e, v6e, L4) and Flex-start (v6e).
"""
import argparse
import csv
import math
import pathlib
import sys

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'harness' / 'plans'))
from make_fu7_10_plans import MAX_H, pack, price, timings  # noqa: E402

ARMS = [('V5', 'v5e'), ('V6', 'v6e'), ('G', 'l4')]
SEEDS = 5


def memlimit_walls():
  out = {}
  for r in csv.DictReader(open(REPO / 'results' / 'analysis' / 'memlimit_20261014' / 'processes.csv')):
    if r['source'] == 'memlimit' and r['exit_code'] == '0' and int(r['size']) in (1280, 1536):
      out[(r['arm'], int(r['size']))] = float(r['wall_s'])
  return out


def main():
  ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
  ap.add_argument('--rc_subset', type=int, default=10)
  ap.add_argument('--targets', default='targets/large/pdb_targets.csv')
  args = ap.parse_args()
  rows = [r for r in csv.DictReader(l for l in open(REPO / args.targets) if not l.startswith('#'))
          if r['group'] in ('large', 'control')]
  manifest = {r['pdb_id']: r for r in csv.DictReader(open(REPO / 'inputs' / 'manifest.csv'))}
  big = memlimit_walls()
  rng = np.random.default_rng(2026)
  subset = set()
  for g in ('large', 'control'):
    ids = sorted(r['pdb_id'] for r in rows if r['group'] == g)
    subset |= {ids[i] for i in sorted(rng.choice(len(ids), size=min(args.rc_subset, len(ids)), replace=False))}
  print(f'targets: {sum(r["group"] == "large" for r in rows)} large, {sum(r["group"] == "control" for r in rows)} control; '
        f'TPU r2 subset {len(subset)}')
  print('| arm | parts | processes | VM h | on demand USD | Spot USD | Flex-start USD | worst case (lifetime, on demand) |')
  print('|---|---|---|---|---|---|---|---|')
  tot = {}
  for arm, platform in ARMS:
    walls, o = timings(arm, manifest)
    def wall(r):
      b = int(r['bucket'])
      return walls[b] if b <= 1024 else big[(arm, b)]
    reps = lambda r: 2 if arm == 'G' or r['pdb_id'] in subset else 1  # noqa: E731
    items = [(r['pdb_id'], SEEDS * reps(r) * wall(r)) for r in rows]
    bins = pack(items, MAX_H[platform] * 3600 - o['setup'] * 60)
    vm_h = sum((o['provision'] + o['setup'] + b['load'] / 60 + o['teardown']) / 60 for b in bins)
    life_h = sum(((math.ceil((o['setup'] + b['load'] / 60) * 1.25 + 20) + 30) / 60) for b in bins)
    life_h = sum(math.ceil(((math.ceil((o['setup'] + b['load'] / 60) * 1.25 + 20)) + 30) / 60) for b in bins) \
        if platform == 'v5e' else life_h
    od, sp = price(platform, 'standard'), price(platform, 'spot')
    fx = price(platform, 'flex_start') if platform == 'v6e' else None
    n = sum(SEEDS * reps(r) for r in rows)
    tot[arm] = (vm_h, od * vm_h, sp * vm_h, fx * vm_h if fx else None)
    print(f'| {arm} | {len(bins)} | {n} | {vm_h:.1f} | {od * vm_h:.2f} | {sp * vm_h:.2f} | '
          f'{fx * vm_h:.2f} | {od * life_h:.2f} |' if fx else
          f'| {arm} | {len(bins)} | {n} | {vm_h:.1f} | {od * vm_h:.2f} | {sp * vm_h:.2f} | - | {od * life_h:.2f} |')
  print('walls used (s): ' + '; '.join(f'{a} 1280 {big[(a, 1280)]:.0f}, 1536 {big[(a, 1536)]:.0f}' for a, _ in ARMS))
  print('TPU r2 subset: ' + ' '.join(sorted(subset)))


if __name__ == '__main__':
  main()
