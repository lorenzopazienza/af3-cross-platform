"""Writes the plans of the warm-cost completion on the L4 Triton and the H100 (label warm_th; drafts/
section10_warm_th_20261111.md, the specification row (section 10 once entered)): Table III's "warm" cost per
structure at 1,024 tokens for the two arms that have none (GT of the main study, H of follow-up 9).

    python3 harness/plans/warm_th/make_warm_th_plans.py [--check]

Each arm is one plan, one VM: the 10 pilot targets (inputs/manifest.csv, pilot rows), seed 1, one fresh run that keeps
its compiled executables (harness/run_plan.py: warm_rerun_seed 1 makes the rep 1 run of seed 1 save them with
--save_fresh_cache), then one warm rerun on the same VM that loads them (--compile_cache warm:...), the mechanism of
the main study's warm reruns and of follow-up 3's. Nothing else changes from the arm's source:
  GT  l4_triton on l4 (g2-standard-8, the launcher's default L4 type and zones), on demand, as the main study's GT;
  H   h100_xla on h100 (a3-highgpu-1g, the launcher's H100 zones), Spot (the project's on-demand H100 quota is 0),
      as follow-up 9.
Both build AlphaFold3 with the Eigen source of our bucket (eigen_source: archive, as every plan since follow-up 7;
the same commit, so the same files). launch_by 2026-11-12 and finish_by 2026-11-13 (the row's stop rule;
cloud/af3_run.sh refuses a launch after launch_by and reports a part not finished by finish_by as incomplete).
Sizing, from the run records of each arm's source (fresh, r1 and r2 where both exist, seed 1, the pilot targets):
  fresh run   the sum over the 10 targets of the median wall of the target's seed-1 process (GT: main study,
              results/analysis/main_20261014/samples.csv; H: follow-up 9, results/analysis/h100_20261104/samples.csv);
  warm rerun  each target's fresh wall minus (1 - k) of its median compile time, k = the main study's G warm compile
              time over its fresh compile time on the same targets (median over targets: the warm process still
              looks up and loads the executable); an estimate, no warm process of GT or H exists;
  overheads   provisioning, setup and teardown, the medians of the source arm's sessions (launcher_events.txt, as
              harness/plans/make_fu7_10_plans.timings): GT the main study's GT sessions, H follow-up 9's;
  job         setup + fresh + warm; deadline ceil(job min x 1.25 + 20) (the rule of follow-ups 3 to 13);
  expected    provisioning + job + teardown, at the price below; worst: the VM lifetime the deadline sets
              (deadline + 30 min) at that price.
Prices: cloud/prices.csv, the highest among the regions each launcher may use (make_fu7_10_plans.price): L4 on demand,
H100 Spot.
Writes harness/plans/warm_th/warm_th_gt_01.yaml, warm_th_h_01.yaml, warm_th_parts.csv, table.md and the parts files
results/parts_warm_th_gt.txt and results/parts_warm_th_h.txt (outside git; FETCH=full on both, so check 2's
bit-identity reads every output file). --check writes nothing and exits 1 if a tracked file differs.
"""
import argparse
import csv
import io
import json
import math
import pathlib
import re
import statistics
import sys

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO / 'harness' / 'plans'))
from make_fu7_10_plans import EIGEN, price, stamp  # noqa: E402

LABEL = 'warm_th'
SPEC = 'drafts/section10_warm_th_20261111.md, the specification row (section 10 once entered)'
LAUNCH_BY, FINISH_BY = '2026-11-12', '2026-11-13'
SEED = 1
MAIN = 'results/analysis/main_20261014/samples.csv'
H100 = 'results/analysis/h100_20261104/samples.csv'
# arm: (platform, config, provisioning, machine type, source samples.csv, source label, source sessions, source text)
ARMS = {'GT': ('l4', 'l4_triton', 'standard', 'g2-standard-8', MAIN, 'main', 'main_gt',
               'the main study\'s GT (l4_triton on g2-standard-8, on demand)'),
        'H': ('h100', 'h100_xla', 'spot', 'a3-highgpu-1g', H100, 'h100', 'h100_h',
              'follow-up 9\'s H (h100_xla on a3-highgpu-1g, Spot)')}


def pilot_targets():
  rows = list(csv.DictReader(open(REPO / 'inputs' / 'manifest.csv')))
  t = sorted(r['pdb_id'] for r in rows if r['pilot'].strip().lower() in ('1', 'true', 'yes'))
  if len(t) != 10:
    raise SystemExit(f'expected the 10 pilot targets in inputs/manifest.csv, found {len(t)}')
  return t


def seed1_processes(samples, platform, label, targets):
  """{target: [(wall, compile)]} of the fresh seed-1 processes (r1, r2) of one arm at AlphaFold3's defaults."""
  out = {}
  for r in csv.DictReader(open(REPO / samples)):
    if (r['platform'] == platform and r['label'] == label and r['target'] in targets and r['seed'] == str(SEED)
            and r['sample'] == '0' and r['repetition'] in ('r1', 'r2') and r['exit_code'] == '0'
            and r['num_recycles'] == '10' and r['num_diffusion_samples'] == '5'):
      out.setdefault(r['target'], []).append((float(r['wall_s']), float(r['compile_s'])))
  missing = sorted(set(targets) - set(out))
  if missing:
    raise SystemExit(f'{samples}: no fresh seed-1 process of {platform} for {missing}')
  return out


def warm_compile_fraction(targets):
  """k: the main study's G warm compile time over its fresh (r1) compile time, median over the pilot targets."""
  fresh, warm = {}, {}
  for r in csv.DictReader(open(REPO / MAIN)):
    if r['platform'] == 'G' and r['label'] == 'main' and r['target'] in targets and r['seed'] == str(SEED) \
            and r['sample'] == '0' and r['exit_code'] == '0':
      if r['repetition'] == 'r1':
        fresh.setdefault(r['target'], []).append(float(r['compile_s']))
      elif r['repetition'] == 'w':
        warm.setdefault(r['target'], []).append(float(r['compile_s']))
  ks = [statistics.median(warm[t]) / statistics.median(fresh[t]) for t in targets if t in warm and t in fresh]
  if len(ks) != len(targets):
    raise SystemExit('the main study\'s G warm reruns do not cover the pilot targets')
  return statistics.median(ks)


def source_sessions(kind):
  if kind == 'main_gt':
    return [s for s in (REPO / 'results' / 'analysis' / 'main_20261014' / 'sessions_main.txt').read_text().split()
            if '_l4_main_gt_' in s]
  return [l.split('#')[0].strip() for l in (REPO / 'scripts' / 'analysis_runs' / 'h100' / 'sessions.txt')
          .read_text().splitlines() if l.split('#')[0].strip()]


def overheads(kind):
  """Median provisioning, setup and teardown minutes of the source sessions (make_fu7_10_plans.timings' rule)."""
  over = []
  for s in source_sessions(kind):
    root = REPO / 'results' / 'af3' / s
    total = 0.0
    for run in sorted(p for p in root.iterdir() if p.is_dir()):
      if (run / 'runs.jsonl').is_file():
        for line in (run / 'runs.jsonl').read_text().splitlines():
          r = json.loads(line)
          if r.get('exit_code') == 0 and r.get('wall_seconds'):
            total += float(r['wall_seconds'])
    ev = {}
    f = root / 'launcher_events.txt'
    for line in (f.read_text().splitlines() if f.is_file() else []):
      m = re.match(r'(\S+)\s+(\w+): (\w+)', line)
      if m:
        ev.setdefault(f'{m.group(2)}:{m.group(3)}', stamp(m.group(1)))
    vm = next((k for k in ev if k.startswith('vm:')), None)
    if vm and all(k in ev for k in ('launch:session', 'job:started', 'job:finished', 'delete:done')):
      mins = lambda a, b: (ev[b] - ev[a]).total_seconds() / 60  # noqa: E731
      over.append({'provision': mins('launch:session', vm), 'teardown': mins('job:finished', 'delete:done'),
                   'setup': mins('job:started', 'job:finished') - total / 60})
  if not over:
    raise SystemExit(f'no complete source session for {kind}')
  return {k: statistics.median(o[k] for o in over) for k in ('provision', 'setup', 'teardown')}, len(over)


def build():
  targets = pilot_targets()
  k = warm_compile_fraction(targets)
  files, parts = {}, []
  for arm, (platform, config, prov, machine, samples, label, kind, src) in ARMS.items():
    procs = seed1_processes(samples, arm, label, targets)
    fresh = sum(statistics.median(w for w, _ in procs[t]) for t in targets)
    warm = sum(statistics.median(w - (1 - k) * c for w, c in procs[t]) for t in targets)
    o, n_sessions = overheads(kind)
    job = o['setup'] + (fresh + warm) / 60
    deadline = math.ceil(job * 1.25 + 20)
    exp_h = (o['provision'] + job + o['teardown']) / 60
    life_h = (deadline + 30) / 60
    usd = price(platform, prov)
    name = f'{LABEL}_{arm.lower()}_01'
    files[f'harness/plans/{LABEL}/{name}.yaml'] = f'''# Warm cost on {arm} ({SPEC}; label {LABEL}): Table III's warm cost per structure at
# 1,024 tokens for {src}; only the targets, the seed and the warm rerun are set here.
# Generated by make_warm_th_plans.py: the 10 pilot targets, seed {SEED}, one fresh run that saves its executables, then one
# warm rerun on the same VM that loads them. Expected job {job:.0f} min (setup {o["setup"]:.1f} min, the median of
# {n_sessions} source sessions; fresh {fresh / 60:.1f} min, the source's seed-{SEED} walls; warm {warm / 60:.1f} min,
# the fresh walls minus (1 - {k:.3f}) x compile); deadline = ceil(job min x 1.25 + 20).

description: warm cost completion, {arm}
label: {LABEL}
weights: gcs
launch_by: {LAUNCH_BY}
finish_by: {FINISH_BY}
{EIGEN}
configs:
  {platform}: {config}

targets:
  default: [{", ".join(targets)}]

seeds: [{SEED}]
fresh_reps: 1
warm_rerun_seed: {SEED}
num_recycles: default
num_diffusion_samples: default
run_timeout_min: 180

deadline_min:
  {platform}: {deadline}
'''
    parts.append({'part': name, 'arm': arm, 'label': LABEL, 'platform': platform, 'config': config,
                  'machine_type': machine, 'provisioning': prov, 'targets': len(targets), 'processes': 2 * len(targets),
                  'fresh_min': round(fresh / 60, 1), 'warm_min': round(warm / 60, 1),
                  'provision_min': round(o['provision'], 1), 'setup_min': round(o['setup'], 1),
                  'teardown_min': round(o['teardown'], 1), 'job_min': round(job), 'deadline_min': deadline,
                  'expected_vm_h': round(exp_h, 2), 'lifetime_h': round(life_h, 2), 'usd_per_h': usd,
                  'expected_usd': round(usd * exp_h, 2), 'worst_usd': round(usd * life_h, 2)})
  buf = io.StringIO()
  w = csv.DictWriter(buf, fieldnames=list(parts[0]), lineterminator='\n')
  w.writeheader()
  w.writerows(parts)
  files[f'harness/plans/{LABEL}/{LABEL}_parts.csv'] = buf.getvalue()
  for p in parts:
    files[f'results/parts_{LABEL}_{p["arm"].lower()}.txt'] = (
        f'# Warm cost completion, arm {p["arm"]} (label {LABEL}; generated by make_warm_th_plans.py): one part, '
        f'{"on demand" if p["provisioning"] == "standard" else "Spot"}; FETCH=full for check 2.\n'
        f'{p["platform"]} harness/plans/{LABEL}/{p["part"]}.yaml SPOT={1 if p["provisioning"] == "spot" else 0} '
        'FETCH=full\n')
  L = ['# Warm cost on GT and H: plans and cost (generated by make_warm_th_plans.py)', '',
       '| plan | platform | config | provisioning | processes | fresh min | warm min | setup min | job min | '
       'deadline min | expected VM h | expected USD | lifetime h | worst USD |',
       '|---|---|---|---|---|---|---|---|---|---|---|---|---|---|']
  L += [f'| {p["part"]} | {p["platform"]} ({p["machine_type"]}) | {p["config"]} | {p["provisioning"]} | '
        f'{p["processes"]} | {p["fresh_min"]} | {p["warm_min"]} | {p["setup_min"]} | {p["job_min"]} | '
        f'{p["deadline_min"]} | {p["expected_vm_h"]} | {p["expected_usd"]} | {p["lifetime_h"]} | {p["worst_usd"]} |'
        for p in parts]
  L += [f'| all | | | | {sum(p["processes"] for p in parts)} | | | | | | '
        f'{sum(p["expected_vm_h"] for p in parts):.2f} | {sum(p["expected_usd"] for p in parts):.2f} | | '
        f'{sum(p["worst_usd"] for p in parts):.2f} |', '',
        f'Fresh: the source arm\'s seed-1 process walls on the pilot targets (median per target). Warm: each fresh wall '
        f'minus (1 - k) x its compile time, k = {k:.3f} (the main study\'s G warm over fresh compile time). Expected: '
        'provisioning, setup and teardown (source-session medians) plus the job. Worst: the VM lifetime set by the '
        'deadline. Prices: the highest of the regions each launcher may use (L4 on demand, H100 Spot).']
  files[f'harness/plans/{LABEL}/table.md'] = '\n'.join(L) + '\n'
  return files


def main():
  ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
  ap.add_argument('--check', action='store_true')
  args = ap.parse_args()
  files = build()
  if args.check:
    absent = [p for p in files if p.startswith('results/') and not (REPO / p).is_file()]
    bad = [p for p, t in files.items() if p not in absent and (not (REPO / p).is_file() or (REPO / p).read_text() != t)]
    stale = sorted(str(p.relative_to(REPO)) for p in HERE.glob('*.yaml') if str(p.relative_to(REPO)) not in files)
    print('up to date' if not bad and not stale else 'DIFFERS: ' + ', '.join(bad + stale))
    return 1 if bad or stale else 0
  for p, t in files.items():
    (REPO / p).parent.mkdir(parents=True, exist_ok=True)
    (REPO / p).write_text(t)
  print(files[f'harness/plans/{LABEL}/table.md'])
  return 0


if __name__ == '__main__':
  sys.exit(main())
