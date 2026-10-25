"""Follow-up 6 (label msa) plan parts, sized like the main study's (notes/analysis_plan.md section 10, row of
2026-10-19), from the main study's own run records.

    python3 harness/plans/msa/make_msa_plans.py [--feat_scale 2.0] [--check]

Arms, with the settings of V5, V6 and G and only the input changed (inputs/official_data/manifest.csv, the
naspaz8-af3 run1 _data.json files), on demand, 60 targets, seeds 1 to 5, one seed per run:
  Go r1  l4_xla on L4, r1;                       Go r2  the same, r2 only (plan key rep_start: 2), launched last;
  V6o    tpu_xla on v6e, r1, plus w (warm rerun of seed 1) on the 10 pilot targets;
  V5o    tpu_xla on v5e, r1, plus w on the 10 pilot targets.
The w runs are defined as W in the main study: in the part of the 10 pilot targets, the r1 run of seed 1 keeps its
compiled executables and a warm rerun loads them (harness/run_plan.py warm_rerun_seed 1).
Sizing, as harness/plans/main was sized (~/Desktop/af3_drafts/main_study/make_main_plans.py): each target's runs in
one part; warm targets and the others in separate parts; first-fit decreasing up to 4 h (L4) or 8 h (TPU) of job
time per VM; deadline ceil(job min x 1.25 + 20). Timings: per arm and token bucket, the median wall of the fresh
and warm processes of the main study's V5, V6 and G sessions (results/analysis/main_20261014/sessions_main.txt;
runs.jsonl wall_seconds), and per platform the median setup, provisioning and teardown from their launcher events.
The official inputs leave the model's work unchanged (AlphaFold3 pads every MSA to 16,384 rows and four template
slots at featurisation, and the model samples 1,024 MSA rows: model/pipeline/pipeline.py msa_crop_size and
max_templates, model/network/evoformer.py num_msa), so only featurisation can take longer: each process's
featurisation seconds are scaled by --feat_scale (default 2) on top of its main-study wall.
Prices: cloud/prices.csv on demand, the highest row among the regions the launcher may use (v5e and v6e
europe-west4; L4 every region of L4_DEFAULT_ZONES); lifetime as cloud/af3_run.sh computes it (Compute Engine deadline
+ 30 min; v5e the watchdog, rounded up to whole hours).
Writes harness/plans/msa/msa_<arm>_<NN>.yaml, harness/plans/msa/parts.csv, harness/plans/msa/table.md, and the parts
files results/parts_msa.txt (every part, in the row's launch order: Go r1, V6o, V5o, Go r2) and
results/parts_msa_<arm>.txt (one per arm, the launch unit of the budget rule). --check writes nothing and exits 1 if
any file differs from what it would write (parts files absent from results/, outside git, are reported, not failed).
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

REPO = pathlib.Path(__file__).resolve().parents[3]
HERE = REPO / 'harness' / 'plans' / 'msa'
ARMS = {  # arm: (main arm, platform, config, rep_start, reps, warm)
    'Go_r1': ('G', 'l4', 'l4_xla', 1, 1, False), 'V6o': ('V6', 'v6e', 'tpu_xla', 1, 1, True),
    'V5o': ('V5', 'v5e', 'tpu_xla', 1, 1, True), 'Go_r2': ('G', 'l4', 'l4_xla', 2, 1, False),
}
ORDER = ['Go_r1', 'V6o', 'V5o', 'Go_r2']
MAX_H = {'l4': 4.0, 'v5e': 8.0, 'v6e': 8.0}
DEADLINE_FACTOR, DEADLINE_PAD_MIN = 1.25, 20
L4_REGIONS = ['europe-west4', 'europe-west1', 'europe-west3', 'europe-west2', 'europe-west6', 'us-east4',
              'us-central1', 'us-east1', 'us-west1', 'us-west4']
LAUNCH_BY, FINISH_BY = '2026-11-02', '2026-11-08'
MANIFEST = 'inputs/official_data/manifest.csv'
SESSION_ARM = re.compile(r'_(v5e|v6e|l4)_main_(v5|v6|g)_\d+$')


def main_sessions():
  names = (REPO / 'results' / 'analysis' / 'main_20261014' / 'sessions_main.txt').read_text().split()
  out = {'V5': [], 'V6': [], 'G': []}
  for s in names:
    m = SESSION_ARM.search(s)
    if m:
      out[{'v5': 'V5', 'v6': 'V6', 'g': 'G'}[m.group(2)]].append(s)
  return out


def stamp(s):
  import datetime as dt
  return dt.datetime.strptime(s, '%Y-%m-%dT%H:%M:%SZ')


def records(session, manifest, feat_scale):
  """[(bucket, kind, seconds)] with featurisation scaled, and the VM overheads in minutes (or None)."""
  root = REPO / 'results' / 'af3' / session
  walls, total = [], 0.0
  for run in sorted(p for p in root.iterdir() if p.is_dir()):
    if not (run / 'runs.jsonl').is_file():
      continue
    kind = 'warm' if re.search(r'_warm\d*$', run.name) else 'fresh'
    for line in (run / 'runs.jsonl').read_text().splitlines():
      r = json.loads(line)
      if r.get('exit_code') == 0 and r.get('wall_seconds'):
        w = float(r['wall_seconds'])
        total += w
        walls.append((int(manifest[r['pdb_id']]['bucket']), kind,
                      w + (feat_scale - 1) * float(r.get('featurisation_seconds') or 0)))
  events = {}
  f = root / 'launcher_events.txt'
  for line in (f.read_text().splitlines() if f.is_file() else []):
    m = re.match(r'(\S+)\s+(\w+): (\w+)', line)
    if m:
      events.setdefault(f'{m.group(2)}:{m.group(3)}', stamp(m.group(1)))
  vm = next((k for k in events if k.startswith('vm:')), None)
  if vm is None or any(k not in events for k in ('launch:session', 'job:started', 'job:finished', 'delete:done')):
    return walls, None
  minutes = lambda a, b: (events[b] - events[a]).total_seconds() / 60  # noqa: E731
  return walls, {'provision': minutes('launch:session', vm), 'teardown': minutes('job:finished', 'delete:done'),
                 'setup': minutes('job:started', 'job:finished') - total / 60}


def featurisation_medians(manifest):
  """{main arm: {bucket: median featurisation seconds of its fresh processes}}, from the same run records; the
  allowance of a follow-up 6 process is --feat_scale times this (scripts/monitor/msa_feat_check.py reads it)."""
  out = {}
  for arm, sessions in main_sessions().items():
    by = {}
    for s in sessions:
      for run in (REPO / 'results' / 'af3' / s).iterdir():
        if run.is_dir() and (run / 'runs.jsonl').is_file() and not re.search(r'_warm\d*$', run.name):
          for line in (run / 'runs.jsonl').read_text().splitlines():
            r = json.loads(line)
            if r.get('exit_code') == 0 and r.get('featurisation_seconds') is not None:
              by.setdefault(int(manifest[r['pdb_id']]['bucket']), []).append(float(r['featurisation_seconds']))
    out[arm] = {b: statistics.median(v) for b, v in by.items()}
  return out


def timings(manifest, feat_scale):
  walls, over = {}, {}
  for arm, sessions in main_sessions().items():
    by = {}
    for s in sessions:
      w, o = records(s, manifest, feat_scale)
      for b, k, x in w:
        by.setdefault((b, k), []).append(x)
      if o:
        over.setdefault(arm, []).append(o)
    walls[arm] = {k: statistics.median(v) for k, v in by.items()}
  over = {a: {k: statistics.median(o[k] for o in v) for k in ('provision', 'setup', 'teardown')} for a, v in over.items()}
  return walls, over


def price(platform):
  rows = [r for r in csv.DictReader(open(REPO / 'cloud' / 'prices.csv'))
          if r['platform'] == platform and r['provisioning'] == 'standard' and r['usd_per_hour']
          and (platform != 'l4' or r['region'] in L4_REGIONS)]
  best = max(rows, key=lambda r: float(r['usd_per_hour']))
  return float(best['usd_per_hour']), best['read_on'], best['region']


def pack(items, capacity):
  bins = []
  for name, sec in sorted(items, key=lambda x: (-x[1], x[0])):
    for b in bins:
      if b['load'] + sec <= capacity:
        b['items'].append(name)
        b['load'] += sec
        break
    else:
      bins.append({'items': [name], 'load': sec})
  return bins


def plan_text(arm, part, n_parts, targets, warm, deadline, job_h, feat_scale):
  main_arm, platform, config, rep_start, reps, _ = ARMS[arm]
  rep = 'r2 only' if rep_start == 2 else 'r1'
  return f'''# Follow-up 6 (notes/analysis_plan.md section 10, row of 2026-10-19), arm {arm.replace("_", " ")} ({rep}), part {part} of {n_parts}.
# The settings of {main_arm} in the main study (config {config} on {platform}, AlphaFold3's defaults, buckets and flags); only the
# input changes: the official _data.json of each target ({MANIFEST}).
# Generated by make_msa_plans.py: {len(targets)} target(s), seeds 1..5, {rep}{", warm rerun of seed 1" if warm else ""}; expected job
# time {job_h:.2f} h from the main study's {main_arm} run records (featurisation x {feat_scale:g}); deadline = ceil(job min x {DEADLINE_FACTOR} + {DEADLINE_PAD_MIN}).

description: follow-up 6, {arm.replace("_", " ")} part {part} of {n_parts}
label: msa
weights: gcs
manifest: {MANIFEST}
launch_by: {LAUNCH_BY}
finish_by: {FINISH_BY}

configs:
  {platform}: {config}

targets:
  default: [{", ".join(targets)}]

seeds: [1, 2, 3, 4, 5]
fresh_reps: {reps}
{"rep_start: 2" + chr(10) if rep_start == 2 else ""}warm_rerun_seed: {1 if warm else "none"}
num_recycles: default
num_diffusion_samples: default
run_timeout_min: 180

deadline_min:
  {platform}: {deadline}
'''


def build(feat_scale):
  """{relative path: text} of every file this script writes."""
  manifest = {r['pdb_id']: r for r in csv.DictReader(open(REPO / 'inputs' / 'manifest.csv'))}
  subset = [r['pdb_id'] for r in csv.DictReader(open(REPO / 'targets' / 'posebusters_subset.csv'))]
  pilot = [t for t in subset if manifest[t]['pilot'] == 'yes']
  assert len(subset) == 60 and len(pilot) == 10
  walls, over = timings(manifest, feat_scale)
  files, parts = {}, []
  for arm in ORDER:
    main_arm, platform, config, rep_start, reps, has_warm = ARMS[arm]
    w, o = walls[main_arm], over[main_arm]
    setup_s = o['setup'] * 60

    def load(t, warm):
      b = int(manifest[t]['bucket'])
      return 5 * reps * w[(b, 'fresh')] + (w.get((b, 'warm'), w[(b, 'fresh')]) if warm else 0)
    groups = [(True, pilot), (False, [t for t in subset if t not in pilot])] if has_warm else [(False, subset)]
    bins = []
    for warm, ts in groups:
      bins += [dict(b, warm=warm) for b in pack([(t, load(t, warm)) for t in ts], MAX_H[platform] * 3600 - setup_s)]
    usd, read_on, region = price(platform)
    for i, b in enumerate(bins, 1):
      name = f'msa_{arm.lower()}_{i:02d}'
      job_min = (setup_s + b['load']) / 60
      deadline = math.ceil(job_min * DEADLINE_FACTOR + DEADLINE_PAD_MIN)
      life_min = deadline + 30
      life_h = math.ceil(life_min / 60) if platform == 'v5e' else life_min / 60
      vm_min = o['provision'] + job_min + o['teardown']
      ts = [t for t in subset if t in b['items']]
      files[f'harness/plans/msa/{name}.yaml'] = plan_text(arm, i, len(bins), ts, b['warm'], deadline, job_min / 60,
                                                          feat_scale)
      parts.append({'part': name, 'arm': arm, 'main_arm': main_arm, 'platform': platform, 'config': config,
                    'rep': 'r2' if rep_start == 2 else 'r1', 'targets': len(ts), 'warm': b['warm'],
                    'processes': len(ts) * 5 * reps + (len(ts) if b['warm'] else 0),
                    'expected_job_h': round(job_min / 60, 2), 'expected_vm_h': round(vm_min / 60, 2),
                    'deadline_min': deadline, 'lifetime_h': round(life_h, 2), 'usd_per_h': usd,
                    'price_region': region, 'price_read_on': read_on, 'expected_usd': round(usd * vm_min / 60, 2),
                    'worst_usd': round(usd * life_h, 2), 'target_list': ' '.join(ts)})
  buf = io.StringIO()
  wr = csv.DictWriter(buf, fieldnames=list(parts[0]), lineterminator='\n')
  wr.writeheader()
  wr.writerows(parts)
  files['harness/plans/msa/parts.csv'] = buf.getvalue()
  head = ('# Follow-up 6 (label msa), cloud/af3_run_many.sh input: PLATFORM PLAN, in the launch order of the section 10 '
          'row\n# of 2026-10-19 (generated by harness/plans/msa/make_msa_plans.py). The budget rule is applied per arm: '
          'launch\n# results/parts_msa_<arm>.txt one arm at a time, after scripts/monitor/credit_check.sh says GO.\n')
  files['results/parts_msa.txt'] = head + ''.join(f'{p["platform"]} harness/plans/msa/{p["part"]}.yaml\n' for p in parts)
  for arm in ORDER:
    files[f'results/parts_msa_{arm.lower()}.txt'] = (
        f'# Follow-up 6, arm {arm.replace("_", " ")} (generated by make_msa_plans.py)\n'
        + ''.join(f'{p["platform"]} harness/plans/msa/{p["part"]}.yaml\n' for p in parts if p['arm'] == arm))
  lines = ['# Follow-up 6 parts (generated by harness/plans/msa/make_msa_plans.py)', '',
           f'On demand; featurisation x {feat_scale:g}; max job hours per part L4 {MAX_H["l4"]:g}, TPU {MAX_H["v5e"]:g}; '
           f'deadline ceil(job min x {DEADLINE_FACTOR} + {DEADLINE_PAD_MIN}); launch by {LAUNCH_BY}, finish by {FINISH_BY}.', '',
           '| arm | platform | parts | processes | expected VM h | expected USD | worst-case USD | largest part: job h, deadline min |',
           '|---|---|---|---|---|---|---|---|']
  tot = [0, 0, 0.0, 0.0, 0.0]
  for arm in ORDER:
    ps = [p for p in parts if p['arm'] == arm]
    big = max(ps, key=lambda p: p['expected_job_h'])
    row = [len(ps), sum(p['processes'] for p in ps), sum(p['expected_vm_h'] for p in ps),
           sum(p['expected_usd'] for p in ps), sum(p['worst_usd'] for p in ps)]
    tot = [a + b for a, b in zip(tot, row)]
    lines.append(f'| {arm.replace("_", " ")} | {ps[0]["platform"]} | {row[0]} | {row[1]} | {row[2]:.1f} | {row[3]:.2f} | '
                 f'{row[4]:.2f} | {big["expected_job_h"]:.2f}, {big["deadline_min"]} |')
  lines.append(f'| **total** | | {tot[0]} | {tot[1]} | {tot[2]:.1f} | {tot[3]:.2f} | {tot[4]:.2f} | |')
  lines += ['', 'Overheads per VM from the main study\'s launcher events (median minutes): ' + '; '.join(
      f'{a}: provisioning {o["provision"]:.1f}, setup {o["setup"]:.1f}, teardown {o["teardown"]:.1f}'
      for a, o in sorted(over.items())) + '.', '',
      'Process wall per bucket (median seconds, featurisation scaled; fresh / warm): ' + '; '.join(
      f'{a}: ' + ', '.join(f'{b} {walls[a].get((b, "fresh"), float("nan")):.0f}/{walls[a].get((b, "warm"), float("nan")):.0f}'
                           for b in (256, 512, 768, 1024)) for a in ('V5', 'V6', 'G')) + '.', '',
      '## Parts', '', '| part | platform | targets | warm | processes | job h | deadline min | lifetime h | expected USD | worst USD |',
      '|---|---|---|---|---|---|---|---|---|---|']
  lines += [f'| {p["part"]} | {p["platform"]} | {p["targets"]} | {"yes" if p["warm"] else "no"} | {p["processes"]} | '
            f'{p["expected_job_h"]:.2f} | {p["deadline_min"]} | {p["lifetime_h"]:.2f} | {p["expected_usd"]:.2f} | '
            f'{p["worst_usd"]:.2f} |' for p in parts]
  files['harness/plans/msa/table.md'] = '\n'.join(lines) + '\n'
  return files


def main():
  ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
  ap.add_argument('--feat_scale', type=float, default=2.0)
  ap.add_argument('--check', action='store_true')
  args = ap.parse_args()
  files = build(args.feat_scale)
  if args.check:
    # results/ is outside git: its parts files are checked when present (a fresh clone regenerates them).
    absent = [p for p in files if p.startswith('results/') and not (REPO / p).is_file()]
    bad = [p for p, t in files.items() if p not in absent and (not (REPO / p).is_file() or (REPO / p).read_text() != t)]
    print(('up to date' if not bad else 'DIFFERS: ' + ', '.join(bad))
          + (f' ({len(absent)} parts file(s) in results/ not generated here)' if absent else ''))
    return 1 if bad else 0
  for old in HERE.glob('msa_*.yaml'):
    old.unlink()
  for p, t in files.items():
    (REPO / p).parent.mkdir(parents=True, exist_ok=True)
    (REPO / p).write_text(t)
  print(files['harness/plans/msa/table.md'].split('## Parts')[0])
  return 0


if __name__ == '__main__':
  sys.exit(main())
