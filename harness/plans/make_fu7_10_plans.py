"""Plan parts of follow-ups 7 (label sw), 8 (seeds) and 9 (h100), sized like the main study's, from the run records of
the arm each new arm derives from. Follow-up 10 (large) has no plans: as specified it selects one target
(drafts/section10_fu7-10.md).

    python3 harness/plans/make_fu7_10_plans.py [--check]

Arms (each with the settings of its source arm; only the factor under study changes):
  seeds  V5x  tpu_xla on v5e and  Gx  l4_xla on L4: 60 targets, seeds 6 to 20 (seeds_beyond_input), r1;
  sw     V6s  tpu_xla on v6e and  Gs  l4_xla on L4: 60 targets, seeds 1 to 5, r1, the alternative stack (plan key stack:
         jax and jaxlib 0.10.1 with libtpu 0.0.41 on TPU, with jax-cuda12-plugin and -pjrt 0.10.1 on GPU);
  h100   H    h100_xla on an H100 80 GB (a3-highgpu-1g): 60 targets, seeds 1 to 5, r1; H r2 (for RC(H)): all 60
         targets, seeds 1 to 5, r2 only (rep_start 2; the 10 pilot targets until the amendment of 2026-11-03);
  smoke  sw_smoke (V6s and Gs stacks) and h100_smoke: 2PV7 (inputs/official_data/smoke_manifest.csv), seed 1, r1.
Sizing (harness/plans/main, ~/Desktop/af3_drafts/main_study/make_main_plans.py): each target's runs in one part,
first-fit decreasing up to 8 h of job on TPU and 4 h on GPU, deadline ceil(job min x 1.25 + 20). Timings: per source
arm and token bucket, the median wall of the fresh processes of its sessions (main study: V5, V6, G; follow-up 3:
A), and the median setup, provisioning and teardown from their launcher events. H: A's walls with the inference part
scaled by 34/62, AlphaFold3's compile-free H100 over A100 time at 1,024 tokens (docs/performance.md).
Prices: cloud/prices.csv, the highest row among the regions the launcher may use; on demand (as the source arms ran)
and as planned (Lorenzo, 2026-10-27: Spot on v5e, L4 and H100; on demand on v6e, which has had no Flex-start capacity). Lifetime as cloud/af3_run.sh
computes it (Compute Engine deadline + 30 min; v5e the watchdog, rounded up to whole hours).
Writes harness/plans/{seeds,sw,h100}/*.yaml, harness/plans/fu7_10_parts.csv, harness/plans/fu7_10_table.md and the
parts files results/parts_<arm>.txt (outside git). --check writes nothing and exits 1 if a tracked file differs.
"""
import argparse
import csv
import datetime as dt
import io
import json
import math
import pathlib
import re
import statistics
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]
PLANS = REPO / 'harness' / 'plans'
LAUNCH_BY, FINISH_BY = '2026-11-11', '2026-11-12'
H_OVER_A = 34 / 62
STACK = {'tpu': {'jax': '0.10.1', 'jaxlib': '0.10.1', 'libtpu': '0.0.41'},
         'gpu': {'jax': '0.10.1', 'jaxlib': '0.10.1', 'jax-cuda12-plugin': '0.10.1', 'jax-cuda12-pjrt': '0.10.1'}}
# arm: (label, source arm, platform, config, seeds, rep_start, targets ('all' or 'pilot'), extra plan lines)
ARMS = {
    'V5x': ('seeds', 'V5', 'v5e', 'tpu_xla', range(6, 21), 1, 'all', 'seeds_beyond_input: true\n'),
    'Gx': ('seeds', 'G', 'l4', 'l4_xla', range(6, 21), 1, 'all', 'seeds_beyond_input: true\n'),
    'V6s': ('sw', 'V6', 'v6e', 'tpu_xla', range(1, 6), 1, 'all', None),
    'Gs': ('sw', 'G', 'l4', 'l4_xla', range(1, 6), 1, 'all', None),
    'H': ('h100', 'H', 'h100', 'h100_xla', range(1, 6), 1, 'all', ''),
    'H_r2': ('h100', 'H', 'h100', 'h100_xla', range(1, 6), 2, 'all', 'rep_start: 2\n'),
}
ORDER = ['V5x', 'Gx', 'V6s', 'Gs', 'H', 'H_r2']
MAX_H = {'v5e': 8.0, 'v6e': 8.0, 'l4': 4.0, 'h100': 4.0}
REGIONS = {'l4': ['europe-west4', 'europe-west1', 'europe-west3', 'europe-west2', 'europe-west6', 'us-east4',
                  'us-central1', 'us-east1', 'us-west1', 'us-west4'], 'h100': ['us-central1', 'us-east4', 'us-east5']}
PLANNED = {'v5e': 'spot', 'l4': 'spot', 'h100': 'spot', 'v6e': 'standard'}   # how each platform's VMs are obtained
# Every plan written here takes the Eigen source of AlphaFold3's build from our bucket (cloud/eigen_pin.sh), not
# from gitlab.com: the same commit, so the same files (plan key eigen_source, harness/run_plan.py).
EIGEN = 'eigen_source: archive\n'
SMOKE = [('sw_smoke', 'v6e', 'tpu_xla', 'V6s'), ('sw_smoke', 'l4', 'l4_xla', 'Gs'), ('h100_smoke', 'h100', 'h100_xla', 'H')]


def sessions(source):
  if source == 'A':
    return [l.split('#')[0].strip() for l in (REPO / 'scripts' / 'analysis_runs' / 'a100' / 'sessions.txt').read_text().splitlines()
            if l.split('#')[0].strip() and re.search(r'_a100_a_\d', l)]
  pat = {'V5': r'_v5e_main_v5_', 'V6': r'_v6e_main_v6_', 'G': r'_l4_main_g_'}[source]
  return [s for s in (REPO / 'results' / 'analysis' / 'main_20261014' / 'sessions_main.txt').read_text().split()
          if re.search(pat, s)]


def stamp(s):
  return dt.datetime.strptime(s, '%Y-%m-%dT%H:%M:%SZ')


def timings(source, manifest):
  """{bucket: median fresh wall s} and the median overheads (min) of a source arm; H from A, inference scaled."""
  base = 'A' if source == 'H' else source
  by, over = {}, []
  for s in sessions(base):
    root = REPO / 'results' / 'af3' / s
    total = 0.0
    for run in sorted(p for p in root.iterdir() if p.is_dir()):
      if not (run / 'runs.jsonl').is_file():
        continue
      for line in (run / 'runs.jsonl').read_text().splitlines():
        r = json.loads(line)
        if r.get('exit_code') != 0 or not r.get('wall_seconds'):
          continue
        total += float(r['wall_seconds'])
        if run.name.endswith('_fresh'):
          w = float(r['wall_seconds'])
          if source == 'H':
            inf = sum((r.get('inference_seconds_by_seed') or {}).values())
            w -= (1 - H_OVER_A) * inf
          by.setdefault(int(manifest[r['pdb_id']]['bucket']), []).append(w)
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
  walls = {b: statistics.median(v) for b, v in by.items()}
  return walls, {k: statistics.median(o[k] for o in over) for k in ('provision', 'setup', 'teardown')}


def price(platform, prov):
  rows = [r for r in csv.DictReader(open(REPO / 'cloud' / 'prices.csv')) if r['platform'] == platform and r['usd_per_hour']
          and (platform not in REGIONS or r['region'] in REGIONS[platform])]
  cands = [r for r in rows if r['provisioning'] == prov] or [r for r in rows if r['provisioning'] == 'standard']
  return max(float(r['usd_per_hour']) for r in cands)


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


def stack_lines(platform):
  fam = 'tpu' if platform in ('v5e', 'v6e') else 'gpu'
  return 'stack:\n  ' + platform + ':\n' + ''.join(f'    {k}: {v}\n' for k, v in STACK[fam].items())


def plan_text(arm, i, n, ts, deadline, job_h):
  label, source, platform, config, seeds, rep_start, _, extra = ARMS[arm]
  seeds = list(seeds)
  why = {'seeds': f'seeds {seeds[0]} to {seeds[-1]} (the frozen inputs list 1 to 5)', 'sw': 'the alternative JAX/XLA stack',
         'h100': 'the chip (H100 80 GB, a3-highgpu-1g)'}[label]
  src = 'A (follow-up 3)' if source == 'H' else f'{source} (main study)'
  return (f'# Follow-up {"7" if label == "sw" else "8" if label == "seeds" else "9"} (drafts/section10_fu7-10.md; label {label}), '
          f'arm {arm.replace("_", " ")}, part {i} of {n}.\n'
          f'# The settings of {src}, config {config} on {platform}; only {why} changes.\n'
          f'# Generated by harness/plans/make_fu7_10_plans.py: {len(ts)} target(s), seeds {seeds[0]}..{seeds[-1]}, '
          f'r{rep_start}; expected job {job_h:.2f} h from the run records of {src}; deadline ceil(job min x 1.25 + 20).\n\n'
          f'description: follow-up {label}, {arm.replace("_", " ")} part {i} of {n}\nlabel: {label}\nweights: gcs\n'
          f'launch_by: {LAUNCH_BY}\nfinish_by: {FINISH_BY}\n' + EIGEN
          + (stack_lines(platform) if label == 'sw' else '') + (extra or '')
          + f'\nconfigs:\n  {platform}: {config}\n\ntargets:\n  default: [{", ".join(ts)}]\n\n'
          f'seeds: [{", ".join(map(str, seeds))}]\nfresh_reps: 1\nwarm_rerun_seed: none\nnum_recycles: default\n'
          f'num_diffusion_samples: default\nrun_timeout_min: 180\n\ndeadline_min:\n  {platform}: {deadline}\n')


def smoke_text(label, platform, config, arm):
  return (f'# Follow-up {"7" if label == "sw_smoke" else "9"} smoke test (label {label}; an engineering check, never a finding):\n'
          f'# 2PV7 (not a study target), its setup-check _data.json (inputs/official_data/smoke_manifest.csv, follow-up 6),\n'
          f'# run end to end on {platform} with the settings of {arm}{" and its alternative stack" if label == "sw_smoke" else ""}. '
          f'Seed 1, r1; deadline 60 min.\n\n'
          f'description: follow-up {label}, 2PV7 on {platform}\nlabel: {label}\nweights: gcs\n'
          f'manifest: inputs/official_data/smoke_manifest.csv\n' + EIGEN
          + (stack_lines(platform) if label == 'sw_smoke' else '')
          + f'\nconfigs:\n  {platform}: {config}\n\ntargets:\n  default: [2PV7]\n\nseeds: [1]\nfresh_reps: 1\n'
          f'warm_rerun_seed: none\nnum_recycles: default\nnum_diffusion_samples: default\nrun_timeout_min: 30\n\n'
          f'deadline_min:\n  {platform}: 60\n')


def eigen_check_text(platform):
  """The follow-up 6 smoke plan of this platform (harness/plans/msa_smoke/msa_smoke_<platform>.yaml) with only the
  Eigen source changed: its 2PV7 output must be bit-identical to that smoke test's (TPU compiles deterministically)."""
  src = (PLANS / 'msa_smoke' / f'msa_smoke_{platform}.yaml').read_text()
  body = src[src.index('\ndescription:') + 1:]
  body = body.replace(f'description: follow-up 6 smoke test, 2PV7 _data.json on {platform}\nlabel: msa_smoke\n',
                      f'description: Eigen archive check, 2PV7 _data.json on {platform}\nlabel: eigen_check\n')
  body = body.replace('manifest: inputs/official_data/smoke_manifest.csv\n', 'manifest: inputs/official_data/smoke_manifest.csv\n' + EIGEN)
  assert 'label: eigen_check' in body and EIGEN in body, platform
  return (f'# Eigen archive check (label eigen_check; an engineering check, never a finding or a contrast): the follow-up 6\n'
          f'# smoke plan harness/plans/msa_smoke/msa_smoke_{platform}.yaml (2PV7 _data.json, seed 1, r1, config tpu_xla) with\n'
          f'# only eigen_source: archive added, so AlphaFold3 is built with the Eigen of cloud/eigen_pin.sh from our bucket.\n'
          f'# Its 2PV7 output must be bit-identical to that smoke session\'s (scripts/monitor/eigen_check.sh). Generated by\n'
          f'# harness/plans/make_fu7_10_plans.py.\n\n' + body)


def build():
  manifest = {r['pdb_id']: r for r in csv.DictReader(open(REPO / 'inputs' / 'manifest.csv'))}
  subset = [r['pdb_id'] for r in csv.DictReader(open(REPO / 'targets' / 'posebusters_subset.csv'))]
  pilot = [t for t in subset if manifest[t]['pilot'] == 'yes']
  files, parts, cache = {}, [], {}
  for arm in ORDER:
    label, source, platform, config, seeds, rep_start, which, extra = ARMS[arm]
    if source not in cache:
      cache[source] = timings(source, manifest)
    walls, o = cache[source]
    ts_all = subset if which == 'all' else pilot
    n_seeds = len(list(seeds))
    load = lambda t: n_seeds * walls[int(manifest[t]['bucket'])]  # noqa: E731
    bins = pack([(t, load(t)) for t in ts_all], MAX_H[platform] * 3600 - o['setup'] * 60)
    for i, b in enumerate(bins, 1):
      name = f'{label}_{arm.lower()}_{i:02d}'
      job_min = o['setup'] + b['load'] / 60
      deadline = math.ceil(job_min * 1.25 + 20)
      life_h = math.ceil((deadline + 30) / 60) if platform == 'v5e' else (deadline + 30) / 60
      vm_h = (o['provision'] + job_min + o['teardown']) / 60
      ts = [t for t in subset if t in b['items']]
      files[f'harness/plans/{label}/{name}.yaml'] = plan_text(arm, i, len(bins), ts, deadline, job_min / 60)
      od, pl = price(platform, 'standard'), price(platform, PLANNED[platform])
      parts.append({'part': name, 'arm': arm, 'label': label, 'platform': platform, 'config': config,
                    'targets': len(ts), 'processes': len(ts) * n_seeds, 'expected_vm_h': round(vm_h, 2),
                    'deadline_min': deadline, 'lifetime_h': round(life_h, 2), 'usd_per_h_on_demand': od,
                    'usd_per_h_planned': pl, 'provisioning': PLANNED[platform],
                    'expected_usd_on_demand': round(od * vm_h, 2), 'worst_usd_on_demand': round(od * life_h, 2),
                    'expected_usd_planned': round(pl * vm_h, 2), 'worst_usd_planned': round(pl * life_h, 2)})
  for label, platform, config, arm in SMOKE:
    files[f'harness/plans/{label}/{label}_{platform}.yaml'] = smoke_text(label, platform, config, arm)
  buf = io.StringIO()
  w = csv.DictWriter(buf, fieldnames=list(parts[0]), lineterminator='\n')
  w.writeheader()
  w.writerows(parts)
  files['harness/plans/fu7_10_parts.csv'] = buf.getvalue()
  for arm in ORDER:
    files[f'results/parts_{arm.lower()}.txt'] = (f'# arm {arm} (generated by harness/plans/make_fu7_10_plans.py)\n'
                                                 + ''.join(f'{p["platform"]} harness/plans/{p["label"]}/{p["part"]}.yaml\n'
                                                           for p in parts if p['arm'] == arm))
  files['results/parts_sw_smoke.txt'] = ''.join(f'{pl} harness/plans/sw_smoke/sw_smoke_{pl}.yaml\n' for pl in ('v6e', 'l4'))
  files['results/parts_h100_smoke.txt'] = 'h100 harness/plans/h100_smoke/h100_smoke_h100.yaml\n'
  for pl in ('v5e', 'v6e'):
    files[f'harness/plans/eigen_check/eigen_check_{pl}.yaml'] = eigen_check_text(pl)
  files['results/parts_eigen_check.txt'] = ''.join(f'{pl} harness/plans/eigen_check/eigen_check_{pl}.yaml\n' for pl in ('v5e', 'v6e'))
  L = ['# Follow-ups 7 to 9: parts and cost (generated by harness/plans/make_fu7_10_plans.py)', '',
       '| arm | label | platform | parts | processes | VM h | on demand: expected, worst USD | planned: expected, worst USD |',
       '|---|---|---|---|---|---|---|---|']
  for arm in ORDER:
    ps = [p for p in parts if p['arm'] == arm]
    s = lambda k: sum(p[k] for p in ps)  # noqa: E731
    L.append(f'| {arm.replace("_", " ")} | {ps[0]["label"]} | {ps[0]["platform"]} | {len(ps)} | {s("processes")} | '
             f'{s("expected_vm_h"):.1f} | {s("expected_usd_on_demand"):.2f}, {s("worst_usd_on_demand"):.2f} | '
             f'{s("expected_usd_planned"):.2f}, {s("worst_usd_planned"):.2f} ({ps[0]["provisioning"]}) |')
  L += ['', 'Process wall per bucket (median s, fresh): ' + '; '.join(
      f'{src}: ' + ', '.join(f'{b} {cache[src][0][b]:.0f}' for b in sorted(cache[src][0])) for src in cache) + '.',
      'Overheads (median min): ' + '; '.join(f'{src}: provisioning {cache[src][1]["provision"]:.1f}, setup '
                                             f'{cache[src][1]["setup"]:.1f}, teardown {cache[src][1]["teardown"]:.1f}'
                                             for src in cache) + '.']
  files['harness/plans/fu7_10_table.md'] = '\n'.join(L) + '\n'
  return files


def main():
  ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
  ap.add_argument('--check', action='store_true')
  args = ap.parse_args()
  files = build()
  if args.check:
    absent = [p for p in files if p.startswith('results/') and not (REPO / p).is_file()]
    bad = [p for p, t in files.items() if p not in absent and (not (REPO / p).is_file() or (REPO / p).read_text() != t)]
    print('up to date' if not bad else 'DIFFERS: ' + ', '.join(bad))
    return 1 if bad else 0
  for label in ('seeds', 'sw', 'h100', 'sw_smoke', 'h100_smoke'):
    for old in (PLANS / label).glob('*.yaml'):
      old.unlink()
  for p, t in files.items():
    (REPO / p).parent.mkdir(parents=True, exist_ok=True)
    (REPO / p).write_text(t)
  print(files['harness/plans/fu7_10_table.md'])
  return 0


if __name__ == '__main__':
  sys.exit(main())
