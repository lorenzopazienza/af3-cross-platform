"""Writes the memory-limit plans for the A100 and the H100 (label memlimit_ah; drafts/section10_memlimit_ah.md, the
specification row), the measurement of harness/plans/memlimit/ (section 10, rows of 2026-10-14) on arms A and H.

    python3 harness/plans/memlimit_ah/make_memlimit_ah_plans.py [--check]

Everything that defines the measurement is imported from harness/plans/memlimit/make_memlimit_plans.py, unchanged:
target 7D5C, seed 1, the padding sizes (1,280 to 5,120 tokens, one fresh process per size with AlphaFold3's own
--buckets), stop rule first_failure, 4 hours per process, and the pessimistic model that sets each deadline (median
wall W of a 7D5C process at the 1,024-token bucket, times (size / 1024)^3, sizes up to the first above 240 min, which
counts 240 min; deadline ceil(job min x 1.25 + 20)). Arms, with the settings of follow-ups 3 and 9:
  A  a100_xla on a2-ultragpu-1g (one A100 80 GB), on demand (the Spot A100 80 GB quota is 0), the launcher's
     A100 zones; W from follow-up 3's 7D5C processes of A, r1 and r2 (results/analysis/a100_20261018/samples.csv);
  H  h100_xla on a3-highgpu-1g (one H100 80 GB), Spot (the on-demand H100 quota is 0), the launcher's H100 zones;
     W from follow-up 9's 7D5C processes of H and H r2 (results/analysis/h100_20261104/samples.csv).
Both plans take the Eigen source from our bucket (eigen_source: archive, as every follow-up since 7; the same commit,
so the same files). Costs (cloud/prices.csv; the highest price among the regions each launcher may use):
  worst     the VM lifetime the deadline sets (deadline + 30 min) at that price;
  expected  provisioning, setup and teardown (follow-up 3 and 9 medians are not separated here: 30 min in all, as
            a round allowance) plus every size run to completion, each size's wall W x r(size), r the ratio of
            AlphaFold3's published compile-free time at that size to its time at 1,024 tokens (docs/performance.md,
            A100 80 GB and H100 80 GB columns: 1,024, 2,048, 3,072, 4,096, 5,120 tokens), interpolated log-log for the
            sizes in between. AlphaFold3 states that both cards fold up to 5,120 tokens with the configurations' memory
            settings, so the expected case runs every size.
Writes harness/plans/memlimit_ah/memlimit_ah_a_01.yaml, memlimit_ah_h_01.yaml, memlimit_ah_parts.csv, table.md and
results/parts_memlimit_ah.txt (outside git). --check writes nothing and exits 1 if a tracked file differs.
"""
import argparse
import csv
import io
import math
import pathlib
import statistics
import sys

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO / 'harness' / 'plans' / 'memlimit'))
from make_memlimit_plans import SEED, SIZES, TARGET, TIMEOUT_MIN, model  # noqa: E402

LABEL = 'memlimit_ah'
SPEC = 'drafts/section10_memlimit_ah.md, the specification row (section 10 once entered)'
OVERHEAD_MIN = 30
# AlphaFold3 docs/performance.md, compile-free seconds on one A100 80 GB and one H100 80 GB.
DOCS = {'A': {1024: 62, 2048: 275, 3072: 703, 4096: 1434, 5120: 2547},
        'H': {1024: 34, 2048: 144, 3072: 367, 4096: 774, 5120: 1416}}
# arm: (platform, config, samples.csv of its follow-up, session label, provisioning, machine type)
ARMS = {'A': ('a100', 'a100_xla', 'results/analysis/a100_20261018/samples.csv', 'a100', 'standard', 'a2-ultragpu-1g'),
        'H': ('h100', 'h100_xla', 'results/analysis/h100_20261104/samples.csv', 'h100', 'spot', 'a3-highgpu-1g')}
REGIONS = {'a100': None, 'h100': ['us-central1', 'us-east4', 'us-east5']}   # as cloud/af3_run.sh's zone lists


def wall_1024(arm):
  platform, config, samples, label, _, _ = ARMS[arm]
  rows = [r for r in csv.DictReader(open(REPO / samples)) if r['target'] == TARGET and r['platform'] == arm
          and r['config'] == config and r['repetition'] in ('r1', 'r2') and r['sample'] == '0' and r['label'] == label]
  if not rows:
    raise SystemExit(f'no {TARGET} r1/r2 process of {arm} in {samples}')
  return statistics.median(float(r['wall_s']) for r in rows), len(rows)


def ratio(arm, size):
  """AlphaFold3's published time at size over its time at 1,024 tokens, log-log between the published sizes."""
  d = DOCS[arm]
  pts = sorted(d)
  lo = max(p for p in pts if p <= size)
  hi = min(p for p in pts if p >= size)
  t = d[lo] if lo == hi else d[lo] * (size / lo) ** (math.log(d[hi] / d[lo]) / math.log(hi / lo))
  return t / d[1024]


def price(platform, prov):
  rows = [r for r in csv.DictReader(open(REPO / 'cloud' / 'prices.csv')) if r['platform'] == platform
          and r['provisioning'] == prov and r['usd_per_hour'] and (not REGIONS[platform] or r['region'] in REGIONS[platform])]
  return max(float(r['usd_per_hour']) for r in rows)


def build():
  files, parts = {}, []
  for arm, (platform, config, samples, _, prov, machine) in ARMS.items():
    wall, n = wall_1024(arm)
    steps, job = model(wall)
    deadline = math.ceil(job * 1.25 + 20)
    last = steps[-1]
    note = (f'{last[0]} would exceed {TIMEOUT_MIN} min and stop the plan' if last[1] == TIMEOUT_MIN
            else 'every size within the limit')
    exp_min = OVERHEAD_MIN + sum(wall * ratio(arm, s) for s in SIZES) / 60
    life_h = (deadline + 30) / 60
    usd = price(platform, prov)
    name = f'{LABEL}_{arm.lower()}_01'
    files[f'harness/plans/{LABEL}/{name}.yaml'] = f'''# Memory limit per chip (M7) on {arm} ({SPEC}; label {LABEL}): the measurement of
# harness/plans/memlimit/ (section 10, rows of 2026-10-14) with the configuration of follow-up {"3" if arm == "A" else "9"} ({config} on {machine}).
# Generated by make_memlimit_ah_plans.py: 7D5C, seed 1, one fresh process per bucket size, stop rule first_failure,
# 4 h per process. Deadline from the pessimistic model of make_memlimit_plans.py: median wall of a 7D5C process of {arm}
# {wall:.0f} s ({n} processes, r1 and r2), x (size/1024)^3; modelled job {job:.0f} min ({note});
# deadline = ceil(job min x 1.25 + 20).

description: memory limit per chip (M7), {arm}
label: {LABEL}
weights: gcs
eigen_source: archive

configs:
  {platform}: {config}

targets:
  default: [{TARGET}]

seeds: [{SEED}]
fresh_reps: 1
warm_rerun_seed: none
buckets: [{", ".join(map(str, SIZES))}]
stop_rule: first_failure
num_recycles: default
num_diffusion_samples: default
run_timeout_min: {TIMEOUT_MIN}

deadline_min:
  {platform}: {deadline}
'''
    parts.append({'part': name, 'arm': arm, 'label': LABEL, 'platform': platform, 'config': config,
                  'machine_type': machine, 'provisioning': prov, 'wall_1024_s': round(wall, 1), 'wall_1024_n': n,
                  'model_job_min': round(job), 'deadline_min': deadline, 'expected_vm_h': round(exp_min / 60, 2),
                  'lifetime_h': round(life_h, 2), 'usd_per_h': usd, 'expected_usd': round(usd * exp_min / 60, 2),
                  'worst_usd': round(usd * life_h, 2)})
  buf = io.StringIO()
  w = csv.DictWriter(buf, fieldnames=list(parts[0]), lineterminator='\n')
  w.writeheader()
  w.writerows(parts)
  files[f'harness/plans/{LABEL}/{LABEL}_parts.csv'] = buf.getvalue()
  files[f'results/parts_{LABEL}.txt'] = (
      f'# Memory limit on A and H (label {LABEL}; generated by make_memlimit_ah_plans.py). A on demand, H Spot;\n'
      '# cloud/af3_run_many.sh needs A100_MACHINE_TYPE=a2-ultragpu-1g for the A100 part.\n'
      + ''.join(f'{p["platform"]} harness/plans/{LABEL}/{p["part"]}.yaml SPOT={1 if p["provisioning"] == "spot" else 0}\n'
                for p in parts))
  L = [f'# Memory limit on A and H: plans and cost (generated by make_memlimit_ah_plans.py)', '',
       '| plan | platform | config | provisioning | W(1,024) s | modelled job min | deadline min | expected VM h | '
       'expected USD | lifetime h | worst USD |', '|---|---|---|---|---|---|---|---|---|---|---|']
  L += [f'| {p["part"]} | {p["platform"]} ({p["machine_type"]}) | {p["config"]} | {p["provisioning"]} | {p["wall_1024_s"]} | '
        f'{p["model_job_min"]} | {p["deadline_min"]} | {p["expected_vm_h"]} | {p["expected_usd"]} | {p["lifetime_h"]} | '
        f'{p["worst_usd"]} |' for p in parts]
  L += [f'| all | | | | | | | {sum(p["expected_vm_h"] for p in parts):.2f} | {sum(p["expected_usd"] for p in parts):.2f} | | '
        f'{sum(p["worst_usd"] for p in parts):.2f} |', '',
        'Expected: every size completes (AlphaFold3 docs/performance.md: both cards fold up to 5,120 tokens), each size '
        'W x r(size), r from the published compile-free times, plus 30 min of provisioning, setup and teardown. Worst: '
        'the VM lifetime set by the deadline. Prices: the highest of the regions each launcher may use.']
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
