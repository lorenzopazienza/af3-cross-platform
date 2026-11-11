"""Warm cost on the L4 Triton and the H100 (label warm_th; notes/analysis_plan.md section 10, the specification row of
2026-11-11 "Warm cost per structure on the L4 Triton and the H100"): Table III's warm cost per structure at 1,024
tokens for GT and H, descriptive, outside every contrast.

    analysis/.venv/bin/python analysis/collect.py --labels warm_th --sessions scripts/analysis_runs/warm_th/sessions.txt \\
        --out results/analysis/warm_th_<DATE> --no_m1
    analysis/.venv/bin/python analysis/warm_th.py --samples results/analysis/warm_th_<DATE>/samples.csv \\
        --sessions scripts/analysis_runs/warm_th/sessions.txt --out results/analysis/warm_th_<DATE>

Sessions: the ones listed in --sessions (the complete session of each part, from the section 10 launch rows; an
interrupted session is recorded there and not listed), exactly one per arm: <stamp>_l4_warm_th_gt_01 (GT, l4_triton on
g2-standard-8) and <stamp>_h100_warm_th_h_01 (H, h100_xla on a3-highgpu-1g). analysis/collect.py codes the arms from
the device and the attention (GT, H). M1 is not computed (collect.py --no_m1): no prediction enters any finding.
Checks, before anything is computed; the analysis stops after writing checks.csv and report.md if one fails, and then
writes no summary.csv (the arm is reported as not completed in section 10):
  0 sessions and arms: every listed session has label warm_th, exactly one per arm, the arm's configuration and
    machine type; every row at AlphaFold3's defaults (10 recycles, 5 samples);
  completeness: per arm, the 10 pilot targets (inputs/manifest.csv), seed 1, each with one fresh process (r1) and
    one warm process (w);
  1 every process exit 0 (followup34.exit_code_check);
  2a, 2b and bit-identity, as the main study's check 2 (contrasts.cache_checks and contrasts.check2): every fresh
    process compiled from an empty cache, every warm process loaded jit_apply_fn with a key its own fresh process
    wrote, and every warm process's outputs equal to its fresh process's, every file byte for byte except the mmCIF
    timestamp line (contrasts.outputs_identical; FETCH=full brings the files). The warm-against-r1 M4 coordinates
    row of the main study is not computed (no RMSD here; the file comparison covers the coordinates).
Measure (M6, plan section 4, interpretations 2 and 3 of section 10's entry of 2026-10-09): analysis/cost.py's
processes, cost_table and summary_table, unchanged: cost per structure = process wall / seeds x the hourly price of
cloud/prices.csv in the region of the run's zone (europe-west4 when the region has no row, noted), on demand, Spot
and Flex-start; warm processes are reported as measured, not amortised. Table III reads the warm row at bucket 1,024,
on-demand price (median_usd_per_structure_standard), as every cost cell of that table, for H too although H ran on
Spot; H's Spot cost (median_usd_per_structure_spot, the Spot price of cloud/prices.csv, read on the date its read_on
column gives) is in the same row and is declared in the report. The fresh rows are written too (the warm run's
source and the run's cost); they enter no comparison and no Table III cell.
Outputs in --out (a new folder; an existing summary.csv stops the run): checks.csv, summary.csv (the columns of
results/analysis/main_20261014/primary_v2/cost/cost_summary.csv, rows GT and H, fresh and warm, every bucket),
cost_per_process.csv, prices_used.csv, report.md, inputs.tsv (path and SHA-256 of every file read); collect.py writes
samples.csv and excluded_sessions.csv, scripts/analysis_runs/warm_th/reproduce.sh writes run.log.
"""
import argparse
import hashlib
import os
import pathlib
import re
import sys
import types

import numpy as np
import pandas as pd

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / 'analysis'))
import contrasts  # noqa: E402
import cost  # noqa: E402
import followup34 as f34  # noqa: E402

LABEL = 'warm_th'
ARMS = ['GT', 'H']
# arm: (session platform, plan part, configuration, machine type)
ARM_RUN = {'GT': ('l4', 'gt', 'l4_triton', 'g2-standard-8'), 'H': ('h100', 'h', 'h100_xla', 'a3-highgpu-1g')}
SESSION = re.compile(r'\d{8}T\d{6}Z_(l4|h100)_warm_th_(gt|h)_01$')
SEED = 1
BUCKET = 1024


def sha256(path):
  return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def read_sessions(path):
  return [l.split('#')[0].strip() for l in pathlib.Path(path).read_text().splitlines() if l.split('#')[0].strip()]


def pilot_targets(manifest=REPO / 'inputs' / 'manifest.csv'):
  m = pd.read_csv(manifest)
  return sorted(m[m.pilot.astype(str).str.strip().str.lower().isin(['1', 'true', 'yes'])].pdb_id)


def session_folders(col):
  """collect.py's session column is the harness run folder (results/af3/<session>/<run>): its session folder name."""
  return col.astype(str).map(lambda s: pathlib.PurePosixPath(s).parent.name if '/' in s else s)


def session_check(d, sessions):
  """Check 0: the listed sessions are the warm_th parts, one per arm, with the arm's configuration and machine type."""
  bad = []
  per_arm = {}
  for s in sessions:
    m = SESSION.match(s)
    if not m:
      bad.append(f'{s}: not a warm_th session name')
      continue
    arm = {'gt': 'GT', 'h': 'H'}[m.group(2)]
    if ARM_RUN[arm][0] != m.group(1):
      bad.append(f'{s}: platform {m.group(1)} for arm {arm}')
    per_arm.setdefault(arm, []).append(s)
  for arm in ARMS:
    if len(per_arm.get(arm, [])) != 1:
      bad.append(f'arm {arm}: {len(per_arm.get(arm, []))} listed sessions, one required')
  listed = set(sessions)
  if set(d.session_folder) != listed:
    bad.append(f'samples.csv sessions {sorted(set(d.session_folder))} are not the listed {sorted(listed)}')
  if (d.label != LABEL).any():
    bad.append(f'labels {sorted(set(d.label))}, not {LABEL}')
  if ((d.num_recycles != 10) | (d.num_diffusion_samples != 5)).any():
    bad.append('a row not at AlphaFold3\'s defaults (10 recycles, 5 samples)')
  for arm in ARMS:
    _, part, config, machine = ARM_RUN[arm]
    g = d[d.session_folder.isin(per_arm.get(arm, []))]
    if len(g) and (set(g.platform) != {arm} or set(g.config) != {config} or set(g.machine_type.astype(str)) != {machine}):
      bad.append(f'arm {arm}: platform {sorted(set(g.platform))}, config {sorted(set(g.config))}, machine type '
                 f'{sorted(set(g.machine_type.astype(str)))}; expected {arm}, {config}, {machine}')
  return {'check': '0 sessions and arms', 'condition': ', '.join(ARMS), 'value': f'{len(sessions)} sessions',
          'detail': 'pass' if not bad else 'FAIL: ' + '; '.join(bad)[:600]}


def completeness(d, targets):
  rows = []
  procs = d.drop_duplicates(['session', 'target', 'seed', 'repetition'])
  for arm in ARMS:
    g = procs[procs.platform == arm]
    bad = []
    for rep in ('r1', 'w'):
      h = g[g.repetition == rep]
      got = sorted(h.target)
      if sorted(set(got)) != targets or len(got) != len(targets) or set(h.seed) != {SEED}:
        bad.append(f'{rep}: {len(got)} processes, targets missing {sorted(set(targets) - set(got))}, seeds {sorted(set(h.seed))}')
    other = sorted(set(g.repetition) - {'r1', 'w'})
    if other:
      bad.append(f'other repetitions {other}')
    rows.append({'check': 'completeness: 10 pilot targets, seed 1, r1 and w', 'condition': arm,
                 'value': f'{len(g)} processes', 'detail': 'pass' if not bad else 'FAIL: ' + '; '.join(bad)[:600]})
  return rows


def checks(d, sessions, targets):
  rows = [session_check(d, sessions)]
  if rows[0]['detail'] != 'pass':
    return rows
  rows += completeness(d, targets)
  rows += f34.exit_code_check(d, ARMS, LABEL)
  cache_rows = contrasts.cache_checks(d)
  for arm in ARMS:
    rows += contrasts.check2(d, pd.DataFrame(columns=['contrast']), cache_rows, arm)
  return rows


def measure(samples):
  """cost.py's per-process table and summary for the arms, as analysis/cost.py computes them."""
  args = types.SimpleNamespace(samples=str(samples), all_settings=False, labels=LABEL, include=None, exclude=None)
  p = cost.processes(args)
  p = p[p.platform.isin(ARMS)]
  prices = cost.load_prices()
  c = cost.cost_table(p, prices, cost.batch_sizes('subset'))
  s = cost.summary_table(c)
  # The price rows the cost table used: per arm and zone region, every provisioning, through cost.price_for itself.
  pu = {}
  for a, z in sorted({(a, z) for a, z in zip(c.platform, c.zone.astype(str))}):
    region = z.rsplit('-', 1)[0] if '-' in z else cost.FALLBACK_REGION
    for prov in cost.PROVISIONING:
      price, read_on, note = cost.price_for(prices, a, z, prov)
      pu[(cost.PRICE_KEY[a], region, prov)] = {'platform': cost.PRICE_KEY[a], 'region': region, 'provisioning': prov,
                                               'usd_per_hour': price, 'read_on': read_on, 'note': note}
  return c, s, pd.DataFrame(list(pu.values()))


def fmt(x, nd=3):
  return '-' if x is None or (isinstance(x, float) and np.isnan(x)) else f'{x:.{nd}g}' if isinstance(x, float) else str(x)


def report(out, rows, s, pu, sessions):
  ok = all(r['detail'] == 'pass' for r in rows)
  L = ['# Warm cost on the L4 Triton and the H100 (warm_th)', '',
       'notes/analysis_plan.md section 10, specification row of 2026-11-11; descriptive, outside every contrast. Code: '
       'analysis/warm_th.py (analysis/cost.py and the check 2 of analysis/contrasts.py, unchanged). Sessions: '
       + ', '.join(sessions) + '.', '', '## Checks (before any result)', '', '| check | condition | value | detail |',
       '|---|---|---|---|']
  L += [f'| {r["check"]} | {r["condition"]} | {r["value"]} | {r["detail"]} |' for r in rows]
  if not ok:
    L += ['', 'A check failed: no cost is computed and summary.csv is not written. Section 10 reports the arm as not '
          'completed.']
    (out / 'report.md').write_text('\n'.join(L) + '\n')
    return
  L += ['', '## Cost per structure (M6; median over processes, USD)', '',
        '| arm | bucket | cache | processes | median wall s | median compile s | on demand | Spot |', '|---|---|---|---|---|---|---|---|']
  for r in s.sort_values(['platform', 'cache', 'bucket']).itertuples():
    L.append(f'| {r.platform} | {r.bucket} | {r.cache} | {r.processes} | {fmt(r.median_wall_s, 4)} | '
             f'{fmt(r.median_compile_s)} | {fmt(r.median_usd_per_structure_standard)} | '
             f'{fmt(r.median_usd_per_structure_spot)} |')
  w = s[(s.cache == 'warm') & (s.bucket == BUCKET)].set_index('platform')
  L += ['', f'Table III (warm, bucket {BUCKET}, on demand as every cost cell there): '
        + '; '.join(f'{a} {fmt(w.loc[a, "median_usd_per_structure_standard"])} USD ({int(w.loc[a, "processes"])} '
                    'processes)' for a in ARMS if a in w.index) + '.',
        'H ran on Spot: its Spot cost at bucket 1024 is '
        + (fmt(w.loc['H', 'median_usd_per_structure_spot']) if 'H' in w.index else '-')
        + ' USD per structure, at the Spot prices of cloud/prices.csv read on '
        + ', '.join(sorted(set(pu[(pu.platform == 'h100') & (pu.provisioning == 'spot')].read_on.dropna().astype(str))))
        + '. The fresh rows are the warm runs\' sources; they enter no comparison and no Table III cell.', '',
        '## Prices used', '', '| platform | region | provisioning | USD/h | read on | note |', '|---|---|---|---|---|---|']
  L += [f'| {r.platform} | {r.region} | {r.provisioning} | {fmt(r.usd_per_hour, 7)} | {r.read_on} | {r.note or ""} |'
        for r in pu.itertuples()]
  (out / 'report.md').write_text('\n'.join(L) + '\n')


def main():
  ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
  ap.add_argument('--samples', required=True)
  ap.add_argument('--sessions', required=True)
  ap.add_argument('--out', required=True)
  ap.add_argument('--manifest', default=str(REPO / 'inputs' / 'manifest.csv'))
  args = ap.parse_args()
  out = pathlib.Path(args.out)
  if (out / 'summary.csv').exists():
    raise SystemExit(f'{out}/summary.csv exists; the output folder must be new')
  out.mkdir(parents=True, exist_ok=True)
  sessions = read_sessions(args.sessions)
  if not sessions:
    raise SystemExit(f'{args.sessions} lists no session')
  d = pd.read_csv(args.samples, low_memory=False)
  d['condition'] = d.platform
  d['session_folder'] = session_folders(d.session)
  rows = checks(d, sessions, pilot_targets(args.manifest))
  pd.DataFrame(rows).to_csv(out / 'checks.csv', index=False)
  read = [pathlib.Path(args.sessions), pathlib.Path(args.samples), pathlib.Path(args.manifest), REPO / 'cloud' / 'prices.csv']
  rel = lambda p: os.path.relpath(os.path.abspath(p), REPO) if os.path.abspath(p).startswith(str(REPO) + os.sep) \
      else pathlib.Path(p).name  # noqa: E731
  pd.DataFrame([{'path': rel(p), 'sha256': sha256(p)} for p in dict.fromkeys(read)]).to_csv(
      out / 'inputs.tsv', sep='\t', index=False)
  for r in rows:
    print(f'>> check {r["check"]} [{r["condition"]}]: {r["value"]} {r["detail"]}', flush=True)
  if not all(r['detail'] == 'pass' for r in rows):
    report(out, rows, None, None, sessions)
    raise SystemExit('a check failed: no cost computed, summary.csv not written (see checks.csv)')
  c, s, pu = measure(args.samples)
  c.to_csv(out / 'cost_per_process.csv', index=False)
  s.to_csv(out / 'summary.csv', index=False)
  pu.to_csv(out / 'prices_used.csv', index=False)
  report(out, rows, s, pu, sessions)
  for r in s[(s.cache == 'warm') & (s.bucket == BUCKET)].itertuples():
    print(f'>> {r.platform} warm {BUCKET}: {r.median_usd_per_structure_standard:.4g} USD per structure on demand, '
          f'{r.median_usd_per_structure_spot:.4g} Spot ({r.processes} processes)', flush=True)
  return 0


if __name__ == '__main__':
  sys.exit(main())
