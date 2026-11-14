"""Returned structures across platforms (section 10, row of drafts/section10_returned_spec_20261114.md, label
returned): a post hoc analysis of frozen outputs, asked by the reviewers, descriptive whatever it shows. No run and no
cloud resource.

    analysis/.venv/bin/python analysis/returned_structures.py --main . --out results/analysis/returned_<DATE> \\
        [--replicates 10000] [--seed 2026]

Question: does the platform change the quality of the structure AlphaFold3 returns? For each paired prediction
(target, seed) of a contrast, the top-ranked sample on side A and the top-ranked sample on side B (the registered
top-1 of the contrast's pairs_seed.csv) are each compared with the experimental structure, and the two qualities are
differenced: B minus A, A the first-named side.

Inputs, relative to --main (the checkout that holds results/):
  results/analysis/main_20261014/primary_v2/samples.csv and contrasts/pairs_seed.csv   (label main)
  results/analysis/cpu60_20261018/samples.csv and pairs_seed.csv                       (label cpu60, C only)
  results/analysis/fu13_20261112/samples.csv                                           (V6f, Gf: label why)
  results/analysis/fu14_20261113b/samples.csv and pairs_seed.csv                       (label f32)
Contrasts (source folder: contrasts):
  main:  HW(V5,V6), HW(V6,G), HW(G,C) (the three primary pairs); yardsticks RC(G) and SD(V6)
  cpu60: HW(G,C) on the 60 targets (follow-up 4)
  fu14:  the ten float32 pairs HW(Pf,Qf) of follow-up 14 (four primary, six secondary); yardstick RC(Gf)
Sides: HW(P,Q) is P r1 against Q r1 on the same seed; RC(P) is P r1 against P r2; SD(P) is seed s against seed s' of
P r1 (every pair of seeds of a target, as the registered SD rows).
Metrics per side: m1_ligand_rmsd of the top-1 sample (ligand RMSD in angstrom from the experimental structure after
pocket superposition, M1 of the plan), iptm and mean_plddt of the same sample. Paired differences d = B - A. A unit
without the RMSD on either side leaves the RMSD statistics (counted in checks.csv); iptm and mean_plddt likewise.
Strata: every paired unit ("all"), and the units where the registered top-1 differs ("flipped"; not for SD, whose
sample indices are unrelated draws).
Statistics per contrast, stratum and metric: median and mean of d, with 95% percentile intervals from the bootstrap
over targets (analysis/contrasts.boot_ci: targets resampled with replacement, units of a drawn target pooled, a fresh
generator from --seed for every statistic, so every statistic sees the same resampled targets); also the median of |d|.
Reading rule (ligand RMSD, median of d, stratum "all" primary, "flipped" beside it): B returns poses farther from the
experimental structure if the interval lies above 0, closer if it lies below 0, "not distinguishable" otherwise; and
"within 0.1 A" if the whole interval lies in [-0.1, 0.1] A (EQUIVALENCE_A). iptm and mean_plddt: the same three-way
reading, descriptive, without a band.
Checks (the analysis stops if 1 or 2 fails): 1, every top-1 of pairs_seed.csv names exactly one row of the samples
table (condition, repetition, target, seed, sample); 2, that row is the argmax of ranking_score of its (condition,
repetition, target, seed) under contrasts.top, and its success (m1_success) equals the registered success_a or
success_b; 3 (reported), units without RMSD, iptm or mean_plddt per contrast and side.
Outputs in --out: units.csv, quality.csv, checks.csv, inputs.tsv, report.md.
"""
import argparse
import hashlib
import pathlib
import re
import sys

import numpy as np
import pandas as pd

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / 'analysis'))
import contrasts  # noqa: E402
import followup34 as f34  # noqa: E402

AN = 'results/analysis'
MAIN = f'{AN}/main_20261014/primary_v2'
CPU60 = f'{AN}/cpu60_20261018'
FU13 = f'{AN}/fu13_20261112'
FU14 = f'{AN}/fu14_20261113b'
EQUIVALENCE_A = 0.1
METRICS = [('ligand_rmsd', 'm1_ligand_rmsd'), ('iptm', 'iptm'), ('mean_plddt', 'mean_plddt')]
FU14_PAIRS = ['HW(V5f,V6f)', 'HW(V6f,Gf)', 'HW(Hf,Af)', 'HW(Gf,Af)', 'HW(V5f,Gf)', 'HW(V5f,Af)', 'HW(Hf,V5f)',
              'HW(V6f,Af)', 'HW(Hf,V6f)', 'HW(Hf,Gf)']
# source -> [(contrast, family)]
CONTRASTS = {
    'main': [('HW(V5,V6)', 'primary pair'), ('HW(V6,G)', 'primary pair'), ('HW(G,C)', 'primary pair'),
             ('RC(G)', 'yardstick'), ('SD(V6)', 'yardstick')],
    'cpu60': [('HW(G,C)', 'follow-up 4')],
    'fu14': [(c, 'float32 pair, follow-up 14') for c in FU14_PAIRS] + [('RC(Gf)', 'yardstick, float32')],
}


def sides(name):
  """(condition, repetition) of side A and side B of a contrast name."""
  m = re.fullmatch(r'HW\((\w+),(\w+)\)', name)
  if m:
    return (m.group(1), 'r1'), (m.group(2), 'r1')
  m = re.fullmatch(r'(RC|SD)\((\w+)\)', name)
  if m:
    c = m.group(2)
    return ((c, 'r1'), (c, 'r2')) if m.group(1) == 'RC' else ((c, 'r1'), (c, 'r1'))
  raise ValueError(f'no sides for contrast {name}')


# ---------------------------------------------------------------- loading

def load_sources(root):
  """{source: (samples table with condition, pairs_seed.csv)}; the samples tables as the registered analyses build
  them (followup34.load_samples, fu14.load_f32 and fu14.load_fu13), sample rows only."""
  import fu14  # noqa: E402 (imported here: it is only needed for the real inputs)
  root = pathlib.Path(root)
  main = f34.load_samples(root / MAIN / 'samples.csv', ['main'])
  c60 = f34.load_samples(root / CPU60 / 'samples.csv', ['cpu60'])
  f32 = fu14.load_f32(root / FU14 / 'samples.csv')
  f13 = fu14.load_fu13(root / FU13 / 'samples.csv')
  out = {
      'main': (f34.union(main), pd.read_csv(root / MAIN / 'contrasts' / 'pairs_seed.csv')),
      'cpu60': (f34.union(main[main.condition != 'C'], pd.concat([main[main.condition == 'C'], c60[c60.condition == 'C']])),
                pd.read_csv(root / CPU60 / 'pairs_seed.csv')),
      'fu14': (f34.union(f32, f13), pd.read_csv(root / FU14 / 'pairs_seed.csv')),
  }
  inputs = [MAIN + '/samples.csv', MAIN + '/contrasts/pairs_seed.csv', CPU60 + '/samples.csv', CPU60 + '/pairs_seed.csv',
            FU13 + '/samples.csv', FU14 + '/samples.csv', FU14 + '/pairs_seed.csv']
  return out, inputs


# ---------------------------------------------------------------- units

def unit_table(samples, pairs, contrast, family, source):
  """One row per registered (target, seed pair) of the contrast, with both top-1 samples' metrics. Checks 1 and 2
  raise SystemExit."""
  sp = pairs[pairs.contrast == contrast]
  if sp.empty:
    raise SystemExit(f'{source}: {contrast} has no rows in pairs_seed.csv')
  (ca, ra), (cb, rb) = sides(contrast)
  units = contrasts.seed_units(samples[samples.condition.isin([ca, cb])])
  rows = []
  for r in sp.itertuples():
    row = {'source': source, 'family': family, 'contrast': contrast, 'target': r.target, 'seed_a': int(r.seed_a),
           'seed_b': int(r.seed_b), 'top1_a': int(r.top1_a), 'top1_b': int(r.top1_b),
           'flip_top1': bool(r.flip_top1)}
    for side, (c, rep, seed, top1, succ) in (('a', (ca, ra, r.seed_a, r.top1_a, r.success_a)),
                                              ('b', (cb, rb, r.seed_b, r.top1_b, r.success_b))):
      g = units.get((c, rep, r.target, int(seed)))
      if g is None or int(top1) not in g.index:
        raise SystemExit(f'check 1: {source} {contrast} {r.target} seed {seed}: no sample {top1} of {c} {rep}')
      t, s, _ = contrasts.top(g)
      if int(t) != int(top1) or bool(s) != bool(succ):
        raise SystemExit(f'check 2: {source} {contrast} {r.target} seed {seed}: registered top-1 {top1} '
                         f'(success {succ}) but the samples give {t} (success {s})')
      x = g.loc[int(top1)]
      for name, col in METRICS:
        row[f'{name}_{side}'] = float(x[col]) if pd.notna(x[col]) else np.nan
    for name, _ in METRICS:
      row[f'd_{name}'] = row[f'{name}_b'] - row[f'{name}_a']
    rows.append(row)
  return pd.DataFrame(rows)


# ---------------------------------------------------------------- statistics

def reading(lo, hi, band=None):
  if lo != lo or hi != hi:
    return 'not computed'
  s = 'B higher' if lo > 0 else ('B lower' if hi < 0 else 'not distinguishable')
  if band is not None and -band <= lo and hi <= band:
    s += f'; within {band:g}'
  return s


def summarize(units, replicates, seed):
  rows = []
  for (source, contrast), u in units.groupby(['source', 'contrast'], sort=False):
    strata = [('all', u)]
    if not contrast.startswith('SD('):
      strata.append(('flipped', u[u.flip_top1]))
    for stratum, g in strata:
      for name, _ in METRICS:
        d = g.dropna(subset=[f'd_{name}'])
        targets = sorted(d.target.unique())
        groups = [d[d.target == t][f'd_{name}'].values for t in targets]
        vals = d[f'd_{name}'].values.astype(float)
        out = {'source': source, 'family': u.family.iloc[0], 'contrast': contrast, 'stratum': stratum, 'metric': name,
               'n_units': len(d), 'n_targets': len(targets), 'units_without_metric': len(g) - len(d)}
        for stat, fn in (('median', contrasts.med), ('mean', contrasts.mean)):
          lo, hi = contrasts.boot_ci(groups, fn, np.random.default_rng(seed), replicates)
          out.update({stat: fn(vals), f'{stat}_ci_low': lo, f'{stat}_ci_high': hi})
        out['median_abs'] = contrasts.med(np.abs(vals))
        out['reading'] = reading(out['median_ci_low'], out['median_ci_high'],
                                 EQUIVALENCE_A if name == 'ligand_rmsd' else None)
        rows.append(out)
  return pd.DataFrame(rows)


def checks_table(units):
  rows = []
  for (source, contrast), u in units.groupby(['source', 'contrast'], sort=False):
    for name, _ in METRICS:
      for side in ('a', 'b'):
        rows.append({'source': source, 'contrast': contrast, 'metric': name, 'side': side, 'units': len(u),
                     'missing': int(u[f'{name}_{side}'].isna().sum())})
  return pd.DataFrame(rows)


def fmt(x, nd=2):
  return 'n/a' if x != x else f'{x:+.{nd}f}'


def report(q, args):
  L = ['# Returned structures across platforms (label returned)', '',
       'Post hoc, descriptive (drafts/section10_returned_spec_20261114.md). d = B minus A, A the first-named side; '
       f'95% percentile intervals, target bootstrap ({args.replicates} replicates, seed {args.seed}). Ligand RMSD in '
       f'angstrom (lower is closer to the experimental structure); band {EQUIVALENCE_A} A.', '',
       '| source | contrast | stratum | metric | units | targets | median d [95% CI] | mean d [95% CI] | median abs d | reading |',
       '|---|---|---|---|---|---|---|---|---|---|']
  for r in q.itertuples():
    nd = 3 if r.metric == 'iptm' else 2
    L.append(f'| {r.source} | {r.contrast} | {r.stratum} | {r.metric} | {r.n_units} | {r.n_targets} | '
             f'{fmt(r.median, nd)} [{fmt(r.median_ci_low, nd)}, {fmt(r.median_ci_high, nd)}] | '
             f'{fmt(r.mean, nd)} [{fmt(r.mean_ci_low, nd)}, {fmt(r.mean_ci_high, nd)}] | {fmt(r.median_abs, nd)} | '
             f'{r.reading} |')
  return '\n'.join(L) + '\n'


def sha256(path):
  h = hashlib.sha256()
  with open(path, 'rb') as f:
    for chunk in iter(lambda: f.read(1 << 20), b''):
      h.update(chunk)
  return h.hexdigest()


def run(sources, out, args, inputs=(), root=None, items=None):
  out = pathlib.Path(out)
  out.mkdir(parents=True, exist_ok=True)
  frames = []
  for source, todo in (items or CONTRASTS).items():
    if source not in sources:
      continue
    samples, pairs = sources[source]
    for contrast, family in todo:
      frames.append(unit_table(samples, pairs, contrast, family, source))
  units = pd.concat(frames, ignore_index=True)
  q = summarize(units, args.replicates, args.seed)
  units.to_csv(out / 'units.csv', index=False)
  q.to_csv(out / 'quality.csv', index=False)
  checks_table(units).to_csv(out / 'checks.csv', index=False)
  if root is not None:
    (out / 'inputs.tsv').write_text(''.join(f'{p}\t{sha256(pathlib.Path(root) / p)}\n' for p in inputs))
  (out / 'report.md').write_text(report(q, args))
  return units, q


def main():
  ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
  ap.add_argument('--main', default='.', help='the checkout that holds results/')
  ap.add_argument('--out', required=True)
  ap.add_argument('--replicates', type=int, default=10000)
  ap.add_argument('--seed', type=int, default=2026)
  args = ap.parse_args()
  out = pathlib.Path(args.out)
  if (out / 'quality.csv').exists():
    raise SystemExit(f'{out}/quality.csv exists; the output folder must be new')
  sources, inputs = load_sources(args.main)
  run(sources, out, args, inputs, args.main)
  print(f'wrote {out}')


if __name__ == '__main__':
  main()
