"""Success of the best of all samples per platform (section 10, row of drafts/section10_best25_spec_20261116.md, label
best25): a post hoc analysis of frozen outputs, descriptive whatever it shows. No run and no cloud resource.

    analysis/.venv/bin/python analysis/best25.py --main . --out results/analysis/best25_<DATE> \\
        [--replicates 10000] [--seed 2026]

Question: do the differences between platforms change what a user finds by looking at all samples, or only which
sample is returned?
Inputs, relative to --main (the per-sample tables only; no structure file is read):
  results/analysis/main_20261014/primary_v2/samples.csv and contrasts/pairs_seed.csv  (V5, V6, G r1 and r2; C r1 of
                                                                                      6 targets)
  results/analysis/cpu60_20261018/samples.csv and pairs_seed.csv                      (C r1 of the other 54 targets)
  results/analysis/a100_20261018/samples.csv; results/analysis/h100_20261104/samples.csv and pairs_seed.csv (A, H)
  results/analysis/main_20261014/primary_v2/contrasts/q3.csv, results/analysis/a100_20261018/q3.csv (check 3 only)
Success is the registered M1 success of a sample (m1_success; a sample without it, as the three glycan targets, is no
success, as registered). Measures per side (condition, repetition) and unit:
  level target, over the seeds S of the contrast (1 to 5; 1 and 2 when the CPU is a side):
    oracle  at least one of the samples of the seeds S succeeds (25 samples; 10 with S = 1, 2);
    main    AlphaFold 3's main output succeeds: the sample with the highest full-precision ranking score over S, exact
            ties to the lowest (seed, sample) (winners.main_output);
  level seed, each (target, seed) of S:
    oracle  at least one of the seed's 5 samples succeeds;
    top-1   the seed's top-ranked sample succeeds (contrasts.top; the registered per-seed top-1).
Rates (rates.csv): per side and seed set, the share of units with each measure, 95% percentile interval from the
  bootstrap over targets (contrasts.boot_ci, --replicates, a fresh generator from --seed for every statistic).
Contrasts (differences.csv), B minus A, A the first-named side, paired on (target) or (target, seed):
  platform: HW(V5,V6), HW(V6,G), HW(H,A) (seeds 1 to 5), HW(G,C) (seeds 1 and 2, 60 targets: follow-up 4's pairing);
  recompilation, the reference: RC(V5), RC(V6), RC(G), RC(H), RC(A) (seeds 1 to 5) and RC(G) on seeds 1 and 2 (the
  reference of HW(G,C)).
  Per contrast, level and measure: the difference of the rates with its interval; the units where exactly one side
  succeeds (A only, B only) and their share with its interval.
Descriptive: no reading rule; the oracle rows are read beside the main-output (target) and top-1 (seed) rows of the
same contrast, and the platform contrasts beside recompilation. No correction for multiple comparisons.
Checks: 1 (stops), every (side, target, seed) of a contrast has exactly 5 samples with a ranking score, and both sides
have the same targets and every seed of S; 2 (stops), the top-1 success of every registered seed unit
(pairs_seed.csv of the main study, follow-up 4 and the H100 analysis) equals the recomputed one; 3 (reported), the
main-output success rate on seeds 1 to 5 equals the registered Q3 rate with 5 seeds (q3.csv) for V5, V6, G and A.
Outputs in --out: units.csv, rates.csv, differences.csv, checks.csv, inputs.tsv, report.md.
"""
import argparse
import pathlib
import sys

import numpy as np
import pandas as pd

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / 'analysis'))
sys.path.insert(0, str(REPO / 'harness'))
import contrasts  # noqa: E402
import followup34 as f34  # noqa: E402
import winners as w  # noqa: E402

AN = 'results/analysis'
MAIN = f'{AN}/main_20261014/primary_v2'
CPU60 = f'{AN}/cpu60_20261018'
A100 = f'{AN}/a100_20261018'
H100 = f'{AN}/h100_20261104'
S5, S2 = (1, 2, 3, 4, 5), (1, 2)
# name, family, side A, side B, seeds, registered pairs source (check 2) and its contrast name
CONTRASTS = [
    ('HW(V5,V6)', 'platform', ('V5', 'r1'), ('V6', 'r1'), S5, 'main', 'HW(V5,V6)'),
    ('HW(V6,G)', 'platform', ('V6', 'r1'), ('G', 'r1'), S5, 'main', 'HW(V6,G)'),
    ('HW(H,A)', 'platform', ('H', 'r1'), ('A', 'r1'), S5, 'h100', 'HW(H,A)'),
    ('HW(G,C)', 'platform', ('G', 'r1'), ('C', 'r1'), S2, 'cpu60', 'HW(G,C)'),
    ('RC(V5)', 'recompilation', ('V5', 'r1'), ('V5', 'r2'), S5, 'main', 'RC(V5)'),
    ('RC(V6)', 'recompilation', ('V6', 'r1'), ('V6', 'r2'), S5, 'main', 'RC(V6)'),
    ('RC(G)', 'recompilation', ('G', 'r1'), ('G', 'r2'), S5, 'main', 'RC(G)'),
    ('RC(H)', 'recompilation', ('H', 'r1'), ('H', 'r2'), S5, 'h100', 'RC(H)'),
    ('RC(A)', 'recompilation', ('A', 'r1'), ('A', 'r2'), S5, 'h100', 'RC(A)'),
    ('RC(G) seeds 1-2', 'recompilation', ('G', 'r1'), ('G', 'r2'), S2, 'main', 'RC(G)'),
]
MEASURES = {'target': ['oracle', 'main'], 'seed': ['oracle', 'top1']}
Q3_CONDITIONS = {'main': ['V5', 'V6', 'G'], 'a100': ['A']}


def load_sources(root):
  root = pathlib.Path(root)
  main = f34.load_samples(root / MAIN / 'samples.csv', ['main'])
  c60 = f34.load_samples(root / CPU60 / 'samples.csv', ['cpu60'])
  a = f34.load_samples(root / A100 / 'samples.csv', ['a100'])
  h = f34.load_samples(root / H100 / 'samples.csv', ['h100'])
  keep = main.condition.isin(['V5', 'V6', 'G', 'C']) & main.repetition.isin(['r1', 'r2'])
  samples = f34.union(main[keep], c60[c60.condition == 'C'], a[(a.condition == 'A') & a.repetition.isin(['r1', 'r2'])],
                      h[(h.condition == 'H') & h.repetition.isin(['r1', 'r2'])])
  pairs = {'main': pd.read_csv(root / MAIN / 'contrasts' / 'pairs_seed.csv'),
           'cpu60': pd.read_csv(root / CPU60 / 'pairs_seed.csv'), 'h100': pd.read_csv(root / H100 / 'pairs_seed.csv')}
  q3 = {'main': pd.read_csv(root / MAIN / 'contrasts' / 'q3.csv'), 'a100': pd.read_csv(root / A100 / 'q3.csv')}
  inputs = [MAIN + '/samples.csv', MAIN + '/contrasts/pairs_seed.csv', MAIN + '/contrasts/q3.csv', CPU60 + '/samples.csv',
            CPU60 + '/pairs_seed.csv', A100 + '/samples.csv', A100 + '/q3.csv', H100 + '/samples.csv',
            H100 + '/pairs_seed.csv']
  return samples, pairs, q3, inputs


def success(x):
  return (x == True).astype(float)  # noqa: E712 - NaN or a missing value is no success


def side_units(samples, side, seeds, contrast):
  """Per-target and per-seed measures of one side on the seeds S; check 1 raises SystemExit."""
  c, rep = side
  g = samples[(samples.condition == c) & (samples.repetition == rep) & samples.seed.isin(seeds)]
  if g.empty:
    raise SystemExit(f'check 1: {contrast}: no samples of {c} {rep}')
  n = g.groupby(['target', 'seed']).agg(k=('sample', 'nunique'), rows=('sample', 'size'),
                                        scored=('ranking_score', 'count'))
  bad = n[(n.k != 5) | (n.rows != 5) | (n.scored != 5)]
  if len(bad):
    raise SystemExit(f'check 1: {contrast} {c} {rep}: (target, seed) without exactly 5 scored samples: '
                     f'{list(bad.index[:5])}')
  seeds_of = g.groupby('target').seed.apply(lambda s: tuple(sorted(s.unique())))
  if (seeds_of != tuple(seeds)).any():
    raise SystemExit(f'check 1: {contrast} {c} {rep}: targets without every seed of {seeds}: '
                     f'{list(seeds_of[seeds_of != tuple(seeds)].index[:5])}')
  seed_rows, target_rows = [], []
  for (t, s), u in g.groupby(['target', 'seed']):
    top, succ, _ = contrasts.top(u.set_index('sample').sort_index())
    seed_rows.append({'target': t, 'seed': int(s), 'oracle': float(success(u.m1_success).max()), 'top1': float(succ),
                      'top1_sample': int(top)})
  for t, u in g.groupby('target'):
    m = w.main_output(u)
    target_rows.append({'target': t, 'oracle': float(success(u.m1_success).max()),
                        'main': float(m.m1_success == True)})  # noqa: E712
  return pd.DataFrame(seed_rows), pd.DataFrame(target_rows)


def contrast_units(samples, spec):
  name, family, a, b, seeds, _, _ = spec
  sa, ta = side_units(samples, a, seeds, name)
  sb, tb = side_units(samples, b, seeds, name)
  if set(ta.target) != set(tb.target):
    raise SystemExit(f'check 1: {name}: the two sides have different targets')
  rows = []
  for level, (xa, xb), key in (('seed', (sa, sb), ['target', 'seed']), ('target', (ta, tb), ['target'])):
    m = xa.merge(xb, on=key, suffixes=('_a', '_b'), validate='one_to_one')
    for r in m.to_dict('records'):
      for meas in MEASURES[level]:
        rows.append({'contrast': name, 'family': family, 'seeds': '-'.join(map(str, seeds)), 'level': level,
                     'measure': meas, 'target': r['target'], 'seed': r.get('seed', np.nan),
                     'side_a': '_'.join(a), 'side_b': '_'.join(b), 'a': r[f'{meas}_a'], 'b': r[f'{meas}_b'],
                     **({'top1_sample_a': r['top1_sample_a'], 'top1_sample_b': r['top1_sample_b']}
                        if level == 'seed' else {})})
  return pd.DataFrame(rows)


def check_registered_top1(units, pairs, specs):
  """Check 2: the recomputed per-seed top-1 and its success equal the registered pairs_seed.csv."""
  n = 0
  for spec in specs:
    name, _, _, _, seeds, src, reg = spec
    p = pairs[src]
    p = p[(p.contrast == reg) & p.seed_a.isin(seeds)]
    u = units[(units.contrast == name) & (units.level == 'seed') & (units.measure == 'top1')]
    m = u.merge(p, left_on=['target', 'seed'], right_on=['target', 'seed_a'], how='outer', indicator=True)
    if (m._merge != 'both').any():
      raise SystemExit(f'check 2: {name}: seed units differ from the registered pairs ({src} {reg})')
    bad = m[(m.top1_sample_a != m.top1_a) | (m.top1_sample_b != m.top1_b) | (m.a != m.success_a.astype(float))
            | (m.b != m.success_b.astype(float))]
    if len(bad):
      raise SystemExit(f'check 2: {name}: top-1 or its success differs from the registered pairs on '
                       f'{list(zip(bad.target, bad.seed))[:5]}')
    n += len(m)
  return n


def rates(units, reps, seed):
  rows, seen = [], set()
  for (name, level, meas), u in units.groupby(['contrast', 'level', 'measure'], sort=False):
    for side in ('a', 'b'):
      key = (u[f'side_{side}'].iloc[0], u.seeds.iloc[0], level, meas)
      if key in seen:
        continue
      seen.add(key)
      groups = [g[side].values for _, g in u.groupby('target')]
      lo, hi = contrasts.boot_ci(groups, contrasts.mean, np.random.default_rng(seed), reps)
      rows.append({'side': key[0], 'seeds': key[1], 'level': level, 'measure': meas, 'units': len(u),
                   'n_targets': u.target.nunique(), 'successes': int(u[side].sum()), 'rate': contrasts.mean(u[side].values),
                   'ci_low': lo, 'ci_high': hi})
  return pd.DataFrame(rows).sort_values(['level', 'measure', 'seeds', 'side'], kind='mergesort')


def differences(units, reps, seed):
  rows = []
  for (name, level, meas), u in units.groupby(['contrast', 'level', 'measure'], sort=False):
    u = u.assign(d=u.b - u.a, disc=(u.b != u.a).astype(float))
    out = {'contrast': name, 'family': u.family.iloc[0], 'seeds': u.seeds.iloc[0], 'level': level, 'measure': meas,
           'units': len(u), 'n_targets': u.target.nunique(), 'rate_a': u.a.mean(), 'rate_b': u.b.mean()}
    for stat, col in (('difference', 'd'), ('discordant', 'disc')):
      groups = [g[col].values for _, g in u.groupby('target')]
      lo, hi = contrasts.boot_ci(groups, contrasts.mean, np.random.default_rng(seed), reps)
      out.update({stat: contrasts.mean(u[col].values), f'{stat}_ci_low': lo, f'{stat}_ci_high': hi})
    out.update({'a_only': int(((u.a == 1) & (u.b == 0)).sum()), 'b_only': int(((u.a == 0) & (u.b == 1)).sum()),
                'targets_discordant': int(u[u.disc == 1].target.nunique())})
    rows.append(out)
  return pd.DataFrame(rows)


def q3_check(units, q3):
  rows = []
  for src, conds in Q3_CONDITIONS.items():
    q = q3[src]
    for c in conds:
      reg = q[(q.n_seeds == 5) & (q.condition == c) & (q.statistic == 'success_rate')]
      u = units[(units.level == 'target') & (units.measure == 'main') & (units.seeds == '1-2-3-4-5')]
      vals = pd.concat([u[u.side_a == f'{c}_r1'][['target', 'a']].rename(columns={'a': 'v'}),
                        u[u.side_b == f'{c}_r1'][['target', 'b']].rename(columns={'b': 'v'})]).drop_duplicates('target')
      mine = vals.v.mean() if len(vals) else np.nan
      regv = float(reg.estimate.iloc[0]) if len(reg) else np.nan
      rows.append({'check': '3 main output = registered Q3 (5 seeds)', 'side': f'{c}_r1',
                   'value': 'equal' if len(reg) and abs(mine - regv) < 1e-12 else
                   ('no registered row' if not len(reg) else 'DIFFERS')})
  return rows


def fmt_pct(x):
  return 'n/a' if x != x else f'{100 * x:.1f}'


def report(r, d, chk, args):
  L = ['# Success of the best of all samples (label best25)', '',
       'Post hoc, descriptive (drafts/section10_best25_spec_20261116.md). Oracle: at least one sample succeeds (target: '
       'all samples of the seeds; seed: the 5 samples of the seed). Main: AlphaFold 3\'s main output (top-ranked over '
       'the seeds). Top-1: the seed\'s top-ranked sample. Percentages with 95% percentile intervals over targets '
       f'({args.replicates} replicates, seed {args.seed}).', '', '## Rates', '',
       '| side | seeds | level | measure | units | successes | rate % [CI] |', '|---|---|---|---|---|---|---|']
  for x in r.to_dict('records'):
    L.append(f'| {x["side"]} | {x["seeds"]} | {x["level"]} | {x["measure"]} | {x["units"]} | {x["successes"]} | '
             f'{fmt_pct(x["rate"])} [{fmt_pct(x["ci_low"])}, {fmt_pct(x["ci_high"])}] |')
  L += ['', '## Differences, B minus A', '', '| contrast | seeds | level | measure | units | A % | B % | '
        'B minus A, points [CI] | A only | B only | discordant % [CI] |', '|---|---|---|---|---|---|---|---|---|---|---|']
  for x in d.to_dict('records'):
    L.append(f'| {x["contrast"]} | {x["seeds"]} | {x["level"]} | {x["measure"]} | {x["units"]} | {fmt_pct(x["rate_a"])} '
             f'| {fmt_pct(x["rate_b"])} | {100 * x["difference"]:+.1f} [{100 * x["difference_ci_low"]:+.1f}, '
             f'{100 * x["difference_ci_high"]:+.1f}] | {x["a_only"]} | {x["b_only"]} | {fmt_pct(x["discordant"])} '
             f'[{fmt_pct(x["discordant_ci_low"])}, {fmt_pct(x["discordant_ci_high"])}] |')
  L += ['', '## Checks', ''] + [f'- {c["check"]}: {c.get("side", "")} {c["value"]}' for c in chk]
  return '\n'.join(L) + '\n'


def run(samples, pairs, q3, out, args, inputs=(), root=None, specs=None):
  out = pathlib.Path(out)
  out.mkdir(parents=True, exist_ok=True)
  specs = specs or CONTRASTS
  units = pd.concat([contrast_units(samples, s) for s in specs], ignore_index=True)
  n = check_registered_top1(units, pairs, specs)
  chk = [{'check': '1 samples complete', 'value': 'pass'},
         {'check': '2 per-seed top-1 and success equal the registered pairs', 'value': f'pass ({n} seed units)'}]
  chk += q3_check(units, q3)
  r = rates(units, args.replicates, args.seed)
  d = differences(units, args.replicates, args.seed)
  units.to_csv(out / 'units.csv', index=False)
  r.to_csv(out / 'rates.csv', index=False)
  d.to_csv(out / 'differences.csv', index=False)
  pd.DataFrame(chk).to_csv(out / 'checks.csv', index=False)
  if root is not None:
    (out / 'inputs.tsv').write_text(''.join(f'{p}\t{w.sha256(pathlib.Path(root) / p)}\n' for p in inputs))
  (out / 'report.md').write_text(report(r, d, chk, args))
  return units, r, d, chk


def main():
  ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
  ap.add_argument('--main', default='.', help='the checkout that holds results/')
  ap.add_argument('--out', required=True)
  ap.add_argument('--replicates', type=int, default=10000)
  ap.add_argument('--seed', type=int, default=2026)
  args = ap.parse_args()
  out = pathlib.Path(args.out)
  if (out / 'differences.csv').exists():
    raise SystemExit(f'{out}/differences.csv exists; the output folder must be new')
  samples, pairs, q3, inputs = load_sources(args.main)
  run(samples, pairs, q3, out, args, inputs, args.main)
  print(f'wrote {out}')


if __name__ == '__main__':
  main()
