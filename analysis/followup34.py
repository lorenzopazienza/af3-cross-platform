"""Shared code of the follow-up 3 (A100 arms, analysis/a100.py) and follow-up 4 (CPU on every target,
analysis/cpu60.py) analyses, notes/analysis_plan.md section 10, rows of 2026-10-16.

Both pair conditions exactly as the main study does (analysis/contrasts.pair_rows: A the first-named side,
paired on (target, seed) and (target, seed, sample)), with the statistics of section 5 (contrasts.summarize) and
the target-clustered bootstrap. Data come from the main study's per-sample table (primary_v2/samples.csv) and the
follow-up's own (analysis/collect.py on its label); RC seed pairs of the main study's conditions are reused from
primary_v2/contrasts/pairs_seed.csv (HW minus RC needs only seed pairs), never recomputed.
"""
import itertools
import pathlib
import sys

import pandas as pd

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / 'analysis'))
import contrasts  # noqa: E402

KEY = ['condition', 'target', 'seed', 'repetition', 'sample']


def load_samples(path, labels):
  """Rows at AlphaFold3's defaults of the given labels, with condition = platform code and an integer sample."""
  d = pd.read_csv(path, low_memory=False)
  d = d[(d.num_recycles == 10) & (d.num_diffusion_samples == 5) & d.label.isin(labels)].copy()
  d['condition'] = d.platform
  return d


def sample_rows(d):
  ok = d[d['sample'].notna()].copy()
  ok['sample'] = ok['sample'].astype(int)
  return ok


def union(*frames):
  """Concatenates per-sample tables; a (condition, target, seed, repetition, sample) in two of them is an error."""
  ok = pd.concat([sample_rows(f) for f in frames], ignore_index=True)
  dup = ok.duplicated(KEY, keep=False)
  if dup.any():
    clash = ok[dup].groupby(['condition', 'target', 'seed', 'repetition']).session.unique()
    raise SystemExit('the same (condition, target, seed, repetition) in two sources:\n' + clash.head(10).to_string())
  return ok


def disjoint_units(a, b, condition):
  """Asserts that no (target, seed) of condition appears in both tables (C: main study and cpu60)."""
  ka = set(map(tuple, a[a.condition == condition][['target', 'seed']].drop_duplicates().values))
  kb = set(map(tuple, b[b.condition == condition][['target', 'seed']].drop_duplicates().values))
  both = sorted(ka & kb)
  if both:
    raise SystemExit(f'{condition}: (target, seed) in both sources: {both[:10]}')
  return len(ka), len(kb)


def one_machine_type(d, conditions):
  """The single machine type of these conditions' sessions; two or more is an error (a follow-up arm never mixes
  machine types)."""
  g = d[d.condition.isin(conditions)]
  types = sorted(g.machine_type.dropna().astype(str).unique())
  if len(types) != 1:
    raise SystemExit(f'{", ".join(conditions)}: sessions of {len(types)} machine types {types}; one is required')
  return types[0]


def build(ok, specs):
  """Seed and sample pairs of the given contrasts. specs: [(name, kind, (condition, rep), (condition, rep),
  primary)]; kind SD pairs the seeds of one (condition, rep) on each target, as contrasts.build_pairs does."""
  units = contrasts.seed_units(ok)
  seed_rows, sample_rows_, cons, counts = [], [], {}, []

  def keys(c, rep):
    return {(k[2], k[3]): k for k in units if k[0] == c and k[1] == rep}
  for name, kind, a, b, primary in specs:
    if kind == 'SD':
      r1 = keys(*a)
      by_target = {}
      for (t, s), k in r1.items():
        by_target.setdefault(t, []).append(s)
      ka, kb = {}, {}
      for t, seeds in by_target.items():
        for s1, s2 in itertools.combinations(sorted(seeds), 2):
          ka[(t, (s1, s2))], kb[(t, (s1, s2))] = r1[(t, s1)], r1[(t, s2)]
    else:
      ka, kb = keys(*a), keys(*b)
    common = sorted(set(ka) & set(kb))
    counts.append({'contrast': name, 'pairs_used': len(common), 'excluded_only_a': len(set(ka) - set(kb)),
                   'excluded_only_b': len(set(kb) - set(ka))})
    for ts in common:
      contrasts.pair_rows(name, kind, ka[ts], kb[ts], units, primary, cons, seed_rows, sample_rows_)
  return pd.DataFrame(seed_rows), pd.DataFrame(sample_rows_), cons, pd.DataFrame(counts)


def main_rc_seed_pairs(primary_dir, names):
  """The main study's RC seed pairs (pairs_seed.csv) of the named contrasts, and their contrasts entries."""
  p = pd.read_csv(pathlib.Path(primary_dir) / 'contrasts' / 'pairs_seed.csv')
  p = p[p.contrast.isin(names)].copy()
  missing = sorted(set(names) - set(p.contrast))
  if missing:
    raise SystemExit(f'not in the main study pairs: {missing}')
  return p, {n: ('RC', not n.startswith('RC(Gd')) for n in names}


def side_by_side(new, old, names, stats):
  """Rows statistic by statistic: the follow-up's estimate and interval next to the main study's."""
  rows = []
  for n in names:
    for st in stats:
      a = new[(new.contrast == n) & (new.statistic == st)]
      b = old[(old.contrast == n) & (old.statistic == st)]
      rows.append({'contrast': n, 'statistic': st,
                   'new_n_targets': int(a.n_targets.iloc[0]) if len(a) else None,
                   'new_estimate': a.estimate.iloc[0] if len(a) else None,
                   'new_ci_low': a.ci_low.iloc[0] if len(a) else None, 'new_ci_high': a.ci_high.iloc[0] if len(a) else None,
                   'main_n_targets': int(b.n_targets.iloc[0]) if len(b) else None,
                   'main_estimate': b.estimate.iloc[0] if len(b) else None,
                   'main_ci_low': b.ci_low.iloc[0] if len(b) else None, 'main_ci_high': b.ci_high.iloc[0] if len(b) else None})
  return pd.DataFrame(rows)


def exit_code_check(d, conditions, label):
  rows = []
  for c in conditions:
    g = d[d.condition == c].drop_duplicates(['session', 'target', 'seed'])
    bad = g[g.exit_code != 0]
    rows.append({'check': f'1 exit codes ({label})', 'condition': c,
                 'value': f'{len(g) - len(bad)}/{len(g)} processes exit 0',
                 'detail': 'pass' if bad.empty and len(g) else ('FAIL: no process' if not len(g) else
                           'FAIL: ' + '; '.join(f'{r.target} {r.repetition} seed {r.seed}: {r.failure}'
                                                for r in bad.itertuples())[:300])})
  return rows


def pct(x):
  return '-' if x is None or pd.isna(x) else f'{100 * x:.1f}%'


def num(x):
  return '-' if x is None or pd.isna(x) else f'{x:.3g}'


def contrast_lines(con):
  out = ['| contrast | role | statistic | n | targets | estimate | 95% CI |', '|---|---|---|---|---|---|---|']
  for r in con.itertuples():
    f = pct if r.statistic.startswith(('flip', 'bit_identical')) else num
    out.append(f'| {r.contrast} | {r.role} | {r.statistic} | {r.n} | {r.n_targets} | {f(r.estimate)} | '
               f'{f(r.ci_low)} to {f(r.ci_high)} |')
  return out
