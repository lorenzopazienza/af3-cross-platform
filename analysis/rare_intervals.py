"""Rare-event intervals, a sensitivity analysis of registered readings (section 10, row of
drafts/section10_rare_spec_20261114.md, label rare): post hoc, asked by the reviewers. The registered readings stay as
they are; this analysis only says whether they would change under an interval suited to few events. No run and no
cloud resource.

    analysis/.venv/bin/python analysis/rare_intervals.py --main . --out results/analysis/rare_<DATE> \\
        [--replicates 10000] [--seed 2026]

Every quantity here is a signed sum of pooled flip rates over the same targets: a single rate (contrasts.summarize),
HW minus RC (contrasts.hw_minus_rc) or D_F = [HWf - RCf] - [HW - RC] (fu13.d_f; an unmeasured RCf is a zero term, as
fu14.zero_like). A pooled rate is the sum of the statistic over the units of the drawn targets divided by their number.
Method: BCa (bias-corrected and accelerated) interval on the registered target bootstrap. The replicates are the
registered ones (np.random.default_rng(--seed).integers(0, T, size=(R, T)) over the sorted common targets, as every
registered interval drew them), so the percentile interval is recomputed first and must equal the registered one
(tolerance 1e-9; a row that does not is marked "not reproduced" and gets no sensitivity reading). Bias correction
z0 = Phi^-1(p), p = (#{theta* < theta} + 0.5 #{theta* = theta}) / R; acceleration a from the jackknife over targets
(leave one target out); 95% limits at the replicate percentiles Phi(z0 + (z0 + z)/(1 - a (z0 + z))), z = -1.96 and
+1.96, by np.percentile's linear interpolation as the registered intervals. Degenerate cases: if every replicate
equals the estimate (no event on any side), the BCa interval is [estimate, estimate] and the row says so; if p is 0
or 1, BCa is undefined and the row says so.
Exact bound beside it, single rates only: the Clopper-Pearson 95% upper limit (two-sided) on the fraction of targets
with at least one event, k of T. It bounds the per-prediction rate from above whatever the correlation within a target
(a target's rate is at most 1 when it has an event, 0 otherwise), so it is conservative.
Readings re-read with the BCa interval, by the registered rule of each family:
  beyond recompilation (HW minus RC): "beyond" if the lower limit is above 0, "not beyond" otherwise;
  agree (follow-up 14, float32 top-1 flip rate): "agree" if the upper limit is below 0.05;
  reduced (D_F): "reduced" if the upper limit is below 0;
  none (zero-event statements): no rule, the intervals are reported.
"changed" is yes when the BCa reading differs from the registered reading recomputed from the registered interval.
Families (all targets of each folder; the post-cutoff subsets are not re-read):
  F1 main primary pairs: HW(V5,V6), HW(V6,G), HW(G,C) minus RC(P) and RC(Q) (secondary), flip_success and flip_matters
     (primary_v2/contrasts/hw_minus_rc.csv);
  F2 follow-up 4: HW(G,C) on 60 targets minus RC(G), flip_success and flip_matters (cpu60_20261018/hw_minus_rc.csv);
  F3 follow-up 11: HW(V5,V6), HW(V6,G) minus RC(P) and RC(Q), flip_success and flip_matters (secondary readings;
     repl_20261106/hw_minus_rc.csv, targets "all ...");
  F4 follow-up 12: HWh(V6,G) minus RC(V6) and RC(G), all three flips, at seeds 1 and 2 (primary_v2 hw_minus_rc.csv)
     and at seeds 1 to 5 (prec5_20261107/hw_minus_rc.csv, targets "all ...", seeds "1 to 5"); HWh(G,C) minus RC(G) at
     seeds 1 and 2 (primary_v2);
  F5 follow-ups 13 and 14, D_F of the four primary float32 pairs, flip_success and flip_matters (fu14_20261113b/d_f.csv);
  F6 follow-up 14, agree: the ten float32 pairs, flip_top1 (fu14_20261113b/contrasts.csv);
  F7 zero-event statements: RC(V5), RC(V6) (primary_v2 contrasts.csv), RC(Gf), RC(V5f), RC(Ad), RC(Hd)
     (fu14_20261113b/contrasts.csv), all three flips.
Every row also gives the events on each term and whether it meets the prompt's criterion (fewer than 10 events on a
term, or a registered lower limit of 0.0 or 0.3 percentage points after rounding to one decimal); rows of the
families that do not meet it are reported too, marked.
Outputs in --out: intervals.csv, inputs.tsv, report.md.
"""
import argparse
import hashlib
import math
import pathlib
import statistics
import sys

import numpy as np
import pandas as pd

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / 'analysis'))

AN = 'results/analysis'
MAIN = f'{AN}/main_20261014/primary_v2'
DIRS = {'main': f'{MAIN}/contrasts', 'cpu60': f'{AN}/cpu60_20261018', 'repl': f'{AN}/repl_20261106',
        'prec5': f'{AN}/prec5_20261107', 'a100': f'{AN}/a100_20261018', 'h100': f'{AN}/h100_20261104',
        'fu14': f'{AN}/fu14_20261113b'}
FLIPS = ['flip_top1', 'flip_success', 'flip_matters']
M1 = ['flip_success', 'flip_matters']
TOL = 1e-9
Z = statistics.NormalDist()


# ---------------------------------------------------------------- the statistic and its intervals

def term_sums(sp, stat, targets):
  """Per target (in the given order) the sum of the statistic and the number of units."""
  g = sp.groupby('target')[stat]
  s = g.apply(lambda v: float(np.sum(v.astype(float)))).reindex(targets)
  n = g.size().reindex(targets)
  return s.values.astype(float), n.values.astype(float)


def combo(terms, stat):
  """terms: [(sign, seed-pair rows or None for a zero term)]. Returns (targets, [(sign, sums, counts)]) on the targets
  common to every measured term."""
  sets = [set(sp.target) for _, sp in terms if sp is not None]
  targets = sorted(set.intersection(*sets)) if sets else []
  parts = [(sign, *term_sums(sp, stat, targets)) for sign, sp in terms if sp is not None]
  return targets, parts


def value(parts, weights):
  """The signed sum of pooled rates for target multiplicities `weights` (shape (..., T))."""
  return sum(sign * (weights @ s) / (weights @ n) for sign, s, n in parts)


def replicates(n_targets, reps, seed):
  """The registered draws as a multiplicity matrix (reps, T)."""
  idx = np.random.default_rng(seed).integers(0, n_targets, size=(reps, n_targets))
  w = np.zeros((reps, n_targets))
  np.add.at(w, (np.repeat(np.arange(reps), n_targets), idx.ravel()), 1.0)
  return w


def jackknife(parts, n_targets):
  w = np.ones((n_targets, n_targets)) - np.eye(n_targets)
  return value(parts, w)


def acceleration(jack):
  d = jack.mean() - jack
  den = 6.0 * (np.sum(d ** 2) ** 1.5)
  return float(np.sum(d ** 3) / den) if den > 0 else 0.0


def bca(theta, boot, jack, level=0.95):
  """(low, high, note). boot: replicate values; jack: leave-one-target-out values."""
  boot = np.asarray(boot, float)
  if np.all(boot == theta):
    return theta, theta, 'degenerate: every replicate equals the estimate'
  p = (np.sum(boot < theta) + 0.5 * np.sum(boot == theta)) / len(boot)
  if p <= 0 or p >= 1:
    return np.nan, np.nan, 'undefined: no replicate on one side of the estimate'
  z0 = Z.inv_cdf(p)
  a = acceleration(np.asarray(jack, float))
  out = []
  for q in ((1 - level) / 2, (1 + level) / 2):
    zq = Z.inv_cdf(q)
    out.append(100 * Z.cdf(z0 + (z0 + zq) / (1 - a * (z0 + zq))))
  lo, hi = np.percentile(boot, out)
  return float(lo), float(hi), f'z0 {z0:.4f}, a {a:.4f}, percentiles {out[0]:.2f} and {out[1]:.2f}'


def binom_cdf(k, n, p):
  return sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k + 1))


def cp_upper(k, n, level=0.95):
  """Two-sided Clopper-Pearson upper limit for k events in n trials."""
  if n == 0:
    return np.nan
  if k >= n:
    return 1.0
  alpha = (1 - level) / 2
  lo, hi = k / n, 1.0
  for _ in range(200):
    mid = (lo + hi) / 2
    if binom_cdf(k, n, mid) > alpha:
      lo = mid
    else:
      hi = mid
  return (lo + hi) / 2


def evaluate(terms, stat, reps, seed):
  targets, parts = combo(terms, stat)
  T = len(targets)
  if T == 0:
    return {'n_targets': 0}
  est = float(value(parts, np.ones(T)))
  boot = value(parts, replicates(T, reps, seed))
  lo, hi = np.percentile(boot, [2.5, 97.5])
  blo, bhi, note = bca(est, boot, jackknife(parts, T))
  r = {'n_targets': T, 'estimate': est, 'pct_low': float(lo), 'pct_high': float(hi), 'bca_low': blo, 'bca_high': bhi,
       'bca_note': note}
  for i, (sign, s, n) in enumerate(parts):
    r[f'events_{i + 1}'] = int(round(s.sum()))
    r[f'units_{i + 1}'] = int(n.sum())
  if len(parts) == 1:
    r['targets_with_event'] = int(np.sum(parts[0][1] > 0))
    r['cp_upper_targets'] = cp_upper(r['targets_with_event'], T)
  return r


# ---------------------------------------------------------------- readings

def read(rule, lo, hi):
  if lo != lo or hi != hi:
    return 'not computed'
  if rule == 'beyond':
    return 'beyond' if lo > 0 else 'not beyond'
  if rule == 'agree':
    return 'agree' if hi < 0.05 else 'agreement not shown'
  if rule == 'reduced':
    return 'reduced' if hi < 0 else 'not reduced'
  return ''


def criterion(r):
  ev = [v for k, v in r.items() if k.startswith('events_')]
  low = r.get('registered_low')
  few = any(e < 10 for e in ev)
  at = low == low and low is not None and round(100 * low, 1) in (0.0, 0.3)
  return 'yes' if few or at else 'no'


# ---------------------------------------------------------------- the families

def pick(df, **kw):
  m = np.ones(len(df), bool)
  for k, v in kw.items():
    m &= (df[k] == v).values if not callable(v) else df[k].map(v).values
  out = df[m]
  if len(out) != 1:
    raise SystemExit(f'registered row {kw}: {len(out)} matches, one required')
  return out.iloc[0]


def rows_of(name, sp):
  out = sp[sp.contrast == name]
  if out.empty:
    raise SystemExit(f'{name}: no seed pairs')
  return out


def specs(t):
  """[(family, quantity, statistic, rule, terms, registered row)] from the tables t (see load)."""
  S = []
  main_hw, main_con, main_sp = t['main_hw'], t['main_con'], t['main_sp']

  def hw_rows(family, hwsrc, rcsrc, table, name, rcs, stats, **sel):
    for rc in rcs:
      for st in stats:
        reg = pick(table, hardware=name, recompilation=rc, statistic=st, **sel)
        S.append((family, f'{name} minus {rc}', st, 'beyond', [(1, rows_of(name, hwsrc)), (-1, rows_of(rc, rcsrc))],
                  reg))
  for name, rcs in (('HW(V5,V6)', ['RC(V5)', 'RC(V6)']), ('HW(V6,G)', ['RC(V6)', 'RC(G)']), ('HW(G,C)', ['RC(G)'])):
    hw_rows('F1 main primary pairs', main_sp, main_sp, main_hw, name, rcs, M1)
  hw_rows('F2 follow-up 4 (60 targets)', t['cpu60_sp'], main_sp, t['cpu60_hw'], 'HW(G,C)', ['RC(G)'], M1)
  all_t = lambda v: str(v).startswith('all')  # noqa: E731
  for name, rcs in (('HW(V5,V6)', ['RC(V5)', 'RC(V6)']), ('HW(V6,G)', ['RC(V6)', 'RC(G)'])):
    hw_rows('F3 follow-up 11', t['repl_sp'], t['repl_sp'], t['repl_hw'], name, rcs, M1, targets=all_t)
  hw_rows('F4 follow-up 12, seeds 1 and 2', main_sp, main_sp, main_hw, 'HWh(V6,G)', ['RC(V6)', 'RC(G)'], FLIPS)
  hw_rows('F4 follow-up 12, seeds 1 and 2', main_sp, main_sp, main_hw, 'HWh(G,C)', ['RC(G)'], FLIPS)
  hw_rows('F4 follow-up 12, seeds 1 to 5', t['prec5_sp'], main_sp, t['prec5_hw'], 'HWh(V6,G)', ['RC(V6)', 'RC(G)'],
          FLIPS, targets=all_t, seeds='1 to 5')
  import fu14  # noqa: E402
  ref = {'main': main_sp, 'a100': t['a100_sp'], 'h100': t['h100_sp']}
  for pair in fu14.PRIMARY:
    name = fu14.hw(*pair)
    hw_name, hw_src = fu14.BF16[pair]
    (rc_name, rc_src), rcf = fu14.RC[pair]
    for st in M1:
      reg = pick(t['fu14_df'], pair=name, statistic=st)
      terms = [(1, rows_of(name, t['fu14_sp'])), (-1, rows_of(rcf, t['fu14_sp']) if rcf else None),
               (-1, rows_of(hw_name, ref[hw_src])), (1, rows_of(rc_name, ref[rc_src]))]
      S.append(('F5 follow-ups 13 and 14, D_F', f'D_F {name} (RCf {rcf or "set to 0"}; {hw_name} minus {rc_name})', st,
                'reduced', terms, reg))
  for pair in fu14.PAIRS:
    name = fu14.hw(*pair)
    S.append(('F6 follow-up 14, agree', name, 'flip_top1', 'agree', [(1, rows_of(name, t['fu14_sp']))],
              pick(t['fu14_con'], contrast=name, statistic='flip_top1')))
  for name, sp, con in (('RC(V5)', main_sp, main_con), ('RC(V6)', main_sp, main_con)) + tuple(
      (n, t['fu14_sp'], t['fu14_con']) for n in ('RC(Gf)', 'RC(V5f)', 'RC(Ad)', 'RC(Hd)')):
    for st in FLIPS:
      S.append(('F7 zero-event statements', name, st, 'none', [(1, rows_of(name, sp))],
                pick(con, contrast=name, statistic=st)))
  return S


def load(root):
  root = pathlib.Path(root)
  p = {k: root / v for k, v in DIRS.items()}
  files = {'main_hw': p['main'] / 'hw_minus_rc.csv', 'main_con': p['main'] / 'contrasts.csv',
           'main_sp': p['main'] / 'pairs_seed.csv', 'cpu60_sp': p['cpu60'] / 'pairs_seed.csv',
           'cpu60_hw': p['cpu60'] / 'hw_minus_rc.csv', 'repl_sp': p['repl'] / 'pairs_seed.csv',
           'repl_hw': p['repl'] / 'hw_minus_rc.csv', 'prec5_sp': p['prec5'] / 'pairs_seed.csv',
           'prec5_hw': p['prec5'] / 'hw_minus_rc.csv', 'a100_sp': p['a100'] / 'pairs_seed.csv',
           'h100_sp': p['h100'] / 'pairs_seed.csv', 'fu14_sp': p['fu14'] / 'pairs_seed.csv',
           'fu14_df': p['fu14'] / 'd_f.csv', 'fu14_con': p['fu14'] / 'contrasts.csv'}
  return {k: pd.read_csv(f) for k, f in files.items()}, [str(f.relative_to(root)) for f in files.values()]


def run(tables, out, args, inputs=(), root=None):
  out = pathlib.Path(out)
  out.mkdir(parents=True, exist_ok=True)
  rows = []
  for family, quantity, stat, rule, terms, reg in specs(tables):
    r = {'family': family, 'quantity': quantity, 'statistic': stat, 'rule': rule}
    r.update(evaluate(terms, stat, args.replicates, args.seed))
    r.update({'registered_estimate': float(reg.estimate), 'registered_low': float(reg.ci_low),
              'registered_high': float(reg.ci_high)})
    same = all(abs(r.get(a, np.nan) - b) <= TOL for a, b in (('estimate', r['registered_estimate']),
                                                             ('pct_low', r['registered_low']),
                                                             ('pct_high', r['registered_high'])))
    r['reproduced'] = 'yes' if same else 'NOT REPRODUCED'
    r['registered_reading'] = read(rule, r['registered_low'], r['registered_high'])
    r['bca_reading'] = read(rule, r['bca_low'], r['bca_high']) if same else 'not read'
    r['changed'] = '' if rule == 'none' or not same else ('yes' if r['bca_reading'] != r['registered_reading'] else 'no')
    r['criterion'] = criterion(r)
    rows.append(r)
  df = pd.DataFrame(rows)
  df.to_csv(out / 'intervals.csv', index=False)
  if root is not None:
    (out / 'inputs.tsv').write_text(''.join(f'{p}\t{sha256(pathlib.Path(root) / p)}\n' for p in inputs))
  (out / 'report.md').write_text(report(df, args))
  bad = int((df.reproduced != 'yes').sum())
  if bad:
    print(f'!! {bad} rows do not reproduce their registered interval; they are not re-read (intervals.csv)')
  return df


def pts(x):
  return 'n/a' if x != x else f'{100 * x:+.1f}'


def report(df, args):
  L = ['# Rare-event intervals (label rare)', '',
       'Post hoc sensitivity analysis (drafts/section10_rare_spec_20261114.md). The registered readings stay; BCa on '
       f'the registered target bootstrap ({args.replicates} replicates, seed {args.seed}); percentage points. '
       'CP: Clopper-Pearson upper limit on the fraction of targets with an event (single rates).', '',
       '| family | quantity | statistic | targets | events | estimate | registered (percentile) | BCa | CP upper | '
       'registered reading | BCa reading | changed | criterion |', '|---|---|---|---|---|---|---|---|---|---|---|---|---|']
  for r in df.to_dict('records'):
    ev = ' / '.join(str(r[k]) for k in sorted(r) if k.startswith('events_') and r[k] == r[k])
    v = r.get('cp_upper_targets', np.nan)
    cp = pts(v) if v == v else ''
    L.append(f'| {r["family"]} | {r["quantity"]} | {r["statistic"]} | {r["n_targets"]} | {ev} | {pts(r["estimate"])} | '
             f'[{pts(r["registered_low"])}, {pts(r["registered_high"])}] | [{pts(r["bca_low"])}, {pts(r["bca_high"])}] | '
             f'{cp} | {r["registered_reading"]} | {r["bca_reading"]} | {r["changed"]} | {r["criterion"]} |')
  ch = df[df.changed == 'yes']
  L += ['', f'Readings that would change under BCa: {len(ch)} of {int((df.rule != "none").sum())}.']
  L += [f'- {r.family}: {r.quantity}, {r.statistic}: {r.registered_reading} -> {r.bca_reading}' for r in ch.itertuples()]
  return '\n'.join(L) + '\n'


def sha256(path):
  h = hashlib.sha256()
  with open(path, 'rb') as f:
    for chunk in iter(lambda: f.read(1 << 20), b''):
      h.update(chunk)
  return h.hexdigest()


def main():
  ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
  ap.add_argument('--main', default='.', help='the checkout that holds results/')
  ap.add_argument('--out', required=True)
  ap.add_argument('--replicates', type=int, default=10000)
  ap.add_argument('--seed', type=int, default=2026)
  args = ap.parse_args()
  out = pathlib.Path(args.out)
  if (out / 'intervals.csv').exists():
    raise SystemExit(f'{out}/intervals.csv exists; the output folder must be new')
  tables, inputs = load(args.main)
  run(tables, out, args, inputs, args.main)
  print(f'wrote {out}')


if __name__ == '__main__':
  main()
