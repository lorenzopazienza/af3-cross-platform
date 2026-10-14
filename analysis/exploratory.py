"""Exploratory analysis of why top-1 flips occur (notes/analysis_plan.md section 10, row of 2026-10-14).

    analysis/.venv/bin/python analysis/exploratory.py --main results/analysis/main_20261014 \\
        [--out results/analysis/main_20261014/exploratory] [--replicates 10000] [--seed 2026]

Exploratory, labelled so in the paper; no primary result changes. Data: the main-study r1 outputs of the primary
analysis after the M4 correction (<main>/primary_v2: contrasts/pairs_seed.csv, contrasts/pairs_sample.csv and
samples.csv), for the contrasts in CONTRASTS. In every contrast A is the first-named condition.
  (a) per-seed top-1 flip rate against A's M3 margin (top-1 minus top-2 ranking_score within the seed): by margin
      quintile (edges from all of the contrast's seed pairs), and a logistic regression of flip on log10 margin
      (slope per tenfold margin, as an odds ratio);
  (b) concentration over targets: how many targets have 0, 1, ... seeds with a flip, and the share of all flips
      that come from the 10% of targets with most flips (ceil(0.1 x targets), ties broken by target ID); next to
      it, the same share expected if every seed pair flipped independently at the contrast's overall rate
      (binomial, simulated);
  (c) perturbation size: the median absolute ranking_score difference of the same (seed, sample) between A and B
      (M5, delta), against the median M3 margin of A (mu); the flip rate predicted from these two quantities by a
      two-sample model (each sample's score moves by independent normal noise with median absolute value delta;
      the margin is exponential with median mu; the top-1 flips when the top-2 overtakes it):
      p = 1/2 - exp(l^2 s^2 / 2) Q(l s), l = ln 2 / mu, s = sqrt(2) delta / 0.6745, Q the normal upper tail;
      against the observed rate. A model, not a fit: it has no free parameter;
  (d) the per-seed flip rate by token bucket (AlphaFold3's padded size, from the frozen manifest);
  (e) for every flip that matters (top-1 changes and the two chosen structures differ in M1 success): the ligand
      RMSD to the crystal, PoseBusters validity and success of the two top-1 samples, and which criterion separates
      them (RMSD on either side of 2 A, or PoseBusters validity).
Intervals: 95% percentile intervals from a paired bootstrap over targets (resample targets with replacement,
--replicates, --seed), as in section 5.
Outputs in --out: a_margin_quintiles.csv, a_logistic.csv, b_seeds_with_flip.csv, b_top_share.csv,
c_perturbation.csv, d_buckets.csv, e_flips_that_matter.csv, e_summary.csv, report.md.
"""

import argparse
import math
import pathlib
import sys

import numpy as np
import pandas as pd

REPO = pathlib.Path(__file__).resolve().parent.parent
CONTRASTS = ['HW(V5,V6)', 'HW(V6,G)', 'HW(V6,GT)', 'RC(G)', 'DU', 'DET', 'PR(V6)', 'PR(G)', 'HWh(V6,G)']
# (condition, repetition) of A and B, as analysis/contrasts.build_pairs pairs them; checked against the pairs'
# top-1 and success columns by check_sides().
SIDES = {'HW(V5,V6)': (('V5', 'r1'), ('V6', 'r1')), 'HW(V6,G)': (('V6', 'r1'), ('G', 'r1')),
         'HW(V6,GT)': (('V6', 'r1'), ('GT', 'r1')), 'RC(G)': (('G', 'r1'), ('G', 'r2')),
         'DU': (('G', 'r1'), ('GT', 'r1')), 'DET': (('G', 'r1'), ('Gd', 'r1')),
         'PR(V6)': (('V6', 'r1'), ('V6h', 'r1')), 'PR(G)': (('G', 'r1'), ('Gh', 'r1')),
         'HWh(V6,G)': (('V6h', 'r1'), ('Gh', 'r1'))}
QUINTILES = 5
TOP_FRACTION = 0.10
MEDIAN_ABS_NORMAL = 0.6744897501960817   # median of |Z|, Z standard normal


def load(main):
  c = main / 'primary_v2' / 'contrasts'
  seed = pd.read_csv(c / 'pairs_seed.csv')
  sample = pd.read_csv(c / 'pairs_sample.csv')
  samples = pd.read_csv(main / 'primary_v2' / 'samples.csv')
  return seed[seed.contrast.isin(CONTRASTS)].copy(), sample[sample.contrast.isin(CONTRASTS)].copy(), samples


def target_index(df):
  targets = sorted(df.target.unique())
  return targets, {t: np.flatnonzero(df.target.values == t) for t in targets}


def boot(df, stat, rng_seed, reps):
  """95% percentile interval of stat(sub-DataFrame) over targets resampled with replacement. Each drawn copy of a
  target is its own unit in column _unit (a target drawn twice is two units, as for per-target statistics)."""
  targets, idx = target_index(df)
  rng = np.random.default_rng(rng_seed)
  vals = []
  for row in rng.integers(0, len(targets), size=(reps, len(targets))):
    rows = [idx[targets[i]] for i in row]
    sub = df.iloc[np.concatenate(rows)].assign(_unit=np.repeat(np.arange(len(rows)), [len(r) for r in rows]))
    v = stat(sub)
    if v is not None and not (isinstance(v, float) and math.isnan(v)):
      vals.append(v)
  return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))) if vals else (np.nan, np.nan)


def logistic(x, y, iters=50):
  """Maximum-likelihood (intercept, slope) of P(y) = 1 / (1 + exp(-(a + b x))), by Newton's method (IRLS).
  None if y has a single value or the fit does not converge (separation)."""
  if y.min() == y.max():
    return None
  X = np.column_stack([np.ones_like(x), x])
  beta = np.zeros(2)
  for _ in range(iters):
    p = 1 / (1 + np.exp(-X @ beta))
    w = p * (1 - p)
    h = X.T @ (X * w[:, None])
    try:
      step = np.linalg.solve(h, X.T @ (y - p))
    except np.linalg.LinAlgError:
      return None
    beta = beta + step
    if np.abs(step).max() < 1e-10:
      return tuple(float(b) for b in beta)
  return None


def q_upper(z):
  return 0.5 * math.erfc(z / math.sqrt(2))


def predicted_flip_rate(delta, mu):
  """The (c) model: P(top-2 overtakes top-1) with normal perturbations (median |e| = delta) and exponential
  margins (median mu)."""
  if not mu > 0:
    return np.nan
  if delta <= 0:
    return 0.0
  lam, s = math.log(2) / mu, math.sqrt(2) * delta / MEDIAN_ABS_NORMAL
  z = lam * s
  # exp(z^2/2) Q(z) written with erfcx-free terms; z stays small for these data, the guard keeps it finite.
  tail = math.exp(z * z / 2) * q_upper(z) if z < 30 else 1 / (z * math.sqrt(2 * math.pi))
  return 0.5 - tail


# (a)
def margin_analysis(seed, args):
  rows, fits = [], []
  for c in CONTRASTS:
    g = seed[seed.contrast == c].copy()
    edges = np.quantile(g.margin_a, np.linspace(0, 1, QUINTILES + 1))
    g['quintile'] = np.clip(np.searchsorted(edges, g.margin_a, side='right') - 1, 0, QUINTILES - 1) + 1
    for q in range(1, QUINTILES + 1):
      h = g[g.quintile == q]
      lo, hi = boot(g, lambda d, q=q: float(d[d.quintile == q].flip_top1.mean()) if (d.quintile == q).any() else None,
                    args.seed, args.replicates)
      rows.append({'contrast': c, 'quintile': q, 'margin_low': edges[q - 1], 'margin_high': edges[q],
                   'margin_median': float(h.margin_a.median()), 'n': len(h), 'flips': int(h.flip_top1.sum()),
                   'flip_rate': float(h.flip_top1.mean()), 'ci_low': lo, 'ci_high': hi})
    x, y = np.log10(g.margin_a.values), g.flip_top1.astype(float).values
    fit = logistic(x, y)

    def slope(d):
      f = logistic(np.log10(d.margin_a.values), d.flip_top1.astype(float).values)
      return f[1] if f else None
    lo, hi = boot(g, slope, args.seed, args.replicates)
    fits.append({'contrast': c, 'n': len(g), 'flips': int(y.sum()),
                 'intercept': fit[0] if fit else np.nan, 'slope_per_log10_margin': fit[1] if fit else np.nan,
                 'slope_ci_low': lo, 'slope_ci_high': hi,
                 'odds_ratio_per_tenfold_margin': math.exp(fit[1]) if fit else np.nan,
                 'odds_ratio_ci_low': math.exp(lo) if not math.isnan(lo) else np.nan,
                 'odds_ratio_ci_high': math.exp(hi) if not math.isnan(hi) else np.nan})
  return pd.DataFrame(rows), pd.DataFrame(fits)


# (b)
def top_share(flips_by_target, n_top):
  v = np.sort(np.asarray(flips_by_target))[::-1]
  return float(v[:n_top].sum() / v.sum()) if v.sum() else np.nan


def concentration(seed, args):
  dist, shares = [], []
  for c in CONTRASTS:
    g = seed[seed.contrast == c]
    per = g.groupby('target').flip_top1.sum().astype(int)
    seeds_per_target = int(g.groupby('target').size().max())
    for k in range(seeds_per_target + 1):
      dist.append({'contrast': c, 'seeds_with_flip': k, 'targets': int((per == k).sum()),
                   'share_of_targets': float((per == k).mean())})
    n_top = math.ceil(TOP_FRACTION * len(per))
    share = top_share(per.sort_index().values, n_top)
    lo, hi = boot(g, lambda d: top_share(d.groupby('_unit').flip_top1.sum().values, n_top), args.seed,
                  args.replicates)
    rate = float(g.flip_top1.mean())
    rng = np.random.default_rng(args.seed)
    sims = rng.binomial(seeds_per_target, rate, size=(args.replicates, len(per)))
    null = [top_share(s, n_top) for s in sims if s.sum()]
    shares.append({'contrast': c, 'targets': len(per), 'top_targets': n_top, 'flips': int(per.sum()),
                   'targets_with_a_flip': int((per > 0).sum()), 'top_share': share, 'ci_low': lo, 'ci_high': hi,
                   'binomial_null_share': float(np.mean(null)) if null else np.nan,
                   'binomial_null_p95': float(np.percentile(null, 95)) if null else np.nan,
                   'top_targets_list': ';'.join(per.sort_values(ascending=False, kind='mergesort').index[:n_top])})
  return pd.DataFrame(dist), pd.DataFrame(shares)


# (c)
def perturbation(seed, sample, args):
  rows = []
  for c in CONTRASTS:
    g, h = seed[seed.contrast == c], sample[sample.contrast == c]
    delta, mu = float(h.m5_ranking_score.median()), float(g.margin_a.median())
    joint = pd.concat([g.assign(_kind='seed'), h.assign(_kind='sample')], ignore_index=True)

    def stats(d):
      dd, mm = d[d._kind == 'sample'].m5_ranking_score.median(), d[d._kind == 'seed'].margin_a.median()
      return dd, mm, predicted_flip_rate(dd, mm), d[d._kind == 'seed'].flip_top1.mean()
    ci = {}
    for i, name in enumerate(('delta', 'mu', 'predicted', 'observed')):
      ci[name] = boot(joint, lambda d, i=i: float(stats(d)[i]), args.seed, args.replicates)
    obs = float(g.flip_top1.mean())
    pred = predicted_flip_rate(delta, mu)
    ratio_ci = boot(joint, lambda d: float(stats(d)[0] / stats(d)[1]), args.seed, args.replicates)
    rows.append({'contrast': c, 'n_seed_pairs': len(g), 'n_sample_pairs': len(h),
                 'median_abs_score_difference': delta, 'delta_ci_low': ci['delta'][0], 'delta_ci_high': ci['delta'][1],
                 'median_margin': mu, 'mu_ci_low': ci['mu'][0], 'mu_ci_high': ci['mu'][1],
                 'delta_over_mu': delta / mu, 'ratio_ci_low': ratio_ci[0], 'ratio_ci_high': ratio_ci[1],
                 'predicted_flip_rate': pred, 'predicted_ci_low': ci['predicted'][0],
                 'predicted_ci_high': ci['predicted'][1], 'observed_flip_rate': obs,
                 'observed_ci_low': ci['observed'][0], 'observed_ci_high': ci['observed'][1]})
  return pd.DataFrame(rows)


# (d)
def buckets(seed, args):
  manifest = pd.read_csv(REPO / 'inputs' / 'manifest.csv', dtype={'pdb_id': str})
  bucket = dict(zip(manifest.pdb_id, manifest.bucket.astype(int)))
  rows = []
  for c in CONTRASTS:
    g = seed[seed.contrast == c].assign(bucket=lambda d: d.target.map(bucket))
    for b, h in g.groupby('bucket'):
      lo, hi = boot(h, lambda d: float(d.flip_top1.mean()), args.seed, args.replicates)
      rows.append({'contrast': c, 'bucket': int(b), 'targets': h.target.nunique(), 'n': len(h),
                   'flips': int(h.flip_top1.sum()), 'flip_rate': float(h.flip_top1.mean()), 'ci_low': lo,
                   'ci_high': hi, 'median_margin': float(h.margin_a.median())})
  return pd.DataFrame(rows)


# (e)
def check_sides(seed, samples):
  """Every seed pair's top-1 and success, recomputed from samples.csv with SIDES, must equal the pair's own."""
  bad = []
  for c, ((ca, ra), (cb, rb)) in SIDES.items():
    for side, (cond, rep), top_col, succ_col in (('a', (ca, ra), 'top1_a', 'success_a'),
                                                 ('b', (cb, rb), 'top1_b', 'success_b')):
      s = samples[(samples.platform == cond) & (samples.repetition == rep)]
      best = (s.sort_values(['ranking_score'], ascending=False, kind='mergesort')
              .groupby(['target', 'seed']).head(1).set_index(['target', 'seed']))
      g = seed[seed.contrast == c]
      key = list(zip(g.target, g[f'seed_{side}']))
      if not set(key) <= set(best.index):
        bad.append(f'{c} {side}: pairs without samples')
        continue
      b = best.loc[key]
      if not ((b['sample'].astype(int).values == g[top_col].values).all()
              and ((b.m1_success == True).values == g[succ_col].astype(bool).values).all()):  # noqa: E712
        bad.append(f'{c} {side}: top-1 or success differs')
  return bad


def flips_that_matter(seed, samples):
  rows = []
  for c in CONTRASTS:
    (ca, ra), (cb, rb) = SIDES[c]
    for r in seed[(seed.contrast == c) & seed.flip_matters].itertuples():
      out = {'contrast': c, 'target': r.target, 'seed_a': r.seed_a, 'seed_b': r.seed_b, 'margin_a': r.margin_a}
      for side, cond, rep, top, s in (('a', ca, ra, r.top1_a, r.seed_a), ('b', cb, rb, r.top1_b, r.seed_b)):
        x = samples[(samples.platform == cond) & (samples.repetition == rep) & (samples.target == r.target)
                    & (samples.seed == s) & (samples['sample'] == top)].iloc[0]
        out.update({f'condition_{side}': cond, f'repetition_{side}': rep, f'top1_{side}': int(top),
                    f'ligand_rmsd_{side}': x.m1_ligand_rmsd, f'pb_valid_{side}': x.m1_pb_valid,
                    f'success_{side}': bool(x.m1_success == True),  # noqa: E712
                    f'pb_failed_{side}': x.m1_pb_failed if isinstance(x.m1_pb_failed, str) else ''})
      ok, bad = ('a', 'b') if out['success_a'] else ('b', 'a')
      rb_ = out[f'ligand_rmsd_{bad}']
      out['separated_by'] = ('RMSD' if isinstance(rb_, float) and rb_ >= 2.0 and out[f'pb_valid_{bad}'] in (True, 'True')
                             else 'PoseBusters' if isinstance(rb_, float) and rb_ < 2.0
                             else 'RMSD and PoseBusters' if isinstance(rb_, float) else 'M1 not computed')
      out['rmsd_difference'] = abs(out['ligand_rmsd_a'] - out['ligand_rmsd_b']) \
          if all(isinstance(out[f'ligand_rmsd_{k}'], float) for k in 'ab') else np.nan
      rows.append(out)
  df = pd.DataFrame(rows)
  summary = (df.groupby(['contrast', 'separated_by']).size().unstack(fill_value=0).reindex(CONTRASTS).fillna(0)
             .astype(int).reset_index()) if len(df) else pd.DataFrame()
  if len(df):
    summary['flips_that_matter'] = summary.drop(columns='contrast').sum(axis=1)
    med = df.groupby('contrast').rmsd_difference.median()
    summary['median_rmsd_difference'] = summary.contrast.map(med)
  return df, summary


def pct(x):
  return '-' if pd.isna(x) else f'{100 * x:.1f}%'


def report(out, qa, la, bd, bs, cp, db, ef, es, args):
  lines = ['# Exploratory: why top-1 flips occur', '',
           'Section 10 row of 2026-10-14; exploratory, no primary result changes. Main-study r1 data after the M4 '
           'correction (primary_v2). A is the first-named condition. Intervals: 95% bootstrap over targets '
           f'({args.replicates} replicates, seed {args.seed}). Code: analysis/exploratory.py.', '',
           '## (a) Flip rate by M3 margin quintile of A', '',
           '| contrast | ' + ' | '.join(f'Q{q}' for q in range(1, QUINTILES + 1)) + ' | odds ratio per tenfold margin |',
           '|---|' + '---|' * (QUINTILES + 1)]
  for c in CONTRASTS:
    q = qa[qa.contrast == c].sort_values('quintile')
    f = la[la.contrast == c].iloc[0]
    lines.append(f'| {c} | ' + ' | '.join(f'{pct(r.flip_rate)} [{pct(r.ci_low)}, {pct(r.ci_high)}]'
                                           for r in q.itertuples())
                 + f' | {f.odds_ratio_per_tenfold_margin:.2f} [{f.odds_ratio_ci_low:.2f}, {f.odds_ratio_ci_high:.2f}] |')
  q = qa[qa.contrast == CONTRASTS[0]].sort_values('quintile')
  lines += ['', f'Quintile margins for {CONTRASTS[0]} (median per quintile): '
            + ', '.join(f'Q{r.quintile} {r.margin_median:.2g}' for r in q.itertuples()) + '.', '',
            '## (b) Concentration over targets', '',
            '| contrast | targets with 0, 1, 2, ... seeds with a flip | flips | share of flips from the top 10% of targets '
            '| binomial null (mean, 95th pct) |', '|---|---|---|---|---|']
  for r in bs.itertuples():
    d = bd[bd.contrast == r.contrast].sort_values('seeds_with_flip')
    lines.append(f'| {r.contrast} | {", ".join(str(x) for x in d.targets)} | {r.flips} | {pct(r.top_share)} '
                 f'[{pct(r.ci_low)}, {pct(r.ci_high)}] ({r.top_targets} targets: {r.top_targets_list.replace(";", ", ")}) '
                 f'| {pct(r.binomial_null_share)}, {pct(r.binomial_null_p95)} |')
  lines += ['', '## (c) Perturbation size against margin', '',
            '| contrast | median abs score difference (delta) | median margin (mu) | delta / mu | predicted flip rate '
            '| observed flip rate |', '|---|---|---|---|---|---|']
  for r in cp.itertuples():
    lines.append(f'| {r.contrast} | {r.median_abs_score_difference:.2g} [{r.delta_ci_low:.2g}, {r.delta_ci_high:.2g}] | '
                 f'{r.median_margin:.2g} [{r.mu_ci_low:.2g}, {r.mu_ci_high:.2g}] | {r.delta_over_mu:.2f} '
                 f'[{r.ratio_ci_low:.2f}, {r.ratio_ci_high:.2f}] | {pct(r.predicted_flip_rate)} '
                 f'[{pct(r.predicted_ci_low)}, {pct(r.predicted_ci_high)}] | {pct(r.observed_flip_rate)} '
                 f'[{pct(r.observed_ci_low)}, {pct(r.observed_ci_high)}] |')
  lines += ['', 'Model (no free parameter): each sample\'s score moves by independent normal noise with median absolute '
            'value delta, margins are exponential with median mu, and the top-1 flips when the top-2 overtakes it.', '',
            '## (d) Flip rate by token bucket', '', '| contrast | ' + ' | '.join(
                f'{b} tokens' for b in sorted(db.bucket.unique())) + ' |', '|---|' + '---|' * db.bucket.nunique()]
  for c in CONTRASTS:
    d = db[db.contrast == c].set_index('bucket')
    lines.append(f'| {c} | ' + ' | '.join(
        f'{pct(d.loc[b].flip_rate)} [{pct(d.loc[b].ci_low)}, {pct(d.loc[b].ci_high)}] ({int(d.loc[b].targets)})'
        if b in d.index else '-' for b in sorted(db.bucket.unique())) + ' |')
  lines += ['', 'In parentheses: targets in the bucket.', '', '## (e) Flips that matter', '']
  if len(es):
    cols = [c for c in es.columns if c not in ('contrast', 'flips_that_matter', 'median_rmsd_difference')]
    lines += ['| contrast | flips that matter | ' + ' | '.join(cols) + ' | median abs(RMSD A - RMSD B) (angstrom) |',
              '|---|---|' + '---|' * (len(cols) + 1)]
    lines += [f'| {r["contrast"]} | {r["flips_that_matter"]} | ' + ' | '.join(str(r[c]) for c in cols)
              + f' | {r["median_rmsd_difference"]:.2f} |' for r in es.to_dict('records')]
    lines += ['', '| contrast | target | seeds | A: sample, RMSD, PB-valid | B: sample, RMSD, PB-valid | separated by |',
              '|---|---|---|---|---|---|']
    lines += [f'| {r.contrast} | {r.target} | {r.seed_a}/{r.seed_b} | {r.top1_a}, {r.ligand_rmsd_a:.2f}, {r.pb_valid_a} '
              f'| {r.top1_b}, {r.ligand_rmsd_b:.2f}, {r.pb_valid_b} | {r.separated_by} |' for r in ef.itertuples()]
  (out / 'report.md').write_text('\n'.join(lines) + '\n')


def main():
  ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
  ap.add_argument('--main', default='results/analysis/main_20261014')
  ap.add_argument('--out')
  ap.add_argument('--replicates', type=int, default=10000)
  ap.add_argument('--seed', type=int, default=2026)
  args = ap.parse_args()
  main_dir = REPO / args.main if not pathlib.Path(args.main).is_absolute() else pathlib.Path(args.main)
  out = pathlib.Path(args.out) if args.out else main_dir / 'exploratory'
  out.mkdir(parents=True, exist_ok=True)
  seed, sample, samples = load(main_dir)
  missing = sorted(set(CONTRASTS) - set(seed.contrast))
  if missing:
    raise SystemExit(f'contrasts not in pairs_seed.csv: {missing}')
  bad = check_sides(seed, samples)
  if bad:
    raise SystemExit('SIDES does not match the pairs: ' + '; '.join(bad))
  qa, la = margin_analysis(seed, args)
  bd, bs = concentration(seed, args)
  cp = perturbation(seed, sample, args)
  db = buckets(seed, args)
  ef, es = flips_that_matter(seed, samples)
  for name, df in (('a_margin_quintiles', qa), ('a_logistic', la), ('b_seeds_with_flip', bd), ('b_top_share', bs),
                   ('c_perturbation', cp), ('d_buckets', db), ('e_flips_that_matter', ef), ('e_summary', es)):
    df.to_csv(out / f'{name}.csv', index=False)
  report(out, qa, la, bd, bs, cp, db, ef, es, args)
  print(f'>> exploratory: {len(CONTRASTS)} contrasts, {len(seed)} seed pairs, {len(ef)} flips that matter; '
        f'report {out / "report.md"}', flush=True)
  return 0


if __name__ == '__main__':
  sys.exit(main())
