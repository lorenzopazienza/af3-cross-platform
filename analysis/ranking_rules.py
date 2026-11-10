"""Ranking rules and their cost (section 10, row C1 of drafts/section10_campaign_ranking_20261110.md, label ranking):
a follow-up analysis of frozen outputs, outside the pre-registered study, descriptive whatever it shows; part (iv) is
exploratory. No run and no cloud resource.

    analysis/.venv/bin/python analysis/ranking_rules.py --main . --out results/analysis/ranking_<DATE> \\
        [--replicates 10000] [--seed 2026]

Inputs, relative to --main (the checkout that holds results/):
  results/analysis/main_20261014/primary_v2/samples.csv   V5, V6, G (r1, r2, seeds 1 to 5); GT, Gd, V6h, Gh (r1) for (iv)
  results/analysis/seeds_20261102/samples.csv             V5 and G, r1, seeds 6 to 20 (follow-up 8)
  results/analysis/h100_20261104/samples.csv              H, r1 and r2, seeds 1 to 5 (follow-up 9)
  results/analysis/a100_20261018/samples.csv              A, r1 and r2, seeds 1 to 5 (follow-up 3; see choice 1)
  results/analysis/main_20261014/primary_v2/cost/cost_summary.csv
Rows are kept at AlphaFold3's defaults (num_recycles 10, num_diffusion_samples 5) of each file's own label (main,
seeds, h100, a100). The script stops with a message if a platform, repetition, seed or target it needs is missing, or
if a (target, seed) does not hold exactly the samples 0 to 4 with both scores.

Rules (a unit is a (target, block of seeds); samples in a block are ordered by (seed, sample), so the plain argmax
breaks exact ties to the lowest (seed, sample) index):
  (i) seed consensus: the sample with the highest full-precision ranking_score among the 5n samples of a block of n
      consecutive seeds, n = 1, 2, 5, 10, 20; HW(V5,G) on seeds 1 to 20 (r1); RC(G) at n = 1 and 5 on seeds 1 to 5.
  (ii) tie tolerance: on each platform the reported set is the samples with top score minus score at most t; a tie is
      declared when the set holds two or more samples; per seed (n = 1) and across seeds 1 to 5 (n = 5); HW(V5,V6),
      HW(V6,G), HW(H,A), RC(G), RC(A), RC(H).
  (iii) rounded score: argmax of ranking_score_summary (ties to the lowest (seed, sample) index) against the
      full-precision argmax; same pairs and n as (ii).
  (iv) exploratory: for every contrast of analysis/exploratory.py (CONTRASTS, sides from its SIDES), the resampling
      model (each (target, seed)'s five first-condition scores plus the whole five-component difference vector,
      second minus first, of every (target, seed) of the other targets; the top-1 flips when the argmax changes),
      against the observed rate and the two-sample model of (c) (exploratory.predicted_flip_rate with delta the
      median absolute same-sample difference and mu the median top-1 minus top-2 margin of the first condition); and
      for HW(V5,G) on seeds 1 to 20, the same with blocks of n = 1, 2, 5 seeds (5n components, block margins).
Metrics: top-1 flip, success (m1_success) flip, success rate on each platform (a = first-named, b = second) and a
minus b; for (ii) also the conflict rate (reported sets disjoint) and the fraction of units with a tie declared on
either platform. Cost: processes per reported prediction (n) and n times the median per bucket from cost_summary.csv
(cache fresh) of the target's token bucket, averaged over the targets, on V5 and G.
Statistics: paired bootstrap over targets (one matrix of target multiplicities, --replicates rows, from --seed, shared
by every statistic), 95% percentile intervals; differences against n = 1 (i), t = 0 (ii) and the full-precision rule
(iii) on the same resampled targets. Rates pool units over the drawn targets (sum of numerators over sum of
denominators; a target drawn twice counts twice).

Choices where the row is silent (simplest reading):
  1. A is not in h100_20261104/samples.csv (it holds H only); A r1 and r2 are read from a100_20261018/samples.csv,
     the file analysis/h100.py takes A from.
  2. (ii): the top-1 flip and the conflict are over all units (a unit with a tie is not a flip); the success flip,
     the success rates and their difference are over units where both platforms report one sample. The set is
     top - score <= t (a margin equal to t counts as a tie; at t = 0 only exact ties do).
  3. HW pairs compare the same seed and sample index on the two conditions; the first-named condition is a.
  4. (iv): "the other targets" excludes every unit of the unit's own target (in a bootstrap resample, every copy of
     it); the model rate is the mean over units of each unit's mean over the other targets' vectors. The two-sample
     model for blocks uses the median block margin (top-1 minus top-2 of 5n samples) and the per-sample delta.
     For a contrast, the seeds are those both conditions ran (seeds 1 and 2 when V6h or Gh is a side).
  5. Cost: median_wall_s of cost_summary.csv is the median wall per process; a main-study process runs one seed, so
     it is the wall per structure. The file has no amortised wall column, so the amortised variant is given for USD
     only (median_usd_per_structure_amortised_standard and _spot). Intervals for cost come from the same target
     bootstrap.
  6. A resample in which a ratio has no denominator (or (iv) has no other target) gives no value for that replicate;
     the interval is taken over the replicates that have one.
Outputs in --out: seed_consensus.csv, tie_tolerance.csv, rounded.csv, model.csv, cost.csv, report.md, inputs.tsv.
"""

import argparse
import hashlib
import pathlib
import sys

import numpy as np
import pandas as pd

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / 'analysis'))
import exploratory as ex  # noqa: E402

MAIN = 'results/analysis/main_20261014/primary_v2/samples.csv'
SEEDS = 'results/analysis/seeds_20261102/samples.csv'
H100 = 'results/analysis/h100_20261104/samples.csv'
A100 = 'results/analysis/a100_20261018/samples.csv'
COST = 'results/analysis/main_20261014/primary_v2/cost/cost_summary.csv'
LABELS = {MAIN: 'main', SEEDS: 'seeds', H100: 'h100', A100: 'a100'}
COLUMNS = ['label', 'platform', 'repetition', 'target', 'bucket', 'seed', 'sample', 'num_recycles',
           'num_diffusion_samples', 'ranking_score', 'ranking_score_summary', 'm1_success']
K = 5                                   # samples per seed
SEEDS_MAIN = [1, 2, 3, 4, 5]
SEEDS_ALL = list(range(1, 21))
N_CONSENSUS = [1, 2, 5, 10, 20]
N_SHORT = [1, 5]
N_MODEL = [1, 2, 5]
TOLERANCES = [0.0, 0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01]
PAIRS = [('HW(V5,V6)', ('V5', 'r1'), ('V6', 'r1')), ('HW(V6,G)', ('V6', 'r1'), ('G', 'r1')),
         ('HW(H,A)', ('H', 'r1'), ('A', 'r1')), ('RC(G)', ('G', 'r1'), ('G', 'r2')),
         ('RC(A)', ('A', 'r1'), ('A', 'r2')), ('RC(H)', ('H', 'r1'), ('H', 'r2'))]
COST_PLATFORMS = ['V5', 'G']
COST_COLS = {'median_wall_s': 'wall_s', 'median_usd_per_structure_standard': 'usd_on_demand',
             'median_usd_per_structure_spot': 'usd_spot',
             'median_usd_per_structure_amortised_standard': 'usd_on_demand_amortised',
             'median_usd_per_structure_amortised_spot': 'usd_spot_amortised'}
METRICS = ['flip_top1', 'flip_success', 'success_a', 'success_b', 'success_difference']
STATUS = ('Follow-up analysis outside the pre-registered study (section 10, row C1, label ranking), descriptive; '
          'part (iv) is exploratory.')


# ---- inputs

def sha256(path):
  h = hashlib.sha256()
  with open(path, 'rb') as f:
    for chunk in iter(lambda: f.read(1 << 20), b''):
      h.update(chunk)
  return h.hexdigest()


def load(path, label):
  d = pd.read_csv(path, usecols=lambda c: c in COLUMNS, low_memory=False)
  missing = sorted(set(COLUMNS) - set(d.columns))
  if missing:
    raise SystemExit(f'{path}: columns missing: {missing}')
  d = d[(d.label == label) & (d.num_recycles == 10) & (d.num_diffusion_samples == K) & d['sample'].notna()].copy()
  d['sample'] = d['sample'].astype(int)
  d['seed'] = d.seed.astype(int)
  d['success'] = d.m1_success.astype(str).str.lower().eq('true')
  return d


class Cube:
  """Scores, two-decimal scores and success of one (platform, repetition), shape (targets, seeds, 5)."""

  def __init__(self, score, summ, succ):
    self.score, self.summ, self.succ = score, summ, succ


def cube(d, platform, rep, targets, seeds, source):
  s = d[(d.platform == platform) & (d.repetition == rep) & d.seed.isin(seeds)]
  if s.empty:
    raise SystemExit(f'{source}: no rows for {platform} {rep} seeds {seeds[0]} to {seeds[-1]}')
  dup = s.duplicated(['target', 'seed', 'sample'])
  if dup.any():
    raise SystemExit(f'{source}: {platform} {rep}: the same (target, seed, sample) twice, e.g. '
                     f'{s[dup].iloc[0][["target", "seed", "sample"]].tolist()}')
  extra = sorted(set(s.target) - set(targets))
  if extra:
    raise SystemExit(f'{source}: {platform} {rep}: targets not in the main study: {extra[:5]}')
  idx = pd.MultiIndex.from_product([targets, seeds, range(K)], names=['target', 'seed', 'sample'])
  f = s.set_index(['target', 'seed', 'sample']).reindex(idx)
  bad = f.ranking_score.isna() | f.ranking_score_summary.isna()
  if bad.any():
    t, sd, k = bad[bad].index[0]
    units = len({(a, b) for a, b, _ in bad[bad].index})
    raise SystemExit(f'{source}: {platform} {rep}: {units} (target, seed) without the {K} samples 0 to {K - 1} '
                     f'with both scores, first {t} seed {sd} sample {k}')
  shape = (len(targets), len(seeds), K)
  return Cube(f.ranking_score.to_numpy(float).reshape(shape), f.ranking_score_summary.to_numpy(float).reshape(shape),
              f.success.fillna(False).to_numpy(bool).reshape(shape))


def join(c1, c2):
  return Cube(*(np.concatenate([getattr(c1, k), getattr(c2, k)], axis=1) for k in ('score', 'summ', 'succ')))


def block(x, n):
  """(T, S, 5) to (T, S / n, 5n): disjoint blocks of n consecutive seeds, samples in (seed, sample) order."""
  t, s, k = x.shape
  if s % n:
    raise ValueError(f'{s} seeds do not split into blocks of {n}')
  return x.reshape(t, s // n, n * k)


# ---- bootstrap

class Boot:
  """Target multiplicities W (replicates x targets), drawn once and shared by every statistic."""

  def __init__(self, n_targets, replicates, seed):
    rng = np.random.default_rng(seed)
    draw = rng.integers(0, n_targets, size=(replicates, n_targets))
    off = (draw + np.arange(replicates)[:, None] * n_targets).ravel()
    self.W = np.bincount(off, minlength=replicates * n_targets).reshape(replicates, n_targets).astype(float)
    self.T = n_targets

  def ratio(self, num, den):
    """Point estimate and bootstrap replicates of sum(num) / sum(den) over targets."""
    est = float(num.sum() / den.sum()) if den.sum() else np.nan
    with np.errstate(invalid='ignore', divide='ignore'):
      bs = (self.W @ num) / (self.W @ den)
    return est, bs


def interval(bs):
  bs = np.asarray(bs, float)
  bs = bs[np.isfinite(bs)]
  return (float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))) if len(bs) else (np.nan, np.nan)


def take(x, pick):
  return np.take_along_axis(x, pick[..., None], -1)[..., 0]


def unit_metrics(succ_a, succ_b, pick_a, pick_b, set_a=None, set_b=None):
  """Per-unit (value, mask) of each metric. succ_* (T, B, k) bool; pick_* (T, B) sample index in the block; set_*
  (T, B, k) bool, the reported sets of rule (ii) (None: one reported sample per platform)."""
  sa, sb = take(succ_a, pick_a).astype(float), take(succ_b, pick_b).astype(float)
  every = np.ones(pick_a.shape, bool)
  both = every if set_a is None else (set_a.sum(-1) == 1) & (set_b.sum(-1) == 1)
  out = {'flip_top1': (both & (pick_a != pick_b), every), 'flip_success': (sa != sb, both),
         'success_a': (sa, both), 'success_b': (sb, both), 'success_difference': (sa - sb, both)}
  if set_a is not None:
    out['conflict'] = (~(set_a & set_b).any(-1), every)
    out['tie_either'] = (~both, every)
  return out


def per_target(val, mask):
  m = mask.reshape(mask.shape[0], -1)
  v = np.where(m, np.asarray(val, float).reshape(m.shape), 0.0)
  return v.sum(1), m.sum(1).astype(float)


def summarise(m, boot, ref=None, keys=None):
  """Rows {metric, units, estimate, ci, and the difference to ref (same resample)} from unit_metrics outputs."""
  rows = []
  for name, (val, mask) in m.items():
    num, den = per_target(val, mask)
    est, bs = boot.ratio(num, den)
    lo, hi = interval(bs)
    row = dict(keys or {}, metric=name, units=int(den.sum()), estimate=est, ci_low=lo, ci_high=hi)
    if ref is not None:
      rnum, rden = per_target(*ref[name])
      rest, rbs = boot.ratio(rnum, rden)
      dlo, dhi = interval(bs - rbs)
      row.update(difference=est - rest, difference_ci_low=dlo, difference_ci_high=dhi)
    rows.append(row)
  return rows


# ---- rules

def consensus(a, b, n):
  sa, sb = block(a.score, n), block(b.score, n)
  return unit_metrics(block(a.succ, n), block(b.succ, n), sa.argmax(-1), sb.argmax(-1))


def seed_consensus(v5, g, g1, g2, boot):
  """(i): HW(V5,G) on seeds 1 to 20 at every n of N_CONSENSUS; RC(G) at n = 1 and 5 on seeds 1 to 5."""
  rows = []
  for pair, a, b, ns in (('HW(V5,G)', v5, g, N_CONSENSUS), ('RC(G)', g1, g2, N_SHORT)):
    ref = consensus(a, b, 1)
    for n in ns:
      rows += summarise(consensus(a, b, n), boot, ref, {'pair': pair, 'n': n, 'seeds': a.score.shape[1],
                                                        'difference_against': 'n = 1'})
  return pd.DataFrame(rows)


def tie_metrics(a, b, n, t):
  sa, sb = block(a.score, n), block(b.score, n)
  set_a = (sa.max(-1, keepdims=True) - sa) <= t
  set_b = (sb.max(-1, keepdims=True) - sb) <= t
  return unit_metrics(block(a.succ, n), block(b.succ, n), sa.argmax(-1), sb.argmax(-1), set_a, set_b)


def tie_tolerance(cubes, boot):
  rows = []
  for pair, ka, kb in PAIRS:
    for n in N_SHORT:
      ref = tie_metrics(cubes[ka], cubes[kb], n, 0.0)
      for t in TOLERANCES:
        rows += summarise(tie_metrics(cubes[ka], cubes[kb], n, t), boot, ref,
                          {'pair': pair, 'n': n, 't': t, 'difference_against': 't = 0'})
  return pd.DataFrame(rows)


def rounded_metrics(a, b, n, rounded):
  key = 'summ' if rounded else 'score'
  pa, pb = block(getattr(a, key), n).argmax(-1), block(getattr(b, key), n).argmax(-1)
  return unit_metrics(block(a.succ, n), block(b.succ, n), pa, pb)


def rounded(cubes, boot):
  rows = []
  for pair, ka, kb in PAIRS:
    for n in N_SHORT:
      ref = rounded_metrics(cubes[ka], cubes[kb], n, False)
      for r in (False, True):
        rows += summarise(rounded_metrics(cubes[ka], cubes[kb], n, r), boot, ref,
                          {'pair': pair, 'n': n, 'ranking': 'ranking_score_summary' if r else 'ranking_score',
                           'difference_against': 'ranking_score'})
  return pd.DataFrame(rows)


# ---- (iv) exploratory model

def flip_matrix(A, D, chunk=256):
  """F[u, v]: the argmax of A[u] + D[v] differs from the argmax of A[u] (the vector D[v] kept whole)."""
  pa = A.argmax(1)
  F = np.zeros((len(A), len(D)), bool)
  for i in range(0, len(A), chunk):
    F[i:i + chunk] = (A[i:i + chunk, None, :] + D[None, :, :]).argmax(-1) != pa[i:i + chunk, None]
  return F


def resampling_model(A, D, tg, n_targets, W=None, chunk=1000):
  """Mean over units u of the mean over units v of the other targets of F[u, v], for each row of target
  multiplicities W (None: every target once). Returns (estimate, replicates)."""
  U = len(A)
  onehot = np.zeros((U, n_targets))
  onehot[np.arange(U), tg] = 1.0
  S = flip_matrix(A, D).astype(float) @ onehot              # (U, T): flips of u over the units of each target
  N = onehot.sum(0)
  own = S[np.arange(U), tg]

  def stat(Wm):
    wu = Wm[:, tg]
    num = Wm @ S.T - wu * own[None, :]
    den = (Wm @ N)[:, None] - wu * N[tg][None, :]
    with np.errstate(invalid='ignore', divide='ignore'):
      inner = num / den
      return (wu * inner).sum(1) / wu.sum(1)
  est = float(stat(np.ones((1, n_targets)))[0])
  if W is None:
    return est, None
  return est, np.concatenate([stat(W[i:i + chunk]) for i in range(0, len(W), chunk)])


def weighted_median(values, tg, W, chunk=500):
  """np.median of the values of the resampled targets (each value repeated by its target's multiplicity)."""
  order = np.argsort(values, kind='mergesort')
  v, t = values[order], tg[order]
  out = []
  for i in range(0, len(W), chunk):
    C = np.cumsum(W[i:i + chunk][:, t], axis=1)
    tot = C[:, -1]
    lo = (C <= ((tot - 1) // 2)[:, None]).sum(1)
    hi = (C <= (tot // 2)[:, None]).sum(1)
    out.append((v[np.minimum(lo, len(v) - 1)] + v[np.minimum(hi, len(v) - 1)]) / 2)
  return np.concatenate(out)


def margins(A):
  s = np.sort(A, axis=1)
  return s[:, -1] - s[:, -2]


def model_row(contrast, n, a_scores, b_scores, boot):
  """a_scores, b_scores (T, B, k): one unit per (target, block)."""
  T, B, k = a_scores.shape
  A, D = a_scores.reshape(-1, k), (b_scores - a_scores).reshape(-1, k)
  tg = np.repeat(np.arange(T), B)
  est, bs = resampling_model(A, D, tg, T, boot.W)
  flip = (A.argmax(1) != (A + D).argmax(1)).reshape(T, B)
  oest, obs = boot.ratio(flip.sum(1).astype(float), np.full(T, float(B)))
  absd, mu_u = np.abs(D).ravel(), margins(A)
  delta, mu = float(np.median(absd)), float(np.median(mu_u))
  dbs, mbs = weighted_median(absd, np.repeat(tg, k), boot.W), weighted_median(mu_u, tg, boot.W)
  two = ex.predicted_flip_rate(delta, mu)
  tbs = np.array([ex.predicted_flip_rate(x, y) for x, y in zip(dbs, mbs)])
  row = {'contrast': contrast, 'n': n, 'targets': T, 'units': T * B, 'observed_flip_rate': oest}
  row['observed_ci_low'], row['observed_ci_high'] = interval(obs)
  row['resampling_model'] = est
  row['resampling_ci_low'], row['resampling_ci_high'] = interval(bs)
  row['resampling_minus_observed'] = est - oest
  row['resampling_minus_observed_ci_low'], row['resampling_minus_observed_ci_high'] = interval(bs - obs)
  row['two_sample_model'] = two
  row['two_sample_ci_low'], row['two_sample_ci_high'] = interval(tbs)
  row.update(median_abs_score_difference=delta, median_margin=mu)
  return row


def model(main, targets, v5, g, boot):
  rows = []
  for c in ex.CONTRASTS:
    (ca, ra), (cb, rb) = ex.SIDES[c]
    seeds = sorted(set(main[(main.platform == ca) & (main.repetition == ra)].seed)
                   & set(main[(main.platform == cb) & (main.repetition == rb)].seed))
    if not seeds:
      raise SystemExit(f'{MAIN}: no common seed for {c} ({ca} {ra}, {cb} {rb})')
    a, b = cube(main, ca, ra, targets, seeds, MAIN), cube(main, cb, rb, targets, seeds, MAIN)
    rows.append(model_row(c, 1, a.score, b.score, boot))
  for n in N_MODEL:
    rows.append(model_row('HW(V5,G)', n, block(v5.score, n), block(g.score, n), boot))
  return pd.DataFrame(rows)


# ---- cost

def cost(cs, buckets, boot):
  rows = []
  for p in COST_PLATFORMS:
    f = cs[(cs.platform == p) & (cs.cache == 'fresh')].set_index('bucket')
    missing = sorted(set(buckets) - set(f.index))
    if missing:
      raise SystemExit(f'{COST}: no fresh row for {p} in buckets {missing}')
    for n in N_CONSENSUS:
      rows.append({'platform': p, 'n': n, 'quantity': 'processes_per_prediction', 'estimate': float(n),
                   'ci_low': float(n), 'ci_high': float(n)})
      for col, name in COST_COLS.items():
        v = f.loc[buckets, col].to_numpy(float)
        lo, hi = interval(n * (boot.W @ v) / boot.T)
        rows.append({'platform': p, 'n': n, 'quantity': f'{name}_per_prediction', 'estimate': n * float(v.mean()),
                     'ci_low': lo, 'ci_high': hi})
  return pd.DataFrame(rows)


# ---- report

def pct(x):
  return '-' if pd.isna(x) else f'{100 * x:.1f}%'


def iv(r, k='estimate', lo='ci_low', hi='ci_high'):
  return f'{pct(r[k])} [{pct(r[lo])}, {pct(r[hi])}]'


def report(out, sc, tt, rd, md, co, n_targets, args):
  L = ['# Ranking rules and their cost', '', STATUS, '',
       f'{n_targets} targets. Intervals: 95% percentile, paired bootstrap over targets ({args.replicates} replicates, '
       f'seed {args.seed}); differences on the same resampled targets. Code: analysis/ranking_rules.py.', '',
       '## (i) Seed consensus (top-1 among the 5n samples of n consecutive seeds)', '',
       '| pair | n | units | top-1 flip | minus n = 1 | success flip | success a | success b |', '|---|---|---|---|---|---|---|---|']
  for (pair, n), g in sc.groupby(['pair', 'n'], sort=False):
    m = g.set_index('metric')
    L.append(f'| {pair} | {n} | {m.loc["flip_top1", "units"]} | {iv(m.loc["flip_top1"])} | '
             f'{iv(m.loc["flip_top1"], "difference", "difference_ci_low", "difference_ci_high")} | '
             f'{iv(m.loc["flip_success"])} | {pct(m.loc["success_a", "estimate"])} | {pct(m.loc["success_b", "estimate"])} |')
  L += ['', '## (ii) Tie tolerance', '',
        '| pair | n | t | top-1 flip | conflict | tie on either | success flip (both single) |', '|---|---|---|---|---|---|---|']
  for (pair, n, t), g in tt.groupby(['pair', 'n', 't'], sort=False):
    m = g.set_index('metric')
    L.append(f'| {pair} | {n} | {t:g} | {iv(m.loc["flip_top1"])} | {iv(m.loc["conflict"])} | '
             f'{iv(m.loc["tie_either"])} | {iv(m.loc["flip_success"])} |')
  L += ['', '## (iii) Rounded score against full precision', '',
        '| pair | n | ranking | top-1 flip | minus full precision | success flip |', '|---|---|---|---|---|---|']
  for (pair, n, r), g in rd.groupby(['pair', 'n', 'ranking'], sort=False):
    m = g.set_index('metric')
    L.append(f'| {pair} | {n} | {r} | {iv(m.loc["flip_top1"])} | '
             f'{iv(m.loc["flip_top1"], "difference", "difference_ci_low", "difference_ci_high")} | '
             f'{iv(m.loc["flip_success"])} |')
  L += ['', '## (iv) Exploratory: resampling model of the top-1 flip', '',
        '| contrast | n | units | observed | resampling model | two-sample model of (c) |', '|---|---|---|---|---|---|']
  for r in md.to_dict('records'):
    L.append(f'| {r["contrast"]} | {r["n"]} | {r["units"]} | {iv(r, "observed_flip_rate", "observed_ci_low", "observed_ci_high")} | '
             f'{iv(r, "resampling_model", "resampling_ci_low", "resampling_ci_high")} | '
             f'{iv(r, "two_sample_model", "two_sample_ci_low", "two_sample_ci_high")} |')
  L += ['', '## Cost per reported prediction (fresh cache; amortised variant for USD only)', '',
        '| platform | n | processes | wall (s) | on demand (USD) | Spot (USD) | on demand amortised | Spot amortised |',
        '|---|---|---|---|---|---|---|---|']
  for (p, n), g in co.groupby(['platform', 'n'], sort=False):
    m = g.set_index('quantity').estimate
    L.append(f'| {p} | {n} | {int(m["processes_per_prediction"])} | {m["wall_s_per_prediction"]:.0f} | '
             f'{m["usd_on_demand_per_prediction"]:.4f} | {m["usd_spot_per_prediction"]:.4f} | '
             f'{m["usd_on_demand_amortised_per_prediction"]:.4f} | {m["usd_spot_amortised_per_prediction"]:.4f} |')
  L += ['', 'Reading: descriptive, no reading rule; (iv) is exploratory.']
  (out / 'report.md').write_text('\n'.join(L) + '\n')


# ---- main

def run(main_root, out, replicates, seed):
  out.mkdir(parents=True, exist_ok=True)
  paths = [MAIN, SEEDS, H100, A100, COST]
  for p in paths:
    if not (main_root / p).is_file():
      raise SystemExit(f'input missing: {main_root / p}')
  with open(out / 'inputs.tsv', 'w') as f:
    f.write('path\tsha256\n')
    for p in paths:
      f.write(f'{p}\t{sha256(main_root / p)}\n')
  d = {p: load(main_root / p, LABELS[p]) for p in (MAIN, SEEDS, H100, A100)}
  main = d[MAIN]
  targets = sorted(main[(main.platform == 'V5') & (main.repetition == 'r1')].target.unique())
  if not targets:
    raise SystemExit(f'{MAIN}: no V5 r1 rows')
  bk = main[main.target.isin(targets)].groupby('target').bucket.nunique()
  if (bk != 1).any():
    raise SystemExit(f'{MAIN}: more than one bucket for {list(bk[bk != 1].index)[:5]}')
  buckets = main.groupby('target').bucket.first().loc[targets].astype(int).tolist()
  cubes = {}
  for (p, r) in {(p, r) for _, a, b in PAIRS for p, r in (a, b)}:
    src = H100 if p == 'H' else A100 if p == 'A' else MAIN
    cubes[(p, r)] = cube(d[src], p, r, targets, SEEDS_MAIN, src)
  late = [s for s in SEEDS_ALL if s not in SEEDS_MAIN]
  v5 = join(cube(main, 'V5', 'r1', targets, SEEDS_MAIN, MAIN), cube(d[SEEDS], 'V5', 'r1', targets, late, SEEDS))
  g = join(cubes[('G', 'r1')], cube(d[SEEDS], 'G', 'r1', targets, late, SEEDS))
  boot = Boot(len(targets), replicates, seed)
  sc = seed_consensus(v5, g, cubes[('G', 'r1')], cubes[('G', 'r2')], boot)
  tt = tie_tolerance(cubes, boot)
  rd = rounded(cubes, boot)
  md = model(main, targets, v5, g, boot)
  co = cost(pd.read_csv(main_root / COST), buckets, boot)
  for name, df in (('seed_consensus', sc), ('tie_tolerance', tt), ('rounded', rd), ('model', md), ('cost', co)):
    df.to_csv(out / f'{name}.csv', index=False)
  report(out, sc, tt, rd, md, co, len(targets), argparse.Namespace(replicates=replicates, seed=seed))
  print(f'>> ranking: {len(targets)} targets, {replicates} replicates; report {out / "report.md"}', flush=True)


def main():
  ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
  ap.add_argument('--main', default=str(REPO), help='checkout that holds results/ (default: this repository)')
  ap.add_argument('--out', required=True)
  ap.add_argument('--replicates', type=int, default=10000)
  ap.add_argument('--seed', type=int, default=2026)
  args = ap.parse_args()
  run(pathlib.Path(args.main).resolve(), pathlib.Path(args.out), args.replicates, args.seed)
  return 0


if __name__ == '__main__':
  sys.exit(main())
