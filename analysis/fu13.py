"""Follow-up 13, why the top-ranked sample flips (label why; notes/analysis_plan.md section 10, row of 2026-11-10).

    analysis/.venv/bin/python analysis/fu13.py --samples <collect.py samples.csv of the why sessions> --out DIR

Conditions, from each session's configuration (harness/configs.yaml, recorded by collect.py as config):
tpu_xla_fu13_p<K> -> P<K> (P0: eps = 0; P5 to P14: eps = 2^-K), tpu_xla_fu13_capture -> V5L or V6L (by device),
l4_xla_fu13_capture -> GL, tpu_xla_fu13_f32 -> V6f, l4_xla_fu13_f32 -> Gf. Main-study conditions V5, V6, G (r1, r2)
from primary_v2/samples.csv and its contrasts/pairs_seed.csv (HW(V5,V6), HW(V6,G), RC(G)); HWh(V6,G) on seeds 1 to 5
from follow-up 12's pairs_seed.csv; the trunk-output differences of follow-up 2 from embed_20261015/summary.csv.

Checks (item c), before anything is compared; the analysis stops if 1 or 2 fails:
  1 every process of the why sessions exits 0 or its failure is recorded (collect.py's failure column);
  2 one machine type per condition;
  3 completeness per condition (missing units leave the contrasts they belong to, section 8; reported);
  4 P0 bit-identical to the main study's V6 r1 for every (target, seed) (contrasts.outputs_identical: every output
    file byte for byte, the mmCIF timestamp line masked); reported, P0 is the reference of P either way;
  5 L on V5 and V6 bit-identical to the main study's V5 r1 and V6 r1, seed 1 (the capture does not change the
    numerics); reported; if it fails, L is read as a comparison of the instrumented program;
  6 every capture file present, with the expected recycles and depths, no ordering error, the same token subset on
    every platform;
  7 V6f r2 bit-identical to V6f r1 (pilot targets, seed 1); a difference calls for the extension of item b.
Statistics (item d): section 5's, the target-clustered bootstrap (--replicates 10000, --seed 2026), per-seed top-1.
  P   P<K> against P0 for each level (contrasts.summarize: flip rates, M4 and M5); eps* per primary pair and statistic
      (top-1 headline, success flip secondary) against the main study's HW rate on seeds 1 and 2 (the five-seed rate
      reported next to it): log2-linear interpolation between the first two adjacent levels whose estimates bracket
      the rate, from the smallest level; "below 2^-14" or "above 2^-5" otherwise; interval from replicates that
      resample the targets once for the whole curve and the HW rate, censored replicates counted at the bound.
  L   D(A, B) = norm(A - B) / norm(A), A first-named, over the captured entries, in units of u = 2^-8: V5-V6, V6-G,
      G-Gr2 (recompilation level), V5-G (secondary); per depth, recycle and representation; first depth in recycle 0
      at which V5-V6 and V6-G exceed G-Gr2; growth, the least-squares slope of log2 D against depth in recycle 0.
  F   D_F = [HW(V6f,Gf) - RC(Gf)] - [HW(V6,G) - RC(G)], seeds 1 to 5, on the same resampled targets; "reduced" if
      the 95% interval lies below 0, "not reduced" otherwise (top-1 primary; success flip and flips that matter
      secondary); HW(V6f,Gf) - HWh(V6,G) and RC(Gf) - RC(G), descriptive; F(V6) = V6 against V6f, F(G) = G against
      Gf; per-seed top-1 success rates; cost: ratios of median process wall time and of device memory, per bucket.
Outputs in --out: checks.csv, completeness.csv, p_curve.csv, p_eps_star.csv, p_contrasts.csv, l_depth.csv,
l_summary.csv, l_first_depth.csv, l_growth.csv, f_reading.csv, f_contrasts.csv, f_secondary.csv, f_success.csv,
f_cost.csv, failures.csv, pairs_seed.csv, pairs_sample.csv, report.md. Figures: analysis/fu13_figures.py.
"""
import argparse
import json
import math
import pathlib
import re
import sys

import numpy as np
import pandas as pd

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / 'analysis'))
import contrasts  # noqa: E402
import followup34 as f34  # noqa: E402

U = 2.0 ** -8                                  # bfloat16 unit roundoff (8 significand bits)
LEVELS = [14, 11, 8, 7, 6, 5]                  # P levels, eps = 2^-K, ascending eps
TARGET_PAIRS = ['HW(V5,V6)', 'HW(V6,G)']
EMBED_ROWS = {'HW(V5,V6)': 'V5 r1 - V6 r1', 'HW(V6,G)': 'V6 r1 - G r1'}
L_PAIRS = [('V5-V6', ('V5L', 'r1'), ('V6L', 'r1'), 'primary'), ('V6-G', ('V6L', 'r1'), ('GL', 'r1'), 'primary'),
           ('G-Gr2', ('GL', 'r1'), ('GL', 'r2'), 'recompilation level'), ('V5-G', ('V5L', 'r1'), ('GL', 'r1'), 'secondary')]
BLOCKS, DEPTH_OUT = 48, 48
PLANNED = {**{f'P{k}': {('r1', s): 60 for s in (1, 2)} for k in [0] + LEVELS},
           'V5L': {('r1', 1): 10}, 'V6L': {('r1', 1): 10}, 'GL': {('r1', 1): 10, ('r2', 1): 10},
           'V6f': {**{('r1', s): 60 for s in range(1, 6)}, ('r2', 1): 10},
           'Gf': {(r, s): 60 for r in ('r1', 'r2') for s in range(1, 6)}}


# ---------------------------------------------------------------- conditions and loading

def condition_of(config, device_platform):
  """Follow-up 13 condition code from the configuration name (and the device's platform code for the TPU capture)."""
  m = re.fullmatch(r'tpu_xla_fu13_p(\d+)', str(config))
  if m:
    return f'P{m.group(1)}'
  if config == 'tpu_xla_fu13_capture':
    return {'V5': 'V5L', 'V6': 'V6L'}.get(str(device_platform)[:2], f'{device_platform}L')
  return {'l4_xla_fu13_capture': 'GL', 'tpu_xla_fu13_f32': 'V6f', 'l4_xla_fu13_f32': 'Gf'}.get(config)


def load_why(path):
  d = pd.read_csv(path, low_memory=False)
  d = d[(d.label == 'why') & (d.num_recycles == 10) & (d.num_diffusion_samples == 5)].copy()
  d['condition'] = [condition_of(c, p) for c, p in zip(d.config, d.platform)]
  if d.condition.isna().any():
    raise SystemExit(f'sessions of label why with configurations not of follow-up 13: {sorted(set(d[d.condition.isna()].config))}')
  return d


def processes(d):
  return d.drop_duplicates(['session', 'target', 'seed'])


# ---------------------------------------------------------------- checks

def check_exit(d):
  rows = []
  for c, g in processes(d).groupby('condition'):
    bad = g[g.exit_code != 0]
    unrecorded = bad[bad.failure.isna() | (bad.failure.astype(str).str.strip() == '')]
    rows.append({'check': '1 every process exits 0 or its failure is recorded', 'condition': c,
                 'value': f'{len(g) - len(bad)}/{len(g)} exit 0; {len(bad)} failed with a recorded cause'
                          f'{"" if unrecorded.empty else f", {len(unrecorded)} without"}',
                 'detail': 'pass' if unrecorded.empty else 'FAIL: ' + '; '.join(
                     f'{r.target} {r.repetition} seed {r.seed}' for r in unrecorded.itertuples())[:300]})
  return rows


def check_machine(d):
  rows = []
  for c in sorted(d.condition.unique()):
    try:
      mt = f34.one_machine_type(d, [c])
      rows.append({'check': '2 one machine type per condition', 'condition': c, 'value': mt, 'detail': 'pass'})
    except SystemExit as e:
      rows.append({'check': '2 one machine type per condition', 'condition': c, 'value': str(e)[:200], 'detail': 'FAIL'})
  return rows


def completeness(d):
  ok = f34.sample_rows(d)
  rows = []
  for c, plan in PLANNED.items():
    for (rep, seed), n in sorted(plan.items()):
      units = ok[(ok.condition == c) & (ok.repetition == rep) & (ok.seed == seed)].groupby('target').size()
      have = int((units == 5).sum())
      rows.append({'condition': c, 'repetition': rep, 'seed': seed, 'planned_targets': n, 'targets_with_5_samples': have,
                   'status': 'complete' if have == n else 'incomplete'})
  return pd.DataFrame(rows)


def identity_rows(new, ref, new_cond, ref_cond, rep_new, rep_ref, seeds, check):
  """outputs_identical of every (target, seed) of new_cond against ref_cond; one summary row and the detail rows."""
  detail = []
  for (t, s), g in new[(new.condition == new_cond) & (new.repetition == rep_new) & new.seed.isin(seeds)].groupby(
      ['target', 'seed']):
    r = ref[(ref.condition == ref_cond) & (ref.repetition == rep_ref) & (ref.target == t) & (ref.seed == s)]
    res = contrasts.outputs_identical(contrasts.output_dir(r), contrasts.output_dir(g), s) if len(r) else \
        {'status': 'missing', 'differ': [], 'missing': ['no reference run']}
    detail.append({'check': check, 'target': t, 'seed': s, 'status': res['status'],
                   'files': '; '.join(res['differ'] + res['missing'])[:300]})
  n_id = sum(x['status'] == 'identical' for x in detail)
  summary = {'check': check, 'condition': f'{new_cond} {rep_new} against {ref_cond} {rep_ref}',
             'value': f'{n_id}/{len(detail)} identical', 'detail': 'pass' if detail and n_id == len(detail) else
             ('no unit' if not detail else 'reported: ' + ', '.join(f'{x["target"]} seed {x["seed"]} {x["status"]}'
                                                                      for x in detail if x['status'] != 'identical')[:250])}
  return summary, detail


# ---------------------------------------------------------------- P

def eps_star(rates, target, levels=LEVELS):
  """(log2 eps*, status) for one curve. rates[i] is the estimate at eps = 2^-levels[i], levels in ascending eps."""
  x = [-k for k in levels]
  f = list(rates)
  if f[0] >= target:
    return (float(x[0]), 'at 2^-%d' % levels[0]) if f[0] == target else (float(x[0]), 'below')
  for i in range(len(f) - 1):
    a, b = f[i] - target, f[i + 1] - target
    if a == 0:
      return float(x[i]), 'interpolated'
    if a * b <= 0 and f[i] != f[i + 1]:
      return float(x[i] + (target - f[i]) / (f[i + 1] - f[i]) * (x[i + 1] - x[i])), 'interpolated'
  return float(x[-1]), 'above'


def unit_sums(sp, name, stat, targets):
  g = sp[sp.contrast == name]
  s = g.groupby('target')[stat].apply(lambda v: float(np.sum(v.astype(float))))
  n = g.groupby('target')[stat].size()
  return np.array([s.get(t, 0.0) for t in targets]), np.array([n.get(t, 0) for t in targets], dtype=float)


def eps_star_table(p_sp, hw_sp, stat, seeds_label, rng_seed, reps):
  """eps* of each primary pair: point estimate and percentile interval (replicates resample targets jointly)."""
  names = [f'P(2^-{k})' for k in LEVELS]
  rows = []
  for pair in TARGET_PAIRS:
    hw = hw_sp[hw_sp.contrast == pair]
    targets = sorted(set.intersection(*[set(p_sp[p_sp.contrast == n].target) for n in names]) & set(hw.target))
    if not targets:
      rows.append({'pair': pair, 'statistic': stat, 'hw_seeds': seeds_label, 'n_targets': 0})
      continue
    lv = [unit_sums(p_sp, n, stat, targets) for n in names]
    hs, hn = unit_sums(hw_sp, pair, stat, targets)
    rates = [s.sum() / n.sum() for s, n in lv]
    h = hs.sum() / hn.sum()
    x, status = eps_star(rates, h)
    rng = np.random.default_rng(rng_seed)
    xs, cens = [], 0
    for idx in rng.integers(0, len(targets), size=(reps, len(targets))):
      rr = [s[idx].sum() / n[idx].sum() for s, n in lv]
      xb, st = eps_star(rr, hs[idx].sum() / hn[idx].sum())
      xs.append(xb)
      cens += st in ('below', 'above')
    lo, hi = np.percentile(xs, [2.5, 97.5])
    rows.append({'pair': pair, 'statistic': stat, 'hw_seeds': seeds_label, 'n_targets': len(targets), 'hw_rate': h,
                 'log2_eps_star': x, 'status': status, 'eps_star_u': 2.0 ** x / U, 'ci_low_u': 2.0 ** lo / U,
                 'ci_high_u': 2.0 ** hi / U, 'censored_replicates': cens, 'replicates': reps,
                 **{f'rate_2^-{k}': r for k, r in zip(LEVELS, rates)}})
  return pd.DataFrame(rows)


def eps_sentence(r, measured=None):
  if not r.get('n_targets'):
    return f'{r["pair"]}: not computed (no common targets)'
  where = {'below': f'below {2.0 ** -LEVELS[0] / U:g} u (2^-{LEVELS[0]}), the smallest level',
           'above': f'above {2.0 ** -LEVELS[-1] / U:g} u (2^-{LEVELS[-1]}), the largest level'}.get(
      r['status'], f'about {r["eps_star_u"]:.2g} times the bfloat16 unit roundoff u')
  s = (f'The cross-platform rate {r["pair"]} of {100 * r["hw_rate"]:.1f}% ({r["statistic"]}, seeds {r["hw_seeds"]}) '
       f'corresponds to a random perturbation at the trunk output of relative size {where} (95% interval '
       f'{r["ci_low_u"]:.2g} to {r["ci_high_u"]:.2g} u; {r["censored_replicates"]} of {r["replicates"]} replicates at a bound).')
  if measured:
    s += f' Measured trunk-output difference between those platforms: {measured}.'
  return s


# ---------------------------------------------------------------- L

def bf16_to_f32(a):
  a = np.asarray(a)
  return (a.astype(np.uint32) << 16).view(np.float32) if a.dtype == np.uint16 else a.astype(np.float32)


def rel_l2(a, b):
  a, b = bf16_to_f32(a).astype(np.float64), bf16_to_f32(b).astype(np.float64)
  na = np.linalg.norm(a)
  return float(np.linalg.norm(a - b) / na) if na > 0 else float('nan')


def expected_keys(last=10):
  keys = {f'single_r{r:02d}_d{d:02d}' for r in (0, last) for d in range(BLOCKS)}
  keys |= {f'pair_r{r:02d}_d{d:02d}' for r in (0, last) for d in range(0, BLOCKS, 2)}
  keys |= {f'{k}_r{r:02d}_d{DEPTH_OUT}' for k in ('single', 'pair') for r in range(last + 1)}
  return keys


def capture_path(rows):
  r = rows.iloc[0]
  return REPO / r.session / r.target / f'fu13_capture_seed-{int(r.seed)}.npz'


def load_capture(path):
  z = np.load(path, allow_pickle=False)
  meta = json.loads(str(z['meta']))
  return {k: z[k] for k in z.files if k != 'meta'}, meta


def check_capture(caps):
  """caps: {(condition, rep, target): (arrays, meta) or None}. Rows of check 6."""
  rows = []
  for (c, rep, t), v in sorted(caps.items()):
    if v is None:
      rows.append({'condition': f'{c} {rep}', 'target': t, 'detail': 'FAIL: capture file missing'})
      continue
    arrays, meta = v
    miss = sorted(expected_keys(meta.get('last_recycle', 10)) - set(arrays))
    err = meta.get('errors') or []
    rows.append({'condition': f'{c} {rep}', 'target': t,
                 'detail': 'pass' if not miss and not err else f'FAIL: {len(miss)} arrays missing {miss[:3]}; errors {err[:2]}'})
  by_t = {}
  for (c, rep, t), v in caps.items():
    if v is not None:
      by_t.setdefault(t, []).append(tuple(v[0]['pair_tokens']))
  for t, subsets in sorted(by_t.items()):
    rows.append({'condition': 'all', 'target': t,
                 'detail': 'pass: same pair token subset' if len(set(subsets)) == 1 else 'FAIL: pair token subsets differ'})
  return rows


def l_depth(caps):
  """Rows (target, pair, kind, recycle, depth, rel_l2, rel_l2_u) for every key both sides of a pair hold."""
  rows = []
  targets = sorted({t for (_, _, t) in caps})
  for t in targets:
    for name, a, b, role in L_PAIRS:
      va, vb = caps.get((*a, t)), caps.get((*b, t))
      if va is None or vb is None:
        continue
      for k in sorted(set(va[0]) & set(vb[0])):
        m = re.fullmatch(r'(single|pair)_r(\d\d)_d(\d\d)', k)
        if not m:
          continue
        d = rel_l2(va[0][k], vb[0][k])
        rows.append({'target': t, 'pair': name, 'role': role, 'kind': m.group(1), 'recycle': int(m.group(2)),
                     'depth': int(m.group(3)), 'rel_l2': d, 'rel_l2_u': d / U})
  return pd.DataFrame(rows)


def l_summary(depth):
  if depth.empty:
    return pd.DataFrame()
  g = depth.groupby(['pair', 'kind', 'recycle', 'depth']).rel_l2_u
  return pd.DataFrame({'n_targets': g.size(), 'median_u': g.median(), 'min_u': g.min(), 'max_u': g.max()}).reset_index()


def first_depth(depth):
  """Per target, the first depth of recycle 0 at which V5-V6 and V6-G exceed G-Gr2 at that depth and kind."""
  rows = []
  if depth.empty:
    return pd.DataFrame(rows)
  r0 = depth[depth.recycle == 0]
  for (t, kind), g in r0.groupby(['target', 'kind']):
    base = g[g.pair == 'G-Gr2'].set_index('depth').rel_l2
    for name in ('V5-V6', 'V6-G'):
      p = g[g.pair == name].set_index('depth').rel_l2.sort_index()
      common = [d for d in p.index if d in base.index]
      hit = [d for d in common if p[d] > base[d]]
      rows.append({'target': t, 'pair': name, 'kind': kind, 'first_depth': hit[0] if hit else None,
                   'depths_compared': len(common), 'recompilation_level_zero_at_first': (base[hit[0]] == 0) if hit else None})
  return pd.DataFrame(rows)


def growth(depth):
  """Per target, pair and kind: least-squares slope of log2 D against depth, recycle 0, depths with D > 0."""
  rows = []
  if depth.empty:
    return pd.DataFrame(rows)
  r0 = depth[(depth.recycle == 0) & (depth.rel_l2 > 0)]
  for (t, name, kind), g in r0.groupby(['target', 'pair', 'kind']):
    if len(g) >= 2:
      slope = float(np.polyfit(g.depth.astype(float), np.log2(g.rel_l2.astype(float)), 1)[0])
    else:
      slope = float('nan')
    rows.append({'target': t, 'pair': name, 'kind': kind, 'slope_bits_per_block': slope, 'n_depths': len(g)})
  return pd.DataFrame(rows)


# ---------------------------------------------------------------- F

def d_f(hwf, rcf, hw, rc, stat, rng_seed, reps):
  """D_F = [HWf - RCf] - [HW - RC] on the targets common to the four seed-pair sets; percentile interval."""
  sets = [hwf, rcf, hw, rc]
  targets = sorted(set.intersection(*[set(x.target) for x in sets]))
  if not targets:
    return {'n_targets': 0, 'estimate': np.nan, 'ci_low': np.nan, 'ci_high': np.nan}
  sums = [unit_sums(x.assign(contrast='x'), 'x', stat, targets) for x in sets]

  def value(idx):
    m = [s[idx].sum() / n[idx].sum() for s, n in sums]
    return (m[0] - m[1]) - (m[2] - m[3]), m
  est, parts = value(np.arange(len(targets)))
  rng = np.random.default_rng(rng_seed)
  vals = [value(idx)[0] for idx in rng.integers(0, len(targets), size=(reps, len(targets)))]
  lo, hi = np.percentile(vals, [2.5, 97.5])
  return {'n_targets': len(targets), 'estimate': float(est), 'ci_low': float(lo), 'ci_high': float(hi),
          'hw_f': parts[0], 'rc_f': parts[1], 'hw': parts[2], 'rc': parts[3]}


def f_reading(ci_high):
  if ci_high != ci_high:
    return 'not computed'
  return 'reduced' if ci_high < 0 else 'not reduced'


def diff_two(a, b, stat, rng_seed, reps):
  """mean(a) - mean(b) on common targets with a target-clustered interval (descriptive comparisons)."""
  targets = sorted(set(a.target) & set(b.target))
  if not targets:
    return {'n_targets': 0, 'estimate': np.nan, 'ci_low': np.nan, 'ci_high': np.nan}
  sa = unit_sums(a.assign(contrast='x'), 'x', stat, targets)
  sb = unit_sums(b.assign(contrast='x'), 'x', stat, targets)
  f = lambda idx: sa[0][idx].sum() / sa[1][idx].sum() - sb[0][idx].sum() / sb[1][idx].sum()  # noqa: E731
  rng = np.random.default_rng(rng_seed)
  vals = [f(idx) for idx in rng.integers(0, len(targets), size=(reps, len(targets)))]
  lo, hi = np.percentile(vals, [2.5, 97.5])
  return {'n_targets': len(targets), 'estimate': float(f(np.arange(len(targets)))), 'ci_low': float(lo),
          'ci_high': float(hi)}


def success_rates(ok, conds, rng_seed, reps):
  rows = []
  units = contrasts.seed_units(ok)
  for c, rep in conds:
    per = {}
    for (cc, rr, t, s), g in units.items():
      if cc == c and rr == rep:
        per.setdefault(t, []).append(float(contrasts.top(g)[1]))
    targets = sorted(per)
    vals = np.concatenate([per[t] for t in targets]) if targets else np.array([])
    lo, hi = contrasts.boot_ci([per[t] for t in targets], contrasts.mean, np.random.default_rng(rng_seed), reps)
    rows.append({'condition': c, 'repetition': rep, 'n': len(vals), 'n_targets': len(targets),
                 'top1_success_rate': float(vals.mean()) if len(vals) else np.nan, 'ci_low': lo, 'ci_high': hi})
  return pd.DataFrame(rows)


def cost(proc, pairs):
  """Ratios of median process wall time and device memory, overall and per bucket: pairs [(new, ref)]."""
  rows = []
  for new, ref in pairs:
    for bucket in ['all'] + sorted(proc.bucket.dropna().astype(int).unique()):
      g = proc if bucket == 'all' else proc[proc.bucket == bucket]
      a, b = g[(g.condition == new) & (g.exit_code == 0)], g[(g.condition == ref) & (g.exit_code == 0)]
      row = {'new': new, 'reference': ref, 'bucket': bucket, 'n_new': len(a), 'n_reference': len(b)}
      for col, name in (('wall_s', 'wall'), ('compiled_needed_bytes', 'compiled_memory'),
                        ('alloc_peak_reserved_bytes', 'peak_reserved')):
        ma, mb = a[col].median() if len(a) else np.nan, b[col].median() if len(b) else np.nan
        row[f'{name}_median_new'], row[f'{name}_median_reference'] = ma, mb
        row[f'{name}_ratio'] = ma / mb if mb and mb == mb and ma == ma else np.nan
      rows.append(row)
  return pd.DataFrame(rows)


# ---------------------------------------------------------------- main

def main():
  ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
  ap.add_argument('--samples', required=True)
  ap.add_argument('--main', default=str(REPO / 'results' / 'analysis' / 'main_20261014' / 'primary_v2'))
  ap.add_argument('--prec5', default=str(REPO / 'results' / 'analysis' / 'prec5_20261107'))
  ap.add_argument('--embed', default=str(REPO / 'results' / 'analysis' / 'embed_20261015' / 'summary.csv'))
  ap.add_argument('--out', required=True)
  ap.add_argument('--replicates', type=int, default=10000)
  ap.add_argument('--seed', type=int, default=2026)
  args = ap.parse_args()
  out = pathlib.Path(args.out)
  out.mkdir(parents=True, exist_ok=True)

  d = load_why(args.samples)
  main_s = f34.load_samples(pathlib.Path(args.main) / 'samples.csv', ['main'])
  main_s = main_s[main_s.condition.isin(['V5', 'V6', 'G'])]
  main_sp = pd.read_csv(pathlib.Path(args.main) / 'contrasts' / 'pairs_seed.csv')
  prec_sp = pd.read_csv(pathlib.Path(args.prec5) / 'pairs_seed.csv')

  chk = check_exit(d) + check_machine(d)
  pd.DataFrame(chk).to_csv(out / 'checks.csv', index=False)
  if any(r['detail'] != 'pass' for r in chk):
    raise SystemExit('check 1 or 2 failed (checks.csv); nothing is compared')
  comp = completeness(d)
  ok = f34.sample_rows(d)
  ref = f34.sample_rows(main_s)
  id_rows = []
  for args_ in (('P0', 'V6', 'r1', 'r1', [1, 2], '4 P0 against the main study V6 r1'),
                ('V5L', 'V5', 'r1', 'r1', [1], '5 L on V5 against the main study V5 r1'),
                ('V6L', 'V6', 'r1', 'r1', [1], '5 L on V6 against the main study V6 r1')):
    s, det = identity_rows(ok, ref, args_[0], args_[1], args_[2], args_[3], args_[4], args_[5])
    chk.append(s)
    id_rows += det
  s, det = identity_rows(ok, ok, 'V6f', 'V6f', 'r2', 'r1', [1], '7 V6f r2 against V6f r1')
  chk.append(s)
  id_rows += det

  # L: captures
  caps = {}
  for c, rep in (('V5L', 'r1'), ('V6L', 'r1'), ('GL', 'r1'), ('GL', 'r2')):
    for t, g in ok[(ok.condition == c) & (ok.repetition == rep)].groupby('target'):
      p = capture_path(g)
      caps[(c, rep, t)] = load_capture(p) if p.is_file() else None
  cap_rows = check_capture(caps)
  chk.append({'check': '6 capture files complete, same token subset', 'condition': 'V5L, V6L, GL r1 and r2',
              'value': f'{sum(r["detail"].startswith("pass") for r in cap_rows)}/{len(cap_rows)} pass',
              'detail': 'pass' if cap_rows and all(r['detail'].startswith('pass') for r in cap_rows) else 'reported'})
  pd.DataFrame(chk).to_csv(out / 'checks.csv', index=False)
  pd.DataFrame(id_rows + [{'check': '6 capture', **r} for r in cap_rows]).to_csv(out / 'checks_detail.csv', index=False)
  caps = {k: v for k, v in caps.items() if v is not None}
  depth = l_depth(caps)
  summ, fd, gr = l_summary(depth), first_depth(depth), growth(depth)

  # P
  specs = [(f'P(2^-{k})', 'PT', ('P0', 'r1'), (f'P{k}', 'r1'), False) for k in LEVELS]
  # F
  fu = pd.concat([ok, ref[(ref.condition.isin(['V6', 'G'])) & (ref.repetition == 'r1')]], ignore_index=True)
  specs_f = [('HW(V6f,Gf)', 'HW', ('V6f', 'r1'), ('Gf', 'r1'), False), ('RC(Gf)', 'RC', ('Gf', 'r1'), ('Gf', 'r2'), False),
             ('RC(V6f)', 'RC', ('V6f', 'r1'), ('V6f', 'r2'), False), ('F(V6)', 'F', ('V6', 'r1'), ('V6f', 'r1'), False),
             ('F(G)', 'F', ('G', 'r1'), ('Gf', 'r1'), False)]
  sp_p, sm_p, cons_p, counts_p = f34.build(ok, specs)
  sp_f, sm_f, cons_f, counts_f = f34.build(fu, specs_f)
  sp_all = pd.concat([sp_p, sp_f], ignore_index=True)
  sm_all = pd.concat([sm_p, sm_f], ignore_index=True)
  p_con = contrasts.summarize(sp_p, sm_p, cons_p, args) if len(sp_p) else pd.DataFrame()
  f_con = contrasts.summarize(sp_f, sm_f, cons_f, args) if len(sp_f) else pd.DataFrame()

  hw_main = main_sp[main_sp.contrast.isin(TARGET_PAIRS)]
  emb = pd.read_csv(args.embed).set_index('pair') if pathlib.Path(args.embed).is_file() else pd.DataFrame()
  eps_rows = []
  for stat in ('flip_top1', 'flip_success'):
    for label, hw in (('1 and 2', hw_main[hw_main.seed_a.isin([1, 2])]), ('1 to 5', hw_main)):
      t = eps_star_table(sp_p, hw, stat, label, args.seed, args.replicates) if len(sp_p) else pd.DataFrame()
      eps_rows.append(t)
  eps_t = pd.concat(eps_rows, ignore_index=True) if eps_rows else pd.DataFrame()
  curve = p_con[p_con.statistic.isin(contrasts.FLIP_STATS)].copy() if len(p_con) else pd.DataFrame()
  if len(curve):
    curve['level'] = curve.contrast.str.extract(r'2\^-(\d+)').astype(int)
    curve['eps'] = 2.0 ** -curve.level
    curve['eps_u'] = curve.eps / U

  # F reading
  f_rows, f_sec = [], []
  hw_ref = main_sp[main_sp.contrast == 'HW(V6,G)']
  rc_ref = main_sp[main_sp.contrast == 'RC(G)']
  hwh = prec_sp[prec_sp.contrast == 'HWh(V6,G)']
  for i, stat in enumerate(contrasts.FLIP_STATS):
    r = d_f(sp_f[sp_f.contrast == 'HW(V6f,Gf)'], sp_f[sp_f.contrast == 'RC(Gf)'], hw_ref, rc_ref, stat, args.seed,
            args.replicates)
    f_rows.append({'statistic': stat, 'role': 'primary' if i == 0 else 'secondary', **r, 'reading': f_reading(r['ci_high'])})
    f_sec.append({'comparison': 'HW(V6f,Gf) minus HWh(V6,G)', 'statistic': stat,
                  **diff_two(sp_f[sp_f.contrast == 'HW(V6f,Gf)'], hwh, stat, args.seed, args.replicates)})
    f_sec.append({'comparison': 'RC(Gf) minus RC(G)', 'statistic': stat,
                  **diff_two(sp_f[sp_f.contrast == 'RC(Gf)'], rc_ref, stat, args.seed, args.replicates)})
  succ = success_rates(fu, [('V6f', 'r1'), ('Gf', 'r1'), ('V6', 'r1'), ('G', 'r1')], args.seed, args.replicates)
  proc = pd.concat([processes(d), processes(main_s[(main_s.condition.isin(['V6', 'G'])) & (main_s.repetition == 'r1')])],
                   ignore_index=True)
  cst = cost(proc, [('V6f', 'V6'), ('Gf', 'G')])
  fails = processes(d)[processes(d).exit_code != 0][['condition', 'repetition', 'target', 'seed', 'session', 'exit_code',
                                                     'failure']]

  outputs = {'completeness': comp, 'p_curve': curve, 'p_eps_star': eps_t, 'p_contrasts': p_con, 'l_depth': depth,
             'l_summary': summ, 'l_first_depth': fd, 'l_growth': gr, 'f_reading': pd.DataFrame(f_rows),
             'f_contrasts': f_con, 'f_secondary': pd.DataFrame(f_sec), 'f_success': succ, 'f_cost': cst,
             'failures': fails, 'pairs_seed': sp_all, 'pairs_sample': sm_all,
             'counts': pd.concat([counts_p, counts_f], ignore_index=True)}
  for name, df in outputs.items():
    df.to_csv(out / f'{name}.csv', index=False)
  (out / 'report.md').write_text(report(chk, comp, curve, eps_t, emb, summ, fd, gr, depth, f_rows, f_sec, succ, cst,
                                        fails))
  print((out / 'report.md').read_text())
  return 0


def measured(emb, depth, pair):
  parts = []
  row = EMBED_ROWS[pair]
  if len(emb) and row in emb.index:
    parts.append(f'follow-up 2, median over the pilot targets: single {emb.loc[row, "single_rel_l2_median"] / U:.2g} u, '
                 f'pair {emb.loc[row, "pair_rel_l2_median"] / U:.2g} u')
  lp = {'HW(V5,V6)': 'V5-V6', 'HW(V6,G)': 'V6-G'}[pair]
  if len(depth):
    g = depth[(depth.pair == lp) & (depth.recycle == 10) & (depth.depth == DEPTH_OUT)]
    if len(g):
      parts.append('L, trunk output of the last recycle: ' + ', '.join(
          f'{k} {g[g.kind == k].rel_l2_u.median():.2g} u' for k in ('single', 'pair') if (g.kind == k).any()))
  return '; '.join(parts)


def report(chk, comp, curve, eps_t, emb, summ, fd, gr, depth, f_rows, f_sec, succ, cst, fails):
  pct, num = f34.pct, f34.num
  L = ['# Follow-up 13: why the top-ranked sample flips', '',
       'notes/analysis_plan.md section 10, row of 2026-11-10; reported whatever it shows. u = 2^-8.', '',
       '## Checks', '', '| check | condition | value | result |', '|---|---|---|---|']
  L += [f'| {r["check"]} | {r["condition"]} | {r["value"]} | {r["detail"]} |' for r in chk]
  inc = comp[comp.status != 'complete']
  L += ['', f'Completeness: {len(comp) - len(inc)} of {len(comp)} planned (condition, repetition, seed) cells complete'
        + ('' if inc.empty else '; incomplete: ' + ', '.join(f'{r.condition} {r.repetition} seed {r.seed} '
                                                             f'({r.targets_with_5_samples}/{r.planned_targets})'
                                                             for r in inc.itertuples())) + '.']
  if len(fails):
    L += ['', 'Failed processes (recorded, not rerun): ' + '; '.join(
        f'{r.condition} {r.repetition} {r.target} seed {r.seed} ({r.failure})' for r in fails.itertuples()) + '.']
  L += ['', '## P: perturbation of the trunk output (descriptive)', '']
  if len(eps_t):
    for r in eps_t[eps_t.n_targets > 0].to_dict('records'):
      L.append('- ' + eps_sentence(r, measured(emb, depth, r['pair']) if r['statistic'] == 'flip_top1' else None))
  L += ['', '| level | eps (u) | statistic | estimate | 95% CI | targets |', '|---|---|---|---|---|---|']
  if len(curve):
    L += [f'| {r.contrast} | {r.eps_u:g} | {r.statistic} | {pct(r.estimate)} | {pct(r.ci_low)} to {pct(r.ci_high)} | '
          f'{r.n_targets} |' for r in curve.sort_values(['statistic', 'eps']).itertuples()]
  L += ['', '## L: layer-wise divergence (descriptive)', '']
  if len(summ):
    for kind in ('single', 'pair'):
      for name in ('V5-V6', 'V6-G', 'G-Gr2'):
        g = summ[(summ.kind == kind) & (summ.pair == name) & (summ.recycle == 0)]
        if len(g):
          L.append(f'- {name}, {kind}, recycle 0: median D at depth 0 {num(g[g.depth == g.depth.min()].median_u.iloc[0])} u, '
                   f'at depth {DEPTH_OUT} {num(g[g.depth == DEPTH_OUT].median_u.iloc[0]) if (g.depth == DEPTH_OUT).any() else "-"} u')
    if len(fd):
      for (name, kind), g in fd.groupby(['pair', 'kind']):
        v = g.first_depth.dropna()
        L.append(f'- first depth of recycle 0 where {name} exceeds G-Gr2 ({kind}): median '
                 f'{num(v.median()) if len(v) else "never"}; per target ' + ', '.join(
                     f'{r.target} {int(r.first_depth) if r.first_depth == r.first_depth and r.first_depth is not None else "never"}'
                     for r in g.itertuples()))
    if len(gr):
      for (name, kind), g in gr.groupby(['pair', 'kind']):
        L.append(f'- growth, {name}, {kind}: median {num(g.slope_bits_per_block.median())} bits per block')
    last = summ[(summ.depth == DEPTH_OUT)].sort_values(['pair', 'kind', 'recycle'])
    L += ['', '| pair | kind | recycle | median D at the trunk output (u) | range |', '|---|---|---|---|---|']
    L += [f'| {r.pair} | {r.kind} | {r.recycle} | {num(r.median_u)} | {num(r.min_u)} to {num(r.max_u)} |'
          for r in last.itertuples()]
  L += ['', '## F: float32 trunk', '', '| statistic | role | D_F | 95% CI | targets | reading |', '|---|---|---|---|---|---|']
  L += [f'| {r["statistic"]} | {r["role"]} | {pct(r["estimate"])} | {pct(r["ci_low"])} to {pct(r["ci_high"])} | '
        f'{r["n_targets"]} | {r["reading"]} |' for r in f_rows]
  L += ['', '| comparison (descriptive) | statistic | estimate | 95% CI |', '|---|---|---|---|']
  L += [f'| {r["comparison"]} | {r["statistic"]} | {pct(r["estimate"])} | {pct(r["ci_low"])} to {pct(r["ci_high"])} |'
        for r in f_sec]
  L += ['', '| condition | top-1 success (per seed) | 95% CI | n |', '|---|---|---|---|']
  L += [f'| {r.condition} {r.repetition} | {pct(r.top1_success_rate)} | {pct(r.ci_low)} to {pct(r.ci_high)} | {r.n} |'
        for r in succ.itertuples()]
  L += ['', '| new / reference | bucket | wall ratio | compiled memory ratio | peak reserved ratio | n |', '|---|---|---|---|---|---|']
  L += [f'| {r.new} / {r.reference} | {r.bucket} | {num(r.wall_ratio)} | {num(r.compiled_memory_ratio)} | '
        f'{num(r.peak_reserved_ratio)} | {r.n_new}, {r.n_reference} |' for r in cst.itertuples()]
  return '\n'.join(L) + '\n'


if __name__ == '__main__':
  sys.exit(main())
