"""Follow-up experiment 1, protein-protein complexes (notes/analysis_plan.md section 10, rows of 2026-10-14;
label ppi; reported as a follow-up outside the pre-registered study, whatever it shows).

    analysis/.venv/bin/python analysis/collect.py --labels ppi --no_m1 --out results/analysis/ppi_<date>
    analysis/ppi/.venv/bin/python analysis/score_dockq.py --collect --out results/analysis/ppi_<date>
    analysis/.venv/bin/python analysis/ppi.py --dir results/analysis/ppi_<date> [--replicates 10000] [--seed 2026]

Inputs in --dir: samples.csv (collect.py) and dockq.csv (score_dockq.py, DockQ 2.1.3). Success of a prediction:
DockQ >= 0.23 against the deposited first biological assembly (primary), DockQ >= 0.49 (secondary); a model that
DockQ cannot score is not a success. The success column the main study's code reads (m1_success) is set to it.

Contrasts (the row): HW(V5,V6), HW(V6,G), RC(V6), RC(G), paired exactly as analysis/contrasts.build_pairs pairs
them in the main study (A the first-named; contrasts.pair_rows), with the main study's statistics
(contrasts.summarize: flip rates of top-1, of top-1 success and of flips that matter, M4 and M5 median and 90th
percentile, the fraction of bit-identical sample pairs; 95% percentile intervals from a paired bootstrap over
targets) and HW minus RC (contrasts.hw_minus_rc: RC(P), and RC(Q) labelled secondary). V5 runs r1 only (the row),
so HW(V5,V6) has no RC(P); only HW(V5,V6) - RC(V6), as the main study's secondary RC(Q), exists.
  M2  top-1 identity per seed (flip_top1 of the contrasts) and overall: per target, the overall top-1 across the
      five seeds (the (seed, sample) with the highest full-precision ranking_score, as AlphaFold3 ranks them) on
      A against B, its flip rate and the flip rate of its success, bootstrap over targets;
  M3  margin (top-1 minus top-2 ranking_score within the seed), per condition and repetition: median, quartiles;
  M4  C-alpha RMSD after global superposition of the complex (both chains, C-alpha atoms of residues present in
      both models, by chain and label_seq_id; no ligand, so no ligand RMSD);
  M5  absolute ranking_score, mean pLDDT and ipTM differences of the same sample.
Also: success rate per condition and repetition (overall top-1 and top-1 per seed, both thresholds); the
statistics above with the secondary threshold; per-target DockQ; the exploratory analyses (a) (flip rate against
the M3 margin) and (c) (perturbation size and the two-sample model) of the flip row (analysis/exploratory.py),
labelled exploratory; and the checks: exit codes, 2a (every fresh process compiled from an empty cache, which RC
needs), the M2 cross-check (contrasts.top_model_check) and DockQ errors.
Outputs in --dir: samples_dockq.csv, checks.csv, success_rates.csv, per_target.csv, contrasts.csv,
contrasts_049.csv, hw_minus_rc.csv, hw_minus_rc_049.csv, overall_top1.csv, m3.csv, pairs_seed.csv,
pairs_sample.csv, exploratory_a_quintiles.csv, exploratory_a_logistic.csv, exploratory_c.csv, report.md.
"""
import argparse
import functools
import pathlib
import sys

import numpy as np
import pandas as pd

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / 'analysis'))
sys.path.insert(0, str(REPO / 'harness'))

import contrasts  # noqa: E402
import exploratory  # noqa: E402

CONTRASTS = [('HW(V5,V6)', 'HW', ('V5', 'r1'), ('V6', 'r1')), ('HW(V6,G)', 'HW', ('V6', 'r1'), ('G', 'r1')),
             ('RC(V6)', 'RC', ('V6', 'r1'), ('V6', 'r2')), ('RC(G)', 'RC', ('G', 'r1'), ('G', 'r2'))]
NAMES = [c[0] for c in CONTRASTS]
SHALLOW = ['7T6A', '7MWR', '8FJ3', '8C26', '8EHE']   # section 10 implementation row: at most 9 paired rows
SUCCESS, SECONDARY = 0.23, 0.49


# ---------------------------------------------------------------- M4 for complexes

def complex_ca(path):
  """{(chain, label_seq_id): C-alpha position} over every polymer chain of a model."""
  import gemmi
  st = gemmi.read_structure(str(path))
  st.setup_entities()
  out = {}
  for ch in st[0]:
    for res in ch.get_polymer():
      ca = res.find_atom('CA', '*')
      if ca and res.label_seq is not None:
        out[(ch.name, res.label_seq)] = np.array(ca.pos.tolist())
  return out


@functools.lru_cache(maxsize=None)
def ca_divergence(target, path_a, path_b):
  """M4 between two predictions of one complex: C-alpha RMSD after global superposition of B on A."""
  from score_ligand import kabsch, rmsd
  try:
    a, b = complex_ca(REPO / path_a), complex_ca(REPO / path_b)
    common = sorted(set(a) & set(b))
    if len(common) < 3:
      raise ValueError(f'{len(common)} common C-alpha atoms')
    xa, xb = np.array([a[k] for k in common]), np.array([b[k] for k in common])
    r, t = kabsch(xb, xa)
    return {'ligand_rmsd': None, 'ca_rmsd_global': round(rmsd(xb @ r.T + t, xa), 4), 'error': None}
  except Exception as e:  # noqa: BLE001 - recorded, as in the main study
    return {'ligand_rmsd': None, 'ca_rmsd_global': None, 'error': f'{type(e).__name__}: {e}'[:300]}


# ---------------------------------------------------------------- data

def load(d):
  s = pd.read_csv(d / 'samples.csv', low_memory=False)
  s = s[(s.label == 'ppi') & (s.num_recycles == 10) & (s.num_diffusion_samples == 5)].copy()
  q = pd.read_csv(d / 'dockq.csv')
  q = q.rename(columns={'model': 'model_path', 'success': 'dockq_success', 'success_049': 'dockq_success_049',
                        'error': 'dockq_error'})
  keep = ['model_path', 'DockQ', 'iRMSD', 'LRMSD', 'fnat', 'dockq_success', 'dockq_success_049', 'dockq_error',
          'dockq_version']
  m = s.merge(q[keep], on='model_path', how='left', validate='many_to_one')
  m['condition'] = m.platform
  ok = m[m['sample'].notna()].copy()
  ok['sample'] = ok['sample'].astype(int)
  dup = ok.duplicated(['condition', 'target', 'seed', 'repetition', 'sample'], keep=False)
  if dup.any():
    raise SystemExit('duplicate (condition, target, seed, repetition, sample) rows')
  return m, ok


def with_success(ok, column):
  o = ok.copy()
  o['m1_success'] = o[column].map(lambda v: v is True or v == 'True' or v == 1)
  return o


# ---------------------------------------------------------------- contrasts

def pairs(ok):
  units = contrasts.seed_units(ok)
  seed_rows, sample_rows, cons = [], [], {}
  for name, kind, a, b in CONTRASTS:
    ka = {(k[2], k[3]): k for k in units if (k[0], k[1]) == a}
    kb = {(k[2], k[3]): k for k in units if (k[0], k[1]) == b}
    for ts in sorted(set(ka) & set(kb)):
      contrasts.pair_rows(name, kind, ka[ts], kb[ts], units, True, cons, seed_rows, sample_rows)
  return pd.DataFrame(seed_rows), pd.DataFrame(sample_rows), cons


def overall_top1(ok, args):
  """Per target, the overall top-1 across seeds on A and on B; flip rate of its identity and of its success."""
  rows, stats = [], []
  for name, _, a, b in CONTRASTS:
    for t in sorted(ok.target.unique()):
      ga = ok[(ok.condition == a[0]) & (ok.repetition == a[1]) & (ok.target == t)]
      gb = ok[(ok.condition == b[0]) & (ok.repetition == b[1]) & (ok.target == t)]
      if ga.empty or gb.empty:
        continue
      ta = ga.sort_values('ranking_score', ascending=False, kind='mergesort').iloc[0]
      tb = gb.sort_values('ranking_score', ascending=False, kind='mergesort').iloc[0]
      rows.append({'contrast': name, 'target': t, 'top1_a': f'{int(ta.seed)}/{int(ta["sample"])}',
                   'top1_b': f'{int(tb.seed)}/{int(tb["sample"])}', 'dockq_a': ta.DockQ, 'dockq_b': tb.DockQ,
                   'success_a': bool(ta.m1_success), 'success_b': bool(tb.m1_success),
                   'flip_top1': (ta.seed, ta['sample']) != (tb.seed, tb['sample']),
                   'flip_success': bool(ta.m1_success) != bool(tb.m1_success)})
  df = pd.DataFrame(rows)
  for name in NAMES:
    g = df[df.contrast == name]
    for st in ('flip_top1', 'flip_success'):
      lo, hi = contrasts.boot_ci([[float(v)] for v in g[st]], contrasts.mean, np.random.default_rng(args.seed),
                                 args.replicates)
      stats.append({'contrast': name, 'statistic': f'overall_{st}', 'n_targets': len(g),
                    'estimate': float(g[st].mean()), 'ci_low': lo, 'ci_high': hi})
  return df, pd.DataFrame(stats)


def success_rates(ok, args):
  rows = []
  for (c, rep), g in ok.groupby(['condition', 'repetition']):
    for col, thr in (('dockq_success', SUCCESS), ('dockq_success_049', SECONDARY)):
      o = with_success(g, col)
      best = o.sort_values('ranking_score', ascending=False, kind='mergesort')
      overall = best.groupby('target').head(1).set_index('target').m1_success.astype(float)
      per_seed = best.groupby(['target', 'seed']).head(1)
      for stat, groups, est in (
          ('success_rate_overall_top1', [[v] for v in overall.values], overall.mean()),
          ('success_rate_top1_per_seed', [h.m1_success.astype(float).values for _, h in per_seed.groupby('target')],
           per_seed.m1_success.astype(float).mean())):
        lo, hi = contrasts.boot_ci(groups, contrasts.mean, np.random.default_rng(args.seed), args.replicates)
        rows.append({'condition': c, 'repetition': rep, 'threshold': thr, 'statistic': stat,
                     'n_targets': g.target.nunique(), 'estimate': float(est), 'ci_low': lo, 'ci_high': hi})
  order = {c: i for i, c in enumerate(['V5', 'V6', 'G'])}
  return pd.DataFrame(rows).sort_values(['threshold', 'statistic', 'condition', 'repetition'],
                                        key=lambda s: s.map(order) if s.name == 'condition' else s)


def m3(ok):
  rows = []
  for (c, rep), g in ok.groupby(['condition', 'repetition']):
    margins = [contrasts.top(u)[2] for _, u in g.groupby(['target', 'seed'])]
    m = np.array(margins, dtype=float)
    rows.append({'condition': c, 'repetition': rep, 'seed_units': len(m), 'median': float(np.median(m)),
                 'q1': float(np.percentile(m, 25)), 'q3': float(np.percentile(m, 75)), 'min': float(m.min()),
                 'zero_margins': int((m == 0).sum())})
  return pd.DataFrame(rows)


def per_target(ok, seed_pairs, sample_pairs, manifest):
  rows = []
  for t in sorted(ok.target.unique(), key=lambda t: int(manifest[t]['draw_order'])):
    mf = manifest[t]
    row = {'target': t, 'draw_order': int(mf['draw_order']), 'tokens': int(mf['num_tokens']),
           'bucket': int(mf['bucket']), 'unpaired_rows': mf['msa_rows'], 'paired_rows': int(mf['paired_rows']),
           'shallow': t in SHALLOW}
    for c, rep in (('V5', 'r1'), ('V6', 'r1'), ('V6', 'r2'), ('G', 'r1'), ('G', 'r2')):
      g = ok[(ok.condition == c) & (ok.repetition == rep) & (ok.target == t)]
      if g.empty:
        continue
      top = g.sort_values('ranking_score', ascending=False, kind='mergesort').iloc[0]
      row[f'{c}{rep}_top1_dockq'] = round(float(top.DockQ), 3) if pd.notna(top.DockQ) else np.nan
      row[f'{c}{rep}_median_dockq'] = round(float(g.DockQ.median()), 3)
      row[f'{c}{rep}_max_dockq'] = round(float(g.DockQ.max()), 3)
    for name in NAMES:
      sp, sm = seed_pairs[(seed_pairs.contrast == name) & (seed_pairs.target == t)], \
          sample_pairs[(sample_pairs.contrast == name) & (sample_pairs.target == t)]
      row[f'{name}_seed_flips'] = int(sp.flip_top1.sum())
      row[f'{name}_success_flips'] = int(sp.flip_success.sum())
      row[f'{name}_m4_median'] = float(sm.m4_ca_rmsd_global.median()) if len(sm) else np.nan
    rows.append(row)
  return pd.DataFrame(rows)


def checks(m, ok):
  rows = []
  for c, g in m.groupby('condition'):
    runs = g.drop_duplicates(['session', 'target'])
    bad = runs[runs.exit_code != 0]
    rows.append({'check': '1 exit codes', 'condition': c, 'value': f'{len(runs) - len(bad)}/{len(runs)} processes exit 0',
                 'detail': 'pass' if bad.empty else 'FAIL'})
    fresh = runs[runs.cache == 'fresh']
    not_fresh = fresh[(fresh.cache_files_before != 0) | (fresh.model_cache_hit == True) | ~(fresh.compile_s > 0)]  # noqa: E712
    rows.append({'check': '2a every fresh process compiled from an empty cache', 'condition': c,
                 'value': f'{len(fresh) - len(not_fresh)}/{len(fresh)}',
                 'detail': 'pass' if not_fresh.empty else 'FAIL: ' + '; '.join(not_fresh.target.head(10))})
  rows += contrasts.top_model_check(ok)
  miss = ok[ok.DockQ.isna()]
  rows.append({'check': 'DockQ for every sample', 'condition': 'all',
               'value': f'{len(ok) - len(miss)}/{len(ok)} samples scored; {ok.dockq_error.notna().sum()} errors',
               'detail': 'pass' if miss.empty else 'FAIL: ' + '; '.join(sorted(set(miss.dockq_error.dropna())))[:300]})
  versions = sorted(ok.dockq_version.dropna().unique())
  rows.append({'check': 'DockQ version', 'condition': 'all', 'value': ', '.join(versions),
               'detail': 'pass' if versions == ['2.1.3'] else 'FAIL'})
  return pd.DataFrame(rows)


def exploratory_ac(seed_pairs, sample_pairs, args):
  exploratory.CONTRASTS = NAMES
  qa, la = exploratory.margin_analysis(seed_pairs, args)
  cp = exploratory.perturbation(seed_pairs, sample_pairs, args)
  return qa, la, cp


# ---------------------------------------------------------------- report

def f3(x):
  return '-' if x is None or (isinstance(x, float) and np.isnan(x)) else f'{x:.3g}'


def pct(x):
  return '-' if x is None or (isinstance(x, float) and np.isnan(x)) else f'{100 * x:.1f}%'


def report(d, chk, sr, con, con49, hwrc, hwrc49, ov, m3df, pt, qa, la, cp, args):
  L = ['# Follow-up 1: protein-protein complexes', '',
       'notes/analysis_plan.md section 10, rows of 2026-10-14; follow-up outside the pre-registered study. '
       f'Code: analysis/ppi.py (bootstrap {args.replicates} replicates, seed {args.seed}). Success: DockQ >= 0.23 '
       '(DockQ 2.1.3, deposited assembly 1); secondary >= 0.49.', '', '## Checks', '',
       '| check | condition | value | detail |', '|---|---|---|---|']
  L += [f'| {r.check} | {r.condition} | {r.value} | {r.detail} |' for r in chk.itertuples()]
  L += ['', '## Success rate (r1 and r2 separately)', '', '| threshold | statistic | condition | rep | targets | rate | 95% CI |',
        '|---|---|---|---|---|---|---|']
  L += [f'| {r.threshold} | {r.statistic} | {r.condition} | {r.repetition} | {r.n_targets} | {pct(r.estimate)} | '
        f'{pct(r.ci_low)} to {pct(r.ci_high)} |' for r in sr.itertuples()]

  def table(c, title):
    out = ['', title, '', '| contrast | role | statistic | n | targets | estimate | 95% CI |', '|---|---|---|---|---|---|---|']
    for r in c.itertuples():
      p = r.statistic.startswith('flip') or r.statistic == 'bit_identical_fraction'
      fm = pct if p else f3
      out.append(f'| {r.contrast} | {r.role} | {r.statistic} | {r.n} | {r.n_targets} | {fm(r.estimate)} | '
                 f'{fm(r.ci_low)} to {fm(r.ci_high)} |')
    return out
  L += table(con, '## Contrasts (DockQ >= 0.23)')
  L += table(con49[con49.statistic.isin(['flip_success', 'flip_matters'])], '## Contrasts, secondary threshold (DockQ >= 0.49): success flips')
  L += ['', '## M2 overall top-1 across seeds (per target)', '', '| contrast | statistic | targets | estimate | 95% CI |',
        '|---|---|---|---|---|']
  L += [f'| {r.contrast} | {r.statistic} | {r.n_targets} | {pct(r.estimate)} | {pct(r.ci_low)} to {pct(r.ci_high)} |'
        for r in ov.itertuples()]
  for h, title in ((hwrc, '## HW minus RC (DockQ >= 0.23)'), (hwrc49, '## HW minus RC (DockQ >= 0.49, secondary)')):
    L += ['', title, '', '| hardware | recompilation | reference | statistic | targets | estimate | 95% CI |',
          '|---|---|---|---|---|---|---|']
    L += [f'| {r.hardware} | {r.recompilation} | {r.reference} | {r.statistic} | {r.n_targets} | {pct(r.estimate)} | '
          f'{pct(r.ci_low)} to {pct(r.ci_high)} |' for r in h.itertuples()]
  L += ['', 'HW(V5,V6) has no RC(P): V5 runs r1 only (the row); HW(V5,V6) - RC(V6) is the main study\'s secondary RC(Q).']
  L += ['', '## M3 margin (top-1 minus top-2 ranking_score within a seed)', '',
        '| condition | rep | seed units | median | Q1 | Q3 | min | zero margins |', '|---|---|---|---|---|---|---|---|']
  L += [f'| {r.condition} | {r.repetition} | {r.seed_units} | {f3(r.median)} | {f3(r.q1)} | {f3(r.q3)} | {f3(r.min)} | '
        f'{r.zero_margins} |' for r in m3df.itertuples()]
  cols = ['V5r1', 'V6r1', 'V6r2', 'Gr1', 'Gr2']
  L += ['', '## Per target (DockQ of the overall top-1; seed flips of top-1 per contrast, of 5)', '',
        '| target | tokens | paired rows | shallow | ' + ' | '.join(cols) + ' | flips HW(V5,V6) | HW(V6,G) | RC(V6) | RC(G) | '
        'success flips HW(V5,V6) | HW(V6,G) | RC(V6) | RC(G) |', '|' + '---|' * (4 + len(cols) + 8)]
  for r in pt.to_dict('records'):
    L.append(f'| {r["target"]} | {r["tokens"]} | {r["paired_rows"]} | {"yes" if r["shallow"] else ""} | '
             + ' | '.join(f3(r.get(f'{c}_top1_dockq')) for c in cols) + ' | '
             + ' | '.join(str(r[f'{n}_seed_flips']) for n in NAMES) + ' | '
             + ' | '.join(str(r[f'{n}_success_flips']) for n in NAMES) + ' |')
  L += ['', '## Exploratory (a): top-1 flip rate by quintile of the M3 margin of A', '',
        '| contrast | quintile | margin range | n | flips | rate | 95% CI |', '|---|---|---|---|---|---|---|']
  L += [f'| {r.contrast} | {r.quintile} | {f3(r.margin_low)} to {f3(r.margin_high)} | {r.n} | {r.flips} | '
        f'{pct(r.flip_rate)} | {pct(r.ci_low)} to {pct(r.ci_high)} |' for r in qa.itertuples()]
  L += ['', '| contrast | n | flips | odds ratio per tenfold margin | 95% CI |', '|---|---|---|---|---|']
  L += [f'| {r.contrast} | {r.n} | {r.flips} | {f3(r.odds_ratio_per_tenfold_margin)} | {f3(r.odds_ratio_ci_low)} to '
        f'{f3(r.odds_ratio_ci_high)} |' for r in la.itertuples()]
  L += ['', '## Exploratory (c): perturbation size and the two-sample model', '',
        '| contrast | delta (median abs score diff) | mu (median margin) | delta/mu | predicted flip rate | observed | 95% CI observed |',
        '|---|---|---|---|---|---|---|']
  L += [f'| {r.contrast} | {f3(r.median_abs_score_difference)} | {f3(r.median_margin)} | {f3(r.delta_over_mu)} | '
        f'{pct(r.predicted_flip_rate)} ({pct(r.predicted_ci_low)} to {pct(r.predicted_ci_high)}) | {pct(r.observed_flip_rate)} | '
        f'{pct(r.observed_ci_low)} to {pct(r.observed_ci_high)} |' for r in cp.itertuples()]
  (d / 'report.md').write_text('\n'.join(L) + '\n')


def main():
  ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
  ap.add_argument('--dir', required=True)
  ap.add_argument('--replicates', type=int, default=10000)
  ap.add_argument('--seed', type=int, default=2026)
  args = ap.parse_args()
  d = REPO / args.dir if not pathlib.Path(args.dir).is_absolute() else pathlib.Path(args.dir)
  import csv
  manifest = {r['pdb_id']: r for r in csv.DictReader(open(REPO / 'inputs' / 'ppi' / 'manifest.csv'))}
  contrasts.divergence = ca_divergence   # M4 for complexes: no ligand
  m, ok = load(d)
  m.to_csv(d / 'samples_dockq.csv', index=False)
  chk = checks(m, ok)
  sr = success_rates(ok, args)
  o23, o49 = with_success(ok, 'dockq_success'), with_success(ok, 'dockq_success_049')
  sp, sm, cons = pairs(o23)
  sp49, sm49, _ = pairs(o49)
  keep = lambda c: c[~c.statistic.str.startswith('m4_ligand')]  # noqa: E731
  con = keep(contrasts.summarize(sp, sm, cons, args))
  con49 = keep(contrasts.summarize(sp49, sm49, cons, args))
  hwrc, hwrc49 = contrasts.hw_minus_rc(sp, cons, args), contrasts.hw_minus_rc(sp49, cons, args)
  ovp, ov = overall_top1(o23, args)
  ovp49, ov49 = overall_top1(o49, args)
  ov = pd.concat([ov, ov49[ov49.statistic == 'overall_flip_success'].assign(statistic='overall_flip_success_049')])
  m3df = m3(ok)
  pt = per_target(o23, sp, sm, manifest)
  qa, la, cp = exploratory_ac(sp, sm, args)
  for name, df in (('checks', chk), ('success_rates', sr), ('contrasts', con), ('contrasts_049', con49),
                   ('hw_minus_rc', hwrc), ('hw_minus_rc_049', hwrc49), ('overall_top1', ov),
                   ('overall_top1_pairs', ovp), ('m3', m3df), ('per_target', pt), ('pairs_seed', sp),
                   ('pairs_sample', sm), ('exploratory_a_quintiles', qa), ('exploratory_a_logistic', la),
                   ('exploratory_c', cp)):
    df.to_csv(d / f'{name}.csv', index=False)
  report(d, chk, sr, con, con49, hwrc, hwrc49, ov, m3df, pt, qa, la, cp, args)
  print((d / 'report.md').read_text())
  return 0


if __name__ == '__main__':
  sys.exit(main())
