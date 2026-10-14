"""M1 for three glycan ligands (notes/analysis_plan.md section 10, row of 2026-10-14): analyses (2) and (3).

    analysis/.venv/bin/python analysis/glycans.py --main results/analysis/main_20261014 [--jobs 8] \\
        [--replicates 10000] [--seed 2026]

For 7OZC (G6S), 7WDT (NGS) and 7WJB (BGC) every prediction lacks O1, which AlphaFold3 removes from ligands in its
glycan sets, so M1 is not computed for them and they count as not a success. Reported side by side:
  (1) primary: section 4 as written, read from <main>/contrasts (not recomputed here);
  (2) section 8: every result that uses M1 on the 57 other targets;
  (3) post-hoc sensitivity: the three targets scored on the heavy atoms AlphaFold3 predicts
      (analysis/score_ligand.py drop_af3_removed: the reference ligand without the atoms AlphaFold3 removed, here O1;
      same pocket, pocket C-alpha superposition, symmetry correction and PoseBusters dock checks), and every result
      that uses M1 on all 60 targets.
The results that use M1: the success rates (overall top-1 and top-1 per seed per condition, and Q3's for n = 1..5),
check 3 (and whether M1 was computed for every sample), Q3, success flips and flips that matter (with HW minus RC),
M4's ligand RMSD, and the training-cutoff sensitivity. All are computed by the functions of the primary analysis
(analysis/contrasts.py, analysis/cutoff.py) with the same bootstrap.

M4 in (2) and (3): pairs of the 57 targets take M4 from <main>/contrasts/pairs_sample.csv, since the rescoring does
not change a ligand without a removed atom; in (3) every SPOT_EVERY-th of those pairs is also recomputed with
drop_af3_removed and compared (m4_spot_check.csv). Pairs of the three targets are recomputed in (3).
Also checked: the rescoring leaves M1 of one sample per target of the 57 unchanged (nochange_check.csv), and (3)
leaves top-1, M3, M5, the bit-identity fractions, and M6 and M7 (cost.py reads no M1 column; its tables recomputed
from the rescored table equal the primary ones) unchanged: unchanged.csv.
M4's protein C-alpha RMSD does not use M1, but in (1), the output of <main>/contrasts, score_ligand.pair_divergence
stopped at the missing O1 before computing it (corrected afterwards: section 10, 2026-10-14, "Correction to the
implementation of M4"; rerun in <main>/primary_v2), so it is missing there for the non-identical sample pairs of the
three targets; (3) computes it for them. It is checked equal on every pair where (1) has it, and its statistics are reported side by side, flagged.

Outputs in <main>/glycans/: m1_rescored.jsonl, samples_rescored.csv, rescored_targets.csv, nochange_check.csv,
m4_spot_check.csv, unchanged.csv, side_by_side.csv, report.md, and per variant (without_three/, rescored/):
contrasts.csv, hw_minus_rc.csv, pairs_seed.csv, pairs_sample.csv, q3.csv, checks_m1.csv,
sensitivity_post_cutoff.csv.
"""

import argparse
import json
import pathlib
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / 'analysis'))
sys.path.insert(0, str(REPO / 'harness'))

import contrasts  # noqa: E402
import cutoff  # noqa: E402

THREE = ('7OZC', '7WDT', '7WJB')
M1_COLUMNS = ['m1_ligand_rmsd', 'm1_pb_valid', 'm1_success', 'm1_pb_failed', 'm1_error']
PAIR_KEYS = ['contrast', 'target', 'seed_a', 'seed_b', 'sample']
SEED_KEYS = ['contrast', 'target', 'seed_a', 'seed_b']
SPOT_EVERY = 25
FROM_PRIMARY = 'from primary pairs_sample.csv'
VARIANTS = (('primary', '(1) primary'), ('without_three', '(2) 57 targets'), ('rescored', '(3) rescored, 60 targets'))


def _rescore(job):
  import score_ligand
  target, path, sha = job
  res = score_ligand.score(target, REPO / path, drop_af3_removed=True)
  res.update(model_sha256=sha, removed_atoms=score_ligand.reference(target, True).removed_atoms)
  return res


def _score_plain_and_rescored(job):
  import score_ligand
  target, path = job
  return target, path, score_ligand.score(target, REPO / path), score_ligand.score(target, REPO / path,
                                                                                  drop_af3_removed=True)


def rescore(ok, out, jobs):
  """{(model_sha256, target): score_ligand result with drop_af3_removed} for every sample of the three targets."""
  cache_path = out / 'm1_rescored.jsonl'
  cache = {}
  if cache_path.exists():
    for line in cache_path.read_text().splitlines():
      c = json.loads(line)
      cache[(c['model_sha256'], c['target'])] = c
  todo = ok[ok.target.isin(THREE)][['target', 'model_path', 'model_sha256']].drop_duplicates()
  need = [tuple(r) for r in todo.itertuples(index=False) if (r.model_sha256, r.target) not in cache]
  if need:
    with ProcessPoolExecutor(max_workers=jobs) as pool, open(cache_path, 'a') as f:
      for res in pool.map(_rescore, need, chunksize=4):
        f.write(json.dumps(res) + '\n')
        cache[(res['model_sha256'], res['target'])] = res
  print(f'>> rescoring: {len(todo) - len(need)} from {cache_path.name}, {len(need)} scored now', flush=True)
  return cache


def with_rescored(df, cache):
  """A copy of the table with M1 of the three targets' samples from the rescoring (as collect.py writes M1)."""
  df = df.copy()
  idx = df.index[df.target.isin(THREE) & df.model_path.notna()]
  res = [cache[(df.at[i, 'model_sha256'], df.at[i, 'target'])] for i in idx]
  df.loc[idx, 'm1_ligand_rmsd'] = [r['ligand_rmsd'] for r in res]
  df.loc[idx, 'm1_pb_valid'] = [r['pb_valid'] for r in res]
  df.loc[idx, 'm1_success'] = [bool(r['success']) for r in res]
  df.loc[idx, 'm1_pb_failed'] = [';'.join(r['pb_failed'] or []) or None for r in res]
  df.loc[idx, 'm1_error'] = [r['error'] for r in res]
  return df


def rescored_targets(ok):
  """Per rescored target, condition and repetition: the samples' ligand RMSD, PoseBusters failures and success."""
  rows = []
  for (t, c, rep), g in ok[ok.target.isin(THREE)].groupby(['target', 'condition', 'repetition']):
    failed = pd.Series(';'.join(g.m1_pb_failed.fillna('')).split(';'))
    failed = failed[failed != ''].value_counts()
    rows.append({'target': t, 'condition': c, 'repetition': rep, 'samples': len(g),
                 'm1_errors': int(g.m1_error.notna().sum()),
                 'ligand_rmsd_median': float(g.m1_ligand_rmsd.median()),
                 'ligand_rmsd_max': float(g.m1_ligand_rmsd.max()),
                 'rmsd_below_2': int((g.m1_ligand_rmsd < 2).sum()),
                 'pb_valid': int((g.m1_pb_valid == True).sum()),  # noqa: E712
                 'success': int((g.m1_success == True).sum()),  # noqa: E712
                 'pb_failed_checks': '; '.join(f'{k} {v}' for k, v in failed.items())})
  return pd.DataFrame(rows)


def nochange_check(ok, jobs):
  """The rescoring on one sample per target of the 57 (r1, first condition, seed and sample): equal to M1 as scored."""
  first = ok[~ok.target.isin(THREE) & ok.model_path.notna() & (ok.repetition == 'r1')] \
      .sort_values(['target', 'condition', 'seed', 'sample']).groupby('target').head(1)
  jobs_ = [(r.target, r.model_path) for r in first.itertuples()]
  rows = []
  with ProcessPoolExecutor(max_workers=jobs) as pool:
    for t, path, plain, res in pool.map(_score_plain_and_rescored, jobs_):
      row = first[(first.target == t) & (first.model_path == path)].iloc[0]
      rows.append({'target': t, 'model_path': path, 'm1_ligand_rmsd': row.m1_ligand_rmsd,
                   'rescored_ligand_rmsd': res['ligand_rmsd'], 'm1_success': row.m1_success,
                   'rescored_success': res['success'], 'rescored_error': res['error'],
                   'same_as_samples_csv': bool(res['ligand_rmsd'] == row.m1_ligand_rmsd
                                               and res['success'] == bool(row.m1_success)
                                               and res['pb_valid'] == bool(row.m1_pb_valid) and res['error'] is None),
                   'same_result_dict': plain == res})
  return pd.DataFrame(rows)


def build_pairs(ok, primary_pairs, recompute):
  """contrasts.build_pairs with M4 of targets outside THREE from the primary pairs; THREE recomputed with the
  rescored reference if recompute. Returns (seed_pairs, sample_pairs, contrasts, excluded, spot checks)."""
  import score_ligand
  calls = {'n': 0}

  def divergence(target, path_a, path_b):
    if target in THREE:
      if not recompute:
        raise RuntimeError(f'{target} is not in this variant')
      return score_ligand.pair_divergence(target, REPO / path_a, REPO / path_b, drop_af3_removed=True)
    calls['n'] += 1
    if recompute and calls['n'] % SPOT_EVERY == 0:
      res = score_ligand.pair_divergence(target, REPO / path_a, REPO / path_b, drop_af3_removed=True)
      return {**res, 'error': FROM_PRIMARY if res['error'] is None else res['error']}
    return {'ligand_rmsd': None, 'ca_rmsd_global': None, 'error': FROM_PRIMARY}

  saved = contrasts.divergence
  contrasts.divergence = divergence
  try:
    seed_pairs, sample_pairs, cons, excluded = contrasts.build_pairs(ok)
  finally:
    contrasts.divergence = saved
  take = sample_pairs.m4_error == FROM_PRIMARY
  merged = sample_pairs.merge(primary_pairs[PAIR_KEYS + ['bit_identical', 'm4_ligand_rmsd', 'm4_ca_rmsd_global',
                                                         'm4_error']],
                              on=PAIR_KEYS, how='left', suffixes=('', '_primary'), validate='one_to_one')
  if merged.bit_identical_primary[take.values].isna().any():
    raise SystemExit('a sample pair of the 57 targets is not in the primary pairs_sample.csv')
  spot = merged[take.values & merged.m4_ligand_rmsd.notna()][PAIR_KEYS + [
      'm4_ligand_rmsd', 'm4_ligand_rmsd_primary', 'm4_ca_rmsd_global', 'm4_ca_rmsd_global_primary']].copy()
  spot['equal'] = (spot.m4_ligand_rmsd == spot.m4_ligand_rmsd_primary) & \
      (spot.m4_ca_rmsd_global == spot.m4_ca_rmsd_global_primary)
  for col in ('m4_ligand_rmsd', 'm4_ca_rmsd_global', 'm4_error'):
    sample_pairs.loc[take, col] = merged.loc[take.values, f'{col}_primary'].values
  sample_pairs['m4_error'] = sample_pairs.m4_error.where(sample_pairs.m4_error.notna(), None)
  if not (merged.bit_identical[take.values] == merged.bit_identical_primary[take.values]).all():
    raise SystemExit('bit_identical differs from the primary pairs for a pair of the 57 targets')
  return seed_pairs, sample_pairs, cons, excluded, spot


def run_variant(name, ok, primary_pairs, args, out, recompute):
  d = out / name
  d.mkdir(parents=True, exist_ok=True)
  seed_pairs, sample_pairs, cons, excluded, spot = build_pairs(ok, primary_pairs, recompute)
  summary = contrasts.summarize(seed_pairs, sample_pairs, cons, args)
  hwrc = contrasts.hw_minus_rc(seed_pairs, cons, args)
  q3rows, answer = contrasts.q3(ok, args)
  chk = pd.DataFrame(contrasts.m1_checks(ok))
  sens, post = cutoff.sensitivity(ok, args)
  for fname, table in (('contrasts.csv', summary), ('hw_minus_rc.csv', hwrc), ('pairs_seed.csv', seed_pairs),
                       ('pairs_sample.csv', sample_pairs), ('q3.csv', q3rows), ('checks_m1.csv', chk),
                       ('excluded.csv', excluded), ('sensitivity_post_cutoff.csv', sens)):
    table.to_csv(d / fname, index=False)
  print(f'>> {name}: {ok.target.nunique()} targets, {len(cons)} contrasts, {len(seed_pairs)} seed pairs, '
        f'{len(sample_pairs)} sample pairs, Q3 answer {answer or "not reached"}', flush=True)
  return {'summary': summary, 'hwrc': hwrc, 'q3': q3rows, 'checks': chk, 'sens': sens, 'seed_pairs': seed_pairs,
          'sample_pairs': sample_pairs, 'spot': spot}


def primary_tables(main):
  c = main / 'contrasts'
  chk = pd.read_csv(c / 'checks.csv')
  return {'summary': pd.read_csv(c / 'contrasts.csv'), 'hwrc': pd.read_csv(c / 'hw_minus_rc.csv'),
          'q3': pd.read_csv(c / 'q3.csv'), 'sens': pd.read_csv(c / 'sensitivity_post_cutoff.csv'),
          'checks': chk[chk.check.str.startswith(('3 ', 'M1 computed'))].reset_index(drop=True),
          'seed_pairs': pd.read_csv(c / 'pairs_seed.csv'), 'sample_pairs': pd.read_csv(c / 'pairs_sample.csv')}


def keyed_rows(t):
  """{(section, name, statistic, reference): {estimate, ci_low, ci_high, n, n_targets, text}} of one variant: every
  result that uses M1 (and the rows of the cutoff sensitivity table, which also has top-1 flips)."""
  rows = {}

  def put(section, name, statistic, reference='', estimate=np.nan, lo=np.nan, hi=np.nan, n=np.nan, n_targets=np.nan,
          text='', uses_m1=True):
    rows[(section, name, statistic, reference)] = {'estimate': estimate, 'ci_low': lo, 'ci_high': hi, 'n': n,
                                                   'n_targets': n_targets, 'text': text, 'uses_m1': uses_m1}

  sens = t['sens']
  for r in sens[sens.set == 'all targets'].itertuples():
    if r.quantity == 'M1 success rate':
      put('success rate, r1', r.name, r.statistic, '', r.estimate, n=r.n, n_targets=r.n_targets)
    elif r.quantity == 'Q3' and r.name == 'answer':
      put('Q3', 'answer', 'seeds needed', '', r.estimate)
  for r in t['q3'].itertuples():
    sec = 'success rate, Q3 (overall top-1 of the first n seeds)' if r.statistic == 'success_rate' else 'Q3'
    put(sec, f'{r.condition} (n = {r.n_seeds})', r.statistic, '', r.estimate, r.ci_low, r.ci_high,
        n_targets=r.n_targets)
  for r in t['checks'].itertuples():
    put('checks', r.condition, r.check, '', text=f'{r.value}: {r.detail}')
  s = t['summary']
  for r in s[s.statistic.isin(['flip_success', 'flip_matters', 'm4_ligand_rmsd_median', 'm4_ligand_rmsd_p90',
                               'm4_errors'])].itertuples():
    sec = 'M4 ligand RMSD' if r.statistic.startswith('m4') else 'flips'
    put(sec, r.contrast, r.statistic, '', r.estimate, r.ci_low, r.ci_high, r.n, r.n_targets,
        text=r.note if isinstance(r.note, str) else '')
  for r in s[s.statistic.isin(['m4_ca_rmsd_global_median', 'm4_ca_rmsd_global_p90'])].itertuples():
    put('M4 C-alpha RMSD (missing in (1) for the three targets)', r.contrast, r.statistic, '', r.estimate, r.ci_low,
        r.ci_high, r.n, r.n_targets, uses_m1=False)
  h = t['hwrc']
  for r in h[h.statistic.isin(['flip_success', 'flip_matters'])].itertuples():
    put('HW minus RC', r.hardware, r.statistic, f'{r.recompilation} ({r.reference})', r.estimate, r.ci_low,
        r.ci_high, n_targets=r.n_targets)
  for r in sens[sens.set != 'all targets'].itertuples():
    put('cutoff sensitivity, released after 2021-09-30', f'{r.quantity}: {r.name}', r.statistic,
        r.reference if isinstance(r.reference, str) else '', r.estimate, r.ci_low, r.ci_high, r.n, r.n_targets,
        uses_m1=r.statistic != 'flip_top1')
  return rows


def side_by_side(tables):
  keyed = {v: keyed_rows(tables[v]) for v, _ in VARIANTS}
  order = list(keyed['primary'])
  order += [k for v, _ in VARIANTS[1:] for k in keyed[v] if k not in order]
  out = []
  for k in order:
    row = {'section': k[0], 'name': k[1], 'statistic': k[2], 'reference': k[3],
           'uses_m1': next(keyed[v][k]['uses_m1'] for v, _ in VARIANTS if k in keyed[v])}
    for v, _ in VARIANTS:
      r = keyed[v].get(k)
      for f in ('estimate', 'ci_low', 'ci_high', 'n', 'n_targets', 'text'):
        row[f'{v}_{f}'] = r[f] if r else (np.nan if f != 'text' else '')
    out.append(row)
  return pd.DataFrame(out)


def unchanged(primary, rescored, cost_equal):
  """What (3) must leave unchanged, compared with the primary outputs: [(check, value, pass)]."""
  rows = []
  a, b = primary['seed_pairs'], rescored['seed_pairs']
  m = a.merge(b, on=SEED_KEYS, how='outer', suffixes=('_p', '_r'), indicator=True)
  for cols, label in ((('top1_a', 'top1_b'), 'M2 top-1 of every (target, seed) pair'),
                      (('flip_top1',), 'top-1 flip of every (target, seed) pair'),
                      (('margin_a', 'margin_b'), 'M3 margin of every (target, seed) pair')):
    same = (m._merge == 'both').all() and all(((m[f'{c}_p'] == m[f'{c}_r']) |
                                               (m[f'{c}_p'].isna() & m[f'{c}_r'].isna())).all() for c in cols)
    rows.append({'check': label, 'value': f'{len(m)} pairs', 'pass': bool(same)})
  a, b = primary['sample_pairs'], rescored['sample_pairs']
  m = a.merge(b, on=PAIR_KEYS, how='outer', suffixes=('_p', '_r'), indicator=True)
  ca_p, ca_r = m.m4_ca_rmsd_global_p, m.m4_ca_rmsd_global_r
  only_r = ca_p.isna() & ca_r.notna()
  same = (m._merge == 'both').all() and bool((ca_p[ca_p.notna()] == ca_r[ca_p.notna()]).all()) and \
      bool(ca_r.notna().all()) and set(m[only_r].target) <= set(THREE)
  rows.append({'check': 'm4_ca_rmsd_global of every sample pair where (1) has it; the others are pairs of the three '
                        'targets, computed in (3) only',
               'value': f'{int(ca_p.notna().sum())} equal pairs; {int(only_r.sum())} pairs of '
                        f'{", ".join(sorted(set(m[only_r].target)))} only in (3)', 'pass': same})
  for c in ('bit_identical', 'm5_ranking_score', 'm5_mean_plddt', 'm5_iptm'):
    same = (m._merge == 'both').all() and ((m[f'{c}_p'] == m[f'{c}_r']) |
                                           (m[f'{c}_p'].isna() & m[f'{c}_r'].isna())).all()
    rows.append({'check': f'{c} of every sample pair', 'value': f'{len(m)} pairs', 'pass': bool(same)})
  keep = lambda s: s[s.statistic.isin(['flip_top1', 'bit_identical_fraction']) |  # noqa: E731
                     s.statistic.str.startswith('m5_')].reset_index(drop=True)
  a, b = keep(primary['summary']), keep(rescored['summary'])
  cols = ['contrast', 'statistic', 'n', 'n_targets', 'estimate', 'ci_low', 'ci_high']
  same = len(a) == len(b) and np.allclose(a[cols[2:]].astype(float), b[cols[2:]].astype(float), rtol=0, atol=0,
                                          equal_nan=True) and (a[cols[:2]].values == b[cols[:2]].values).all()
  rows.append({'check': 'contrasts.csv: flip_top1, M5 and bit-identity rows (estimate and 95% CI)',
               'value': f'{len(a)} rows', 'pass': bool(same)})
  a = primary['hwrc'][primary['hwrc'].statistic == 'flip_top1'].reset_index(drop=True)
  b = rescored['hwrc'][rescored['hwrc'].statistic == 'flip_top1'].reset_index(drop=True)
  same = len(a) == len(b) and np.allclose(a[['estimate', 'ci_low', 'ci_high']], b[['estimate', 'ci_low', 'ci_high']],
                                          rtol=0, atol=0, equal_nan=True)
  rows.append({'check': 'hw_minus_rc.csv: flip_top1 rows', 'value': f'{len(a)} rows', 'pass': bool(same)})
  rows += cost_equal
  return pd.DataFrame(rows)


def cost_unchanged(main, rescored_csv, out):
  """M6 and M7: cost.py reads no M1 column; its CSV outputs from the rescored table equal the primary ones."""
  src = (REPO / 'analysis' / 'cost.py').read_text()
  rows = [{'check': 'analysis/cost.py reads no M1 column', 'value': f'{src.count("m1_")} occurrences of "m1_"',
           'pass': 'm1_' not in src}]
  tmp = out / 'rescored' / 'cost_check'
  subprocess.run([sys.executable, str(REPO / 'analysis' / 'cost.py'), '--samples', str(rescored_csv), '--labels',
                  'main', '--out', str(tmp)], check=True, capture_output=True)
  for f in sorted((main / 'cost').glob('*.csv')):
    same = (tmp / f.name).is_file() and (tmp / f.name).read_bytes() == f.read_bytes()
    rows.append({'check': f'M6/M7 cost/{f.name} from the rescored table, byte for byte', 'value': '', 'pass': same})
  return rows


def pct(x):
  return '-' if pd.isna(x) else f'{100 * x:.1f}%'


def cell(row, v):
  e, lo, hi, txt = row[f'{v}_estimate'], row[f'{v}_ci_low'], row[f'{v}_ci_high'], row[f'{v}_text']
  if row['section'] == 'checks':
    return txt or '-'
  if row['statistic'] == 'seeds needed':
    return 'not reached' if pd.isna(e) else str(int(e))
  if row['statistic'] == 'm4_errors':
    return '-' if pd.isna(row[f'{v}_n']) else f'{int(row[f"{v}_n"])} pairs'
  if row['statistic'].startswith('m4_'):
    return '-' if pd.isna(e) else f'{e:.3g} [{lo:.3g}, {hi:.3g}]' if not pd.isna(lo) else f'{e:.3g}'
  base = pct(e) if pd.isna(lo) else f'{pct(e)} [{pct(lo)}, {pct(hi)}]'
  nt = row[f'{v}_n_targets']
  return base if pd.isna(e) or pd.isna(nt) else f'{base} ({int(nt)})'


def report(table, rt, nochange, spot, unch, out):
  lines = ['# M1 for three glycan ligands: primary, 57 targets, rescored', '',
           'Section 10 row of 2026-10-14 (notes/analysis_plan.md). (1) primary, section 4 as written (the three '
           'targets count as not a success; from contrasts/). (2) section 8, the 57 targets other than 7OZC, 7WDT, '
           '7WJB. (3) post-hoc sensitivity: the three targets scored on the heavy atoms AlphaFold3 predicts '
           '(reference ligand without O1; same pocket, pocket C-alpha superposition, symmetry correction and '
           'PoseBusters dock checks), all 60 targets. Same functions and bootstrap as the primary analysis '
           '(10,000 replicates over targets, seed 2026). Code: analysis/glycans.py.', '',
           'Percentages: estimate [95% CI] (targets). M4: median or 90th percentile of ligand RMSD in angstrom '
           '[95% CI].', '',
           '| section | name | statistic | (1) primary | (2) 57 targets | (3) rescored |', '|---|---|---|---|---|---|']
  for row in table.to_dict('records'):
    name = row['name'] + (f' vs {row["reference"]}' if row['reference'] else '')
    flag = '' if row['uses_m1'] else ' (no M1)'
    lines.append(f'| {row["section"]} | {name} | {row["statistic"]}{flag} | {cell(row, "primary")} | '
                 f'{cell(row, "without_three")} | {cell(row, "rescored")} |')
  lines += ['', '## The three targets, rescored (all conditions and repetitions)', '',
            '| target | condition | rep | samples | RMSD median | RMSD < 2 A | PB-valid | success | PoseBusters checks failed |',
            '|---|---|---|---|---|---|---|---|---|']
  lines += [f'| {r.target} | {r.condition} | {r.repetition} | {r.samples} | {r.ligand_rmsd_median:.3f} | '
            f'{r.rmsd_below_2} | {r.pb_valid} | {r.success} | {r.pb_failed_checks or "-"} |' for r in rt.itertuples()]
  lines += ['', '## Checks of the rescoring', '',
            f'- Rescoring of one sample per target of the 57 (none has an atom AlphaFold3 removes): '
            f'{int(nochange.same_as_samples_csv.sum())}/{len(nochange)} equal to samples.csv, '
            f'{int(nochange.same_result_dict.sum())}/{len(nochange)} identical result dicts (nochange_check.csv).',
            f'- M4 of the 57 targets taken from the primary pairs; spot check, every {SPOT_EVERY}th non-identical pair '
            f'recomputed with the rescoring option: {int(spot.equal.sum())}/{len(spot)} equal (m4_spot_check.csv).',
            '', '## Unchanged by (3) (unchanged.csv)', '', '| check | value | pass |', '|---|---|---|']
  lines += [f'| {r["check"]} | {r["value"]} | {"yes" if r["pass"] else "NO"} |' for r in unch.to_dict('records')]
  (out / 'report.md').write_text('\n'.join(lines) + '\n')


def main():
  ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
  ap.add_argument('--main', default='results/analysis/main_20261014')
  ap.add_argument('--out')
  ap.add_argument('--jobs', type=int, default=8)
  ap.add_argument('--replicates', type=int, default=10000)
  ap.add_argument('--seed', type=int, default=2026)
  args = ap.parse_args()
  main_dir = REPO / args.main if not pathlib.Path(args.main).is_absolute() else pathlib.Path(args.main)
  out = pathlib.Path(args.out) if args.out else main_dir / 'glycans'
  out.mkdir(parents=True, exist_ok=True)
  load_args = argparse.Namespace(samples=main_dir / 'samples.csv', labels='main', include=None, exclude=None,
                                 condition='platform', replicates=args.replicates, seed=args.seed)
  d, ok = contrasts.load(load_args)
  primary = primary_tables(main_dir)

  cache = rescore(ok, out, args.jobs)
  ok_rescored = with_rescored(ok, cache)
  raw = pd.read_csv(main_dir / 'samples.csv')
  with_rescored(raw, cache).to_csv(out / 'samples_rescored.csv', index=False)
  rt = rescored_targets(ok_rescored)
  rt.to_csv(out / 'rescored_targets.csv', index=False)
  nochange = nochange_check(ok, args.jobs)
  nochange.to_csv(out / 'nochange_check.csv', index=False)

  tables = {'primary': primary}
  tables['without_three'] = run_variant('without_three', ok[~ok.target.isin(THREE)], primary['sample_pairs'],
                                        load_args, out, recompute=False)
  tables['rescored'] = run_variant('rescored', ok_rescored, primary['sample_pairs'], load_args, out, recompute=True)
  spot = tables['rescored']['spot']
  spot.to_csv(out / 'm4_spot_check.csv', index=False)
  written = {k: pd.read_csv(out / 'rescored' / f) for k, f in (('seed_pairs', 'pairs_seed.csv'),
                                                               ('sample_pairs', 'pairs_sample.csv'),
                                                               ('summary', 'contrasts.csv'),
                                                               ('hwrc', 'hw_minus_rc.csv'))}
  unch = unchanged(primary, written, cost_unchanged(main_dir, out / 'samples_rescored.csv', out))
  unch.to_csv(out / 'unchanged.csv', index=False)
  table = side_by_side(tables)
  table.to_csv(out / 'side_by_side.csv', index=False)
  report(table, rt, nochange, spot, unch, out)
  bad = (not nochange.same_as_samples_csv.all()) or (not spot.equal.all()) or (not unch['pass'].all())
  print(f'>> no-change check {int(nochange.same_as_samples_csv.sum())}/{len(nochange)}; M4 spot check '
        f'{int(spot.equal.sum())}/{len(spot)}; unchanged {int(unch["pass"].sum())}/{len(unch)}; '
        f'report {out / "report.md"}', flush=True)
  return 1 if bad else 0


if __name__ == '__main__':
  sys.exit(main())
