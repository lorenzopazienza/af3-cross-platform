"""Training-cutoff sensitivity analysis (notes/analysis_plan.md section 10, row of 2026-10-13; secondary).

    analysis/.venv/bin/python analysis/cutoff.py [--out results/analysis/<name>]     # the release-date table only

The released AlphaFold3 weights were trained on structures up to 2021-09-30. The release-date table lists, for the
60 targets, the RCSB initial release date from the frozen target records (targets/posebusters_subset.csv,
column rcsb_initial_release) and whether the target was released after the cutoff. analysis/contrasts.py then
reports, next to the full-set values and computed by the same functions on the post-cutoff targets only:
  - M1 success rate per condition, r1: of the overall top-1 across seeds (as AlphaFold3 ranks them; the
    definition of pilot check 3) and, secondary, of the top-1 per seed;
  - Q3 (analysis/contrasts.q3);
  - the top-1 and success flip rates of the section 5 primary hardware pairs (analysis/contrasts.summarize), and
    HW minus RC for them (analysis/contrasts.hw_minus_rc), with the same bootstrap.
The primary analysis is unchanged. Outputs (by contrasts.py): release_dates.csv, sensitivity_post_cutoff.csv and a
section of report.md.
"""
import argparse
import pathlib

import numpy as np
import pandas as pd

REPO = pathlib.Path(__file__).resolve().parent.parent
CUTOFF = '2021-09-30'
FLIPS = ['flip_top1', 'flip_success']


def release_table(path=REPO / 'targets' / 'posebusters_subset.csv'):
  t = pd.read_csv(path, dtype=str)[['pdb_id', 'rcsb_initial_release', 'bucket', 'pilot']]
  t['post_cutoff'] = t.rcsb_initial_release > CUTOFF
  return t.sort_values(['rcsb_initial_release', 'pdb_id']).reset_index(drop=True)


def success_rates(ok, label):
  """M1 success rate per condition on r1: overall top-1 across seeds, and top-1 per seed (secondary)."""
  import contrasts
  r1 = ok[ok.repetition == 'r1']
  rows = []
  for c in sorted(r1.condition.unique(), key=contrasts.rank):
    g = r1[r1.condition == c].sort_values('ranking_score', ascending=False, kind='mergesort')
    overall = g.groupby('target').head(1)
    per_seed = g.groupby(['target', 'seed']).head(1)
    for stat, top in (('success_rate_overall_top1', overall), ('success_rate_top1_per_seed', per_seed)):
      rows.append({'set': label, 'quantity': 'M1 success rate', 'name': c, 'statistic': stat, 'n': len(top),
                   'n_targets': top.target.nunique(), 'estimate': float((top.m1_success == True).mean())  # noqa: E712
                   if len(top) else np.nan, 'ci_low': np.nan, 'ci_high': np.nan, 'reference': ''})
  return rows


def block(ok, args, label):
  """The sensitivity quantities on one set of targets, as rows of one table."""
  import contrasts
  rows = success_rates(ok, label)
  q3rows, answer = contrasts.q3(ok, args)
  for r in q3rows.itertuples():
    rows.append({'set': label, 'quantity': 'Q3', 'name': f'{r.condition} (n = {r.n_seeds})', 'statistic': r.statistic,
                 'n': np.nan, 'n_targets': r.n_targets, 'estimate': r.estimate, 'ci_low': r.ci_low,
                 'ci_high': r.ci_high, 'reference': ''})
  rows.append({'set': label, 'quantity': 'Q3', 'name': 'answer', 'statistic': 'seeds needed', 'n': np.nan,
               'n_targets': np.nan, 'estimate': answer if answer else np.nan, 'ci_low': np.nan, 'ci_high': np.nan,
               'reference': ''})                                    # NaN: not reached
  seed_pairs, sample_pairs, cons, _ = contrasts.build_pairs(ok)
  primary = {k: v for k, v in cons.items() if v == ('HW', True)}
  rc = {k: v for k, v in cons.items() if v[0] == 'RC'}
  if primary:
    s = contrasts.summarize(seed_pairs, sample_pairs, primary, args)
    for r in s[s.statistic.isin(FLIPS)].itertuples():
      rows.append({'set': label, 'quantity': 'primary pair flips', 'name': r.contrast, 'statistic': r.statistic,
                   'n': r.n, 'n_targets': r.n_targets, 'estimate': r.estimate, 'ci_low': r.ci_low,
                   'ci_high': r.ci_high, 'reference': ''})
    h = contrasts.hw_minus_rc(seed_pairs, {**primary, **rc}, args)
    for r in h[h.statistic.isin(FLIPS)].itertuples():
      rows.append({'set': label, 'quantity': 'HW minus RC', 'name': r.hardware, 'statistic': r.statistic, 'n': np.nan,
                   'n_targets': r.n_targets, 'estimate': r.estimate, 'ci_low': r.ci_low, 'ci_high': r.ci_high,
                   'reference': f'{r.recompilation} ({r.reference})'})
  return rows


def sensitivity(ok, args, releases=None):
  """Full set and post-cutoff targets, side by side: (table, post-cutoff targets present in ok)."""
  t = release_table() if releases is None else releases
  post = set(t[t.post_cutoff].pdb_id)
  sub = ok[ok.target.isin(post)]
  rows = block(ok, args, 'all targets') + (block(sub, args, f'released after {CUTOFF}') if len(sub) else [])
  return pd.DataFrame(rows), sorted(set(sub.target))


def report_lines(table, post_targets, releases):
  def pct(x):
    return '-' if pd.isna(x) else f'{100 * x:.1f}%'
  n_post = int(releases.post_cutoff.sum())
  lines = ['', '## Training-cutoff sensitivity (secondary; section 10, 2026-10-13)', '',
           f'Targets released after {CUTOFF}: {n_post} of {len(releases)} in the frozen records '
           f'({len(post_targets)} present in these runs). Same functions as the primary analysis, on those targets '
           'only (release_dates.csv, sensitivity_post_cutoff.csv).', '',
           '| quantity | name | statistic | all targets | released after cutoff |', '|---|---|---|---|---|']
  full = table[table.set == 'all targets'].set_index(['quantity', 'name', 'statistic', 'reference'])
  sub = table[table.set != 'all targets'].set_index(['quantity', 'name', 'statistic', 'reference']) \
      if (table.set != 'all targets').any() else table.iloc[0:0].set_index(['quantity', 'name', 'statistic', 'reference'])
  for key in full.index:
    a = full.loc[key]
    b = sub.loc[key] if key in sub.index else None
    fmt = (lambda r: '-' if r is None else (f'{pct(r.estimate)} [{pct(r.ci_low)}, {pct(r.ci_high)}]'
                                            if not pd.isna(r.ci_low) else pct(r.estimate))) \
        if key[2] != 'seeds needed' else (lambda r: '-' if r is None else 'not reached' if pd.isna(r.estimate)
                                          else f'{int(r.estimate)}')
    name = key[1] + (f' vs {key[3]}' if key[3] and key[0] == 'HW minus RC' else '')
    lines.append(f'| {key[0]} | {name} | {key[2]} | {fmt(a)} | {fmt(b)} |')
  return lines


def main():
  ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
  ap.add_argument('--out', default='results/analysis/release_dates')
  args = ap.parse_args()
  out = REPO / args.out if not pathlib.Path(args.out).is_absolute() else pathlib.Path(args.out)
  out.mkdir(parents=True, exist_ok=True)
  t = release_table()
  t.to_csv(out / 'release_dates.csv', index=False)
  print(f'{len(t)} targets; released after {CUTOFF}: {int(t.post_cutoff.sum())}; on or before: '
        f'{int((~t.post_cutoff).sum())}; written {out / "release_dates.csv"}')


if __name__ == '__main__':
  main()
