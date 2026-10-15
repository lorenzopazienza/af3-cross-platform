"""Checks of the primary rerun after the section 10 correction to M4 (row of 2026-10-14).

    analysis/.venv/bin/python scripts/analysis_runs/main_20261014/primary_v2/compare_with_primary.py

(a) M4's C-alpha RMSD of primary_v2/contrasts/pairs_sample.csv equals the one of glycans/rescored/pairs_sample.csv in
    every sample pair (analysis (3), whose protein superposition does not depend on the ligand).
(b) Every other output is identical to the existing primary output (main_20261014): every file byte for byte, except
    the expected differences, which are listed and checked to be only these: in pairs_sample.csv the column
    m4_ca_rmsd_global (only in pairs that had no value before); in contrasts.csv the rows m4_ca_rmsd_global_median and
    m4_ca_rmsd_global_p90; in report.md the lines of those rows.
Prints a summary; exit code 1 if a check fails.
"""
import difflib
import pathlib
import sys

import pandas as pd

# The outputs compared (the script lives in scripts/analysis_runs/main_20261014/primary_v2/).
HERE = pathlib.Path(__file__).resolve().parents[4] / 'results' / 'analysis' / 'main_20261014' / 'primary_v2'
OLD = HERE.parent
GLYCANS = OLD / 'glycans'
KEYS = ['contrast', 'target', 'seed_a', 'seed_b', 'sample']
CA = 'm4_ca_rmsd_global'
CA_STATS = ('m4_ca_rmsd_global_median', 'm4_ca_rmsd_global_p90')
COMPARED = ['samples.csv', 'excluded_sessions.csv', 'm1_cache.jsonl', 'contrasts', 'cost', 'decision']


def files():
  out = []
  for name in COMPARED:
    p = OLD / name
    out += sorted(f.relative_to(OLD) for f in p.rglob('*') if f.is_file()) if p.is_dir() else [pathlib.Path(name)]
  new = sorted(f.relative_to(HERE) for name in COMPARED for f in ((HERE / name).rglob('*') if (HERE / name).is_dir()
                                                                   else [HERE / name]) if f.is_file())
  return out, sorted(set(new) - set(out))


def text(path):
  return pd.read_csv(path, dtype=str, keep_default_na=False)


def check_pairs_sample(old, new):
  a, b = text(old), text(new)
  ok = list(a.columns) == list(b.columns) and len(a) == len(b) and (a[KEYS].values == b[KEYS].values).all()
  cols = [c for c in a.columns if not (a[c] == b[c]).all()]
  changed = a[CA] != b[CA]
  only_filled = bool((a.loc[changed, CA] == '').all() and (b.loc[changed, CA] != '').all())
  targets = sorted(set(a.loc[changed, 'target']))
  ok = ok and cols == [CA] and only_filled
  return ok, (f'differs only in column(s) {cols}; {int(changed.sum())} pairs, all empty before and filled now, '
              f'targets {", ".join(targets)}; every other cell identical' if ok else f'UNEXPECTED: columns {cols}')


def check_contrasts(old, new):
  a, b = text(old), text(new)
  ok = list(a.columns) == list(b.columns) and len(a) == len(b) and \
      (a[['contrast', 'statistic']].values == b[['contrast', 'statistic']].values).all()
  rows = ~(a == b).all(axis=1)
  stats = sorted(set(a.loc[rows, 'statistic']))
  ok = ok and set(stats) <= set(CA_STATS)
  return ok, (f'{int(rows.sum())} rows differ, all statistics in {stats} ({a.loc[rows, "contrast"].nunique()} '
              f'contrasts); every other row identical' if ok else f'UNEXPECTED rows: {stats}')


def check_report(old, new):
  a, b = old.read_text().splitlines(), new.read_text().splitlines()
  removed = [l for l in difflib.ndiff(a, b) if l.startswith('- ')]
  added = [l for l in difflib.ndiff(a, b) if l.startswith('+ ')]
  ok = len(a) == len(b) and all(any(f'| {s} |' in l for s in CA_STATS) for l in removed + added)
  return ok, (f'{len(removed)} lines differ, all M4 C-alpha rows; every other line identical' if ok
              else 'UNEXPECTED lines: ' + ' / '.join(l[:90] for l in (removed + added)[:6]))


EXPECTED = {pathlib.Path('contrasts/pairs_sample.csv'): check_pairs_sample,
            pathlib.Path('contrasts/contrasts.csv'): check_contrasts,
            pathlib.Path('contrasts/report.md'): check_report}


def main():
  fails = 0
  print('(a) M4 C-alpha RMSD against glycans/rescored/pairs_sample.csv')
  new = pd.read_csv(HERE / 'contrasts' / 'pairs_sample.csv')
  res = pd.read_csv(GLYCANS / 'rescored' / 'pairs_sample.csv')
  m = new.merge(res, on=KEYS, how='outer', suffixes=('', '_rescored'), indicator=True)
  equal = (m._merge == 'both') & (m[CA] == m[f'{CA}_rescored'])
  missing = int(new[CA].isna().sum())
  print(f'    {int(equal.sum())}/{len(m)} pairs equal ({len(new)} in primary_v2, {len(res)} in rescored); '
        f'pairs without a C-alpha RMSD in primary_v2: {missing}')
  fails += int(not equal.all() or missing)
  print('(b) every other output against the existing primary output')
  old_files, extra = files()
  n_same = 0
  for rel in old_files:
    a, b = OLD / rel, HERE / rel
    if not b.is_file():
      print(f'    MISSING {rel}')
      fails += 1
    elif a.read_bytes() == b.read_bytes():
      n_same += 1
      if rel in EXPECTED:
        print(f'    UNEXPECTED: {rel} identical (the correction should change it)')
        fails += 1
    elif rel in EXPECTED:
      ok, msg = EXPECTED[rel](a, b)
      print(f'    {rel}: {msg}')
      fails += int(not ok)
    else:
      print(f'    DIFFERS {rel}')
      fails += 1
  print(f'    {n_same}/{len(old_files)} files byte-identical; {len(EXPECTED)} expected to differ, checked above')
  if extra:
    print(f'    files only in primary_v2: {", ".join(map(str, extra))}')
  print('ALL CHECKS PASS' if not fails else f'{fails} CHECK(S) FAIL')
  return 1 if fails else 0


if __name__ == '__main__':
  sys.exit(main())
