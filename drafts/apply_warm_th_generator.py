"""Applies the warm_th change to the paper generator (analysis/paper_assets_isc.py) and its pending-results tests
(analysis/tests/test_paper_assets_pending.py) in a checkout of branch paper; row of 2026-11-11 "Warm cost per structure
on the L4 Triton and the H100" (drafts/section10_warm_th_20261111.md). Exact-anchor replacements: every anchor must
occur exactly once, otherwise nothing is written and the anchor is named. Run when no other session edits those files:

    python3 drafts/apply_warm_th_generator.py ~/Desktop/af3-paper

What it changes (nothing while results/analysis/warm_th_<DATE>/ is absent):
  warm_th_summary(main_dir)  reads warm_th_<DATE>/summary.csv and checks.csv (every check pass; one warm row at 1024
                             per arm, GT and H), records both in sources.tsv, adds a README note; None when absent;
  table_q2(..., warm_th=)    adds those rows to the warm column (refuses an arm that already has one) and, when
                             present, adds to the Table III note's warm sentence that the L4 Triton and the H100 were
                             measured later and H's Spot cost, extends the table's role line, and no longer appends
                             the "% TODO warm_th" line (kept while absent);
  main                       passes warm_th_summary(main_dir);
  tests                      three tests on synthetic folders (absent, present, refusals).
"""
import pathlib
import sys

GEN = 'analysis/paper_assets_isc.py'
TEST = 'analysis/tests/test_paper_assets_pending.py'

EDITS = {GEN: [
("""memlimit_ah=None):
  \"\"\"Table III (task of 2026-11-10, A3): compilation and inference split, cost per structure fresh, with compilation
  amortized (plan M6, second variant) and from warm reruns, A100 and H100 rows, and one memory metric on every device
  (M7 as the plan defines it: compiled peak plus arguments).\"\"\"""",
"""memlimit_ah=None,
             warm_th=None):
  \"\"\"Table III (task of 2026-11-10, A3): compilation and inference split, cost per structure fresh, with compilation
  amortized (plan M6, second variant) and from warm reruns, A100 and H100 rows, and one memory metric on every device
  (M7 as the plan defines it: compiled peak plus arguments). warm_th (warm_th_summary, row of 2026-11-11): the warm
  rows of the L4 Triton and the H100, added to the warm column; None leaves both cells empty.\"\"\""""),
("""  w = cost_all[cost_all.cache == 'warm']
  rows = []
""",
"""  w = cost_all[cost_all.cache == 'warm']
  if warm_th is not None:
    clash = sorted(set(warm_th.platform) & set(w[w.bucket == 1024].platform))
    if clash:
      raise SystemExit(f'warm_th: {clash} already have a warm row at 1024')
    w = pd.concat([w, warm_th], ignore_index=True)
  rows = []
"""),
("""  nwarm = w[(w.platform.isin(['V5', 'V6', 'G'])) & (w.bucket == 1024)].processes
""",
"""  nwarm = w[(w.platform.isin(['V5', 'V6', 'G', 'GT', 'H'])) & (w.bucket == 1024)].processes
  warm_later = ''
  if warm_th is not None:
    hs = warm_th[warm_th.platform == 'H'].iloc[0]
    warm_later = (' The L4 Triton and the H100 were measured the same way in a later run. The H100 ran on Spot; at '
                  f'its Spot price, its warm cost is {usd3(hs.median_usd_per_structure_spot)} USD.')
"""),
("""'v5e, v6e, L4 and L4 Triton; follow-ups 3 and 9 for the A100 and H100; memory limit descriptive',""",
"""'v5e, v6e, L4 and L4 Triton; follow-ups 3 and 9 for the A100 and H100; memory limit descriptive'
              + ('; warm cost of the L4 Triton and H100 descriptive, measured later' if warm_th is not None else ''),"""),
("""f'({int(nwarm.min())} of them at 1024).',""",
"""f'({int(nwarm.min())} of them at 1024).' + warm_later,"""),
("""  nf = out / 'tables' / 'TIII_q2_note.tex'
  nf.write_text(nf.read_text() + '% TODO warm_th: the warm cost of the L4 Triton and the H100 (empty cells) is written '
                'when results/analysis/warm_th_<DATE>/ exists; see analysis/paper_assets_isc.py, table_q2.\\n')""",
"""  if warm_th is None:
    nf = out / 'tables' / 'TIII_q2_note.tex'
    nf.write_text(nf.read_text() + '% TODO warm_th: the warm cost of the L4 Triton and the H100 (empty cells) is written '
                  'when results/analysis/warm_th_<DATE>/ exists; see analysis/paper_assets_isc.py, table_q2.\\n')"""),
("""def memlimit_cell(ml):""",
"""WARM_TH_COLUMNS = ['platform', 'bucket', 'cache', 'processes', 'median_usd_per_structure_standard',
                   'median_usd_per_structure_spot']


def warm_th_summary(main_dir):
  \"\"\"The warm rows at 1024 of the warm-cost completion on GT and H (results/analysis/warm_th_<DATE>/summary.csv, the
  columns of the main study's cost_summary.csv; analysis/warm_th.py writes it only when every check passed), or None.
  Its checks.csv must read pass on every row; one warm row per arm, GT and H, is required. Row of 2026-11-11.\"\"\"
  d = one_output_dir(main_dir, 'warm_th', ['summary.csv', 'checks.csv'])
  if d is None:
    return None
  rel = lambda f: str((d / f).relative_to(main_dir))  # noqa: E731
  ck = read(main_dir, rel('checks.csv'))
  if not len(ck) or (ck.detail != 'pass').any():
    raise SystemExit(f'{d.name}/checks.csv: a check did not pass')
  sm = read(main_dir, rel('summary.csv'))
  missing = [c for c in WARM_TH_COLUMNS if c not in sm.columns]
  w = sm[(sm.cache == 'warm') & (sm.bucket == 1024)] if not missing else sm
  if missing or sorted(w.platform) != ['GT', 'H']:
    raise SystemExit(f'{d.name}/summary.csv: columns missing {missing}, warm arms at 1024 {sorted(w.platform)}')
  NOTES.append(f'Table III: warm cells of GT and H from {d.name}/summary.csv (analysis/warm_th.py; every check of '
               'checks.csv passed).')
  return w.reset_index(drop=True)


def memlimit_cell(ml):"""),
("""memlimit_ah=memlimit_ah_summary(main_dir))""",
"""memlimit_ah=memlimit_ah_summary(main_dir), warm_th=warm_th_summary(main_dir))"""),
], TEST: [
("""Synthetic output folders only; no session and no real follow-up 13 or memlimit_ah output is read.""",
"""Row of 2026-11-11: the warm cost on GT and H (warm_th analysis, the format of the main study's cost_summary.csv).
Synthetic output folders only; no session and no real follow-up 13, memlimit_ah or warm_th output is read."""),
]}

TESTS_APPENDED = '''

# ---- warm cost on GT and H (label warm_th, row of 2026-11-11; analysis/warm_th.py)

def cost_rows(arm, warm=True, compile_s=50.0):
  rows = []
  for b in (256, 512, 1024):
    rows.append({'platform': arm, 'bucket': b, 'cache': 'fresh', 'processes': 10, 'targets': 2, 'median_wall_s': 100.0,
                 'median_inference_s': 60.0, 'median_compile_s': compile_s, 'median_usd_per_structure_standard': 0.1,
                 'median_usd_per_structure_spot': 0.05, 'median_usd_per_structure_amortised_standard': 0.08,
                 'median_usd_per_structure_amortised_spot': 0.04})
  if warm:
    rows.append({'platform': arm, 'bucket': 1024, 'cache': 'warm', 'processes': 2, 'targets': 2, 'median_wall_s': 70.0,
                 'median_inference_s': 60.0, 'median_compile_s': 3.0, 'median_usd_per_structure_standard': 0.07,
                 'median_usd_per_structure_spot': 0.035})
  return rows


PRICE_DATES = {'main': ['2026-10-08', '2026-10-09'], 'a100': ['2026-10-16'], 'h100': ['2026-10-27']}


def table_q2_inputs():
  """Minimal synthetic inputs of table_q2: the main study (V5, V6, G, GT, C), follow-ups 3 and 9 (A, H)."""
  cost = pd.DataFrame(cost_rows('V5') + cost_rows('V6') + cost_rows('G') + cost_rows('GT', warm=False)
                      + cost_rows('C', warm=False))
  gpu = pd.DataFrame(cost_rows('A') + cost_rows('H', warm=False))
  mem = lambda arms: pd.DataFrame([{'platform': a, 'bucket': 1024, 'compiled_peak_plus_args_gib': 3.0,  # noqa: E731
                                    'compiled_peak_gib': 2.0, 'device_limit_gib': 20.0} for a in arms])
  memlimit = pd.DataFrame([{'arm': a, 'session': 's', 'sizes_run': 2, 'largest_completed_size': 3584,
                            'stopping_size': 4096, 'cause': 'oom'} for a in ('V5', 'V6', 'G', 'GT')])
  return cost, mem(['V5', 'V6', 'G', 'GT']), memlimit, gpu, mem(['A', 'H'])


def warm_th_outputs(main, checks_ok=True, arms=('GT', 'H')):
  d = main / isc.AN / 'warm_th_20261113'
  rows = []
  for a in arms:
    rows += cost_rows(a)
  sm = pd.DataFrame(rows)
  h_warm = (sm.cache == 'warm') & (sm.platform == 'H')
  sm.loc[h_warm, 'median_usd_per_structure_standard'] = 0.2468
  sm.loc[h_warm, 'median_usd_per_structure_spot'] = 0.1357
  write(d, 'summary.csv', sm)
  write(d, 'checks.csv', pd.DataFrame([{'check': '0', 'condition': 'GT, H', 'value': '', 'detail': 'pass'},
                                       {'check': '2b', 'condition': 'H', 'value': '',
                                        'detail': 'pass' if checks_ok else 'FAIL: 7D5C'}]))
  return d


def test_warm_th_absent_keeps_the_cells_empty_and_the_note(tmp_path, clean):
  main, out = tmp_path / 'main', tmp_path / 'out'
  (main / isc.AN).mkdir(parents=True)
  assert isc.warm_th_summary(main) is None
  (out / 'tables').mkdir(parents=True)
  df = isc.table_q2(out, *table_q2_inputs(), price_dates=PRICE_DATES, batch1024=8)
  assert list(df[df.arm.isin(['GT', 'H'])]['USD warm']) == ['', '']
  assert df[df.arm == 'G'].iloc[0]['USD warm'] == isc.usd3(0.07)
  note = (out / 'tables' / 'TIII_q2_note.tex').read_text()
  assert '% TODO warm_th' in note and 'measured the same way in a later run' not in note


def test_warm_th_fills_the_two_cells_and_updates_the_note(tmp_path, clean):
  main, out = tmp_path / 'main', tmp_path / 'out'
  warm_th_outputs(main)
  w = isc.warm_th_summary(main)
  assert sorted(w.platform) == ['GT', 'H'] and (w.cache == 'warm').all() and (w.bucket == 1024).all()
  assert f'{isc.AN}/warm_th_20261113/summary.csv' in isc.SOURCES and f'{isc.AN}/warm_th_20261113/checks.csv' in isc.SOURCES
  (out / 'tables').mkdir(parents=True)
  df = isc.table_q2(out, *table_q2_inputs(), price_dates=PRICE_DATES, batch1024=8, warm_th=w)
  cell = df.set_index('arm')['USD warm']
  assert cell['GT'] == isc.usd3(0.07) and cell['H'] == isc.usd3(0.2468) and cell['A'] == isc.usd3(0.07)
  note = (out / 'tables' / 'TIII_q2_note.tex').read_text()
  assert 'TODO warm_th' not in note and 'measured the same way in a later run' in note
  assert f'its warm cost is {isc.usd3(0.1357)} USD' in note and '(2 of them at 1024)' in note
  assert any('warm_th_20261113' in n for n in isc.NOTES)


def test_warm_th_refuses_failed_checks_missing_arms_and_double_rows(tmp_path, clean):
  main = tmp_path / 'main'
  warm_th_outputs(main, checks_ok=False)
  with pytest.raises(SystemExit, match='a check did not pass'):
    isc.warm_th_summary(main)
  main = tmp_path / 'main2'
  warm_th_outputs(main, arms=('GT',))
  with pytest.raises(SystemExit, match='warm arms'):
    isc.warm_th_summary(main)
  main = tmp_path / 'main3'
  warm_th_outputs(main)
  w = isc.warm_th_summary(main)
  cost, mem, memlimit, gpu, gmem = table_q2_inputs()
  (tmp_path / 'out' / 'tables').mkdir(parents=True)
  with pytest.raises(SystemExit, match='already have a warm row'):
    isc.table_q2(tmp_path / 'out', pd.concat([cost, pd.DataFrame(cost_rows('GT'))[lambda x: x.cache == 'warm']]), mem,
                 memlimit, gpu, gmem, price_dates=PRICE_DATES, warm_th=w)
'''


def main():
  root = pathlib.Path(sys.argv[1]).expanduser() if len(sys.argv) > 1 else pathlib.Path('.')
  new = {}
  for rel, edits in EDITS.items():
    s = (root / rel).read_text()
    if 'def warm_th_summary' in s:
      raise SystemExit(f'{rel} already defines warm_th_summary: applied before?')
    for old, rep in edits:
      n = s.count(old)
      if n != 1:
        raise SystemExit(f'{rel}: anchor found {n} times, nothing written:\n{old[:200]}')
      s = s.replace(old, rep)
    new[rel] = s
  new[TEST] = new[TEST].rstrip('\n') + '\n' + TESTS_APPENDED
  for rel, s in new.items():
    if '\u2014' in s:
      raise SystemExit(f'{rel}: em dash')
    (root / rel).write_text(s)
    print(f'written {rel}')
  return 0


if __name__ == '__main__':
  sys.exit(main())
