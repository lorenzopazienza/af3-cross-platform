"""Rare-event intervals (drafts/section10_rare_spec_20261114.md, label rare): BCa, the exact target bound, the readings,
and the reproduction of registered intervals, on synthetic seed pairs. The registered tables are built here with the
registered code (contrasts.hw_minus_rc, contrasts.summarize, fu14.d_f_table), so the tests check that
analysis/rare_intervals.py recomputes them exactly. No real AlphaFold3 output is read."""
import math
import pathlib
import sys
import types

import numpy as np
import pandas as pd
import pytest

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'analysis'))
import contrasts  # noqa: E402
import fu13  # noqa: E402
import fu14  # noqa: E402
import rare_intervals as ri  # noqa: E402

REPS = 300
ARGS = types.SimpleNamespace(replicates=REPS, seed=2026)
TARGETS = [f'T{i:02d}' for i in range(12)]


def seed_pairs(name, kind, rate, rng, targets=TARGETS, seeds=(1, 2, 3, 4, 5)):
  rows = []
  for t in targets:
    for s in seeds:
      f = rng.random() < rate
      succ = f and rng.random() < 0.5
      rows.append({'contrast': name, 'kind': kind, 'role': '', 'target': t, 'seed_a': s, 'seed_b': s, 'top1_a': 0,
                   'top1_b': int(f), 'success_a': True, 'success_b': not succ, 'margin_a': 0.001, 'margin_b': 0.001,
                   'flip_top1': f, 'flip_success': succ, 'flip_matters': f and succ})
  return pd.DataFrame(rows)


def sample_pairs(sp):
  """One sample pair per seed pair, enough for contrasts.summarize."""
  return pd.DataFrame({'contrast': sp.contrast, 'kind': sp.kind, 'role': '', 'target': sp.target, 'seed_a': sp.seed_a,
                       'seed_b': sp.seed_b, 'sample': 0, 'bit_identical': False, 'm4_ligand_rmsd': 0.1,
                       'm4_ca_rmsd_global': 0.1, 'm4_error': None, 'm5_ranking_score': 0.0, 'm5_mean_plddt': 0.0,
                       'm5_iptm': 0.0})


def kind_of(name):
  return name.split('(')[0]


def build(names_rates, rng, **kw):
  sp = pd.concat([seed_pairs(n, kind_of(n), r, rng, **kw) for n, r in names_rates], ignore_index=True)
  cons = {n: (kind_of(n), True) for n, _ in names_rates}
  return sp, cons


@pytest.fixture(scope='module')
def tables():
  rng = np.random.default_rng(7)
  main_sp, main_cons = build([('HW(V5,V6)', 0.2), ('HW(V6,G)', 0.25), ('HW(G,C)', 0.2), ('HWh(V6,G)', 0.1),
                              ('HWh(G,C)', 0.1), ('RC(V5)', 0.0), ('RC(V6)', 0.0), ('RC(G)', 0.08)], rng)
  two = main_sp.contrast.str.startswith('HWh') & (main_sp.seed_a > 2)
  main_sp = main_sp[~two].reset_index(drop=True)          # section 11 arms: seeds 1 and 2
  gc = main_sp.contrast == 'HW(G,C)'
  main_sp = main_sp[~gc | main_sp.target.isin(TARGETS[:4])].reset_index(drop=True)   # C: a few targets
  cpu60_sp, cpu60_cons = build([('HW(G,C)', 0.2)], rng)
  repl_sp, repl_cons = build([('HW(V5,V6)', 0.2), ('HW(V6,G)', 0.2), ('RC(V5)', 0.0), ('RC(V6)', 0.0),
                              ('RC(G)', 0.07)], rng)
  prec5_sp, prec5_cons = build([('HWh(V6,G)', 0.1)], rng)
  a100_sp, _ = build([('HW(G,A)', 0.2)], rng)
  h100_sp, _ = build([('HW(H,A)', 0.2), ('RC(H)', 0.1)], rng)
  fu14_names = [(fu14.hw(*p), 0.01) for p in fu14.PAIRS] + [('RC(Gf)', 0.0), ('RC(V5f)', 0.0)]
  fu14_sp, fu14_cons = build(fu14_names, rng)
  pilot, pilot_cons = build([('RC(Ad)', 0.0), ('RC(Hd)', 0.0)], rng, targets=TARGETS[:4])
  fu14_sp = pd.concat([fu14_sp, pilot], ignore_index=True)
  fu14_cons.update(pilot_cons)
  rc_main = main_sp[main_sp.contrast.str.startswith('RC(')]
  t = {'main_sp': main_sp, 'main_hw': contrasts.hw_minus_rc(main_sp, main_cons, ARGS),
       'main_con': contrasts.summarize(main_sp, sample_pairs(main_sp), main_cons, ARGS),
       'cpu60_sp': cpu60_sp,
       'cpu60_hw': contrasts.hw_minus_rc(pd.concat([cpu60_sp, rc_main]), {**cpu60_cons, **{
           'RC(G)': ('RC', True)}}, ARGS),
       'repl_sp': repl_sp, 'repl_hw': contrasts.hw_minus_rc(repl_sp, repl_cons, ARGS).assign(targets='all 12'),
       'prec5_sp': prec5_sp,
       'prec5_hw': contrasts.hw_minus_rc(pd.concat([prec5_sp, rc_main]), {
           **prec5_cons, 'RC(V6)': ('RC', True), 'RC(G)': ('RC', True)}, ARGS).assign(targets='all 12', seeds='1 to 5'),
       'a100_sp': a100_sp, 'h100_sp': h100_sp, 'fu14_sp': fu14_sp,
       'fu14_con': contrasts.summarize(fu14_sp, sample_pairs(fu14_sp), fu14_cons, ARGS)}
  t['fu14_df'] = fu14.d_f_table(fu14_sp, {'main': main_sp, 'a100': a100_sp, 'h100': h100_sp}, ARGS)
  return t


# ---- the statistic reproduces the registered intervals

def test_single_rate_equals_boot_ci():
  sp = seed_pairs('HW(X,Y)', 'HW', 0.1, np.random.default_rng(1))
  groups = [sp[sp.target == t].flip_top1.astype(float).values for t in sorted(sp.target.unique())]
  lo, hi = contrasts.boot_ci(groups, contrasts.mean, np.random.default_rng(2026), REPS)
  r = ri.evaluate([(1, sp)], 'flip_top1', REPS, 2026)
  assert r['pct_low'] == pytest.approx(lo, abs=1e-12) and r['pct_high'] == pytest.approx(hi, abs=1e-12)
  assert r['estimate'] == pytest.approx(sp.flip_top1.mean(), abs=1e-12)


def test_difference_equals_hw_minus_rc():
  rng = np.random.default_rng(3)
  sp = pd.concat([seed_pairs('HW(X,Y)', 'HW', 0.2, rng), seed_pairs('RC(X)', 'RC', 0.05, rng, targets=TARGETS[2:])])
  reg = contrasts.hw_minus_rc(sp, {'HW(X,Y)': ('HW', True), 'RC(X)': ('RC', True)}, ARGS)
  for st in ri.FLIPS:
    g = reg[reg.statistic == st].iloc[0]
    r = ri.evaluate([(1, sp[sp.contrast == 'HW(X,Y)']), (-1, sp[sp.contrast == 'RC(X)'])], st, REPS, 2026)
    assert r['n_targets'] == len(TARGETS) - 2
    assert (r['estimate'], r['pct_low'], r['pct_high']) == pytest.approx((g.estimate, g.ci_low, g.ci_high), abs=1e-12)


def test_d_f_with_zero_term_equals_fu13():
  rng = np.random.default_rng(4)
  hwf, hw, rc = (seed_pairs(n, 'HW', p, rng) for n, p in (('a', 0.05), ('b', 0.2), ('c', 0.08)))
  for st in ri.FLIPS:
    reg = fu13.d_f(hwf, fu14.zero_like(hwf), hw, rc, st, 2026, REPS)
    r = ri.evaluate([(1, hwf), (-1, None), (-1, hw), (1, rc)], st, REPS, 2026)
    assert (r['estimate'], r['pct_low'], r['pct_high']) == pytest.approx(
        (reg['estimate'], reg['ci_low'], reg['ci_high']), abs=1e-12)


# ---- BCa

def test_bca_degenerate_without_events():
  sp = seed_pairs('RC(X)', 'RC', 0.0, np.random.default_rng(5))
  r = ri.evaluate([(1, sp)], 'flip_top1', REPS, 2026)
  assert (r['bca_low'], r['bca_high']) == (0.0, 0.0) and r['bca_note'].startswith('degenerate')
  assert r['targets_with_event'] == 0


def test_bca_undefined_when_one_side_empty():
  lo, hi, note = ri.bca(0.5, np.array([1.0, 2.0, 3.0]), np.array([1.0, 1.1]))
  assert lo != lo and hi != hi and note.startswith('undefined')


def test_acceleration_of_the_mean_is_the_closed_form():
  x = np.random.default_rng(6).exponential(size=40)
  n = len(x)
  jack = np.array([(x.sum() - xi) / (n - 1) for xi in x])
  d = x - x.mean()
  want = np.sum(d ** 3) / (6 * np.sum(d ** 2) ** 1.5)
  assert ri.acceleration(jack) == pytest.approx(want, rel=1e-10)


def test_bca_close_to_percentile_for_symmetric_data():
  rng = np.random.default_rng(8)
  sp = pd.DataFrame({'target': np.repeat([f't{i}' for i in range(200)], 5), 'v': rng.normal(size=1000)})
  r = ri.evaluate([(1, sp)], 'v', 4000, 2026)
  width = r['pct_high'] - r['pct_low']
  assert abs(r['bca_low'] - r['pct_low']) < 0.1 * width and abs(r['bca_high'] - r['pct_high']) < 0.1 * width


def test_bca_shifts_with_skew():
  """Few events: the replicate distribution is right-skewed; BCa moves both limits up against the percentile ones."""
  rng = np.random.default_rng(9)
  sp = seed_pairs('HW(X,Y)', 'HW', 0.0, rng, targets=[f't{i}' for i in range(60)])
  sp.loc[sp.index[:2], 'flip_top1'] = True          # two events in one target
  sp.loc[sp.index[100], 'flip_top1'] = True         # one in another
  r = ri.evaluate([(1, sp)], 'flip_top1', 4000, 2026)
  assert r['bca_high'] >= r['pct_high'] and r['bca_low'] >= r['pct_low']


def test_bca_uses_the_registered_draws():
  w = ri.replicates(5, 3, 2026)
  idx = np.random.default_rng(2026).integers(0, 5, size=(3, 5))
  assert np.array_equal(w, np.stack([np.bincount(row, minlength=5) for row in idx]).astype(float))


# ---- exact bound

def test_cp_upper_values():
  assert ri.cp_upper(0, 60) == pytest.approx(1 - 0.025 ** (1 / 60), abs=1e-9)
  assert ri.cp_upper(60, 60) == 1.0
  u = ri.cp_upper(3, 60)
  assert ri.binom_cdf(3, 60, u) == pytest.approx(0.025, abs=1e-9)
  assert math.isnan(ri.cp_upper(0, 0))


# ---- readings

def test_readings():
  assert ri.read('beyond', 0.001, 0.1) == 'beyond' and ri.read('beyond', 0.0, 0.1) == 'not beyond'
  assert ri.read('agree', 0.0, 0.049) == 'agree' and ri.read('agree', 0.0, 0.05) == 'agreement not shown'
  assert ri.read('reduced', -0.2, -0.01) == 'reduced' and ri.read('reduced', -0.2, 0.0) == 'not reduced'
  assert ri.read('beyond', np.nan, 0.1) == 'not computed' and ri.read('none', 0, 0) == ''


def test_criterion():
  assert ri.criterion({'events_1': 12, 'events_2': 9, 'registered_low': 0.02}) == 'yes'
  assert ri.criterion({'events_1': 40, 'events_2': 30, 'registered_low': 0.003}) == 'yes'
  assert ri.criterion({'events_1': 40, 'events_2': 30, 'registered_low': 0.0004}) == 'yes'   # 0.0 after rounding
  assert ri.criterion({'events_1': 40, 'events_2': 30, 'registered_low': 0.02}) == 'no'


# ---- end to end on synthetic registered tables

def test_run_reproduces_every_registered_row(tables, tmp_path):
  df = ri.run(tables, tmp_path, ARGS)
  assert (df.reproduced == 'yes').all(), df[df.reproduced != 'yes'][['quantity', 'statistic']]
  fam = df.family.str[:2].value_counts().to_dict()
  assert fam == {'F1': 10, 'F2': 2, 'F3': 8, 'F4': 15, 'F5': 8, 'F6': 10, 'F7': 18}
  assert set(df[df.rule == 'none'].changed) == {''}
  assert set(df[df.rule != 'none'].changed) <= {'yes', 'no'}
  assert (tmp_path / 'intervals.csv').is_file() and (tmp_path / 'report.md').read_text().startswith('# Rare-event')
  zero = df[(df.family.str.startswith('F7')) & (df.quantity == 'RC(V5)')]
  assert zero.cp_upper_targets.tolist() == pytest.approx([ri.cp_upper(0, len(TARGETS))] * 3)


def test_run_flags_a_row_that_does_not_reproduce(tables, tmp_path):
  bad = dict(tables)
  bad['cpu60_hw'] = tables['cpu60_hw'].assign(ci_low=tables['cpu60_hw'].ci_low + 0.01)
  df = ri.run(bad, tmp_path, ARGS)
  f2 = df[df.family.str.startswith('F2')]
  assert (f2.reproduced == 'NOT REPRODUCED').all() and (f2.bca_reading == 'not read').all()
  assert (df[~df.family.str.startswith('F2')].reproduced == 'yes').all()


def test_missing_registered_row_stops(tables, tmp_path):
  bad = dict(tables)
  bad['fu14_con'] = tables['fu14_con'][tables['fu14_con'].contrast != 'RC(Ad)']
  with pytest.raises(SystemExit):
    ri.run(bad, tmp_path, ARGS)
