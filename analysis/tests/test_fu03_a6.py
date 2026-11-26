"""Follow-up 3, the section 11 reading of PR(A), HWh(V6,A) and HWh(A,C) (analysis/fu03_a6.py): the checks, the
differences and the readings on synthetic per-sample tables, with the registered tables built by the registered code
(followup34.build, contrasts.summarize, contrasts.hw_minus_rc). No real AlphaFold 3 output is read."""
import pathlib
import sys
import types

import numpy as np
import pandas as pd
import pytest

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'analysis'))
import contrasts  # noqa: E402
import followup34 as f34  # noqa: E402
import fu03_a6  # noqa: E402

ARGS = types.SimpleNamespace(replicates=200, seed=2026)
TARGETS = [f'T{i:02d}' for i in range(fu03_a6.N_TARGETS)]
# (condition, repetition): seeds; r2 copies r1 (V6 exactly, A and G with noise), every other arm draws its own scores.
ARMS = {('A', 'r1'): fu03_a6.S15, ('A', 'r2'): fu03_a6.S15, ('V6', 'r1'): fu03_a6.S15, ('V6', 'r2'): fu03_a6.S15,
        ('G', 'r1'): fu03_a6.S15, ('G', 'r2'): fu03_a6.S15, ('Ah', 'r1'): fu03_a6.S12, ('V6h', 'r1'): fu03_a6.S12,
        ('Gh', 'r1'): fu03_a6.S12, ('C', 'r1'): fu03_a6.S12}
A100_SPECS = [('PR(A)', 'PR', ('A', 'r1'), ('Ah', 'r1'), False), ('HWh(V6,A)', 'HWh', ('V6h', 'r1'), ('Ah', 'r1'), False),
              ('HWh(A,C)', 'HWh', ('Ah', 'r1'), ('C', 'r1'), False), ('HW(V6,A)', 'HW', ('V6', 'r1'), ('A', 'r1'), False),
              ('HW(A,C)', 'HW', ('A', 'r1'), ('C', 'r1'), False), ('RC(A)', 'RC', ('A', 'r1'), ('A', 'r2'), True)]
MAIN_SPECS = [('RC(V6)', 'RC', ('V6', 'r1'), ('V6', 'r2'), True), ('RC(G)', 'RC', ('G', 'r1'), ('G', 'r2'), True),
              ('HWh(V6,G)', 'HWh', ('V6h', 'r1'), ('Gh', 'r1'), False)]


def samples(seed=0, noise=0.02):
  rng = np.random.default_rng(seed)
  base, rows = {}, []
  for (c, rep), seeds in ARMS.items():
    for t in TARGETS:
      for s in seeds:
        for k in range(5):
          if rep == 'r1':
            base[(c, t, s, k)] = (float(rng.random()), bool(rng.random() < 0.3))
            score, succ = base[(c, t, s, k)]
          else:
            r1 = base[(c, t, s, k)]
            score, succ = (r1[0] if c == 'V6' else r1[0] + float(rng.normal(0, noise))), r1[1]
          rows.append({'session': f'results/af3/20261018T000000Z_{c}_{rep}/run', 'condition': c, 'platform': c,
                       'repetition': rep, 'target': t, 'seed': s, 'sample': k, 'ranking_score': score,
                       'm1_success': succ, 'mean_plddt': 90.0, 'iptm': 0.9, 'model_path': np.nan})
  return pd.DataFrame(rows)


def registered_tables(ok):
  """The registered outputs, built by the registered code."""
  out = {}
  for src, specs in (('a100', A100_SPECS), ('main', MAIN_SPECS)):
    sp, sm, cons, _ = f34.build(ok, specs)
    out[f'{src}_sp'] = sp
    out[f'{src}_con'] = contrasts.summarize(sp, sm, cons, ARGS)
  a_sp, m_sp = out['a100_sp'], out['main_sp']
  out['a100_hw'] = contrasts.hw_minus_rc(
      pd.concat([a_sp[a_sp.contrast == 'HW(V6,A)'], a_sp[a_sp.contrast == 'RC(A)'], m_sp[m_sp.contrast == 'RC(V6)']]),
      {'HW(V6,A)': ('HW', False), 'RC(A)': ('RC', True), 'RC(V6)': ('RC', True)}, ARGS)
  out['main_hw'] = contrasts.hw_minus_rc(m_sp, {'HWh(V6,G)': ('HWh', False), 'RC(V6)': ('RC', True),
                                                'RC(G)': ('RC', True)}, ARGS)
  con = out['a100_con']
  t = []
  for name in fu03_a6.T22B_ROWS:
    row = {'contrast': name, 'targets': str(fu03_a6.N_TARGETS)}
    for st, col in fu03_a6.T22B_COLS.items():
      r = con[(con.contrast == name) & (con.statistic == st)].iloc[0]
      row[col] = fu03_a6.cell(r)
    t.append(row)
  out['t22b'] = pd.DataFrame(t)
  return out


@pytest.fixture(scope='module')
def data():
  ok = f34.union(samples())
  return ok, registered_tables(ok)


def copy(tables):
  return {k: v.copy() for k, v in tables.items()}


def test_run_passes_and_writes(data, tmp_path):
  ok, tables = data
  rt, res, chk = fu03_a6.run(ok, copy(tables), tmp_path, ARGS)
  assert all(c['value'].startswith('pass') for c in chk)
  assert {c['check'][0] for c in chk} == {'1', '2', '3', '4', '5'}
  assert len(res) == 3 * len(fu03_a6.ROWS)
  for f in ('rates.csv', 'results.csv', 'checks.csv', 'report.md'):
    assert (tmp_path / f).exists()
  # RC(V6) is 0 by construction (r2 copies r1), and V6h and Ah draw their own scores: above the RC level.
  p1 = res[(res.id == 'P1') & (res.statistic == 'flip_top1')].iloc[0]
  assert p1.position == 'above the RC level' and p1.reading == 'attributed to the platform'
  rc = rt[(rt.contrast == 'RC(V6)')]
  assert (rc.estimate == 0).all()


def test_seed_sets_of_the_references(data, tmp_path):
  ok, tables = data
  _, res, _ = fu03_a6.run(ok, copy(tables), tmp_path, ARGS)
  first = res.drop_duplicates('id').set_index('id')
  assert first.loc['P1', 'pairs_x'] == 120 and first.loc['P1', 'pairs_y'] == 300
  assert first.loc['P2', 'pairs_y'] == 300 and first.loc['Ha2', 'pairs_y'] == 120
  assert first.loc['R1', 'pairs_x'] == 300 and first.loc['Hd1', 'pairs_x'] == 120
  assert (res[res.id == 'Hb3'].reading == 'not computed').all()


def test_plan_rows_equal_hw_minus_rc(data, tmp_path):
  ok, tables = data
  _, res, _ = fu03_a6.run(ok, copy(tables), tmp_path, ARGS)
  sp = fu03_a6.registered_pairs(tables)
  ref = fu03_a6.plan_rows_hw_minus_rc(sp, ARGS)
  ref = ref[ref.reference == 'RC(P)'].set_index(['hardware', 'statistic'])
  for r in res[res.label == 'plan'].itertuples():
    h = ref.loc[(r.quantity, r.statistic)]
    assert abs(h.estimate - r.estimate) < 1e-12 and abs(h.ci_low - r.ci_low) < 1e-12 and abs(h.ci_high - r.ci_high) < 1e-12


@pytest.mark.parametrize('lo,hi,pos,rule', [(0.01, 0.05, 'above the RC level', 'attributed to the platform'),
                                            (0.0, 0.04, 'at the RC level', 'attributed to the default matmul precision'),
                                            (-0.02, 0.04, 'at the RC level', 'attributed to the default matmul precision'),
                                            (-0.05, -0.01, 'below the RC level',
                                             'attributed to the default matmul precision'),
                                            (np.nan, np.nan, 'not computed', 'not computed')])
def test_position_and_rule(lo, hi, pos, rule):
  assert fu03_a6.position(lo, hi) == pos
  assert fu03_a6.reading('rule', lo, hi) == rule


def test_other_readings():
  assert fu03_a6.reading('pr', -0.04, 0.09) == 'not distinguished from recompilation'
  assert fu03_a6.reading('hw', 0.01, 0.2) == 'a difference beyond recompilation'
  assert fu03_a6.reading('shrink', -0.13, -0.01) == 'smaller at default precision'
  assert fu03_a6.position(-0.1, 0.02, '0 (no shrink shown)') == 'at 0 (no shrink shown)'
  with pytest.raises(ValueError):
    fu03_a6.reading('other', 0.0, 0.1)


def test_t22b_mismatch_stops_on_that_contrast(data, tmp_path):
  ok, tables = data
  t = copy(tables)
  t['t22b'].loc[t['t22b'].contrast == 'HWh(A,C)', 'success flip (%)'] = '9.9 [0.0, 9.9]'
  with pytest.raises(SystemExit, match=r'check 4: HWh\(A,C\) differs .*stop on this contrast'):
    fu03_a6.run(ok, t, tmp_path, ARGS)


def test_top1_mismatch_stops(data, tmp_path):
  ok, tables = data
  t = copy(tables)
  sp = t['a100_sp']
  i = sp.index[sp.contrast == 'HWh(V6,A)'][0]
  sp.loc[i, 'top1_b'] = (int(sp.loc[i, 'top1_b']) + 1) % 5
  with pytest.raises(SystemExit, match=r'check 2: HWh\(V6,A\) side B'):
    fu03_a6.run(ok, t, tmp_path, ARGS)


def test_missing_seed_stops(data, tmp_path):
  ok, tables = data
  t = copy(tables)
  sp = t['a100_sp']
  t['a100_sp'] = sp.drop(sp.index[(sp.contrast == 'PR(A)') & (sp.seed_a == 2)][:1])
  with pytest.raises(SystemExit, match=r'check 1: PR\(A\)'):
    fu03_a6.run(ok, t, tmp_path, ARGS)


def test_registered_rate_mismatch_stops(data, tmp_path):
  ok, tables = data
  t = copy(tables)
  con = t['a100_con']
  con.loc[(con.contrast == 'RC(A)') & (con.statistic == 'flip_top1'), 'estimate'] += 0.001
  with pytest.raises(SystemExit, match=r'check 3: the registered RC\(A\) flip_top1'):
    fu03_a6.run(ok, t, tmp_path, ARGS)


def test_registered_hw_minus_rc_mismatch_stops(data, tmp_path):
  ok, tables = data
  t = copy(tables)
  hw = t['main_hw']
  hw.loc[hw.index[0], 'ci_low'] -= 0.01
  with pytest.raises(SystemExit, match=r'check 5'):
    fu03_a6.run(ok, t, tmp_path, ARGS)
