"""Success of the best of all samples (drafts/section10_best25_spec_20261116.md, label best25): the oracle and main-output
measures, the seed sets, the differences and discordant counts, and the checks of analysis/best25.py on synthetic
per-sample tables, with the seed pairs built by the registered code (followup34.build). No real AlphaFold 3 output is
read."""
import pathlib
import sys
import types

import numpy as np
import pandas as pd
import pytest

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'analysis'))
import best25 as b  # noqa: E402
import followup34 as f34  # noqa: E402

ARGS = types.SimpleNamespace(replicates=200, seed=2026)
TARGETS = [f'T{i}' for i in range(6)]
S5, S2 = b.S5, b.S2
SPECS = [('HW(V5,V6)', 'platform', ('V5', 'r1'), ('V6', 'r1'), S5, 'main', 'HW(V5,V6)'),
         ('HW(G,C)', 'platform', ('G', 'r1'), ('C', 'r1'), S2, 'main', 'HW(G,C)'),
         ('RC(G)', 'recompilation', ('G', 'r1'), ('G', 'r2'), S5, 'main', 'RC(G)'),
         ('RC(G) seeds 1-2', 'recompilation', ('G', 'r1'), ('G', 'r2'), S2, 'main', 'RC(G)')]


def samples(seed=0):
  rng = np.random.default_rng(seed)
  rows = []
  for c, reps, seeds in (('V5', ['r1'], S5), ('V6', ['r1'], S5), ('G', ['r1', 'r2'], S5), ('C', ['r1'], S2)):
    for rep in reps:
      for t in TARGETS:
        for s in seeds:
          for k in range(5):
            succ = rng.random() < 0.25
            rows.append({'session': f'{c}-{rep}', 'label': 'main', 'condition': c, 'platform': c, 'repetition': rep,
                         'target': t, 'seed': s, 'sample': k, 'ranking_score': float(rng.random()),
                         'm1_success': succ if t != 'T5' else np.nan, 'm1_ligand_rmsd': 1.0, 'iptm': 0.9,
                         'mean_plddt': 90.0, 'model_path': np.nan, 'num_recycles': 10, 'num_diffusion_samples': 5})
  return pd.DataFrame(rows)


@pytest.fixture()
def data():
  ok = f34.union(samples())
  sp, _, _, _ = f34.build(ok, [('HW(V5,V6)', 'HW', ('V5', 'r1'), ('V6', 'r1'), True),
                               ('HW(G,C)', 'HW', ('G', 'r1'), ('C', 'r1'), True),
                               ('RC(G)', 'RC', ('G', 'r1'), ('G', 'r2'), True)])
  q3 = {'main': pd.DataFrame(columns=['n_seeds', 'condition', 'statistic', 'estimate']),
        'a100': pd.DataFrame(columns=['n_seeds', 'condition', 'statistic', 'estimate'])}
  return ok, {'main': sp}, q3


def test_measures(data, tmp_path):
  ok, pairs, q3 = data
  units, r, d, chk = b.run(ok, pairs, q3, tmp_path, ARGS, specs=SPECS)
  g = ok[(ok.condition == 'V6') & (ok.repetition == 'r1') & (ok.target == 'T1')]
  u = units[(units.contrast == 'HW(V5,V6)') & (units.level == 'target') & (units.target == 'T1')]
  assert u[u.measure == 'oracle'].b.iloc[0] == float((g.m1_success == True).any())  # noqa: E712
  best = g.sort_values(['ranking_score'], ascending=False).iloc[0]
  assert u[u.measure == 'main'].b.iloc[0] == float(best.m1_success == True)  # noqa: E712
  s = units[(units.contrast == 'HW(V5,V6)') & (units.level == 'seed') & (units.measure == 'oracle')]
  assert len(s) == len(TARGETS) * 5
  gs = g[g.seed == 3]
  assert s[(s.target == 'T1') & (s.seed == 3)].b.iloc[0] == float((gs.m1_success == True).any())  # noqa: E712
  # T5 has no M1 success anywhere: no success on any side, as registered
  assert (units[units.target == 'T5'][['a', 'b']] == 0).all().all()


def test_seed_sets(data, tmp_path):
  ok, pairs, q3 = data
  units, r, d, chk = b.run(ok, pairs, q3, tmp_path, ARGS, specs=SPECS)
  gc = units[(units.contrast == 'HW(G,C)') & (units.level == 'seed')]
  assert set(gc.seed) == {1, 2}
  g = ok[(ok.condition == 'G') & (ok.repetition == 'r1') & (ok.target == 'T2') & ok.seed.isin([1, 2])]
  t = units[(units.contrast == 'HW(G,C)') & (units.level == 'target') & (units.measure == 'oracle') & (units.target == 'T2')]
  assert t.a.iloc[0] == float((g.m1_success == True).any())  # noqa: E712
  assert set(r[r.side == 'G_r1'].seeds) == {'1-2-3-4-5', '1-2'}


def test_differences_and_discordance(data, tmp_path):
  ok, pairs, q3 = data
  units, r, d, chk = b.run(ok, pairs, q3, tmp_path, ARGS, specs=SPECS)
  u = units[(units.contrast == 'RC(G)') & (units.level == 'target') & (units.measure == 'oracle')]
  row = d[(d.contrast == 'RC(G)') & (d.level == 'target') & (d.measure == 'oracle')].iloc[0]
  assert row.difference == pytest.approx((u.b - u.a).mean())
  assert row.a_only == int(((u.a == 1) & (u.b == 0)).sum()) and row.b_only == int(((u.a == 0) & (u.b == 1)).sum())
  assert row.discordant == pytest.approx((u.a != u.b).mean())
  assert row.difference_ci_low <= row.difference <= row.difference_ci_high
  rr = r[(r.side == 'V5_r1') & (r.level == 'seed') & (r.measure == 'top1')].iloc[0]
  assert rr.rate == pytest.approx(pairs['main'][pairs['main'].contrast == 'HW(V5,V6)'].success_a.astype(float).mean())
  for f in ('units.csv', 'rates.csv', 'differences.csv', 'checks.csv', 'report.md'):
    assert (tmp_path / f).is_file()


def test_check_2_stops(data, tmp_path):
  ok, pairs, q3 = data
  sp = pairs['main'].copy()
  i = sp.index[sp.contrast == 'RC(G)'][0]
  sp.loc[i, 'success_b'] = not sp.loc[i, 'success_b']
  with pytest.raises(SystemExit, match='check 2'):
    b.run(ok, {'main': sp}, q3, tmp_path, ARGS, specs=SPECS)


def test_check_1_stops_on_a_missing_sample(data, tmp_path):
  ok, pairs, q3 = data
  drop = ok[(ok.condition == 'V6') & (ok.target == 'T0') & (ok.seed == 4) & (ok['sample'] == 2)].index
  with pytest.raises(SystemExit, match='check 1'):
    b.run(ok.drop(drop), pairs, q3, tmp_path, ARGS, specs=SPECS)


def test_q3_check(data, tmp_path):
  ok, pairs, q3 = data
  g = ok[(ok.condition == 'V5') & (ok.repetition == 'r1')]
  rate = np.mean([float(x.sort_values('ranking_score', ascending=False).iloc[0].m1_success == True)  # noqa: E712
                  for _, x in g.groupby('target')])
  q3 = dict(q3, main=pd.DataFrame([{'n_seeds': 5, 'condition': 'V5', 'statistic': 'success_rate', 'estimate': rate},
                                   {'n_seeds': 5, 'condition': 'V6', 'statistic': 'success_rate', 'estimate': -1.0}]))
  units, r, d, chk = b.run(ok, pairs, q3, tmp_path, ARGS, specs=SPECS)
  v = {c['side']: c['value'] for c in chk if c['check'].startswith('3')}
  assert v['V5_r1'] == 'equal' and v['V6_r1'] == 'DIFFERS' and v['G_r1'] == 'no registered row'
