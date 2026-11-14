"""Returned structures across platforms (drafts/section10_returned_spec_20261114.md, label returned): units, checks,
statistics and readings of analysis/returned_structures.py on synthetic per-sample tables, with the seed pairs built
by the registered code (followup34.build). No real AlphaFold3 output is read."""
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
import returned_structures as rs  # noqa: E402

ARGS = types.SimpleNamespace(replicates=200, seed=2026)
TARGETS = [f'T{i}' for i in range(6)]
SPECS = [('HW(V5,V6)', 'HW', ('V5', 'r1'), ('V6', 'r1'), True), ('HW(V6,G)', 'HW', ('V6', 'r1'), ('G', 'r1'), True),
         ('HW(G,C)', 'HW', ('G', 'r1'), ('C', 'r1'), True), ('RC(G)', 'RC', ('G', 'r1'), ('G', 'r2'), True),
         ('SD(V6)', 'SD', ('V6', 'r1'), ('V6', 'r1'), True)]
ITEMS = {'main': [(n, 'test') for n, *_ in SPECS]}


def samples(seed=0):
  rng = np.random.default_rng(seed)
  rows = []
  for c, reps in (('V5', ['r1']), ('V6', ['r1']), ('G', ['r1', 'r2']), ('C', ['r1'])):
    for rep in reps:
      for t in TARGETS:
        for s in (1, 2):
          for k in range(5):
            rmsd = float(rng.gamma(2.0, 1.0))
            rows.append({'session': f'{c}-{rep}', 'label': 'main', 'condition': c, 'platform': c, 'repetition': rep,
                         'target': t, 'seed': s, 'sample': k, 'ranking_score': float(rng.random()),
                         'm1_ligand_rmsd': rmsd, 'm1_success': rmsd < 2.0, 'iptm': float(rng.random()),
                         'mean_plddt': float(60 + 30 * rng.random()), 'model_path': np.nan,
                         'num_recycles': 10, 'num_diffusion_samples': 5})
  return pd.DataFrame(rows)


@pytest.fixture()
def source():
  ok = f34.union(samples())
  sp, _, _, _ = f34.build(ok, SPECS)
  return ok, sp


def test_sides():
  assert rs.sides('HW(V5f,V6f)') == (('V5f', 'r1'), ('V6f', 'r1'))
  assert rs.sides('RC(Gf)') == (('Gf', 'r1'), ('Gf', 'r2'))
  assert rs.sides('SD(V6)') == (('V6', 'r1'), ('V6', 'r1'))
  with pytest.raises(ValueError):
    rs.sides('F(V5)')


def test_units_hold_the_registered_top1_metrics(source, tmp_path):
  ok, sp = source
  units, q = rs.run({'main': (ok, sp)}, tmp_path, ARGS, items=ITEMS)
  assert len(units) == len(sp)
  u = units[(units.contrast == 'HW(V6,G)')].iloc[0]
  g = ok[(ok.condition == 'V6') & (ok.repetition == 'r1') & (ok.target == u.target) & (ok.seed == u.seed_a)]
  h = ok[(ok.condition == 'G') & (ok.repetition == 'r1') & (ok.target == u.target) & (ok.seed == u.seed_b)]
  a, b = g.loc[g.ranking_score.idxmax()], h.loc[h.ranking_score.idxmax()]
  assert (u.top1_a, u.top1_b) == (a['sample'], b['sample'])
  assert u.d_ligand_rmsd == pytest.approx(b.m1_ligand_rmsd - a.m1_ligand_rmsd)
  assert u.d_iptm == pytest.approx(b.iptm - a.iptm) and u.d_mean_plddt == pytest.approx(b.mean_plddt - a.mean_plddt)
  sd = units[units.contrast == 'SD(V6)']
  assert len(sd) == len(TARGETS) and (sd.seed_a != sd.seed_b).all()


def test_strata_and_statistics(source, tmp_path):
  ok, sp = source
  units, q = rs.run({'main': (ok, sp)}, tmp_path, ARGS, items=ITEMS)
  assert set(q[q.contrast == 'SD(V6)'].stratum) == {'all'}
  assert set(q[q.contrast == 'HW(V5,V6)'].stratum) == {'all', 'flipped'}
  assert len(q) == 4 * 2 * 3 + 1 * 3
  row = q[(q.contrast == 'HW(V5,V6)') & (q.stratum == 'all') & (q.metric == 'ligand_rmsd')].iloc[0]
  u = units[units.contrast == 'HW(V5,V6)']
  assert row['median'] == pytest.approx(np.median(u.d_ligand_rmsd))
  groups = [u[u.target == t].d_ligand_rmsd.values for t in sorted(u.target.unique())]
  lo, hi = contrasts.boot_ci(groups, contrasts.med, np.random.default_rng(2026), ARGS.replicates)
  assert (row.median_ci_low, row.median_ci_high) == pytest.approx((lo, hi))
  fl = q[(q.contrast == 'HW(V5,V6)') & (q.stratum == 'flipped') & (q.metric == 'ligand_rmsd')].iloc[0]
  assert fl.n_units == int(u.flip_top1.sum())
  for f in ('units.csv', 'quality.csv', 'checks.csv', 'report.md'):
    assert (tmp_path / f).is_file()


def test_missing_rmsd_is_counted(source, tmp_path):
  ok, sp = source
  ok = ok.copy()
  r = sp[sp.contrast == 'HW(V5,V6)'].iloc[0]
  hit = (ok.condition == 'V6') & (ok.repetition == 'r1') & (ok.target == r.target) & (ok.seed == r.seed_b) & \
      (ok['sample'] == r.top1_b)
  ok.loc[hit, 'm1_ligand_rmsd'] = np.nan
  units, q = rs.run({'main': (ok, sp)}, tmp_path, ARGS, items=ITEMS)
  row = q[(q.contrast == 'HW(V5,V6)') & (q.stratum == 'all') & (q.metric == 'ligand_rmsd')].iloc[0]
  assert row.units_without_metric == 1
  chk = pd.read_csv(tmp_path / 'checks.csv')
  assert int(chk[(chk.contrast == 'HW(V5,V6)') & (chk.metric == 'ligand_rmsd') & (chk.side == 'b')].missing.iloc[0]) == 1


def test_check_2_stops_on_a_wrong_top1(source, tmp_path):
  ok, sp = source
  sp = sp.copy()
  i = sp.index[sp.contrast == 'HW(V6,G)'][0]
  sp.loc[i, 'top1_a'] = (sp.loc[i, 'top1_a'] + 1) % 5
  with pytest.raises(SystemExit, match='check 2'):
    rs.run({'main': (ok, sp)}, tmp_path, ARGS, items=ITEMS)


def test_check_1_stops_on_a_missing_sample(source, tmp_path):
  ok, sp = source
  r = sp[sp.contrast == 'HW(G,C)'].iloc[0]
  drop = (ok.condition == 'C') & (ok.target == r.target) & (ok.seed == r.seed_b) & (ok['sample'] == r.top1_b)
  with pytest.raises(SystemExit, match='check 1'):
    rs.run({'main': (ok[~drop], sp)}, tmp_path, ARGS, items=ITEMS)


def test_contrast_without_pairs_stops(source, tmp_path):
  ok, sp = source
  with pytest.raises(SystemExit):
    rs.run({'main': (ok, sp[sp.contrast != 'RC(G)'])}, tmp_path, ARGS, items=ITEMS)


def test_reading():
  assert rs.reading(0.01, 0.05, 0.1) == 'B higher; within 0.1'
  assert rs.reading(-0.3, -0.01, 0.1) == 'B lower'
  assert rs.reading(-0.05, 0.05, 0.1) == 'not distinguishable; within 0.1'
  assert rs.reading(-0.2, 0.05, 0.1) == 'not distinguishable'
  assert rs.reading(-0.2, 0.05) == 'not distinguishable'
  assert rs.reading(np.nan, 0.05, 0.1) == 'not computed'


def test_default_contrasts_cover_the_prompt():
  names = {c for items in rs.CONTRASTS.values() for c, _ in items}
  assert {'HW(V5,V6)', 'HW(V6,G)', 'HW(G,C)'} <= names
  assert len([c for c, _ in rs.CONTRASTS['fu14'] if c.startswith('HW(')]) == 10
