"""Returned poses compared directly, further contrasts (drafts/section10_winners_ext_spec_20261116.md, label
winners_ext): the sides of DU and HW(G,C), the seeds of the main output, the sources kept apart, the checks and the
outputs of analysis/winners_ext.py on synthetic per-sample tables, with the seed pairs built by the registered code
(followup34.build) and a synthetic pose RMSD in place of the structure files. No real AlphaFold 3 output is read."""
import pathlib
import sys
import types

import numpy as np
import pandas as pd
import pytest

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'analysis'))
import followup34 as f34  # noqa: E402
import winners as w  # noqa: E402
import winners_ext as we  # noqa: E402

ARGS = types.SimpleNamespace(replicates=200, seed=2026, jobs=1)
ITEMS = {'main': ['HW(G,C)', 'DU'], 'cpu60': ['HW(G,C)']}
TARGETS = [f'T{i}' for i in range(5)]
SPECS = [('HW(G,C)', 'HW', ('G', 'r1'), ('C', 'r1'), True), ('DU', 'DU', ('G', 'r1'), ('GT', 'r1'), True)]
POS = {}


def fake_pose_rmsd(task):
  target, a, b = task
  if target == 'T4':
    return None, 'glycan: atom O1 missing'
  return abs(POS[a] - POS[b]), None


def samples(seed=0, c_targets=TARGETS):
  rng = np.random.default_rng(seed)
  rows = []
  for c, seeds, targets in (('G', (1, 2, 3, 4, 5), TARGETS), ('GT', (1, 2, 3, 4, 5), TARGETS), ('C', (1, 2), c_targets)):
    for t in targets:
      for s in seeds:
        for k in range(5):
          path = f'{c}/r1/{t}/{s}/{k}.cif'
          POS[path] = float(rng.normal(0, 3))
          exp = abs(POS[path])
          rows.append({'session': f'{c}-r1', 'label': 'main', 'condition': c, 'platform': c, 'repetition': 'r1',
                       'target': t, 'seed': s, 'sample': k, 'ranking_score': float(rng.random()),
                       'm1_ligand_rmsd': exp, 'm1_success': exp < 2.0, 'm1_pb_valid': rng.random() < 0.8,
                       'iptm': 0.9, 'mean_plddt': 90.0, 'model_path': path, 'num_recycles': 10,
                       'num_diffusion_samples': 5})
  return pd.DataFrame(rows)


@pytest.fixture()
def sources(monkeypatch):
  monkeypatch.setattr(w, 'POSE_RMSD', fake_pose_rmsd)
  main = f34.union(samples(0, TARGETS[:2]))
  c60 = f34.union(samples(1))                      # another draw: the follow-up 4 source has its own samples
  sp_main, _, _, _ = f34.build(main.assign(model_path=np.nan), SPECS)
  sp_c60, _, _, _ = f34.build(c60.assign(model_path=np.nan), SPECS[:1])
  return {'main': (main, sp_main), 'cpu60': (c60, sp_c60)}


def test_sides():
  assert we.sides('DU') == (('G', 'r1'), ('GT', 'r1'))
  assert we.sides('HW(G,C)') == (('G', 'r1'), ('C', 'r1'))
  assert we.sides('RC(GT)') == (('GT', 'r1'), ('GT', 'r2'))


def test_units_per_source_and_seeds(sources, tmp_path):
  df, s, chk = we.run(sources, tmp_path, ARGS, items=ITEMS)
  seed = df[df.level == 'seed']
  assert len(seed[(seed.source == 'main') & (seed.contrast == 'HW(G,C)')]) == 2 * 2
  assert len(seed[(seed.source == 'main') & (seed.contrast == 'DU')]) == 5 * 5
  assert len(seed[(seed.source == 'cpu60') & (seed.contrast == 'HW(G,C)')]) == 5 * 2
  tgt = df[df.level == 'target']
  gc = tgt[tgt.contrast == 'HW(G,C)']
  assert (gc.seeds == 2).all() and set(gc[gc.source == 'main'].target) == {'T0', 'T1'}
  assert (tgt[tgt.contrast == 'DU'].seeds == 5).all()
  # the main output of G in HW(G,C) is chosen among seeds 1 and 2 only
  main, _ = sources['cpu60']
  g = main[(main.condition == 'G') & (main.target == 'T2') & main.seed.isin([1, 2])]
  best = g.sort_values('ranking_score', ascending=False).iloc[0]
  r = gc[(gc.source == 'cpu60') & (gc.target == 'T2')].iloc[0]
  assert r.returned_a == str((int(best.seed), int(best['sample'])))


def test_du_side_b_is_gt(sources, tmp_path):
  df, s, chk = we.run(sources, tmp_path, ARGS, items=ITEMS)
  r = df[(df.contrast == 'DU') & (df.level == 'seed')].iloc[0]
  assert r.path_a.startswith('G/') and r.path_b.startswith('GT/')
  assert r.pose_rmsd == pytest.approx(abs(POS[r.path_a] - POS[r.path_b]))


def test_summary_kept_apart_by_source(sources, tmp_path):
  df, s, chk = we.run(sources, tmp_path, ARGS, items=ITEMS)
  rows = s[(s.contrast == 'HW(G,C)') & (s.level == 'seed') & (s.stratum == 'all')]
  assert set(rows.source) == {'main', 'cpu60'}
  u = df[(df.source == 'cpu60') & (df.level == 'seed')].dropna(subset=['pose_rmsd'])
  r = rows[rows.source == 'cpu60'].iloc[0]
  assert r.pose_rmsd_median == pytest.approx(np.median(u.pose_rmsd))
  assert r.units_with_pose_rmsd == 4 * 2                     # T4 has no pose RMSD
  assert len(chk) == 3 and set(zip(chk.source, chk.contrast)) == {('main', 'HW(G,C)'), ('main', 'DU'),
                                                                  ('cpu60', 'HW(G,C)')}
  text = (tmp_path / 'report.md').read_text()
  assert 'winners_ext' in text and '| cpu60 | HW(G,C) | seed | all |' in text
  for f in ('units_seed.csv', 'units_target.csv', 'summary.csv', 'checks.csv', 'errors.csv', 'report.md'):
    assert (tmp_path / f).is_file()


def test_check_2_stops_on_a_wrong_top1(sources, tmp_path):
  main, sp = sources['main']
  sp = sp.copy()
  i = sp.index[sp.contrast == 'DU'][0]
  sp.loc[i, 'top1_b'] = (sp.loc[i, 'top1_b'] + 1) % 5
  with pytest.raises(SystemExit, match='check 2'):
    we.run({'main': (main, sp)}, tmp_path, ARGS, items={'main': ['DU']})
