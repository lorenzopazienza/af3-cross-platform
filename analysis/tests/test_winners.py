"""Returned poses compared directly (drafts/section10_winners_spec_20261114.md, label winners): units, the main output
across seeds, the counts, the checks and the outputs of analysis/winners.py on synthetic per-sample tables, with the
seed pairs built by the registered code (followup34.build) and a synthetic pose RMSD in place of the structure files.
No real AlphaFold3 output is read."""
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

ARGS = types.SimpleNamespace(replicates=200, seed=2026, jobs=1)
TARGETS = [f'T{i}' for i in range(5)]
SPECS = [('HW(V5,V6)', 'HW', ('V5', 'r1'), ('V6', 'r1'), True), ('RC(G)', 'RC', ('G', 'r1'), ('G', 'r2'), True)]
ITEMS = {'main': ['HW(V5,V6)', 'RC(G)']}
POS = {}   # model path -> a 1-D "pose" coordinate; the synthetic pose RMSD is the distance between two of them


def fake_pose_rmsd(task):
  target, a, b = task
  if target == 'T4':
    return None, 'glycan: atom O1 missing'
  return abs(POS[a] - POS[b]), None


def samples(seed=0):
  rng = np.random.default_rng(seed)
  rows = []
  for c, reps in (('V5', ['r1']), ('V6', ['r1']), ('G', ['r1', 'r2'])):
    for rep in reps:
      for t in TARGETS:
        for s in (1, 2, 3):
          for k in range(5):
            path = f'{c}/{rep}/{t}/{s}/{k}.cif'
            POS[path] = float(rng.normal(0, 3))
            exp = abs(POS[path])
            rows.append({'session': f'{c}-{rep}', 'label': 'main', 'condition': c, 'platform': c, 'repetition': rep,
                         'target': t, 'seed': s, 'sample': k, 'ranking_score': float(rng.random()),
                         'm1_ligand_rmsd': exp, 'm1_success': exp < 2.0, 'm1_pb_valid': rng.random() < 0.8,
                         'iptm': 0.9, 'mean_plddt': 90.0, 'model_path': path, 'num_recycles': 10,
                         'num_diffusion_samples': 5})
  return pd.DataFrame(rows)


@pytest.fixture()
def source(monkeypatch):
  monkeypatch.setattr(w, 'POSE_RMSD', fake_pose_rmsd)
  ok = f34.union(samples())
  sp, _, _, _ = f34.build(ok.assign(model_path=np.nan), SPECS)   # build reads no structure file
  return ok, sp


def test_main_output_rule():
  g = pd.DataFrame({'seed': [1, 1, 2, 2], 'sample': [0, 1, 0, 1], 'ranking_score': [0.5, 0.9, 0.9, 0.1]})
  r = w.main_output(g)
  assert (r.seed, r['sample']) == (1, 1)          # exact tie at 0.9: the lower seed


def test_units_and_measures(source, tmp_path):
  ok, sp = source
  df, s, chk = w.run({'main': (ok, sp)}, tmp_path, ARGS, items=ITEMS)
  seed = df[(df.level == 'seed') & (df.contrast == 'HW(V5,V6)')]
  assert len(seed) == len(TARGETS) * 3
  r = seed[seed.target == 'T0'].iloc[0]
  assert r.pose_rmsd == pytest.approx(abs(POS[r.path_a] - POS[r.path_b]))
  assert r.distinct == (r.pose_rmsd > 2.0)
  lo, hi = sorted([r.rmsd_exp_a, r.rmsd_exp_b])
  assert r.binding_mode_change == (r.pose_rmsd > 2.0 and lo <= 2.0 and hi > 4.0)
  tgt = df[(df.level == 'target') & (df.contrast == 'HW(V5,V6)')]
  assert len(tgt) == len(TARGETS) and (tgt.seeds == 3).all()
  g = ok[(ok.condition == 'V5') & (ok.target == 'T1')]
  best = g.sort_values('ranking_score', ascending=False).iloc[0]
  assert tgt[tgt.target == 'T1'].iloc[0].returned_a == str((int(best.seed), int(best['sample'])))


def test_glycan_like_errors_are_listed(source, tmp_path):
  ok, sp = source
  df, s, chk = w.run({'main': (ok, sp)}, tmp_path, ARGS, items=ITEMS)
  assert df[df.target == 'T4'].pose_rmsd.isna().all()
  e = pd.read_csv(tmp_path / 'errors.csv')
  assert set(e.target) == {'T4'}
  row = s[(s.contrast == 'HW(V5,V6)') & (s.level == 'seed') & (s.stratum == 'all')].iloc[0]
  assert row.units_with_pose_rmsd == 4 * 3 and row.distinct_of == 4 * 3


def test_summary_statistics(source, tmp_path):
  ok, sp = source
  df, s, chk = w.run({'main': (ok, sp)}, tmp_path, ARGS, items=ITEMS)
  u = df[(df.contrast == 'RC(G)') & (df.level == 'seed')].dropna(subset=['pose_rmsd'])
  row = s[(s.contrast == 'RC(G)') & (s.level == 'seed') & (s.stratum == 'all')].iloc[0]
  assert row.pose_rmsd_median == pytest.approx(np.median(u.pose_rmsd))
  assert row.pose_rmsd_p90 == pytest.approx(np.percentile(u.pose_rmsd, 90))
  assert row.pose_rmsd_max == pytest.approx(u.pose_rmsd.max())
  assert row.distinct_n == int((u.pose_rmsd > 2).sum())
  ch = s[(s.contrast == 'RC(G)') & (s.level == 'seed') & (s.stratum == 'changed')].iloc[0]
  assert ch.units == int(df[(df.contrast == 'RC(G)') & (df.level == 'seed')].changed.sum())
  for f in ('units_seed.csv', 'units_target.csv', 'summary.csv', 'checks.csv', 'errors.csv', 'report.md'):
    assert (tmp_path / f).is_file()


def test_check_3_compares_with_the_earlier_counts(source, tmp_path):
  ok, sp = source
  df, s, chk = w.run({'main': (ok, sp)}, tmp_path, ARGS, items=ITEMS)
  r = chk[chk.contrast == 'HW(V5,V6)'].iloc[0]
  assert 'earlier count 16' in r.detail and r.value.endswith(f'of {len(TARGETS)} targets')


def test_check_2_stops_on_a_wrong_top1(source, tmp_path):
  ok, sp = source
  sp = sp.copy()
  i = sp.index[sp.contrast == 'RC(G)'][0]
  sp.loc[i, 'top1_b'] = (sp.loc[i, 'top1_b'] + 1) % 5
  with pytest.raises(SystemExit, match='check 2'):
    w.run({'main': (ok, sp)}, tmp_path, ARGS, items=ITEMS)


def test_identical_paths_give_zero():
  assert w.pose_rmsd_real(('T', 'same.cif', 'same.cif')) == (0.0, None)
