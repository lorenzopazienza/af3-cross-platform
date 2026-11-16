"""Which criterion changes in a success flip (drafts/section10_pb_flips_spec_20261116.md, label pb_flips): the
thresholds against posebusters' dock configuration, the excess, the criterion and the class of each flip, the
reproduction check that stops the analysis, and the outputs of analysis/pb_flips.py, on synthetic per-sample tables
with the seed pairs built by the registered code (followup34.build) and synthetic rescoring and pose functions in place
of the structure files. No real AlphaFold 3 output is read."""
import pathlib
import sys
import types

import numpy as np
import pandas as pd
import pytest
import yaml

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'analysis'))
import followup34 as f34  # noqa: E402
import pb_flips as pf  # noqa: E402
import winners as w  # noqa: E402

ARGS = types.SimpleNamespace(replicates=200, seed=2026, jobs=1)
TARGETS = [f'T{i}' for i in range(8)]
SPECS = [('HW(V5,V6)', 'HW', ('V5', 'r1'), ('V6', 'r1'), True), ('HW(V6,G)', 'HW', ('V6', 'r1'), ('G', 'r1'), True),
         ('RC(G)', 'RC', ('G', 'r1'), ('G', 'r2'), True)]
ITEMS = {'main': ['HW(V5,V6)', 'HW(V6,G)', 'RC(G)']}
PB = ['minimum_distance_to_protein', 'bond_lengths', 'internal_energy']
POS, EXC = {}, {}      # path -> 1-D pose; path -> {check: excess}


def fake_pose(task):
  t, a, b = task
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
            POS[path] = float(rng.normal(0, 1.5))
            rmsd = round(abs(POS[path]) + 0.5, 4)
            failed = [PB[int(rng.integers(0, 3))]] if rng.random() < 0.3 else []
            EXC[path] = {ch: (float(rng.uniform(0.01, 0.3)) if ch in failed else -float(rng.uniform(0.01, 0.5)))
                         for ch in pf.CHECKS}
            rows.append({'session': f'{c}-{rep}', 'label': 'main', 'condition': c, 'platform': c, 'repetition': rep,
                         'target': t, 'seed': s, 'sample': k, 'ranking_score': float(rng.random()),
                         'm1_ligand_rmsd': rmsd, 'm1_pb_valid': not failed, 'm1_pb_failed': ';'.join(failed) or np.nan,
                         'm1_success': rmsd < 2.0 and not failed, 'm1_error': np.nan, 'iptm': 0.9, 'mean_plddt': 90.0,
                         'model_path': path, 'num_recycles': 10, 'num_diffusion_samples': 5})
  return pd.DataFrame(rows)


def fake_rescore(task, table=None):
  t, path = task
  x = table.loc[path]
  failed = x.m1_pb_failed if isinstance(x.m1_pb_failed, str) else ''
  out = {'target': t, 'path': path, 're_ligand_rmsd': x.m1_ligand_rmsd, 're_pb_valid': bool(x.m1_pb_valid),
         're_pb_failed': failed, 're_success': bool(x.m1_success), 're_error': None, 'std_failed': failed,
         'full_agrees': True}
  for c in pf.CHECKS:
    out[f'excess:{c}'] = EXC[path][c]
    out[f'pass:{c}'] = c not in failed.split(';')
  return out


@pytest.fixture()
def source(monkeypatch):
  ok = f34.union(samples())
  table = ok.set_index('model_path', drop=False)
  monkeypatch.setattr(w, 'POSE_RMSD', fake_pose)
  monkeypatch.setattr(pf, 'RESCORE', lambda task: fake_rescore(task, table))
  sp, _, _, _ = f34.build(ok.assign(model_path=np.nan), SPECS)
  return ok, sp


def test_thresholds_match_the_dock_config():
  import posebusters
  cfg = yaml.safe_load((pathlib.Path(posebusters.__file__).parent / 'config' / 'dock.yml').read_text())
  par = {}
  for m in cfg['modules']:
    p = m.get('parameters', {})
    for out in m['chosen_binary_test_output']:
      name = m.get('rename_outputs', {}).get(out, out).lower().replace(' ', '_')
      par[name + m.get('rename_suffix', '') if out == 'no_clashes' else name] = (m['function'], p)
  assert par['bond_lengths'][1]['threshold_bad_bond_length'] == pf.CHECKS['bond_lengths'][2]
  assert par['bond_angles'][1]['threshold_bad_angle'] == pf.CHECKS['bond_angles'][2]
  assert par['internal_steric_clash'][1]['threshold_clash'] == pf.CHECKS['internal_steric_clash'][2]
  assert par['aromatic_ring_flatness'][1]['threshold_flatness'] == pf.CHECKS['aromatic_ring_flatness'][2]
  assert par['non-aromatic_ring_non-flatness'][1]['threshold_flatness'] == \
      pf.CHECKS['non-aromatic_ring_non-flatness'][2]
  assert par['double_bond_flatness'][1]['threshold_flatness'] == pf.CHECKS['double_bond_flatness'][2]
  assert par['internal_energy'][1]['threshold_energy_ratio'] == pf.CHECKS['internal_energy'][2]
  assert par['protein-ligand_maximum_distance'][1]['max_distance'] == pf.CHECKS['protein-ligand_maximum_distance'][2]
  for k in ('protein', 'organic_cofactors', 'inorganic_cofactors', 'waters'):
    assert par[f'volume_overlap_with_{k}'][1]['clash_cutoff'] == pf.CHECKS[f'volume_overlap_with_{k}'][2]
  for m in cfg['modules']:
    if m['function'] == 'intermolecular_distance':
      assert m['parameters']['clash_cutoff'] == 0.75 and m['parameters']['radius_scale'] == 1.0
  standard = {m.get('rename_outputs', {}).get(o, o).lower().replace(' ', '_') for m in cfg['modules']
              for o in m['chosen_binary_test_output']}
  assert len(standard) == 22 and set(pf.CHECKS) | set(pf.BINARY) == standard


def test_excess_and_values():
  assert pf.excess(0.80, 'lower', 0.75) == pytest.approx(-1 / 15)
  assert pf.excess(0.70, 'lower', 0.75) == pytest.approx(1 / 15)
  assert pf.excess(110.0, 'upper', 100.0) == pytest.approx(0.10)
  assert np.isnan(pf.excess(np.nan, 'upper', 1.0))
  assert pf.rmsd_excess(2.2) == pytest.approx(0.10) and pf.rmsd_excess(1.8) == pytest.approx(-0.10)
  row = pd.Series({'longest_bond_relative_length': 1.10, 'shortest_bond_relative_length': 0.70,
                   'most_extreme_relative_angle': 1.2, 'shortest_noncovalent_relative_distance': 0.65})
  assert pf.check_value(row, 'bond_lengths') == pytest.approx(0.30)          # the short bond: 1 - 0.70
  assert pf.check_value(row, 'bond_angles') == pytest.approx(0.2)
  assert pf.check_value(row, 'internal_steric_clash') == pytest.approx(0.35)
  assert np.isnan(pf.check_value(row, 'internal_energy'))


def test_criterion():
  r = types.SimpleNamespace(error_fail='', rmsd_fail=2.5, pb_valid_fail=True)
  assert pf.criterion(r) == 'RMSD'
  r.pb_valid_fail = False
  assert pf.criterion(r) == 'RMSD and PoseBusters'
  r.rmsd_fail = 1.0
  assert pf.criterion(r) == 'PoseBusters'
  r.error_fail = 'ScoreError'
  assert pf.criterion(r) == 'not scored'


def test_flips_classes_and_summary(source, tmp_path):
  ok, sp = source
  flips, s, chk = pf.run({'main': (ok, sp)}, tmp_path, ARGS, items=ITEMS)
  assert len(flips) == int(sp.flip_success.sum()) and len(flips) > 0
  for r in flips.to_dict('records'):
    fail = r['failing_side']
    assert not sp[(sp.contrast == r['contrast']) & (sp.target == r['target']) & (sp.seed_a == r['seed'])].iloc[0][
        f'success_{fail}']
    failing = r['failing_criteria'].split(';')
    ex = [pf.rmsd_excess(r['rmsd_fail']) if c == 'RMSD' else EXC[r['path_fail']][c] for c in failing]
    assert r['worst_excess_fail'] == pytest.approx(max(ex))
    assert r['pose_rmsd'] == pytest.approx(abs(POS[r['path_fail']] - POS[r['path_ok']]))
    want = 'pose change' if r['pose_rmsd'] > 2 else 'near-threshold crossing' if max(ex) <= 0.10 else 'other'
    assert r['class'] == want
  row = s[(s.contrast == 'RC(G)') & (s.stratum == 'all flips')].iloc[0]
  g = flips[flips.contrast == 'RC(G)']
  assert row.flips == len(g)
  assert row['class: near-threshold crossing'] == int((g['class'] == 'near-threshold crossing').sum())
  grp = s[(s.contrast == 'group: platform') & (s.stratum == 'all flips')].iloc[0]
  assert grp.flips == int(flips.contrast.isin(['HW(V5,V6)', 'HW(V6,G)']).sum())
  for f in ('flips.csv', 'rescored.csv', 'summary.csv', 'checks.csv', 'reproduction.csv', 'report.md'):
    assert (tmp_path / f).is_file()
  assert 'pb_flips' in (tmp_path / 'report.md').read_text()


def test_reproduction_check_stops(source, tmp_path, monkeypatch):
  ok, sp = source
  table = ok.set_index('model_path', drop=False)

  def off(task):
    out = fake_rescore(task, table)
    out['re_ligand_rmsd'] = out['re_ligand_rmsd'] + 0.001
    return out
  monkeypatch.setattr(pf, 'RESCORE', off)
  with pytest.raises(SystemExit, match='check 3'):
    pf.run({'main': (ok, sp)}, tmp_path, ARGS, items=ITEMS)
  rep = pd.read_csv(tmp_path / 'reproduction.csv')
  assert not rep['3:ligand_rmsd'].any() and not (tmp_path / 'summary.csv').exists()


def test_sign_check_stops(source, tmp_path, monkeypatch):
  ok, sp = source
  table = ok.set_index('model_path', drop=False)

  def flipped_sign(task):
    out = fake_rescore(task, table)
    out['excess:volume_overlap_with_protein'] = abs(out['excess:volume_overlap_with_protein'])   # > 0 but passes
    return out
  monkeypatch.setattr(pf, 'RESCORE', flipped_sign)
  with pytest.raises(SystemExit, match='check 3, 3b or 4'):
    pf.run({'main': (ok, sp)}, tmp_path, ARGS, items=ITEMS)


def test_check_2_stops_on_a_wrong_top1(source, tmp_path):
  ok, sp = source
  sp = sp.copy()
  i = sp.index[sp.contrast == 'RC(G)'][0]
  sp.loc[i, 'top1_b'] = (sp.loc[i, 'top1_b'] + 1) % 5
  with pytest.raises(SystemExit, match='check 2'):
    pf.run({'main': (ok, sp)}, tmp_path, ARGS, items=ITEMS)
