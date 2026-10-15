"""analysis/ppi.py (follow-up experiment 1, section 10, 2026-10-14) on synthetic tables and structures."""

import types

import numpy as np
import pandas as pd
import pytest

import ppi

gemmi = pytest.importorskip('gemmi')


def _complex_cif(path, shift=(0.0, 0.0, 0.0), rotate=False, move_b=0.0):
  """Two short poly-Ala chains A and B with C-alpha atoms only."""
  st = gemmi.Structure()
  model = gemmi.Model('1')
  for name, x0 in (('A', 0.0), ('B', 8.0)):
    ch = gemmi.Chain(name)
    for i in range(6):
      res = gemmi.Residue()
      res.name, res.seqid, res.label_seq = 'ALA', gemmi.SeqId(i + 1, ' '), i + 1   # AlphaFold3 writes label_seq_id
      res.entity_type = gemmi.EntityType.Polymer
      x, y, z = x0 + 1.5 * i + (move_b if name == 'B' else 0.0), 0.7 * i * i, 0.3 * i
      if rotate:
        x, y = -y, x
      a = gemmi.Atom()
      a.name, a.element = 'CA', gemmi.Element('C')
      a.pos = gemmi.Position(x + shift[0], y + shift[1], z + shift[2])
      res.add_atom(a)
      ch.add_residue(res)
    model.add_chain(ch)
  st.add_model(model)
  st.setup_entities()
  st.make_mmcif_document().write_file(str(path))
  return path


def test_ca_divergence_is_invariant_to_rigid_motion(tmp_path, monkeypatch):
  monkeypatch.setattr(ppi, 'REPO', tmp_path)
  a = _complex_cif(tmp_path / 'a.cif')
  b = _complex_cif(tmp_path / 'b.cif', shift=(5.0, -3.0, 2.0), rotate=True)
  ppi.ca_divergence.cache_clear()
  r = ppi.ca_divergence('X', 'a.cif', 'b.cif')
  assert r['error'] is None and r['ligand_rmsd'] is None and r['ca_rmsd_global'] == pytest.approx(0.0, abs=1e-3)


def test_ca_divergence_uses_both_chains(tmp_path, monkeypatch):
  monkeypatch.setattr(ppi, 'REPO', tmp_path)
  _complex_cif(tmp_path / 'a.cif')
  _complex_cif(tmp_path / 'b.cif', move_b=3.0)   # chain B slides along x: no rigid fit of the complex removes it
  ppi.ca_divergence.cache_clear()
  assert ppi.ca_divergence('X', 'a.cif', 'b.cif')['ca_rmsd_global'] > 0.5
  assert len(ppi.complex_ca(tmp_path / 'a.cif')) == 12


def _ok():
  rows = []
  # Target T1: V6 r1 top-1 is seed 1 sample 0 (success), V6 r2 top-1 is seed 2 sample 1 (failure).
  for rep, best in (('r1', (1, 0)), ('r2', (2, 1))):
    for seed in (1, 2):
      for k in range(3):
        score = 0.9 if (seed, k) == best else 0.5 - 0.1 * k + 0.01 * seed
        dq = 0.8 if (seed, k) == (1, 0) else 0.1
        rows.append({'condition': 'V6', 'repetition': rep, 'target': 'T1', 'seed': seed, 'sample': k,
                     'ranking_score': score, 'DockQ': dq, 'dockq_success': dq >= 0.23, 'dockq_success_049': dq >= 0.49})
  return pd.DataFrame(rows)


def test_overall_top1_flip_of_identity_and_success(monkeypatch):
  monkeypatch.setattr(ppi, 'CONTRASTS', [('RC(V6)', 'RC', ('V6', 'r1'), ('V6', 'r2'))])
  monkeypatch.setattr(ppi, 'NAMES', ['RC(V6)'])
  pairs, stats = ppi.overall_top1(ppi.with_success(_ok(), 'dockq_success'),
                                  types.SimpleNamespace(seed=2026, replicates=50))
  r = pairs.iloc[0]
  assert (r.top1_a, r.top1_b, r.success_a, r.success_b) == ('1/0', '2/1', True, False)
  assert bool(r.flip_top1) and bool(r.flip_success)
  assert set(stats.statistic) == {'overall_flip_top1', 'overall_flip_success'} and (stats.estimate == 1.0).all()


def test_success_rates_overall_and_per_seed():
  sr = ppi.success_rates(_ok(), types.SimpleNamespace(seed=2026, replicates=50))
  g = sr[(sr.repetition == 'r1') & (sr.threshold == 0.23)].set_index('statistic').estimate
  assert g['success_rate_overall_top1'] == 1.0
  assert g['success_rate_top1_per_seed'] == 0.5   # seed 1: (1, 0) succeeds; seed 2: top-1 sample 0, DockQ 0.1
  r2 = sr[(sr.repetition == 'r2') & (sr.threshold == 0.23)].set_index('statistic').estimate
  assert r2['success_rate_overall_top1'] == 0.0


def test_m3_margins():
  m = ppi.m3(ppi.with_success(_ok(), 'dockq_success')).set_index('repetition')
  assert m.loc['r1', 'seed_units'] == 2 and m.loc['r1', 'zero_margins'] == 0
  assert m.loc['r1', 'median'] == pytest.approx(np.median([0.9 - 0.41, 0.52 - 0.42]))


def test_load_merges_dockq_by_model_path(tmp_path):
  pd.DataFrame([{'label': 'ppi', 'num_recycles': 10, 'num_diffusion_samples': 5, 'platform': 'G', 'target': 'T1',
                 'seed': 1, 'repetition': 'r1', 'sample': k, 'model_path': f'm{k}.cif'} for k in range(2)]
               + [{'label': 'main', 'num_recycles': 10, 'num_diffusion_samples': 5, 'platform': 'G', 'target': 'X',
                   'seed': 1, 'repetition': 'r1', 'sample': 0, 'model_path': 'x.cif'}]).to_csv(tmp_path / 'samples.csv')
  pd.DataFrame([{'model': f'm{k}.cif', 'DockQ': 0.1 + k, 'iRMSD': 1, 'LRMSD': 1, 'fnat': 0.5, 'success': k == 1,
                 'success_049': False, 'error': None, 'dockq_version': '2.1.3'} for k in range(2)]).to_csv(
      tmp_path / 'dockq.csv')
  m, ok = ppi.load(tmp_path)
  assert list(ok.target) == ['T1', 'T1'] and list(ok.DockQ) == [0.1, 1.1] and list(ok.condition) == ['G', 'G']
