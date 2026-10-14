"""analysis/score_dockq.py on DockQ's own example (1A2K, DockQ v2.1.3 README) and on AlphaFold3-style mmCIF
files. Runs in the PPI environment (analysis/requirements-ppi.txt):

    analysis/ppi/.venv/bin/python -m pytest analysis/tests/test_score_dockq.py
"""

import pathlib

import pytest

pytest.importorskip('DockQ')
gemmi = pytest.importorskip('gemmi')

import score_dockq  # noqa: E402

DATA = pathlib.Path(__file__).resolve().parent / 'data' / 'dockq'
MODEL, NATIVE = DATA / '1A2K_r_l_b.model.pdb.gz', DATA / '1A2K_r_l_b.pdb.gz'


def test_version_is_the_pinned_one():
  assert score_dockq.dockq_version() == '2.1.3'


def test_readme_case_native_ab_against_model_ba():
  r = score_dockq.score(MODEL, NATIVE, {'A': 'B', 'B': 'A'})
  assert r['error'] is None
  assert r['DockQ'] == pytest.approx(0.994, abs=5e-4)
  assert r['iRMSD'] == pytest.approx(0.0, abs=5e-4) and r['LRMSD'] == pytest.approx(0.0, abs=5e-4)
  assert r['fnat'] == pytest.approx(0.983, abs=5e-4) and r['fnonnat'] == pytest.approx(0.008, abs=5e-4)
  assert r['clashes'] == 0 and r['success'] and r['success_049']


@pytest.mark.parametrize('chain_map,dockq', [({'A': 'B', 'C': 'C'}, 0.511), ({'B': 'A', 'C': 'C'}, 0.453)])
def test_readme_cases_other_interfaces(chain_map, dockq):
  r = score_dockq.score(MODEL, NATIVE, chain_map)
  assert r['DockQ'] == pytest.approx(dockq, abs=5e-4)
  assert r['success'] and r['success_049'] == (dockq >= 0.49)


def _as_af3_cif(src, dst, rename, shift=None):
  """The native as an AlphaFold3-like mmCIF: chains renamed (A, B), optionally one chain moved away."""
  st = gemmi.read_structure(str(src))
  st.setup_entities()
  st.remove_ligands_and_waters()
  for ch in list(st[0]):
    if ch.name not in rename:
      st[0].remove_chain(ch.name)
  for ch in st[0]:
    if shift and ch.name == shift:
      for res in ch:
        for a in res:
          a.pos = gemmi.Position(a.pos.x + 40.0, a.pos.y, a.pos.z)
    ch.name = rename[ch.name]
  st.setup_entities()
  st.make_mmcif_document().write_file(str(dst))
  return dst


def test_native_against_itself_as_mmcif(tmp_path):
  cif = _as_af3_cif(NATIVE, tmp_path / 'm.cif', {'A': 'A', 'B': 'B'})
  r = score_dockq.score(cif, NATIVE, {'A': 'A', 'B': 'B'})
  assert r['error'] is None and r['DockQ'] == pytest.approx(1.0, abs=1e-6) and r['fnat'] == 1.0


def test_separated_chains_are_not_a_success(tmp_path):
  cif = _as_af3_cif(NATIVE, tmp_path / 'm.cif', {'A': 'A', 'B': 'B'}, shift='B')
  r = score_dockq.score(cif, NATIVE, {'A': 'A', 'B': 'B'})
  assert r['error'] is None and r['fnat'] == 0.0 and r['DockQ'] < 0.23 and not r['success']


def test_missing_chain_is_recorded_not_raised(tmp_path):
  r = score_dockq.score(MODEL, NATIVE, {'A': 'Z', 'B': 'A'})
  assert r['success'] is False and 'model chain Z not found' in r['error']
