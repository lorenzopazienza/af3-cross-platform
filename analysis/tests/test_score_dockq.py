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


def test_collect_reads_the_harness_layout(tmp_path, monkeypatch):
  # <session>/<run>/<T>/af3_output/<T>/seed-S_sample-K/<T>_seed-S_sample-K_model.cif, as harness/run_af3.py writes.
  import csv
  import gzip
  import json
  monkeypatch.setattr(score_dockq, 'REPO', tmp_path)
  (tmp_path / 'targets' / 'ppi').mkdir(parents=True)
  with open(tmp_path / 'targets' / 'ppi' / 'targets.csv', 'w', newline='') as f:
    csv.writer(f).writerows([['pdb_id', 'chain_1', 'chain_2'], ['1A2K', 'A', 'B']])
  nat = tmp_path / 'data' / 'ppi' / 'assembly1'
  nat.mkdir(parents=True)
  plain = _as_af3_cif(NATIVE, tmp_path / 'native.cif', {'A': 'A', 'B': 'B'})
  (nat / '1A2K-assembly1.cif.gz').write_bytes(gzip.compress(plain.read_bytes()))
  run = tmp_path / 'results' / 'af3' / 'S_l4_ppi_g_01' / 'l4_xla_rep1_seed1_fresh'
  run.mkdir(parents=True)
  (run / 'session.json').write_text(json.dumps({'label': 'ppi'}))
  sd = run / '1A2K' / 'af3_output' / '1A2K' / 'seed-1_sample-0'
  sd.mkdir(parents=True)
  _as_af3_cif(NATIVE, sd / '1A2K_seed-1_sample-0_model.cif', {'A': 'A', 'B': 'B'})
  out = tmp_path / 'out'
  out.mkdir()
  score_dockq.collect(out)
  rows = list(csv.DictReader(open(out / 'dockq.csv')))
  assert len(rows) == 1 and rows[0]['target'] == '1A2K' and rows[0]['seed'] == '1' and rows[0]['sample'] == '0'
  assert float(rows[0]['DockQ']) == pytest.approx(1.0, abs=1e-6) and rows[0]['success'] == 'True'
