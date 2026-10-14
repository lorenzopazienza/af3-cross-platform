"""M1 for three glycan ligands (section 10, 2026-10-14): the rescoring of analysis (3) and analysis/glycans.py."""

import csv
import pathlib

import pandas as pd
import pytest

import glycans
import score_ligand

REPO = pathlib.Path(__file__).resolve().parents[2]
TARGETS = [r['pdb_id'] for r in csv.DictReader(open(REPO / 'targets' / 'posebusters_subset.csv'))]
needs_data = pytest.mark.skipif(not (REPO / 'data' / 'rcsb' / '7WJB.cif').exists(),
                                reason='data/rcsb/ not present (gitignored; see data/README or scripts)')
needs_af3 = pytest.mark.skipif(not score_ligand.AF3_COMPONENT_SETS.exists(),
                               reason='AlphaFold3 build data (chemical_component_sets.pickle) not present')


@needs_af3
def test_the_rule_drops_o1_of_glycans_only():
  assert {'G6S', 'NGS', 'BGC', 'GLF', 'GCO'} <= score_ligand.af3_glycans()
  assert score_ligand.af3_removed_atoms('BGC', ['C1', 'O1', 'O5']) == {'O1'}
  assert score_ligand.af3_removed_atoms('GLF', ['C1', 'F1', 'O5']) == set()       # a glycan without O1
  assert score_ligand.af3_removed_atoms('ATP', ['O1', 'C1']) == set()              # O1 of a non-glycan stays
  assert score_ligand.af3_removed_atoms('BGC', ['O1A', 'O11']) == set()            # only the atom named O1


@needs_af3
@needs_data
def test_only_the_three_targets_have_an_atom_removed():
  removed = {t: score_ligand.reference(t, True).removed_atoms for t in TARGETS}
  assert {t: r for t, r in removed.items() if r} == {t: ['O1'] for t in glycans.THREE}


@needs_af3
@needs_data
@pytest.mark.parametrize('target', [t for t in TARGETS if t not in glycans.THREE])
def test_rescoring_changes_nothing_without_a_removed_atom(target):
  from rdkit import Chem
  a, b = score_ligand.reference(target), score_ligand.reference(target, True)
  assert b.removed_atoms == []
  assert b.ligand_names == a.ligand_names and b.pocket == a.pocket
  assert Chem.MolToMolBlock(b.ligand) == Chem.MolToMolBlock(a.ligand)


@needs_af3
@needs_data
@pytest.mark.parametrize('target', ['7U3J', '7BCP', '7DKT'])   # a non-glycan, and the two glycans without O1
def test_rescored_m1_equals_m1_without_a_removed_atom(target):
  chain, seqid = score_ligand.crystal_site(target)
  xtal = REPO / 'data' / 'rcsb' / f'{target}.cif'
  kw = dict(pb_protein='crystal', protein_chain=chain, ligand_seqid=seqid)
  assert score_ligand.score(target, xtal, **kw) == score_ligand.score(target, xtal, drop_af3_removed=True, **kw)


def _crystal_without(target, atom, tmp_path):
  """The crystal structure with the named atom deleted from the reference ligand instance, as AlphaFold3 writes it."""
  import gemmi
  chain, seqid = score_ligand.crystal_site(target)
  st = gemmi.read_structure(str(REPO / 'data' / 'rcsb' / f'{target}.cif'))
  ccd = score_ligand.reference(target).ccd
  n = 0
  for ch in st[0]:
    for res in ch:
      if res.name == ccd and res.seqid.num == seqid:
        for i in reversed(range(len(res))):
          if res[i].name == atom:
            del res[i]
            n += 1
  assert n >= 1
  st.setup_entities()
  path = tmp_path / f'{target}_without_{atom}.cif'
  st.make_mmcif_document().write_file(str(path))
  return path, chain, seqid


@needs_af3
@needs_data
def test_7wjb_rescored_on_the_atoms_alphafold3_predicts(tmp_path):
  full, cut = score_ligand.reference('7WJB'), score_ligand.reference('7WJB', True)
  assert full.ccd == 'BGC' and cut.removed_atoms == ['O1']
  assert cut.ligand_names == [n for n in full.ligand_names if n != 'O1']
  assert cut.ligand.GetNumAtoms() == full.ligand.GetNumAtoms() - 1 == 11
  assert cut.pocket == full.pocket                                   # the pocket of the full ligand
  path, chain, seqid = _crystal_without('7WJB', 'O1', tmp_path)
  kw = dict(pb_protein='crystal', protein_chain=chain, ligand_seqid=seqid)
  primary = score_ligand.score('7WJB', path, **kw)
  assert primary['success'] is False and "missing in the prediction: ['O1']" in primary['error']
  rescored = score_ligand.score('7WJB', path, drop_af3_removed=True, **kw)
  assert rescored['error'] is None and rescored['ligand_rmsd'] == 0.0
  assert rescored['pb_valid'] is True and rescored['success'] is True
  # M4: the same O1-less model against itself, on the rescored reference only
  assert score_ligand.pair_divergence('7WJB', path, path)['error']
  m4 = score_ligand.pair_divergence('7WJB', path, path, drop_af3_removed=True)
  assert m4['error'] is None and m4['ligand_rmsd'] == 0.0


def _ok():
  rows = []
  for t in ('5SB2', '7WJB'):
    for s in (0, 1):
      rows.append({'target': t, 'sample': s, 'model_path': f'{t}/{s}.cif', 'model_sha256': f'{t}{s}',
                   'm1_ligand_rmsd': None if t == '7WJB' else 0.5, 'm1_pb_valid': None if t == '7WJB' else True,
                   'm1_success': t != '7WJB', 'm1_pb_failed': None,
                   'm1_error': "ligand atoms missing in the prediction: ['O1']" if t == '7WJB' else None})
  return pd.DataFrame(rows)


def test_with_rescored_replaces_m1_of_the_three_targets_only():
  ok = _ok()
  cache = {('7WJB0', '7WJB'): {'ligand_rmsd': 0.4, 'pb_valid': True, 'success': True, 'pb_failed': [], 'error': None},
           ('7WJB1', '7WJB'): {'ligand_rmsd': 2.5, 'pb_valid': False, 'success': False,
                               'pb_failed': ['minimum_distance_to_protein'], 'error': None}}
  new = glycans.with_rescored(ok, cache)
  pd.testing.assert_frame_equal(new[new.target == '5SB2'], ok[ok.target == '5SB2'])
  g = new[new.target == '7WJB']
  assert list(g.m1_ligand_rmsd) == [0.4, 2.5] and list(g.m1_success) == [True, False]
  assert list(g.m1_pb_failed) == [None, 'minimum_distance_to_protein'] and g.m1_error.isna().all()
  assert ok[ok.target == '7WJB'].m1_error.notna().all()               # the input table is not modified


def test_unchanged_flags_a_changed_top1():
  seed = pd.DataFrame({'contrast': ['HW(V5,V6)'] * 2, 'target': ['A', 'B'], 'seed_a': [1, 1], 'seed_b': [1, 1],
                       'top1_a': [0, 1], 'top1_b': [0, 2], 'flip_top1': [False, True], 'margin_a': [0.1, 0.2],
                       'margin_b': [0.1, 0.3]})
  sample = pd.DataFrame({'contrast': ['HW(V5,V6)'], 'target': ['A'], 'seed_a': [1], 'seed_b': [1], 'sample': [0],
                         'bit_identical': [False], 'm4_ca_rmsd_global': [0.1], 'm5_ranking_score': [0.01],
                         'm5_mean_plddt': [0.2], 'm5_iptm': [0.0]})
  summary = pd.DataFrame({'contrast': ['HW(V5,V6)'], 'statistic': ['flip_top1'], 'n': [2], 'n_targets': [2],
                          'estimate': [0.5], 'ci_low': [0.0], 'ci_high': [1.0]})
  hwrc = pd.DataFrame({'statistic': ['flip_top1'], 'estimate': [0.5], 'ci_low': [0.0], 'ci_high': [1.0]})
  t = {'seed_pairs': seed, 'sample_pairs': sample, 'summary': summary, 'hwrc': hwrc}
  assert glycans.unchanged(t, t, [])['pass'].all()
  changed = dict(t, seed_pairs=seed.assign(top1_b=[0, 3]))
  res = glycans.unchanged(t, changed, []).set_index('check')['pass']
  assert not res['M2 top-1 of every (target, seed) pair'] and res['M3 margin of every (target, seed) pair']


def test_unchanged_accepts_c_alpha_rmsd_added_only_for_the_three_targets():
  def tables(targets, ca):
    sample = pd.DataFrame({'contrast': 'RC(G)', 'target': targets, 'seed_a': 1, 'seed_b': 1, 'sample': [0, 1],
                           'bit_identical': False, 'm4_ca_rmsd_global': ca, 'm5_ranking_score': 0.0,
                           'm5_mean_plddt': 0.0, 'm5_iptm': 0.0})
    seed = pd.DataFrame({'contrast': ['RC(G)'], 'target': ['A'], 'seed_a': [1], 'seed_b': [1], 'top1_a': [0],
                         'top1_b': [0], 'flip_top1': [False], 'margin_a': [0.1], 'margin_b': [0.1]})
    empty = pd.DataFrame({'contrast': pd.Series([], dtype=str), 'statistic': pd.Series([], dtype=str), 'n': [], 'n_targets': [], 'estimate': [], 'ci_low': [],
                          'ci_high': []})
    return {'seed_pairs': seed, 'sample_pairs': sample, 'summary': empty, 'hwrc': empty}
  check = 'm4_ca_rmsd_global of every sample pair where (1) has it; the others are pairs of the three targets, ' \
          'computed in (3) only'
  res = glycans.unchanged(tables(['A', '7WJB'], [0.2, None]), tables(['A', '7WJB'], [0.2, 0.3]), [])
  assert res.set_index('check')['pass'][check]
  res = glycans.unchanged(tables(['A', 'B'], [0.2, None]), tables(['A', 'B'], [0.2, 0.3]), [])
  assert not res.set_index('check')['pass'][check]
