"""Targets of follow-up 10, larger inputs on the whole PDB (label large; notes/analysis_plan.md section 10, row of
2026-11-04 "Follow-up experiment 10, larger inputs on the whole PDB", OSF addendum https://osf.io/nsbgp): the 34 large
and 34 control targets of targets/large/pdb_targets_amended.csv, in the columns of targets/posebusters_subset.csv,
and the M1 reference ligand of each.

Run inside the pinned AlphaFold3 environment (the CCD and the CIF reader are AlphaFold3's, as in the screen):

    cd third_party/alphafold3
    .venv/bin/python <repository>/targets/large/build_large_subset.py

Reads, offline: targets/large/pdb_targets_amended.csv (SHA-256 checked against the value the row and the OSF
addendum record) and the deposited mmCIF files the screen cached, data/pdb_large_screen/cif/<PDB>.cif.gz. Nothing
is queried.

Rows: the groups large and control (the 10 of "information 1537-2048" are not run), large first, each in PDB ID
order. Columns: those of targets/posebusters_subset.csv, then the follow-up 10 columns. Values:
  - pdb_id, ligand_ccd, num_tokens, bucket, sequence: the screen's (AlphaFold3's token count, the RCSB canonical
    sequence); sequence_length its length; sequence_sha256 its SHA-256 (the key of data/msa/);
  - posebusters_id empty (not PoseBusters complexes; the M1 reference is the deposited model, below);
  - n_protein_chains: the deposited polymer chains (one entity, rule 1); n_input_chains 2 (the chain and the ligand);
  - protein_chain_id: the auth_asym_id of the pocket chain (rule 3; the screen records its label_asym_id, kept in
    protein_label_asym), so gemmi and analysis/score_ligand.py find it by name;
  - bin: the token range of the bucket (le256, 257-512, 513-768, 769-1024, 1025-1280, 1281-1536); pilot no;
  - source_file: the cached mmCIF; source_sha256, rcsb_sha256: its SHA-256 as downloaded (gzip);
    rcsb_url the download URL; rcsb_initial_release the screen's release date; rcsb_latest_revision the last
    revision date in the mmCIF (_pdbx_audit_revision_history);
  - group, stratum, variant_A, method, resolution, cluster30, ligand_asym: the screen's;
  - tpu_r2: yes for the 20 targets of the r2 subset on V5 and V6, drawn as the row says
    (numpy.random.default_rng(2026).choice without replacement, 10 per group, from each group's PDB IDs sorted as
    strings, large first) and checked equal to the 20 the row lists.

M1 reference (analysis/score_ligand.py, row item d: "the reference is the rule-2 instance of the deposited model"):
data/large_refs/<PDB>_ligand.sdf, the ligand_asym instance of model 1 of the cached mmCIF, heavy atoms, the first
alternate location of each atom (as gemmi's remove_alternative_conformations keeps it, which score_ligand applies to
the crystal), with bond orders and formal charges from AlphaFold3's CCD entry (rdkit_utils.mol_from_ccd_cif, the
converter AlphaFold3 itself uses) and the CCD atom names as the atom property atom_name. Atoms of the CCD entry that
the deposited instance lacks are left out (the reference is what was modelled; the count is recorded).

Outputs: targets/large/large_subset.csv, targets/large/sources.json (SHA-256 of every input and output, counts,
reference ligands), data/large_refs/<PDB>_ligand.sdf. A rerun writes the same CSV and SDF files byte for byte.
"""
import collections
import csv
import datetime
import gzip
import hashlib
import io
import json
import pathlib
import subprocess
import sys

import numpy as np
from rdkit import Chem

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'inputs'))
import af3_featurise  # noqa: E402
from alphafold3.cpp import cif_dict  # noqa: E402
from alphafold3.data.tools import rdkit_utils  # noqa: E402

TARGETS = REPO / 'targets' / 'large' / 'pdb_targets_amended.csv'
TARGETS_SHA256 = '7536e8995f822e42236c5e3ea3380264131828814c1ad6b2a28831fe9e6e44f4'   # the row and the OSF addendum
CIF_DIR = REPO / 'data' / 'pdb_large_screen' / 'cif'
REF_DIR = REPO / 'data' / 'large_refs'
OUT = REPO / 'targets' / 'large' / 'large_subset.csv'
SOURCES = REPO / 'targets' / 'large' / 'sources.json'
SPEC_ROW = ('notes/analysis_plan.md section 10, row of 2026-11-04 (follow-up experiment 10, larger inputs on the whole '
            'PDB, label large); OSF addendum https://osf.io/nsbgp')
SUBSET_COLS = ['pdb_id', 'ligand_ccd', 'posebusters_id', 'n_protein_chains', 'n_input_chains', 'protein_chain_id',
               'sequence_length', 'num_tokens', 'bucket', 'bin', 'pilot', 'sequence_sha256', 'source_file',
               'source_sha256', 'rcsb_url', 'rcsb_sha256', 'rcsb_initial_release', 'rcsb_latest_revision', 'sequence']
LARGE_COLS = ['group', 'stratum', 'variant_A', 'method', 'resolution', 'cluster30', 'ligand_asym', 'protein_label_asym',
              'tpu_r2']
GROUPS = ('large', 'control')
SEED, PER_GROUP = 2026, 10
ROW_SUBSET = {'large': '7DIA 7QUR 8C8V 8SX7 8TZG 8V1G 8XB3 9BWT 9GZ2 9X5O'.split(),
              'control': '7UMP 8BLP 8CEY 9ETQ 9HKX 9OHN 9OMM 9P4Y 9ZB9 9ZNO'.split()}
BINS = ((256, 'le256'), (512, '257-512'), (768, '513-768'), (1024, '769-1024'), (1280, '1025-1280'),
        (1536, '1281-1536'))


def sha256(data):
  return hashlib.sha256(data).hexdigest()


def read_targets():
  data = TARGETS.read_bytes()
  if sha256(data) != TARGETS_SHA256:
    raise SystemExit(f'{TARGETS.relative_to(REPO)}: SHA-256 {sha256(data)} is not the registered {TARGETS_SHA256}')
  rows = list(csv.DictReader(l for l in data.decode().splitlines() if not l.startswith('#')))
  return [r for r in rows if r['group'] in GROUPS], sha256(data)


def tpu_r2_subset(rows):
  """The row's draw: one generator, large first, choice of 10 without replacement from the sorted PDB IDs."""
  rng = np.random.default_rng(SEED)
  out = {}
  for g in GROUPS:
    ids = sorted(r['pdb_id'] for r in rows if r['group'] == g)
    out[g] = [ids[i] for i in sorted(rng.choice(len(ids), size=PER_GROUP, replace=False))]
    if out[g] != ROW_SUBSET[g]:
      raise SystemExit(f'the r2 subset drawn for {g} {out[g]} is not the one the row lists {ROW_SUBSET[g]}')
  return {t for g in GROUPS for t in out[g]}


def first_altloc_atoms(d, asym):
  """{atom name: (x, y, z)} of label_asym_id asym in model 1, heavy atoms, the first alternate location of the
  instance (atoms without an alternate location always), as gemmi's remove_alternative_conformations keeps them."""
  col = lambda k: d.get(f'_atom_site.{k}', [])  # noqa: E731
  model, el, alt = col('pdbx_PDB_model_num'), col('type_symbol'), col('label_alt_id')
  names, asyms = col('label_atom_id'), col('label_asym_id')
  xs, ys, zs = col('Cartn_x'), col('Cartn_y'), col('Cartn_z')
  first_model = model[0]
  chosen, out = None, {}
  for i in range(len(names)):
    if model[i] != first_model or asyms[i] != asym or el[i] in ('H', 'D'):
      continue
    a = alt[i] if alt else '.'
    if a not in ('.', '?'):
      chosen = chosen or a
      if a != chosen:
        continue
    out.setdefault(names[i], (float(xs[i]), float(ys[i]), float(zs[i])))
  return out


def auth_chain(d, label_asym):
  col = lambda k: d.get(f'_atom_site.{k}', [])  # noqa: E731
  auth = {a for a, l in zip(col('auth_asym_id'), col('label_asym_id')) if l == label_asym}
  if len(auth) != 1:
    raise SystemExit(f'label_asym_id {label_asym}: {len(auth)} auth_asym_id values {sorted(auth)}')
  return auth.pop()


def reference_sdf(ccd, code, atoms):
  """SDF text of the CCD component with the given heavy-atom coordinates, and the CCD heavy atoms left out."""
  mol = rdkit_utils.mol_from_ccd_cif(ccd.get(code), force_parse=True, sort_alphabetically=False,
                                     remove_hydrogens=False)
  mol = Chem.RemoveAllHs(mol)                        # hydrogens become implicit; atom properties are kept
  missing = sorted(a.GetProp('atom_name') for a in mol.GetAtoms() if a.GetProp('atom_name') not in atoms)
  if missing:
    Chem.Kekulize(mol, clearAromaticFlags=True)
    rw = Chem.RWMol(mol)
    for i in sorted((a.GetIdx() for a in rw.GetAtoms() if a.GetProp('atom_name') not in atoms), reverse=True):
      for nb in rw.GetAtomWithIdx(i).GetNeighbors():   # a removed neighbour becomes an implicit hydrogen
        nb.SetNoImplicit(False)
      rw.RemoveAtom(i)
    mol = rw.GetMol()
    Chem.SanitizeMol(mol)
  conf = Chem.Conformer(mol.GetNumAtoms())
  for a in mol.GetAtoms():
    conf.SetAtomPosition(a.GetIdx(), atoms[a.GetProp('atom_name')])
  mol.RemoveAllConformers()
  mol.AddConformer(conf, assignId=True)
  Chem.AssignStereochemistryFrom3D(mol)
  Chem.CreateAtomStringPropertyList(mol, 'atom_name')
  mol.SetProp('_Name', code)
  return Chem.MolToMolBlock(mol) + '$$$$\n', missing, mol.GetNumAtoms()


def bin_of(bucket):
  return next(name for b, name in BINS if b == bucket)


def main():
  started = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds')
  rows, targets_sha = read_targets()
  if [sum(r['group'] == g for r in rows) for g in GROUPS] != [34, 34]:
    raise SystemExit('expected 34 large and 34 control targets')
  if len({r['pdb_id'] for r in rows}) != len(rows):
    raise SystemExit('a PDB ID twice; input file names would collide')
  r2 = tpu_r2_subset(rows)
  ccd = af3_featurise.ccd()
  REF_DIR.mkdir(parents=True, exist_ok=True)
  out, refs, cif_sha = [], {}, {}
  for r in sorted(rows, key=lambda r: (GROUPS.index(r['group']), r['pdb_id'])):
    pdb = r['pdb_id']
    gz = (CIF_DIR / f'{pdb}.cif.gz').read_bytes()
    cif_sha[pdb] = sha256(gz)
    d = cif_dict.from_string(gzip.decompress(gz).decode())
    pocket = r['pocket_chains'].split()
    if len(pocket) != 1:
      raise SystemExit(f'{pdb}: pocket chains {pocket}; rule 3 gives one')
    revisions = sorted(d.get('_pdbx_audit_revision_history.revision_date', []))
    atoms = first_altloc_atoms(d, r['ligand_asym'])
    sdf, missing, n_atoms = reference_sdf(ccd, r['ligand_ccd'], atoms)
    (REF_DIR / f'{pdb}_ligand.sdf').write_text(sdf)
    refs[pdb] = {'ligand_ccd': r['ligand_ccd'], 'ligand_asym': r['ligand_asym'], 'heavy_atoms': n_atoms,
                 'ccd_heavy_atoms_missing': missing, 'sha256': sha256(sdf.encode())}
    seq = r['sequence']
    out.append({
        'pdb_id': pdb, 'ligand_ccd': r['ligand_ccd'], 'posebusters_id': '', 'n_protein_chains': r['chains_deposited'],
        'n_input_chains': 2, 'protein_chain_id': auth_chain(d, pocket[0]), 'sequence_length': len(seq),
        'num_tokens': r['num_tokens'], 'bucket': r['bucket'], 'bin': bin_of(int(r['bucket'])), 'pilot': 'no',
        'sequence_sha256': sha256(seq.encode()), 'source_file': f'data/pdb_large_screen/cif/{pdb}.cif.gz',
        'source_sha256': cif_sha[pdb], 'rcsb_url': f'https://files.rcsb.org/download/{pdb}.cif.gz',
        'rcsb_sha256': cif_sha[pdb], 'rcsb_initial_release': r['release'],
        'rcsb_latest_revision': revisions[-1] if revisions else '', 'sequence': seq,
        'group': r['group'], 'stratum': r['stratum'], 'variant_A': r['variant_A'], 'method': r['method'],
        'resolution': r['resolution'], 'cluster30': r['cluster30'], 'ligand_asym': r['ligand_asym'],
        'protein_label_asym': pocket[0], 'tpu_r2': 'yes' if pdb in r2 else 'no'})
  if any(int(r['sequence_length']) != int(t['residues']) for r, t in zip(out, sorted(
      rows, key=lambda r: (GROUPS.index(r['group']), r['pdb_id'])))):
    raise SystemExit('sequence length differs from the screen')
  buf = io.StringIO()
  w = csv.DictWriter(buf, SUBSET_COLS + LARGE_COLS, lineterminator='\n')
  w.writeheader()
  w.writerows(out)
  text = buf.getvalue()
  OUT.write_text(text)
  count = lambda key, g: dict(sorted(collections.Counter(str(r[key]) for r in out if r['group'] == g).items()))  # noqa: E731
  SOURCES.write_text(json.dumps({
      'built_utc': started,
      'built_by': 'targets/large/build_large_subset.py',
      'specification': SPEC_ROW,
      'rule': 'the rows of groups large and control of targets/large/pdb_targets_amended.csv (the screen of the row, '
              'item a), large first, each in PDB ID order; pilot no; tpu_r2: the row\'s r2 subset (item c)',
      'inputs_sha256': {'targets/large/pdb_targets_amended.csv': targets_sha},
      'rcsb_mmcif_sha256': cif_sha,
      'reference_ligands': {'folder': 'data/large_refs', 'files': refs,
                            'how': 'ligand_asym instance of model 1, heavy atoms, first alternate location; bond '
                                   'orders and charges from the AlphaFold3 CCD (rdkit_utils.mol_from_ccd_cif)'},
      'alphafold3': {'commit': subprocess.check_output(['git', '-C', str(af3_featurise.AF3_DIR), 'rev-parse', 'HEAD'],
                                                       text=True).strip(),
                     'buckets': list(af3_featurise.BUCKETS)},
      'counts': {g: {'targets': sum(r['group'] == g for r in out), 'by_bucket': count('bucket', g),
                     'by_method': count('method', g), 'variant_A': sum(r['variant_A'] == 'yes' for r in out if r['group'] == g),
                     'tpu_r2': sum(r['tpu_r2'] == 'yes' for r in out if r['group'] == g)} for g in GROUPS},
      'output': {'path': 'targets/large/large_subset.csv', 'sha256': sha256(text.encode())},
  }, indent=2) + '\n')
  print(f'{len(out)} targets -> {OUT.relative_to(REPO)}; reference ligands -> {REF_DIR.relative_to(REPO)}/')
  for pdb, ref in refs.items():
    if ref['ccd_heavy_atoms_missing']:
      print(f'  {pdb} {ref["ligand_ccd"]}: CCD heavy atoms not modelled: {ref["ccd_heavy_atoms_missing"]}')


if __name__ == '__main__':
  main()
