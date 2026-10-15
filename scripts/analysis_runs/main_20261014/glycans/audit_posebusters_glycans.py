"""Descriptive check (4) of the section 10 row of 2026-10-14: AlphaFold3's glycan O1 rule on PoseBusters V1 and V2.

    analysis/.venv/bin/python scripts/analysis_runs/main_20261014/glycans/audit_posebusters_glycans.py

Outside every contrast; no weights, no cloud. For every ligand of PoseBusters V1 (the 428 entries of
data/posebusters/posebusters_benchmark_set) and V2 (the 308 entries of targets/posebusters_v2_screen.csv), by CCD
code: its CCD entry in the 2026-10-08 components.cif of the AlphaFold3 install (the file its build_data step used;
SHA-256 printed), membership in AlphaFold3's GLYCAN_LINKING_LIGANDS and GLYCAN_OTHER_LIGANDS
(chemical_component_sets.pickle of the same install), whether it has an atom named O1, O1's
pdbx_leaving_atom_flag, O1's heavy-atom neighbours (from _chem_comp_bond), and the number of connected fragments of
the heavy-atom graph with and without O1. AlphaFold3 at its default --fix_standalone_glycans=false removes O1 from
a ligand in either set that has no bondedAtomPairs to O1 (rule_applies).
Writes audit_posebusters_glycans.csv and audit_posebusters_glycans.txt next to this script.
"""
import csv
import hashlib
import pathlib
import sys

import gemmi

REPO = pathlib.Path(__file__).resolve().parents[4]
HERE = REPO / 'results' / 'analysis' / 'main_20261014' / 'glycans'   # outputs
sys.path.insert(0, str(REPO / 'analysis'))
import score_ligand  # noqa: E402

CCD = REPO / 'third_party' / 'alphafold3' / '.venv' / 'lib' / 'python3.12' / 'site-packages' / 'share' / 'libcifpp' / \
    'components.cif'
V1_DIR = REPO / 'data' / 'posebusters' / 'posebusters_benchmark_set'
V2_CSV = REPO / 'targets' / 'posebusters_v2_screen.csv'
STUDY = REPO / 'targets' / 'posebusters_subset.csv'


def sets():
  with open(score_ligand.AF3_COMPONENT_SETS, 'rb') as f:
    s = score_ligand._SetsUnpickler(f).load()
  return frozenset(s['glycans_linking']), frozenset(s['glycans_other'])


def fragments(nodes, edges):
  parent = {n: n for n in nodes}

  def find(x):
    while parent[x] != x:
      parent[x] = parent[parent[x]]
      x = parent[x]
    return x
  for a, b in edges:
    if a in parent and b in parent:
      parent[find(a)] = find(b)
  return len({find(n) for n in nodes})


def entry(block):
  """Heavy atoms {name: (element, leaving flag)}, heavy-atom bonds, type, name of one CCD block."""
  atoms = {}
  t = block.find('_chem_comp_atom.', ['atom_id', 'type_symbol', '?pdbx_leaving_atom_flag'])
  for row in t:
    name, el = gemmi.cif.as_string(row[0]), gemmi.cif.as_string(row[1]).upper()
    if el not in ('H', 'D'):
      atoms[name] = (el, gemmi.cif.as_string(row[2]) if row.has(2) else '?')
  bonds = []
  for row in block.find('_chem_comp_bond.', ['atom_id_1', 'atom_id_2']):
    a, b = gemmi.cif.as_string(row[0]), gemmi.cif.as_string(row[1])
    if a in atoms and b in atoms:
      bonds.append((a, b))
  return atoms, bonds, gemmi.cif.as_string(block.find_value('_chem_comp.type') or ''), \
      gemmi.cif.as_string(block.find_value('_chem_comp.name') or '')


def main():
  sha = hashlib.sha256(CCD.read_bytes()).hexdigest()
  v1 = sorted(p.name for p in V1_DIR.iterdir() if p.is_dir())
  v2 = sorted(r['posebusters_id'] for r in csv.DictReader(open(V2_CSV)))
  study = {r['ligand_ccd']: r['pdb_id'] for r in csv.DictReader(open(STUDY))}
  codes = {}
  for set_name, ids in (('V1', v1), ('V2', v2)):
    for pb in ids:
      codes.setdefault(pb.split('_', 1)[1], {'V1': [], 'V2': []})[set_name].append(pb)
  linking, other = sets()
  doc = gemmi.cif.read(str(CCD))
  rows = []
  for code in sorted(codes):
    block = doc.find_block(code)
    if block is None:
      rows.append({'ccd': code, 'in_ccd': False})
      continue
    atoms, bonds, ctype, cname = entry(block)
    has_o1 = 'O1' in atoms
    nbrs = sorted({b if a == 'O1' else a for a, b in bonds if 'O1' in (a, b)})
    n_all = fragments(atoms, bonds)
    n_wo = fragments([a for a in atoms if a != 'O1'], [(a, b) for a, b in bonds if 'O1' not in (a, b)]) \
        if has_o1 else n_all
    glycan = 'linking' if code in linking else 'other' if code in other else ''
    rows.append({'ccd': code, 'in_ccd': True, 'name': cname, 'chem_comp_type': ctype,
                 'v1_entries': ';'.join(codes[code]['V1']), 'v2_entries': ';'.join(codes[code]['V2']),
                 'study_target': study.get(code, ''), 'af3_glycan_set': glycan, 'heavy_atoms': len(atoms),
                 'has_O1': has_o1, 'O1_element': atoms['O1'][0] if has_o1 else '',
                 'O1_leaving_flag': atoms['O1'][1] if has_o1 else '',
                 'O1_heavy_neighbours': ';'.join(f'{n}({atoms[n][0]})' for n in nbrs),
                 'fragments_with_O1': n_all, 'fragments_without_O1': n_wo,
                 'rule_applies': bool(glycan and has_o1)})
  out = HERE / 'audit_posebusters_glycans.csv'
  keys = list(next(r for r in rows if r['in_ccd']))
  with open(out, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=keys, restval='')
    w.writeheader()
    w.writerows(rows)

  lines = [f'components.cif SHA-256 {sha}', f'glycan sets: {len(linking)} linking, {len(other)} other',
           f'V1: {len(v1)} entries; V2: {len(v2)} entries; distinct CCD codes: {len(codes)}',
           f'CCD codes not in components.cif: {[r["ccd"] for r in rows if not r["in_ccd"]]}', '']
  for set_name in ('V1', 'V2'):
    rs = [r for r in rows if r['in_ccd'] and r[f'{set_name.lower()}_entries']]
    n_entries = lambda sel: sum(len(r[f'{set_name.lower()}_entries'].split(';')) for r in sel)  # noqa: E731
    gly = [r for r in rs if r['af3_glycan_set']]
    app = [r for r in gly if r['rule_applies']]
    split = [r for r in app if r['fragments_without_O1'] > r['fragments_with_O1']]
    flag_y = [r for r in app if r['O1_leaving_flag'] == 'Y']
    lines += [f'{set_name}: {len(rs)} CCD codes ({n_entries(rs)} entries)',
              f'  in an AlphaFold3 glycan set: {len(gly)} codes ({n_entries(gly)} entries): '
              + ', '.join(f'{r["ccd"]}({r["af3_glycan_set"]})' for r in gly),
              f'  with an atom named O1 (O1 removed at the default): {len(app)} codes ({n_entries(app)} entries): '
              + ', '.join(r['ccd'] for r in app),
              f'    of which O1 flagged leaving (Y): {len(flag_y)}; O1 not leaving: '
              + (', '.join(f'{r["ccd"]}({r["O1_leaving_flag"]})' for r in app if r['O1_leaving_flag'] != 'Y') or '-'),
              f'    of which deleting O1 splits the heavy-atom graph: {len(split)}'
              + (': ' + ', '.join(f'{r["ccd"]} ({r["fragments_with_O1"]} -> {r["fragments_without_O1"]})'
                                  for r in split) if split else ''),
              f'  glycan without O1 (not affected): ' + (', '.join(r['ccd'] for r in gly if not r['has_O1']) or '-'),
              '']
  lines += ['Affected codes (both sets): ccd | type | set | O1 element, flag | O1 heavy neighbours | fragments '
            'with -> without O1 | V1 | V2 | study']
  for r in rows:
    if r.get('rule_applies'):
      lines.append(f'  {r["ccd"]} | {r["chem_comp_type"]} | {r["af3_glycan_set"]} | {r["O1_element"]}, '
                   f'{r["O1_leaving_flag"]} | {r["O1_heavy_neighbours"]} | {r["fragments_with_O1"]} -> '
                   f'{r["fragments_without_O1"]} | {r["v1_entries"] or "-"} | {r["v2_entries"] or "-"} | '
                   f'{r["study_target"] or "-"}')
  (HERE / 'audit_posebusters_glycans.txt').write_text('\n'.join(lines) + '\n')
  print('\n'.join(lines))
  return 0


if __name__ == '__main__':
  sys.exit(main())
