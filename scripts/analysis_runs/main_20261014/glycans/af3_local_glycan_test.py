"""Descriptive check (4) of the section 10 row of 2026-10-14: AlphaFold3's own input processing on six glycans.

    cd third_party/alphafold3 && .venv/bin/python ../../scripts/analysis_runs/main_20261014/glycans/af3_local_glycan_test.py

Local, no weights, no cloud; AlphaFold3 code is called, not edited. For PRP, MMA, 6MJ, BGC, G6S and GLF, each as
the only ligand next to a short protein chain (inputs/af3_featurise.protein_ligand_json; no bondedAtomPairs),
AlphaFold3 v3.0.4 featurises the input (alphafold3.data.featurisation.featurise_input, as run_alphafold.py calls it,
with fix_standalone_glycans false and true). From the featurised batch:
  - the ligand atoms that enter the model: the ligand chain's atoms in token_atoms_layout (one token per ligand
    atom), checked against ref_atom_name_chars and ref_mask;
  - the ligand's bond graph that enters the model: tokens_to_ligand_ligand_bonds (the CCD intra-ligand bonds that
    AlphaFold3 adds in RefStructure.compute_features and keeps only between atoms present), and its number of
    connected fragments.
Compared with the CCD heavy atoms of AlphaFold3's own CCD (alphafold3.constants.chemical_components.Ccd).
Writes af3_local_glycan_test.txt and af3_local_glycan_test.json next to this script.
"""
import json
import pathlib
import subprocess
import sys

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[4]
HERE = REPO / 'results' / 'analysis' / 'main_20261014' / 'glycans'   # outputs
sys.path.insert(0, str(REPO / 'inputs'))
import af3_featurise  # noqa: E402  (puts third_party/alphafold3 on the path)

from alphafold3.common import folding_input  # noqa: E402
from alphafold3.constants import chemical_component_sets  # noqa: E402
from alphafold3.constants import chemical_components  # noqa: E402
from alphafold3.data import featurisation  # noqa: E402
from alphafold3.version import __version__ as AF3_VERSION  # noqa: E402

LIGANDS = ['PRP', 'MMA', '6MJ', 'BGC', 'G6S', 'GLF']
PROTEIN = 'GSHMSKLVVLGAGGVGKSAL'   # any short chain; the ligand is a separate chain without bonds to it
LIGAND_CHAIN = 'B'


def fragments(nodes, edges):
  parent = {n: n for n in nodes}

  def find(x):
    while parent[x] != x:
      parent[x] = parent[parent[x]]
      x = parent[x]
    return x
  for a, b in edges:
    parent[find(a)] = find(b)
  return len({find(n) for n in nodes})


def ccd_heavy(ccd, code):
  e = ccd[code]
  atoms = [n for n, el in zip(e['_chem_comp_atom.atom_id'], e['_chem_comp_atom.type_symbol']) if el not in ('H', 'D')]
  flags = dict(zip(e['_chem_comp_atom.atom_id'], e.get('_chem_comp_atom.pdbx_leaving_atom_flag', ['?'] * len(atoms))))
  bonds = [(a, b) for a, b in zip(e.get('_chem_comp_bond.atom_id_1', []), e.get('_chem_comp_bond.atom_id_2', []))
           if a in atoms and b in atoms]
  return atoms, bonds, flags


def run(code, fix, ccd):
  fold_input = folding_input.Input.from_json(af3_featurise.protein_ligand_json(f'glycan_{code}', PROTEIN, code, [1]))
  batch = featurisation.featurise_input(fold_input, ccd=ccd, buckets=af3_featurise.BUCKETS,
                                        ref_max_modified_date=af3_featurise.REF_MAX_MODIFIED_DATE,
                                        fix_standalone_glycans=fix)[0]
  layout = batch['token_atoms_layout']
  layout = layout.item() if isinstance(layout, np.ndarray) else layout   # a 0-d object array holding the AtomLayout
  n_tokens = int(batch['seq_mask'].sum())
  names, tokens = [], []
  for tok in range(n_tokens):
    for k in range(layout.shape[1]):
      if layout.chain_id[tok, k] == LIGAND_CHAIN and layout.atom_name[tok, k]:
        names.append(str(layout.atom_name[tok, k]))
        tokens.append(tok)
  chars = batch['ref_atom_name_chars']
  ref_names = [''.join(chr(c + 32) for c in chars[t, 0] if c) for t in tokens]
  ref_mask = [bool(batch['ref_mask'][t, 0]) for t in tokens]
  idx, mask = batch['tokens_to_ligand_ligand_bonds:gather_idxs'], batch['tokens_to_ligand_ligand_bonds:gather_mask']
  name_of = dict(zip(tokens, names))
  bonds = sorted({tuple(sorted((name_of[int(a)], name_of[int(b)]))) for (a, b), m in zip(idx, mask)
                  if m.all() and int(a) in name_of and int(b) in name_of})
  heavy, ccd_bonds, flags = ccd_heavy(ccd, code)
  return {'ligand': code, 'fix_standalone_glycans': fix,
          'af3_glycan_set': 'linking' if code in chemical_component_sets.GLYCAN_LINKING_LIGANDS else
          'other' if code in chemical_component_sets.GLYCAN_OTHER_LIGANDS else '',
          'ccd_heavy_atoms': len(heavy), 'ccd_heavy_bonds': len(ccd_bonds),
          'ccd_fragments': fragments(heavy, ccd_bonds),
          'O1_leaving_flag': flags.get('O1', ''),
          'atoms_in_model': names, 'n_atoms_in_model': len(names),
          'dropped': [a for a in heavy if a not in names],
          'ref_atom_names_match_layout': ref_names == names, 'ref_mask_all_true': all(ref_mask),
          'bonds_in_model': len(bonds), 'fragments_in_model': fragments(names, bonds),
          'fragment_atoms': _fragment_lists(names, bonds)}


def _fragment_lists(nodes, edges):
  adj = {n: set() for n in nodes}
  for a, b in edges:
    adj[a].add(b)
    adj[b].add(a)
  seen, out = set(), []
  for n in nodes:
    if n in seen:
      continue
    stack, comp = [n], []
    while stack:
      x = stack.pop()
      if x not in seen:
        seen.add(x)
        comp.append(x)
        stack.extend(adj[x] - seen)
    out.append(sorted(comp))
  return sorted(out, key=len, reverse=True)


def main():
  af3 = REPO / 'third_party' / 'alphafold3'
  git = lambda *a: subprocess.run(['git', '-C', str(af3), *a], capture_output=True, text=True).stdout.strip()  # noqa: E731
  commit, tag = git('rev-parse', 'HEAD'), git('describe', '--tags', '--always')
  src_changes = git('status', '--short', '--untracked-files=no', '--', 'src') or 'none'
  ccd = chemical_components.Ccd()
  results = [run(code, fix, ccd) for code in LIGANDS for fix in (False, True)]
  (HERE / 'af3_local_glycan_test.json').write_text(json.dumps(results, indent=1) + '\n')
  lines = [f'AlphaFold3 checkout {tag} ({commit}; the package\'s __version__ string reads {AF3_VERSION}); '
           f'changes under src/: {src_changes}', f'featurisation only (no weights); protein chain {PROTEIN}, ligand '
           f'chain {LIGAND_CHAIN}, no bondedAtomPairs; numpy {np.__version__}', '',
           'ligand | set | fix_standalone_glycans | CCD heavy atoms | atoms in model | dropped | O1 leaving flag | '
           'bonds in model (CCD) | fragments in model (CCD) | ref features match layout']
  for r in results:
    lines.append(f'{r["ligand"]} | {r["af3_glycan_set"] or "-"} | {str(r["fix_standalone_glycans"]).lower()} | '
                 f'{r["ccd_heavy_atoms"]} | {r["n_atoms_in_model"]} | {",".join(r["dropped"]) or "-"} | '
                 f'{r["O1_leaving_flag"] or "-"} | {r["bonds_in_model"]} ({r["ccd_heavy_bonds"]}) | '
                 f'{r["fragments_in_model"]} ({r["ccd_fragments"]}) | '
                 f'{r["ref_atom_names_match_layout"] and r["ref_mask_all_true"]}')
  lines += ['', 'Atoms in the model and fragments:']
  for r in results:
    lines.append(f'{r["ligand"]} fix={str(r["fix_standalone_glycans"]).lower()}: {" ".join(r["atoms_in_model"])}')
    if r['fragments_in_model'] > 1:
      lines.append('    fragments: ' + ' | '.join(' '.join(f) for f in r['fragment_atoms']))
  (HERE / 'af3_local_glycan_test.txt').write_text('\n'.join(lines) + '\n')
  print('\n'.join(lines))
  return 0


if __name__ == '__main__':
  sys.exit(main())
