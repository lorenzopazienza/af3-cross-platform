"""Builds the follow-up 6 test fixtures (run once, from the repository root, in the AlphaFold3 venv):

    third_party/alphafold3/.venv/bin/python analysis/tests/fixtures/make_msa_fixtures.py

msa_template_6yqw_A.cif: chain A of the PDB entry 6YQW (data/rcsb/6YQW.cif), cut and written as AlphaFold3's data
pipeline writes a template (alphafold3 structure.filter(chain_id=...).to_mmcif(), data/pipeline.py).
msa_data_6yqw.json: a synthetic AlphaFold3 _data.json for 6YQW, written by folding_input.Input.to_json() as
run_alphafold.py writes <name>_data.json: the official input (data/inputs_official/6YQW.json) with the first 24
rows of the frozen ColabFold MSA as unpairedMsa, an empty pairedMsa and the template above mapped residue to
residue. Not a study input: the tests use it to check that a _data.json with templates runs through the harness.
"""
import dataclasses
import json
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / 'third_party' / 'alphafold3'))
from alphafold3 import structure  # noqa: E402
from alphafold3.common import folding_input  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
s = structure.from_mmcif((REPO / 'data' / 'rcsb' / '6YQW.cif').read_text(), fix_mse_residues=True,
                         fix_arginines=True, include_water=False)
mmcif = s.filter(chain_id='A').to_mmcif()
(HERE / 'msa_template_6yqw_A.cif').write_text(mmcif)
base = list(folding_input.load_fold_inputs_from_path(REPO / 'data' / 'inputs_official' / '6YQW.json'))[0]
frozen = json.loads((REPO / 'data' / 'inputs' / '6YQW.json').read_text())
msa = '\n'.join(next(e['protein'] for e in frozen['sequences'] if 'protein' in e)['unpairedMsa'].splitlines()[:48]) + '\n'
prot = base.protein_chains[0]
n = len(prot.sequence)
chain = folding_input.ProteinChain(
    id=prot.id, sequence=prot.sequence, ptms=prot.ptms, description=prot.description, unpaired_msa=msa, paired_msa='',
    templates=[folding_input.Template(mmcif=mmcif, query_to_template_map={i: i for i in range(n)})])
chains = [chain if c is prot else c for c in base.chains]
d = dataclasses.replace(base, chains=chains)
(HERE / 'msa_data_6yqw.json').write_text(d.to_json())
print('wrote', HERE / 'msa_template_6yqw_A.cif', HERE / 'msa_data_6yqw.json', 'query length', n)
