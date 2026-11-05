"""Targets of follow-up 10 (label large) on the whole PDB: the selection rule of the draft specification row, applied
read-only to the RCSB PDB. No study input is built: no MSA, no input JSON.

Run inside the pinned AlphaFold3 environment (token counts and the CCD come from AlphaFold3; the mmCIF files are read
with AlphaFold3's own CIF reader):

    cd third_party/alphafold3
    .venv/bin/python <repository>/targets/large/pdb_screen.py

Every RCSB answer is cached under data/pdb_large_screen/ (search results, GraphQL entries, one mmCIF per checked entry,
token counts), so a rerun reproduces the screen offline; the screen date is the date of the first search, recorded in
search.json and in the outputs.

Source: RCSB PDB entries released after 2021-09-30 (AlphaFold3's training cutoff), X-ray or EM, resolution at most
3.0 A, protein only (no nucleic acid entity), exactly one protein entity, at least one non-polymer entity.
Rule (fixed before any prediction; nothing has been predicted for these targets):
  1. one protein entity, polypeptide(L), whose RCSB canonical sequence has the 20 standard letters only; no nucleic
     acid in the entry; not one of the main study's 60 or follow-up 11's 72 PoseBusters entries;
  2. one ligand: the candidate ligands are the non-polymer components not in the screen's additive, ion and cofactor
     list (ADDITIVES), of formula weight 150 to 1,000 Da, in AlphaFold3's CCD. Of all their instances in model 1 of
     the deposited mmCIF, the one with the most protein heavy atoms within 4 A of its heavy atoms; ties: the lowest
     label_asym_id. An entry whose best instance touches no protein atom is excluded;
  3. single-chain pocket: every protein heavy atom within 6 A of that instance's heavy atoms (all alternate locations
     counted) belongs to one protein chain (label_asym_id);
  4. size: AlphaFold3's token count of that chain plus one copy of the ligand by CCD code (empty MSA, no templates;
     inputs/af3_featurise.py): large 1,025 to 1,536; information only 1,537 to 2,048; control at most 1,024;
  5. one target per protein: RCSB sequence clusters at 30% identity (Search API grouping; an entity not clustered yet
     is its own cluster); within a cluster, members in order of
     resolution, then release date, then PDB ID, the first that meets rules 1 to 4;
  6. control: clusters with a member of at most 1,024 tokens and not in the large group, in strata of experimental
     method (X-ray, EM) and resolution band (<= 2.0, 2.0 to 2.5, 2.5 to 3.0 A); in each stratum as many targets as
     the large group has there. numpy.random.default_rng(2026) permutes each stratum's clusters (strata in a fixed
     order, cluster IDs sorted as strings); clusters are taken in that order, each with its first member of that stratum that
     meets rules 1 to 4, until the stratum is full; a cluster taken in one stratum is not used again;
  7. variant A (sensitivity subset): the selected targets whose deposited model has exactly one polymer chain.
Recorded per entry, not a rule: the number of the ligand instance's heavy atoms modelled against the CCD's.
Outputs: targets/large/pdb_screen_entries.csv (every checked entry, its status and reason at each step),
targets/large/pdb_targets.csv (the large and control groups), targets/large/pdb_screen_counts.md (counts per step).
"""
import collections
import csv
import datetime
import gzip
import hashlib
import json
import pathlib
import sys
import time
import urllib.request

import numpy as np
from rdkit import Chem
from rdkit import RDLogger

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'inputs'))
import af3_featurise  # noqa: E402
import http_util  # noqa: E402
from alphafold3.cpp import cif_dict  # noqa: E402

CACHE = REPO / 'data' / 'pdb_large_screen'
OUT = REPO / 'targets' / 'large'
SEARCH = 'https://search.rcsb.org/rcsbsearch/v2/query'
GQL = 'https://data.rcsb.org/graphql'
CIF_URL = 'https://files.rcsb.org/download/{}.cif.gz'
CUTOFF = '2021-09-30'
MAX_RES = 3.0
METHODS = {'X-RAY DIFFRACTION': 'X-ray', 'ELECTRON MICROSCOPY': 'EM'}
# The additive, ion and cofactor list of ~/Desktop/af3_drafts/fu10_pdb/pdb_large_screen.py, unchanged.
ADDITIVES = set('''GOL EDO PEG PG4 PGE 1PE P6G PE4 PE5 PE8 2PE 12P 15P MPD MRD IPA EOH MOH DMS ACT ACY FMT TRS EPE MES HEPES
MPO PO4 SO4 NO3 CL BR IOD NA K MG CA ZN MN FE FE2 CO NI CU CU1 CD HG SR BA CS RB LI AL GA PT AU AG YB SM GD TB EU
ACE NH2 BME DTT TCEP CIT FLC TAR MLI MAL SUC TRE GLC BGC NAG MAN BMA FUC GAL NDG SIA UNL UNX OXY HOH DOD PEO
BOG LDA LMT DDM UMQ CPS CHS CLR Y01 OLA OLC PLM MYR STE HEX DAO LMU DMU 7PH PC1 POV PEE PLC D10 HTG SOG BNG LMN
FAD FMN NAD NAP NDP NAI NADP HEM HEC HEA SF4 FES F3S ATP ADP AMP ANP ACP AGS GTP GDP GNP GCP GSP SAM SAH COA'''.split())
# The amended ligand rule (--ligand_rule amended), proposed after the screen's lists were found to let lipids,
# nucleotide cofactors, buffers and modified residues through: also exclude components whose CCD type is not
# non-polymer, with a PSI-MOD annotation (post-translational groups), with an acyclic carbon chain of 8 or more atoms
# (fatty acids, phospholipids, detergents, alkanes), with a nucleoside phosphate (nucleotides, CoA and its thioesters,
# FAD, NAD and analogues), with a polyethylene glycol run of three ethylene-glycol units (PEG), without carbon or with an
# element outside ORGANIC (ions, metal clusters and metal complexes), those in EXTRA (buffers, chelators, a detergent,
# polyamines, and cofactors and chromophores without a nucleotide, bilins and free pyridoxal phosphate among them, and
# ergosterol, as cholesterol is in ADDITIVES), and covalently linked instances (RCSB instance
# annotation HAS_COVALENT_LINKAGE), as PoseBusters' ligands are non-covalent.
EXTRA = set('BTB BCN TLA SPM SPK CLA BLA BLR H4B TPP MTE ZXF FNR P1T FOL THH PC G3H AGP B3P NTA EDT CXS NHE PIN CA7'
            ' LBV IE5 CYC PLP PMP ERG'.split())
ORGANIC = set('C H D N O S P F CL BR I B SE SI'.split())
PEG = Chem.MolFromSmarts('[#8;!R]~[#6;!R]~[#6;!R]~[#8;!R]~[#6;!R]~[#6;!R]~[#8;!R]~[#6;!R]~[#6;!R]~[#8;!R]')
CHAIN8 = Chem.MolFromSmarts('[#6;!R]~[#6;!R]~[#6;!R]~[#6;!R]~[#6;!R]~[#6;!R]~[#6;!R]~[#6;!R]')
NUCLEOTIDE = Chem.MolFromSmarts('[#15]~[#8]~[#6]~[#6;R]1~[#8;R]~[#6;R](~[#7;R])~[#6;R]~[#6;R]1')      # 5' phosphate
NUCLEOTIDE23 = Chem.MolFromSmarts('[#15]~[#8]~[#6;R]1~[#6;R]~[#8;R]~[#6;R](~[#7;R])~[#6;R]1')        # 2' or 3' phosphate
RDLogger.DisableLog('rdApp.*')
LIGAND_RULE = 'screen'
FW = (150.0, 1000.0)
CONTACT, POCKET = 4.0, 6.0
STANDARD = set('ACDEFGHIKLMNPQRSTVWY')
LARGE, INFO = (1025, 1536), (1537, 2048)
SEARCH_LARGE = (950, 2048)    # residues: a ligand of at most 1,000 Da has fewer than 75 heavy atoms
SEARCH_CONTROL = (1, 1024)
BANDS = ((0.0, 2.0, '<=2.0'), (2.0, 2.5, '2.0-2.5'), (2.5, 3.0, '2.5-3.0'))
SEED = 2026
NA_TYPES = ('polydeoxyribonucleotide', 'polyribonucleotide', 'polydeoxyribonucleotide/polyribonucleotide hybrid',
            'peptide nucleic acid')


def post(url, body):
  for t in range(6):
    try:
      req = urllib.request.Request(url, json.dumps(body).encode(), {'Content-Type': 'application/json',
                                                                    'User-Agent': http_util.USER_AGENT})
      with urllib.request.urlopen(req, timeout=180, context=http_util.SSL_CONTEXT) as r:
        txt = r.read()
        return json.loads(txt) if txt else {}
    except Exception as e:  # network or server error: retry, then stop (a rerun resumes from the cache)
      print('  retry', t + 1, e, flush=True)
      time.sleep(10 * (t + 1))
  raise SystemExit(f'no answer from {url}; rerun to resume from the cache')


def node(attr, op, value):
  return {'type': 'terminal', 'service': 'text', 'parameters': {'attribute': attr, 'operator': op, 'value': value}}


def search(name, lo, hi):
  path = CACHE / 'search.json'
  cache = json.loads(path.read_text()) if path.exists() else {}
  if name not in cache:
    nodes = [node('rcsb_entry_info.polymer_entity_count_nucleic_acid', 'equals', 0),
             node('rcsb_entry_info.polymer_entity_count_protein', 'equals', 1),
             node('rcsb_entry_info.nonpolymer_entity_count', 'greater_or_equal', 1),
             node('rcsb_accession_info.initial_release_date', 'greater', CUTOFF),
             node('rcsb_entry_info.resolution_combined', 'less_or_equal', MAX_RES),
             node('exptl.method', 'in', sorted(METHODS)),
             node('entity_poly.rcsb_sample_sequence_length', 'range',
                  {'from': lo, 'to': hi, 'include_lower': True, 'include_upper': True})]
    q = {'query': {'type': 'group', 'logical_operator': 'and', 'nodes': nodes}, 'return_type': 'entry',
         'request_options': {'return_all_hits': True, 'results_content_type': ['experimental']}}
    ids = sorted(h['identifier'] for h in post(SEARCH, q).get('result_set', []))
    cache[name] = {'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds'),
                   'residues': [lo, hi], 'ids': ids}
    CACHE.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cache, indent=0))
  return cache[name]


Q = '''query($ids:[String!]!){entries(entry_ids:$ids){rcsb_id exptl{method}
rcsb_entry_info{resolution_combined deposited_polymer_entity_instance_count}
rcsb_accession_info{initial_release_date}
polymer_entities{entity_poly{type pdbx_seq_one_letter_code_can rcsb_sample_sequence_length}
rcsb_polymer_entity_container_identifiers{asym_ids}}
nonpolymer_entities{rcsb_nonpolymer_entity_container_identifiers{asym_ids}
nonpolymer_comp{chem_comp{id formula_weight}}}}}'''


def entries(ids):
  path = CACHE / 'entries.json.gz'
  cache = json.loads(gzip.decompress(path.read_bytes())) if path.exists() else {}
  todo = sorted(set(ids) - set(cache))
  for i in range(0, len(todo), 200):
    data = post(GQL, {'query': Q, 'variables': {'ids': todo[i:i + 200]}})
    if data.get('errors'):
      raise SystemExit(f'GraphQL error: {data["errors"][:1]}')
    for e in data['data']['entries'] or []:
      cache[e['rcsb_id']] = e
    if (i // 200) % 10 == 9 or i + 200 >= len(todo):
      path.write_bytes(gzip.compress(json.dumps(cache).encode()))
      print(f'  GraphQL: {min(i + 200, len(todo))}/{len(todo)} entries', flush=True)
  return cache


def clusters(ids):
  """{entry ID: RCSB 30% sequence cluster} from the Search API (polymer entities grouped by sequence identity 30%;
  the GraphQL cluster field is empty). An entity RCSB has not clustered yet gets its own cluster, 'none_<entry>'."""
  path = CACHE / 'clusters30.json'
  cache = json.loads(path.read_text()) if path.exists() else {}
  todo = sorted(set(ids) - set(cache))
  for i in range(0, len(todo), 1000):
    chunk = todo[i:i + 1000]
    q = {'query': {'type': 'group', 'logical_operator': 'and', 'nodes': [
            node('rcsb_polymer_entity_container_identifiers.entry_id', 'in', chunk),
            node('entity_poly.rcsb_entity_polymer_type', 'exact_match', 'Protein')]},
         'return_type': 'polymer_entity',
         'request_options': {'group_by': {'aggregation_method': 'sequence_identity', 'similarity_cutoff': 30},
                             'group_by_return_type': 'groups', 'return_all_hits': True}}
    out = post(SEARCH, q)
    for g in out.get('group_set', []):
      for m in g['result_set']:
        cache[m['identifier'].split('_')[0]] = g['identifier']
    for pid in chunk:
      cache.setdefault(pid, f'none_{pid}')
    path.write_text(json.dumps(cache, indent=0, sort_keys=True))
    print(f'  clusters: {min(i + 1000, len(todo))}/{len(todo)} entries', flush=True)
  return cache


def cif(pdb):
  path = CACHE / 'cif' / f'{pdb}.cif.gz'
  if not path.exists():
    path.parent.mkdir(parents=True, exist_ok=True)
    for t in range(6):
      try:
        with urllib.request.urlopen(urllib.request.Request(CIF_URL.format(pdb), headers={'User-Agent': http_util.USER_AGENT}),
                                    timeout=300, context=http_util.SSL_CONTEXT) as r:
          data = r.read()
        break
      except Exception as e:  # retry, then stop
        print('  retry', pdb, t + 1, e, flush=True)
        time.sleep(10 * (t + 1))
    else:
      raise SystemExit(f'{pdb}: mmCIF not downloaded; rerun to resume')
    path.with_suffix('.partial').write_bytes(data)
    path.with_suffix('.partial').rename(path)
    time.sleep(0.2)
  return gzip.decompress(path.read_bytes()).decode()


_ANN = None


def covalent(pdb):
  """label_asym_ids of the entry's non-polymer instances that RCSB annotates HAS_COVALENT_LINKAGE (cached)."""
  global _ANN
  path = CACHE / 'instance_annotations.json'
  if _ANN is None:
    _ANN = json.loads(path.read_text()) if path.exists() else {}
  if not any(k.split()[0] == pdb for k in _ANN):
    q = '''query($ids:[String!]!){entries(entry_ids:$ids){rcsb_id nonpolymer_entities{nonpolymer_comp{chem_comp{id}}
nonpolymer_entity_instances{rcsb_nonpolymer_entity_instance_container_identifiers{asym_id}
rcsb_nonpolymer_instance_annotation{type}}}}}'''
    out = post(GQL, {'query': q, 'variables': {'ids': [pdb]}})
    for e in out['data']['entries'] or []:
      for ne in e['nonpolymer_entities'] or []:
        for inst in ne['nonpolymer_entity_instances'] or []:
          _ANN[f"{pdb} {inst['rcsb_nonpolymer_entity_instance_container_identifiers']['asym_id']}"] = [
              ne['nonpolymer_comp']['chem_comp']['id'],
              sorted({a['type'] for a in inst['rcsb_nonpolymer_instance_annotation'] or []})]
    _ANN.setdefault(f'{pdb} -', ['', []])
    path.write_text(json.dumps(_ANN, indent=0))
  return {k.split()[1] for k, v in _ANN.items() if k.split()[0] == pdb and 'HAS_COVALENT_LINKAGE' in v[1]}


def pocket(pdb, protein_entity_asyms, ligand_comps):
  """Rule 2 (instance) and rule 3 (single-chain pocket) from model 1 of the deposited mmCIF."""
  skip = covalent(pdb) if LIGAND_RULE == 'amended' else set()
  d = cif_dict.from_string(cif(pdb))
  col = {k: np.array(d.get(f'_atom_site.{k}', [])) for k in ('pdbx_PDB_model_num', 'type_symbol', 'label_asym_id',
                                                                'label_comp_id', 'Cartn_x', 'Cartn_y', 'Cartn_z')}
  model = col['pdbx_PDB_model_num']
  keep = (model == model[0]) & ~np.isin(col['type_symbol'], ['H', 'D']) if len(model) else np.zeros(0, bool)
  xyz = np.stack([col[k][keep].astype(float) for k in ('Cartn_x', 'Cartn_y', 'Cartn_z')], axis=1)
  asym_k, comp_k = col['label_asym_id'][keep], col['label_comp_id'][keep]
  prot = np.isin(asym_k, list(protein_entity_asyms))
  best = None
  for a in sorted(set(asym_k[np.isin(comp_k, list(ligand_comps)) & ~prot]) - skip):
    lig = asym_k == a
    ccd = comp_k[lig][0]
    dist = np.sqrt(((xyz[prot][:, None, :] - xyz[lig][None, :, :]) ** 2).sum(-1)).min(1)
    contacts = int((dist <= CONTACT).sum())
    chains = sorted(set(asym_k[prot][dist <= POCKET]))
    cand = {'ligand_asym': str(a), 'ligand_ccd': str(ccd), 'contacts': contacts, 'pocket_chains': [str(c) for c in chains],
            'ligand_atoms_modelled': int(lig.sum())}
    if best is None or contacts > best['contacts']:
      best = cand
  return best


def featurise(seq, ccd, cache):
  key = hashlib.sha256(json.dumps({'sequence': seq, 'ccd': ccd}).encode()).hexdigest()
  if key not in cache:
    try:
      cache[key] = af3_featurise.featurise(af3_featurise.protein_ligand_json('x', seq, ccd, seeds=[1]))
    except Exception as e:  # AlphaFold3 rejects the input
      cache[key] = {'error': f'{type(e).__name__}: {e}'[:200]}
    (CACHE / 'token_counts.json').write_text(json.dumps(cache, indent=0, sort_keys=True))
  return cache[key]


def band_of(res):
  return next(name for lo, hi, name in BANDS if lo < res <= hi or (lo == 0.0 and res <= hi))


def static(e, exclude, cl30):
  """Rules 1 and 2 from the RCSB record: (row, reason or None)."""
  pid = e['rcsb_id']
  methods = [m['method'] for m in e['exptl']]
  res = (e['rcsb_entry_info']['resolution_combined'] or [None])[0]
  row = {'pdb_id': pid, 'method': '/'.join(METHODS.get(m, m) for m in methods), 'resolution': res,
         'release': e['rcsb_accession_info']['initial_release_date'][:10],
         'chains_deposited': e['rcsb_entry_info']['deposited_polymer_entity_instance_count']}
  polys = e['polymer_entities'] or []
  if len(methods) != 1 or methods[0] not in METHODS or res is None or res > MAX_RES:
    return row, 'method or resolution'
  if any(p['entity_poly']['type'] in NA_TYPES for p in polys):
    return row, 'nucleic acid in the entry'
  prot = [p for p in polys if p['entity_poly']['type'] == 'polypeptide(L)']
  if len(prot) != 1 or len(polys) != 1:
    return row, 'not one polypeptide(L) entity'
  seq = ''.join((prot[0]['entity_poly']['pdbx_seq_one_letter_code_can'] or '').split())
  row.update(residues=len(seq), sequence=seq, cluster30=cl30.get(pid),
             protein_asyms=' '.join(prot[0]['rcsb_polymer_entity_container_identifiers']['asym_ids'] or []))
  if not seq or set(seq) - STANDARD:
    return row, 'non-standard letters in the canonical sequence'
  if pid in exclude:
    return row, 'a main-study or follow-up 11 entry'
  ccd = af3_featurise.ccd()
  ligs = sorted({n['nonpolymer_comp']['chem_comp']['id'] for n in (e['nonpolymer_entities'] or [])
                 if n['nonpolymer_comp']['chem_comp']['id'] not in ADDITIVES
                 and FW[0] <= (n['nonpolymer_comp']['chem_comp']['formula_weight'] or 0) <= FW[1]})
  row['drug_like'] = ' '.join(ligs)
  if not ligs:
    return row, 'no drug-like ligand'
  in_ccd = [l for l in ligs if l in ccd]
  row['drug_like_in_ccd'] = ' '.join(in_ccd)
  if not in_ccd:
    return row, 'no drug-like ligand in the AlphaFold3 CCD'
  if LIGAND_RULE == 'amended':
    kept = [l for l in in_ccd if amended_ok(l, ccd)]
    row['drug_like_in_ccd'] = ' '.join(kept)
    if not kept:
      return row, 'no drug-like ligand under the amended rule'
  return row, None


_PSIMOD = None


def amended_ok(code, ccd):
  """The component-level part of the amended ligand rule (the instance-level part, non-covalent, is in check)."""
  global _PSIMOD
  if _PSIMOD is None:
    _PSIMOD = json.loads((CACHE / 'psimod.json').read_text()) if (CACHE / 'psimod.json').exists() else {}
  if code in EXTRA:
    return False
  c = ccd.get(code) or {}
  if (c.get('_chem_comp.type') or ['?'])[0].lower() != 'non-polymer':
    return False
  elements = {x.upper() for x in c.get('_chem_comp_atom.type_symbol', [])}
  if 'C' not in elements or elements - ORGANIC:
    return False
  if code not in _PSIMOD:
    q = '''query($ids:[String!]!){chem_comps(comp_ids:$ids){chem_comp{id} rcsb_chem_comp_annotation{type}}}'''
    out = post(GQL, {'query': q, 'variables': {'ids': [code]}})
    for cc in out['data']['chem_comps'] or []:
      _PSIMOD[cc['chem_comp']['id']] = any(a['type'] == 'PSI-MOD' for a in cc['rcsb_chem_comp_annotation'] or [])
    _PSIMOD.setdefault(code, False)
    (CACHE / 'psimod.json').write_text(json.dumps(_PSIMOD, indent=0, sort_keys=True))
  if _PSIMOD[code]:
    return False
  types = c.get('_pdbx_chem_comp_descriptor.type', [])
  descs = c.get('_pdbx_chem_comp_descriptor.descriptor', [])
  smiles = next((d for t, d in zip(types, descs) if t == 'SMILES_CANONICAL'), None) or \
      next((d for t, d in zip(types, descs) if t == 'SMILES'), None)
  mol = (Chem.MolFromSmiles(smiles) or Chem.MolFromSmiles(smiles, sanitize=False)) if smiles else None
  if mol is not None:
    Chem.FastFindRings(mol)   # ring information for the [R] queries, also on an unsanitized molecule
  if mol is None:
    return True    # no descriptor: the chain and nucleotide tests cannot apply
  return not any(mol.HasSubstructMatch(q) for q in (CHAIN8, NUCLEOTIDE, NUCLEOTIDE23, PEG))


def check(row, tokcache):
  """Rules 2 (instance), 3 and 4 on one entry that passed rules 1 and 2: sets the row's fields, returns a reason."""
  p = pocket(row['pdb_id'], set(row['protein_asyms'].split()), set(row['drug_like_in_ccd'].split()))
  if p is None:
    return 'no instance of a drug-like ligand in model 1'
  row.update(ligand_ccd=p['ligand_ccd'], ligand_asym=p['ligand_asym'], contacts=p['contacts'],
             pocket_chains=' '.join(p['pocket_chains']), ligand_atoms_modelled=p['ligand_atoms_modelled'])
  if p['contacts'] == 0:
    return 'the ligand touches no protein atom'
  if len(p['pocket_chains']) != 1:
    return 'pocket on more than one chain'
  f = featurise(row['sequence'], p['ligand_ccd'], tokcache)
  if 'error' in f:
    return 'AlphaFold3 featurisation failed: ' + f['error']
  row.update(num_tokens=f['num_tokens'], bucket=f['bucket'],
             ligand_atoms_ccd=f['num_tokens'] - row['residues'])
  return None


def main():
  global LIGAND_RULE
  if len(sys.argv) > 1:
    if sys.argv[1:] != ['--ligand_rule', 'amended'] and sys.argv[1:] != ['--ligand_rule', 'screen']:
      raise SystemExit('usage: pdb_screen.py [--ligand_rule screen|amended]')
    LIGAND_RULE = sys.argv[2]
  sfx = '' if LIGAND_RULE == 'screen' else '_amended'
  started = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds')
  exclude = {r['pdb_id'] for f in ('targets/posebusters_subset.csv', 'targets/repl/repl_subset.csv')
             if (REPO / f).is_file() for r in csv.DictReader(open(REPO / f))}
  if not (REPO / 'targets' / 'repl' / 'repl_subset.csv').is_file():   # branch fu11-12 holds it before its merge
    alt = pathlib.Path.home() / 'Desktop' / 'af3-fu11' / 'targets' / 'repl' / 'repl_subset.csv'
    exclude |= {r['pdb_id'] for r in csv.DictReader(open(alt))}
  s_large, s_ctrl = search('large', *SEARCH_LARGE), search('control', *SEARCH_CONTROL)
  print(f'search: {len(s_large["ids"])} entries of {SEARCH_LARGE[0]} to {SEARCH_LARGE[1]} residues, '
        f'{len(s_ctrl["ids"])} of at most {SEARCH_CONTROL[1]} ({s_large["utc"]})', flush=True)
  ent = entries(s_large['ids'] + s_ctrl['ids'])
  cl30 = clusters(s_large['ids'] + s_ctrl['ids'])
  tpath = CACHE / 'token_counts.json'
  tokcache = json.loads(tpath.read_text()) if tpath.exists() else {}
  rows = {}
  for pid in sorted(set(s_large['ids']) | set(s_ctrl['ids'])):
    if pid not in ent:
      rows[pid] = {'pdb_id': pid, 'status': 'excluded', 'reason': 'not in the GraphQL answer'}
      continue
    row, why = static(ent[pid], exclude, cl30)
    row.update(status='excluded' if why else 'candidate', reason=why or '', checked='no')
    rows[pid] = row
  order = lambda r: (float(r['resolution']), r['release'], r['pdb_id'])  # noqa: E731

  # Large and information bands: every candidate of 950 to 2,048 residues is checked (rules 2 to 4), so the counts
  # after each step are complete.
  big = [r for r in rows.values() if r['status'] == 'candidate' and r['residues'] >= SEARCH_LARGE[0]]
  for n, r in enumerate(sorted(big, key=lambda r: r['pdb_id']), 1):
    why = check(r, tokcache)
    r.update(checked='yes', status='excluded' if why else 'eligible', reason=why or '')
    if n % 25 == 0:
      print(f'  checked {n}/{len(big)} large-band candidates', flush=True)
  in_band = lambda r, b: r['status'] == 'eligible' and b[0] <= r['num_tokens'] <= b[1]  # noqa: E731
  groups = {}
  for name, band in (('large', LARGE), ('information', INFO)):
    by = collections.defaultdict(list)
    for r in rows.values():
      if in_band(r, band):
        by[r['cluster30']].append(r)
    groups[name] = sorted((min(m, key=order) for m in by.values()), key=lambda r: r['pdb_id'])
  large = groups['large']
  quota = collections.Counter((r['method'], band_of(float(r['resolution']))) for r in large)
  large_clusters = {r['cluster30'] for r in large}

  # Control: strata in a fixed order; one permutation per stratum from one generator.
  rng = np.random.default_rng(SEED)
  strata = [(m, b) for m in ('X-ray', 'EM') for _, _, b in BANDS]
  pool = [r for r in rows.values() if r['status'] == 'candidate' and r['residues'] <= SEARCH_CONTROL[1]
          and r['cluster30'] not in large_clusters]
  control, used, tried = [], set(), collections.Counter()
  for st in strata:
    by = collections.defaultdict(list)
    for r in pool:
      if (r['method'], band_of(float(r['resolution']))) == st:
        by[r['cluster30']].append(r)
    ids = sorted(by)
    perm = rng.permutation(len(ids))
    need = quota.get(st, 0)
    for i in perm:
      if need == 0:
        break
      cid = ids[i]
      if cid in used:
        continue
      for r in sorted(by[cid], key=order):
        if r['checked'] == 'no':
          why = check(r, tokcache)
          r.update(checked='yes', status='excluded' if why else 'eligible', reason=why or '')
          tried[st] += 1
        if r['status'] == 'eligible' and r['num_tokens'] <= 1024:
          control.append(dict(r, stratum=f'{st[0]} {st[1]}'))
          used.add(cid)
          need -= 1
          break
    print(f'  control {st}: {quota.get(st, 0) - need} of {quota.get(st, 0)} ({tried[st]} entries checked, '
          f'{len(ids)} clusters in the stratum)', flush=True)

  OUT.mkdir(parents=True, exist_ok=True)
  cols = ['pdb_id', 'status', 'reason', 'checked', 'method', 'resolution', 'release', 'chains_deposited', 'residues',
          'cluster30', 'drug_like', 'drug_like_in_ccd', 'ligand_ccd', 'ligand_asym', 'contacts', 'pocket_chains',
          'ligand_atoms_modelled', 'ligand_atoms_ccd', 'num_tokens', 'bucket']
  head = (f'# targets/large/pdb_screen.py, run {started}; RCSB search of {s_large["utc"]}; cache data/pdb_large_screen/; '
          f'AlphaFold3 buckets {list(af3_featurise.BUCKETS)}; ligand rule {LIGAND_RULE}\n')
  with open(OUT / f'pdb_screen_entries{sfx}.csv', 'w', newline='') as f:
    f.write(head)
    w = csv.DictWriter(f, cols, extrasaction='ignore', lineterminator='\n')
    w.writeheader()
    w.writerows(sorted(rows.values(), key=lambda r: r['pdb_id']))
  sel = [dict(r, group='large', stratum=f'{r["method"]} {band_of(float(r["resolution"]))}') for r in large] + \
        [dict(r, group='control') for r in control] + \
        [dict(r, group='information 1537-2048', stratum='') for r in groups['information']]
  for r in sel:
    r['variant_A'] = 'yes' if int(r['chains_deposited']) == 1 else 'no'
  with open(OUT / f'pdb_targets{sfx}.csv', 'w', newline='') as f:
    f.write(head)
    w = csv.DictWriter(f, ['group', 'stratum', 'variant_A', 'pdb_id'] + cols[4:] + ['sequence'], extrasaction='ignore',
                       lineterminator='\n')
    w.writeheader()
    w.writerows(sel)
  report(rows, large, control, groups['information'], quota, s_large, s_ctrl, started, sfx)


def report(rows, large, control, info, quota, s_large, s_ctrl, started, sfx):
  allr = list(rows.values())
  big = [r for r in allr if r.get('residues', 0) >= SEARCH_LARGE[0]]
  L = [f'# Follow-up 10 (large) on the whole PDB: counts per step (targets/large/pdb_screen.py, ligand rule {LIGAND_RULE})', '',
       f'Screen run {started}; RCSB search {s_large["utc"]}: {len(s_large["ids"])} entries of {SEARCH_LARGE[0]} to '
       f'{SEARCH_LARGE[1]} residues and {len(s_ctrl["ids"])} of at most {SEARCH_CONTROL[1]} (released after {CUTOFF}, '
       f'X-ray or EM, resolution <= {MAX_RES} A, protein only, one protein entity, a non-polymer entity).', '',
       '## Large band (950 to 2,048 residues in the search): every entry checked', '',
       '| step | entries | X-ray | EM | clusters (30%) |', '|---|---|---|---|---|']

  def line(name, rs):
    return (f'| {name} | {len(rs)} | {sum(r["method"] == "X-ray" for r in rs)} | {sum(r["method"] == "EM" for r in rs)} | '
            f'{len({r.get("cluster30") for r in rs})} |')
  steps = [('from the search', lambda r: True),
           ('rule 1: one polypeptide(L) entity, standard letters, no nucleic acid, not a PoseBusters study entry',
            lambda r: r['reason'] not in ('method or resolution', 'nucleic acid in the entry', 'not one polypeptide(L) entity',
                                          'non-standard letters in the canonical sequence', 'a main-study or follow-up 11 entry',
                                          'not in the GraphQL answer')),
           ('rule 2: a drug-like ligand in the AlphaFold3 CCD, an instance touching the protein',
            lambda r: r['checked'] == 'yes' and r['reason'] not in ('no instance of a drug-like ligand in model 1',
                                                                    'the ligand touches no protein atom')),
           ('rule 3: single-chain pocket (6 A)', lambda r: r['checked'] == 'yes' and (
               r['reason'] == '' or r['reason'].startswith('AlphaFold3 featurisation'))),
           ('rule 4: featurised', lambda r: r['status'] == 'eligible')]
  for name, f in steps:
    L.append(line(name, [r for r in big if f(r)]))
  elig = [r for r in big if r['status'] == 'eligible']
  for name, (lo, hi) in (('tokens 1,025 to 1,536 (large)', LARGE), ('tokens 1,537 to 2,048 (information)', INFO),
                         ('tokens at most 1,024', (0, 1024))):
    L.append(line(name, [r for r in elig if lo <= r['num_tokens'] <= hi]))
  L += ['', f'Entries without an RCSB 30% cluster (each its own cluster): '
        f'{sum(str(r.get("cluster30", "")).startswith("none_") for r in allr if r.get("status") != "excluded" or r.get("checked") == "yes")}.',
        '', f'Rule 3 removes {sum(r["reason"] == "pocket on more than one chain" for r in big)} entries of the large band.', '',
        '## Selected (rule 5: one per 30% cluster)', '',
        '| group | targets | variant A (one deposited chain) | by stratum | by bucket |', '|---|---|---|---|---|']
  for name, rs in (('large 1,025 to 1,536', large), ('control at most 1,024', control), ('information 1,537 to 2,048', info)):
    st = collections.Counter(f'{r["method"]} {band_of(float(r["resolution"]))}' for r in rs)
    bk = collections.Counter(r['bucket'] for r in rs)
    L.append(f'| {name} | {len(rs)} | {sum(int(r["chains_deposited"]) == 1 for r in rs)} | '
             f'{", ".join(f"{k}: {v}" for k, v in sorted(st.items()))} | {", ".join(f"{k}: {v}" for k, v in sorted(bk.items()))} |')
  ctl_checked = [r for r in allr if r.get('residues', 0) < SEARCH_LARGE[0] and r.get('checked') == 'yes']
  L += ['', f'Control: {len(ctl_checked)} entries checked in draw order; reasons: '
        + ', '.join(f'{k or "eligible"} {v}' for k, v in collections.Counter(r['reason'] for r in ctl_checked).items())]
  (OUT / f'pdb_screen_counts{sfx}.md').write_text('\n'.join(L) + '\n')
  print('\n'.join(L))


if __name__ == '__main__':
  main()
