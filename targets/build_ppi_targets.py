"""Selects the protein-protein targets of follow-up experiment 1 (notes/analysis_plan.md section 10, row
"Follow-up experiment 1, protein-protein complexes" of 2026-10-14), by its fixed rule.

    analysis/ppi/.venv/bin/python targets/build_ppi_targets.py

Environment: analysis/requirements-ppi.txt (gemmi for coordinates, Biopython for alignments). Public APIs only:
the RCSB Search API (search.rcsb.org), Data API (data.rcsb.org, GraphQL) and file server (files.rcsb.org);
downloads cached in data/ppi/ (never committed).

Rule, in order (the pool size after each step is recorded):
  1. RCSB Search API, entries: initial release 2021-10-01 to 2024-12-31; experimental method X-ray diffraction;
     resolution 2.5 A or better; exactly two protein polymer entities; no nucleic-acid (or hybrid) entity. The
     filters are added one query at a time, each query and its count recorded.
  2. Data API, per entry: re-checks 1 (release date in range, X-ray as the only method, resolution, two polymer
     entities, both protein), then the first biological assembly (assembly_id 1) must contain exactly one chain
     (instance) of each of the two protein entities and no other polymer chain (instances counted over
     pdbx_struct_assembly_gen: asym IDs times operators).
  3. Each chain 30 to 700 residues and at most 1,000 in total, counted on the entity's canonical sequence
     (entity_poly.pdbx_seq_one_letter_code_can; the AlphaFold3 input sequence, as in the main study).
  4. The two chains in contact: at least 10 residue pairs (one residue from each chain) with a heavy-atom pair
     within 5 A, in the coordinates of the first biological assembly (files.rcsb.org <ID>-assembly1.cif;
     first model, first alternative conformation, hydrogens removed).
  5. Redundancy at 30% sequence identity: the entries taken in random order (numpy default_rng(2026)
     permutation of the pool sorted by ID); an entry is dropped if either of its chains is at least 30%
     identical to a chain of an entry already kept. Identity of two chains (the rule does not define it; this
     definition is recorded): identical aligned residue pairs / length of the shorter chain, in an optimal
     global alignment with BLOSUM62, gap open -11, gap extend -1 (BLAST's defaults) and free end gaps
     (Bio.Align.PairwiseAligner).
  6. Draw: a permutation of the kept pool (sorted by ID) with numpy default_rng(2026); the first 30 are the
     targets (all if fewer). The rest of the permutation, in order, is the replacement list: a target whose
     input cannot be built is replaced by the next one in it (inputs/ppi/build_ppi_inputs.py, before any run).
Outputs in targets/ppi/: queries.json (every Search and Data API query, its UTC date and its result count),
pool_steps.csv (pool size after each step), candidates.csv (every entry after step 1 with each check's
outcome), redundancy.csv (the clustering: order, kept or dropped, and for a drop the kept entry and chain, and
the identity), draw.csv (the whole draw order) and targets.csv (the 30 targets: PDB ID, chains, lengths,
sequences, tokens, bucket, release date), and sources.json.
"""
import csv
import datetime
import gzip
import hashlib
import http.client as http_client
import io
import json
import pathlib
import re
import sys
import time
import urllib.error
import urllib.request

import gemmi
import numpy as np
from Bio.Align import PairwiseAligner, substitution_matrices

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / 'inputs'))
import http_util  # noqa: E402  (system CA bundle on python.org builds; the project's user agent)
OUT = REPO / 'targets' / 'ppi'
CACHE = REPO / 'data' / 'ppi'
SEARCH = 'https://search.rcsb.org/rcsbsearch/v2/query'
GRAPHQL = 'https://data.rcsb.org/graphql'
FILES = 'https://files.rcsb.org/download'
USER_AGENT = http_util.USER_AGENT
DATE_FROM, DATE_TO = '2021-10-01', '2024-12-31'
MAX_RESOLUTION = 2.5
MIN_LEN, MAX_LEN, MAX_TOTAL = 30, 700, 1000
CONTACT_A, MIN_CONTACT_PAIRS = 5.0, 10
IDENTITY = 0.30
SEED = 2026
N_TARGETS = 30
BUCKETS = (256, 512, 768, 1024, 1280)
QUERY_LOG = []


def utc_now():
  return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds')


def http(url, data=None, tries=6):
  body = json.dumps(data).encode() if data is not None else None
  for attempt in range(tries):
    try:
      req = urllib.request.Request(url, data=body, headers={'User-Agent': USER_AGENT,
                                                            **({'Content-Type': 'application/json'} if body else {})})
      with urllib.request.urlopen(req, timeout=120, context=http_util.SSL_CONTEXT) as r:
        return r.read()
    except urllib.error.HTTPError as e:
      if e.code == 404:
        raise
      if attempt == tries - 1:
        raise
    except (urllib.error.URLError, http_client.HTTPException, OSError):   # includes dropped connections
      if attempt == tries - 1:
        raise
    time.sleep(5 * 2 ** attempt)


def terminal(attribute, operator, value):
  return {'type': 'terminal', 'service': 'text', 'parameters': {'attribute': attribute, 'operator': operator,
                                                                'value': value}}


SEARCH_STEPS = [
    ('release 2021-10-01 to 2024-12-31', terminal('rcsb_accession_info.initial_release_date', 'range',
                                                  {'from': DATE_FROM, 'to': DATE_TO, 'include_lower': True,
                                                   'include_upper': True})),
    ('X-ray diffraction', terminal('exptl.method', 'exact_match', 'X-RAY DIFFRACTION')),
    ('resolution <= 2.5 A', terminal('rcsb_entry_info.resolution_combined', 'less_or_equal', MAX_RESOLUTION)),
    ('exactly two protein entities', terminal('rcsb_entry_info.polymer_entity_count_protein', 'equals', 2)),
    ('no nucleic-acid entity', terminal('rcsb_entry_info.polymer_entity_count_nucleic_acid', 'equals', 0)),
    ('no nucleic-acid hybrid entity', terminal('rcsb_entry_info.polymer_entity_count_nucleic_acid_hybrid', 'equals', 0)),
]


def search(nodes, ids=False):
  query = {'query': {'type': 'group', 'logical_operator': 'and', 'nodes': nodes}, 'return_type': 'entry',
           'request_options': {'results_content_type': ['experimental'],
                               **({'return_all_hits': True} if ids else {'return_counts': True})}}
  out = json.loads(http(SEARCH, query))
  ids_out = sorted(r['identifier'] for r in out.get('result_set', [])) if ids else None
  QUERY_LOG.append({'utc': utc_now(), 'service': 'search', 'query': query, 'total_count': out.get('total_count'),
                    'ids_sha256': hashlib.sha256('\n'.join(ids_out).encode()).hexdigest() if ids else None})
  return (out.get('total_count'), ids_out)


GQL = '''query($ids: [String!]!) { entries(entry_ids: $ids) {
  rcsb_id
  rcsb_accession_info { initial_release_date }
  exptl { method }
  rcsb_entry_info { resolution_combined polymer_entity_count polymer_entity_count_protein
                    polymer_entity_count_nucleic_acid polymer_entity_count_nucleic_acid_hybrid }
  assemblies { rcsb_assembly_container_identifiers { assembly_id }
               pdbx_struct_assembly_gen { asym_id_list oper_expression } }
  polymer_entities { rcsb_polymer_entity_container_identifiers { entity_id asym_ids auth_asym_ids }
                     entity_poly { type pdbx_seq_one_letter_code_can rcsb_entity_polymer_type } }
} }'''


def entry_metadata(ids):
  path = CACHE / 'metadata.json'
  cached = json.loads(path.read_text()) if path.exists() else {}
  todo = [i for i in ids if i not in cached]
  for k in range(0, len(todo), 100):
    batch = todo[k:k + 100]
    out = json.loads(http(GRAPHQL, {'query': GQL, 'variables': {'ids': batch}}))
    if out.get('errors'):
      raise SystemExit(f'GraphQL errors: {out["errors"][:2]}')
    for e in out['data']['entries']:
      cached[e['rcsb_id']] = e
    QUERY_LOG.append({'utc': utc_now(), 'service': 'data (graphql)', 'entries_requested': len(batch),
                      'entries_returned': len(out['data']['entries'])})
    path.write_text(json.dumps(cached))
  return {i: cached[i] for i in ids if i in cached}


def operator_count(expr):
  """Number of operators an oper_expression generates: '1', '1,2', '(1-3)', '(1,2)(3,4)' (a product)."""
  groups = re.findall(r'\(([^)]*)\)', expr) or [expr]
  n = 1
  for g in groups:
    count = 0
    for part in g.split(','):
      part = part.strip()
      if '-' in part:
        a, b = part.split('-')
        count += int(b) - int(a) + 1 if a.isdigit() and b.isdigit() else 1
      elif part:
        count += 1
    n *= count
  return n


def check_entry(e):
  """(reason or '', dict of chains) for steps 2 and 3."""
  info = e['rcsb_entry_info']
  date = e['rcsb_accession_info']['initial_release_date'][:10]
  if not DATE_FROM <= date <= DATE_TO:
    return f'release date {date}', {}
  methods = sorted({x['method'] for x in e.get('exptl') or []})
  if methods != ['X-RAY DIFFRACTION']:
    return f'methods {methods}', {}
  res = (info.get('resolution_combined') or [None])[0]
  if res is None or res > MAX_RESOLUTION:
    return f'resolution {res}', {}
  if info['polymer_entity_count'] != 2 or info['polymer_entity_count_protein'] != 2:
    return f'polymer entities {info["polymer_entity_count"]}, protein {info["polymer_entity_count_protein"]}', {}
  ents = e['polymer_entities']
  asym_entity = {a: p['rcsb_polymer_entity_container_identifiers']['entity_id'] for p in ents
                 for a in p['rcsb_polymer_entity_container_identifiers']['asym_ids']}
  a1 = [a for a in e.get('assemblies') or [] if a['rcsb_assembly_container_identifiers']['assembly_id'] == '1']
  if not a1:
    return 'no assembly 1', {}
  counts = {}
  for gen in a1[0]['pdbx_struct_assembly_gen'] or []:
    ops = operator_count(gen['oper_expression'])
    for a in gen['asym_id_list']:
      if a in asym_entity:
        counts.setdefault(asym_entity[a], []).extend([a] * ops)
  per_entity = {eid: len(v) for eid, v in counts.items()}
  if sorted(per_entity) != sorted(p['rcsb_polymer_entity_container_identifiers']['entity_id'] for p in ents) \
     or any(n != 1 for n in per_entity.values()):
    return f'assembly 1 polymer chains per entity {per_entity}', {}
  chains = {}
  for p in ents:
    ids = p['rcsb_polymer_entity_container_identifiers']
    asym = counts[ids['entity_id']][0]
    auth = ids['auth_asym_ids'][ids['asym_ids'].index(asym)] if asym in ids['asym_ids'] and ids.get('auth_asym_ids') else ''
    seq = re.sub(r'\s', '', p['entity_poly']['pdbx_seq_one_letter_code_can'])
    chains[ids['entity_id']] = {'entity': ids['entity_id'], 'asym_id': asym, 'auth_asym_id': auth, 'sequence': seq,
                                'length': len(seq), 'poly_type': p['entity_poly']['type']}
  lens = [c['length'] for c in chains.values()]
  if min(lens) < MIN_LEN or max(lens) > MAX_LEN:
    return f'chain lengths {lens}', chains
  if sum(lens) > MAX_TOTAL:
    return f'total length {sum(lens)}', chains
  return '', chains


def assembly_file(pdb_id):
  path = CACHE / 'assembly1' / f'{pdb_id}-assembly1.cif.gz'
  if not path.exists():
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.part')
    tmp.write_bytes(http(f'{FILES}/{pdb_id}-assembly1.cif.gz'))
    tmp.replace(path)   # atomic: a cached file is always complete
  return path


def contact_pairs(pdb_id):
  """(residue pairs within CONTACT_A between the two protein chains of assembly 1, note)."""
  st = gemmi.read_structure(str(assembly_file(pdb_id)))
  st.setup_entities()
  st.remove_hydrogens()
  st.remove_alternative_conformations()
  model = st[0]
  chains = [ch for ch in model if len(ch.get_polymer()) and ch.get_polymer().check_polymer_type()
            in (gemmi.PolymerType.PeptideL, gemmi.PolymerType.PeptideD)]
  if len(chains) != 2:
    return None, f'{len(chains)} protein chains in the assembly file'
  a, b = chains
  ns = gemmi.NeighborSearch(model, st.cell, 6).populate()
  pairs = set()
  for res in a.get_polymer():
    for atom in res:
      for mark in ns.find_atoms(atom.pos, '\0', radius=CONTACT_A):
        cra = mark.to_cra(model)
        if cra.chain.name == b.name and cra.atom.pos.dist(atom.pos) <= CONTACT_A:
          pairs.add((str(res.seqid), str(cra.residue.seqid)))
  return len(pairs), f'chains {a.name}, {b.name}'


def aligner():
  al = PairwiseAligner()
  al.mode = 'global'
  al.substitution_matrix = substitution_matrices.load('BLOSUM62')
  al.open_gap_score, al.extend_gap_score = -11, -1
  al.target_end_gap_score = al.query_end_gap_score = 0.0
  return al


def identity(al, s, t):
  aln = al.align(s, t)[0]
  return aln.counts().identities / min(len(s), len(t))


def bucket(tokens):
  return next(b for b in BUCKETS if tokens <= b)


def write_csv(path, rows, fields=None):
  fields = fields or list(rows[0])
  with open(path, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore', lineterminator='\n')
    w.writeheader()
    w.writerows(rows)


def main():
  OUT.mkdir(parents=True, exist_ok=True)
  CACHE.mkdir(parents=True, exist_ok=True)
  steps = []
  nodes = []
  for label, node in SEARCH_STEPS:
    nodes.append(node)
    count, _ = search(list(nodes))
    steps.append({'step': f'1 search: {label}', 'pool': count})
    print(f'{steps[-1]["step"]}: {count}', flush=True)
  count, ids = search(list(nodes), ids=True)
  meta = entry_metadata(ids)
  rows, passed = [], []
  reasons_2, reasons_3 = 0, 0
  for pid in ids:
    e = meta.get(pid)
    reason, chains = check_entry(e) if e else ('no metadata', {})
    row = {'pdb_id': pid, 'release_date': e['rcsb_accession_info']['initial_release_date'][:10] if e else '',
           'resolution': (e['rcsb_entry_info'].get('resolution_combined') or [None])[0] if e else '',
           'step2_3': reason or 'pass', 'contact_pairs': '', 'contact_note': '', 'step4': ''}
    if chains:
      c1, c2 = sorted(chains.values(), key=lambda c: c['entity'])
      row.update(chain_1=c1['auth_asym_id'], chain_2=c2['auth_asym_id'], length_1=c1['length'], length_2=c2['length'])
    rows.append(row)
    if not reason:
      passed.append((pid, chains))
    elif reason.startswith(('chain lengths', 'total length')):
      reasons_3 += 1
    else:
      reasons_2 += 1
  n2 = len(ids) - reasons_2
  steps.append({'step': '2 data API: re-check of 1, assembly 1 one chain per protein entity', 'pool': n2})
  steps.append({'step': '3 chain lengths 30 to 700, total <= 1000', 'pool': len(passed)})
  print(f'{steps[-2]["step"]}: {n2}\n{steps[-1]["step"]}: {len(passed)}', flush=True)

  pool = []
  by_id = {r['pdb_id']: r for r in rows}
  for k, (pid, chains) in enumerate(passed, 1):
    try:
      n, note = contact_pairs(pid)
    except urllib.error.HTTPError as err:
      n, note = None, f'assembly file: HTTP {err.code}'
    r = by_id[pid]
    r.update(contact_pairs='' if n is None else n, contact_note=note,
             step4='pass' if n is not None and n >= MIN_CONTACT_PAIRS else ('contact not computed' if n is None
                                                                             else f'{n} residue pairs'))
    if r['step4'] == 'pass':
      pool.append((pid, chains))
    if k % 200 == 0:
      print(f'   contacts: {k}/{len(passed)}', flush=True)
  steps.append({'step': '4 contact: >= 10 residue pairs within 5 A (assembly 1)', 'pool': len(pool)})
  print(f'{steps[-1]["step"]}: {len(pool)}', flush=True)
  QUERY_LOG.append({'utc': utc_now(), 'service': 'files (assembly1.cif.gz)', 'entries': len(passed)})

  order = [pool[i] for i in np.random.default_rng(SEED).permutation(len(pool))]
  al = aligner()
  kept, red_rows = [], []
  for rank, (pid, chains) in enumerate(order, 1):
    seqs = [c['sequence'] for c in sorted(chains.values(), key=lambda c: c['entity'])]
    hit = None
    for kid, kseqs in kept:
      for i, s in enumerate(seqs):
        for j, t in enumerate(kseqs):
          ident = identity(al, s, t)
          if ident >= IDENTITY:
            hit = (kid, i + 1, j + 1, ident)
            break
        if hit:
          break
      if hit:
        break
    red_rows.append({'order': rank, 'pdb_id': pid, 'kept': not hit, 'dropped_for': hit[0] if hit else '',
                     'chain': hit[1] if hit else '', 'kept_chain': hit[2] if hit else '',
                     'identity': round(hit[3], 4) if hit else ''})
    if not hit:
      kept.append((pid, seqs))
    if rank % 200 == 0:
      print(f'   redundancy: {rank}/{len(order)}, kept {len(kept)}', flush=True)
  steps.append({'step': '5 redundancy at 30% identity (random order, seed 2026)', 'pool': len(kept)})
  print(f'{steps[-1]["step"]}: {len(kept)}', flush=True)

  kept_ids = sorted(k for k, _ in kept)
  draw = [kept_ids[i] for i in np.random.default_rng(SEED).permutation(len(kept_ids))]
  chains_of = dict(pool)
  draw_rows = []
  for k, pid in enumerate(draw, 1):
    c1, c2 = sorted(chains_of[pid].values(), key=lambda c: c['entity'])
    tokens = c1['length'] + c2['length']
    draw_rows.append({'draw_order': k, 'role': 'target' if k <= N_TARGETS else 'replacement', 'pdb_id': pid,
                      'release_date': by_id[pid]['release_date'], 'resolution': by_id[pid]['resolution'],
                      'entity_1': c1['entity'], 'chain_1': c1['auth_asym_id'], 'asym_1': c1['asym_id'],
                      'length_1': c1['length'], 'entity_2': c2['entity'], 'chain_2': c2['auth_asym_id'],
                      'asym_2': c2['asym_id'], 'length_2': c2['length'], 'tokens': tokens, 'bucket': bucket(tokens),
                      'contact_pairs': by_id[pid]['contact_pairs'], 'sequence_1': c1['sequence'],
                      'sequence_2': c2['sequence']})
  steps.append({'step': f'6 draw: first {N_TARGETS} of a permutation (seed 2026)', 'pool': min(N_TARGETS, len(draw))})

  write_csv(OUT / 'pool_steps.csv', steps)
  write_csv(OUT / 'candidates.csv', rows, ['pdb_id', 'release_date', 'resolution', 'step2_3', 'chain_1', 'chain_2',
                                          'length_1', 'length_2', 'contact_pairs', 'contact_note', 'step4'])
  write_csv(OUT / 'redundancy.csv', red_rows)
  write_csv(OUT / 'draw.csv', draw_rows)
  write_csv(OUT / 'targets.csv', [r for r in draw_rows if r['role'] == 'target'])
  (OUT / 'queries.json').write_text(json.dumps(QUERY_LOG, indent=1) + '\n')
  (OUT / 'sources.json').write_text(json.dumps({
      'built_utc': utc_now(), 'script': 'targets/build_ppi_targets.py', 'search_api': SEARCH, 'data_api': GRAPHQL,
      'files': FILES, 'seed': SEED, 'identity_threshold': IDENTITY,
      'identity_definition': 'identical aligned pairs / length of the shorter chain; global alignment, BLOSUM62, '
                             'gap open -11, extend -1, free end gaps (Bio.Align.PairwiseAligner)',
      'biopython': __import__('Bio').__version__, 'gemmi': gemmi.__version__, 'numpy': np.__version__,
      'search_ids_sha256': hashlib.sha256('\n'.join(ids).encode()).hexdigest(), 'pool_steps': steps}, indent=2) + '\n')
  print('\n'.join(f'{s["step"]}: {s["pool"]}' for s in steps))
  return 0


if __name__ == '__main__':
  sys.exit(main())
