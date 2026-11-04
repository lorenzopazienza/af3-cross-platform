"""Feasibility screen of a new follow-up 10 (label large) on Runs N' Poses: the main study's selection rule applied to
its systems, with exact token counts from AlphaFold3's own featurisation. No study input is made; nothing is drawn.

Run inside the pinned AlphaFold3 environment, from the repository root's third_party/alphafold3:

    cd third_party/alphafold3
    .venv/bin/python ../../targets/large/rnp_screen.py

Sources: data/runs_n_poses/annotations.csv (the Runs N' Poses annotations, its SHA-256 recorded in the output's
header) and, per entry, the RCSB Data API (GraphQL; the polymer entities of the entry with their type, canonical
one-letter sequence and label and author chain IDs, and the entry's initial release date), cached once in
data/runs_n_poses_screen/rcsb_entries.json (a cached entry is never asked for again). Token counts are cached in
data/runs_n_poses_screen/token_counts.json (keyed like the main builder's: the SHA-256 of the input JSON).

Rule, the main study's (targets/build_posebusters_subset.py) read on Runs N' Poses systems:
  1. one protein chain in the system (system_id <entry>__<biounit>__<protein chains>__<ligand chains>) and exactly one
     proper ligand chain among its ligand chains (ligand_is_proper; ions and other non-proper ligands of the system
     are not part of the input, as the main study leaves out cofactors);
  2. no nucleic acid in the entry (no polymer entity of a DNA, RNA or hybrid type);
  3. the protein sequence is the canonical one (entity_poly.pdbx_seq_one_letter_code_can) of the entity whose label
     chain IDs (else author chain IDs) contain the system's chain; the entity must be polypeptide(L) and the sequence
     the 20 standard amino acids only;
  4. the ligand CCD code in AlphaFold3's bundled CCD;
  5. AlphaFold3 featurises the input (that chain plus one copy of the ligand by CCD code, empty MSA, no templates);
     its token count and bucket are recorded. Systems with more than 1,536 residues are not featurised (they have
     more than 1,536 tokens).
Also recorded: whether the entry is among the main study's 60 or follow-up 11's 72 targets (by PDB ID).
Output: targets/large/rnp_screen.csv (one row per system of step 1, status eligible or excluded with the reason).
"""
import collections
import csv
import hashlib
import json
import pathlib
import sys
import time
import urllib.request

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'inputs'))
import af3_featurise  # noqa: E402
import http_util  # noqa: E402

ANN = REPO / 'data' / 'runs_n_poses' / 'annotations.csv'
CACHE = REPO / 'data' / 'runs_n_poses_screen'
OUT = REPO / 'targets' / 'large' / 'rnp_screen.csv'
GRAPHQL = 'https://data.rcsb.org/graphql'
QUERY = """query($ids:[String!]!){entries(entry_ids:$ids){rcsb_id rcsb_accession_info{initial_release_date}
polymer_entities{entity_poly{type pdbx_seq_one_letter_code_can}
rcsb_polymer_entity_container_identifiers{asym_ids auth_asym_ids}}}}"""
NA_TYPES = ('polydeoxyribonucleotide', 'polyribonucleotide', 'polydeoxyribonucleotide/polyribonucleotide hybrid',
            'peptide nucleic acid')
STANDARD = set('ACDEFGHIKLMNPQRSTVWY')
MAX_RESIDUES = 1536
COLS = ['system_id', 'entry_pdb_id', 'ligand_ccd', 'protein_chain', 'status', 'reason', 'release_date', 'sequence_length',
        'num_tokens', 'bucket', 'in_main_60', 'in_repl_72', 'pocket_cluster', 'sequence_sha256', 'sequence']


def sha256(b):
  return hashlib.sha256(b).hexdigest()


def candidates():
  """Systems with one protein chain and exactly one proper ligand chain (step 1)."""
  by = collections.defaultdict(list)
  for r in csv.DictReader(open(ANN)):
    by[r['system_id']].append(r)
  out = []
  for sid, rs in sorted(by.items()):
    _, _, prot, _ = sid.split('__')
    proper = [r for r in rs if r['ligand_is_proper'] == 'True']
    if len(prot.split('_')) == 1 and len(proper) == 1:
      out.append((sid, proper[0]))
  return out


def entries(ids):
  CACHE.mkdir(parents=True, exist_ok=True)
  path = CACHE / 'rcsb_entries.json'
  cache = json.loads(path.read_text()) if path.exists() else {}
  todo = sorted(set(ids) - set(cache))
  for i in range(0, len(todo), 100):
    body = json.dumps({'query': QUERY, 'variables': {'ids': todo[i:i + 100]}}).encode()
    req = urllib.request.Request(GRAPHQL, body, {'Content-Type': 'application/json', 'User-Agent': http_util.USER_AGENT})
    for attempt in range(5):
      try:
        data = json.load(urllib.request.urlopen(req, timeout=60, context=http_util.SSL_CONTEXT))['data']['entries']
        break
      except Exception as e:  # network or server error: retry, then stop
        print('  retry', i, e, flush=True)
        time.sleep(10 * (attempt + 1))
    else:
      raise SystemExit('RCSB did not answer; rerun to resume from the cache')
    for e in data or []:
      cache[e['rcsb_id']] = e
    path.write_text(json.dumps(cache, indent=0, sort_keys=True))
    print(f'  RCSB: {min(i + 100, len(todo))}/{len(todo)} entries', flush=True)
    time.sleep(0.5)
  return cache


def main():
  cands = candidates()
  ann_sha = sha256(ANN.read_bytes())
  rcsb = entries([s.split('__')[0].upper() for s, _ in cands])
  main60 = {r['pdb_id'] for r in csv.DictReader(open(REPO / 'targets' / 'posebusters_subset.csv'))}
  repl72 = {r['pdb_id'] for r in csv.DictReader(open(REPO / 'targets' / 'repl' / 'repl_subset.csv'))}
  ccd = af3_featurise.ccd()
  tpath = CACHE / 'token_counts.json'
  tokens = json.loads(tpath.read_text()) if tpath.exists() else {}
  rows = []
  for n, (sid, ann) in enumerate(cands, 1):
    entry = sid.split('__')[0].upper()
    chain = sid.split('__')[2].split('.')[-1]
    lig = ann['ligand_ccd_code']
    row = {'system_id': sid, 'entry_pdb_id': entry, 'ligand_ccd': lig, 'protein_chain': chain, 'status': 'excluded',
           'in_main_60': 'yes' if entry in main60 else 'no', 'in_repl_72': 'yes' if entry in repl72 else 'no',
           'pocket_cluster': ann['cluster']}
    rows.append(row)
    e = rcsb.get(entry)
    if not e:
      row['reason'] = 'entry not found at RCSB'
      continue
    row['release_date'] = ((e.get('rcsb_accession_info') or {}).get('initial_release_date') or '')[:10]
    ents = e.get('polymer_entities') or []
    if any(x['entity_poly']['type'] in NA_TYPES for x in ents):
      row['reason'] = 'nucleic acid in the entry'
      continue
    ids = lambda x, k: (x['rcsb_polymer_entity_container_identifiers'].get(k) or [])  # noqa: E731
    hit = [x for x in ents if chain in ids(x, 'asym_ids')] or [x for x in ents if chain in ids(x, 'auth_asym_ids')]
    if len(hit) != 1:
      row['reason'] = f'chain {chain} matches {len(hit)} polymer entities'
      continue
    kind, seq = hit[0]['entity_poly']['type'], ''.join((hit[0]['entity_poly']['pdbx_seq_one_letter_code_can'] or '').split())
    if kind != 'polypeptide(L)':
      row['reason'] = f'entity type {kind}'
      continue
    if not seq or set(seq) - STANDARD:
      row['reason'] = 'non-standard letters in sequence: ' + ''.join(sorted(set(seq) - STANDARD))
      continue
    row.update(sequence=seq, sequence_length=len(seq), sequence_sha256=sha256(seq.encode()))
    if lig not in ccd:
      row['reason'] = 'ligand not in AlphaFold3 CCD'
      continue
    if len(seq) > MAX_RESIDUES:
      row['reason'] = f'more than {MAX_RESIDUES} residues ({len(seq)})'
      continue
    text = af3_featurise.protein_ligand_json(sid, seq, lig, seeds=[1])
    key = sha256(json.dumps({'sequence': seq, 'ccd': lig}).encode())   # the name does not change the count
    if key not in tokens:
      t0 = time.time()
      try:
        tokens[key] = af3_featurise.featurise(text)
      except Exception as ex:  # AlphaFold3 rejects the input
        tokens[key] = {'error': f'{type(ex).__name__}: {ex}'[:200]}
      tpath.write_text(json.dumps(tokens, indent=0, sort_keys=True))
      print(f'  [{n}/{len(cands)}] {sid}: {tokens[key]} ({time.time() - t0:.1f} s)', flush=True)
    res = tokens[key]
    if 'error' in res:
      row['reason'] = 'AlphaFold3 featurisation failed: ' + res['error']
      continue
    row.update(num_tokens=res['num_tokens'], bucket=res['bucket'], status='eligible')
  OUT.parent.mkdir(parents=True, exist_ok=True)
  with open(OUT, 'w', newline='') as f:
    f.write(f'# targets/large/rnp_screen.py: data/runs_n_poses/annotations.csv sha256 {ann_sha}; RCSB GraphQL cache '
            f'data/runs_n_poses_screen/rcsb_entries.json; AlphaFold3 buckets {list(af3_featurise.BUCKETS)}\n')
    w = csv.DictWriter(f, COLS, extrasaction='ignore', lineterminator='\n')
    w.writeheader()
    w.writerows(rows)
  print(collections.Counter(r['status'] for r in rows))
  print(collections.Counter(r['reason'].split(':')[0].split(' (')[0] for r in rows if r['status'] == 'excluded'))
  print(collections.Counter(r['bucket'] for r in rows if r['status'] == 'eligible'))


if __name__ == '__main__':
  main()
