"""Frozen AlphaFold3 inputs of follow-up experiment 1 (protein-protein complexes; notes/analysis_plan.md
section 10, row of 2026-10-14), validated by AlphaFold3.

Run inside the pinned AlphaFold3 environment, after targets/build_ppi_targets.py:

    cd third_party/alphafold3
    .venv/bin/python ../../inputs/ppi/build_ppi_inputs.py

Targets: targets/ppi/draw.csv in draw order; the first 30 whose input can be built. A target is replaced by the
next entry of the draw, before any run, only for a permanent cause: the ColabFold server answers ERROR when the
sequence is submitted (its answer for an input it rejects), or AlphaFold3's own parser or featurisation rejects
the input. Every replaced entry is logged in inputs/ppi/replacements.csv with its cause. Anything transient
(server refusals and backoffs, HTTP errors, timeouts, an outage, a ticket that does not complete, a result
without the expected files) stops the script without replacing anything: rerun it later and it resumes from the
cache, until every drawn target's MSAs are built (section 10 row of 2026-10-14, as clarified by Lorenzo).

Per target, data/inputs/ppi/<PDB ID>.json (dialect alphafold3, version 1, modelSeeds 1 to 5): the two protein
chains A and B (canonical entity sequences from the draw; no ligands, ions or waters), each with
  unpairedMsa  the main study's unpaired MSA for that sequence: ColabFold server (https://api.colabfold.com),
               /ticket/msa, mode "env", uniref.a3m plus the environmental A3M without its repeated query
               (inputs/build_msas.py: same client, cache data/msa/<sha256 of sequence>.a3m, delays, backoff);
  pairedMsa    the chain's block of the server's pairing result: /ticket/pair, mode "pairgreedy" (ColabFold's
               default pairing for its default MSA mode, mmseqs2_uniref_env: colabfold/batch.py and
               colabfold.py run_mmseqs2 at commit 2934b1b6), queries >101 (chain A) and >102 (chain B);
               pair.a3m holds one block per query separated by NUL bytes, row i of each block paired with row i
               of the other, parsed as ColabFold parses it; cached in data/msa/pair/<sha256 of A|B>.*;
  templates    none.
Each written file is loaded with the loader run_alphafold.py uses for --json_path and featurised with
AlphaFold3's own pipeline (first seed, no inference). Frozen inputs are never overwritten with different content.
Writes inputs/ppi/manifest.csv (the columns of inputs/manifest.csv, plus chain and paired-MSA columns),
inputs/ppi/manifest.json (build environment) and inputs/ppi/replacements.csv.
"""
import csv
import datetime
import hashlib
import importlib.metadata
import io
import json
import pathlib
import platform
import subprocess
import sys
import tarfile
import time

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import af3_featurise  # noqa: E402
import build_msas  # noqa: E402
import http_util  # noqa: E402

REPO = af3_featurise.REPO_ROOT
DRAW = REPO / 'targets' / 'ppi' / 'draw.csv'
MSA_DIR = REPO / 'data' / 'msa'
PAIR_DIR = MSA_DIR / 'pair'
INPUT_DIR = REPO / 'data' / 'inputs' / 'ppi'
SEEDS = (1, 2, 3, 4, 5)
N_TARGETS = 30
PAIR_MODE = 'pairgreedy'
COLABFOLD_COMMIT = '2934b1b61ea1f6646c40aad47086e8f28863a2db'


class BuildError(Exception):
  """A permanent cause: the input of the target cannot be built, and the target is replaced."""


def stop(target_reason):
  """A transient failure: stop without replacing anything; a rerun resumes from the cache."""
  raise SystemExit(f'!! stopped (transient, nothing replaced): {target_reason}. Rerun later; completed MSAs and '
                   'inputs are cached.')


def ticket(endpoint, q, mode, server):
  """(tar.gz bytes, provenance) of one ColabFold ticket, as inputs/build_msas.py queries the server. A submission
  the server answers with ERROR is permanent (BuildError); every other failure stops (transient)."""
  prov, log = {'submit_utc': utc_now(), 'polls': 0}, []
  while True:
    headers, body = build_msas.call(f'{build_msas.HOST}/{endpoint}', server.backoff, data={'q': q, 'mode': mode})
    out = json.loads(body)
    log.append({'utc': utc_now(), 'call': 'submit', 'response': out})
    if out.get('status') in ('UNKNOWN', 'RATELIMIT', 'MAINTENANCE'):
      server.backoff.wait(f'submit status {out.get("status")}')
      continue
    if out.get('status') == 'ERROR':
      raise BuildError(f'the server rejected the submission ({endpoint}, mode {mode}): {out}')
    break
  prov.update(ticket=out.get('id'), submit_response=out, submit_headers={
      k: v for k, v in headers.items() if k.lower() in ('date', 'server')})
  while out.get('status') in ('UNKNOWN', 'RUNNING', 'PENDING'):
    time.sleep(build_msas.POLL_S)
    _, body = build_msas.call(f'{build_msas.HOST}/ticket/{prov["ticket"]}', server.backoff)
    out = json.loads(body)
    prov['polls'] += 1
  log.append({'utc': utc_now(), 'call': 'final status', 'response': out})
  if out.get('status') != 'COMPLETE':
    stop(f'ticket {prov["ticket"]} ({endpoint}) ended with {out}')
  _, tar_bytes = build_msas.call(f'{build_msas.HOST}/result/download/{prov["ticket"]}', server.backoff)
  server.backoff.reset()
  prov.update(complete_utc=utc_now(), final_response=out, call_log=log)
  return tar_bytes, prov


def sha256(data: bytes) -> str:
  return hashlib.sha256(data).hexdigest()


def utc_now():
  return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds')


class Server:
  """Serial access to the ColabFold server with the main study's delay between queries."""

  def __init__(self):
    self.backoff, self.last = build_msas.Backoff(), None

  def pace(self):
    if self.last is not None:
      time.sleep(max(0, build_msas.DELAY_BETWEEN_QUERIES_S - (time.time() - self.last)))
    self.last = time.time()


def unpaired(seq, server, commit):
  """The cached unpaired MSA of seq (data/msa/), queried as inputs/build_msas.py does if missing."""
  key = sha256(seq.encode())
  a3m_path, side_path = MSA_DIR / f'{key}.a3m', MSA_DIR / f'{key}.json'
  if not (a3m_path.exists() and side_path.exists()):
    server.pace()
    t0 = time.time()
    tar_bytes, prov = ticket('ticket/msa', f'>101\n{seq}\n', build_msas.MODE, server)
    log = prov.pop('call_log')
    try:
      combined, members = build_msas.combine(tar_bytes, seq)
    except SystemExit as e:
      stop(f'unpaired MSA of a {len(seq)}-residue sequence: {e}')
    MSA_DIR.mkdir(parents=True, exist_ok=True)
    (MSA_DIR / f'{key}.tar.gz').write_bytes(tar_bytes)
    a3m_path.write_text(combined)
    side_path.write_text(json.dumps({
        'sequence_sha256': key, 'sequence_length': len(seq), 'sequence': seq, 'server': build_msas.HOST,
        'endpoint': '/ticket/msa', 'mode': build_msas.MODE, 'user_agent': http_util.USER_AGENT, **prov,
        'query_seconds': round(time.time() - t0, 1), 'tar_sha256': sha256(tar_bytes), 'tar_bytes': len(tar_bytes),
        'members': members, 'a3m_sha256': sha256(combined.encode()), 'a3m_rows': combined.count('\n>') + 1,
        'combine_rule': f'{build_msas.UNIREF_A3M} + {build_msas.ENV_A3M} without its query record',
        'built_by': 'inputs/ppi/build_ppi_inputs.py (inputs/build_msas.py client)', 'repo_commit': commit,
        'call_log': log}, indent=2) + '\n')
  side = json.loads(side_path.read_text())
  a3m = a3m_path.read_text()
  if sha256(a3m.encode()) != side['a3m_sha256']:
    raise SystemExit(f'{a3m_path}: SHA-256 does not match its sidecar')
  return a3m, side


def parse_pair_a3m(text, n_queries=2, first=101):
  """{query number: A3M block} of a ColabFold pair.a3m (colabfold.py run_mmseqs2's parser)."""
  blocks, m, update = {}, None, True
  for line in io.StringIO(text):
    if '\x00' in line:
      line = line.replace('\x00', '')
      update = True
    if line.startswith('>') and update:
      m = int(line[1:].rstrip())
      update = False
      blocks.setdefault(m, [])
    if m is not None and line:
      blocks[m].append(line)
  want = list(range(first, first + n_queries))
  if sorted(blocks) != want:
    stop(f'paired MSA: query blocks {sorted(blocks)}, expected {want}')
  return {k: ''.join(v) for k, v in blocks.items()}


def a3m_records(text):
  """[(header, sequence)] of an A3M text."""
  out, header, seq = [], None, []
  for line in text.splitlines():
    if line.startswith('>'):
      if header is not None:
        out.append((header, ''.join(seq)))
      header, seq = line[1:], []
    elif line.strip():
      seq.append(line.strip())
  if header is not None:
    out.append((header, ''.join(seq)))
  return out


def paired(seqs, server, commit):
  """[A3M block of chain A, of chain B] and the sidecar, from the cache or the server."""
  key = sha256('|'.join(seqs).encode())
  side_path = PAIR_DIR / f'{key}.json'
  if not side_path.exists():
    server.pace()
    t0 = time.time()
    q = ''.join(f'>{101 + i}\n{s}\n' for i, s in enumerate(seqs))
    tar_bytes, prov = ticket('ticket/pair', q, PAIR_MODE, server)
    with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode='r:gz') as tar:
      names = [m.name for m in tar.getmembers() if m.isfile()]
      if 'pair.a3m' not in names:
        stop(f'paired MSA: no pair.a3m in the result ({names})')
      text = tar.extractfile('pair.a3m').read().decode()
    blocks = parse_pair_a3m(text, len(seqs))
    PAIR_DIR.mkdir(parents=True, exist_ok=True)
    (PAIR_DIR / f'{key}.tar.gz').write_bytes(tar_bytes)
    side = {'sequences_sha256': key, 'sequences': list(seqs), 'server': build_msas.HOST, 'endpoint': '/ticket/pair',
            'mode': PAIR_MODE, 'colabfold_reference_commit': COLABFOLD_COMMIT, 'user_agent': http_util.USER_AGENT,
            **prov, 'query_seconds': round(time.time() - t0, 1), 'tar_sha256': sha256(tar_bytes),
            'tar_bytes': len(tar_bytes), 'members': sorted(names), 'blocks': {}, 'built_by': 'inputs/ppi/build_ppi_inputs.py',
            'repo_commit': commit}
    for i, s in enumerate(seqs):
      block = blocks[101 + i]
      path = PAIR_DIR / f'{key}.{"AB"[i]}.a3m'
      path.write_text(block)
      side['blocks']['AB'[i]] = {'a3m_sha256': sha256(block.encode()), 'rows': len(a3m_records(block))}
    side_path.write_text(json.dumps(side, indent=2) + '\n')
  side = json.loads(side_path.read_text())
  out = []
  for i, s in enumerate(seqs):
    block = (PAIR_DIR / f'{key}.{"AB"[i]}.a3m').read_text()
    if sha256(block.encode()) != side['blocks']['AB'[i]]['a3m_sha256']:
      raise SystemExit(f'{key}.{"AB"[i]}.a3m: SHA-256 does not match its sidecar')
    recs = a3m_records(block)
    if not recs or recs[0][1] != s:
      stop(f'paired MSA: chain {"AB"[i]} block does not start with its query sequence')
    out.append(block)
  if len({side['blocks'][c]['rows'] for c in 'AB'}) != 1:
    stop(f'paired MSA: blocks have different row counts {side["blocks"]}')
  return out, side


def input_json(name, seqs, unpaired_a3m, paired_a3m):
  return json.dumps({
      'name': name, 'modelSeeds': list(SEEDS),
      'sequences': [{'protein': {'id': c, 'sequence': s, 'unpairedMsa': u, 'pairedMsa': p, 'templates': []}}
                    for c, s, u, p in zip('AB', seqs, unpaired_a3m, paired_a3m)],
      'dialect': 'alphafold3', 'version': 1}, indent=2) + '\n'


def git_head(path):
  return subprocess.run(['git', '-C', str(path), 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()


def build(row, server, commit):
  seqs = [row['sequence_1'], row['sequence_2']]
  ua = [unpaired(s, server, commit) for s in seqs]
  pa, pside = paired(seqs, server, commit)
  text = input_json(row['pdb_id'], seqs, [a for a, _ in ua], pa)
  path = INPUT_DIR / f'{row["pdb_id"]}.json'
  if path.exists() and path.read_text() != text:
    raise SystemExit(f'{path} exists with different content; frozen inputs are not overwritten.')
  if not path.exists():
    path.write_text(text)
  t0 = time.time()
  try:
    result = af3_featurise.featurise_path(path)
  except Exception as e:  # noqa: BLE001 - AlphaFold3 rejects the input
    path.unlink()
    raise BuildError(f'AlphaFold3 rejected the input: {type(e).__name__}: {e}'[:300])
  return {
      'pdb_id': row['pdb_id'], 'ligand_ccd': '', 'pilot': 'no', 'input_path': str(path.relative_to(REPO)),
      'input_sha256': sha256(text.encode()), 'input_bytes': len(text.encode()),
      'sequence_sha256': ';'.join(s['sequence_sha256'] for _, s in ua),
      'msa_sha256': ';'.join(s['a3m_sha256'] for _, s in ua), 'msa_rows': ';'.join(str(s['a3m_rows']) for _, s in ua),
      'msa_server': build_msas.HOST, 'msa_mode': build_msas.MODE,
      'msa_query_utc': ';'.join(s['submit_utc'] for _, s in ua),
      'paired_msa_sha256': ';'.join(pside['blocks'][c]['a3m_sha256'] for c in 'AB'),
      'paired_rows': pside['blocks']['A']['rows'], 'paired_mode': PAIR_MODE, 'paired_query_utc': pside['submit_utc'],
      'num_tokens': result['num_tokens'], 'bucket': result['bucket'], 'num_seeds': result['num_seeds'],
      'validated': 'af3_featurised', 'featurise_seconds': round(time.time() - t0, 1),
      'draw_order': row['draw_order'], 'release_date': row['release_date'], 'chain_A': row['chain_1'],
      'chain_B': row['chain_2'], 'length_A': row['length_1'], 'length_B': row['length_2']}


def main():
  INPUT_DIR.mkdir(parents=True, exist_ok=True)
  draw = list(csv.DictReader(open(DRAW)))
  commit = git_head(REPO)
  server, rows, replaced = Server(), [], []
  for row in draw:
    if len(rows) == N_TARGETS:
      break
    print(f'[{len(rows) + 1}/{N_TARGETS}] draw {row["draw_order"]} {row["pdb_id"]} '
          f'({row["length_1"]} + {row["length_2"]} residues)', flush=True)
    try:
      rows.append(build(row, server, commit))
      print(f'    {rows[-1]["num_tokens"]} tokens, bucket {rows[-1]["bucket"]}, unpaired rows {rows[-1]["msa_rows"]}, '
            f'paired rows {rows[-1]["paired_rows"]}', flush=True)
    except BuildError as e:   # permanent only; transient failures stop the script (SystemExit) and replace nothing
      replaced.append({'pdb_id': row['pdb_id'], 'draw_order': row['draw_order'], 'reason': str(e), 'utc': utc_now()})
      print(f'    cannot be built, permanent cause ({e}); replaced by the next draw', flush=True)
      rejected = [r for r in replaced[-3:] if r['reason'].startswith('AlphaFold3 rejected')]
      if len(rejected) == 3 and [int(r['draw_order']) for r in rejected] == list(range(int(row['draw_order']) - 2, int(row['draw_order']) + 1)):
        stop('AlphaFold3 rejected three consecutive draws, more likely a problem of this environment than of the '
             'inputs; check before rerunning')
  if len(rows) < N_TARGETS:
    print(f'!! only {len(rows)} inputs could be built from {len(draw)} draws')
  with open(HERE / 'manifest.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, list(rows[0]), lineterminator='\n')
    w.writeheader()
    w.writerows(rows)
  with open(HERE / 'replacements.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, ['pdb_id', 'draw_order', 'reason', 'utc'], lineterminator='\n')
    w.writeheader()
    w.writerows(replaced)
  packages = {}
  for name in ('alphafold3', 'jax', 'jaxlib', 'numpy', 'rdkit'):
    try:
      packages[name] = importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
      packages[name] = None
  (HERE / 'manifest.json').write_text(json.dumps({
      'built_utc': utc_now(), 'built_by': 'inputs/ppi/build_ppi_inputs.py', 'repo_commit': commit,
      'alphafold3_commit': git_head(af3_featurise.AF3_DIR), 'af3_buckets': list(af3_featurise.BUCKETS),
      'validation': 'folding_input.load_fold_inputs_from_path + featurisation.featurise_input (first seed, no inference)',
      'seeds': list(SEEDS), 'python': platform.python_version(), 'packages': packages,
      'draw_csv_sha256': sha256(DRAW.read_bytes()), 'inputs': len(rows), 'replaced': len(replaced),
      'unpaired': f'{build_msas.HOST} /ticket/msa mode {build_msas.MODE}',
      'paired': f'{build_msas.HOST} /ticket/pair mode {PAIR_MODE} (ColabFold {COLABFOLD_COMMIT})'}, indent=2) + '\n')
  print(f'Wrote inputs/ppi/manifest.csv with {len(rows)} validated inputs; {len(replaced)} replaced.')


if __name__ == '__main__':
  main()
