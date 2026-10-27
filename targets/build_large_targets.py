"""Selects the larger-input targets of follow-up 10 (drafts/section10_fu7-10.md; label large): the source and the
eligibility of the main 60 (targets/build_posebusters_subset.py) with a token window above the main study's 1,024.

Run inside the pinned AlphaFold3 environment, as the main builder (token counts from AlphaFold3's own featurisation):

    cd third_party/alphafold3
    .venv/bin/python ../../targets/build_large_targets.py --rule main|chains [--cache DIR] [--out DIR]

Rules (--rule):
  main     the main builder's steps 1 to 5 unchanged (PoseBusters V2, exactly one protein chain and no nucleic acid,
           the RCSB canonical sequence, the ligand in AlphaFold3's CCD, featurisation of that chain plus one copy of
           the ligand), with the token window MIN_TOKENS to MAX_TOKENS in place of "at most 1,024". The literal
           reading of the follow-up; it reads the main study's screen (targets/posebusters_v2_screen.csv), whose
           token counts come from the same featurisation, and needs no download.
  chains   step 2 relaxed: every protein chain of the PoseBusters protein file (still no nucleic acid), each mapped
           to its RCSB entity's canonical sequence as in step 3 (all must map, polypeptide(L), 20 standard letters);
           the input is every such chain (ids A, B, ...) plus one copy of the ligand, empty MSA, no templates. Steps
           4 and 5 as the main builder, with the token window. A complex whose canonical sequences alone exceed
           MAX_TOKENS is excluded without featurisation (every residue is at least one token).
Then, for either rule: bins by AlphaFold3 bucket (1,280, 1,536, 2,048); N_TARGETS allocated over the bins as the main
builder's allocate() does (all eligible if fewer); within each bin, in bin order, drawn without replacement with
numpy.random.default_rng(SAMPLE_SEED), as the main builder. Fewer than MIN_TARGETS eligible: the rule is reported as
infeasible (exit 1) and only the screen is written.

Downloads (rule chains only): one mmCIF per candidate from the RCSB PDB, cached in --cache/rcsb/ (default data/rcsb/,
shared with the main builder); token counts cached in --cache/large_token_counts.json. Outputs in --out (default
targets/): large_screen_<rule>.csv (every V2 identifier with its reason), and if feasible large_subset_<rule>.csv and
large_sources_<rule>.json. The frozen inputs are made afterwards, as the main study's (MSAs, then build_inputs), only
after the specification row is committed.
"""
import argparse
import collections
import csv
import io
import json
import pathlib
import sys
import time
import zipfile

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import build_posebusters_subset as main_rule  # noqa: E402
from build_posebusters_subset import af3_featurise  # noqa: E402

REPO = af3_featurise.REPO_ROOT
MIN_TOKENS, MAX_TOKENS = 1100, 2000
BINS = (('1280', 1025, 1280), ('1536', 1281, 1536), ('2048', 1537, 2048))
N_TARGETS, MIN_TARGETS = 20, 15
SAMPLE_SEED = main_rule.SAMPLE_SEED
CHAIN_IDS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'


def bin_of(tokens):
  return next(name for name, lo, hi in BINS if lo <= tokens <= hi)


def allocate(counts, total):
  """The main builder's allocate() over this rule's bins: fewest eligible first, each min(eligible, remaining // bins left)."""
  order = sorted(counts, key=lambda b: (counts[b], [n for n, _, _ in BINS].index(b)))
  alloc, remaining = {}, total
  for i, b in enumerate(order):
    alloc[b] = min(counts[b], remaining // (len(order) - i))
    remaining -= alloc[b]
  return alloc


def chains_ligand_json(name, sequences, ccd_code, seeds):
  """As af3_featurise.protein_ligand_json, with one protein entry per chain."""
  return json.dumps({
      'name': name, 'modelSeeds': list(seeds),
      'sequences': [{'protein': {'id': CHAIN_IDS[i], 'sequence': s, 'unpairedMsa': '', 'pairedMsa': '', 'templates': []}}
                    for i, s in enumerate(sequences)] + [{'ligand': {'id': CHAIN_IDS[len(sequences)], 'ccdCodes': [ccd_code]}}],
      'dialect': 'alphafold3', 'version': 1}, indent=2)


def screen_main():
  rows = list(csv.DictReader(open(REPO / 'targets' / 'posebusters_v2_screen.csv')))
  out = []
  for r in rows:
    row = {k: r[k] for k in ('posebusters_id', 'pdb_id', 'ligand_ccd', 'n_protein_chains', 'num_tokens')}
    t = int(r['num_tokens']) if r['num_tokens'] else None
    if t is None:
      row.update(status='excluded', reason=r['reason'])
    elif not MIN_TOKENS <= t <= MAX_TOKENS:
      row.update(status='excluded', reason=f'{t} tokens outside {MIN_TOKENS} to {MAX_TOKENS}')
    else:
      row.update(status='eligible', reason='', bin=bin_of(t))
    out.append(row)
  return out


def screen_chains(cache):
  pb_dir = REPO / 'data' / 'posebusters'
  zf = zipfile.ZipFile(io.BytesIO((pb_dir / main_rule.ZIP_NAME).read_bytes()))
  v1_ids = {l.strip() for l in zf.read('posebusters_benchmark_set_ids.txt').decode().splitlines() if l.strip()}
  v2_ids = sorted(l.strip() for l in (pb_dir / 'posebusters_pdb_ccd_ids.txt').read_text().splitlines() if l.strip())
  ccd = af3_featurise.ccd()
  cache_path = cache / 'large_token_counts.json'
  token_cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
  out = []
  for n, pb_id in enumerate(v2_ids, 1):
    pdb, lig = pb_id.split('_')
    row = {'posebusters_id': pb_id, 'pdb_id': pdb, 'ligand_ccd': lig, 'status': 'excluded', 'reason': '',
           'n_protein_chains': '', 'num_tokens': ''}
    out.append(row)
    if pb_id not in v1_ids:
      row['reason'] = 'not in Zenodo set'
      continue
    pdb_bytes = zf.read(f'posebusters_benchmark_set/{pb_id}/{pb_id}_protein.pdb')
    chains, has_na = main_rule.protein_chains(pdb_bytes.decode())
    row['n_protein_chains'] = len(chains)
    if has_na or not chains:
      row['reason'] = 'nucleic acid present' if has_na else 'no protein chain'
      continue
    path = cache / 'rcsb' / f'{pdb}.cif'
    fresh = not path.exists()
    cif = main_rule.cached(path, main_rule.RCSB_URL.format(pdb=pdb)).decode()
    if fresh:
      time.sleep(0.2)
    seqs, why = [], ''
    for c in chains:
      kind, seq, info = main_rule.rcsb_entity(cif, c)
      if kind is None or kind != 'polypeptide(L)' or set(seq) - main_rule.STANDARD_LETTERS:
        why = (f'chain {c} not found in RCSB entity_poly' if kind is None else f'chain {c}: entity type {kind}'
               if kind != 'polypeptide(L)' else f'chain {c}: non-standard letters in sequence')
        break
      seqs.append(seq)
    row['rcsb_initial_release'] = info['initial_release']
    if why:
      row['reason'] = why
      continue
    if lig not in ccd:
      row['reason'] = 'ligand not in AlphaFold3 CCD'
      continue
    row['sequence_lengths'] = '+'.join(str(len(s)) for s in seqs)
    if sum(map(len, seqs)) > MAX_TOKENS:
      row['reason'] = f'{sum(map(len, seqs))} residues > {MAX_TOKENS} (not featurised)'
      continue
    text = chains_ligand_json(pb_id, seqs, lig, seeds=[1])
    key = main_rule.sha256(text.encode())
    if key not in token_cache:
      try:
        token_cache[key] = af3_featurise.featurise(text)
      except Exception as e:   # AlphaFold3 rejects the input
        token_cache[key] = {'error': f'{type(e).__name__}: {e}'[:200]}
      cache_path.write_text(json.dumps(token_cache, indent=1))
      print(f'  [{n}/{len(v2_ids)}] {pb_id}: {token_cache[key]}', flush=True)
    res = token_cache[key]
    if 'error' in res:
      row['reason'] = 'AlphaFold3 featurisation failed: ' + res['error']
      continue
    t = res['num_tokens']
    row['num_tokens'] = t
    if not MIN_TOKENS <= t <= MAX_TOKENS:
      row['reason'] = f'{t} tokens outside {MIN_TOKENS} to {MAX_TOKENS}'
      continue
    row.update(status='eligible', bin=bin_of(t))
  return out


def main():
  ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
  ap.add_argument('--rule', choices=('main', 'chains'), required=True)
  ap.add_argument('--cache', default=str(REPO / 'data'))
  ap.add_argument('--out', default=str(REPO / 'targets'))
  args = ap.parse_args()
  out = pathlib.Path(args.out)
  screen = screen_main() if args.rule == 'main' else screen_chains(pathlib.Path(args.cache))
  cols = ['posebusters_id', 'pdb_id', 'ligand_ccd', 'status', 'reason', 'n_protein_chains', 'sequence_lengths',
          'num_tokens', 'bin', 'rcsb_initial_release']
  with open(out / f'large_screen_{args.rule}.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, cols, extrasaction='ignore', lineterminator='\n')
    w.writeheader()
    w.writerows(screen)
  eligible = [r for r in screen if r['status'] == 'eligible']   # sorted by identifier
  by_bin = collections.OrderedDict((b, [r for r in eligible if r['bin'] == b]) for b, _, _ in BINS)
  counts = {b: len(v) for b, v in by_bin.items()}
  print(f'rule {args.rule}: {len(eligible)} eligible in {MIN_TOKENS} to {MAX_TOKENS} tokens, by bucket {counts}')
  if len(eligible) < MIN_TARGETS:
    print(f'INFEASIBLE: fewer than {MIN_TARGETS} eligible targets; nothing selected')
    return 1
  alloc = allocate(counts, min(N_TARGETS, len(eligible)))
  rng = np.random.default_rng(SAMPLE_SEED)
  chosen = []
  for b, rows in by_bin.items():
    chosen += [rows[i] for i in (sorted(rng.choice(len(rows), size=alloc[b], replace=False)) if alloc[b] else [])]
  chosen.sort(key=lambda r: r['posebusters_id'])
  with open(out / f'large_subset_{args.rule}.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, cols, extrasaction='ignore', lineterminator='\n')
    w.writeheader()
    w.writerows(chosen)
  (out / f'large_sources_{args.rule}.json').write_text(json.dumps({
      'rule': args.rule, 'token_window': [MIN_TOKENS, MAX_TOKENS], 'bins': [list(b) for b in BINS],
      'n_targets': N_TARGETS, 'min_targets': MIN_TARGETS, 'sample_seed': SAMPLE_SEED, 'allocation': alloc,
      'eligible_by_bin': counts,
      'exclusion_reasons': dict(collections.Counter(
          'outside the token window' if 'tokens outside' in r['reason'] else
          f'more than {MAX_TOKENS} residues' if 'residues >' in r['reason'] else r['reason'].split(':')[0]
          for r in screen if r['status'] == 'excluded'))}, indent=2) + '\n')
  print('selected:', ', '.join(f'{r["posebusters_id"]} ({r["num_tokens"]})' for r in chosen))
  return 0


if __name__ == '__main__':
  sys.exit(main())
