"""DockQ (version 2) of protein-protein predictions: success of follow-up experiment 1 (notes/analysis_plan.md
section 10, row "Follow-up experiment 1, protein-protein complexes" of 2026-10-14).

    analysis/ppi/.venv/bin/python analysis/score_dockq.py --model M.cif --native N.cif --chain_map A:A,B:B
    analysis/ppi/.venv/bin/python analysis/score_dockq.py --collect [--out results/analysis/ppi_<date>]

Environment: analysis/requirements-ppi.txt (DockQ 2.1.3 declares numpy<2, so it is pinned apart from the
analysis environment, whose numpy is 2.5.3). The DockQ version is recorded with every result.

score(): the DockQ of the native interface between the two chains, with DockQ's own sequence alignment of model
to native chains (DockQ.run_on_all_native_interfaces with the given native-to-model chain map). mmCIF files are
read with DockQ's MMCIFParser (author chain IDs), PDB files with DockQ's load_PDB. Success: DockQ >= 0.23
(primary), >= 0.49 (secondary). A model that cannot be scored is not a success; the reason is recorded.

--collect: every sample model (seed-S_sample-K/*_model.cif) of the sessions in results/af3/ with label ppi,
against the deposited first biological assembly of its target (data/ppi/assembly1/<ID>-assembly1.cif.gz, the
coordinates the target rule used), chain map: the entry's chain_1 -> model chain A, chain_2 -> B
(targets/ppi/targets.csv). Results cached by model SHA-256 in <out>/dockq_cache.jsonl; table <out>/dockq.csv.
"""
import argparse
import gzip
import hashlib
import importlib.metadata
import json
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent
SUCCESS, SECONDARY = 0.23, 0.49
KEYS = ('DockQ', 'iRMSD', 'LRMSD', 'fnat', 'fnonnat', 'F1', 'clashes')


def dockq_version():
  return importlib.metadata.version('DockQ')


def load(path, chains):
  from DockQ.DockQ import load_PDB
  from DockQ.parsers import MMCIFParser
  path = str(path)
  if path.endswith(('.cif', '.cif.gz')):
    model = MMCIFParser(QUIET=True).get_structure('-', (gzip.open if path.endswith('.gz') else open)(path, 'rt'),
                                                  chains=chains, parse_hetatms=False, auth_chains=True, model_number=0)
    model.id = path
    return model
  return load_PDB(path, chains=chains)


def score(model_path, native_path, chain_map):
  """DockQ of the native interface; chain_map maps native chain -> model chain. Never raises."""
  out = {'model': str(model_path), 'native': str(native_path), 'chain_map': chain_map, **{k: None for k in KEYS},
         'success': False, 'success_049': False, 'error': None, 'dockq_version': dockq_version()}
  try:
    from DockQ.DockQ import run_on_all_native_interfaces
    model = load(model_path, list(chain_map.values()))
    native = load(native_path, list(chain_map.keys()))
    for c in chain_map:
      if c not in [ch.id for ch in native]:
        raise ValueError(f'native chain {c} not found')
    for c in chain_map.values():
      if c not in [ch.id for ch in model]:
        raise ValueError(f'model chain {c} not found')
    results, _ = run_on_all_native_interfaces(model, native, chain_map=chain_map)
    if len(results) != 1:
      raise ValueError(f'{len(results)} native interfaces, expected 1')
    r = next(iter(results.values()))
    out.update({k: (float(r[k]) if k != 'clashes' else int(r[k])) for k in KEYS})
    out['success'] = out['DockQ'] >= SUCCESS
    out['success_049'] = out['DockQ'] >= SECONDARY
  except Exception as e:  # noqa: BLE001 - recorded, never fatal for a batch
    out['error'] = f'{type(e).__name__}: {e}'[:300]
  return out


def sha256(path):
  return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def collect(out):
  import csv
  targets = {r['pdb_id']: r for r in csv.DictReader(open(REPO / 'targets' / 'ppi' / 'targets.csv'))}
  cache_path = out / 'dockq_cache.jsonl'
  cache = {}
  if cache_path.exists():
    for line in cache_path.read_text().splitlines():
      c = json.loads(line)
      cache[(c['model_sha256'], c['target'])] = c
  rows = []
  for sj in sorted((REPO / 'results' / 'af3').glob('*/*/session.json')):
    try:
      if json.loads(sj.read_text()).get('label') != 'ppi':
        continue
    except ValueError:
      continue
    for cif in sorted(sj.parent.glob('*/af3_output/*/seed-*_sample-*/*_model.cif')):
      target = cif.parts[-4]
      t = targets.get(target)
      if t is None:
        continue
      key = (sha256(cif), target)
      if key not in cache:
        native = REPO / 'data' / 'ppi' / 'assembly1' / f'{target}-assembly1.cif.gz'
        res = score(cif, native, {t['chain_1']: 'A', t['chain_2']: 'B'})
        res.update(model_sha256=key[0], target=target, model=str(cif.relative_to(REPO)),
                   native=str(native.relative_to(REPO)))
        cache[key] = res
        with open(cache_path, 'a') as f:
          f.write(json.dumps(res) + '\n')
      seed, sample = cif.parent.name.replace('seed-', '').split('_sample-')
      rows.append({'session': sj.parent.parent.name, 'run': sj.parent.name, 'target': target, 'seed': int(seed),
                   'sample': int(sample), **{k: cache[key][k] for k in (*KEYS, 'success', 'success_049', 'error',
                                                                         'dockq_version', 'model', 'model_sha256')}})
  with open(out / 'dockq.csv', 'w', newline='') as f:
    if rows:
      w = csv.DictWriter(f, list(rows[0]), lineterminator='\n')
      w.writeheader()
      w.writerows(rows)
  print(f'{len(rows)} models scored with DockQ {dockq_version()}; {out / "dockq.csv"}')


def main():
  ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
  ap.add_argument('--model')
  ap.add_argument('--native')
  ap.add_argument('--chain_map', default='A:A,B:B', help='native:model pairs, for example A:B,B:A')
  ap.add_argument('--collect', action='store_true')
  ap.add_argument('--out', default=str(REPO / 'results' / 'analysis' / 'ppi'))
  args = ap.parse_args()
  if args.collect:
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    collect(out)
    return 0
  if not (args.model and args.native):
    ap.error('give --model and --native, or --collect')
  chain_map = dict(p.split(':') for p in args.chain_map.split(','))
  print(json.dumps(score(args.model, args.native, chain_map), indent=2))
  return 0


if __name__ == '__main__':
  sys.exit(main())
