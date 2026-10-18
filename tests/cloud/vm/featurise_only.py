"""Test double for harness/featurise_only.py on the fake VM (AlphaFold3 is fake there): checks the frozen input
against the manifest as the real script does and writes host.json and one features.jsonl line per (target, seed)
with deterministic fake hashes, plus an arrays file with --save_arrays."""
import argparse
import csv
import hashlib
import json
import os
import pathlib

ap = argparse.ArgumentParser()
ap.add_argument('--manifest', required=True)
ap.add_argument('--targets', required=True)
ap.add_argument('--seeds', required=True)
ap.add_argument('--out', required=True)
ap.add_argument('--save_arrays', action='store_true')
a = ap.parse_args()
man = {r['pdb_id']: r for r in csv.DictReader(open(a.manifest))}
out = pathlib.Path(a.out)
out.mkdir(parents=True, exist_ok=True)
(out / 'host.json').write_text(json.dumps({'cpu': 'fake', 'packages': {'numpy': '2.4.1'}, 'weights_seen':
                                           sorted(os.listdir(os.path.expanduser('~/af3_weights_run')))
                                           if os.path.isdir(os.path.expanduser('~/af3_weights_run')) else None}) + '\n')
for t in a.targets.split(','):
  frozen = pathlib.Path(man[t]['input_path'])
  assert hashlib.sha256(frozen.read_bytes()).hexdigest() == man[t]['input_sha256'], t
  d = out / t
  d.mkdir(exist_ok=True)
  for s in a.seeds.split(','):
    arrays = [{'name': 'aatype', 'model_input': True, 'dtype': 'int32', 'shape': [256],
               'sha256': hashlib.sha256(f'{t}{s}'.encode()).hexdigest()}]
    with open(d / 'features.jsonl', 'a') as f:
      f.write(json.dumps({'target': t, 'seed': int(s), 'arrays': arrays, 'error': None}) + '\n')
    if a.save_arrays:
      (d / 'arrays').mkdir(exist_ok=True)
      (d / 'arrays' / f'seed-{s}.npz').write_bytes(b'fake')
  print(f'>> {t}: seeds {a.seeds} (fake featurisation)', flush=True)
