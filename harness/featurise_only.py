"""Follow-up 5, host-side featurisation (notes/analysis_plan.md section 10, row of 2026-10-18; label feat): AlphaFold3's
own featurisation of frozen inputs, without the model, the weights or an accelerator.

    third_party/alphafold3/.venv/bin/python harness/featurise_only.py --manifest inputs/feat/manifest.csv \\
        --targets 5SD5 --seeds 1,2,3,4,5 --out results/af3/<session>/featurise [--save_arrays]

For each target: the frozen input (data/inputs/...) must match the manifest's SHA-256. For each seed, the input every
arm ran is rebuilt as harness/run_af3.py builds it (modelSeeds set to [seed], json.dumps(indent=2) plus a newline;
its SHA-256 is the run_input_sha256 of the arms' runs.jsonl), loaded with the loader run_alphafold.py uses for
--json_path (folding_input.load_fold_inputs_from_path), and featurised by the call run_alphafold.py's
predict_structure makes before inference (v3.0.4 lines 577 to 587; 578 to 588 with af3_tpu.patch, the file
every arm ran, SHA-256 c62bfaca...):

    ccd = chemical_components.Ccd(user_ccd=fold_input.user_ccd)
    featurised_examples = featurisation.featurise_input(
        fold_input=fold_input, buckets=buckets, ccd=ccd, verbose=True,
        ref_max_modified_date=ref_max_modified_date, conformer_max_iterations=conformer_max_iterations,
        resolve_msa_overlaps=resolve_msa_overlaps, fix_standalone_glycans=fix_standalone_glycans)

with the values main() passes (v3.0.4 lines 989 and 1064 to 1068; 996 and 1071 to 1075 patched), read from run_alphafold.py's own flags parsed with the command
line the harness gives AlphaFold3 (--json_path, --run_data_pipeline=false; no --buckets, so AlphaFold3's default
list). AlphaFold3 is imported, never changed. Each seed is its own call on its own one-seed input, as each arm's
process featurised one seed; featurise_input builds a new pipeline and np.random.RandomState(seed) per call.

Outputs in --out:
  <target>/features.jsonl  one line per (target, seed): target, seed, input and run-input SHA-256, tokens, bucket,
                           seconds, and per feature its name, dtype, shape and SHA-256 (of the C-order bytes; the
                           model's inputs, model_input true, are the arrays the model receives on the device)
  <target>/arrays/seed-<S>.npz  with --save_arrays: every model-input array, for the max absolute difference
                           (analysis/feat.py)
  host.json                the host: CPU model, flags, family/model/stepping, vCPUs, OS release, kernel, glibc,
                           Python, numpy (and its BLAS), rdkit, AlphaFold3 version, the SHA-256 of the CCD pickles
                           build_data wrote, Compute Engine machine type and zone
Exit status: 0 if every (target, seed) was featurised, 1 otherwise (each failure is a line with an error).
"""
import argparse
import csv
import dataclasses
import datetime
import hashlib
import importlib.metadata as md
import json
import os
import pathlib
import platform
import sys
import time
import traceback

REPO = pathlib.Path(__file__).resolve().parent.parent
AF3_DIR = REPO / 'third_party' / 'alphafold3'
sys.path.insert(0, str(REPO / 'harness'))
from run_af3 import gce_metadata, host_info, sha256_file  # noqa: E402

# The command line harness/run_af3.py gives run_alphafold.py, minus what does not reach featurisation (paths of
# the output, weights and compilation cache, the backend and attention flags read only by the model).
AF3_ARGV = ['run_alphafold.py', '--json_path=input.json', '--run_data_pipeline=false']


def load_af3():
  """Imports AlphaFold3 (run_alphafold.py and the modules it uses) and parses run_alphafold.py's flags."""
  sys.path.insert(0, str(AF3_DIR))
  from absl import flags
  import run_alphafold
  flags.FLAGS(AF3_ARGV)
  return run_alphafold


def featurisation_args(run_alphafold):
  """The keyword values main() passes to process_fold_input and on to featurise_input."""
  return {
      'buckets': tuple(int(bucket) for bucket in run_alphafold._BUCKETS.value),
      'ref_max_modified_date': datetime.date.fromisoformat(run_alphafold._MAX_TEMPLATE_DATE.value),
      'conformer_max_iterations': run_alphafold._CONFORMER_MAX_ITERATIONS.value,
      'resolve_msa_overlaps': run_alphafold._RESOLVE_MSA_OVERLAPS.value,
      'fix_standalone_glycans': run_alphafold._FIX_STANDALONE_GLYCANS.value,
  }


def featurise(run_alphafold, fold_input, args):
  """run_alphafold.predict_structure's featurisation, with its own module objects."""
  chemical_components = run_alphafold.chemical_components
  featurisation = run_alphafold.featurisation
  ccd = chemical_components.Ccd(user_ccd=fold_input.user_ccd)
  return featurisation.featurise_input(
      fold_input=fold_input,
      buckets=args['buckets'],
      ccd=ccd,
      verbose=True,
      ref_max_modified_date=args['ref_max_modified_date'],
      conformer_max_iterations=args['conformer_max_iterations'],
      resolve_msa_overlaps=args['resolve_msa_overlaps'],
      fix_standalone_glycans=args['fix_standalone_glycans'],
  )


def run_input_text(frozen_text, seed):
  """The one-seed input harness/run_af3.py writes to work/input_seeds.json."""
  data = json.loads(frozen_text)
  if seed not in data['modelSeeds']:
    raise SystemExit(f'seed {seed} not in the frozen input seeds {data["modelSeeds"]}')
  data['modelSeeds'] = [seed]
  return json.dumps(data, indent=2) + '\n'


def array_records(run_alphafold, batch):
  """name, dtype, shape and SHA-256 of every feature: the arrays run_alphafold.py's ModelRunner.run_inference sends
  to the device (model_input true: those utils.remove_invalidly_typed_feats keeps), then the others, which stay on
  the host (model_input false: empty_output_struc, the coordinate-free output structure, hashed as its mmCIF; the
  atom layouts, by their fields; a missing bond layout, as None)."""
  import numpy as np
  kept = run_alphafold.utils.remove_invalidly_typed_feats(batch)
  out = []
  for name in sorted(batch):
    v = batch[name]
    if name in kept:
      a = np.asarray(v)
      out.append({'name': name, 'model_input': True, 'dtype': str(a.dtype), 'shape': list(a.shape),
                  'sha256': hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()})
      continue
    obj = v.item() if isinstance(v, np.ndarray) and v.dtype == object and v.shape == () else v
    if obj is None:
      data, how = b'None', 'none'
    elif hasattr(obj, 'to_mmcif'):
      data, how = obj.to_mmcif().encode(), 'mmcif'
    elif dataclasses.is_dataclass(obj):   # AtomLayout: arrays of names, ids and types
      parts = []
      for field in dataclasses.fields(obj):
        f = np.asarray(getattr(obj, field.name))
        # Object arrays (strings) hold pointers: their elements are hashed as text instead.
        body = ('\x1f'.join(map(str, f.ravel())).encode() if f.dtype == object
                else np.ascontiguousarray(f).tobytes())
        parts.append(f'{field.name}|{f.dtype}|{f.shape}|'.encode() + body)
      data, how = b'\0'.join(parts), 'dataclass fields'
    elif isinstance(v, np.ndarray) and v.dtype != object:
      data, how = np.ascontiguousarray(v).tobytes(), 'bytes'
    else:
      raise ValueError(f'feature {name}: {type(obj).__name__} has no stable serialisation to hash')
    out.append({'name': name, 'model_input': False, 'dtype': str(getattr(v, 'dtype', type(v).__name__)),
                'shape': list(getattr(v, 'shape', ())), 'hashed_as': how, 'sha256': hashlib.sha256(data).hexdigest()})
  return out


def cpuinfo():
  """CPU fields of /proc/cpuinfo (first processor), or the macOS sysctl equivalents."""
  out = {}
  try:
    for line in open('/proc/cpuinfo'):
      if not line.strip():
        break
      k, _, v = line.partition(':')
      k = k.strip()
      if k in ('vendor_id', 'cpu family', 'model', 'model name', 'stepping', 'microcode', 'flags'):
        out[k] = v.strip()
  except OSError:
    if sys.platform == 'darwin':
      import subprocess
      for k, key in (('model name', 'machdep.cpu.brand_string'), ('flags', 'machdep.cpu.features')):
        v = subprocess.run(['sysctl', '-n', key], capture_output=True, text=True).stdout.strip()
        if v:
          out[k] = v.lower() if k == 'flags' else v
  flags = out.get('flags', '').split()
  out['avx512f'] = ('avx512f' in flags) if flags else None
  out['avx512_flags'] = sorted(f for f in flags if f.startswith('avx512'))
  return out


def os_release():
  out = {}
  try:
    for line in open('/etc/os-release'):
      k, _, v = line.strip().partition('=')
      if k in ('PRETTY_NAME', 'ID', 'VERSION_ID', 'VERSION_CODENAME'):
        out[k] = v.strip('"')
  except OSError:
    out['PRETTY_NAME'] = platform.platform()
  return out


def version(package):
  try:
    return md.version(package)
  except md.PackageNotFoundError:
    return None


def describe_host(run_alphafold):
  import numpy as np
  try:
    glibc = os.confstr('CS_GNU_LIBC_VERSION')
  except (ValueError, OSError, AttributeError):
    glibc = None
  try:
    blas = np.show_config(mode='dicts').get('Build Dependencies', {}).get('blas')
  except Exception as e:  # numpy older than 1.26
    blas = f'unknown ({type(e).__name__})'
  converters = pathlib.Path(run_alphafold.chemical_components.__file__).parent / 'converters'
  pickles = {p.name: sha256_file(p) for p in sorted(converters.glob('*.pickle'))}
  return {
      **host_info(), 'cpuinfo': cpuinfo(), 'os_release': os_release(), 'kernel': platform.release(),
      'glibc': glibc, 'python': platform.python_version(),
      'packages': {p: version(p) for p in ('alphafold3', 'numpy', 'rdkit', 'scipy', 'jax', 'jaxlib')},
      'numpy_blas': blas,
      'threads_env': {k: os.environ.get(k) for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')},
      'ccd_pickles_sha256': pickles,
      'af3_argv': AF3_ARGV, 'featurisation_args': {k: str(v) for k, v in featurisation_args(run_alphafold).items()},
      'gce_min_cpu_platform': gce_metadata('cpu-platform'),
  }


def main():
  ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
  ap.add_argument('--manifest', required=True)
  ap.add_argument('--targets', required=True, help='comma-separated PDB IDs')
  ap.add_argument('--seeds', default='1,2,3,4,5')
  ap.add_argument('--out', required=True)
  ap.add_argument('--save_arrays', action='store_true')
  args = ap.parse_args()
  out = pathlib.Path(args.out)
  out.mkdir(parents=True, exist_ok=True)
  seeds = [int(s) for s in args.seeds.split(',')]
  with open(REPO / args.manifest) as f:
    manifest = {r['pdb_id']: r for r in csv.DictReader(f)}
  rows = []
  for t in args.targets.split(','):
    if t not in manifest:
      raise SystemExit(f'{t}: not in {args.manifest}')
    frozen = REPO / manifest[t]['input_path']
    if sha256_file(frozen) != manifest[t]['input_sha256']:
      raise SystemExit(f'{manifest[t]["input_path"]}: SHA-256 differs from {args.manifest}')
    rows.append(manifest[t])

  run_alphafold = load_af3()
  from alphafold3.common import folding_input
  fargs = featurisation_args(run_alphafold)
  host = describe_host(run_alphafold)
  (out / 'host.json').write_text(json.dumps(host, indent=2) + '\n')
  failures = 0
  for row in rows:
    t = row['pdb_id']
    tdir = out / t
    tdir.mkdir(parents=True, exist_ok=True)
    frozen_text = (REPO / row['input_path']).read_text()
    for seed in seeds:
      rec = {'target': t, 'seed': seed, 'manifest': args.manifest, 'input_sha256': row['input_sha256']}
      work = tdir / f'input_seed{seed}.json'
      try:
        text = run_input_text(frozen_text, seed)
        work.write_text(text)
        rec['run_input_sha256'] = hashlib.sha256(text.encode()).hexdigest()
        fold_inputs = list(folding_input.load_fold_inputs_from_path(work))
        if len(fold_inputs) != 1:
          raise ValueError(f'expected one fold input, got {len(fold_inputs)}')
        t0 = time.time()
        batches = featurise(run_alphafold, fold_inputs[0], fargs)
        rec['featurisation_seconds'] = round(time.time() - t0, 2)
        if len(batches) != 1:
          raise ValueError(f'expected one featurised example, got {len(batches)}')
        batch = batches[0]
        rec['num_tokens'] = int(batch['seq_mask'].sum())
        rec['bucket'] = int(batch['seq_mask'].shape[0])
        rec['arrays'] = array_records(run_alphafold, batch)
        if args.save_arrays:
          import numpy as np
          (tdir / 'arrays').mkdir(exist_ok=True)
          kept = run_alphafold.utils.remove_invalidly_typed_feats(batch)
          np.savez_compressed(tdir / 'arrays' / f'seed-{seed}.npz', **{k: np.asarray(v) for k, v in kept.items()})
        rec['error'] = None
      except Exception as e:  # recorded, never silent; the other seeds go on
        failures += 1
        rec['error'] = f'{type(e).__name__}: {e}'[:500]
        traceback.print_exc()
      finally:
        work.unlink(missing_ok=True)
      with open(tdir / 'features.jsonl', 'a') as f:
        f.write(json.dumps(rec) + '\n')
      print(f'>> {t} seed {seed}: ' + (f'{len(rec["arrays"])} arrays, {rec["featurisation_seconds"]} s'
                                     if rec['error'] is None else f'FAILED {rec["error"]}'), flush=True)
  return 1 if failures else 0


if __name__ == '__main__':
  sys.exit(main())
