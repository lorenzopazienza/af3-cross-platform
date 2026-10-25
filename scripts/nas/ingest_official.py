#!/usr/bin/env python3
"""Follow-up 6 (label msa): checks a copy of a NAS data-pipeline run on the Mac, freezes it, and for naspaz8-af3 run1
builds the official inference inputs. notes/analysis_plan.md section 10, rows of 2026-10-19 (specification) and
2026-10-20 (data-pipeline implementation; runner scripts/nas/run_official_pipeline.sh).

Lorenzo copies the run folder from the VM to the Mac (this script never connects to the NAS or its VMs), into
data/nas/<host>_<run>/, then runs from the repository root:

    python3 scripts/nas/ingest_official.py naspaz8-af3_run1      # the 60 targets: checks, freeze, inference inputs
    python3 scripts/nas/ingest_official.py naspaz8-af3_run2      # 10 pilot targets: checks and freeze only
    python3 scripts/nas/ingest_official.py naspaz6-af3_run1      # 10 pilot targets: checks and freeze only
    python3 scripts/nas/ingest_official.py --smoke PATH          # the 2PV7 _data.json of the setup check

Checks, in order; the first failure stops the script and nothing is built or frozen:
  1. the runner's output manifest (outputs_sha256.txt): every listed file is present with that SHA-256, and every file
     under output/ is listed;
  2. results.csv: the last row of each target (a failed attempt keeps an earlier row) has exit_code 0, its
     _data.json exists with the SHA-256 the row records, and the unpaired and paired MSA sequence counts and the
     template count the row records match the file;
  3. the targets: run1 must hold all 60 targets of inputs/official/manifest.csv with exit_code 0, otherwise it stops
     and names the missing or failed ones (the stop rule of the 2026-10-19 row: if the pipeline has not finished all
     60 targets by 2026-10-28, no inference arm runs); run2 and naspaz6-af3 run1 must hold the 10 pilot targets;
  4. each _data.json against its official input (data/inputs_official/<target>.json, checked against
     inputs/official/manifest.csv): the same name, model seeds, ligands, dialect and protein sequence and chain id;
     the protein gains unpairedMsa, pairedMsa and templates and nothing else changes.
Freeze record, before any inference: results/freeze/msa_<host>_<run>/files_sha256.tsv (every file of the copy,
with size and SHA-256) and summary.md (its SHA-256 is the freeze identifier of the delivery, with the runner's
manifest SHA-256 and, for run1, the inference-input manifest's).
Inference inputs (naspaz8-af3 run1 only; the other deliveries are never inference inputs):
data/inputs_official_data/<target>.json, each the run1 <target>_data.json byte-unchanged (SHA-256 checked after the
copy), outside git like the frozen ColabFold inputs (data/inputs/); inputs/official_data/manifest.csv, in the
columns harness/run_af3.py and harness/run_plan.py read (pdb_id, pilot, input_path, input_sha256, num_tokens,
bucket) plus MSA rows, templates and their source; inputs/official_data/manifest.csv.sha256. num_tokens and bucket
come from AlphaFold3's own featurisation of each _data.json (third_party/alphafold3/.venv; templates featurised
from their inline mmCIF, no database), compared with the main study's (inputs/manifest.csv); --no_featurise copies
the main study's values instead (the same sequences and ligands give the same tokens), and says so.
--smoke PATH: the 2PV7 _data.json of the setup check, or the copy of /data/smoke/output/ from naspaz8-af3, which holds
several setup-check runs: every *_data.json under it must have SHA-256
6e8904a1e74569308573b09358298520f4cdd4d4e8a1b91dff8ae9a8c3b588fb (section 10, 2026-10-20), their number is printed and
the first in path order is used; copied to
data/inputs_official_data/smoke/2PV7.json with inputs/official_data/smoke_manifest.csv (not a study target).
"""
import argparse
import csv
import datetime
import hashlib
import io
import json
import pathlib
import shutil
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]
NAS = REPO / 'data' / 'nas'
OFFICIAL_MANIFEST = REPO / 'inputs' / 'official' / 'manifest.csv'
MAIN_MANIFEST = REPO / 'inputs' / 'manifest.csv'
DATA_DIR = REPO / 'data' / 'inputs_official_data'
OUT_MANIFEST = REPO / 'inputs' / 'official_data' / 'manifest.csv'
SMOKE_MANIFEST = REPO / 'inputs' / 'official_data' / 'smoke_manifest.csv'
FREEZE = REPO / 'results' / 'freeze'
RUNS = {'naspaz8-af3_run1': 'all', 'naspaz8-af3_run2': 'pilot', 'naspaz6-af3_run1': 'pilot'}
SMOKE_SHA = '6e8904a1e74569308573b09358298520f4cdd4d4e8a1b91dff8ae9a8c3b588fb'   # 2PV7, setup check (2026-10-20)
ADDED = ('unpairedMsa', 'pairedMsa', 'templates')
COLUMNS = ['pdb_id', 'ligand_ccd', 'pilot', 'input_path', 'input_sha256', 'input_bytes', 'sequence_sha256',
           'num_tokens', 'bucket', 'num_seeds', 'main_num_tokens', 'main_bucket', 'tokens_from', 'unpaired_msa_rows',
           'paired_msa_rows', 'templates', 'templates_featurised', 'source_run', 'source_path', 'source_sha256',
           'pipeline_seconds', 'vm_mem_peak_mib', 'rcsb_initial_release', 'released_on_or_before_cutoff']


class Stop(SystemExit):
  pass


def sha256(path, chunk=1 << 24):
  h = hashlib.sha256()
  with open(path, 'rb') as f:
    while block := f.read(chunk):
      h.update(block)
  return h.hexdigest()


def a3m_rows(text):
  return sum(1 for line in (text or '').splitlines() if line.startswith('>'))


def protein_of(d):
  ps = [e['protein'] for e in d['sequences'] if 'protein' in e]
  if len(ps) != 1:
    raise Stop(f'{d.get("name")}: expected one protein entry, found {len(ps)}')
  return ps[0]


def check_manifest(src):
  """Step 1: outputs_sha256.txt against the files, and every output file listed."""
  m = src / 'outputs_sha256.txt'
  if not m.is_file():
    raise Stop(f'{m} missing: the runner writes it when every target is done; copy the whole run folder again')
  listed = {}
  for line in m.read_text().splitlines():
    if line.strip():
      h, _, rel = line.partition('  ')
      listed[rel.strip()] = h.strip()
  bad = [rel for rel, h in listed.items() if not (src / rel).is_file() or sha256(src / rel) != h]
  if bad:
    raise Stop(f'{len(bad)} file(s) differ from outputs_sha256.txt or are missing: {bad[:5]}')
  present = {str(p.relative_to(src)) for p in (src / 'output').rglob('*') if p.is_file()}
  extra = sorted(present - set(listed))
  if extra:
    raise Stop(f'{len(extra)} file(s) under output/ are not in outputs_sha256.txt: {extra[:5]}')
  return listed


def local_path(src, vm_path):
  rel = vm_path[vm_path.index('/output/') + 1:] if '/output/' in vm_path else vm_path
  return src / rel


def check_results(src, want):
  """Steps 2 and 3: the last results.csv row per target, its file and counts; the expected targets all exit 0."""
  rows = list(csv.DictReader(open(src / 'results.csv')))
  last = {}
  for r in rows:
    last[r['pdb_id']] = r
  failed = sorted(t for t in want if t in last and last[t]['exit_code'] != '0')
  missing = sorted(t for t in want if t not in last)
  if failed or missing:
    raise Stop(f'{src.name}: {len(want) - len(failed) - len(missing)} of {len(want)} targets done with exit_code 0'
               + (f'; failed: {", ".join(failed)}' if failed else '') + (f'; not run: {", ".join(missing)}' if missing else '')
               + ('. Stop rule (section 10, 2026-10-19): if the data pipeline has not finished all 60 targets by '
                  '2026-10-28, no inference arm runs and the follow-up is descriptive only.' if len(want) == 60 else ''))
  extra = sorted(set(last) - set(want))
  if extra:
    raise Stop(f'{src.name}: results.csv has targets outside the run: {extra}')
  out = {}
  for t in want:
    r = last[t]
    p = local_path(src, r['data_json'])
    if not p.is_file():
      raise Stop(f'{t}: {r["data_json"]} not in the copy ({p})')
    if sha256(p) != r['data_json_sha256']:
      raise Stop(f'{t}: {p.name} SHA-256 differs from results.csv')
    d = json.loads(p.read_text())
    prot = protein_of(d)
    counts = (str(a3m_rows(prot.get('unpairedMsa'))), str(a3m_rows(prot.get('pairedMsa'))),
              str(len(prot.get('templates') or [])))
    if counts != (r['unpaired_seqs'], r['paired_seqs'], r['templates']):
      raise Stop(f'{t}: MSA or template counts {counts} differ from results.csv '
                 f'{(r["unpaired_seqs"], r["paired_seqs"], r["templates"])}')
    out[t] = (r, p, d)
  return out


def dense(obj):
  """A JSON value without the empty fields AlphaFold3's writer adds (None, [], {}), recursively."""
  if isinstance(obj, dict):
    return {k: dense(v) for k, v in obj.items() if v not in (None, [], {})}
  if isinstance(obj, list):
    return [dense(v) for v in obj]
  return obj


def check_against_input(t, d, official):
  """Step 4: the _data.json is its official input plus MSAs and templates. AlphaFold3 writes it with its own JSON
  writer (folding_input.Input.to_json: version 4, and empty modifications, bondedAtomPairs and userCCD); the
  comparison ignores the version and those empty fields, and everything else must be equal."""
  ip = REPO / official['official_input_path']
  if sha256(ip) != official['official_input_sha256']:
    raise Stop(f'{t}: {ip} SHA-256 differs from {OFFICIAL_MANIFEST.relative_to(REPO)}')
  src = dense(json.loads(ip.read_text()))
  raw_protein = protein_of(d)   # presence of the added fields, before empty ones (templates: []) are dropped
  d = dense(d)
  extra = sorted(set(d) - set(src) - {'version'})
  if extra:
    raise Stop(f'{t}: top-level fields {extra} not in the official input')
  for k in ('name', 'modelSeeds', 'dialect'):
    if d.get(k) != src.get(k):
      raise Stop(f'{t}: {k} {d.get(k)!r} differs from the official input ({src.get(k)!r})')
  if len(d['sequences']) != len(src['sequences']):
    raise Stop(f'{t}: {len(d["sequences"])} entries, the official input has {len(src["sequences"])}')
  for a, b in zip(src['sequences'], d['sequences']):
    (ka, va), = a.items()
    (kb, vb), = b.items()
    if ka != kb:
      raise Stop(f'{t}: entry {ka} became {kb}')
    if ka == 'protein':
      if vb.get('sequence') != va.get('sequence') or vb.get('id') != va.get('id'):
        raise Stop(f'{t}: protein sequence or chain id changed')
      if hashlib.sha256(vb['sequence'].encode()).hexdigest() != official['sequence_sha256']:
        raise Stop(f'{t}: protein sequence SHA-256 differs from the manifest')
      rest = {k: v for k, v in vb.items() if k not in ADDED}
      if rest != va or not all(k in raw_protein for k in ADDED):
        raise Stop(f'{t}: protein fields beyond {", ".join(ADDED)} changed, or one of them is missing')
    elif vb != va:
      raise Stop(f'{t}: {ka} entry changed')


FEATURISE = r'''
import dataclasses, json, sys
sys.path.insert(0, sys.argv[1] + '/harness')
import featurise_only as fo   # run_alphafold.py's own featurisation call and flag values (follow-up 5)
ra = fo.load_af3()
from alphafold3.common import folding_input
args = fo.featurisation_args(ra)
out = {}
for p in sys.argv[2:]:
  fi = list(folding_input.load_fold_inputs_from_path(p))[0]
  fi = dataclasses.replace(fi, rng_seeds=[fi.rng_seeds[0]])   # tokens and templates do not depend on the seed
  b = fo.featurise(ra, fi, args)[0]
  mask = b['template_atom_mask'].reshape(b['template_atom_mask'].shape[0], -1)
  out[p] = {'num_tokens': int(b['seq_mask'].sum()), 'bucket': int(b['seq_mask'].shape[0]),
            'templates_featurised': int((mask.sum(-1) > 0).sum())}
print('@@' + json.dumps(out))
'''


def featurise(paths):
  """num_tokens, bucket and featurised template count of each input, from AlphaFold3's own featurisation."""
  py = REPO / 'third_party' / 'alphafold3' / '.venv' / 'bin' / 'python'
  if not py.is_file():
    raise Stop(f'{py} not found: AlphaFold3 venv needed for the featurisation (or --no_featurise)')
  rc = subprocess.run([str(py), '-c', FEATURISE, str(REPO)] + [str(p) for p in paths], capture_output=True, text=True,
                      cwd=REPO)
  line = next((l for l in rc.stdout.splitlines() if l.startswith('@@')), None)
  if rc.returncode != 0 or line is None:
    raise Stop('featurisation failed:\n' + (rc.stderr or rc.stdout)[-2000:])
  return json.loads(line[2:])


def freeze(src, extra_lines):
  """results/freeze/msa_<run>/: every file of the copy with size and SHA-256; summary.md; returns the identifier."""
  out = FREEZE / f'msa_{src.name}'
  out.mkdir(parents=True, exist_ok=True)
  rows = [(str(p.relative_to(src)), p.stat().st_size, sha256(p)) for p in sorted(src.rglob('*'))
          if p.is_file() and p.name != '.lock']
  tsv = out / 'files_sha256.tsv'
  tsv.write_text('path\tbytes\tsha256\n' + ''.join(f'{r}\t{n}\t{h}\n' for r, n, h in rows))
  ident = sha256(tsv)
  (out / 'summary.md').write_text('\n'.join([
      f'# Freeze: follow-up 6 data-pipeline delivery {src.name}', '',
      f'Made {datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")} by '
      'scripts/nas/ingest_official.py, before any inference run, from the copy in '
      f'{src.relative_to(REPO)}/ (checked against the runner\'s outputs_sha256.txt and results.csv).', '',
      f'Freeze identifier: SHA-256 of files_sha256.tsv = `{ident}`', '',
      f'- files: {len(rows)}, {sum(n for _, n, _ in rows):,} bytes',
      f'- runner output manifest outputs_sha256.txt: `{sha256(src / "outputs_sha256.txt")}`'] + extra_lines) + '\n')
  return ident, out


def build_inputs(checked, official, main, tokens):
  DATA_DIR.mkdir(parents=True, exist_ok=True)
  rows = []
  for t in sorted(checked):
    r, p, d = checked[t]
    dest = DATA_DIR / f'{t}.json'
    shutil.copyfile(p, dest)
    h = sha256(dest)
    if h != r['data_json_sha256']:
      raise Stop(f'{t}: the copy {dest} is not byte-identical to the delivered _data.json')
    prot = protein_of(d)
    o, m = official[t], main[t]
    tk = tokens.get(str(dest)) if tokens else None
    rows.append({
        'pdb_id': t, 'ligand_ccd': o['ligand_ccd'], 'pilot': o['pilot'], 'input_path': str(dest.relative_to(REPO)),
        'input_sha256': h, 'input_bytes': dest.stat().st_size, 'sequence_sha256': o['sequence_sha256'],
        'num_tokens': tk['num_tokens'] if tk else m['num_tokens'], 'bucket': tk['bucket'] if tk else m['bucket'],
        'num_seeds': len(d['modelSeeds']), 'main_num_tokens': m['num_tokens'], 'main_bucket': m['bucket'],
        'tokens_from': 'af3_featurised' if tk else 'main study manifest (--no_featurise)',
        'unpaired_msa_rows': a3m_rows(prot.get('unpairedMsa')), 'paired_msa_rows': a3m_rows(prot.get('pairedMsa')),
        'templates': len(prot.get('templates') or []),
        'templates_featurised': tk['templates_featurised'] if tk else '',
        'source_run': p.relative_to(NAS).parts[0], 'source_path': str(p.relative_to(NAS)),
        'source_sha256': r['data_json_sha256'], 'pipeline_seconds': r['seconds'], 'vm_mem_peak_mib': r['vm_mem_peak_mib'],
        'rcsb_initial_release': o['rcsb_initial_release'],
        'released_on_or_before_cutoff': o['released_on_or_before_cutoff']})
  buf = io.StringIO()
  w = csv.DictWriter(buf, fieldnames=COLUMNS, lineterminator='\n')
  w.writeheader()
  w.writerows(rows)
  OUT_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
  OUT_MANIFEST.write_text(buf.getvalue())
  digest = sha256(OUT_MANIFEST)
  (OUT_MANIFEST.parent / 'manifest.csv.sha256').write_text(f'{digest}  manifest.csv\n')
  return rows, digest


def ingest(run, no_featurise):
  if run not in RUNS:
    raise Stop(f'{run}: not one of the deliveries of the follow-up 6 row ({", ".join(RUNS)})')
  src = NAS / run
  if not src.is_dir():
    raise Stop(f'{src} not found: copy /data/runs/{run}/ from the VM into data/nas/ first')
  official = {r['pdb_id']: r for r in csv.DictReader(open(OFFICIAL_MANIFEST))}
  main = {r['pdb_id']: r for r in csv.DictReader(open(MAIN_MANIFEST))}
  want = sorted(official) if RUNS[run] == 'all' else sorted(t for t, r in official.items() if r['pilot'] == 'yes')
  print(f'>> {run}: {len(want)} targets expected')
  listed = check_manifest(src)
  print(f'   1. outputs_sha256.txt: {len(listed)} files, all present with their SHA-256, none unlisted')
  checked = check_results(src, want)
  print(f'   2-3. results.csv: {len(checked)} of {len(want)} targets exit 0; each _data.json matches its SHA-256 and counts')
  for t, (_, _, d) in checked.items():
    check_against_input(t, d, official[t])
  print('   4. every _data.json is its official input plus unpairedMsa, pairedMsa and templates')
  size = sum(p.stat().st_size for _, p, _ in checked.values())
  print(f'   _data.json total: {size:,} bytes ({size / 2**20:.0f} MiB), largest '
        f'{max(p.stat().st_size for _, p, _ in checked.values()) / 2**20:.1f} MiB')
  if RUNS[run] != 'all':
    ident, out = freeze(src, [f'- role: determinism check only, never an inference input; {len(checked)} pilot targets'])
    print(f'>> freeze {out.relative_to(REPO)}: {ident}')
    return 0
  tokens = None
  if not no_featurise:
    print('   featurising each _data.json with AlphaFold3 (tokens, bucket, templates) ...', flush=True)
    tokens = featurise([p for _, p, _ in (checked[t] for t in sorted(checked))])
    tokens = {str(DATA_DIR / f'{t}.json'): tokens[str(checked[t][1])] for t in checked}
  rows, digest = build_inputs(checked, official, main, tokens)
  changed = [r for r in rows if str(r['bucket']) != r['main_bucket'] or str(r['num_tokens']) != r['main_num_tokens']]
  print(f'>> {len(rows)} inference inputs in {DATA_DIR.relative_to(REPO)}/; manifest {OUT_MANIFEST.relative_to(REPO)} '
        f'SHA-256 {digest}')
  print(f'   tokens and bucket ({rows[0]["tokens_from"]}): ' + (
      'all equal to the main study\'s' if not changed else
      'CHANGED for ' + ', '.join(f'{r["pdb_id"]} ({r["main_num_tokens"]}->{r["num_tokens"]}, bucket '
                                 f'{r["main_bucket"]}->{r["bucket"]})' for r in changed)))
  ident, out = freeze(src, [f'- inference-input manifest inputs/official_data/manifest.csv: `{digest}`',
                            f'- inference inputs: {len(rows)} files in {DATA_DIR.relative_to(REPO)}/, {size:,} bytes, '
                            'each byte-identical to its delivered _data.json'])
  print(f'>> freeze {out.relative_to(REPO)}: {ident}')
  return 0


def smoke(path):
  """PATH is the 2PV7 _data.json, or a copy of /data/smoke/output, which holds several setup-check runs: then every
  *_data.json under it must have SMOKE_SHA, their number is reported, and the first in path order is used."""
  p = pathlib.Path(path)
  found = sorted(p.rglob('*_data.json')) if p.is_dir() else [p]
  if not found:
    raise Stop(f'{p}: no *_data.json')
  hashes = {f: sha256(f) for f in found}
  bad = {f: h for f, h in hashes.items() if h != SMOKE_SHA}
  if bad:
    raise Stop(f'{len(bad)} of {len(found)} _data.json under {p} are not the 2PV7 setup-check file ({SMOKE_SHA}): '
               + '; '.join(f'{f.relative_to(p) if p.is_dir() else f}: {h}' for f, h in list(bad.items())[:5]))
  p = found[0]
  print(f'>> {len(found)} 2PV7 _data.json found, all with SHA-256 {SMOKE_SHA}; using {p}')
  h = SMOKE_SHA
  d = json.loads(p.read_text())
  dest = DATA_DIR / 'smoke' / '2PV7.json'
  dest.parent.mkdir(parents=True, exist_ok=True)
  shutil.copyfile(p, dest)
  tk = featurise([dest])[str(dest)]
  prots = [e['protein'] for e in d['sequences'] if 'protein' in e]
  row = {'pdb_id': '2PV7', 'ligand_ccd': '', 'pilot': 'no', 'input_path': str(dest.relative_to(REPO)),
         'input_sha256': sha256(dest), 'input_bytes': dest.stat().st_size, 'sequence_sha256': '',
         'num_tokens': tk['num_tokens'], 'bucket': tk['bucket'], 'num_seeds': len(d['modelSeeds']),
         'main_num_tokens': '', 'main_bucket': '', 'tokens_from': 'af3_featurised',
         'unpaired_msa_rows': sum(a3m_rows(q.get('unpairedMsa')) for q in prots),
         'paired_msa_rows': sum(a3m_rows(q.get('pairedMsa')) for q in prots),
         'templates': sum(len(q.get('templates') or []) for q in prots), 'templates_featurised': tk['templates_featurised'],
         'source_run': f'setup check 2PV7 (section 10, 2026-10-20); 1 of {len(found)} identical copies',
         'source_path': str(p.relative_to(REPO)) if p.is_relative_to(REPO) else p.name, 'source_sha256': h,
         'pipeline_seconds': '', 'vm_mem_peak_mib': '', 'rcsb_initial_release': '', 'released_on_or_before_cutoff': ''}
  buf = io.StringIO()
  w = csv.DictWriter(buf, fieldnames=COLUMNS, lineterminator='\n')
  w.writeheader()
  w.writerow(row)
  SMOKE_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
  SMOKE_MANIFEST.write_text(buf.getvalue())
  print(f'>> 2PV7: {tk["num_tokens"]} tokens, bucket {tk["bucket"]}, seeds {d["modelSeeds"]}, {row["templates"]} templates '
        f'({tk["templates_featurised"]} featurised); {SMOKE_MANIFEST.relative_to(REPO)}')
  return 0


def main():
  ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
  ap.add_argument('run', nargs='?', help=', '.join(RUNS))
  ap.add_argument('--no_featurise', action='store_true')
  ap.add_argument('--smoke', help='the 2PV7 _data.json of the setup check')
  args = ap.parse_args()
  try:
    return smoke(args.smoke) if args.smoke else ingest(args.run, args.no_featurise)
  except Stop as e:
    print(f'STOP: {e}', file=sys.stderr)
    return 1


if __name__ == '__main__':
  sys.exit(main())
