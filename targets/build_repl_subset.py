"""Targets of follow-up 11, replication on new targets (label repl; notes/analysis_plan.md section 10, row of
2026-11-03): every PoseBusters V2 complex that meets the main study's selection rule and was not drawn into the 60.

Run inside the pinned AlphaFold3 environment, like the main builder:

    cd third_party/alphafold3
    .venv/bin/python ../../targets/build_repl_subset.py

It runs the main builder's screen unchanged (targets/build_posebusters_subset.screen_v2, steps 1 to 5 of its rule)
on the main builder's cached sources, read offline: the PoseBusters zip and the V2 identifier list in data/posebusters/
(their SHA-256 checked against targets/sources.json, where the main builder recorded them after its Zenodo MD5
check), the RCSB mmCIF files in data/rcsb/ and the token counts in data/posebusters/token_counts.json (a missing count
is featurised with AlphaFold3, as the main builder does). Checks, before anything is written: the screen it computes is
byte-identical to the committed targets/posebusters_v2_screen.csv, and no complex of the result is among the 60 of
targets/posebusters_subset.csv (by PoseBusters identifier and by PDB ID).

Rule (fixed by the row, no choice left): all rows with status "eligible" in the screen, ordered by identifier; pilot
"no" for every one; every other column as the main builder writes it for its 60 (targets/posebusters_subset.csv
columns, n_input_chains 2, RCSB release dates from the cached mmCIF revision history).

Outputs: targets/repl/repl_subset.csv and targets/repl/sources.json.
"""

import collections
import csv
import datetime
import hashlib
import io
import json
import pathlib
import subprocess
import sys
import zipfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import build_posebusters_subset as bps  # noqa: E402

REPO = bps.REPO
OUT_DIR = REPO / 'targets' / 'repl'
SPEC_ROW = 'notes/analysis_plan.md section 10, row of 2026-11-03 (follow-up experiment 11, label repl)'


def sha256_file(path):
  return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def csv_text(cols, rows, **extra):
  buf = io.StringIO()
  w = csv.DictWriter(buf, cols, extrasaction='ignore', lineterminator='\n')
  w.writeheader()
  for r in rows:
    w.writerow(dict(r, **extra))
  return buf.getvalue()


def main():
  started = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds')
  recorded = json.loads((REPO / 'targets' / 'sources.json').read_text())['posebusters']
  pb_dir = bps.DATA / 'posebusters'
  zip_bytes = (pb_dir / bps.ZIP_NAME).read_bytes()
  ids_bytes = (pb_dir / 'posebusters_pdb_ccd_ids.txt').read_bytes()
  for name, data, want in ((bps.ZIP_NAME, zip_bytes, recorded['sha256']),
                           ('posebusters_pdb_ccd_ids.txt', ids_bytes, recorded['v2_ids_sha256'])):
    if bps.sha256(data) != want:
      raise SystemExit(f'data/posebusters/{name}: SHA-256 differs from targets/sources.json')
  v2_ids = sorted(l.strip() for l in ids_bytes.decode().splitlines() if l.strip())
  zf = zipfile.ZipFile(io.BytesIO(zip_bytes))
  v1_ids = {l.strip() for l in zf.read('posebusters_benchmark_set_ids.txt').decode().splitlines() if l.strip()}
  cache_path = pb_dir / 'token_counts.json'
  token_cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
  n_cached = len(token_cache)

  screen = bps.screen_v2(zf, v2_ids, v1_ids, bps.af3_featurise.ccd(), token_cache, cache_path)
  # The main builder then marks its 60 as selected; the screen it writes is that, so mark them the same way here.
  main = list(csv.DictReader(open(REPO / 'targets' / 'posebusters_subset.csv')))
  main_ids, main_pdb = {r['posebusters_id'] for r in main}, {r['pdb_id'] for r in main}
  for r in screen:
    if r['posebusters_id'] in main_ids:
      r['status'] = 'selected'
  committed = (REPO / 'targets' / 'posebusters_v2_screen.csv').read_text()
  if csv_text(bps.SCREEN_COLS, screen) != committed:
    raise SystemExit('the screen computed here differs from targets/posebusters_v2_screen.csv; nothing written')
  if sum(r['status'] == 'selected' for r in screen) != len(main_ids):
    raise SystemExit('the 60 of targets/posebusters_subset.csv are not all selected in the screen')

  repl = sorted((r for r in screen if r['status'] == 'eligible'), key=lambda r: r['posebusters_id'])
  overlap = sorted({r['posebusters_id'] for r in repl} & main_ids) + sorted({r['pdb_id'] for r in repl} & main_pdb)
  if overlap:
    raise SystemExit(f'complexes among the 60: {overlap}')
  if len({r['pdb_id'] for r in repl}) != len(repl):
    raise SystemExit('two complexes share a PDB ID; input file names would collide')
  OUT_DIR.mkdir(parents=True, exist_ok=True)
  text = csv_text(bps.SUBSET_COLS, repl, n_input_chains=2, pilot='no')
  (OUT_DIR / 'repl_subset.csv').write_text(text)
  sources = {
      'built_utc': started,
      'built_by': 'targets/build_repl_subset.py',
      'specification': SPEC_ROW,
      'rule': 'every row with status eligible in targets/posebusters_v2_screen.csv (the main study rule, steps 1 to 5 '
              'of targets/build_posebusters_subset.py, not drawn into the 60), ordered by identifier; pilot no',
      'screen': {'recomputed_byte_identical_to': 'targets/posebusters_v2_screen.csv',
                 'sha256': hashlib.sha256(committed.encode()).hexdigest()},
      'inputs_sha256': {
          'data/posebusters/' + bps.ZIP_NAME: bps.sha256(zip_bytes),
          'data/posebusters/posebusters_pdb_ccd_ids.txt': bps.sha256(ids_bytes),
          'targets/posebusters_subset.csv': sha256_file(REPO / 'targets' / 'posebusters_subset.csv'),
          'targets/sources.json': sha256_file(REPO / 'targets' / 'sources.json'),
      },
      'token_cache': {'path': 'data/posebusters/token_counts.json', 'entries_before': n_cached,
                      'entries_after': len(token_cache)},
      'rcsb_mmcif_sha256': {r['pdb_id']: r['rcsb_sha256'] for r in repl},
      'alphafold3': {'commit': subprocess.check_output(
                         ['git', '-C', str(bps.af3_featurise.AF3_DIR), 'rev-parse', 'HEAD'], text=True).strip(),
                     'buckets': list(bps.af3_featurise.BUCKETS)},
      'counts': {
          'targets': len(repl),
          'by_bucket': dict(sorted(collections.Counter(int(r['bucket']) for r in repl).items())),
          'by_bin': dict(collections.Counter(r['bin'] for r in repl)),
          'released_after_2021-09-30': sum(r['rcsb_initial_release'][:10] > '2021-09-30' for r in repl),
      },
      'output': {'path': 'targets/repl/repl_subset.csv', 'sha256': hashlib.sha256(text.encode()).hexdigest()},
  }
  (OUT_DIR / 'sources.json').write_text(json.dumps(sources, indent=2) + '\n')
  print(json.dumps(sources['counts'], indent=2))


if __name__ == '__main__':
  main()
