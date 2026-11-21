"""Follow-up 17 (label dtp): read-only copies from the results bucket of the files FETCH=light left there, for the
processes the analysis compares byte for byte (analysis/dtp.py check 5, contrasts.outputs_identical, which expects every
output file: *_data.json and the full *_confidences.json as well as what the light fetch brought). A copy of follow-up
14's script with this follow-up's processes.

    python3 scripts/analysis_runs/dtp/fetch_full_outputs.py [--dry-run | --check]

Processes: V6fd r1, seed 1, the 10 pilot targets (light fetch) and V6fd r2 (FETCH=full already, checked only), 20 in
all. Sessions from scripts/analysis_runs/dtp/sessions.txt.
For each process: gcloud storage rsync --recursive of gs://af2-tpu-benchmark-af3-results/af3/<session>/<run>/<target>/
af3_output into the local folder, only *_data.json and *_confidences.json (not the summary ones), as
scripts/monitor/main_20261013/fetch_and_crosscheck_warm.sh did for the main study. Nothing is written to the bucket.
About 0.3 GB to copy (the V6fd r1 pilot processes). --dry-run prints the commands; --check copies nothing and exits 1 if a needed file is missing locally
(scripts/analysis_runs/dtp/reproduce.sh runs it first). Standard library only.
"""
import argparse
import csv
import pathlib
import re
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parents[3]
BUCKET = 'gs://af2-tpu-benchmark-af3-results/af3'
ONLY = r'^(?!.*(_data\.json|(?<!_summary)_confidences\.json)$)'      # rsync --exclude: everything else
SESSIONS = REPO / 'scripts' / 'analysis_runs' / 'dtp' / 'sessions.txt'


def units():
  """[(session, run, target, seed)] of the processes the identity checks compare."""
  pilot = {r['pdb_id'] for r in csv.DictReader(open(REPO / 'inputs' / 'manifest.csv')) if r['pilot'] == 'yes'}
  sessions = [l.strip() for l in SESSIONS.read_text().splitlines() if l.strip() and not l.startswith('#')]
  out = []
  for s in sessions:
    root = REPO / 'results' / 'af3' / s
    for run in sorted(p for p in root.iterdir() if p.is_dir() and p.name.endswith('_fresh')):
      seed = int(re.search(r'_seed(\d+)_', run.name).group(1))
      for t in sorted(p.name for p in run.iterdir() if (p / 'af3_output').is_dir()):
        keep = (('_dtp_v6fd_r2_' in s)
                or ('_dtp_v6fd_' in s and 'dtp_f32def_rep1_seed1_' in run.name and t in pilot))
        if keep:
          out.append((s, run.name, t, seed))
  return sorted(set(out))


def missing(session, run, target, seed):
  d = REPO / 'results' / 'af3' / session / run / target / 'af3_output' / target
  need = [d / f'{target}_data.json', d / f'{target}_confidences.json'] + \
      [d / f'seed-{seed}_sample-{k}' / f'{target}_seed-{seed}_sample-{k}_confidences.json' for k in range(5)]
  return [p for p in need if not p.is_file()]


def main():
  ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
  g = ap.add_mutually_exclusive_group()
  g.add_argument('--dry-run', action='store_true')
  g.add_argument('--check', action='store_true')
  args = ap.parse_args()
  us = units()
  todo = [u for u in us if missing(*u)]
  print(f'>> {len(us)} processes compared byte for byte; {len(todo)} lack full outputs locally')
  if args.check:
    for u in todo[:10]:
      print(f'   missing: {u[0]}/{u[1]}/{u[2]} ({len(missing(*u))} files)')
    if todo:
      print('!! run scripts/analysis_runs/dtp/fetch_full_outputs.py first (read-only copy from the results bucket)')
    return 1 if todo else 0
  bad = 0
  for i, (s, run, t, seed) in enumerate(todo, 1):
    src, dst = f'{BUCKET}/{s}/{run}/{t}/af3_output', REPO / 'results' / 'af3' / s / run / t / 'af3_output'
    cmd = ['gcloud', 'storage', 'rsync', '--recursive', f'--exclude={ONLY}', src, str(dst)]
    if args.dry_run:
      print(' '.join(cmd))
      continue
    r = subprocess.run(cmd, capture_output=True, text=True)
    left = missing(s, run, t, seed)
    bad += bool(r.returncode or left)
    print(f'[{i}/{len(todo)}] {s}/{run}/{t}: {"ok" if not r.returncode and not left else "FAILED " + (r.stderr or "")[-200:]}',
          flush=True)
  if not args.dry_run:
    print(f'>> {len(todo) - bad} of {len(todo)} copied; {bad} failed')
  return 1 if bad else 0


if __name__ == '__main__':
  sys.exit(main())
