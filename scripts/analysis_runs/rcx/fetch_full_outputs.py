"""Follow-up 15: read-only copies from the results bucket of the files FETCH=light left there, for the processes the
analysis compares byte for byte (analysis/rcx.py, the strict bit-identity of item e 1, contrasts.outputs_identical,
which expects every output file: *_data.json and the full *_confidences.json as well as what the light fetch brought).

    python3 scripts/analysis_runs/rcx/fetch_full_outputs.py [--dry-run | --check]

Processes: Af r1 and Hf r1 of follow-up 14 (scripts/analysis_runs/fu14/sessions.txt), seeds 1 to 5, the 10 pilot
targets (light fetch: 100 processes to copy), and Af r2 and Hf r2 of follow-up 15 (FETCH=full already, checked only;
scripts/analysis_runs/rcx/sessions.txt), 200 in all. GT r1 and r2 (part A3) are compared on the files of the light
fetch and need nothing here. For each process: gcloud storage rsync --recursive of
gs://af2-tpu-benchmark-af3-results/af3/<session>/<run>/<target>/af3_output into the local folder, only *_data.json and
*_confidences.json (not the summary ones), as scripts/analysis_runs/fu14/fetch_full_outputs.py did. Nothing is written
to the bucket. About 4.5 GB to copy (follow-up 14's full sessions hold about 45 MB per process). --dry-run prints the
commands; --check copies nothing and exits 1 if a needed file is missing locally (scripts/analysis_runs/rcx/
reproduce.sh runs it first). Standard library only.
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
SESSIONS = {'fu14': REPO / 'scripts' / 'analysis_runs' / 'fu14' / 'sessions.txt',
            'rcx': REPO / 'scripts' / 'analysis_runs' / 'rcx' / 'sessions.txt'}


def listed(which):
  f = SESSIONS[which]
  if not f.is_file():
    return []
  return [l.split('#')[0].strip() for l in f.read_text().splitlines() if l.split('#')[0].strip()]


def units():
  """[(session, run, target, seed)] of the processes the strict identity check compares."""
  pilot = {r['pdb_id'] for r in csv.DictReader(open(REPO / 'inputs' / 'manifest.csv')) if r['pilot'] == 'yes'}
  sessions = [s for s in listed('fu14') if re.search(r'_(a100_f32_af|h100_f32_hf)_\d+$', s)]
  sessions += [s for s in listed('rcx') if re.search(r'_(a100_rcx_af_r2|h100_rcx_hf_r2)_\d+$', s)]
  out = []
  for s in sessions:
    root = REPO / 'results' / 'af3' / s
    for run in sorted(p for p in root.iterdir() if p.is_dir() and p.name.endswith('_fresh')):
      seed = int(re.search(r'_seed(\d+)_', run.name).group(1))
      for t in sorted(p.name for p in run.iterdir() if (p / 'af3_output').is_dir()):
        if t in pilot:
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
      print('!! run scripts/analysis_runs/rcx/fetch_full_outputs.py first (read-only copy from the results bucket)')
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
