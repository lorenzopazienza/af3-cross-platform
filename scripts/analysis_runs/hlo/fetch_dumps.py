"""Follow-up 16: read-only copies from the results bucket of what a process of the listed sessions lacks locally (its
xla_dump/ folder or a full output file), for the case the launcher's final fetch did not bring everything (both parts
run with FETCH=full, so normally nothing is copied).

    python3 scripts/analysis_runs/hlo/fetch_dumps.py [--dry-run | --check]

Processes: the 16 of scripts/analysis_runs/hlo/sessions.txt (2 sessions, runs tpu_xla_hlo_rep1/rep2_seed1_fresh,
targets 7U3J, 7NPL, 7XQZ and 7D5C). For each process that lacks a file: gcloud storage rsync --recursive of
gs://af2-tpu-benchmark-af3-results/af3/<session>/<run>/<target> into the local folder. Nothing is written to the
bucket. --dry-run prints the commands; --check copies nothing and exits 1 if a needed file is missing locally
(scripts/analysis_runs/hlo/reproduce.sh runs it first). The registered references (main study V5 and V6 r1 seed 1)
were fetched in full at the time and are checked too (never copied here). Standard library only.
"""
import argparse
import pathlib
import re
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parents[3]
BUCKET = 'gs://af2-tpu-benchmark-af3-results/af3'
SESSIONS = REPO / 'scripts' / 'analysis_runs' / 'hlo' / 'sessions.txt'
TARGETS = ('7U3J', '7NPL', '7XQZ', '7D5C')
RUNS = ('tpu_xla_hlo_rep1_seed1_fresh', 'tpu_xla_hlo_rep2_seed1_fresh')
REF = {'v5e': '20261013T132954Z_v5e_main_v5_01', 'v6e': '20261013T205257Z_v6e_main_v6_01'}
sys.path.insert(0, str(REPO / 'analysis'))
from hlo import DUMP_OPT as OPT, DUMP_PRE as PRE  # noqa: E402  (standard library only, as this script)


def listed():
  if not SESSIONS.is_file():
    return []
  return [l.split('#')[0].strip() for l in SESSIONS.read_text().splitlines() if l.split('#')[0].strip()]


def outputs(d, t, seed=1):
  out = d / 'af3_output' / t
  return [out / f'{t}_data.json', out / f'{t}_confidences.json'] + [
      out / f'seed-{seed}_sample-{k}' / f'{t}_seed-{seed}_sample-{k}{s}' for k in range(5)
      for s in ('_model.cif', '_summary_confidences.json', '_confidences.json')]


def missing(session, run, t, dump=True):
  d = REPO / 'results' / 'af3' / session / run / t
  miss = [str(p.relative_to(d)) for p in outputs(d, t) if not p.is_file()]
  if dump:
    names = [p.name for p in (d / 'xla_dump').iterdir()] if (d / 'xla_dump').is_dir() else []
    if sum(bool(OPT.search(n)) for n in names) != 1 or sum(bool(PRE.search(n)) for n in names) != 1:
      miss.append('xla_dump/ (one optimised and one unoptimised dump of jit_apply_fn)')
  return miss


def main():
  ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
  g = ap.add_mutually_exclusive_group()
  g.add_argument('--dry-run', action='store_true')
  g.add_argument('--check', action='store_true')
  args = ap.parse_args()
  ss = listed()
  units = [(s, r, t) for s in ss for r in RUNS for t in TARGETS]
  todo = [u for u in units if missing(*u)]
  refs = [(REF[c], 'tpu_xla_rep1_seed1_fresh', t) for c in REF for t in TARGETS]
  ref_bad = [u for u in refs if missing(*u, dump=False)]
  print(f'>> {len(units)} processes of {len(ss)} sessions; {len(todo)} lack a file locally; '
        f'{len(refs) - len(ref_bad)} of {len(refs)} registered references complete')
  for u in ref_bad:
    print(f'!! reference {u[0]}/{u[1]}/{u[2]} lacks {missing(*u, dump=False)[:3]}')
  if args.check:
    for u in todo:
      print(f'   missing: {u[0]}/{u[1]}/{u[2]}: {missing(*u)[:3]}')
    if todo:
      print('!! run scripts/analysis_runs/hlo/fetch_dumps.py first (read-only copy from the results bucket)')
    return 1 if todo or ref_bad or len(ss) != 2 else 0
  bad = 0
  for i, (s, run, t) in enumerate(todo, 1):
    src, dst = f'{BUCKET}/{s}/{run}/{t}', REPO / 'results' / 'af3' / s / run / t
    cmd = ['gcloud', 'storage', 'rsync', '--recursive', src, str(dst)]
    if args.dry_run:
      print(' '.join(cmd))
      continue
    r = subprocess.run(cmd, capture_output=True, text=True)
    left = missing(s, run, t)
    bad += bool(r.returncode or left)
    print(f'[{i}/{len(todo)}] {s}/{run}/{t}: {"ok" if not r.returncode and not left else "FAILED " + (r.stderr or "")[-200:]}',
          flush=True)
  if not args.dry_run:
    print(f'>> {len(todo) - bad} of {len(todo)} copied; {bad} failed')
  return 1 if bad else 0


if __name__ == '__main__':
  sys.exit(main())
