"""Follow-up 17 (label dtp): the complete session of each part, checked against its plan, read-only.

    python3 scripts/analysis_runs/dtp/check_sessions.py [--write] [--a-only] [--local-full SESSION]

For every part of harness/plans/dtp_parts.csv, every session folder results/af3/<stamp>_<platform>_<part> is a
candidate (a launch that created no VM leaves only results/af3/.launcher/ and is not one). A candidate is complete if:
final_status.json has exit_code 0 and final_upload_complete; plan_runs.jsonl has exactly the plan's runs
(harness/run_plan.py --describe), each with harness exit code 0; each run's runs.jsonl has one line per planned
target, each exit_code 0; each process has its 5 sample mmCIFs and summary confidences locally, and, for V6fd r2
(FETCH=full), every full output file as well; and a wrapper log in results/af3/.many/ for that session prints "Plan
finished: every harness run exited 0". Exactly one complete session per part is required. Prints one line per part
and the candidates that are not complete with the reason; --write writes the complete sessions to
scripts/analysis_runs/dtp/sessions.txt under its header. --a-only accepts part B parts without a session (part B may
not have run, by its gate: they are then listed as not run and are not a problem). The functions are follow-up 15's
(scripts/analysis_runs/rcx/check_sessions.py), with this follow-up's parts. Standard library only.
"""
import argparse
import csv
import json
import pathlib
import re
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parents[3]
AF3 = REPO / 'results' / 'af3'
SESSIONS = REPO / 'scripts' / 'analysis_runs' / 'dtp' / 'sessions.txt'
B_ARMS = ('Afd', 'Hfd')
FULL_ARMS = ('V6fd_r2',)
FINISHED = 'Plan finished: every harness run exited 0'


def describe(plan, platform):
  out = subprocess.run([sys.executable, str(REPO / 'harness' / 'run_plan.py'), '--plan', plan, '--platform', platform,
                        '--describe'], capture_output=True, text=True, cwd=REPO, check=True)
  return json.loads(out.stdout)


def plan_finished_logs():
  """{session name: wrapper log path} for every wrapper log that prints the finished line for a session."""
  found = {}
  for log in sorted((AF3 / '.many').glob('*/*.log')):
    t = log.read_text(errors='replace')
    if FINISHED in t:
      for s in set(re.findall(r'\d{8}T\d{6}Z_(?:v6e|a100|h100|l4)_dtp_\w+_\d\d', t)):
        found.setdefault(s, log)
  return found


def check(session, d, finished, local_full=False, upload=True):
  root = AF3 / session
  try:
    fs = json.loads((root / 'final_status.json').read_text())
  except (OSError, ValueError):
    return 'no final_status.json'
  if fs.get('exit_code') != 0 or (not fs.get('final_upload_complete') and upload):
    return f'final_status exit {fs.get("exit_code")}, upload complete {fs.get("final_upload_complete")}'
  runs = [json.loads(l) for l in (root / 'plan_runs.jsonl').read_text().splitlines() if l.strip()] \
      if (root / 'plan_runs.jsonl').is_file() else []
  planned = [r['run'] for r in d['runs']]
  if sorted(r['run'] for r in runs) != sorted(planned):
    return f'plan_runs.jsonl runs {sorted(r["run"] for r in runs)} != planned {sorted(planned)}'
  if any(r.get('harness_exit_code') != 0 for r in runs):
    return 'a harness run did not exit 0'
  n = 0
  for r in d['runs']:
    lines = [json.loads(l) for l in (root / r['run'] / 'runs.jsonl').read_text().splitlines() if l.strip()] \
        if (root / r['run'] / 'runs.jsonl').is_file() else []
    if sorted(x['pdb_id'] for x in lines) != sorted(d['targets']) or any(x['exit_code'] != 0 for x in lines):
      return f'{r["run"]}: {len(lines)} processes, exit codes {sorted({x["exit_code"] for x in lines})}'
    seed = int(re.search(r'_seed(\d+)_', r['run']).group(1))
    for t in d['targets']:
      out = root / r['run'] / t / 'af3_output' / t
      cifs = list(out.glob(f'seed-{seed}_sample-*/{t}_seed-{seed}_sample-*_model.cif'))
      sums = list(out.glob(f'seed-{seed}_sample-*/{t}_seed-{seed}_sample-*_summary_confidences.json'))
      if len(cifs) != 5 or len(sums) != 5:
        return f'{r["run"]}/{t}: {len(cifs)} mmCIFs, {len(sums)} summary confidences locally'
      if local_full:
        full = [out / f'{t}_data.json', out / f'{t}_confidences.json'] + [
            out / f'seed-{seed}_sample-{k}' / f'{t}_seed-{seed}_sample-{k}_confidences.json' for k in range(5)]
        if not all(f.is_file() for f in full):
          return f'{r["run"]}/{t}: not every full output local'
      n += 1
  if session not in finished:
    return f'no wrapper log prints "{FINISHED}"'
  return n


def main():
  ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
  ap.add_argument('--write', action='store_true')
  ap.add_argument('--a-only', action='store_true')
  ap.add_argument('--local-full', action='append', default=[], metavar='SESSION')
  args = ap.parse_args()
  parts = list(csv.DictReader(open(REPO / 'harness' / 'plans' / 'dtp_parts.csv')))
  finished = plan_finished_logs()
  chosen, bad, total = [], 0, 0
  for p in parts:
    d = describe(f'harness/plans/dtp/{p["part"]}.yaml', p['platform'])
    expected = len(d['runs']) * len(d['targets'])
    cands = sorted(s.name for s in AF3.glob(f'*_{p["platform"]}_{p["part"]}') if s.is_dir())
    ok, notes = [], []
    for s in cands:
      r = check(s, d, finished, local_full=s in args.local_full or p['arm'] in FULL_ARMS,
                upload=s not in args.local_full)
      if isinstance(r, int) and r == expected:
        ok.append((s, r))
      else:
        notes.append(f'{s}: {r if not isinstance(r, int) else f"{r} processes, expected {expected}"}')
    if len(ok) == 1:
      chosen.append(ok[0][0])
      total += ok[0][1]
      print(f'{p["part"]:14s} {ok[0][0]}  {ok[0][1]}/{expected} processes exit 0, outputs local, plan finished'
            + ('  [final upload incomplete; every output file local, --local-full]' if ok[0][0] in args.local_full else '')
            + (f'  (not listed: {"; ".join(notes)})' if notes else ''))
    elif p['arm'] in B_ARMS and not cands and args.a_only:
      print(f'{p["part"]:14s} not run (part B)')
    else:
      bad += 1
      print(f'{p["part"]:14s} PROBLEM: {len(ok)} complete sessions; ' + '; '.join(notes))
  print(f'>> {len(chosen)} of {len(parts)} parts with exactly one complete session; {total} processes; {bad} problems')
  if args.write and not bad:
    header = [l for l in SESSIONS.read_text().splitlines() if l.startswith('#')]
    SESSIONS.write_text('\n'.join(header + chosen) + '\n')
    print(f'>> written {SESSIONS.relative_to(REPO)}')
  return 1 if bad else 0


if __name__ == '__main__':
  sys.exit(main())
