#!/usr/bin/env python3
"""Follow-up 6 (label msa): does each part's featurisation stay within the allowance its plan was sized with?

    python3 scripts/monitor/msa_feat_check.py [SESSION ...]     # default: every fetched Go r1 session

harness/plans/msa/make_msa_plans.py sized every part with each process's featurisation time scaled by the factor
written in the plan's header ("featurisation x 2"), against the main study's median featurisation seconds of the same
arm and token bucket (make_msa_plans.featurisation_medians). For each session (a fetched results/af3/<session>/ of
a follow-up 6 part) this prints, from its runs.jsonl records: the processes done against the part's planned count,
the featurisation seconds observed against the allowance (factor x the main-study median, summed over the processes
done), the observed factor (observed over the main-study medians), the processes above their own allowance, and the
job time against the part's expected job time and deadline (parts.csv; the start from launcher_events.txt, the
end from the VM's final_status.json, or launcher_events.txt without it). A part whose observed featurisation exceeds its allowance, or whose job ran longer than expected, gets a
WARNING line. Nothing is changed: plans are regenerated only by hand, with a section 10 row. Exit 0 without
warnings, 1 with any, 2 if a session is not a follow-up 6 part. Reads only local files.
"""
import csv
import datetime as dt
import json
import pathlib
import re
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'harness' / 'plans' / 'msa'))
import make_msa_plans as plans  # noqa: E402

SESSION = re.compile(r'^\d{8}T\d{6}Z_(l4|v5e|v6e)_(msa_[a-z0-9_]+_\d{2})$')


def factor_of(plan_text):
  m = re.search(r'featurisation x ([0-9.]+)', plan_text)
  return float(m.group(1)) if m else None


def job_minutes(session_dir):
  f = session_dir / 'launcher_events.txt'
  ev = {}
  for line in (f.read_text().splitlines() if f.is_file() else []):
    m = re.match(r'(\S+)\s+(\w+): (\w+)', line)
    if m:
      ev.setdefault(f'{m.group(2)}:{m.group(3)}', dt.datetime.strptime(m.group(1), '%Y-%m-%dT%H:%M:%SZ'))
  if 'job:started' not in ev:
    return None
  # The VM's own finish time (final_status.json, written by the job) when present: the launcher's 'job: finished'
  # event is the time the laptop read it, later than the finish if the laptop was offline meanwhile.
  fs = session_dir / 'final_status.json'
  if fs.is_file():
    fin = json.loads(fs.read_text()).get('finished_utc')
    if fin:
      return (dt.datetime.strptime(fin, '%Y-%m-%dT%H:%M:%SZ') - ev['job:started']).total_seconds() / 60
  if 'job:finished' in ev:
    return (ev['job:finished'] - ev['job:started']).total_seconds() / 60
  return None


def check(session_dir, parts, medians, manifest):
  """(lines, warnings) for one session."""
  m = SESSION.match(session_dir.name)
  if not m or m.group(2) not in parts:
    raise SystemExit(f'{session_dir.name}: not a session of a follow-up 6 part (harness/plans/msa/parts.csv)')
  part = parts[m.group(2)]
  plan = REPO / 'harness' / 'plans' / 'msa' / f'{part["part"]}.yaml'
  factor = factor_of(plan.read_text())
  med = medians[part['main_arm']]
  procs = []
  for runs in sorted(session_dir.glob('*/runs.jsonl')):
    for line in runs.read_text().splitlines():
      r = json.loads(line)
      if r.get('exit_code') == 0 and r.get('featurisation_seconds') is not None:
        b = int(r.get('bucket') or manifest[r['pdb_id']]['bucket'])
        procs.append((r['pdb_id'], runs.parent.name, b, float(r['featurisation_seconds']), med[b]))
  observed = sum(p[3] for p in procs)
  base = sum(p[4] for p in procs)
  allowance = factor * base
  over = [p for p in procs if p[3] > factor * p[4]]
  lines = [f'{session_dir.name} ({part["arm"].replace("_", " ")}, plan {part["part"]}, featurisation x {factor:g})',
           f'   processes done: {len(procs)} of {part["processes"]}',
           f'   featurisation: observed {observed:.0f} s, allowance {allowance:.0f} s (x {factor:g} the main study\'s '
           f'{base:.0f} s); observed factor {observed / base:.2f}' if base else '   featurisation: no process done yet',
           f'   processes above their own allowance: {len(over)}'
           + (' (' + ', '.join(f'{t} {run.split("_rep")[-1] if "_rep" in run else run} {o:.0f} s > {factor * b_:.0f} s'
                               for t, run, _, o, b_ in over[:5]) + (', ...' if len(over) > 5 else '') + ')' if over else '')]
  warnings = []
  if base and observed > allowance:
    warnings.append(f'WARNING {part["part"]}: featurisation {observed:.0f} s exceeds the plan\'s allowance {allowance:.0f} s '
                    f'(observed factor {observed / base:.2f} > {factor:g}); the plan does not hold for this part. No plan '
                    'was changed: regenerate by hand (make_msa_plans.py --feat_scale) with a section 10 row if needed.')
  jm = job_minutes(session_dir)
  exp_min = float(part['expected_job_h']) * 60
  if jm is not None:
    lines.append(f'   job: {jm:.0f} min, expected {exp_min:.0f} min, deadline {part["deadline_min"]} min')
    if jm > exp_min:
      warnings.append(f'WARNING {part["part"]}: the job took {jm:.0f} min, more than the expected {exp_min:.0f} min '
                      f'(deadline {part["deadline_min"]} min).')
  else:
    lines.append(f'   job: not finished (expected {exp_min:.0f} min, deadline {part["deadline_min"]} min)')
  return lines, warnings


def main(argv):
  parts = {p['part']: p for p in csv.DictReader(open(REPO / 'harness' / 'plans' / 'msa' / 'parts.csv'))}
  manifest = {r['pdb_id']: r for r in csv.DictReader(open(REPO / 'inputs' / 'manifest.csv'))}
  medians = plans.featurisation_medians(manifest)
  af3 = REPO / 'results' / 'af3'
  sessions = [af3 / s for s in argv] if argv else sorted(p for p in af3.glob('*_l4_msa_go_r1_*') if p.is_dir())
  if not sessions:
    print('no Go r1 session fetched yet')
    return 0
  warnings = []
  for s in sessions:
    lines, w = check(s, parts, medians, manifest)
    print('\n'.join(lines))
    warnings += w
  print('\n'.join(warnings) if warnings else 'every part within its featurisation allowance and expected job time')
  return 1 if warnings else 0


if __name__ == '__main__':
  try:
    sys.exit(main(sys.argv[1:]))
  except SystemExit as e:
    if isinstance(e.code, str):
      print(e.code, file=sys.stderr)
      sys.exit(2)
    raise
