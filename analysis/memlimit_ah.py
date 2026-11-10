"""Memory limit per chip on A and H (label memlimit_ah; notes/analysis_plan.md section 10, the specification row of
2026-11-10 "Memory limit per chip on the A100 and the H100" and its analysis row): analysis/memlimit.py's logic,
unchanged, on the sessions of label memlimit_ah.

    analysis/.venv/bin/python analysis/memlimit_ah.py --sessions scripts/analysis_runs/memlimit_ah/sessions.txt \\
        --out results/analysis/memlimit_ah_<DATE>

analysis/memlimit.py reads only label memlimit and the arms V5, V6, G and GT, and its outputs of 2026-10-14 are frozen,
so it is not edited: this module calls its functions as they are (process_rows, main_study_rows, summary, report,
error_memory through process_rows, and cost.price_for), with the arm list set to A and H while it runs. The only parts
written here are what memlimit.py hard-codes: which sessions (the ones listed in --sessions, each of label
memlimit_ah, one per arm; the specification row's "complete session" of each part, a preempted H session being
recorded and not listed), the platform and configuration of the arms (a100 with a100_xla -> A, h100 with h100_xla ->
H), and the 1,024-token reference process of each arm, read with memlimit.main_study_rows from the follow-up that
measured it (follow-up 3 for A, results/analysis/a100_20261018/samples.csv; follow-up 9 for H,
results/analysis/h100_20261104/samples.csv; 7D5C, fresh, r1, seed 1), its source column naming that follow-up.
Prices: memlimit.py's, the on-demand list price of the zone's region (cost.price_for, 'standard'), for H too, although
H ran on Spot (as M6 compares platforms at on-demand prices; the Spot price is in cloud/prices.csv).
Checks, before anything is computed (the analysis stops if one fails): every listed session exists and has label
memlimit_ah; exactly one listed session per arm, each of a100 or h100; every process record has a size.
Outputs in --out (a new folder): processes.csv, summary.csv (the columns and format of
results/analysis/memlimit_20261014/summary.csv), report.md and inputs.tsv (path and SHA-256 of every file read);
run.log is written by scripts/analysis_runs/memlimit_ah/reproduce.sh. If every size of an arm completes, its
stopping_size and cause are empty, as memlimit.summary writes them: the result is "at least 5,120 tokens".
"""
import argparse
import contextlib
import hashlib
import json
import os
import pathlib
import re
import sys

import pandas as pd

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / 'analysis'))
import cost  # noqa: E402
import memlimit  # noqa: E402

LABEL = 'memlimit_ah'
ARMS = ['A', 'H']
CODE = {('a100', 'a100_xla'): 'A', ('h100', 'h100_xla'): 'H'}
SAMPLES = {'A': REPO / 'results' / 'analysis' / 'a100_20261018' / 'samples.csv',
           'H': REPO / 'results' / 'analysis' / 'h100_20261104' / 'samples.csv'}
SOURCE = {'A': 'follow-up 3 (7D5C, r1, seed 1)', 'H': 'follow-up 9 (7D5C, r1, seed 1)'}
SESSION = re.compile(r'\d{8}T\d{6}Z_(a100|h100)_memlimit_ah_[ah]_01$')


@contextlib.contextmanager
def arms_a_h():
  """memlimit.ARMS set to A and H while memlimit's functions run, then restored."""
  saved = memlimit.ARMS
  memlimit.ARMS = ARMS
  try:
    yield
  finally:
    memlimit.ARMS = saved


def sha256(path):
  return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def read_sessions(path):
  return [l.split('#')[0].strip() for l in pathlib.Path(path).read_text().splitlines() if l.split('#')[0].strip()]


def session_records(root, sessions, read):
  """memlimit.session_records for the listed sessions of label memlimit_ah; raises SystemExit on a failed check."""
  out, per_arm = [], {}
  for name in sessions:
    m = SESSION.match(name)
    if not m or not (root / name).is_dir():
      raise SystemExit(f'{name}: not a memlimit_ah session folder under {root}')
    for sj in sorted((root / name).glob('*/session.json')):
      s = json.loads(sj.read_text())
      read.append(sj)
      if s.get('label') != LABEL:
        raise SystemExit(f'{sj}: label {s.get("label")!r}, not {LABEL}')
      arm = CODE.get((m.group(1), s.get('config_name')))
      if arm is None:
        raise SystemExit(f'{sj}: {m.group(1)} with configuration {s.get("config_name")!r} is not arm A or H')
      per_arm.setdefault(arm, set()).add(name)
      for rj in sorted(sj.parent.glob('*/run.json')):
        r = json.loads(rj.read_text())
        read.append(rj)
        if not (r.get('af3_buckets') or r.get('bucket')):
          raise SystemExit(f'{rj}: no size')
        out.append((name, arm, (s.get('host') or {}).get('gce_zone'), r))
  bad = {a: sorted(v) for a, v in per_arm.items() if len(v) != 1}
  if bad:
    raise SystemExit(f'more than one session for an arm: {bad}')
  return out, sorted(per_arm)


def process_table(root, sessions, samples, read):
  prices = cost.load_prices()
  records, arms = session_records(root, sessions, read)
  rows = memlimit.process_rows(records, prices)
  with arms_a_h():
    for arm in arms:
      if pathlib.Path(samples[arm]).is_file():
        read.append(pathlib.Path(samples[arm]))
        ref = [r for r in memlimit.main_study_rows(prices, samples[arm]) if r['arm'] == arm]
        rows += [dict(r, source=SOURCE[arm]) for r in ref]
  t = pd.DataFrame(rows)
  if t.empty:
    return t
  t['arm'] = pd.Categorical(t.arm, ARMS, ordered=True)
  return t.sort_values(['arm', 'size']).reset_index(drop=True)


def main():
  ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
  ap.add_argument('--sessions', required=True)
  ap.add_argument('--out', required=True)
  ap.add_argument('--root', default=str(REPO / 'results' / 'af3'))
  args = ap.parse_args()
  out = pathlib.Path(args.out)
  if (out / 'summary.csv').exists():
    raise SystemExit(f'{out}/summary.csv exists; the output folder must be new')
  out.mkdir(parents=True, exist_ok=True)
  read = [pathlib.Path(args.sessions), REPO / 'cloud' / 'prices.csv']
  t = process_table(pathlib.Path(args.root), read_sessions(args.sessions), SAMPLES, read)
  with arms_a_h():
    s = memlimit.summary(t)
    t.to_csv(out / 'processes.csv', index=False)
    s.to_csv(out / 'summary.csv', index=False)
    memlimit.report(t, s, out)
  rep = (out / 'report.md').read_text().replace(
      "Section 10 rows of 2026-10-14. One row per AlphaFold3 process (7D5C, seed 1, padded with --buckets); the 1,024 "
      "rows are the main study's 7D5C process (fresh, r1, seed 1).",
      'Memory limit on A and H (label memlimit_ah), section 10 rows of 2026-11-10 (specification) and of its analysis; '
      'analysis/memlimit.py\'s logic through analysis/memlimit_ah.py. One row per AlphaFold3 process (7D5C, seed 1, '
      'padded with --buckets); the 1,024 rows are the 7D5C process of follow-up 3 (A) and follow-up 9 (H) (fresh, '
      'r1, seed 1). Prices: on demand, for H too (H ran on Spot).').replace(
      '\\* main study, bucket 1,024.', '\\* follow-up 3 (A) or 9 (H), bucket 1,024.')
  (out / 'report.md').write_text(rep)
  # Paths as the repository names them (results/ may be a symlink, so they are not resolved).
  rel = lambda p: os.path.relpath(os.path.abspath(p), REPO) if os.path.abspath(p).startswith(str(REPO) + os.sep) \
      else pathlib.Path(p).name  # noqa: E731
  pd.DataFrame([{'path': rel(p), 'sha256': sha256(p)} for p in dict.fromkeys(read)]).to_csv(
      out / 'inputs.tsv', sep='\t', index=False)
  print(rep)
  for r in s.itertuples():
    print(f'>> {r.arm}: {r.sizes_run} sizes run, largest completed {r.largest_completed_size}, stopping size '
          f'{r.stopping_size if r.stopping_size == r.stopping_size else "none"} {r.cause}', flush=True)
  return 0


if __name__ == '__main__':
  sys.exit(main())
