"""Cross-session check of notes/analysis_plan.md section 10 (row of 2026-10-13), draft. Engineering check only.

    analysis/.venv/bin/python scripts/monitor/main_20261013/cross_session_check.py plan
    analysis/.venv/bin/python scripts/monitor/main_20261013/cross_session_check.py run --runs-finished SCRATCH

For the 10 pilot targets, each pilot2 r1 run is compared with the main-study r1 run of the same condition, target
and seed, with harness/compare_runs.py --strict (every output file byte for byte, only the mmCIF timestamp line
masked): seeds 1 and 2 for V5, V6, G, Gd, V6h and Gh, and C on its 6 targets (512 tokens or fewer). Expected
bit-identical: V5, V6, V6h, Gd and C; G and Gh are reported descriptively (default autotuning varies between
compiles; G added to the row on 2026-10-13, commit 0dec27a, before any comparison).
The result is identical or not per (condition, target, seed); it enters no contrast.

plan: pairs the runs from records only (session.json label and config, runs.jsonl exit codes; no output is
      opened) and lists, per (condition, target, seed), the pilot2 and main sessions found, or what is missing.
run:  only after the main runs finish (--runs-finished): read-only copies from the results bucket of the files a
      light fetch left there (full *_confidences.json and *_data.json) for the target folders compared, then one
      scratch session per pair in SCRATCH holding only the two runs' target folders (symlinks), compare_runs
      --strict on it, and only its strict verdict kept: cross_session.tsv and cross_session_summary.txt here.
Pilot2 sessions: results/analysis/pilot2_20261012/sessions_pilot2.txt (exclusions applied there). Main sessions:
every results/af3 session with label main whose run has the target with exit code 0; two candidates for one
pair are reported, never chosen between.
"""
import csv
import json
import pathlib
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parents[3]
AF3 = REPO / 'results' / 'af3'
HERE = REPO / 'results' / 'analysis' / 'main_monitor_20261013'   # outputs
BUCKET = 'gs://af2-tpu-benchmark-af3-results/af3'
LIGHT = r'^(?!.*(_data\.json|(?<!_summary)_confidences\.json)$)'
CONDITIONS = {  # condition: (platform in the session name, config)
    'V5': ('v5e', 'tpu_xla'), 'V6': ('v6e', 'tpu_xla'), 'G': ('l4', 'l4_xla'), 'Gd': ('l4', 'l4_xla_det'),
    'V6h': ('v6e', 'tpu_xla_highest'), 'Gh': ('l4', 'l4_xla_highest'), 'C': ('cpu', 'cpu_xla')}
EXPECT_IDENTICAL = {'V5', 'V6', 'Gd', 'V6h', 'C'}
SEEDS = (1, 2)


def targets(cond):
  rows = list(csv.DictReader(open(REPO / 'inputs' / 'manifest.csv')))
  pilot = sorted(r['pdb_id'] for r in rows if r['pilot'] == 'yes')
  return [t for t in pilot if cond != 'C' or int(next(r for r in rows if r['pdb_id'] == t)['num_tokens']) <= 512]


def candidates(sessions, label, cond, target, seed):
  """Sessions whose r1 run of this seed, with this condition's config, holds the target with exit code 0."""
  platform, config = CONDITIONS[cond]
  run = f'{config}_rep1_seed{seed}_fresh'
  out = []
  for s in sessions:
    if f'_{platform}_' not in s.name:
      continue
    rd = s / run
    sj = rd / 'session.json'
    if not (sj.is_file() and (rd / 'runs.jsonl').is_file()):
      continue
    meta = json.loads(sj.read_text())
    if meta.get('label') != label or meta.get('config_name') != config:
      continue
    recs = [json.loads(l) for l in (rd / 'runs.jsonl').read_text().splitlines()]
    if any(r['pdb_id'] == target and r.get('exit_code') == 0 for r in recs):
      out.append((s.name, run))
  return out


def pairs():
  pilot = [AF3 / s for s in (REPO / 'results/analysis/pilot2_20261012/sessions_pilot2.txt').read_text().split()]
  main = sorted(p for p in AF3.glob('2026*_main_*') if p.is_dir())
  for cond in CONDITIONS:
    for t in targets(cond):
      for seed in SEEDS:
        yield cond, t, seed, candidates(pilot, 'pilot2', cond, t, seed), candidates(main, 'main', cond, t, seed)


def plan():
  n = ok = 0
  for cond, t, seed, p, m in pairs():
    n += 1
    status = 'ready' if len(p) == 1 and len(m) == 1 else \
        ('no pilot2 run' if not p else 'several pilot2 runs' if len(p) > 1 else
         'no main run yet' if not m else 'several main runs: ' + ', '.join(s for s, _ in m))
    ok += status == 'ready'
    print(f'{cond:4} {t} seed {seed}: pilot2 {p[0][0] if len(p) == 1 else "-":36} main {m[0][0] if len(m) == 1 else "-":36} {status}')
  print(f'{ok}/{n} pairs ready')


def run(scratch):
  scratch = pathlib.Path(scratch)
  rows = []
  for cond, t, seed, p, m in pairs():
    if len(p) != 1 or len(m) != 1:
      rows.append([cond, t, seed, p[0][0] if p else '', m[0][0] if m else '', 'not compared', '', '', ''])
      continue
    work = scratch / f'{cond}_{t}_seed{seed}'
    subprocess.run(['rm', '-rf', str(work)], check=True)
    for (session, run_name), created in zip((p[0], m[0]), ('2000-01-01T00:00:00Z', '2000-01-02T00:00:00Z')):
      src = AF3 / session / run_name / t
      subprocess.run(['gcloud', 'storage', 'rsync', '--recursive', f'--exclude={LIGHT}',
                      f'{BUCKET}/{session}/{run_name}/{t}', str(src)], check=True, capture_output=True)
      dst = work / f'{session}__{run_name}'
      dst.mkdir(parents=True)
      (dst / 'session.json').write_text(json.dumps({'created_utc': created}))   # pilot2 first: the reference
      (dst / t).symlink_to(src)
    subprocess.run([sys.executable, str(REPO / 'harness' / 'compare_runs.py'), '--session', str(work), '--strict'],
                   check=True, capture_output=True)
    comps = json.loads((work / 'compare_runs.json').read_text())['comparisons']
    s = comps[0]['strict'] if len(comps) == 1 else {'status': f'{len(comps)} comparisons', 'identical': [], 'files': 0,
                                                       'differ': [], 'missing': []}
    rows.append([cond, t, seed, p[0][0], m[0][0], s['status'], f'{len(s["identical"])}/{s["files"]}',
                 len(s['differ']), len(s['missing'])])
  with open(HERE / 'cross_session.tsv', 'w') as f:
    f.write('condition\ttarget\tseed\tpilot2_session\tmain_session\tstrict\tfiles_identical\tfiles_differ\tfiles_missing\n')
    f.writelines('\t'.join(str(x) for x in r) + '\n' for r in rows)
  lines = []
  for cond in CONDITIONS:
    r = [x for x in rows if x[0] == cond]
    ident = sum(x[5] == 'identical' for x in r)
    exp = 'expected identical' if cond in EXPECT_IDENTICAL else 'descriptive'
    verdict = '' if cond not in EXPECT_IDENTICAL else ('as expected' if ident == len(r) else 'MISMATCH: investigate')
    lines.append(f'{cond:4} ({exp}): {ident}/{len(r)} (target, seed) bit-identical {verdict}')
  (HERE / 'cross_session_summary.txt').write_text('\n'.join(lines) + '\n')
  print('\n'.join(lines))


if __name__ == '__main__':
  if len(sys.argv) >= 2 and sys.argv[1] == 'plan':
    plan()
  elif len(sys.argv) == 4 and sys.argv[1] == 'run' and sys.argv[2] == '--runs-finished':
    run(sys.argv[3])
  else:
    sys.exit(__doc__)
