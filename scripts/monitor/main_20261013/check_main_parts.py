"""Main study, while it runs: checks 1 and 2 (2a, 2b) and provenance on the finished parts, from run records only.

    analysis/.venv/bin/python scripts/monitor/main_20261013/check_main_parts.py [OUT_DIR]

Reads only records: results/af3/.many/*/parts/*/{result,line} (which parts finished with launcher exit 0, and
their sessions), and per session launch_record.txt, launcher_events.txt, final_status.json, plan_runs.jsonl,
<run>/session.json and <run>/runs.jsonl. Never opens a model, *_confidences.json or *_ranking_scores.csv (no
outcome). The planned cells come from harness/run_plan.py's expand of the part's plan (as on the VM).

Per part:
  check 1  every planned (run, target) process has a record with exit code 0 (or a recorded failure, which fails
           the check), no unplanned process, every harness run in plan_runs.jsonl exit 0, final_status exit 0;
  2a       every fresh process started with an empty cache and compiled jit_apply_fn (no cache hit, compile
           seconds recorded);
  2b       (parts with a warm run) every warm process loaded jit_apply_fn from the cache with a key that its source
           fresh process (rep 1, same seed and target, same session) wrote;
  prov     label main; config as the plan; weights SHA-256 one value across all parts; launch commit in the
           expected set (EXPECTED_COMMITS) with 0 uncommitted files; provisioning standard.
Writes OUT_DIR/parts.csv and prints one line per part plus totals per arm. Bit-identity of the warm parts is
checked separately with harness/compare_runs.py --strict (crosscheck_warm.sh).
"""
import csv
import json
import pathlib
import re
import sys

REPO = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / 'harness'))
from run_plan import expand  # noqa: E402

# Commits from which main-study launches are expected: the commit that added the plans (501637f) and every later
# commit on this branch that changes nothing the launcher uploads to the VMs (af3_tpu, harness, targets, inputs,
# cloud, data), from git.
import subprocess  # noqa: E402
BASE = '501637fd5dcc0a285bf43d15a27d90cc1b2a8920'
UPLOADED = ['af3_tpu', 'harness', 'targets', 'inputs', 'cloud', 'data']


def expected_commits():
  later = subprocess.run(['git', '-C', str(REPO), 'rev-list', f'{BASE}..HEAD'], capture_output=True, text=True,
                         check=True).stdout.split()
  same = [c for c in later if not subprocess.run(['git', '-C', str(REPO), 'diff', '--name-only', BASE, c, '--'] + UPLOADED,
                                                 capture_output=True, text=True, check=True).stdout.strip()]
  return {BASE, *same}


EXPECTED_COMMITS = expected_commits()
CONFIG = {'v5': 'tpu_xla', 'v6': 'tpu_xla', 'g': 'l4_xla', 'gd': 'l4_xla_det', 'gt': 'l4_triton',
          'v6h': 'tpu_xla_highest', 'gh': 'l4_xla_highest', 'c': 'cpu_xla'}


def rec(path):
  out = {}
  for line in path.read_text().splitlines() if path.is_file() else []:
    if '=' in line:
      k, v = line.split('=', 1)
      out[k] = v
  return out


def finished_parts():
  """[(platform, plan, session)] of parts whose launcher exited 0, from the wrapper state."""
  out = []
  for d in sorted((REPO / 'results' / 'af3' / '.many').glob('2*/parts/*')):
    result = (d / 'result').read_text().strip() if (d / 'result').is_file() else ''
    m = re.match(r'ok: launcher exit 0, session (\S+)', result)
    if m:
      platform, plan = (d / 'line').read_text().split()[:2]
      out.append((platform, plan, m.group(1)))
  return out


def check_part(platform, plan, session):
  arm = pathlib.Path(plan).stem.split('_')[1]
  ses = REPO / 'results' / 'af3' / session
  desc = expand(REPO / plan, platform, REPO / 'inputs' / 'manifest.csv')
  lr = rec(ses / 'launch_record.txt')
  problems = {'1': [], '2a': [], '2b': [], 'prov': []}
  status = json.loads((ses / 'final_status.json').read_text()) if (ses / 'final_status.json').is_file() else {}
  if status.get('exit_code') != 0:
    problems['1'].append(f'final_status exit {status.get("exit_code")}')
  harness = {json.loads(l)['run']: json.loads(l).get('harness_exit_code')
             for l in (ses / 'plan_runs.jsonl').read_text().splitlines()} if (ses / 'plan_runs.jsonl').is_file() else {}
  procs, weights, written = {}, set(), {}
  n_fresh = n_warm = 0
  for run in desc['runs']:
    rdir = ses / run['run']
    if harness.get(run['run']) != 0:
      problems['1'].append(f'{run["run"]}: harness exit {harness.get(run["run"])}')
    sj = json.loads((rdir / 'session.json').read_text()) if (rdir / 'session.json').is_file() else {}
    if sj.get('label') != 'main' or sj.get('config_name') != CONFIG[arm]:
      problems['prov'].append(f'{run["run"]}: label {sj.get("label")} config {sj.get("config_name")}')
    weights.add((sj.get('weights') or {}).get('sha256'))
    records = [json.loads(l) for l in (rdir / 'runs.jsonl').read_text().splitlines()] if (rdir / 'runs.jsonl').is_file() else []
    seen = {r['pdb_id'] for r in records}
    for t in desc['targets']:
      if t not in seen:
        problems['1'].append(f'{run["run"]} {t}: no record')
    for r in records:
      t = r['pdb_id']
      procs[(run['run'], t)] = r.get('exit_code')
      if t not in desc['targets']:
        problems['1'].append(f'{run["run"]} {t}: not planned')
      if r.get('exit_code') != 0:
        problems['1'].append(f'{run["run"]} {t}: exit {r.get("exit_code")} {r.get("failure")}')
      c, cc = r.get('compile') or {}, r.get('compile_cache') or {}
      weights.add(r.get('weights_sha256'))
      if run['cache'] == 'fresh':
        n_fresh += 1
        if (cc.get('before') or {}).get('files') != 0 or c.get('model_cache_hit') or not c.get('model_seconds'):
          problems['2a'].append(f'{run["run"]} {t}')
        if run.get('save_cache'):
          written[(t, run['seed'])] = set(cc.get('model_entries_written') or [])
  for run in desc['runs']:
    if run['cache'] != 'warm':
      continue
    src = next(r for r in desc['runs'] if r['run'] == run['source_run'])
    for r in [json.loads(l) for l in (ses / run['run'] / 'runs.jsonl').read_text().splitlines()]:
      n_warm += 1
      c = r.get('compile') or {}
      if not (c.get('model_cache_hit') and c.get('model_cache_key') in written.get((r['pdb_id'], src['seed']), set())):
        problems['2b'].append(f'{run["run"]} {r["pdb_id"]}: key {c.get("model_cache_key")}')
  if lr.get('COMMIT') not in EXPECTED_COMMITS or lr.get('DIRTY_FILES') != '0':
    problems['prov'].append(f'commit {lr.get("COMMIT")} dirty {lr.get("DIRTY_FILES")}')
  if lr.get('PROVISIONING') != 'standard' or lr.get('PLAN') != plan or lr.get('PLATFORM') != platform:
    problems['prov'].append(f'record {lr.get("PLATFORM")} {lr.get("PLAN")} {lr.get("PROVISIONING")}')
  planned = len(desc['runs']) * len(desc['targets'])
  return {'part': pathlib.Path(plan).stem, 'arm': arm, 'platform': platform, 'session': session,
          'zone': lr.get('ZONE'), 'planned': planned, 'recorded': len(procs),
          'exit0': sum(1 for v in procs.values() if v == 0), 'fresh': n_fresh, 'warm': n_warm,
          'check1': 'pass' if not problems['1'] and len(procs) == planned else 'FAIL: ' + '; '.join(problems['1'])[:300],
          'check2a': 'pass' if not problems['2a'] else 'FAIL: ' + '; '.join(problems['2a'])[:300],
          'check2b': ('pass' if not problems['2b'] else 'FAIL: ' + '; '.join(problems['2b'])[:300]) if n_warm else '-',
          'provenance': 'pass' if not problems['prov'] else 'FAIL: ' + '; '.join(problems['prov'])[:300],
          'weights': ';'.join(sorted(w for w in weights if w))}


def main():
  out = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else REPO / 'results' / 'analysis' / 'main_monitor_20261013'
  rows = [check_part(*p) for p in finished_parts()]
  shas = {w for r in rows for w in r['weights'].split(';')}
  for r in rows:
    if len(shas) != 1:
      r['provenance'] = (r['provenance'] + '; weights differ across parts').replace('pass; ', 'FAIL: ')
  with open(out / 'parts.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
  for r in sorted(rows, key=lambda r: r['part']):
    print(f'{r["part"]:12} {r["session"]:36} {r["zone"] or "-":15} procs {r["exit0"]:3}/{r["planned"]:3} | '
          f'1 {r["check1"][:4]} | 2a {r["check2a"][:4]} ({r["fresh"]}) | 2b {r["check2b"][:4]} ({r["warm"]}) | '
          f'prov {r["provenance"][:4]}')
  print('weights SHA-256 values across all parts:', len(shas))
  arms = {}
  for r in rows:
    a = arms.setdefault(r['arm'], [0, 0, 0, 0, 0])
    a[0] += 1; a[1] += r['planned']; a[2] += r['exit0']; a[3] += r['fresh']; a[4] += r['warm']
  for arm, (n, planned, ok, fresh, warm) in sorted(arms.items()):
    print(f'arm {arm:4}: {n} part(s) finished, {ok}/{planned} processes exit 0, {fresh} fresh, {warm} warm')
  bad = [r for r in rows if 'FAIL' in r['check1'] + r['check2a'] + r['check2b'] + r['provenance']]
  for r in bad:
    print('!!', r['part'], r['check1'], r['check2a'], r['check2b'], r['provenance'])
  print('ALL PASS' if not bad else f'{len(bad)} PART(S) FAIL')
  return 1 if bad else 0


if __name__ == '__main__':
  sys.exit(main())
