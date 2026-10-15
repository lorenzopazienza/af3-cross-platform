"""Follow-up launches (wrapper run 20261014T223755Z: follow-up 2 embed and follow-up 1 ppi, notes/analysis_plan.md
section 10, rows of 2026-10-14): check 1 and provenance on the finished parts, from run records only.

    analysis/.venv/bin/python scripts/monitor/followups_20261015/check_followup_parts.py [OUT_DIR]

Reads only records: results/af3/.many/20261014T223755Z/parts/*/{result,line} (parts whose launcher exited 0 and
their sessions), and per session launch_record.txt, final_status.json, plan_runs.jsonl, <run>/session.json and
<run>/runs.jsonl, plus the embeddings files' sizes on disk. Never opens a model, a confidences file, a ranking
file or an embeddings array (no outcome). The planned cells come from harness/run_plan.py's expand of the part's
plan, with the plan's own manifest (as on the VM).

Per part:
  check 1  every planned (run, target) process has a record with exit code 0, no unplanned process, every harness
           run in plan_runs.jsonl exit 0, final_status exit 0;
  prov     label (embed or ppi) and config as the plan; weights SHA-256 one value across all parts and equal to
           the main study's; launch commit in the expected set with 0 uncommitted files; provisioning standard;
           manifest as the plan (session.json args.manifest; ppi: inputs/ppi/manifest.csv) and every process's
           input SHA-256 equal to that manifest's input_sha256;
           embed: save_embeddings on in session.json and every process, one embeddings file per process listed
           in run.json and present locally with the recorded size (fetched); ppi: save_embeddings off.
Writes OUT_DIR/parts.csv and prints one line per part.
"""
import csv
import json
import pathlib
import re
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / 'harness'))
from run_plan import expand  # noqa: E402

WRAPPER = REPO / 'results' / 'af3' / '.many' / '20261014T223755Z'
# The commit that added the follow-up plans and inputs, and every later commit that changes nothing the launcher
# uploads to the VMs.
BASE = '104fb275bd8492e48f94dd7c2f4a6e3da25690af'
UPLOADED = ['af3_tpu', 'harness', 'targets', 'inputs', 'cloud', 'data']
CONFIG = {'v5': 'tpu_xla', 'v6': 'tpu_xla', 'g': 'l4_xla', 'gd': 'l4_xla_det'}
MAIN_PARTS = REPO / 'results' / 'analysis' / 'main_monitor_20261013' / 'parts.csv'


def expected_commits():
  later = subprocess.run(['git', '-C', str(REPO), 'rev-list', f'{BASE}..HEAD'], capture_output=True, text=True,
                         check=True).stdout.split()
  same = [c for c in later if not subprocess.run(['git', '-C', str(REPO), 'diff', '--name-only', BASE, c, '--'] + UPLOADED,
                                                 capture_output=True, text=True, check=True).stdout.strip()]
  return {BASE, *same}


def rec(path):
  out = {}
  for line in path.read_text().splitlines() if path.is_file() else []:
    if '=' in line:
      k, v = line.split('=', 1)
      out[k] = v
  return out


def parts():
  """[(platform, plan, session or None, result text)] for every part of the wrapper run."""
  out = []
  for d in sorted((WRAPPER / 'parts').glob('*')):
    result = (d / 'result').read_text().strip() if (d / 'result').is_file() else ''
    platform, plan = (d / 'line').read_text().split()[:2]
    m = re.match(r'ok: launcher exit 0, session (\S+)', result)
    out.append((platform, plan, m.group(1) if m else None, result))
  return out


def check_part(platform, plan, session, commits):
  kind, arm = pathlib.Path(plan).stem.split('_')[:2]
  ses = REPO / 'results' / 'af3' / session
  desc = expand(REPO / plan, platform, REPO / 'inputs' / 'manifest.csv')
  manifest_rel = desc.get('manifest') or 'inputs/manifest.csv'
  with open(REPO / manifest_rel) as f:
    manifest = {r['pdb_id']: r for r in csv.DictReader(f)}
  lr = rec(ses / 'launch_record.txt')
  p1, prov = [], []
  status = json.loads((ses / 'final_status.json').read_text()) if (ses / 'final_status.json').is_file() else {}
  if status.get('exit_code') != 0:
    p1.append(f'final_status exit {status.get("exit_code")}')
  harness = {json.loads(l)['run']: json.loads(l).get('harness_exit_code')
             for l in (ses / 'plan_runs.jsonl').read_text().splitlines()} if (ses / 'plan_runs.jsonl').is_file() else {}
  procs, weights, emb_files, emb_bytes = {}, set(), 0, 0
  for run in desc['runs']:
    rdir = ses / run['run']
    if harness.get(run['run']) != 0:
      p1.append(f'{run["run"]}: harness exit {harness.get(run["run"])}')
    sj = json.loads((rdir / 'session.json').read_text()) if (rdir / 'session.json').is_file() else {}
    if sj.get('label') != kind or sj.get('config_name') != CONFIG[arm]:
      prov.append(f'{run["run"]}: label {sj.get("label")} config {sj.get("config_name")}')
    if (sj.get('args') or {}).get('manifest') != manifest_rel:
      prov.append(f'{run["run"]}: manifest {(sj.get("args") or {}).get("manifest")}')
    if bool(sj.get('save_embeddings')) != (kind == 'embed'):
      prov.append(f'{run["run"]}: save_embeddings {sj.get("save_embeddings")}')
    weights.add((sj.get('weights') or {}).get('sha256'))
    records = [json.loads(l) for l in (rdir / 'runs.jsonl').read_text().splitlines()] if (rdir / 'runs.jsonl').is_file() else []
    seen = {r['pdb_id'] for r in records}
    for t in desc['targets']:
      if t not in seen:
        p1.append(f'{run["run"]} {t}: no record')
    for r in records:
      t = r['pdb_id']
      procs[(run['run'], t)] = r.get('exit_code')
      weights.add(r.get('weights_sha256'))
      if t not in desc['targets']:
        p1.append(f'{run["run"]} {t}: not planned')
      if r.get('exit_code') != 0:
        p1.append(f'{run["run"]} {t}: exit {r.get("exit_code")} {r.get("failure")}')
      if t not in manifest or r.get('input_sha256') != manifest[t]['input_sha256']:
        prov.append(f'{run["run"]} {t}: input SHA-256 not the manifest\'s')
      files = r.get('embedding_files') or []
      if kind == 'embed':
        ok = bool(r.get('save_embeddings')) and len(files) == 1 and all(
            (rdir / t / f['path']).is_file() and (rdir / t / f['path']).stat().st_size == f['bytes'] for f in files)
        if not ok:
          prov.append(f'{run["run"]} {t}: embeddings {files}')
        emb_files += len(files)
        emb_bytes += sum(f['bytes'] for f in files)
      elif r.get('save_embeddings') or files:
        prov.append(f'{run["run"]} {t}: embeddings saved')
  if lr.get('COMMIT') not in commits or lr.get('DIRTY_FILES') != '0':
    prov.append(f'commit {lr.get("COMMIT")} dirty {lr.get("DIRTY_FILES")}')
  if lr.get('PROVISIONING') != 'standard' or lr.get('PLAN') != plan or lr.get('PLATFORM') != platform:
    prov.append(f'record {lr.get("PLATFORM")} {lr.get("PLAN")} {lr.get("PROVISIONING")}')
  planned = len(desc['runs']) * len(desc['targets'])
  return {'part': pathlib.Path(plan).stem, 'kind': kind, 'arm': arm, 'platform': platform, 'session': session,
          'zone': lr.get('ZONE'), 'commit': (lr.get('COMMIT') or '')[:7], 'dirty_files': lr.get('DIRTY_FILES'),
          'manifest': manifest_rel, 'planned': planned, 'recorded': len(procs),
          'exit0': sum(1 for v in procs.values() if v == 0), 'embedding_files': emb_files,
          'embedding_mb': round(emb_bytes / 1e6, 1),
          'check1': 'pass' if not p1 and len(procs) == planned else 'FAIL: ' + '; '.join(p1)[:300],
          'provenance': 'pass' if not prov else 'FAIL: ' + '; '.join(prov)[:300],
          'weights': ';'.join(sorted(w for w in weights if w))}


def main():
  out = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else REPO / 'results' / 'analysis' / 'followups_20261015'
  commits = expected_commits()
  rows, pending = [], []
  for platform, plan, session, result in parts():
    if session:
      rows.append(check_part(platform, plan, session, commits))
    else:
      pending.append((pathlib.Path(plan).stem, result))
  shas = {w for r in rows for w in r['weights'].split(';')}
  main_shas = {r['weights'] for r in csv.DictReader(open(MAIN_PARTS))} if MAIN_PARTS.is_file() else set()
  for r in rows:
    if len(shas) != 1 or (main_shas and shas != main_shas):
      r['provenance'] = (r['provenance'] + '; weights differ across parts or from the main study').replace('pass; ', 'FAIL: ')
  with open(out / 'parts.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n')
    w.writeheader()
    w.writerows(rows)
  for r in sorted(rows, key=lambda r: (r['kind'], r['part'])):
    print(f'{r["part"]:10} {r["session"]:34} {r["zone"] or "-":15} {r["commit"]} dirty {r["dirty_files"]} '
          f'{r["manifest"]:24} procs {r["exit0"]:3}/{r["planned"]:3} | 1 {r["check1"][:4]} | prov {r["provenance"][:4]}'
          + (f' | npz {r["embedding_files"]} ({r["embedding_mb"]} MB)' if r['kind'] == 'embed' else ''))
  print(f'weights SHA-256 values across finished parts: {len(shas)} ({";".join(sorted(shas))[:16]}...); '
        f'same as the main study: {shas == main_shas}')
  for part, result in pending:
    print(f'not finished: {part}: {result}')
  bad = [r for r in rows if 'FAIL' in r['check1'] + r['provenance']]
  for r in bad:
    print('!!', r['part'], r['check1'], r['provenance'])
  print('ALL FINISHED PARTS PASS' if not bad else f'{len(bad)} PART(S) FAIL')
  return 1 if bad else 0


if __name__ == '__main__':
  sys.exit(main())
