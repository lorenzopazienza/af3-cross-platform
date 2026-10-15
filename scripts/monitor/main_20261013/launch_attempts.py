"""Every main-study and GT smoke launch attempt, from the wrapper logs and the launcher records (no outputs read).

    python3 scripts/monitor/main_20261013/launch_attempts.py > attempts.tsv
"""
import pathlib
import re

REPO = pathlib.Path(__file__).resolve().parents[3]
L = REPO / 'results' / 'af3' / '.launcher'


def rec(path):
  out = {}
  for line in path.read_text().splitlines() if path.is_file() else []:
    if '=' in line:
      k, v = line.split('=', 1)
      out[k] = v
  return out


def attempt(session, log_text):
  r = rec(L / session / 'launch_record.txt')
  ev = (L / session / 'launcher_events.txt').read_text() if (L / session / 'launcher_events.txt').is_file() else ''
  tried = re.findall(r'Tried, in order:\s*(.*)', log_text)
  quota = sorted(set(re.findall(r'zone ([a-z0-9-]+): QUOTA_EXCEEDED', log_text)))
  stock = len(re.findall(r'STOCKOUT|does not have enough resources', log_text))
  exit_ = re.findall(r'launcher: exit code (\d+)', ev) or re.findall(r'^>> exit code (\d+)', log_text, re.M)
  job = re.findall(r'job: finished, exit code (\d+)', ev)
  return {'session': session, 'created': r.get('CREATED') == '1', 'zone': r.get('ZONE', '') if r.get('CREATED') == '1' else '',
          'launched_utc': r.get('LAUNCHED_UTC', ''), 'job_exit': job[-1] if job else '', 'launcher_exit': exit_[-1] if exit_ else 'running',
          'zones_tried': len(tried[-1].split()) if tried else '', 'capacity_errors': stock, 'quota_zones': ','.join(quota)}


rows = []
for log in sorted((REPO / 'results' / 'af3' / '.many').glob('2*/*.log')):
  text = log.read_text(errors='replace')
  ses = re.findall(r'^   session: results/af3/([^/]+)/', text, re.M)
  if ses:
    rows.append((log.parent.name, log.stem.split('.')[0], log.stem.split('.')[1] if '.' in log.stem else 'attempt1',
                 attempt(ses[-1], text)))
for d in sorted(L.glob('*_l4_gt_smoke')):
  text = '\n'.join(p.read_text(errors='replace') for p in d.glob('*.txt'))
  rows.append(('direct', 'gt_smoke', '', attempt(d.name, text)))
print('wrapper\tpart\tattempt\tsession\tlaunched_utc\tcreated\tzone\tjob_exit\tlauncher_exit\tzones_tried\tcapacity_errors\tquota_zones')
for w, part, att, a in rows:
  print('\t'.join(str(x) for x in (w, part, att, a['session'], a['launched_utc'], a['created'], a['zone'], a['job_exit'],
                                   a['launcher_exit'], a['zones_tried'], a['capacity_errors'], a['quota_zones'])))
