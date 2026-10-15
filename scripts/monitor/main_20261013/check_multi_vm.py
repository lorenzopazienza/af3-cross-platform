"""Monitor (records only): every main-study part with more than one session that created a VM.

    python3 scripts/monitor/main_20261013/check_multi_vm.py

Reads results/af3/.launcher/<session>/launch_record.txt (CREATED=1) and, when present,
results/af3/<session>/final_status.json, for sessions named <stamp>_<platform>_main_<arm>_<NN>. Prints each part
with two or more VM-creating sessions, with each session's job exit code and completion time, so that the duplicate
session rule (notes/analysis_plan.md section 10, 2026-10-13) can be applied at analysis time. Known relaunches
recorded in section 10 (main_gd_08) are labelled as such; anything else is flagged.
"""
import json
import pathlib
import re

REPO = pathlib.Path(__file__).resolve().parents[3]
KNOWN_RELAUNCH = {('l4', 'main_gd_08')}
NAME = re.compile(r'^(\d{8}T\d{6}Z)_(cpu|l4|v5e|v6e)_(main_[a-z0-9]+_\d+)$')
parts = {}
for rec in sorted((REPO / 'results' / 'af3' / '.launcher').glob('*/launch_record.txt')):
  m = NAME.match(rec.parent.name)
  if not m or 'CREATED=1' not in rec.read_text().split():
    continue
  st = REPO / 'results' / 'af3' / rec.parent.name / 'final_status.json'
  final = json.loads(st.read_text()) if st.is_file() else {}
  parts.setdefault((m.group(2), m.group(3)), []).append((rec.parent.name, final.get('exit_code'), final.get('finished_utc')))
multi = {k: v for k, v in parts.items() if len(v) > 1}
print(f'{len(parts)} main parts with a VM-creating session; {len(multi)} with more than one')
for (platform, plan), sessions in sorted(multi.items()):
  tag = 'known relaunch (section 10)' if (platform, plan) in KNOWN_RELAUNCH else 'FLAG: more than one VM-creating session'
  print(f'  {platform} {plan}: {tag}')
  for s, code, done in sessions:
    print(f'    {s}: job exit {code if code is not None else "none (no final status)"}, completed {done or "-"}')
