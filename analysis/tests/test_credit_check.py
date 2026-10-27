"""scripts/monitor/credit_check.sh (follow-up 6 budget rule): a session's VM time is counted once, although its
launcher_events.txt is in both results/af3/<session>/ and results/af3/.launcher/<session>/."""
import pathlib
import shutil
import subprocess

REPO = pathlib.Path(__file__).resolve().parents[2]
EVENTS = ('2026-10-30T00:00:00Z  vm: af3-run-l4-x running in europe-west4-a\n'
          '2026-10-30T01:00:00Z  delete: done\n')


def test_a_session_in_both_folders_is_counted_once(tmp_path):
  (tmp_path / 'harness' / 'plans' / 'msa').mkdir(parents=True)
  shutil.copy(REPO / 'harness' / 'plans' / 'msa' / 'parts.csv', tmp_path / 'harness' / 'plans' / 'msa' / 'parts.csv')
  for d in ('results/af3', 'results/af3/.launcher'):
    s = tmp_path / d / '20261030T000000Z_l4_msa_go_r1_01'
    s.mkdir(parents=True)
    s.joinpath('launcher_events.txt').write_text(EVENTS)
  out = subprocess.run(['bash', str(REPO / 'scripts' / 'monitor' / 'credit_check.sh'), 'v6o', '500'], cwd=tmp_path,
                       capture_output=True, text=True).stdout
  assert 'from 1 session record(s): inference arms 0.90 USD' in out   # 1 h x 0.897 USD/h, not 2 h
