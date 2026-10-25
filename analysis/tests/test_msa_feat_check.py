"""scripts/monitor/msa_feat_check.py (follow-up 6): each part's featurisation against the allowance its plan was sized
with, on synthetic session records; it warns and changes nothing."""
import importlib.util
import json
import pathlib

import pytest

REPO = pathlib.Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('msa_feat_check', REPO / 'scripts' / 'monitor' / 'msa_feat_check.py')
fc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fc)
PART = {'part': 'msa_go_r1_01', 'arm': 'Go_r1', 'main_arm': 'G', 'processes': '20', 'expected_job_h': '3.82',
        'deadline_min': '307'}
MEDIANS = {'G': {256: 10.0, 1024: 30.0}}


def session(tmp_path, feats, job_min=None, name='20261030T000000Z_l4_msa_go_r1_01'):
  s = tmp_path / name
  run = s / 'l4_xla_rep1_seed1_fresh'
  run.mkdir(parents=True)
  run.joinpath('runs.jsonl').write_text(''.join(json.dumps({'pdb_id': t, 'bucket': b, 'exit_code': 0,
                                                            'featurisation_seconds': f}) + '\n' for t, b, f in feats))
  if job_min is not None:
    s.joinpath('launcher_events.txt').write_text(
        f'2026-10-30T00:00:00Z  job: started on the VM\n2026-10-30T{job_min // 60:02d}:{job_min % 60:02d}:00Z  job: finished, exit 0\n')
  return s


def test_within_the_allowance_no_warning(tmp_path):
  lines, warnings = fc.check(session(tmp_path, [('T1', 256, 15.0), ('T2', 1024, 50.0)], job_min=200), {'msa_go_r1_01': PART},
                             MEDIANS, {})
  assert warnings == []
  text = '\n'.join(lines)
  assert 'observed 65 s, allowance 80 s' in text and 'observed factor 1.62' in text and 'processes done: 2 of 20' in text
  assert 'above their own allowance: 0' in text


def test_above_the_allowance_warns_and_names_the_processes(tmp_path):
  lines, warnings = fc.check(session(tmp_path, [('T1', 256, 30.0), ('T2', 1024, 70.0)]), {'msa_go_r1_01': PART}, MEDIANS, {})
  assert len(warnings) == 1 and 'exceeds the plan\'s allowance 80 s' in warnings[0] and 'No plan' in warnings[0]
  assert 'above their own allowance: 2' in '\n'.join(lines) and 'job: not finished' in '\n'.join(lines)


def test_a_job_longer_than_expected_warns(tmp_path):
  _, warnings = fc.check(session(tmp_path, [('T1', 256, 12.0)], job_min=240), {'msa_go_r1_01': PART}, MEDIANS, {})
  assert len(warnings) == 1 and 'more than the expected 229 min' in warnings[0]


def test_the_plans_record_their_factor_and_a_foreign_session_is_refused(tmp_path):
  assert fc.factor_of((REPO / 'harness' / 'plans' / 'msa' / 'msa_go_r1_01.yaml').read_text()) == 2.0
  with pytest.raises(SystemExit, match='not a session of a follow-up 6 part'):
    fc.check(session(tmp_path, [], name='20261030T000000Z_l4_main_g_01'), {'msa_go_r1_01': PART}, MEDIANS, {})
