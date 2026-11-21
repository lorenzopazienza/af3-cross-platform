"""Follow-ups 17 to 21 (labels dtp, tri, rcf, aut, xvm): synthetic tests of the budget rule before each wrapper, the
gate of part B, the stop rule before each part (scripts/monitor/prep_budget.py and the guarded launcher
scripts/monitor/prep_guarded_launch.sh), the fallback allowance of the on-demand relaunch after one preemption and
the caps of the specification rows. Fake session folders only; no cloud command."""
import csv
import os
import pathlib
import subprocess
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'scripts' / 'monitor'))
import prep_budget as pb  # noqa: E402

LABELS = ('dtp', 'tri', 'rcf', 'aut', 'xvm')
# caps written in the specification rows (item g), after the fallback rule of 2026-11-21
CAPS = {'dtp': 590, 'tri': 340, 'rcf': 275, 'aut': 115, 'xvm': 170}


def parts(label):
  return list(csv.DictReader(open(REPO / 'harness' / 'plans' / f'{label}_parts.csv')))


def budget(args, tmp_path, now='2026-12-01T08:00:00Z', marker=None):
  env = dict(os.environ, NOW=now, RESULTS_AF3=str(tmp_path / 'af3'))
  if marker:
    env['PREP_MARKER'] = str(marker)
  else:
    env.pop('PREP_MARKER', None)
    env['PREP_MARKER'] = str(tmp_path / 'no_marker.txt')
  return subprocess.run([sys.executable, str(REPO / 'scripts' / 'monitor' / 'prep_budget.py')] + args,
                        capture_output=True, text=True, cwd=REPO, env=env)


def session(tmp_path, platform, part, start, end=None, code='0', prov='spot'):
  s = tmp_path / 'af3' / f'{start.replace("-", "").replace(":", "")}_{platform}_{part}'
  s.mkdir(parents=True)
  ev = f'{start} launch: session\n{start} vm: running\n'
  if end:
    ev += f'{end} delete: done\n{end} launcher: exit code {code}\n'
  (s / 'launcher_events.txt').write_text(ev)
  (s / 'launch_record.txt').write_text(f'PROVISIONING={prov}\n')
  return s


def part_a_done(tmp_path, label, start='2026-12-01T08:00:00Z', end='2026-12-01T08:10:00Z'):
  for p in parts(label):
    if p['row_part'] == 'A':
      session(tmp_path, p['platform'], p['part'], start, end=end, prov=p['provisioning'])


# ---------------------------------------------------------------- caps and fallback allowance

@pytest.mark.parametrize('label', LABELS)
def test_caps_are_the_rule_on_the_worst_case_with_fallback(label):
  assert pb.LABELS[label]['cap'] == CAPS[label]
  assert pb.cap_of(parts(label)) == CAPS[label]


def test_fallback_allowance_is_on_demand_except_on_the_h100():
  assert pb.OD_FALLBACK == ('l4', 'a100', 'v6e')
  l4 = next(p for p in parts('dtp') if p['platform'] == 'l4')
  h = next(p for p in parts('tri') if p['platform'] == 'h100')
  assert pb.worst_fb(l4) == pytest.approx(float(l4['worst_usd_planned']) + 0.75 * float(l4['worst_usd_on_demand']))
  assert pb.worst_fb(h) == pytest.approx(1.25 * float(h['worst_usd_planned']))          # Spot again: no H100 quota
  assert pb.realistic_fb(h) == pytest.approx(float(h['expected_usd_planned']) * (1 - 0.05) + 0.1 * float(h['expected_usd_planned']))
  for label in LABELS:
    for p in parts(label):
      assert float(p['expected_usd_planned']) <= pb.realistic_fb(p) <= pb.worst_fb(p) + 1e-9


def test_costs_table_lists_every_part_of_every_row():
  r = subprocess.run([sys.executable, str(REPO / 'scripts' / 'monitor' / 'prep_budget.py'), 'costs', 'all'],
                     capture_output=True, text=True, cwd=REPO)
  assert r.returncode == 0
  rows = [l for l in r.stdout.splitlines() if l.startswith('| 1') or l.startswith('| 2')]
  assert len(rows) == 8 and all(f'(rule {pb.LABELS[l.split("|")[2].strip()]["cap"]:.0f})' in l for l in rows)


# ---------------------------------------------------------------- budget rule before each wrapper

def test_check_floor_dates_and_usage(tmp_path):
  (tmp_path / 'af3').mkdir()
  r = budget(['check', 'dtp', 'v6e', '1000'], tmp_path)
  assert r.returncode == 0 and r.stdout.strip().endswith('GO'), r.stdout + r.stderr
  w = sum(pb.worst_fb(p) for p in parts('dtp') if p['platform'] == 'l4')
  r = budget(['check', 'dtp', 'l4', f'{w + 20:.2f}'], tmp_path)                       # floor 25 not kept
  assert r.returncode == 1 and '< 25 USD' in r.stdout
  r = budget(['check', 'xvm', 'l4', '1000'], tmp_path, now='2026-12-07T00:00:00Z')
  assert r.returncode == 1 and 'launch after 2026-12-06' in r.stdout
  r = budget(['check', 'aut', 'l4', '1000'], tmp_path, now='2026-12-08T20:00:00Z')
  assert r.returncode == 1 and 'launch after 2026-12-06' in r.stdout
  assert budget(['check', 'tri', 'l4', '1000'], tmp_path).returncode == 2              # no L4 wrapper in tri
  assert budget(['check', 'tri', 'h100', 'x'], tmp_path).returncode == 2


def test_expected_end_after_finish_by_is_refused(tmp_path, monkeypatch, capsys):
  (tmp_path / 'af3').mkdir()
  monkeypatch.setenv('NOW', '2026-12-06T23:00:00Z')
  monkeypatch.setenv('RESULTS_AF3', str(tmp_path / 'af3'))
  monkeypatch.setenv('PREP_MARKER', str(tmp_path / 'm.txt'))
  monkeypatch.setitem(pb.LABELS['aut'], 'slots', {'l4': 1})                    # one launcher: about 56 h
  assert pb.check('aut', 'l4', 1000.0) == 1
  assert 'cannot finish by 2026-12-08' in capsys.readouterr().out
  monkeypatch.setitem(pb.LABELS['aut'], 'slots', {'l4': 24})
  assert pb.check('aut', 'l4', 1000.0) == 0


def test_running_wrappers_of_other_follow_ups_are_subtracted(tmp_path):
  session(tmp_path, 'h100', 'tri_ht_01', '2026-12-01T07:00:00Z')                       # running, no end
  r = budget(['check', 'dtp', 'l4', '1000'], tmp_path)
  assert r.returncode == 0 and 'running wrapper tri h100' in r.stdout, r.stdout
  need = sum(pb.worst_fb(p) for p in parts('tri') if p['platform'] == 'h100')
  w = sum(pb.worst_fb(p) for p in parts('dtp') if p['platform'] == 'l4')
  r = budget(['check', 'dtp', 'l4', f'{need + w + 10:.2f}'], tmp_path)                 # enough without tri only
  assert r.returncode == 1 and '< 25 USD' in r.stdout


@pytest.mark.parametrize('label,wrapper', [('dtp', 'a100'), ('dtp', 'h100'), ('tri', 'a100'), ('xvm', 'a100')])
def test_gate_of_part_b_waits_for_part_a(tmp_path, label, wrapper):
  (tmp_path / 'af3').mkdir()
  r = budget(['check', label, wrapper, '1000'], tmp_path)
  assert r.returncode == 1 and 'part A has not ended' in r.stdout
  part_a_done(tmp_path, label)
  r = budget(['check', label, wrapper, '1000'], tmp_path, now='2026-12-01T09:00:00Z')
  assert r.returncode == 0 and r.stdout.strip().endswith('GO'), r.stdout + r.stderr


# ---------------------------------------------------------------- stop rule before each part

def test_guard_marker_order_cap_and_unknown_plan(tmp_path):
  (tmp_path / 'af3').mkdir()
  r = budget(['guard', 'harness/plans/dtp/dtp_gfd_01.yaml'], tmp_path)
  assert r.returncode == 0 and r.stdout.strip().endswith('GO'), r.stdout + r.stderr
  r = budget(['guard', 'harness/plans/dtp/dtp_afd_01.yaml'], tmp_path)
  assert r.returncode == 1 and 'part A has not ended' in r.stdout
  marker = tmp_path / 'marker.txt'
  marker.write_text('stopped\n')
  r = budget(['guard', 'harness/plans/aut/aut_gna_01.yaml'], tmp_path, marker=marker)
  assert r.returncode == 1 and 'marker' in r.stdout
  session(tmp_path, 'a100', 'rcf_af_r2b_01', '2026-11-27T00:00:00Z', end='2026-12-01T00:00:00Z', code='1')  # 96 h lost
  r = budget(['guard', 'harness/plans/rcf/rcf_af_r2b_02.yaml'], tmp_path)
  assert r.returncode == 1 and 'projected spend' in r.stdout
  assert budget(['guard', 'harness/plans/rcx/rcx_af_r2_01.yaml'], tmp_path).returncode == 1
  r = budget(['guard', 'harness/plans/xvm/xvm_g_r3_01.yaml'], tmp_path, now='2026-12-07T00:00:00Z')
  assert r.returncode == 1 and 'launch after 2026-12-06' in r.stdout


def test_guard_prices_a_fallback_part_on_demand(tmp_path):
  (tmp_path / 'af3').mkdir()
  p = next(x for x in parts('aut') if x['part'] == 'aut_gna_01')
  env = dict(os.environ, NOW='2026-12-01T08:00:00Z', RESULTS_AF3=str(tmp_path / 'af3'),
             PREP_MARKER=str(tmp_path / 'm.txt'), PROVISIONING='standard')
  r = subprocess.run([sys.executable, str(REPO / 'scripts' / 'monitor' / 'prep_budget.py'), 'guard',
                      'harness/plans/aut/aut_gna_01.yaml'], capture_output=True, text=True, cwd=REPO, env=env)
  assert r.returncode == 0 and f'this part {float(p["expected_usd_on_demand"]):.2f}' in r.stdout, r.stdout
  assert 'guard aut_gna_01 (standard)' in r.stdout


def test_spent_counts_the_lost_spot_session_and_the_on_demand_rerun(tmp_path):
  session(tmp_path, 'l4', 'aut_gna_01', '2026-12-01T00:00:00Z', end='2026-12-01T01:00:00Z', code='1')
  session(tmp_path, 'l4', 'aut_gna_01', '2026-12-01T02:00:00Z', end='2026-12-01T04:00:00Z', prov='standard')
  p = next(x for x in parts('aut') if x['part'] == 'aut_gna_01')
  r = budget(['spent', 'aut'], tmp_path, now='2026-12-01T05:00:00Z')
  want = float(p['usd_per_h_planned']) + 2 * float(p['usd_per_h_on_demand'])
  assert r.returncode == 0 and float(r.stdout) == pytest.approx(want, abs=0.01)
  r = budget(['spent', 'all'], tmp_path, now='2026-12-01T05:00:00Z')
  assert float(r.stdout) == pytest.approx(want, abs=0.01)


def guarded(tmp_path, plan, now, platform='l4'):
  stub = tmp_path / 'stub.sh'
  stub.write_text('echo "stub launcher PLAN=$PLAN"\n')
  (tmp_path / 'af3').mkdir(exist_ok=True)
  env = dict(os.environ, PLAN=plan, PLATFORM=platform, NOW=now, RESULTS_AF3=str(tmp_path / 'af3'),
             PREP_MARKER=str(tmp_path / 'marker.txt'), PREP_GUARD_LOG=str(tmp_path / 'guard.tsv'),
             PREP_LAUNCHER=str(stub))
  return subprocess.run(['bash', str(REPO / 'scripts' / 'monitor' / 'prep_guarded_launch.sh')], capture_output=True,
                        text=True, cwd=REPO, env=env)


def test_guarded_launcher_execs_on_go_and_writes_the_marker_only_for_the_cap_or_the_date(tmp_path):
  r = guarded(tmp_path, 'harness/plans/xvm/xvm_g_r3_01.yaml', '2026-12-01T08:00:00Z')
  assert r.returncode == 0 and 'stub launcher PLAN=harness/plans/xvm/xvm_g_r3_01.yaml' in r.stdout, r.stdout + r.stderr
  assert not (tmp_path / 'marker.txt').exists() and (tmp_path / 'guard.tsv').read_text().count('\n') == 1
  r = guarded(tmp_path, 'harness/plans/xvm/xvm_a_r3_01.yaml', '2026-12-01T08:00:00Z', 'a100')   # part A not ended
  assert r.returncode == 3 and 'stub launcher' not in r.stdout and not (tmp_path / 'marker.txt').exists()
  r = guarded(tmp_path, 'harness/plans/xvm/xvm_g_r3_02.yaml', '2026-12-07T00:00:00Z')
  assert r.returncode == 3 and 'launch after 2026-12-06' in (tmp_path / 'marker.txt').read_text()


def test_credit_check_script_usage():
  for args in (['det', 'l4', '1'], ['dtp', 'l4', 'x']):
    r = subprocess.run(['bash', str(REPO / 'scripts' / 'monitor' / 'credit_check_prep.sh')] + args,
                       capture_output=True, text=True, cwd=REPO)
    assert r.returncode == 2
