"""harness/plans/warm_th/ (label warm_th; drafts/section10_warm_th_20261111.md, the specification row) read through
harness/run_plan.py as the VMs read them, against the arms' source plans; scripts/monitor/credit_check_warm_th.sh on
a stub gcloud, a synthetic ledger and synthetic session folders. No cloud call."""
import csv
import os
import pathlib
import subprocess
import sys

import pytest

from run_plan import expand

REPO = pathlib.Path(__file__).resolve().parents[2]
PLANS = REPO / 'harness' / 'plans'
WT = PLANS / 'warm_th'
MANIFEST = REPO / 'inputs' / 'manifest.csv'
PILOT = sorted(r['pdb_id'] for r in csv.DictReader(open(MANIFEST)) if r['pilot'].strip().lower() in ('1', 'true', 'yes'))
# arm: (plan, platform, config, the source plan of the arm)
ARMS = {'gt': ('warm_th_gt_01.yaml', 'l4', 'l4_triton', PLANS / 'main' / 'main_gt_01.yaml'),
        'h': ('warm_th_h_01.yaml', 'h100', 'h100_xla', PLANS / 'h100' / 'h100_h_01.yaml')}


@pytest.mark.skipif(not (REPO / 'results' / 'analysis' / 'h100_20261104' / 'samples.csv').exists(),
                    reason='follow-up 9 samples.csv not present (results/ is gitignored)')
def test_warm_th_plans_are_up_to_date():
  out = subprocess.run([sys.executable, str(WT / 'make_warm_th_plans.py'), '--check'], capture_output=True, text=True)
  assert out.returncode == 0 and 'up to date' in out.stdout, out.stdout + out.stderr


@pytest.mark.parametrize('arm', sorted(ARMS))
def test_plan_is_a_fresh_run_and_its_warm_rerun_on_the_pilot_targets(arm):
  plan, platform, config, source = ARMS[arm]
  d = expand(WT / plan, platform, MANIFEST)
  assert d['label'] == 'warm_th' and d['configs'] == [config] and d['weights'] == 'gcs'
  assert sorted(d['targets']) == PILOT and len(PILOT) == 10
  assert d['seeds'] == [1] and d['fresh_reps'] == 1 and d['warm_rerun_seed'] == 1 and d['processes'] == 20
  assert d['eigen_source'] == 'archive' and d['launch_by'] == '2026-11-12' and d['finish_by'] == '2026-11-13'
  fresh, warm = d['runs']
  assert fresh == {'run': f'{config}_rep1_seed1_fresh', 'config': config, 'rep': 1, 'seed': 1, 'cache': 'fresh',
                   'libtpu': None, 'save_cache': True}
  assert warm['cache'] == 'warm' and warm['source_run'] == fresh['run'] and warm['run'] == f'{config}_seed1_warm'
  ref = expand(source, platform, MANIFEST)                                   # the main study's GT, follow-up 9's H
  for k in ('configs', 'weights', 'num_recycles', 'num_diffusion_samples', 'run_timeout_min', 'libtpu',
            'compare_reference'):
    assert d[k] == ref[k], k
  assert not {'buckets', 'stop_rule', 'save_embeddings', 'manifest', 'stack', 'seeds_beyond_input', 'rep_start'} & set(d)
  assert 0 < d['deadline_min'] < ref['deadline_min']


def test_parts_and_parts_files():
  parts = {p['arm']: p for p in csv.DictReader(open(WT / 'warm_th_parts.csv'))}
  assert parts['GT']['provisioning'] == 'standard' and parts['H']['provisioning'] == 'spot'
  assert parts['GT']['machine_type'] == 'g2-standard-8' and parts['H']['machine_type'] == 'a3-highgpu-1g'
  assert sum(float(p['worst_usd']) for p in parts.values()) < 15                   # the gate's cap (30) holds a relaunch
  for arm, (plan, platform, _, _) in ARMS.items():
    f = REPO / 'results' / f'parts_warm_th_{arm}.txt'
    if f.exists():
      lines = [l for l in f.read_text().splitlines() if l and not l.startswith('#')]
      assert lines == [f'{platform} harness/plans/warm_th/{plan} SPOT={1 if arm == "h" else 0} FETCH=full']


GATE = REPO / 'scripts' / 'monitor' / 'credit_check_warm_th.sh'


def gate(tmp_path, arm='gt', quota='10', balance=300.0, now='2026-11-12T08:00:00Z', sessions=(), p_open=True):
  """Runs the gate with a stub gcloud (quota value, or 'fail'), a ledger and a session folder of its own.
  sessions: (name, events text) pairs written as results/af3/<name>/launcher_events.txt."""
  tmp_path.mkdir(parents=True, exist_ok=True)
  g = tmp_path / 'gcloud'
  body = ('exit 1' if quota == 'fail' else
          "echo '{\"dimensionsInfos\": [{\"applicableLocations\": [\"global\"], \"details\": {\"value\": \"%s\"}}]}'" % quota)
  g.write_text('#!/bin/bash\n' + body + '\n')
  g.chmod(0o755)
  ledger = tmp_path / 'ledger.tsv'
  ledger.write_text(''.join(f'2026-11-10T09:20:00Z\tl4\t{a}\tok\t1\n' for a in ('l5', 'l6', 'lg', 'v6f', 'gf'))
                    + ('2026-11-11T04:24:24Z\tv6e\tp\tlaunch\t46.69\n' if p_open else
                       '2026-11-12T04:24:24Z\tv6e\tp\tok\t46.69\n'))
  af3 = tmp_path / 'af3'
  for name, ev in sessions:
    (af3 / name).mkdir(parents=True, exist_ok=True)
    (af3 / name / 'launcher_events.txt').write_text(ev)
  af3.mkdir(exist_ok=True)
  log = tmp_path / 'gate.tsv'
  env = dict(os.environ, GCLOUD=str(g), LEDGER=str(ledger), RESULTS_AF3=str(af3), GATE_LOG=str(log), NOW=now)
  r = subprocess.run(['bash', str(GATE), arm, str(balance)], capture_output=True, text=True, cwd=REPO, env=env)
  return r.returncode, r.stdout, log.read_text() if log.exists() else ''


def test_gate_go_with_the_quota_and_follow_up_13_p_counted(tmp_path):
  rc, out, log = gate(tmp_path)
  assert rc == 0 and out.strip().endswith('GO'), out
  assert 'not yet finished (follow-up 13) arm p' in out and 'counted: gt, h' in out
  assert log.startswith('2026-11-12T08:00:00Z\tgt\tGPUS_ALL_REGIONS=10\tgpu_other_live=0') and log.rstrip().endswith('GO')


def test_gate_counts_live_gpu_sessions_of_other_launchers_but_not_stale_ones(tmp_path):
  live = [(f'20261112T0{i}0000Z_l4_why_gf_{i:02d}', '2026-11-12T00:00:00Z launch: session x\n') for i in range(1, 10)]
  stale = [('20261012T052814Z_l4_pilot_l4d_2', '2026-10-12T05:28:14Z launch: session x\n')]
  rc, out, _ = gate(tmp_path, sessions=live + stale)
  assert rc == 1 and 'GPUS_ALL_REGIONS 10 < 11 needed' in out and 'GPU sessions of other launchers running: 9' in out
  rc, out, _ = gate(tmp_path / 'b', quota='fail', sessions=stale)
  assert rc == 0 and 'unread' in out
  rc, out, _ = gate(tmp_path / 'c', quota='fail', sessions=live[:1])
  assert rc == 1 and 'GPUS_ALL_REGIONS unread and a GPU session of another launcher is running' in out


def test_gate_floor_cap_and_dates(tmp_path):
  rc, out, _ = gate(tmp_path, balance=60)                    # 60 - 37.80 (P, 7 levels) - 6.46 < 25
  assert rc == 1 and '< 25 USD' in out
  assert gate(tmp_path / 'b', balance=60, p_open=False)[0] == 0
  rc, out, _ = gate(tmp_path / 'c', now='2026-11-13T00:00:00Z')
  assert rc == 1 and 'launch after 2026-11-12' in out
  rc, out, _ = gate(tmp_path / 'd', now='2026-11-12T23:30:00Z')      # the last launch still ends on 2026-11-13
  assert rc == 0 and 'to about 2026-11-13 01:00 UTC' in out


def test_gate_second_arm_and_relaunch(tmp_path):
  gt_live = ('20261112T080000Z_l4_warm_th_gt_01', '2026-11-12T08:00:05Z vm: created\n')
  rc, out, _ = gate(tmp_path, arm='h', sessions=[gt_live])
  assert rc == 0 and 'counted: h (' in out and 'running: gt' in out              # the running arm's cost is in "running"
  rc, out, _ = gate(tmp_path / 'b', arm='gt', sessions=[gt_live])
  assert rc == 1 and 'a session of arm gt is running' in out
  done = ('20261112T080000Z_l4_warm_th_gt_01', '2026-11-12T08:00:05Z vm: created\n2026-11-12T09:30:05Z delete: done\n'
                                               '2026-11-12T09:30:06Z launcher: exit 0\n')
  rc, out, _ = gate(tmp_path / 'c', arm='gt', sessions=[done], now='2026-11-12T10:00:00Z')
  assert rc == 1 and 'arm gt has a finished session' in out and 'spend so far from 1 session(s) 1.35' in out
  preempted = ('20261112T080005Z_h100_warm_th_h_01', '2026-11-12T08:00:10Z vm: created\n2026-11-12T08:30:10Z delete: done\n'
                                                     '2026-11-12T08:30:11Z launcher: exit 1\n')
  rc, out, _ = gate(tmp_path / 'd', arm='h', sessions=[done, preempted], now='2026-11-12T10:00:00Z')
  assert rc == 0 and 'counted: h (5.11 USD)' in out and 'spend so far from 2 session(s) ' in out


def test_gate_usage():
  r = subprocess.run(['bash', str(GATE), 'a', '100'], capture_output=True, text=True, cwd=REPO)
  assert r.returncode == 2 and 'usage' in r.stderr
  r = subprocess.run(['bash', str(GATE), 'gt'], capture_output=True, text=True, cwd=REPO)
  assert r.returncode == 2

