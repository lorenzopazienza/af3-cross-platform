"""harness/plans/memlimit/ against the section 10 row of 2026-10-14 ("Memory limit per chip"), read through
harness/run_plan.py as the VMs read them; the --buckets option and the stop rule of harness/run_plan.py and
harness/run_af3.py. End-to-end launches on fake VMs: tests/cloud/suite_7_memlimit.sh."""

import json
import os
import pathlib
import subprocess
import sys

import pytest

import run_af3
import run_plan
from run_plan import expand

REPO = pathlib.Path(__file__).resolve().parents[2]
PLANS = REPO / 'harness' / 'plans'
MANIFEST = REPO / 'inputs' / 'manifest.csv'
SIZES = [1280, 1536, 2048, 2560, 3072, 3584, 4096, 4608, 5120]
ARMS = {'v5': ('v5e', 'tpu_xla'), 'v6': ('v6e', 'tpu_xla'), 'g': ('l4', 'l4_xla'), 'gt': ('l4', 'l4_triton')}


@pytest.mark.parametrize('arm', sorted(ARMS))
def test_memlimit_plan_is_the_specified_measurement(arm):
  platform, config = ARMS[arm]
  d = expand(PLANS / 'memlimit' / f'memlimit_{arm}.yaml', platform, MANIFEST)
  assert d['label'] == 'memlimit' and d['weights'] == 'gcs' and d['targets'] == ['7D5C'] and d['tokens'] == {'7D5C': 1023}
  assert d['seeds'] == [1] and d['fresh_reps'] == 1 and d['warm_rerun_seed'] is None and d['configs'] == [config]
  assert d['num_recycles'] is None and d['num_diffusion_samples'] is None          # AlphaFold3's defaults
  assert d['buckets'] == SIZES and d['stop_rule'] == 'first_failure' and d['run_timeout_min'] == 240
  assert [r['buckets'] for r in d['runs']] == SIZES and d['processes'] == 9
  assert all(r['cache'] == 'fresh' and not r['save_cache'] for r in d['runs'])
  assert [r['run'] for r in d['runs']] == [f'{config}_bucket{s}_rep1_seed1_fresh' for s in SIZES]
  assert 0 < d['deadline_min'] < 9 * 240                                            # sized by make_memlimit_plans.py


@pytest.mark.skipif(not (REPO / 'results' / 'analysis' / 'main_20261014' / 'primary_v2' / 'samples.csv').exists(),
                    reason='main-study samples.csv not present (results/ is gitignored)')
def test_plans_are_what_the_generator_writes():
  before = {p.name: p.read_text() for p in (PLANS / 'memlimit').glob('memlimit_*.yaml')}
  subprocess.run([sys.executable, str(PLANS / 'memlimit' / 'make_memlimit_plans.py')], check=True, capture_output=True)
  assert {p.name: p.read_text() for p in (PLANS / 'memlimit').glob('memlimit_*.yaml')} == before


@pytest.mark.parametrize('plan', sorted(p for p in PLANS.rglob('*.yaml') if not {'memlimit', 'embed', 'ppi', 'feat', 'msa', 'msa_smoke', 'sw_smoke', 'h100_smoke', 'eigen_check',
                                                         'repl', 'large', 'memlimit_ah'} & set(p.parts)))
def test_other_plans_get_no_new_keys(plan):
  for platform in run_plan.PLATFORMS:
    try:
      d = expand(plan, platform, MANIFEST)
    except SystemExit:
      continue                                                                       # not a plan for this platform
    assert not {'buckets', 'stop_rule', 'save_embeddings', 'manifest'} & set(d)
    assert not any('buckets' in r for r in d['runs'])


def test_plan_refuses_bad_buckets(tmp_path):
  base = (PLANS / 'memlimit' / 'memlimit_g.yaml').read_text()
  for bad in ('buckets: [1536, 1280]', 'buckets: [1280, 1280]', 'buckets: [0, 1280]'):
    p = tmp_path / 'p.yaml'
    p.write_text(base.replace('buckets: [1280, 1536, 2048, 2560, 3072, 3584, 4096, 4608, 5120]', bad))
    with pytest.raises(SystemExit):
      expand(p, 'l4', MANIFEST)
  p.write_text(base.replace('warm_rerun_seed: none', 'warm_rerun_seed: 1'))
  with pytest.raises(SystemExit, match='cannot be combined'):
    expand(p, 'l4', MANIFEST)
  p.write_text(base.replace('stop_rule: first_failure', 'stop_rule: never'))
  with pytest.raises(SystemExit, match='stop_rule'):
    expand(p, 'l4', MANIFEST)


def test_parse_buckets():
  assert run_af3.parse_buckets('1280') == [1280] and run_af3.parse_buckets('1280,1536') == [1280, 1536]
  for bad in ('1536,1280', 'x', '0', '1280,1280'):
    with pytest.raises(SystemExit):
      run_af3.parse_buckets(bad)


def test_failure_detail():
  log = 'Featurising...\nTraceback (most recent call last):\njaxlib.xla_extension.XlaRuntimeError: RESOURCE_EXHAUSTED: ' \
        'Out of memory while trying to allocate 30.1GiB.\nother line\n'
  assert run_af3.failure_detail(None, False, log) is None
  assert run_af3.failure_detail('oom', False, log).startswith('jaxlib.xla_extension.XlaRuntimeError: RESOURCE_EXHAUSTED')
  assert run_af3.failure_detail('timeout', True, log) == 'killed by the harness at --timeout_min'
  assert run_af3.failure_detail('error', False, 'a\nValueError: bad input\nb\n') == 'ValueError: bad input'
  assert run_af3.failure_detail('signal 9', False, 'a\nlast\n') == 'last'


def test_stop_rule_logs_later_runs_as_not_run(tmp_path):
  d = expand(PLANS / 'memlimit' / 'memlimit_v6.yaml', 'v6e', MANIFEST)
  stop_run = d['runs'][4]['run']                                                     # 3072
  (tmp_path / stop_run).mkdir()
  (tmp_path / stop_run / 'runs.jsonl').write_text(json.dumps(
      {'pdb_id': '7D5C', 'exit_code': 1, 'timed_out': False, 'failure': 'oom', 'failure_detail': 'RESOURCE_EXHAUSTED: x',
       'wall_seconds': 50.0, 'padded_bucket': 3072}) + '\n')
  log = tmp_path / 'plan_runs.jsonl'
  stop = run_plan.apply_stop_rule(log, tmp_path, d, 5, 1)
  lines = [json.loads(l) for l in log.read_text().splitlines()]
  assert [l['buckets'] for l in lines] == SIZES[5:]
  assert all(l['harness_exit_code'] is None and l['not_run'] == 'stop rule' and l['stopped_by'] == stop_run for l in lines)
  assert stop['stopped_at_bucket'] == 3072 and stop['last_completed_bucket'] == 2560
  assert stop['processes'][0]['failure'] == 'oom' and len(stop['not_run']) == 4
  assert json.loads((tmp_path / 'stop_rule.json').read_text()) == stop


# ---- memory limit on A and H (label memlimit_ah; drafts/section10_memlimit_ah.md)

AH = PLANS / 'memlimit_ah'


@pytest.mark.skipif(not (REPO / 'results' / 'analysis' / 'a100_20261018' / 'samples.csv').exists(),
                    reason='follow-up 3 samples.csv not present (results/ is gitignored)')
def test_memlimit_ah_plans_are_up_to_date():
  out = subprocess.run([sys.executable, str(AH / 'make_memlimit_ah_plans.py'), '--check'], capture_output=True, text=True)
  assert out.returncode == 0 and 'up to date' in out.stdout, out.stdout + out.stderr


@pytest.mark.parametrize('arm,platform,config', [('a', 'a100', 'a100_xla'), ('h', 'h100', 'h100_xla')])
def test_memlimit_ah_is_the_memlimit_measurement_with_the_follow_up_configuration(arm, platform, config):
  new = expand(AH / f'memlimit_ah_{arm}_01.yaml', platform, MANIFEST)
  ref = expand(PLANS / 'memlimit' / 'memlimit_g.yaml', 'l4', MANIFEST)
  assert new['label'] == 'memlimit_ah' and new['configs'] == [config] and new['eigen_source'] == 'archive'
  for k in ('targets', 'seeds', 'buckets', 'stop_rule', 'run_timeout_min', 'num_recycles', 'num_diffusion_samples'):
    assert new[k] == ref[k], k
  assert [r['run'].replace(config, 'C') for r in new['runs']] == [r['run'].replace('l4_xla', 'C') for r in ref['runs']]


GATE = REPO / 'scripts' / 'monitor' / 'credit_check_memlimit_ah.sh'


def gate(tmp_path, quota, gf=None, live_l4=False, balance=574.88, now='2026-11-11T08:00:00Z'):
  """Runs the gate with a stub gcloud (quota value, or 'fail'), a ledger and a session folder of its own."""
  g = tmp_path / 'gcloud'
  body = ('exit 1' if quota == 'fail' else
          "echo '{\"dimensionsInfos\": [{\"applicableLocations\": [\"global\"], \"details\": {\"value\": \"%s\"}}]}'" % quota)
  g.write_text('#!/bin/bash\n' + body + '\n')
  g.chmod(0o755)
  ledger = tmp_path / 'ledger.tsv'
  ledger.write_text(''.join(f'2026-11-10T09:20:00Z\tl4\t{a}\tok\t1\n' for a in ('l5', 'l6', 'lg', 'v6f', 'p'))
                    + (f'2026-11-11T07:00:00Z\tl4\tgf\t{gf}\t281.84\n' if gf else ''))
  af3 = tmp_path / 'af3'
  s = af3 / '20261111T070000Z_l4_why_gf_01'
  s.mkdir(parents=True, exist_ok=True)
  (s / 'launcher_events.txt').write_text('2026-11-11T07:00:00Z launch: session\n'
                                         + ('' if live_l4 else '2026-11-11T07:30:00Z launcher: exit 0\n'))
  env = dict(os.environ, GCLOUD=str(g), LEDGER=str(ledger), RESULTS_AF3=str(af3), GATE_LOG=str(tmp_path / 'gate.tsv'),
             NOW=now)
  r = subprocess.run(['bash', str(GATE), str(balance)], capture_output=True, text=True, cwd=REPO, env=env)
  return r.returncode, r.stdout, (tmp_path / 'gate.tsv').read_text() if (tmp_path / 'gate.tsv').exists() else ''


def test_gate_start_condition_quota_of_10_while_gf_runs(tmp_path):
  rc, out, log = gate(tmp_path, '10', gf=None, live_l4=True)
  assert rc == 0 and out.strip().endswith('GO') and 'GPUS_ALL_REGIONS=10' in log and log.rstrip().endswith('GO')


def test_gate_quota_8_waits_for_gf(tmp_path):
  rc, out, _ = gate(tmp_path, '8', gf=None, live_l4=True)
  assert rc == 1 and 'GPUS_ALL_REGIONS 8 < 10' in out and 'Gf has not finished' in out
  rc, out, _ = gate(tmp_path, '8', gf='ok', live_l4=False)
  assert rc == 0, out
  rc, out, _ = gate(tmp_path, '8', gf='stopped', live_l4=True)
  assert rc == 1 and 'an L4 arm of follow-up 13 is running' in out


def test_gate_unreadable_quota_falls_back_to_gf(tmp_path):
  rc, out, log = gate(tmp_path, 'fail', gf=None)
  assert rc == 1 and 'GPUS_ALL_REGIONS unread' in out and 'GPUS_ALL_REGIONS=unread' in log
  assert gate(tmp_path, 'fail', gf='ok')[0] == 0


def test_gate_floor_counts_unfinished_follow_up_13_arms_and_dates(tmp_path):
  rc, out, _ = gate(tmp_path, '10', gf=None, balance=260)    # 260 - 148.99 (gf, not finished) - 94.32 < 25
  assert rc == 1 and 'not yet finished (follow-up 13) arm gf' in out and '< 25 USD' in out
  assert gate(tmp_path, '10', gf='ok', balance=260)[0] == 0
  rc, out, _ = gate(tmp_path, '10', gf='ok', now='2026-12-03T00:00:00Z')
  assert rc == 1 and 'launch after 2026-12-02' in out
