"""harness/plans/memlimit/ against the section 10 row of 2026-10-14 ("Memory limit per chip"), read through
harness/run_plan.py as the VMs read them; the --buckets option and the stop rule of harness/run_plan.py and
harness/run_af3.py. End-to-end launches on fake VMs: tests/cloud/suite_7_memlimit.sh."""

import json
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


@pytest.mark.parametrize('plan', sorted(p for p in PLANS.rglob('*.yaml') if not {'memlimit', 'embed', 'ppi', 'feat', 'msa', 'msa_smoke', 'sw_smoke', 'h100_smoke', 'eigen_check'} & set(p.parts)))
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
