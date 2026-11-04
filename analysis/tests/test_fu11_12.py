"""Follow-ups 11 (repl) and 12 (prec5) (notes/analysis_plan.md section 10, rows of 2026-11-03): the targets of 11, the
plans as harness/plans/make_fu11_12_plans.py writes them, the budget rule with two arms at once, the settings diff,
and the two analyses on synthetic per-sample tables. No real AlphaFold3 output is read."""
import csv
import json
import os
import pathlib
import subprocess
import sys
import types

import pandas as pd
import pytest

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'harness'))
sys.path.insert(0, str(REPO / 'analysis' / 'tests'))
import run_plan  # noqa: E402

import contrasts  # noqa: E402
import followup34 as f34  # noqa: E402
import prec5  # noqa: E402
import repl  # noqa: E402
from test_fu7_10 import no_files, rows, write  # noqa: E402,F401  (no_files: autouse, no model file is read)

PLANS = REPO / 'harness' / 'plans'
ARGS = types.SimpleNamespace(seed=2026, replicates=40)
SUBSET = REPO / 'targets' / 'posebusters_subset.csv'
REPL = REPO / 'targets' / 'repl' / 'repl_subset.csv'


# ---- targets of follow-up 11

def test_repl_targets_are_every_eligible_complex_not_among_the_60():
  screen = list(csv.DictReader(open(REPO / 'targets' / 'posebusters_v2_screen.csv')))
  sub = list(csv.DictReader(open(REPL)))
  main = list(csv.DictReader(open(SUBSET)))
  assert [r['posebusters_id'] for r in sub] == sorted(r['posebusters_id'] for r in screen if r['status'] == 'eligible')
  assert len(sub) == 72 and {r['pilot'] for r in sub} == {'no'}
  assert not {r['pdb_id'] for r in sub} & {r['pdb_id'] for r in main}
  assert list(sub[0]) == list(main[0])                                   # the main subset's columns
  by_id = {r['posebusters_id']: r for r in screen}
  for r in sub:                                                           # token counts and buckets of the screen
    assert (r['num_tokens'], r['bucket'], r['bin']) == tuple(by_id[r['posebusters_id']][k] for k in ('num_tokens', 'bucket', 'bin'))
    assert r['rcsb_initial_release'] and r['sequence'] and r['n_input_chains'] == '2'
  src = json.loads((REPO / 'targets' / 'repl' / 'sources.json').read_text())
  assert src['counts']['by_bucket'] == {'256': 5, '512': 67}
  assert src['output']['sha256'] == __import__('hashlib').sha256(REPL.read_bytes()).hexdigest()


def test_input_builders_keep_their_defaults():
  msas = (REPO / 'inputs' / 'build_msas.py').read_text()
  inputs = (REPO / 'inputs' / 'build_inputs.py').read_text()
  assert "SUBSET = REPO / 'targets' / 'posebusters_subset.csv'" in msas and "default=str(SUBSET.relative_to(REPO))" in msas
  assert "SUBSET = REPO / 'targets' / 'posebusters_subset.csv'" in inputs and "INPUT_DIR = REPO / 'data' / 'inputs'" in inputs
  assert "ap.add_argument('--manifest', default='inputs/manifest.csv')" in inputs
  assert "ap.add_argument('--exclusions', help=" in inputs                 # off unless given
  out = subprocess.run([sys.executable, str(REPO / 'inputs' / 'build_msas.py'), '--help'], capture_output=True, text=True)
  assert out.returncode == 0 and '--subset' in out.stdout


# ---- plans

def describe(path, platform, manifest=None):
  return run_plan.expand(path, platform, REPO / 'inputs' / 'manifest.csv')


@pytest.fixture
def repl_manifest(tmp_path, monkeypatch):
  """inputs/repl/manifest.csv as build_inputs.py will write it (pdb_id, pilot, num_tokens, bucket), in a temporary
  repository root: run_plan reads a plan's manifest key relative to its REPO."""
  if (REPO / 'inputs' / 'repl' / 'manifest.csv').is_file():
    return
  (tmp_path / 'inputs' / 'repl').mkdir(parents=True)
  pd.read_csv(REPL, dtype=str)[['pdb_id', 'pilot', 'num_tokens', 'bucket']].to_csv(
      tmp_path / 'inputs' / 'repl' / 'manifest.csv', index=False)
  monkeypatch.setattr(run_plan, 'REPO', tmp_path)


def test_generator_output_is_up_to_date():
  for g in ('make_fu11_12_plans.py', 'make_fu7_10_plans.py'):
    rc = subprocess.run([sys.executable, str(PLANS / g), '--check'], capture_output=True, text=True)
    assert rc.returncode == 0, g + rc.stdout + rc.stderr


def test_repl_arms_cover_the_72_targets_once_with_r1_and_r2(repl_manifest):
  want = sorted(pd.read_csv(REPL, dtype=str).pdb_id)
  for arm, platform, config in (('v5r', 'v5e', 'tpu_xla'), ('v6r', 'v6e', 'tpu_xla'), ('gr', 'l4', 'l4_xla')):
    ts = []
    for p in sorted((PLANS / 'repl').glob(f'repl_{arm}_*.yaml')):
      d = describe(p, platform)
      assert d['label'] == 'repl' and d['configs'] == [config] and d['seeds'] == [1, 2, 3, 4, 5]
      assert d['fresh_reps'] == 2 and d['warm_rerun_seed'] is None and d['manifest'] == 'inputs/repl/manifest.csv'
      assert d['eigen_source'] == 'archive' and d['finish_by'] == '2026-11-14' and 'launch_by' not in d
      assert {r['rep'] for r in d['runs']} == {1, 2} and not d.get('seeds_beyond_input')
      ts += d['targets']
    assert sorted(ts) == want, arm


def test_prec5_arms_run_seeds_3_to_5_of_the_60_at_highest_precision():
  want = sorted(pd.read_csv(SUBSET, dtype=str).pdb_id)
  for arm, platform, config in (('v6h5', 'v6e', 'tpu_xla_highest'), ('gh5', 'l4', 'l4_xla_highest')):
    ts = []
    for p in sorted((PLANS / 'prec5').glob(f'prec5_{arm}_*.yaml')):
      d = describe(p, platform)
      assert d['label'] == 'prec5' and d['configs'] == [config] and d['seeds'] == [3, 4, 5] and d['fresh_reps'] == 1
      assert 'manifest' not in d and not d.get('seeds_beyond_input') and d['eigen_source'] == 'archive'
      ts += d['targets']
    assert sorted(ts) == want, arm
  assert set(json.loads((REPO / 'inputs' / 'manifest.json').read_text())['seeds']) >= {3, 4, 5}   # in modelSeeds


def test_config_diff_shows_only_the_intended_differences():
  r = subprocess.run([sys.executable, str(PLANS / 'config_diff_fu11_12.py'), '--out', os.devnull], capture_output=True,
                     text=True)
  assert r.returncode == 0, r.stdout[-2000:] + r.stderr
  assert 'Settings that differ beyond the intended ones: 0' in r.stdout
  section = lambda arm: r.stdout.split(f'## {arm} against')[1].split('\n## ')[0]  # noqa: E731
  assert '| input manifest | `inputs/manifest.csv` | `inputs/repl/manifest.csv` | intended |' in section('V5r')
  assert '| seeds of the arm | `1 to 2` | `3 to 5` | intended |' in section('V6h5')
  assert '| config name | `tpu_xla_highest` | `tpu_xla_highest` | same |' in section('V6h5')
  assert '| provisioning | `standard` | `flex_start` | intended |' in section('V6r')


# ---- budget rule, two arms at once

def credit(tmp_path, *args, now='2026-11-05T00:00:00Z'):
  for f in ('harness/plans/fu11_12_parts.csv', 'harness/plans/fu7_10_parts.csv', 'cloud/prices.csv'):
    (tmp_path / f).parent.mkdir(parents=True, exist_ok=True)
    (tmp_path / f).write_text((REPO / f).read_text())
  r = subprocess.run(['bash', str(REPO / 'scripts' / 'monitor' / 'credit_check_fu11_12.sh'), *args], cwd=tmp_path,
                     capture_output=True, text=True, env=dict(os.environ, NOW=now))
  return r.returncode, r.stdout + r.stderr


def expected(arm, parts='fu11_12_parts.csv'):
  return sum(float(r['expected_usd_planned']) for r in csv.DictReader(open(PLANS / parts))
             if r['arm'].lower().replace(' ', '_') == arm)


def session(tmp_path, name, events, prov='spot', where='results/af3/.launcher'):
  s = tmp_path / where / name
  s.mkdir(parents=True, exist_ok=True)
  (s / 'launcher_events.txt').write_text(events)
  (s / 'launch_record.txt').write_text(f'PROVISIONING={prov}\n')


RUNNING = '2026-11-04T20:00:00Z  launch: session x\n2026-11-04T20:01:00Z  vm: af3-run-x running in us-central1-a\n'


def test_credit_check_subtracts_every_running_arm(tmp_path):
  code, out = credit(tmp_path, 'gr', '200')
  assert code == 0 and f'arm gr (l4 spot): expected cost {expected("gr"):.2f} USD' in out and 'running arm ' not in out
  session(tmp_path, '20261104T200000Z_h100_h100_h_r2_01', RUNNING)              # H r2 still running
  session(tmp_path, '20261104T200000Z_v5e_repl_v5r_01', RUNNING, prov='standard')
  session(tmp_path, '20261103T000000Z_l4_sw_gs_01', '2026-11-03T00:00:00Z  vm: x\n2026-11-03T01:00:00Z  delete: done\n'
          '2026-11-03T01:00:01Z  launcher: exit code 0\n', where='results/af3')  # exited: not subtracted
  code, out = credit(tmp_path, 'gr', '200')
  left = 200 - expected('h_r2', 'fu7_10_parts.csv') - expected('v5r') - expected('gr')
  assert f'running arm h_r2 (1 session(s) whose launcher has not exited): expected {expected("h_r2", "fu7_10_parts.csv"):.2f}' in out
  assert 'running arm v5r' in out and 'running arm gs' not in out
  assert f'balance minus running arms minus expected {left:.2f} USD' in out and code == 0
  code, out = credit(tmp_path, 'gr', f'{200 - left + 24:.2f}')                  # 24 USD left: below the floor
  assert code == 1 and '< 25 USD' in out


def test_credit_check_cap_spend_and_the_date_of_follow_up_12(tmp_path):
  session(tmp_path, '20261104T000000Z_l4_prec5_gh5_01', '2026-11-04T00:00:00Z  vm: x\n'
          '2026-11-04T10:00:00Z  delete: done\n2026-11-04T10:00:01Z  launcher: exit code 0\n', where='results/af3')
  spot = next(float(r['usd_per_h_planned']) for r in csv.DictReader(open(PLANS / 'fu11_12_parts.csv')) if r['platform'] == 'l4')
  code, out = credit(tmp_path, 'v6h5', '500')
  assert code == 0 and f'from 1 session record(s): {10 * spot:.2f} USD' in out and '(cap of prec5: 24)' in out
  code, out = credit(tmp_path, 'v6h5', '500', now='2026-11-14T20:00:00Z')       # 7.9 h cannot end by 2026-11-14
  assert code == 1 and 'cannot finish by 2026-11-14' in out
  code, out = credit(tmp_path, 'v5r', '500', now='2026-11-14T20:00:00Z')        # follow-up 11: printed, not a condition
  assert code == 0 and 'expected duration' in out
  assert credit(tmp_path, 'v5x', '1')[0] == 2 and credit(tmp_path, 'gr', 'x')[0] == 2


# ---- analyses

T_NEW, T_MAIN = ['R1', 'R2', 'R3', 'R4'], ['M1', 'M2', 'M3', 'M4']


def repl_world(tmp_path, monkeypatch, extra_target=None):
  write([{'pdb_id': t, 'pilot': 'no', 'num_tokens': 300, 'bucket': 512 if t != 'R4' else 256} for t in T_NEW],
        tmp_path / 'manifest.csv')
  write([{'pdb_id': t, 'rcsb_initial_release': '2022-01-01' if t in ('R3', 'R4') else '2020-01-01'} for t in T_NEW],
        tmp_path / 'targets.csv')
  write([{'pdb_id': t, 'rcsb_initial_release': '2022-01-01', 'pilot': 'no'} for t in T_MAIN], tmp_path / 'main_targets.csv')
  write([{'pdb_id': t, 'bucket': 512} for t in T_MAIN], tmp_path / 'main_manifest.csv')
  main = tmp_path / 'main' / 'contrasts'
  main.mkdir(parents=True)
  m = pd.DataFrame(rows('V5', 'main', 'r1', T_MAIN, range(1, 6)) + rows('V5', 'main', 'r2', T_MAIN, range(1, 6))
                   + rows('V6', 'main', 'r1', T_MAIN, range(1, 6), shift=0.2) + rows('V6', 'main', 'r2', T_MAIN, range(1, 6), shift=0.2)
                   + rows('G', 'main', 'r1', T_MAIN, range(1, 6)) + rows('G', 'main', 'r2', T_MAIN, range(1, 6)))
  m['condition'] = m.platform
  sp, sm, cons, _ = f34.build(f34.sample_rows(m), repl.SPECS)
  sp.to_csv(main / 'pairs_seed.csv', index=False)
  contrasts.summarize(sp, sm, cons, ARGS).to_csv(main / 'contrasts.csv', index=False)
  contrasts.hw_minus_rc(sp, cons, ARGS).to_csv(main / 'hw_minus_rc.csv', index=False)
  ts = T_NEW + ([extra_target] if extra_target else [])
  new = (rows('V5', 'repl', 'r1', ts, range(1, 6)) + rows('V5', 'repl', 'r2', ts, range(1, 6))
         + rows('V6', 'repl', 'r1', ts, range(1, 6), shift=0.2) + rows('V6', 'repl', 'r2', ts, range(1, 6), shift=0.2)
         + rows('G', 'repl', 'r1', ts, range(1, 6), shift=0.2) + rows('G', 'repl', 'r2', ts, range(1, 6), shift=0.2))
  out = tmp_path / 'out'
  monkeypatch.setattr(sys, 'argv', ['repl.py', '--samples', str(write(new, tmp_path / 'samples.csv')),
                                    '--main', str(tmp_path / 'main'), '--manifest', str(tmp_path / 'manifest.csv'),
                                    '--targets', str(tmp_path / 'targets.csv'),
                                    '--main_targets', str(tmp_path / 'main_targets.csv'),
                                    '--main_manifest', str(tmp_path / 'main_manifest.csv'),
                                    '--out', str(out), '--replicates', '40'])
  return out


def test_repl_reading_rule_next_to_main_and_pooled(tmp_path, monkeypatch):
  out = repl_world(tmp_path, monkeypatch)
  assert repl.main() == 0
  rule = pd.read_csv(out / 'reading_rule.csv').set_index(['targets', 'pair', 'statistic'])
  assert rule.loc[('all 4', 'HW(V5,V6)', 'flip_top1'), 'answer'] == 'replicates'          # every top-1 flips, RC 0
  assert rule.loc[('all 4', 'HW(V5,V6)', 'flip_top1'), 'role'] == 'headline'
  assert rule.loc[('all 4', 'HW(V6,G)', 'flip_top1'), 'answer'].startswith('does not replicate')
  assert rule.loc[('post-cutoff 2', 'HW(V5,V6)', 'flip_success'), 'answer'].startswith('does not replicate')
  assert ('post-cutoff 2', 'HW(V5,V6)', 'flip_top1') not in rule.index                     # M1 results only
  side = pd.read_csv(out / 'next_to_main.csv')
  assert {'all 4', 'bucket 512: 3', 'bucket 512: 4'} <= set(side.targets)
  assert set(side[side.source == 'main study'].targets) == {'all 4', 'bucket 512: 4'}
  pooled = pd.read_csv(out / 'pooled.csv')
  assert set(pooled.targets) == {'pooled 8', 'pooled post-cutoff 6'}
  hw = pooled[(pooled.quantity == 'HW(V5,V6) minus RC(V5)') & (pooled.statistic == 'flip_top1')]
  assert float(hw.estimate.iloc[0]) == 1.0
  assert (pd.read_csv(out / 'completeness.csv').status == 'complete').all()
  assert 'HW(V5,V6), flip_top1 (headline): replicates' in (out / 'report.md').read_text()


def test_repl_stops_on_a_target_of_the_main_study(tmp_path, monkeypatch):
  repl_world(tmp_path, monkeypatch, extra_target='M1')
  with pytest.raises(SystemExit, match='a check failed'):
    repl.main()


def prec5_world(tmp_path, monkeypatch, new_seeds=range(3, 6), hwh_shift=0.0):
  targets = ['T1', 'T2', 'T3', 'T4']
  write([{'pdb_id': t, 'rcsb_initial_release': '2022-01-01' if t in ('T3', 'T4') else '2020-01-01', 'pilot': 'no'}
         for t in targets], tmp_path / 'targets.csv')
  main = tmp_path / 'main' / 'contrasts'
  main.mkdir(parents=True)
  m = pd.DataFrame(rows('V6', 'main', 'r1', targets, range(1, 6)) + rows('V6', 'main', 'r2', targets, range(1, 6))
                   + rows('G', 'main', 'r1', targets, range(1, 6), shift=0.2) + rows('G', 'main', 'r2', targets, range(1, 6), shift=0.2)
                   + rows('V6h', 'main', 'r1', targets, [1, 2]) + rows('Gh', 'main', 'r1', targets, [1, 2], shift=hwh_shift))
  m['condition'] = m.platform
  m.to_csv(tmp_path / 'main' / 'samples.csv', index=False)
  specs = prec5.SPECS + [('RC(V6)', 'RC', ('V6', 'r1'), ('V6', 'r2'), True), ('RC(G)', 'RC', ('G', 'r1'), ('G', 'r2'), True)]
  sp, sm, cons, _ = f34.build(f34.sample_rows(m), specs)
  sp.to_csv(main / 'pairs_seed.csv', index=False)
  contrasts.summarize(sp, sm, cons, ARGS).to_csv(main / 'contrasts.csv', index=False)
  contrasts.hw_minus_rc(sp, cons, ARGS).to_csv(main / 'hw_minus_rc.csv', index=False)
  new = rows('V6h', 'prec5', 'r1', targets, new_seeds) + rows('Gh', 'prec5', 'r1', targets, new_seeds, shift=hwh_shift)
  out = tmp_path / 'out'
  monkeypatch.setattr(sys, 'argv', ['prec5.py', '--samples', str(write(new, tmp_path / 'samples.csv')),
                                    '--main', str(tmp_path / 'main'), '--targets', str(tmp_path / 'targets.csv'),
                                    '--out', str(out), '--replicates', '40'])
  return out


def test_prec5_five_seeds_next_to_two_and_the_section_11_reading(tmp_path, monkeypatch):
  out = prec5_world(tmp_path, monkeypatch)
  assert prec5.main() == 0
  chk = pd.read_csv(out / 'checks.csv')
  assert (chk.detail == 'pass').all() and chk.check.str.contains('seeds 1 and 2 equal the main study').any()
  con = pd.read_csv(out / 'contrasts.csv')
  five = con[(con.seeds == '1 to 5') & (con.targets == 'all 4') & (con.statistic == 'flip_top1')].set_index('contrast')
  assert five.loc['PR(G)', 'estimate'] == 1.0 and five.loc['HWh(V6,G)', 'estimate'] == 0.0
  assert five.loc['HW(V6,G)', 'estimate'] == 1.0 and five.loc['PR(V6)', 'n'] == 20         # 4 targets x 5 seeds
  assert set(con.seeds) == {'1 to 5', '1 and 2 (main study)'}
  rule = pd.read_csv(out / 'reading_rule.csv')
  r = rule[(rule.seeds == '1 to 5') & (rule.targets == 'all 4') & (rule.statistic == 'flip_top1')].reading.iloc[0]
  assert r.startswith('shrinks to the RC level')
  assert set(pd.read_csv(out / 'completeness.csv').status) == {'complete'}


def test_prec5_platform_reading_and_refusal_of_other_seeds(tmp_path, monkeypatch):
  out = prec5_world(tmp_path, monkeypatch, hwh_shift=0.2)
  assert prec5.main() == 0
  rule = pd.read_csv(out / 'reading_rule.csv')
  r = rule[(rule.seeds == '1 to 5') & (rule.targets == 'all 4') & (rule.statistic == 'flip_top1')].reading.iloc[0]
  assert r.startswith('does not shrink')
  (tmp_path / 'b').mkdir()
  prec5_world(tmp_path / 'b', monkeypatch, new_seeds=[2, 3])
  with pytest.raises(SystemExit, match='a check failed'):
    prec5.main()
