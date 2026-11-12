"""Follow-up 14 (label f32; notes/analysis_plan.md section 10, row of 2026-11-12): synthetic tests of every reading
rule of analysis/fu14.py, of the plans, configurations and config diff, of the budget rule, the stop rule (the A100
wrapper's guarded launcher) and the session check. No follow-up 14 output exists or is read."""
import json
import os
import pathlib
import re
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'analysis'))
sys.path.insert(0, str(REPO / 'harness'))
sys.path.insert(0, str(REPO / 'harness' / 'plans'))
import fu13  # noqa: E402
import fu13_probe  # noqa: E402
import fu14  # noqa: E402
from run_af3 import read_configs  # noqa: E402


class Args:
  seed, replicates = 2026, 400


# ---------------------------------------------------------------- conditions, design

def test_condition_of_maps_every_follow_up_14_configuration():
  assert fu14.condition_of('tpu_xla_fu13_f32', 'V5h') == 'V5f'       # collect.py codes highest precision as V5h
  assert fu14.condition_of('tpu_xla_fu13_f32', 'V6h') == 'V6f'
  assert fu14.condition_of('a100_xla_fu14_f32', 'Ah') == 'Af'
  assert fu14.condition_of('h100_xla_fu14_f32', 'Hh') == 'Hf'
  assert fu14.condition_of('a100_xla_det', 'A') == 'Ad'
  assert fu14.condition_of('h100_xla_det', 'H') == 'Hd'
  assert fu14.condition_of('l4_xla_fu13_f32', 'Gh') == 'Gf'
  assert fu14.condition_of('a100_xla', 'A') is None


def test_planned_units_and_pairs_are_the_specification():
  n = {c: sum(p.values()) for c, p in fu14.PLANNED.items()}
  assert n == {'V5f': 310, 'Af': 300, 'Hf': 300, 'Ad': 100, 'Hd': 100}
  pairs = {frozenset(p) for p in fu14.PAIRS}
  assert len(fu14.PAIRS) == 10 and len(pairs) == 10 and all(len(p) == 2 and p <= set(fu14.FLOAT32) for p in pairs)
  assert fu14.PRIMARY == [('V5f', 'V6f'), ('V6f', 'Gf'), ('Hf', 'Af'), ('Gf', 'Af')]
  assert set(fu14.BF16) == set(fu14.PAIRS) and set(fu14.RC) == set(fu14.PRIMARY)
  assert fu14.RC[('Gf', 'Af')][1] == 'RC(Gf)' and fu14.RC[('Hf', 'Af')][1] is None and fu14.RC[('V5f', 'V6f')][1] is None


# ---------------------------------------------------------------- reading rules

def test_agree_threshold_is_strict_below_five_points():
  assert fu14.THRESHOLD == 0.05
  assert fu14.agree_reading(0.0) == 'agree' and fu14.agree_reading(0.0499) == 'agree'
  assert fu14.agree_reading(0.05) == 'agreement not shown' and fu14.agree_reading(0.2) == 'agreement not shown'
  assert fu14.agree_reading(float('nan')) == 'not computed'


def con_rows(highs):
  rows = []
  for (p, q), hi in zip(fu14.PAIRS, highs):
    rows.append({'contrast': fu14.hw(p, q), 'statistic': 'flip_top1', 'estimate': hi / 2, 'ci_low': 0.0, 'ci_high': hi,
                 'n': 300})
  return pd.DataFrame(rows)


def test_generality_needs_all_ten_pairs():
  a = fu14.agree_table(con_rows([0.01] * 10))
  assert (a.reading == 'agree').all() and fu14.generality(a).startswith('agree on every accelerator')
  a = fu14.agree_table(con_rows([0.01] * 9 + [0.051]))          # one secondary pair above: no generality
  assert fu14.generality(a).startswith('not shown') and 'HW(Hf,Gf)' in fu14.generality(a).split('; not ')[1]
  a = fu14.agree_table(con_rows([0.01] * 9))                     # a pair not computed (missing arm)
  assert a.reading.iloc[-1] == 'not computed' and fu14.generality(a).startswith('not shown')
  assert a[a.pair == 'HW(V6f,Gf)'].note.iloc[0].startswith('known at specification')


def seed_pairs(name, rates, targets=60, per=5, seed=0):
  """Synthetic seed pairs: on each target, per seed pairs whose flips have the given rate per statistic."""
  rng = np.random.default_rng(seed)
  rows = []
  for t in range(targets):
    for s in range(per):
      rows.append({'contrast': name, 'target': f'T{t}', 'seed_a': s + 1, 'seed_b': s + 1,
                   **{st: int(rng.random() < r) for st, r in rates.items()}})
  return pd.DataFrame(rows)


RATES = lambda x: {'flip_top1': x, 'flip_success': x / 4, 'flip_matters': x / 6}  # noqa: E731


def test_d_f_with_the_float32_recompilation_set_to_zero_is_an_upper_bound():
  hwf = seed_pairs('HWf', RATES(0.05), seed=1)
  rcf = seed_pairs('RCf', RATES(0.3), seed=2)
  hw_, rc_ = seed_pairs('HW', RATES(0.4), seed=3), seed_pairs('RC', RATES(0.05), seed=4)
  bound = fu13.d_f(hwf, fu14.zero_like(hwf), hw_, rc_, 'flip_top1', 2026, 400)
  exact = fu13.d_f(hwf, rcf, hw_, rc_, 'flip_top1', 2026, 400)
  assert bound['estimate'] >= exact['estimate'] and bound['ci_high'] >= exact['ci_high']
  assert bound['rc_f'] == 0.0 and fu13.f_reading(bound['ci_high']) == 'reduced'      # a reduced bound: D_F reduced too


def test_d_f_table_reads_each_primary_pair(monkeypatch):
  sp = pd.concat([seed_pairs(fu14.hw(*p), RATES(0.01), seed=i) for i, p in enumerate(fu14.PAIRS)]
                 + [seed_pairs('RC(Gf)', RATES(0.0), seed=20)], ignore_index=True)
  ref = {'main': pd.concat([seed_pairs('HW(V5,V6)', RATES(0.2), seed=30), seed_pairs('RC(V5)', RATES(0.0), seed=31),
                            seed_pairs('HW(V6,G)', RATES(0.2), seed=32), seed_pairs('RC(G)', RATES(0.07), seed=33)]),
         'h100': pd.concat([seed_pairs('HW(H,A)', RATES(0.2), seed=34), seed_pairs('RC(H)', RATES(0.09), seed=35)]),
         'a100': seed_pairs('HW(G,A)', RATES(0.07), seed=36)}
  t = fu14.d_f_table(sp, ref, Args)
  top = t[t.statistic == 'flip_top1'].set_index('pair')
  assert list(top.index) == [fu14.hw(*p) for p in fu14.PRIMARY]
  assert top.loc['HW(V5f,V6f)', 'reading'] == 'reduced' and top.loc['HW(Hf,Af)', 'reading'] == 'reduced'
  assert top.loc['HW(Hf,Af)', 'float32_recompilation'].startswith('set to 0')
  assert top.loc['HW(Gf,Af)', 'float32_recompilation'] == 'RC(Gf)'
  assert set(t.reading) <= {'reduced', 'not reduced'}


def test_settings_check_reads_the_run_folders(tmp_path):
  def proc(cond, bf16, env, device):
    run = tmp_path / f's_{cond}' / 'run'
    (run / 'T1').mkdir(parents=True)
    (run / 'T1' / 'af3_entry_stats.json').write_text(json.dumps({'fu13': {'global_config_bfloat16': bf16}} if bf16 else {}))
    (run / 'session.json').write_text(json.dumps({'xla_env': env}))
    return {'session': str(run.relative_to(tmp_path)), 'target': 'T1', 'seed': 1, 'repetition': 'r1',
            'condition': cond, 'device_kind': device}
  d = pd.DataFrame([proc('Af', 'none', {'JAX_DEFAULT_MATMUL_PRECISION': 'highest'}, 'NVIDIA A100-SXM4-80GB'),
                    proc('Hd', None, {'XLA_FLAGS': '--xla_gpu_enable_triton_gemm=false --xla_gpu_deterministic_ops=true'},
                         'NVIDIA H100 80GB HBM3'),
                    proc('V5f', 'all', {'JAX_DEFAULT_MATMUL_PRECISION': 'highest'}, 'TPU v5 lite')])
  rows, detail = fu14.check_settings(d, root=tmp_path)
  res = {r['condition']: r['detail'] for r in rows}
  assert res['Af'] == 'pass' and res['Hd'] == 'pass' and res['V5f'].startswith('reported')
  assert len(detail) == 1 and 'bfloat16 all' in detail[0]['process']


def test_memory_table_and_completeness():
  ok = pd.DataFrame([{'condition': 'Af', 'repetition': 'r1', 'seed': s, 'target': f'T{t}', 'sample': k}
                     for s in range(1, 6) for t in range(60) for k in range(5)])
  comp = fu14.completeness(ok)
  assert comp[comp.condition == 'Af'].status.eq('complete').all()
  assert comp[comp.condition == 'Ad'].status.eq('not run').all()
  proc = pd.DataFrame([{'condition': 'V5f', 'bucket': 1024, 'compiled_peak_bytes': 8 * 2 ** 30, 'compiled_args_bytes':
                        2 ** 30, 'device_bytes_limit': 15.75 * 2 ** 30, 'exit_code': 0}])
  m = fu14.memory(proc)
  assert m.max_compiled_gib.iloc[0] == 9.0 and round(m.fraction_of_limit.iloc[0], 3) == round(9 / 15.75, 3)


# ---------------------------------------------------------------- configurations, plans, config diff

def test_new_configurations_are_their_sources_plus_the_follow_up_14_settings_only():
  c = read_configs(REPO / 'harness' / 'configs.yaml')
  strip = lambda x: {k: v for k, v in x.items() if k != 'description'}  # noqa: E731
  for new, base in (('a100_xla_fu14_f32', 'a100_xla'), ('h100_xla_fu14_f32', 'h100_xla')):
    cfg = strip(c[new])
    env = dict(cfg.pop('env'))
    assert env.pop('AF3_FU13_BFLOAT16') == 'none' and env.pop('JAX_DEFAULT_MATMUL_PRECISION') == 'highest'
    assert {**cfg, 'env': env} == strip(c[base]), new
    fu13_probe.settings(c[new]['env'])                  # the probe accepts them
  for new, base in (('a100_xla_det', 'a100_xla'), ('h100_xla_det', 'h100_xla')):
    cfg = strip(c[new])
    env = dict(cfg.pop('env'))
    assert env['XLA_FLAGS'] == c[base]['env']['XLA_FLAGS'] + ' --xla_gpu_deterministic_ops=true'
    env['XLA_FLAGS'] = c[base]['env']['XLA_FLAGS']
    assert {**cfg, 'env': env} == strip(c[base]), new
  assert c['l4_xla_det']['env']['XLA_FLAGS'] == c['l4_xla']['env']['XLA_FLAGS'] + ' --xla_gpu_deterministic_ops=true'


def test_plans_are_up_to_date_and_match_the_specification():
  out = subprocess.run([sys.executable, str(REPO / 'harness' / 'plans' / 'make_fu14_plans.py'), '--check'],
                       capture_output=True, text=True)
  assert out.returncode == 0, out.stdout + out.stderr
  parts = pd.read_csv(REPO / 'harness' / 'plans' / 'fu14_parts.csv')
  assert parts.groupby('arm').processes.sum().to_dict() == {'V5f': 300, 'V5f_r2': 10, 'Af': 300, 'Hf': 300, 'Ad': 100, 'Hd': 100}
  prov = parts.groupby('platform').provisioning.unique().apply(list).to_dict()
  assert prov == {'v5e': ['standard'], 'a100': ['spot'], 'h100': ['spot']}
  for arm, slots in (('V5f', 4), ('Af', 5), ('Hf', 5), ('Ad', 5), ('Hd', 5)):
    assert (parts.arm == arm).sum() % slots == 0, arm                   # parts in multiples of the slots (item f)
  f32 = parts[parts.arm.isin(['V5f', 'V5f_r2', 'Af', 'Hf'])]
  assert f32.expected_usd_planned.sum() < 300                          # the cap of item g at the expected factor
  assert parts[parts.arm.isin(['Ad', 'Hd'])].worst_usd_planned.sum() <= 62  # item b2's own cap
  assert (parts.worst_vm_h <= 8.5).all()
  configs = read_configs(REPO / 'harness' / 'configs.yaml')
  for p in sorted((REPO / 'harness' / 'plans' / 'f32').glob('*.yaml')):
    t = p.read_text()
    assert ('label: f32\n' in t and 'launch_by: 2026-11-18\n' in t and 'finish_by: 2026-11-20\n' in t
            and 'eigen_source: archive\n' in t), p
    assert re.search(r'configs:\n  \w+: (\S+)', t).group(1) in configs, p
    assert ('rep_start: 2\n' in t) == ('v5f_r2' in p.name), p
    assert ('fresh_reps: 2\n' in t) == ('_ad_' in p.name or '_hd_' in p.name), p


def test_config_diff_finds_nothing_beyond_the_allowed_settings(tmp_path):
  out = subprocess.run([sys.executable, str(REPO / 'harness' / 'plans' / 'config_diff_fu14.py'), '--out',
                        str(tmp_path / 'd.md')], capture_output=True, text=True, cwd=REPO)
  if 'No such file' in out.stderr or 'FileNotFoundError' in out.stderr:
    pytest.skip('source sessions not here')
  assert out.returncode == 0, out.stdout[-2000:] + out.stderr[-2000:]
  assert (tmp_path / 'd.md').read_text().rstrip().endswith('Settings that differ beyond the allowed ones: 0')


@pytest.mark.parametrize('script', ['make_fu7_10_plans.py', 'make_fu10_plans.py', 'make_fu11_12_plans.py',
                                    'make_fu13_plans.py'])
def test_earlier_plans_are_unchanged(script):
  out = subprocess.run([sys.executable, str(REPO / 'harness' / 'plans' / script), '--check'], capture_output=True, text=True)
  assert out.returncode == 0 and 'up to date' in out.stdout, out.stdout + out.stderr


# ---------------------------------------------------------------- budget rule and stop rule

def budget(args, tmp_path, now='2026-11-13T18:00:00Z', marker=None):
  env = dict(os.environ, NOW=now, RESULTS_AF3=str(tmp_path / 'af3'), F32_MARKER=str(marker or tmp_path / 'marker.txt'))
  return subprocess.run([sys.executable, str(REPO / 'scripts' / 'monitor' / 'fu14_budget.py')] + args,
                        capture_output=True, text=True, cwd=REPO, env=env)


def session(tmp_path, platform, part, start, end=None, code='0', prov='spot'):
  s = tmp_path / 'af3' / f'{start.replace("-", "").replace(":", "")}_{platform}_{part}'
  s.mkdir(parents=True)
  ev = f'{start} launch: session\n{start} vm: running\n'
  if end:
    ev += f'{end} delete: done\n{end} launcher: exit {code}\n'
  (s / 'launcher_events.txt').write_text(ev)
  (s / 'launch_record.txt').write_text(f'PROVISIONING={prov}\n')
  return s


def test_budget_check_go_floor_cap_dates_and_det(tmp_path):
  (tmp_path / 'af3').mkdir()
  r = budget(['check', 'a100', '326.57'], tmp_path)
  assert r.returncode == 0 and r.stdout.strip().endswith('GO'), r.stdout + r.stderr
  r = budget(['check', 'h100', '110'], tmp_path)
  assert r.returncode == 1 and '< 25 USD' in r.stdout
  r = budget(['check', 'v5e', '326.57'], tmp_path, now='2026-11-19T00:00:00Z')
  assert r.returncode == 1 and 'launch after 2026-11-18' in r.stdout
  r = budget(['check', 'det', '80'], tmp_path)                              # below worst case + 25
  assert r.returncode == 1 and 'worst case of item b2' in r.stdout
  assert budget(['check', 'det', '200'], tmp_path).returncode == 0
  session(tmp_path, 'h100', 'f32_hf_01', '2026-11-13T18:00:00Z')            # a float32 session still running
  r = budget(['check', 'det', '200'], tmp_path, now='2026-11-13T19:00:00Z')
  assert r.returncode == 1 and 'float32 arms have not ended' in r.stdout
  assert budget(['check', 'x', '1'], tmp_path).returncode == 2


def test_guard_stops_at_the_marker_and_at_the_cap(tmp_path):
  (tmp_path / 'af3').mkdir()
  r = budget(['guard', 'harness/plans/f32/f32_af_01.yaml'], tmp_path)
  assert r.returncode == 0 and r.stdout.strip().endswith('GO'), r.stdout + r.stderr
  marker = tmp_path / 'marker.txt'
  marker.write_text('stopped\n')
  r = budget(['guard', 'harness/plans/f32/f32_af_02.yaml'], tmp_path, marker=marker)
  assert r.returncode == 1 and 'marker' in r.stdout
  marker.unlink()
  # spend so far: a v5e session that ran 200 h on demand (an absurd overrun) pushes the projection over 300 USD
  session(tmp_path, 'v5e', 'f32_v5f_01', '2026-11-05T00:00:00Z', end='2026-11-13T08:00:00Z', prov='standard')
  r = budget(['guard', 'harness/plans/f32/f32_af_03.yaml'], tmp_path)
  assert r.returncode == 1 and 'projected float32 spend' in r.stdout
  r = budget(['guard', 'harness/plans/f32/f32_ad_01.yaml'], tmp_path)        # item b2 parts are not under this cap
  assert r.returncode == 0


def test_guarded_launcher_execs_the_launcher_on_go_and_writes_the_marker_on_no_go(tmp_path):
  stub = tmp_path / 'stub.sh'
  stub.write_text('echo "stub launcher PLAN=$PLAN"\n')
  (tmp_path / 'af3').mkdir()
  env = dict(os.environ, PLAN='harness/plans/f32/f32_af_01.yaml', PLATFORM='a100', NOW='2026-11-13T18:00:00Z',
             RESULTS_AF3=str(tmp_path / 'af3'), F32_MARKER=str(tmp_path / 'marker.txt'),
             F32_GUARD_LOG=str(tmp_path / 'guard.tsv'), F32_LAUNCHER=str(stub))
  r = subprocess.run(['bash', str(REPO / 'scripts' / 'monitor' / 'fu14_guarded_launch.sh')], capture_output=True,
                     text=True, cwd=REPO, env=env)
  assert r.returncode == 0 and 'stub launcher PLAN=harness/plans/f32/f32_af_01.yaml' in r.stdout, r.stdout + r.stderr
  assert not (tmp_path / 'marker.txt').exists() and (tmp_path / 'guard.tsv').read_text().count('\n') == 1
  r = subprocess.run(['bash', str(REPO / 'scripts' / 'monitor' / 'fu14_guarded_launch.sh')], capture_output=True,
                     text=True, cwd=REPO, env=dict(env, NOW='2026-11-19T00:00:00Z'))
  assert r.returncode == 3 and 'stub launcher' not in r.stdout and 'NO GO' in r.stdout
  assert 'launch after 2026-11-18' in (tmp_path / 'marker.txt').read_text()


def test_credit_check_script_usage():
  r = subprocess.run(['bash', str(REPO / 'scripts' / 'monitor' / 'credit_check_fu14.sh'), 'v6e', '1'], capture_output=True,
                     text=True, cwd=REPO)
  assert r.returncode == 2


# ---------------------------------------------------------------- end to end, on relabelled earlier rows

PRIMARY = REPO / 'results' / 'analysis' / 'main_20261014' / 'primary_v2'
A100 = REPO / 'results' / 'analysis' / 'a100_20261018'
H100 = REPO / 'results' / 'analysis' / 'h100_20261104'
FU13 = REPO / 'results' / 'analysis' / 'fu13_20261112'


@pytest.mark.skipif(not all((p / 'samples.csv').is_file() for p in (PRIMARY, A100, H100, FU13)),
                    reason='earlier analysis outputs not here')
def test_main_runs_end_to_end_on_relabelled_rows(tmp_path):
  """fu14.main on earlier rows relabelled as follow-up 14 conditions (V5 as V5f, A as Af and Ad, H as Hf and Hd);
  checks that every output is written, the checks run and the readings take their allowed values. No follow-up 14
  output exists or is read."""
  pilot = ['7U3J', '7NP6', '7BTT']
  others = ['8G6P', '7LCU']
  rows = []

  def take(path, label, platform, rep, seeds, config, as_platform, rep_as=None, targets=pilot + others):
    m = pd.read_csv(path, low_memory=False)
    g = m[(m.label == label) & (m.platform == platform) & (m.repetition == rep) & m.seed.isin(seeds)
          & m.target.isin(targets) & m['sample'].notna()].copy()
    g['label'], g['config'], g['platform'] = 'f32', config, as_platform
    if rep_as:
      g['repetition'] = rep_as
    rows.append(g)
  take(PRIMARY / 'samples.csv', 'main', 'V5', 'r1', [1, 2, 3, 4, 5], 'tpu_xla_fu13_f32', 'V5h')
  take(PRIMARY / 'samples.csv', 'main', 'V5', 'r1', [1], 'tpu_xla_fu13_f32', 'V5h', rep_as='r2', targets=pilot)
  take(A100 / 'samples.csv', 'a100', 'A', 'r1', [1, 2, 3, 4, 5], 'a100_xla_fu14_f32', 'Ah')
  take(H100 / 'samples.csv', 'h100', 'H', 'r1', [1, 2, 3, 4, 5], 'h100_xla_fu14_f32', 'Hh')
  take(A100 / 'samples.csv', 'a100', 'A', 'r1', [1, 2, 3, 4, 5], 'a100_xla_det', 'A', targets=pilot)
  take(A100 / 'samples.csv', 'a100', 'A', 'r1', [1, 2, 3, 4, 5], 'a100_xla_det', 'A', rep_as='r2', targets=pilot)
  d = pd.concat(rows, ignore_index=True)
  d.to_csv(tmp_path / 'samples.csv', index=False)
  out = tmp_path / 'out'
  r = subprocess.run([sys.executable, str(REPO / 'analysis' / 'fu14.py'), '--samples', str(tmp_path / 'samples.csv'),
                      '--out', str(out), '--replicates', '200'], capture_output=True, text=True, cwd=REPO)
  assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-3000:]
  for name in ('checks', 'completeness', 'contrasts', 'agree', 'd_f', 'secondary', 'success', 'det_identity', 'cost',
               'memory', 'pairs_seed'):
    assert (out / f'{name}.csv').is_file(), name
  chk = pd.read_csv(out / 'checks.csv')
  assert (chk[chk.check.str.startswith(('1 ', '2 '))].detail == 'pass').all()
  assert chk[chk.check.str.startswith('5 ')].detail.iloc[0] == 'pass'      # relabelled V5 r1 against itself
  assert chk[chk.check.str.startswith('4 ')].detail.str.startswith('reported').any()   # main-study runs: no fu13 key
  agree = pd.read_csv(out / 'agree.csv')
  assert len(agree) == 10 and set(agree.reading) <= {'agree', 'agreement not shown', 'not computed'}
  dft = pd.read_csv(out / 'd_f.csv')
  assert set(dft.pair) == {fu14.hw(*p) for p in fu14.PRIMARY} and set(dft.reading) <= {'reduced', 'not reduced'}
  rep = (out / 'report.md').read_text()
  assert "follow-up 13's f_reading.csv" in rep        # with 200 replicates the interval differs; see the test below
  det = pd.read_csv(out / 'det_identity.csv')
  assert det.set_index('condition').loc['Hd', 'value'] == 'not run'
  assert '## Agree' in rep and 'Generality (item e 3)' in rep and '## Item b2' in rep


@pytest.mark.skipif(not (FU13 / 'samples.csv').is_file(), reason='follow-up 13 outputs not here')
def test_follow_up_13_pairs_are_rebuilt_exactly():
  """The seed pairs of HW(V6f,Gf) and RC(Gf) that fu14 builds from follow-up 13's samples equal follow-up 13's own
  pairs_seed.csv, so D_F of (V6f,Gf) with 10,000 replicates and seed 2026 repeats follow-up 13's value."""
  f13 = fu14.load_fu13(FU13 / 'samples.csv')
  import followup34 as f34
  sp, _, _, _ = f34.build(f34.sample_rows(f13), [('HW(V6f,Gf)', 'HW', ('V6f', 'r1'), ('Gf', 'r1'), True),
                                                ('RC(Gf)', 'RC', ('Gf', 'r1'), ('Gf', 'r2'), False)])
  old = pd.read_csv(FU13 / 'pairs_seed.csv')
  cols = ['contrast', 'target', 'seed_a', 'seed_b', 'flip_top1', 'flip_success', 'flip_matters']
  for name in ('HW(V6f,Gf)', 'RC(Gf)'):
    a = sp[sp.contrast == name][cols].sort_values(cols[1:3]).reset_index(drop=True).astype(str)
    b = old[old.contrast == name][cols].sort_values(cols[1:3]).reset_index(drop=True).astype(str)
    assert len(a) == 300 and a.equals(b), name
