"""Follow-up 15 (label rcx; notes/analysis_plan.md section 10, row of 2026-11-14): synthetic tests of the readings of
analysis/rcx.py (bit-identity, the D_F sensitivity, the stack check), of the plans and the config diff, of the budget
rule, the gate of part A3 and the stop rule (the guarded launcher), and an end-to-end run on relabelled earlier rows.
No follow-up 15 output exists or is read."""
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
import fu14  # noqa: E402
import rcx  # noqa: E402
from run_af3 import read_configs  # noqa: E402

PARTS = REPO / 'harness' / 'plans' / 'rcx_parts.csv'


class Args:
  seed, replicates = 2026, 400


# ---------------------------------------------------------------- design

def test_conditions_and_contrasts_are_the_specification():
  assert rcx.CONFIG == {'a100_xla_fu14_f32': 'Af', 'h100_xla_fu14_f32': 'Hf', 'l4_triton': 'GT'}
  assert rcx.RCS == [('RC(Af)', 'Af'), ('RC(Hf)', 'Hf'), ('RC(GT)', 'GT')]
  assert rcx.TARGETS == {'Af': 'pilot', 'Hf': 'pilot', 'GT': 'all'} and rcx.SEEDS == [1, 2, 3, 4, 5]
  assert rcx.MACHINE == {'Af': 'a2-ultragpu-1g', 'Hf': 'a3-highgpu-1g', 'GT': 'g2-standard-8'}


# ---------------------------------------------------------------- bit-identity

def write_outputs(root, t, seed, full, stamp='2026-11-15', coords='1.000'):
  d = root / t
  for k in range(5):
    s = d / f'seed-{seed}_sample-{k}'
    s.mkdir(parents=True, exist_ok=True)
    (s / f'{t}_seed-{seed}_sample-{k}_model.cif').write_text(
        f'data_x\n_ma_model_list.model_group_name "AF3 {stamp}"\nATOM {coords}\n')
    (s / f'{t}_seed-{seed}_sample-{k}_summary_confidences.json').write_text('{"iptm": 0.9}')
    if full:
      (s / f'{t}_seed-{seed}_sample-{k}_confidences.json').write_text('{"pae": [1]}')
  (d / f'{t}_model.cif').write_text(f'data_x\n_ma_model_list.model_group_name "AF3 {stamp}"\nATOM {coords}\n')
  (d / f'{t}_ranking_scores.csv').write_text('seed,sample,ranking_score\n')
  (d / f'{t}_summary_confidences.json').write_text('{"iptm": 0.9}')
  if full:
    (d / f'{t}_confidences.json').write_text('{"pae": [1]}')
    (d / f'{t}_data.json').write_text('{}')
  return d


def test_light_identity_masks_the_timestamp_and_finds_a_coordinate_change(tmp_path):
  a = write_outputs(tmp_path / 'a', 'T1', 1, full=False, stamp='2026-10-13')
  b = write_outputs(tmp_path / 'b', 'T1', 1, full=True, stamp='2026-11-15')      # extra full files are not compared
  assert rcx.light_identical(a, b, 1)['status'] == 'identical'
  c = write_outputs(tmp_path / 'c', 'T1', 1, full=False, coords='1.001')
  r = rcx.light_identical(a, c, 1)
  assert r['status'] == 'differ' and 'T1_model.cif' in r['differ']
  (b / 'seed-1_sample-3' / 'T1_seed-1_sample-3_summary_confidences.json').unlink()
  assert rcx.light_identical(a, b, 1)['status'] == 'missing'


def test_identity_reading_bit_identical_not_bit_identical_or_not_determined():
  u = lambda *ss: [{'status': s} for s in ss]  # noqa: E731
  assert rcx.identity_reading(u('identical', 'identical')) == 'recompilation bit-identical'
  assert rcx.identity_reading(u('identical', 'differ', 'missing')).startswith('not bit-identical (1 of 3')
  assert rcx.identity_reading(u('identical', 'missing')).startswith('not determined (1 of 2')
  assert rcx.identity_reading([]) == 'no unit'


def test_strict_identity_needs_every_output_file(tmp_path):
  import contrasts
  a = write_outputs(tmp_path / 'a', 'T1', 2, full=False)
  b = write_outputs(tmp_path / 'b', 'T1', 2, full=True)
  assert contrasts.outputs_identical(a, b, 2)['status'] == 'missing'           # why Af r1 and Hf r1 are fetched in full
  a = write_outputs(tmp_path / 'a', 'T1', 2, full=True, stamp='2026-11-12')
  assert contrasts.outputs_identical(a, b, 2)['status'] == 'identical'


# ---------------------------------------------------------------- D_F sensitivity

def seed_pairs(name, rates, targets, per=5, seed=0):
  rng = np.random.default_rng(seed)
  rows = []
  for t in targets:
    for s in range(per):
      rows.append({'contrast': name, 'target': t, 'seed_a': s + 1, 'seed_b': s + 1,
                   **{st: int(rng.random() < r) for st, r in rates.items()}})
  return pd.DataFrame(rows)


RATES = lambda x: {'flip_top1': x, 'flip_success': x / 4, 'flip_matters': x / 6}  # noqa: E731
ALL = [f'T{i:02d}' for i in range(60)]
PILOT = ALL[:10]


def test_d_f_partial_with_rc_zero_is_the_registered_d_f_draw_for_draw():
  hwf = seed_pairs('HW(Hf,Af)', RATES(0.02), ALL, seed=1)
  hw, rc = seed_pairs('HW(H,A)', RATES(0.2), ALL, seed=2), seed_pairs('RC(H)', RATES(0.09), ALL, seed=3)
  zero = seed_pairs('RC(Hf)', RATES(0.0), PILOT, seed=4)
  for st in ('flip_top1', 'flip_success'):
    reg = fu13.d_f(hwf, fu14.zero_like(hwf), hw, rc, st, 2026, 400)
    part = rcx.d_f_partial(hwf, zero, hw, rc, st, 2026, 400)
    assert part['n_targets'] == 60
    for k in ('estimate', 'ci_low', 'ci_high'):
      assert part[k] == pytest.approx(reg[k], abs=1e-12), (st, k)


def test_d_f_partial_measured_term_lowers_d_f_and_counts_replicates_without_the_pilot_targets():
  hwf = seed_pairs('HW(Hf,Af)', RATES(0.02), ALL, seed=1)
  hw, rc = seed_pairs('HW(H,A)', RATES(0.2), ALL, seed=2), seed_pairs('RC(H)', RATES(0.09), ALL, seed=3)
  rcf = seed_pairs('RC(Hf)', RATES(0.3), PILOT, seed=5)
  reg = fu13.d_f(hwf, fu14.zero_like(hwf), hw, rc, 'flip_top1', 2026, 400)
  part = rcx.d_f_partial(hwf, rcf, hw, rc, 'flip_top1', 2026, 400)
  assert part['rc_f'] > 0 and part['estimate'] < reg['estimate'] and part['ci_high'] <= reg['ci_high'] + 1e-12
  one = seed_pairs('RC(Hf)', RATES(0.5), ['T00'], seed=6)                       # one pilot target: often not drawn
  r = rcx.d_f_partial(hwf, one, hw, rc, 'flip_top1', 2026, 400)
  assert 0 < r['replicates_without_rc_f'] < 400


def test_d_f_sensitivity_table_has_the_four_versions_per_statistic():
  f14 = seed_pairs('HW(Hf,Af)', RATES(0.02), ALL, seed=1)
  h = pd.concat([seed_pairs('HW(H,A)', RATES(0.2), ALL, seed=2), seed_pairs('RC(H)', RATES(0.09), ALL, seed=3)])
  sp = seed_pairs('RC(Hf)', RATES(0.0), PILOT, seed=4)
  t = rcx.d_f_sensitivity(sp, f14, h, PILOT, Args)
  assert len(t) == 12 and set(t.reading) <= {'reduced', 'not reduced'}
  v = t[t.statistic == 'flip_top1'].set_index('version')
  assert list(v.index) == ['registered: 60 targets, RC(Hf) set to 0', '(i) 60 targets, RC(Hf) measured on the pilot targets',
                           '(ii) 10 pilot targets, RC(Hf) measured', '(ii) 10 pilot targets, RC(Hf) set to 0']
  assert v.iloc[0].estimate == pytest.approx(v.iloc[1].estimate) and v.iloc[2].n_targets == 10
  assert rcx.d_f_sensitivity(sp.iloc[0:0], f14, h, PILOT, Args).empty          # RC(Hf) not measured


# ---------------------------------------------------------------- settings and stack

def fake_session(root, name, driver='580.178.04', python='3.12.16', af3='3.0.5.dev0+g85c4d2050.d20261115', freeze='a==1\n'):
  s = root / name
  run = s / 'x_rep2_seed1_fresh'
  run.mkdir(parents=True)
  (s / 'device_check.txt').write_text(f'| NVIDIA-SMI {driver}   Driver Version: {driver}     CUDA Version: 13.0 |\n')
  (run / 'session.json').write_text(json.dumps({'host': {'platform': 'Linux-6.8.0-1070-gcp-x86_64-with-glibc2.35'},
                                                'probe': {'python': python}}))
  (s / 'setup.json').write_text(json.dumps({'packages': {'alphafold3': af3, 'jax': '0.10.2'}}))
  (s / 'pip_freeze.txt').write_text(freeze)
  return s


def test_stack_of_strips_the_build_date_and_check_stack_reports_differences(tmp_path, monkeypatch):
  af3 = tmp_path / 'af3'
  fake_session(af3, '20261013T143616Z_l4_main_gt_01', python='3.12.15', af3='3.0.5.dev0+g85c4d2050.d20261013')
  fake_session(af3, '20261115T100000Z_l4_rcx_gt_r2_01')
  a = rcx.stack_of(af3 / '20261013T143616Z_l4_main_gt_01')
  b = rcx.stack_of(af3 / '20261115T100000Z_l4_rcx_gt_r2_01')
  assert a['driver'] == '580.178.04' and a['packages'] == b['packages'] and a['pip_freeze_sha256'] == b['pip_freeze_sha256']
  monkeypatch.setattr(rcx, 'r1_sessions', lambda c, args: ['20261013T143616Z_l4_main_gt_01'])
  d = pd.DataFrame([{'condition': 'GT', 'session': 'results/af3/20261115T100000Z_l4_rcx_gt_r2_01/x_rep2_seed1_fresh'}])
  rows, table = rcx.check_stack(d, None, root=af3)
  assert rows[0]['detail'].startswith('reported, differs in python: 3.12.16 (r1 3.12.15)')
  assert set(table[~table.same].field) == {'python'}
  fake_session(af3, '20261115T110000Z_l4_rcx_gt_r2_02', python='3.12.15', driver='580.999.99')
  d = pd.DataFrame([{'condition': 'GT', 'session': 'results/af3/20261115T110000Z_l4_rcx_gt_r2_02/x'}])
  rows, _ = rcx.check_stack(d, None, root=af3)
  assert 'driver: 580.999.99' in rows[0]['detail'] and 'python' not in rows[0]['detail']


def test_settings_check_float32_on_the_gpus_and_bfloat16_on_gt(tmp_path):
  def proc(cond, bf16, env, device):
    run = tmp_path / f's_{cond}' / 'run'
    (run / 'T1').mkdir(parents=True)
    (run / 'T1' / 'af3_entry_stats.json').write_text(json.dumps({'fu13': {'global_config_bfloat16': bf16}} if bf16 else {}))
    (run / 'session.json').write_text(json.dumps({'xla_env': env}))
    return {'session': str(run.relative_to(tmp_path)), 'target': 'T1', 'seed': 1, 'repetition': 'r2',
            'condition': cond, 'device_kind': device}
  d = pd.DataFrame([proc('Af', 'none', {'JAX_DEFAULT_MATMUL_PRECISION': 'highest'}, 'NVIDIA A100-SXM4-80GB'),
                    proc('Hf', None, {'JAX_DEFAULT_MATMUL_PRECISION': 'highest'}, 'NVIDIA H100 80GB HBM3'),
                    proc('GT', None, {'XLA_FLAGS': '--xla_gpu_enable_triton_gemm=false'}, 'NVIDIA L4')])
  rows, detail = rcx.check_settings(d, root=tmp_path)
  res = {r['condition']: r['detail'] for r in rows}
  assert res['Af'] == 'pass' and res['GT'] == 'pass' and res['Hf'].startswith('reported')
  assert len(detail) == 1


def test_completeness_per_arm():
  ok = pd.DataFrame([{'condition': 'Af', 'repetition': 'r2', 'seed': s, 'target': t, 'sample': k}
                     for s in range(1, 6) for t in PILOT for k in range(5)])
  comp = rcx.completeness(ok, PILOT, ALL)
  assert comp[comp.condition == 'Af'].status.eq('complete').all()
  assert comp[comp.condition == 'GT'].status.eq('not run').all() and comp[comp.condition == 'GT'].planned_targets.eq(60).all()


# ---------------------------------------------------------------- plans and config diff

def test_plans_are_up_to_date_and_match_the_specification():
  out = subprocess.run([sys.executable, str(REPO / 'harness' / 'plans' / 'make_rcx_plans.py'), '--check'],
                       capture_output=True, text=True)
  assert out.returncode == 0, out.stdout + out.stderr
  parts = pd.read_csv(PARTS)
  assert parts.groupby('arm').processes.sum().to_dict() == {'Af_r2': 50, 'Hf_r2': 50, 'GT_r2': 300}
  assert parts.groupby('arm').targets.sum().to_dict() == {'Af_r2': 10, 'Hf_r2': 10, 'GT_r2': 60}
  assert set(parts.provisioning) == {'spot'}
  for arm, slots in (('Af_r2', 4), ('Hf_r2', 3), ('GT_r2', 8)):
    assert (parts.arm == arm).sum() % slots == 0, arm
  a4, a3 = parts[parts.row_part == 'A4'], parts[parts.row_part == 'A3']
  assert a4.worst_usd_planned.sum() + a3.worst_usd_planned.sum() <= 62          # the inclusion rule of part A3
  assert (parts.worst_vm_h <= 4.5).all()
  configs = read_configs(REPO / 'harness' / 'configs.yaml')
  manifest = pd.read_csv(REPO / 'inputs' / 'manifest.csv')
  pilot = set(manifest[manifest.pilot == 'yes'].pdb_id)
  seen = {'af': set(), 'hf': set(), 'gt': set()}
  for p in sorted((REPO / 'harness' / 'plans' / 'rcx').glob('*.yaml')):
    t = p.read_text()
    assert ('label: rcx\n' in t and 'launch_by: 2026-11-17\n' in t and 'finish_by: 2026-11-18\n' in t
            and 'eigen_source: archive\n' in t and 'rep_start: 2\n' in t and 'fresh_reps: 1\n' in t), p
    cfg = re.search(r'configs:\n  \w+: (\S+)', t).group(1)
    assert cfg in configs and cfg in rcx.CONFIG, p
    ts = set(re.search(r'default: \[([^\]]*)\]', t).group(1).split(', '))
    seen[p.name.split('_')[1]] |= ts
  assert seen['af'] == pilot and seen['hf'] == pilot and len(seen['gt']) == 60
  lines = (REPO / 'results' / 'parts_rcx_h100.txt').read_text() if (REPO / 'results' / 'parts_rcx_h100.txt').is_file() else ''
  assert 'us-east4' not in lines.split('\n', 1)[-1]


def test_configurations_are_unchanged_by_follow_up_15():
  out = subprocess.run(['git', 'diff', '--quiet', 'HEAD', '--', 'harness/configs.yaml'], cwd=REPO)
  assert out.returncode == 0


def test_config_diff_finds_nothing_beyond_the_allowed_settings(tmp_path):
  out = subprocess.run([sys.executable, str(REPO / 'harness' / 'plans' / 'config_diff_rcx.py'), '--out',
                        str(tmp_path / 'd.md')], capture_output=True, text=True, cwd=REPO)
  if 'No such file' in out.stderr or 'FileNotFoundError' in out.stderr:
    pytest.skip('r1 sessions not here')
  assert out.returncode == 0, out.stdout[-2000:] + out.stderr[-2000:]
  assert (tmp_path / 'd.md').read_text().rstrip().endswith('Settings that differ beyond the allowed ones: 0')


@pytest.mark.parametrize('script', ['make_fu13_plans.py', 'make_fu14_plans.py'])
def test_earlier_plans_are_unchanged(script):
  out = subprocess.run([sys.executable, str(REPO / 'harness' / 'plans' / script), '--check'], capture_output=True, text=True)
  assert out.returncode == 0 and 'up to date' in out.stdout, out.stdout + out.stderr


# ---------------------------------------------------------------- budget rule, gate of part A3, stop rule

def budget(args, tmp_path, now='2026-11-15T08:00:00Z', marker=None):
  env = dict(os.environ, NOW=now, RESULTS_AF3=str(tmp_path / 'af3'), RCX_MARKER=str(marker or tmp_path / 'marker.txt'))
  return subprocess.run([sys.executable, str(REPO / 'scripts' / 'monitor' / 'rcx_budget.py')] + args,
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


def a4_done(tmp_path, end='2026-11-15T09:00:00Z'):
  for p in pd.read_csv(PARTS).query('row_part == "A4"').itertuples():
    session(tmp_path, p.platform, p.part, '2026-11-15T08:00:00Z', end=end)


def test_budget_check_a4_floor_cap_dates(tmp_path):
  (tmp_path / 'af3').mkdir()
  for w in ('a100', 'h100'):
    r = budget(['check', w, '87'], tmp_path)
    assert r.returncode == 0 and r.stdout.strip().endswith('GO'), r.stdout + r.stderr
  r = budget(['check', 'h100', '40'], tmp_path)                                  # 40 - 20.14 < 25
  assert r.returncode == 1 and '< 25 USD' in r.stdout
  r = budget(['check', 'a100', '87'], tmp_path, now='2026-11-18T00:00:00Z')
  assert r.returncode == 1 and 'launch after 2026-11-17' in r.stdout
  assert budget(['check', 'v5e', '87'], tmp_path).returncode == 2


def test_gate_of_part_a3_waits_for_part_a4_and_its_spend(tmp_path):
  (tmp_path / 'af3').mkdir()
  r = budget(['check', 'l4', '87'], tmp_path)
  assert r.returncode == 1 and 'part A4 has not ended' in r.stdout
  a4_done(tmp_path)                                         # 7 parts, 1 h each at their Spot price: about 32 USD
  r = budget(['check', 'l4', '55'], tmp_path, now='2026-11-15T10:00:00Z')
  assert r.returncode == 0 and r.stdout.strip().endswith('GO'), r.stdout + r.stderr
  r = budget(['check', 'l4', '30'], tmp_path, now='2026-11-15T10:00:00Z')        # 30 - 13.10 < 25
  assert r.returncode == 1 and '< 25 USD' in r.stdout
  session(tmp_path, 'a100', 'rcx_af_r2_01', '2026-11-15T09:10:00Z', end='2026-11-15T15:00:00Z', code='1')  # spend 17.6
  r = budget(['check', 'l4', '87'], tmp_path, now='2026-11-15T16:00:00Z')
  assert r.returncode == 1 and 'projected spend' in r.stdout                     # about 32 + 17.6 + 13.1 > 62


def test_guard_marker_cap_and_order(tmp_path):
  (tmp_path / 'af3').mkdir()
  r = budget(['guard', 'harness/plans/rcx/rcx_af_r2_01.yaml'], tmp_path)
  assert r.returncode == 0 and r.stdout.strip().endswith('GO'), r.stdout + r.stderr
  r = budget(['guard', 'harness/plans/rcx/rcx_gt_r2_01.yaml'], tmp_path)
  assert r.returncode == 1 and 'part A4 has not ended' in r.stdout
  marker = tmp_path / 'marker.txt'
  marker.write_text('stopped\n')
  r = budget(['guard', 'harness/plans/rcx/rcx_hf_r2_01.yaml'], tmp_path, marker=marker)
  assert r.returncode == 1 and 'marker' in r.stdout
  marker.unlink()
  session(tmp_path, 'h100', 'rcx_hf_r2_02', '2026-11-15T00:00:00Z', end='2026-11-15T08:00:00Z', code='1')  # 52 USD lost
  r = budget(['guard', 'harness/plans/rcx/rcx_hf_r2_02.yaml'], tmp_path, now='2026-11-15T08:30:00Z')
  assert r.returncode == 1 and 'projected spend' in r.stdout
  assert budget(['guard', 'harness/plans/f32/f32_af_01.yaml'], tmp_path).returncode == 1
  r = budget(['spent'], tmp_path)
  assert r.returncode == 0 and float(r.stdout) == pytest.approx(8 * 6.538448, abs=0.01)


def test_guard_counts_part_a3_only_once_it_has_started(tmp_path):
  a4_done(tmp_path)
  r = budget(['guard', 'harness/plans/rcx/rcx_gt_r2_01.yaml'], tmp_path, now='2026-11-15T10:00:00Z')
  assert r.returncode == 0 and 'still needed by the other parts 0.00' in r.stdout, r.stdout


def guarded(tmp_path, plan, now):
  stub = tmp_path / 'stub.sh'
  stub.write_text('echo "stub launcher PLAN=$PLAN"\n')
  (tmp_path / 'af3').mkdir(exist_ok=True)
  env = dict(os.environ, PLAN=plan, PLATFORM='a100', NOW=now, RESULTS_AF3=str(tmp_path / 'af3'),
             RCX_MARKER=str(tmp_path / 'marker.txt'), RCX_GUARD_LOG=str(tmp_path / 'guard.tsv'), RCX_LAUNCHER=str(stub))
  return subprocess.run(['bash', str(REPO / 'scripts' / 'monitor' / 'rcx_guarded_launch.sh')], capture_output=True,
                        text=True, cwd=REPO, env=env)


def test_guarded_launcher_execs_on_go_and_writes_the_marker_only_for_the_cap_or_the_date(tmp_path):
  r = guarded(tmp_path, 'harness/plans/rcx/rcx_af_r2_01.yaml', '2026-11-15T08:00:00Z')
  assert r.returncode == 0 and 'stub launcher PLAN=harness/plans/rcx/rcx_af_r2_01.yaml' in r.stdout, r.stdout + r.stderr
  assert not (tmp_path / 'marker.txt').exists() and (tmp_path / 'guard.tsv').read_text().count('\n') == 1
  r = guarded(tmp_path, 'harness/plans/rcx/rcx_gt_r2_01.yaml', '2026-11-15T08:00:00Z')   # part A4 not ended
  assert r.returncode == 3 and 'stub launcher' not in r.stdout and not (tmp_path / 'marker.txt').exists()
  r = guarded(tmp_path, 'harness/plans/rcx/rcx_af_r2_02.yaml', '2026-11-18T00:00:00Z')
  assert r.returncode == 3 and 'launch after 2026-11-17' in (tmp_path / 'marker.txt').read_text()


def test_credit_check_script_usage():
  r = subprocess.run(['bash', str(REPO / 'scripts' / 'monitor' / 'credit_check_rcx.sh'), 'det', '1'], capture_output=True,
                     text=True, cwd=REPO)
  assert r.returncode == 2


# ---------------------------------------------------------------- end to end, on relabelled earlier rows

PRIMARY = REPO / 'results' / 'analysis' / 'main_20261014' / 'primary_v2'
FU14 = REPO / 'results' / 'analysis' / 'fu14_20261113b'
H100 = REPO / 'results' / 'analysis' / 'h100_20261104'


@pytest.mark.skipif(not all((p / 'samples.csv').is_file() for p in (PRIMARY, FU14, H100)),
                    reason='earlier analysis outputs not here')
def test_main_runs_end_to_end_on_relabelled_rows(tmp_path):
  """rcx.main on the r1 rows themselves relabelled as follow-up 15's r2 (Af and Hf of follow-up 14 on three pilot
  targets, GT of the main study on those and two more): every output is written, the checks run, the light identity of
  GT reads "recompilation bit-identical" (r2 is r1's own files), RC(GT) is 0, and the D_F sensitivity is computed."""
  pilot, others = ['7U3J', '7NP6', '7BTT'], ['8G6P', '7LCU']
  rows = []

  def take(path, label, platform, seeds, config, targets):
    m = pd.read_csv(path, low_memory=False)
    g = m[(m.label == label) & (m.platform == platform) & (m.repetition == 'r1') & m.seed.isin(seeds)
          & m.target.isin(targets) & m['sample'].notna()].copy()
    g['label'], g['config'], g['repetition'] = 'rcx', config, 'r2'
    rows.append(g)
  take(FU14 / 'samples.csv', 'f32', 'Ah', [1, 2, 3, 4, 5], 'a100_xla_fu14_f32', pilot)
  take(FU14 / 'samples.csv', 'f32', 'Hh', [1, 2, 3, 4, 5], 'h100_xla_fu14_f32', pilot)
  take(PRIMARY / 'samples.csv', 'main', 'GT', [1, 2, 3, 4, 5], 'l4_triton', pilot + others)
  d = pd.concat(rows, ignore_index=True)
  assert set(d.config) == set(rcx.CONFIG) and len(d) > 0
  d.to_csv(tmp_path / 'samples.csv', index=False)
  out = tmp_path / 'out'
  r = subprocess.run([sys.executable, str(REPO / 'analysis' / 'rcx.py'), '--samples', str(tmp_path / 'samples.csv'),
                      '--out', str(out), '--replicates', '200'], capture_output=True, text=True, cwd=REPO)
  assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-3000:]
  for name in ('checks', 'completeness', 'contrasts', 'identity', 'registered', 'rc_gt_vs_g', 'd_f_sensitivity', 'stack',
               'pairs_seed'):
    assert (out / f'{name}.csv').is_file(), name
  chk = pd.read_csv(out / 'checks.csv')
  assert (chk[chk.check.str.startswith(('1 ', '2'))].detail == 'pass').all()
  assert (chk[chk.check.str.startswith('4b')].detail == 'pass').all()           # r2 is r1: the same stack
  ident = pd.read_csv(out / 'identity.csv').set_index('condition')
  assert ident.loc['GT r2 against GT r1', 'detail'] == 'recompilation bit-identical'
  con = pd.read_csv(out / 'contrasts.csv')
  top = con[(con.contrast == 'RC(GT)') & (con.statistic == 'flip_top1')]
  assert top.estimate.iloc[0] == 0 and top.n.iloc[0] == 25
  assert set(pd.read_csv(out / 'registered.csv').contrast) == set(rcx.REGISTERED)
  dfs = pd.read_csv(out / 'd_f_sensitivity.csv')
  assert len(dfs) == 12 and set(dfs.reading) <= {'reduced', 'not reduced'}
  rep = (out / 'report.md').read_text()
  assert "follow-up 14's d_f.csv" in rep and '## RC(GT) against RC(G)' in rep


@pytest.mark.skipif(not all((p / 'pairs_seed.csv').is_file() for p in (FU14, H100)), reason='earlier outputs not here')
def test_registered_d_f_of_hf_af_is_reproduced_exactly():
  """With 10,000 replicates and seed 2026, the registered D_F(Hf,Af) recomputed from follow-up 14's and follow-up 9's
  seed pairs equals follow-up 14's d_f.csv, so the sensitivity of item e 2 uses the registered draws."""
  f14, h = pd.read_csv(FU14 / 'pairs_seed.csv'), pd.read_csv(H100 / 'pairs_seed.csv')
  sp = f14[f14.contrast == 'HW(Hf,Af)'].iloc[0:0].assign(contrast='RC(Hf)')
  sp = pd.concat([sp, seed_pairs('RC(Hf)', RATES(0.0), ['7U3J'])], ignore_index=True)

  class A:
    seed, replicates = 2026, 10000
  t = rcx.d_f_sensitivity(sp, f14, h, ['7U3J'], A)
  assert rcx.registered_d_f_equal(t, FU14) is True
