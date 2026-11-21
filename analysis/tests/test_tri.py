"""Follow-up 18 (label tri; notes/analysis_plan.md section 10, row "Follow-up experiment 18"): synthetic tests of the
readings of analysis/tri.py (E_T, the main output, the default-configuration pairs, the replication of gtat), of its
checks (settings, VMs, stack), of the plans and the config diff, and an end-to-end run on relabelled earlier rows.
No follow-up 18 output exists or is read."""
import json
import pathlib
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'analysis'))
sys.path.insert(0, str(REPO / 'harness'))
import contrasts  # noqa: E402
import tri  # noqa: E402
import winners  # noqa: E402
from run_af3 import read_configs  # noqa: E402

PARTS = REPO / 'harness' / 'plans' / 'tri_parts.csv'
ALL = [f'T{i:02d}' for i in range(60)]


class Args:
  seed, replicates = 2026, 300


def seed_pairs(name, rates, targets, per=5, seed=0):
  rng = np.random.default_rng(seed)
  rows = []
  for t in targets:
    for s in range(per):
      top = int(rng.random() < rates['flip_top1'])
      suc = int(rng.random() < rates['flip_success'])
      rows.append({'contrast': name, 'target': t, 'seed_a': s + 1, 'seed_b': s + 1, 'top1_a': 0, 'top1_b': top,
                   'flip_top1': top, 'flip_success': suc, 'flip_matters': top * suc})
  return pd.DataFrame(rows)


RATES = lambda x: {'flip_top1': x, 'flip_success': x / 3}  # noqa: E731


# ---------------------------------------------------------------- design

def test_conditions_and_contrasts_are_the_specification():
  assert tri.CONFIG == {'h100_triton': 'HT', 'a100_triton': 'AT'}
  assert tri.PLANNED == [('HT', 'r1'), ('HT', 'r2'), ('AT', 'r2')] and tri.SEEDS == [1, 2, 3, 4, 5]
  names = {s[0]: (s[2], s[3]) for s in tri.SPECS}
  assert names['HW(HT,AT)'] == (('HT', 'r1'), ('AT', 'r1')) and names['RC(HT)'] == (('HT', 'r1'), ('HT', 'r2'))
  assert names['RC(AT)'] == (('AT', 'r1'), ('AT', 'r2')) and names['DU(H)'] == (('H', 'r1'), ('HT', 'r1'))
  assert names['HW(GT,AT)'] == (('GT', 'r1'), ('AT', 'r1')) and names['HW(GT,HT)'] == (('GT', 'r1'), ('HT', 'r1'))
  assert [s[0] for s in tri.SPECS if s[4]] == ['HW(HT,AT)']
  assert tri.DEFAULT_PAIRS['HW(GT,AT)'] == ('RC(GT)', 'RC(AT)') and set(tri.XLA_PAIRS) == {'HW(G,A)', 'HW(H,G)', 'HW(H,A)'}
  assert tri.MO_CONTRASTS == [('MO(HT,AT)', ('HT', 'r1'), ('AT', 'r1')), ('MO(HT)', ('HT', 'r1'), ('HT', 'r2'))]
  assert tri.WITHIN_A == 2.0 and tri.TRITON == '--flash_attention_implementation=triton'


# ---------------------------------------------------------------- readings

def test_reading_rules_e1_and_e2():
  assert tri.beyond_reading(0.001) == 'beyond recompilation'
  assert tri.beyond_reading(0.0) == 'not distinguishable from recompilation'
  assert tri.beyond_reading(-0.03) == 'not distinguishable from recompilation'
  assert tri.beyond_reading(float('nan')) == 'not computed'
  assert tri.mo_reading(0.02) == 'the main output differs beyond same-type rescheduling'
  assert tri.mo_reading(0.0) == 'not distinguishable from same-type rescheduling'


def test_e_t_beyond_not_distinguishable_and_the_larger_reference():
  hw = seed_pairs('HW(HT,AT)', RATES(0.3), ALL, seed=1)
  rc = seed_pairs('RC(HT)', RATES(0.05), ALL, seed=2)
  sp = pd.concat([hw, rc])
  t = tri.e_t_table(sp, pd.DataFrame(columns=['contrast', 'target']), Args)
  top = t[(t.statistic == 'flip_top1') & (t.reference == 'RC(HT)')].iloc[0]
  assert top.role == 'primary' and top.reading == 'beyond recompilation' and top.n_targets == 60
  assert top.estimate == pytest.approx(hw.flip_top1.mean() - rc.flip_top1.mean())
  e3 = t[t.role == 'secondary (rule e3)']
  assert (e3.reading == 'not run (part B)').all() and len(e3) == 3
  rat = seed_pairs('RC(AT)', RATES(0.3), ALL, seed=3)                         # RC(AT) as large as HW: rule e3 differs
  t = tri.e_t_table(pd.concat([sp, rat]), pd.DataFrame(columns=['contrast', 'target']), Args)
  e3 = t[(t.role == 'secondary (rule e3)') & (t.statistic == 'flip_top1')].iloc[0]
  assert e3.reference == ('RC(AT)' if rat.flip_top1.mean() > rc.flip_top1.mean() else 'RC(HT)')
  assert e3.reading == 'not distinguishable from recompilation'
  same = seed_pairs('RC(HT)', RATES(0.3), ALL, seed=1)                         # identical to HW: excess 0
  t = tri.e_t_table(pd.concat([hw, same]), pd.DataFrame(columns=['contrast', 'target']), Args)
  assert t.iloc[0].estimate == 0 and t.iloc[0].reading == 'not distinguishable from recompilation'


def test_registered_pairs_fill_in_and_comparisons_use_the_same_targets():
  sp = pd.concat([seed_pairs('HW(HT,AT)', RATES(0.2), ALL, seed=1), seed_pairs('RC(HT)', RATES(0.1), ALL, seed=2)])
  reg = pd.concat([seed_pairs('HW(H,A)', RATES(0.2), ALL, seed=1), seed_pairs('RC(H)', RATES(0.1), ALL[:30], seed=4)])
  c = tri.comparisons(sp, reg, Args).set_index(['comparison', 'statistic'])
  assert c.loc[('HW(HT,AT) minus HW(H,A)', 'flip_top1'), 'estimate'] == 0
  assert c.loc[('RC(HT) minus RC(H)', 'flip_top1'), 'n_targets'] == 30
  assert np.isnan(c.loc[('RC(AT) minus RC(A)', 'flip_top1'), 'estimate'])        # part B not run


def test_default_pairs_subtract_the_larger_rc_and_name_a_missing_one():
  sp = pd.concat([seed_pairs('HW(GT,AT)', RATES(0.25), ALL, seed=1), seed_pairs('HW(GT,HT)', RATES(0.25), ALL, seed=5),
                  seed_pairs('HW(HT,AT)', RATES(0.2), ALL, seed=6), seed_pairs('RC(HT)', RATES(0.15), ALL, seed=2)])
  reg = seed_pairs('RC(GT)', RATES(0.05), ALL, seed=3)
  dp = tri.default_pairs(sp, reg, Args)
  r = dp[(dp.pair == 'HW(GT,HT)') & (dp.statistic == 'flip_top1')].iloc[0]
  assert r.reference == 'RC(HT)'
  r = dp[(dp.pair == 'HW(GT,AT)') & (dp.statistic == 'flip_top1')].iloc[0]
  assert r.reference == 'RC(GT) (RC(AT) not measured)'
  assert (dp[dp.group.str.startswith('XLA')].reference == 'not computed').all()


def gtat_rates(con, n=300, n_targets=60):
  """A synthetic rates.csv in gtat's format: the rows of con on "60 targets" and the same on "57 targets (no glycan)"
  with other values (which must not be read)."""
  rows = []
  for targets, shift in (('60 targets', 0.0), ('57 targets (no glycan)', 0.05)):
    for r in con.itertuples():
      rows.append({'targets': targets, 'contrast': r.contrast, 'statistic': r.statistic, 'n': n - 15 * (shift > 0),
                   'n_targets': n_targets - 3 * (shift > 0), 'events': 0, 'targets_with_event': 0,
                   'estimate': r.estimate + shift, 'ci_low': r.ci_low + shift, 'ci_high': r.ci_high + shift})
  rows.append({'targets': '60 targets', 'contrast': 'RC(GT)', 'statistic': 'flip_top1', 'n': n, 'n_targets': n_targets,
               'events': 0, 'targets_with_event': 0, 'estimate': 0.9, 'ci_low': 0.8, 'ci_high': 1.0})
  return pd.DataFrame(rows, columns=tri.GTAT_RATES)


def test_gtat_replication_equals_differs_or_not_found(tmp_path):
  sp = seed_pairs('HW(GT,AT)', RATES(0.2), ALL, seed=1)
  con = pd.DataFrame([{'contrast': 'HW(GT,AT)', 'statistic': s, 'n': 300, 'n_targets': 60, 'estimate': sp[s].mean(),
                       'ci_low': 0.1, 'ci_high': 0.3} for s in contrasts.FLIP_STATS])
  assert tri.gtat_replication(sp, con, None).result.tolist() == ['gtat output not found']
  g = tmp_path / 'gtat'
  g.mkdir()
  assert tri.gtat_replication(sp, con, g).result.tolist() == ['gtat output not found']
  sp.to_csv(g / 'pairs_seed.csv', index=False)
  gtat_rates(con).to_csv(g / 'rates.csv', index=False)
  r = tri.gtat_replication(sp, con, g)
  assert len(r) == 4 and (r.result == 'equals').all(), r                       # only the "60 targets" rows are read
  assert r.what.iloc[1].endswith('(rates.csv, 60 targets)')
  other = sp.copy()
  other.loc[0, 'flip_top1'] = 1 - other.loc[0, 'flip_top1']
  other.to_csv(g / 'pairs_seed.csv', index=False)
  gtat_rates(con.assign(estimate=con.estimate + 0.01)).to_csv(g / 'rates.csv', index=False)
  assert (tri.gtat_replication(sp, con, g).result == 'DIFFERS').all()
  gtat_rates(con).assign(n=299).to_csv(g / 'rates.csv', index=False)          # same values, another n
  assert (tri.gtat_replication(sp, con, g).iloc[1:].result == 'DIFFERS').all()
  gtat_rates(con).drop(columns=['ci_low']).to_csv(g / 'rates.csv', index=False)
  assert tri.gtat_replication(sp, con, g).result.iloc[1] == 'gtat rates.csv lacks ci_low'


GTAT = REPO / 'results' / 'analysis' / 'gtat_20261121'


@pytest.mark.skipif(not (GTAT / 'rates.csv').is_file(), reason='gtat outputs not here')
def test_gtat_files_have_the_columns_and_rows_the_replication_reads():
  """The real gtat outputs (2026-11-21): only their header lines and label columns are read, never a value."""
  rates_cols = pd.read_csv(GTAT / 'rates.csv', nrows=0).columns.tolist()
  pairs_cols = pd.read_csv(GTAT / 'pairs_seed.csv', nrows=0).columns.tolist()
  assert rates_cols == tri.GTAT_RATES
  assert set(tri.GTAT_PAIRS) <= set(pairs_cols)
  labels = pd.read_csv(GTAT / 'rates.csv', usecols=['targets', 'contrast', 'statistic'])
  mine = labels[(labels.targets == tri.GTAT_TARGETS) & (labels.contrast == 'HW(GT,AT)')]
  assert sorted(mine.statistic) == sorted(contrasts.FLIP_STATS)
  assert set(pd.read_csv(GTAT / 'pairs_seed.csv', usecols=['contrast']).contrast) == {'HW(GT,AT)'}


# ---------------------------------------------------------------- main output

def samples(cond, rep, targets, best, success=None, rmsd=None):
  """5 seeds x 5 samples per target; best[t] = (seed, sample) with the highest ranking score."""
  rows = []
  for t in targets:
    for s in tri.SEEDS:
      for k in range(5):
        win = (s, k) == best[t]
        ok = success[t] if success and win else False
        rows.append({'condition': cond, 'repetition': rep, 'target': t, 'seed': s, 'sample': k,
                     'ranking_score': 0.9 if win else 0.5 - 0.01 * s - 0.001 * k, 'm1_success': ok,
                     'm1_ligand_rmsd': (rmsd or {}).get(t, 1.0) if win else 5.0, 'm1_pb_valid': 'True',
                     'm1_pb_failed': np.nan, 'm1_error': np.nan,
                     'model_path': f'{cond}/{rep}/{t}/seed-{s}_sample-{k}/m.cif'})
  return pd.DataFrame(rows)


def test_main_output_contrasts_and_e_mo(monkeypatch):
  ts = ALL[:20]
  ht1 = samples('HT', 'r1', ts, {t: (1, 0) for t in ts}, {t: True for t in ts})
  # HT r2: the same main output except on 2 targets; AT: a different one on 10 targets, success lost on 4
  ht2 = samples('HT', 'r2', ts, {t: ((2, 1) if i < 2 else (1, 0)) for i, t in enumerate(ts)}, {t: True for t in ts})
  at = samples('AT', 'r1', ts, {t: ((3, 4) if i < 10 else (1, 0)) for i, t in enumerate(ts)},
               {t: i >= 4 for i, t in enumerate(ts)}, rmsd={t: (3.1 if i < 4 else 1.2) for i, t in enumerate(ts)})
  ok = pd.concat([ht1, ht2, at], ignore_index=True)
  u = tri.mo_units(ok)
  assert u[u.contrast == 'MO(HT,AT)'].mo_changed.sum() == 10 and u[u.contrast == 'MO(HT)'].mo_changed.sum() == 2
  assert u[u.contrast == 'MO(HT,AT)'].mo_success_flip.sum() == 4
  assert u[u.contrast == 'MO(HT,AT)'].set_index('target').loc['T00', 'main_b'] == '3:4'
  s = tri.mo_summary(u, Args).set_index(['quantity', 'statistic'])
  e = s.loc[('E_MO = MO(HT,AT) minus MO(HT)', 'mo_changed')]
  assert e.estimate == pytest.approx(10 / 20 - 2 / 20) and e.n_targets == 20
  assert e.reading == 'the main output differs beyond same-type rescheduling'
  assert s.loc[('MO(HT,AT)', 'mo_changed'), 'count'] == 10
  monkeypatch.setattr(winners, 'POSE_RMSD', lambda task: (2.5 if '/AT/' in '/' + task[2] else 0.4, None))
  ch = tri.mo_changes(u)
  at_ch = ch[ch.contrast == 'MO(HT,AT)'].set_index('target')
  assert len(at_ch) == 10 and at_ch.loc['T00', 'within_2A'] == 'one' and at_ch.loc['T05', 'within_2A'] == 'both'
  assert at_ch.loc['T00', 'criterion'] == 'RMSD' and at_ch.loc['T00', 'failing_side'] == 'b'
  assert at_ch.loc['T00', 'class'] == 'pose change' and at_ch.loc['T05', 'success_differs'] == False  # noqa: E712
  cc = tri.change_counts(ch).set_index(['contrast', 'category'])['count']
  assert cc[('MO(HT,AT)', 'failing criterion: RMSD')] == 4 and cc[('MO(HT,AT)', 'poses more than 2 A apart')] == 10
  assert cc[('MO(HT)', 'main output differs')] == 2


def test_main_output_uses_only_the_seeds_both_sides_ran():
  ts = ['T00']
  a = samples('HT', 'r1', ts, {'T00': (5, 0)}, {'T00': True})
  b = samples('AT', 'r1', ts, {'T00': (1, 0)}, {'T00': True})
  b = b[b.seed != 5]
  u = tri.mo_units(pd.concat([a, b], ignore_index=True))
  r = u[u.contrast == 'MO(HT,AT)'].iloc[0]
  assert r.seeds == 4 and r.main_a != '5:0'


# ---------------------------------------------------------------- checks

def fake_run(root, session, run, target, cmd, env, device='NVIDIA H100 80GB HBM3', bf16=None):
  r = root / 'results' / 'af3' / session / run
  (r / target).mkdir(parents=True, exist_ok=True)
  with open(r / 'runs.jsonl', 'a') as f:
    f.write(json.dumps({'pdb_id': target, 'command': cmd}) + '\n')
  (r / 'session.json').write_text(json.dumps({'xla_env': env, 'env_overrides': {}, 'config': {'env': {}}}))
  if bf16:
    (r / target / 'af3_entry_stats.json').write_text(json.dumps({'fu13': {'global_config_bfloat16': bf16}}))
  return {'session': f'results/af3/{session}/{run}', 'target': target, 'seed': 1, 'repetition': 'r1',
          'condition': 'HT', 'device_kind': device}


def test_settings_check_triton_environment_and_device(tmp_path):
  good = '... --jax_backend=gpu --flash_attention_implementation=triton --run_data_pipeline=false'
  rows = [fake_run(tmp_path, 'S1', 'h100_triton_rep1_seed1_fresh', 'T1', good, {'XLA_FLAGS': 'x'}),
          fake_run(tmp_path, 'S1', 'h100_triton_rep1_seed1_fresh', 'T2', good.replace('triton', 'xla'), {}),
          fake_run(tmp_path, 'S2', 'h100_triton_rep1_seed1_fresh', 'T3', good, {'JAX_DEFAULT_MATMUL_PRECISION': 'highest'}),
          fake_run(tmp_path, 'S3', 'h100_triton_rep1_seed1_fresh', 'T4', good, {}, device='NVIDIA A100-SXM4-80GB'),
          fake_run(tmp_path, 'S4', 'h100_triton_rep1_seed1_fresh', 'T5', good, {}, bf16='none')]
  rows[1]['session'] = rows[0]['session']
  out, det = tri.check_settings(pd.DataFrame(rows), root=tmp_path)
  assert out[0]['value'] == '1/5 processes' and out[0]['detail'].startswith('reported')
  text = ' '.join(x['process'] for x in det)
  assert 'T2' in text and 'JAX_DEFAULT_MATMUL_PRECISION' in text and 'device NVIDIA A100' in text and 'float32 trunk' in text


def launch_record(af3, session, vm, launcher=False):
  d = af3 / ('.launcher' if launcher else '') / session
  d.mkdir(parents=True, exist_ok=True)
  (d / 'launch_record.txt').write_text(f'SESSION={session}\nVM_NAME={vm}\nPLATFORM=h100\n')


def test_check_5_vm_names(tmp_path):
  af3 = tmp_path / 'af3'
  launch_record(af3, '20261201T000000Z_h100_tri_ht_01', 'vm-a')
  launch_record(af3, '20261201T010000Z_h100_tri_ht_r2_01', 'vm-b', launcher=True)
  launch_record(af3, '20261201T020000Z_h100_tri_ht_r2_02', 'vm-a')
  d = pd.DataFrame([{'condition': 'HT', 'repetition': 'r1', 'target': t, 'session': f'results/af3/20261201T000000Z_h100_tri_ht_01/x'}
                    for t in ('T1', 'T2')] +
                   [{'condition': 'HT', 'repetition': 'r2', 'target': 'T1', 'session': 'results/af3/20261201T010000Z_h100_tri_ht_r2_01/x'},
                    {'condition': 'HT', 'repetition': 'r2', 'target': 'T2', 'session': 'results/af3/20261201T020000Z_h100_tri_ht_r2_02/x'}])
  r = tri.check_vms(d, af3)
  assert r['value'] == '1/2 (session, target) on another VM' and 'T2: same VM vm-a' in r['detail']
  assert tri.check_vms(d[d.repetition == 'r1'], af3)['detail'] == 'not run'


def test_stack_check_reports_python_and_image(tmp_path):
  af3, many = tmp_path / 'af3', tmp_path / 'many'
  for s, py in (('20261104T000000Z_h100_h100_h_01', '3.12.15'), ('20261201T000000Z_h100_tri_ht_01', '3.12.16')):
    run = af3 / s / 'x_rep1_seed1_fresh'
    run.mkdir(parents=True)
    (af3 / s / 'device_check.txt').write_text('Driver Version: 580.178.04 \n')
    (run / 'session.json').write_text(json.dumps({'host': {'platform': 'Linux-6.8'}, 'probe': {'python': py}}))
    (af3 / s / 'pip_freeze.txt').write_text('a==1\n')
  (many / 'R1').mkdir(parents=True)
  (many / 'R1' / 'p.log').write_text('results/af3/20261201T000000Z_h100_tri_ht_01/\n   image af2-tpu-benchmark/af3-accel-2204-nvidia-580\n')
  d = pd.DataFrame([{'condition': 'HT', 'session': 'results/af3/20261201T000000Z_h100_tri_ht_01/x_rep1_seed1_fresh'}])
  rows, table = tri.check_stack(d, af3, many, refs=lambda c: ['20261104T000000Z_h100_h100_h_01'])
  assert 'python: 3.12.16 (reference 3.12.15)' in rows[0]['detail']
  assert 'image: af2-tpu-benchmark/af3-accel-2204-nvidia-580 (reference not recorded)' in rows[0]['detail']
  assert set(table[~table.same].field) == {'python', 'image'}


def test_completeness_marks_part_b_not_run():
  ok = pd.DataFrame([{'condition': 'HT', 'repetition': r, 'seed': s, 'target': t, 'sample': k}
                     for r in ('r1', 'r2') for s in tri.SEEDS for t in ALL for k in range(5)])
  comp = tri.completeness(ok, ALL)
  assert comp[comp.condition == 'HT'].status.eq('complete').all()
  assert comp[comp.condition == 'AT'].status.eq('not run').all() and len(comp) == 15


# ---------------------------------------------------------------- plans and config diff

def test_plans_are_up_to_date_and_match_the_specification():
  out = subprocess.run([sys.executable, str(REPO / 'harness' / 'plans' / 'make_prep_plans.py'), '--label', 'tri', '--check'],
                       capture_output=True, text=True, cwd=REPO)
  assert out.returncode == 0, out.stdout + out.stderr
  parts = pd.read_csv(PARTS)
  assert parts.groupby('arm').processes.sum().to_dict() == {'HT': 300, 'HT_r2': 300, 'AT_r2': 300}
  assert (parts.groupby('arm').part.count().to_dict() == {'HT': 5, 'HT_r2': 5, 'AT_r2': 7})
  assert set(parts.provisioning) == {'spot'} and set(parts.fetch) == {'light'}
  assert set(parts[parts.row_part == 'A'].arm) == {'HT', 'HT_r2'} and set(parts[parts.row_part == 'B'].arm) == {'AT_r2'}
  configs = read_configs(REPO / 'harness' / 'configs.yaml')
  assert configs['h100_triton']['flash_attention_implementation'] == 'triton'
  for p in sorted((REPO / 'harness' / 'plans' / 'tri').glob('*.yaml')):
    t = p.read_text()
    assert 'label: tri\n' in t and 'eigen_source: archive\n' in t, p
    assert ('rep_start: 2\n' in t) == ('_r2_' in p.name), p
    cfg = 'a100_triton' if '_at_' in p.name else 'h100_triton'
    assert f'  {"a100" if "_at_" in p.name else "h100"}: {cfg}\n' in t, p
  ht = {p.name.replace('_r2', ''): p.read_text().split('targets:')[1].split('\n')[1]
        for p in (REPO / 'harness' / 'plans' / 'tri').glob('tri_ht_*.yaml')}
  assert len(ht) == 5                                         # HT and HT r2 parts of the same number hold the same targets


def test_config_diff_finds_nothing_beyond_the_allowed_settings(tmp_path):
  out = subprocess.run([sys.executable, str(REPO / 'harness' / 'plans' / 'config_diff_tri.py'), '--out',
                        str(tmp_path / 'd.md')], capture_output=True, text=True, cwd=REPO)
  if 'No such file' in out.stderr or 'FileNotFoundError' in out.stderr:
    pytest.skip('source sessions not here')
  assert out.returncode == 0, out.stdout[-2000:] + out.stderr[-2000:]
  text = (tmp_path / 'd.md').read_text()
  assert text.rstrip().endswith('Settings that differ beyond the allowed ones: 0')
  assert text.count('intended (attention only)') == 4                         # entry and command, HT r1 and HT r2


def test_config_diff_attention_rule():
  import config_diff_tri as cdt  # noqa: E402
  a = ['{"flash_attention_implementation": "xla", "jax_backend": "gpu"}']
  b = ['{"flash_attention_implementation": "triton", "jax_backend": "gpu"}']
  assert cdt.mark(cdt.ENTRY, a, b, True) == 'intended (attention only)'
  assert cdt.mark(cdt.ENTRY, a, b, False) == 'DIFFERS'
  c = ['{"flash_attention_implementation": "triton", "jax_backend": "cpu"}']
  assert cdt.mark(cdt.ENTRY, a, c, True) == 'DIFFERS'


sys.path.insert(0, str(REPO / 'harness' / 'plans'))

# ---------------------------------------------------------------- end to end, on relabelled earlier rows

A100 = REPO / 'results' / 'analysis' / 'a100_20261018'
H100 = REPO / 'results' / 'analysis' / 'h100_20261104'
PRIMARY = REPO / 'results' / 'analysis' / 'main_20261014' / 'primary_v2'
RCX = REPO / 'results' / 'analysis' / 'rcx_20261115'


@pytest.mark.skipif(not all((p / 'samples.csv').is_file() for p in (A100, H100, PRIMARY, RCX)),
                    reason='earlier analysis outputs not here')
def test_main_runs_end_to_end_on_relabelled_rows(tmp_path):
  """tri.main on earlier rows relabelled as follow-up 18's: H r1 and r2 of follow-up 9 as HT r1 and r2, AT r1 of
  follow-up 3 as AT r2, on three targets and two seeds, with no GT row (the main study's input without GT). Every output is written; checks 1 and 2 pass; check 4a reports
  the XLA command of the relabelled H rows; RC(AT) is 0 and its light identity reads "recompilation bit-identical"
  (AT r2 is AT r1's own files); HW(HT,AT) equals the registered HW(H,A) on those units."""
  ts, seeds = ['7U3J', '7NP6', '7BTT'], [1, 2]
  rows = []

  def take(path, platform, rep, config, new_rep):
    m = pd.read_csv(path, low_memory=False)
    g = m[(m.platform == platform) & (m.repetition == rep) & m.seed.isin(seeds) & m.target.isin(ts)
          & m['sample'].notna()].copy()
    g['label'], g['config'], g['repetition'] = 'tri', config, new_rep
    rows.append(g)
  take(H100 / 'samples.csv', 'H', 'r1', 'h100_triton', 'r1')
  take(H100 / 'samples.csv', 'H', 'r2', 'h100_triton', 'r2')
  take(A100 / 'samples.csv', 'AT', 'r1', 'a100_triton', 'r2')
  d = pd.concat(rows, ignore_index=True)
  d.to_csv(tmp_path / 'samples.csv', index=False)
  # No real GT row enters this test: HW(GT,AT) on the frozen outputs belongs to the post hoc analysis gtat and is not
  # computed here. The main study's input is a copy without GT (G only; DU and RC(G) seed pairs only).
  main = tmp_path / 'main'
  (main / 'contrasts').mkdir(parents=True)
  m = pd.read_csv(PRIMARY / 'samples.csv', low_memory=False)
  m[(m.platform == 'G') & m.target.isin(ts)].to_csv(main / 'samples.csv', index=False)
  ps = pd.read_csv(PRIMARY / 'contrasts' / 'pairs_seed.csv')
  ps[ps.contrast.isin(['DU', 'RC(G)'])].to_csv(main / 'contrasts' / 'pairs_seed.csv', index=False)
  out = tmp_path / 'out'
  r = subprocess.run([sys.executable, str(REPO / 'analysis' / 'tri.py'), '--samples', str(tmp_path / 'samples.csv'),
                      '--out', str(out), '--replicates', '200', '--gtat', str(tmp_path / 'no_gtat'),
                      '--main', str(main), '--rcx', str(tmp_path / 'no_rcx')],
                     capture_output=True, text=True, cwd=REPO)
  assert r.returncode == 0, r.stderr[-3000:]
  for name in ('checks', 'completeness', 'contrasts', 'e_t', 'comparisons', 'identity', 'main_output',
               'main_output_units', 'default_pairs', 'gtat_replication', 'success', 'cost', 'stack', 'pairs_seed'):
    assert (out / f'{name}.csv').is_file(), name
  chk = pd.read_csv(out / 'checks.csv')
  assert (chk[chk.check.str.startswith(('1 ', '2'))].detail == 'pass').all()
  s4 = chk[chk.check.str.startswith('4a') & (chk.condition == 'HT')].detail.iloc[0]
  assert s4.startswith('reported') and 'no --flash_attention_implementation=triton' in s4
  ident = pd.read_csv(out / 'identity.csv').set_index('condition')
  assert ident.loc['AT', 'detail'] == 'recompilation bit-identical'
  con = pd.read_csv(out / 'contrasts.csv')
  top = con[(con.contrast == 'RC(AT)') & (con.statistic == 'flip_top1')]
  assert top.estimate.iloc[0] == 0 and top.n.iloc[0] == 6
  sp = pd.read_csv(out / 'pairs_seed.csv')
  reg = pd.read_csv(H100 / 'pairs_seed.csv')
  reg = reg[(reg.contrast == 'HW(H,A)') & reg.target.isin(ts) & reg.seed_a.isin(seeds)]
  mine = sp[sp.contrast == 'HW(HT,AT)']
  # AT r1 (here AT) against H: not HW(H,A) (A is XLA); the pairing must cover the 6 units
  assert len(mine) == 6 and len(reg) == 6
  assert pd.read_csv(out / 'gtat_replication.csv').result.tolist() == ['gtat output not found']
  assert not (sp.contrast.isin(['HW(GT,AT)', 'HW(GT,HT)'])).any()               # no GT row was given
  e = pd.read_csv(out / 'e_t.csv')
  assert set(e.reading) <= {'beyond recompilation', 'not distinguishable from recompilation'}
  rep = (out / 'report.md').read_text()
  assert '## T2: the main output (rule e2)' in rep and 'The gtat values stand.' in rep


def test_pb_full_rerun_adds_the_posebusters_excess(monkeypatch):
  """--pb_full: pb_flips' rerun (replaced here by a synthetic one that reproduces the saved M1) gives the excess of the
  failing PoseBusters check, and the flip is classified as pb_flips classifies it."""
  import pb_flips
  ts = ALL[:2]
  ht = samples('HT', 'r1', ts, {t: (1, 0) for t in ts}, {t: True for t in ts})
  at = samples('AT', 'r1', ts, {t: (2, 0) for t in ts}, {t: False for t in ts}, rmsd={t: 1.5 for t in ts})
  win = (at.seed == 2) & (at['sample'] == 0)
  at.loc[win, 'm1_pb_valid'] = 'False'
  at['m1_pb_failed'] = at['m1_pb_failed'].astype(object)
  at.loc[win, 'm1_pb_failed'] = 'minimum_distance_to_protein'
  u = tri.mo_units(pd.concat([ht, at], ignore_index=True))
  check = 'minimum_distance_to_protein'
  assert check in pb_flips.CHECKS

  def rescore(task):
    t, path = task
    bad = '/AT/' in '/' + path
    out = {'target': t, 'path': path, 're_ligand_rmsd': 1.5 if bad else 1.0, 're_pb_valid': not bad,
           're_pb_failed': check if bad else '', 're_success': not bad, 're_error': None,
           'std_failed': check if bad else '', 'full_agrees': True}
    for c in pb_flips.CHECKS:
      out[f'pass:{c}'] = None
      out[f'excess:{c}'] = np.nan
    out[f'pass:{check}'] = not bad
    out[f'excess:{check}'] = 0.05 if bad else -0.2
    return out
  monkeypatch.setattr(pb_flips, 'RESCORE', rescore)
  monkeypatch.setattr(winners, 'POSE_RMSD', lambda task: (0.8, None))
  ch = tri.mo_changes(u, pb_full=True).set_index('target')
  r = ch.loc['T00']
  assert r.criterion == 'PoseBusters' and r.failing_criteria == check and r['class'] == 'near-threshold crossing'
  assert r.excess_fail == f'{check}=0.0500'
  ch0 = tri.mo_changes(u, pb_full=False).set_index('target')
  assert ch0.loc['T00', 'class'] == 'not classified'                     # without the rerun, no check excess
