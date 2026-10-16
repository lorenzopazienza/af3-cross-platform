"""analysis/cpu60.py, analysis/a100.py and analysis/followup34.py (follow-ups 3 and 4, section 10, 2026-10-16) on
synthetic per-sample tables: pairing, the reuse of the main study's 6 C targets, the machine-type check, the checks
that stop the A100 analysis, the follow-up labels and codes, and M6 at both regions."""

import types

import numpy as np
import pandas as pd
import pytest

import a100
import collect
import contrasts
import cost
import cpu60
import followup34 as f34

ARGS = types.SimpleNamespace(seed=2026, replicates=40)


@pytest.fixture(autouse=True)
def no_files(monkeypatch):
  # Coordinates and M4 without model files: a model path encodes its coordinates after "@".
  monkeypatch.setattr(contrasts, 'coords_text', lambda path: path.split('@')[1])
  monkeypatch.setattr(contrasts, 'divergence', lambda t, a, b: {'ligand_rmsd': 1.0, 'ca_rmsd_global': 0.5, 'error': None})


def rows(condition, targets, seeds, rep, label='main', shift=0.0, machine='g2-standard-8', exit_code=0, coords='x',
         session=None):
  out = []
  for t in targets:
    for s in seeds:
      for k in range(5):
        score = 0.9 - 0.1 * k + (shift if k == 1 else 0.0)   # shift > 0.1 makes sample 1 the top-1
        out.append({'session': session or f'S_{condition}_{rep}', 'run': f'run_{rep}_seed{s}', 'label': label,
                    'platform': condition, 'target': t, 'seed': s, 'repetition': rep, 'sample': k,
                    'cache': 'warm' if rep == 'w' else 'fresh', 'num_recycles': 10, 'num_diffusion_samples': 5,
                    'ranking_score': score, 'm1_success': k == 0, 'mean_plddt': 80.0, 'iptm': 0.8,
                    'model_path': f'm/{condition}/{rep}/{t}/{s}/{k}@{coords}{k}', 'machine_type': machine,
                    'exit_code': exit_code, 'failure': None if exit_code == 0 else 'oom', 'cache_files_before': 0,
                    'model_cache_hit': rep == 'w', 'model_cache_key': f'key_{t}_{s}' if rep == 'w' else None,
                    'model_cache_written': f'key_{t}_{s}' if rep == 'r1' else None, 'num_tokens': 300, 'bucket': 512,
                    'zone': 'us-central1-a', 'wall_s': 100.0, 'inference_s': 60.0, 'compile_s': 30.0})
  d = pd.DataFrame(out)
  d['condition'] = d.platform
  return d


def test_union_refuses_a_unit_from_two_sources():
  a = rows('C', ['T1'], [1], 'r1')
  with pytest.raises(SystemExit, match='two sources'):
    f34.union(a, a.assign(session='other'))


def test_disjoint_c_units_and_the_reuse_of_the_main_studys_targets():
  main = rows('C', ['T1', 'T2'], [1, 2], 'r1')
  c60 = rows('C', ['T3', 'T4', 'T5'], [1, 2], 'r1', label='cpu60', session='S_cpu60')
  assert f34.disjoint_units(main, c60, 'C') == (4, 6)
  with pytest.raises(SystemExit, match='both sources'):
    f34.disjoint_units(main, rows('C', ['T2'], [2], 'r1', label='cpu60'), 'C')


def test_one_machine_type_refuses_two():
  d = pd.concat([rows('A', ['T1'], [1], 'r1', machine='a2-ultragpu-1g'),
                 rows('A', ['T2'], [1], 'r1', machine='a2-highgpu-1g', session='S2')])
  with pytest.raises(SystemExit, match='2 machine types'):
    f34.one_machine_type(d, ['A'])
  assert f34.one_machine_type(d[d.machine_type == 'a2-ultragpu-1g'], ['A']) == 'a2-ultragpu-1g'


def test_build_pairs_flips_and_seed_yardstick():
  ok = f34.union(rows('G', ['T1', 'T2'], [1, 2, 3], 'r1'), rows('A', ['T1', 'T2'], [1, 2, 3], 'r1', shift=0.2, coords='y'))
  sp, sm, cons, counts = f34.build(ok, [('HW(G,A)', 'HW', ('G', 'r1'), ('A', 'r1'), False),
                                        ('SD(A)', 'SD', ('A', 'r1'), ('A', 'r1'), True)])
  hw = sp[sp.contrast == 'HW(G,A)']
  assert len(hw) == 6 and hw.flip_top1.all() and hw.flip_success.all()   # A's top-1 is sample 1, a failure
  assert len(sp[sp.contrast == 'SD(A)']) == 2 * 3                          # 3 seed pairs per target
  assert set(counts.contrast) == {'HW(G,A)', 'SD(A)'} and cons['HW(G,A)'] == ('HW', False)
  hs = sm[sm.contrast == 'HW(G,A)']
  assert len(hs) == 6 * 5 and not hs.bit_identical.any() and (hs.m4_ca_rmsd_global == 0.5).all()
  assert sm[sm.contrast == 'SD(A)'].bit_identical.all()   # within A the fake coordinates repeat


def _primary(tmp_path, names=('RC(G)', 'RC(V5)', 'RC(V6)', 'RC(Gd)')):
  (tmp_path / 'contrasts').mkdir(parents=True)
  seed = pd.DataFrame([{'contrast': n, 'kind': 'RC', 'role': 'recompilation', 'target': t, 'seed_a': 1, 'seed_b': 1,
                        'top1_a': 0, 'top1_b': 0, 'success_a': True, 'success_b': True, 'margin_a': 0.1, 'margin_b': 0.1,
                        'flip_top1': False, 'flip_success': False, 'flip_matters': False}
                       for n in names for t in ('T1', 'T2', 'T3')])
  seed.to_csv(tmp_path / 'contrasts' / 'pairs_seed.csv', index=False)
  pd.DataFrame([{'contrast': 'HW(G,C)', 'kind': 'HW', 'role': 'primary pair', 'statistic': 'flip_top1', 'n': 12,
                 'n_targets': 6, 'estimate': 0.25, 'ci_low': 0.0, 'ci_high': 0.5, 'note': ''}]).to_csv(
      tmp_path / 'contrasts' / 'contrasts.csv', index=False)
  return tmp_path


def test_cpu60_contrasts_use_main_and_cpu60_c_targets(tmp_path):
  targets = ['T1', 'T2', 'T3', 'T4', 'T5']
  main = pd.concat([rows(c, targets, [1, 2], 'r1') for c in ('G', 'V5', 'V6', 'GT', 'Gd', 'Gh')]
                   + [rows('C', ['T1', 'T2'], [1, 2], 'r1', machine='n2-highmem-16', shift=0.2)])
  c60 = rows('C', ['T3', 'T4', 'T5'], [1, 2], 'r1', label='cpu60', machine='n2-highmem-16', session='S_cpu60')
  checks = cpu60.checks_first(main, c60)
  assert all(r['detail'] == 'pass' for r in checks) and checks[-1]['value'] == '4 + 6'
  counts, con, six, hwrc, sp, sm, cpu = cpu60.analyse(main, c60, ARGS, _primary(tmp_path))
  hw = sp[sp.contrast == 'HW(G,C)']
  assert sorted(hw.target.unique()) == targets and len(hw) == 10
  assert hw[hw.target.isin(['T1', 'T2'])].flip_top1.all() and not hw[hw.target.isin(['T3', 'T4', 'T5'])].flip_top1.any()
  r = con[(con.contrast == 'HW(G,C)') & (con.statistic == 'flip_top1')].iloc[0]
  assert r.n_targets == 5 and r.estimate == pytest.approx(0.4)
  assert set(con.contrast) == set(cpu60.NAMES)
  s = six[(six.contrast == 'HW(G,C)') & (six.statistic == 'flip_top1')].iloc[0]
  assert s.main_estimate == 0.25 and s.main_n_targets == 6 and s.new_n_targets == 5    # the 6-target value unchanged
  assert {'RC(G)', 'RC(V5)', 'RC(V6)', 'RC(Gd)'} >= set(hwrc.recompilation)
  assert sorted(cpu.target.unique()) == targets


def test_cpu60_check_reports_a_failed_process():
  main = rows('C', ['T1'], [1, 2], 'r1', machine='n2-highmem-16')
  c60 = rows('C', ['T3'], [1, 2], 'r1', label='cpu60', machine='n2-highmem-16', exit_code=1)
  checks = cpu60.checks_first(main, c60)
  assert checks[0]['detail'].startswith('FAIL')


def _a100_tables(machine='a2-ultragpu-1g', exit_code=0):
  targets = ['T1', 'T2']
  a = pd.concat([rows('A', targets, [1, 2], 'r1', label='a100', machine=machine),
                 rows('A', targets, [1, 2], 'r2', label='a100', machine=machine),
                 rows('A', targets, [1], 'w', label='a100', machine=machine),
                 rows('Ah', targets, [1, 2], 'r1', label='a100', machine=machine),
                 rows('AT', targets, [1, 2], 'r1', label='a100', machine=machine, exit_code=exit_code)])
  main = pd.concat([rows(c, targets, [1, 2], 'r1') for c in ('G', 'GT', 'V5', 'V6', 'V6h')]
                   + [rows('C', ['T1'], [1, 2], 'r1', machine='n2-highmem-16')])
  return a, main


def test_a100_checks_pass_and_stop(monkeypatch):
  monkeypatch.setattr(contrasts, 'warm_identity',
                      lambda d, c: [(t, 1, {'status': 'identical', 'differ': [], 'missing': []}) for t in ('T1', 'T2')])
  a, main = _a100_tables()
  ok = f34.union(main, a)
  checks, passed = a100.checks_first(a, ok)
  assert passed, checks
  assert {r['check'][:2] for r in checks} >= {'ma', '1 ', '2a', '2b', '2 '}
  a, main = _a100_tables(exit_code=1)
  assert not a100.checks_first(a, f34.union(main, a))[1]                    # check 1 fails: stop
  a, main = _a100_tables(machine='a2-highgpu-1g')
  assert not a100.checks_first(a, f34.union(main, a))[1]                    # the wrong machine type: stop
  a, main = _a100_tables()
  a.loc[a.condition == 'Ah', 'machine_type'] = 'a2-highgpu-1g'
  with pytest.raises(SystemExit, match='machine types'):                    # two machine types: refused
    a100.checks_first(a, f34.union(main, a))


def test_a100_contrasts_and_both_region_prices(tmp_path):
  a, main = _a100_tables()
  ok = f34.union(main, a)
  counts, con, hwrc, q3rows, q3answer, sp, sm = a100.analyse(ok, ARGS, _primary(tmp_path, ('RC(G)', 'RC(V6)')))
  assert set(con.contrast) == {s[0] for s in a100.CHECK2_SPECS + a100.SPECS}
  assert set(hwrc.hardware) == {'HW(G,A)', 'HW(V6,A)'} and set(hwrc.recompilation) == {'RC(G)', 'RC(V6)', 'RC(A)'}
  assert set(q3rows[q3rows.statistic == 'success_rate'].condition) == {'V5', 'V6', 'G', 'GT', 'C', 'A', 'AT'}
  du = sp[sp.contrast == 'DU(A)']
  assert len(du) == 4 and (du.kind == 'DU').all()
  c = pd.DataFrame([{'session': 'S', 'platform': 'A', 'target': 'T1', 'bucket': 512, 'repetition': 'r1',
                     'zone': 'us-central1-a', 'wall_per_structure_s': 3600.0}])
  per, med = a100.both_regions(c, a100.a100_price_table())
  assert per.iloc[0]['usd_per_structure_standard_us-central1'] == pytest.approx(5.027702)
  assert per.iloc[0]['usd_per_structure_standard_europe-west4'] == pytest.approx(5.535713)
  assert not np.isnan(per.iloc[0]['usd_per_structure_flex_start_europe-west4'])


def test_follow_up_codes_labels_and_prices(tmp_path):
  assert collect.platform_of('NVIDIA A100-SXM4-80GB', 'xla', None, 'a100_xla') == 'A'
  assert collect.platform_of('NVIDIA A100-SXM4-80GB', 'triton', None, 'a100_triton') == 'AT'
  assert collect.platform_of('NVIDIA A100-SXM4-80GB', 'xla', 'highest', 'a100_xla_highest') == 'Ah'
  assert collect.platform_of('NVIDIA L4', 'xla', None, 'l4_xla') == 'G'                      # unchanged
  assert {cost.PRICE_KEY[c] for c in ('A', 'Ah', 'AT')} == {'a100'}
  d = pd.concat([rows('A', ['T1'], [1], 'r1', label='a100'), rows('A', ['T2'], [1], 'r1', label='smoke')])
  d.to_csv(tmp_path / 's.csv', index=False)
  loaded = f34.load_samples(tmp_path / 's.csv', ['a100'])
  assert sorted(loaded.target.unique()) == ['T1']                                            # the smoke test never enters
