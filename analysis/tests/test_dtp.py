"""Follow-up 17 (label dtp; notes/analysis_plan.md section 10, row "Follow-up experiment 17"): synthetic tests of the
readings of analysis/dtp.py (agree, D_T, the attribution of rule e3, the decomposition, the generality sentence), of
check 4 (the precision variable must be absent), of the main-output counts, of the plans and the config diff, and an
end-to-end run on relabelled rows of follow-up 13. No follow-up 17 output exists or is read."""
import json
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
import dtp  # noqa: E402
from run_af3 import read_configs  # noqa: E402

PARTS = REPO / 'harness' / 'plans' / 'dtp_parts.csv'
ALL = [f'T{i:02d}' for i in range(60)]


class Args:
  seed, replicates = 2026, 300


def seed_pairs(name, rates, targets=ALL, per=5, seed=0):
  rng = np.random.default_rng(seed)
  rows = []
  for t in targets:
    for s in range(per):
      rows.append({'contrast': name, 'target': t, 'seed_a': s + 1, 'seed_b': s + 1,
                   **{st: int(rng.random() < r) for st, r in rates.items()}})
  return pd.DataFrame(rows)


RATES = lambda x: {'flip_top1': x, 'flip_success': x / 5, 'flip_matters': x / 6}  # noqa: E731


# ---------------------------------------------------------------- design

def test_conditions_cells_and_pairs_are_the_specification():
  assert dtp.CONFIG == {'tpu_xla_dtp_f32def': 'V6fd', 'l4_xla_dtp_f32def': 'Gfd', 'a100_xla_dtp_f32def': 'Afd',
                        'h100_xla_dtp_f32def': 'Hfd'}
  assert dtp.CELLS == {'c_bd': ('HW(V6,G)', 'main'), 'c_bh': ('HWh(V6,G)', 'prec5'), 'c_fd': ('HW(V6fd,Gfd)', 'here'),
                       'c_fh': ('HW(V6f,Gf)', 'fu13')}
  pairs = {frozenset(p) for p in dtp.B_PAIRS}
  assert len(dtp.B_PAIRS) == 6 and len(pairs) == 6 and all(len(p) == 2 for p in pairs)
  assert set().union(*pairs) == {'V6fd', 'Gfd', 'Afd', 'Hfd'}
  assert dtp.THRESHOLD == 0.05 and dtp.SEEDS == [1, 2, 3, 4, 5]
  assert dtp.PLANNED['V6fd'][('r2', 1)] == 10 and dtp.PLANNED['Gfd'][('r2', 5)] == 60
  assert dtp.B_DT[('Hfd', 'Afd')][2] is None and dtp.B_DT[('Gfd', 'Afd')][2] == 'RC(Gfd)'


# ---------------------------------------------------------------- readings

def test_agree_reading_threshold():
  assert dtp.agree_reading(0.0499) == 'agree'
  assert dtp.agree_reading(0.05) == 'agreement not shown'
  assert dtp.agree_reading(np.nan) == 'not computed'


def test_attribution_follows_rule_e3_in_order():
  assert dtp.attribution('agree', 'reduced') == 'float32 activations suffice'
  assert dtp.attribution('agree', 'not reduced').startswith('float32 activations suffice (note')
  assert dtp.attribution('agreement not shown', 'not reduced') == 'both are needed'
  assert dtp.attribution('agreement not shown', 'reduced') == 'both contribute'
  assert dtp.attribution('agreement not shown', 'not computed') == 'not computed'


def test_generality_needs_all_six_pairs():
  agree = pd.DataFrame({'pair': [dtp.hw(*p) for p in dtp.B_PAIRS], 'reading': ['agree'] * 6})
  assert dtp.generality(agree).startswith('the float32 trunk at the default matmul precision makes')
  agree.loc[3, 'reading'] = 'agreement not shown'
  g = dtp.generality(agree)
  assert g.startswith('not shown') and 'HW(V6fd,Afd)' in g.split('; agreement not shown')[1]
  agree.loc[1, 'reading'] = 'not computed'
  assert dtp.generality(agree).startswith('not computed')


def test_d_t_reads_reduced_when_the_new_cell_agrees_and_not_reduced_when_it_does_not():
  ref = {'main': pd.concat([seed_pairs('HW(V6,G)', RATES(0.2), seed=1), seed_pairs('RC(G)', RATES(0.07), seed=2)])}
  sp = pd.concat([seed_pairs('HW(V6fd,Gfd)', RATES(0.0), seed=3), seed_pairs('RC(Gfd)', RATES(0.0), seed=4)])
  t = dtp.d_t_table(sp, ref, Args)
  top = t[t.statistic == 'flip_top1'].iloc[0]
  assert top.role == 'primary' and top.reading == 'reduced' and top.estimate == pytest.approx(top.rc - top.hw)
  sp = pd.concat([seed_pairs('HW(V6fd,Gfd)', RATES(0.2), seed=5), seed_pairs('RC(Gfd)', RATES(0.07), seed=6)])
  assert dtp.d_t_table(sp, ref, Args).query('statistic == "flip_top1"').reading.iloc[0] == 'not reduced'


def test_decomposition_terms_on_known_cells():
  one = {'flip_top1': 1.0, 'flip_success': 1.0, 'flip_matters': 1.0}
  zero = {k: 0.0 for k in one}
  cp = {'c_bd': seed_pairs('HW(V6,G)', one), 'c_bh': seed_pairs('HWh(V6,G)', one),
        'c_fd': seed_pairs('HW(V6fd,Gfd)', zero), 'c_fh': seed_pairs('HW(V6f,Gf)', zero)}
  t = dtp.decomposition(cp, Args).set_index(['term', 'statistic'])
  assert t.loc[('T_d', 'flip_top1'), 'estimate'] == -1 and t.loc[('P_f', 'flip_top1'), 'estimate'] == 0
  assert t.loc[('I', 'flip_top1'), 'estimate'] == 0 and t.loc[('I', 'flip_top1'), 'n_targets'] == 60
  cp['c_fd'] = seed_pairs('HW(V6fd,Gfd)', {k: 0.5 for k in one}, seed=9)     # half: P_f negative, I negative
  t = dtp.decomposition(cp, Args).set_index(['term', 'statistic'])
  p_f, i = t.loc[('P_f', 'flip_top1'), 'estimate'], t.loc[('I', 'flip_top1'), 'estimate']
  assert p_f < 0 and i == pytest.approx(p_f)                                   # c_bh - c_bd = 0


def test_part_b_d_t_sets_rc_hfd_to_zero_and_uses_rc_gfd():
  ref = {'h100': pd.concat([seed_pairs('HW(H,A)', RATES(0.2), seed=1), seed_pairs('RC(H)', RATES(0.09), seed=2)]),
         'a100': seed_pairs('HW(G,A)', RATES(0.2), seed=3), 'main': seed_pairs('RC(G)', RATES(0.07), seed=4),
         'fu14': pd.concat([seed_pairs('HW(Hf,Af)', RATES(0.02), seed=5), seed_pairs('HW(Gf,Af)', RATES(0.02), seed=6)])}
  sp = pd.concat([seed_pairs('HW(Hfd,Afd)', RATES(0.01), seed=7), seed_pairs('HW(Gfd,Afd)', RATES(0.01), seed=8),
                  seed_pairs('RC(Gfd)', RATES(0.05), seed=9)])
  t = dtp.part_b_table(sp, ref, Args)
  h = t[(t.comparison == 'D_T of HW(Hfd,Afd)') & (t.statistic == 'flip_top1')].iloc[0]
  g = t[(t.comparison == 'D_T of HW(Gfd,Afd)') & (t.statistic == 'flip_top1')].iloc[0]
  assert h.float32_recompilation.startswith('set to 0') and g.float32_recompilation == 'RC(Gfd)'
  assert h.reading == 'reduced' and len(t) == 12
  assert dtp.part_b_table(sp[sp.contrast == 'RC(Gfd)'], ref, Args).empty            # part B not run


# ---------------------------------------------------------------- checks

def proc(tmp_path, cond, bf16, env, device, name):
  run = tmp_path / name / 'run'
  (run / 'T1').mkdir(parents=True)
  (run / 'T1' / 'af3_entry_stats.json').write_text(json.dumps({'fu13': {'global_config_bfloat16': bf16}} if bf16 else {}))
  (run / 'session.json').write_text(json.dumps({'xla_env': env}))
  return {'session': str(run.relative_to(tmp_path)), 'target': 'T1', 'seed': 1, 'repetition': 'r1', 'condition': cond,
          'device_kind': device}


def test_check_4_requires_float32_and_the_absence_of_the_precision_variable(tmp_path):
  d = pd.DataFrame([
      proc(tmp_path, 'V6fd', 'none', {}, 'TPU v6 lite', 'a'),
      proc(tmp_path, 'Gfd', 'none', {'JAX_DEFAULT_MATMUL_PRECISION': 'highest'}, 'NVIDIA L4', 'b'),   # FU13's cell
      proc(tmp_path, 'Afd', None, {}, 'NVIDIA A100-SXM4-80GB', 'c'),                                  # bfloat16 trunk
      proc(tmp_path, 'Hfd', 'none', {}, 'NVIDIA A100-SXM4-80GB', 'd')])                               # wrong device
  rows, detail = dtp.check_settings(d, root=tmp_path)
  res = {r['condition']: r['detail'] for r in rows}
  assert res['V6fd'] == 'pass'
  assert all(res[c].startswith('reported') for c in ('Gfd', 'Afd', 'Hfd')) and len(detail) == 3
  assert 'JAX_DEFAULT_MATMUL_PRECISION highest' in res['Gfd']


def test_completeness_marks_part_b_not_run():
  ok = pd.DataFrame([{'condition': 'Gfd', 'repetition': r, 'seed': s, 'target': t, 'sample': k}
                     for r in ('r1', 'r2') for s in range(1, 6) for t in ALL for k in range(5)])
  comp = dtp.completeness(ok)
  assert comp[comp.condition == 'Gfd'].status.eq('complete').all()
  assert comp[comp.condition.isin(['Afd', 'Hfd'])].status.eq('not run').all()
  assert set(comp[comp.condition == 'Hfd'].part) == {'B'}


def test_main_output_counts_differences_of_the_returned_sample_and_its_success():
  rows = []
  for c in ('V6fd', 'Gfd'):
    for t in ('T1', 'T2'):
      for s in range(1, 6):
        for k in range(5):
          score = 0.5 + 0.01 * k + 0.001 * s
          if c == 'Gfd' and t == 'T2' and s == 3 and k == 0:
            score = 0.99                                                      # another main output on T2
          rows.append({'condition': c, 'repetition': 'r1', 'target': t, 'seed': s, 'sample': k, 'ranking_score': score,
                       'm1_success': not (c == 'Gfd' and t == 'T2' and s == 3 and k == 0)})
  mo, det = dtp.main_output_table(pd.DataFrame(rows))
  r = mo.set_index('cell').loc['c_fd']
  assert r.targets == 2 and r.main_output_differs == 1 and r.main_output_success_differs == 1
  assert mo.set_index('cell').loc['c_bd'].targets == 0


# ---------------------------------------------------------------- plans and config diff

def test_plans_are_up_to_date_and_match_the_specification():
  out = subprocess.run([sys.executable, str(REPO / 'harness' / 'plans' / 'make_prep_plans.py'), '--label', 'dtp',
                        '--check'], capture_output=True, text=True)
  assert out.returncode == 0, out.stdout + out.stderr
  parts = pd.read_csv(PARTS)
  assert parts.groupby('arm').processes.sum().to_dict() == {'V6fd': 300, 'V6fd_r2': 10, 'Gfd': 600, 'Afd': 300,
                                                            'Hfd': 300}
  assert parts.groupby('arm').size().to_dict() == {'V6fd': 4, 'V6fd_r2': 1, 'Gfd': 60, 'Afd': 12, 'Hfd': 8}
  assert set(parts[parts.arm.isin(['Afd', 'Hfd'])].row_part) == {'B'}
  assert parts.set_index('arm').loc['V6fd_r2', 'fetch'] == 'full'
  configs = read_configs(REPO / 'harness' / 'configs.yaml')
  for cfg in dtp.CONFIG:
    env = configs[cfg].get('env') or {}
    assert env.get('AF3_FU13_BFLOAT16') == 'none' and 'JAX_DEFAULT_MATMUL_PRECISION' not in env, cfg
  for p in sorted((REPO / 'harness' / 'plans' / 'dtp').glob('*.yaml')):
    t = p.read_text()
    assert 'label: dtp\n' in t and 'launch_by: 2026-12-06\n' in t and 'finish_by: 2026-12-08\n' in t, p
    assert re.search(r'configs:\n  \w+: (\S+)', t).group(1) in dtp.CONFIG, p
  f = REPO / 'results' / 'parts_dtp_v6e.txt'
  if f.is_file():
    lines = [l for l in f.read_text().splitlines() if l and not l.startswith('#')]
    assert len(lines) == 5 and all('ZONES=europe-west4-a,europe-west4-b,europe-west4-c,us-east5-a,us-east5-b '
                                   'REQUEST_VALID_FOR=15m' in l for l in lines)
  for name in ('parts_dtp_l4.txt', 'parts_dtp_h100.txt'):     # us-east4 tried last (setup failures of 2026-11-12)
    f = REPO / 'results' / name
    if f.is_file():
      for line in f.read_text().splitlines()[1:]:
        zones = re.search(r'ZONES=(\S+)', line).group(1).split(',')
        east4 = [i for i, z in enumerate(zones) if z.startswith('us-east4')]
        assert east4 and east4 == list(range(len(zones) - len(east4), len(zones))), (name, line)
  f = REPO / 'results' / 'parts_dtp_a100.txt'
  if f.is_file():
    assert 'ZONES=' not in f.read_text()                       # A100 zones come from A100_REGION_CAPS at launch


def test_config_diff_finds_nothing_beyond_the_allowed_settings(tmp_path):
  out = subprocess.run([sys.executable, str(REPO / 'harness' / 'plans' / 'config_diff_dtp.py'), '--out',
                        str(tmp_path / 'd.md')], capture_output=True, text=True, cwd=REPO)
  if 'No such file' in out.stderr or 'FileNotFoundError' in out.stderr:
    pytest.skip('source sessions not here')
  assert out.returncode == 0, out.stdout[-2000:] + out.stderr[-2000:]
  text = (tmp_path / 'd.md').read_text()
  assert text.rstrip().endswith('Settings that differ beyond the allowed ones: 0')
  assert text.count('JAX_DEFAULT_MATMUL_PRECISION in the new arm\'s environment: none (as intended)') == 4


# ---------------------------------------------------------------- end to end, on relabelled rows of follow-up 13

AN = REPO / 'results' / 'analysis'
NEEDED = [AN / 'fu13_20261112', AN / 'main_20261014' / 'primary_v2', AN / 'prec5_20261107', AN / 'fu14_20261113b',
          AN / 'a100_20261018', AN / 'h100_20261104']


@pytest.mark.skipif(not all((p / 'samples.csv').is_file() for p in NEEDED), reason='earlier analysis outputs not here')
def test_main_runs_end_to_end_on_relabelled_rows(tmp_path):
  """dtp.main on V6f and Gf of follow-up 13 (r1, r2) relabelled as V6fd and Gfd on three targets: every output is
  written, checks 1 and 2 pass, check 4 reports every process (follow-up 13 had JAX_DEFAULT_MATMUL_PRECISION=highest),
  check 5 passes (V6f r2 is bit-identical to r1), c_fd is the c_fh of those targets, and part B reads "not run"."""
  m = pd.read_csv(AN / 'fu13_20261112' / 'samples.csv', low_memory=False)
  g = m[(m.label == 'why') & m.config.isin(['tpu_xla_fu13_f32', 'l4_xla_fu13_f32'])
        & m.target.isin(['7U3J', '7NP6', '7BTT']) & m['sample'].notna()].copy()
  g['config'] = g.config.map({'tpu_xla_fu13_f32': 'tpu_xla_dtp_f32def', 'l4_xla_fu13_f32': 'l4_xla_dtp_f32def'})
  g['label'] = 'dtp'
  g.to_csv(tmp_path / 'samples.csv', index=False)
  out = tmp_path / 'out'
  r = subprocess.run([sys.executable, str(REPO / 'analysis' / 'dtp.py'), '--samples', str(tmp_path / 'samples.csv'),
                      '--out', str(out), '--replicates', '100'], capture_output=True, text=True, cwd=REPO)
  assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-3000:]
  for name in ('checks', 'completeness', 'contrasts', 'cells', 'd_t', 'decomposition', 'attribution', 'agree',
               'recompilation', 'part_b', 'main_output', 'success', 'cost', 'memory', 'stack', 'pairs_seed'):
    assert (out / f'{name}.csv').is_file(), name
  chk = pd.read_csv(out / 'checks.csv')
  assert (chk[chk.check.str.startswith(('1 ', '2 '))].detail == 'pass').all()
  assert chk[chk.check.str.startswith('4 ')].detail.str.startswith('reported').all()
  assert chk[chk.check.str.startswith('5 ')].detail.iloc[0] == 'pass'
  sp = pd.read_csv(out / 'pairs_seed.csv')
  fu13_sp = pd.read_csv(AN / 'fu13_20261112' / 'pairs_seed.csv')
  mine = sp[sp.contrast == 'HW(V6fd,Gfd)'].flip_top1.mean()
  ref = fu13_sp[(fu13_sp.contrast == 'HW(V6f,Gf)') & fu13_sp.target.isin(['7U3J', '7NP6', '7BTT'])].flip_top1.mean()
  assert mine == pytest.approx(ref)
  attr = pd.read_csv(out / 'attribution.csv').iloc[0]
  assert attr.generality.startswith('not computed') and attr.attribution != ''
  assert '## Readings (item e)' in (out / 'report.md').read_text()
