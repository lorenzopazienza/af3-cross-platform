"""Follow-up 20 (label aut; notes/analysis_plan.md section 10, row "Follow-up experiment 20"): synthetic tests of the
readings of analysis/aut.py (rules e1 to e3), of the settings check and the cost, of the plans, and an end-to-end run
on relabelled earlier rows. No follow-up 20 output exists or is read."""
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
import aut  # noqa: E402
from run_af3 import read_configs  # noqa: E402

PARTS = REPO / 'harness' / 'plans' / 'aut_parts.csv'
ALL = [f'T{i:02d}' for i in range(60)]


class Args:
  seed, replicates = 2026, 400


def seed_pairs(name, rates, targets, per=5, seed=0):
  rng = np.random.default_rng(seed)
  return pd.DataFrame([{'contrast': name, 'target': t, 'seed_a': s + 1, 'seed_b': s + 1,
                        **{st: bool(rng.random() < r) for st, r in rates.items()}}
                       for t in targets for s in range(per)])


RATES = lambda x: {'flip_top1': x, 'flip_success': x / 4, 'flip_matters': x / 6}  # noqa: E731


def test_conditions_and_contrasts_are_the_specification():
  assert aut.CONFIG == {'l4_xla_noautotune': 'Gna'} and aut.FLAG == '--xla_gpu_autotune_level=0'
  assert [(s[0], s[2], s[3]) for s in aut.SPECS] == [('RC(Gna)', ('Gna', 'r1'), ('Gna', 'r2')),
                                                      ('NA(G)', ('G', 'r1'), ('Gna', 'r1')),
                                                      ('HW(V6,Gna)', ('V6', 'r1'), ('Gna', 'r1'))]
  assert aut.REGISTERED == ['HW(V6,G)', 'HW(V6,Gd)', 'RC(G)']


def test_rule_e1_bit_identical_only_if_every_unit_is_identical():
  assert aut.identity_reading({'units': 300, 'identical': 300, 'value': ''}) == aut.BIT_IDENTICAL
  r = aut.identity_reading({'units': 300, 'identical': 299, 'value': '299/300 identical, 0 differ, 1 lack files'})
  assert r.startswith('not bit-identical without autotuning: 1 of 300')            # missing files: rule 1 does not hold
  assert aut.identity_reading({'units': 0, 'identical': 0, 'value': 'not run'}) == 'not run'


def test_rule_e2_and_the_three_statements_of_rule_e3():
  assert aut.reduced_reading(-0.01) == 'reduced without autotuning'
  assert aut.reduced_reading(0.0) == 'not reduced' and aut.reduced_reading(float('nan')) == 'not computed'
  red, nred = 'reduced without autotuning', 'not reduced'
  assert aut.statement(aut.BIT_IDENTICAL, red).startswith('turning autotuning off removed')
  assert aut.statement('not bit-identical without autotuning: 3 of 300', red).startswith('turning autotuning off reduced')
  assert aut.statement('not bit-identical without autotuning: 3 of 300', nred).startswith('autotuning alone does not account')
  assert aut.statement(aut.BIT_IDENTICAL, nred).startswith('autotuning alone does not account')  # rule 2 decides first


def test_rc_minus_rc_g_reads_reduced_when_rc_gna_is_zero():
  import xvm
  sp = seed_pairs('RC(Gna)', RATES(0.0), ALL, seed=1)
  g = seed_pairs('RC(G)', RATES(0.073), ALL, seed=2)
  rows = pd.DataFrame(xvm.diff_rows(sp, 'RC(Gna)', 'RC(G)', 'x', Args, other=g))
  top = rows[rows.statistic == 'flip_top1'].iloc[0]
  assert top.n_targets == 60 and top.estimate < 0 and aut.reduced_reading(top.ci_high) == 'reduced without autotuning'
  same = pd.DataFrame(xvm.diff_rows(g.assign(contrast='RC(Gna)'), 'RC(Gna)', 'RC(G)', 'x', Args, other=g))
  assert (same.estimate == 0).all() and aut.reduced_reading(same.ci_high.iloc[0]) == 'not reduced'


def test_settings_check_needs_the_flag_as_a_token_and_the_l4(tmp_path):
  def proc(name, flags, device):
    run = tmp_path / name / 'run'
    run.mkdir(parents=True)
    (run / 'session.json').write_text(json.dumps({'xla_env': {'XLA_FLAGS': flags}}))
    return {'session': str(run.relative_to(tmp_path)), 'target': 'T1', 'seed': 1, 'repetition': 'r1', 'condition': 'Gna',
            'device_kind': device}
  d = pd.DataFrame([proc('a', '--xla_gpu_enable_triton_gemm=false --xla_gpu_autotune_level=0', 'NVIDIA L4'),
                    proc('b', '--xla_gpu_enable_triton_gemm=false --xla_gpu_autotune_level=04', 'NVIDIA L4'),
                    proc('c', '--xla_gpu_autotune_level=0', 'NVIDIA A100-SXM4-80GB')])
  rows, detail = aut.check_settings(d, root=tmp_path)
  assert rows[0]['value'] == '1/3 processes' and rows[0]['detail'].startswith('reported') and len(detail) == 2


def test_compile_cost_and_completeness():
  proc = pd.DataFrame([{'condition': c, 'bucket': b, 'exit_code': 0, 'compile_s': v}
                       for c, v in (('Gna', 30.0), ('G', 20.0)) for b in (256, 512)])
  c = aut.compile_cost(proc).set_index('bucket')
  assert c.loc['all', 'compile_ratio'] == pytest.approx(1.5) and c.loc[512, 'compile_ratio'] == pytest.approx(1.5)
  ok = pd.DataFrame([{'condition': 'Gna', 'repetition': r, 'seed': s, 'target': t, 'sample': k}
                     for r in ('r1',) for s in aut.SEEDS for t in ALL for k in range(5)])
  comp = aut.completeness(ok, ALL)
  assert comp[comp.repetition == 'r1'].status.eq('complete').all() and comp[comp.repetition == 'r2'].status.eq('not run').all()


def test_plans_are_up_to_date_and_match_the_specification():
  out = subprocess.run([sys.executable, str(REPO / 'harness' / 'plans' / 'make_prep_plans.py'), '--label', 'aut', '--check'],
                       capture_output=True, text=True)
  assert out.returncode == 0, out.stdout + out.stderr
  parts = pd.read_csv(PARTS)
  assert len(parts) == 24 and parts.processes.sum() == 600 and parts.targets.sum() == 60
  assert set(parts.arm) == {'Gna'} and set(parts.row_part) == {'A'} and set(parts.provisioning) == {'spot'}
  configs = read_configs(REPO / 'harness' / 'configs.yaml')
  assert aut.FLAG in str(configs['l4_xla_noautotune']['env']['XLA_FLAGS']).split()
  seen = set()
  for p in sorted((REPO / 'harness' / 'plans' / 'aut').glob('*.yaml')):
    t = p.read_text()
    assert ('label: aut\n' in t and 'fresh_reps: 2\n' in t and 'rep_start: 1\n' in t and 'eigen_source: archive\n' in t
            and '  l4: l4_xla_noautotune\n' in t), p
    seen |= set(re.search(r'default: \[([^\]]*)\]', t).group(1).split(', '))
  assert seen == set(pd.read_csv(REPO / 'inputs' / 'manifest.csv').pdb_id)


PRIMARY = REPO / 'results' / 'analysis' / 'main_20261014' / 'primary_v2'


@pytest.mark.skipif(not (PRIMARY / 'samples.csv').is_file(), reason='earlier analysis outputs not here')
def test_main_runs_end_to_end_on_relabelled_rows(tmp_path):
  """aut.main on G r1 and r2 of the main study relabelled as Gna, four targets: every output is written, NA(G) is 0,
  RC(Gna) equals RC(G) on these targets (RC(Gna) minus RC(G) is 0, "not reduced", the third statement), and the
  settings check reports the missing flag without stopping."""
  targets = ['7U3J', '7NP6', '7BTT', '8G6P']
  m = pd.read_csv(PRIMARY / 'samples.csv', low_memory=False)
  g = m[(m.label == 'main') & (m.platform == 'G') & m.repetition.isin(['r1', 'r2']) & m.target.isin(targets)
        & m['sample'].notna()].copy()
  g['label'], g['config'] = 'aut', 'l4_xla_noautotune'
  g.to_csv(tmp_path / 'samples.csv', index=False)
  out = tmp_path / 'out'
  r = subprocess.run([sys.executable, str(REPO / 'analysis' / 'aut.py'), '--samples', str(tmp_path / 'samples.csv'),
                      '--out', str(out), '--replicates', '200'], capture_output=True, text=True, cwd=REPO)
  assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-3000:]
  for name in ('checks', 'completeness', 'contrasts', 'identity', 'rc_vs_g', 'registered', 'cost', 'stack', 'pairs_seed'):
    assert (out / f'{name}.csv').is_file(), name
  chk = pd.read_csv(out / 'checks.csv').set_index('check')
  assert chk.loc['1 every process exits 0 or its failure is recorded', 'detail'] == 'pass'
  assert chk.loc['4a the flag and the device in every process', 'detail'].startswith('reported')
  con = pd.read_csv(out / 'contrasts.csv')
  assert con[(con.contrast == 'NA(G)') & (con.statistic == 'flip_top1')].estimate.iloc[0] == 0
  vs = pd.read_csv(out / 'rc_vs_g.csv')
  top = vs[vs.statistic == 'flip_top1'].iloc[0]
  assert top.estimate == pytest.approx(0) and top.reading == 'not reduced' and top.n_targets == 4
  assert set(pd.read_csv(out / 'registered.csv').contrast) == set(aut.REGISTERED)
  rep = (out / 'report.md').read_text()
  assert 'Rule 3: autotuning alone does not account' in rep
  cst = pd.read_csv(out / 'cost.csv')
  allb = cst[cst.bucket == 'all'].iloc[0]                                       # G: all 60 targets, Gna: these four
  assert allb.n_new == 20 and allb.n_reference == 300 and 0 < allb.wall_ratio < 10 and allb.compile_ratio == allb.compile_ratio
