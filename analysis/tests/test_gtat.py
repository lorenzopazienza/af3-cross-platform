"""The L4 against the A100 with Triton attention (drafts/section10_gtat_spec_20261121.md, label gtat): the seed pairs,
the statistics, the readings and the checks of analysis/gtat.py on synthetic per-sample tables, with the registered
tables built by the registered code (followup34.build, contrasts.summarize, contrasts.hw_minus_rc). No real
AlphaFold 3 output is read."""
import pathlib
import sys
import types

import numpy as np
import pandas as pd
import pytest

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'analysis'))
import contrasts  # noqa: E402
import followup34 as f34  # noqa: E402
import gtat  # noqa: E402

ARGS = types.SimpleNamespace(replicates=200, seed=2026)
TARGETS = [f'T{i:02d}' for i in range(57)] + list(gtat.GLYCAN_TARGETS)
ARMS = {('G', 'r1'): ('main', 'l4_main_g_01', 'xla', 'g2-standard-8', 'NVIDIA L4'),
        ('G', 'r2'): ('main', 'l4_main_g_02', 'xla', 'g2-standard-8', 'NVIDIA L4'),
        ('GT', 'r1'): ('main', 'l4_main_gt_01', 'triton', 'g2-standard-8', 'NVIDIA L4'),
        ('GT', 'r2'): ('rcx', 'l4_rcx_gt_r2_01', 'triton', 'g2-standard-8', 'NVIDIA L4'),
        ('A', 'r1'): ('a100', 'a100_a100_a_01', 'xla', 'a2-ultragpu-1g', 'NVIDIA A100-SXM4-80GB'),
        ('A', 'r2'): ('a100', 'a100_a100_a_02', 'xla', 'a2-ultragpu-1g', 'NVIDIA A100-SXM4-80GB'),
        ('AT', 'r1'): ('a100', 'a100_a100_at_01', 'triton', 'a2-ultragpu-1g', 'NVIDIA A100-SXM4-80GB')}


def session(c, rep):
  return f'20261018T000000Z_{ARMS[(c, rep)][1]}'


def samples(seed=0):
  """Per-sample rows; every side draws its own scores, so the top-1 flips often, and r2 copies r1 with small noise."""
  rng = np.random.default_rng(seed)
  base = {}
  rows = []
  for (c, rep), (label, _, att, mt, dev) in ARMS.items():
    for t in TARGETS:
      for s in gtat.SEEDS:
        for k in range(5):
          key = (c, t, s, k)
          if rep == 'r1':
            base[key] = (float(rng.random()), bool(rng.random() < 0.3))
            score, succ = base[key]
          else:
            score, succ = base[key][0] + float(rng.normal(0, 0.02)), base[key][1]
          rows.append({'session': f'results/af3/{session(c, rep)}/run_seed{s}', 'label': label, 'platform': c,
                       'condition': c, 'config': c, 'attention': att, 'machine_type': mt, 'device_kind': dev,
                       'repetition': rep, 'target': t, 'seed': s, 'sample': k, 'exit_code': 0,
                       'ranking_score': score, 'm1_success': False if t in gtat.GLYCAN_TARGETS else succ,
                       'm1_error': "ligand atoms missing in the prediction: ['O1']" if t in gtat.GLYCAN_TARGETS
                       else np.nan,
                       'iptm': 0.9, 'mean_plddt': 90.0, 'model_path': np.nan, 'num_recycles': 10,
                       'num_diffusion_samples': 5})
  return pd.DataFrame(rows)


def registered(ok, specs):
  sp, sm, cons, _ = f34.build(ok, specs)
  return sp, contrasts.summarize(sp, sm, cons, ARGS), cons


@pytest.fixture()
def data():
  d = samples()
  ok = f34.union(d)
  main_sp, _, _ = registered(ok, [('RC(G)', 'RC', ('G', 'r1'), ('G', 'r2'), True),
                                  ('DU', 'DU', ('G', 'r1'), ('GT', 'r1'), True)])
  a_specs = [('RC(A)', 'RC', ('A', 'r1'), ('A', 'r2'), True), ('HW(G,A)', 'HW', ('G', 'r1'), ('A', 'r1'), False),
             ('DU(A)', 'DU', ('A', 'r1'), ('AT', 'r1'), True)]
  a_sp, a_con, a_cons = registered(ok, a_specs)
  rc_g = main_sp[main_sp.contrast == 'RC(G)']
  a_hw = contrasts.hw_minus_rc(pd.concat([a_sp, rc_g], ignore_index=True), {**a_cons, 'RC(G)': ('RC', True)}, ARGS)
  a_hw = a_hw[a_hw.hardware == 'HW(G,A)'].reset_index(drop=True)
  rcx_sp, rcx_con, _ = registered(ok, [('RC(GT)', 'RC', ('GT', 'r1'), ('GT', 'r2'), True)])
  tables = {'main_sp': main_sp, 'a100_sp': a_sp, 'a100_con': a_con, 'a100_hw': a_hw, 'rcx_sp': rcx_sp,
            'rcx_con': rcx_con}
  sides = {'GT': d[(d.platform == 'GT') & (d.repetition == 'r1')], 'AT': d[d.platform == 'AT']}
  freeze = sorted({session(c, r) for c, r in ARMS})
  lists = {'AT': [session('AT', 'r1')], 'freeze': freeze}
  return sides, tables, lists, ok


def test_seed_pairs_equal_the_registered_code(data, tmp_path):
  sides, tables, lists, ok = data
  sp, rates, hwrc, vs, chk = gtat.run(sides, tables, lists, tmp_path, ARGS)
  ref, _, _, _ = f34.build(ok, [('HW(GT,AT)', 'HW', ('GT', 'r1'), ('AT', 'r1'), False)])
  cols = ['target', 'seed_a', 'seed_b', 'top1_a', 'top1_b', 'success_a', 'success_b', 'flip_top1', 'flip_success',
          'flip_matters']
  pd.testing.assert_frame_equal(sp[cols].reset_index(drop=True), ref[cols].reset_index(drop=True), check_dtype=False)
  assert len(sp) == 300 and sp.flip_top1.any()
  r = rates[(rates.targets == '60 targets') & (rates.contrast == 'HW(GT,AT)')].set_index('statistic')
  assert r.loc['flip_top1', 'estimate'] == pytest.approx(sp.flip_top1.mean())
  assert r.loc['flip_top1', 'ci_low'] <= r.loc['flip_top1', 'estimate'] <= r.loc['flip_top1', 'ci_high']
  for f in ('pairs_seed.csv', 'rates.csv', 'hw_minus_rc.csv', 'versus_hw_g_a.csv', 'registered.csv', 'checks.csv',
            'report.md'):
    assert (tmp_path / f).is_file()


def test_hw_minus_rc_gt(data, tmp_path):
  sides, tables, lists, ok = data
  sp, rates, hwrc, vs, chk = gtat.run(sides, tables, lists, tmp_path, ARGS)
  rc = tables['rcx_sp'][tables['rcx_sp'].contrast == 'RC(GT)']
  row = hwrc[(hwrc.targets == '60 targets') & (hwrc.statistic == 'flip_top1')].iloc[0]
  assert row.estimate == pytest.approx(sp.flip_top1.mean() - rc.flip_top1.mean())
  assert row.recompilation == 'RC(GT)' and row.n_targets == 60
  assert row.reading == gtat.read_rc(row.ci_low)
  # no RC(AT) exists: one reference only, three statistics on each target set
  assert len(hwrc) == 6 and set(hwrc.recompilation) == {'RC(GT)'}


def test_glycan_subset(data, tmp_path):
  sides, tables, lists, ok = data
  sp, rates, hwrc, vs, chk = gtat.run(sides, tables, lists, tmp_path, ARGS)
  assert set(rates.n_targets) == {60, 57} and set(hwrc[hwrc.targets.str.startswith('57')].n_targets) == {57}
  g = sp[sp.target.isin(gtat.GLYCAN_TARGETS)]
  assert not g.success_a.any() and not g.success_b.any() and not g.flip_success.any()
  r57 = rates[(rates.targets.str.startswith('57')) & (rates.contrast == 'HW(GT,AT)')].set_index('statistic')
  assert r57.loc['flip_top1', 'estimate'] == pytest.approx(sp[~sp.target.isin(gtat.GLYCAN_TARGETS)].flip_top1.mean())
  assert [c['value'].startswith('as expected') for c in chk if c['check'].startswith('5')] == [True]


def test_paired_difference_is_the_bootstrap_of_hw_minus_rc(data):
  sides, tables, lists, ok = data
  x = tables['a100_sp'][tables['a100_sp'].contrast == 'HW(G,A)']
  y = tables['main_sp'][tables['main_sp'].contrast == 'RC(G)']
  mine = gtat.paired_difference(x, y, ARGS).set_index('statistic')
  ref = contrasts.hw_minus_rc(pd.concat([x, y]), {'HW(G,A)': ('HW', False), 'RC(G)': ('RC', True)}, ARGS)
  ref = ref[ref.reference == 'RC(P)'].set_index('statistic')
  for st in contrasts.FLIP_STATS:
    for col in ('estimate', 'ci_low', 'ci_high'):
      assert mine.loc[st, col] == pytest.approx(ref.loc[st, col], abs=1e-12)


def test_readings():
  assert gtat.read_rc(0.001) == 'beyond RC(GT)' and gtat.read_rc(0.0) == 'not beyond RC(GT)'
  assert gtat.read_rc(float('nan')) == 'not computed'
  assert gtat.read_vs(0.01, 0.2) == 'larger with Triton' and gtat.read_vs(-0.2, -0.01) == 'smaller with Triton'
  assert gtat.read_vs(-0.1, 0.1) == 'no difference shown' and gtat.read_vs(0.0, 0.1) == 'no difference shown'


def test_check_1_stops(data, tmp_path):
  sides, tables, lists, ok = data
  with pytest.raises(SystemExit, match='check 1'):
    gtat.run(sides, tables, dict(lists, AT=['20261018T000000Z_a100_a100_at_02']), tmp_path, ARGS)
  with pytest.raises(SystemExit, match='check 1'):
    gtat.run(dict(sides, GT=sides['GT'].assign(attention='xla')), tables, lists, tmp_path, ARGS)
  with pytest.raises(SystemExit, match='check 1'):
    gtat.run(sides, tables, dict(lists, freeze=[session('AT', 'r1')]), tmp_path, ARGS)


def test_check_2_stops(data, tmp_path):
  sides, tables, lists, ok = data
  at = sides['AT']
  drop = at[(at.target == 'T03') & (at.seed == 2) & (at['sample'] == 4)].index
  with pytest.raises(SystemExit, match='check 2'):
    gtat.run(dict(sides, AT=at.drop(drop)), tables, lists, tmp_path, ARGS)


def test_check_3_stops(data, tmp_path):
  sides, tables, lists, ok = data
  sp = tables['a100_sp'].copy()
  i = sp.index[sp.contrast == 'DU(A)'][0]
  sp.loc[i, 'top1_b'] = (sp.loc[i, 'top1_b'] + 1) % 5
  with pytest.raises(SystemExit, match='check 3'):
    gtat.run(sides, dict(tables, a100_sp=sp), lists, tmp_path, ARGS)


def test_check_4_stops(data, tmp_path):
  sides, tables, lists, ok = data
  bad = tables['rcx_con'].assign(ci_high=tables['rcx_con'].ci_high + 0.01)
  with pytest.raises(SystemExit, match='check 4'):
    gtat.run(sides, dict(tables, rcx_con=bad), lists, tmp_path, ARGS)
  bad = tables['a100_hw'].assign(estimate=tables['a100_hw'].estimate + 0.01)
  with pytest.raises(SystemExit, match='check 4'):
    gtat.run(sides, dict(tables, a100_hw=bad), lists, tmp_path, ARGS)
