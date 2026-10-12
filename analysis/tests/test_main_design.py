"""The main-study design (notes/analysis_plan.md section 7 and the section 10 rows of 2026-10-12) through
analysis/contrasts.py on synthetic data: label main, seeds 1 to 5, Q3 for n = 1 to 5, DU (GT against G), Gd with
5 seeds, V6h and Gh with 2 seeds, C on its subset of targets. Scaled down: 8 targets (C on 6), the warm rerun on
3 of them."""

import argparse

import pandas as pd
import pytest

import contrasts

TARGETS = [f'T{i}' for i in range(1, 9)]
C_TARGETS = TARGETS[:6]
WARM = TARGETS[:3]
SEEDS5, SEEDS2 = (1, 2, 3, 4, 5), (1, 2)
DESIGN = {  # condition: (targets, seeds, repetitions with r1/r2, warm)
    'V5': (TARGETS, SEEDS5, ('r1', 'r2'), True), 'V6': (TARGETS, SEEDS5, ('r1', 'r2'), True),
    'G': (TARGETS, SEEDS5, ('r1', 'r2'), True), 'Gd': (TARGETS, SEEDS5, ('r1', 'r2'), True),
    'GT': (TARGETS, SEEDS5, ('r1',), False), 'V6h': (TARGETS, SEEDS2, ('r1',), False),
    'Gh': (TARGETS, SEEDS2, ('r1',), False), 'C': (C_TARGETS, SEEDS2, ('r1',), False),
}


def table(flips=(), label='main', only=None):
  """Rows of the design; top-1 is sample 0 except for (condition, repetition, target) in flips (sample 1)."""
  rows = []
  for cond, (targets, seeds, reps, warm) in DESIGN.items():
    if only and cond not in only:
      continue
    cells = [(rep, t, s) for rep in reps for t in targets for s in seeds]
    cells += [('w', t, 1) for t in WARM] if warm else []
    for rep, t, s in cells:
      for k in range(5):
        score = 0.95 if (cond, rep, t) in flips and k == 1 else 0.9 - 0.1 * k
        rows.append({'session': f'{label}_{cond}_{rep}', 'label': label, 'platform': cond, 'libtpu': None,
                     'target': t, 'seed': s, 'repetition': rep, 'sample': k, 'num_recycles': 10,
                     'num_diffusion_samples': 5, 'exit_code': 0, 'failure': None, 'ranking_score': score,
                     'mean_plddt': 80.0, 'iptm': 0.8, 'm1_success': k == 0, 'model_path': None})
  return pd.DataFrame(rows)


def run(df, tmp_path, labels='main', replicates=200):
  path = tmp_path / 'samples.csv'
  df.to_csv(path, index=False)
  args = argparse.Namespace(samples=str(path), labels=labels, include=None, exclude=None, condition='platform',
                            replicates=replicates, seed=2026)
  d, ok = contrasts.load(args)
  sp, sm, cons, excluded = contrasts.build_pairs(ok)
  return args, d, ok, sp, sm, cons, excluded.set_index('contrast')


SECTION5 = {'RC(V5)', 'RC(V6)', 'RC(G)', 'W(V5)', 'W(V6)', 'W(G)', 'SD(V5)', 'SD(V6)', 'SD(G)', 'SD(GT)', 'SD(C)',
            'HW(V5,V6)', 'HW(V5,G)', 'HW(V5,GT)', 'HW(V5,C)', 'HW(V6,G)', 'HW(V6,GT)', 'HW(V6,C)', 'HW(G,C)',
            'HW(GT,C)', 'DU'}
SECTION11 = {'PR(V6)', 'PR(G)', 'HWh(V6,G)', 'HWh(G,C)'}
GD = {'RC(Gd)', 'W(Gd)', 'DET', 'HW(V6,Gd)', 'HW(Gd,C)'}


def test_main_design_contrasts_and_pairs(tmp_path):
  # A pilot2 copy of G is in the table: --labels main must leave it out (otherwise two sessions clash).
  df = pd.concat([table(), table(label='pilot2', only={'G'})])
  _, d, ok, sp, sm, cons, ex = run(df, tmp_path)
  assert set(d.label) == {'main'}
  assert set(cons) == SECTION5 | SECTION11 | GD
  assert cons['DU'] == ('DU', True) and contrasts.role('DU', True) == 'default use'
  for name in ('HW(V5,V6)', 'HW(V6,G)', 'HW(G,C)'):
    assert contrasts.role(*cons[name]) == 'primary pair'
  for name in ('HW(V5,GT)', 'HW(V6,GT)', 'HW(GT,C)', 'HW(V5,G)', 'HW(V6,Gd)', 'HW(Gd,C)'):
    assert contrasts.role(*cons[name]) == 'secondary pair'
  n, nc = len(TARGETS), len(C_TARGETS)
  expect = {'RC(G)': (n * 5, 0, 0), 'RC(Gd)': (n * 5, 0, 0), 'W(G)': (3, n * 5 - 3, 0), 'W(Gd)': (3, n * 5 - 3, 0),
            'DU': (n * 5, 0, 0), 'DET': (n * 5, 0, 0), 'HW(V6,Gd)': (n * 5, 0, 0),
            'SD(G)': (n * 10, 0, 0), 'SD(GT)': (n * 10, 0, 0), 'SD(C)': (nc, 0, 0),
            'HW(G,C)': (nc * 2, n * 5 - nc * 2, 0), 'HW(Gd,C)': (nc * 2, n * 5 - nc * 2, 0),
            'PR(V6)': (n * 2, n * 3, 0), 'PR(G)': (n * 2, n * 3, 0), 'HWh(V6,G)': (n * 2, 0, 0),
            'HWh(G,C)': (nc * 2, n * 2 - nc * 2, 0)}
  for name, (used, only_a, only_b) in expect.items():
    assert (ex.loc[name].pairs_used, ex.loc[name].excluded_only_a, ex.loc[name].excluded_only_b) == \
        (used, only_a, only_b), name


def test_du_and_gd_statistics(tmp_path):
  # GT moves its top-1 on T1 (all 5 seeds); Gd r2 on T2 (all 5 seeds); nothing else moves.
  args, _, ok, sp, sm, cons, _ = run(table(flips={('GT', 'r1', 'T1'), ('Gd', 'r2', 'T2')}), tmp_path)
  s = contrasts.summarize(sp, sm, cons, args).set_index(['contrast', 'statistic'])
  assert s.loc[('DU', 'flip_top1')].estimate == pytest.approx(5 / 40)
  assert s.loc[('DU', 'flip_matters')].estimate == pytest.approx(5 / 40)
  assert s.loc[('RC(Gd)', 'flip_top1')].estimate == pytest.approx(5 / 40)
  assert s.loc[('DET', 'flip_top1')].estimate == 0 and s.loc[('RC(G)', 'flip_top1')].estimate == 0
  r = s.loc[('DU', 'flip_top1')]
  assert r.n == 40 and r.n_targets == 8 and r.ci_low <= r.estimate <= r.ci_high
  h = contrasts.hw_minus_rc(sp, cons, args).set_index(['hardware', 'recompilation', 'statistic'])
  assert h.loc[('HW(V6,Gd)', 'RC(Gd)', 'flip_top1')].estimate == pytest.approx(0 - 5 / 40)
  assert 'DU' not in h.index.get_level_values(0)        # DU is not a hardware pair: no HW - RC row


def test_q3_for_one_to_five_seeds(tmp_path):
  args, _, ok, *_ = run(table(), tmp_path)
  rows, answer = contrasts.q3(ok, args)
  assert sorted(rows.n_seeds.unique()) == [1, 2, 3, 4, 5]
  rates = rows[rows.statistic == 'success_rate']
  for n in (1, 2, 3, 4, 5):
    conds = set(rates[rates.n_seeds == n].condition)
    # C has seeds 1 and 2 only; the h arms and Gd are outside Q3 (sections 10 and 11).
    assert conds == ({'V5', 'V6', 'G', 'GT', 'C'} if n <= 2 else {'V5', 'V6', 'G', 'GT'})
  assert (rates.estimate == 1.0).all() and answer == 1


def test_checks_on_the_main_design(tmp_path):
  args, d, ok, sp, sm, cons, _ = run(table(), tmp_path)
  c = contrasts.checks(d, ok, sm, args)
  one = c[c.check == '1 exit codes']
  assert set(one.condition) == set(DESIGN) and all(v.split(' ')[0].split('/')[0] == v.split(' ')[0].split('/')[1]
                                                   for v in one.value)
  assert set(c[c.check.str.startswith('3 ')].condition) == {'V6'}
  cover = c[c.check == '2 every arm with a warm rerun is present'].iloc[0]
  assert cover.detail == 'pass'
