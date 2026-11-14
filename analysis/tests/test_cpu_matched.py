"""Matched recompilation reference for the CPU pair (drafts/section10_cpu_matched_spec_20261114.md, label cpu_matched):
the reproduction check, the seed restriction and the readings of analysis/cpu_matched.py on synthetic seed pairs, with
the registered tables built by the registered contrasts.hw_minus_rc. No real AlphaFold3 output is read."""
import pathlib
import sys
import types

import numpy as np
import pandas as pd
import pytest

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'analysis'))
import contrasts  # noqa: E402
import cpu_matched as cm  # noqa: E402

ARGS = types.SimpleNamespace(replicates=300, seed=2026)


def pairs(name, kind, targets, seeds, rate_by_seed, rng):
  rows = []
  for t in targets:
    for s in seeds:
      f = rng.random() < rate_by_seed(s)
      succ = f and rng.random() < 0.5
      rows.append({'contrast': name, 'kind': kind, 'role': '', 'target': t, 'seed_a': s, 'seed_b': s, 'top1_a': 0,
                   'top1_b': int(f), 'success_a': True, 'success_b': not succ, 'margin_a': 0.0, 'margin_b': 0.0,
                   'flip_top1': f, 'flip_success': succ, 'flip_matters': f and succ})
  return pd.DataFrame(rows)


@pytest.fixture()
def tables():
  rng = np.random.default_rng(3)
  t60 = [f'T{i:02d}' for i in range(20)]
  rc = pairs('RC(G)', 'RC', t60, [1, 2, 3, 4, 5], lambda s: 0.02 if s <= 2 else 0.3, rng)   # seeds 3 to 5 flip more
  hw_main = pairs('HW(G,C)', 'HW', t60[:6], [1, 2], lambda s: 0.2, rng)
  hw60 = pairs('HW(G,C)', 'HW', t60, [1, 2], lambda s: 0.2, rng)
  cons = {'HW(G,C)': ('HW', True), 'RC(G)': ('RC', True)}
  main_sp = pd.concat([rc, hw_main], ignore_index=True)
  return {'main_sp': main_sp, 'main_hw': contrasts.hw_minus_rc(main_sp, cons, ARGS),
          'cpu60_sp': hw60, 'cpu60_hw': contrasts.hw_minus_rc(pd.concat([hw60, rc]), cons, ARGS)}


def test_matched_uses_the_same_seeds(tables, tmp_path):
  df = cm.run(tables, tmp_path, ARGS)
  r = df[(df.pair == 'follow-up 4, 60 targets') & (df.statistic == 'flip_top1')].iloc[0]
  hw = tables['cpu60_sp']
  rc = tables['main_sp'][(tables['main_sp'].contrast == 'RC(G)') & tables['main_sp'].seed_a.isin([1, 2])]
  assert r.estimate == pytest.approx(hw.flip_top1.mean() - rc.flip_top1.mean())
  assert r.hw_seeds == '1 2' and r.n_targets == 20
  reg = tables['cpu60_hw']
  g = reg[(reg.recompilation == 'RC(G)') & (reg.statistic == 'flip_top1')].iloc[0]
  assert r.registered_estimate == pytest.approx(g.estimate)
  assert r.estimate > r.registered_estimate      # the synthetic RC(G) flips less on seeds 1 and 2


def test_main_pair_on_its_own_targets(tables, tmp_path):
  df = cm.run(tables, tmp_path, ARGS)
  assert (df[df.pair == 'main study, 6 targets'].n_targets == 6).all()
  assert len(df) == 6 and set(df.changed) <= {'yes', 'no'}
  assert (tmp_path / 'matched.csv').is_file() and (tmp_path / 'report.md').read_text().startswith('# Matched')


def test_reproduction_check_stops(tables, tmp_path):
  bad = dict(tables)
  bad['cpu60_hw'] = tables['cpu60_hw'].assign(ci_low=tables['cpu60_hw'].ci_low + 0.01)
  with pytest.raises(SystemExit, match='not reproduced'):
    cm.run(bad, tmp_path, ARGS)


def test_reading():
  assert cm.read(0.001) == 'beyond' and cm.read(0.0) == 'not beyond' and cm.read(float('nan')) == 'not computed'
