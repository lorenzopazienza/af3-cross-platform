"""analysis/exploratory.py (section 10, 2026-10-14, exploratory analysis of top-1 flips)."""

import argparse
import math
import pathlib

import numpy as np
import pandas as pd
import pytest

import exploratory as ex

REPO = pathlib.Path(__file__).resolve().parents[2]
MAIN = REPO / 'results' / 'analysis' / 'main_20261014'
needs_main = pytest.mark.skipif(not (MAIN / 'primary_v2' / 'contrasts' / 'pairs_seed.csv').exists(),
                                reason='primary_v2 outputs not present (results/ is gitignored)')


def test_logistic_recovers_known_coefficients():
  rng = np.random.default_rng(0)
  x = rng.normal(size=20000)
  y = (rng.random(20000) < 1 / (1 + np.exp(-(-1.0 + 0.8 * x)))).astype(float)
  a, b = ex.logistic(x, y)
  assert abs(a + 1.0) < 0.05 and abs(b - 0.8) < 0.05
  assert ex.logistic(x, np.zeros_like(x)) is None                     # one outcome only: no fit


@pytest.mark.parametrize('delta,mu', [(0.0003, 0.0012), (0.0007, 0.0012), (0.0001, 0.0012), (0.002, 0.001)])
def test_predicted_flip_rate_matches_a_simulation_of_its_model(delta, mu):
  rng = np.random.default_rng(1)
  n = 400000
  sigma = delta / ex.MEDIAN_ABS_NORMAL
  margin = rng.exponential(mu / math.log(2), n)
  flips = (rng.normal(0, sigma, n) - rng.normal(0, sigma, n)) > margin
  assert abs(ex.predicted_flip_rate(delta, mu) - flips.mean()) < 0.003


def test_predicted_flip_rate_limits():
  assert ex.predicted_flip_rate(0.0, 0.001) == 0.0
  assert 0.49 < ex.predicted_flip_rate(10.0, 1e-6) <= 0.5                # perturbation >> margin: a coin flip
  assert math.isnan(ex.predicted_flip_rate(0.001, 0.0))


def test_top_share():
  assert ex.top_share([5, 0, 0, 0, 0, 0, 0, 0, 0, 5], 1) == 0.5
  assert ex.top_share([1] * 10, 1) == 0.1
  assert math.isnan(ex.top_share([0, 0], 1))


def test_bootstrap_counts_a_target_drawn_twice_as_two_units():
  df = pd.DataFrame({'target': ['A', 'A', 'B', 'B'], 'flip_top1': [True, True, False, False]})
  seen = []
  ex.boot(df, lambda d: seen.append(d.groupby('_unit').size().tolist()) or 0.0, 0, 50)
  assert all(sizes == [2, 2] for sizes in seen)                        # two units of two seeds, whatever was drawn
  lo, hi = ex.boot(df, lambda d: float(d.flip_top1.mean()), 0, 2000)
  assert lo == 0.0 and hi == 1.0


def test_margin_quintiles_and_slope_on_synthetic_pairs():
  rng = np.random.default_rng(2)
  rows = []
  for t in range(60):
    for s in range(1, 6):
      m = 10 ** rng.uniform(-5, -2)
      rows.append({'contrast': 'HW(V5,V6)', 'target': f'T{t:02d}', 'seed_a': s, 'seed_b': s, 'margin_a': m,
                   'flip_top1': bool(rng.random() < 1 / (1 + np.exp(-(-6 - 1.5 * np.log10(m)))))})
  seed = pd.DataFrame(rows)
  args = argparse.Namespace(seed=2026, replicates=200)
  saved = ex.CONTRASTS
  ex.CONTRASTS = ['HW(V5,V6)']
  try:
    qa, la = ex.margin_analysis(seed, args)
  finally:
    ex.CONTRASTS = saved
  assert list(qa.quintile) == [1, 2, 3, 4, 5] and qa.n.sum() == 300
  assert qa.flip_rate.iloc[0] > qa.flip_rate.iloc[-1]                 # small margins flip more
  assert la.slope_per_log10_margin.iloc[0] < 0 and la.slope_ci_high.iloc[0] < 0


@needs_main
def test_sides_match_the_pairs_of_the_primary_analysis():
  seed, _, samples = ex.load(MAIN)
  assert ex.check_sides(seed, samples) == []


@needs_main
def test_flips_that_matter_are_every_flip_that_matters_and_differ_in_success():
  seed, _, samples = ex.load(MAIN)
  ef, es = ex.flips_that_matter(seed, samples)
  assert len(ef) == int(seed.flip_matters.sum())
  assert (ef.success_a != ef.success_b).all()
  assert int(es.flips_that_matter.sum()) == len(ef)
  bad = ef.apply(lambda r: r.ligand_rmsd_b if r.success_a else r.ligand_rmsd_a, axis=1)
  assert ((ef.separated_by == 'PoseBusters') == (bad < 2.0)).all()
