"""Exploratory (c) on follow-up 12's five seeds (analysis/prec5_exploratory.py; section 10, row of 2026-11-10 "Two
computations on frozen outputs for the paper", part 1): the runner calls exploratory.perturbation unchanged on the
three contrasts, restores the module, checks the pairs, refuses to write into the source folder or over an output,
and records the inputs' SHA-256. Synthetic data only; no real AlphaFold3 output is read."""
import hashlib
import pathlib
import subprocess
import sys
import types

import numpy as np
import pandas as pd
import pytest

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'analysis'))
import exploratory as ex  # noqa: E402
import prec5_exploratory as pe  # noqa: E402

TARGETS = [f'T{i}' for i in range(6)]


def synthetic(rng, contrasts=pe.CONTRASTS + ['HW(V6,G)'], scale=None):
  """pairs_seed and pairs_sample rows as analysis/contrasts builds them, with a known per-contrast shift size."""
  seed_rows, sample_rows = [], []
  for j, c in enumerate(contrasts):
    s = scale or 0.0005 * (j + 1)
    for t in TARGETS:
      for sd in pe.SEEDS:
        a = rng.uniform(0.5, 0.9, 5)
        d = rng.normal(0, s, 5)
        b = a + d
        oa, ob = np.argsort(-a), np.argsort(-b)
        seed_rows.append({'contrast': c, 'kind': c.split('(')[0], 'role': 'synthetic', 'target': t, 'seed_a': sd,
                          'seed_b': sd, 'top1_a': int(oa[0]), 'top1_b': int(ob[0]), 'success_a': True,
                          'success_b': True, 'margin_a': float(a[oa[0]] - a[oa[1]]),
                          'margin_b': float(b[ob[0]] - b[ob[1]]), 'flip_top1': bool(oa[0] != ob[0]),
                          'flip_success': False, 'flip_matters': False})
        for k in range(5):
          sample_rows.append({'contrast': c, 'kind': c.split('(')[0], 'role': 'synthetic', 'target': t,
                              'seed_a': sd, 'seed_b': sd, 'sample': k, 'bit_identical': False,
                              'm4_ligand_rmsd': 0.1, 'm4_ca_rmsd_global': 0.1, 'm4_error': '',
                              'm5_ranking_score': float(abs(d[k])), 'm5_mean_plddt': 0.0, 'm5_iptm': 0.0})
  return pd.DataFrame(seed_rows), pd.DataFrame(sample_rows)


def tree(tmp_path, rng):
  root = tmp_path / 'main'
  src = root / pe.SOURCE
  src.mkdir(parents=True)
  seed, sample = synthetic(rng)
  seed.to_csv(src / 'pairs_seed.csv', index=False)
  sample.to_csv(src / 'pairs_sample.csv', index=False)
  return root, src


def args(reps=200):
  return types.SimpleNamespace(seed=2026, replicates=reps)


def only(df):
  return df[df.contrast.isin(pe.CONTRASTS)].reset_index(drop=True)


def test_point_estimates_are_the_definitions_of_c():
  seed, sample = (only(d) for d in synthetic(np.random.default_rng(1)))
  cp = pe.run(seed, sample, args())
  assert list(cp.contrast) == pe.CONTRASTS and list(cp.columns) == pe.COLUMNS
  for r in cp.itertuples():
    g, h = seed[seed.contrast == r.contrast], sample[sample.contrast == r.contrast]
    delta, mu = h.m5_ranking_score.median(), g.margin_a.median()
    assert r.median_abs_score_difference == pytest.approx(delta)
    assert r.median_margin == pytest.approx(mu)
    assert r.delta_over_mu == pytest.approx(delta / mu)
    assert r.predicted_flip_rate == pytest.approx(ex.predicted_flip_rate(delta, mu))
    assert r.observed_flip_rate == pytest.approx(g.flip_top1.mean())
    assert (r.n_seed_pairs, r.n_sample_pairs) == (len(TARGETS) * 5, len(TARGETS) * 25)
    for lo, est, hi in ((r.delta_ci_low, r.median_abs_score_difference, r.delta_ci_high),
                        (r.observed_ci_low, r.observed_flip_rate, r.observed_ci_high),
                        (r.predicted_ci_low, r.predicted_flip_rate, r.predicted_ci_high)):
      assert lo <= est + 1e-12 and est <= hi + 1e-12


def test_same_result_as_perturbation_itself_and_module_restored():
  seed, sample = (only(d) for d in synthetic(np.random.default_rng(2)))
  before = list(ex.CONTRASTS)
  cp = pe.run(seed, sample, args())
  assert ex.CONTRASTS == before
  saved = ex.CONTRASTS
  try:
    ex.CONTRASTS = list(pe.CONTRASTS)
    direct = ex.perturbation(seed, sample, args())
  finally:
    ex.CONTRASTS = saved
  pd.testing.assert_frame_equal(cp, direct[pe.COLUMNS])


def test_module_restored_after_an_error():
  before = list(ex.CONTRASTS)
  with pytest.raises(Exception):
    pe.run(pd.DataFrame({'contrast': []}), pd.DataFrame({'contrast': []}), args())
  assert ex.CONTRASTS == before


def test_larger_shift_gives_larger_ratio_and_prediction():
  seed, sample = (only(d) for d in synthetic(np.random.default_rng(3)))
  cp = pe.run(seed, sample, args()).set_index('contrast')
  # synthetic scales grow with the contrast's position in CONTRASTS
  assert cp.loc['PR(V6)', 'delta_over_mu'] < cp.loc['PR(G)', 'delta_over_mu'] < cp.loc['HWh(V6,G)', 'delta_over_mu']
  assert cp.loc['PR(V6)', 'predicted_flip_rate'] < cp.loc['HWh(V6,G)', 'predicted_flip_rate']


def test_check_flags_bad_pairs():
  seed, sample = (only(d) for d in synthetic(np.random.default_rng(4)))
  assert pe.check(seed, sample, len(TARGETS)) == []
  assert any('targets' in b for b in pe.check(seed, sample, len(TARGETS) + 1))
  s2 = seed[~((seed.contrast == 'PR(G)') & (seed.seed_a == 5))]
  assert any('PR(G)' in b and 'seeds' in b for b in pe.check(s2, sample, len(TARGETS)))
  s3 = seed.copy()
  s3.loc[0, 'flip_top1'] = not s3.loc[0, 'flip_top1']
  assert any('flip_top1' in b for b in pe.check(s3, sample, len(TARGETS)))
  m = sample.copy()
  m.loc[0, 'm5_ranking_score'] = np.nan
  assert any('missing' in b for b in pe.check(seed, m, len(TARGETS)))
  assert any('sample pairs' in b for b in pe.check(seed, sample.iloc[1:], len(TARGETS)))
  assert any('no rows' in b for b in pe.check(seed[seed.contrast != 'HWh(V6,G)'], sample, len(TARGETS)))


def cli(root, out, reps=50):
  return subprocess.run([sys.executable, str(REPO / 'analysis' / 'prec5_exploratory.py'), '--main', str(root),
                         '--out', str(out), '--replicates', str(reps), '--seed', '2026', '--targets',
                         str(len(TARGETS))], capture_output=True, text=True)


def test_cli_writes_outputs_with_hashes_and_leaves_source_unchanged(tmp_path):
  root, src = tree(tmp_path, np.random.default_rng(5))
  before = {p.name: p.read_bytes() for p in src.iterdir()}
  out = root / 'results' / 'analysis' / 'prec5_exploratory_20990101'
  r = cli(root, out)
  assert r.returncode == 0, r.stderr
  assert sorted(p.name for p in out.iterdir()) == ['exploratory_c.csv', 'inputs.tsv']
  cp = pd.read_csv(out / 'exploratory_c.csv')
  assert list(cp.columns) == pe.COLUMNS and list(cp.contrast) == pe.CONTRASTS
  inp = pd.read_csv(out / 'inputs.tsv', sep='\t')
  assert list(inp.path) == [str(pe.SOURCE / 'pairs_seed.csv'), str(pe.SOURCE / 'pairs_sample.csv')]
  for p, h in zip(inp.path, inp.sha256):
    assert hashlib.sha256((root / p).read_bytes()).hexdigest() == h
  assert {p.name: p.read_bytes() for p in src.iterdir()} == before
  # a second run into the same folder is refused
  r2 = cli(root, out)
  assert r2.returncode != 0 and 'must be new' in r2.stderr


def test_cli_refuses_to_write_into_the_source_folder(tmp_path):
  root, src = tree(tmp_path, np.random.default_rng(6))
  r = cli(root, src / 'sub')
  assert r.returncode != 0 and 'must not change' in r.stderr
  assert sorted(p.name for p in src.iterdir()) == ['pairs_sample.csv', 'pairs_seed.csv']


def test_header_matches_the_main_study_c_table():
  """COLUMNS is the header of main_20261014/exploratory/c_perturbation.csv, as exploratory.perturbation writes it."""
  seed, sample = (only(d) for d in synthetic(np.random.default_rng(7)))
  saved = ex.CONTRASTS
  try:
    ex.CONTRASTS = list(pe.CONTRASTS)
    assert list(ex.perturbation(seed, sample, args(20)).columns) == pe.COLUMNS
  finally:
    ex.CONTRASTS = saved
