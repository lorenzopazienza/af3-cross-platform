"""Ranking rules and their cost (section 10, row C1 of drafts/section10_campaign_ranking_20261110.md): the rules (i) to
(iv), the cost and the CLI of analysis/ranking_rules.py on synthetic per-sample tables. No real AlphaFold3 output is
read."""
import pathlib
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'analysis'))
import exploratory as ex  # noqa: E402
import ranking_rules as rr  # noqa: E402


def cube_of(score, succ=None, summ=None):
  score = np.asarray(score, float)
  return rr.Cube(score, np.round(score, 2) if summ is None else np.asarray(summ, float),
                 np.zeros(score.shape, bool) if succ is None else np.asarray(succ, bool))


def metric(m, name):
  val, mask = m[name]
  return float(np.where(mask, val, 0).sum() / mask.sum()) if mask.sum() else np.nan


# ---- (i) seed consensus

def test_consensus_picks_the_block_argmax_over_disjoint_consecutive_seeds():
  rng = np.random.default_rng(0)
  score = rng.random((3, 20, 5))
  for n in rr.N_CONSENSUS:
    b = rr.block(score, n)
    assert b.shape == (3, 20 // n, 5 * n)
    for t in range(3):
      for j in range(20 // n):
        seeds = range(j * n, (j + 1) * n)                       # seeds j n + 1 to (j + 1) n, 1-based
        flat = score[t, list(seeds)].ravel()                     # (seed, sample) order
        assert np.array_equal(b[t, j], flat)
        k = b[t, j].argmax()
        assert score[t, seeds[0] + k // 5, k % 5] == flat.max()  # the global argmax of the block
  # every seed in exactly one block
  idx = rr.block(np.arange(20 * 5).reshape(1, 20, 5), 5)[0]
  assert sorted(idx.ravel()) == list(range(100))


def test_consensus_flip_and_success():
  a = np.zeros((1, 2, 5))
  b = np.zeros((1, 2, 5))
  a[0, 0, 1], a[0, 1, 3] = 0.9, 0.8                              # a: seed 1 sample 1 wins the block of 2
  b[0, 0, 1], b[0, 1, 3] = 0.8, 0.9                              # b: seed 2 sample 3 wins it
  succ = np.zeros((1, 2, 5), bool)
  succ[0, 1, 3] = True
  m1 = rr.consensus(cube_of(a, succ), cube_of(b, succ), 1)
  assert metric(m1, 'flip_top1') == 0.0                          # per seed the same sample wins
  m2 = rr.consensus(cube_of(a, succ), cube_of(b, succ), 2)
  assert metric(m2, 'flip_top1') == 1.0 and metric(m2, 'flip_success') == 1.0
  assert metric(m2, 'success_a') == 0.0 and metric(m2, 'success_b') == 1.0 and metric(m2, 'success_difference') == -1.0


# ---- (ii) tie tolerance

def test_t0_equals_plain_argmax_and_has_no_ties_without_exact_ties():
  rng = np.random.default_rng(1)
  a, b = cube_of(rng.random((4, 5, 5))), cube_of(rng.random((4, 5, 5)))
  for n in rr.N_SHORT:
    t0 = rr.tie_metrics(a, b, n, 0.0)
    plain = rr.consensus(a, b, n)
    assert metric(t0, 'tie_either') == 0.0
    for k in rr.METRICS:
      assert metric(t0, k) == metric(plain, k)
    assert metric(t0, 'conflict') == metric(plain, 'flip_top1')


def test_margin_below_t_declares_a_tie_and_removes_the_flip():
  a = cube_of([[[0.800, 0.799, 0.1, 0.1, 0.1]]])                 # margin 0.001 on a
  b = cube_of([[[0.799, 0.800, 0.1, 0.1, 0.1]]])                 # b picks the other sample
  assert metric(rr.tie_metrics(a, b, 1, 0.0), 'flip_top1') == 1.0
  m = rr.tie_metrics(a, b, 1, 0.002)
  assert metric(m, 'tie_either') == 1.0 and metric(m, 'flip_top1') == 0.0
  assert metric(m, 'conflict') == 0.0                            # the sets {0, 1} and {0, 1} overlap
  assert np.isnan(metric(m, 'flip_success'))                     # no unit with one sample on both platforms
  assert metric(rr.tie_metrics(a, b, 1, 0.0005), 'tie_either') == 0.0


def test_conflict_is_disjoint_sets():
  a = cube_of([[[0.900, 0.899, 0.1, 0.1, 0.1]]])                 # a reports {0, 1} at t = 0.002
  b = cube_of([[[0.1, 0.1, 0.700, 0.699, 0.1]]])                 # b reports {2, 3}
  m = rr.tie_metrics(a, b, 1, 0.002)
  assert metric(m, 'conflict') == 1.0 and metric(m, 'flip_top1') == 0.0
  c = cube_of([[[0.1, 0.900, 0.899, 0.1, 0.1]]])                 # c reports {1, 2}: overlaps a
  assert metric(rr.tie_metrics(a, c, 1, 0.002), 'conflict') == 0.0


# ---- (iii) rounded

def test_rounded_ties_go_to_the_lowest_seed_sample_index():
  score = np.array([[[0.801, 0.804, 0.1, 0.1, 0.1], [0.1, 0.1, 0.1, 0.1, 0.803]]])
  summ = np.round(score, 2)                                      # 0.80 three times
  a = cube_of(score, summ=summ)
  b = cube_of(score, summ=summ)
  pick = rr.block(a.summ, 2).argmax(-1)
  assert pick[0, 0] == 0                                         # seed 1 sample 0, the lowest index
  assert rr.block(a.score, 2).argmax(-1)[0, 0] == 1              # full precision: seed 1 sample 1
  succ = np.zeros(score.shape, bool)
  succ[0, 0, 1] = True
  m_full = rr.rounded_metrics(cube_of(score, succ, summ), b, 2, False)
  m_round = rr.rounded_metrics(cube_of(score, succ, summ), b, 2, True)
  assert metric(m_full, 'success_a') == 1.0 and metric(m_round, 'success_a') == 0.0


def test_rounded_flip_appears_when_rounding_hides_a_difference():
  a = cube_of([[[0.804, 0.801, 0.1, 0.1, 0.1]]])
  b = cube_of([[[0.801, 0.806, 0.1, 0.1, 0.1]]], summ=[[[0.80, 0.81, 0.1, 0.1, 0.1]]])
  assert metric(rr.rounded_metrics(a, b, 1, False), 'flip_top1') == 1.0
  assert metric(rr.rounded_metrics(a, b, 1, True), 'flip_top1') == 1.0
  c = cube_of([[[0.801, 0.804, 0.1, 0.1, 0.1]]])                 # rounded 0.80 and 0.80: lowest index on both
  assert metric(rr.rounded_metrics(a, c, 1, True), 'flip_top1') == 0.0


# ---- (iv) model

def test_model_is_zero_for_zero_differences():
  rng = np.random.default_rng(2)
  A = rng.random((12, 5))
  tg = np.repeat(np.arange(4), 3)
  est, bs = rr.resampling_model(A, np.zeros_like(A), tg, 4, rr.Boot(4, 50, 2026).W)
  assert est == 0.0 and np.nanmax(bs) == 0.0
  assert ex.predicted_flip_rate(0.0, 0.1) == 0.0


def test_model_keeps_the_five_components_together():
  A = np.array([[1.0, 0.9, 0, 0, 0], [1.0, 0.9, 0, 0, 0]])
  D = np.array([[0.3, 0.3, 0, 0, 0], [-0.2, -0.2, 0, 0, 0]])     # shifts that move the top two together
  tg = np.array([0, 1])
  assert rr.resampling_model(A, D, tg, 2)[0] == 0.0              # mixing components (-0.2 with 0.3) would flip
  D2 = np.array([[0.0, 0.2, 0, 0, 0], [0, 0, 0, 0, 0]])         # target 0's vector flips target 1's unit
  assert rr.resampling_model(A, D2, tg, 2)[0] == 0.5
  # the own target's vectors are not used: target 0 holds units 0 and 1, target 1 unit 2
  A3 = np.tile([1.0, 0.85, 0, 0, 0], (3, 1))
  D3 = np.array([[0.0, 0.2, 0, 0, 0], [0.0, 0.2, 0, 0, 0], [0, 0, 0, 0, 0]])
  assert rr.resampling_model(A3, D3, np.array([0, 0, 1]), 2)[0] == pytest.approx(1 / 3)   # with them: 2/3


def test_model_bootstrap_with_every_target_once_equals_the_estimate():
  rng = np.random.default_rng(3)
  A, D = rng.random((15, 5)), 0.05 * rng.standard_normal((15, 5))
  tg = np.repeat(np.arange(5), 3)
  est, bs = rr.resampling_model(A, D, tg, 5, np.ones((2, 5)))
  assert np.allclose(bs, est)
  F = rr.flip_matrix(A, D)
  brute = np.mean([F[u, tg != tg[u]].mean() for u in range(15)])
  assert est == pytest.approx(brute)


def test_weighted_median_matches_numpy_median():
  rng = np.random.default_rng(4)
  vals, tg = rng.random(20), np.repeat(np.arange(5), 4)
  W = rr.Boot(5, 30, 7).W
  got = rr.weighted_median(vals, tg, W)
  for w, g in zip(W, got):
    assert g == np.median(np.repeat(vals, w[tg].astype(int)))


# ---- cost

def test_cost_is_n_times_the_bucket_median_averaged_over_targets():
  cs = pd.DataFrame([{'platform': p, 'bucket': b, 'cache': c, 'median_wall_s': w * m,
                      'median_usd_per_structure_standard': 0.1 * w * m, 'median_usd_per_structure_spot': 0.05 * w * m,
                      'median_usd_per_structure_amortised_standard': 0.08 * w * m,
                      'median_usd_per_structure_amortised_spot': 0.04 * w * m}
                     for p, m in (('V5', 1.0), ('G', 2.0), ('V6', 9.0)) for b, w in ((256, 100.0), (512, 200.0))
                     for c in ('fresh', 'warm')])
  cs.loc[cs.cache == 'warm', 'median_wall_s'] = 1.0
  co = rr.cost(cs, [256, 512, 512], rr.Boot(3, 20, 2026))
  assert set(co.platform) == {'V5', 'G'}
  get = lambda p, n, q: co[(co.platform == p) & (co.n == n) & (co.quantity == q)].estimate.iloc[0]  # noqa: E731
  for n in rr.N_CONSENSUS:
    assert get('V5', n, 'processes_per_prediction') == n
    assert get('V5', n, 'wall_s_per_prediction') == pytest.approx(n * (100 + 200 + 200) / 3)
    assert get('G', n, 'usd_spot_per_prediction') == pytest.approx(n * 0.05 * 2 * 500 / 3)


# ---- end to end

def samples(label, platform, rep, targets, seeds, rng, buckets):
  out = []
  for t in targets:
    for s in seeds:
      for k in range(5):
        x = float(rng.random())
        out.append({'session': 's', 'label': label, 'platform': platform, 'repetition': rep, 'cache': 'fresh',
                    'target': t, 'bucket': buckets[t], 'seed': s, 'sample': k, 'num_recycles': 10,
                    'num_diffusion_samples': 5, 'exit_code': 0, 'ranking_score': x,
                    'ranking_score_summary': round(x, 2), 'm1_success': bool(rng.random() < 0.5)})
  return out


def tree(root, drop=None):
  rng = np.random.default_rng(5)
  targets = ['T1', 'T2', 'T3', 'T4']
  buckets = {'T1': 256, 'T2': 256, 'T3': 512, 'T4': 512}
  main = []
  for p in ('V5', 'V6', 'G', 'Gd'):
    for r in ('r1', 'r2'):
      main += samples('main', p, r, targets, rr.SEEDS_MAIN, rng, buckets)
  main += samples('main', 'GT', 'r1', targets, rr.SEEDS_MAIN, rng, buckets)
  for p in ('V6h', 'Gh'):
    main += samples('main', p, 'r1', targets, [1, 2], rng, buckets)
  main += samples('main', 'G', 'w', targets, [1], rng, buckets)
  seeds = [r for p in ('V5', 'G') for r in samples('seeds', p, 'r1', targets, range(6, 21), rng, buckets)]
  h100 = [r for rep in ('r1', 'r2') for r in samples('h100', 'H', rep, targets, rr.SEEDS_MAIN, rng, buckets)]
  a100 = [r for rep in ('r1', 'r2') for r in samples('a100', 'A', rep, targets, rr.SEEDS_MAIN, rng, buckets)]
  cs = pd.DataFrame([{'platform': p, 'bucket': b, 'cache': 'fresh', 'processes': 10, 'targets': 2, 'median_wall_s': 100.0 * b,
                      'median_usd_per_structure_standard': 0.1, 'median_usd_per_structure_amortised_standard': 0.09,
                      'median_usd_per_structure_spot': 0.05, 'median_usd_per_structure_amortised_spot': 0.04}
                     for p in ('V5', 'G') for b in (256, 512)])
  frames = {rr.MAIN: pd.DataFrame(main), rr.SEEDS: pd.DataFrame(seeds), rr.H100: pd.DataFrame(h100),
            rr.A100: pd.DataFrame(a100), rr.COST: cs}
  if drop:
    path, cond = drop
    frames[path] = frames[path][~cond(frames[path])]
  for path, df in frames.items():
    (root / path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(root / path, index=False)
  return root


def cli(root, out, *extra):
  return subprocess.run([sys.executable, str(REPO / 'analysis' / 'ranking_rules.py'), '--main', str(root), '--out',
                         str(out), '--replicates', '40', *extra], capture_output=True, text=True)


def test_cli_end_to_end_writes_every_output_and_is_deterministic(tmp_path):
  root = tree(tmp_path / 'main')
  r1, r2 = cli(root, tmp_path / 'o1'), cli(root, tmp_path / 'o2')
  assert r1.returncode == 0, r1.stdout + r1.stderr
  assert r2.returncode == 0, r2.stdout + r2.stderr
  names = ['seed_consensus.csv', 'tie_tolerance.csv', 'rounded.csv', 'model.csv', 'cost.csv', 'report.md', 'inputs.tsv']
  for n in names:
    assert (tmp_path / 'o1' / n).read_bytes() == (tmp_path / 'o2' / n).read_bytes(), n
  sc = pd.read_csv(tmp_path / 'o1' / 'seed_consensus.csv')
  assert sorted(sc[sc.pair == 'HW(V5,G)'].n.unique()) == rr.N_CONSENSUS
  assert sorted(sc[sc.pair == 'RC(G)'].n.unique()) == rr.N_SHORT
  assert (sc[(sc.pair == 'HW(V5,G)') & (sc.metric == 'flip_top1')].set_index('n').units.to_dict()
          == {1: 80, 2: 40, 5: 16, 10: 8, 20: 4})
  tt = pd.read_csv(tmp_path / 'o1' / 'tie_tolerance.csv')
  assert sorted(tt.t.unique()) == rr.TOLERANCES and set(tt.pair) == {p for p, _, _ in rr.PAIRS}
  assert (tt[tt.t == 0].difference == 0).all()
  md = pd.read_csv(tmp_path / 'o1' / 'model.csv')
  assert list(md.contrast) == ex.CONTRASTS + ['HW(V5,G)'] * 3
  assert md[md.contrast == 'PR(V6)'].units.iloc[0] == 8           # V6h ran seeds 1 and 2
  inputs = pd.read_csv(tmp_path / 'o1' / 'inputs.tsv', sep='\t')
  assert list(inputs.path) == [rr.MAIN, rr.SEEDS, rr.H100, rr.A100, rr.COST] and (inputs.sha256.str.len() == 64).all()
  rep = (tmp_path / 'o1' / 'report.md').read_text()
  assert 'outside the pre-registered study' in rep and 'descriptive' in rep and 'exploratory' in rep
  r3 = cli(root, tmp_path / 'o3', '--seed', '7')
  assert r3.returncode == 0
  assert (tmp_path / 'o3' / 'seed_consensus.csv').read_bytes() != (tmp_path / 'o1' / 'seed_consensus.csv').read_bytes()
  for p in [tmp_path / 'o1' / n for n in names] + [pathlib.Path(rr.__file__), pathlib.Path(__file__)]:
    assert chr(0x2014) not in p.read_text()       # no em dash


@pytest.mark.parametrize('drop, message', [
    ((rr.MAIN, lambda d: d.platform == 'V6'), 'no rows for V6 r1'),
    ((rr.SEEDS, lambda d: d.seed == 20), 'without the 5 samples'),
    ((rr.A100, lambda d: d.repetition == 'r2'), 'no rows for A r2'),
    ((rr.H100, lambda d: (d.target == 'T2') & (d.seed == 3) & (d['sample'] == 4)), 'without the 5 samples'),
    ((rr.COST, lambda d: (d.platform == 'G') & (d.bucket == 512)), 'no fresh row for G'),
])
def test_cli_stops_when_inputs_are_missing(tmp_path, drop, message):
  r = cli(tree(tmp_path / 'main', drop), tmp_path / 'o')
  assert r.returncode != 0 and message in r.stderr, r.stderr
