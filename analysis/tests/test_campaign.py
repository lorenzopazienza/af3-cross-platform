"""Campaign simulation (notes/analysis_plan.md section 10, row A2 of drafts/section10_campaign_ranking_20261110.md;
label campaign): the expectation and the bootstrap of analysis/campaign.py, the top-1 across seeds of unit (b), the
sign of the success points, the rerun reference on the HW pair's seeds and the CLI on a synthetic --main tree. No real
AlphaFold3 output is read."""
import pathlib
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'analysis'))
import campaign  # noqa: E402

PAIR_COLS = ['contrast', 'kind', 'role', 'target', 'seed_a', 'seed_b', 'top1_a', 'top1_b', 'success_a', 'success_b',
             'margin_a', 'margin_b', 'flip_top1', 'flip_success', 'flip_matters']
MAIN_T = ['T1', 'T2', 'T3', 'T4']
REPL_T = ['R1', 'R2', 'R3']
SEEDS = [1, 2, 3, 4, 5]


# ---- synthetic tables

def sample_rows(label, platform, rep, targets, seeds, rng, n=5):
  return [{'label': label, 'platform': platform, 'repetition': rep, 'target': t, 'seed': s, 'sample': k,
           'ranking_score': float(rng.uniform(0.5, 0.9)), 'm1_success': bool(rng.random() < 0.5),
           'num_recycles': 10, 'num_diffusion_samples': 5}
          for t in targets for s in seeds for k in range(n)]


def seed_top(d, platform, rep, target, seed):
  u = d[(d.platform == platform) & (d.repetition == rep) & (d.target == target) & (d.seed == seed)]
  r = u.sort_values('ranking_score', ascending=False, kind='mergesort').iloc[0]
  return int(r['sample']), bool(r.m1_success == True)  # noqa: E712


def pairs(d, contrast, targets, seeds):
  """pairs_seed rows of contrast as analysis/contrasts builds them (per-seed top-1 on each side)."""
  kind, inner = contrast.split('(')[0], contrast[contrast.index('(') + 1:-1].split(',')
  a, b = ((inner[0], 'r1'), (inner[0], 'r2')) if kind == 'RC' else ((inner[0], 'r1'), (inner[1], 'r1'))
  rows = []
  for t in targets:
    for s in seeds:
      ta, sa = seed_top(d, *a, t, s)
      tb, sb = seed_top(d, *b, t, s)
      rows.append({'contrast': contrast, 'kind': kind, 'role': 'x', 'target': t, 'seed_a': s, 'seed_b': s,
                   'top1_a': ta, 'top1_b': tb, 'success_a': sa, 'success_b': sb, 'margin_a': 0.01,
                   'margin_b': 0.01, 'flip_top1': ta != tb, 'flip_success': sa != sb,
                   'flip_matters': ta != tb and sa != sb})
  return rows


def write_tree(root, seed=0):
  """A synthetic --main checkout with the five folders of the row (samples.csv and pairs_seed.csv). In main, G r2 is
  G r1 except that on seeds 3 to 5 sample 0 (never the r1 top-1) scores 10 and succeeds: RC(G) changes on every one
  of those seeds and nowhere else."""
  rng = np.random.default_rng(seed)
  smp = {
      'main': sample_rows('main', 'G', 'r1', MAIN_T, SEEDS, rng) + sample_rows('main', 'V5', 'r1', MAIN_T, SEEDS, rng)
      + sample_rows('main', 'V5', 'r2', MAIN_T, SEEDS, rng) + sample_rows('main', 'V6', 'r1', MAIN_T, SEEDS, rng)
      + sample_rows('main', 'V6', 'r2', MAIN_T, SEEDS, rng) + sample_rows('main', 'C', 'r1', ['T1'], [1, 2], rng)
      + sample_rows('main', 'Gd', 'r1', MAIN_T, SEEDS, rng),
      'a100': sample_rows('a100', 'A', 'r1', MAIN_T, SEEDS, rng) + sample_rows('a100', 'A', 'r2', MAIN_T, SEEDS, rng),
      'h100': sample_rows('h100', 'H', 'r1', MAIN_T, SEEDS, rng) + sample_rows('h100', 'H', 'r2', MAIN_T, SEEDS, rng),
      'cpu60': sample_rows('cpu60', 'C', 'r1', MAIN_T[1:], [1, 2], rng),
      'repl': sum((sample_rows('repl', p, r, REPL_T, SEEDS, rng) for p in ('V5', 'V6', 'G') for r in ('r1', 'r2')), []),
  }
  for r in smp['main']:
    if r['platform'] == 'G' and r['seed'] >= 3 and r['sample'] == 0:
      r['ranking_score'] = 0.1   # never the r1 top-1
  g2 = [dict(r, repetition='r2') for r in smp['main'] if r['platform'] == 'G']
  for r in g2:
    if r['seed'] >= 3 and r['sample'] == 0:
      r['ranking_score'], r['m1_success'] = 10.0, True
  smp['main'] += g2
  frames = {k: pd.DataFrame(v) for k, v in smp.items()}
  allm = pd.concat(frames.values(), ignore_index=True)
  prs = {
      'main': sum((pairs(allm[allm.label == 'main'], c, MAIN_T, SEEDS)
                   for c in ('HW(V6,G)', 'HW(V5,V6)', 'RC(V6)', 'RC(G)', 'RC(V5)')), []),
      'a100': pairs(allm, 'HW(G,A)', MAIN_T, SEEDS) + pairs(allm, 'RC(A)', MAIN_T, SEEDS),
      'h100': pairs(allm, 'HW(H,A)', MAIN_T, SEEDS) + pairs(allm, 'RC(A)', MAIN_T, SEEDS)
      + pairs(allm, 'RC(H)', MAIN_T, SEEDS),
      'cpu60': pairs(allm, 'HW(G,C)', MAIN_T, [1, 2]),
      'repl': sum((pairs(allm[allm.label == 'repl'], c, REPL_T, SEEDS)
                   for c in ('HW(V6,G)', 'HW(V5,V6)', 'RC(V6)', 'RC(G)', 'RC(V5)')), []),
  }
  for k in campaign.FOLDERS:
    for rel, df in ((campaign.SAMPLES[k], frames[k]), (campaign.PAIRS[k], pd.DataFrame(prs[k], columns=PAIR_COLS))):
      p = pathlib.Path(root) / rel
      p.parent.mkdir(parents=True, exist_ok=True)
      df.to_csv(p, index=False)
  return root


def jobs(flips, succ_from, succ_to, per_target=5):
  n = len(flips)
  return pd.DataFrame({'target': [f'T{i // per_target:03d}' for i in range(n)], 'seed': [i % per_target for i in range(n)],
                       'changed': np.asarray(flips, float),
                       'success_changed': (np.asarray(succ_from) != np.asarray(succ_to)).astype(float),
                       'success_points': np.asarray(succ_to, float) - np.asarray(succ_from, float)})


@pytest.fixture
def toy():
  rng = np.random.default_rng(1)
  n = 60
  return jobs(rng.random(n) < 0.3, rng.random(n) < 0.5, rng.random(n) < 0.6)


# ---- expectation and bootstrap

@pytest.mark.parametrize('share', campaign.SHARES)
def test_expectation_is_share_times_pair_rate(toy, share):
  values, counts = campaign.stack(toy)
  res = {st: est for st, est, _, _ in campaign.summarize(values, counts, share, np.random.default_rng(0), 50)}
  assert res['changed'] == pytest.approx(share * toy.changed.mean())
  assert res['success_changed'] == pytest.approx(share * toy.success_changed.mean())
  assert res['success_points'] == pytest.approx(share * 100 * toy.success_points.mean())


def test_share_zero_gives_zero_and_one_gives_the_hw_rate(toy):
  values, counts = campaign.stack(toy)
  zero = campaign.summarize(values, counts, 0.0, np.random.default_rng(0), 200)
  assert all(est == 0 and lo == 0 and hi == 0 for _, est, lo, hi in zero)
  one = {st: (est, lo, hi) for st, est, lo, hi in campaign.summarize(values, counts, 1.0, np.random.default_rng(0), 200)}
  ref = {st: (est, lo, hi) for st, est, lo, hi in campaign.summarize(values, counts, None, np.random.default_rng(0), 200)}
  assert one['changed'][0] == pytest.approx(toy.changed.mean())
  assert one['success_points'][0] == pytest.approx(100 * toy.success_points.mean())
  assert one == ref   # s = 1 moves every job: the plain target bootstrap of the pair (same targets drawn)


def test_interval_contains_estimate_in_a_simple_case():
  d = jobs([True, False] * 50, [False] * 100, [True, False, False, False] * 25)   # 20 targets x 5 jobs
  values, counts = campaign.stack(d)
  for share in (0.25, 0.5, 1.0):
    for st, est, lo, hi in campaign.summarize(values, counts, share, np.random.default_rng(3), 2000):
      assert lo <= est <= hi, (share, st, est, lo, hi)


def test_draw_moves_about_share_of_jobs():
  d = jobs([True] * 200, [False] * 200, [False] * 200)
  values, counts = campaign.stack(d)
  vals = campaign.draw(values, counts, 0.25, np.random.default_rng(5), 400)
  assert vals[:, 0].mean() == pytest.approx(0.25, abs=0.01)
  assert (vals[:, 1] == 0).all() and (vals[:, 2] == 0).all()


# ---- units

def test_unit_b_top_across_seeds_is_the_global_argmax():
  s = pd.DataFrame({'target': ['X'] * 6 + ['Y'] * 4, 'seed': [1, 1, 1, 2, 2, 3, 1, 1, 2, 2],
                    'sample': [0, 1, 2, 0, 3, 0, 0, 1, 0, 1],
                    'ranking_score': [0.80, 0.70, 0.60, 0.50, 0.90, 0.99, 0.4, 0.6, 0.6, 0.3],
                    'success': [False, False, False, False, True, False, True, False, True, False]})
  keys = pd.DataFrame({'target': ['X', 'X', 'Y', 'Y'], 'seed': [1, 2, 1, 2]})   # seed 3 of X is not shared
  top = campaign.top_across_seeds(s, keys, 'test')
  assert (top.loc['X', 'seed'], top.loc['X', 'sample'], bool(top.loc['X', 'success'])) == (2, 3, True)
  assert (top.loc['Y', 'seed'], top.loc['Y', 'sample']) == (1, 1)   # tie at 0.6: lowest seed
  with pytest.raises(SystemExit, match='no samples'):
    campaign.top_across_seeds(s, pd.DataFrame({'target': ['X'], 'seed': [4]}), 'test')


def test_unit_b_changed_compares_seed_and_sample():
  p = pd.DataFrame({'platform': 'P', 'repetition': 'r1', 'target': ['X', 'X'], 'seed': [1, 2], 'sample': [0, 0],
                    'ranking_score': [0.9, 0.8], 'success': [False, True]})
  q = p.assign(platform='Q', ranking_score=[0.8, 0.9])   # same sample index, other seed
  get = lambda c: {'P': p, 'Q': q}[c[0]]  # noqa: E731
  b = campaign.unit_b(get, ('P', 'r1'), ('Q', 'r1'), pd.DataFrame({'target': ['X', 'X'], 'seed': [1, 2]}), 'test')
  assert b.changed.tolist() == [1.0] and b.success_changed.tolist() == [1.0] and b.success_points.tolist() == [1.0]


def test_success_points_sign_is_q_minus_p():
  rows = pd.DataFrame({'target': ['X'], 'seed_a': [1], 'seed_b': [1], 'success_a': [True], 'success_b': [False],
                       'flip_top1': [True], 'flip_success': [True]})
  # HW(G,A) with P = A: side b is the all-P campaign; G (Q) succeeds where A fails: + 1
  first, second, side = campaign.sides('HW(G,A)', 'A')
  assert (first, second, side) == (('A', 'r1'), ('G', 'r1'), 'b')
  assert campaign.unit_a(rows, side).success_points.tolist() == [1.0]
  # HW(G,C) with P = G: side a is the all-P campaign; C (Q) fails where G succeeds: - 1
  assert campaign.sides('HW(G,C)', 'G')[2] == 'a'
  assert campaign.unit_a(rows, 'a').success_points.tolist() == [-1.0]
  # RC: r2 minus r1
  assert campaign.sides('RC(A)', 'A') == (('A', 'r1'), ('A', 'r2'), 'a')
  values, counts = campaign.stack(campaign.unit_a(rows, 'b'))
  est = {st: e for st, e, _, _ in campaign.summarize(values, counts, 0.5, np.random.default_rng(0), 10)}
  assert est['success_points'] == pytest.approx(50.0)


def test_missing_success_counts_as_no_success():
  assert campaign.truth([True, 'True', 'False', np.nan, '', 1.0, 0]).tolist() == [True, True, False, False, False,
                                                                                    True, False]


# ---- the analysis on a synthetic tree

def test_deterministic_for_a_fixed_seed(tmp_path):
  write_tree(tmp_path)
  c1, r1, _ = campaign.analyse(tmp_path, 60, 2026)
  c2, r2, _ = campaign.analyse(tmp_path, 60, 2026)
  pd.testing.assert_frame_equal(c1, c2)
  pd.testing.assert_frame_equal(r1, r2)
  c3, _, _ = campaign.analyse(tmp_path, 60, 7)
  assert not c1.ci_low.equals(c3.ci_low) or not c1.ci_high.equals(c3.ci_high)


def test_campaign_rows_match_the_pairs(tmp_path):
  write_tree(tmp_path)
  camp, _, _ = campaign.analyse(tmp_path, 30, 2026)
  sp = pd.read_csv(tmp_path / campaign.PAIRS['a100'])
  hw = sp[sp.contrast == 'HW(G,A)']
  row = camp[(camp.policy.str.startswith('(1)')) & (camp.unit == 'a') & (camp.statistic == 'changed')]
  assert row.estimate.tolist() == pytest.approx([s * hw.flip_top1.mean() for s in campaign.SHARES])
  assert set(row.n_jobs) == {20} and set(row.n_targets) == {4}
  p3 = camp[camp.policy.str.startswith('(3)')]
  assert set(p3.n_jobs[p3.unit == 'a']) == {8} and set(p3.n_jobs[p3.unit == 'b']) == {4}
  assert set(camp.source[camp.policy.str.startswith('(4)')]) == {campaign.FOLDERS['main'], campaign.FOLDERS['repl']}


def test_reference_is_rc_on_the_same_seeds(tmp_path):
  write_tree(tmp_path)   # RC(G) of main changes only on seeds 3 to 5
  _, refs, _ = campaign.analyse(tmp_path, 30, 2026)
  g = refs[(refs.contrast == 'RC(G)') & (refs.statistic == 'changed')].set_index(['policy', 'source', 'unit'])
  main = campaign.FOLDERS['main']
  p3 = [k for k in g.index if k[0].startswith('(3)')]
  assert {k[1] for k in p3} == {main}
  assert g.loc[(p3[0][0], main, 'a'), 'estimate'] == 0 and g.loc[(p3[0][0], main, 'a'), 'n_jobs'] == 8
  assert g.loc[(p3[0][0], main, 'b'), 'estimate'] == 0     # top-1 across seeds 1 and 2 only
  p4 = [k for k in g.index if k[0].startswith('(4)') and k[1] == main][0][0]
  assert g.loc[(p4, main, 'a'), 'estimate'] == pytest.approx(3 / 5) and g.loc[(p4, main, 'a'), 'n_jobs'] == 20
  assert g.loc[(p4, main, 'b'), 'estimate'] == 1.0         # seeds 3 to 5 hold the r2 winner
  assert set(refs.role[(refs.contrast == 'RC(G)') & refs.policy.str.startswith('(4)')]) == {'RC(Q), printed beside'}
  assert set(refs.contrast[refs.policy.str.startswith('(1)')]) == {'RC(A)'}
  assert set(refs.source[refs.policy.str.startswith('(2)')]) == {campaign.FOLDERS['h100']}
  assert set(refs.contrast[refs.policy.str.startswith('(5)')]) == {'RC(V5)'}


def run_cli(main, out, reps=30):
  return subprocess.run([sys.executable, str(REPO / 'analysis' / 'campaign.py'), '--main', str(main), '--out', str(out),
                         '--replicates', str(reps)], capture_output=True, text=True)


def test_cli_end_to_end(tmp_path):
  main = write_tree(tmp_path / 'checkout')
  out = tmp_path / 'out'
  r = run_cli(main, out)
  assert r.returncode == 0, r.stderr
  for f in ('campaign.csv', 'references.csv', 'report.md', 'inputs.tsv'):
    assert (out / f).is_file()
  camp = pd.read_csv(out / 'campaign.csv')
  assert list(camp.columns) == ['policy', 'unit', 'share', 'statistic', 'estimate', 'ci_low', 'ci_high', 'n_targets',
                                'n_jobs', 'source']
  assert len(camp) == 7 * 2 * len(campaign.SHARES) * 3     # 7 (policy, source) campaigns
  refs = pd.read_csv(out / 'references.csv')
  assert list(refs.columns)[:7] == ['policy', 'unit', 'contrast', 'statistic', 'estimate', 'ci_low', 'ci_high']
  inp = pd.read_csv(out / 'inputs.tsv', sep='\t')
  assert set(inp.path) == set(campaign.PAIRS.values()) | set(campaign.SAMPLES.values())
  assert all(len(h) == 64 for h in inp.sha256)
  rep = (out / 'report.md').read_text()
  for words in ('follow-up analysis', 'outside the pre-registered study', 'descriptive',
                'The shares are assumptions, not measurements', 'C ran r1 only'):
    assert words.lower() in rep.lower()
  dash = chr(0x2014)
  assert dash not in rep and dash not in (REPO / 'analysis' / 'campaign.py').read_text()


def test_cli_stops_on_a_missing_contrast(tmp_path):
  main = write_tree(tmp_path / 'checkout')
  p = main / campaign.PAIRS['h100']
  d = pd.read_csv(p)
  d[d.contrast != 'HW(H,A)'].to_csv(p, index=False)
  r = run_cli(main, tmp_path / 'out')
  assert r.returncode != 0 and 'HW(H,A)' in r.stderr and 'missing' in r.stderr
  assert not (tmp_path / 'out' / 'campaign.csv').exists()


def test_cli_stops_on_a_missing_platform(tmp_path):
  main = write_tree(tmp_path / 'checkout')
  p = main / campaign.SAMPLES['h100']
  d = pd.read_csv(p)
  d[d.repetition != 'r1'].to_csv(p, index=False)
  r = run_cli(main, tmp_path / 'out')
  assert r.returncode != 0 and 'platform H repetition r1 is missing' in r.stderr


def test_help():
  r = subprocess.run([sys.executable, str(REPO / 'analysis' / 'campaign.py'), '--help'], capture_output=True, text=True)
  assert r.returncode == 0 and '--replicates' in r.stdout and '--main' in r.stdout
