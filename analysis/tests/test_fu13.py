"""Follow-up 13 (label why; notes/analysis_plan.md section 10, row of 2026-11-10): synthetic tests of every reading
rule of analysis/fu13.py, of the probe's pure functions (harness/fu13_probe.py), of the plans and configurations, and
of the budget script. No run output is read (none exists before the launch); engineering checks of the probe on CPU
with random weights are recorded in the implementation row, never here as results."""
import ast
import json
import os
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
sys.path.insert(0, str(REPO / 'harness' / 'plans'))
import fu13  # noqa: E402
import fu13_probe  # noqa: E402
from run_af3 import read_configs  # noqa: E402

U = 2.0 ** -8


class Args:
  seed, replicates = 2026, 400


# ---------------------------------------------------------------- conditions, checks

def test_condition_of_maps_every_follow_up_13_configuration():
  assert fu13.condition_of('tpu_xla_fu13_p0', 'V6') == 'P0'
  assert fu13.condition_of('tpu_xla_fu13_p14', 'V6') == 'P14'
  assert fu13.condition_of('tpu_xla_fu13_capture', 'V5') == 'V5L'
  assert fu13.condition_of('tpu_xla_fu13_capture', 'V6') == 'V6L'
  assert fu13.condition_of('l4_xla_fu13_capture', 'G') == 'GL'
  assert fu13.condition_of('tpu_xla_fu13_f32', 'V6h') == 'V6f'      # collect.py codes highest precision as V6h
  assert fu13.condition_of('l4_xla_fu13_f32', 'Gh') == 'Gf'
  assert fu13.condition_of('tpu_xla', 'V6') is None


def test_planned_units_are_the_specification_counts():
  n = sum(v for plan in fu13.PLANNED.values() for v in plan.values())
  assert n == 1790      # P 840, L 40, V6f 310, Gf 600 (item b)
  assert sum(fu13.PLANNED['Gf'].values()) == 600 and sum(fu13.PLANNED['V6f'].values()) == 310


def proc_rows(**kw):
  base = dict(session='s', target='T1', seed=1, repetition='r1', condition='Gf', exit_code=0, failure=np.nan)
  return {**base, **kw}


def test_check_1_passes_a_recorded_out_of_memory_stop_and_fails_an_unrecorded_one():
  d = pd.DataFrame([proc_rows(), proc_rows(target='T2', exit_code=1, failure='oom')])
  assert all(r['detail'] == 'pass' for r in fu13.check_exit(d))
  d = pd.DataFrame([proc_rows(), proc_rows(target='T2', exit_code=1, failure=np.nan)])
  assert fu13.check_exit(d)[0]['detail'].startswith('FAIL')


# ---------------------------------------------------------------- P

def test_eps_star_interpolates_in_log2_between_the_first_bracketing_levels():
  rates = [0.0, 0.02, 0.10, 0.15, 0.25, 0.40]          # at 2^-14, -11, -8, -7, -6, -5
  x, st = fu13.eps_star(rates, 0.20)
  assert st == 'interpolated' and x == pytest.approx(-7 + 0.5 * 1)   # halfway between 2^-7 and 2^-6
  x, st = fu13.eps_star(rates, 0.10)
  assert st == 'interpolated' and x == -8


def test_eps_star_is_censored_below_and_above_the_levels():
  assert fu13.eps_star([0.3] * 6, 0.2) == (-14.0, 'below')
  assert fu13.eps_star([0.0, 0.01, 0.02, 0.03, 0.04, 0.05], 0.2) == (-5.0, 'above')


def test_eps_star_takes_the_first_bracket_from_the_smallest_level_on_a_non_monotone_curve():
  x, st = fu13.eps_star([0.0, 0.25, 0.15, 0.30, 0.30, 0.30], 0.20)
  assert st == 'interpolated' and -14 < x < -11


def seed_pairs(contrast, rates, n_targets=20, seeds=(1, 2), stat='flip_top1'):
  """Seed pairs with a given flip rate per contrast: the first round(rate * n) units flip."""
  rows = []
  units = [(f'T{i:02d}', s) for i in range(n_targets) for s in seeds]
  k = round(rates * len(units))
  for j, (t, s) in enumerate(units):
    rows.append({'contrast': contrast, 'target': t, 'seed_a': s, 'seed_b': s, stat: j < k,
                 'flip_top1': j < k, 'flip_success': False, 'flip_matters': False})
  return pd.DataFrame(rows)


def test_eps_star_table_and_sentence():
  rates = dict(zip(fu13.LEVELS, [0.0, 0.025, 0.1, 0.15, 0.25, 0.4]))
  sp = pd.concat([seed_pairs(f'P(2^-{k})', r) for k, r in rates.items()], ignore_index=True)
  hw = pd.concat([seed_pairs('HW(V5,V6)', 0.20), seed_pairs('HW(V6,G)', 0.50)], ignore_index=True)
  t = fu13.eps_star_table(sp, hw, 'flip_top1', '1 and 2', 2026, 200)
  a = t[t.pair == 'HW(V5,V6)'].iloc[0]
  assert a.status == 'interpolated' and a.log2_eps_star == pytest.approx(-6.5)
  assert a.eps_star_u == pytest.approx(2 ** -6.5 / U) and a.ci_low_u <= a.eps_star_u <= a.ci_high_u
  b = t[t.pair == 'HW(V6,G)'].iloc[0]
  assert b.status == 'above' and b.censored_replicates > 0
  s = fu13.eps_sentence(a.to_dict(), 'single 1 u, pair 8 u')
  assert s.startswith('The cross-platform rate HW(V5,V6) of 20.0%') and 'times the bfloat16 unit roundoff u' in s
  assert 'a random perturbation at the trunk output of relative size about' in s and 'Measured trunk-output difference' in s
  assert 'above 8 u (2^-5)' in fu13.eps_sentence(b.to_dict())


# ---------------------------------------------------------------- L

def test_bfloat16_patterns_decode_exactly():
  x = np.array([1.0, -2.5, 3.140625, 0.0], dtype=np.float32)
  bits = (x.view(np.uint32) >> 16).astype(np.uint16)
  assert np.array_equal(fu13.bf16_to_f32(bits), x)


def test_expected_keys_of_a_capture():
  k = fu13.expected_keys(10)
  assert len(k) == 2 * 48 + 2 * 24 + 11 * 2
  assert 'pair_r00_d02' in k and 'pair_r00_d01' not in k and 'single_r05_d48' in k and 'single_r05_d10' not in k


def capture(base, scale_by_depth, last=10, seed=0, subset=(0, 1, 2)):
  """A synthetic capture: base arrays plus a difference of relative size scale_by_depth(d) at recycle 0 and 10."""
  rng = np.random.default_rng(seed)
  arrays = {'pair_tokens': np.array(subset, dtype=np.int32)}
  for key in fu13.expected_keys(last):
    kind, r, d = re.fullmatch(r'(single|pair)_r(\d\d)_d(\d\d)', key).groups()
    b = base[kind]
    noise = rng.standard_normal(b.shape)
    noise *= scale_by_depth(int(d)) * np.linalg.norm(b) / np.linalg.norm(noise)
    arrays[key] = (b + noise).astype(np.float32)
  return arrays, {'last_recycle': last, 'errors': []}


def test_layerwise_first_depth_growth_and_units():
  rng = np.random.default_rng(1)
  base = {'single': rng.standard_normal((5, 8)), 'pair': rng.standard_normal((3, 3, 4))}
  zero = lambda d: 0.0  # noqa: E731
  caps = {('V5L', 'r1', 'T'): capture(base, zero, seed=1),
          ('V6L', 'r1', 'T'): capture(base, lambda d: 0.0 if d < 4 else U * 2.0 ** (d - 4), seed=2),
          ('GL', 'r1', 'T'): capture(base, zero, seed=3),
          ('GL', 'r2', 'T'): capture(base, lambda d: 0.0 if d < 10 else 1e-6, seed=4)}
  depth = fu13.l_depth(caps)
  v56 = depth[(depth.pair == 'V5-V6') & (depth.kind == 'single') & (depth.recycle == 0)].set_index('depth')
  assert v56.rel_l2_u[4] == pytest.approx(1.0, rel=1e-3) and v56.rel_l2_u[5] == pytest.approx(2.0, rel=1e-3)
  fd = fu13.first_depth(depth)
  r = fd[(fd.pair == 'V5-V6') & (fd.kind == 'single')].iloc[0]
  assert r.first_depth == 4 and r.recompilation_level_zero_at_first
  rp = fd[(fd.pair == 'V5-V6') & (fd.kind == 'pair')].iloc[0]
  assert rp.first_depth == 4                       # pair at even depths only
  gr = fu13.growth(depth)
  g = gr[(gr.pair == 'V5-V6') & (gr.kind == 'single')].iloc[0]
  assert g.slope_bits_per_block == pytest.approx(1.0, abs=0.01)
  s = fu13.l_summary(depth)
  assert set(s.columns) >= {'pair', 'kind', 'recycle', 'depth', 'median_u', 'min_u', 'max_u'}
  assert (s[(s.recycle == 5)].depth == 48).all()   # other recycles: trunk output only


def test_first_depth_never_when_the_platform_pair_stays_below_recompilation():
  rng = np.random.default_rng(2)
  base = {'single': rng.standard_normal((4, 6)), 'pair': rng.standard_normal((2, 2, 3))}
  caps = {('V5L', 'r1', 'T'): capture(base, lambda d: 0.0, seed=1),
          ('V6L', 'r1', 'T'): capture(base, lambda d: 1e-4, seed=2),
          ('GL', 'r1', 'T'): capture(base, lambda d: 0.0, seed=3),
          ('GL', 'r2', 'T'): capture(base, lambda d: 1e-1, seed=4)}
  fd = fu13.first_depth(fu13.l_depth(caps))
  assert fd[fd.pair == 'V5-V6'].first_depth.isna().all()


def test_capture_check_flags_missing_arrays_ordering_errors_and_subset_mismatch():
  rng = np.random.default_rng(3)
  base = {'single': rng.standard_normal((3, 4)), 'pair': rng.standard_normal((2, 2, 2))}
  good = capture(base, lambda d: 0.0)
  bad = capture(base, lambda d: 0.0, subset=(0, 2))
  missing = ({k: v for k, v in good[0].items() if k != 'single_r10_d47'}, good[1])
  err = (good[0], {'last_recycle': 10, 'errors': ['recycle 3 output after 140 block events']})
  rows = fu13.check_capture({('V5L', 'r1', 'A'): good, ('V6L', 'r1', 'A'): good, ('GL', 'r1', 'B'): missing,
                             ('GL', 'r2', 'B'): err, ('V5L', 'r1', 'C'): None, ('V6L', 'r1', 'D'): good,
                             ('GL', 'r1', 'D'): bad})
  det = {(r['condition'], r['target']): r['detail'] for r in rows}
  assert det[('V5L r1', 'A')] == 'pass' and det[('all', 'A')].startswith('pass')
  assert det[('GL r1', 'B')].startswith('FAIL') and det[('GL r2', 'B')].startswith('FAIL')
  assert det[('V5L r1', 'C')] == 'FAIL: capture file missing' and det[('all', 'D')].startswith('FAIL')


# ---------------------------------------------------------------- F

def fpairs(name, rate, n=30, seeds=5):
  return seed_pairs(name, rate, n_targets=n, seeds=tuple(range(1, seeds + 1)))


def test_f_reduced_when_the_float32_effect_is_at_the_recompilation_level():
  r = fu13.d_f(fpairs('a', 0.10), fpairs('b', 0.10), fpairs('c', 0.30), fpairs('d', 0.06), 'flip_top1', 2026, 400)
  assert r['estimate'] == pytest.approx(-0.24, abs=1e-9) and r['ci_high'] < 0
  assert fu13.f_reading(r['ci_high']) == 'reduced'


def test_f_not_reduced_when_the_effect_is_unchanged_or_larger():
  r = fu13.d_f(fpairs('a', 0.30), fpairs('b', 0.07), fpairs('c', 0.30), fpairs('d', 0.07), 'flip_top1', 2026, 400)
  assert r['estimate'] == pytest.approx(0.0) and fu13.f_reading(r['ci_high']) == 'not reduced'
  r = fu13.d_f(fpairs('a', 0.50), fpairs('b', 0.0), fpairs('c', 0.30), fpairs('d', 0.07), 'flip_top1', 2026, 400)
  assert r['ci_low'] > 0 and fu13.f_reading(r['ci_high']) == 'not reduced'
  assert fu13.f_reading(float('nan')) == 'not computed'


def test_f_uses_only_targets_common_to_the_four_sets():
  r = fu13.d_f(fpairs('a', 0.1, n=30), fpairs('b', 0.1, n=20), fpairs('c', 0.3), fpairs('d', 0.1), 'flip_top1', 2026, 50)
  assert r['n_targets'] == 20


def test_cost_ratios_per_bucket():
  proc = pd.DataFrame([{'condition': c, 'bucket': b, 'exit_code': 0, 'wall_s': w, 'compiled_needed_bytes': m,
                        'alloc_peak_reserved_bytes': m}
                       for c, f in (('V6', 1), ('V6f', 3)) for b in (256, 1024)
                       for w, m in ((100 * f * b / 256, 10 * f), (110 * f * b / 256, 10 * f))])
  c = fu13.cost(proc, [('V6f', 'V6')])
  assert set(c.bucket) == {'all', 256, 1024}
  assert (c.wall_ratio.round(9) == 3).all() and (c.compiled_memory_ratio == 3).all()


# ---------------------------------------------------------------- the probe (harness/fu13_probe.py)

def test_probe_settings_off_by_default_and_validated():
  assert fu13_probe.settings({'PATH': '/bin'}) == {}
  s = fu13_probe.settings({'AF3_FU13_PERTURB_EPS': '2^-8', 'AF3_FU13_PERTURB_SEED': '2026'})
  assert s['perturb_eps'] == 2.0 ** -8 and s['perturb_seed'] == 2026
  assert fu13_probe.settings({'AF3_FU13_PERTURB_EPS': '0', 'AF3_FU13_PERTURB_SEED': '2026'})['perturb_eps'] == 0.0
  assert fu13_probe.settings({'AF3_FU13_BFLOAT16': 'none'}) == {'bfloat16': 'none'}
  assert fu13_probe.settings({'AF3_FU13_CAPTURE': '1'}) == {'capture': True}
  for bad in ({'AF3_FU13_TYPO': '1'}, {'AF3_FU13_PERTURB_EPS': '0.01', 'AF3_FU13_PERTURB_SEED': '1'},
              {'AF3_FU13_PERTURB_EPS': '0'}, {'AF3_FU13_BFLOAT16': 'intermediate'}, {'AF3_FU13_CAPTURE': 'yes'}):
    with pytest.raises(SystemExit):
      fu13_probe.settings(bad)


def test_probe_name_hash_is_the_sha256_prefix():
  import hashlib
  assert fu13_probe.name_hash('7U3J') == int(hashlib.sha256(b'7U3J').hexdigest()[:8], 16)


def test_pair_subset_rule():
  n = 300
  seq_mask = np.r_[np.ones(n), np.zeros(212)]
  is_protein = np.r_[np.ones(n - 38), np.zeros(38 + 212)]
  s = fu13_probe.pair_subset(seq_mask, is_protein)
  assert len(s) == 64 + 38 and s[0] == 0 and (n - 39) in s and np.all(np.diff(s) > 0)
  assert set(range(n - 38, n)) <= set(s) and max(s) < n
  small = fu13_probe.pair_subset(np.ones(50), np.r_[np.ones(40), np.zeros(10)])
  assert list(small) == list(range(50))


def test_af3_entry_imports_the_probe_only_under_an_af3_fu13_variable():
  src = (REPO / 'harness' / 'af3_entry.py').read_text()
  tree = ast.parse(src)
  imports = [n for n in ast.walk(tree) if isinstance(n, ast.Import) and any(a.name == 'fu13_probe' for a in n.names)]
  assert len(imports) == 1
  assert "if any(k.startswith('AF3_FU13_') for k in os.environ):\n    import fu13_probe" in src
  assert "if _FU13:   # follow-up 13 only" in src     # no "fu13" key in earlier records


# ---------------------------------------------------------------- configurations and plans

def git_show(rev, path):
  try:
    out = subprocess.run(['git', 'show', f'{rev}:{path}'], capture_output=True, text=True, cwd=REPO)
  except OSError:
    return None
  return out.stdout if out.returncode == 0 else None


def test_earlier_configurations_are_unchanged(tmp_path):
  before = git_show('c808d63', 'harness/configs.yaml')   # the commit before follow-up 13
  if before is None:
    pytest.skip('no git history here')
  (tmp_path / 'c.yaml').write_text(before)
  old, new = read_configs(tmp_path / 'c.yaml'), read_configs(REPO / 'harness' / 'configs.yaml')
  assert all(new[k] == old[k] for k in old)
  added = set(new) - set(old)
  assert added == {f'tpu_xla_fu13_p{k}' for k in (0, 14, 11, 8, 7, 6, 5)} | {
      'tpu_xla_fu13_capture', 'l4_xla_fu13_capture', 'tpu_xla_fu13_f32', 'l4_xla_fu13_f32'}
  for k in added:   # each is tpu_xla or l4_xla plus follow-up 13 settings only
    base = old['tpu_xla' if k.startswith('tpu') else 'l4_xla']
    cfg = {kk: v for kk, v in new[k].items() if kk != 'description'}
    env = {kk: v for kk, v in cfg.pop('env', {}).items()
           if not kk.startswith('AF3_FU13_') and not (k.endswith('f32') and kk == 'JAX_DEFAULT_MATMUL_PRECISION')}
    if env:
      cfg['env'] = env
    assert cfg == {kk: v for kk, v in base.items() if kk != 'description'}, k
    fu13_probe.settings(new[k].get('env', {}))


def test_plans_are_up_to_date_and_match_the_specification():
  out = subprocess.run([sys.executable, str(REPO / 'harness' / 'plans' / 'make_fu13_plans.py'), '--check'],
                       capture_output=True, text=True)
  assert out.returncode == 0, out.stdout + out.stderr
  parts = pd.read_csv(REPO / 'harness' / 'plans' / 'fu13_parts.csv')
  assert parts.groupby('arm').processes.sum().to_dict() == {'L5': 10, 'L6': 10, 'LG': 20, 'V6f': 310, 'P': 840, 'Gf': 600}
  assert set(parts.provisioning) == {'standard', 'flex_start'} and (parts[parts.platform == 'v6e'].provisioning == 'flex_start').all()
  assert parts.worst_usd_planned.sum() == pytest.approx(417, abs=0.5)      # the cap of item g
  configs = read_configs(REPO / 'harness' / 'configs.yaml')
  for p in sorted((REPO / 'harness' / 'plans' / 'why').glob('*.yaml')):
    t = p.read_text()
    assert 'label: why\n' in t and 'finish_by: 2026-11-30\n' in t and 'eigen_source: archive\n' in t, p
    cfg = re.search(r'configs:\n  \w+: (\S+)', t).group(1)
    assert cfg in configs and 'fu13' in cfg, p
  p_parts = parts[parts.arm == 'P']
  assert list(p_parts.config) == [f'tpu_xla_fu13_p{k}' for k in (0, 5, 6, 7, 8, 11, 14)]   # P0 first, then 2^-5 down


@pytest.mark.parametrize('script', ['make_fu7_10_plans.py', 'make_fu10_plans.py', 'make_fu11_12_plans.py'])
def test_earlier_plans_are_unchanged(script):
  out = subprocess.run([sys.executable, str(REPO / 'harness' / 'plans' / script), '--check'], capture_output=True, text=True)
  assert out.returncode == 0 and 'up to date' in out.stdout, out.stdout + out.stderr


# ---------------------------------------------------------------- budget rule

def credit(arm, bal, now):
  return subprocess.run(['bash', str(REPO / 'scripts' / 'monitor' / 'credit_check_fu13.sh'), arm, str(bal)],
                        capture_output=True, text=True, cwd=REPO, env=dict(os.environ, NOW=now))


def test_credit_check_go_floor_cap_and_dates():
  r = credit('p', 574.88, '2026-11-11T08:00:00Z')
  assert r.returncode == 0 and r.stdout.strip().endswith('GO')
  r = credit('gf', 160, '2026-11-11T08:00:00Z')
  assert r.returncode == 1 and '< 25 USD' in r.stdout
  r = credit('l5', 574.88, '2026-11-26T00:00:00Z')
  assert r.returncode == 1 and 'launch after 2026-11-25' in r.stdout
  r = credit('gf', 574.88, '2026-11-30T12:00:00Z')
  assert r.returncode == 1 and 'cannot finish by 2026-11-30' in r.stdout
  assert credit('x', 1, '2026-11-11T08:00:00Z').returncode == 2


# ---------------------------------------------------------------- end to end, on relabelled main-study rows

PRIMARY = REPO / 'results' / 'analysis' / 'main_20261014' / 'primary_v2'


@pytest.mark.skipif(not (PRIMARY / 'samples.csv').is_file(), reason='main-study outputs not here')
def test_main_runs_end_to_end_on_relabelled_main_study_rows(tmp_path):
  """fu13.main on main-study rows relabelled as follow-up 13 conditions (P levels with ranking scores permuted so
  that flips grow with eps; synthetic capture files); checks that every output is written and the checks pass.
  No follow-up 13 output exists or is read."""
  m = pd.read_csv(PRIMARY / 'samples.csv', low_memory=False)
  m = m[(m.label == 'main') & (m.num_recycles == 10) & (m.num_diffusion_samples == 5) & m['sample'].notna()]
  targets = ['7U3J', '7NP6', '7BTT']
  m = m[m.target.isin(targets)]
  rows = []

  def take(platform, rep, seeds, config, cond_platform, session, rep_as=None, shuffle=0):
    g = m[(m.platform == platform) & (m.repetition == rep) & m.seed.isin(seeds)].copy()
    g['label'], g['config'], g['platform'], g['session'] = 'why', config, cond_platform, session
    if rep_as:
      g['repetition'] = rep_as
    if shuffle:
      for (t, s), idx in g.groupby(['target', 'seed']).groups.items():
        if (hash((t, s)) + shuffle) % 3 == 0 or shuffle > 3:
          g.loc[idx, 'ranking_score'] = g.loc[idx, 'ranking_score'].values[::-1]
    rows.append(g)
  for i, k in enumerate([0] + fu13.LEVELS):
    take('V6', 'r1', [1, 2], f'tpu_xla_fu13_p{k}', 'V6', f'synthetic/p{k}', shuffle=i)
  cap = tmp_path / 'cap'
  for cond, pl, rep, cfg in (('V5L', 'V5', 'r1', 'tpu_xla_fu13_capture'), ('V6L', 'V6', 'r1', 'tpu_xla_fu13_capture'),
                             ('GL', 'G', 'r1', 'l4_xla_fu13_capture'), ('GL', 'G', 'r2', 'l4_xla_fu13_capture')):
    take(pl, rep, [1], cfg, pl, str(cap / f'{cond}_{rep}'))
  take('V6', 'r1', [1, 2, 3, 4, 5], 'tpu_xla_fu13_f32', 'V6h', 'synthetic/v6f')
  take('V6', 'r1', [1], 'tpu_xla_fu13_f32', 'V6h', 'synthetic/v6f_r2', rep_as='r2')
  take('G', 'r1', [1, 2, 3, 4, 5], 'l4_xla_fu13_f32', 'Gh', 'synthetic/gf_r1')
  take('G', 'r2', [1, 2, 3, 4, 5], 'l4_xla_fu13_f32', 'Gh', 'synthetic/gf_r2')
  d = pd.concat(rows, ignore_index=True)
  d.to_csv(tmp_path / 'samples.csv', index=False)
  rng = np.random.default_rng(0)
  base = {'single': rng.standard_normal((6, 5)), 'pair': rng.standard_normal((4, 4, 3))}
  for j, (cond, rep) in enumerate((('V5L', 'r1'), ('V6L', 'r1'), ('GL', 'r1'), ('GL', 'r2'))):
    for t in targets:
      arrays, meta = capture(base, lambda dd, j=j: 0.0 if j == 0 else U * 2.0 ** (dd / 12 - j), seed=j)
      folder = cap / f'{cond}_{rep}' / t
      folder.mkdir(parents=True)
      np.savez(folder / 'fu13_capture_seed-1.npz', meta=np.array(json.dumps(meta)), **arrays)
  out = tmp_path / 'out'
  r = subprocess.run([sys.executable, str(REPO / 'analysis' / 'fu13.py'), '--samples', str(tmp_path / 'samples.csv'),
                      '--out', str(out), '--replicates', '200'], capture_output=True, text=True, cwd=REPO)
  assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-3000:]
  for name in ('checks', 'completeness', 'p_curve', 'p_eps_star', 'l_depth', 'l_summary', 'l_first_depth', 'l_growth',
               'f_reading', 'f_secondary', 'f_success', 'f_cost', 'pairs_seed'):
    assert (out / f'{name}.csv').is_file(), name
  chk = pd.read_csv(out / 'checks.csv')
  assert (chk[chk.check.str.startswith(('1 ', '2 '))].detail == 'pass').all()
  for c in ('4 ', '5 ', '7 '):   # relabelled rows point at the main-study files themselves: identical
    assert (chk[chk.check.str.startswith(c)].detail == 'pass').all(), c
  assert chk[chk.check.str.startswith('6 ')].detail.iloc[0] == 'pass'
  curve = pd.read_csv(out / 'p_curve.csv')
  assert set(curve.level) == set(fu13.LEVELS)
  f = pd.read_csv(out / 'f_reading.csv')
  assert list(f.statistic) == ['flip_top1', 'flip_success', 'flip_matters'] and set(f.reading) <= {'reduced', 'not reduced'}
  rep = (out / 'report.md').read_text()
  assert '## P: perturbation' in rep and '## L: layer-wise' in rep and '## F: float32 trunk' in rep
