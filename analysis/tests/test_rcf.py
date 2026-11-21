"""Follow-up 19 (label rcf; notes/analysis_plan.md section 10, row "Follow-up experiment 19"): synthetic tests of
analysis/rcf.py (D_F with RC(Hf) measured on 60 targets, the identity split between the 50 new and the 10 pilot
targets, the checks), of the plans, and an end-to-end run on relabelled earlier rows. No follow-up 19 output exists or
is read."""
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
import fu13  # noqa: E402
import fu14  # noqa: E402
import rcf  # noqa: E402
from run_af3 import read_configs  # noqa: E402

PARTS = REPO / 'harness' / 'plans' / 'rcf_parts.csv'
ALL = [f'T{i:02d}' for i in range(60)]
PILOT, NEW = ALL[:10], ALL[10:]


class Args:
  seed, replicates = 2026, 400


def seed_pairs(name, rates, targets, per=5, seed=0):
  rng = np.random.default_rng(seed)
  return pd.DataFrame([{'contrast': name, 'target': t, 'seed_a': s + 1, 'seed_b': s + 1,
                        **{st: bool(rng.random() < r) for st, r in rates.items()}}
                       for t in targets for s in range(per)])


RATES = lambda x: {'flip_top1': x, 'flip_success': x / 4, 'flip_matters': x / 6}  # noqa: E731


def test_conditions_are_the_specification():
  assert rcf.CONFIG == {'a100_xla_fu14_f32': 'Af', 'h100_xla_fu14_f32': 'Hf'} and rcf.SEEDS == [1, 2, 3, 4, 5]


# ---------------------------------------------------------------- D_F

def refs():
  f14 = seed_pairs('HW(Hf,Af)', RATES(0.02), ALL, seed=1)
  h = pd.concat([seed_pairs('HW(H,A)', RATES(0.2), ALL, seed=2), seed_pairs('RC(H)', RATES(0.09), ALL, seed=3)])
  return f14, h


def test_d_f_with_rc_zero_on_60_targets_is_the_registered_d_f_draw_for_draw():
  f14, h = refs()
  t = rcf.d_f_table(seed_pairs('RC(Hf)', RATES(0.0), ALL, seed=4), f14, h, Args)
  assert len(t) == 6 and t.same_draws.all()
  for st, g in t.groupby('statistic'):
    a, b = g.iloc[0], g.iloc[1]
    assert a.version.startswith('registered') and b.version == 'RC(Hf) measured on the 60 targets'
    for k in ('estimate', 'ci_low', 'ci_high'):
      assert a[k] == pytest.approx(b[k], abs=1e-12), (st, k)
  assert 'the same reading' in rcf.d_f_sentence(t) and 'Same draws as the registered D_F: yes' in rcf.d_f_sentence(t)


def test_d_f_measured_term_and_a_missing_target():
  f14, h = refs()
  t = rcf.d_f_table(seed_pairs('RC(Hf)', RATES(0.3), ALL, seed=5), f14, h, Args).set_index(['statistic', 'version'])
  reg = t.loc[('flip_top1', 'registered: 60 targets, RC(Hf) set to 0')]
  meas = t.loc[('flip_top1', 'RC(Hf) measured on the 60 targets')]
  assert meas.rc_f > 0 and meas.estimate < reg.estimate
  t = rcf.d_f_table(seed_pairs('RC(Hf)', RATES(0.0), ALL[1:], seed=4), f14, h, Args)
  assert not t[t.version.str.startswith('RC(Hf) measured')].same_draws.any()
  assert t[t.version.str.startswith('RC(Hf) measured')].n_targets.eq(59).all()
  assert 'no (the target lists differ' in rcf.d_f_sentence(t)
  assert 'not computed' in rcf.d_f_sentence(rcf.d_f_table(seed_pairs('x', RATES(0), ALL).iloc[0:0], f14, h, Args))


@pytest.mark.skipif(not all((REPO / 'results' / 'analysis' / f / 'pairs_seed.csv').is_file()
                            for f in ('fu14_20261113b', 'h100_20261104')), reason='earlier outputs not here')
def test_registered_d_f_of_hf_af_is_reproduced_exactly():
  """With 10,000 replicates and seed 2026 the registered D_F(Hf,Af) recomputed here equals follow-up 14's d_f.csv."""
  f14 = pd.read_csv(REPO / 'results' / 'analysis' / 'fu14_20261113b' / 'pairs_seed.csv')
  h = pd.read_csv(REPO / 'results' / 'analysis' / 'h100_20261104' / 'pairs_seed.csv')

  class A:
    seed, replicates = 2026, 10000
  t = rcf.d_f_table(seed_pairs('RC(Hf)', RATES(0.0), sorted(set(f14.target))), f14, h, A)
  import rcx
  assert rcx.registered_d_f_equal(t, REPO / 'results' / 'analysis' / 'fu14_20261113b') is True
  assert t[t.version.str.startswith('RC(Hf) measured')].same_draws.all()


# ---------------------------------------------------------------- checks and identity

def test_completeness_check_targets_and_pilot_reproduction():
  rows = [{'condition': c, 'repetition': 'r2', 'seed': s, 'target': t, 'sample': k}
          for c in rcf.CONDS for s in rcf.SEEDS for t in NEW for k in range(5)]
  pil = [{**r, 'target': t} for r in rows[:5] for t in PILOT]
  comp = rcf.completeness(pd.DataFrame(rows), pd.DataFrame(pil), PILOT, NEW)
  assert comp[comp.source == 'follow-up 19'].status.eq('complete').all()
  f15 = comp[comp.source == 'follow-up 15'].set_index(['condition', 'seed']).status       # Af seed 1 only
  assert f15[('Af', 1)] == 'complete' and f15.drop(('Af', 1)).eq('not run').all()
  assert rcf.check_targets(pd.DataFrame({'target': NEW}), PILOT)[0]['detail'] == 'pass'
  assert rcf.check_targets(pd.DataFrame({'target': ['T00', 'T20']}), PILOT)[0]['detail'] .endswith('pilot targets T00')
  sp = pd.concat([seed_pairs('RC(Af)', RATES(0.0), ALL, seed=1), seed_pairs('RC(Hf)', RATES(0.0), ALL, seed=2)])
  reg = sp[sp.target.isin(PILOT)]
  assert all(r['detail'] == 'pass' for r in rcf.check_pilot_reproduced(sp, reg, PILOT))
  reg = reg.copy()
  reg.loc[reg.index[0], 'flip_top1'] = True
  assert rcf.check_pilot_reproduced(sp, reg, PILOT)[0]['detail'].startswith('reported')


def write_outputs(root, t, seed, full, coords='1.000', stamp='x'):
  d = root / t
  for k in range(5):
    s = d / f'seed-{seed}_sample-{k}'
    s.mkdir(parents=True, exist_ok=True)
    (s / f'{t}_seed-{seed}_sample-{k}_model.cif').write_text(
        f'data_x\n_ma_model_list.model_group_name "AF3 {stamp}"\nATOM {coords}\n')
    (s / f'{t}_seed-{seed}_sample-{k}_summary_confidences.json').write_text('{"iptm": 0.9}')
    if full:
      (s / f'{t}_seed-{seed}_sample-{k}_confidences.json').write_text('{"pae": [1]}')
  (d / f'{t}_model.cif').write_text(f'data_x\n_ma_model_list.model_group_name "AF3 {stamp}"\nATOM {coords}\n')
  (d / f'{t}_ranking_scores.csv').write_text('seed,sample,ranking_score\n')
  (d / f'{t}_summary_confidences.json').write_text('{"iptm": 0.9}')
  if full:
    (d / f'{t}_confidences.json').write_text('{"pae": [1]}')
    (d / f'{t}_data.json').write_text('{}')
  return d


def test_identity_light_on_the_new_targets_strict_on_the_pilot(tmp_path):
  rows = []
  for c in ('Af', 'Hf'):
    for rep in ('r1', 'r2'):
      for t, full in (('T00', True), ('T10', False)):
        d = write_outputs(tmp_path / f'{c}_{rep}', t, 1, full=full or rep == 'r1', stamp=rep)
        rows += [{'condition': c, 'repetition': rep, 'target': t, 'seed': 1, 'sample': k,
                  'model_path': str(d / f'seed-1_sample-{k}' / f'{t}_seed-1_sample-{k}_model.cif')} for k in range(5)]
  ok = pd.DataFrame(rows)
  ident, det = rcf.identity(ok, ['T00'], ['T10'])
  r = {(x['condition'], x['targets']): x for x in ident}
  assert r[('Af', 'new 50')]['detail'] == 'recompilation bit-identical'
  assert r[('Af', 'pilot 10')]['detail'] == 'recompilation bit-identical'
  assert r[('Af', 'all 60')]['detail'] == 'recompilation bit-identical' and len(det) == 4
  (tmp_path / 'Hf_r2' / 'T00' / 'T00_confidences.json').write_text('{"pae": [2]}')   # a full file the light fetch lacks
  ident, _ = rcf.identity(ok, ['T00'], ['T10'])
  r = {(x['condition'], x['targets']): x for x in ident}
  assert r[('Hf', 'pilot 10')]['detail'].startswith('not bit-identical (1 of 1')
  assert r[('Hf', 'all 60')]['detail'].startswith('not bit-identical (1 of 2')


# ---------------------------------------------------------------- plans

def test_plans_are_up_to_date_and_match_the_specification():
  out = subprocess.run([sys.executable, str(REPO / 'harness' / 'plans' / 'make_prep_plans.py'), '--label', 'rcf', '--check'],
                       capture_output=True, text=True)
  assert out.returncode == 0, out.stdout + out.stderr
  parts = pd.read_csv(PARTS)
  assert parts.groupby('arm').processes.sum().to_dict() == {'Af_r2b': 250, 'Hf_r2b': 250}
  assert parts.groupby('arm').size().to_dict() == {'Af_r2b': 12, 'Hf_r2b': 9}
  assert set(parts.provisioning) == {'spot'} and set(parts.row_part) == {'A'}
  configs = read_configs(REPO / 'harness' / 'configs.yaml')
  manifest = pd.read_csv(REPO / 'inputs' / 'manifest.csv')
  pilot = set(manifest[manifest.pilot == 'yes'].pdb_id)
  seen = {'af': set(), 'hf': set()}
  for p in sorted((REPO / 'harness' / 'plans' / 'rcf').glob('*.yaml')):
    t = p.read_text()
    assert 'label: rcf\n' in t and 'rep_start: 2\n' in t and 'fresh_reps: 1\n' in t and 'eigen_source: archive\n' in t, p
    cfg = re.search(r'configs:\n  \w+: (\S+)', t).group(1)
    assert cfg in configs and cfg in rcf.CONFIG, p
    seen[p.name.split('_')[1]] |= set(re.search(r'default: \[([^\]]*)\]', t).group(1).split(', '))
  assert seen['af'] == seen['hf'] == set(manifest.pdb_id) - pilot and len(seen['af']) == 50


# ---------------------------------------------------------------- end to end, on relabelled earlier rows

FU14 = REPO / 'results' / 'analysis' / 'fu14_20261113b'
RCX = REPO / 'results' / 'analysis' / 'rcx_20261115'
H100 = REPO / 'results' / 'analysis' / 'h100_20261104'


@pytest.mark.skipif(not all((p / 'samples.csv').is_file() for p in (FU14, RCX, H100)), reason='earlier outputs not here')
def test_main_runs_end_to_end_on_relabelled_rows(tmp_path):
  """rcf.main on Af r1 and Hf r1 of follow-up 14 relabelled as follow-up 19's r2 on three non-pilot targets: every
  output is written, RC on the new targets is 0 and light-identical (r2 is r1's own files), the pilot targets come from
  follow-up 15 and reproduce its seed pairs, and D_F is computed; with 200 replicates the registered D_F cannot be
  reproduced, so the analysis says so and exits 1."""
  new = ['8G6P', '7LCU', '5SB2']
  m = pd.read_csv(FU14 / 'samples.csv', low_memory=False)
  g = m[(m.label == 'f32') & m.platform.isin(['Ah', 'Hh']) & m.config.isin(rcf.CONFIG) & (m.repetition == 'r1')
        & m.target.isin(new) & m['sample'].notna()].copy()
  g['label'], g['repetition'] = 'rcf', 'r2'
  g.to_csv(tmp_path / 'samples.csv', index=False)
  out = tmp_path / 'out'
  r = subprocess.run([sys.executable, str(REPO / 'analysis' / 'rcf.py'), '--samples', str(tmp_path / 'samples.csv'),
                      '--out', str(out), '--replicates', '200'], capture_output=True, text=True, cwd=REPO)
  assert r.returncode == 1 and 'registered D_F was not reproduced' in r.stderr, r.stdout[-2000:] + r.stderr[-3000:]
  for name in ('checks', 'completeness', 'contrasts', 'identity', 'd_f', 'stack', 'pairs_seed'):
    assert (out / f'{name}.csv').is_file(), name
  chk = pd.read_csv(out / 'checks.csv')
  assert (chk[chk.check.str.startswith(('1 ', '2', '3b', '4b', '5 '))].detail == 'pass').all(), chk.to_string()
  con = pd.read_csv(out / 'contrasts.csv')
  top = con[con.statistic == 'flip_top1'].set_index('contrast')
  assert top.loc['RC(Af) new 50', 'estimate'] == 0 and top.loc['RC(Af) new 50', 'n_targets'] == 3
  assert top.loc['RC(Af)', 'n_targets'] == 13
  ident = pd.read_csv(out / 'identity.csv').set_index(['condition', 'targets'])
  assert ident.loc[('Af', 'new 50'), 'detail'] == 'recompilation bit-identical'
  d = pd.read_csv(out / 'd_f.csv')
  assert len(d) == 6 and set(d.reading) <= {'reduced', 'not reduced'}
  assert 'DIFFERS from' in (out / 'report.md').read_text()
