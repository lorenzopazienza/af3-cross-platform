"""Section 10 rows of 2026-10-13: the training-cutoff sensitivity analysis (analysis/cutoff.py, called by
analysis/contrasts.py) and the duplicate-session rule (analysis/collect.py). Synthetic data, no outputs read."""

import argparse
import json
import sys

import pandas as pd
import pytest

import collect
import contrasts
import cutoff

# ---- release dates ---------------------------------------------------------------------------------------------


def test_release_table_from_the_frozen_records():
  t = cutoff.release_table()
  assert len(t) == 60 and int(t.post_cutoff.sum()) == 39 and int((~t.post_cutoff).sum()) == 21
  pre = t[~t.post_cutoff]
  assert pre.iloc[0].pdb_id == '7D5C' and pre.iloc[0].rcsb_initial_release == '2021-02-24'
  assert pre.iloc[-1].pdb_id == '7NF3' and pre.iloc[-1].rcsb_initial_release == '2021-08-04'
  assert set(pre[pre.pilot == 'yes'].pdb_id) == {'7D5C', '7BTT', '7NPL', '7NP6'}
  assert pre.bucket.value_counts().to_dict() == {'512': 11, '256': 6, '768': 2, '1024': 2}


def test_a_release_on_the_cutoff_date_counts_as_before_it(tmp_path):
  p = tmp_path / 't.csv'
  pd.DataFrame([{'pdb_id': 'A', 'rcsb_initial_release': '2021-09-30', 'bucket': '256', 'pilot': 'no'},
                {'pdb_id': 'B', 'rcsb_initial_release': '2021-10-01', 'bucket': '256', 'pilot': 'no'}]).to_csv(p, index=False)
  t = cutoff.release_table(p).set_index('pdb_id')
  assert not t.loc['A'].post_cutoff and t.loc['B'].post_cutoff

# ---- sensitivity analysis --------------------------------------------------------------------------------------


TARGETS = ('T1', 'T2', 'T3', 'T4')
RELEASES = pd.DataFrame([{'pdb_id': t, 'rcsb_initial_release': d, 'bucket': '256', 'pilot': 'no'}
                         for t, d in zip(TARGETS, ('2022-01-01', '2022-06-01', '2021-03-01', '2020-01-01'))])
RELEASES['post_cutoff'] = RELEASES.rcsb_initial_release > cutoff.CUTOFF


def table(flips=(), failing=()):
  """V6 and G (r1, r2) and C (r1) on TARGETS, seeds 1 and 2; top-1 is sample 0 unless (cond, rep, target) is in
  flips (then sample 1). Sample 0 is the only success, except for targets in failing (no success)."""
  rows = []
  for cond, reps in (('V6', ('r1', 'r2')), ('G', ('r1', 'r2')), ('C', ('r1',))):
    for rep in reps:
      for t in TARGETS:
        for seed in (1, 2):
          for k in range(5):
            score = 0.95 if (cond, rep, t) in flips and k == 1 else 0.9 - 0.1 * k
            rows.append({'session': f'{cond}_{rep}', 'label': 'main', 'platform': cond, 'libtpu': None, 'target': t,
                         'seed': seed, 'repetition': rep, 'sample': k, 'num_recycles': 10, 'num_diffusion_samples': 5,
                         'exit_code': 0, 'failure': None, 'ranking_score': score, 'mean_plddt': 80.0, 'iptm': 0.8,
                         'm1_success': k == 0 and t not in failing, 'model_path': None})
  return pd.DataFrame(rows)


def load(df, tmp_path, replicates=200):
  path = tmp_path / 'samples.csv'
  df.to_csv(path, index=False)
  args = argparse.Namespace(samples=str(path), labels=None, include=None, exclude=None, condition='platform',
                            replicates=replicates, seed=2026, out=None)
  d, ok = contrasts.load(args)
  return args, d, ok


def get(t, set_, quantity, name, statistic, reference=''):
  r = t[(t.set == set_) & (t.quantity == quantity) & (t.name == name) & (t.statistic == statistic) &
        (t.reference == reference)]
  assert len(r) == 1, (set_, quantity, name, statistic, reference, len(r))
  return r.iloc[0]


POST = f'released after {cutoff.CUTOFF}'


def test_sensitivity_restricts_to_post_cutoff_targets(tmp_path):
  # G moves its top-1 on T3 (pre-cutoff) only; T4 (pre-cutoff) never succeeds.
  args, _, ok = load(table(flips={('G', 'r1', 'T3')}, failing={'T4'}), tmp_path)
  t, post = cutoff.sensitivity(ok, args, RELEASES)
  t = t.fillna({'reference': ''})
  assert post == ['T1', 'T2']
  assert get(t, 'all targets', 'M1 success rate', 'V6', 'success_rate_overall_top1').estimate == pytest.approx(3 / 4)
  assert get(t, POST, 'M1 success rate', 'V6', 'success_rate_overall_top1').estimate == 1.0
  assert get(t, POST, 'M1 success rate', 'V6', 'success_rate_top1_per_seed').n == 4
  assert get(t, 'all targets', 'primary pair flips', 'HW(V6,G)', 'flip_top1').estimate == pytest.approx(2 / 8)
  assert get(t, POST, 'primary pair flips', 'HW(V6,G)', 'flip_top1').estimate == 0
  assert get(t, POST, 'primary pair flips', 'HW(V6,G)', 'flip_top1').n_targets == 2
  h = get(t, 'all targets', 'HW minus RC', 'HW(V6,G)', 'flip_top1', 'RC(V6) (RC(P))')
  assert h.estimate == pytest.approx(2 / 8)
  assert get(t, POST, 'HW minus RC', 'HW(V6,G)', 'flip_top1', 'RC(V6) (RC(P))').estimate == 0
  assert set(t[t.quantity == 'primary pair flips'].statistic) == {'flip_top1', 'flip_success'}
  assert set(t[(t.quantity == 'primary pair flips') & (t.set == POST)].name) == {'HW(V6,G)', 'HW(G,C)'}
  assert {'Q3'} <= set(t[t.set == POST].quantity)


def test_post_cutoff_values_are_the_primary_functions_on_the_subset(tmp_path):
  args, _, ok = load(table(flips={('G', 'r1', 'T1'), ('V6', 'r2', 'T2')}), tmp_path)
  t, _ = cutoff.sensitivity(ok, args, RELEASES)
  sub = ok[ok.target.isin(['T1', 'T2'])]
  sp, sm, cons, _ = contrasts.build_pairs(sub)
  s = contrasts.summarize(sp, sm, {'HW(V6,G)': cons['HW(V6,G)']}, args).set_index('statistic')
  r = get(t, POST, 'primary pair flips', 'HW(V6,G)', 'flip_top1')
  assert (r.estimate, r.ci_low, r.ci_high) == tuple(s.loc['flip_top1', ['estimate', 'ci_low', 'ci_high']])
  q, _ = contrasts.q3(sub, args)
  assert len(t[(t.set == POST) & (t.quantity == 'Q3')]) == len(q) + 1      # plus the answer row


def test_contrasts_main_writes_the_sensitivity_apart_from_unchanged_outputs(tmp_path, monkeypatch):
  df = table(flips={('G', 'r1', 'T3')})
  df['target'] = df.target.map({'T1': '7U3J', 'T2': '7XQZ', 'T3': '7D5C', 'T4': '7BTT'})   # 7D5C, 7BTT: pre-cutoff
  args, _, ok = load(df, tmp_path, replicates=100)
  out = tmp_path / 'out'
  monkeypatch.setattr(sys, 'argv', ['contrasts.py', '--samples', args.samples, '--out', str(out), '--replicates', '100'])
  assert contrasts.main() == 0
  sp, sm, cons, _ = contrasts.build_pairs(ok)
  direct = contrasts.summarize(sp, sm, cons, args)
  pd.testing.assert_frame_equal(pd.read_csv(out / 'contrasts.csv'), pd.read_csv(_csv(direct, tmp_path)))
  rel = pd.read_csv(out / 'release_dates.csv')
  assert len(rel) == 60 and int(rel.post_cutoff.sum()) == 39
  sens = pd.read_csv(out / 'sensitivity_post_cutoff.csv')
  assert set(sens.set) == {'all targets', POST}
  text = (out / 'report.md').read_text()
  assert '## Training-cutoff sensitivity' in text and text.index('## Pairs excluded') < text.index('## Training-cutoff')


def _csv(df, tmp_path):
  p = tmp_path / 'direct.csv'
  df.to_csv(p, index=False)
  return p

# ---- duplicate sessions ----------------------------------------------------------------------------------------


def session(root, name, finished=None, codes=(0, 0), runs=('l4_triton_rep1_seed1_fresh',)):
  s = root / name
  for run in runs:
    rd = s / run
    rd.mkdir(parents=True)
    (rd / 'session.json').write_text(json.dumps({
        'label': 'main', 'config_name': 'l4_triton', 'config': {'flash_attention_implementation': 'triton'},
        'probe': {'device_kind': 'NVIDIA L4', 'packages': {'jax': '0.10.2'}}, 'compile_cache': {'mode': 'fresh'}}))
    (rd / 'runs.jsonl').write_text(''.join(json.dumps({'pdb_id': f'T{i}', 'output_dir': str(rd / 'none'), 'seeds': [1],
                                                       'exit_code': c}) + '\n' for i, c in enumerate(codes)))
  if finished:
    (s / 'final_status.json').write_text(json.dumps({'exit_code': 0, 'finished_utc': finished}))
  return s


def test_two_complete_sessions_keep_the_first_completed_and_exclude_every_other(tmp_path):
  late_launch = session(tmp_path, '20261013T180000Z_l4_main_gt_04', finished='2026-10-13T21:00:00Z')
  early_launch = session(tmp_path, '20261013T175500Z_l4_main_gt_04', finished='2026-10-13T22:00:00Z')
  interrupted = session(tmp_path, '20261013T170000Z_l4_main_gt_04')                       # no final status
  other = session(tmp_path, '20261013T180500Z_l4_main_gt_05', finished='2026-10-13T23:00:00Z')
  kept, excluded = collect.resolve_duplicate_sessions(collect.run_folders([tmp_path]))
  assert {f.parent for f in kept} == {late_launch, other}       # completion time decides, not launch time
  ex = {e['session']: e for e in excluded}
  assert set(ex) == {early_launch.name, interrupted.name}
  assert ex[early_launch.name]['fully_exit_0'] is True and ex[interrupted.name]['fully_exit_0'] is False
  assert all(e['kept_session'] == late_launch.name for e in excluded)


def test_one_complete_session_leaves_the_part_unchanged(tmp_path):
  a = session(tmp_path, '20261013T180000Z_l4_main_gt_04', finished='2026-10-13T21:00:00Z')
  b = session(tmp_path, '20261013T170000Z_l4_main_gt_04', finished='2026-10-13T20:00:00Z', codes=(0, 1))
  kept, excluded = collect.resolve_duplicate_sessions(collect.run_folders([tmp_path]))
  assert excluded == [] and {f.parent for f in kept} == {a, b}   # rule not triggered: contrasts.load refuses duplicates


def test_same_plan_on_two_platforms_is_two_parts(tmp_path):
  session(tmp_path, '20261010T215616Z_v5e_pilot', finished='2026-10-11T01:00:00Z')
  session(tmp_path, '20261012T115650Z_v6e_pilot', finished='2026-10-12T14:00:00Z')
  kept, excluded = collect.resolve_duplicate_sessions(collect.run_folders([tmp_path]))
  assert excluded == [] and len(kept) == 2


def test_collect_main_drops_and_lists_excluded_sessions(tmp_path, monkeypatch):
  root = tmp_path / 'af3'
  session(root, '20261013T180000Z_l4_main_gt_04', finished='2026-10-13T21:00:00Z')
  dup = session(root, '20261013T175500Z_l4_main_gt_04', finished='2026-10-13T22:00:00Z')
  out = tmp_path / 'out'
  monkeypatch.setattr(sys, 'argv', ['collect.py', '--root', str(root), '--out', str(out), '--no_m1'])
  assert collect.main() == 0
  samples = pd.read_csv(out / 'samples.csv')
  assert len(samples) == 2      # the kept session's two processes (one row each, no models); four with the duplicate
  ex = pd.read_csv(out / 'excluded_sessions.csv')
  assert list(ex.session) == [dup.name] and list(ex.kept_session) == ['20261013T180000Z_l4_main_gt_04']
