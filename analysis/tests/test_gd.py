"""The 2026-10-11 amendment (notes/analysis_plan.md, section 10) on synthetic data: the Gd condition
(analysis/collect.py), its secondary contrasts RC(Gd), DET, HW(V6,Gd) and HW(Gd,C) (analysis/contrasts.py, computed
as the section 5 contrasts), and pilot check 2 read as 2a, 2b, then bit-identity ignoring the mmCIF timestamp line."""

import argparse
import json

import pandas as pd
import pytest

import collect
import contrasts
import cost
from run_af3 import read_configs
from run_plan import expand

TARGETS = ('T1', 'T2', 'T3')
PILOT = [('V5', ('r1', 'r2', 'w')), ('V6', ('r1', 'r2', 'w')), ('G', ('r1', 'r2', 'w')), ('C', ('r1',))]
MATCHED = [('V6h', ('r1',)), ('Gh', ('r1',))]
GD = [('Gd', ('r1', 'r2', 'w'))]
NEW = {'RC(Gd)': 'RC', 'W(Gd)': 'W', 'DET': 'DET', 'HW(V6,Gd)': 'HW', 'HW(Gd,C)': 'HW'}

CIF = '''data_{t}
#
_entry.id {t}
#
_ma_model_list.model_group_id   1
_ma_model_list.model_group_name "AlphaFold-beta-20231127 (3.0.2 @ {stamp})"
_ma_model_list.model_name       "Top ranked model"
#
loop_
_atom_site.group_PDB
_atom_site.label_asym_id
_atom_site.label_seq_id
_atom_site.label_atom_id
_atom_site.Cartn_x
_atom_site.Cartn_y
_atom_site.Cartn_z
_atom_site.B_iso_or_equiv
ATOM A 1 CA {x} 0.000 0.000 80.00
#
'''


def table(conditions, flips=(), models=None, same_as=None):
  """TARGETS x seeds 1, 2 x 5 samples for each (condition, repetitions); top-1 is sample 0 except for
  (condition, repetition, target) in flips (sample 1). Sample 0 is the only success. With models (a folder),
  the G-family conditions (G, Gd) get mmCIF files whose coordinates depend on (target, seed, sample) only.
  same_as={'Gd': 'G'} gives Gd exactly G's rows apart from the condition."""
  rows = []
  for cond, reps in conditions:
    src = (same_as or {}).get(cond, cond)
    for rep in reps:
      for t in TARGETS:
        for seed in (1, 2):
          for k in range(5):
            score = 0.95 if (src, rep, t) in flips and k == 1 else 0.9 - 0.1 * k
            path = None
            if models is not None and cond in ('G', 'Gd'):
              f = models / cond / rep / t / f'seed-{seed}_sample-{k}' / 'model.cif'
              f.parent.mkdir(parents=True, exist_ok=True)
              f.write_text(CIF.format(t=t, stamp=f'2026-10-11 0{k}:00:00 {rep}', x=f'{seed}.{k}00'))
              path = str(f)
            rows.append({'session': f'{cond}_{rep}', 'label': 'pilot2', 'platform': cond, 'libtpu': None,
                         'target': t, 'seed': seed, 'repetition': rep, 'sample': k, 'num_recycles': 10,
                         'num_diffusion_samples': 5, 'exit_code': 0, 'failure': None, 'ranking_score': score,
                         'mean_plddt': 80.0, 'iptm': 0.8, 'm1_success': k == 0, 'model_path': path})
  return pd.DataFrame(rows)


def run(df, tmp_path, condition='platform', replicates=300):
  path = tmp_path / 'samples.csv'
  df.to_csv(path, index=False)
  args = argparse.Namespace(samples=str(path), labels=None, include=None, exclude=None, condition=condition,
                            replicates=replicates, seed=2026)
  d, ok = contrasts.load(args)
  sp, sm, cons, excluded = contrasts.build_pairs(ok)
  return args, d, ok, sp, sm, cons, excluded


# Gd's mapping ----------------------------------------------------------------------------------------------------

def test_gd_condition_code():
  assert collect.platform_of('NVIDIA L4', 'xla', None, 'l4_xla_det') == 'Gd'
  assert collect.platform_of('NVIDIA L4', 'xla', None, 'l4_xla') == 'G'
  assert collect.platform_of('NVIDIA L4', 'xla', 'highest', 'l4_xla_highest') == 'Gh'
  assert collect.platform_of('NVIDIA L4', 'triton', None, 'l4_triton') == 'GT'
  assert collect.platform_of('TPU v6 lite', 'xla', None, 'tpu_xla') == 'V6'
  assert collect.platform_of('NVIDIA L4', 'xla') == 'G'   # records without config_name keep the earlier codes


def session(folder, config, label, xla_env=None):
  folder.mkdir(parents=True)
  (folder / 'session.json').write_text(json.dumps({
      'label': label, 'config_name': config, 'config': {'flash_attention_implementation': 'xla'},
      'probe': {'device_kind': 'NVIDIA L4', 'packages': {'jax': '0.10.2'}}, 'xla_env': xla_env or {},
      'compile_cache': {'mode': 'fresh'}}))
  (folder / 'runs.jsonl').write_text(json.dumps({'pdb_id': 'T1', 'output_dir': str(folder / 'none'),
                                                 'seeds': [1], 'exit_code': 1, 'failure': 'error'}) + '\n')
  return folder


def test_collect_reads_gd_from_the_config_and_selects_by_label(tmp_path):
  g = session(tmp_path / 's1' / 'l4_xla_rep1_seed1_fresh', 'l4_xla', 'pilot2')
  gd = session(tmp_path / 's2' / 'l4_xla_det_rep1_seed1_fresh', 'l4_xla_det', 'pilot2',
               {'XLA_FLAGS': '--xla_gpu_enable_triton_gemm=false --xla_gpu_deterministic_ops=true'})
  diag = session(tmp_path / 's3' / 'l4_xla_det_rep1_seed1_fresh', 'l4_xla_det', 'diag')
  old = session(tmp_path / 's4' / 'l4_xla_rep1_seed1_fresh', 'l4_xla', 'pilot')
  for folder, code in ((g, 'G'), (gd, 'Gd'), (diag, 'Gd')):
    rows, _ = collect.rows_of_run(folder, {})
    assert (rows[0]['platform'], rows[0]['repetition']) == (code, 'r1')
  assert rows[0]['config'] == 'l4_xla_det' and 'config' in collect.COLUMNS
  assert set(collect.run_folders([tmp_path], ['pilot2'])) == {g, gd}
  assert set(collect.run_folders([tmp_path])) == {g, gd, diag, old}


def test_gd_config_is_g_plus_deterministic_ops():
  configs = read_configs(contrasts.REPO / 'harness' / 'configs.yaml')
  det, base = configs['l4_xla_det'], configs['l4_xla']
  assert {k: v for k, v in det.items() if k not in ('description', 'env')} == \
      {k: v for k, v in base.items() if k not in ('description', 'env')}
  assert det['env'] == dict(base['env'], XLA_FLAGS=base['env']['XLA_FLAGS'] + ' --xla_gpu_deterministic_ops=true')


def test_gd_plans_repeat_the_g_design():
  plans, manifest = contrasts.REPO / 'harness' / 'plans', contrasts.REPO / 'inputs' / 'manifest.csv'

  def cells(desc):
    return {(t, r['seed'], 'w' if r['cache'] == 'warm' else f'r{r["rep"]}') for r in desc['runs'] for t in desc['targets']}
  g = [expand(plans / f'pilot_l4_{i}.yaml', 'l4', manifest) for i in (1, 2)]
  gd = [expand(plans / f'pilot_l4d_{i}.yaml', 'l4', manifest) for i in (1, 2)]
  for a, b in zip(g, gd):
    assert (b['label'], b['weights'], b['configs']) == ('pilot2', 'gcs', ['l4_xla_det'])
    assert sorted(a['targets']) == sorted(b['targets']) and cells(a) == cells(b)
    assert b['num_recycles'] is None and b['num_diffusion_samples'] is None


def test_gd_is_priced_and_measured_as_the_l4():
  prices = cost.load_prices()
  for prov in cost.PROVISIONING:
    assert cost.price_for(prices, 'Gd', 'us-east4-c', prov) == cost.price_for(prices, 'G', 'us-east4-c', prov)
  p = pd.DataFrame([{'platform': 'Gd', 'bucket': 256, 'exit_code': 0, 'num_tokens': 200, 'compiled_peak_bytes': 1.0,
                     'compiled_args_bytes': 1.0, 'alloc_peak_reserved_bytes': 0, 'alloc_peak_in_use_bytes': 5.0,
                     'device_bytes_limit': 10.0, 'max_rss_bytes': 1.0, 'failure': None}])
  assert pd.isna(cost.memory_table(p).iloc[0].alloc_peak_reserved_gib)   # CUDA allocator: no reservation


# The four contrasts -----------------------------------------------------------------------------------------------

def test_gd_contrasts_and_only_those(tmp_path):
  args, _, ok, _, _, cons, _ = run(table(PILOT + MATCHED + GD), tmp_path)
  before = set(run(table(PILOT + MATCHED), tmp_path)[5])
  assert set(cons) == before | set(NEW)
  for name, kind in NEW.items():
    assert cons[name] == (kind, False), 'the 2026-10-11 contrasts are secondary'
  assert not any('Gd' in c for c in before)                 # no SD(Gd), no HW(V5,Gd), no DU
  rows, _ = contrasts.q3(ok, args)
  assert not rows.condition.str.contains('Gd').any()        # Gd stays out of Q3


def test_flip_rates_of_the_gd_contrasts(tmp_path):
  # Gd r1: top-1 moves on T1 and T2; Gd r2: on T1. G, V6 and C never move.
  flips = {('Gd', 'r1', 'T1'), ('Gd', 'r1', 'T2'), ('Gd', 'r2', 'T1')}
  args, _, _, sp, sm, cons, _ = run(table(PILOT + GD, flips), tmp_path)
  s = contrasts.summarize(sp, sm, cons, args).set_index(['contrast', 'statistic'])
  assert s.loc[('RC(Gd)', 'flip_top1')].estimate == pytest.approx(2 / 6)       # T2 only
  for name in ('DET', 'HW(V6,Gd)', 'HW(Gd,C)'):
    assert s.loc[(name, 'flip_top1')].estimate == pytest.approx(4 / 6)
    assert s.loc[(name, 'flip_matters')].estimate == pytest.approx(4 / 6)
    r = s.loc[(name, 'flip_top1')]
    assert r.n == 6 and r.n_targets == 3 and r.ci_low <= r.estimate <= r.ci_high
  h = contrasts.hw_minus_rc(sp, cons, args).set_index(['hardware', 'recompilation', 'statistic'])
  assert h.loc[('HW(V6,Gd)', 'RC(V6)', 'flip_top1')].estimate == pytest.approx(4 / 6)
  assert h.loc[('HW(V6,Gd)', 'RC(Gd)', 'flip_top1')].estimate == pytest.approx(4 / 6 - 2 / 6)
  assert h.loc[('HW(V6,Gd)', 'RC(Gd)', 'flip_top1')].reference == 'RC(Q), secondary'
  assert h.loc[('HW(Gd,C)', 'RC(Gd)', 'flip_top1')].estimate == pytest.approx(4 / 6 - 2 / 6)
  assert set(h.loc['HW(Gd,C)'].index.get_level_values(0)) == {'RC(Gd)'}     # no RC(C)
  assert 'DET' not in h.index.get_level_values(0)


def test_gd_equal_to_g_gives_zero_det_and_the_g_statistics(tmp_path):
  # Gd carries exactly G's data (rankings and coordinates): DET is zero everywhere and bit-identical, and RC(Gd),
  # HW(V6,Gd), HW(Gd,C) equal RC(G), HW(V6,G), HW(G,C) in every statistic and interval (same pairing, statistics
  # and bootstrap with seed 2026).
  flips = {('G', 'r1', 'T1'), ('G', 'r2', 'T3'), ('V6', 'r1', 'T2')}
  df = table(PILOT + GD, flips, models=tmp_path / 'models', same_as={'Gd': 'G'})
  args, _, _, sp, sm, cons, _ = run(df, tmp_path)
  s = contrasts.summarize(sp, sm, cons, args)
  det = s[s.contrast == 'DET'].set_index('statistic')
  for st in contrasts.FLIP_STATS:
    assert det.loc[st].estimate == 0 and det.loc[st].ci_low == 0 and det.loc[st].ci_high == 0
  for st in ('m5_ranking_score', 'm5_mean_plddt', 'm5_iptm', 'm4_ligand_rmsd', 'm4_ca_rmsd_global'):
    assert det.loc[f'{st}_median'].estimate == 0 and det.loc[f'{st}_p90'].estimate == 0
  assert det.loc['bit_identical_fraction'].estimate == 1 and det.loc['bit_identical_fraction'].n == 30
  cols = ['statistic', 'n', 'n_targets', 'estimate', 'ci_low', 'ci_high', 'note']
  for gd, g in (('RC(Gd)', 'RC(G)'), ('HW(V6,Gd)', 'HW(V6,G)'), ('HW(Gd,C)', 'HW(G,C)'), ('W(Gd)', 'W(G)')):
    pd.testing.assert_frame_equal(s[s.contrast == gd][cols].reset_index(drop=True),
                                  s[s.contrast == g][cols].reset_index(drop=True))
  assert stat(s, 'HW(V6,Gd)', 'flip_top1') == pytest.approx(4 / 6)       # not trivially zero
  h = contrasts.hw_minus_rc(sp, cons, args)
  hcols = ['reference', 'statistic', 'n_targets', 'estimate', 'ci_low', 'ci_high']
  for gd, g in (('HW(V6,Gd)', 'HW(V6,G)'), ('HW(Gd,C)', 'HW(G,C)')):
    pd.testing.assert_frame_equal(h[h.hardware == gd][hcols].reset_index(drop=True),
                                  h[h.hardware == g][hcols].reset_index(drop=True))


def stat(s, name, statistic):
  return s[(s.contrast == name) & (s.statistic == statistic)].iloc[0].estimate


def test_gd_leaves_section_5_and_11_unchanged(tmp_path):
  flips = {('G', 'r1', 'T1'), ('V6', 'r2', 'T2'), ('V5', 'r1', 'T3'), ('V6h', 'r1', 'T3'), ('Gh', 'r1', 'T1'),
           ('Gd', 'r1', 'T2'), ('Gd', 'r2', 'T3'), ('Gd', 'w', 'T1')}
  args, d0, ok0, sp0, sm0, cons0, ex0 = run(table(PILOT + MATCHED, flips), tmp_path)
  args, d1, ok1, sp1, sm1, cons1, ex1 = run(table(PILOT + MATCHED + GD, flips), tmp_path)
  assert list(cons1)[:len(cons0)] == list(cons0)
  pd.testing.assert_frame_equal(sp0, sp1[sp1.contrast.isin(cons0)].reset_index(drop=True))
  pd.testing.assert_frame_equal(sm0, sm1[sm1.contrast.isin(cons0)].reset_index(drop=True))
  pd.testing.assert_frame_equal(ex0, ex1[ex1.contrast.isin(cons0)].reset_index(drop=True))
  s0, s1 = contrasts.summarize(sp0, sm0, cons0, args), contrasts.summarize(sp1, sm1, cons1, args)
  pd.testing.assert_frame_equal(s0, s1[s1.contrast.isin(cons0)].reset_index(drop=True))
  h0, h1 = contrasts.hw_minus_rc(sp0, cons0, args), contrasts.hw_minus_rc(sp1, cons1, args)
  pd.testing.assert_frame_equal(h0, h1[h1.hardware.isin(cons0)].reset_index(drop=True))
  (q0, a0), (q1, a1) = contrasts.q3(ok0, args), contrasts.q3(ok1, args)
  pd.testing.assert_frame_equal(q0, q1)
  assert a0 == a1
  c0, c1 = contrasts.checks(d0, ok0, sm0, args), contrasts.checks(d1, ok1, sm1, args)
  keep = c1.condition.isin(set(c0.condition)) & ~c1.check.str.startswith('2 every arm')
  pd.testing.assert_frame_equal(c0[~c0.check.str.startswith('2 every arm')].reset_index(drop=True),
                                c1[keep].reset_index(drop=True))


def test_platform_libtpu_names(tmp_path):
  df = table(PILOT + GD)
  df.loc[df.platform.isin(['V5', 'V6']), 'libtpu'] = '0.0.43.2'
  _, _, _, _, _, cons, _ = run(df, tmp_path, condition='platform_libtpu')
  assert {'RC(Gd_none)', 'W(Gd_none)', 'DET(G_none,Gd_none)', 'HW(V6_0.0.43.2,Gd_none)', 'HW(Gd_none,C_none)'} <= set(cons)
  assert not any('Gd' in c and c.startswith(('SD', 'HW(V5')) for c in cons)


# Check 2: 2a, 2b, then bit-identity ignoring the timestamp line -----------------------------------------------------

def outputs(root, t='T1', seed=1, stamp='2026-10-11 01:02:03', x='1.000', conf='{"pae": [1.0]}', light=False):
  """An AlphaFold3 output folder of one target and seed, as AlphaFold3 writes it; light: as FETCH=light fetches it."""
  d = root / t / 'af3_output' / t
  d.mkdir(parents=True)
  files = {f'{t}_model.cif': CIF.format(t=t, stamp=stamp, x=x), f'{t}_summary_confidences.json': '{"ptm": 0.5}',
           f'{t}_confidences.json': conf, f'{t}_ranking_scores.csv': 'seed,sample,ranking_score\n1,0,0.9\n',
           f'{t}_data.json': '{"name": "T1"}', 'TERMS_OF_USE.md': 'terms\n'}
  for k in range(5):
    sd = f'seed-{seed}_sample-{k}'
    files[f'{sd}/{t}_{sd}_model.cif'] = CIF.format(t=t, stamp=stamp, x=x)
    files[f'{sd}/{t}_{sd}_summary_confidences.json'] = '{"ptm": 0.5}'
    files[f'{sd}/{t}_{sd}_confidences.json'] = conf
  for name, text in files.items():
    if light and (name.endswith('_data.json') or
                  (name.endswith('_confidences.json') and not name.endswith('_summary_confidences.json'))):
      continue
    (d / name).parent.mkdir(parents=True, exist_ok=True)
    (d / name).write_text(text)
  return d


def test_bit_identity_ignores_the_timestamp_line_and_nothing_else(tmp_path):
  ref = outputs(tmp_path / 'r1')
  assert contrasts.outputs_identical(ref, outputs(tmp_path / 'w', stamp='2026-10-12 23:59:59'), 1)['status'] == \
      'identical'
  # A change anywhere else fails: coordinates, another _ma_model_list line, confidences, the version in the stamp
  # line is masked with it (the whole line is AlphaFold3's version and time of writing).
  r = contrasts.outputs_identical(ref, outputs(tmp_path / 'x', x='1.001'), 1)
  assert r['status'] == 'differ' and 'T1_model.cif' in r['differ'] and len(r['differ']) == 6
  other = outputs(tmp_path / 'name')
  cif = other / 'seed-1_sample-3' / 'T1_seed-1_sample-3_model.cif'
  cif.write_text(cif.read_text().replace('"Top ranked model"', '"Top ranked model "'))
  assert contrasts.outputs_identical(ref, other, 1)['differ'] == ['seed-1_sample-3/T1_seed-1_sample-3_model.cif']
  r = contrasts.outputs_identical(ref, outputs(tmp_path / 'conf', conf='{"pae": [1.0000001]}'), 1)
  assert r['status'] == 'differ' and len(r['differ']) == 6 and all(n.endswith('_confidences.json') for n in r['differ'])
  terms = outputs(tmp_path / 'terms')
  (terms / 'TERMS_OF_USE.md').write_text('other\n')
  assert contrasts.outputs_identical(ref, terms, 1)['differ'] == ['TERMS_OF_USE.md']


def test_masking_is_limited_to_the_tag_at_the_start_of_a_line(tmp_path):
  a, b = tmp_path / 'a.cif', tmp_path / 'b.cif'
  a.write_text('_ma_model_list.model_group_name "x @ 1"\n_ma_model_list.model_group_id 1\n# model_group_name x\n')
  b.write_text('_ma_model_list.model_group_name "y @ 2"\n_ma_model_list.model_group_id 1\n# model_group_name x\n')
  assert contrasts.masked(a) == contrasts.masked(b)
  b.write_text('_ma_model_list.model_group_name "y @ 2"\n_ma_model_list.model_group_id 2\n# model_group_name x\n')
  assert contrasts.masked(a) != contrasts.masked(b)
  b.write_text('_ma_model_list.model_group_name "y @ 2"\n_ma_model_list.model_group_id 1\n# model_group_name y\n')
  assert contrasts.masked(a) != contrasts.masked(b)
  # Outside an mmCIF nothing is masked.
  j1, j2 = tmp_path / 'a.json', tmp_path / 'b.json'
  j1.write_text('_ma_model_list.model_group_name 1\n')
  j2.write_text('_ma_model_list.model_group_name 2\n')
  assert contrasts.masked(j1) != contrasts.masked(j2)


def test_bit_identity_needs_the_full_confidences(tmp_path):
  # FETCH=light leaves the full confidences and data.json in the bucket on both sides: not a pass.
  r = contrasts.outputs_identical(outputs(tmp_path / 'r1', light=True), outputs(tmp_path / 'w', light=True), 1)
  assert r['status'] == 'missing' and 'T1_data.json' in r['missing'] and len(r['missing']) == 7
  r = contrasts.outputs_identical(outputs(tmp_path / 'a'), tmp_path / 'nothing' / 'T1', 1)
  assert r['status'] == 'missing'


def check_table(tmp_path, cond='Gd', warm_key='K1', stamp='2026-10-12 00:00:00', x='1.000', fresh_before=0):
  """One condition, target T1 seed 1: r1 (wrote K1), r2 and w, with output folders on disk."""
  rows = []
  for rep, hit, key, written, before in (('r1', False, None, 'K1', fresh_before), ('r2', False, None, 'K2', 0),
                                         ('w', True, warm_key, None, 2)):
    d = outputs(tmp_path / rep, stamp=stamp if rep == 'w' else '2026-10-11 00:00:00', x=x if rep == 'w' else '1.000')
    for k in range(5):
      rows.append({'session': f'{cond}_{rep}', 'condition': cond, 'target': 'T1', 'seed': 1, 'repetition': rep,
                   'sample': k, 'exit_code': 0, 'failure': None, 'cache_files_before': before,
                   'model_cache_hit': hit, 'model_cache_key': key, 'model_cache_written': written,
                   'model_path': str(d / f'seed-1_sample-{k}' / f'T1_seed-1_sample-{k}_model.cif')})
  return pd.DataFrame(rows)


def check_rows(d):
  """{'2a' | '2b' | '2 bit-identity' | '2 overall': row} of check 2 on Gd, in the order check2 gives them."""
  key = lambda c: c.split(' ')[0] if c.startswith(('2a', '2b')) else ' '.join(c.split(' ')[:2]).rstrip(':')  # noqa: E731
  return {key(r['check']): r for r in contrasts.check2(d, pd.DataFrame(), contrasts.cache_checks(d), 'Gd')}


def test_check_2_in_order_on_gd(tmp_path):
  r = check_rows(check_table(tmp_path / 'ok'))
  assert list(r) == ['2a', '2b', '2 bit-identity', '2 overall']
  assert r['2a']['detail'] == r['2b']['detail'] == r['2 bit-identity']['detail'] == r['2 overall']['detail'] == 'pass'
  assert r['2 bit-identity']['value'] == '1/1 targets'


def test_check_2_fails_on_differing_outputs(tmp_path):
  r = check_rows(check_table(tmp_path, x='1.001'))
  assert r['2 bit-identity']['detail'].startswith('FAIL: T1 seed 1 differ') and r['2 overall']['detail'] == 'FAIL: bit-identity'


def test_check_2_bit_identity_is_read_only_after_2a_and_2b(tmp_path):
  # The warm run loaded another executable (G's, say): 2b fails, and bit-identity is not read even if it holds.
  r = check_rows(check_table(tmp_path / 'b', warm_key='K_of_G'))
  assert r['2b']['detail'] == 'FAIL: T1'
  assert r['2 bit-identity']['detail'] == 'not read (2a or 2b failed); pass'
  assert r['2 overall']['detail'] == 'FAIL: 2b'
  r = check_rows(check_table(tmp_path / 'a', fresh_before=4))
  assert r['2 overall']['detail'] == 'FAIL: 2a'


def test_check_2_covers_every_arm_with_a_warm_rerun(tmp_path):
  d = check_table(tmp_path)
  d = d[d.repetition != 'w']
  r = check_rows(d)
  assert r['2b']['detail'] == 'FAIL: no warm rerun' and r['2 overall']['detail'] == 'FAIL: 2b, bit-identity'
  args = argparse.Namespace()
  c = contrasts.checks(check_table(tmp_path / 'all'), pd.DataFrame(columns=['condition']), pd.DataFrame(), args)
  row = c[c.check == '2 every arm with a warm rerun is present'].iloc[0]
  assert row.detail == 'FAIL: absent V5, V6, G' and row.value == '1/4 arms'


def test_m1_check(tmp_path):
  ok = pd.DataFrame({'condition': ['C'] * 3, 'target': ['T1', 'T2', 'T3'], 'm1_success': [True, False, None],
                     'm1_error': [None, 'ligand not found', None]})
  c = contrasts.checks(ok.assign(session='s', exit_code=0, failure=None, repetition='r1', seed=1, sample=0), ok,
                       pd.DataFrame(), argparse.Namespace())
  row = c[c.check == 'M1 computed for every sample'].iloc[0]
  assert row.value == '1/3 samples' and row.detail == 'FAIL, targets: T2, T3'
