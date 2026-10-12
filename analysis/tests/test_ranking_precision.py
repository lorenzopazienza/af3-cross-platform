"""Full-precision ranking_score (notes/analysis_plan.md section 10, 2026-10-12): analysis/collect.py takes it from
AlphaFold3's <target>_ranking_scores.csv, keeps the rounded summary value apart, and analysis/contrasts.py checks
that every run's overall top-1 is AlphaFold3's top-level model with no tie at the top. Synthetic run folders."""

import argparse
import json

import pandas as pd

import collect
import contrasts

CIF = '''data_T1
#
_ma_model_list.model_group_name "AlphaFold-beta-20231127 (3.0.2 @ {stamp})"
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
# AlphaFold3's own scores: sample 3 is the best; the summaries round every one to 0.97 (as in 7BTT on V6).
FULL = {0: 0.9701, 1: 0.9712, 2: 0.9688, 3: 0.9733, 4: 0.9702}


def run_folder(tmp_path, scores=FULL, top_sample=3, csv=True, name='tpu_xla_rep1_seed1_fresh'):
  run = tmp_path / 'ses' / name
  out = run / 'T1' / 'af3_output' / 'T1'
  for k, v in scores.items():
    sd = out / f'seed-1_sample-{k}'
    sd.mkdir(parents=True)
    (sd / f'T1_seed-1_sample-{k}_summary_confidences.json').write_text(
        json.dumps({'ranking_score': round(v, 2), 'ptm': 0.9, 'iptm': 0.9, 'has_clash': 0.0}))
    (sd / f'T1_seed-1_sample-{k}_model.cif').write_text(CIF.format(stamp='2026-10-12 01:00:00', x=f'{k}.000'))
  # The top-level model: the top sample's structure, written at another time.
  (out / 'T1_model.cif').write_text(CIF.format(stamp='2026-10-12 01:05:00', x=f'{top_sample}.000'))
  if csv:
    (out / 'T1_ranking_scores.csv').write_text(
        'seed,sample,ranking_score\n' + ''.join(f'1,{k},{v!r}\n' for k, v in scores.items()))
  (run / 'session.json').write_text(json.dumps({
      'label': 'pilot2', 'config_name': 'tpu_xla', 'config': {'flash_attention_implementation': 'xla'},
      'probe': {'device_kind': 'TPU v6 lite', 'packages': {'jax': '0.10.2'}}, 'compile_cache': {'mode': 'fresh'}}))
  (run / 'runs.jsonl').write_text(json.dumps({'pdb_id': 'T1', 'output_dir': str(run / 'T1' / 'af3_output'),
                                              'seeds': [1], 'exit_code': 0}) + '\n')
  rows, _ = collect.rows_of_run(run, {})
  d = pd.DataFrame(rows)
  d['condition'] = d.platform
  d['m1_success'] = False   # M1 is scored later (collect.main); not needed here
  return d


def test_collect_uses_alphafold3_full_precision_scores(tmp_path):
  d = run_folder(tmp_path).set_index('sample')
  assert d.ranking_score.to_dict() == FULL
  assert set(d.ranking_score_summary) == {0.97}
  assert 'ranking_score_summary' in collect.COLUMNS
  # The top-1 of M2 is AlphaFold3's (sample 3), not the lowest index among rounded ties (sample 0).
  assert contrasts.top(d)[0] == 3


def test_missing_ranking_scores_csv_leaves_the_score_empty(tmp_path):
  d = run_folder(tmp_path, csv=False)
  assert d.ranking_score.isna().all() and set(d.ranking_score_summary) == {0.97}


def by_check(rows):
  return {r['check'].split(' ')[1] + ' ' + r['check'].split(' ')[2]: r for r in rows}


def test_top_model_check_passes_on_alphafold3s_own_top(tmp_path):
  r = by_check(contrasts.top_model_check(run_folder(tmp_path)))
  assert r['overall top-1']['value'] == '1/1 (run, target)' and r['overall top-1']['detail'] == 'pass'
  assert r['no full-precision']['detail'] == 'pass'


def test_top_model_check_fails_when_the_top_level_model_is_another_sample(tmp_path):
  r = by_check(contrasts.top_model_check(run_folder(tmp_path, top_sample=0)))
  assert r['overall top-1']['detail'].startswith('FAIL') and 'seed 1 sample 3 is not the top-level model' in \
      r['overall top-1']['detail']


def test_top_model_check_fails_on_a_full_precision_tie(tmp_path):
  r = by_check(contrasts.top_model_check(run_folder(tmp_path, scores={**FULL, 1: 0.9733})))
  assert r['no full-precision']['detail'].startswith('FAIL') and r['no full-precision']['value'].startswith('1 tie')


def test_top_model_check_fails_without_full_precision_scores(tmp_path):
  r = by_check(contrasts.top_model_check(run_folder(tmp_path, csv=False)))
  assert 'no full-precision ranking_score' in r['overall top-1']['detail']


def test_outputs_have_a_role_column_not_primary(tmp_path):
  rows = []
  for plat, reps in (('V6', ('r1', 'r2')), ('G', ('r1', 'r2')), ('V5', ('r1',))):
    for rep in reps:
      for t in ('T1', 'T2'):
        for k in range(5):
          rows.append({'session': f'{plat}_{rep}', 'label': 'x', 'platform': plat, 'libtpu': None, 'target': t,
                       'seed': 1, 'repetition': rep, 'sample': k, 'num_recycles': 10, 'num_diffusion_samples': 5,
                       'exit_code': 0, 'failure': None, 'ranking_score': 0.9 - 0.1 * k, 'mean_plddt': 80.0,
                       'iptm': 0.8, 'm1_success': k == 0, 'model_path': None})
  path = tmp_path / 'samples.csv'
  pd.DataFrame(rows).to_csv(path, index=False)
  args = argparse.Namespace(samples=str(path), labels=None, include=None, exclude=None, condition='platform',
                            replicates=50, seed=2026)
  _, ok = contrasts.load(args)
  sp, sm, cons, _ = contrasts.build_pairs(ok)
  s, h = contrasts.summarize(sp, sm, cons, args), contrasts.hw_minus_rc(sp, cons, args)
  for frame in (sp, sm, s, h):
    assert 'primary' not in frame.columns and 'role' in frame.columns
  roles = s.drop_duplicates('contrast').set_index('contrast').role.to_dict()
  assert roles['HW(V6,G)'] == 'primary pair' and roles['HW(V5,V6)'] == 'primary pair'
  assert roles['HW(V5,G)'] == 'secondary pair' and roles['RC(V6)'] == 'recompilation'
  assert set(h.role) == {'primary pair', 'secondary pair'}
