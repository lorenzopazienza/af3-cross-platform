"""analysis/msa.py (follow-up 6, notes/analysis_plan.md section 10, row of 2026-10-19) on synthetic data: Neff as
fixed in the module, the template audit on an AlphaFold3-written template, the determinism comparison by field, and
the inference part (completeness, contrasts, IN, official minus ColabFold, success rates and the reading rule, on
all targets and on the post-cutoff ones). No real output is read."""
import csv
import hashlib
import json
import pathlib
import types

import numpy as np
import pandas as pd
import pytest

import contrasts
import msa

DATA = pathlib.Path(__file__).resolve().parent / 'fixtures'
ARGS = types.SimpleNamespace(seed=2026, replicates=40)


def a3m(*rows):
  return ''.join(f'>s{i}\n{r}\n' for i, r in enumerate(rows))


def test_neff_by_hand():
  # Q and R1 identical, R2 0.9 to both, R3 0.6 or less to all: weights 1/3, 1/3, 1/3, 1.
  rows = msa.a3m_rows(a3m('ACDEFGHIKL', 'ACDEFGHIKL', 'ACDEFGHIKM', 'ACDEFGWWWW'))
  assert msa.neff(msa.msa_matrix(rows)) == pytest.approx(2.0)
  assert msa.neff(msa.msa_matrix(rows), block=1) == pytest.approx(2.0)          # blocks do not change it


def test_insertions_are_removed_and_a_gap_matches_a_gap():
  rows = msa.a3m_rows(a3m('ACDEFGHIKL', '-----FGHIK', '-----FGHIKmmm'.replace('K', 'Kq')))
  assert rows == ['ACDEFGHIKL', '-----FGHIK', '-----FGHIK']
  # the two gapped rows are identical (gap = gap), and 0.5 to the query: weights 1, 1/2, 1/2
  assert msa.neff(msa.msa_matrix(rows)) == pytest.approx(2.0)
  assert msa.neff(msa.msa_matrix([])) == 0.0
  with pytest.raises(ValueError, match='query length'):
    msa.msa_matrix(['ACDE', 'ACD'])


def test_template_info_and_audit_on_an_alphafold3_template(tmp_path, monkeypatch):
  info = msa.template_info((DATA / 'msa_template_6yqw_A.cif').read_text())
  assert info['pdb_id'] == '6YQW' and info['chain'] == 'A' and info['release_date'] == '2021-04-14'
  d = json.loads((DATA / 'msa_data_6yqw.json').read_text())
  p = tmp_path / '6YQW.json'
  p.write_text(json.dumps(d))
  sha = hashlib.sha256(p.read_bytes()).hexdigest()
  monkeypatch.setattr(msa, 'REPO', tmp_path)
  official = {'6YQW': {'input_path': '6YQW.json', 'input_sha256': sha, 'ligand_ccd': '82I',
                       'released_on_or_before_cutoff': 'yes'}}
  cache = tmp_path / 'rcsb.json'
  cache.write_text(json.dumps({'6YQW': {'release': '2021-04-14', 'ligands': ['82I', 'SO4']}}))
  t = msa.template_audit(official, cache, fetch=False)
  r = t.iloc[0]
  assert (r.template_pdb_id, r.template_chain, r.release_date) == ('6YQW', 'A', '2021-04-14')
  assert r.identity >= 0.95 and r.own_entry and r.identity_ge_95 and r.contains_target_ligand == 'yes'
  cache.write_text('{}')
  assert msa.template_audit(official, cache, fetch=False).iloc[0].contains_target_ligand == 'unknown'


def test_determinism_reports_differences_by_field(tmp_path, monkeypatch):
  monkeypatch.setattr(msa, 'NAS', tmp_path)
  base = json.loads((DATA / 'msa_data_6yqw.json').read_text())
  other = json.loads(json.dumps(base))
  prot = other['sequences'][0]['protein']
  prot['unpairedMsa'] = prot['unpairedMsa'] + '>extra\n' + prot['sequence'] + '\n'
  for run, d in ((msa.RUN1, base), (msa.RUN2, base), (msa.NAS6, other)):
    f = tmp_path / run / 'output' / '6YQW' / '6yqw' / '6yqw_data.json'
    f.parent.mkdir(parents=True)
    f.write_text(json.dumps(d))
    with open(tmp_path / run / 'results.csv', 'w', newline='') as fh:
      w = csv.writer(fh)
      w.writerow(['pdb_id', 'data_json', 'data_json_sha256'])
      w.writerow(['6YQW', f'/data/runs/{run}/output/6YQW/6yqw/6yqw_data.json', hashlib.sha256(f.read_bytes()).hexdigest()])
  r = msa.determinism({'6YQW': {'pilot': 'yes'}}).iloc[0]
  assert r.run2_identical and not r.naspaz6_identical and r.run2_difference == ''
  diff = json.loads(r.naspaz6_difference)
  assert diff['unpairedMsa']['rows'][1] == diff['unpairedMsa']['rows'][0] + 1
  assert diff['templates']['same_hits_same_order'] and diff['pairedMsa']['same_rows_same_order']


# ---- inference

@pytest.fixture(autouse=True)
def no_files(monkeypatch):
  monkeypatch.setattr(contrasts, 'coords_text', lambda path: path.split('@')[1])
  monkeypatch.setattr(contrasts, 'divergence', lambda t, a, b: {'ligand_rmsd': 1.0, 'ca_rmsd_global': 0.5, 'error': None})


TARGETS = ['T1', 'T2', 'T3', 'T4']
PILOT = ['T1']
POST = {'T3', 'T4'}


def rows(platform, label, rep, targets, seeds, shift=0.0, coords='x', sha=None):
  out = []
  for t in targets:
    for s in seeds:
      for k in range(5):
        out.append({'session': f'S_{label}_{platform}_{rep}', 'run': f'run_{rep}_seed{s}', 'label': label,
                    'platform': platform, 'target': t, 'seed': s, 'repetition': rep, 'sample': k, 'num_recycles': 10,
                    'num_diffusion_samples': 5, 'ranking_score': 0.9 - 0.1 * k + (shift if k == 1 else 0.0),
                    'm1_success': k == 0, 'mean_plddt': 80.0, 'iptm': 0.8,
                    'model_path': f'm/{label}/{platform}/{rep}/{t}/{s}/{k}@{coords}{k}', 'exit_code': 0,
                    'input_sha256': (sha or {}).get(t, f'cf_{t}')})
  return out


def write_world(tmp, v6o_shift=0.2, drop_go_r2=False):
  official = {t: {'input_sha256': f'off_{t}', 'pilot': 'yes' if t in PILOT else 'no',
                  'released_on_or_before_cutoff': 'no' if t in POST else 'yes'} for t in TARGETS}
  sha = {t: official[t]['input_sha256'] for t in TARGETS}
  seeds = [1, 2]
  m = (rows('V5', 'main', 'r1', TARGETS, seeds) + rows('V6', 'main', 'r1', TARGETS, seeds)
       + rows('G', 'main', 'r1', TARGETS, seeds))
  o = (rows('V5', 'msa', 'r1', TARGETS, seeds, sha=sha) + rows('V5', 'msa', 'w', PILOT, [1], sha=sha)
       + rows('V6', 'msa', 'r1', TARGETS, seeds, shift=v6o_shift, coords='y', sha=sha)
       + rows('V6', 'msa', 'w', PILOT, [1], shift=v6o_shift, coords='y', sha=sha)
       + rows('G', 'msa', 'r1', TARGETS, seeds, sha=sha) + ([] if drop_go_r2 else rows('G', 'msa', 'r2', TARGETS, seeds, sha=sha)))
  main = tmp / 'main'
  (main / 'contrasts').mkdir(parents=True)
  pd.DataFrame(m).to_csv(main / 'samples.csv', index=False)
  pd.DataFrame([{'contrast': c, 'target': t, 'seed_a': s, 'seed_b': s, 'flip_top1': False, 'flip_success': False,
                 'flip_matters': False} for c in ('HW(V5,V6)', 'HW(V6,G)', 'HW(V5,G)') for t in TARGETS for s in seeds]
               ).to_csv(main / 'contrasts' / 'pairs_seed.csv', index=False)
  pd.DataFrame(o).to_csv(tmp / 'samples.csv', index=False)
  return official, types.SimpleNamespace(samples=tmp / 'samples.csv', main=main, seed=2026, replicates=40)


def test_inference_contrasts_rule_and_comparisons(tmp_path, monkeypatch):
  monkeypatch.setattr(msa, 'PLAN', {('Go', 'r1'): 8, ('Go', 'r2'): 8, ('V6o', 'r1'): 8, ('V5o', 'r1'): 8,
                                    ('V6o', 'w'): 1, ('V5o', 'w'): 1})
  official, args = write_world(tmp_path)
  res = msa.inference(args, official, tmp_path)
  assert (res['completeness'].status == 'complete').all()
  con = res['contrasts']
  allc = con[con.targets == 'all 60']
  get = lambda c, st: float(allc[(allc.contrast == c) & (allc.statistic == st)].estimate.iloc[0])  # noqa: E731
  assert get('HW(V5o,V6o)', 'flip_top1') == 1.0 and get('HW(V6o,Go)', 'flip_top1') == 1.0
  assert get('HW(V5o,Go)', 'flip_top1') == 0.0 and get('RC(Go)', 'flip_top1') == 0.0
  assert get('W(V6o)', 'bit_identical_fraction') == 1.0 and get('IN(V6)', 'flip_top1') == 1.0
  assert set(allc.contrast) >= {'SD(V5o)', 'SD(V6o)', 'SD(Go)', 'IN(V5)', 'IN(V6)', 'IN(G)', 'W(V5o)'}
  rule = res['reading_rule']
  first = rule[rule.targets == 'all 60'].iloc[0]                     # the headline is written first
  assert (first.statistic, first.role) == ('flip_top1', 'headline')
  assert set(rule[rule.targets == 'all 60'].role[1:]) == {'secondary'}
  assert first.branch == 'the main conclusion holds with official inputs' and first.incomplete_arms == ''
  omc = res['official_minus_colabfold']
  o = omc[(omc.targets == 'all 60') & (omc.statistic == 'flip_top1')].set_index('official')
  assert o.loc['HW(V5o,V6o)', 'reading'] == 'higher' and o.loc['HW(V5o,Go)', 'reading'] == 'similar'
  # post-cutoff: only the statistics that use M1
  assert set(con[con.targets == 'post-cutoff 39'].statistic) == set(msa.M1_STATS)
  assert set(rule[rule.targets == 'post-cutoff 39'].statistic) == set(msa.M1_STATS)
  sub = omc[omc.targets == 'post-cutoff 39']
  assert set(sub.n_targets) == {2}
  sr = res['success_rates']
  assert set(sr[sr.targets == 'all 60'].condition) == {'V5', 'V6', 'G', 'V5o', 'V6o', 'Go'}
  assert float(sr[(sr.targets == 'all 60') & (sr.condition == 'V6o')].success_rate.iloc[0]) == 0.0   # top-1 sample 1


def test_an_incomplete_arm_is_reported_never_imputed(tmp_path, monkeypatch):
  monkeypatch.setattr(msa, 'PLAN', {('Go', 'r1'): 8, ('Go', 'r2'): 8, ('V6o', 'r1'): 8, ('V5o', 'r1'): 8,
                                    ('V6o', 'w'): 1, ('V5o', 'w'): 1})
  official, args = write_world(tmp_path, drop_go_r2=True)
  res = msa.inference(args, official, tmp_path)
  comp = res['completeness'].set_index(['condition', 'repetition'])
  assert comp.loc[('Go', 'r2'), 'status'] == 'incomplete' and comp.loc[('Go', 'r2'), 'units_with_5_samples'] == 0
  assert 'RC(Go)' not in set(res['contrasts'].contrast)          # no r2 pairs: not computed, not imputed
  rule = res['reading_rule']
  assert (rule.incomplete_arms == 'Go').all()
  assert rule.branch.str.startswith('not applied: RC(Go) is not available').all()
  assert rule.branch.str.contains("RC\\(G\\) is not substituted").all()
  assert (rule['HW(V6o,Go) minus RC(Go), ci_low'] == 'not available').all()
  hw = res['hw_minus_rc']
  assert hw.estimate.isna().all() and hw.note.str.startswith('not available').all() and set(hw.recompilation) == {'RC(Go)'}


def test_an_input_that_is_not_the_official_data_json_stops(tmp_path):
  official, args = write_world(tmp_path)
  d = pd.read_csv(args.samples)
  d.loc[0, 'input_sha256'] = 'something else'
  d.to_csv(args.samples, index=False)
  with pytest.raises(SystemExit, match='not the official _data.json'):
    msa.msa_samples(args.samples, official)
