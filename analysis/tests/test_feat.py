"""analysis/feat.py (follow-up 5, notes/analysis_plan.md section 10, row of 2026-10-18) on synthetic sessions: the
checks, the comparison and the decision rule. No real output is read."""
import json
import sys

import numpy as np
import pandas as pd
import pytest

import feat

TARGETS = ['T1', 'T2']
SEEDS = [1, 2]
FEATURES = [('aatype', True), ('ref_pos', True), ('empty_output_struc', False)]


def host(family, **over):
  h = {'cpu': f'cpu {family}', 'cpu_count': 8, 'python': '3.12.15', 'kernel': '6.8.0', 'glibc': '2.35',
       'gce_machine_type': f'mt-{family}', 'cpuinfo': {'avx512f': family != 'milan'},
       'os_release': {'PRETTY_NAME': 'Ubuntu 22.04.5 LTS'},
       'packages': {'numpy': '2.4.1', 'rdkit': '2025.9.4', 'scipy': '1.17.0', 'alphafold3': f'3.0.5.dev0+g85c4d2050.d{family}'},
       # The pickles differ on every host, as on the VMs (not byte-reproducible); they are reported, not checked.
       'ccd_pickles_sha256': {'ccd.pickle': f'aa{family}', 'chemical_component_sets.pickle': f'bb{family}'}}
  h.update(over)
  return h


CIF = 'f1bef735f3d74171fb7e79af6b46fb1a3e753f6602f5b08ec90312013e33244b'


def session(root, family, stamp='20261019T000000Z', changed=None, drop=None, arrays=True, host_over=None, cif=CIF):
  """One feat session; changed = {(target, seed, feature)}: a different hash (and array) on this host; cif: the
  components.cif build_data read (ccd.json), None for no ccd.json."""
  name = f'{stamp}_cpu_feat_{family}'
  d = root / name / 'featurise'
  d.mkdir(parents=True)
  if cif:
    (root / name / 'ccd.json').write_text(json.dumps({'used_sha256': cif, 'used_bytes': 519231791}))
  (d / 'host.json').write_text(json.dumps(host(family, **(host_over or {}))))
  for t in TARGETS:
    (d / t).mkdir()
    for s in SEEDS:
      if drop == (t, s):
        continue
      recs = []
      arrs = {}
      for name_, model in FEATURES:
        bump = changed and (t, s, name_) in changed
        recs.append({'name': name_, 'model_input': model, 'dtype': 'float32', 'shape': [4],
                     'sha256': f'{t}{s}{name_}' + ('x' if bump else '')})
        if model:
          arrs[name_] = np.arange(4, dtype=np.float32) + s + (np.float32(1e-6) if bump else 0)
      with open(d / t / 'features.jsonl', 'a') as f:
        f.write(json.dumps({'target': t, 'seed': s, 'run_input_sha256': f'in{t}{s}', 'arrays': recs, 'error': None}) + '\n')
      if arrays:
        (d / t / 'arrays').mkdir(exist_ok=True)
        np.savez(d / t / 'arrays' / f'seed-{s}.npz', **arrs)
  return name


def run(tmp_path, monkeypatch, sessions):
  (tmp_path / 'sessions.txt').write_text('# four hosts\n' + '\n'.join(sessions) + '\n')
  (tmp_path / 'm.csv').write_text('pdb_id,input_path\n' + ''.join(f'{t},x\n' for t in TARGETS))
  out = tmp_path / 'out'
  monkeypatch.setattr(feat, 'REPO', tmp_path)
  monkeypatch.setattr(sys, 'argv', ['feat.py', '--sessions', str(tmp_path / 'sessions.txt'), '--root', str(tmp_path / 'af3'),
                                    '--manifest', 'm.csv', '--seeds', '1,2', '--out', str(out)])
  return out


def test_identical_everywhere_excludes_the_host(tmp_path, monkeypatch):
  root = tmp_path / 'af3'
  out = run(tmp_path, monkeypatch, [session(root, f) for f in feat.FAMILIES])
  assert feat.main() == 0
  dec = json.loads((out / 'decision.json').read_text())
  assert dec['branch'] == 'host excluded' and dec['model_inputs']['different'] == 0
  assert dec['model_inputs']['compared'] == len(TARGETS) * len(SEEDS) * 2 == dec['model_inputs']['identical']
  assert dec['host_side'] == {'compared': len(TARGETS) * len(SEEDS), 'identical': len(TARGETS) * len(SEEDS),
                              'different': 0, 'units_with_a_difference': 0}
  assert pd.read_csv(out / 'differences.csv').empty
  assert 'host is excluded as a source of the HW differences' in (out / 'report.md').read_text()


def test_one_differing_array_triggers_5b_and_is_listed_with_its_difference(tmp_path, monkeypatch):
  root = tmp_path / 'af3'
  names = [session(root, f, changed={('T2', 1, 'ref_pos')} if f == 'genoa' else None) for f in feat.FAMILIES]
  out = run(tmp_path, monkeypatch, names)
  assert feat.main() == 0
  dec = json.loads((out / 'decision.json').read_text())
  assert dec['branch'] == '5b' and dec['model_inputs']['different'] == 1 and dec['host_side']['different'] == 0
  assert dec['model_inputs']['units_with_a_difference'] == 1
  d = pd.read_csv(out / 'differences.csv')
  assert list(d[['target', 'seed', 'feature']].itertuples(index=False, name=None)) == [('T2', 1, 'ref_pos')]
  assert d.host_groups[0] == 'milan+cascadelake+icelake | genoa'
  assert d.max_abs_difference[0] == pytest.approx(1e-6, rel=0.1)   # float32 spacing near 2 to 6 and d.differing_elements[0] == 4
  c = pd.read_csv(out / 'counts.csv').set_index('feature')
  assert c.loc['ref_pos', 'different'] == 1 and c.loc['ref_pos', 'identical'] == 3


def test_a_host_side_difference_alone_is_reported_but_does_not_trigger_5b(tmp_path, monkeypatch):
  # Implementation row of 2026-10-18: the rule reads the 59 model inputs only.
  root = tmp_path / 'af3'
  names = [session(root, f, arrays=False, changed={('T1', 2, 'empty_output_struc')} if f == 'icelake' else None)
           for f in feat.FAMILIES]
  out = run(tmp_path, monkeypatch, names)
  assert feat.main() == 0
  dec = json.loads((out / 'decision.json').read_text())
  assert dec['branch'] == 'host excluded' and dec['model_inputs']['different'] == 0
  assert dec['host_side']['different'] == 1 and dec['host_side']['units_with_a_difference'] == 1
  d = pd.read_csv(out / 'differences.csv')
  assert list(d.feature) == ['empty_output_struc'] and not d.model_input[0]
  assert not d.arrays_compared[0] and pd.isna(d.max_abs_difference[0])
  rep = (out / 'report.md').read_text()
  assert 'Differing host-side entries (reported apart; they do not decide)' in rep
  assert 'Differing model-input arrays' not in rep


def test_model_input_and_host_side_differences_together_trigger_5b(tmp_path, monkeypatch):
  root = tmp_path / 'af3'
  names = [session(root, f, changed={('T1', 1, 'aatype'), ('T1', 1, 'empty_output_struc')} if f == 'milan' else None)
           for f in feat.FAMILIES]
  out = run(tmp_path, monkeypatch, names)
  assert feat.main() == 0
  dec = json.loads((out / 'decision.json').read_text())
  assert dec['branch'] == '5b' and dec['model_inputs']['different'] == 1 and dec['host_side']['different'] == 1
  assert pd.read_csv(out / 'differences.csv').host_groups.tolist() == ['milan | genoa+cascadelake+icelake'] * 2


@pytest.mark.parametrize('case', ['missing_seed', 'two_sessions', 'ccd', 'no_ccd_json', 'numpy', 'input'])
def test_a_failed_check_stops_before_any_comparison(tmp_path, monkeypatch, case):
  root = tmp_path / 'af3'
  names = []
  for f in feat.FAMILIES:
    over = {}
    if case == 'numpy' and f == 'genoa':
      over['packages'] = dict(host(f)['packages'], numpy='2.4.0')
    cif = CIF
    if f == 'milan' and case in ('ccd', 'no_ccd_json'):
      cif = '3db18637' + CIF[8:] if case == 'ccd' else None
    names.append(session(root, f, drop=('T1', 2) if case == 'missing_seed' and f == 'icelake' else None, host_over=over,
                         cif=cif))
  if case == 'two_sessions':
    names.append(session(root, 'milan', stamp='20261019T010000Z'))
  if case == 'input':
    p = root / names[1] / 'featurise' / 'T1' / 'features.jsonl'
    p.write_text(p.read_text().replace('"inT11"', '"changed"'))
  out = run(tmp_path, monkeypatch, names)
  with pytest.raises(SystemExit, match='a check failed'):
    feat.main()
  chk = pd.read_csv(out / 'checks.csv')
  assert (chk.detail != 'pass').any() and not (out / 'decision.json').exists()


def test_alphafold3_build_label_is_not_a_difference():
  assert feat.base_version('3.0.5.dev0+g85c4d2050.d20261019') == feat.base_version('3.0.5.dev0+g85c4d2050.d20261020')


def test_a_session_that_is_not_a_feat_session_is_refused(tmp_path):
  with pytest.raises(SystemExit, match='not a follow-up 5 session'):
    feat.load(tmp_path, ['20261019T000000Z_cpu_cpu60_7d5c'])


def test_ccd_pickles_that_differ_are_reported_not_checked(tmp_path, monkeypatch):
  # Section 10, "Follow-up 5, analysis stopped at a pre-specified check": the pickles build_data writes are not
  # byte-reproducible; the check reads the components.cif every host's build_data read (ccd.json).
  root = tmp_path / 'af3'
  out = run(tmp_path, monkeypatch, [session(root, f) for f in feat.FAMILIES])
  assert feat.main() == 0
  chk = pd.read_csv(out / 'checks.csv')
  assert (chk.detail == 'pass').all() and not chk.check.str.contains('pickle').any()
  row = chk[chk.check.str.contains('components.cif')]
  assert len(row) == 1 and row.value.iloc[0].count(CIF[:12]) == 4
  h = pd.read_csv(out / 'hosts.csv')
  assert list(h.components_cif_sha256) == [CIF] * 4
  assert h.ccd_pickle_sha256.nunique() == 4 and h.chemical_component_sets_pickle_sha256.nunique() == 4
