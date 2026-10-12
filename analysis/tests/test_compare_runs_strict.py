"""harness/compare_runs.py --strict: every AlphaFold3 output file byte for byte, only the mmCIF timestamp line
(_ma_model_list.model_group_name) ignored (notes/analysis_plan.md section 10, 2026-10-11). Synthetic folders."""

import json
import sys

import compare_runs

CIF = '''data_{t}
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
ATOM A 2 CA 3.800 0.000 0.000 80.00
#
'''


def outputs(root, t='T1', seeds=(1,), stamp='2026-10-11 01:02:03', x='1.000', conf='{"pae": [1.0]}', light=False):
  """An AlphaFold3 output folder of one target as AlphaFold3 writes it; light: as FETCH=light fetches it."""
  d = root / t / 'af3_output' / t
  d.mkdir(parents=True)
  files = {f'{t}_model.cif': CIF.format(t=t, stamp=stamp, x=x), f'{t}_summary_confidences.json': '{"ptm": 0.5}',
           f'{t}_confidences.json': conf, f'{t}_ranking_scores.csv': 'seed,sample,ranking_score\n1,0,0.9\n',
           f'{t}_data.json': '{"name": "T1"}', 'TERMS_OF_USE.md': 'terms\n'}
  for seed in seeds:
    for k in range(5):
      sd = f'seed-{seed}_sample-{k}'
      files[f'{sd}/{t}_{sd}_model.cif'] = CIF.format(t=t, stamp=stamp, x=f'{k}{x[1:]}')
      files[f'{sd}/{t}_{sd}_summary_confidences.json'] = '{"ptm": 0.5}'
      files[f'{sd}/{t}_{sd}_confidences.json'] = conf
  for name, text in files.items():
    if light and (name.endswith('_data.json') or
                  (name.endswith('_confidences.json') and not name.endswith('_summary_confidences.json'))):
      continue
    (d / name).parent.mkdir(parents=True, exist_ok=True)
    (d / name).write_text(text)
  return d


def test_only_the_timestamp_line_is_ignored(tmp_path):
  ref = outputs(tmp_path / 'r1')
  s = compare_runs.strict_compare(ref, outputs(tmp_path / 'w', stamp='2026-10-12 23:59:59'))
  assert s['status'] == 'identical' and s['files'] == 21 and len(s['identical']) == 21 and s['target_files_compared']
  s = compare_runs.strict_compare(ref, outputs(tmp_path / 'x', x='1.001'))
  assert s['status'] == 'differ' and len(s['differ']) == 6 and all(n.endswith('_model.cif') for n in s['differ'])
  s = compare_runs.strict_compare(ref, outputs(tmp_path / 'c', conf='{"pae": [1.0000001]}'))
  assert len(s['differ']) == 6 and all(n.endswith('_confidences.json') for n in s['differ'])
  other = outputs(tmp_path / 'name')
  cif = other / 'seed-1_sample-3' / 'T1_seed-1_sample-3_model.cif'
  cif.write_text(cif.read_text().replace('"Top ranked model"', '"Top ranked model "'))
  assert compare_runs.strict_compare(ref, other)['differ'] == ['seed-1_sample-3/T1_seed-1_sample-3_model.cif']
  terms = outputs(tmp_path / 'terms')
  (terms / 'TERMS_OF_USE.md').write_text('other\n')
  assert compare_runs.strict_compare(ref, terms)['differ'] == ['TERMS_OF_USE.md']


def test_masking_needs_the_tag_as_the_first_token_of_an_mmcif_line(tmp_path):
  a, b = tmp_path / 'a.cif', tmp_path / 'b.cif'
  a.write_text('_ma_model_list.model_group_name "x @ 1"\n_ma_model_list.model_group_id 1\n# model_group_name x\n')
  b.write_text('_ma_model_list.model_group_name "y @ 2"\n_ma_model_list.model_group_id 1\n# model_group_name x\n')
  assert compare_runs.masked(a) == compare_runs.masked(b)
  b.write_text('_ma_model_list.model_group_name "y @ 2"\n_ma_model_list.model_group_id 1\n# model_group_name y\n')
  assert compare_runs.masked(a) != compare_runs.masked(b)
  b.write_text('_ma_model_list.model_group_name_x "x @ 1"\n_ma_model_list.model_group_id 1\n# model_group_name x\n')
  assert compare_runs.masked(a) != compare_runs.masked(b)
  j1, j2 = tmp_path / 'a.json', tmp_path / 'b.json'
  j1.write_text('_ma_model_list.model_group_name 1\n')
  j2.write_text('_ma_model_list.model_group_name 2\n')
  assert compare_runs.masked(j1) != compare_runs.masked(j2)


def test_light_fetch_is_missing_not_a_pass(tmp_path):
  s = compare_runs.strict_compare(outputs(tmp_path / 'a', light=True), outputs(tmp_path / 'b', light=True))
  assert s['status'] == 'missing' and 'T1_data.json' in s['missing'] and 'T1_confidences.json' in s['missing']
  assert len(s['missing']) == 7 and len(s['identical']) == 14
  s = compare_runs.strict_compare(outputs(tmp_path / 'c'), outputs(tmp_path / 'd', light=True))
  assert s['status'] == 'missing' and len(s['missing']) == 7


def test_seeds(tmp_path):
  # Different seeds: nothing to compare. Overlapping seeds: the common seed's samples, not the ranked target files.
  s = compare_runs.strict_compare(outputs(tmp_path / 'a', seeds=(1,)), outputs(tmp_path / 'b', seeds=(2,)))
  assert s['status'] == 'no common seed' and s['files'] == 0
  s = compare_runs.strict_compare(outputs(tmp_path / 'c', seeds=(1, 2)), outputs(tmp_path / 'd', seeds=(1,)))
  assert s['status'] == 'identical' and not s['target_files_compared'] and s['seeds'] == [1]
  assert s['files'] == 15 and all(n.startswith('seed-1_') for n in s['identical'])


def session(tmp_path, warm_stamp, warm_x='1.000'):
  ses = tmp_path / 'ses'
  for run, created, stamp, x in (('tpu_xla_rep1_seed1_fresh', '2026-10-11T01:00:00Z', '2026-10-11 01:00:00', '1.000'),
                                 ('tpu_xla_seed1_warm', '2026-10-11T02:00:00Z', warm_stamp, warm_x)):
    (ses / run).mkdir(parents=True)
    (ses / run / 'session.json').write_text(json.dumps({'created_utc': created}))
    outputs(ses / run, stamp=stamp, x=x)
  return ses


def cli(monkeypatch, capsys, *argv):
  monkeypatch.setattr(sys, 'argv', ['compare_runs.py', *argv])
  assert compare_runs.main() == 0
  return capsys.readouterr().out


def test_cli_strict_and_unchanged_without_it(tmp_path, monkeypatch, capsys):
  ses = session(tmp_path, '2026-10-11 02:00:00')
  out = cli(monkeypatch, capsys, '--session', str(ses))
  assert 'strict' not in out and len(out.splitlines()) == 2
  report = json.loads((ses / 'compare_runs.json').read_text())
  assert [set(c) for c in report['comparisons']] == [{'label', 'reference', 'run', 'summary', 'samples'}]
  out = cli(monkeypatch, capsys, '--session', str(ses), '--strict', '--out', 'strict')
  assert out.splitlines()[2] == ('    strict (all files, mmCIF timestamp line ignored): identical, '
                                 '21/21 files identical')
  assert json.loads((ses / 'strict.json').read_text())['comparisons'][0]['strict']['status'] == 'identical'
  ses2 = session(tmp_path / 'two', '2026-10-11 02:00:00', warm_x='1.001')
  out = cli(monkeypatch, capsys, '--session', str(ses2), '--strict')
  assert 'strict (all files, mmCIF timestamp line ignored): differ, 15/21 files identical; differ: ' in out
  assert 'and 3 more' in out
