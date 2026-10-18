"""harness/featurise_only.py (follow-up 5, notes/analysis_plan.md section 10, row of 2026-10-18): its featurisation
call is run_alphafold.py's, and on this machine it is deterministic on repeat. The second test runs AlphaFold3's
featurisation on two small frozen inputs (needs third_party/alphafold3/.venv with build_data done, and data/inputs/)."""
import ast
import hashlib
import json
import pathlib
import subprocess
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parents[2]
AF3 = REPO / 'third_party' / 'alphafold3'
RUN_AF = AF3 / 'run_alphafold.py'
AF3_PY = AF3 / '.venv' / 'bin' / 'python'
FEAT = REPO / 'harness' / 'featurise_only.py'
RUN_AF_SHA256 = 'c62bfaca9476c57c519429abc7f57d8db05165487165cbb463308e98b5f7ace4'   # every arm's session.json


def functions(path):
  return {n.name: n for n in ast.walk(ast.parse(path.read_text())) if isinstance(n, ast.FunctionDef)}


def call_keywords(fn, callee):
  calls = [n for n in ast.walk(fn) if isinstance(n, ast.Call) and ast.unparse(n.func).endswith(callee)]
  assert len(calls) == 1, f'{fn.name}: {len(calls)} calls to {callee}'
  return {k.arg: ast.unparse(k.value) for k in calls[0].keywords}


def test_run_alphafold_is_the_file_every_arm_ran():
  if not RUN_AF.is_file():
    pytest.skip('no AlphaFold3 checkout')
  assert hashlib.sha256(RUN_AF.read_bytes()).hexdigest() == RUN_AF_SHA256


def test_featurisation_call_matches_run_alphafold_v304():
  if not RUN_AF.is_file():
    pytest.skip('no AlphaFold3 checkout')
  af, ours = functions(RUN_AF), functions(FEAT)
  # predict_structure: Ccd(user_ccd=fold_input.user_ccd), then featurise_input with these keywords.
  ccd_lines = [ast.unparse(n) for n in ast.walk(af['predict_structure']) if isinstance(n, ast.Assign)
               and ast.unparse(n.targets[0]) == 'ccd']
  assert ccd_lines == ['ccd = chemical_components.Ccd(user_ccd=fold_input.user_ccd)']
  assert [ast.unparse(n) for n in ast.walk(ours['featurise']) if isinstance(n, ast.Assign)
          and ast.unparse(n.targets[0]) == 'ccd'] == ccd_lines
  theirs = call_keywords(af['predict_structure'], 'featurisation.featurise_input')
  mine = call_keywords(ours['featurise'], 'featurisation.featurise_input')
  assert theirs == {'fold_input': 'fold_input', 'buckets': 'buckets', 'ccd': 'ccd', 'verbose': 'True',
                    'ref_max_modified_date': 'ref_max_modified_date',
                    'conformer_max_iterations': 'conformer_max_iterations',
                    'resolve_msa_overlaps': 'resolve_msa_overlaps', 'fix_standalone_glycans': 'fix_standalone_glycans'}
  assert set(mine) == set(theirs) and mine['verbose'] == 'True'
  assert all(mine[k] == ("args['%s']" % k) for k in theirs if k not in ('fold_input', 'ccd', 'verbose'))
  # process_fold_input hands its values to predict_structure unchanged ...
  passed = call_keywords(af['process_fold_input'], 'predict_structure')
  assert all(passed[k] == k for k in ('buckets', 'ref_max_modified_date', 'conformer_max_iterations',
                                      'resolve_msa_overlaps', 'fix_standalone_glycans'))
  # ... and main() computes them from its flags, as featurisation_args does.
  main_kw = call_keywords(af['main'], 'process_fold_input')
  assert main_kw['buckets'] == 'tuple((int(bucket) for bucket in _BUCKETS.value))'
  assert main_kw['ref_max_modified_date'] == 'max_template_date'
  assert 'max_template_date = datetime.date.fromisoformat(_MAX_TEMPLATE_DATE.value)' in [
      ast.unparse(n) for n in ast.walk(af['main']) if isinstance(n, ast.Assign)]
  ret = [n for n in ast.walk(ours['featurisation_args']) if isinstance(n, ast.Return)][0].value
  args = {ast.literal_eval(k): ast.unparse(v) for k, v in zip(ret.keys, ret.values)}
  assert args['buckets'] == 'tuple((int(bucket) for bucket in run_alphafold._BUCKETS.value))'
  assert args['ref_max_modified_date'] == 'datetime.date.fromisoformat(run_alphafold._MAX_TEMPLATE_DATE.value)'
  for k in ('conformer_max_iterations', 'resolve_msa_overlaps', 'fix_standalone_glycans'):
    assert args[k] == 'run_alphafold.' + main_kw[k]
  # The one-seed input is the one harness/run_af3.py writes.
  run_af3 = (REPO / 'harness' / 'run_af3.py').read_text()
  assert "data['modelSeeds'] = seeds" in run_af3 and "run_input.write_text(json.dumps(data, indent=2) + '\\n')" in run_af3


def featurise(out):
  rc = subprocess.run([str(AF3_PY), str(FEAT), '--manifest', 'inputs/feat/manifest.csv', '--targets', '6YQW,8EY4',
                       '--seeds', '1,2', '--out', str(out)], cwd=REPO, capture_output=True, text=True)
  assert rc.returncode == 0, rc.stdout[-2000:] + rc.stderr[-2000:]
  return {(d['target'], d['seed']): d for t in ('6YQW', '8EY4')
          for d in map(json.loads, (out / t / 'features.jsonl').read_text().splitlines())}


@pytest.mark.skipif(not AF3_PY.is_file() or not (REPO / 'data' / 'inputs' / '6YQW.json').is_file(),
                    reason='needs the AlphaFold3 venv and the frozen inputs')
def test_two_small_targets_run_and_are_deterministic(tmp_path):
  a, b = featurise(tmp_path / 'a'), featurise(tmp_path / 'b')    # two separate processes
  assert set(a) == {('6YQW', 1), ('6YQW', 2), ('8EY4', 1), ('8EY4', 2)}
  for k in a:
    assert a[k]['error'] is None and a[k]['arrays'] == b[k]['arrays'], k
    assert a[k]['run_input_sha256'] == b[k]['run_input_sha256']
    assert sum(r['model_input'] for r in a[k]['arrays']) >= 50
  ref = {r['name']: r['sha256'] for r in a[('6YQW', 1)]['arrays']}
  ref2 = {r['name']: r['sha256'] for r in a[('6YQW', 2)]['arrays']}
  assert ref['ref_pos'] != ref2['ref_pos']              # the seed reaches the ligand's RDKit conformer
  host = json.loads((tmp_path / 'a' / 'host.json').read_text())
  assert host['packages']['numpy'] and host['packages']['rdkit'] and host['ccd_pickles_sha256']
