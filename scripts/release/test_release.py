"""scripts/release/release.py: categories, session kinds, the weights guard, the *_data.json check, and refused
builds that keep nothing.

    analysis/.venv/bin/python -m pytest -q scripts/release/test_release.py
"""
import hashlib
import json
import re
import pathlib
import sys
import types

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import release  # noqa: E402


@pytest.mark.parametrize('path,cat', [
    ('r/7U3J/af3_output/7U3J/seed-1_sample-0/7U3J_seed-1_sample-0_model.cif', 'model_cif'),
    ('r/7U3J/af3_output/7U3J/7U3J_model.cif', 'model_cif'),
    ('r/7U3J/af3_output/7U3J/seed-1_sample-0/7U3J_seed-1_sample-0_summary_confidences.json', 'ranking_summary'),
    ('r/7U3J/af3_output/7U3J/7U3J_ranking_scores.csv', 'ranking_summary'),
    ('r/7U3J/af3_output/7U3J/seed-1_sample-0/7U3J_seed-1_sample-0_confidences.json', 'full_confidences'),
    ('r/7U3J/af3_output/7U3J/7U3J_data.json', 'data_json'),
    ('r/7U3J/af3_output/7U3J/seed-1_embeddings/7U3J_seed-1_embeddings.npz', 'embeddings'),
    ('r/7U3J/af3_output/7U3J/TERMS_OF_USE.md', 'terms'), ('r/7U3J/run_alphafold.log', 'af3_logs'),
    ('r/7U3J/run.json', 'records'), ('session.json', 'records'), ('launch_record.txt', 'records'),
    ('r/7U3J/work/input_seeds.json', 'work'), ('r/7U3J/work/jax_cache/jit_apply_fn-abc-cache', 'work'),
    ('xla_dump.tgz', 'other_binary'),
    ('r/7XQZ/xla_dump/module_0537.jit_apply_fn.cl_970057490.after_codegen.txt', 'xla_dump'),
    ('r/7XQZ/xla_dump/module_0534.jit_apply_fn.cl_970057490/module.mlir', 'xla_dump'),
    ('r/8EYE/fu13_capture_seed-1.npz', 'captures'), ('featurise/7T6A/arrays/seed-1.npz', 'features')])
def test_category(path, cat):
  assert release.category(path) == cat


@pytest.fixture(autouse=True)
def _no_local_inventory(tmp_path, monkeypatch):
  monkeypatch.setattr(release, 'OUT_DIR', tmp_path / 'no_inventory')   # the plain files of a record do not depend on results/release


def _file(tmp_path, name, data=b'x' * 10):
  p = tmp_path / name
  p.parent.mkdir(parents=True, exist_ok=True)
  p.write_bytes(data)
  return p


@pytest.mark.parametrize('name', ['af3.bin', 'af3.bin.zst', 'AF3.BIN.ZST', 'params.bin', 'w.bin.zst'])
def test_guard_refuses_weights_names(tmp_path, name):
  with pytest.raises(release.Refused, match='name of the model weights'):
    release.guard(_file(tmp_path, name), f'af3/S/{name}')


def test_guard_refuses_weights_content_and_size(tmp_path, monkeypatch):
  p = _file(tmp_path, 'x_model.cif', b'not really a model')
  monkeypatch.setattr(release, 'WEIGHTS_SHA256', hashlib.sha256(p.read_bytes()).hexdigest())
  with pytest.raises(release.Refused, match='SHA-256 of the model weights'):
    release.guard(p, 'af3/S/x_model.cif')
  monkeypatch.setattr(release, 'WEIGHTS_SHA256', '0' * 64)
  monkeypatch.setattr(release, 'WEIGHTS_BYTES', p.stat().st_size)
  with pytest.raises(release.Refused, match='size of the model weights'):
    release.guard(p, 'af3/S/x_model.cif')


def test_guard_refuses_files_over_the_limit(tmp_path, monkeypatch):
  monkeypatch.setattr(release, 'MAX_MEMBER_BYTES', 5)
  with pytest.raises(release.Refused, match='over 1 GiB'):
    release.guard(_file(tmp_path, 'big_confidences.json'), 'af3/S/big_confidences.json')


def test_guard_refuses_other_buckets_and_allows_provenance(tmp_path):
  ok = _file(tmp_path, 'session.json', b'{"weights": {"file": "af3.bin.zst", "sha256": "' + release.WEIGHTS_SHA256.encode()
             + b'"}, "upload": "gs://af2-tpu-benchmark-af3-results/af3/S"}')
  size, sha = release.guard(ok, 'af3/S/session.json')
  assert size == ok.stat().st_size and sha == hashlib.sha256(ok.read_bytes()).hexdigest()
  bad = _file(tmp_path, 'launch_record.txt', b'WEIGHTS_GCS_URI=gs://private-weights/af3.bin.zst\n')
  with pytest.raises(release.Refused, match='private-weights'):
    release.guard(bad, 'af3/S/launch_record.txt')


INPUT = {'name': '7U3J', 'dialect': 'alphafold3', 'version': 1, 'modelSeeds': [1, 2],
         'sequences': [{'protein': {'id': 'A', 'sequence': 'MKV', 'unpairedMsa': '>q\nMKV\n', 'pairedMsa': '',
                                    'templates': []}}, {'ligand': {'id': 'B', 'ccdCodes': ['ATP']}}]}


def _data_json(seed=1, **change):
  d = json.loads(json.dumps(INPUT))
  d.update(version=4, modelSeeds=[seed])
  d['sequences'][0]['protein']['modifications'] = []
  for k, v in change.items():
    d['sequences'][0]['protein'][k] = v
  return d


def _session(root, name='S_l4_main_g_01', weights=None, data=None):
  r = root / name / 'l4_xla_rep1_seed1_fresh'
  s = r / '7U3J'
  _file(s, 'af3_output/7U3J/seed-1_sample-0/7U3J_seed-1_sample-0_model.cif', b'data_\nATOM 1\n')
  _file(s, 'af3_output/7U3J/TERMS_OF_USE.md', b'# ALPHAFOLD 3 OUTPUT TERMS OF USE\n')
  _file(s, 'af3_output/7U3J/7U3J_data.json', json.dumps(data or _data_json()).encode())
  _file(s, 'run.json', b'{"exit_code": 0}')
  inp = root.parent / 'repo' / 'data' / 'inputs' / '7U3J.json'
  _file(inp.parent, inp.name, json.dumps(INPUT).encode())
  _file(r, 'runs.jsonl', json.dumps({'pdb_id': '7U3J', 'exit_code': 0, 'seeds': [1], 'input_path': 'data/inputs/7U3J.json',
                                     'input_sha256': hashlib.sha256(inp.read_bytes()).hexdigest()}).encode() + b'\n')
  _file(r, 'session.json', json.dumps({'weights': {'file': 'af3.bin.zst', 'sha256': weights or release.WEIGHTS_SHA256}}).encode())
  return s


def _args(root, out, **kw):
  return types.SimpleNamespace(root=str(root), repo=str(root.parent / 'repo'), out=str(out), archives=kw.get('archives'),
                               sessions=kw.get('sessions'), allow_incomplete_inputs_check=False,
                               paper_ref=kw.get('paper_ref', 'none'), code_ref=kw.get('code_ref', 'none'))


def _passed_check(root, out):
  assert release.check_inputs(types.SimpleNamespace(root=str(root), repo=str(root.parent / 'repo'), out=str(out),
                                                    sessions=None)) == 0


def test_build_writes_terms_and_verifies(tmp_path, monkeypatch):
  monkeypatch.setattr(release, 'freeze_sessions', lambda: {'S_l4_main_g_01': 'main'})
  monkeypatch.setattr(release, 'unexplained_files', lambda: {})
  _session(tmp_path / 'af3')
  out = tmp_path / 'out'
  _passed_check(tmp_path / 'af3', out)
  assert release.build(_args(tmp_path / 'af3', out, archives='af3_main_structures.tar.gz,session_records.tar.gz')) == 0
  man = (out / 'R2' / 'af3_main_structures.tar.gz.sha256').read_text()
  assert 'LEGALLY_BINDING_TERMS_OF_USE.txt' in man and 'OUTPUT_TERMS_OF_USE.md' in man and '_data.json' not in man
  assert 'af3/S_l4_main_g_01/l4_xla_rep1_seed1_fresh/7U3J/af3_output/7U3J/seed-1_sample-0/7U3J_seed-1_sample-0_model.cif' in man
  assert 'run.json' in (out / 'R1' / 'session_records.tar.gz.sha256').read_text()
  assert release.verify(types.SimpleNamespace(out=str(out))) == 0
  first = (out / 'R2' / 'SHA256SUMS').read_bytes()
  _passed_check(tmp_path / 'af3', tmp_path / 'out2')
  assert release.build(_args(tmp_path / 'af3', tmp_path / 'out2', archives='af3_main_structures.tar.gz')) == 0
  assert (tmp_path / 'out2' / 'R2' / 'SHA256SUMS').read_bytes() == first   # deterministic


def test_refused_build_keeps_nothing(tmp_path, monkeypatch):
  monkeypatch.setattr(release, 'freeze_sessions', lambda: {'S_l4_main_g_01': 'main'})
  monkeypatch.setattr(release, 'unexplained_files', lambda: {})
  s = _session(tmp_path / 'af3')
  _file(s, 'af3.bin', b'pretend weights')   # a records-category file by its name: selected, then refused
  out = tmp_path / 'out'
  _passed_check(tmp_path / 'af3', out)
  assert release.build(_args(tmp_path / 'af3', out)) == 2
  assert not [p for p in out.rglob('*') if p.is_file() and not p.name.startswith('inputs_check')]   # the check's own report stays


def test_unexpected_large_output_is_refused(tmp_path, monkeypatch):
  monkeypatch.setattr(release, 'freeze_sessions', lambda: {'S_l4_main_g_01': 'main'})
  monkeypatch.setattr(release, 'unexplained_files', lambda: {})
  _session(tmp_path / 'af3')
  out = tmp_path / 'out'
  _file(out / 'R2', 'stray.bin.zst', b'y' * 100)
  _passed_check(tmp_path / 'af3', out)
  monkeypatch.setattr(release, 'MAX_MEMBER_BYTES', 50)
  assert release.build(_args(tmp_path / 'af3', out, archives='af3_logs.tar.gz,session_records.tar.gz')) == 2
  assert not (out / 'R1' / 'session_records.tar.gz').exists()


def test_excluded_unexplained_files_are_not_packaged(tmp_path, monkeypatch):
  monkeypatch.setattr(release, 'freeze_sessions', lambda: {'S_l4_main_g_01': 'main'})
  monkeypatch.setattr(release, 'unexplained_files', lambda: {('S_l4_main_g_01', 'l4_xla_rep1_seed1_fresh/7U3J/run.json'): 'bucket only'})
  _session(tmp_path / 'af3')
  out = tmp_path / 'out'
  assert release.build(_args(tmp_path / 'af3', out, archives='session_records.tar.gz')) == 0
  assert '7U3J/run.json' not in (out / 'R1' / 'session_records.tar.gz.sha256').read_text()


def test_classify_sessions():
  freeze = {'S_main': 'main', 'S_excl': 'pilot2, excluded'}
  assert release.classify('S_main', ['r/runs.jsonl'], freeze) == ('main', 'data freeze')
  assert release.classify('S_excl', ['launch_record.txt'], freeze)[0] == 'attempt'     # never ran: R1 records
  assert release.classify('20261013T111330Z_l4_gt_smoke', ['r/runs.jsonl'], freeze)[0] == 'superseded'
  assert release.classify('20261008T205351Z_v5e_bisect', ['r/runs.jsonl'], freeze)[0] == 'out'
  assert release.classify('20261012T052814Z_l4_pilot_l4d_2', ['r/runs.jsonl'], freeze)[0] == 'superseded'
  assert set(release.SUPERSEDED) >= {'20261009T154308Z_v5e_pilot', '20261010T215557Z_l4_diag_l4_determinism'}


def test_superseded_archive_maps_rows_and_needs_official_weights(tmp_path, monkeypatch):
  monkeypatch.setattr(release, 'freeze_sessions', lambda: {})
  monkeypatch.setattr(release, 'unexplained_files', lambda: {})
  _session(tmp_path / 'af3', name='20261013T111330Z_l4_gt_smoke')
  out = tmp_path / 'out'
  _passed_check(tmp_path / 'af3', out)
  assert release.build(_args(tmp_path / 'af3', out, archives='af3_superseded_engineering.tar.gz')) == 0
  import tarfile
  with tarfile.open(out / 'R2' / 'af3_superseded_engineering.tar.gz') as t:
    table = t.extractfile('SUPERSEDED_SESSIONS.csv').read().decode()
  assert '20261013T111330Z_l4_gt_smoke,"superseded or engineering, not used in any reported result","2026-10-12 and 2026-10-13, GT smoke' in table
  (tmp_path / 'af3' / '20261013T111330Z_l4_gt_smoke' / 'l4_xla_rep1_seed1_fresh' / 'session.json').write_text(
      json.dumps({'weights': {'file': 'random_weights.bin.zst', 'sha256': 'c' * 64}}))
  assert release.build(_args(tmp_path / 'af3', tmp_path / 'out2', archives='af3_superseded_engineering.tar.gz')) == 2


def test_attempt_launcher_records_go_to_r1_even_if_unexplained(tmp_path, monkeypatch):
  monkeypatch.setattr(release, 'freeze_sessions', lambda: {'S_v6e_pilot': 'pilot2, excluded'})
  monkeypatch.setattr(release, 'unexplained_files', lambda: {('S_v6e_pilot', 'launch_record.txt'): 'bucket only'})
  _file(tmp_path / 'af3' / 'S_v6e_pilot', 'launch_record.txt', b'SESSION=S_v6e_pilot\n')
  out = tmp_path / 'out'
  assert release.build(_args(tmp_path / 'af3', out, archives='attempt_superseded_records.tar.gz')) == 0
  assert 'af3/S_v6e_pilot/launch_record.txt' in (out / 'R1' / 'attempt_superseded_records.tar.gz.sha256').read_text()


@pytest.mark.parametrize('change,expect', [({}, []), ({'unpairedMsa': '>q\nMKV\n>x\nMKL\n'}, ['unpairedMsa differs']),
                                           ({'sequence': 'MKL'}, ['sequence differs']),
                                           ({'modifications': [{'ptmType': 'SEP', 'ptmPosition': 1}]}, ['modifications differs'])])
def test_compare_data_json(change, expect):
  d = release.compare_data_json(INPUT, _data_json(**change), [1])
  assert [x.split(': ', 1)[1] for x in d] == expect
  assert release.compare_data_json(INPUT, _data_json(seed=3), [3])[0].startswith('modelSeeds')


def test_r2_build_needs_a_passing_inputs_check(tmp_path, monkeypatch):
  monkeypatch.setattr(release, 'freeze_sessions', lambda: {'S_l4_main_g_01': 'main'})
  monkeypatch.setattr(release, 'unexplained_files', lambda: {})
  root, out = tmp_path / 'af3', tmp_path / 'out'
  _session(root, data=_data_json(unpairedMsa='>q\nMKV\n>x\nMKL\n'))
  assert release.build(_args(root, out, archives='af3_main_structures.tar.gz')) == 2          # no check yet
  assert release.check_inputs(types.SimpleNamespace(root=str(root), repo=str(tmp_path / 'repo'), out=str(out),
                                                    sessions=None)) == 2
  assert json.loads((out / 'inputs_check.json').read_text())['status'] == 'differences'
  assert release.build(_args(root, out, archives='af3_main_structures.tar.gz')) == 2          # check failed
  assert not (out / 'R2').exists() or not any((out / 'R2').iterdir())


def test_analysis_outputs_go_to_r2_with_terms_and_lists_to_r1(tmp_path, monkeypatch):
  monkeypatch.setattr(release, 'freeze_sessions', lambda: {'S_l4_main_g_01': 'main'})
  monkeypatch.setattr(release, 'unexplained_files', lambda: {})
  root, repo, out = tmp_path / 'af3', tmp_path / 'repo', tmp_path / 'out'
  _session(root)
  _file(repo / 'results' / 'analysis' / 'main_20261014', 'samples.csv', b'target,ranking_score\n7U3J,0.9\n')
  _file(repo / 'results' / 'analysis' / 'main_20261014', 'sessions_main.txt', b'S_l4_main_g_01\n')
  _file(repo / 'results' / 'paper_assets', 'README.md', b'# Paper assets\n')
  _file(repo / 'results', 'parts_followups.txt', b'l4 harness/plans/ppi/ppi_g_01.yaml\n')
  _file(repo / 'results' / 'af3' / '.many' / '20261014T223755Z', 'summary.txt', b'13 part(s)\n')
  _file(repo / 'results' / 'af3' / '.launcher' / 'S_l4_main_g_01', 'launch_record.txt', b'SESSION=S_l4_main_g_01\n')
  assert {(a[0], a[1]) for a in release.ARCHIVES} >= {('R2', 'analysis_outputs.tar.gz'),
                                                      ('R1', 'session_lists_launch_records.tar.gz')}
  # no check-inputs has run: analysis_outputs (made from the repository) still builds
  assert release.build(_args(root, out, archives='analysis_outputs.tar.gz,session_lists_launch_records.tar.gz')) == 0
  r2 = (out / 'R2' / 'analysis_outputs.tar.gz.sha256').read_text()
  assert 'results/analysis/main_20261014/samples.csv' in r2 and 'results/paper_assets/README.md' in r2
  assert 'LEGALLY_BINDING_TERMS_OF_USE.txt' in r2 and 'OUTPUT_TERMS_OF_USE.md' in r2
  r1 = (out / 'R1' / 'session_lists_launch_records.tar.gz.sha256').read_text()
  for path in ('results/analysis/main_20261014/sessions_main.txt', 'results/parts_followups.txt',
               'results/af3/.many/20261014T223755Z/summary.txt', 'results/af3/.launcher/S_l4_main_g_01/launch_record.txt'):
    assert path in r1
  assert 'samples.csv' not in r1 and not (out / 'R1' / 'analysis_outputs.tar.gz').exists()
  # an R2 archive made from session folders still needs check-inputs
  assert release.build(_args(root, tmp_path / 'out2', archives='af3_main_structures.tar.gz')) == 2


def test_guard_refuses_email_addresses_off_the_list(tmp_path):
  p = _file(tmp_path, 'job.log', b'WARNING: authenticated as someone@example.org which is the active account\n')
  with pytest.raises(release.Refused, match='email address'):
    release.guard(p, 'af3/S/job.log')
  assert release.EMAIL.search(b'pip install pkg @ git+https://host/repo') is None   # not an address


def _wrapper_log(repo, text):
  return _file(repo / 'results' / 'af3' / '.many' / 'W', 'memlimit_v5.log', text)


def test_listed_file_is_packaged_redacted_and_recorded(tmp_path, monkeypatch):
  monkeypatch.setattr(release, 'freeze_sessions', lambda: {})
  monkeypatch.setattr(release, 'unexplained_files', lambda: {})
  monkeypatch.setattr(release, 'REDACT', {'results/af3/.many/W/memlimit_v5.log': 'gcloud account email'})
  root, repo, out = tmp_path / 'af3', tmp_path / 'repo', tmp_path / 'out'
  root.mkdir()
  original = b'This command is authenticated as a.b@example.org which is the active account.\nagain a.b@example.org\n'
  src = _wrapper_log(repo, original)
  assert release.build(_args(root, out, archives='session_lists_launch_records.tar.gz')) == 0
  assert src.read_bytes() == original                                    # the source is never changed
  import tarfile
  with tarfile.open(out / 'R1' / 'session_lists_launch_records.tar.gz') as t:
    released = t.extractfile('results/af3/.many/W/memlimit_v5.log').read()
  assert released == original.replace(b'a.b@example.org', b'<account>') and b'@' not in released
  rows = list(release.csv.DictReader(open(out / 'R1' / 'REDACTIONS.tsv'), delimiter='\t'))
  assert len(rows) == 1 and rows[0]['path'] == 'results/af3/.many/W/memlimit_v5.log'
  assert rows[0]['original_sha256'] == hashlib.sha256(original).hexdigest()
  assert rows[0]['released_sha256'] == hashlib.sha256(released).hexdigest()
  assert rows[0]['replaced'].startswith('2 occurrence(s) of 1 email address(es) replaced by <account>')
  assert 'a.b@example.org' not in (out / 'R1' / 'REDACTIONS.tsv').read_text()
  man = (out / 'R1' / 'session_lists_launch_records.tar.gz.sha256').read_text()
  assert f'{hashlib.sha256(released).hexdigest()}  results/af3/.many/W/memlimit_v5.log' in man
  assert 'REDACTIONS.tsv' in (out / 'R1' / 'SHA256SUMS').read_text()
  assert release.verify(types.SimpleNamespace(out=str(out))) == 0
  first = (out / 'R1' / 'session_lists_launch_records.tar.gz').read_bytes()
  assert release.build(_args(root, tmp_path / 'out2', archives='session_lists_launch_records.tar.gz')) == 0
  assert (tmp_path / 'out2' / 'R1' / 'session_lists_launch_records.tar.gz').read_bytes() == first   # deterministic


def test_stale_redaction_entry_and_refusal_restore_redactions(tmp_path, monkeypatch):
  monkeypatch.setattr(release, 'freeze_sessions', lambda: {})
  monkeypatch.setattr(release, 'unexplained_files', lambda: {})
  monkeypatch.setattr(release, 'REDACT', {'results/af3/.many/W/memlimit_v5.log': 'gcloud account email'})
  root, repo, out = tmp_path / 'af3', tmp_path / 'repo', tmp_path / 'out'
  root.mkdir()
  _wrapper_log(repo, b'no address here\n')
  assert release.build(_args(root, out, archives='session_lists_launch_records.tar.gz')) == 2   # stale entry
  _wrapper_log(repo, b'authenticated as a.b@example.org\n')
  assert release.build(_args(root, out, archives='session_lists_launch_records.tar.gz')) == 0
  before = (out / 'R1' / 'REDACTIONS.tsv').read_bytes()
  _file(repo / 'results' / 'af3' / '.many' / 'W', 'other.log', b'mail me: c@example.org\n')   # not on the list
  assert release.build(_args(root, out, archives='session_lists_launch_records.tar.gz')) == 2
  assert (out / 'R1' / 'REDACTIONS.tsv').read_bytes() == before                             # restored, not lost


# ---------------------------------------------------------------- the whole study (decisions of 2026-11-08)

def _old_session(root, name, kind_files=True):
  """A session folder of any kind with one record, one model, one full confidence file, one log."""
  r = root / name / 'run1' / 'T1'
  _file(r, 'run.json', b'{"exit_code": 0}')
  _file(r, 'run_alphafold.log', b'I log line\n')
  _file(r, 'af3_output/T1/T1_model.cif', b'data_\nATOM 1\n')
  _file(r, 'af3_output/T1/T1_confidences.json', b'{"pae": [[0.1]]}')
  _file(r, 'af3_output/T1/TERMS_OF_USE.md', b'# TERMS\n')
  _file(root / name, 'job.log', b'job\n')


# The archives of 2026-10-15, as they were: session archives by (kinds, categories).
ARCHIVES_20261015 = {
    'session_records.tar.gz': (['main', 'pilot2', 'memlimit', 'ppi', 'embed'], ['records']),
    'attempt_superseded_records.tar.gz': (['superseded', 'attempt'], ['records']),
    'af3_main_structures.tar.gz': (['main'], ['model_cif', 'ranking_summary', 'terms']),
    'af3_main_full_confidences.tar.gz': (['main'], ['full_confidences']),
    'af3_pilot2_structures.tar.gz': (['pilot2'], ['model_cif', 'ranking_summary', 'terms']),
    'af3_pilot2_full_confidences.tar.gz': (['pilot2'], ['full_confidences']),
    'af3_memlimit.tar.gz': (['memlimit'], ['model_cif', 'ranking_summary', 'terms', 'full_confidences']),
    'af3_ppi_structures.tar.gz': (['ppi'], ['model_cif', 'ranking_summary', 'terms']),
    'af3_ppi_full_confidences.tar.gz': (['ppi'], ['full_confidences']),
    'af3_embed_structures.tar.gz': (['embed'], ['model_cif', 'ranking_summary', 'terms', 'full_confidences']),
    'af3_embed_embeddings.tar': (['embed'], ['embeddings']),
    'af3_logs.tar.gz': (['main', 'pilot2', 'memlimit', 'ppi', 'embed'], ['af3_logs']),
    'af3_superseded_engineering.tar.gz': (['superseded'], ['model_cif', 'ranking_summary', 'terms', 'full_confidences', 'af3_logs', 'embeddings']),
}


def test_archives_of_20261015_keep_their_definition_and_members(tmp_path, monkeypatch):
  specs = {a[1]: a for a in release.ARCHIVES}
  for name, (kinds, cats) in ARCHIVES_20261015.items():
    assert (specs[name][2], specs[name][3]) == (kinds, cats)
  old = {'S_main': 'main', 'S_pilot': 'pilot2', 'S_mem': 'memlimit', 'S_ppi': 'ppi', 'S_embed': 'embed'}
  new = {f'S_{k}': k for k in release.FOLLOWUP_KINDS}
  root = tmp_path / 'af3'
  for s in list(old) + list(new):
    _old_session(root, s)
  kinds_of = dict(old)
  before = {n: release.members(specs[n], root, tmp_path, None, kinds_of, {}) for n in ARCHIVES_20261015}
  kinds_of.update(new)
  after = {n: release.members(specs[n], root, tmp_path, None, kinds_of, {}) for n in ARCHIVES_20261015}
  assert before == after                     # the new kinds add no member to an archive of 2026-10-15
  assert any(before.values())
  for n, m in before.items():
    assert not [x for x, _ in m if any(f'/S_{k}/' in x for k in release.FOLLOWUP_KINDS)], n


def test_freeze_archive_keeps_the_files_of_20261015_and_takes_every_folder(tmp_path):
  repo = tmp_path / 'repo'
  for f in ('manifest_sha256.tsv', 'sessions.tsv', 'bucket_objects.tsv', 'comparison.tsv', 'summary.md', 'INDEX.md',
            'followups34_20261018/summary.md', 'feat_20261018/manifest_sha256.tsv', 'followups612_20261108/manifest_sha256.tsv',
            'msa_naspaz8-af3_run1/files_sha256.tsv', 'msa_naspaz6-af3_run1/files_sha256.tsv', 'msa_naspaz8-af3_run2/summary.md'):
    _file(repo / 'results' / 'freeze', f)
  for f in ('make_freeze_manifest.py', 'reproduce.sh', 'reproduce_followups612.sh', 'freeze_index.py'):
    _file(repo / 'scripts' / 'freeze', f)
  spec = next(a for a in release.ARCHIVES if a[1] == 'freeze.tar.gz')
  got = {m for m, _ in release.members(spec, tmp_path, repo, None, {}, {})}
  assert got == {f'results/freeze/{f}' for f in (
      'manifest_sha256.tsv', 'sessions.tsv', 'bucket_objects.tsv', 'comparison.tsv', 'summary.md', 'INDEX.md',
      'followups34_20261018/summary.md', 'feat_20261018/manifest_sha256.tsv', 'followups612_20261108/manifest_sha256.tsv',
      'msa_naspaz8-af3_run1/files_sha256.tsv', 'msa_naspaz6-af3_run1/files_sha256.tsv', 'msa_naspaz8-af3_run2/summary.md')} | {
      f'scripts/freeze/{f}' for f in ('make_freeze_manifest.py', 'reproduce.sh', 'reproduce_followups612.sh', 'freeze_index.py')}


def test_globs_that_missed_files(tmp_path):
  repo = tmp_path / 'repo'
  _file(repo / 'scripts' / 'analysis_runs' / 'large', 'sessions.txt', b'S\n')
  _file(repo / 'results' / 'paper_assets_fu678' / 'tables', 'T1.tex', b't\n')
  _file(repo / 'results' / 'paper_assets_fu10_11_12', 'README.md', b'r\n')
  _file(repo / 'results' / 'paper_assets', 'README.md', b'r\n')
  specs = {a[1]: a for a in release.ARCHIVES}
  lists = {m for m, _ in release.members(specs['session_lists_launch_records.tar.gz'], tmp_path, repo, None, {}, {})}
  outs = {m for m, _ in release.members(specs['analysis_outputs.tar.gz'], tmp_path, repo, None, {}, {})}
  assert 'scripts/analysis_runs/large/sessions.txt' in lists
  assert {'results/paper_assets_fu678/tables/T1.tex', 'results/paper_assets_fu10_11_12/README.md', 'results/paper_assets/README.md'} <= outs


def test_r1_has_the_new_inputs_and_manifests(tmp_path):
  repo = tmp_path / 'repo'
  wanted = ['data/inputs_repl/A.json', 'data/inputs_large/B.json', 'inputs/official/manifest.csv', 'inputs/repl/manifest.csv',
            'inputs/large/manifest.csv', 'targets/repl/repl_subset.csv', 'targets/large/large_subset.csv']
  for f in wanted:
    _file(repo, f)
  got = {m for spec in release.ARCHIVES if spec[0] == 'R1' and spec[2] == 'repo'
         for m, _ in release.members(spec, tmp_path, repo, None, {}, {})}
  assert set(wanted) <= got


PIPELINE_ARCHIVES = ['inputs_official_pipeline.tar.gz', *[f'nas_{d}.tar.gz' for d in release.NAS_DELIVERIES]]


def test_pipeline_results_are_in_r2_and_nowhere_in_r1(tmp_path):
  repo = tmp_path / 'repo'
  pipeline = ['data/inputs_official/C.json', 'data/inputs_official_data/C.json', 'data/inputs_official_data/smoke/2PV7.json',
              'data/inputs_official.tar.gz', 'inputs/official_data/manifest.csv', 'inputs/official_data/manifest.csv.sha256',
              'inputs/official_data/smoke_manifest.csv']
  deliveries = [f'data/nas/{d}/output/T/T/T_data.json' for d in release.NAS_DELIVERIES]
  for f in pipeline + deliveries:
    _file(repo, f)
  r1 = {m for spec in release.ARCHIVES if spec[0] == 'R1' and spec[2] == 'repo' for m, _ in release.members(spec, tmp_path, repo, None, {}, {})}
  assert not r1
  specs = {a[1]: a for a in release.ARCHIVES}
  assert {specs[a][0] for a in PIPELINE_ARCHIVES} == {'R2'}
  assert {m for m, _ in release.members(specs['inputs_official_pipeline.tar.gz'], tmp_path, repo, None, {}, {})} == set(pipeline)


def test_exclusion_leaves_the_rest_of_the_inputs_in_r1_and_layout_agrees(tmp_path):
  repo = tmp_path / 'repo'
  for f in ('inputs/manifest.csv', 'inputs/official/manifest.csv', 'inputs/official_data/manifest.csv'):
    _file(repo, f)
  spec = next(a for a in release.ARCHIVES if a[1] == 'inputs_targets_manifests.tar.gz')
  got = {m for m, _ in release.members(spec, tmp_path, repo, None, {}, {})}
  assert got == {'inputs/manifest.csv', 'inputs/official/manifest.csv'}
  assert {str(p.relative_to(repo)) for p in release.archive_repo_files(spec[1], spec[3], repo)} == got


@pytest.mark.parametrize('name', PIPELINE_ARCHIVES)
def test_moved_archives_carry_the_notice_and_the_terms(tmp_path, monkeypatch, name):
  import tarfile
  repo = tmp_path / 'repo'
  monkeypatch.setattr(release, 'freeze_sessions', lambda: {})
  monkeypatch.setattr(release, 'unexplained_files', lambda: {})
  root = repo / 'results' / 'af3'
  _file(root / 'S', 'TERMS_OF_USE.md', b'# ALPHAFOLD 3 OUTPUT TERMS OF USE\n')
  _file(repo, 'data/inputs_official_data/C.json', b'{}')
  for d in release.NAS_DELIVERIES:
    _file(repo, f'data/nas/{d}/results.csv', b'a,b\n')
  out = tmp_path / 'out'
  args = _args(root, out, archives=name)
  args.repo = str(repo)
  assert release.build(args) == 0
  assert not (out / 'R1').exists()
  with tarfile.open(out / 'R2' / name) as t:
    assert t.extractfile('LEGALLY_BINDING_TERMS_OF_USE.txt').read().decode() == release.LEGALLY_BINDING
    assert t.extractfile('OUTPUT_TERMS_OF_USE.md').read() == b'# ALPHAFOLD 3 OUTPUT TERMS OF USE\n'


def test_legally_binding_file_says_what_the_terms_require():
  flat = ' '.join(release.LEGALLY_BINDING.split())
  assert 'By using this information, you agree to AlphaFold 3 Output Terms of Use found at https://github.com/google-deepmind/alphafold3/blob/main/OUTPUT_TERMS_OF_USE.md.' in flat
  assert 'To request access to the AlphaFold 3 model parameters, follow the process set out at https://github.com/google-deepmind/alphafold3. You may only use these if received directly from Google.' in flat
  assert 'WEIGHTS_TERMS_OF_USE.md' in flat


@pytest.mark.parametrize('rel', ['data/inputs_official_data/C.json', 'data/inputs_official_data/smoke/2PV7.json',
                                 'data/nas/naspaz8-af3_run1/output/T/T/T_data.json'])
def test_pipeline_results_are_never_modified(tmp_path, rel):
  p = _file(tmp_path, 'x.json', b'{"p": "/home/someone/x"}')
  with pytest.raises(release.Refused, match='released unmodified'):
    release.prepare(p, rel, 'inputs_official_pipeline.tar.gz')


def test_nas_archives_hold_the_delivery_and_its_freeze(tmp_path):
  repo = tmp_path / 'repo'
  for d in release.NAS_DELIVERIES:
    _file(repo / 'data' / 'nas' / d, 'results.csv', b'a,b\n')
    _file(repo / 'results' / 'freeze' / f'msa_{d}', 'files_sha256.tsv', b'x\n')
  _file(repo / 'data' / 'nas' / 'logs', 'host.txt', b'not a delivery\n')
  specs = {a[1]: a for a in release.ARCHIVES}
  for d in release.NAS_DELIVERIES:
    got = {m for m, _ in release.members(specs[f'nas_{d}.tar.gz'], tmp_path, repo, None, {}, {})}
    assert got == {f'data/nas/{d}/results.csv', f'results/freeze/msa_{d}/files_sha256.tsv'}


@pytest.mark.parametrize('kind', ['cpu60', 'a100', 'msa', 'sw', 'seeds', 'h100', 'repl', 'prec5', 'large'])
def test_new_kinds_have_their_archives_and_records(tmp_path, kind):
  assert kind in release.RELEASED_KINDS
  specs = {a[1]: a for a in release.ARCHIVES}
  root = tmp_path / 'af3'
  _old_session(root, f'S_{kind}')
  kinds_of = {f'S_{kind}': kind}
  assert [m for m, _ in release.members(specs[f'af3_{kind}_structures.tar.gz'], root, tmp_path, None, kinds_of, {})
          if m.endswith('_model.cif')]
  assert [m for m, _ in release.members(specs[f'af3_{kind}_full_confidences.tar.gz'], root, tmp_path, None, kinds_of, {})]
  assert [m for m, _ in release.members(specs['session_records_followups.tar.gz'], root, tmp_path, None, kinds_of, {})
          if m.endswith('/job.log')]
  assert [m for m, _ in release.members(specs['af3_logs_followups.tar.gz'], root, tmp_path, None, kinds_of, {})]


def test_feat_session_is_a_freeze_session_with_features(tmp_path):
  assert release.classify('20261018T161643Z_cpu_feat_milan', ['plan_runs.jsonl', 'job.log'], {'20261018T161643Z_cpu_feat_milan': 'feat'})[0] == 'feat'
  assert release.category('featurise/7T6A/arrays/seed-1.npz') == 'features'
  assert release.category('featurise/7T6A/features.jsonl') == 'records'
  specs = {a[1]: a for a in release.ARCHIVES}
  root = tmp_path / 'af3'
  _file(root / 'S_feat' / 'featurise' / '7T6A' / 'arrays', 'seed-1.npz', b'PK\x03\x04')
  _file(root / 'S_feat' / 'featurise' / '7T6A', 'features.jsonl', b'{}\n')
  got = [m for m, _ in release.members(specs['af3_feat_features.tar.gz'], root, tmp_path, None, {'S_feat': 'feat'}, {})]
  assert got == ['af3/S_feat/featurise/7T6A/arrays/seed-1.npz']


def test_layout_of_the_records_stays_within_zenodos_file_limit():
  for rec, plain in release.PLAIN_FILES.items():
    assert 2 * sum(1 for a in release.ARCHIVES if a[0] == rec) + plain <= release.ZENODO_MAX_FILES


# ---------------------------------------------------------------- sessions outside the freeze

@pytest.mark.parametrize('session,row', [
    ('20261026T134950Z_l4_msa_smoke_l4', '2026-10-26, Follow-up 6, smoke'),
    ('20261026T164415Z_l4_msa_go_r1_02', '2026-10-27, Follow-up 6, Go r1'),
    ('20261027T153322Z_v6e_msa_v6o_02', '2026-10-28, Follow-up 6, V6o'),
    ('20261029T111747Z_v5e_eigen_check_v5e', '2026-10-29, Follow-ups 7 to 9, eigen_check'),
    ('20261029T120749Z_l4_sw_smoke_l4', '2026-10-29, Follow-up 7, sw_smoke'),
    ('20261030T041222Z_v5e_seeds_v5x_03', '2026-10-30, Follow-up 8, V5x'),
    ('20261031T110141Z_l4_seeds_gx_04', '2026-11-01, Follow-up 8, Gx'),
    ('20261101T213113Z_l4_sw_gs_04', '2026-11-02, Follow-up 7, Gs'),
    ('20261103T210601Z_h100_h100_smoke_h100', '2026-11-03, Follow-up 9'),
    ('20261104T223023Z_l4_repl_gr_01', '2026-11-05, Follow-up 11, Gr'),
    ('20261105T151041Z_v5e_repl_v5r_03', '2026-11-05, Follow-up 11, V5r'),
    ('20261106T223614Z_l4_large_gl_01', '2026-11-07 and 2026-11-08, Follow-up 10, Gl'),
    ('20261105T224627Z_v5e_large_v5l_01', '2026-11-07, Follow-up 10, V5l'),
    ('20261016T231247Z_a100_a100_smoke', '2026-10-17, Follow-ups 3 and 4'),
    ('20261016T230421Z_cpu_cpu60_7nu0', '2026-10-17, Follow-ups 3 and 4')])
def test_outside_sessions_map_to_their_row(session, row):
  kind, why = release.classify(session, ['r/runs.jsonl'], {})
  assert kind == 'outside' and why.startswith(row)
  assert release.classify(session, ['launch_record.txt'], {})[0] == 'outside'   # with or without run records


def test_a_frozen_session_of_the_same_arm_stays_in_its_kind():
  s = '20261104T223023Z_l4_repl_gr_01'
  assert release.classify(s, ['r/runs.jsonl'], {s: 'repl'})[0] == 'repl'


@pytest.mark.parametrize('session', ['20261201T000000Z_l4_large_gl_01', '20261030T000000Z_v5e_msa_v5o_01',
                                     '20261110T000000Z_v6e_prec5_v6h5_01', '20261110T000000Z_v5e_eigen_check_v5e'])
def test_a_session_no_row_maps_is_unmapped(session):
  assert release.classify(session, ['r/runs.jsonl'], {})[0] == 'unmapped'
  assert release.classify(session, ['launch_record.txt'], {})[0] == 'unmapped'


def test_unmapped_session_stops_the_build_and_the_check_and_is_listed(tmp_path, monkeypatch):
  monkeypatch.setattr(release, 'freeze_sessions', lambda: {})
  monkeypatch.setattr(release, 'unexplained_files', lambda: {})
  root, out = tmp_path / 'af3', tmp_path / 'out'
  name = '20261201T000000Z_l4_large_gl_01'
  _old_session(root, name)
  (tmp_path / 'repo').mkdir()
  assert release.build(_args(root, out, archives='outside_freeze_records.tar.gz')) == 2
  assert not (out / 'R1').exists()
  assert release.check_inputs(types.SimpleNamespace(root=str(root), repo=str(tmp_path / 'repo'), out=str(out), sessions=None)) == 2
  inv = types.SimpleNamespace(root=str(root), listing=None, bucket_objects=None, launcher=None, out=str(tmp_path / 'inv'))
  assert release.inventory(inv) == 2
  assert name in (tmp_path / 'inv' / 'unmapped_sessions.csv').read_text()


def test_outside_archive_holds_records_only_and_its_table(tmp_path, monkeypatch):
  monkeypatch.setattr(release, 'freeze_sessions', lambda: {})
  monkeypatch.setattr(release, 'unexplained_files', lambda: {})
  root, out = tmp_path / 'af3', tmp_path / 'out'
  name = '20261104T223023Z_l4_repl_gr_01'
  _old_session(root, name)
  (tmp_path / 'repo').mkdir()
  assert release.build(_args(root, out, archives='outside_freeze_records.tar.gz')) == 0
  import tarfile
  with tarfile.open(out / 'R1' / 'outside_freeze_records.tar.gz') as t:
    names = t.getnames()
    table = t.extractfile('OUTSIDE_SESSIONS.csv').read().decode()
  assert f'af3/{name}/job.log' in names and f'af3/{name}/run1/T1/run.json' in names
  assert not [n for n in names if n.endswith(('_model.cif', '_confidences.json', 'TERMS_OF_USE.md', 'run_alphafold.log'))]
  assert f'{name},"{release.OUTSIDE_LABEL}","2026-11-05, Follow-up 11, Gr' in table


def test_outside_mapping_covers_the_local_launcher_records():
  lau = release.REPO / 'results' / 'af3' / '.launcher'
  if not lau.is_dir() or not (release.FREEZE / 'sessions.tsv').is_file():
    pytest.skip('no local launcher records')
  frozen = release.freeze_sessions()
  kinds = [release.classify(d.name, ['launch_record.txt'], frozen)[0] for d in lau.iterdir() if d.is_dir()]
  assert 'unmapped' not in kinds and kinds.count('outside') == 384


def test_row_explained_launcher_records_are_released(monkeypatch, tmp_path):
  f = tmp_path / 'comparison.tsv'
  f.write_text('session\tpath\tstatus\tnote\n20261031T204043Z_l4_seeds_gx_04\tlaunch_record.txt\tlocal only\t\n'
               'S\tjob.log\tMISMATCH\tlocal 1 B\nS\t.uploaded\tlocal only\texpected: written after\n')
  monkeypatch.setattr(release, 'FREEZE', tmp_path)
  assert release.unexplained_files() == {('S', 'job.log'): 'MISMATCH'}
  assert len(release.ROW_EXPLAINED) == 18


# ---------------------------------------------------------------- the seed allowance of follow-up 8

def test_seeds_beyond_the_input_only_for_follow_up_8():
  dj = _data_json(seed=7)
  assert release.compare_data_json(INPUT, dj, [7])[0].startswith('modelSeeds')                                 # no general relaxation
  assert release.compare_data_json(INPUT, dj, [7], release.SEEDS_BEYOND_INPUT['seeds']) == []                  # follow-up 8
  assert release.compare_data_json(INPUT, _data_json(seed=21), [21], release.SEEDS_BEYOND_INPUT['seeds'])      # beyond its seeds
  assert release.compare_data_json(INPUT, _data_json(seed=8), [7], release.SEEDS_BEYOND_INPUT['seeds'])        # not the process's seed
  assert set(release.SEEDS_BEYOND_INPUT) == {'seeds'} and release.SEEDS_BEYOND_INPUT['seeds'] == frozenset(range(6, 21))


@pytest.mark.parametrize('kind,status', [('seeds', 'pass'), ('prec5', 'differences'), ('main', 'differences')])
def test_check_inputs_gives_the_allowance_to_the_seeds_kind_only(tmp_path, monkeypatch, kind, status):
  name = f'S_{kind}'
  monkeypatch.setattr(release, 'freeze_sessions', lambda: {name: kind})
  monkeypatch.setattr(release, 'unexplained_files', lambda: {})
  root, repo = tmp_path / 'af3', tmp_path / 'repo'
  _session(root, name=name, data=_data_json(seed=7))
  runs = root / name / 'l4_xla_rep1_seed1_fresh' / 'runs.jsonl'
  r = json.loads(runs.read_text())
  r['seeds'] = [7]
  runs.write_text(json.dumps(r) + '\n')
  rc = release.check_inputs(types.SimpleNamespace(root=str(root), repo=str(repo), out=str(tmp_path / 'out'), sessions=None))
  assert json.loads((tmp_path / 'out' / 'inputs_check.json').read_text())['status'] == status
  assert rc == (0 if status == 'pass' else 2)


# ---------------------------------------------------------------- privacy

CONTACT = b'someone.real@example.org'


@pytest.mark.parametrize('text,cls,replacement', [
    (b'host 10.164.0.12:8470 up', 'private_ipv4', b'<private-ip>'),
    (b'gateway 192.168.1.40', 'private_ipv4', b'<private-ip>'),
    (b'peer 172.20.3.4 ok', 'private_ipv4', b'<private-ip>'),
    (b'tailnet 100.101.2.3', 'private_ipv4', b'<private-ip>'),
    (b'link 169.254.169.254', 'private_ipv4', b'<private-ip>'),
    (b'ping nas.local now', 'local_host', b'<private-host>'),
    (b'ssh truenas.lan', 'local_host', b'<private-host>'),
    (b'ssh root@naspaz8 ok', 'nas_host', b'<nas-host>'),
    (b'host naspaz6.', 'nas_host', b'<nas-host>'),
    (b'cd /home/someone/repo', 'home_path', b'/home/<user>/repo'),
    (b'open /Users/someone/Desktop/x', 'home_path', b'/Users/<user>/Desktop/x')])
def test_private_network_data_is_redacted_with_a_row(tmp_path, text, cls, replacement):
  p = _file(tmp_path, 'job.log', text + b'\n')
  size, sha, data, row = release.prepare(p, 'results/af3/.many/W/job.log', 'A.tar.gz')
  assert p.read_bytes() == text + b'\n'                             # the source is never changed
  assert replacement in data and row['original_sha256'] == hashlib.sha256(text + b'\n').hexdigest()
  assert row['released_sha256'] == sha == hashlib.sha256(data).hexdigest() and size == len(data)
  assert cls in row['replaced'] and text.split()[-1] not in row['replaced'].encode()
  assert not release.private_findings(data)


@pytest.mark.parametrize('text', [
    b'version 12.5.2.1', b'libtpu 0.0.43.2', b'8.8.8.8', b'172.32.0.1', b'172.15.0.1', b'100.63.0.1', b'192.169.0.1',
    b'0.8 10.5 10.25,10.1', b'[10.1, 10.2, 10.3, 10.4]', b'/usr/local/lib/python3.12', b'/home/<user>/repo', b'~/.local/bin',
    b'threading.local()', b'jax.local_devices', b'ANDREA', b'num_tokens: 1234567890123456789',
    b'naspaz8-af3_run1', b'/data/runs/naspaz6-af3_run1/output', b'host=naspaz8-af3 kernel', b'naspaz6_identical,naspaz6_difference',
    b'"unpairedMsa": ">q\\nMKVsk-AAAAAAAAAAAAAAAAAAAAAAAAAA\\n"', b'AAKIAAAAAAAAAAAAAAAAAAAA', b'sha256=abcdef0123456789',
    b'SMILES C[C@H]1CC[C@@H](O)C1', b'https://github.com/maintainer/alphafold-on-tpu'])
def test_things_that_are_not_private_network_data(text):
  assert release.private_findings(text) == {} and release.EMAIL.search(text) is None


@pytest.mark.parametrize('text,cls', [
    (b'PrivateKey = ' + b'A' * 42 + b'E=\n', 'wireguard_key'),
    (b'ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIabcdefghijklmnop key', 'ssh_key'),
    (b'-----BEGIN OPENSSH PRIVATE KEY-----', 'ssh_key'),
    (b'token ghp_abcdefghijklmnopqrstuvwxyz0123', 'api_token'),
    (b'key AIza' + b'B' * 35, 'api_token'),
    (b'auth ya29.abcdefghijklmnopqrstuvwxyz', 'api_token'),
    (b'aws AKIAABCDEFGHIJKLMNOP', 'api_token'),
    (b'api_key=abcdef0123456789abcdef', 'api_token'),
    (b'Authorization: Bearer abcdef0123456789abcdef', 'api_token'),
    (b'ZENODO_TOKEN password: abcdef0123456789', 'api_token')])
def test_secrets_are_refused(tmp_path, text, cls):
  p = _file(tmp_path, 'job.log', text)
  with pytest.raises(release.Refused, match=cls):
    release.guard(p, 'results/af3/.many/W/job.log')
  with pytest.raises(release.Refused, match=cls):
    release.prepare(p, 'results/af3/.many/W/job.log', 'A.tar.gz')


def test_user_agent_is_redacted_on_the_list(tmp_path, monkeypatch):
  rel = 'data/msa/abc.json'
  monkeypatch.setattr(release, 'REDACT', {rel: release.REDACT_REASONS[0][1]})
  original = json.dumps({'server': 'https://api.colabfold.com', 'user_agent': f'alphafold-on-tpu/0.1 (mailto:{CONTACT.decode()})',
                         'sequence': 'MKV', 'mode': 'env'}).encode()
  p = _file(tmp_path / 'msa', 'abc.json', original)
  size, sha, data, row = release.prepare(p, rel, 'msa_provenance.tar.gz')
  d = json.loads(data)
  assert d['user_agent'] == 'alphafold-on-tpu/0.1 (mailto:<account>)' and d['sequence'] == 'MKV'
  assert CONTACT not in data and p.read_bytes() == original
  assert row['original_sha256'] == hashlib.sha256(original).hexdigest() and row['released_sha256'] == hashlib.sha256(data).hexdigest()
  assert row['replaced'].startswith('1 occurrence(s) of 1 email address(es) replaced by <account>') and 'user_agent' in row['replaced']
  assert CONTACT.decode() not in row['replaced']


def test_user_agent_off_the_list_is_refused(tmp_path):
  p = _file(tmp_path, 'abc.json', json.dumps({'user_agent': f'x (mailto:{CONTACT.decode()})'}).encode())
  with pytest.raises(release.Refused, match='email address'):
    release.guard(p, 'data/msa/never_listed.json')


def test_redact_list_reasons_and_paths(tmp_path):
  f = tmp_path / 'list.txt'
  f.write_text('data/msa/a.json\nresults/af3/.many/W/x.log\n')
  got = release.load_redact_list(f)
  assert 'user_agent' in got['data/msa/a.json'] and 'authenticated as' in got['results/af3/.many/W/x.log']
  f.write_text('some/other/place.txt\n')
  with pytest.raises(StopIteration):
    release.load_redact_list(f)                 # a path of a kind no reason covers is not silently listed


def test_contact_address_only_once_and_only_in_a_readme(tmp_path, monkeypatch):
  monkeypatch.setattr(release, 'CONTACT_SHA256', hashlib.sha256(CONTACT).hexdigest())
  once = _file(tmp_path / 'a', 'README.md', b'Contact: ' + CONTACT + b'\n')
  release.guard(once, 'README.md')
  with pytest.raises(release.Refused, match='email address'):
    release.guard(_file(tmp_path / 'b', 'README.md', b'Contact: ' + CONTACT + b', again ' + CONTACT + b'\n'), 'README.md')
  with pytest.raises(release.Refused, match='email address'):
    release.guard(_file(tmp_path / 'c', 'README.md', b'Contact: other@example.org\n'), 'README.md')
  with pytest.raises(release.Refused, match='email address'):
    release.guard(_file(tmp_path / 'd', 'job.log', b'Contact: ' + CONTACT + b'\n'), 'job.log')
  assert release.scan_member(('A', 'README.md', str(once)))[2] == {'email': 1}


def test_alphafold3_outputs_are_never_modified(tmp_path):
  p = _file(tmp_path, 'T_model.cif', b'data_\n# /home/someone/x\n')
  with pytest.raises(release.Refused, match='released unmodified'):
    release.prepare(p, 'af3/S/r/T/af3_output/T/T_model.cif', 'af3_main_structures.tar.gz')
  p = _file(tmp_path, 'run_alphafold.log', b'loading /home/someone/x\n')
  assert b'/home/<user>/x' in release.prepare(p, 'af3/S/r/T/run_alphafold.log', 'af3_logs.tar.gz')[2]


def test_nested_archive_is_scanned(tmp_path):
  import tarfile
  bad = tmp_path / 'bad.tar.gz'
  with tarfile.open(bad, 'w:gz') as t:
    t.add(_file(tmp_path, 'in.txt', b'ssh root@naspaz8\n'), arcname='x/in.txt')
  with pytest.raises(release.Refused, match='nested archive'):
    release.guard(bad, 'data/inputs_official.tar.gz')
  good = tmp_path / 'good.tar.gz'
  with tarfile.open(good, 'w:gz') as t:
    t.add(_file(tmp_path, 'in2.json', b'{"name": "7D5C"}'), arcname='x/in2.json')
  release.guard(good, 'data/inputs_official.tar.gz')


def test_arrays_are_not_scanned_as_text(tmp_path):
  p = _file(tmp_path, 'seed-1.npz', b'a@b.org 10.0.0.1 /home/x')
  assert release.prepare(p, 'af3/S/featurise/T/arrays/seed-1.npz', 'af3_feat_features.tar.gz')[2] is None


def test_plain_files_of_a_record_are_not_redacted_but_refused(tmp_path):
  p = _file(tmp_path, 'inventory.md', b'Source: /home/someone/x\n')
  with pytest.raises(release.Refused, match='plain file'):
    release.plain_guard(p, 'release/inventory.md')


def test_the_release_tooling_passes_its_own_guard():
  for name in ('release.py', 'RUNBOOK.md'):
    release.plain_guard(release.HERE / name, f'release/{name}')
  for name in ('records/LICENCES.md', 'records/R1/README.md', 'records/R2/README.md'):
    release.plain_guard(release.HERE / name, 'README.md')


def test_redacted_build_records_every_class_and_verifies(tmp_path, monkeypatch):
  monkeypatch.setattr(release, 'freeze_sessions', lambda: {'S_large': 'large'})
  monkeypatch.setattr(release, 'unexplained_files', lambda: {})
  root, out = tmp_path / 'af3', tmp_path / 'out'
  _old_session(root, 'S_large')
  _file(root / 'S_large', 'finish.log', b'cd /home/someone/repo; ping 10.1.2.3\n')
  (tmp_path / 'repo').mkdir()
  assert release.build(_args(root, out, archives='session_records_followups.tar.gz')) == 0
  rows = list(release.csv.DictReader(open(out / 'R1' / 'REDACTIONS.tsv'), delimiter='\t'))
  assert [r['path'] for r in rows] == ['af3/S_large/finish.log']
  assert 'home_path' in rows[0]['replaced'] and 'private_ipv4' in rows[0]['replaced'] and 'someone' not in rows[0]['replaced']
  assert release.verify(types.SimpleNamespace(out=str(out))) == 0


def test_a_record_over_the_limits_stops(tmp_path, monkeypatch):
  monkeypatch.setattr(release, 'ZENODO_MAX_FILES', 2)
  d = tmp_path / 'R2'
  for i in range(3):
    _file(d, f'a{i}.txt')
  assert release.finish_or_refuse([d]) == 2


def test_real_freeze_is_the_final_one():
  index = release.REPO / 'results' / 'freeze' / 'INDEX.md'
  if not index.is_file():
    pytest.skip('no local freeze')
  assert f'**Final freeze for the paper:** `{release.FREEZE_ID}`' in index.read_text()
  assert release.sha256_file(release.FREEZE / 'manifest_sha256.tsv') == release.FREEZE_ID


def test_a_record_has_no_folders(tmp_path):
  d = tmp_path / 'R2'
  _file(d / 'release', 'a.txt')
  assert release.finish_or_refuse([d]) == 2


def test_r1_tooling_goes_in_as_flat_files(tmp_path, monkeypatch):
  monkeypatch.setattr(release, 'OUT_DIR', tmp_path / 'results_release')
  _file(tmp_path / 'results_release' / 'inventory', 'OUTSIDE_SESSIONS.csv', b'session,label,section_10_row\n')
  d = tmp_path / 'R1'
  d.mkdir()
  assert release.finish_or_refuse([d]) == 0
  names = {p.name for p in d.iterdir()}
  assert {'README.md', 'LICENCES.md', 'release_release.py', 'release_RUNBOOK.md', 'release_SUPERSEDED_SESSIONS.csv',
          'release_OUTSIDE_SESSIONS.csv', 'SHA256SUMS'} <= names
  assert not [p for p in d.iterdir() if p.is_dir()]


def test_archives_command_splits_by_machine(capsys):
  release.archives(types.SimpleNamespace(built_on='laptop'))
  laptop = set(capsys.readouterr().out.strip().split(','))
  release.archives(types.SimpleNamespace(built_on='vm'))
  vm = set(capsys.readouterr().out.strip().split(','))
  assert laptop | vm == {a[1] for a in release.ARCHIVES} and not laptop & vm
  assert {'freeze.tar.gz', 'msa_provenance.tar.gz', 'nas_naspaz8-af3_run1.tar.gz', 'session_records_followups.tar.gz', 'analysis_outputs.tar.gz'} <= laptop
  assert {'attempt_superseded_records.tar.gz', 'outside_freeze_records.tar.gz', 'af3_large_full_confidences.tar.gz', 'af3_feat_features.tar.gz'} <= vm


def test_merge_redactions_keeps_both_machines_rows(tmp_path):
  d = tmp_path / 'out' / 'R1'
  row = {'archive': 'A', 'path': 'p', 'original_sha256': 'a', 'released_sha256': 'b', 'replaced': 'x'}
  release.write_redactions(d if d.mkdir(parents=True) is None else d, [row])
  other = tmp_path / 'vm' / 'R1'
  other.mkdir(parents=True)
  release.write_redactions(other, [dict(row, archive='B', path='q')])
  assert release.merge_redactions(types.SimpleNamespace(out=str(tmp_path / 'out'), record='R1', **{'from': str(other / 'REDACTIONS.tsv')})) == 0
  assert [r['archive'] for r in release.csv.DictReader(open(d / 'REDACTIONS.tsv'), delimiter='\t')] == ['A', 'B']


def test_vm_steps_script_is_valid_and_copies_outputs_only_for_released_sessions():
  import re
  import subprocess
  sh = release.HERE / 'vm_steps.sh'
  subprocess.run(['bash', '-n', str(sh)], check=True)
  text = sh.read_text()
  pattern = re.search(r"OUTPUTS='(.+)'", text).group(1)
  for name in ('x_model.cif', 'x_confidences.json', 'x_summary_confidences.json', 'x_data.json', 'x_ranking_scores.csv',
               'TERMS_OF_USE.md', 'run_alphafold.log'):
    assert re.fullmatch(pattern, f'r/T/af3_output/T/{name}')
  for name in ('run.json', 'runs.jsonl', 'job.log', 'launch_record.txt', 'session.json'):
    assert not re.fullmatch(pattern, f'r/{name}')


DOI = {'R1': '10.5281/zenodo.24057470', 'R2': '10.5281/zenodo.24057544'}
TAG = 'zenodo-v1'   # the repository tag of this release, moved from the first build's commit (366159b) to the release commit
# The OSF registrations, public since 2026-11-17: the project, the main study and the addenda of follow-ups 10, 11, 13 to 16.
OSF = ['https://osf.io/qtz3v', 'https://osf.io/fegsy', 'https://osf.io/nsbgp', 'https://osf.io/64bpy', 'https://osf.io/c4kh7',
       'https://osf.io/eafkx', 'https://osf.io/h9yw8', 'https://osf.io/hcxm7']
PAPER_POINTER = 'Paper: see the related identifiers of this Zenodo record (arXiv preprint and conference version, added when available).'
# The relations of Zenodo's deposit metadata reference (developers.zenodo.org, read on 2026-11-08); isRelatedTo is not one.
ZENODO_RELATIONS = {'isCitedBy', 'cites', 'isSupplementTo', 'isSupplementedBy', 'isContinuedBy', 'continues', 'isDescribedBy',
                    'describes', 'hasMetadata', 'isMetadataFor', 'isNewVersionOf', 'isPreviousVersionOf', 'isPartOf', 'hasPart',
                    'isReferencedBy', 'references', 'isDocumentedBy', 'documents', 'isCompiledBy', 'compiles', 'isVariantFormOf',
                    'isOriginalFormof', 'isIdenticalTo', 'isAlternateIdentifier', 'isReviewedBy', 'reviews', 'isDerivedFrom',
                    'isSourceOf', 'requires', 'isRequiredBy', 'isObsoletedBy', 'obsoletes'}


def test_runbook_tags_the_build_and_fills_in_no_paper_field():
  text = (release.HERE / 'RUNBOOK.md').read_text()
  flat = ' '.join(text.split())
  assert f'git tag -f -a {TAG} -m "Zenodo release, freeze 4ec185c5, arXiv version"' in text and f'git push origin {TAG}' in text
  assert 'build from that tag' in flat and f'--code_ref {TAG} --paper_ref {TAG}' in text
  assert 'zenodo-4ec185c5' not in text and 'paper-arxiv-v1' not in text
  assert 'TITLE=' not in text and '<paper' not in text and '<arXiv' not in text and 'angle bracket' not in text and 'placeholder' not in text
  assert 'added later as related works ("is supplement to")' in flat and 'no new version needed' in flat


def test_r2_licence_is_custom_and_r1_is_cc_by():
  r1 = json.loads((release.HERE / 'records' / 'R1' / 'zenodo.json').read_text())['metadata']
  r2 = json.loads((release.HERE / 'records' / 'R2' / 'zenodo.json').read_text())['metadata']
  assert r1['license'] == 'cc-by-4.0' and 'rights' not in r1
  assert 'license' not in r2
  assert r2['rights'] == [{'title': {'en': 'AlphaFold 3 Output Terms of Use'},
                           'link': 'https://github.com/google-deepmind/alphafold3/blob/main/OUTPUT_TERMS_OF_USE.md'}]


def test_runbook_never_sends_r2_metadata_through_the_deposit_api():
  # that endpoint takes only licence ids and defaults to CC0 for a dataset: R2's licence is set by hand in the form
  text = (release.HERE / 'RUNBOOK.md').read_text()
  assert '$API/$R1_ID"' in text and '--data @$HOME/repo/scripts/release/records/R1/zenodo.json' in text
  assert 'zenodo_R2.json' not in text and '"$API/$R2_ID"' not in text
  assert 'other-nc' not in text and 'Add custom' in text


@pytest.mark.parametrize('record', ['R1', 'R2'])
def test_zenodo_metadata(record):
  m = json.loads((release.HERE / 'records' / record / 'zenodo.json').read_text())['metadata']
  assert m['upload_type'] == 'dataset' and m['access_right'] == 'open'
  assert m['creators'] == [{'name': 'Pazienza, Lorenzo', 'orcid': '0009-0006-1687-0125'}]
  assert m['contributors'][0]['type'] == 'ContactPerson' and m['contributors'][0]['orcid'] == '0009-0006-1687-0125'
  address = release.EMAIL.findall(m['notes'].encode())
  assert len(address) == 1 and hashlib.sha256(address[0]).hexdigest() == release.CONTACT_SHA256
  assert 'github.com/lorenzopazienza/alphafold-on-tpu' in str(m['related_identifiers'])
  assert 'arXiv' not in str(m['related_identifiers']) and PAPER_POINTER in m['description']   # added on Zenodo when available
  assert not re.search(r'<paper|&lt;paper|<arXiv|<commit', json.dumps(m))
  other = 'R2' if record == 'R1' else 'R1'
  links = {r['identifier']: r['relation'] for r in m['related_identifiers']}
  assert links[DOI[other]] == {'R1': 'isSourceOf', 'R2': 'isDerivedFrom'}[record]
  assert set(links.values()) <= ZENODO_RELATIONS
  assert '<DOI of record' not in str(m) and '&lt;DOI' not in str(m)
  assert release.FREEZE_ID in m['description'] and DOI[other] in m['description'] and '439 sessions' in m['description']
  for url in OSF:
    assert links[url] == 'isDocumentedBy', url


@pytest.mark.parametrize('record', ['R1', 'R2'])
def test_readme_carries_the_decisions(record):
  text = (release.HERE / 'records' / record / 'README.md').read_text()
  for needed in (release.FREEZE_ID, '0009-0006-1687-0125', DOI['R1'], DOI['R2'], 'REDACTIONS.tsv', 'LICENCES.md',
                 '439 sessions', *OSF):
    assert needed in text
  assert '<DOI of record' not in text
  assert PAPER_POINTER in text and f'alphafold-on-tpu, the repository commit tagged {TAG}.' in text
  assert 'DRAFT' not in text and 'not yet published' not in text and 'angle brackets' not in text
  assert not re.search(r'<paper|<arXiv|<commit', text)
  address = release.EMAIL.findall(text.encode())
  assert len(address) == 1 and hashlib.sha256(address[0]).hexdigest() == release.CONTACT_SHA256
  for spec in release.ARCHIVES:
    if spec[0] == record:
      assert f'`{spec[1]}`' in text, spec[1]


def test_r2_readme_gives_the_notice_the_binding_file_and_the_citation():
  text = (release.HERE / 'records' / 'R2' / 'README.md').read_text()
  flat = ' '.join(text.split())
  head = ' '.join(text[:text.index('## What this record contains')].split())
  assert 'By using this information, you agree to AlphaFold 3 Output Terms of Use found at https://github.com/google-deepmind/alphafold3/blob/main/OUTPUT_TERMS_OF_USE.md.' in head
  assert 'LEGALLY_BINDING_TERMS_OF_USE.txt' in head and 'Legally Binding Terms of Use' in head
  assert 'Accurate structure prediction of biomolecular interactions with AlphaFold 3' in head and 'Abramson, J.' in head
  assert 'the structure predictions and all related information provided by AlphaFold 3' in flat


def test_r1_readme_holds_no_pipeline_archive():
  text = (release.HERE / 'records' / 'R1' / 'README.md').read_text()
  for name in PIPELINE_ARCHIVES:
    assert name not in text.split('## What this record contains')[1].split('Data freeze identifier')[0]


def test_nas_table_says_what_each_delivery_recorded():
  text = (release.HERE / 'records' / 'R2' / 'README.md').read_text()
  table = text.split('### Where the follow-up 6 data pipeline ran')[1].split('###')[0]
  assert 'UGREEN DXP8800 Plus' in table and 'UGREEN iDX6011 Pro' in table and 'TrueNAS' in table
  for d in release.NAS_DELIVERIES:
    assert f'`{d}`' in table and f'data/nas/{d}/provenance.txt' in table
    p = release.REPO / 'data' / 'nas' / d / 'provenance.txt'
    if p.is_file():      # the deliveries are local data; synthetic checks of the wording above run without them
      rec = p.read_text()
      assert re.search(r'vcpu=(\d+)', rec).group(1) + ' vCPU' in table
      assert f"{int(re.search(r'mem_total_mib=(\d+)', rec).group(1)):,} MiB" in table
      assert re.search(r'^cpu=(.+)$', rec, re.M).group(1) in table and re.search(r' os=(.+)$', rec, re.M).group(1) in table
  assert not release.private_findings(text.encode())


def test_redaction_note_names_the_class_and_never_the_user(tmp_path):
  p = _file(tmp_path, 'job.log', b'cd /Users/someone/x\n')
  note = release.prepare(p, 'results/af3/.many/W/job.log', 'A')[3]['replaced']
  assert note == '1 home_path replaced by .../<user>'


# ---------------------------------------------------------------- labels after the first build (freeze 4ec185c5, 2026-11-17)

# The archives of the first build (commit 366159b, then tagged zenodo-v1), as uploaded to the drafts on 2026-11-09: name,
# record, kinds or 'repo', categories or patterns.
FIRST_BUILD = {spec[1]: spec for spec in release.ARCHIVES[:len(release.FIRST_BUILD_ARCHIVES)]}


def test_first_build_archives_are_first_and_unchanged():
  assert len(release.FIRST_BUILD_ARCHIVES) == 45
  assert release.FIRST_BUILD_ARCHIVES == [a[1] for a in release.ARCHIVES[:45]]
  assert {n for n in release.FIRST_BUILD_ARCHIVES if n.startswith('af3_')} >= {
      'af3_main_structures.tar.gz', 'af3_large_full_confidences.tar.gz', 'af3_logs_followups.tar.gz', 'af3_feat_features.tar.gz'}
  for name, spec in FIRST_BUILD.items():
    if spec[2] != 'repo':
      assert not set(spec[2]) & set(release.LATER_KINDS + release.OUTSIDE_KINDS[1:]), name
  assert set(release.BATCH_OF) == {a[1] for a in release.ARCHIVES}
  assert {release.BATCH_OF[n] for n in release.FIRST_BUILD_ARCHIVES} == {'zenodo-v1'}


def test_first_build_archives_keep_their_members_without_the_new_kinds(tmp_path):
  old = {'S_main': 'main', 'S_pilot': 'pilot2', 'S_mem': 'memlimit', 'S_ppi': 'ppi', 'S_embed': 'embed',
         'S_att': 'attempt', 'S_out': 'outside', **{f'S_{k}': k for k in release.FOLLOWUP_KINDS}}
  new = {f'S_{k}': k for k in release.LATER_KINDS + ('outside_followups1316',)}
  root = tmp_path / 'af3'
  for s in list(old) + list(new):
    _old_session(root, s)
    _file(root / s / 'run1' / 'T1' / 'xla_dump', 'module_0001.jit_apply_fn.after_codegen.txt', b'HloModule m\n')
    _file(root / s / 'run1' / 'T1', 'fu13_capture_seed-1.npz', b'PK\x03\x04')
  monkey = dict(old)
  before = {n: release.members(FIRST_BUILD[n], root, tmp_path, None, monkey, {}) for n in FIRST_BUILD if FIRST_BUILD[n][2] != 'repo'}
  monkey.update(new)
  after = {n: release.members(FIRST_BUILD[n], root, tmp_path, None, monkey, {}) for n in before}
  assert before == after and any(before.values())
  for n, m in after.items():
    assert not [x for x, _ in m if '/xla_dump/' in x or '_capture' in x], n   # the new categories reach no old archive


@pytest.mark.parametrize('kind', ['fu13', 'fu14', 'rcx', 'hlo', 'memlimit_ah', 'warm_th'])
def test_later_labels_have_their_archives_and_records(tmp_path, kind):
  assert kind in release.RELEASED_KINDS and kind in release.LATER_KINDS
  specs = {a[1]: a for a in release.ARCHIVES}
  root = tmp_path / 'af3'
  _old_session(root, f'S_{kind}')
  kinds_of = {f'S_{kind}': kind}
  lab = next(x for x in release.LATER_LABELS if x['kind'] == kind)
  outputs = [n for n, (rec, cats) in lab['archives'].items() if 'model_cif' in cats]
  assert len(outputs) == 1 and specs[outputs[0]][0] == 'R2'
  got = [m for m, _ in release.members(specs[outputs[0]], root, tmp_path, None, kinds_of, {})]
  assert [m for m in got if m.endswith('_model.cif')] and [m for m in got if m.endswith('_confidences.json')]
  assert [m for m in got if m.endswith('run_alphafold.log')] and not [m for m in got if m.endswith(('job.log', 'run.json'))]
  recs = [m for m, _ in release.members(specs['session_records_followups1316.tar.gz'], root, tmp_path, None, kinds_of, {})]
  assert f'af3/S_{kind}/job.log' in recs and not [m for m in recs if m.endswith('_model.cif')]
  assert release.BUILT_ON[outputs[0]] if outputs[0] in release.BUILT_ON else True
  assert release.BUILT_ON.get(outputs[0], 'vm') == 'vm' and release.BUILT_ON['session_records_followups1316.tar.gz'] == 'laptop'


def test_grouped_archives_and_their_contents(tmp_path):
  specs = {a[1]: a for a in release.ARCHIVES}
  assert specs['af3_memlimit_ah_warm_th.tar.gz'][2] == ['memlimit_ah', 'warm_th']
  assert 'af3_fu13_captures.tar.gz' not in specs                      # left out (decision of 2026-11-17)
  assert not [a for a in release.ARCHIVES if a[2] != 'repo' and 'captures' in a[3]] and 'captures' in release.LEFT_OUT
  assert specs['hlo_xla_dumps.tar.gz'][2:] == (['hlo'], ['xla_dump']) and specs['hlo_xla_dumps.tar.gz'][0] == release.HLO_DUMPS_RECORD
  root = tmp_path / 'af3'
  _old_session(root, 'S_hlo')
  _file(root / 'S_hlo' / 'run1' / 'T1' / 'xla_dump', 'module_0001.jit_apply_fn.after_codegen.txt', b'HloModule m\n')
  _old_session(root, 'S_fu13')
  _file(root / 'S_fu13' / 'run1' / 'T1', 'fu13_capture_seed-1.npz', b'PK\x03\x04')
  kinds_of = {'S_hlo': 'hlo', 'S_fu13': 'fu13'}
  dumps = [m for m, _ in release.members(specs['hlo_xla_dumps.tar.gz'], root, tmp_path, None, kinds_of, {})]
  assert dumps == ['af3/S_hlo/run1/T1/xla_dump/module_0001.jit_apply_fn.after_codegen.txt']
  for spec in release.ARCHIVES:
    if spec[2] != 'repo':
      assert not [m for m, _ in release.members(spec, root, tmp_path, None, kinds_of, {}) if '_capture' in m], spec[1]
  recs = [m for m, _ in release.members(specs['session_records_followups1316.tar.gz'], root, tmp_path, None, kinds_of, {})]
  assert not [m for m in recs if '/xla_dump/' in m or m.endswith('.npz')]


@pytest.mark.parametrize('session,row', [
    ('20261110T092603Z_v5e_why_l5_01', '2026-11-12, Follow-up 13 (label why), complete, item 1'),
    ('20261110T092001Z_l4_why_lg_01', '2026-11-10, Follow-up 13, LG stopped'),
    ('20261110T190412Z_l4_why_gf_29', '2026-11-12, Follow-up 13 (label why), complete, item 5'),
    ('20261112T094029Z_v6e_why_p_06', 'the four rows of 2026-11-12 on arm P'),
    ('20261112T203854Z_v5e_f32_v5f_r2_01', '2026-11-13, Follow-up 14 (label f32), complete, items 3 and 4'),
    ('20261113T102625Z_a100_f32_af_15', 'rows of 2026-11-12 and 2026-11-13 on the A100'),
    ('20261112T201157Z_h100_f32_hf_01', '2026-11-13, Follow-up 14 (label f32), complete, item 2'),
    ('20261113T122618Z_a100_f32_ad_01', '2026-11-13, Follow-up 14 (label f32), complete, items 4 and 10'),
    ('20261114T191329Z_a100_rcx_af_r2_03', '2026-11-15, Follow-up 15 (label rcx), complete, item 2'),
    ('20261115T010417Z_l4_rcx_gt_r2_08', '2026-11-15, Follow-up 15 (label rcx), complete, item 4'),
    ('20261116T140122Z_v5e_hlo_v5_01', '2026-11-16, Follow-up 16 (label hlo), complete')])
def test_later_outside_sessions_map_to_their_row(session, row):
  for paths in (['r/runs.jsonl'], ['launch_record.txt']):
    kind, why = release.classify(session, paths, {})
    assert kind == 'outside_followups1316' and why.startswith(row)
  assert release.classify(session, ['r/runs.jsonl'], {session: 'fu13'})[0] == 'fu13'   # a frozen session keeps its kind


@pytest.mark.parametrize('session', ['20261120T000000Z_l4_why_gf_01', '20261116T150000Z_v5e_hlo_v5_01',
                                     '20261110T101013Z_a100_memlimit_ah_a_01', '20261112T000000Z_h100_f32_hd_01',
                                     '20261125T000000Z_l4_newlabel_01', '20261109T000001Z_v5e_pilot'])
def test_later_sessions_no_row_maps_are_unmapped(session):
  assert release.classify(session, ['r/runs.jsonl'], {})[0] == 'unmapped'
  assert release.classify(session, ['launch_record.txt'], {})[0] == 'unmapped'


def test_sessions_before_the_first_build_keep_their_old_classes():
  assert release.classify('20261108T235959Z_v5e_pilot', ['launch_record.txt'], {})[0] == 'attempt'
  assert release.classify('20261008T205351Z_v5e_bisect', ['r/runs.jsonl'], {})[0] == 'out'
  assert release.classify('S_l4_main_g_01', ['r/runs.jsonl'], {})[0] == 'out'   # names without a stamp: as before


def test_later_mapping_covers_the_local_launcher_records():
  lau = release.REPO / 'results' / 'af3' / '.launcher'
  if not lau.is_dir() or not (release.FREEZE / 'sessions.tsv').is_file():
    pytest.skip('no local launcher records')
  frozen = release.freeze_sessions()
  kinds = [release.classify(d.name, ['launch_record.txt'], frozen)[0] for d in lau.iterdir() if d.is_dir()]
  assert 'unmapped' not in kinds and kinds.count('outside_followups1316') == 247


def test_later_outside_archive_holds_records_only_and_its_table(tmp_path, monkeypatch):
  monkeypatch.setattr(release, 'freeze_sessions', lambda: {})
  monkeypatch.setattr(release, 'unexplained_files', lambda: {})
  root, out = tmp_path / 'af3', tmp_path / 'out'
  name = '20261114T221412Z_l4_rcx_gt_r2_03'
  _old_session(root, name)
  (tmp_path / 'repo').mkdir()
  assert release.build(_args(root, out, archives='outside_freeze_records_followups1316.tar.gz,outside_freeze_records.tar.gz')) == 0
  import tarfile
  assert not (out / 'R1' / 'outside_freeze_records.tar.gz').exists()      # the first build's archive takes none of them
  with tarfile.open(out / 'R1' / 'outside_freeze_records_followups1316.tar.gz') as t:
    names = t.getnames()
    table = t.extractfile('OUTSIDE_SESSIONS.csv').read().decode()
  assert f'af3/{name}/job.log' in names and not [n for n in names if n.endswith(('_model.cif', '_confidences.json'))]
  assert f'{name},"{release.OUTSIDE_LABEL}","2026-11-15, Follow-up 15 (label rcx), complete, item 4' in table


def test_check_inputs_covers_the_later_kinds(tmp_path, monkeypatch):
  monkeypatch.setattr(release, 'freeze_sessions', lambda: {'S_fu14': 'fu14'})
  monkeypatch.setattr(release, 'unexplained_files', lambda: {})
  _session(tmp_path / 'af3', name='S_fu14', data=_data_json(unpairedMsa='>q\nMKV\n>x\nMKL\n'))
  rc = release.check_inputs(types.SimpleNamespace(root=str(tmp_path / 'af3'), repo=str(tmp_path / 'repo'),
                                                  out=str(tmp_path / 'out'), sessions=None))
  assert rc == 2 and json.loads((tmp_path / 'out' / 'inputs_check.json').read_text())['processes'] == 1


# ---------------------------------------------------------------- systemd unit names are not addresses

@pytest.mark.parametrize('text', [
    b'Stopping getty@tty1.service - Getty on tty1...', b'Stopped serial-getty@ttyS0.service.', b'user@1000.service: done',
    b'"unit": "user-runtime-dir@1000.service"', b'systemd-fsck@dev-disk-by\\x2duuid-12ab.service ok',
    b'container-getty@1.service', b'dbus-:1.2-org.freedesktop.problems@0.service', b'lvm2-pvscan@8:1.service',
    b'run-user@0.mount', b'sshd@3-10.0.0.2:22.socket', b'x@y.timer', b'x@y.slice', b'x@y.scope', b'x@y.path',
    b'x@y.swap', b'x@y.device', b'x@y.automount'])
def test_systemd_unit_names_are_not_addresses(tmp_path, text):
  assert release.find_emails(text) == [] and release.redact_emails(text) == text
  if b'10.0.0.2' not in text:
    release.guard(_file(tmp_path, 'job.log', text + b'\n'), 'af3/S/job.log')


@pytest.mark.parametrize('text,address', [
    (b'mail a.b@example.org now', b'a.b@example.org'), (b'x@corp.target', b'x@corp.target'),       # target is a TLD
    (b'getty@tty1.services', b'getty@tty1.services'), (b'a+getty@tty1.service', b'a+getty@tty1.service'),
    (b'ops@db.service.example.com', b'ops@db.service.example.com'), (b'%u@tty1.service', b'%u@tty1.service'),
    (b'getty@tty1.service and a.b@example.org', b'a.b@example.org'),
    (b'authenticated as a.b@example.org which', b'a.b@example.org')])
def test_addresses_are_still_found_next_to_unit_names(tmp_path, text, address):
  assert address in release.find_emails(text)
  assert set(release.EMAIL.findall(text)) >= set(release.find_emails(text))   # never more than the plain pattern
  with pytest.raises(release.Refused, match='email address'):
    release.guard(_file(tmp_path, 'job.log', text), 'af3/S/job.log')
  assert address not in release.redact_emails(text)


def test_unit_names_survive_the_redaction_of_a_listed_file(tmp_path, monkeypatch):
  rel = 'results/af3/.many/W/x.log'
  monkeypatch.setattr(release, 'REDACT', {rel: 'gcloud account email'})
  p = _file(tmp_path, 'x.log', b'authenticated as a.b@example.org\nStopped getty@tty1.service.\n')
  data, row = release.prepare(p, rel, 'session_lists_launch_records.tar.gz')[2:]
  assert data == b'authenticated as <account>\nStopped getty@tty1.service.\n'
  assert row['replaced'].startswith('1 occurrence(s) of 1 email address(es)')


def test_the_unit_types_are_not_top_level_domains():
  # IANA root zone list, version 2026111700: SERVICES and TARGET are TLDs; none of the exempt unit types is.
  tlds_seen = {'services', 'target'}
  assert not {t.decode() for t in release.SYSTEMD_UNIT_TYPES} & tlds_seen
  assert b'target' not in release.SYSTEMD_UNIT_TYPES


# ---------------------------------------------------------------- XLA dumps of follow-up 16

def test_dump_constants_are_measured_and_large_ones_refused(tmp_path):
  small = b'ENTRY %main (p: bf16[384,384]) -> f32[] {\n  %c = f32[] constant(0)\n  %d = s32[3]{0} constant({1, 2, 3})\n}\n'
  assert release.dump_constant_chars(small) == len(b'0)') + len(b'{1, 2, 3})')
  release.guard(_file(tmp_path / 'a', 'm.after_codegen.txt', small), 'af3/S/r/T/xla_dump/m.after_codegen.txt')
  big = b'%w = f32[128,128]{1,0} constant({' + b', '.join([b'0.12345'] * 16384) + b'})\n'
  with pytest.raises(release.Refused, match='printed constants'):
    release.guard(_file(tmp_path / 'b', 'm.after_codegen.txt', big), 'af3/S/r/T/xla_dump/m.after_codegen.txt')
  mlir = b'%c = stablehlo.constant dense<"0x' + b'AB' * 40000 + b'"> : tensor<20000xf32>\n'
  with pytest.raises(release.Refused, match='printed constants'):
    release.guard(_file(tmp_path / 'c', 'module.mlir', mlir), 'af3/S/r/T/xla_dump/m/module.mlir')
  blob = b'%c = stablehlo.constant dense_resource<__elided__> : tensor<4xf32>\n{-#\n  dialect_resources: {}\n#-}\n'
  with pytest.raises(release.Refused, match='resource blob'):
    release.guard(_file(tmp_path / 'd', 'module.mlir', blob), 'af3/S/r/T/xla_dump/m/module.mlir')
  # the same text outside an XLA dump is not measured
  release.guard(_file(tmp_path / 'e', 'notes.txt', big), 'af3/S/r/T/notes.txt')


def test_real_dumps_pass_the_constant_check():
  folders = sorted((release.REPO / 'results' / 'af3').glob('*_hlo_v*_01/*/*/xla_dump'))
  if not folders:
    pytest.skip('no local follow-up 16 dumps')
  for f in folders[:2]:
    for p in f.rglob('*'):
      if p.is_file() and ('jit_apply_fn' in str(p)):
        release.check_dump_constants(p.read_bytes(), str(p))


def test_dumps_archive_carries_the_apache_licence_and_the_notice(tmp_path, monkeypatch):
  import tarfile
  monkeypatch.setattr(release, 'freeze_sessions', lambda: {'S_hlo': 'hlo'})
  monkeypatch.setattr(release, 'unexplained_files', lambda: {})
  root, repo, out = tmp_path / 'af3', tmp_path / 'repo', tmp_path / 'out'
  s = _session(root, name='S_hlo')
  _file(s / 'xla_dump', 'module_0537.jit_apply_fn.after_codegen.txt', b'HloModule m\n// /home/someone/af3/model.py:12\n')
  _passed_check(root, out)
  assert release.build(_args(root, out, archives='hlo_xla_dumps.tar.gz')) == 2        # no LICENSES/Apache-2.0
  _file(repo / 'LICENSES', 'Apache-2.0', b'Apache License\nVersion 2.0, January 2004\n')
  assert release.build(_args(root, out, archives='hlo_xla_dumps.tar.gz')) == 0
  rec = release.HLO_DUMPS_RECORD
  with tarfile.open(out / rec / 'hlo_xla_dumps.tar.gz') as t:
    assert t.extractfile('LICENSE_ALPHAFOLD3_CODE_APACHE-2.0.txt').read().startswith(b'Apache License')
    notice = t.extractfile('NOTICE_XLA_DUMPS.txt').read().decode()
    dump = t.extractfile('af3/S_hlo/l4_xla_rep1_seed1_fresh/7U3J/xla_dump/module_0537.jit_apply_fn.after_codegen.txt').read()
  assert 'Apache License 2.0' in notice and 'do not hold the model parameters' in notice
  assert b'/home/<user>/af3' in dump and b'someone' not in dump
  rows = list(release.csv.DictReader(open(out / rec / 'REDACTIONS.tsv'), delimiter='\t'))
  assert [r['archive'] for r in rows] == ['hlo_xla_dumps.tar.gz']


def test_repository_has_the_apache_licence_text():
  text = (release.REPO / 'LICENSES' / 'Apache-2.0').read_text()
  assert text.lstrip().startswith('Apache License') and 'Version 2.0, January 2004' in text
  assert 'END OF TERMS AND CONDITIONS' in text


# ---------------------------------------------------------------- paper assets from git

def _git(repo, *a):
  import subprocess
  subprocess.run(['git', '-C', str(repo), *a], check=True, capture_output=True)


def test_paper_assets_come_from_a_git_ref(tmp_path, monkeypatch):
  import tarfile
  monkeypatch.setattr(release, 'freeze_sessions', lambda: {})
  monkeypatch.setattr(release, 'unexplained_files', lambda: {})
  root, repo, out = tmp_path / 'af3', tmp_path / 'repo', tmp_path / 'out'
  root.mkdir()
  _file(repo / 'results' / 'analysis' / 'fu13_20261112', 'report.md', b'# fu13\n')
  _file(repo / 'results' / 'paper_assets_isc' / 'tables', 'T1.tex', b'old\n')
  _file(repo / 'results' / 'paper_assets_fu9', 'README.md', b'r\n')
  _file(root / 'S', 'TERMS_OF_USE.md', b'# ALPHAFOLD 3 OUTPUT TERMS OF USE\n')
  _git(repo, 'init', '-q')
  _file(repo / 'paper' / 'assets' / 'tables', 'Tmain.tex', b'committed\n')
  _git(repo, 'add', 'paper/assets')
  _git(repo, '-c', 'user.name=t', '-c', 'user.email=t@example.org', 'commit', '-q', '-m', 'assets')
  _git(repo, 'tag', 'paper-test')
  _file(repo / 'paper' / 'assets' / 'tables', 'Tmain.tex', b'uncommitted change\n')
  _file(repo / 'paper' / 'assets', 'untracked.tex', b'never committed\n')
  args = _args(root, out, archives='analysis_outputs.tar.gz', paper_ref=None)
  args.repo = str(repo)
  assert release.build(args) == 2                                   # the ref must be given
  args.paper_ref = 'no-such-ref'
  assert release.build(args) == 2
  args.paper_ref = 'paper-test'
  assert release.build(args) == 0
  with tarfile.open(out / 'R2' / 'analysis_outputs.tar.gz') as t:
    names = set(t.getnames())
    assert t.extractfile('paper/assets/tables/Tmain.tex').read() == b'committed\n'
    commit = t.extractfile('PAPER_ASSETS_COMMIT.txt').read().decode()
  assert 'paper/assets/untracked.tex' not in names and 'results/analysis/fu13_20261112/report.md' in names
  assert 'results/paper_assets_fu9/README.md' in names and not [n for n in names if 'paper_assets_isc' in n]
  assert commit.startswith('paper/assets/ from git ref paper-test, commit ') and len(commit.split()[-1]) == 40


# ---------------------------------------------------------------- the copy of the code in R1 (decision D5 of 2026-11-18)

def _code_repo(tmp_path):
  """A git repository: a commit tagged CODE_TAG, then an uncommitted change and an untracked file."""
  repo = tmp_path / 'repo'
  _file(repo, 'README.md', b'# code\n')
  for name in release.CODE_REQUIRED:
    _file(repo, name, f'licence file {name}\n'.encode())
  _file(repo / 'scripts', 'run.sh', b'#!/bin/bash\necho run\n').chmod(0o755)
  _file(repo / 'paper' / 'assets' / 'tables', 'T1.tex', b'computed from outputs\n')
  _file(repo / 'paper', 'main.tex', b'\\documentclass{article}\n')
  _git(repo, 'init', '-q')
  _git(repo, 'add', '-A')
  _git(repo, '-c', 'user.name=t', '-c', 'user.email=t@example.org', 'commit', '-q', '-m', 'release')
  _git(repo, 'tag', '-a', release.CODE_TAG, '-m', 'release')
  _file(repo, 'README.md', b'# changed after the tag\n')
  _file(repo, 'untracked.txt', b'never committed\n')
  return repo


def _code_build(tmp_path, monkeypatch, repo, ref):
  monkeypatch.setattr(release, 'freeze_sessions', lambda: {})
  monkeypatch.setattr(release, 'unexplained_files', lambda: {})
  (tmp_path / 'af3').mkdir(exist_ok=True)
  args = _args(tmp_path / 'af3', tmp_path / 'out', archives=release.CODE_ARCHIVE, code_ref=ref)
  args.repo = str(repo)
  return release.build(args)


def test_code_archive_spec():
  assert ('R1', release.CODE_ARCHIVE, 'git', [release.CODE_TAG]) in release.ARCHIVES
  assert release.CODE_ARCHIVE == 'code_alphafold-on-tpu_zenodo-v1.tar.gz' and release.CODE_PREFIX == 'alphafold-on-tpu-zenodo-v1/'
  assert release.BUILT_ON[release.CODE_ARCHIVE] == 'laptop' and release.CODE_ARCHIVE not in release.FIRST_BUILD_ARCHIVES
  assert release.CODE_EXCLUDES == ('paper/assets',)


def test_code_archive_is_the_tagged_commit(tmp_path, monkeypatch):
  import tarfile
  repo = _code_repo(tmp_path)
  assert _code_build(tmp_path, monkeypatch, repo, release.CODE_TAG) == 0
  archive = tmp_path / 'out' / 'R1' / release.CODE_ARCHIVE
  commit = release.git_commit(repo, release.CODE_TAG)
  with tarfile.open(archive) as t:
    members = {m.name: m for m in t}
    assert t.extractfile('alphafold-on-tpu-zenodo-v1/README.md').read() == b'# code\n'     # committed, not the change
    text = t.extractfile('CODE_COMMIT.txt').read().decode()
  assert sorted(members) == sorted(['CODE_COMMIT.txt', 'alphafold-on-tpu-zenodo-v1/README.md',
                                    'alphafold-on-tpu-zenodo-v1/paper/main.tex', 'alphafold-on-tpu-zenodo-v1/scripts/run.sh']
                                   + [release.CODE_PREFIX + n for n in release.CODE_REQUIRED])
  assert members['alphafold-on-tpu-zenodo-v1/scripts/run.sh'].mode == 0o755
  assert members['alphafold-on-tpu-zenodo-v1/README.md'].mode == 0o644
  assert {m.mtime for m in members.values()} == {release.MTIME}
  assert text.split('\n')[0] == commit and len(commit) == 40 and 'without paper/assets' in text
  man = (tmp_path / 'out' / 'R1' / f'{release.CODE_ARCHIVE}.sha256').read_text()
  assert f'{hashlib.sha256(b"# code" + bytes([10])).hexdigest()}  alphafold-on-tpu-zenodo-v1/README.md' in man
  assert release.CODE_ARCHIVE in (tmp_path / 'out' / 'R1' / 'SHA256SUMS').read_text()
  assert not (tmp_path / 'out' / 'R1' / 'REDACTIONS.tsv').exists()
  first = archive.read_bytes()
  args = types.SimpleNamespace(out=str(tmp_path / 'out'), previous=None, repo=str(repo))
  assert release.verify(args) == 0
  monkeypatch.setattr(release, 'CODE_EXCLUDES', ())          # the commit now has a file the archive lacks
  assert release.verify(args) == 1
  monkeypatch.setattr(release, 'CODE_EXCLUDES', ('paper/assets',))
  assert release.verify(types.SimpleNamespace(out=str(tmp_path / 'out'), previous=None, repo=str(tmp_path / 'af3'))) == 0
  (tmp_path / 'out' / 'R1' / release.CODE_ARCHIVE).unlink()
  assert _code_build(tmp_path, monkeypatch, repo, 'HEAD') == 0                            # same commit, another ref
  assert archive.read_bytes() == first                                                    # deterministic


def test_code_archive_needs_the_ref_of_the_tagged_commit(tmp_path, monkeypatch):
  repo = _code_repo(tmp_path)
  assert _code_build(tmp_path, monkeypatch, repo, None) == 2                              # the ref must be given
  assert _code_build(tmp_path, monkeypatch, repo, 'no-such-ref') == 2
  _git(repo, 'add', '-A')
  _git(repo, '-c', 'user.name=t', '-c', 'user.email=t@example.org', 'commit', '-q', '-m', 'after the tag')
  assert _code_build(tmp_path, monkeypatch, repo, 'HEAD') == 2                            # not the tagged commit
  assert not (tmp_path / 'out' / 'R1').exists() or not list((tmp_path / 'out' / 'R1').glob('code_*'))
  assert _code_build(tmp_path, monkeypatch, repo, 'none') == 0                            # a test build without it


@pytest.mark.parametrize('path,data,refused', [
    ('notes/x.md', b'write to someone@example.org\n', True),
    ('scripts/release/test_release.py', b'someone@example.org\n', False),
    ('scripts/x.sh', b'gsutil cp gs://another-bucket/x .\n', True),
    ('scripts/x.sh', b'gsutil cp gs://af2-tpu-benchmark-af3-weights/x .\n', False),
    ('notes/y.md', b'log /Users/someone/x\n', True),
    ('targets/ppi/run.log', b'/Users/someone/x\n', False),
    ('notes/y.md', b'host naspaz8 here\n', True),
    ('notes/analysis_plan.md', b'host naspaz8 here\n', False),
    ('notes/y.md', b'ip 192.168.1.20\n', True),
    ('notes/y.md', b'token ghp_' + b'A' * 36 + b'\n', True),
    ('scripts/release/test_release.py', b'gs://another-bucket /Users/someone 192.168.1.20 naspaz8\n', False),
    ('scripts/release/test_release.py', b'token ghp_' + b'A' * 36 + b'\n', False),  # the guard's tests: made-up fixtures
    ('scripts/release/release.py', b'token ghp_' + b'A' * 36 + b'\n', True),
])
def test_code_guard_refuses_what_it_does_not_list(tmp_path, path, data, refused):
  src = _file(tmp_path, 'f', data)
  if refused:
    with pytest.raises(release.Refused):
      release.check_code_file(src, path)
  else:
    assert release.check_code_file(src, path)[0] == len(data)


def test_code_archive_refuses_a_commit_without_its_licence_files(tmp_path, monkeypatch):
  repo = _code_repo(tmp_path)
  _git(repo, 'rm', '-q', 'NOTICE')
  _git(repo, '-c', 'user.name=t', '-c', 'user.email=t@example.org', 'commit', '-q', '-m', 'no notice')
  _git(repo, 'tag', '-f', '-a', release.CODE_TAG, '-m', 'moved')
  assert _code_build(tmp_path, monkeypatch, repo, release.CODE_TAG) == 2
  assert not list((tmp_path / 'out').rglob('code_*'))


def test_code_licence_files_pass_the_guard():
  assert release.CODE_REQUIRED == ('LICENSE', 'NOTICE', 'LICENSES/Apache-2.0', 'LICENSES/MIT-alphafold-tpu-benchmark')
  for name in release.CODE_REQUIRED:
    assert release.check_code_file(release.REPO / name, name)[0] > 0
  notice = (release.REPO / 'NOTICE').read_text()
  assert '4e9c6b64f3026bc9316b12590fd4cd16559ab51f' in notice and 'Lorenzo Pazienza and Ihab El Bani' in notice
  assert 'LICENSES/MIT-alphafold-tpu-benchmark' in notice
  for path in [l.strip() for l in notice.splitlines() if l.startswith('  ')]:
    assert (release.REPO / path).is_file(), path
  text = (release.REPO / 'LICENSES' / 'MIT-alphafold-tpu-benchmark').read_text()
  assert text.startswith('MIT License') and 'Copyright (c) 2026 Lorenzo Pazienza and Ihab El Bani' in text


def test_code_guard_allows_the_contact_in_any_file(tmp_path, monkeypatch):
  monkeypatch.setattr(release, 'CONTACT_SHA256', hashlib.sha256(b'me@example.net').hexdigest())
  src = _file(tmp_path, 'f', b'Contact: me@example.net, again me@example.net\n')
  assert release.check_code_file(src, 'paper/main.tex')[0] > 0
  with pytest.raises(release.Refused):
    release.check_code_file(_file(tmp_path, 'g', b'me@example.net and other@example.net\n'), 'paper/main.tex')


def test_code_build_refuses_a_file_it_does_not_list_and_keeps_nothing(tmp_path, monkeypatch):
  repo = _code_repo(tmp_path)
  _file(repo / 'notes', 'n.md', b'see /Users/someone/data\n')
  _git(repo, 'add', 'notes/n.md')
  _git(repo, '-c', 'user.name=t', '-c', 'user.email=t@example.org', 'commit', '-q', '-m', 'n')
  _git(repo, 'tag', '-f', '-a', release.CODE_TAG, '-m', 'moved')
  assert _code_build(tmp_path, monkeypatch, repo, release.CODE_TAG) == 2
  assert not list((tmp_path / 'out').rglob('code_*'))


# ---------------------------------------------------------------- incremental versions

def _previous(tmp_path, files):
  """A previous version's record folders: {record: {name: bytes}}; SHA256SUMS, Zenodo listings, REDACTIONS.tsv."""
  prev = tmp_path / 'prev'
  for rec, items in files.items():
    d = prev / rec
    d.mkdir(parents=True)
    (d / 'SHA256SUMS').write_text(''.join(f'{hashlib.sha256(b).hexdigest()}  {n}\n' for n, b in sorted(items.items())))
    listing = [{'filename': n, 'filesize': len(b)} for n, b in items.items()] + [{'filename': 'SHA256SUMS', 'filesize': 10}]
    (prev / f'zenodo_files_{release.ZENODO_IDS[rec]}.json').write_text(json.dumps(listing))
    release.write_redactions(d, [{'archive': 'af3_main_structures.tar.gz', 'path': 'af3/S/job.log', 'original_sha256': 'a',
                                  'released_sha256': 'b', 'replaced': '1 home_path'},
                                 {'archive': 'analysis_outputs.tar.gz', 'path': 'results/analysis/x/log', 'original_sha256': 'c',
                                  'released_sha256': 'd', 'replaced': '1 home_path'}])
  return prev


def test_a_later_version_keeps_what_it_does_not_rebuild(tmp_path):
  prev = _previous(tmp_path, {'R2': {'af3_main_structures.tar.gz': b'old main', 'af3_main_structures.tar.gz.sha256': b'm',
                                     'analysis_outputs.tar.gz': b'old analysis', 'analysis_outputs.tar.gz.sha256': b'a',
                                     'af3_retired.tar.gz': b'gone', 'README.md': b'old readme', 'LICENCES.md': b'l'}})
  out = tmp_path / 'out'
  (out / 'R2').mkdir(parents=True)
  release.write_archive(out / 'R2' / 'analysis_outputs.tar.gz', [], [('results/analysis/y/log', b'new log\n')])
  _file(out / 'R2', 'analysis_outputs.tar.gz.sha256', f'{hashlib.sha256(b"new log\n").hexdigest()}  results/analysis/y/log\n'.encode())
  new_analysis = (out / 'R2' / 'analysis_outputs.tar.gz').read_bytes()
  release.write_archive(out / 'R2' / 'af3_fu13.tar.gz', [], [('af3/S/x', b'x')])
  release.write_redactions(out / 'R2', [{'archive': 'analysis_outputs.tar.gz', 'path': 'results/analysis/y/log',
                                         'original_sha256': 'e', 'released_sha256': 'f', 'replaced': '1 home_path'}])
  assert release.sums(types.SimpleNamespace(out=str(out), previous=str(prev))) == 0
  sums = dict(reversed(l.split('  ', 1)) for l in (out / 'R2' / 'SHA256SUMS').read_text().splitlines())
  assert sums['af3_main_structures.tar.gz'] == hashlib.sha256(b'old main').hexdigest()          # kept, not here
  assert sums['analysis_outputs.tar.gz'] == hashlib.sha256(new_analysis).hexdigest()            # rebuilt
  assert 'af3_retired.tar.gz' not in sums and 'af3_fu13.tar.gz' in sums
  rows = {(r['archive'], r['path']) for r in release.csv.DictReader(open(out / 'R2' / 'REDACTIONS.tsv'), delimiter='\t')}
  assert rows == {('af3_main_structures.tar.gz', 'af3/S/job.log'), ('analysis_outputs.tar.gz', 'results/analysis/y/log')}
  changes = {r['file']: r['action'] for r in release.csv.DictReader(open(out / 'zenodo_changes_R2.csv'))}
  assert changes['af3_main_structures.tar.gz'] == 'keep' and changes['af3_main_structures.tar.gz.sha256'] == 'keep'
  assert changes['analysis_outputs.tar.gz'] == 'replace' and changes['analysis_outputs.tar.gz.sha256'] == 'replace'
  assert changes['af3_fu13.tar.gz'] == 'add' and changes['af3_retired.tar.gz'] == 'delete'
  assert changes['README.md'] == 'replace' and changes['SHA256SUMS'] == 'replace'
  assert release.verify(types.SimpleNamespace(out=str(out), previous=str(prev))) == 0
  assert release.verify(types.SimpleNamespace(out=str(out), previous=None)) == 1           # without it, the kept files are missing


def test_a_later_version_over_the_limits_counts_the_kept_files(tmp_path, monkeypatch):
  prev = _previous(tmp_path, {'R2': {'af3_main_structures.tar.gz': b'x' * 100}})
  _file(tmp_path / 'out' / 'R2', 'af3_fu13.tar.gz', b'y')
  monkeypatch.setattr(release, 'ZENODO_MAX_BYTES', 50)
  assert release.sums(types.SimpleNamespace(out=str(tmp_path / 'out'), previous=str(prev))) == 2


def test_archives_of_a_batch_and_the_vm_lists(tmp_path, capsys):
  release.archives(types.SimpleNamespace(built_on='vm', batch='followups1316'))
  vm = capsys.readouterr().out.strip().split(',')
  assert vm == ['outside_freeze_records_followups1316.tar.gz', 'af3_fu13.tar.gz',
                'af3_fu14.tar.gz', 'af3_rcx.tar.gz', 'af3_hlo.tar.gz', 'hlo_xla_dumps.tar.gz', 'af3_memlimit_ah_warm_th.tar.gz']
  release.archives(types.SimpleNamespace(built_on='laptop', batch='followups1316'))
  assert capsys.readouterr().out.strip() == 'session_records_followups1316.tar.gz,code_alphafold-on-tpu_zenodo-v1.tar.gz'
  inv = tmp_path / 'inv'
  release.write_csv(_file(inv, 'sessions.csv').parent / 'sessions.csv',
                    [{'session': 'A', 'kind': 'main'}, {'session': 'B', 'kind': 'fu13'}, {'session': 'C', 'kind': 'outside'},
                     {'session': 'D', 'kind': 'outside_followups1316'}, {'session': 'E', 'kind': 'warm_th'}])
  assert release.vm_lists(types.SimpleNamespace(inventory=str(inv), out=str(tmp_path / 'b'), batch='followups1316')) == 0
  assert (tmp_path / 'b' / 'sessions_release.txt').read_text() == 'B\nE\n'
  assert (tmp_path / 'b' / 'sessions_outside.txt').read_text() == 'D\n'
  assert release.vm_lists(types.SimpleNamespace(inventory=str(inv), out=str(tmp_path / 'c'), batch=None)) == 0
  assert (tmp_path / 'c' / 'sessions_release.txt').read_text() == 'A\nB\nE\n'
  assert (tmp_path / 'c' / 'sessions_outside.txt').read_text() == 'C\nD\n'


def test_the_records_stay_within_the_limits_with_the_new_archives():
  r1 = sum(1 for a in release.ARCHIVES if a[0] == 'R1')
  r2 = sum(1 for a in release.ARCHIVES if a[0] == 'R2')
  assert (r1, r2) == ({'R2': 12, 'R1': 13}[release.HLO_DUMPS_RECORD], {'R2': 42, 'R1': 41}[release.HLO_DUMPS_RECORD])
  assert 2 * r1 + release.PLAIN_FILES['R1'] <= 43 and 2 * r2 + release.PLAIN_FILES['R2'] <= 88
