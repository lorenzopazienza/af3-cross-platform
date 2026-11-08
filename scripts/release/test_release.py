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
    ('xla_dump.tgz', 'other_binary')])
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
                               sessions=kw.get('sessions'), allow_incomplete_inputs_check=False)


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
# The relations of Zenodo's deposit metadata reference (developers.zenodo.org, read on 2026-11-08); isRelatedTo is not one.
ZENODO_RELATIONS = {'isCitedBy', 'cites', 'isSupplementTo', 'isSupplementedBy', 'isContinuedBy', 'continues', 'isDescribedBy',
                    'describes', 'hasMetadata', 'isMetadataFor', 'isNewVersionOf', 'isPreviousVersionOf', 'isPartOf', 'hasPart',
                    'isReferencedBy', 'references', 'isDocumentedBy', 'documents', 'isCompiledBy', 'compiles', 'isVariantFormOf',
                    'isOriginalFormof', 'isIdenticalTo', 'isAlternateIdentifier', 'isReviewedBy', 'reviews', 'isDerivedFrom',
                    'isSourceOf', 'requires', 'isRequiredBy', 'isObsoletedBy', 'obsoletes'}


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
  assert '$API/$R1_ID"' in text and '--data @$HOME/zenodo_R1.json' in text
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
  assert 'github.com/lorenzopazienza/alphafold-on-tpu' in str(m['related_identifiers']) and 'arXiv:<arXiv identifier>' in str(m['related_identifiers'])
  other = 'R2' if record == 'R1' else 'R1'
  links = {r['identifier']: r['relation'] for r in m['related_identifiers']}
  assert links[DOI[other]] == {'R1': 'isSourceOf', 'R2': 'isDerivedFrom'}[record]
  assert set(links.values()) <= ZENODO_RELATIONS
  assert '<DOI of record' not in str(m) and '&lt;DOI' not in str(m)
  assert release.FREEZE_ID in m['description'] and DOI[other] in m['description']


@pytest.mark.parametrize('record', ['R1', 'R2'])
def test_readme_carries_the_decisions(record):
  text = (release.HERE / 'records' / record / 'README.md').read_text()
  for needed in (release.FREEZE_ID, '0009-0006-1687-0125', DOI['R1'], DOI['R2'], 'REDACTIONS.tsv', 'LICENCES.md'):
    assert needed in text
  assert '<DOI of record' not in text
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
