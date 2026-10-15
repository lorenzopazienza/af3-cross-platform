"""scripts/release/release.py: categories, session kinds, the weights guard, the *_data.json check, and refused
builds that keep nothing.

    analysis/.venv/bin/python -m pytest -q scripts/release/test_release.py
"""
import hashlib
import json
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
