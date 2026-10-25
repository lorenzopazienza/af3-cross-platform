"""scripts/nas/ingest_official.py (follow-up 6, notes/analysis_plan.md section 10, rows of 2026-10-19 and 2026-10-20)
on synthetic NAS deliveries: checks, stop messages, the inference inputs and the freeze record. The AlphaFold3
featurisation test (tokens, bucket, templates from the inline mmCIF) needs the AlphaFold3 venv."""
import csv
import hashlib
import importlib.util
import json
import pathlib

import pytest

REPO = pathlib.Path(__file__).resolve().parents[2]
DATA = REPO / 'analysis' / 'tests' / 'fixtures'
spec = importlib.util.spec_from_file_location('ingest_official', REPO / 'scripts' / 'nas' / 'ingest_official.py')
ing = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ing)
FIXTURE = json.loads((DATA / 'msa_data_6yqw.json').read_text())
TARGETS = {'6YQW': 'yes', '7AAA': 'no', '7BBB': 'no'}


def sha(b):
  return hashlib.sha256(b).hexdigest()


def official_input(name):
  d = json.loads(json.dumps(FIXTURE))
  d['name'] = name
  d['version'] = 1
  for k in ('bondedAtomPairs', 'userCCD'):
    d.pop(k)
  p = d['sequences'][0]['protein']
  for k in ('modifications', 'unpairedMsa', 'pairedMsa', 'templates'):
    p.pop(k)
  return d


def data_json(name):
  d = json.loads(json.dumps(FIXTURE))
  d['name'] = name
  return d


@pytest.fixture
def world(tmp_path, monkeypatch):
  for k, v in {'REPO': tmp_path, 'NAS': tmp_path / 'data' / 'nas',
               'OFFICIAL_MANIFEST': tmp_path / 'inputs' / 'official' / 'manifest.csv',
               'MAIN_MANIFEST': tmp_path / 'inputs' / 'manifest.csv', 'DATA_DIR': tmp_path / 'data' / 'inputs_official_data',
               'OUT_MANIFEST': tmp_path / 'inputs' / 'official_data' / 'manifest.csv',
               'SMOKE_MANIFEST': tmp_path / 'inputs' / 'official_data' / 'smoke_manifest.csv',
               'FREEZE': tmp_path / 'results' / 'freeze'}.items():
    monkeypatch.setattr(ing, k, v)
  (tmp_path / 'inputs' / 'official').mkdir(parents=True)
  (tmp_path / 'data' / 'inputs_official').mkdir(parents=True)
  orows, mrows = [], []
  seq = FIXTURE['sequences'][0]['protein']['sequence']
  for t, pilot in TARGETS.items():
    raw = (json.dumps(official_input(t), indent=2) + '\n').encode()
    (tmp_path / 'data' / 'inputs_official' / f'{t}.json').write_bytes(raw)
    orows.append({'pdb_id': t, 'ligand_ccd': '82I', 'pilot': pilot, 'official_input_path': f'data/inputs_official/{t}.json',
                  'official_input_sha256': sha(raw), 'sequence_sha256': sha(seq.encode()),
                  'rcsb_initial_release': '2021-04-14', 'released_on_or_before_cutoff': 'no'})
    mrows.append({'pdb_id': t, 'num_tokens': '117', 'bucket': '256'})
  for path, rows in ((ing.OFFICIAL_MANIFEST, orows), (ing.MAIN_MANIFEST, mrows)):
    with open(path, 'w', newline='') as f:
      w = csv.DictWriter(f, fieldnames=list(rows[0]))
      w.writeheader()
      w.writerows(rows)
  return tmp_path


def delivery(world, run, targets, fail=(), rows_before=(), tamper=None):
  src = world / 'data' / 'nas' / run
  out_lines, rows = [], []
  for t in rows_before:
    rows.append([t, run.split('_')[0], run.split('_')[1], '3', 's', 'e', '10', '1', '2', '1', '', '', '', '', ''])
  for t in targets:
    rel = f'output/{t}/{t.lower()}/{t.lower()}_data.json'
    p = src / rel
    p.parent.mkdir(parents=True)
    raw = data_json(t)
    if tamper == ('sequence', t):
      raw['sequences'][0]['protein']['sequence'] = 'A' + raw['sequences'][0]['protein']['sequence'][1:]
    b = json.dumps(raw, indent=2).encode()
    p.write_bytes(b)
    out_lines.append(f'{sha(b)}  {rel}')
    rc = '1' if t in fail else '0'
    rows.append([t, run.split('_')[0], run.split('_')[1], '3', 's', 'e', '600', '1000', '9000', rc,
                 f'/data/runs/{run}/{rel}', sha(b), '24', '0', '1'])
  (src / 'outputs_sha256.txt').write_text('\n'.join(out_lines) + '\n')
  with open(src / 'results.csv', 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['pdb_id', 'host', 'run', 'jackhmmer_n_cpu', 'start_utc', 'end_utc', 'seconds', 'vm_mem_baseline_mib',
                'vm_mem_peak_mib', 'exit_code', 'data_json', 'data_json_sha256', 'unpaired_seqs', 'paired_seqs', 'templates'])
    w.writerows(rows)
  (src / 'provenance.txt').write_text('host=test\n')
  return src


def test_run1_builds_byte_unchanged_inputs_a_manifest_and_a_freeze(world):
  src = delivery(world, 'naspaz8-af3_run1', list(TARGETS), rows_before=['7AAA'])   # a failed attempt, then success
  assert ing.ingest('naspaz8-af3_run1', no_featurise=True) == 0
  rows = list(csv.DictReader(open(ing.OUT_MANIFEST)))
  assert [r['pdb_id'] for r in rows] == sorted(TARGETS)
  for r in rows:
    delivered = src / 'output' / r['pdb_id'] / r['pdb_id'].lower() / f'{r["pdb_id"].lower()}_data.json'
    assert (world / r['input_path']).read_bytes() == delivered.read_bytes()
    assert r['input_sha256'] == r['source_sha256'] == sha(delivered.read_bytes())
    assert (r['num_tokens'], r['bucket'], r['unpaired_msa_rows'], r['templates']) == ('117', '256', '24', '1')
    assert r['tokens_from'].startswith('main study manifest')
  digest = (ing.OUT_MANIFEST.parent / 'manifest.csv.sha256').read_text().split()[0]
  assert digest == sha(ing.OUT_MANIFEST.read_bytes())
  summary = (world / 'results' / 'freeze' / 'msa_naspaz8-af3_run1' / 'summary.md').read_text()
  assert digest in summary and 'before any inference run' in summary
  tsv = (world / 'results' / 'freeze' / 'msa_naspaz8-af3_run1' / 'files_sha256.tsv').read_text()
  assert 'results.csv' in tsv and 'outputs_sha256.txt' in tsv


def test_a_failed_target_stops_with_the_stop_rule(world):
  delivery(world, 'naspaz8-af3_run1', list(TARGETS), fail=['7BBB'])
  with pytest.raises(ing.Stop, match='failed: 7BBB'):
    ing.ingest('naspaz8-af3_run1', no_featurise=True)
  assert not ing.OUT_MANIFEST.exists() and not (world / 'results').exists()


def test_a_missing_target_stops(world):
  delivery(world, 'naspaz8-af3_run1', ['6YQW', '7AAA'])
  with pytest.raises(ing.Stop, match='not run: 7BBB'):
    ing.ingest('naspaz8-af3_run1', no_featurise=True)


def test_a_file_that_differs_from_the_runner_manifest_stops(world):
  src = delivery(world, 'naspaz8-af3_run1', list(TARGETS))
  p = src / 'output' / '7AAA' / '7aaa' / '7aaa_data.json'
  p.write_bytes(p.read_bytes() + b' ')
  with pytest.raises(ing.Stop, match='differ from outputs_sha256.txt'):
    ing.ingest('naspaz8-af3_run1', no_featurise=True)


def test_an_unlisted_output_file_stops(world):
  src = delivery(world, 'naspaz8-af3_run1', list(TARGETS))
  (src / 'output' / '6YQW' / 'extra.txt').write_text('x')
  with pytest.raises(ing.Stop, match='not in outputs_sha256.txt'):
    ing.ingest('naspaz8-af3_run1', no_featurise=True)


def test_a_changed_sequence_stops(world):
  delivery(world, 'naspaz8-af3_run1', list(TARGETS), tamper=('sequence', '7AAA'))
  with pytest.raises(ing.Stop, match='protein sequence'):
    ing.ingest('naspaz8-af3_run1', no_featurise=True)


@pytest.mark.parametrize('run', ['naspaz8-af3_run2', 'naspaz6-af3_run1'])
def test_determinism_runs_are_checked_and_frozen_but_never_inputs(world, run):
  delivery(world, run, ['6YQW'])
  assert ing.ingest(run, no_featurise=True) == 0
  assert not ing.OUT_MANIFEST.exists() and not ing.DATA_DIR.exists()
  assert 'never an inference input' in (world / 'results' / 'freeze' / f'msa_{run}' / 'summary.md').read_text()


def test_an_unknown_run_is_refused(world):
  with pytest.raises(ing.Stop, match='not one of the deliveries'):
    ing.ingest('naspaz8-af3_run3', no_featurise=True)


@pytest.mark.skipif(not (REPO / 'third_party' / 'alphafold3' / '.venv' / 'bin' / 'python').is_file(),
                    reason='needs the AlphaFold3 venv')
def test_alphafold3_featurises_the_data_json_with_its_template_and_no_database():
  out = ing.featurise([DATA / 'msa_data_6yqw.json'])[str(DATA / 'msa_data_6yqw.json')]
  main = {r['pdb_id']: r for r in csv.DictReader(open(REPO / 'inputs' / 'manifest.csv'))}['6YQW']
  assert out == {'num_tokens': int(main['num_tokens']), 'bucket': int(main['bucket']), 'templates_featurised': 1}


def test_smoke_refuses_any_file_but_the_2pv7_setup_check(world, tmp_path):
  f = tmp_path / '2pv7_data.json'
  f.write_text(json.dumps(FIXTURE))
  with pytest.raises(ing.Stop, match='are not the 2PV7 setup-check file'):
    ing.smoke(f)
  assert not ing.SMOKE_MANIFEST.exists()
  assert ing.SMOKE_SHA == '6e8904a1e74569308573b09358298520f4cdd4d4e8a1b91dff8ae9a8c3b588fb'


def test_smoke_folder_needs_every_copy_identical_and_picks_one(world, tmp_path, monkeypatch):
  out = tmp_path / 'smoke_output'
  for d in ('2pv7', '2pv7_20261019_101500', '2pv7_20261020_090000'):
    (out / d).mkdir(parents=True)
    (out / d / '2pv7_data.json').write_text('same bytes')
  monkeypatch.setattr(ing, 'SMOKE_SHA', sha(b'same bytes'))
  monkeypatch.setattr(ing, 'featurise', lambda paths: {str(paths[0]): {'num_tokens': 600, 'bucket': 768,
                                                                        'templates_featurised': 0}})
  monkeypatch.setattr(ing.json, 'loads', lambda text: {'modelSeeds': [1], 'sequences': []})
  assert ing.smoke(out) == 0
  row = next(csv.DictReader(open(ing.SMOKE_MANIFEST)))
  assert row['source_run'].endswith('1 of 3 identical copies') and row['input_sha256'] == sha(b'same bytes')
  (out / '2pv7_20261020_090000' / '2pv7_data.json').write_text('other bytes')
  with pytest.raises(ing.Stop, match='1 of 3 _data.json'):
    ing.smoke(out)
