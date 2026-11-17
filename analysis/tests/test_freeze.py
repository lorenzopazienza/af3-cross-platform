"""The data freeze (scripts/freeze/make_freeze_manifest.py, freeze_index.py): the session selection of each flag, against the
local records of the four earlier freezes; the follow-up 6 to 12 and 13 to 16 extensions; the comparison and the index on
synthetic folders. No bucket and no gcloud: every gcloud helper is replaced, and subprocess is refused."""
import csv
import hashlib
import importlib.util
import pathlib
import subprocess
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parents[2]
FREEZE = REPO / 'results' / 'freeze'


def load(name):
  spec = importlib.util.spec_from_file_location(name, REPO / 'scripts' / 'freeze' / f'{name}.py')
  mod = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(mod)
  return mod


mf = load('make_freeze_manifest')
fi = load('freeze_index')
EARLIER = [('sessions.tsv', {}), ('followups34_20261018/sessions.tsv', {'followups34': True}),
           ('feat_20261018/sessions.tsv', {'feat': True}), ('followups612_20261108/sessions.tsv', {'followups612': True})]
HAVE_RECORDS = (FREEZE / 'followups612_20261108' / 'sessions.tsv').is_file() and (mf.AF3 / '.many').is_dir()
needs_records = pytest.mark.skipif(not HAVE_RECORDS, reason='results/freeze or results/af3/.many is absent')
LABELS = {'msa': 20, 'sw': 9, 'seeds': 27, 'h100': 6, 'repl': 18, 'prec5': 6, 'large': 49}
LABELS_1316 = {'fu13': 79, 'fu14': 44, 'rcx': 15, 'hlo': 2, 'memlimit_ah': 2, 'warm_th': 2}


def recorded(rel):
  """(session, kind) of a freeze's sessions.tsv, as text."""
  lines = (FREEZE / rel).read_text().splitlines()[1:]
  return [tuple(l.split('\t')[:2]) for l in lines]


@needs_records
@pytest.mark.parametrize('rel,kw', EARLIER)
def test_earlier_flags_give_the_session_list_of_their_freeze(rel, kw):
  assert mf.sessions(**kw) == recorded(rel)


def expected_identifier(rel, want):
  assert hashlib.sha256((FREEZE / rel).read_bytes()).hexdigest() == want


@needs_records
def test_the_four_earlier_freezes_are_untouched():
  expected_identifier('manifest_sha256.tsv', '14e1fc4c12b6573de9389d9aba079d5fb5f30b290c9303f75c5de4f478b9d8f9')
  expected_identifier('followups34_20261018/manifest_sha256.tsv', 'fc297328ae079155da0f9e2ac0781b5d207f75e1fafd8d5ccfd7f5f7a8edd9c8')
  expected_identifier('feat_20261018/manifest_sha256.tsv', 'a4fb532a2e3ba7a9e691fb35e43feeadd004497af0f6749ff0327039bc5d31bf')
  expected_identifier('followups612_20261108/manifest_sha256.tsv', '689230120499e5ba9d903051553c6554f3412f462456461757398f8fd3560a1a')


@needs_records
@pytest.mark.parametrize('rel,kw', EARLIER)
def test_earlier_session_lists_are_byte_identical_to_their_sessions_tsv(rel, kw):
  # The first two columns of each earlier freeze's sessions.tsv, byte for byte, as the selection writes them.
  want = ''.join('\t'.join(l.split('\t')[:2]) + '\n' for l in (FREEZE / rel).read_text().splitlines()[1:])
  assert ''.join(f'{s}\t{k}\n' for s, k in mf.sessions(**kw)) == want


@needs_records
def test_new_flag_1316_adds_exactly_the_144_sessions_after_the_612_list():
  old, new = mf.sessions(followups612=True), mf.sessions(followups1316=True)
  assert new[:len(old)] == old                                # implies --followups612, in the same order
  added = new[len(old):]
  assert len(added) == 144 == sum(LABELS_1316.values()) and mf.FOLLOWUP1316 == LABELS_1316
  for label, n in LABELS_1316.items():
    assert [s for s, k in added if k == label] == mf.session_list(label) and len(mf.session_list(label)) == n
    assert all(mf.label_of_1316(s) == label for s, k in added if k == label)   # the name carries the label's word
  names = [s for s, _ in new]
  assert len(set(names)) == len(names)                        # no duplicates, no overlap with earlier labels
  assert not {k for _, k in old} & set(LABELS_1316)
  assert not any('_smoke' in s for s, _ in added)
  assert mf.check_local_folders(new) is None                  # every listed session has a local folder


@needs_records
def test_every_unlisted_local_folder_of_the_new_labels_is_explained():
  skipped = mf.not_frozen_1316(mf.sessions(followups1316=True))
  assert {s for s, _, _ in skipped} == set(mf.NOT_FROZEN_1316) and len(skipped) == 20
  assert {k for _, k, _ in skipped} == {'fu13', 'fu14', 'rcx'}
  assert all(mf.label_of_1316(s) == k for s, k, _ in skipped)
  assert mf.label_of_1316('20261014T123400Z_v5e_memlimit_01') is None          # the earlier memlimit label is not memlimit_ah


@needs_records
def test_new_flag_adds_exactly_the_135_sessions_after_the_feat_list():
  feat, new = mf.sessions(feat=True), mf.sessions(followups612=True)
  assert new[:len(feat)] == feat                              # implies --feat, in the same order, so the rows stay a prefix
  added = new[len(feat):]
  assert len(added) == 135 and mf.FOLLOWUP612 == LABELS
  for label, n in LABELS.items():
    assert [s for s, k in added if k == label] == mf.session_list(label) and len(mf.session_list(label)) == n
  names = [s for s, _ in new]
  assert len(set(names)) == len(names)                        # no duplicates, no overlap with earlier labels
  assert not {k for _, k in feat} & set(LABELS)
  for s, k in added:
    assert f'_{k}_' in s and not s.startswith('#')            # the label is the follow-up's own; no smoke session
    assert '_smoke' not in s


def world(tmp_path, monkeypatch, lists, lists1316=None, tags=None):
  """A results tree with sessions.txt lists and no earlier freeze's records (the base list is empty); lists1316 are the
  --followups1316 labels, tags the words their session names carry."""
  lists1316 = lists1316 or {}
  runs = tmp_path / 'scripts' / 'analysis_runs'
  for label, names in {**lists, **lists1316}.items():
    (runs / label).mkdir(parents=True)
    (runs / label / 'sessions.txt').write_text('# header\n\n' + ''.join(f'{n}  # note\n' for n in names))
  monkeypatch.setattr(mf, 'ANALYSIS_RUNS', runs)
  monkeypatch.setattr(mf, 'AF3', tmp_path / 'results' / 'af3')
  monkeypatch.setattr(mf, 'FOLLOWUP612', {k: len(v) for k, v in lists.items()})
  monkeypatch.setattr(mf, 'FOLLOWUP1316', {k: len(v) for k, v in lists1316.items()})
  monkeypatch.setattr(mf, 'NAME_TAG_1316', tags or {k: k for k in lists1316})
  monkeypatch.setattr(mf, 'NOT_FROZEN_1316', {})
  monkeypatch.setattr(mf, 'FINAL_UPLOAD_CUT', {})
  (tmp_path / 'previous.tsv').write_text('session\tpath\tbytes\tsha256\tcrc32c\n')
  monkeypatch.setattr(mf, 'PREVIOUS_1316', tmp_path / 'previous.tsv')
  (tmp_path / 'results' / 'af3').mkdir(parents=True)
  pil = tmp_path / 'p'
  pil.mkdir()
  (pil / 'sessions_pilot2.txt').write_text('')
  (pil / 'sessions_excluded.txt').write_text('')
  (pil / 'sessions_main.txt').write_text('')
  (pil / 'summary.txt').write_text('')
  monkeypatch.setattr(mf, 'PILOT2', pil)
  monkeypatch.setattr(mf, 'MAIN', pil)
  monkeypatch.setattr(mf, 'MEMLIMIT_WRAPPER', pil / 'summary.txt')
  monkeypatch.setattr(mf, 'FOLLOWUP_WRAPPER', pil / 'summary.txt')
  monkeypatch.setattr(mf, 'FOLLOWUP34_WRAPPERS', [])
  monkeypatch.setattr(mf, 'FEAT_WRAPPERS', [])
  monkeypatch.setattr(mf, 'PREFIX_CHECKED', tuple(lists) + tuple(lists1316))
  monkeypatch.setattr(subprocess, 'run', lambda *a, **k: (_ for _ in ()).throw(AssertionError(f'no subprocess: {a}')))
  return tmp_path


def test_session_list_reads_names_only(tmp_path, monkeypatch):
  world(tmp_path, monkeypatch, {'msa': ['S1', 'S2']})
  assert mf.session_list('msa') == ['S1', 'S2']
  assert mf.sessions(followups612=True) == [('S1', 'msa'), ('S2', 'msa')]


def test_a_missing_local_folder_is_an_error(tmp_path, monkeypatch):
  w = world(tmp_path, monkeypatch, {'msa': ['S1', 'S2']})
  (w / 'results' / 'af3' / 'S1').mkdir()
  with pytest.raises(SystemExit, match=r'1 session\(s\).*S2 \(msa\)'):
    mf.check_local_folders(mf.sessions(followups612=True))
  mf.check_local_folders([('gone', 'pilot2, excluded: never ran')])            # an earlier label keeps its record


def test_duplicates_and_other_lengths_are_errors(tmp_path, monkeypatch):
  world(tmp_path, monkeypatch, {'msa': ['S1'], 'sw': ['S1']})
  with pytest.raises(SystemExit, match='already in the freeze or listed twice'):
    mf.sessions(followups612=True)
  world_dir = tmp_path / 'second'
  world_dir.mkdir()
  world(world_dir, monkeypatch, {'msa': ['S1', 'S1']})
  with pytest.raises(SystemExit, match='listed twice'):
    mf.sessions(followups612=True)
  monkeypatch.setattr(mf, 'FOLLOWUP612', {'msa': 3})
  with pytest.raises(SystemExit, match='lists 2 sessions, not 3'):
    mf.sessions(followups612=True)


def nas_world(tmp_path, monkeypatch, good=True):
  d = tmp_path / 'freeze' / 'msa_host_run1'
  d.mkdir(parents=True)
  (d / 'files_sha256.tsv').write_text('path\tbytes\tsha256\na\t1\tx\n')
  ident = hashlib.sha256((d / 'files_sha256.tsv').read_bytes()).hexdigest()
  (d / 'summary.md').write_text(f'# Freeze\n\nMade 2026-10-26T13:40:16+00:00 by x\n\nFreeze identifier: SHA-256 of files_sha256.tsv = `{ident}`\n\n- files: 1, 1 bytes\n')
  monkeypatch.setattr(mf, 'FREEZE_DIR', tmp_path / 'freeze')
  monkeypatch.setattr(mf, 'NAS_FREEZES', {'msa_host_run1': ident if good else '0' * 64})
  return ident


def session_folder(w, name, plan_ok=True):
  s = w / 'results' / 'af3' / name
  (s / 'run1' / 'T1' / 'af3_output' / 'T1').mkdir(parents=True)
  (s / 'final_status.json').write_text('{"exit_code": 0, "final_upload_complete": true}')
  (s / 'plan_runs.jsonl').write_text('{"run": "run1", "seed": 1, "harness_exit_code": 0}\n')
  (s / 'run1' / 'runs.jsonl').write_text('{"pdb_id": "T1", "exit_code": 0}\n')
  t = s / 'run1' / 'T1'
  (t / 'run.json').write_text('{}')
  top = t / 'af3_output' / 'T1'
  for n in ('T1_model.cif', 'T1_ranking_scores.csv', 'T1_summary_confidences.json'):
    (top / n).write_text('x')
  for k in range(5 if plan_ok else 4):
    d = top / f'seed-1_sample-{k}'
    d.mkdir()
    (d / f'T1_seed-1_sample-{k}_model.cif').write_text('x')
    (d / f'T1_seed-1_sample-{k}_summary_confidences.json').write_text('x')
  (s / 'job.log').write_text('line 1\nline 2 appended later\n')
  (s / '.uploaded').write_text('')
  return s


def test_real_mode_labels_the_expected_differences_and_lists_nas(tmp_path, monkeypatch):
  w = world(tmp_path, monkeypatch, {'msa': ['S1']})
  ident = nas_world(tmp_path, monkeypatch)
  s = session_folder(w, 'S1')
  files = sorted(p for p in s.rglob('*') if p.is_file())
  rel = {str(p.relative_to(s)): p for p in files}
  monkeypatch.setattr(mf, 'local_crc32c', lambda folder: {k: 'crc' for k in rel})
  remote = {k: (p.stat().st_size, 'crc') for k, p in rel.items() if k not in ('.uploaded', 'job.log')}
  remote['job.log'] = (7, 'old')
  remote['run1/T1/af3_output/T1/T1_data.json'] = (5, 'd')
  monkeypatch.setattr(mf, 'bucket_objects', lambda b, sess: remote)
  monkeypatch.setattr(mf, 'bucket_copy_is_prefix', lambda b, sess, r, p: True)
  monkeypatch.setattr(mf, 'bucket_sessions', lambda b: ['S1', '20261101T000000Z_l4_msa_smoke_x'])
  out = tmp_path / 'out'
  monkeypatch.setattr(sys, 'argv', ['m', '--bucket', 'gs://b', '--followups612', '--out', str(out)])
  assert mf.main() == 0
  rows = {r['path']: r for r in csv.DictReader(open(out / 'comparison.tsv'), delimiter='\t')}
  assert rows['job.log']['status'] == 'MISMATCH' and rows['job.log']['note'].startswith('expected: appended after the final upload (local copy longer, the bucket copy a byte prefix')
  assert rows['.uploaded']['status'] == 'local only' and rows['.uploaded']['note'].startswith('expected')
  assert rows['run1/T1/af3_output/T1/T1_data.json']['note'].startswith('expected: FETCH=light')
  text = (out / 'summary.md').read_text()
  assert f'| results/freeze/msa_host_run1 | `{ident}` |' in text and 'Flag of this freeze: --followups612' in text
  assert 'not labelled expected above: 0' in text and '20261101T000000Z_l4_msa_smoke_x' in text   # listed, not compared
  assert 'msa: 1 |' in text
  monkeypatch.setattr(mf, 'bucket_copy_is_prefix', lambda b, sess, r, p: False)
  assert mf.main() == 0
  assert 'not labelled expected above: 1' in (out / 'summary.md').read_text()


def test_a_changed_nas_freeze_stops_the_run(tmp_path, monkeypatch):
  nas_world(tmp_path, monkeypatch, good=False)
  with pytest.raises(SystemExit, match='not the identifier'):
    mf.nas_freezes()


def test_local_only_uses_no_gcloud_and_reports_incomplete_folders(tmp_path, monkeypatch):
  w = world(tmp_path, monkeypatch, {'msa': ['S1', 'S2']})
  nas_world(tmp_path, monkeypatch)
  session_folder(w, 'S1')
  session_folder(w, 'S2', plan_ok=False)
  out = tmp_path / 'dry'
  monkeypatch.setattr(sys, 'argv', ['m', '--local_only', '--followups612', '--out', str(out)])
  assert mf.main() == 0                                                         # subprocess.run is refused by world()
  text = (out / 'summary.md').read_text()
  assert 'NOT a freeze' in text and '| msa | 2 |' in text
  check = list(csv.DictReader(open(out / 'session_check.tsv'), delimiter='\t'))
  assert {r['session'] for r in check} == {'S2'} and len(check) == 2           # sample 4: model and summary confidences
  assert all(r.split('\t')[4] == '' for r in (out / 'manifest_sha256.tsv').read_text().splitlines()[1:])
  monkeypatch.setattr(sys, 'argv', ['m', '--local_only', '--followups612', '--out', str(tmp_path / 'freeze' / 'x')])
  with pytest.raises(SystemExit, match='does not write under results/freeze'):
    mf.main()


def test_bucket_or_local_only_is_required(monkeypatch):
  for argv in (['m'], ['m', '--local_only', '--bucket', 'gs://b']):
    monkeypatch.setattr(sys, 'argv', argv)
    with pytest.raises(SystemExit):
      mf.main()


def session_freeze(folder, made, kinds, rows):
  folder.mkdir(parents=True, exist_ok=True)
  (folder / 'manifest_sha256.tsv').write_text(rows)
  ident = hashlib.sha256(rows.encode()).hexdigest()
  (folder / 'sessions.tsv').write_text('session\tkind\n' + ''.join(f'{k}{i}\t{k}\n' for k in kinds for i in range(2)))
  (folder / 'summary.md').write_text(f'# F\n\nMade {made} by x\n\nFreeze identifier: SHA-256 of manifest_sha256.tsv = `{ident}`\n')
  return ident


def test_index_lists_every_freeze_and_names_the_final_one(tmp_path):
  root = tmp_path / 'freeze'
  h = 'session\tpath\n'
  session_freeze(root, '2026-10-15T09:00:00+00:00', ['main'], h + 'a\n')
  session_freeze(root / 'feat_20261018', '2026-10-18T18:00:00+00:00', ['main', 'feat'], h + 'a\nb\n')
  text, sessions, nas, final = fi.build(root)
  assert final is None and 'Final freeze for the paper: none yet' in text
  nas_dir = root / 'msa_host_run1'
  nas_dir.mkdir()
  (nas_dir / 'files_sha256.tsv').write_text('x\n')
  nid = hashlib.sha256(b'x\n').hexdigest()
  (nas_dir / 'summary.md').write_text(f'Made 2026-10-26T13:40:16+00:00 by y\n\nFreeze identifier: SHA-256 of files_sha256.tsv = `{nid}`\n\n- files: 1, 2 bytes\n')
  good = session_freeze(root / 'followups612_20261115', '2026-11-15T10:00:00+00:00', ['main', 'feat'] + list(LABELS), h + 'a\nb\nc\n')
  text, sessions, nas, final = fi.build(root)
  assert final['identifier'] == good and f'**Final freeze for the paper:** `{good}`' in text
  assert [s['flag'] for s in sessions] == ['(none)', '--feat', '--followups612']
  assert [s['holds_previous'] for s in sessions] == ['-', 'yes', 'yes']
  assert nas[0]['identifier'] == nid and nas[0]['check'] == 'ok' and '1, 2 bytes' in text
  (root / 'followups612_20261115' / 'manifest_sha256.tsv').write_text(h + 'changed\n')
  text, sessions, _, final = fi.build(root)
  assert final is None and sessions[-1]['holds_previous'] == 'NO' and 'DIFFERS from summary.md' in text


@needs_records
def test_index_of_the_local_freezes():
  text, sessions, nas, final = fi.build(FREEZE)
  ids = {s['identifier'][:8]: s for s in sessions}
  assert {'14e1fc4c', 'fc297328', 'a4fb532a'} <= set(ids) and all(s['check'] == 'ok' for s in sessions)
  assert ids['14e1fc4c']['flag'] == '(none)' and ids['fc297328']['flag'] == '--followups34' and ids['a4fb532a']['flag'] == '--feat'
  assert ids['fc297328']['holds_previous'] == 'yes' and ids['a4fb532a']['holds_previous'] == 'yes'
  assert {n['identifier'][:8] for n in nas} >= {'3c21b4af', '781f4ee3', 'ed7f634e'}


S13 = '20261110T000000Z_l4_why_gf_01'
S13B = '20261110T000001Z_l4_why_gf_02'
S16 = '20261116T000000Z_v6e_hlo_v6_01'


def world1316(tmp_path, monkeypatch):
  return world(tmp_path, monkeypatch, {'msa': ['M1']}, {'fu13': [S13], 'hlo': [S16]}, {'fu13': 'why', 'hlo': 'hlo'})


def test_1316_selection_and_missing_folder(tmp_path, monkeypatch):
  w = world1316(tmp_path, monkeypatch)
  sel = mf.sessions(followups1316=True)
  assert sel == [('M1', 'msa'), (S13, 'fu13'), (S16, 'hlo')]           # implies --followups612, then the new labels
  assert mf.sessions(followups612=True) == [('M1', 'msa')]              # without the flag, nothing new
  for s in ('M1', S13):
    (w / 'results' / 'af3' / s).mkdir()
  with pytest.raises(SystemExit, match=rf'1 session\(s\).*{S16} \(hlo\)'):
    mf.check_local_folders(sel)


def test_1316_duplicates_with_earlier_labels_and_other_lengths(tmp_path, monkeypatch):
  world(tmp_path, monkeypatch, {'msa': ['M1']}, {'fu13': ['M1']})
  with pytest.raises(SystemExit, match='fu13: sessions already in the freeze'):
    mf.sessions(followups1316=True)
  monkeypatch.setattr(mf, 'FOLLOWUP1316', {'fu13': 2})
  with pytest.raises(SystemExit, match='lists 1 sessions, not 2'):
    mf.sessions(followups1316=True)


def test_unlisted_local_folders_need_a_row(tmp_path, monkeypatch):
  w = world1316(tmp_path, monkeypatch)
  sel = mf.sessions(followups1316=True)
  for s in ('M1', S13, S16, S13B, '20261014T000000Z_v5e_main_01'):
    (w / 'results' / 'af3' / s).mkdir()
  with pytest.raises(SystemExit, match=rf'1 local session folder.*{S13B} \(fu13\)'):
    mf.not_frozen_1316(sel)
  monkeypatch.setattr(mf, 'NOT_FROZEN_1316', {S13B: 'preempted; row of 2026-11-10'})
  assert mf.not_frozen_1316(sel) == [(S13B, 'fu13', 'preempted; row of 2026-11-10')]
  monkeypatch.setattr(mf, 'NOT_FROZEN_1316', {S13: 'contradiction'})
  with pytest.raises(SystemExit, match='both listed and recorded as not frozen'):
    mf.not_frozen_1316(sel)


def test_1316_real_mode_labels_the_cut_upload_and_counts_the_dumps(tmp_path, monkeypatch):
  w = world1316(tmp_path, monkeypatch)
  nas_world(tmp_path, monkeypatch)
  for s in ('M1', S13, S16):
    session_folder(w, s)
  dump = w / 'results' / 'af3' / S16 / 'run1' / 'T1' / 'xla_dump'
  dump.mkdir()
  (dump / 'module_0001.jit_apply_fn.after_optimizations.txt').write_text('HloModule x\n')
  (dump / 'module_0001.jit_apply_fn.before_optimizations.txt').write_text('HloModule\n')
  (w / 'results' / 'af3' / S13B).mkdir()
  monkeypatch.setattr(mf, 'NOT_FROZEN_1316', {S13B: 'preempted; row of 2026-11-10'})
  monkeypatch.setattr(mf, 'FINAL_UPLOAD_CUT', {S13: 'fu13'})
  def local_crc(folder):
    return {str(p.relative_to(folder)): 'crc' for p in folder.rglob('*') if p.is_file()}
  monkeypatch.setattr(mf, 'local_crc32c', local_crc)
  def remote(b, sess):
    folder = w / 'results' / 'af3' / sess
    got = {r: (( folder / r).stat().st_size, c) for r, c in local_crc(folder).items()}
    if sess == S13:
      got.pop('setup.json', None)
      got.pop('run1/runs.jsonl')                             # local only, explained by the cut final upload
    return got
  monkeypatch.setattr(mf, 'bucket_objects', remote)
  monkeypatch.setattr(mf, 'bucket_sessions', lambda b: ['M1', S13, S13B, S16, '20261112T000000Z_a100_f32_af_01'])
  out = tmp_path / 'out'
  monkeypatch.setattr(sys, 'argv', ['m', '--bucket', 'gs://b', '--followups1316', '--out', str(out)])
  assert mf.main() == 0
  rows = {(r['session'], r['path']): r for r in csv.DictReader(open(out / 'comparison.tsv'), delimiter='\t')}
  assert rows[(S13, 'run1/runs.jsonl')]['note'].startswith('expected: the final upload was cut off')
  assert rows[(S16, 'run1/runs.jsonl')]['status'] == 'same'
  text = (out / 'summary.md').read_text()
  assert 'not labelled expected above: 0' in text and '| fu13: 1 |' in text and '| hlo: 1 |' in text
  assert f'| {S13B} | fu13 | preempted; row of 2026-11-10 |' in text
  assert f'| {S16} | 2 | 22 |' in text and 'Flag of this freeze: --followups1316' in text
  assert S13B in text.split('not in the lists above:')[1] and '_f32_af_01' in text   # listed, not compared
  manifest = (out / 'manifest_sha256.tsv').read_text()
  assert 'run1/T1/xla_dump/module_0001.jit_apply_fn.after_optimizations.txt' in manifest   # dumps frozen like any file
  assert S13B not in manifest


def test_1316_local_only(tmp_path, monkeypatch):
  w = world1316(tmp_path, monkeypatch)
  nas_world(tmp_path, monkeypatch)
  for s in ('M1', S13):
    session_folder(w, s)
  session_folder(w, S16, plan_ok=False)
  out = tmp_path / 'dry'
  monkeypatch.setattr(sys, 'argv', ['m', '--local_only', '--followups1316', '--out', str(out)])
  assert mf.main() == 0
  text = (out / 'summary.md').read_text()
  assert '| follow-ups 13 to 16 | 2 |' in text and 'Follow-ups 6 to 16: what looks incomplete' in text
  assert {r['session'] for r in csv.DictReader(open(out / 'session_check.tsv'), delimiter='\t')} == {S16}
  assert f'| {S16} | 0 | 0 |' in text


def test_index_prefers_the_1316_freeze(tmp_path):
  root = tmp_path / 'freeze'
  h = 'session\tpath\n'
  session_freeze(root, '2026-10-15T09:00:00+00:00', ['main'], h + 'a\n')
  old = session_freeze(root / 'followups612_20261108', '2026-11-08T16:00:00+00:00', ['main'] + list(LABELS), h + 'a\nb\n')
  some = session_freeze(root / 'partial', '2026-11-09T16:00:00+00:00', ['main', 'fu13'], h + 'a\nb\nc\n')
  text, sessions, _, final = fi.build(root)
  assert final['identifier'] == old and sessions[-1]['flag'].startswith('unknown')
  (root / 'partial' / 'sessions.tsv').unlink()
  (root / 'partial' / 'manifest_sha256.tsv').unlink()
  new = session_freeze(root / 'followups1316_20261118', '2026-11-18T10:00:00+00:00',
                       ['main'] + list(LABELS) + list(LABELS_1316), h + 'a\nb\nc\n')
  text, sessions, _, final = fi.build(root)
  assert final['identifier'] == new and f'**Final freeze for the paper:** `{new}`' in text
  assert [s['flag'] for s in sessions] == ['(none)', '--followups612', '--followups1316']
  assert [s['holds_previous'] for s in sessions] == ['-', 'yes', 'yes']
  assert 'followups1316_20261118 (final)' in text


@needs_records
def test_index_of_the_local_freezes_is_unchanged_until_the_new_freeze():
  text, sessions, _, final = fi.build(FREEZE)
  if not any(s['flag'] == '--followups1316' for s in sessions):
    assert final['identifier'].startswith('68923012') and text == (FREEZE / 'INDEX.md').read_text()


def test_files_added_changed_or_removed_since_the_previous_freeze(tmp_path):
  prev = tmp_path / 'followups612_x' / 'manifest_sha256.tsv'
  prev.parent.mkdir()
  prev.write_text('session\tpath\tbytes\tsha256\tcrc32c\nA\ta.cif\t1\taa\t\nA\tb.cif\t2\tbb\t\nB\tc.cif\t3\tcc\t\n')
  same = [('A', 'a.cif', 1, 'aa', ''), ('A', 'b.cif', 2, 'bb', ''), ('B', 'c.cif', 3, 'cc', ''), ('N', 'new.cif', 9, 'nn', '')]
  text = '\n'.join(mf.since_previous(same, prev))
  assert 'added 0 (0 bytes), changed 0, removed 0' in text and 'CHANGED' not in text      # a new session is not "added"
  rows = same[:2] + [('A', 'r/T_data.json', 5, 'dd', ''), ('A', 'r/T_seed-1_sample-0_confidences.json', 7, 'ee', '')]
  text = '\n'.join(mf.since_previous(rows, prev))
  assert 'added 2 (12 bytes), changed 0, removed 1: CHANGED OR REMOVED' in text
  assert '| added | A | _data.json | 1 |' in text and '| added | A | _confidences.json | 1 |' in text
  assert '| removed | B | c.cif | 1 |' in text
  rows = [('A', 'a.cif', 1, 'zz', '')] + same[1:]
  assert 'changed 1' in '\n'.join(mf.since_previous(rows, prev))


def test_index_accepts_files_added_later_to_earlier_sessions():
  h = b'session\tpath\n'
  old = h + b'A\ta\nB\tb\n'
  assert fi.holds(old, old + b'C\tc\n') == 'yes'
  assert fi.holds(old, h + b'A\ta\nA\ta2\nB\tb\nC\tc\n') == 'yes, plus 1 files added later to its sessions'
  assert fi.holds(old, h + b'A\ta\nB\tb changed\nC\tc\n') == 'NO'
  assert fi.holds(old, h + b'B\tb\nA\ta\n') == 'NO'                                  # order kept
