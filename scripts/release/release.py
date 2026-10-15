"""Data release for Zenodo: inventory, layout, input check, archive building and verification (prepare and dry-run
only; this script never contacts a bucket or Zenodo). Build plan: scripts/release/RUNBOOK.md.

    python3 scripts/release/release.py inventory [--listing results/release/inventory/bucket_listing.txt]
    python3 scripts/release/release.py layout
    python3 scripts/release/release.py check-inputs --root DIR --out DIR [--sessions S1,S2] [--repo .]
    python3 scripts/release/release.py build --root DIR --out DIR [--archives A,B] [--sessions S1,S2] [--repo .]
    python3 scripts/release/release.py sums --out DIR
    python3 scripts/release/release.py verify --out DIR

Sessions (classify()): the data freeze's sessions (results/freeze/sessions.tsv: main, pilot2, memlimit, ppi, embed);
"superseded" sessions with the official weights that no reported result uses (SUPERSEDED, each with its section 10
row; released apart in R2, af3_superseded_engineering.tar.gz, after check_official_weights()); "attempt" sessions that
never ran (launcher records only; released in R1); everything else stays out (random-weight engineering sessions).
The freeze's unexplained files stay out, except launcher records of attempts that never ran.
inventory: sorts every object of a read-only bucket listing (`gcloud storage ls -l -r 'gs://BUCKET/af3/**'`) or, with
--root, of a local copy, by session kind and category (category()). Writes inventory/{summary.csv, sessions.csv,
excluded_sessions.csv, excluded_files.csv, inventory.md}.
layout: the archives and records of ARCHIVES with file counts, raw and estimated compressed sizes (gzip ratios
measured on local samples, RATIO), against Zenodo's per-record limits. Writes layout.csv, layout.md.
check-inputs: compares every released process's *_data.json with its frozen input (compare_data_json() says what is
compared) and writes inputs_check.csv and inputs_check.json into --out; exit 2 on any difference. The R2 build
refuses to start unless this passed on the same root.
build: writes the archives from --root (a folder holding <session>/ folders: the local freeze copy results/af3, or a
copy of the bucket's af3/ prefix) and from the repository (--repo), into <out>/<record>/: deterministic tar (sorted
members, fixed mtime, owner and mode; gzip header without time or name), one SHA-256 manifest per archive
(<archive>.sha256, `sha256sum -c` format), then sums. Every candidate file passes guard() first; any refusal stops
the build and removes what it wrote. --sessions limits a build to some sessions (a test subset).
sums: copies scripts/release/records/<record>/README.md (and, for R1, this script, RUNBOOK.md, the inventory, the
layout, the exclusion lists and SUPERSEDED_SESSIONS.csv under release/) into each record folder and writes its
SHA256SUMS; run it after archives built elsewhere are copied in.
verify: recomputes SHA256SUMS and every member's SHA-256 from the archives.

guard() refuses: a file named af3.bin or af3.bin.zst, or any *.bin / *.bin.zst; a file whose SHA-256 is the
weights' (WEIGHTS_SHA256) or whose size is the weights' (WEIGHTS_BYTES); any file over 1 GiB (none is expected among
the members); a text file that names a gs:// bucket other than ALLOWED_BUCKETS; a text file that contains an email
address, unless it is on the redaction list (REDACT): those are packaged as a copy with every address replaced by
<account> (the files in results/ are not changed), and each redaction is recorded in <record>/REDACTIONS.tsv
(archive, path, original and released SHA-256, what was replaced). Text that records the weights'
SHA-256 or file name (session.json, run.json, weights.json) is provenance and stays in. After a build, any file over
1 GiB in --out that is not one of the expected archives is refused too. Standard library only.
"""
import argparse
import csv
import gzip
import hashlib
import io
import json
import os
import pathlib
import re
import sys
import tarfile

HERE = pathlib.Path(__file__).resolve().parent            # scripts/release: this script, records/<R>/README.md
REPO = HERE.parents[1]
OUT_DIR = REPO / 'results' / 'release'   # inventory and layout outputs
FREEZE = REPO / 'results' / 'freeze'
FREEZE_ID = '14e1fc4c12b6573de9389d9aba079d5fb5f30b290c9303f75c5de4f478b9d8f9'
WEIGHTS_SHA256 = '74d0258616917cd122f5eab6d076afe4a8930e96823851e65e4f777dfb1f33ff'
WEIGHTS_BYTES = 1020545840
WEIGHTS_NAMES = {'af3.bin', 'af3.bin.zst'}
MAX_MEMBER_BYTES = 1 << 30
ALLOWED_BUCKETS = {'af2-tpu-benchmark-af3-results'}
TEXT_SUFFIXES = {'.json', '.jsonl', '.txt', '.log', '.md', '.csv', '.tsv', '.out', '.sh', '.py', '.cif', ''}
MTIME = 1792022400   # 2026-10-15T00:00:00Z, every member's mtime
ZENODO_MAX_BYTES, ZENODO_MAX_FILES = 50 * 10 ** 9, 100
# gzip -6 ratios measured on 12 random local files per category (2026-10-15); embeddings do not compress.
RATIO = {'model_cif': 0.255, 'ranking_summary': 0.063, 'full_confidences': 0.124, 'terms': 0.35, 'af3_logs': 0.067,
         'records': 0.15, 'embeddings': 1.0, 'repo': 0.331}


def category(path):
  """The release category of a path inside a session folder."""
  name = path.rsplit('/', 1)[-1]
  if '/work/' in f'/{path}' or name in {'.uploaded', '.wrapper_done'}:
    return 'work'               # local scratch (inputs with seeds, compile caches): never released
  if name.endswith('_data.json'):
    return 'data_json'
  if name.endswith('_embeddings.npz'):
    return 'embeddings'
  if name.endswith('_model.cif'):
    return 'model_cif'
  if name.endswith(('_summary_confidences.json', '_ranking_scores.csv')):
    return 'ranking_summary'
  if name.endswith('_confidences.json'):
    return 'full_confidences'
  if name == 'TERMS_OF_USE.md':
    return 'terms'
  if name == 'run_alphafold.log':
    return 'af3_logs'
  if name.endswith(('.tgz', '.tar', '.gz', '.zst')):
    return 'other_binary'
  return 'records'              # harness and launcher records and logs (JSON, text)


# ---------------------------------------------------------------- the data freeze

def freeze_sessions():
  """{session: kind} of the data freeze."""
  out = {}
  for r in csv.DictReader(open(FREEZE / 'sessions.tsv'), delimiter='\t'):
    out[r['session']] = 'pilot2, excluded' if r['kind'].startswith('pilot2, excluded') else r['kind']
  return out


def unexplained_files():
  """{(session, path): status} the freeze could not explain (excluded from the release)."""
  out = {}
  for r in csv.DictReader(open(FREEZE / 'comparison.tsv'), delimiter='\t'):
    if r['status'] != 'same' and not r['note'].startswith('expected'):
      out[(r['session'], r['path'])] = r['status']
  return out


# Sessions outside the freeze that used the official weights and are released, apart, as "superseded or
# engineering, not used in any reported result" (decision of 2026-10-15), each with the section 10 row that records
# it. Four of them are not named in section 10; their row is the one that records their arm or check.
P0_PROV = '2026-10-10, Provenance of the pilot sessions of 2026-10-09 (first pilot, superseded by the repeated pilot: 2026-10-11, Scope of the repeated pilot)'
P0_L4 = '2026-10-10, L4 arm (G) of the pilot (first pilot, superseded by the repeated pilot: 2026-10-11, Scope of the repeated pilot)'
P0_WARM = '2026-10-10, V5 warm-cache check (section 6, check 2) (first pilot; session not named in the row)'
P0_H = '2026-10-10, Numeric precision: section 11 added (first-pilot V6h and Gh arms; session not named in the row)'
SUPERSEDED = {
    '20261009T154308Z_v5e_pilot': P0_PROV, '20261009T154318Z_v6e_pilot': P0_PROV,
    '20261009T154331Z_cpu_pilot_cpu_1': P0_PROV, '20261009T172653Z_cpu_pilot_cpu_2': P0_PROV,
    '20261009T172743Z_cpu_pilot_cpu_3': P0_PROV, '20261009T172752Z_cpu_pilot_cpu_4': P0_PROV,
    '20261009T230055Z_l4_pilot_l4_1': P0_L4, '20261010T013052Z_l4_pilot_l4_2': P0_L4,
    '20261010T134201Z_v5e_v5e_warm_completion': P0_WARM,
    '20261010T134219Z_v6e_pilot_v6h': P0_H, '20261010T134240Z_l4_pilot_l4h_1': P0_H, '20261010T150023Z_l4_pilot_l4h_2': P0_H,
    '20261010T215557Z_l4_diag_l4_determinism': '2026-10-11, L4 determinism diagnostic, and Result of the L4 determinism diagnostic (label diag)',
    '20261013T111330Z_l4_gt_smoke': '2026-10-12 and 2026-10-13, GT smoke test (label smoke, engineering check only)',
    '20261012T052814Z_l4_pilot_l4d_2': '2026-10-12, Repeated pilot, launch incidents (interrupted mid-run, replaced by 20261012T065347Z_l4_pilot_l4d_2; pilot2, excluded)',
}
RELEASED_KINDS = ('main', 'pilot2', 'memlimit', 'ppi', 'embed', 'superseded', 'attempt')


def classify(session, paths, kinds):
  """(kind, reason): a freeze kind, 'superseded' (SUPERSEDED), 'attempt' (no VM run records: launcher records
  only, released in R1), or 'out' (not released) with the reason."""
  ran = any(p.endswith(('run.json', 'runs.jsonl')) for p in paths)
  if session in SUPERSEDED:
    return 'superseded', SUPERSEDED[session]
  if not ran:
    return 'attempt', 'attempt that never ran (no VM run records); launcher records released in R1'
  if session in kinds:
    return kinds[session], 'data freeze'
  if session < '20261009T15':
    return 'out', 'engineering session before the study (random weights, probes, stack validation)'
  return 'out', 'run outside the study, not in the freeze and not listed as superseded'


# ---------------------------------------------------------------- inventory

def read_listing(path):
  """[(session, path, bytes)] from a `gcloud storage ls -l -r` listing."""
  rows = []
  for line in open(path):
    parts = line.split()
    if len(parts) != 3 or not parts[0].isdigit():
      continue
    rel = parts[2].split('/af3/', 1)[1]
    session, _, p = rel.partition('/')
    rows.append((session, p, int(parts[0])))
  return rows


def walk(root):
  rows = []
  for s in sorted(d for d in pathlib.Path(root).iterdir() if d.is_dir() and not d.name.startswith('.')):
    for p in sorted(s.rglob('*')):
      if p.is_file():
        rows.append((s.name, str(p.relative_to(s)), p.stat().st_size))
  return rows


def inventory(args):
  rows = walk(args.root) if args.root else read_listing(args.listing)
  kinds, odd = freeze_sessions(), unexplained_files()
  out = pathlib.Path(args.out)
  out.mkdir(parents=True, exist_ok=True)
  summary, per_session, excluded_files = {}, {}, []
  by_session = {}
  for s, p, b in rows:
    by_session.setdefault(s, []).append((p, b))
  excluded_sessions = []
  for s, items in sorted(by_session.items()):
    kind, reason = classify(s, [p for p, _ in items], kinds)
    if kind == 'out':
      excluded_sessions.append({'session': s, 'files': len(items), 'bytes': sum(b for _, b in items), 'reason': reason})
      continue
    for p, b in items:
      if (s, p) in odd and kind != 'attempt':
        excluded_files.append({'session': s, 'path': p, 'bytes': b, 'freeze_status': odd[(s, p)]})
        continue
      c = category(p)
      k = (kind, c)
      summary[k] = [summary.get(k, [0, 0])[0] + 1, summary.get(k, [0, 0])[1] + b]
      ps = per_session.setdefault(s, {'session': s, 'kind': kind})
      ps[f'{c}_files'] = ps.get(f'{c}_files', 0) + 1
      ps[f'{c}_bytes'] = ps.get(f'{c}_bytes', 0) + b
  # freeze files absent from this listing (pilot2 excluded sessions without a bucket folder, local-only files)
  missing = sorted(set(kinds) - set(by_session))
  for s in missing:
    excluded_sessions.append({'session': s, 'files': 0, 'bytes': 0, 'reason': f'in the freeze ({kinds[s]}) but absent from this listing'})
  write_csv(out / 'summary.csv', [{'kind': k, 'category': c, 'files': v[0], 'bytes': v[1]} for (k, c), v in sorted(summary.items())])
  cats = sorted({c for _, c in summary})
  write_csv(out / 'sessions.csv', [{'session': r['session'], 'kind': r['kind'],
                                    **{f'{c}_{x}': r.get(f'{c}_{x}', 0) for c in cats for x in ('files', 'bytes')}}
                                   for r in per_session.values()])
  write_csv(out / 'excluded_sessions.csv', excluded_sessions)
  write_csv(out / 'excluded_files.csv', excluded_files)
  kinds_order = ['main', 'pilot2', 'memlimit', 'ppi', 'embed', 'superseded', 'attempt']
  lines = ['# Results bucket inventory', '', f'Source: {args.root or args.listing} ({len(rows)} objects, '
           f'{sum(b for *_, b in rows) / 2 ** 30:.2f} GiB). Sessions and kinds from the data freeze `{FREEZE_ID}`.', '',
           '| kind | sessions | ' + ' | '.join(cats) + ' | total |', '|---|---|' + '---|' * (len(cats) + 1)]
  for k in kinds_order:
    n = sum(1 for r in per_session.values() if r['kind'] == k)
    cells = [summary.get((k, c), [0, 0]) for c in cats]
    lines.append(f'| {k} | {n} | ' + ' | '.join(f'{f} / {gib(b)}' for f, b in cells) +
                 f' | {sum(f for f, _ in cells)} / {gib(sum(b for _, b in cells))} |')
  lines += ['', 'Cells: files / GiB.', '', f'## Excluded sessions ({len(excluded_sessions)})', '',
            '| session | files | GiB | reason |', '|---|---|---|---|']
  lines += [f'| {r["session"]} | {r["files"]} | {gib(r["bytes"])} | {r["reason"]} |' for r in excluded_sessions]
  lines += ['', f'## Excluded files: the freeze\'s unexplained differences ({len(excluded_files)})', '',
            '| session | path | bytes | freeze status |', '|---|---|---|---|']
  lines += [f'| {r["session"]} | {r["path"]} | {r["bytes"]} | {r["freeze_status"]} |' for r in excluded_files]
  not_in_listing = [k for k in odd if k[0] in by_session and k[1] not in {p for p, _ in by_session[k[0]]}]
  lines += ['', f'Unexplained freeze files not in this listing (local only, so not in any bucket-built archive): '
            f'{len(not_in_listing)}', '']
  (out / 'inventory.md').write_text('\n'.join(lines) + '\n')
  print('\n'.join(lines[:12 + len(kinds_order)]))
  return 0


def gib(b):
  return f'{b / 2 ** 30:.2f}'


def write_csv(path, rows):
  with open(path, 'w', newline='') as f:
    if rows:
      w = csv.DictWriter(f, list(rows[0]), lineterminator='\n')
      w.writeheader()
      w.writerows(rows)
    else:
      f.write('')


# ---------------------------------------------------------------- layout

STRUCTURES = ['model_cif', 'ranking_summary', 'terms']
# (record, archive, kinds, categories); repo archives list repository paths instead of kinds.
ARCHIVES = [
    ('R1', 'inputs_targets_manifests.tar.gz', 'repo', ['data/inputs/*.json', 'data/inputs/ppi/*.json', 'inputs/**',
                                                       'targets/**']),
    ('R1', 'freeze.tar.gz', 'repo', ['results/freeze/manifest_sha256.tsv', 'results/freeze/sessions.tsv',
                                     'results/freeze/bucket_objects.tsv', 'results/freeze/comparison.tsv',
                                     'results/freeze/summary.md', 'scripts/freeze/make_freeze_manifest.py',
                                     'scripts/freeze/reproduce.sh']),
    ('R1', 'session_lists_launch_records.tar.gz', 'repo', ['results/analysis/**/sessions_*.txt',
                                                          'results/analysis/main_monitor_20261013/ccd_codes.json',
                                                          'results/parts_*.txt', 'results/af3/.many/**',
                                                          'results/af3/.launcher/**']),
    ('R1', 'session_records.tar.gz', ['main', 'pilot2', 'memlimit', 'ppi', 'embed'], ['records']),
    ('R1', 'attempt_superseded_records.tar.gz', ['superseded', 'attempt'], ['records']),
    ('R2', 'af3_main_structures.tar.gz', ['main'], STRUCTURES),
    ('R2', 'af3_main_full_confidences.tar.gz', ['main'], ['full_confidences']),
    ('R2', 'af3_pilot2_structures.tar.gz', ['pilot2'], STRUCTURES),
    ('R2', 'af3_pilot2_full_confidences.tar.gz', ['pilot2'], ['full_confidences']),
    ('R2', 'af3_memlimit.tar.gz', ['memlimit'], STRUCTURES + ['full_confidences']),
    ('R2', 'af3_ppi_structures.tar.gz', ['ppi'], STRUCTURES),
    ('R2', 'af3_ppi_full_confidences.tar.gz', ['ppi'], ['full_confidences']),
    ('R2', 'af3_embed_structures.tar.gz', ['embed'], STRUCTURES + ['full_confidences']),
    ('R2', 'af3_embed_embeddings.tar', ['embed'], ['embeddings']),
    ('R2', 'af3_logs.tar.gz', ['main', 'pilot2', 'memlimit', 'ppi', 'embed'], ['af3_logs']),
    ('R2', 'af3_superseded_engineering.tar.gz', ['superseded'], STRUCTURES + ['full_confidences', 'af3_logs', 'embeddings']),
    # derived analysis outputs carry per-sample values computed from AlphaFold3 outputs: R2, under the Output Terms
    ('R2', 'analysis_outputs.tar.gz', 'repo', ['results/analysis/**', 'results/paper_assets/**']),
]
# Where each archive is built (decision of 2026-10-15): R1 from the repository and the local freeze copy on the
# laptop; the rest from a copy of the bucket on a VM in the bucket's region (scripts/release/RUNBOOK.md).
BUILT_ON = {'inputs_targets_manifests.tar.gz': 'laptop', 'freeze.tar.gz': 'laptop',
            'session_lists_launch_records.tar.gz': 'laptop', 'session_records.tar.gz': 'laptop',
            'analysis_outputs.tar.gz': 'laptop'}
SUPERSEDED_LABEL = 'superseded or engineering, not used in any reported result'
LEFT_OUT = {'data_json': 'AlphaFold3\'s copy of each run\'s input; check-inputs compares every one with its frozen '
                         'input on the build VM and the R2 build refuses to start unless all match',
            'work': 'local scratch (per-run inputs with seeds, compile caches); not in the bucket',
            'other_binary': 'engineering dumps (XLA and TPU log archives), none in a study session'}
RECORDS = {'R1': 'Inputs, targets, manifests, data freeze, session lists and run records (CC BY 4.0)',
           'R2': 'AlphaFold3 outputs and the analysis outputs derived from them (licence "Other": the AlphaFold 3 '
                 'Output Terms of Use; no Creative Commons)'}


def layout(args):
  summary = list(csv.DictReader(open(pathlib.Path(args.inventory) / 'summary.csv')))
  rows = []
  for record, name, kinds, cats in ARCHIVES:
    if kinds == 'repo':
      files = [p for pat in cats for p in repo_files(REPO, pat)]
      raw = sum(p.stat().st_size for p in files)
      rows.append({'record': record, 'archive': name, 'contents': ', '.join(cats), 'files': len(files), 'raw_bytes': raw,
                   'estimated_bytes': int(raw * RATIO['repo'])})
      continue
    sel = [r for r in summary if r['kind'] in kinds and r['category'] in cats]
    raw = sum(int(r['bytes']) for r in sel)
    est = sum(int(int(r['bytes']) * RATIO[r['category']]) for r in sel)
    rows.append({'record': record, 'archive': name, 'contents': f'{", ".join(kinds)}: {", ".join(cats)}',
                 'files': sum(int(r['files']) for r in sel), 'raw_bytes': raw, 'estimated_bytes': est})
  out = pathlib.Path(args.out)
  write_csv(out / 'layout.csv', rows)
  left = {}
  for r in summary:
    if r['category'] in LEFT_OUT:
      left.setdefault(r['category'], [0, 0])
      left[r['category']][0] += int(r['files'])
      left[r['category']][1] += int(r['bytes'])
  lines = ['# Release layout (proposal)', '', f'Data freeze `{FREEZE_ID}`. Zenodo limits per record: 50 GB and 100 files.', '']
  for rec, title in RECORDS.items():
    g = [r for r in rows if r['record'] == rec]
    n_files = 2 * len(g) + 2 + (13 if rec == 'R1' else 0)   # archives, manifests, README.md, SHA256SUMS; R1: release/ (12), REDACTIONS.tsv
    est = sum(r['estimated_bytes'] for r in g)
    lines += [f'## {rec}: {title}', '', f'{len(g)} archives, {n_files} files with manifests, README.md, SHA256SUMS '
              f'and the terms or licence; estimated {est / 1e9:.1f} GB '
              f'({"within" if est < ZENODO_MAX_BYTES and n_files <= ZENODO_MAX_FILES else "OVER"} the limits).', '',
              '| archive | contents | files | raw GiB | estimated GiB |', '|---|---|---|---|---|']
    lines += [f'| {r["archive"]} | {r["contents"]} | {r["files"]} | {gib(r["raw_bytes"])} | {gib(r["estimated_bytes"])} |'
              for r in g]
    lines.append('')
  lines += ['## Left out', '', '| category | files | GiB | why |', '|---|---|---|---|']
  lines += [f'| {c} | {v[0]} | {gib(v[1])} | {LEFT_OUT[c]} |' for c, v in sorted(left.items())]
  lines += ['', 'Also left out: random-weight engineering sessions and other runs outside the freeze, the freeze\'s unexplained files '
            'except launcher records of attempts that never ran (inventory/excluded_sessions.csv, inventory/excluded_files.csv), and the AlphaFold3 model weights, '
            'which are never packaged (guard() in release.py).', '']
  (out / 'layout.md').write_text('\n'.join(lines) + '\n')
  print('\n'.join(lines))
  return 0


def repo_files(repo, pattern):
  """Files matching pattern under repo. A pattern ending in '/**' means every file below that folder: pathlib's glob
  matches only folders with a trailing '**' before Python 3.13, so rglob is used for it (same result on every
  version)."""
  base = pathlib.Path(repo)
  found = (base / pattern[:-3]).rglob('*') if pattern.endswith('/**') else base.glob(pattern)
  return sorted(p for p in found if p.is_file() and '__pycache__' not in p.parts and not p.name.startswith('.'))


# ---------------------------------------------------------------- guard

class Refused(Exception):
  pass


GS_URI = re.compile(rb'gs://([a-z0-9][a-z0-9._-]{1,220})')
EMAIL = re.compile(rb'[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}')
REDACTED_EMAIL = b'<account>'
# The only packaged files allowed to contain an email address: a copy of each is packaged with every address replaced
# by <account>, and the redaction recorded in REDACTIONS.tsv of the record. Keys are archive member paths. The files in
# results/ are never changed. Found by scanning every releasable local text file (2026-10-15): the gcloud account in
# the "authenticated as" lines of two wrapper logs.
REDACT = {
    'results/af3/.many/20261014T123400Z/memlimit_v5.log': 'gcloud account email, in the lines that say authenticated as',
    'results/af3/.many/20261014T123400Z/memlimit_v5.attempt2.log': 'gcloud account email, in the lines that say authenticated as',
}
REDACTIONS_HEADER = ['archive', 'path', 'original_sha256', 'released_sha256', 'replaced']


def sha256_file(path):
  h = hashlib.sha256()
  with open(path, 'rb') as f:
    while block := f.read(1 << 22):
      h.update(block)
  return h.hexdigest()


def guard(path, rel):
  """(size, SHA-256) of a file to package, or raise Refused. A text file that contains an email address is refused
  unless it is on the redaction list (REDACT), whose files prepare() packages redacted."""
  name = rel.rsplit('/', 1)[-1].lower()
  if name in WEIGHTS_NAMES or name.endswith(('.bin', '.bin.zst')):
    raise Refused(f'{rel}: the name of the model weights')
  size = os.path.getsize(path)
  if size == WEIGHTS_BYTES:
    raise Refused(f'{rel}: the size of the model weights ({WEIGHTS_BYTES} bytes)')
  if size > MAX_MEMBER_BYTES:
    raise Refused(f'{rel}: {size} bytes, over 1 GiB')
  sha = sha256_file(path)
  if sha == WEIGHTS_SHA256:
    raise Refused(f'{rel}: the SHA-256 of the model weights')
  if pathlib.Path(name).suffix in TEXT_SUFFIXES:
    data = pathlib.Path(path).read_bytes()
    others = {m.decode() for m in GS_URI.findall(data)} - ALLOWED_BUCKETS
    if others:
      raise Refused(f'{rel}: names a bucket other than the results bucket ({", ".join(sorted(others))})')
    if rel not in REDACT and EMAIL.search(data):
      raise Refused(f'{rel}: contains an email address and is not on the redaction list')
  return size, sha


def prepare(path, rel, archive):
  """(size, SHA-256, bytes or None, redaction row or None) of a file to package. Files on REDACT are packaged as a
  copy with every email address replaced by <account> (the source file is not changed)."""
  size, sha = guard(path, rel)
  if rel not in REDACT:
    return size, sha, None, None
  original = pathlib.Path(path).read_bytes()
  found = EMAIL.findall(original)
  if not found:
    raise Refused(f'{rel}: on the redaction list but contains no email address (stale list)')
  data = EMAIL.sub(REDACTED_EMAIL, original)
  if EMAIL.search(data):
    raise Refused(f'{rel}: an email address survived the redaction')
  new_sha = hashlib.sha256(data).hexdigest()
  return len(data), new_sha, data, {
      'archive': archive, 'path': rel, 'original_sha256': sha, 'released_sha256': new_sha,
      'replaced': f'{len(found)} occurrence(s) of {len(set(found))} email address(es) replaced by <account> '
                  f'({REDACT[rel]})'}


def write_redactions(d, rows):
  """Adds rows to <record>/REDACTIONS.tsv (one row per redacted file; rows of other builds kept)."""
  f = d / 'REDACTIONS.tsv'
  old = list(csv.DictReader(open(f), delimiter='\t')) if f.is_file() else []
  keep = {(r['archive'], r['path']): r for r in old}
  keep.update({(r['archive'], r['path']): r for r in rows})
  with open(f, 'w', newline='') as h:
    w = csv.DictWriter(h, REDACTIONS_HEADER, delimiter='\t', lineterminator='\n')
    w.writeheader()
    w.writerows(sorted(keep.values(), key=lambda r: (r['archive'], r['path'])))


# ---------------------------------------------------------------- build

LEGALLY_BINDING = '''By using this information, you agree to AlphaFold 3 Output Terms of Use found at
https://github.com/google-deepmind/alphafold3/blob/main/OUTPUT_TERMS_OF_USE.md.

To request access to the AlphaFold 3 model parameters, follow the process set out at
https://github.com/google-deepmind/alphafold3. You may only use these if received directly from Google. Use is
subject to terms of use available at https://github.com/google-deepmind/alphafold3/blob/main/WEIGHTS_TERMS_OF_USE.md.
'''


def session_kinds(root, sessions_filter=None):
  """{session: kind} of every session folder under root (classify(): freeze kind, superseded, attempt or out)."""
  freeze, out = freeze_sessions(), {}
  for d in sorted(x for x in pathlib.Path(root).iterdir() if x.is_dir() and not x.name.startswith('.')):
    if sessions_filter and d.name not in sessions_filter:
      continue
    out[d.name] = classify(d.name, [str(p.relative_to(d)) for p in d.rglob('*') if p.is_file()], freeze)[0]
  return out


def check_official_weights(session_dir):
  """A superseded or engineering session is released only if every run used the official weights."""
  shas = {(json.loads(f.read_text()).get('weights') or {}).get('sha256') for f in session_dir.glob('*/session.json')}
  if shas != {WEIGHTS_SHA256}:
    raise Refused(f'{session_dir.name}: not every run used the official weights ({sorted(map(str, shas))}); '
                  'random-weight sessions stay out')


def members(spec, root, repo, sessions_filter, kinds_of, odd):
  """[(archive member path, source path)] of one archive, sorted."""
  _, name, kinds, cats = spec
  out = []
  if kinds == 'repo':
    for pat in cats:
      out += [(str(p.relative_to(repo)), p) for p in repo_files(repo, pat)]
    return sorted(set(out))
  for s in sorted(kinds_of):
    if kinds_of[s] not in kinds or (sessions_filter and s not in sessions_filter):
      continue
    if kinds_of[s] == 'superseded' and any(c != 'records' for c in cats):
      check_official_weights(pathlib.Path(root) / s)
    for p in sorted((pathlib.Path(root) / s).rglob('*')):
      rel = str(p.relative_to(pathlib.Path(root) / s))
      # the freeze's unexplained files stay out, except launcher records of attempts that never ran
      if p.is_file() and category(rel) in cats and ((s, rel) not in odd or kinds_of[s] == 'attempt'):
        out.append((f'af3/{s}/{rel}', p))
  return out


# ---------------------------------------------------------------- *_data.json against the frozen inputs

INPUT_DEFAULTS = {'modifications': []}   # fields AlphaFold3 adds to a sequence entry, with the value it adds
TOP_LEVEL_SAME = ('name', 'dialect')
TOP_LEVEL_EMPTY = ('bondedAtomPairs', 'userCCD')


def compare_data_json(inp, dj, seeds):
  """Differences between a frozen input and AlphaFold3's *_data.json of one process (empty list: same). Compared:
  name and dialect; modelSeeds of the data JSON equal to the process's seeds and among the input's; bondedAtomPairs
  and userCCD absent or empty in both; the same number and order of sequence entries, each of the same type (protein,
  ligand, ...) with every field equal (id, sequence, unpairedMsa, pairedMsa, templates, ccdCodes, ...), a field
  missing from the input being allowed only with AlphaFold3's default (modifications: []). version is not compared
  (AlphaFold3 writes its own dialect version)."""
  diff = []
  for k in TOP_LEVEL_SAME:
    if inp.get(k) != dj.get(k):
      diff.append(f'{k}: {inp.get(k)!r} != {dj.get(k)!r}')
  if list(dj.get('modelSeeds') or []) != list(seeds) or not set(seeds) <= set(inp.get('modelSeeds') or []):
    diff.append(f'modelSeeds: data {dj.get("modelSeeds")}, process {seeds}, input {inp.get("modelSeeds")}')
  for k in TOP_LEVEL_EMPTY:
    if inp.get(k) or dj.get(k):
      diff.append(f'{k}: input {inp.get(k)!r}, data {dj.get(k)!r}')
  a, b = inp.get('sequences') or [], dj.get('sequences') or []
  if len(a) != len(b):
    return diff + [f'sequences: {len(a)} entries in the input, {len(b)} in the data JSON']
  for i, (ea, eb) in enumerate(zip(a, b)):
    if list(ea) != list(eb):
      diff.append(f'sequence {i}: type {list(ea)} != {list(eb)}')
      continue
    (t, va), (_, vb) = next(iter(ea.items())), next(iter(eb.items()))
    for k in sorted(set(va) | set(vb)):
      if k not in va and k in INPUT_DEFAULTS and vb.get(k) == INPUT_DEFAULTS[k]:
        continue
      if va.get(k) != vb.get(k):
        diff.append(f'sequence {i} ({t} {va.get("id")}): {k} differs')
  return diff


def check_inputs(args):
  """Every released process's *_data.json against its frozen input (input_path in runs.jsonl, whose SHA-256 must be
  the recorded input_sha256). Writes <out>/inputs_check.csv and inputs_check.json; exit 2 on any difference."""
  root, repo, out = pathlib.Path(args.root), pathlib.Path(args.repo), pathlib.Path(args.out)
  out.mkdir(parents=True, exist_ok=True)
  sessions_filter = set(args.sessions.split(',')) if args.sessions else None
  kinds_of = session_kinds(root, sessions_filter)
  rows, inputs = [], {}
  for s, kind in sorted(kinds_of.items()):
    if kind not in ('main', 'pilot2', 'memlimit', 'ppi', 'embed', 'superseded'):
      continue
    for runs in sorted((root / s).glob('*/runs.jsonl')):
      for line in runs.read_text().splitlines():
        r = json.loads(line)
        t, ip = r['pdb_id'], r.get('input_path')
        dj = next((root / s / runs.parent.name / t / 'af3_output').glob('*/*_data.json'), None)
        row = {'session': s, 'run': runs.parent.name, 'target': t, 'input_path': ip, 'exit_code': r.get('exit_code')}
        if ip not in inputs:
          f = repo / ip
          inputs[ip] = (json.loads(f.read_text()), sha256_file(f)) if f.is_file() else (None, None)
        inp, sha = inputs[ip]
        if inp is None or sha != r.get('input_sha256'):
          row.update(result='differs', detail=f'frozen input missing or its SHA-256 is not the recorded {r.get("input_sha256")}')
        elif dj is None:
          row.update(result='no data JSON' + (' (process failed)' if r.get('exit_code') != 0 else ''), detail='')
        else:
          d = compare_data_json(inp, json.loads(dj.read_text()), r.get('seeds') or [])
          row.update(result='same' if not d else 'differs', detail='; '.join(d)[:500])
        rows.append(row)
  write_csv(out / 'inputs_check.csv', rows)
  n = {k: sum(1 for r in rows if r['result'] == k) for k in sorted({r['result'] for r in rows})}
  missing = sum(1 for r in rows if r['result'] == 'no data JSON')
  status = 'differences' if n.get('differs') else ('incomplete' if missing or not rows else 'pass')
  summary = {'root': str(root.resolve()), 'sessions': len({r['session'] for r in rows}), 'processes': len(rows),
             'results': n, 'status': status, 'compared': compare_data_json.__doc__.split('Compared:')[1].strip()}
  (out / 'inputs_check.json').write_text(json.dumps(summary, indent=1) + '\n')
  print(json.dumps({k: v for k, v in summary.items() if k != 'compared'}))
  for r in [r for r in rows if r['result'] == 'differs'][:10]:
    print(f'!! {r["session"]}/{r["run"]}/{r["target"]}: {r["detail"]}')
  return 2 if status == 'differences' else 0


def require_inputs_check(out, root, allow_incomplete):
  """R2 is built only after check-inputs passed on the same root."""
  f = out / 'inputs_check.json'
  if not f.is_file():
    raise Refused('no inputs_check.json in --out: run check-inputs on this root first')
  c = json.loads(f.read_text())
  ok = {'pass'} | ({'incomplete'} if allow_incomplete else set())
  if c.get('root') != str(pathlib.Path(root).resolve()) or c.get('status') not in ok:
    raise Refused(f'check-inputs: status {c.get("status")} on {c.get("root")}; the R2 build needs "pass" on this root')


def tarinfo(name, size):
  ti = tarfile.TarInfo(name)
  ti.size, ti.mtime, ti.mode, ti.uid, ti.gid, ti.uname, ti.gname = size, MTIME, 0o644, 0, 0, '', ''
  return ti


def write_archive(path, items, extra):
  """Deterministic tar (gzip without time or name for .tar.gz). items: [(member, source, size, sha, bytes or None)]."""
  raw = open(path, 'wb')
  fileobj = gzip.GzipFile(filename='', mode='wb', fileobj=raw, mtime=0, compresslevel=6) if path.name.endswith('.gz') else raw
  with tarfile.open(fileobj=fileobj, mode='w', format=tarfile.GNU_FORMAT) as tar:
    for member, data in extra:
      tar.addfile(tarinfo(member, len(data)), io.BytesIO(data))
    for member, src, size, _, data in items:
      if data is not None:   # a redacted copy
        tar.addfile(tarinfo(member, size), io.BytesIO(data))
        continue
      with open(src, 'rb') as f:
        tar.addfile(tarinfo(member, size), f)
  if fileobj is not raw:
    fileobj.close()
  raw.close()


def build(args):
  root, repo, out = pathlib.Path(args.root), pathlib.Path(args.repo), pathlib.Path(args.out)
  wanted = set(args.archives.split(',')) if args.archives else None
  sessions_filter = set(args.sessions.split(',')) if args.sessions else None
  kinds_of, odd = session_kinds(root, sessions_filter), unexplained_files()
  written, restore = [], {}   # restore: REDACTIONS.tsv as it was before this build
  try:
    # every R2 archive made from session outputs needs a passing check-inputs on the same root (not the
    # repository-made analysis_outputs.tar.gz, which holds no session file)
    if any(spec[0] == 'R2' and spec[2] != 'repo' and (not wanted or spec[1] in wanted) for spec in ARCHIVES):
      require_inputs_check(out, root, args.allow_incomplete_inputs_check)
    for spec in ARCHIVES:
      record, name, kinds, _ = spec
      if wanted and name not in wanted:
        continue
      cand = members(spec, root, repo, sessions_filter, kinds_of, odd)
      if not cand:
        print(f'   {record}/{name}: no files in this root, skipped')
        continue
      prepared = [(m, src, *prepare(src, m, name)) for m, src in cand]   # every file is checked before anything is written
      items = [(m, src, size, sha, data) for m, src, size, sha, data, _ in prepared]
      redactions = [row for *_, row in prepared if row]
      extra = []
      if record == 'R2':
        terms = next((src for m, src in cand if m.endswith('/TERMS_OF_USE.md')), None) or \
            next(pathlib.Path(root).rglob('TERMS_OF_USE.md'), None)
        if terms is None:
          raise Refused(f'{name}: no AlphaFold3 TERMS_OF_USE.md to ship with the outputs')
        extra = [('LEGALLY_BINDING_TERMS_OF_USE.txt', LEGALLY_BINDING.encode()),
                 ('OUTPUT_TERMS_OF_USE.md', pathlib.Path(terms).read_bytes())]
      if 'superseded' in kinds:
        present = sorted({m.split('/')[1] for m, _ in cand} & set(SUPERSEDED))
        extra.append(('SUPERSEDED_SESSIONS.csv', superseded_table(present).encode()))
      d = out / record
      d.mkdir(parents=True, exist_ok=True)
      write_archive(d / name, items, extra)
      written.append(d / name)
      lines = [f'{hashlib.sha256(data).hexdigest()}  {m}' for m, data in extra] + [f'{sha}  {m}' for m, _, _, sha, _ in items]
      (d / f'{name}.sha256').write_text('\n'.join(lines) + '\n')
      written.append(d / f'{name}.sha256')
      if redactions:
        f = d / 'REDACTIONS.tsv'
        if f not in restore:
          restore[f] = f.read_bytes() if f.is_file() else None
        write_redactions(d, redactions)
      print(f'   {record}/{name}: {len(items)} files, {sum(s for _, _, s, _, _ in items) / 2 ** 20:.1f} MiB raw, '
            f'{(d / name).stat().st_size / 2 ** 20:.1f} MiB archive' + (f', {len(redactions)} redacted' if redactions else ''))
    expected = {spec[1] for spec in ARCHIVES}
    for p in out.rglob('*'):
      if p.is_file() and p.stat().st_size > MAX_MEMBER_BYTES and p.name not in expected:
        raise Refused(f'{p}: over 1 GiB and not an expected archive')
  except Refused as e:
    for p in written:
      p.unlink(missing_ok=True)
    for f, content in restore.items():
      if content is None:
        f.unlink(missing_ok=True)
      else:
        f.write_bytes(content)
    print(f'!! refused, nothing kept: {e}', file=sys.stderr)
    return 2
  return finish_records(sorted({p.parent for p in written} | {f.parent for f in restore}))


def superseded_table(sessions):
  return 'session,label,section_10_row\n' + ''.join(f'{s},"{SUPERSEDED_LABEL}","{SUPERSEDED[s]}"\n' for s in sessions)


def finish_records(dirs):
  """README.md (and for R1 the release tooling and lists) into each record folder, then SHA256SUMS of everything in
  it. Run again (command sums) after archives built elsewhere are copied in."""
  for d in dirs:
    plain = [(HERE / 'records' / d.name / 'README.md', d / 'README.md')]
    if d.name == 'R1':   # the release tooling and its lists travel with record R1, as plain files
      plain += [(HERE / 'release.py', d / 'release' / 'release.py'), (HERE / 'RUNBOOK.md', d / 'release' / 'RUNBOOK.md')]
      plain += [(OUT_DIR / n, d / 'release' / pathlib.Path(n).name) for n in
                ('layout.md', 'layout.csv', 'inventory/inventory.md', 'inventory/summary.csv',
                 'inventory/sessions.csv', 'inventory/excluded_sessions.csv', 'inventory/excluded_files.csv')]
    for src, dst in plain:
      if src.is_file():
        guard(src, str(dst.relative_to(d)))
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(src.read_bytes())
    files = sorted(p for p in d.rglob('*') if p.is_file() and p.name != 'SHA256SUMS')
    if len(files) > ZENODO_MAX_FILES - 1:
      print(f'!! {d.name}: {len(files) + 1} files, over Zenodo\'s {ZENODO_MAX_FILES} per record', file=sys.stderr)
    if d.name == 'R1':
      (d / 'release' / 'SUPERSEDED_SESSIONS.csv').write_text(superseded_table(sorted(SUPERSEDED)))
      files = sorted(p for p in d.rglob('*') if p.is_file() and p.name != 'SHA256SUMS')
    (d / 'SHA256SUMS').write_text(''.join(f'{sha256_file(p)}  {p.relative_to(d)}\n' for p in files))
    print(f'   {d.name}/SHA256SUMS: {len(files)} files')
  return 0


def sums(args):
  return finish_records(sorted(d for d in pathlib.Path(args.out).iterdir() if d.is_dir() and d.name in RECORDS))


def verify(args):
  bad = 0
  for sums in sorted(pathlib.Path(args.out).glob('*/SHA256SUMS')):
    d = sums.parent
    for line in sums.read_text().splitlines():
      sha, name = line.split('  ', 1)
      if sha256_file(d / name) != sha:
        print(f'!! {d.name}/{name}: SHA-256 differs')
        bad += 1
    for man in sorted(d.glob('*.sha256')):
      want = dict(reversed(l.split('  ', 1)) for l in man.read_text().splitlines())
      got = {}
      with tarfile.open(d / man.name[:-len('.sha256')]) as tar:
        for m in tar:
          got[m.name] = hashlib.sha256(tar.extractfile(m).read()).hexdigest()
      if got != want:
        print(f'!! {d.name}/{man.name}: {len(set(got.items()) ^ set(want.items()))} member(s) differ')
        bad += 1
      else:
        print(f'   {d.name}/{man.name[:-7]}: {len(got)} members match')
  print('verify: ' + ('OK' if not bad else f'{bad} problem(s)'))
  return 1 if bad else 0


def main():
  ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
  sub = ap.add_subparsers(dest='cmd', required=True)
  p = sub.add_parser('inventory')
  p.add_argument('--listing', default=str(OUT_DIR / 'inventory' / 'bucket_listing.txt'))
  p.add_argument('--root', help='walk a local copy instead of reading a listing')
  p.add_argument('--out', default=str(OUT_DIR / 'inventory'))
  p = sub.add_parser('layout')
  p.add_argument('--inventory', default=str(OUT_DIR / 'inventory'))
  p.add_argument('--out', default=str(OUT_DIR))
  p = sub.add_parser('build')
  p.add_argument('--root', required=True, help='folder holding <session>/ folders')
  p.add_argument('--repo', default=str(REPO))
  p.add_argument('--out', required=True)
  p.add_argument('--archives')
  p.add_argument('--sessions')
  p.add_argument('--allow_incomplete_inputs_check', action='store_true',
                 help='test subsets only: accept a check-inputs run that found processes without a data JSON')
  p = sub.add_parser('check-inputs')
  p.add_argument('--root', required=True)
  p.add_argument('--repo', default=str(REPO))
  p.add_argument('--out', required=True)
  p.add_argument('--sessions')
  p = sub.add_parser('sums')
  p.add_argument('--out', required=True)
  p = sub.add_parser('verify')
  p.add_argument('--out', required=True)
  args = ap.parse_args()
  return {'inventory': inventory, 'layout': layout, 'build': build, 'check-inputs': check_inputs, 'sums': sums,
          'verify': verify}[args.cmd](args)


if __name__ == '__main__':
  sys.exit(main())
