"""Data freeze manifest of the pilot2, main-study, memory-limit and follow-up sessions, and a comparison with the
results bucket.

    analysis/.venv/bin/python scripts/freeze/make_freeze_manifest.py --bucket gs://BUCKET [--out results/freeze]
        [--followups34]

Sessions: the 13 pilot2 sessions of results/analysis/pilot2_20261012/sessions_pilot2.txt, the pilot2 sessions
excluded there, the 53 main sessions of results/analysis/main_20261014/sessions_main.txt, and the 4 memory-limit
sessions the wrapper run results/af3/.many/20261014T123400Z/summary.txt lists (section 10, 2026-10-14), and the 13
follow-up sessions (4 embed, 9 ppi) the wrapper run results/af3/.many/20261014T223755Z/summary.txt lists (added
2026-10-15). With --followups34 (added 2026-10-18; the freeze extended to follow-ups 3 and 4, section 10, rows of
2026-10-17 and 2026-10-18), also the sessions that ended with launcher exit 0 in the follow-up 4 wrapper runs (54
cpu60: 49 in 20261016T230400Z, the 5 relaunches in 20261017T112542Z) and the follow-up 3 ones (12 a100: 7 A in
20261017T112657Z, 2 Ah and 3 AT in 20261017T211156Z); preempted and smoke sessions are not frozen. Without it, the
session list is the one of freeze 14e1fc4c (2026-10-15), unchanged. For the follow-up 3 and 4 sessions, a log the
launcher appended to after the final upload (local copy longer) is also downloaded from the bucket (read only, to a
temporary file) and must be a byte prefix of the local copy; otherwise it stays an unexplained MISMATCH.
Local side: every file of each results/af3/<session>/ folder, with its size, SHA-256 (hashlib) and CRC32C
(`gcloud storage hash --skip-md5`, computed on this machine). Bucket side: the object list of
<bucket>/af3/<session>/ with size and CRC32C (`gcloud storage objects list`, read only); also every other session
folder under <bucket>/af3/ named like a pilot2 or main session (listed, not compared file by file). Nothing is
written to the bucket.
Comparison per file (path relative to the session folder): same (both sides, equal CRC32C and size), mismatch,
local only, bucket only. Expected differences are labelled: bucket-only *_data.json and full *_confidences.json
(FETCH=light leaves them in the bucket), local-only files the launcher writes after the VM's last upload.
Outputs in --out: sessions.tsv, manifest_sha256.tsv (the frozen local files), bucket_objects.tsv, comparison.tsv,
summary.md; the SHA-256 of manifest_sha256.tsv, which identifies the freeze, is in summary.md.
"""
import argparse
import datetime
import hashlib
import json
import pathlib
import re
import subprocess
import sys
import tempfile

REPO = pathlib.Path(__file__).resolve().parents[2]
AF3 = REPO / 'results' / 'af3'
PILOT2 = REPO / 'results' / 'analysis' / 'pilot2_20261012'
MAIN = REPO / 'results' / 'analysis' / 'main_20261014'
MEMLIMIT_WRAPPER = AF3 / '.many' / '20261014T123400Z' / 'summary.txt'
FOLLOWUP_WRAPPER = AF3 / '.many' / '20261014T223755Z' / 'summary.txt'
FOLLOWUP34_WRAPPERS = [('cpu60', '20261016T230400Z'), ('cpu60', '20261017T112542Z'), ('a100', '20261017T112657Z'),
                       ('a100', '20261017T211156Z')]
FETCH_LIGHT = re.compile(r'(_data\.json|(?<!summary)_confidences\.json)$')
# Session-level files the VM job or the launcher writes or appends to after the VM's final upload
# (cloud/vm_af3_job.sh upload_out, cloud/vm_job_finish.sh, cloud/af3_run_many.sh): a later local copy.
AFTER_UPLOAD_LOCAL = re.compile(r'^(\.uploaded|\.wrapper_done|finish\.log|upload_final\.log)$')
AFTER_UPLOAD_APPENDED = re.compile(r'^(job\.log|upload_final\.log)$')


def sessions(followups34=False):
  out = [(s, 'pilot2') for s in (PILOT2 / 'sessions_pilot2.txt').read_text().split()]
  for line in (PILOT2 / 'sessions_excluded.txt').read_text().splitlines():
    s, _, why = line.partition('\t')
    if s.strip():
      out.append((s.strip(), f'pilot2, excluded: {why.strip()}'))
  out += [(s, 'main') for s in (MAIN / 'sessions_main.txt').read_text().split()]
  out += [(m.group(1), 'memlimit') for m in re.finditer(r'ok: launcher exit 0, session (\S+)', MEMLIMIT_WRAPPER.read_text())]
  for m in re.finditer(r'ok: launcher exit 0, session (\S+)', FOLLOWUP_WRAPPER.read_text()):
    out.append((m.group(1), 'embed' if '_embed_' in m.group(1) else 'ppi'))
  if followups34:
    for kind, w in FOLLOWUP34_WRAPPERS:
      out += [(s, kind) for s in re.findall(r'ok: launcher exit 0, session (\S+)',
                                            (AF3 / '.many' / w / 'summary.txt').read_text())]
  return out


def sha256(path, chunk=1 << 24):
  h = hashlib.sha256()
  with open(path, 'rb') as f:
    while block := f.read(chunk):
      h.update(block)
  return h.hexdigest()


def local_crc32c(folder):
  """{relative path: base64 CRC32C} of every file in folder, from `gcloud storage hash` (local, no network)."""
  out = subprocess.run(['gcloud', 'storage', 'hash', '--skip-md5', f'{folder}/**'], capture_output=True, text=True)
  if out.returncode != 0:
    raise SystemExit(f'gcloud storage hash {folder}: {out.stderr[-500:]}')
  crc = {}
  # The output is YAML documents (crc32c_hash, digest_format, url) separated by "---".
  for d in out.stdout.split('---'):
    fields = dict(l.strip().partition(': ')[::2] for l in d.strip().splitlines() if ': ' in l)
    if 'url' in fields and 'crc32c_hash' in fields:
      crc[str(pathlib.Path(fields['url']).resolve().relative_to(folder.resolve()))] = fields['crc32c_hash']
  return crc


def bucket_objects(bucket, session):
  """{relative path: (size, base64 CRC32C)} of <bucket>/af3/<session>/ (read only)."""
  uri = f'{bucket.rstrip("/")}/af3/{session}/**'
  out = subprocess.run(['gcloud', 'storage', 'objects', 'list', uri, '--format=json(name,size,crc32c_hash)'],
                       capture_output=True, text=True)
  if out.returncode != 0:
    if 'matched no objects' in out.stderr or 'One or more URLs matched no objects' in out.stderr:
      return {}
    raise SystemExit(f'gcloud storage objects list {uri}: {out.stderr[-500:]}')
  prefix = f'af3/{session}/'
  return {o['name'][len(prefix):]: (int(o['size']), o.get('crc32c_hash')) for o in json.loads(out.stdout or '[]')
          if o['name'].startswith(prefix)}


def bucket_copy_is_prefix(bucket, session, rel, local_path):
  """True if <bucket>/af3/<session>/<rel> is a byte prefix of the local file (read-only download to a temporary
  file, deleted afterwards)."""
  with tempfile.TemporaryDirectory() as d:
    dst = pathlib.Path(d) / 'object'
    out = subprocess.run(['gcloud', 'storage', 'cp', f'{bucket.rstrip("/")}/af3/{session}/{rel}', str(dst)],
                         capture_output=True, text=True)
    if out.returncode != 0:
      raise SystemExit(f'gcloud storage cp {session}/{rel}: {out.stderr[-500:]}')
    remote = dst.read_bytes()
  return local_path.read_bytes()[:len(remote)] == remote


def bucket_sessions(bucket):
  out = subprocess.run(['gcloud', 'storage', 'ls', f'{bucket.rstrip("/")}/af3/'], capture_output=True, text=True)
  if out.returncode != 0:
    raise SystemExit(f'gcloud storage ls: {out.stderr[-500:]}')
  return sorted(p.rstrip('/').rsplit('/', 1)[-1] for p in out.stdout.split() if p.endswith('/'))


def main():
  ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
  ap.add_argument('--bucket', required=True, help='the results bucket, gs://NAME')
  ap.add_argument('--out', default=str(REPO / 'results' / 'freeze'))
  ap.add_argument('--followups34', action='store_true', help='also freeze the follow-up 3 and 4 sessions')
  args = ap.parse_args()
  out = pathlib.Path(args.out)
  out.mkdir(parents=True, exist_ok=True)
  started = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds')
  rows_s, rows_m, rows_b, rows_c = [], [], [], []
  totals = {}
  for session, kind in sessions(args.followups34):
    folder = AF3 / session
    local = sorted(p for p in folder.rglob('*') if p.is_file()) if folder.is_dir() else []
    crc = local_crc32c(folder) if local else {}
    remote = bucket_objects(args.bucket, session)
    lmap = {}
    for p in local:
      rel = str(p.relative_to(folder))
      size = p.stat().st_size
      lmap[rel] = (size, crc.get(rel))
      rows_m.append((session, rel, size, sha256(p), crc.get(rel) or ''))
    for rel, (size, c) in sorted(remote.items()):
      rows_b.append((session, rel, size, c or ''))
    for rel in sorted(lmap.keys() | remote.keys()):
      l, r = lmap.get(rel), remote.get(rel)
      if l and r:
        status = 'same' if (l[0], l[1]) == (r[0], r[1]) else 'MISMATCH'
        note = '' if status == 'same' else f'local {l[0]} B {l[1]}, bucket {r[0]} B {r[1]}'
        if status != 'same' and AFTER_UPLOAD_APPENDED.match(rel) and l[0] > r[0]:
          if kind not in ('cpu60', 'a100'):
            note = 'expected: appended after the final upload (local copy longer); ' + note
          elif bucket_copy_is_prefix(args.bucket, session, rel, folder / rel):
            note = ('expected: appended after the final upload (local copy longer, the bucket copy a byte prefix of '
                    'it); ' + note)
          else:
            note = 'local copy longer, but the bucket copy is not a prefix of it; ' + note
      elif l:
        status = 'local only'
        note = 'expected: written after the final upload' if AFTER_UPLOAD_LOCAL.match(rel) else ''
      else:
        status = 'bucket only'
        note = 'expected: FETCH=light leaves it in the bucket' if FETCH_LIGHT.search(rel) else ''
      rows_c.append((session, rel, status, note))
      key = ('pilot2, excluded' if kind.startswith('pilot2, excluded') else kind,
             status + (f' ({note.split(";")[0].replace("expected: ", "")})' if note.startswith('expected') else ''))
      t = totals.setdefault(key, [0, 0])
      t[0] += 1
      t[1] += (l or r)[0]
    rows_s.append((session, kind, folder.is_dir(), len(local), sum(v[0] for v in lmap.values()), len(remote),
                   sum(v[0] for v in remote.values())))
    print(f'{session:44} {kind[:16]:16} local {len(local):6} files, bucket {len(remote):6} objects', flush=True)

  def tsv(name, header, rows):
    with open(out / name, 'w') as f:
      f.write('\t'.join(header) + '\n')
      for r in rows:
        f.write('\t'.join(str(x) for x in r) + '\n')
  tsv('sessions.tsv', ['session', 'kind', 'local_folder', 'local_files', 'local_bytes', 'bucket_objects',
                       'bucket_bytes'], rows_s)
  tsv('manifest_sha256.tsv', ['session', 'path', 'bytes', 'sha256', 'crc32c'], rows_m)
  tsv('bucket_objects.tsv', ['session', 'path', 'bytes', 'crc32c'], rows_b)
  tsv('comparison.tsv', ['session', 'path', 'status', 'note'], rows_c)
  ours = {s for s, _, *_ in rows_s}
  named = r'_(pilot|main|memlimit|embed|ppi|cpu60|a100)_?' if args.followups34 else r'_(pilot|main|memlimit|embed|ppi)_?'
  others = [s for s in bucket_sessions(args.bucket) if s not in ours and re.search(named, s) and s >= '20261010T2']
  manifest_sha = sha256(out / 'manifest_sha256.tsv')
  lines = ['# Data freeze: pilot2, main-study, memory-limit and follow-up sessions', '',
           f'Made {started} by scripts/freeze/make_freeze_manifest.py (read-only bucket listing; nothing written to '
           'the bucket). Bucket CRC32C from `gcloud storage objects list`, local CRC32C from `gcloud storage hash`.', '',
           f'Freeze identifier: SHA-256 of manifest_sha256.tsv = `{manifest_sha}`', '',
           '| sessions | local files | local bytes | bucket objects | bucket bytes |', '|---|---|---|---|---|']
  for kind in ('pilot2', 'pilot2, excluded', 'main', 'memlimit', 'embed', 'ppi') + (('cpu60', 'a100') if args.followups34
                                                                                  else ()):
    sel = [r for r in rows_s if (r[1].startswith('pilot2, excluded') if kind == 'pilot2, excluded' else r[1] == kind)]
    lines.append(f'| {kind}: {len(sel)} | {sum(r[3] for r in sel)} | {sum(r[4] for r in sel):,} | '
                 f'{sum(r[5] for r in sel)} | {sum(r[6] for r in sel):,} |')
  lines += ['', '| kind | status | files | bytes |', '|---|---|---|---|']
  lines += [f'| {k[0]} | {k[1]} | {v[0]} | {v[1]:,} |' for k, v in sorted(totals.items())]
  odd = [r for r in rows_c if r[2] != 'same' and not r[3].startswith('expected')]
  lines += ['', f'## Unexplained: files missing on either side or mismatched, not labelled expected above: {len(odd)}',
            '']
  groups = {}
  for s, rel, st, _ in odd:
    pattern = re.sub(r'seed-\d+_sample-\d+', 'seed-S_sample-K', re.sub(r'^[^/]+/[0-9A-Z]{4}/', '<run>/<target>/',
                                                                      re.sub(r'^[^/]+/(?=[^/]+$)', '<run>/', rel)))
    groups.setdefault((st, pattern), []).append(f'{s}/{rel}')
  lines += ['| status | path pattern | files | example |', '|---|---|---|---|']
  lines += [f'| {st} | {pat} | {len(v)} | {v[0]} |' for (st, pat), v in sorted(groups.items())]
  sel = [r for r in rows_s if not r[2]]
  if sel:
    lines += ['', 'Sessions without a local folder: ' + ', '.join(f'{r[0]} ({r[1]})' for r in sel)]
  lines += ['', f'Other session folders in the bucket named like pilot2, main, memory-limit or follow-up sessions, not in the lists above: '
            f'{", ".join(others) if others else "none"}']
  (out / 'summary.md').write_text('\n'.join(lines) + '\n')
  print('\n'.join(lines))
  return 0


if __name__ == '__main__':
  sys.exit(main())
