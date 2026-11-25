"""Index of the data freezes: results/freeze/INDEX.md, read only from the freeze folders.

    python3 scripts/freeze/freeze_index.py [--freeze_dir results/freeze]

Session freezes (scripts/freeze/make_freeze_manifest.py): the folder results/freeze/ itself (14e1fc4c) and every subfolder
holding sessions.tsv and manifest_sha256.tsv. For each: the identifier, computed again as the SHA-256 of
manifest_sha256.tsv and checked against its summary.md; the flag, recognized from the labels in sessions.tsv; the date, from
"Made" in summary.md; the sessions by label. NAS freezes (scripts/nas/ingest_official.py): every subfolder holding
files_sha256.tsv, read the same way. The final freeze for the paper is the newest session freeze that holds every label of
follow-ups 6 to 21 (flag --followups1721, added 2026-11-25); until one exists, the newest that holds every label of
follow-ups 6 to 16 (flag --followups1316, added 2026-11-17); until that exists, the newest that holds every label of
follow-ups 6 to 12 (flag --followups612); until that exists the index says so and names none. Each freeze is cumulative, so
the final one's identifier covers every earlier session freeze: the index checks that each freeze's manifest rows are, byte
for byte, the first rows of the next one's ("yes"), or (added 2026-11-17) that every one of them is, byte for byte and in
the same order, a row of the next one's, which also holds files added later to sessions of the earlier freeze ("yes, plus N
files added later to its sessions": the full outputs that scripts/analysis_runs/fu13/fetch_full_outputs.py copied from the
bucket into two main-study sessions on 2026-11-12). The NAS freezes stay separate. Standard library only.
"""
import argparse
import collections
import csv
import hashlib
import pathlib
import re
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]
LABELS_612 = ('msa', 'sw', 'seeds', 'h100', 'repl', 'prec5', 'large')
LABELS_1316 = ('fu13', 'fu14', 'rcx', 'hlo', 'memlimit_ah', 'warm_th')
LABELS_1721 = ('dtp', 'tri', 'rcf', 'aut', 'xvm')


def sha256(path):
  h = hashlib.sha256()
  with open(path, 'rb') as f:
    while block := f.read(1 << 24):
      h.update(block)
  return h.hexdigest()


def flag_of(labels):
  if set(LABELS_612) | set(LABELS_1316) | set(LABELS_1721) <= labels:
    return '--followups1721'
  if labels & set(LABELS_1721):
    return 'unknown (only some follow-up 6 to 21 labels)'
  if set(LABELS_612) | set(LABELS_1316) <= labels:
    return '--followups1316'
  if labels & set(LABELS_1316):
    return 'unknown (only some follow-up 6 to 16 labels)'
  if set(LABELS_612) <= labels:
    return '--followups612'
  if labels & set(LABELS_612):
    return 'unknown (only some follow-up 6 to 12 labels)'
  if 'feat' in labels:
    return '--feat'
  if {'cpu60', 'a100'} & labels:
    return '--followups34'
  return '(none)'


def holds(previous, manifest):
  """'yes' if the earlier manifest is a byte prefix of this one; 'yes, plus N files added later to its sessions' if each
  of its rows is a row of this one, in the same order; otherwise 'NO'."""
  if manifest.startswith(previous):
    return 'yes'
  old, new = previous.splitlines(keepends=True), manifest.splitlines(keepends=True)
  sessions = {line.split(b'\t', 1)[0] for line in old[1:]}
  i = added = 0
  for line in new:
    if i < len(old) and line == old[i]:
      i += 1
    elif line.split(b'\t', 1)[0] in sessions:
      added += 1
  if i < len(old):
    return 'NO'
  return f'yes, plus {added} files added later to its sessions'


def read_summary(folder):
  text = (folder / 'summary.md').read_text() if (folder / 'summary.md').is_file() else ''
  made = re.search(r'^Made (\S+)', text, re.M)
  ident = re.search(r'identifier: SHA-256 of \S+ = `([0-9a-f]{64})`', text)
  return text, made.group(1) if made else '?', ident.group(1) if ident else None


def session_freeze(folder, root):
  _, made, stated = read_summary(folder)
  got = sha256(folder / 'manifest_sha256.tsv')
  with open(folder / 'sessions.tsv', newline='') as f:
    rows = list(csv.DictReader(f, delimiter='\t'))
  counts = collections.Counter('pilot2, excluded' if r['kind'].startswith('pilot2, excluded') else r['kind'] for r in rows)
  name = str(folder.relative_to(root.parent))
  return {'name': name, 'identifier': got, 'check': 'ok' if got == stated else f'DIFFERS from summary.md ({stated})',
          'flag': flag_of(set(counts)), 'made': made, 'sessions': len(rows), 'labels': counts}


def nas_freeze(folder, root):
  text, made, stated = read_summary(folder)
  got = sha256(folder / 'files_sha256.tsv')
  files = re.search(r'^- files: (.+)$', text, re.M)
  return {'name': str(folder.relative_to(root.parent)), 'identifier': got,
          'check': 'ok' if got == stated else f'DIFFERS from summary.md ({stated})', 'made': made,
          'files': files.group(1) if files else '?'}


def build(root):
  sessions, nas = [], []
  for folder in [root] + sorted(p for p in root.iterdir() if p.is_dir()):
    if (folder / 'sessions.tsv').is_file() and (folder / 'manifest_sha256.tsv').is_file():
      sessions.append(session_freeze(folder, root))
    elif (folder / 'files_sha256.tsv').is_file():
      nas.append(nas_freeze(folder, root))
  sessions.sort(key=lambda s: s['made'])
  previous = None
  for s in sessions:
    manifest = (root.parent / s['name'] / 'manifest_sha256.tsv').read_bytes()
    s['holds_previous'] = '-' if previous is None else holds(previous, manifest)
    previous = manifest
  candidates = ([s for s in sessions if s['flag'] == '--followups1721' and s['check'] == 'ok'] or
                [s for s in sessions if s['flag'] == '--followups1316' and s['check'] == 'ok'] or
                [s for s in sessions if s['flag'] == '--followups612' and s['check'] == 'ok'])
  final = candidates[-1] if candidates else None
  L = ['# Data freezes', '', 'Generated by scripts/freeze/freeze_index.py, read only from the freeze folders; do not edit.', '']
  if final:
    L += [f'**Final freeze for the paper:** `{final["identifier"]}` ({final["name"]}, made {final["made"]}). It holds every '
          'earlier session freeze; the follow-up 6 NAS deliveries keep their own freezes, below.', '']
  else:
    L += ['**Final freeze for the paper: none yet.** It is the newest freeze made with --followups1721 '
          '(DATE=YYYYMMDD bash scripts/freeze/reproduce_followups1721.sh), or, until one exists, with --followups1316, '
          'or with --followups612.', '']
  L += ['## Session freezes', '', '| folder | identifier | identifier check | flag | made | sessions | earlier rows byte-identical | by label |',
        '|---|---|---|---|---|---|---|---|']
  for s in sessions:
    by = ', '.join(f'{k} {v}' for k, v in s['labels'].items())
    mark = ' (final)' if final and s is final else ''
    L.append(f'| {s["name"]}{mark} | `{s["identifier"]}` | {s["check"]} | {s["flag"]} | {s["made"]} | {s["sessions"]} | {s["holds_previous"]} | {by} |')
  L += ['', '## Follow-up 6 NAS deliveries (not in the bucket)', '', '| folder | identifier | identifier check | made | files |',
        '|---|---|---|---|---|']
  L += [f'| {n["name"]} | `{n["identifier"]}` | {n["check"]} | {n["made"]} | {n["files"]} |' for n in nas]
  return '\n'.join(L) + '\n', sessions, nas, final


def main():
  ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
  ap.add_argument('--freeze_dir', default=str(REPO / 'results' / 'freeze'))
  args = ap.parse_args()
  root = pathlib.Path(args.freeze_dir).resolve()
  text, *_ = build(root)
  (root / 'INDEX.md').write_text(text)
  print(text)
  return 0


if __name__ == '__main__':
  sys.exit(main())
