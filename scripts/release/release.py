"""Data release for Zenodo: inventory, layout, input check, archive building and verification (prepare and dry-run
only; this script never contacts a bucket or Zenodo). Build plan: scripts/release/RUNBOOK.md.

    python3 scripts/release/release.py inventory [--listing results/release/inventory/bucket_listing.txt]
    python3 scripts/release/release.py inventory --root results/af3 --bucket_objects TSV --launcher results/af3/.launcher
    python3 scripts/release/release.py layout
    python3 scripts/release/release.py scan --root DIR --out DIR [--repo .] [--archives A,B]
    python3 scripts/release/release.py check-inputs --root DIR --out DIR [--sessions S1,S2] [--repo .]
    python3 scripts/release/release.py build --root DIR --out DIR [--archives A,B] [--sessions S1,S2] [--repo .]
    python3 scripts/release/release.py archives --built_on laptop|vm
    python3 scripts/release/release.py merge-redactions --out DIR --record R1 --from FILE
    python3 scripts/release/release.py sums --out DIR
    python3 scripts/release/release.py verify --out DIR

Sessions (classify()): the data freeze's sessions (FREEZE/sessions.tsv: main, pilot2, memlimit, ppi, embed, and the
follow-ups cpu60, a100, feat, msa, sw, seeds, h100, repl, prec5, large); "superseded" sessions with the official weights
that no reported result uses (SUPERSEDED, each with its section 10 row; released apart in R2,
af3_superseded_engineering.tar.gz, after check_official_weights()); "outside" sessions of follow-ups 3 to 12 that are not
in the freeze (smoke, preempted, interrupted, relaunched-over, eigen_check; each mapped by OUTSIDE_RULES or OUTSIDE_NAMED
to its section 10 row; records only, released in R1); "attempt" sessions that never ran (launcher records only; released
in R1); everything else stays out (random-weight engineering sessions). A session named like a follow-up 6 to 12 session
that is neither in the freeze nor mapped to a row is "unmapped": inventory lists it and exits 2, and build and
check-inputs stop. The freeze's unexplained files stay out, except launcher records of attempts that never ran and the
18 launcher records the row of 2026-11-01 explains (ROW_EXPLAINED).
inventory: sorts every object of a read-only bucket listing (`gcloud storage ls -l -r 'gs://BUCKET/af3/**'`) or, with
--root, of a local copy, by session kind and category (category()). Writes inventory/{summary.csv, sessions.csv,
excluded_sessions.csv, excluded_files.csv, unmapped_sessions.csv, OUTSIDE_SESSIONS.csv, inventory.md}. With --root,
--bucket_objects uses a freeze's bucket_objects.tsv for the frozen sessions (no bucket read) and --launcher adds the local
launcher records of sessions without a local folder: an estimate before the read-only listing of the runbook.
scan: guard()'s privacy checks on every local member of the chosen archives, without building: privacy_scan.csv (by
archive, class and action, counts only, never a value) and privacy_files.tsv (path, class, occurrences).
layout: the archives and records of ARCHIVES with file counts, raw and estimated compressed sizes (gzip ratios
measured on local samples, RATIO), against Zenodo's per-record limits. Writes layout.csv, layout.md.
check-inputs: compares every released process's *_data.json with its frozen input (compare_data_json() says what is
compared; seeds 6 to 20 of follow-up 8 have an explicit allowance, SEEDS_BEYOND_INPUT) and writes inputs_check.csv and
inputs_check.json into --out; exit 2 on any difference. The R2 build refuses to start unless this passed on the same
root.
build: writes the archives from --root (a folder holding <session>/ folders: the local freeze copy results/af3, or a
copy of the bucket's af3/ prefix) and from the repository (--repo), into <out>/<record>/: deterministic tar (sorted
members, fixed mtime, owner and mode; gzip header without time or name), one SHA-256 manifest per archive
(<archive>.sha256, `sha256sum -c` format), then sums. Every candidate file passes guard() first; any refusal stops
the build and removes what it wrote. --sessions limits a build to some sessions (a test subset).
sums: copies scripts/release/records/<record>/README.md and records/LICENCES.md (and, for R1, this script, RUNBOOK.md,
the inventory, the layout, the exclusion lists, SUPERSEDED_SESSIONS.csv and OUTSIDE_SESSIONS.csv, as release_<name>, since
a Zenodo record has no folders) into each record folder and writes its SHA256SUMS; it stops if a record is over Zenodo's
limits. Run it after archives built elsewhere are copied in.
archives: prints the comma-separated names of the archives built on the laptop or on the VM (BUILT_ON).
merge-redactions: adds the rows of a REDACTIONS.tsv written elsewhere (the other machine's build) to a record's.
verify: recomputes SHA256SUMS and every member's SHA-256 from the archives.

guard() refuses: a file named af3.bin or af3.bin.zst, or any *.bin / *.bin.zst; a file whose SHA-256 is the
weights' (WEIGHTS_SHA256) or whose size is the weights' (WEIGHTS_BYTES); any file over 1 GiB (none is expected among
the members); a text file that names a gs:// bucket other than ALLOWED_BUCKETS; a text file that contains an email
address, unless it is on the redaction list (REDACT, from records/redact_email_paths.txt) or is a record's README.md
carrying only the contact address, once; a secret (a WireGuard or ssh key, a token that looks like an API key: PRIVATE,
action refuse), since a secret must be looked at and revoked, not hidden; a redaction that would touch an AlphaFold3
output file (those are released unmodified); and, inside a nested .tar.gz or .tgz member, any of the above. Packaged
as a copy with the text replaced, with a row in <record>/REDACTIONS.tsv (archive, path, original and released SHA-256,
what was replaced; never the value): email addresses on the list (<account>), and in any text file, private IPv4
addresses (<private-ip>), .local and .lan host names (<private-host>), bare NAS host names (<nas-host>) and home paths
(/home/<user>). The labels of the follow-up 6 deliveries (naspaz8-af3_run1 and the like) stay: they are part of frozen
files, file names and identifiers. The files in results/ and data/ are never changed. Text that records the weights'
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
FREEZE = REPO / 'results' / 'freeze' / 'followups612_20261108'   # the final freeze (results/freeze/INDEX.md)
FREEZE_ID = '689230120499e5ba9d903051553c6554f3412f462456461757398f8fd3560a1a'
WEIGHTS_SHA256 = '74d0258616917cd122f5eab6d076afe4a8930e96823851e65e4f777dfb1f33ff'
WEIGHTS_BYTES = 1020545840
WEIGHTS_NAMES = {'af3.bin', 'af3.bin.zst'}
MAX_MEMBER_BYTES = 1 << 30
ALLOWED_BUCKETS = {'af2-tpu-benchmark-af3-results'}
# Not scanned as text: arrays and archives (random bytes would match the patterns); nested .tar.gz and .tgz members are
# opened and scanned instead (NESTED_SUFFIXES).
BINARY_SUFFIXES = ('.npz', '.npy', '.zst', '.tar', '.gz', '.tgz', '.bin')
NESTED_SUFFIXES = ('.tar.gz', '.tgz')
MTIME = 1792022400   # 2026-10-15T00:00:00Z, every member's mtime
ZENODO_MAX_BYTES, ZENODO_MAX_FILES = 50 * 10 ** 9, 100
ARCHIVE_ABOUT_MAX_BYTES = 12 * 10 ** 9   # the plan keeps every archive under about 10 GB: a larger one is reported
# gzip -6 ratios measured on 12 random local files per category (2026-10-15); embeddings do not compress.
RATIO = {'model_cif': 0.255, 'ranking_summary': 0.063, 'full_confidences': 0.124, 'terms': 0.35, 'af3_logs': 0.067,
         'records': 0.15, 'embeddings': 1.0, 'features': 0.92, 'repo': 0.331}


def category(path):
  """The release category of a path inside a session folder."""
  name = path.rsplit('/', 1)[-1]
  if '/work/' in f'/{path}' or name in {'.uploaded', '.wrapper_done'}:
    return 'work'               # local scratch (inputs with seeds, compile caches): never released
  if name.endswith('_data.json'):
    return 'data_json'
  if name.endswith('_embeddings.npz'):
    return 'embeddings'
  if name.endswith('.npz'):
    return 'features'           # follow-up 5: the feature arrays of a featurisation run
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


# Unexplained in the comparison only because the comparison labels by pattern: the launchers of these six Gx sessions ran
# on the laptop while it was offline and could never upload their own records, and the sessions were copied from the
# bucket by hand (rows of 2026-11-01 and of the freeze, 2026-11-08). The files are the laptop's own launch records, with
# their SHA-256 in the manifest, so they are released with the session.
ROW_EXPLAINED = {(f'{stamp}_l4_seeds_gx_{part}', name)
                 for stamp, part in (('20261031T204043Z', '04'), ('20261031T204048Z', '06'), ('20261031T204053Z', '07'),
                                     ('20261031T204059Z', '12'), ('20261031T204104Z', '13'), ('20261031T204109Z', '14'))
                 for name in ('launch_record.txt', 'launcher_events.txt', 'uploaded_files.txt')}


def unexplained_files():
  """{(session, path): status} the freeze could not explain (excluded from the release)."""
  out = {}
  for r in csv.DictReader(open(FREEZE / 'comparison.tsv'), delimiter='\t'):
    if r['status'] != 'same' and not r['note'].startswith('expected') and (r['session'], r['path']) not in ROW_EXPLAINED:
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
# Sessions of follow-ups 3 to 12 outside the freeze: their records are released, their outputs are not (decision of
# 2026-11-08). Each is mapped to the section 10 row that explains it. Rules: (plan prefix after the platform, first and
# last launch stamp the row covers, row). The windows keep a later session of the same arm from being mapped by accident.
R_MSA_SMOKE = '2026-10-26, Follow-up 6, smoke tests (label msa_smoke; engineering check, not a study target)'
R_GO_R1 = '2026-10-27, Follow-up 6, Go r1, launch and relaunch (attempts with no VM, setup failures on the GitLab overload)'
R_V6O = '2026-10-28, Follow-up 6, V6o, launch and relaunches (attempts with no VM, setup failures on the GitLab overload)'
R_V5O = '2026-10-29, Follow-up 6, V5o, launch (attempts with no VM)'
R_EIGEN = '2026-10-29, Follow-ups 7 to 9, eigen_check (engineering check, never a finding; attempts with no VM, one setup failure)'
R_SW_SMOKE = '2026-10-29, Follow-up 7, sw_smoke (engineering check, never a finding; attempts with no VM)'
R_V5X = '2026-10-30, Follow-up 8, V5x: Spot record (attempts with no VM, preempted VMs, VMs deleted before the job); 2026-10-31, none enters the analysis'
R_GX = '2026-11-01, Follow-up 8, Gx (Spot VMs deleted by Google during the job, relaunched unchanged)'
R_GS = '2026-11-02, Follow-up 7, Gs (Spot VMs deleted by Google during the job, relaunched unchanged)'
R_H_SMOKE = '2026-11-03, Follow-up 9, amendment of H r2 (smoke test, engineering check, never a finding)'
R_GR = '2026-11-05, Follow-up 11, Gr (interrupted sessions kept in the record and in no analysis)'
R_V5R = '2026-11-05, Follow-up 11, V5r (attempts with no VM, setup failures on the apt lock)'
R_GL = '2026-11-07 and 2026-11-08, Follow-up 10, Gl (Spot preemptions and lack of capacity, relaunched unchanged)'
R_V5L = '2026-11-07, Follow-up 10, V5l (create requests refused for lack of capacity, no VM)'
OUTSIDE_RULES = [
    ('msa_smoke_', '20261026T134900Z', '20261026T144000Z', R_MSA_SMOKE),
    ('msa_go_r1_', '20261026T162600Z', '20261026T222000Z', R_GO_R1),
    ('msa_v6o_', '20261027T115900Z', '20261027T174200Z', R_V6O),
    ('msa_v5o_', '20261028T095200Z', '20261028T095900Z', R_V5O),
    ('eigen_check_', '20261029T110500Z', '20261029T115300Z', R_EIGEN),
    ('sw_smoke_', '20261029T120700Z', '20261029T123700Z', R_SW_SMOKE),
    ('seeds_v5x_', '20261029T125100Z', '20261030T220700Z', R_V5X),
    ('seeds_gx_', '20261031T110100Z', '20261101T021600Z', R_GX),
    ('sw_gs_', '20261101T213000Z', '20261102T020800Z', R_GS),
    ('h100_smoke_', '20261103T210600Z', '20261103T210700Z', R_H_SMOKE),
    ('repl_gr_', '20261104T223000Z', '20261105T074700Z', R_GR),
    ('repl_v5r_', '20261104T223000Z', '20261105T172900Z', R_V5R),
    ('large_v5l_', '20261105T224600Z', '20261107T080000Z', R_V5L),
    ('large_gl_', '20261106T223600Z', '20261107T200000Z', R_GL),
]
# Follow-ups 3 and 4: the sessions the rows name that had a VM and a job but are not in the freeze. Attempts of these
# follow-ups with no VM keep the plain "attempt" rule.
R_A100_FIRST = '2026-10-16, Follow-up 3, first smoke-test launch (stopped at the weights step, no AlphaFold3 process ran)'
R_A100_SMOKE = '2026-10-17, Follow-ups 3 and 4, first launches (smoke tests, engineering checks only)'
R_CPU60 = '2026-10-17, Follow-ups 3 and 4, first launches (Spot VMs preempted, relaunched unchanged; enter no analysis)'
OUTSIDE_NAMED = {
    '20261016T183749Z_a100_a100_smoke': R_A100_FIRST,
    '20261016T231247Z_a100_a100_smoke': R_A100_SMOKE, '20261016T232522Z_a100_a100_smoke_h': R_A100_SMOKE,
    '20261016T234440Z_a100_a100_smoke_t': R_A100_SMOKE,
    '20261016T230421Z_cpu_cpu60_7nu0': R_CPU60, '20261017T080130Z_cpu_cpu60_7rc3': R_CPU60,
    '20261017T080933Z_cpu_cpu60_5sd5': R_CPU60, '20261017T081626Z_cpu_cpu60_6yqw': R_CPU60,
    '20261017T082024Z_cpu_cpu60_6zpb': R_CPU60,
}
OUTSIDE_LABEL = 'outside the freeze: records only, no output released, not used in any reported result'
FU612_SESSION = re.compile(r'^(\d{8}T\d{6}Z)_[a-z0-9]+_((?:msa|sw|seeds|h100|repl|prec5|large|eigen_check)(?:_.*)?)$')
EARLY_KINDS = ('main', 'pilot2', 'memlimit', 'ppi', 'embed')
FOLLOWUP_KINDS = ('cpu60', 'a100', 'feat', 'msa', 'sw', 'seeds', 'h100', 'repl', 'prec5', 'large')
RELEASED_KINDS = EARLY_KINDS + FOLLOWUP_KINDS + ('superseded', 'attempt', 'outside')


def outside_row(session):
  """The section 10 row of a session outside the freeze, or None."""
  if session in OUTSIDE_NAMED:
    return OUTSIDE_NAMED[session]
  m = FU612_SESSION.match(session)
  if m:
    for prefix, first, last, row in OUTSIDE_RULES:
      if m.group(2).startswith(prefix) and first <= m.group(1) <= last:
        return row
  return None


def classify(session, paths, kinds):
  """(kind, reason): a freeze kind, 'superseded' (SUPERSEDED), 'outside' (a session of follow-ups 3 to 12 outside the
  freeze, mapped to its row: records only, released in R1), 'unmapped' (a follow-up 6 to 12 session that no row
  explains: stops the build), 'attempt' (no VM run records: launcher records only, released in R1), or 'out' (not
  released) with the reason."""
  ran = any(p.endswith(('run.json', 'runs.jsonl')) for p in paths)
  if session in SUPERSEDED:
    return 'superseded', SUPERSEDED[session]
  if session not in kinds:
    row = outside_row(session)
    if row:
      return 'outside', row
    if FU612_SESSION.match(session):
      return 'unmapped', 'named like a follow-up 6 to 12 session, not in the freeze, and no section 10 row maps it'
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


def walk_estimate(root, bucket_objects, launcher):
  """An inventory with no bucket read: the frozen sessions from the freeze's bucket_objects.tsv plus their local files,
  the others from their local folders, and sessions with no local folder from their local launcher records."""
  rows, seen = [], set()
  if bucket_objects:
    for r in csv.DictReader(open(bucket_objects), delimiter='\t'):
      rows.append((r['session'], r['path'], int(r['bytes'])))
      seen.add((r['session'], r['path']))
  rows += [x for x in walk(root) if (x[0], x[1]) not in seen]
  if launcher:
    have = {s for s, *_ in rows}
    rows += [x for x in walk(launcher) if x[0] not in have]
  return rows


def inventory(args):
  rows = walk_estimate(args.root, args.bucket_objects, args.launcher) if args.root else read_listing(args.listing)
  kinds, odd = freeze_sessions(), unexplained_files()
  out = pathlib.Path(args.out)
  out.mkdir(parents=True, exist_ok=True)
  summary, per_session, excluded_files = {}, {}, []
  by_session = {}
  for s, p, b in rows:
    by_session.setdefault(s, []).append((p, b))
  excluded_sessions, unmapped, outside = [], [], []
  for s, items in sorted(by_session.items()):
    kind, reason = classify(s, [p for p, _ in items], kinds)
    if kind == 'unmapped':
      unmapped.append({'session': s, 'files': len(items), 'bytes': sum(b for _, b in items), 'reason': reason})
      continue
    if kind == 'outside':
      outside.append(s)
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
  write_csv(out / 'unmapped_sessions.csv', unmapped)
  (out / 'OUTSIDE_SESSIONS.csv').write_text(outside_table(outside))
  kinds_order = list(RELEASED_KINDS)
  source = pathlib.Path(args.root or args.listing).name + (' and a freeze\'s bucket_objects.tsv' if args.root and args.bucket_objects else '')
  lines = ['# Results bucket inventory', '', f'Source: {source} ({len(rows)} objects, '
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
  if unmapped:
    lines += ['', f'## Sessions no section 10 row maps ({len(unmapped)}): the build stops until each is mapped', '',
              '| session | files | reason |', '|---|---|---|']
    lines += [f'| {r["session"]} | {r["files"]} | {r["reason"]} |' for r in unmapped]
  (out / 'inventory.md').write_text('\n'.join(lines) + '\n')
  print('\n'.join(lines[:7 + len(kinds_order)]))
  if unmapped:
    print(f'!! {len(unmapped)} session(s) no section 10 row maps: ' + ', '.join(r['session'] for r in unmapped), file=sys.stderr)
    return 2
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
NAS_DELIVERIES = ('naspaz8-af3_run1', 'naspaz6-af3_run1', 'naspaz8-af3_run2')
ARCHIVES = [
    ('R1', 'inputs_targets_manifests.tar.gz', 'repo', ['data/inputs/*.json', 'data/inputs/ppi/*.json', 'inputs/**',
                                                       'targets/**']),
    ('R1', 'inputs_repl_large.tar.gz', 'repo', ['data/inputs_repl/**', 'data/inputs_large/**']),
    ('R1', 'msa_provenance.tar.gz', 'repo', ['data/msa/*.json', 'data/msa/pair/*.json']),
    ('R1', 'freeze.tar.gz', 'repo', ['results/freeze/**', 'scripts/freeze/**']),
    ('R1', 'session_lists_launch_records.tar.gz', 'repo', ['results/analysis/**/sessions_*.txt',
                                                          'results/analysis/main_monitor_20261013/ccd_codes.json',
                                                          'scripts/analysis_runs/*/sessions.txt',
                                                          'results/parts_*.txt', 'results/af3/.many/**',
                                                          'results/af3/.launcher/**']),
    ('R1', 'session_records.tar.gz', ['main', 'pilot2', 'memlimit', 'ppi', 'embed'], ['records']),
    ('R1', 'session_records_followups.tar.gz', list(FOLLOWUP_KINDS), ['records']),
    ('R1', 'attempt_superseded_records.tar.gz', ['superseded', 'attempt'], ['records']),
    ('R1', 'outside_freeze_records.tar.gz', ['outside'], ['records']),
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
    # follow-ups 3 to 12: two archives per follow-up, none above about 10 GB
    *[spec for k in ('cpu60', 'a100', 'msa', 'sw', 'seeds', 'h100', 'repl', 'prec5', 'large')
      for spec in (('R2', f'af3_{k}_structures.tar.gz', [k], STRUCTURES),
                   ('R2', f'af3_{k}_full_confidences.tar.gz', [k], ['full_confidences']))],
    ('R2', 'af3_feat_features.tar.gz', ['feat'], ['features']),
    ('R2', 'af3_logs_followups.tar.gz', list(FOLLOWUP_KINDS), ['af3_logs']),
    # Results of AlphaFold3's data pipeline (follow-up 6) are treated as Output (decision of 2026-11-08): the Terms define
    # Output as "the structure predictions and all related information provided by AlphaFold 3", and AlphaFold 3 as the
    # Code and Model Parameters. So they are in R2, with the three deliveries that produced them and the manifest that
    # carries their MSA and template counts.
    ('R2', 'inputs_official_pipeline.tar.gz', 'repo', ['data/inputs_official/**', 'data/inputs_official_data/**',
                                                       'data/inputs_official.tar.gz', 'inputs/official_data/**']),
    *[('R2', f'nas_{d}.tar.gz', 'repo', [f'data/nas/{d}/**', f'results/freeze/msa_{d}/**']) for d in NAS_DELIVERIES],
    # derived analysis outputs carry per-sample values computed from AlphaFold3 outputs: R2, under the Output Terms
    ('R2', 'analysis_outputs.tar.gz', 'repo', ['results/analysis/**', 'results/paper_assets/**',
                                               'results/paper_assets_*/**']),
]
# Files a repo archive's patterns reach but another archive holds: record R1 keeps only CC BY 4.0 material.
ARCHIVE_EXCLUDES = {'inputs_targets_manifests.tar.gz': ['inputs/official_data/**']}
# Where each archive is built (decision of 2026-10-15): R1 from the repository and the local freeze copy on the
# laptop; the rest from a copy of the bucket on a VM in the bucket's region (scripts/release/RUNBOOK.md).
BUILT_ON = {spec[1]: 'laptop' for spec in ARCHIVES if spec[2] == 'repo'}
BUILT_ON.update({'session_records.tar.gz': 'laptop', 'session_records_followups.tar.gz': 'laptop'})
SUPERSEDED_LABEL = 'superseded or engineering, not used in any reported result'
LEFT_OUT = {'data_json': 'AlphaFold3\'s copy of each run\'s input; check-inputs compares every one with its frozen '
                         'input on the build VM and the R2 build refuses to start unless all match',
            'work': 'local scratch (per-run inputs with seeds, compile caches); not in the bucket',
            'other_binary': 'engineering dumps (XLA and TPU log archives), none in a study session'}
# Plain files of a record besides the archives and their manifests: README.md, LICENCES.md, REDACTIONS.tsv, SHA256SUMS;
# R1 also, with the prefix release_ because a Zenodo record has no folders: release.py, RUNBOOK.md, layout.md, layout.csv,
# inventory.md, summary.csv, sessions.csv, excluded_sessions.csv, excluded_files.csv, SUPERSEDED_SESSIONS.csv,
# OUTSIDE_SESSIONS.csv, inputs_check.csv and inputs_check.json.
PLAIN_FILES = {'R1': 17, 'R2': 4}
RECORDS = {'R1': 'Inputs, targets, manifests, data freeze, session lists and run records (CC BY 4.0)',
           'R2': 'AlphaFold3 outputs, the results of its data pipeline with the follow-up 6 deliveries, and the analysis '
                 'outputs derived from them (custom licence: the AlphaFold 3 Output Terms of Use; no Creative Commons)'}


def layout(args):
  summary = list(csv.DictReader(open(pathlib.Path(args.inventory) / 'summary.csv')))
  rows = []
  for record, name, kinds, cats in ARCHIVES:
    if kinds == 'repo':
      files = archive_repo_files(name, cats, REPO)
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
    n_files = 2 * len(g) + PLAIN_FILES[rec]   # archives, their manifests, the plain files
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
  lines += ['', 'Also left out: random-weight engineering sessions, the AlphaFold3 outputs of the sessions of follow-ups 3 to 12 outside the freeze (their records are released), the freeze\'s unexplained files '
            'except launcher records of attempts that never ran (inventory/excluded_sessions.csv, inventory/excluded_files.csv), and the AlphaFold3 model weights, '
            'which are never packaged (guard() in release.py).', '']
  (out / 'layout.md').write_text('\n'.join(lines) + '\n')
  print('\n'.join(lines))
  return 0


def archive_repo_files(name, patterns, repo):
  """Files of a repo archive: what its patterns match, less its ARCHIVE_EXCLUDES."""
  out = {p for pat in patterns for p in repo_files(repo, pat)}
  out -= {p for pat in ARCHIVE_EXCLUDES.get(name, ()) for p in repo_files(repo, pat)}
  return sorted(out)


def repo_files(repo, pattern):
  """Files matching pattern under repo. A pattern ending in '/**' means every file below that folder (or below every
  folder a wildcard in it matches): pathlib's glob matches only folders with a trailing '**' before Python 3.13, so
  rglob is used for it (same result on every version)."""
  base = pathlib.Path(repo)
  if pattern.endswith('/**'):
    found = [p for d in base.glob(pattern[:-3]) for p in d.rglob('*')]
  else:
    found = base.glob(pattern)
  return sorted(p for p in found if p.is_file() and '__pycache__' not in p.parts and not p.name.startswith('.'))


# ---------------------------------------------------------------- guard

class Refused(Exception):
  pass


GS_URI = re.compile(rb'gs://([a-z0-9][a-z0-9._-]{1,220})')
EMAIL = re.compile(rb'[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}')
REDACTED_EMAIL = b'<account>'
# The contact address, by its SHA-256 so that this script (released in R1) does not carry it: allowed only in a record's
# README.md, exactly once (decision of 2026-11-08). Attribution and contact do not come from the provenance files.
CONTACT_SHA256 = '5639004dce512da3abe0761ebfde8c1f960704cdc40e754de1db2c0570ac3dc5'
CONTACT_FILES = {'README.md'}
# The only packaged files allowed to contain an email address: a copy of each is packaged with every address replaced
# by <account>, and the redaction recorded in REDACTIONS.tsv of the record. The list holds archive member paths, found
# by `release.py scan` over every local member; the files in results/ and data/ are never changed.
REDACT_LIST = HERE / 'records' / 'redact_email_paths.txt'
REDACT_REASONS = (('data/msa/', 'the contact address the ColabFold server asks for, in the user_agent field of an MSA provenance file'),
                  ('results/af3/.many/', 'gcloud account email, in the lines that say authenticated as'))


def load_redact_list(path=REDACT_LIST):
  out = {}
  if pathlib.Path(path).is_file():
    for rel in pathlib.Path(path).read_text().splitlines():
      if rel.strip():
        out[rel.strip()] = next(why for prefix, why in REDACT_REASONS if rel.startswith(prefix))
  return out


REDACT = load_redact_list()
REDACTIONS_HEADER = ['archive', 'path', 'original_sha256', 'released_sha256', 'replaced']

# Private network data and secrets in text files (decision of 2026-11-08; `release.py scan` counts them).
# (class, pattern, action, replacement). "redact": a copy with a row in REDACTIONS.tsv, because the file is a needed record
# and the value is not. "refuse": a secret must be looked at and revoked, not hidden in a published copy.
DOTTED_QUAD = re.compile(rb'\.\d{1,3}\.\d{1,3}\.')
PRIVATE = {
    'private_ipv4': (re.compile(rb'(?<![\d.])(?:10\.\d{1,3}\.\d{1,3}\.\d{1,3}|192\.168\.\d{1,3}\.\d{1,3}'
                                rb'|172\.(?:1[6-9]|2\d|3[01])\.\d{1,3}\.\d{1,3}|169\.254\.\d{1,3}\.\d{1,3}'
                                rb'|100\.(?:6[4-9]|[7-9]\d|1[01]\d|12[0-7])\.\d{1,3}\.\d{1,3})(?![\d.])'),
                     'redact', b'<private-ip>'),
    'local_host': (re.compile(rb'(?<![\w./-])[A-Za-z0-9][A-Za-z0-9-]*(?:\.[A-Za-z0-9-]+)*\.(?:local|lan)(?![\w(-]|\.\w)'),
                   'redact', b'<private-host>'),
    # the delivery labels stay: they are part of frozen files, file names and identifiers (results/freeze/INDEX.md)
    'nas_host': (re.compile(rb'naspaz[0-9]+(?!-af3(?:_run[12])?(?![\w-])|_identical|_difference)'), 'redact', b'<nas-host>'),
    'home_path': (re.compile(rb'(/Users|/home)/[A-Za-z0-9._-]+'), 'redact', rb'\1/<user>'),
    'wireguard_key': (re.compile(rb'(?<![A-Za-z0-9+/=])[A-Za-z0-9+/]{42}[AEIMQUYcgkosw048]=(?![A-Za-z0-9+/=])'), 'refuse', None),
    'ssh_key': (re.compile(rb'(?:ssh-(?:rsa|ed25519|dss)|ecdsa-sha2-nistp\d+)[ \t]+AAAA[A-Za-z0-9+/=]{20,}'
                           rb'|-----BEGIN [A-Z ]*PRIVATE KEY-----'), 'refuse', None),
    'api_token': (re.compile(rb'(?<![A-Za-z0-9_-])(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}'
                             rb'|glpat-[A-Za-z0-9_-]{16,}|AIza[0-9A-Za-z_-]{35}|ya29\.[0-9A-Za-z_-]{20,}|AKIA[0-9A-Z]{16}'
                             rb'|sk-[A-Za-z0-9_-]{20,}|hf_[A-Za-z]{30,}|xox[abprs]-[A-Za-z0-9-]{10,}|zenodo_[A-Za-z0-9]{20,})'
                             rb'|(?i:api[_-]?key|access[_-]?token|auth[_-]?token|client[_-]?secret|passw(?:or)?d|bearer)'
                             rb'["\'\s:=]+[A-Za-z0-9_\-./+]{16,}'), 'refuse', None),
}
TOKEN_WORDS = (b'ghp_', b'gho_', b'ghu_', b'ghs_', b'ghr_', b'github_pat_', b'glpat-', b'aiza', b'ya29.', b'akia', b'sk-', b'hf_', b'xox',
               b'zenodo_', b'api_key', b'api-key', b'apikey', b'access_token', b'access-token', b'accesstoken', b'auth_token',
               b'auth-token', b'authtoken', b'client_secret', b'client-secret', b'clientsecret', b'passw', b'bearer')
# Cheap tests before the patterns, so that the bytes that matter least (numeric JSON, most of the release) cost least.
PRIVATE_QUICK = {'private_ipv4': lambda d: DOTTED_QUAD.search(d), 'local_host': lambda d: b'.local' in d or b'.lan' in d,
                 'nas_host': lambda d: b'naspaz' in d, 'home_path': lambda d: b'/home/' in d or b'/Users/' in d,
                 'wireguard_key': lambda d: b'=' in d, 'ssh_key': lambda d: b'ssh-' in d or b'ecdsa-' in d or b'PRIVATE KEY' in d,
                 'api_token': lambda d: (lambda low: any(w in low for w in TOKEN_WORDS))(d.lower())}
# AlphaFold3's own output files are released unmodified: a redaction in one of them stops the build instead.
UNMODIFIED_CATEGORIES = {'model_cif', 'ranking_summary', 'full_confidences', 'terms', 'embeddings', 'features'}
# The same for the results of its data pipeline, which are treated as Output: the inference inputs and each delivered _data.json.
PIPELINE_RESULT = re.compile(r'data/inputs_official_data/.*\.json|data/nas/[^/]+/output/.*_data\.json')


def sha256_file(path):
  h = hashlib.sha256()
  with open(path, 'rb') as f:
    while block := f.read(1 << 22):
      h.update(block)
  return h.hexdigest()


def private_findings(data):
  """{class: occurrences} of PRIVATE in some bytes."""
  out = {}
  for name, (pattern, _, _) in PRIVATE.items():
    if PRIVATE_QUICK[name](data):
      n = sum(1 for _ in pattern.finditer(data))
      if n:
        out[name] = n
  return out


def check_nested(path, rel):
  """A nested archive cannot be redacted: anything guard() would act on inside it stops the build."""
  with tarfile.open(path) as tar:
    for m in tar:
      if not m.isfile():
        continue
      data = tar.extractfile(m).read()
      found = sorted(private_findings(data)) + (['email address'] if b'@' in data and EMAIL.search(data) else [])
      others = {g.decode() for g in GS_URI.findall(data)} - ALLOWED_BUCKETS
      if found or others:
        raise Refused(f'{rel}: member {m.name} holds {", ".join(found + sorted(others))}; a nested archive cannot be redacted')


def is_text_member(name):
  return not name.endswith(BINARY_SUFFIXES)


def inspect(path, rel):
  """(size, SHA-256, {private class: occurrences}, the bytes of a scanned file or None), or raise Refused. See guard()."""
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
  if name.endswith(NESTED_SUFFIXES):
    check_nested(path, rel)
  if not is_text_member(name):
    return size, sha, {}, None
  data = pathlib.Path(path).read_bytes()
  others = {m.decode() for m in GS_URI.findall(data)} - ALLOWED_BUCKETS
  if others:
    raise Refused(f'{rel}: names a bucket other than the results bucket ({", ".join(sorted(others))})')
  if rel not in REDACT and b'@' in data:
    found = EMAIL.findall(data)
    contact = rel in CONTACT_FILES and len(found) == 1 and hashlib.sha256(found[0]).hexdigest() == CONTACT_SHA256
    if found and not contact:
      raise Refused(f'{rel}: contains an email address and is not on the redaction list')
  findings = private_findings(data)
  secrets = sorted(k for k in findings if PRIVATE[k][1] == 'refuse')
  if secrets:
    raise Refused(f'{rel}: holds what looks like a secret ({", ".join(secrets)}): look at it and revoke it')
  return size, sha, findings, data


def guard(path, rel):
  """(size, SHA-256) of a file to package, or raise Refused. A text file that contains an email address is refused
  unless it is on the redaction list (REDACT), whose files prepare() packages redacted, or is a record's README.md with
  the contact address once. Secrets are refused (PRIVATE, action refuse); private network data is left to prepare()."""
  return inspect(path, rel)[:2]


def prepare(path, rel, archive):
  """(size, SHA-256, bytes or None, redaction row or None) of a file to package. Files on REDACT are packaged as a
  copy with every email address replaced by <account>, and any text file with private network data as a copy with it
  replaced (PRIVATE); the source file is not changed. AlphaFold3's own output files are never modified."""
  size, sha, findings, original = inspect(path, rel)
  on_list = rel in REDACT
  if not on_list and not findings:
    return size, sha, None, None
  notes, data = [], original
  if on_list:
    found = EMAIL.findall(original or b'')
    if not found:
      raise Refused(f'{rel}: on the redaction list but contains no email address (stale list)')
    data = EMAIL.sub(REDACTED_EMAIL, original)
    if EMAIL.search(data):
      raise Refused(f'{rel}: an email address survived the redaction')
    notes.append(f'{len(found)} occurrence(s) of {len(set(found))} email address(es) replaced by <account> ({REDACT[rel]})')
  if rel.startswith('af3/') and rel.count('/') >= 2 and category(rel.split('/', 2)[2]) in UNMODIFIED_CATEGORIES:
    raise Refused(f'{rel}: an AlphaFold3 output file with {", ".join(sorted(findings)) or "an email address"}; '
                  'these are released unmodified, so look at it before going on')
  if PIPELINE_RESULT.fullmatch(rel):
    raise Refused(f'{rel}: a result of AlphaFold3\'s data pipeline with {", ".join(sorted(findings)) or "an email address"}; '
                  'these are released unmodified, so look at it before going on')
  for name in findings:
    pattern, _, replacement = PRIVATE[name]
    data, n = pattern.subn(replacement, data)
    notes.append(f'{n} {name} replaced by {replacement.decode().replace(chr(92) + "1", ".../")}'.replace('.../' + '/', '.../'))
  left = private_findings(data)
  if left:
    raise Refused(f'{rel}: survived the redaction: {", ".join(sorted(left))}')
  new_sha = hashlib.sha256(data).hexdigest()
  return len(data), new_sha, data, {'archive': archive, 'path': rel, 'original_sha256': sha, 'released_sha256': new_sha,
                                    'replaced': '; '.join(notes)}


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
  """{session: kind} of every session folder under root (classify(): freeze kind, superseded, outside, attempt or out).
  A session no section 10 row maps stops here, with the list."""
  freeze, out = freeze_sessions(), {}
  for d in sorted(x for x in pathlib.Path(root).iterdir() if x.is_dir() and not x.name.startswith('.')):
    if sessions_filter and d.name not in sessions_filter:
      continue
    out[d.name] = classify(d.name, [str(p.relative_to(d)) for p in d.rglob('*') if p.is_file()], freeze)[0]
  unmapped = sorted(s for s, k in out.items() if k == 'unmapped')
  if unmapped:
    raise Refused(f'{len(unmapped)} session(s) outside the freeze that no section 10 row maps: {", ".join(unmapped)}')
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
    return sorted((str(p.relative_to(repo)), p) for p in archive_repo_files(name, cats, repo))
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
# Follow-up 8 (label seeds) ran seeds 6 to 20 on the frozen inputs, which list seeds 1 to 5: its plans pass
# --seeds_beyond_input (harness/run_af3.py; specification row of 2026-10-29). Allowed for that kind and those seeds only.
SEEDS_BEYOND_INPUT = {'seeds': frozenset(range(6, 21))}
TOP_LEVEL_SAME = ('name', 'dialect')
TOP_LEVEL_EMPTY = ('bondedAtomPairs', 'userCCD')


def compare_data_json(inp, dj, seeds, beyond_input=frozenset()):
  """Differences between a frozen input and AlphaFold3's *_data.json of one process (empty list: same). Compared:
  name and dialect; modelSeeds of the data JSON equal to the process's seeds and among the input's (or, for follow-up 8
  only, among seeds 6 to 20: SEEDS_BEYOND_INPUT); bondedAtomPairs
  and userCCD absent or empty in both; the same number and order of sequence entries, each of the same type (protein,
  ligand, ...) with every field equal (id, sequence, unpairedMsa, pairedMsa, templates, ccdCodes, ...), a field
  missing from the input being allowed only with AlphaFold3's default (modifications: []). version is not compared
  (AlphaFold3 writes its own dialect version)."""
  diff = []
  for k in TOP_LEVEL_SAME:
    if inp.get(k) != dj.get(k):
      diff.append(f'{k}: {inp.get(k)!r} != {dj.get(k)!r}')
  if list(dj.get('modelSeeds') or []) != list(seeds) or not set(seeds) <= set(inp.get('modelSeeds') or []) | set(beyond_input):
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
  try:
    kinds_of = session_kinds(root, sessions_filter)
  except Refused as e:
    print(f'!! {e}', file=sys.stderr)
    return 2
  rows, inputs = [], {}
  for s, kind in sorted(kinds_of.items()):
    if kind not in EARLY_KINDS + FOLLOWUP_KINDS + ('superseded',):
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
          d = compare_data_json(inp, json.loads(dj.read_text()), r.get('seeds') or [], SEEDS_BEYOND_INPUT.get(kind, frozenset()))
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
  odd = unexplained_files()
  written, restore = [], {}   # restore: REDACTIONS.tsv as it was before this build
  try:
    kinds_of = session_kinds(root, sessions_filter)
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
      if 'outside' in kinds:
        extra.append(('OUTSIDE_SESSIONS.csv', outside_table(sorted({m.split('/')[1] for m, _ in cand})).encode()))
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
      if (d / name).stat().st_size > ARCHIVE_ABOUT_MAX_BYTES:
        print(f'!! {record}/{name}: {(d / name).stat().st_size / 1e9:.1f} GB, above the plan of about 10 GB per archive', file=sys.stderr)
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
  return finish_or_refuse(sorted({p.parent for p in written} | {f.parent for f in restore}))


def superseded_table(sessions):
  return 'session,label,section_10_row\n' + ''.join(f'{s},"{SUPERSEDED_LABEL}","{SUPERSEDED[s]}"\n' for s in sessions)


def outside_table(sessions):
  return 'session,label,section_10_row\n' + ''.join(f'{s},"{OUTSIDE_LABEL}","{outside_row(s)}"\n' for s in sessions)


def plain_guard(src, rel):
  """A plain file of a record is not redacted: private network data in one stops the build."""
  findings = inspect(src, rel)[2]
  if findings:
    raise Refused(f'{rel}: a plain file of the record with {", ".join(sorted(findings))}')


def finish_or_refuse(dirs):
  try:
    return finish_records(dirs)
  except Refused as e:
    print(f'!! refused: {e}', file=sys.stderr)
    return 2


def finish_records(dirs):
  """README.md (and for R1 the release tooling and lists) into each record folder, then SHA256SUMS of everything in
  it. Run again (command sums) after archives built elsewhere are copied in."""
  for d in dirs:
    plain = [(HERE / 'records' / d.name / 'README.md', d / 'README.md'), (HERE / 'records' / 'LICENCES.md', d / 'LICENCES.md')]
    if d.name == 'R1':   # the release tooling and its lists travel with record R1, as plain files
      plain += [(HERE / 'release.py', d / 'release_release.py'), (HERE / 'RUNBOOK.md', d / 'release_RUNBOOK.md')]
      plain += [(OUT_DIR / n, d / f'release_{pathlib.Path(n).name}') for n in
                ('layout.md', 'layout.csv', 'inventory/inventory.md', 'inventory/summary.csv',
                 'inventory/sessions.csv', 'inventory/excluded_sessions.csv', 'inventory/excluded_files.csv',
                 'inventory/OUTSIDE_SESSIONS.csv')]
    for src, dst in plain:
      if src.is_file():
        plain_guard(src, str(dst.relative_to(d)))
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(src.read_bytes())
    if d.name == 'R1':
      (d / 'release_SUPERSEDED_SESSIONS.csv').write_text(superseded_table(sorted(SUPERSEDED)))
    files = sorted(p for p in d.rglob('*') if p.is_file() and p.name != 'SHA256SUMS')
    nested = [p for p in files if p.parent != d]
    if nested:
      raise Refused(f'{d.name}: {nested[0].relative_to(d)} is in a folder, and a Zenodo record has none')
    (d / 'SHA256SUMS').write_text(''.join(f'{sha256_file(p)}  {p.relative_to(d)}\n' for p in files))
    total = sum(p.stat().st_size for p in d.rglob('*') if p.is_file())
    print(f'   {d.name}/SHA256SUMS: {len(files)} files; record {len(files) + 1} files, {total / 1e9:.2f} GB')
    if len(files) + 1 > ZENODO_MAX_FILES or total > ZENODO_MAX_BYTES:
      raise Refused(f'{d.name}: {len(files) + 1} files, {total / 1e9:.1f} GB, over Zenodo\'s {ZENODO_MAX_FILES} files or '
                    f'{ZENODO_MAX_BYTES / 1e9:.0f} GB per record')
  return 0


def sums(args):
  return finish_or_refuse(sorted(d for d in pathlib.Path(args.out).iterdir() if d.is_dir() and d.name in RECORDS))


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


def scan_member(item):
  """(archive, member, {class: occurrences}, refusal or None), with the checks of guard() and prepare() and no build."""
  archive, rel, src = item
  try:
    size, sha, findings, data = inspect(src, rel)
  except Refused as e:
    return archive, rel, {}, str(e)
  found = dict(findings)
  if data is not None and b'@' in data:
    emails = EMAIL.findall(data)
    if emails:
      found['email'] = len(emails)
  return archive, rel, found, None


def scan(args):
  """guard()'s privacy checks over every local member of the chosen archives. Writes counts and paths, never a value."""
  import multiprocessing
  root, repo, out = pathlib.Path(args.root), pathlib.Path(args.repo), pathlib.Path(args.out)
  out.mkdir(parents=True, exist_ok=True)
  wanted = set(args.archives.split(',')) if args.archives else None
  try:
    kinds_of = session_kinds(root)
  except Refused as e:
    print(f'!! {e}', file=sys.stderr)
    return 2
  odd = unexplained_files()
  items = []
  for spec in ARCHIVES:
    if not wanted or spec[1] in wanted:
      items += [(spec[1], m, str(src)) for m, src in members(spec, root, repo, None, kinds_of, odd)]
  totals, files, refusals = {}, [], []
  with multiprocessing.Pool(args.jobs) as pool:
    for archive, rel, found, refusal in pool.imap_unordered(scan_member, items, chunksize=64):
      if refusal:
        refusals.append({'archive': archive, 'path': rel, 'refusal': refusal})
      for cls, n in found.items():
        action = 'on the list, redact' if cls == 'email' and rel in REDACT else 'contact' if cls == 'email' else PRIVATE[cls][1]
        t = totals.setdefault((archive, cls, action), [0, 0])
        t[0] += 1
        t[1] += n
        files.append({'archive': archive, 'path': rel, 'class': cls, 'occurrences': n})
  write_csv(out / 'privacy_scan.csv', [{'archive': a, 'class': c, 'action': x, 'files': v[0], 'occurrences': v[1]}
                                       for (a, c, x), v in sorted(totals.items())])
  with open(out / 'privacy_files.tsv', 'w', newline='') as f:
    w = csv.DictWriter(f, ['archive', 'path', 'class', 'occurrences'], delimiter='\t', lineterminator='\n')
    w.writeheader()
    w.writerows(sorted(files, key=lambda r: (r['archive'], r['path'], r['class'])))
  write_csv(out / 'privacy_refusals.csv', refusals)
  print(f'scanned {len(items)} members; {len(files)} findings in privacy_files.tsv; {len(refusals)} refusals')
  for (a, c, x), v in sorted(totals.items()):
    print(f'   {a}: {c} ({x}): {v[0]} files, {v[1]} occurrences')
  for r in refusals[:20]:
    print(f'!! {r["refusal"]}')
  return 2 if refusals else 0


def archives(args):
  print(','.join(spec[1] for spec in ARCHIVES if BUILT_ON.get(spec[1], 'vm') == args.built_on))
  return 0


def merge_redactions(args):
  d = pathlib.Path(args.out) / args.record
  rows = list(csv.DictReader(open(getattr(args, 'from')), delimiter='\t'))
  write_redactions(d, rows)
  print(f'{d}/REDACTIONS.tsv: {len(rows)} rows merged, {sum(1 for _ in open(d / "REDACTIONS.tsv")) - 1} in all')
  return 0


def main():
  ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
  sub = ap.add_subparsers(dest='cmd', required=True)
  p = sub.add_parser('inventory')
  p.add_argument('--listing', default=str(OUT_DIR / 'inventory' / 'bucket_listing.txt'))
  p.add_argument('--root', help='walk a local copy instead of reading a listing')
  p.add_argument('--bucket_objects', help='with --root: a bucket_objects.tsv of a freeze, for the sizes of bucket-only files')
  p.add_argument('--launcher', help='with --root: results/af3/.launcher, for sessions with no local folder')
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
  p = sub.add_parser('scan')
  p.add_argument('--root', required=True)
  p.add_argument('--repo', default=str(REPO))
  p.add_argument('--out', required=True)
  p.add_argument('--archives')
  p.add_argument('--jobs', type=int, default=6)
  p = sub.add_parser('archives')
  p.add_argument('--built_on', required=True, choices=['laptop', 'vm'])
  p = sub.add_parser('merge-redactions')
  p.add_argument('--out', required=True)
  p.add_argument('--record', required=True, choices=sorted(RECORDS))
  p.add_argument('--from', required=True)
  p = sub.add_parser('sums')
  p.add_argument('--out', required=True)
  p = sub.add_parser('verify')
  p.add_argument('--out', required=True)
  args = ap.parse_args()
  return {'inventory': inventory, 'layout': layout, 'build': build, 'check-inputs': check_inputs, 'scan': scan, 'sums': sums,
          'archives': archives, 'merge-redactions': merge_redactions, 'verify': verify}[args.cmd](args)


if __name__ == '__main__':
  sys.exit(main())
