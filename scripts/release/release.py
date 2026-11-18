"""Data release for Zenodo: inventory, layout, input check, archive building and verification (prepare and dry-run
only; this script never contacts a bucket or Zenodo). Build plan: scripts/release/RUNBOOK.md.

    python3 scripts/release/release.py inventory [--listing results/release/inventory/bucket_listing.txt]
    python3 scripts/release/release.py inventory --root results/af3 --bucket_objects TSV --launcher results/af3/.launcher
    python3 scripts/release/release.py layout
    python3 scripts/release/release.py scan --root DIR --out DIR [--repo .] [--archives A,B]
    python3 scripts/release/release.py check-inputs --root DIR --out DIR [--sessions S1,S2] [--repo .]
    python3 scripts/release/release.py build --root DIR --out DIR [--archives A,B] [--sessions S1,S2] [--repo .] [--paper_ref REF]
        [--code_ref REF]
    python3 scripts/release/release.py archives --built_on laptop|vm [--batch B]
    python3 scripts/release/release.py vm-lists --out DIR [--batch B]
    python3 scripts/release/release.py merge-redactions --out DIR --record R1 --from FILE
    python3 scripts/release/release.py sums --out DIR [--previous DIR]
    python3 scripts/release/release.py verify --out DIR [--previous DIR] [--repo .]

Sessions (classify()): the data freeze's sessions (FREEZE/sessions.tsv: main, pilot2, memlimit, ppi, embed, the
follow-ups cpu60, a100, feat, msa, sw, seeds, h100, repl, prec5, large, and the labels added after the first build,
LATER_LABELS: fu13, fu14, rcx, hlo, memlimit_ah, warm_th, whose sessions outside the freeze are "outside_<batch>",
records only, each mapped to its row by the label's rules; a session of these labels no rule maps, or any session
launched after the first build that no label covers, is "unmapped" and stops the build); "superseded" sessions with the official weights
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
archives: prints the comma-separated names of the archives built on the laptop or on the VM (BUILT_ON); with --batch,
only those of that batch (BATCH_OF), the archives an incremental version builds on the VM.
vm-lists: the VM's copy lists (sessions_release.txt, sessions_outside.txt) from the inventory, for one batch or all.
Incremental versions: sums --previous DIR keeps the files of the previous version that are not rebuilt (their SHA-256 in
SHA256SUMS, their rows in REDACTIONS.tsv) and writes zenodo_changes_<record>.csv (keep, replace, add or delete for each
file of the Zenodo draft); verify --previous accepts those kept files. build --paper_ref REF puts the paper's assets
(paper/assets at that git commit, read with git archive) into analysis_outputs.tar.gz; build --code_ref REF writes the
copy of the code into R1 (CODE_ARCHIVE: the commit tagged CODE_TAG, without paper/assets, never redacted).
merge-redactions: adds the rows of a REDACTIONS.tsv written elsewhere (the other machine's build) to a record's.
verify: recomputes SHA256SUMS and every member's SHA-256 from the archives, and compares the code copy with its commit
when the repository (--repo) holds it.

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
1 GiB in --out that is not one of the expected archives is refused too. Email addresses are found with systemd unit
names left out (find_emails()). An XLA dump file is refused if its printed constants exceed XLA_DUMP_CONSTANT_LIMIT
characters or it holds an MLIR resource blob (the model weights must never be inside one). Standard library and git.
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
import subprocess
import sys
import tarfile
import tempfile

HERE = pathlib.Path(__file__).resolve().parent            # scripts/release: this script, records/<R>/README.md
REPO = HERE.parents[1]
OUT_DIR = REPO / 'results' / 'release'   # inventory and layout outputs
FREEZE = REPO / 'results' / 'freeze' / 'followups1316_20261117'   # the final freeze (results/freeze/INDEX.md)
FREEZE_ID = '4ec185c5a929738439de3b916a9c8475c4c3abd218fa140bfd460acf917d4486'
# The first build (uploaded to the two drafts on 2026-11-09) was made from freeze 68923012, at commit 366159b, then tagged
# zenodo-v1 (the tag now names the commit of this release, CODE_TAG; 'zenodo-v1' stays the name of the first build's
# batch in BATCH_OF); its archives are kept byte for byte unless a later batch rebuilds them (BATCH_OF,
# REBUILT_EACH_VERSION; RUNBOOK.md).
FIRST_BUILD = {'tag': 'zenodo-v1', 'freeze_id': '689230120499e5ba9d903051553c6554f3412f462456461757398f8fd3560a1a',
               'listing_cut': '20261109T000000Z'}
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
         'records': 0.15, 'embeddings': 1.0, 'features': 0.92, 'repo': 0.331,
         # measured on the local build of 2026-11-17: af3_fu13_captures.tar.gz 6,451.0 MiB raw to 5,153.6 MiB (the captures
         # are bfloat16 bits in an uncompressed .npz; xz and byte shuffling reach 0.70 at best), hlo_xla_dumps.tar.gz
         # 1,220.9 MiB to 151.3 MiB
         'captures': 0.799, 'xla_dump': 0.124}


def category(path):
  """The release category of a path inside a session folder."""
  name = path.rsplit('/', 1)[-1]
  if '/work/' in f'/{path}' or name in {'.uploaded', '.wrapper_done'}:
    return 'work'               # local scratch (inputs with seeds, compile caches): never released
  if '/xla_dump/' in f'/{path}':
    return 'xla_dump'           # follow-up 16: XLA's dumps of the compiled model (HLO and MLIR text, options, reports)
  if name.endswith('.npz') and '_capture' in name:
    return 'captures'           # follow-up 13: the layer-wise captures (fu13_capture_seed-<s>.npz)
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

# ---------------------------------------------------------------- labels released after the first build
# Adding a label to a later Zenodo version is one entry of LATER_LABELS (and, for a new batch, one of BATCHES):
# its kind, archives, records, rows and the classification of its sessions all follow from it.
#   kind      the freeze kind of its frozen sessions (FREEZE/sessions.tsv)
#   plans     the plan prefixes of its session names (<stamp>_<platform>_<plan>...)
#   batch     the release batch that adds it (BATCHES): names its R1 archives of run records
#             (session_records_<batch>.tar.gz) and of sessions outside the freeze (outside_freeze_records_<batch>.tar.gz)
#   archives  {archive: (record, categories)}: its outputs, grouped by follow-up; labels that name the same archive
#             share it. Built on the VM (from the bucket), like every archive made from session outputs.
#   rules     (plan prefix, first and last launch stamp, section 10 row) for each of its sessions outside the freeze:
#             records only, in R1. A session named like the label that is neither frozen nor mapped stops the build.
BATCHES = {
    'followups1316': {'freeze': 'followups1316_20261117', 'freeze_id': FREEZE_ID, 'sessions': 439,
                      'what': 'follow-ups 13 to 16, memlimit_ah and warm_th'},
}
CURRENT_BATCH = 'followups1316'
OUTPUTS = ['model_cif', 'ranking_summary', 'terms', 'full_confidences', 'af3_logs']
# The record of follow-up 16's XLA dumps is Lorenzo's decision (records/LICENCES.md, "XLA dumps of follow-up 16"):
# R2 is the proposal, by the reading of decision (5) of 2026-11-08; set 'R1' here if he decides otherwise.
HLO_DUMPS_RECORD = 'R2'
R13 = '2026-11-12, Follow-up 13 (label why), complete'
R14 = '2026-11-13, Follow-up 14 (label f32), complete'
R15 = '2026-11-15, Follow-up 15 (label rcx), complete'
LATER_LABELS = [
    {'kind': 'fu13', 'plans': ('why_',), 'batch': 'followups1316',
     'archives': {'af3_fu13.tar.gz': ('R2', OUTPUTS)},   # its layer captures are left out (LEFT_OUT, decision of 2026-11-17)
     'rules': [('why_l5_', '20261110T091900Z', '20261110T115300Z',
                f'{R13}, item 1 (L5: the first VM stopped at setup on the apt lock; attempts with no VM, no capacity)'),
               ('why_lg_', '20261110T092000Z', '20261110T182500Z',
                '2026-11-10, Follow-up 13, LG stopped, and fix of the layer-wise capture on the L4 and relaunch of LG; '
                f'{R13}, item 3 (the first LG session, all 20 processes exit 1, kept and not listed; attempts with no VM)'),
               ('why_gf_', '20261110T101300Z', '20261111T023300Z',
                f'{R13}, item 5 (Gf: attempts with no VM, no capacity; why_gf_29 without a job, its VM orphaned)'),
               ('why_p_', '20261111T042400Z', '20261112T110300Z',
                f'the four rows of 2026-11-12 on arm P; {R13}, item 6 (P: Flex-start and on-demand attempts without '
                'capacity, quota, a queued request deleted before any job)')]},
    {'kind': 'fu14', 'plans': ('f32_',), 'batch': 'followups1316',
     'archives': {'af3_fu14.tar.gz': ('R2', OUTPUTS)},
     'rules': [('f32_v5f_', '20261112T201200Z', '20261113T024000Z',
                f'{R14}, items 3 and 4 (v5e: attempts with no VM, no capacity; V5f r2, setup exit 100 on the apt lock)'),
               ('f32_af_', '20261112T201200Z', '20261113T105000Z',
                f'rows of 2026-11-12 and 2026-11-13 on the A100; {R14}, items 2 to 9 (A100: Spot preemptions, setup '
                'exit 100 in us-east4-c, attempts with no VM, the image family 404, a launcher ended during the copy, '
                'gcloud failures without network)'),
               ('f32_hf_', '20261112T201100Z', '20261112T231500Z',
                f'{R14}, item 2 (H100: Spot preemptions before any process finished)'),
               ('f32_ad_', '20261113T122600Z', '20261113T133700Z',
                f'{R14}, items 4 and 10 (item b2, Ad: attempts with no VM, no capacity)')]},
    {'kind': 'rcx', 'plans': ('rcx_',), 'batch': 'followups1316',
     'archives': {'af3_rcx.tar.gz': ('R2', OUTPUTS)},
     'rules': [('rcx_af_r2_', '20261114T191300Z', '20261114T204900Z',
                f'{R15}, item 2 (Af r2: Spot preemptions during setup, before any AlphaFold 3 run)'),
               ('rcx_gt_r2_', '20261114T221300Z', '20261115T034300Z',
                f'{R15}, item 4 (GT r2: Spot preemptions; finished processes kept and not used)')]},
    {'kind': 'hlo', 'plans': ('hlo_',), 'batch': 'followups1316',
     'archives': {'af3_hlo.tar.gz': ('R2', OUTPUTS), 'hlo_xla_dumps.tar.gz': (HLO_DUMPS_RECORD, ['xla_dump'])},
     'rules': [('hlo_v5_', '20261116T140100Z', '20261116T141900Z',
                '2026-11-16, Follow-up 16 (label hlo), complete (v5e: three attempts with no VM, no capacity)')]},
    {'kind': 'memlimit_ah', 'plans': ('memlimit_ah_',), 'batch': 'followups1316',
     'archives': {'af3_memlimit_ah_warm_th.tar.gz': ('R2', OUTPUTS)}, 'rules': []},
    {'kind': 'warm_th', 'plans': ('warm_th_',), 'batch': 'followups1316',
     'archives': {'af3_memlimit_ah_warm_th.tar.gz': ('R2', OUTPUTS)}, 'rules': []},
]
LATER_KINDS = tuple(lab['kind'] for lab in LATER_LABELS)
OUTSIDE_KINDS = ('outside',) + tuple(f'outside_{b}' for b in BATCHES)
RELEASED_KINDS = EARLY_KINDS + FOLLOWUP_KINDS + LATER_KINDS + ('superseded', 'attempt') + OUTSIDE_KINDS
STAMPED_SESSION = re.compile(r'^(\d{8}T\d{6}Z)_[a-z0-9]+_(.+)$')


def later_label(session):
  """(LATER_LABELS entry, launch stamp, plan part of the name) of a session named like a later label, or None."""
  m = STAMPED_SESSION.match(session)
  if m:
    for lab in LATER_LABELS:
      if m.group(2).startswith(lab['plans']):
        return lab, m.group(1), m.group(2)
  return None


def outside_row(session):
  """The section 10 row of a session outside the freeze, or None."""
  if session in OUTSIDE_NAMED:
    return OUTSIDE_NAMED[session]
  m = FU612_SESSION.match(session)
  if m:
    for prefix, first, last, row in OUTSIDE_RULES:
      if m.group(2).startswith(prefix) and first <= m.group(1) <= last:
        return row
  hit = later_label(session)
  if hit:
    lab, stamp, plan = hit
    for prefix, first, last, row in lab['rules']:
      if plan.startswith(prefix) and first <= stamp <= last:
        return row
  return None


def classify(session, paths, kinds):
  """(kind, reason): a freeze kind, 'superseded' (SUPERSEDED), 'outside' (a session of follow-ups 3 to 12 outside the
  freeze, mapped to its row: records only, released in R1), 'outside_<batch>' (the same for a later label, LATER_LABELS),
  'unmapped' (a session of follow-ups 6 to 12 or of a later label that no row explains, or any session launched after
  the first build that no label covers: stops the build), 'attempt' (no VM run records: launcher records only, released
  in R1), or 'out' (not released) with the reason."""
  ran = any(p.endswith(('run.json', 'runs.jsonl')) for p in paths)
  if session in SUPERSEDED:
    return 'superseded', SUPERSEDED[session]
  if session not in kinds:
    hit = later_label(session)
    row = outside_row(session)
    if row:
      return (f'outside_{hit[0]["batch"]}' if hit else 'outside'), row
    if hit:
      return 'unmapped', f'named like a {hit[0]["kind"]} session, not in the freeze, and no section 10 row maps it'
    if FU612_SESSION.match(session):
      return 'unmapped', 'named like a follow-up 6 to 12 session, not in the freeze, and no section 10 row maps it'
    m = STAMPED_SESSION.match(session)
    if m and m.group(1) >= FIRST_BUILD['listing_cut']:
      return 'unmapped', ('launched after the first build, not in the freeze and of no label in LATER_LABELS: add its '
                          'label or decide that it stays out')
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
    # derived analysis outputs carry per-sample values computed from AlphaFold3 outputs: R2, under the Output Terms;
    # the paper's own assets come from a git ref (build --paper_ref, PAPER_ASSETS), not from these folders
    ('R2', 'analysis_outputs.tar.gz', 'repo', ['results/analysis/**', 'results/paper_assets/**',
                                               'results/paper_assets_*/**']),
]
FIRST_BUILD_ARCHIVES = [spec[1] for spec in ARCHIVES]


def later_archives():
  """The archive specs LATER_LABELS defines, in table order: per batch, the R1 run records of its frozen sessions and
  the R1 records of its sessions outside the freeze, then each label's archives (shared ones merged)."""
  specs, order = {}, []
  for batch in BATCHES:
    labs = [lab for lab in LATER_LABELS if lab['batch'] == batch]
    for name, rec, kinds, cats in ((f'session_records_{batch}.tar.gz', 'R1', [lab['kind'] for lab in labs], ['records']),
                                   (f'outside_freeze_records_{batch}.tar.gz', 'R1', [f'outside_{batch}'], ['records'])):
      specs[name] = (rec, name, kinds, cats)
      order.append(name)
    for lab in labs:
      for name, (rec, cats) in lab['archives'].items():
        if name in specs:
          assert specs[name][0] == rec and specs[name][3] == cats, name
          specs[name][2].append(lab['kind'])
        else:
          specs[name] = (rec, name, [lab['kind']], list(cats))
          order.append(name)
  return [specs[n] for n in order]


ARCHIVES += later_archives()
# A copy of the code in R1 (decision D5 of 2026-11-18), so that the code has a DOI: `git archive` of the commit tagged
# CODE_TAG (build --code_ref REF; the commit REF names must be the tag's, since the archive's name carries the tag), every
# file under CODE_PREFIX, with CODE_COMMIT.txt giving the full hash. Committed files only, packaged byte for byte with
# their mode (0644, or 0755 for executables): the copy is never redacted, so it stays identical to the public commit, and
# anything guard() would refuse or redact stops the build unless CODE_ALLOWED lists it (check_code_file()). CODE_EXCLUDES
# stay out: paper/assets/ holds tables and figures computed from AlphaFold 3 outputs, which belong in R2 under the Output
# Terms (analysis_outputs.tar.gz takes them from the same commit, build --paper_ref).
CODE_TAG = 'zenodo-v1'
CODE_ARCHIVE = f'code_alphafold-on-tpu_{CODE_TAG}.tar.gz'
CODE_PREFIX = f'alphafold-on-tpu-{CODE_TAG}/'
CODE_COMMIT_FILE = 'CODE_COMMIT.txt'
CODE_EXCLUDES = ('paper/assets',)
# The licence files the copy must carry (the build refuses a commit without them): the repository's MIT licence, NOTICE
# (the files copied from github.com/lorenzopazienza/alphafold-tpu-benchmark, MIT, Lorenzo Pazienza and Ihab El Bani), that
# repository's licence, and the Apache License 2.0 of the AlphaFold 3 lines of af3_tpu/af3_tpu.patch.
CODE_REQUIRED = ('LICENSE', 'NOTICE', 'LICENSES/Apache-2.0', 'LICENSES/MIT-alphafold-tpu-benchmark')
ARCHIVES.append(('R1', CODE_ARCHIVE, 'git', [CODE_TAG]))
# Files a repo archive's patterns reach but another archive holds: record R1 keeps only CC BY 4.0 material. The ISC
# draft's assets (results/paper_assets_isc) are superseded by the paper's own, taken from the tagged paper commit.
ARCHIVE_EXCLUDES = {'inputs_targets_manifests.tar.gz': ['inputs/official_data/**'],
                    'analysis_outputs.tar.gz': ['results/paper_assets_isc/**']}
# The paper's tables and figures (paper/assets of the paper commit that is tagged for the release), read from git with
# `git archive` (build --paper_ref REF) into analysis_outputs.tar.gz, with PAPER_ASSETS_COMMIT.txt naming the commit.
PAPER_ASSETS = ('analysis_outputs.tar.gz', 'paper/assets')
# Further members of an archive, from the repository: (member, repository path or None for the text given).
ARCHIVE_NOTICES = {'hlo_xla_dumps.tar.gz': [
    ('LICENSE_ALPHAFOLD3_CODE_APACHE-2.0.txt', 'LICENSES/Apache-2.0'),
    ('NOTICE_XLA_DUMPS.txt', None)]}
XLA_DUMPS_NOTICE = '''XLA dumps of follow-up 16 (label hlo): the text files XLA wrote while JAX compiled AlphaFold 3's model
(AlphaFold 3 v3.0.4) on TPU v5e and v6e: HLO and MLIR text before and after optimisation, buffer assignment and memory
reports, compiler options and flags. They are compiled from AlphaFold 3's code, Copyright 2024 DeepMind Technologies
Limited, licensed under the Apache License 2.0 (LICENSE_ALPHAFOLD3_CODE_APACHE-2.0.txt); they carry the names and
shapes of the model's parameters and the source locations of that code.

They do not hold the model parameters (weights): the parameters are arguments of the compiled function, not
constants. Checked before release on every one of the 21,696 files: no printed constant in the HLO text is longer than
16 characters (XLA elides large constants as {...}), the hex-encoded constants in the MLIR files total 80,384 bytes over
all files and at most 5,024 bytes in one file, against 1,145,669,864 bytes of parameters named like weights in the entry
of each compiled model and a weights file of 1,020,545,840 bytes. The release build refuses any dump file whose printed
constants exceed 65,536 characters, or that holds an MLIR resource blob (release.py, XLA_DUMP_CONSTANT_LIMIT).

Changes: home paths are replaced by /home/<user> (REDACTIONS.tsv of the record lists each file, with the SHA-256 of the
original and of the released copy); nothing else is changed.
'''
# Where each archive is built (decision of 2026-10-15): R1 from the repository and the local freeze copy on the
# laptop; the rest from a copy of the bucket on a VM in the bucket's region (scripts/release/RUNBOOK.md).
BUILT_ON = {spec[1]: 'laptop' for spec in ARCHIVES if spec[2] in ('repo', 'git')}
BUILT_ON.update({'session_records.tar.gz': 'laptop', 'session_records_followups.tar.gz': 'laptop'})
BUILT_ON.update({f'session_records_{b}.tar.gz': 'laptop' for b in BATCHES})
# The batch that made each archive: the first build's archives are kept byte for byte in a later version unless
# rebuilt; the archives that hold the freeze, the lists and the analyses change with every version.
BATCH_OF = {name: 'zenodo-v1' for name in FIRST_BUILD_ARCHIVES}
BATCH_OF.update({spec[1]: lab_batch for spec in ARCHIVES for lab_batch in
                 {b for b in BATCHES if spec[1] in {f'session_records_{b}.tar.gz', f'outside_freeze_records_{b}.tar.gz'}
                  or any(spec[1] in lab['archives'] for lab in LATER_LABELS if lab['batch'] == b)}})
BATCH_OF[CODE_ARCHIVE] = CURRENT_BATCH   # new in this release; a later code tag is a new archive (RUNBOOK.md, Part C)
REBUILT_EACH_VERSION = ('freeze.tar.gz', 'session_lists_launch_records.tar.gz', 'analysis_outputs.tar.gz')
SUPERSEDED_LABEL = 'superseded or engineering, not used in any reported result'
LEFT_OUT = {'data_json': 'AlphaFold3\'s copy of each run\'s input; check-inputs compares every one with its frozen '
                         'input on the build VM and the R2 build refuses to start unless all match',
            'work': 'local scratch (per-run inputs with seeds, compile caches); not in the bucket',
            'other_binary': 'engineering dumps (XLA and TPU log archives), none in a study session',
            'captures': 'the layer captures of follow-up 13 (bfloat16 bits in uncompressed .npz, 0.80 of their size under gzip, '
                        '0.70 at best): R2 would pass 50 GB with them (decision of 2026-11-17); they stay in the results '
                        'bucket and in the freeze manifest with their SHA-256, available on request'}
# Plain files of a record besides the archives and their manifests: README.md, LICENCES.md, REDACTIONS.tsv, SHA256SUMS;
# R1 also, with the prefix release_ because a Zenodo record has no folders: release.py, RUNBOOK.md, layout.md, layout.csv,
# inventory.md, summary.csv, sessions.csv, excluded_sessions.csv, excluded_files.csv, SUPERSEDED_SESSIONS.csv,
# OUTSIDE_SESSIONS.csv, inputs_check.csv and inputs_check.json (the first build's check, kept), and the check of each
# later batch, inputs_check_<batch>.csv and .json.
PLAIN_FILES = {'R1': 17 + 2 * len(BATCHES), 'R2': 4}
RECORDS = {'R1': 'Inputs, targets, manifests, data freeze, session lists and run records (CC BY 4.0)',
           'R2': 'AlphaFold3 outputs, the results of its data pipeline with the follow-up 6 deliveries, the XLA dumps of '
                 'follow-up 16, and the analysis outputs derived from them (custom licence: the AlphaFold 3 Output Terms '
                 'of Use; no Creative Commons)'}


def layout(args):
  summary = list(csv.DictReader(open(pathlib.Path(args.inventory) / 'summary.csv')))
  rows = []
  for record, name, kinds, cats in ARCHIVES:
    if kinds == 'git':   # an estimate from the files tracked at HEAD; the build reads the tagged commit
      files = [REPO / p for p in git_run(REPO, 'ls-files', '-z').decode().split('\0')
               if p and not code_excluded(p) and (REPO / p).is_file()]
      raw = sum(p.stat().st_size for p in files)
      rows.append({'record': record, 'archive': name, 'contents': f'git archive of the commit tagged {CODE_TAG}, without '
                   f'{", ".join(CODE_EXCLUDES)}', 'files': len(files), 'raw_bytes': raw,
                   'estimated_bytes': int(raw * RATIO['repo'])})
      continue
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
  lines += ['', 'Also left out: random-weight engineering sessions, the AlphaFold3 outputs of the sessions of follow-ups 3 to 16 outside the freeze (their records are released), the freeze\'s unexplained files '
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
# systemd unit names of template instances (NAME, at sign, INSTANCE, dot, TYPE; in the job logs of the VMs) look like
# addresses to EMAIL. Only the unit types that are not top-level domains are exempt (IANA root zone list, version
# 2026111700): `target` is a TLD, so a unit of type target stays an address. A unit is matched as a whole token (systemd's
# characters: letters, digits, ':', '-', '_', '.', backslash), so an address that merely contains a unit name (with a
# '+' before it, a domain after the type, or the TLD `services`) is still an address; and at each position the unit is
# tried first, so it is never half-redacted. Tests: scripts/release/test_release.py.
SYSTEMD_UNIT_TYPES = (b'service', b'socket', b'device', b'mount', b'automount', b'swap', b'path', b'timer', b'slice',
                      b'scope')
SYSTEMD_UNIT = (rb'(?<![A-Za-z0-9_.:\\%+-])[A-Za-z0-9_.:\\-]+@[A-Za-z0-9_.:\\-]*\.(?:' + b'|'.join(SYSTEMD_UNIT_TYPES)
                + rb')(?!\.?[A-Za-z0-9_\\-])')
EMAIL_OR_UNIT = re.compile(rb'(?P<unit>' + SYSTEMD_UNIT + rb')|(?P<email>' + EMAIL.pattern + rb')')


def find_emails(data):
  """Every email address in some bytes (EMAIL), systemd unit names left out."""
  if b'@' not in data:
    return []
  return [m.group('email') for m in EMAIL_OR_UNIT.finditer(data) if m.group('email')]


def redact_emails(data):
  """The bytes with every email address replaced by <account>; systemd unit names are kept."""
  return EMAIL_OR_UNIT.sub(lambda m: m.group(0) if m.group('unit') else REDACTED_EMAIL, data)
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
UNMODIFIED_CATEGORIES = {'model_cif', 'ranking_summary', 'full_confidences', 'terms', 'embeddings', 'features', 'captures'}
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
      found = sorted(private_findings(data)) + (['email address'] if find_emails(data) else [])
      others = {g.decode() for g in GS_URI.findall(data)} - ALLOWED_BUCKETS
      if found or others:
        raise Refused(f'{rel}: member {m.name} holds {", ".join(found + sorted(others))}; a nested archive cannot be redacted')


def is_text_member(name):
  return not name.endswith(BINARY_SUFFIXES)


# An XLA dump must not carry the model weights as constants. The weights are arguments of the compiled function; the
# constants of the dumps of follow-up 16 are small tables and vectors of AlphaFold 3's code (at most 15,411 characters
# of printed constants in one file, 2026-11-17). Any dump file above this, or with an MLIR resource blob (where MLIR
# keeps large constants), is refused: look at it before anything is released.
XLA_DUMP_CONSTANT_LIMIT = 65536
HLO_CONSTANT = re.compile(rb'\bconstant\(')
MLIR_DENSE = re.compile(rb'\bdense<')


def dump_constant_chars(data):
  """Characters of printed constant literals in an XLA dump file: HLO text constant(...) and MLIR dense<...>."""
  total = 0
  for m in HLO_CONSTANT.finditer(data):
    i = j = m.end()
    depth = 1
    while depth and j < len(data):
      c = data[j]
      depth += (c == 40) - (c == 41)   # ( and )
      j += 1
    total += j - i
  for m in MLIR_DENSE.finditer(data):
    end = data.find(b'>', m.end())
    total += (end if end >= 0 else len(data)) - m.end()
  return total


def check_dump_constants(data, rel):
  if b'dense_resource' in data or b'dialect_resources' in data:
    raise Refused(f'{rel}: an XLA dump with an MLIR resource blob (large constants): look at it for the model weights')
  n = dump_constant_chars(data)
  if n > XLA_DUMP_CONSTANT_LIMIT:
    raise Refused(f'{rel}: an XLA dump with {n} characters of printed constants (limit {XLA_DUMP_CONSTANT_LIMIT}): '
                  'look at it for the model weights')


def inspect(path, rel, code=False):
  """(size, SHA-256, {private class: occurrences}, the bytes of a scanned file or None), or raise Refused. See guard().
  code: a file of the code copy (rel is its repository path), with the buckets and addresses of CODE_ALLOWED."""
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
  if '/xla_dump/' in rel:
    check_dump_constants(data, rel)
  tests = code and rel in CODE_ALLOWED['guard_tests']
  others = set() if tests else \
      {m.decode() for m in GS_URI.findall(data)} - ALLOWED_BUCKETS - (CODE_ALLOWED['bucket'] if code else set())
  if others:
    raise Refused(f'{rel}: names a bucket other than the results bucket ({", ".join(sorted(others))})')
  if code and b'@' in data:
    found = [] if tests else [e for e in find_emails(data) if hashlib.sha256(e).hexdigest() != CONTACT_SHA256]
    if found:
      raise Refused(f'{rel}: contains an email address other than the contact (code copy, CODE_ALLOWED)')
  elif rel not in REDACT and b'@' in data:
    found = find_emails(data)
    contact = rel in CONTACT_FILES and len(found) == 1 and hashlib.sha256(found[0]).hexdigest() == CONTACT_SHA256
    if found and not contact:
      raise Refused(f'{rel}: contains an email address and is not on the redaction list')
  findings = private_findings(data)
  secrets = sorted(k for k in findings if PRIVATE[k][1] == 'refuse')
  if secrets and not tests:
    raise Refused(f'{rel}: holds what looks like a secret ({", ".join(secrets)}): look at it and revoke it')
  return size, sha, findings, data


# The code copy (CODE_ARCHIVE) holds the public commit byte for byte, so it is never redacted: what guard() would refuse
# or redact in it stops the build, except what is listed here, each by repository path, as of the release commit. The
# contact address (CONTACT_SHA256) is allowed in any file of the code: the README, CITATION.cff, the papers and the
# records' READMEs and zenodo.json give it as the contact. A new file that needs an entry stops the build: look at it first.
# NOTICE and LICENSES/ go through the same checks and need no entry: they hold no address, bucket, path or host name
# (test_code_licence_files_pass_the_guard); CODE_REQUIRED makes them part of every copy.
CODE_ALLOWED = {
    # the guard's own tests: every address, bucket, path, host name, network address, key and token in them is made up
    # for a test (each looked at on 2026-11-18)
    'guard_tests': {'scripts/release/test_release.py'},
    # buckets the code names: the private weights bucket and the earlier AlphaFold 2 results bucket (names only; neither
    # grants any access), and the made-up names of the tests and of a usage line
    'bucket': {'af2-tpu-benchmark-af3-weights', 'af2-tpu-benchmark-results', 'bucket', 'test-bucket', 'private-weights',
               'af3-results-test', 'fake-deps', 'followup-results', 'memlimit-results', 'pilot-results'},
    # a recorded log of the target selection (the same file is redacted in inputs_targets_manifests.tar.gz); the fake home
    # folder of two cloud tests; made-up paths in test strings
    'home_path': {'targets/ppi/run.log', 'tests/cloud/ctrlc.py', 'tests/cloud/resume.py', 'tests/many/test_run_many.sh',
                  'analysis/tests/test_paper_assets_isc.py'},
    # the names of the two NAS machines of follow-up 6 in the plan, the MSA ingest and its test
    'nas_host': {'notes/analysis_plan.md', 'analysis/msa.py', 'analysis/tests/test_ingest_official.py'},
}


def check_code_file(path, rel):
  """(size, SHA-256) of a file of the code copy (rel: its repository path), or raise Refused: guard()'s checks, where
  CODE_ALLOWED lists what the public code holds by design; anything else guard() would redact stops the build."""
  size, sha, findings, _ = inspect(path, rel, code=True)
  left = sorted(k for k in findings if rel not in CODE_ALLOWED.get(k, set()) | CODE_ALLOWED['guard_tests'])
  if left:
    raise Refused(f'{rel}: holds {", ".join(left)}; the code copy is never redacted: change the file before the release '
                  'commit, or list it in CODE_ALLOWED with a reason')
  return size, sha


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
    found = find_emails(original or b'')
    if not found:
      raise Refused(f'{rel}: on the redaction list but contains no email address (stale list)')
    data = redact_emails(original)
    if find_emails(data):
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
  if kinds == 'git':
    return out   # read from git by build() (build --code_ref, export_code())
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
    if kind not in EARLY_KINDS + FOLLOWUP_KINDS + LATER_KINDS + ('superseded',):
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


def write_archive(path, items, extra, modes=None):
  """Deterministic tar (gzip without time or name for .tar.gz). items: [(member, source, size, sha, bytes or None)].
  modes: {member: mode} for members that are not 0644 (the executables of the code copy)."""
  modes = modes or {}
  raw = open(path, 'wb')
  fileobj = gzip.GzipFile(filename='', mode='wb', fileobj=raw, mtime=0, compresslevel=6) if path.name.endswith('.gz') else raw
  with tarfile.open(fileobj=fileobj, mode='w', format=tarfile.GNU_FORMAT) as tar:
    for member, data in extra:
      tar.addfile(tarinfo(member, len(data)), io.BytesIO(data))
    for member, src, size, _, data in items:
      ti = tarinfo(member, size)
      ti.mode = modes.get(member, ti.mode)
      if data is not None:   # a redacted copy
        tar.addfile(ti, io.BytesIO(data))
        continue
      with open(src, 'rb') as f:
        tar.addfile(ti, f)
  if fileobj is not raw:
    fileobj.close()
  raw.close()


def build(args):
  root, repo, out = pathlib.Path(args.root), pathlib.Path(args.repo), pathlib.Path(args.out)
  wanted = set(args.archives.split(',')) if args.archives else None
  sessions_filter = set(args.sessions.split(',')) if args.sessions else None
  odd = unexplained_files()
  written, restore = [], {}   # restore: REDACTIONS.tsv as it was before this build
  paper_tmp = tempfile.TemporaryDirectory(prefix='af3_paper_assets_')
  code_tmp = tempfile.TemporaryDirectory(prefix='af3_code_')
  try:
    kinds_of = session_kinds(root, sessions_filter)
    paper = None   # (ref, commit, export root) of the paper assets
    if not wanted or PAPER_ASSETS[0] in wanted:
      ref = getattr(args, 'paper_ref', None)
      if not ref:
        raise Refused(f'{PAPER_ASSETS[0]} takes the paper assets from git: pass --paper_ref REF (the tagged paper commit), '
                      'or --paper_ref none for a test build without them')
      if ref != 'none':
        paper = (ref, export_paper_assets(repo, ref, pathlib.Path(paper_tmp.name)), pathlib.Path(paper_tmp.name))
    code = None    # (ref, commit, [(member, source, repository path, mode)]) of the code copy
    if not wanted or CODE_ARCHIVE in wanted:
      ref = getattr(args, 'code_ref', None)
      if not ref:
        raise Refused(f'{CODE_ARCHIVE} is a git archive: pass --code_ref REF (the commit tagged {CODE_TAG}), '
                      'or --code_ref none for a test build without it')
      if ref != 'none':
        code = export_code(repo, ref, pathlib.Path(code_tmp.name))
    # every R2 archive made from session outputs needs a passing check-inputs on the same root (not the
    # repository-made analysis_outputs.tar.gz, which holds no session file)
    if any(spec[0] == 'R2' and spec[2] != 'repo' and (not wanted or spec[1] in wanted) for spec in ARCHIVES):
      require_inputs_check(out, root, args.allow_incomplete_inputs_check)
    for spec in ARCHIVES:
      record, name, kinds, _ = spec
      if wanted and name not in wanted:
        continue
      cand = members(spec, root, repo, sessions_filter, kinds_of, odd)
      if name == PAPER_ASSETS[0] and paper:
        cand = sorted(cand + archive_tree(paper[2], PAPER_ASSETS[1]))
      if kinds == 'git' and code:
        cand = [(m, src) for m, src, _, _ in code[2]]
      if not cand:
        print(f'   {record}/{name}: ' + ('--code_ref none, skipped' if kinds == 'git' else 'no files in this root, skipped'))
        continue
      if kinds == 'git':   # the code copy is checked, never redacted (check_code_file())
        prepared = [(m, src, *check_code_file(src, path), None, None) for m, src, path, _ in code[2]]
      else:                # every file is checked before anything is written
        prepared = [(m, src, *prepare(src, m, name)) for m, src in cand]
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
      if kinds not in ('repo', 'git') and any(k in OUTSIDE_KINDS for k in kinds):
        extra.append(('OUTSIDE_SESSIONS.csv', outside_table(sorted({m.split('/')[1] for m, _ in cand})).encode()))
      if name == PAPER_ASSETS[0] and paper:
        extra.append(('PAPER_ASSETS_COMMIT.txt', f'{PAPER_ASSETS[1]}/ from git ref {paper[0]}, commit {paper[1]}\n'.encode()))
      if kinds == 'git':
        extra.append((CODE_COMMIT_FILE, code_commit_text(code[1]).encode()))
      for member, src in ARCHIVE_NOTICES.get(name, ()):
        if src and not (repo / src).is_file():
          raise Refused(f'{name}: {src} is missing from the repository')
        extra.append((member, (repo / src).read_bytes() if src else XLA_DUMPS_NOTICE.encode()))
      d = out / record
      d.mkdir(parents=True, exist_ok=True)
      write_archive(d / name, items, extra, {m: mode for m, _, _, mode in code[2]} if kinds == 'git' else None)
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
  finally:
    paper_tmp.cleanup()
    code_tmp.cleanup()
  return finish_or_refuse(sorted({p.parent for p in written} | {f.parent for f in restore}))


def git_run(repo, *a):
  """The output of a git command in the repository, or raise Refused."""
  r = subprocess.run(['git', '-C', str(repo), *a], capture_output=True)
  if r.returncode:
    raise Refused(f'git {" ".join(a)}: {r.stderr.decode().strip() or "exit " + str(r.returncode)}')
  return r.stdout


def git_commit(repo, ref):
  return git_run(repo, 'rev-parse', '--verify', '--quiet', f'{ref}^{{commit}}').decode().strip()


def code_excluded(path):
  return any(path == x or path.startswith(x + '/') for x in CODE_EXCLUDES)


def code_tree(repo, commit):
  """{repository path: (bytes, mode)} of every file of a commit (`git archive`), less CODE_EXCLUDES. A link, or any
  member that is neither a file nor a folder, is refused: the copy holds files only."""
  out = {}
  with tarfile.open(fileobj=io.BytesIO(git_run(repo, 'archive', '--format=tar', commit))) as tar:
    for m in tar:
      if m.isdir():
        continue
      if not m.isfile():
        raise Refused(f'git archive of {commit}: {m.name} is not a regular file; the code copy holds files only')
      if not code_excluded(m.name):
        out[m.name] = (tar.extractfile(m).read(), 0o755 if m.mode & 0o111 else 0o644)
  if not out:
    raise Refused(f'git archive of {commit}: no file')
  return out


def code_commit_text(commit):
  """CODE_COMMIT.txt: the full hash on the first line, then what the copy is (the same bytes for any ref of the commit)."""
  return (f'{commit}\n\nThe files of {CODE_PREFIX} are the commit above, tagged {CODE_TAG}, of '
          f'https://github.com/lorenzopazienza/alphafold-on-tpu, read with git archive, byte for byte, without '
          f'{", ".join(CODE_EXCLUDES)} (tables and figures computed from AlphaFold 3 outputs: record R2, {PAPER_ASSETS[0]}).\n')


def export_code(repo, ref, dest):
  """(ref, commit, [(member, source, repository path, mode)]) of the code copy: the files of the commit REF names, less
  CODE_EXCLUDES, written under dest. The commit must be the one tag CODE_TAG names, since the archive's name carries it."""
  commit = git_commit(repo, ref)
  tagged = git_commit(repo, CODE_TAG)
  if commit != tagged:
    raise Refused(f'--code_ref {ref} is commit {commit[:12]}, but tag {CODE_TAG} names {tagged[:12]}: {CODE_ARCHIVE} '
                  f'must hold the tagged commit (move the tag, or set CODE_TAG for a new version)')
  tree = code_tree(repo, commit)
  missing = [p for p in CODE_REQUIRED if p not in tree]
  if missing:
    raise Refused(f'commit {commit[:12]} lacks {", ".join(missing)}: the code copy must carry its licence files (CODE_REQUIRED)')
  out = []
  for path, (data, mode) in sorted(tree.items()):
    target = (dest / path).resolve()
    if not str(target).startswith(str(dest.resolve()) + os.sep):
      raise Refused(f'git archive of {ref}: member {path} outside the export folder')
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)
    out.append((CODE_PREFIX + path, target, path, mode))
  return ref, commit, out


def export_paper_assets(repo, ref, dest):
  """Writes the files of PAPER_ASSETS[1] at git ref REF of the repository under dest (`git archive`, so only committed
  files, as of that commit) and returns the commit."""
  def git(*a):
    return git_run(repo, *a)
  commit = git('rev-parse', '--verify', '--quiet', f'{ref}^{{commit}}').decode().strip()
  with tarfile.open(fileobj=io.BytesIO(git('archive', '--format=tar', commit, '--', PAPER_ASSETS[1]))) as tar:
    for m in tar:
      if m.isfile():
        target = (dest / m.name).resolve()
        if not str(target).startswith(str(dest.resolve()) + os.sep):
          raise Refused(f'git archive of {ref}: member {m.name} outside {PAPER_ASSETS[1]}')
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(tar.extractfile(m).read())
  if not any(dest.rglob('*')):
    raise Refused(f'{ref}: no {PAPER_ASSETS[1]} at commit {commit}')
  return commit


def archive_tree(root, folder):
  """[(member, source)] of every file below root/folder, members relative to root."""
  return sorted((str(p.relative_to(root)), p) for p in (pathlib.Path(root) / folder).rglob('*') if p.is_file())


def superseded_table(sessions):
  return 'session,label,section_10_row\n' + ''.join(f'{s},"{SUPERSEDED_LABEL}","{SUPERSEDED[s]}"\n' for s in sessions)


def outside_table(sessions):
  return 'session,label,section_10_row\n' + ''.join(f'{s},"{OUTSIDE_LABEL}","{outside_row(s)}"\n' for s in sessions)


def plain_guard(src, rel):
  """A plain file of a record is not redacted: private network data in one stops the build."""
  findings = inspect(src, rel)[2]
  if findings:
    raise Refused(f'{rel}: a plain file of the record with {", ".join(sorted(findings))}')


def finish_or_refuse(dirs, previous=None):
  try:
    return finish_records(dirs, previous)
  except Refused as e:
    print(f'!! refused: {e}', file=sys.stderr)
    return 2


# A later version keeps the files of the previous one that it does not rebuild (incremental build, RUNBOOK.md): the
# previous version's record folder is given as --previous DIR, holding <record>/SHA256SUMS, <record>/REDACTIONS.tsv and
# Zenodo's file listing of each draft, zenodo_files_<id>.json (results/release/vm_build_20261109/ for the first build).
ZENODO_IDS = {'R1': '24057470', 'R2': '24057544'}
KEPT_PLAIN = ('release_inputs_check.csv', 'release_inputs_check.json')   # the first build's check-inputs, kept


def load_previous(previous, record):
  """{file name: (SHA-256, bytes)} of a record of the previous version."""
  prev = pathlib.Path(previous)
  sums_file = prev / record / 'SHA256SUMS'
  listing = prev / f'zenodo_files_{ZENODO_IDS[record]}.json'
  if not sums_file.is_file() or not listing.is_file():
    raise Refused(f'--previous {previous}: needs {record}/SHA256SUMS and {listing.name}')
  sizes = {f['filename']: f['filesize'] for f in json.loads(listing.read_text())}
  out = {}
  for line in sums_file.read_text().splitlines():
    sha, name = line.split('  ', 1)
    out[name] = (sha, sizes.get(name))
  out['SHA256SUMS'] = (None, sizes.get('SHA256SUMS'))
  return out


def keepable(record, name):
  """Whether a file of the previous version may be kept unchanged: an archive of this release (or its manifest) or the
  first build's check-inputs report."""
  names = {spec[1] for spec in ARCHIVES if spec[0] == record}
  return name in names or (name.endswith('.sha256') and name[:-7] in names) or (record == 'R1' and name in KEPT_PLAIN)


def finish_records(dirs, previous=None):
  """README.md (and for R1 the release tooling and lists) into each record folder, then SHA256SUMS of everything in
  it. Run again (command sums) after archives built elsewhere are copied in. With previous (a later version), the files
  of the previous version that are not here and may be kept (keepable()) are listed in SHA256SUMS with their recorded
  SHA-256, their rows of REDACTIONS.tsv are carried over, and <out>/zenodo_changes_<record>.csv says what to do with
  each file of the draft: keep, replace, add or delete. <out>/zenodo_files_<id>.json lists every file the record will
  hold, with its size: with the record folders' SHA256SUMS and REDACTIONS.tsv, the --previous of the next version."""
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
    kept, prev = {}, {}
    if previous:
      prev = load_previous(previous, d.name)
      here = {p.name for p in d.iterdir() if p.is_file()}
      kept = {n: v for n, v in prev.items() if n not in here and n != 'SHA256SUMS' and keepable(d.name, n)}
      unknown = [n for n, (_, size) in kept.items() if size is None]
      if unknown:
        raise Refused(f'{d.name}: no size in the previous Zenodo listing for {", ".join(sorted(unknown))}')
      kept_archives = {n for n in kept if not n.endswith('.sha256') and n not in KEPT_PLAIN}
      f = pathlib.Path(previous) / d.name / 'REDACTIONS.tsv'
      if f.is_file():
        rows = [r for r in csv.DictReader(open(f), delimiter='\t') if r['archive'] in kept_archives]
        if rows:
          write_redactions(d, rows)
    files = sorted(p for p in d.rglob('*') if p.is_file() and p.name != 'SHA256SUMS')
    nested = [p for p in files if p.parent != d]
    if nested:
      raise Refused(f'{d.name}: {nested[0].relative_to(d)} is in a folder, and a Zenodo record has none')
    rows = {str(p.relative_to(d)): (sha256_file(p), p.stat().st_size) for p in files}
    rows.update(kept)
    (d / 'SHA256SUMS').write_text(''.join(f'{sha}  {n}\n' for n, (sha, _) in sorted(rows.items())))
    n_files = len(rows) + 1
    total = sum(size for _, size in rows.values()) + (d / 'SHA256SUMS').stat().st_size
    print(f'   {d.name}/SHA256SUMS: {len(rows)} files ({len(kept)} kept from the previous version); record {n_files} '
          f'files, {total / 1e9:.2f} GB')
    # what the draft will hold once uploaded, in the form of Zenodo's listing: the --previous of the next version
    listing = [{'filename': n, 'filesize': size, 'checksum_sha256': sha} for n, (sha, size) in sorted(rows.items())]
    listing.append({'filename': 'SHA256SUMS', 'filesize': (d / 'SHA256SUMS').stat().st_size})
    (d.parent / f'zenodo_files_{ZENODO_IDS[d.name]}.json').write_text(json.dumps(listing, indent=1) + '\n')
    if previous:
      changes = []
      for n in sorted(set(rows) | set(prev) | {'SHA256SUMS'}):
        if n == 'SHA256SUMS':
          changes.append({'file': n, 'action': 'replace', 'bytes': (d / n).stat().st_size})
        elif n not in rows:
          changes.append({'file': n, 'action': 'delete', 'bytes': prev[n][1]})
        elif n not in prev:
          changes.append({'file': n, 'action': 'add', 'bytes': rows[n][1]})
        else:
          changes.append({'file': n, 'action': 'keep' if rows[n][0] == prev[n][0] else 'replace', 'bytes': rows[n][1]})
      write_csv(d.parent / f'zenodo_changes_{d.name}.csv', changes)
      count = {a: sum(1 for c in changes if c['action'] == a) for a in ('keep', 'replace', 'add', 'delete')}
      sent = sum(c['bytes'] for c in changes if c['action'] in ('replace', 'add'))
      print(f'   {d.name}: draft files to keep {count["keep"]}, replace {count["replace"]}, add {count["add"]}, '
            f'delete {count["delete"]}; to upload {sent / 1e9:.2f} GB ({d.parent}/zenodo_changes_{d.name}.csv)')
    if n_files > ZENODO_MAX_FILES or total > ZENODO_MAX_BYTES:
      raise Refused(f'{d.name}: {n_files} files, {total / 1e9:.1f} GB, over Zenodo\'s {ZENODO_MAX_FILES} files or '
                    f'{ZENODO_MAX_BYTES / 1e9:.0f} GB per record')
  return 0


def sums(args):
  previous = getattr(args, 'previous', None)
  return finish_or_refuse(sorted(d for d in pathlib.Path(args.out).iterdir() if d.is_dir() and d.name in RECORDS), previous)


def verify(args):
  """Every file of each SHA256SUMS and every member of each archive here. With --previous, a file listed but not here
  must be one the previous version holds with the same SHA-256 (kept, not rebuilt)."""
  previous = getattr(args, 'previous', None)
  bad = 0
  for sums_file in sorted(pathlib.Path(args.out).glob('*/SHA256SUMS')):
    d = sums_file.parent
    prev = load_previous(previous, d.name) if previous else {}
    kept = 0
    for line in sums_file.read_text().splitlines():
      sha, name = line.split('  ', 1)
      if not (d / name).is_file() and prev.get(name, (None,))[0] == sha:
        kept += 1
        continue
      if not (d / name).is_file() or sha256_file(d / name) != sha:
        print(f'!! {d.name}/{name}: SHA-256 differs or file missing')
        bad += 1
    if kept:
      print(f'   {d.name}: {kept} file(s) kept from the previous version, SHA-256 as it recorded')
    for man in sorted(d.glob('*.sha256')):
      archive = d / man.name[:-len('.sha256')]
      if not archive.is_file():
        if prev.get(archive.name, (None,))[0]:
          continue   # the archive is kept from the previous version
        print(f'!! {d.name}/{archive.name}: missing')
        bad += 1
        continue
      want = dict(reversed(l.split('  ', 1)) for l in man.read_text().splitlines())
      got = {}
      with tarfile.open(archive) as tar:
        for m in tar:
          got[m.name] = hashlib.sha256(tar.extractfile(m).read()).hexdigest()
      if got != want:
        print(f'!! {d.name}/{man.name}: {len(set(got.items()) ^ set(want.items()))} member(s) differ')
        bad += 1
      else:
        print(f'   {d.name}/{man.name[:-7]}: {len(got)} members match')
        if archive.name == CODE_ARCHIVE:
          bad += verify_code(archive, got, pathlib.Path(getattr(args, 'repo', None) or REPO))
  print('verify: ' + ('OK' if not bad else f'{bad} problem(s)'))
  return 1 if bad else 0


def verify_code(archive, got, repo):
  """The code copy against git: its files must be those of the commit CODE_COMMIT.txt names, less CODE_EXCLUDES, byte
  for byte. 0 if they are, or if that commit is not in the repository (then only the manifest is checked); 1 otherwise."""
  with tarfile.open(archive) as tar:
    commit = tar.extractfile(CODE_COMMIT_FILE).read().decode().split('\n', 1)[0].strip()
  if not re.fullmatch(r'[0-9a-f]{40}', commit):
    print(f'!! {archive.name}: {CODE_COMMIT_FILE} does not start with a full commit hash')
    return 1
  try:
    tree = code_tree(repo, git_commit(repo, commit))
  except Refused:
    print(f'   {archive.name}: commit {commit[:12]} is not in {repo}; checked against its manifest only')
    return 0
  want = {CODE_PREFIX + p: hashlib.sha256(data).hexdigest() for p, (data, _) in tree.items()}
  have = {m: s for m, s in got.items() if m != CODE_COMMIT_FILE}
  if have != want:
    print(f'!! {archive.name}: {len(set(have.items()) ^ set(want.items()))} file(s) differ from commit {commit[:12]}')
    return 1
  print(f'   {archive.name}: {len(have)} files, the same as commit {commit[:12]} (without {", ".join(CODE_EXCLUDES)})')
  return 0


def scan_member(item):
  """(archive, member, {class: occurrences}, refusal or None), with the checks of guard() and prepare() and no build."""
  archive, rel, src = item
  try:
    size, sha, findings, data = inspect(src, rel)
  except Refused as e:
    return archive, rel, {}, str(e)
  found = dict(findings)
  if data is not None and b'@' in data:
    emails = find_emails(data)
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
  """The archives built on one machine; with --batch, only those of that batch (an incremental version)."""
  batch = getattr(args, 'batch', None)
  print(','.join(spec[1] for spec in ARCHIVES if BUILT_ON.get(spec[1], 'vm') == args.built_on
                 and (not batch or BATCH_OF[spec[1]] == batch)))
  return 0


def vm_lists(args):
  """The VM's copy lists from the inventory's sessions.csv: sessions_release.txt (sessions whose outputs and records
  are released) and sessions_outside.txt (sessions outside the freeze: records only, copied without outputs). With
  --batch, only the sessions of that batch's labels; without it, every released session (a full rebuild)."""
  rows = list(csv.DictReader(open(pathlib.Path(args.inventory) / 'sessions.csv')))
  if args.batch:
    if args.batch not in BATCHES:
      print(f'!! unknown batch {args.batch}', file=sys.stderr)
      return 2
    inside = {lab['kind'] for lab in LATER_LABELS if lab['batch'] == args.batch}
    outside = {f'outside_{args.batch}'}
  else:
    outside = set(OUTSIDE_KINDS)
    inside = {r['kind'] for r in rows} - outside
  out = pathlib.Path(args.out)
  out.mkdir(parents=True, exist_ok=True)
  for name, kinds in (('sessions_release.txt', inside), ('sessions_outside.txt', outside)):
    sel = sorted(r['session'] for r in rows if r['kind'] in kinds)
    (out / name).write_text(''.join(f'{s}\n' for s in sel))
    print(f'{out / name}: {len(sel)} sessions')
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
  p.add_argument('--paper_ref', help=f'git ref of the paper commit whose {PAPER_ASSETS[1]}/ goes into {PAPER_ASSETS[0]} '
                 '(needed when that archive is built; "none" for a test build without them)')
  p.add_argument('--code_ref', help=f'git ref of the commit tagged {CODE_TAG}, copied into {CODE_ARCHIVE} (needed when '
                 'that archive is built; "none" for a test build without it)')
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
  p.add_argument('--batch', choices=sorted(set(BATCH_OF.values())), help='only the archives of this batch')
  p = sub.add_parser('vm-lists')
  p.add_argument('--inventory', default=str(OUT_DIR / 'inventory'))
  p.add_argument('--out', required=True)
  p.add_argument('--batch', help='a batch of BATCHES (incremental build); omit for every released session')
  p = sub.add_parser('merge-redactions')
  p.add_argument('--out', required=True)
  p.add_argument('--record', required=True, choices=sorted(RECORDS))
  p.add_argument('--from', required=True)
  p = sub.add_parser('sums')
  p.add_argument('--out', required=True)
  p.add_argument('--previous', help='the previous version (incremental build): keep its files that are not rebuilt')
  p = sub.add_parser('verify')
  p.add_argument('--out', required=True)
  p.add_argument('--previous', help='the previous version: files listed but not here must be its, unchanged')
  p.add_argument('--repo', default=str(REPO), help=f'the repository whose commit {CODE_ARCHIVE} must match')
  args = ap.parse_args()
  return {'inventory': inventory, 'layout': layout, 'build': build, 'check-inputs': check_inputs, 'scan': scan, 'sums': sums,
          'archives': archives, 'vm-lists': vm_lists, 'merge-redactions': merge_redactions, 'verify': verify}[args.cmd](args)


if __name__ == '__main__':
  sys.exit(main())
