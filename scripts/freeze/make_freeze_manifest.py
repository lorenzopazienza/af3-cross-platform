"""Data freeze manifest of the pilot2, main-study, memory-limit and follow-up sessions, and a comparison with the
results bucket.

    analysis/.venv/bin/python scripts/freeze/make_freeze_manifest.py --bucket gs://BUCKET [--out results/freeze]
        [--followups34] [--feat] [--followups612] [--followups1316] [--followups1721]
    analysis/.venv/bin/python scripts/freeze/make_freeze_manifest.py --local_only --out FOLDER [--followups612]
        [--followups1316] [--followups1721]

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
temporary file) and must be a byte prefix of the local copy; otherwise it stays an unexplained MISMATCH. With --feat
(added 2026-10-18, implies --followups34), also the 4 follow-up 5 sessions (label feat) of wrapper run 20261018T161637Z
(section 10, row "Follow-up 5, launch"), with the same prefix check. With --followups612 (added 2026-11-08, implies
--feat, so the freeze stays cumulative and one identifier covers everything; section 10, the launch, completion and
analysis rows of follow-ups 6 to 12, 2026-10-26 to 2026-11-08), also exactly the 135 sessions of
scripts/analysis_runs/{msa,sw,seeds,h100,repl,prec5,large}/sessions.txt (20, 9, 27, 6, 18, 6, 49), labelled by follow-up, with
the same prefix check; smoke, preempted, interrupted and relaunched-over sessions are in no list and are not frozen. A
session of those lists without a local folder, a session in two lists or in an earlier label, or a list of another
length stops the script before anything is hashed. The follow-up 6 NAS deliveries are not in the bucket and keep their
own freezes (results/freeze/msa_<host>_<run>/): the SHA-256 of each files_sha256.tsv is computed again, must equal the
identifier in NAS_FREEZES and in its summary.md, and is listed in the new summary.md; they are not refrozen.
With --followups1316 (added 2026-11-17, implies --followups612, so the freeze stays cumulative and one identifier covers
everything; section 10, the launch, relaunch, completion, correction and analysis rows of follow-ups 13 to 16, of the
memory limit on A and H and of the warm cost on GT and H, 2026-11-10 to 2026-11-17), also exactly the 144 sessions of
scripts/analysis_runs/{fu13,fu14,rcx,hlo,memlimit_ah,warm_th}/sessions.txt (79, 44, 15, 2, 2, 2), labelled by those
names, with the same checks and the same prefix check. Their smoke, preempted, interrupted, relaunched-over and
retried sessions are in no list and are not frozen: every local folder named like a session of those labels
(NAME_TAG_1316) that no list holds must be in NOT_FROZEN_1316 with the section 10 row that explains it, and is listed in
summary.md; one that is not stops the script before anything is hashed, as does an explained session that a list holds.
The local-only files of 20261113T001513Z_v5e_f32_v5f_06 (FINAL_UPLOAD_CUT: its final upload was cut off and the
launcher fetched every file from the VM; row of 2026-11-13, follow-up 14 complete) are labelled expected. Follow-up 16's
XLA dumps (<run>/<target>/xla_dump/ of the hlo sessions) are frozen like every other file; their count and bytes are
in summary.md. summary.md also lists the files of the sessions of freeze 68923012 (PREVIOUS_1316) added, changed or
removed locally since it was made (since_previous): added files are frozen with the rest (the full outputs that
scripts/analysis_runs/fu13/fetch_full_outputs.py copied from the bucket into two main-study sessions on 2026-11-12);
a changed or removed one is flagged there for the row.
With --followups1721 (added 2026-11-25, implies --followups1316, so the freeze stays cumulative and one identifier covers
everything; section 10, the launch, relaunch, launch-rule and completion rows of follow-ups 17 to 21, 2026-11-24), also
exactly the 168 sessions of scripts/analysis_runs/{dtp,tri,rcf,aut,xvm}/sessions.txt (85, 17, 21, 24, 21), labelled by
those names, with the same checks and the same prefix check. Their session folder names carry the label itself
(NAME_TAG_1721). Their smoke, preempted, lost, relaunched-over and timed-out sessions are in no list and are not frozen:
every local folder named like a session of those labels that no list holds must be in NOT_FROZEN_1721 with the section 10
row that explains it, and is listed in summary.md; one that is not stops the script before anything is hashed, as does an
explained session that a list holds. summary.md lists the files of the sessions of freeze 4ec185c5 (PREVIOUS_1721) added,
changed or removed locally since it was made, as above.
--local_only (added 2026-11-08, a dry run, not a freeze): no gcloud at all, no bucket, no CRC32C; writes sessions.tsv,
manifest_sha256.tsv (crc32c empty, so its SHA-256 is not a freeze identifier), session_check.tsv (session_problems: what looks
incomplete next to each session's final_status.json, plan_runs.jsonl, runs.jsonl and run.json files) and summary.md, into an
--out that must lie outside results/freeze.
Local side: every file of each results/af3/<session>/ folder, with its size, SHA-256 (hashlib) and CRC32C
(`gcloud storage hash --skip-md5`, computed on this machine). Bucket side: the object list of
<bucket>/af3/<session>/ with size and CRC32C (`gcloud storage objects list`, read only); also every other session
folder under <bucket>/af3/ named like a pilot2 or main session (listed, not compared file by file). Nothing is
written to the bucket.
Comparison per file (path relative to the session folder): same (both sides, equal CRC32C and size), mismatch,
local only, bucket only. Expected differences are labelled: bucket-only *_data.json and full *_confidences.json
(FETCH=light leaves them in the bucket), local-only files the launcher writes after the VM's last upload, and with
--followups1316 (and --followups1721) the local-only files of the FINAL_UPLOAD_CUT session.
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
FEAT_WRAPPERS = [('feat', '20261018T161637Z')]
FREEZE_DIR = REPO / 'results' / 'freeze'
ANALYSIS_RUNS = REPO / 'scripts' / 'analysis_runs'
# Follow-ups 6 to 12: label and number of sessions of scripts/analysis_runs/<label>/sessions.txt.
FOLLOWUP612 = {'msa': 20, 'sw': 9, 'seeds': 27, 'h100': 6, 'repl': 18, 'prec5': 6, 'large': 49}
# Follow-ups 13 to 16 and the two later measurements (--followups1316): label, number of sessions of
# scripts/analysis_runs/<label>/sessions.txt, and the word the session folder names carry.
FOLLOWUP1316 = {'fu13': 79, 'fu14': 44, 'rcx': 15, 'hlo': 2, 'memlimit_ah': 2, 'warm_th': 2}
NAME_TAG_1316 = {'fu13': 'why', 'fu14': 'f32', 'rcx': 'rcx', 'hlo': 'hlo', 'memlimit_ah': 'memlimit_ah', 'warm_th': 'warm_th'}
# Local session folders of those labels that no list holds, each with the section 10 row that explains it; not frozen.
# A local folder of those labels that is neither listed nor here stops the script before anything is hashed.
NOT_FROZEN_1316 = {
    '20261110T092001Z_l4_why_lg_01': 'fu13 LG, all 20 processes exit 1 (jax.debug.callback without a CPU device); '
                                     'rows of 2026-11-10 (LG stopped; fix and relaunch) and 2026-11-12 (complete, item 3)',
    '20261110T092603Z_v5e_why_l5_01': 'fu13 L5, setup exit 100 (apt lock) before any harness run; row of 2026-11-12 '
                                      '(complete, item 1)',
    '20261112T201157Z_h100_f32_hf_01': 'fu14 Hf, Spot preemption before any process finished; row of 2026-11-13 '
                                       '(complete, item 2)',
    '20261112T201203Z_h100_f32_hf_02': 'fu14 Hf, Spot preemption before any process finished; row of 2026-11-13 '
                                       '(complete, item 2)',
    '20261112T201217Z_h100_f32_hf_05': 'fu14 Hf, Spot preemption before any process finished; row of 2026-11-13 '
                                       '(complete, item 2)',
    '20261112T201228Z_a100_f32_af_01': 'fu14 Af, Spot preemption; row of 2026-11-13 (complete, item 2)',
    '20261112T201248Z_a100_f32_af_09': 'fu14 Af, Spot preemption; row of 2026-11-13 (complete, item 2)',
    '20261112T223637Z_a100_f32_af_15': 'fu14 Af, Spot preemption; row of 2026-11-13 (complete, item 2)',
    '20261113T001931Z_a100_f32_af_13': 'fu14 Af, Spot preemption; row of 2026-11-13 (complete, item 2)',
    '20261112T203710Z_a100_f32_af_13': 'fu14 Af, setup exit 100 (apt, us-east4-c) before any AlphaFold 3 run; rows of '
                                       '2026-11-12 (supervision) and 2026-11-13 (complete, item 3)',
    '20261112T204116Z_a100_f32_af_02': 'fu14 Af, setup exit 100 (apt, us-east4-c) before any AlphaFold 3 run; rows of '
                                       '2026-11-12 (relaunches without us-east4) and 2026-11-13 (complete, item 3)',
    '20261112T203854Z_v5e_f32_v5f_r2_01': 'fu14 V5f r2, setup exit 100 (apt lock) before any AlphaFold 3 run; rows of '
                                          '2026-11-12 (relaunches without us-east4) and 2026-11-13 (complete, item 3)',
    '20261114T191329Z_a100_rcx_af_r2_03': 'rcx Af r2, Spot preemption during setup, before any AlphaFold 3 run; row of '
                                          '2026-11-15 (complete, item 2)',
    '20261114T191334Z_a100_rcx_af_r2_01': 'rcx Af r2, Spot preemption during setup, before any AlphaFold 3 run; row of '
                                          '2026-11-15 (complete, item 2)',
    '20261114T221407Z_l4_rcx_gt_r2_08': 'rcx GT r2, Spot preemption after AlphaFold 3 runs had started (its finished '
                                        'processes not used); row of 2026-11-15 (complete, item 4)',
    '20261114T221412Z_l4_rcx_gt_r2_03': 'rcx GT r2, Spot preemption after AlphaFold 3 runs had started (its finished '
                                        'processes not used); row of 2026-11-15 (complete, item 4)',
    '20261114T221417Z_l4_rcx_gt_r2_04': 'rcx GT r2, Spot preemption after AlphaFold 3 runs had started (its finished '
                                        'processes not used); row of 2026-11-15 (complete, item 4)',
    '20261114T221422Z_l4_rcx_gt_r2_05': 'rcx GT r2, Spot preemption after AlphaFold 3 runs had started (its finished '
                                        'processes not used); row of 2026-11-15 (complete, item 4)',
    '20261114T221426Z_l4_rcx_gt_r2_06': 'rcx GT r2, Spot preemption after AlphaFold 3 runs had started (its finished '
                                        'processes not used); row of 2026-11-15 (complete, item 4)',
    '20261115T010417Z_l4_rcx_gt_r2_08': 'rcx GT r2, Spot preemption 3 min after the job started, before any AlphaFold 3 '
                                        'run; row of 2026-11-15 (complete, item 4)',
}
# A listed session whose final upload to the bucket was cut off and whose launcher fetched every file from the VM: its
# local-only files are expected (row of 2026-11-13, follow-up 14 complete).
FINAL_UPLOAD_CUT = {'20261113T001513Z_v5e_f32_v5f_06': 'fu14'}
# The freeze that --followups1316 extends: its files are compared with the local ones of the same sessions (summary.md).
PREVIOUS_1316 = FREEZE_DIR / 'followups612_20261108' / 'manifest_sha256.tsv'
# Follow-ups 17 to 21 (--followups1721): label and number of sessions of scripts/analysis_runs/<label>/sessions.txt, in
# the order of the follow-ups; the session folder names carry the label itself.
FOLLOWUP1721 = {'dtp': 85, 'tri': 17, 'rcf': 21, 'aut': 24, 'xvm': 21}
NAME_TAG_1721 = {'dtp': 'dtp', 'tri': 'tri', 'rcf': 'rcf', 'aut': 'aut', 'xvm': 'xvm'}
# Local session folders of those labels that no list holds, each with the section 10 row that explains it; not frozen.
# A local folder of those labels that is neither listed nor here stops the script before anything is hashed. The launch
# attempts that created no VM have no session folder (results/af3/.launcher/ only).
_FU17_GFD = ('dtp Gfd, Spot VM lost during the run, the part relaunched on demand in the second round (results/'
             'dtp_fallback.tsv); its finished processes not used; row of 2026-11-24 (follow-up 17 complete, item 2)')
_FU18_HT = ('tri HT, Spot VM lost during the run (VM NOT_FOUND), the part relaunched unchanged on Spot; its finished '
            'processes not used; row of 2026-11-24 (follow-up 18 complete, item 1)')
_FU21_G = ('xvm G r3, Spot VM lost at 10:29Z to 10:30Z, the part relaunched on demand in the second round; its finished '
           'processes not used; row of 2026-11-24 (follow-up 21 complete, item 1)')
NOT_FROZEN_1721 = {
    **{s: _FU17_GFD for s in (
        '20261123T234814Z_l4_dtp_gfd_01', '20261123T234819Z_l4_dtp_gfd_02', '20261123T234824Z_l4_dtp_gfd_03',
        '20261123T234829Z_l4_dtp_gfd_04', '20261123T234833Z_l4_dtp_gfd_05', '20261123T234844Z_l4_dtp_gfd_07',
        '20261123T234849Z_l4_dtp_gfd_08', '20261123T234854Z_l4_dtp_gfd_09', '20261123T234904Z_l4_dtp_gfd_11',
        '20261123T234908Z_l4_dtp_gfd_12', '20261123T234918Z_l4_dtp_gfd_14', '20261123T234929Z_l4_dtp_gfd_16',
        '20261123T234939Z_l4_dtp_gfd_18', '20261123T234944Z_l4_dtp_gfd_19', '20261123T234949Z_l4_dtp_gfd_20',
        '20261124T000216Z_l4_dtp_gfd_27', '20261124T000222Z_l4_dtp_gfd_28', '20261124T000423Z_l4_dtp_gfd_15',
        '20261124T001347Z_l4_dtp_gfd_31', '20261124T001754Z_l4_dtp_gfd_26', '20261124T003153Z_l4_dtp_gfd_34',
        '20261124T003158Z_l4_dtp_gfd_35', '20261124T004047Z_l4_dtp_gfd_38', '20261124T004447Z_l4_dtp_gfd_39',
        '20261124T004452Z_l4_dtp_gfd_40', '20261124T005349Z_l4_dtp_gfd_42', '20261124T005418Z_l4_dtp_gfd_43',
        '20261124T005846Z_l4_dtp_gfd_44', '20261124T014015Z_l4_dtp_gfd_60', '20261124T015227Z_l4_dtp_gfd_58',
        '20261124T015318Z_l4_dtp_gfd_59')},
    '20261124T023724Z_l4_dtp_gfd_13': ('dtp Gfd, relaunched unchanged on Spot in the second round and preempted again, '
                                       'then run on demand in the third round; its finished processes not used; row of '
                                       '2026-11-24 (follow-up 17 complete, item 2)'),
    '20261124T101000Z_a100_dtp_afd_11': ('dtp Afd, Spot VM lost in us-central1, the part moved on demand; row of '
                                         '2026-11-24 (follow-up 17 complete, item 3)'),
    '20261124T102117Z_a100_dtp_afd_10': ('dtp Afd, Spot VM lost in us-central1, the part moved on demand; its finished '
                                         'process not used; row of 2026-11-24 (follow-up 17 complete, item 3)'),
    '20261123T234813Z_h100_tri_smoke_ht': ('tri smoke test (HT, 7U3J seed 1, label smoke), passed, not a part; row of '
                                           '2026-11-24 (follow-up 18 complete, item 1)'),
    **{s: _FU18_HT for s in (
        '20261124T002206Z_h100_tri_ht_01', '20261124T002211Z_h100_tri_ht_02', '20261124T002216Z_h100_tri_ht_03',
        '20261124T002221Z_h100_tri_ht_r2_01', '20261124T003335Z_h100_tri_ht_r2_04')},
    '20261124T113837Z_h100_rcf_hf_r2b_04': ('rcf Hf r2b, reached the deadline because its apt setup took 68 min, after '
                                            'seeds 1 to 4; relaunched unchanged on Spot; its finished processes not '
                                            'used; rows of 2026-11-24 (Hf r2b relaunch; follow-up 19 complete, item 1)'),
    '20261124T113903Z_h100_rcf_hf_r2b_09': ('rcf Hf r2b, Spot preemption at about 12:10Z; relaunched unchanged on Spot; '
                                            'its finished processes not used; rows of 2026-11-24 (Hf r2b relaunch; '
                                            'follow-up 19 complete, item 1)'),
    '20261124T144617Z_h100_rcf_hf_r2b_09': ('rcf Hf r2b, Spot preemption again at about 14:58Z; completed in the '
                                            "wrapper's second round on Spot; its finished processes not used; row of "
                                            '2026-11-24 (follow-up 19 complete, item 1)'),
    **{s: _FU21_G for s in (
        '20261124T095048Z_l4_xvm_g_r3_01', '20261124T095053Z_l4_xvm_g_r3_08', '20261124T095058Z_l4_xvm_g_r3_07',
        '20261124T095102Z_l4_xvm_g_r3_09', '20261124T095107Z_l4_xvm_g_r3_10', '20261124T095113Z_l4_xvm_g_r3_11',
        '20261124T095118Z_l4_xvm_g_r3_12', '20261124T095123Z_l4_xvm_g_r3_02', '20261124T095129Z_l4_xvm_g_r3_03',
        '20261124T095132Z_l4_xvm_g_r3_04', '20261124T095138Z_l4_xvm_g_r3_05')},
}
# The freeze that --followups1721 extends: its files are compared with the local ones of the same sessions (summary.md).
PREVIOUS_1721 = FREEZE_DIR / 'followups1316_20261117' / 'manifest_sha256.tsv'
PREFIX_CHECKED = (('cpu60', 'a100', 'feat') + tuple(FOLLOWUP612) + tuple(FOLLOWUP1316)
                  + tuple(FOLLOWUP1721))   # appended logs: prefix check
# Follow-up 6 data-pipeline deliveries: not in the bucket; their own freezes (scripts/nas/ingest_official.py).
NAS_FREEZES = {'msa_naspaz8-af3_run1': '3c21b4afbdfbcf310e7e289ea7ff326c2e1579ed18cea425a1616dd8f1c7b9da',
               'msa_naspaz6-af3_run1': '781f4ee3221df5f2e8f09b9dec91ad919a87c80480eba838eedb946cf0c131b2',
               'msa_naspaz8-af3_run2': 'ed7f634e3e0832462ab52ea918f0d47954fee7a8ca8fda98c18add3c9ff3d0c4'}
FETCH_LIGHT = re.compile(r'(_data\.json|(?<!summary)_confidences\.json)$')
# Session-level files the VM job or the launcher writes or appends to after the VM's final upload
# (cloud/vm_af3_job.sh upload_out, cloud/vm_job_finish.sh, cloud/af3_run_many.sh): a later local copy.
AFTER_UPLOAD_LOCAL = re.compile(r'^(\.uploaded|\.wrapper_done|finish\.log|upload_final\.log)$')
AFTER_UPLOAD_APPENDED = re.compile(r'^(job\.log|upload_final\.log)$')


def session_list(label):
  """The sessions of scripts/analysis_runs/<label>/sessions.txt, in file order; comments and blank lines left out."""
  path = ANALYSIS_RUNS / label / 'sessions.txt'
  return [line.split('#')[0].strip() for line in path.read_text().splitlines() if line.split('#')[0].strip()]


def add_lists(out, labels):
  """Appends the sessions of scripts/analysis_runs/<label>/sessions.txt for each label, in order; stops on a list of
  another length, a session listed twice or a session already in the freeze."""
  earlier = {s for s, _ in out}
  for label, want in labels.items():
    listed = session_list(label)
    if len(listed) != want:
      raise SystemExit(f'scripts/analysis_runs/{label}/sessions.txt lists {len(listed)} sessions, not {want}')
    again = sorted(s for s in listed if s in earlier) + sorted({s for s in listed if listed.count(s) > 1})
    if again:
      raise SystemExit(f'{label}: sessions already in the freeze or listed twice: {", ".join(again)}')
    earlier |= set(listed)
    out += [(s, label) for s in listed]


def sessions(followups34=False, feat=False, followups612=False, followups1316=False, followups1721=False):
  followups1316 = followups1316 or followups1721
  followups612 = followups612 or followups1316
  feat = feat or followups612
  out = [(s, 'pilot2') for s in (PILOT2 / 'sessions_pilot2.txt').read_text().split()]
  for line in (PILOT2 / 'sessions_excluded.txt').read_text().splitlines():
    s, _, why = line.partition('\t')
    if s.strip():
      out.append((s.strip(), f'pilot2, excluded: {why.strip()}'))
  out += [(s, 'main') for s in (MAIN / 'sessions_main.txt').read_text().split()]
  out += [(m.group(1), 'memlimit') for m in re.finditer(r'ok: launcher exit 0, session (\S+)', MEMLIMIT_WRAPPER.read_text())]
  for m in re.finditer(r'ok: launcher exit 0, session (\S+)', FOLLOWUP_WRAPPER.read_text()):
    out.append((m.group(1), 'embed' if '_embed_' in m.group(1) else 'ppi'))
  if followups34 or feat:
    for kind, w in FOLLOWUP34_WRAPPERS + (FEAT_WRAPPERS if feat else []):
      out += [(s, kind) for s in re.findall(r'ok: launcher exit 0, session (\S+)',
                                            (AF3 / '.many' / w / 'summary.txt').read_text())]
  if followups612:
    add_lists(out, FOLLOWUP612)
  if followups1316:
    add_lists(out, FOLLOWUP1316)
  if followups1721:
    add_lists(out, FOLLOWUP1721)
  return out


def check_local_folders(selection):
  """Stops if a session of follow-ups 6 to 21 (lists of --followups612, --followups1316 and --followups1721) has no
  local folder. Earlier
  labels keep their record: sessions without a folder are listed in summary.md (the excluded pilot2 attempts of freeze
  14e1fc4c)."""
  missing = [f'{s} ({k})' for s, k in selection
             if (k in FOLLOWUP612 or k in FOLLOWUP1316 or k in FOLLOWUP1721) and not (AF3 / s).is_dir()]
  if missing:
    raise SystemExit(f'{len(missing)} session(s) of the follow-up lists have no folder under results/af3/, nothing '
                     'was frozen: ' + ', '.join(missing))


def label_of(name, tags):
  """The label whose word (tags: label -> word) the session folder name carries, or None."""
  for label, tag in tags.items():
    if re.search(rf'^\d{{8}}T\d{{6}}Z_[a-z0-9]+_{tag}_', name):
      return label
  return None


def label_of_1316(name):
  """The --followups1316 label whose word the session folder name carries, or None."""
  return label_of(name, NAME_TAG_1316)


def label_of_1721(name):
  """The --followups1721 label whose word the session folder name carries, or None."""
  return label_of(name, NAME_TAG_1721)


def not_frozen(selection, tags, explained, span, table):
  """[(session, label, reason)] of the local folders of the labels in tags that no list holds, each explained by
  explained. Stops, before anything is hashed, on a folder that no list holds and no row explains, or on an explained
  session that a list holds."""
  listed = {s for s, _ in selection}
  both = sorted(s for s in explained if s in listed)
  if both:
    raise SystemExit('sessions both listed and recorded as not frozen: ' + ', '.join(both))
  out, odd = [], []
  for d in sorted(p for p in AF3.iterdir() if p.is_dir()):
    label = label_of(d.name, tags)
    if label is None or d.name in listed:
      continue
    if d.name in explained:
      out.append((d.name, label, explained[d.name]))
    else:
      odd.append(f'{d.name} ({label})')
  if odd:
    raise SystemExit(f'{len(odd)} local session folder(s) of the follow-up {span} labels are in no list and no section '
                     f'10 row explains them ({table}), nothing was frozen: ' + ', '.join(odd))
  return out


def not_frozen_1316(selection):
  """not_frozen for the --followups1316 labels (NAME_TAG_1316, NOT_FROZEN_1316)."""
  return not_frozen(selection, NAME_TAG_1316, NOT_FROZEN_1316, '13 to 16', 'NOT_FROZEN_1316')


def not_frozen_1721(selection):
  """not_frozen for the --followups1721 labels (NAME_TAG_1721, NOT_FROZEN_1721)."""
  return not_frozen(selection, NAME_TAG_1721, NOT_FROZEN_1721, '17 to 21', 'NOT_FROZEN_1721')


def hlo_dumps(selection):
  """[(session, files, bytes)] of the xla_dump/ folders of each hlo session (follow-up 16's HLO dumps)."""
  out = []
  for s, k in selection:
    if k != 'hlo':
      continue
    files = [p for p in (AF3 / s).glob('*/*/xla_dump/**/*') if p.is_file()]
    out.append((s, len(files), sum(p.stat().st_size for p in files)))
  return out


def nas_freezes():
  """The follow-up 6 NAS freezes: (folder, identifier, files and bytes line, date), each identifier computed again."""
  rows = []
  for name, want in NAS_FREEZES.items():
    d = FREEZE_DIR / name
    if not (d / 'files_sha256.tsv').is_file() or not (d / 'summary.md').is_file():
      raise SystemExit(f'results/freeze/{name}: files_sha256.tsv or summary.md is missing')
    got, text = sha256(d / 'files_sha256.tsv'), (d / 'summary.md').read_text()
    if got != want or got not in text:
      raise SystemExit(f'results/freeze/{name}: SHA-256 of files_sha256.tsv is {got}, not the identifier {want} of its '
                       'freeze and its summary.md')
    files = re.search(r'^- files: (.+)$', text, re.M)
    made = re.search(r'^Made (\S+)', text, re.M)
    rows.append((name, got, files.group(1) if files else '?', made.group(1) if made else '?'))
  return rows


def session_problems(folder):
  """What looks incomplete in one session folder, next to its own records (local only; a dry-run check)."""
  out = []
  status = folder / 'final_status.json'
  if not status.is_file():
    out.append('no final_status.json')
  else:
    st = json.loads(status.read_text())
    if st.get('exit_code') != 0:
      out.append(f'final_status.json: exit_code {st.get("exit_code")}')
    if st.get('final_upload_complete') is not True:
      out.append('final_status.json: final upload not complete')
  plan = folder / 'plan_runs.jsonl'
  runs = [json.loads(l) for l in plan.read_text().splitlines() if l.strip()] if plan.is_file() else []
  if not runs:
    out.append('no plan_runs.jsonl lines')
  for r in runs:
    run, seed = folder / r['run'], r.get('seed')
    if r.get('harness_exit_code') != 0:
      out.append(f'{r["run"]}: harness exit {r.get("harness_exit_code")}')
    if not (run / 'runs.jsonl').is_file():
      out.append(f'{r["run"]}: no runs.jsonl')
      continue
    for line in (run / 'runs.jsonl').read_text().splitlines():
      x = json.loads(line)
      t = x['pdb_id']
      if x.get('exit_code') != 0:
        out.append(f'{r["run"]}/{t}: exit {x.get("exit_code")}')
      if not (run / t / 'run.json').is_file():
        out.append(f'{r["run"]}/{t}: no run.json')
      top = run / t / 'af3_output' / t
      for name in (f'{t}_model.cif', f'{t}_ranking_scores.csv', f'{t}_summary_confidences.json'):
        if not (top / name).is_file():
          out.append(f'{r["run"]}/{t}: no {name}')
      for k in range(5):
        base = f'seed-{seed}_sample-{k}'
        for name in (f'{t}_{base}_model.cif', f'{t}_{base}_summary_confidences.json'):
          if not (top / base / name).is_file():
            out.append(f'{r["run"]}/{t}: no {base}/{name}')
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
  ap.add_argument('--bucket', help='the results bucket, gs://NAME (required unless --local_only)')
  ap.add_argument('--out', default=str(REPO / 'results' / 'freeze'))
  ap.add_argument('--followups34', action='store_true', help='also freeze the follow-up 3 and 4 sessions')
  ap.add_argument('--feat', action='store_true', help='also freeze the follow-up 3, 4 and 5 sessions')
  ap.add_argument('--followups612', action='store_true', help='also freeze the follow-up 3 to 12 sessions (implies --feat)')
  ap.add_argument('--followups1316', action='store_true',
                  help='also freeze the follow-up 13 to 16, memlimit_ah and warm_th sessions (implies --followups612)')
  ap.add_argument('--followups1721', action='store_true',
                  help='also freeze the follow-up 17 to 21 sessions (implies --followups1316)')
  ap.add_argument('--local_only', action='store_true', help='dry run: no bucket and no gcloud; not a freeze')
  args = ap.parse_args()
  if bool(args.bucket) == args.local_only:
    ap.error('give --bucket, or --local_only without it')
  args.followups1316 = args.followups1316 or args.followups1721
  args.followups612 = args.followups612 or args.followups1316
  args.feat = args.feat or args.followups612
  selection = sessions(args.followups34, args.feat, args.followups612, args.followups1316, args.followups1721)
  check_local_folders(selection)
  skipped = not_frozen_1316(selection) if args.followups1316 else []
  skipped1721 = not_frozen_1721(selection) if args.followups1721 else []
  dumps = hlo_dumps(selection) if args.followups1316 else []
  nas = nas_freezes() if args.followups612 else []
  flag = '--followups1721' if args.followups1721 else '--followups1316' if args.followups1316 else '--followups612'
  previous = PREVIOUS_1721 if args.followups1721 else None
  out = pathlib.Path(args.out)
  if args.local_only:
    if out.resolve().is_relative_to(FREEZE_DIR.resolve()):
      raise SystemExit('--local_only does not write under results/freeze: its output is not a freeze')
    return local_only(selection, nas, out, skipped, dumps, flag, skipped1721, previous)
  out.mkdir(parents=True, exist_ok=True)
  started = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds')
  rows_s, rows_m, rows_b, rows_c = [], [], [], []
  totals = {}
  for session, kind in selection:
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
          if kind not in PREFIX_CHECKED:
            note = 'expected: appended after the final upload (local copy longer); ' + note
          elif bucket_copy_is_prefix(args.bucket, session, rel, folder / rel):
            note = ('expected: appended after the final upload (local copy longer, the bucket copy a byte prefix of '
                    'it); ' + note)
          else:
            note = 'local copy longer, but the bucket copy is not a prefix of it; ' + note
      elif l:
        status = 'local only'
        note = 'expected: written after the final upload' if AFTER_UPLOAD_LOCAL.match(rel) else ''
        if not note and FINAL_UPLOAD_CUT.get(session) == kind:
          note = 'expected: the final upload was cut off, every file fetched from the VM (row of 2026-11-13)'
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
  named = (r'_(pilot|main|memlimit|embed|ppi|cpu60|a100|feat|msa|sw|seeds|h100|repl|prec5|large|why|f32|rcx|hlo|warm_th)_?'
           r'|_(dtp|tri|rcf|aut|xvm)_' if args.followups1721 else
           r'_(pilot|main|memlimit|embed|ppi|cpu60|a100|feat|msa|sw|seeds|h100|repl|prec5|large|why|f32|rcx|hlo|warm_th)_?'
           if args.followups1316 else
           r'_(pilot|main|memlimit|embed|ppi|cpu60|a100|feat|msa|sw|seeds|h100|repl|prec5|large)_?' if args.followups612 else
           r'_(pilot|main|memlimit|embed|ppi|cpu60|a100|feat)_?' if args.feat else
           r'_(pilot|main|memlimit|embed|ppi|cpu60|a100)_?' if args.followups34 else r'_(pilot|main|memlimit|embed|ppi)_?')
  others = [s for s in bucket_sessions(args.bucket) if s not in ours and re.search(named, s) and s >= '20261010T2']
  manifest_sha = sha256(out / 'manifest_sha256.tsv')
  lines = ['# Data freeze: pilot2, main-study, memory-limit and follow-up sessions', '',
           f'Made {started} by scripts/freeze/make_freeze_manifest.py (read-only bucket listing; nothing written to '
           'the bucket). Bucket CRC32C from `gcloud storage objects list`, local CRC32C from `gcloud storage hash`.', '',
           f'Freeze identifier: SHA-256 of manifest_sha256.tsv = `{manifest_sha}`', '',
           '| sessions | local files | local bytes | bucket objects | bucket bytes |', '|---|---|---|---|---|']
  for kind in ('pilot2', 'pilot2, excluded', 'main', 'memlimit', 'embed', 'ppi') + (
      ('cpu60', 'a100') if args.followups34 or args.feat else ()) + (('feat',) if args.feat else ()) + (
      tuple(FOLLOWUP612) if args.followups612 else ()) + (tuple(FOLLOWUP1316) if args.followups1316 else ()) + (
      tuple(FOLLOWUP1721) if args.followups1721 else ()):
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
  lines += report_1316(skipped, dumps)
  lines += report_1721(skipped1721) if args.followups1721 else []
  if args.followups1316:
    lines += since_previous(rows_m, previous)
  if nas:
    lines += ['', '## Follow-up 6 NAS deliveries: not in the bucket, not refrozen; each its own freeze', '',
              f'(Flag of this freeze: {flag}. SHA-256 of each files_sha256.tsv computed again for this summary.)', '',
              '| freeze folder | identifier | files | made |', '|---|---|---|---|']
    lines += [f'| results/freeze/{n} | `{i}` | {f} | {m} |' for n, i, f, m in nas]
  (out / 'summary.md').write_text('\n'.join(lines) + '\n')
  print('\n'.join(lines))
  return 0


def since_previous(rows_m, previous=None):
  """summary.md lines: the files of the sessions of the freeze --followups1316 extends that were added, changed (bytes or
  SHA-256) or removed locally since it was made, by session and kind of file. Added files are frozen with the rest."""
  previous = previous or PREVIOUS_1316
  old = {}
  with open(previous) as f:
    next(f)
    for line in f:
      s, rel, size, sha, _ = line.rstrip('\n').split('\t')
      old[(s, rel)] = (int(size), sha)
  sessions = {s for s, _ in old}
  new = {(s, rel): (int(size), sha) for s, rel, size, sha, _ in rows_m if s in sessions}
  added = sorted(k for k in new if k not in old)
  changed = sorted(k for k in new if k in old and new[k] != old[k])
  gone = sorted(k for k in old if k not in new)
  name = previous.parent.name
  lines = ['', f'## Sessions of results/freeze/{name}: files added, changed or removed locally since it was made', '',
           f'added {len(added)} ({sum(new[k][0] for k in added):,} bytes), changed {len(changed)}, removed {len(gone)}'
           + ('' if not (changed or gone) else ': CHANGED OR REMOVED FROZEN FILES, explain each in the row'), '']
  groups = {}
  for what, keys in (('added', added), ('changed', changed), ('removed', gone)):
    for s, rel in keys:
      kind = FETCH_LIGHT.search(rel)
      groups.setdefault((what, s, kind.group(1) if kind else rel.rsplit('/', 1)[-1]), []).append(rel)
  if groups:
    lines += ['| change | session | file | files |', '|---|---|---|---|']
    lines += [f'| {w} | {s} | {k} | {len(v)} |' for (w, s, k), v in sorted(groups.items())]
  return lines


def report_1316(skipped, dumps):
  """summary.md lines of --followups1316: the local folders not frozen, with their rows, and follow-up 16's HLO dumps."""
  if not skipped and not dumps:
    return []
  lines = ['', f'## Follow-ups 13 to 16: local session folders in no list, not frozen: {len(skipped)}', '',
           '| session | label | why not frozen (section 10) |', '|---|---|---|']
  lines += [f'| {s} | {k} | {why} |' for s, k, why in skipped]
  lines += ['', '## Follow-up 16 (hlo): XLA dumps, frozen like every other file', '',
            '| session | dump files | dump bytes |', '|---|---|---|']
  lines += [f'| {s} | {n} | {b:,} |' for s, n, b in dumps]
  lines.append(f'| all | {sum(d[1] for d in dumps)} | {sum(d[2] for d in dumps):,} |')
  return lines


def report_1721(skipped):
  """summary.md lines of --followups1721: the local folders of its labels not frozen, with their rows."""
  lines = ['', f'## Follow-ups 17 to 21: local session folders in no list, not frozen: {len(skipped)}', '',
           '| session | label | why not frozen (section 10) |', '|---|---|---|']
  return lines + [f'| {s} | {k} | {why} |' for s, k, why in skipped]


def local_only(selection, nas, out, skipped=(), dumps=(), flag='--followups612', skipped1721=(), previous=None):
  """The dry run: local files only, no gcloud, no CRC32C, no comparison. Not a freeze."""
  out.mkdir(parents=True, exist_ok=True)
  started = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds')
  rows_s, rows_m, rows_p, names = [], [], [], {}
  for session, kind in selection:
    folder = AF3 / session
    local = sorted(p for p in folder.rglob('*') if p.is_file()) if folder.is_dir() else []
    size = 0
    for p in local:
      rel, n = str(p.relative_to(folder)), p.stat().st_size
      size += n
      rows_m.append((session, rel, n, sha256(p), ''))
      if '/' not in rel:
        names.setdefault(kind, {}).setdefault(rel, set()).add(session)
    rows_s.append((session, kind, folder.is_dir(), len(local), size))
    if kind in FOLLOWUP612 or kind in FOLLOWUP1316 or kind in FOLLOWUP1721:
      rows_p += [(session, kind, x) for x in session_problems(folder)]
    print(f'{session:44} {kind[:16]:16} local {len(local):6} files', flush=True)

  def tsv(name, header, rows):
    with open(out / name, 'w') as f:
      f.write('\t'.join(header) + '\n')
      for r in rows:
        f.write('\t'.join(str(x) for x in r) + '\n')
  tsv('sessions.tsv', ['session', 'kind', 'local_folder', 'local_files', 'local_bytes'], rows_s)
  tsv('manifest_sha256.tsv', ['session', 'path', 'bytes', 'sha256', 'crc32c'], rows_m)
  tsv('session_check.tsv', ['session', 'kind', 'problem'], rows_p)
  kinds = list(dict.fromkeys('pilot2, excluded' if k.startswith('pilot2, excluded') else k for _, k in selection))
  lines = ['# Local dry run, NOT a freeze (no bucket, no CRC32C, no comparison)', '', f'Made {started}.', '',
           '| label | sessions | local files | local bytes |', '|---|---|---|---|']
  for kind in kinds:
    sel = [r for r in rows_s if (r[1].startswith('pilot2, excluded') if kind == 'pilot2, excluded' else r[1] == kind)]
    lines.append(f'| {kind} | {len(sel)} | {sum(r[3] for r in sel)} | {sum(r[4] for r in sel):,} |')
  new = [r for r in rows_s if r[1] in FOLLOWUP612]
  lines.append(f'| follow-ups 6 to 12 | {len(new)} | {sum(r[3] for r in new)} | {sum(r[4] for r in new):,} |')
  new = [r for r in rows_s if r[1] in FOLLOWUP1316]
  if new:
    lines.append(f'| follow-ups 13 to 16 | {len(new)} | {sum(r[3] for r in new)} | {sum(r[4] for r in new):,} |')
  new = [r for r in rows_s if r[1] in FOLLOWUP1721]
  if new:
    lines.append(f'| follow-ups 17 to 21 | {len(new)} | {sum(r[3] for r in new)} | {sum(r[4] for r in new):,} |')
  lines.append(f'| all | {len(rows_s)} | {sum(r[3] for r in rows_s)} | {sum(r[4] for r in rows_s):,} |')
  last = {'--followups1721': 21, '--followups1316': 16}.get(flag, 12)
  lines += ['', f'## Follow-ups 6 to {last}: what looks incomplete next to the session '
            f'records: {len(rows_p)} (session_check.tsv)', '']
  for session in dict.fromkeys(r[0] for r in rows_p):
    sel = [r[2] for r in rows_p if r[0] == session]
    lines.append(f'- {session}: {len(sel)}: ' + '; '.join(sel[:3]) + (' ...' if len(sel) > 3 else ''))
  lines += ['', f'## Follow-ups 6 to {last}: top-level files not in every session of the label', '']
  for kind in tuple(FOLLOWUP612) + (tuple(FOLLOWUP1316) if last >= 16 else ()) + (tuple(FOLLOWUP1721) if last >= 21 else ()):
    n = sum(1 for r in rows_s if r[1] == kind)
    odd = {f: v for f, v in names.get(kind, {}).items() if len(v) != n}
    if odd:
      lines.append(f'- {kind}: ' + '; '.join(f'{f} in {len(v)} of {n}' for f, v in sorted(odd.items())))
  if nas:
    lines += ['', '## Follow-up 6 NAS deliveries (identifiers computed again)', '']
    lines += [f'- results/freeze/{n}: `{i}`, {f}, made {m}' for n, i, f, m in nas]
  lines += report_1316(skipped, dumps)
  lines += report_1721(skipped1721) if last >= 21 else []
  if last >= 16:
    lines += since_previous(rows_m, previous)
  (out / 'summary.md').write_text('\n'.join(lines) + '\n')
  print('\n'.join(lines))
  return 0


if __name__ == '__main__':
  sys.exit(main())
