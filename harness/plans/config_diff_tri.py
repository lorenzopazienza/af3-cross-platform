"""Follow-up 18 (label tri): every setting of each new arm against the sessions of its source arm, as
harness/plans/config_diff_rcx.py does for follow-up 15 (config_diff_fu7_10's functions, imported).

    python3 harness/plans/config_diff_tri.py [--out harness/plans/config_diff_tri.md]

Arms: HT r1 and HT r2 (h100_triton) against H r1 and r2 of follow-up 9 (h100_xla; scripts/analysis_runs/h100/
sessions.txt); AT r2 (a100_triton) against AT r1 of follow-up 3 (scripts/analysis_runs/a100/sessions.txt).
Source side, as recorded in the source sessions: launch_record.txt, every fresh harness run's session.json and
command, the image each session was created from as its wrapper log printed it, and the Eigen source of the build.
New side, as the arm will run: its plans (harness/plans/tri), harness/configs.yaml, harness/run_af3.py, cloud/
af3_run.sh's defaults, af3_tpu/, Spot (make_prep_plans.LABELS['tri']['provisioning']), the A100 machine type
a2-ultragpu-1g and the image the launch sets (IMAGE below).
Allowed differences (row "Follow-up experiment 18", items a and f): the label; the repetitions; the provisioning;
the image family (project copies with the same NVIDIA driver; check 4 of analysis/tri.py compares the stack session
by session); the Eigen source (the archive of the same commit, as every plan since follow-up 7); for HT only, the
attention: the configuration name and description, the entry's flash_attention_implementation (triton for xla) and
the same word in the run_alphafold command. The entry and the command of HT are compared once more with the attention
set back to xla: they must then be the same as H's. Any other difference is DIFFERS and the script exits 1.
"""
import argparse
import json
import pathlib
import re
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'harness' / 'plans'))
import config_diff_fu7_10 as cd  # noqa: E402
import config_diff_rcx as cr  # noqa: E402
import make_prep_plans as mk  # noqa: E402

IMG, EIGEN = cr.IMG, cr.EIGEN
IMAGE = {'h100': 'af2-tpu-benchmark/af3-accel-2204-nvidia-580',
         'a100': 'af2-tpu-benchmark/af3-accel-2204-nvidia-580-atr1 (else af2-tpu-benchmark/af3-accel-2204-nvidia-580)'}
ENTRY = 'config (harness/configs.yaml entry, without its description)'
CMD = 'run_alphafold command'
ALLOWED = {'label', cd.REPS, 'provisioning', IMG, EIGEN}
ATTENTION = {'config name', 'config description', ENTRY, CMD}
# new arm: (plans glob, platform, source arm, attention changes)
ARMS = {'HT r1': ('tri/tri_ht_[0-9]*.yaml', 'h100', 'H', True), 'HT r2': ('tri/tri_ht_r2_*.yaml', 'h100', 'H', True),
        'AT r2': ('tri/tri_at_r2_*.yaml', 'a100', 'AT', False)}
A100_MACHINE_TYPE = 'a2-ultragpu-1g'
_source_sessions = cd.source_sessions


def source_sessions(arm):
  if arm in ('H', 'AT'):
    return mk.source_sessions(arm)
  return _source_sessions(arm)


cd.source_sessions = source_sessions     # cd.recorded looks the name up at call time


def to_xla(values):
  """The values with the Triton attention set back to XLA (entry JSON and command)."""
  out = []
  for v in values:
    v = re.sub(r'"flash_attention_implementation": "triton"', '"flash_attention_implementation": "xla"', v)
    out.append(v.replace('--flash_attention_implementation=triton', '--flash_attention_implementation=xla'))
  return sorted(out)


def mark(k, a, b, attention):
  if a == b:
    return 'same'
  if k in ALLOWED:
    return 'intended'
  if attention and k in ('config name', 'config description'):
    return 'intended (attention)'
  if attention and k in (ENTRY, CMD):
    return 'intended (attention only)' if to_xla(b) == sorted(a) else 'DIFFERS'
  return 'DIFFERS'


def main():
  ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
  ap.add_argument('--out', default=str(REPO / 'harness' / 'plans' / 'config_diff_tri.md'))
  args = ap.parse_args()
  cd.PLANNED = mk.LABELS['tri']['provisioning']          # cd.planned reads the provisioning of each platform from it
  L = ['# Follow-up 18: settings of each new arm against its source arm (harness/plans/config_diff_tri.py)', '',
       'Source: as recorded in the source sessions (H: follow-up 9; AT r1: follow-up 3), with the image each '
       'session\'s wrapper log printed and the Eigen source of its build. New: as the plans of harness/plans/tri, '
       'harness/configs.yaml, harness/run_af3.py and cloud/af3_run.sh resolve them, Spot, with the image the launch '
       f'sets and the A100 machine type {A100_MACHINE_TYPE}. Allowed differences: the label, the repetitions, the '
       'provisioning, the image family, the Eigen source and, for HT, the attention alone. Every other row must be '
       'the same.', '']
  bad = 0
  for arm, (glob, platform, source, attention) in ARMS.items():
    src, n_src = cd.recorded(source, platform)
    sessions = source_sessions(source)
    imgs = {}
    for s in sessions:
      imgs.setdefault(cr.image_of(s) or 'not in any wrapper log', []).append(s)
    src[IMG] = sorted(f'{k} ({len(v)} sessions)' for k, v in imgs.items())
    src[EIGEN] = sorted({cr.eigen_of(s) for s in sessions})
    one = json.loads(src[cd.STACK][0])
    new, n_new = cd.planned(glob, platform, one)
    if platform == 'a100':
      new['machine type'] = [A100_MACHINE_TYPE]
    new[IMG] = [f'{IMAGE[platform]} (set at launch)']
    new[EIGEN] = ['archive 3.4.0 commit 3147391d946bb4b6c68edd901f2add6ac1f31f8c (plan key eigen_source: archive)']
    del src['image'], new['image']
    L += [f'## {arm} against {source} ({n_src} source sessions, {n_new} plans)', '',
          '| setting | source (recorded) | new (planned) | |', '|---|---|---|---|']
    for k in src:
      a, b = src[k], new.get(k)
      m = mark(k, a, b, attention)
      bad += m == 'DIFFERS'
      fmt = lambda v: '<br>'.join(f'`{x}`' for x in v) if v else '(none)'  # noqa: E731
      L.append(f'| {k} | {fmt(a)} | {fmt(b)} | {m} |')
    for k in sorted(set(new) - set(src)):
      bad += 1
      L.append(f'| {k} | (none) | {new[k]} | DIFFERS |')
    L.append('')
  L.append(f'Settings that differ beyond the allowed ones: {bad}')
  pathlib.Path(args.out).write_text('\n'.join(L) + '\n')
  print('\n'.join(L))
  return 1 if bad else 0


if __name__ == '__main__':
  sys.exit(main())
