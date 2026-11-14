"""Follow-up 15: every setting of each r2 arm against the r1 sessions it repeats (Af and Hf of follow-up 14, GT of the
main study), as harness/plans/config_diff_fu14.py does for follow-up 14 (config_diff_fu7_10's functions, imported).

    python3 harness/plans/config_diff_rcx.py [--out harness/plans/config_diff_rcx.md]

Source side, as recorded in the r1 sessions (Af and Hf: scripts/analysis_runs/fu14/sessions.txt; GT: results/analysis/
main_20261014/sessions_main.txt): launch_record.txt, every fresh harness run's session.json and command, and, unlike
the earlier diffs, the image each session was created from as its wrapper log printed it ("image PROJECT/FAMILY" in
results/af3/.many/*/<part>*.log; the launcher's default at the session's commit when no log names the session), and
the Eigen source of the build (eigen.json; none means the GitLab clone of tag 3.4.0, before follow-up 7).
New side, as the arm will run: its plans (harness/plans/rcx), harness/configs.yaml, harness/run_af3.py, cloud/
af3_run.sh's defaults, af3_tpu/, the A100 machine type a2-ultragpu-1g (A100_MACHINE_TYPE at launch), Spot on every
platform (make_rcx_plans.PROVISIONING) and the image the launch sets (IMAGE below, IMAGE_PROJECT and IMAGE_FAMILY).
Allowed differences (row of 2026-11-14, items a and f): the label; the repetitions (r2); the provisioning (GT r1 ran
on demand); the image family (the project copies of item a, which carry the same NVIDIA driver; check 4 of the
analysis compares driver, kernel and Python packages session by session); the Eigen source of GT (the archive of the
same commit, 3147391d, as every plan since follow-up 7). Any other difference is DIFFERS and the script exits 1.
"""
import argparse
import json
import pathlib
import re
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'harness' / 'plans'))
import config_diff_fu7_10 as cd  # noqa: E402
import make_rcx_plans as mk  # noqa: E402

IMG = 'image (as the wrapper log printed it for each session)'
EIGEN = 'Eigen source of the AlphaFold3 build'
IMAGE = {'a100': 'af2-tpu-benchmark/af3-accel-2204-nvidia-580', 'h100': 'af2-tpu-benchmark/af3-accel-2204-nvidia-580',
         'l4': 'af2-tpu-benchmark/af3-accel-2204-nvidia-580-gtr1'}
ALLOWED = {'label', cd.REPS, 'provisioning', IMG, EIGEN}
# r2 arm: (plans glob, platform, r1 condition)
ARMS = {'Af r2': ('rcx/rcx_af_r2_*.yaml', 'a100', 'Af'), 'Hf r2': ('rcx/rcx_hf_r2_*.yaml', 'h100', 'Hf'),
        'GT r2': ('rcx/rcx_gt_r2_*.yaml', 'l4', 'GT')}
A100_MACHINE_TYPE = 'a2-ultragpu-1g'
_source_sessions = cd.source_sessions


def source_sessions(arm):
  if arm in ('Af', 'Hf', 'GT'):
    return mk.source_sessions(arm)
  return _source_sessions(arm)


cd.source_sessions = source_sessions     # cd.recorded looks the name up at call time


def image_of(session, many=None):
  """The image the session's VM was created from, as its wrapper log printed it, or None."""
  many = many or REPO / 'results' / 'af3' / '.many'
  for log in sorted(many.glob('*/*.log')):
    text = log.read_text(errors='replace')
    if f'results/af3/{session}/' in text:
      m = re.findall(r'^\s+image (\S+?)(?:,|$)', text, re.M)
      if m:
        return m[-1]
  return None


def eigen_of(session):
  f = REPO / 'results' / 'af3' / session / 'eigen.json'
  if not f.is_file():
    return 'git clone of gitlab.com/libeigen/eigen tag 3.4.0 (before follow-up 7)'
  j = json.loads(f.read_text())
  return f'{j.get("source")} {j.get("tag")} commit {j.get("commit")}'


def main():
  ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
  ap.add_argument('--out', default=str(REPO / 'harness' / 'plans' / 'config_diff_rcx.md'))
  args = ap.parse_args()
  cd.PLANNED = mk.PROVISIONING          # cd.planned reads the provisioning of each platform from this name
  L = ['# Follow-up 15: settings of each r2 arm against the r1 sessions it repeats (harness/plans/config_diff_rcx.py)', '',
       'Source: as recorded in the r1 sessions (Af and Hf: follow-up 14; GT: main study), with the image each '
       'session\'s wrapper log printed and the Eigen source of its build. New: as the plans of harness/plans/rcx, '
       'harness/configs.yaml, harness/run_af3.py and cloud/af3_run.sh resolve them, Spot, with the image the launch '
       f'sets ({IMAGE["a100"]} for Af r2 and Hf r2, {IMAGE["l4"]} for GT r2) and the A100 machine type '
       f'{A100_MACHINE_TYPE}. Allowed differences: the label, the repetitions, the provisioning, the image family and '
       'the Eigen source of GT (the archive of the same commit). Every other row must be the same.', '']
  bad = 0
  for arm, (glob, platform, source) in ARMS.items():
    src, n_src = cd.recorded(source, platform)
    sessions = source_sessions(source)
    imgs = {}
    for s in sessions:
      imgs.setdefault(image_of(s) or 'not in any wrapper log', []).append(s)
    src[IMG] = sorted(f'{k} ({len(v)} sessions)' for k, v in imgs.items())
    src[EIGEN] = sorted({eigen_of(s) for s in sessions})
    one = json.loads(src[cd.STACK][0])
    new, n_new = cd.planned(glob, platform, one)
    if platform == 'a100':
      new['machine type'] = [A100_MACHINE_TYPE]
    new[IMG] = [f'{IMAGE[platform]} (set at launch)']
    new[EIGEN] = ['archive 3.4.0 commit 3147391d946bb4b6c68edd901f2add6ac1f31f8c (plan key eigen_source: archive)']
    del src['image'], new['image']
    L += [f'## {arm} against {source} r1 ({n_src} r1 sessions, {n_new} plans)', '',
          '| setting | r1 (recorded) | r2 (planned) | |', '|---|---|---|---|']
    for k in src:
      a, b = src[k], new.get(k)
      if k == EIGEN and source != 'GT':
        b = [x.split(' (')[0] for x in b]
      mark = 'same' if a == b else ('intended' if k in ALLOWED else 'DIFFERS')
      bad += mark == 'DIFFERS'
      fmt = lambda v: '<br>'.join(f'`{x}`' for x in v) if v else '(none)'  # noqa: E731
      L.append(f'| {k} | {fmt(a)} | {fmt(b)} | {mark} |')
    L.append('')
  L.append(f'Settings that differ beyond the allowed ones: {bad}')
  pathlib.Path(args.out).write_text('\n'.join(L) + '\n')
  print('\n'.join(L))
  return 1 if bad else 0


if __name__ == '__main__':
  sys.exit(main())
