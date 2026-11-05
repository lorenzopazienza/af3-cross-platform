"""Follow-up 10 (label large): every setting of V5l, V6l and Gl against the main-study arm each derives from (V5, V6,
G), as harness/plans/config_diff_fu11_12.py does for follow-ups 11 and 12 (config_diff_fu7_10's functions, imported).

    python3 harness/plans/config_diff_fu10.py [--out harness/plans/config_diff_fu10.md]

Source side, as recorded in the main-study sessions of V5, V6 and G (results/analysis/main_20261014/sessions_main.txt).
New side, as the arm will run: its plans (harness/plans/large), harness/configs.yaml, harness/run_af3.py,
cloud/af3_run.sh's defaults, af3_tpu/, with the provisioning of harness/plans/make_fu10_plans.PROPOSED. Intended
differences (row of 2026-11-04, item c): the label, the input manifest (follow-up 10's own targets), the repetitions
(r2 on a subset of the targets on V5l and V6l, on all on Gl; no warm reruns) and the provisioning (how the VM is
obtained, not the computation). Any other difference is DIFFERS and the script exits 1. Until
inputs/large/manifest.csv exists, the plans are expanded against a temporary manifest made from
targets/large/large_subset.csv (the same targets and token counts; the report says so).
"""
import argparse
import csv
import json
import pathlib
import sys
import tempfile

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'harness' / 'plans'))
import config_diff_fu7_10 as cd  # noqa: E402
from make_fu10_plans import MANIFEST, PROPOSED, SUBSET  # noqa: E402

INPUT_MANIFEST = 'input manifest'
# new arm: (plans glob, platform, source arm, intended differences)
ARMS = {
    'V5l': ('large/large_v5l_*.yaml', 'v5e', 'V5', {'label', INPUT_MANIFEST, cd.REPS, 'provisioning'}),
    'V6l': ('large/large_v6l_*.yaml', 'v6e', 'V6', {'label', INPUT_MANIFEST, cd.REPS, 'provisioning'}),
    'Gl': ('large/large_gl_*.yaml', 'l4', 'G', {'label', INPUT_MANIFEST, cd.REPS, 'provisioning'}),
}


def temporary_manifest():
  """A repository-like folder holding inputs/large/manifest.csv made from the targets CSV, or None if the real one
  exists. harness/run_plan.expand reads a plan's manifest key relative to its module's REPO only."""
  if (REPO / MANIFEST).is_file():
    return None
  tmp = pathlib.Path(tempfile.mkdtemp())
  (tmp / MANIFEST).parent.mkdir(parents=True)
  rows = list(csv.DictReader(open(REPO / SUBSET)))
  with open(tmp / MANIFEST, 'w', newline='') as f:
    w = csv.DictWriter(f, ['pdb_id', 'pilot', 'num_tokens', 'bucket'], extrasaction='ignore', lineterminator='\n')
    w.writeheader()
    w.writerows(rows)
  return tmp


def main():
  ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
  ap.add_argument('--out', default=str(REPO / 'harness' / 'plans' / 'config_diff_fu10.md'))
  args = ap.parse_args()
  cd.PLANNED = PROPOSED          # cd.planned reads the provisioning of each platform from this name
  tmp = temporary_manifest()
  L = ['# Follow-up 10: settings of each new arm against its source arm (harness/plans/config_diff_fu10.py)', '',
       'Source arm: as recorded in its main-study sessions. New arm: as its plans, harness/configs.yaml, '
       'harness/run_af3.py and cloud/af3_run.sh resolve them, with the provisioning of the row (on demand on v5e, '
       'Flex-start on v6e, Spot on L4). Intended differences: the label, the input manifest, the repetitions and the '
       'provisioning (how the VM is obtained, not the computation).', '']
  if tmp:
    L += [f'{MANIFEST} does not exist yet: the plans are expanded against the targets of {SUBSET} (rerun this '
          'script once the inputs are built).', '']
  bad = 0
  for arm, (glob, platform, source, intended) in ARMS.items():
    src, n_src = cd.recorded(source, platform)
    one = json.loads(src[cd.STACK][0])
    saved = cd.run_plan.REPO
    if tmp:
      cd.run_plan.REPO = tmp
    try:
      new, n_new = cd.planned(glob, platform, one)
    finally:
      cd.run_plan.REPO = saved
    L += [f'## {arm} against {source} ({n_src} source sessions, {n_new} plans)', '',
          '| setting | source arm (recorded) | new arm (planned) | |', '|---|---|---|---|']
    for k in src:
      same = src[k] == new.get(k)
      mark = 'same' if same else ('intended' if k in intended else 'DIFFERS')
      bad += mark == 'DIFFERS'
      fmt = lambda v: '<br>'.join(f'`{x}`' for x in v) if v else '(none)'  # noqa: E731
      L.append(f'| {k} | {fmt(src[k])} | {fmt(new.get(k))} | {mark} |')
    extra = sorted(set(new) - set(src))
    if extra:
      L.append(f'| (settings only on the new side: {", ".join(extra)}) | | | DIFFERS |')
      bad += 1
    L.append('')
  L.append(f'Settings that differ beyond the intended ones: {bad}')
  pathlib.Path(args.out).write_text('\n'.join(L) + '\n')
  print('\n'.join(L))
  return 1 if bad else 0


if __name__ == '__main__':
  sys.exit(main())
