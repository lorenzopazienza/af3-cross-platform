"""Follow-ups 11 and 12: every setting of each new arm against the main-study arm it derives from, as
harness/plans/config_diff_fu7_10.py does for follow-ups 7 to 9 (its functions, imported).

    python3 harness/plans/config_diff_fu11_12.py [--out harness/plans/config_diff_fu11_12.md]

Source side, as recorded in the main-study sessions of V5, V6, G, V6h and Gh (results/analysis/main_20261014/
sessions_main.txt). New side, as the arm will run: its plans (harness/plans/{repl,prec5}), harness/configs.yaml,
harness/run_af3.py, cloud/af3_run.sh's defaults, af3_tpu/, with the provisioning proposed in
harness/plans/make_fu11_12_plans.PROPOSED. Intended differences (rows of 2026-11-03): the label, the input manifest
(follow-up 11's own targets), the seeds (follow-up 12: 3 to 5), the repetitions (no warm reruns) and the provisioning
(how the VM is obtained, not the computation). Any other difference is DIFFERS and the script exits 1. Until
inputs/repl/manifest.csv exists, the follow-up 11 plans are expanded against a temporary manifest made from
targets/repl/repl_subset.csv (the same targets and token counts; the report says so).
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
from make_fu11_12_plans import PROPOSED, REPL_MANIFEST  # noqa: E402

MANIFEST = 'input manifest'
# new arm: (plans glob, platform, source arm, intended differences)
ARMS = {
    'V5r': ('repl/repl_v5r_*.yaml', 'v5e', 'V5', {'label', MANIFEST, cd.REPS, 'provisioning'}),
    'V6r': ('repl/repl_v6r_*.yaml', 'v6e', 'V6', {'label', MANIFEST, cd.REPS, 'provisioning'}),
    'Gr': ('repl/repl_gr_*.yaml', 'l4', 'G', {'label', MANIFEST, cd.REPS, 'provisioning'}),
    'V6h5': ('prec5/prec5_v6h5_*.yaml', 'v6e', 'V6h', {'label', cd.SEEDS, 'provisioning'}),
    'Gh5': ('prec5/prec5_gh5_*.yaml', 'l4', 'Gh', {'label', cd.SEEDS, 'provisioning'}),
}


def temporary_repl_manifest():
  """A repository-like folder holding inputs/repl/manifest.csv made from the targets CSV, or None if the real one
  exists. harness/run_plan.expand reads a plan's manifest key relative to its module's REPO only."""
  if (REPO / REPL_MANIFEST).is_file():
    return None
  tmp = pathlib.Path(tempfile.mkdtemp())
  (tmp / 'inputs' / 'repl').mkdir(parents=True)
  rows = list(csv.DictReader(open(REPO / 'targets' / 'repl' / 'repl_subset.csv')))
  with open(tmp / REPL_MANIFEST, 'w', newline='') as f:
    w = csv.DictWriter(f, ['pdb_id', 'pilot', 'num_tokens', 'bucket'], extrasaction='ignore', lineterminator='\n')
    w.writeheader()
    w.writerows(rows)
  return tmp


def main():
  ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
  ap.add_argument('--out', default=str(REPO / 'harness' / 'plans' / 'config_diff_fu11_12.md'))
  args = ap.parse_args()
  cd.PLANNED = PROPOSED          # cd.planned reads the provisioning of each platform from this name
  tmp = temporary_repl_manifest()
  L = ['# Follow-ups 11 and 12: settings of each new arm against its source arm '
       '(harness/plans/config_diff_fu11_12.py)', '',
       'Source arm: as recorded in its main-study sessions. New arm: as its plans, harness/configs.yaml, '
       'harness/run_af3.py and cloud/af3_run.sh resolve them, with the proposed provisioning (on demand on v5e, '
       'Flex-start on v6e, Spot on L4). Intended differences: the label, the input manifest (follow-up 11), the seeds '
       '(follow-up 12), the repetitions and the provisioning (how the VM is obtained, not the computation).', '']
  if tmp:
    L += [f'{REPL_MANIFEST} does not exist yet: the follow-up 11 plans are expanded against the targets of '
          'targets/repl/repl_subset.csv (rerun this script once the inputs are built).', '']
  bad = 0
  for arm, (glob, platform, source, intended) in ARMS.items():
    src, n_src = cd.recorded(source, platform)
    one = json.loads(src[cd.STACK][0])
    saved = cd.run_plan.REPO
    if tmp and glob.startswith('repl/'):
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
    L.append('')
  L.append(f'Settings that differ beyond the intended ones: {bad}')
  pathlib.Path(args.out).write_text('\n'.join(L) + '\n')
  print('\n'.join(L))
  return 1 if bad else 0


if __name__ == '__main__':
  sys.exit(main())
