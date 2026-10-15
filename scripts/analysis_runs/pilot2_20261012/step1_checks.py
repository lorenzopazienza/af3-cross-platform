"""Pilot2, Phase 2 Step 1 on the completed arms: check 1 and check 2 (2a, 2b, then bit-identity).

    analysis/.venv/bin/python scripts/analysis_runs/pilot2_20261012/step1_checks.py STEP_DIR [--bit_identity]

STEP_DIR holds samples.csv from analysis/collect.py --no_m1 (no M1, no outcome). Check 1: every process
exit 0, and every (target, seed, repetition) of the arm's plan files present. Check 2 for the arms with a
warm rerun (V5, V6, G, Gd): 2a and 2b from the run records (analysis/contrasts.cache_checks), then, only with
--bit_identity, the strict comparison of every output file (analysis/contrasts.check2). Writes
STEP_DIR/checks.csv and prints the rows. No contrast is built.
"""

import pathlib
import sys

import pandas as pd

REPO = pathlib.Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO / 'analysis'), str(REPO / 'harness')]
import contrasts  # noqa: E402
from run_plan import expand  # noqa: E402

PLANS = {'V5': [('pilot.yaml', 'v5e')], 'V6': [('pilot.yaml', 'v6e')], 'V6h': [('pilot_v6h.yaml', 'v6e')],
         'G': [('pilot_l4_1.yaml', 'l4'), ('pilot_l4_2.yaml', 'l4')],
         'Gd': [('pilot_l4d_1.yaml', 'l4'), ('pilot_l4d_2.yaml', 'l4')],
         'Gh': [('pilot_l4h_1.yaml', 'l4'), ('pilot_l4h_2.yaml', 'l4')],
         'C': [(f'pilot_cpu_{i}.yaml', 'cpu') for i in (1, 2, 3, 4)]}
WARM = ['V5', 'V6', 'G', 'Gd']


def planned(arm):
  cells = set()
  for plan, platform in PLANS[arm]:
    desc = expand(REPO / 'harness' / 'plans' / plan, platform, REPO / 'inputs' / 'manifest.csv')
    for run in desc['runs']:
      rep = 'w' if run['cache'] == 'warm' else f'r{run["rep"]}'
      cells |= {(t, run['seed'], rep) for t in desc['targets']}
  return cells


def main():
  step = pathlib.Path(sys.argv[1])
  d = pd.read_csv(step / 'samples.csv')
  d['condition'] = d.platform
  rows = []
  for arm in [a for a in PLANS if a in set(d.condition)]:
    g = d[d.condition == arm]
    procs = g.drop_duplicates(['session', 'target', 'seed'])
    failed = procs[procs.exit_code != 0]
    have = {(r.target, int(r.seed), r.repetition) for r in procs.itertuples()}
    want = planned(arm)
    missing, extra = sorted(want - have), sorted(have - want)
    ok = failed.empty and not missing and not extra
    detail = 'pass' if ok else 'FAIL: ' + '; '.join(
        [f'{r.target} {r.repetition} seed {r.seed}: exit {r.exit_code} {r.failure}' for r in failed.itertuples()]
        + [f'missing {m}' for m in missing] + [f'not in plan {e}' for e in extra])
    rows.append({'check': '1 exit codes and planned runs present', 'condition': arm,
                 'value': f'{len(procs) - len(failed)}/{len(procs)} processes exit 0; {len(want & have)}/{len(want)} '
                          'planned (target, seed, repetition)', 'detail': detail[:600]})
  cache_rows = contrasts.cache_checks(d)
  for arm in PLANS:
    if arm not in WARM:
      rows += [dict(r, check=r['check'] + ' (informative: no warm rerun in this arm)')
               for r in cache_rows if r['condition'] == arm]
  for arm in WARM:
    if '--bit_identity' in sys.argv:
      rows += contrasts.check2(d, pd.DataFrame(), cache_rows, arm)
    else:
      rows += [r for r in cache_rows if r['condition'] == arm]
  out = pd.DataFrame(rows)
  out.to_csv(step / ('checks_with_bit_identity.csv' if '--bit_identity' in sys.argv else 'checks.csv'), index=False)
  for r in out.itertuples():
    print(f'{r.condition:4} | {r.check} | {r.value} | {r.detail}')


if __name__ == '__main__':
  main()
