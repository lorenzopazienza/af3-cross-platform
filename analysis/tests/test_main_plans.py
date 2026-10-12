"""harness/plans/main/ against notes/analysis_plan.md section 7 and the section 10 rows of 2026-10-12 (main study
design and execution), read through harness/run_plan.py as the VMs read them; and harness/plans/gt_smoke.yaml."""

import csv
import math
import pathlib

import pytest

from run_plan import expand

REPO = pathlib.Path(__file__).resolve().parents[2]
PLANS = REPO / 'harness' / 'plans'
MANIFEST = REPO / 'inputs' / 'manifest.csv'
ROWS = {r['pdb_id']: r for r in csv.DictReader(open(MANIFEST))}
SUBSET = [r['pdb_id'] for r in csv.DictReader(open(REPO / 'targets' / 'posebusters_subset.csv'))]
PILOT = sorted(t for t in SUBSET if ROWS[t]['pilot'] == 'yes')
SMALL = sorted(t for t in PILOT if int(ROWS[t]['num_tokens']) <= 512)
DESIGN = {  # arm: (platform, config, seeds, fresh reps, warm rerun of seed 1 on the pilot targets, targets)
    'v5': ('v5e', 'tpu_xla', 5, 2, True, SUBSET), 'v6': ('v6e', 'tpu_xla', 5, 2, True, SUBSET),
    'g': ('l4', 'l4_xla', 5, 2, True, SUBSET), 'gd': ('l4', 'l4_xla_det', 5, 2, True, SUBSET),
    'gt': ('l4', 'l4_triton', 5, 1, False, SUBSET), 'v6h': ('v6e', 'tpu_xla_highest', 2, 1, False, SUBSET),
    'gh': ('l4', 'l4_xla_highest', 2, 1, False, SUBSET), 'c': ('cpu', 'cpu_xla', 2, 1, False, SMALL),
}
MAIN = sorted((PLANS / 'main').glob('main_*.yaml'))


def arm_of(path):
  return path.stem.split('_')[1]


def expanded(path):
  return expand(path, DESIGN[arm_of(path)][0], MANIFEST)


def test_targets_and_parts_file():
  assert len(SUBSET) == len(set(SUBSET)) == 60 and set(ROWS) == set(SUBSET)
  assert len(PILOT) == 10 and len(SMALL) == 6
  lines = [l.split() for l in open(PLANS / 'main' / 'parts_all.txt') if l.strip() and not l.startswith('#')]
  assert sorted(pathlib.Path(l[1]).name for l in lines) == sorted(p.name for p in MAIN)
  assert all(l[0] == DESIGN[arm_of(pathlib.Path(l[1]))][0] and (REPO / l[1]).is_file() for l in lines)


@pytest.mark.parametrize('arm', sorted(DESIGN))
def test_each_arm_has_exactly_the_design_cells(arm):
  platform, config, seeds, reps, warm, targets = DESIGN[arm]
  cells, parts_of = {}, {}
  for path in [p for p in MAIN if arm_of(p) == arm]:
    d = expanded(path)
    assert (d['label'], d['weights'], d['configs']) == ('main', 'gcs', [config])
    assert d['num_recycles'] is None and d['num_diffusion_samples'] is None
    runs = {r['run']: r for r in d['runs']}
    for r in d['runs']:
      rep = 'w' if r['cache'] == 'warm' else f'r{r["rep"]}'
      if rep == 'w':   # the warm rerun reads the cache its own r1 wrote, in the same part (same VM)
        src = runs[r['source_run']]
        assert src['seed'] == r['seed'] == 1 and src['rep'] == 1 and src['save_cache']
      for t in d['targets']:
        cells[(t, r['seed'], rep)] = cells.get((t, r['seed'], rep), 0) + 1
        parts_of.setdefault(t, set()).add(path.stem)
  want = {(t, s, f'r{r}') for t in targets for s in range(1, seeds + 1) for r in range(1, reps + 1)}
  want |= {(t, 1, 'w') for t in PILOT} if warm else set()
  assert set(cells) == want and set(cells.values()) == {1}
  assert all(len(p) == 1 for p in parts_of.values()), 'all runs of an (arm, target) in one part'


def test_deadlines_fit_the_vm_lifetime():
  for path in MAIN:
    d = expanded(path)
    life = d['deadline_min'] + 30          # cloud/af3_run.sh: MAX_RUN_DURATION = deadline + 30 min
    assert 0 < d['deadline_min'] and life >= d['deadline_min'] + 10
    if DESIGN[arm_of(path)][0] == 'l4':
      assert life <= 6 * 60, f'{path.name}: an L4 part should fit about 5 to 6 hours'


def test_gt_smoke_plan():
  d = expand(PLANS / 'gt_smoke.yaml', 'l4', MANIFEST)
  assert (d['label'], d['weights'], d['configs'], d['targets']) == ('smoke', 'gcs', ['l4_triton'], ['7U3J'])
  assert [r['run'] for r in d['runs']] == ['l4_triton_rep1_seed1_fresh'] and d['processes'] == 1
  assert d['deadline_min'] == math.ceil((8.4 + 127 / 60) * 1.25 + 20)
  for other in ('cpu', 'v5e', 'v6e'):
    with pytest.raises(SystemExit):
      expand(PLANS / 'gt_smoke.yaml', other, MANIFEST)
