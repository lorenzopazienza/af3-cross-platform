"""harness/plans/embed/ and harness/plans/ppi/ against the section 10 rows of 2026-10-14 (follow-up experiments 1
and 2), read through harness/run_plan.py as the VMs read them; the save_embeddings and manifest plan keys.
End-to-end launches on fake VMs: tests/cloud/suite_8_followups.sh."""

import csv
import pathlib

import pytest

from run_plan import expand

REPO = pathlib.Path(__file__).resolve().parents[2]
PLANS = REPO / 'harness' / 'plans'
MANIFEST = REPO / 'inputs' / 'manifest.csv'
PILOT = sorted(r['pdb_id'] for r in csv.DictReader(open(MANIFEST)) if r['pilot'] == 'yes')
EMBED = {'v5': ('v5e', 'tpu_xla', 1), 'v6': ('v6e', 'tpu_xla', 1), 'g': ('l4', 'l4_xla', 2), 'gd': ('l4', 'l4_xla_det', 1)}
PPI = {'v5': ('v5e', 'tpu_xla', 1), 'v6': ('v6e', 'tpu_xla', 2), 'g': ('l4', 'l4_xla', 2)}
PPI_MANIFEST = REPO / 'inputs' / 'ppi' / 'manifest.csv'


@pytest.mark.parametrize('arm', sorted(EMBED))
def test_embed_plan_is_the_specified_experiment(arm):
  platform, config, reps = EMBED[arm]
  d = expand(PLANS / 'embed' / f'embed_{arm}.yaml', platform, MANIFEST)
  assert d['label'] == 'embed' and d['weights'] == 'gcs' and d['save_embeddings'] is True
  assert sorted(d['targets']) == PILOT and d['seeds'] == [1] and d['configs'] == [config]
  assert d['fresh_reps'] == reps and d['warm_rerun_seed'] is None and d['processes'] == 10 * reps
  assert d['num_recycles'] is None and d['num_diffusion_samples'] is None and 'manifest' not in d
  assert all(r['cache'] == 'fresh' for r in d['runs'])


def test_embed_parts_file():
  lines = [l.split() for l in (PLANS / 'embed' / 'parts.txt').read_text().splitlines() if l and not l.startswith('#')]
  assert sorted(l[1] for l in lines) == sorted(f'harness/plans/embed/embed_{a}.yaml' for a in EMBED)
  assert all(any(x.startswith('ZONES=') for x in l) == (l[0] == 'v6e') for l in lines)


def test_plan_keys_are_validated(tmp_path):
  base = (PLANS / 'embed' / 'embed_g.yaml').read_text()
  p = tmp_path / 'p.yaml'
  p.write_text(base.replace('save_embeddings: true', 'save_embeddings: yes'))
  with pytest.raises(SystemExit, match='save_embeddings'):
    expand(p, 'l4', MANIFEST)
  for bad in ('/etc/manifest.csv', '../outside.csv'):
    p.write_text(base.replace('save_embeddings: true', f'manifest: {bad}'))
    with pytest.raises(SystemExit, match='manifest'):
      expand(p, 'l4', MANIFEST)


needs_ppi = pytest.mark.skipif(not PPI_MANIFEST.exists() or not list((PLANS / 'ppi').glob('ppi_*.yaml')),
                               reason='follow-up 1 inputs and plans not built yet')


@needs_ppi
@pytest.mark.parametrize('arm', sorted(PPI))
def test_ppi_parts_cover_every_target_once(arm):
  platform, config, reps = PPI[arm]
  targets = sorted(r['pdb_id'] for r in csv.DictReader(open(PPI_MANIFEST)))
  assert len(targets) == 30
  seen = []
  for plan in sorted((PLANS / 'ppi').glob(f'ppi_{arm}_*.yaml')):
    d = expand(plan, platform, MANIFEST)          # the plan's manifest key overrides this one
    assert d['label'] == 'ppi' and d['weights'] == 'gcs' and d['manifest'] == 'inputs/ppi/manifest.csv'
    assert d['seeds'] == [1, 2, 3, 4, 5] and d['fresh_reps'] == reps and d['configs'] == [config]
    assert d['warm_rerun_seed'] is None and 'save_embeddings' not in d
    seen += d['targets']
  assert sorted(seen) == targets


@needs_ppi
def test_ppi_parts_file_lists_every_part():
  lines = [l.split() for l in (PLANS / 'ppi' / 'parts.txt').read_text().splitlines() if l and not l.startswith('#')]
  assert sorted(l[1] for l in lines) == sorted(f'harness/plans/ppi/{p.name}' for p in (PLANS / 'ppi').glob('ppi_*.yaml'))
  assert all(any(x.startswith('ZONES=') for x in l) == (l[0] == 'v6e') for l in lines)
