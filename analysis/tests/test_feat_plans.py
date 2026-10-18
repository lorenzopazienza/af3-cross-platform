"""Follow-up 5 plans, manifest and prices (notes/analysis_plan.md section 10, row of 2026-10-18; label feat):
harness/plans/feat/, inputs/feat/manifest.csv, results/parts_feat.txt and the cloud/prices.csv rows."""
import csv
import json
import math
import pathlib
import re
import subprocess
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'harness'))
sys.path.insert(0, str(REPO / 'analysis'))
import cost  # noqa: E402
import run_plan  # noqa: E402

CCD = 'f1bef735f3d74171fb7e79af6b46fb1a3e753f6602f5b08ec90312013e33244b'
HOSTS = {'milan': ('n2d-standard-32', 'AMD Milan', 'ubuntu-os-cloud', 'ubuntu-2204-lts'),
         'genoa': ('c3d-standard-30', 'AMD Genoa', 'ubuntu-os-accelerator-images', 'ubuntu-accel-2204-amd64-tpu-v5e-v5p-v6e'),
         'cascadelake': ('c2-standard-8', 'Intel Cascade Lake', 'ubuntu-os-accelerator-images',
                         'ubuntu-accelerator-2204-amd64-with-nvidia-580'),
         'icelake': ('n2-highmem-16', 'Intel Ice Lake', 'ubuntu-os-cloud', 'ubuntu-2204-lts')}


def describe(name, platform='cpu'):
  return run_plan.expand(REPO / 'harness' / 'plans' / 'feat' / f'feat_{name}.yaml', platform, REPO / 'inputs/manifest.csv')


def manifest_ids(path):
  return [r['pdb_id'] for r in csv.DictReader(open(REPO / path))]


@pytest.mark.parametrize('name', sorted(HOSTS))
def test_each_plan_is_featurisation_only_on_its_host(name):
  d = describe(name)
  assert d['featurise_only'] and d['weights'] == 'none' and d['label'] == 'feat' and d['platform'] == 'cpu'
  assert sorted(d['targets']) == sorted(manifest_ids('inputs/manifest.csv') + manifest_ids('inputs/ppi/manifest.csv'))
  assert len(d['targets']) == 90 and d['seeds'] == [1, 2, 3, 4, 5] and d['save_arrays'] and d['ccd_sha256'] == CCD
  assert tuple(d['host'][k] for k in run_plan.HOST_KEYS) == HOSTS[name]
  head = (REPO / 'harness' / 'plans' / 'feat' / f'feat_{name}.yaml').read_text()
  job = int(re.search(r'= (\d+) min;\n# deadline', head).group(1))
  assert d['deadline_min'] == math.ceil(job * 2.5 + 20)


def test_featurise_plans_refuse_an_accelerator_platform_and_weights(tmp_path):
  with pytest.raises(SystemExit, match='PLATFORM cpu'):
    describe('milan', 'l4')
  text = (REPO / 'harness' / 'plans' / 'feat' / 'feat_milan.yaml').read_text()
  for bad, msg in [(text.replace('weights: none', 'weights: gcs'), 'weights: none'),
                   (text.replace('  machine_type: n2d-standard-32', '  machine_type: g2-standard-8'), 'CPU-only'),
                   (text.replace('  image_family: ubuntu-2204-lts\n', ''), 'fixes its host')]:
    p = tmp_path / 'plan.yaml'
    p.write_text(bad)
    with pytest.raises(SystemExit, match=msg):
      run_plan.expand(p, 'cpu', REPO / 'inputs/manifest.csv')


def test_other_plans_describe_as_before():
  d = run_plan.expand(REPO / 'harness' / 'plans' / 'cpu60' / 'cpu60_7d5c.yaml', 'cpu', REPO / 'inputs/manifest.csv')
  assert 'featurise_only' not in d and 'host' not in d and d['weights'] == 'gcs'


def test_feat_manifest_is_the_two_frozen_manifests():
  rc = subprocess.run([sys.executable, 'inputs/feat/build_feat_manifest.py', '--check'], cwd=REPO, capture_output=True,
                      text=True)
  assert rc.returncode == 0, rc.stdout
  rows = {r['pdb_id']: r for r in csv.DictReader(open(REPO / 'inputs/feat/manifest.csv'))}
  for src in ('inputs/manifest.csv', 'inputs/ppi/manifest.csv'):
    for r in csv.DictReader(open(REPO / src)):
      assert rows[r['pdb_id']]['source_manifest'] == src
      assert all(rows[r['pdb_id']][k] == r[k] for k in ('input_path', 'input_sha256', 'num_tokens', 'bucket'))


def test_parts_file_has_one_cpu_part_per_host_family():
  lines = [l.split() for l in (REPO / 'results' / 'parts_feat.txt').read_text().splitlines()
           if l.strip() and not l.startswith('#')]
  assert lines == [['cpu', f'harness/plans/feat/feat_{n}.yaml'] for n in ('milan', 'genoa', 'cascadelake', 'icelake')]


def test_new_cpu_prices_leave_the_c_arm_price_unchanged():
  # analysis/cost.py keys prices by (platform, region, provisioning): the n2-highmem-16 rows must stay last.
  p = cost.load_prices()
  assert p[('cpu', 'europe-west4', 'standard')] == (1.153824, '2026-10-08')
  assert p[('cpu', 'europe-west4', 'spot')] == (0.321312, '2026-10-08')
  rows = [r for r in csv.DictReader(open(REPO / 'cloud' / 'prices.csv')) if r['platform'] == 'cpu']
  have = {(r['machine_type'], r['provisioning']): float(r['usd_per_hour']) for r in rows}
  for mt, *_ in HOSTS.values():
    assert have[(mt, 'standard')] > have[(mt, 'spot')] > 0
