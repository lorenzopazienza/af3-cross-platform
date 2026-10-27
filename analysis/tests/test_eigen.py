"""The Eigen archive (cloud/eigen_pin.sh): plan key eigen_source, cloud/vm_eigen_source.sh on a synthetic archive with a
fake gcloud, and scripts/monitor/eigen_check.sh on synthetic sessions. No network, no cloud."""
import json
import os
import pathlib
import shutil
import subprocess
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'harness'))
import run_plan  # noqa: E402
from test_compare_runs_strict import outputs  # noqa: E402

PLANS = REPO / 'harness' / 'plans'


def describe(path, platform):
  return run_plan.expand(path, platform, REPO / 'inputs' / 'manifest.csv')


# ---- plans

def test_followups_7_to_9_and_the_check_take_the_archive_and_nothing_else_does():
  with_key = sorted(p for p in PLANS.rglob('*.yaml') if 'eigen_source:' in p.read_text())
  folders = {p.parent.name for p in with_key}
  assert folders == {'seeds', 'sw', 'h100', 'sw_smoke', 'h100_smoke', 'eigen_check'}
  for p in with_key:
    platform = next(iter(run_plan.read_configs(p)['configs']))
    assert describe(p, platform)['eigen_source'] == 'archive', p
  for p in sorted((PLANS / 'msa').glob('*.yaml')) + sorted((PLANS / 'msa_smoke').glob('*.yaml')):
    platform = next(iter(run_plan.read_configs(p)['configs']))
    assert 'eigen_source' not in describe(p, platform), p          # follow-up 6 unchanged


def test_eigen_check_plans_are_the_follow_up_6_smoke_plans_plus_the_key():
  for pl in ('v5e', 'v6e'):
    a = describe(PLANS / 'msa_smoke' / f'msa_smoke_{pl}.yaml', pl)
    b = describe(PLANS / 'eigen_check' / f'eigen_check_{pl}.yaml', pl)
    assert b.pop('eigen_source') == 'archive' and b.pop('label') == 'eigen_check' and a.pop('label') == 'msa_smoke'
    for d in (a, b):
      d.pop('plan'), d.pop('description')
    assert a == b


def test_eigen_source_values(tmp_path):
  base = (PLANS / 'eigen_check' / 'eigen_check_v6e.yaml').read_text()
  for value, want in (('git', None), ('archive', 'archive')):
    p = tmp_path / f'{value}.yaml'
    p.write_text(base.replace('eigen_source: archive', f'eigen_source: {value}'))
    assert describe(p, 'v6e').get('eigen_source') == want
  p = tmp_path / 'bad.yaml'
  p.write_text(base.replace('eigen_source: archive', 'eigen_source: gitlab'))
  with pytest.raises(SystemExit):
    describe(p, 'v6e')


# ---- cloud/vm_eigen_source.sh

def fake_root(tmp_path, files=3, wrong_commit=False):
  """A repository copy with the real vm_eigen_source.sh, a tiny Eigen-like archive, its pin, and a fake gcloud."""
  root = tmp_path / 'repo'
  (root / 'cloud').mkdir(parents=True)
  shutil.copy(REPO / 'cloud' / 'vm_eigen_source.sh', root / 'cloud')
  src = tmp_path / 'eigen-git'
  src.mkdir()
  for i in range(files):
    (src / f'h{i}.h').write_text(f'// header {i}\n')
  git = ['git', '-C', str(src), '-c', 'user.name=t', '-c', 'user.email=t@invalid']
  subprocess.run(['git', 'init', '-q', str(src)], check=True)
  subprocess.run(git + ['add', '.'], check=True)
  subprocess.run(git + ['commit', '-q', '-m', 'x'], check=True)
  commit = subprocess.run(git + ['rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()
  deps = tmp_path / 'bucket'
  deps.mkdir()
  archive = deps / 'eigen-test.tar.gz'
  with open(archive, 'wb') as f:
    tar = subprocess.run(git + ['archive', '--format=tar', '--prefix=eigen/', commit], capture_output=True, check=True)
    f.write(subprocess.run(['gzip', '-n'], input=tar.stdout, capture_output=True, check=True).stdout)
  sha = subprocess.run(['shasum', '-a', '256', str(archive)], capture_output=True, text=True).stdout.split()[0]
  (root / 'cloud' / 'eigen_pin.sh').write_text(
      f'export EIGEN_TAG="test"\nexport EIGEN_COMMIT="{"0" * 40 if wrong_commit else commit}"\n'
      f'export EIGEN_ARCHIVE_NAME="eigen-test.tar.gz"\nexport EIGEN_FILES="{files}"\n'
      f'export EIGEN_ARCHIVE_URI="gs://test-bucket/deps/eigen-test.tar.gz"\nexport EIGEN_ARCHIVE_SHA256="{sha}"\n')
  binp = tmp_path / 'bin'
  binp.mkdir()
  (binp / 'gcloud').write_text(f'#!/bin/bash\n[ "$1 $2" = "storage cp" ] || exit 9\ncp "{deps}/$(basename "$3")" "$4"\n')
  (binp / 'gcloud').chmod(0o755)
  return root, deps, binp


def vm_eigen(root, binp, tmp_path, env_source='archive'):
  out = tmp_path / 'out'
  out.mkdir(exist_ok=True)
  env = dict(os.environ, PATH=f'{binp}:{os.environ["PATH"]}', EIGEN_DEPS_DIR=str(tmp_path / 'deps'))
  env.pop('EIGEN_SOURCE', None)
  if env_source:
    env['EIGEN_SOURCE'] = env_source
  return subprocess.run(['bash', str(root / 'cloud' / 'vm_eigen_source.sh'), str(out)], capture_output=True, text=True,
                        env=env), out


def test_vm_eigen_source_installs_the_pinned_archive_and_records_it(tmp_path):
  root, _, binp = fake_root(tmp_path)
  r, out = vm_eigen(root, binp, tmp_path)
  assert r.returncode == 0, r.stderr
  folder = pathlib.Path(r.stdout.strip())
  assert folder == tmp_path / 'deps' / 'eigen-src' / 'eigen' and sorted(p.name for p in folder.iterdir()) == ['h0.h', 'h1.h', 'h2.h']
  j = json.loads((out / 'eigen.json').read_text())
  assert j['source'] == 'archive' and j['files'] == 3 and j['fetchcontent_source_dir'] == str(folder)
  assert 'gitlab.com/libeigen/eigen.git' in j['replaces']


def test_vm_eigen_source_refuses_another_file_or_commit_and_needs_the_opt_in(tmp_path):
  root, deps, binp = fake_root(tmp_path / 'a')
  (deps / 'eigen-test.tar.gz').write_bytes(b'not the archive')
  r, out = vm_eigen(root, binp, tmp_path / 'a')
  assert r.returncode == 3 and 'SHA-256' in r.stderr and not (out / 'eigen.json').exists()
  assert not (tmp_path / 'a' / 'deps' / 'eigen-src').exists()
  root, _, binp = fake_root(tmp_path / 'b', wrong_commit=True)
  r, _ = vm_eigen(root, binp, tmp_path / 'b')
  assert r.returncode == 3 and 'carries commit' in r.stderr
  root, _, binp = fake_root(tmp_path / 'c')
  r, _ = vm_eigen(root, binp, tmp_path / 'c', env_source=None)
  assert r.returncode == 3


# ---- scripts/monitor/eigen_check.sh

REF = '20261026T142437Z_v6e_msa_smoke_v6e'
RUN = 'tpu_xla_rep1_seed1_fresh'


def check_tree(tmp_path, fetch='full', ref_eigen=False, differ=False):
  new = tmp_path / 'results' / 'af3' / 'NEW_v6e_eigen_check_v6e'
  (new / RUN).mkdir(parents=True)
  (new / RUN / 'session.json').write_text('{"created_utc": "2026-11-01T00:00:00Z"}')
  outputs(new / RUN, t='2PV7', x='1.500' if differ else '1.000', light=fetch == 'light')
  pin = {l.split('=')[0].split()[-1]: l.split('"')[1] for l in (REPO / 'cloud' / 'eigen_pin.sh').read_text().splitlines()
         if l.startswith('export')}
  (new / 'launch_record.txt').write_text(f'EIGEN_SOURCE=archive\nFETCH={fetch}\n')
  (new / 'eigen.json').write_text(json.dumps({'source': 'archive', 'archive_sha256': pin['EIGEN_ARCHIVE_SHA256'],
                                              'commit': pin['EIGEN_COMMIT']}))
  ref = tmp_path / 'results' / 'af3' / REF
  ref.mkdir(parents=True)
  (ref / 'launch_record.txt').write_text('FETCH=light\n' + ('EIGEN_SOURCE=archive\n' if ref_eigen else ''))
  outputs(tmp_path / 'results' / 'eigen_check_ref' / REF / RUN, t='2PV7')   # the full copy from the bucket
  return subprocess.run(['bash', str(REPO / 'scripts' / 'monitor' / 'eigen_check.sh'), 'v6e', new.name],
                        cwd=tmp_path, capture_output=True, text=True)


def test_eigen_check_passes_on_identical_outputs(tmp_path):
  r = check_tree(tmp_path)
  assert r.returncode == 0 and r.stdout.strip().splitlines()[-1].startswith('PASS'), r.stdout + r.stderr


@pytest.mark.parametrize('kw, why', [({'differ': True}, 'outputs differ'), ({'fetch': 'light'}, 'fetched light'),
                                     ({'ref_eigen': True}, 'did not clone Eigen')])
def test_eigen_check_fails(tmp_path, kw, why):
  r = check_tree(tmp_path, **kw)
  assert r.returncode == 1 and why in r.stdout, r.stdout + r.stderr
