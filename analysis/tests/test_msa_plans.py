"""Follow-up 6 (label msa) harness support, plans, smoke plans, parts files and settings (notes/analysis_plan.md
section 10, row of 2026-10-19): rep_start and the plan dates in harness/run_plan.py, the plans as
harness/plans/msa/make_msa_plans.py writes them, the settings against the main study (config_diff.py), and a
_data.json with templates through the harness's one-seed input."""
import csv
import json
import pathlib
import shutil
import subprocess
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parents[2]
DATA = REPO / 'analysis' / 'tests' / 'fixtures'
sys.path.insert(0, str(REPO / 'harness'))
import featurise_only  # noqa: E402
import run_plan  # noqa: E402

PLANS = REPO / 'harness' / 'plans' / 'msa'


@pytest.fixture
def repo_with_manifest(tmp_path, monkeypatch):
  """run_plan reading the msa plans against a manifest like the one the ingest writes (same targets and buckets)."""
  (tmp_path / 'inputs' / 'official_data').mkdir(parents=True)
  rows = list(csv.DictReader(open(REPO / 'inputs' / 'manifest.csv')))
  with open(tmp_path / 'inputs' / 'official_data' / 'manifest.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=['pdb_id', 'pilot', 'input_path', 'input_sha256', 'num_tokens', 'bucket'])
    w.writeheader()
    w.writerows({k: r[k] for k in ('pdb_id', 'pilot', 'num_tokens', 'bucket')} | {'input_path': f'x/{r["pdb_id"]}.json',
                 'input_sha256': 'x'} for r in rows)
  (tmp_path / 'inputs' / 'official_data' / 'smoke_manifest.csv').write_text(
      'pdb_id,pilot,input_path,input_sha256,num_tokens,bucket\n2PV7,no,x/2PV7.json,x,600,768\n')
  monkeypatch.setattr(run_plan, 'REPO', tmp_path)
  return tmp_path


def describe(plan, platform):
  return run_plan.expand(plan, platform, REPO / 'inputs' / 'manifest.csv')


def test_rep_start_runs_r2_only_and_refuses_a_warm_rerun(tmp_path, repo_with_manifest):
  d = describe(PLANS / 'msa_go_r2_01.yaml', 'l4')
  assert d['rep_start'] == 2 and {r['rep'] for r in d['runs']} == {2}
  assert all(r['run'].startswith('l4_xla_rep2_seed') and not r['save_cache'] for r in d['runs'])
  bad = tmp_path / 'bad.yaml'
  bad.write_text((PLANS / 'msa_go_r2_01.yaml').read_text().replace('warm_rerun_seed: none', 'warm_rerun_seed: 1'))
  with pytest.raises(SystemExit, match='rep_start above 1'):
    describe(bad, 'l4')


def test_main_study_plans_describe_as_before():
  for p in sorted((REPO / 'harness' / 'plans' / 'main').glob('main_*.yaml')):
    platform = run_plan.read_configs(p)['deadline_min']
    d = describe(p, next(iter(platform)))
    assert not {'rep_start', 'launch_by', 'finish_by'} & set(d)
    assert {r['rep'] for r in d['runs'] if r['rep']} <= {1, 2}


def test_every_msa_plan_has_the_arm_settings_the_dates_and_the_official_inputs(repo_with_manifest):
  subset = [r['pdb_id'] for r in csv.DictReader(open(REPO / 'targets' / 'posebusters_subset.csv'))]
  pilot = [r['pdb_id'] for r in csv.DictReader(open(REPO / 'inputs' / 'manifest.csv')) if r['pilot'] == 'yes']
  by_arm = {}
  for p in sorted(PLANS.glob('msa_*.yaml')):
    arm = p.stem.rsplit('_', 1)[0]
    platform = {'msa_go_r1': 'l4', 'msa_go_r2': 'l4', 'msa_v6o': 'v6e', 'msa_v5o': 'v5e'}[arm]
    d = describe(p, platform)
    assert d['label'] == 'msa' and d['weights'] == 'gcs' and d['manifest'] == 'inputs/official_data/manifest.csv'
    assert (d['launch_by'], d['finish_by']) == ('2026-11-02', '2026-11-08')
    assert d['configs'] == [{'l4': 'l4_xla', 'v6e': 'tpu_xla', 'v5e': 'tpu_xla'}[platform]]
    assert d['seeds'] == [1, 2, 3, 4, 5] and d['num_recycles'] is None and d['num_diffusion_samples'] is None
    assert 'buckets' not in d and 'save_embeddings' not in d and d['libtpu'] == []
    warm = d['warm_rerun_seed'] == 1
    assert not warm or (sorted(d['targets']) == sorted(pilot) and arm in ('msa_v5o', 'msa_v6o'))
    by_arm.setdefault(arm, []).extend(d['targets'])
  for arm, ts in by_arm.items():
    assert sorted(ts) == sorted(subset), arm                    # every target once per arm


@pytest.mark.skipif(not (REPO / 'results' / 'parts_msa.txt').is_file(), reason='results/ is outside git: run make_msa_plans.py')
def test_parts_files_follow_the_launch_order():
  lines = [l.split() for l in (REPO / 'results' / 'parts_msa.txt').read_text().splitlines() if l and not l.startswith('#')]
  arms = [pathlib.Path(p).stem.rsplit('_', 1)[0] for _, p in lines]
  order = [a for i, a in enumerate(arms) if i == 0 or arms[i - 1] != a]
  assert order == ['msa_go_r1', 'msa_v6o', 'msa_v5o', 'msa_go_r2']
  for arm in ('go_r1', 'v6o', 'v5o', 'go_r2'):
    per = [l.split() for l in (REPO / 'results' / f'parts_msa_{arm}.txt').read_text().splitlines() if l and not l.startswith('#')]
    assert per == [l for l, a in zip(lines, arms) if a == f'msa_{arm}']


def test_generator_output_is_up_to_date():
  rc = subprocess.run([sys.executable, str(PLANS / 'make_msa_plans.py'), '--check'], capture_output=True, text=True)
  assert rc.returncode == 0, rc.stdout + rc.stderr


def test_settings_match_the_main_study_apart_from_the_input():
  if not (REPO / 'results' / 'analysis' / 'main_20261014' / 'sessions_main.txt').is_file():
    pytest.skip('main-study session records not present')
  rc = subprocess.run([sys.executable, str(PLANS / 'config_diff.py'), '--out', '/dev/null'], capture_output=True, text=True)
  assert rc.returncode == 0 and 'Settings that differ beyond the input: 0' in rc.stdout, rc.stdout[-2000:]


def test_smoke_plans_run_2pv7_on_each_stack_with_the_arm_settings(repo_with_manifest):
  for platform, config in (('v5e', 'tpu_xla'), ('v6e', 'tpu_xla'), ('l4', 'l4_xla')):
    d = describe(REPO / 'harness' / 'plans' / 'msa_smoke' / f'msa_smoke_{platform}.yaml', platform)
    assert d['label'] == 'msa_smoke' and d['targets'] == ['2PV7'] and d['seeds'] == [1] and d['configs'] == [config]
    assert d['processes'] == 1 and d['deadline_min'] == 60


def test_the_one_seed_input_of_a_data_json_keeps_msas_and_templates():
  text = (DATA / 'msa_data_6yqw.json').read_text()
  for seed in (1, 4):
    d = json.loads(featurise_only.run_input_text(text, seed))   # the input harness/run_af3.py writes per process
    src = json.loads(text)
    assert d['modelSeeds'] == [seed]
    d['modelSeeds'] = src['modelSeeds']
    assert d == src                                            # MSAs and inline template mmCIF unchanged


@pytest.mark.skipif(not (REPO / 'third_party' / 'alphafold3' / '.venv' / 'bin' / 'python').is_file(),
                    reason='needs the AlphaFold3 venv')
def test_alphafold3_featurises_the_one_seed_input_with_its_template(tmp_path):
  p = tmp_path / 'input_seeds.json'
  p.write_text(featurise_only.run_input_text((DATA / 'msa_data_6yqw.json').read_text(), 3))
  code = ('import sys; sys.path.insert(0, "harness"); import featurise_only as fo; ra = fo.load_af3(); '
          'from alphafold3.common import folding_input; fi = list(folding_input.load_fold_inputs_from_path(sys.argv[1]))[0]; '
          'b = fo.featurise(ra, fi, fo.featurisation_args(ra)); m = b[0]["template_atom_mask"]; '
          'print("@@", list(fi.rng_seeds), len(b), int((m.reshape(m.shape[0], -1).sum(-1) > 0).sum()))')
  rc = subprocess.run([str(REPO / 'third_party' / 'alphafold3' / '.venv' / 'bin' / 'python'), '-c', code, str(p)],
                      capture_output=True, text=True, cwd=REPO)
  line = next(l for l in rc.stdout.splitlines() if l.startswith('@@'))
  assert line == '@@ [3] 1 1'                              # one seed, as set by the harness; the template featurised
