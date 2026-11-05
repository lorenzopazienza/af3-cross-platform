"""Follow-up 10, larger inputs on the whole PDB (label large; notes/analysis_plan.md section 10, row of 2026-11-04): the
targets as targets/large/build_large_subset.py writes them, the M1 reference, the plans as
harness/plans/make_fu10_plans.py writes them, the settings diff, the budget rule, and the analysis on synthetic
per-sample tables. No real AlphaFold3 output is read."""
import csv
import hashlib
import json
import os
import pathlib
import subprocess
import sys
import types

import pandas as pd
import pytest

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'harness'))
sys.path.insert(0, str(REPO / 'analysis' / 'tests'))
import run_plan  # noqa: E402

import large  # noqa: E402
import score_ligand  # noqa: E402
from test_fu7_10 import no_files, rows, write  # noqa: E402,F401  (no_files: autouse, no model file is read)

PLANS = REPO / 'harness' / 'plans'
SUBSET = REPO / 'targets' / 'large' / 'large_subset.csv'
ROW_SUBSET = ('7DIA 7QUR 8C8V 8SX7 8TZG 8V1G 8XB3 9BWT 9GZ2 9X5O 7UMP 8BLP 8CEY 9ETQ 9HKX 9OHN 9OMM 9P4Y 9ZB9 9ZNO').split()
HAVE_DATA = (REPO / 'data' / 'large_refs').is_dir() and (REPO / 'data' / 'pdb_large_screen' / 'cif').is_dir()


# ---- targets

def test_large_subset_is_the_registered_selection_in_the_main_columns():
  sub = list(csv.DictReader(open(SUBSET)))
  screen = [r for r in csv.DictReader(l for l in open(REPO / 'targets' / 'large' / 'pdb_targets_amended.csv')
                                      if not l.startswith('#')) if r['group'] in ('large', 'control')]
  main = list(csv.DictReader(open(REPO / 'targets' / 'posebusters_subset.csv')))
  repl = list(csv.DictReader(open(REPO / 'targets' / 'repl' / 'repl_subset.csv')))
  assert list(sub[0])[:len(main[0])] == list(main[0])                       # the main subset's columns first
  assert [r['group'] for r in sub] == ['large'] * 34 + ['control'] * 34
  assert sorted(r['pdb_id'] for r in sub) == sorted(r['pdb_id'] for r in screen)
  assert not {r['pdb_id'] for r in sub} & ({r['pdb_id'] for r in main} | {r['pdb_id'] for r in repl})
  by = {r['pdb_id']: r for r in screen}
  for r in sub:
    s = by[r['pdb_id']]
    assert (r['ligand_ccd'], r['num_tokens'], r['bucket'], r['sequence'], r['ligand_asym']) == \
        (s['ligand_ccd'], s['num_tokens'], s['bucket'], s['sequence'], s['ligand_asym'])
    assert r['sequence_sha256'] == hashlib.sha256(r['sequence'].encode()).hexdigest()
    assert r['pilot'] == 'no' and r['n_input_chains'] == '2' and r['protein_label_asym'] == s['pocket_chains']
  assert sorted(r['pdb_id'] for r in sub if r['tpu_r2'] == 'yes') == sorted(ROW_SUBSET)
  src = json.loads((REPO / 'targets' / 'large' / 'sources.json').read_text())
  assert src['output']['sha256'] == hashlib.sha256(SUBSET.read_bytes()).hexdigest()
  assert src['inputs_sha256']['targets/large/pdb_targets_amended.csv'] == \
      '7536e8995f822e42236c5e3ea3380264131828814c1ad6b2a28831fe9e6e44f4'
  assert src['counts']['large']['variant_A'] == 17 and src['counts']['control']['variant_A'] == 11


def test_input_builder_reads_the_columns_the_large_subset_has():
  text = (REPO / 'inputs' / 'build_inputs.py').read_text()
  for col in ('pdb_id', 'sequence', 'sequence_sha256', 'ligand_ccd', 'num_tokens', 'pilot'):
    assert f"t['{col}']" in text or f't["{col}"]' in text
  assert "SUBSET = REPO / 'targets' / 'posebusters_subset.csv'" in text and "INPUT_DIR = REPO / 'data' / 'inputs'" in text


# ---- M1 reference

def test_score_ligand_finds_the_large_targets_after_the_60_and_the_72():
  row = score_ligand._subset_row('8EXL')
  assert score_ligand._is_large(row) and row['ligand_asym']
  assert score_ligand._crystal_path('8EXL', row) == REPO / 'data' / 'pdb_large_screen' / 'cif' / '8EXL.cif.gz'
  main = next(csv.DictReader(open(REPO / 'targets' / 'posebusters_subset.csv')))
  assert not score_ligand._is_large(score_ligand._subset_row(main['pdb_id']))
  assert score_ligand._crystal_path(main['pdb_id'], main) == REPO / 'data' / 'rcsb' / f'{main["pdb_id"]}.cif'


@pytest.mark.skipif(not HAVE_DATA, reason='data/large_refs or the screen cache is absent')
def test_large_reference_is_the_deposited_instance_and_self_scores():
  res = score_ligand.self_test('8EXL')
  assert res['success'] and res['ligand_rmsd'] == 0.0 and res['error'] is None
  ref = score_ligand.reference('9FTR')                                       # two CCD atoms not modelled
  assert len(ref.ligand_names) == 18
  bad = score_ligand.self_test('9DRA')            # ligand only in altloc B against protein conformation A
  assert not bad['success'] and bad['ligand_rmsd'] == 0.0 and 'minimum_distance_to_protein' in bad['pb_failed']


# ---- plans

def describe(path, platform):
  return run_plan.expand(path, platform, REPO / 'inputs' / 'manifest.csv')


@pytest.fixture
def large_manifest(tmp_path, monkeypatch):
  """inputs/large/manifest.csv as build_inputs.py will write it, in a temporary repository root."""
  if (REPO / 'inputs' / 'large' / 'manifest.csv').is_file():
    return
  (tmp_path / 'inputs' / 'large').mkdir(parents=True)
  pd.read_csv(SUBSET, dtype=str)[['pdb_id', 'pilot', 'num_tokens', 'bucket']].to_csv(
      tmp_path / 'inputs' / 'large' / 'manifest.csv', index=False)
  monkeypatch.setattr(run_plan, 'REPO', tmp_path)


def test_generator_output_is_up_to_date():
  rc = subprocess.run([sys.executable, str(PLANS / 'make_fu10_plans.py'), '--check'], capture_output=True, text=True)
  assert rc.returncode == 0, rc.stdout + rc.stderr


def test_arms_cover_the_68_targets_with_r2_where_the_row_puts_it(large_manifest):
  want = sorted(pd.read_csv(SUBSET, dtype=str).pdb_id)
  for arm, platform, config in (('v5l', 'v5e', 'tpu_xla'), ('v6l', 'v6e', 'tpu_xla'), ('gl', 'l4', 'l4_xla')):
    ts, r2 = [], []
    for p in sorted((PLANS / 'large').glob(f'large_{arm}_*.yaml')):
      d = describe(p, platform)
      assert d['label'] == 'large' and d['configs'] == [config] and d['seeds'] == [1, 2, 3, 4, 5]
      assert d['warm_rerun_seed'] is None and d['manifest'] == 'inputs/large/manifest.csv'
      assert d['eigen_source'] == 'archive' and d['finish_by'] == '2026-11-14' and 'launch_by' not in d
      assert d['run_timeout_min'] == 180 and not d.get('seeds_beyond_input')
      ts += d['targets']
      if d['fresh_reps'] == 2:
        r2 += d['targets']
    assert sorted(ts) == want, arm                                          # every target once
    assert sorted(r2) == (want if arm == 'gl' else sorted(ROW_SUBSET)), arm


def test_parts_respect_size_memory_and_timeout():
  parts = list(csv.DictReader(open(PLANS / 'fu10_parts.csv')))
  assert {p['provisioning'] for p in parts if p['platform'] == 'v5e'} == {'standard'}
  assert {p['provisioning'] for p in parts if p['platform'] == 'v6e'} == {'flex_start'}
  assert {p['provisioning'] for p in parts if p['platform'] == 'l4'} == {'spot'}
  assert max(int(p['longest_process_s']) for p in parts) < 180 * 60
  big = [p for p in parts if p['platform'] == 'l4' and float(p['expected_vm_h']) > 4.5]
  assert big and all(p['targets'] == '1' for p in big)                     # a 1,536-token L4 target alone in its part
  total = sum(float(p['expected_usd_planned']) for p in parts)
  assert 184 < total < 186 and total * 1.2 < 223                            # the row: 184.47 expected, cap 221


def test_config_diff_shows_only_the_intended_differences():
  r = subprocess.run([sys.executable, str(PLANS / 'config_diff_fu10.py'), '--out', os.devnull], capture_output=True,
                     text=True)
  assert r.returncode == 0, r.stdout[-2000:] + r.stderr
  assert 'Settings that differ beyond the intended ones: 0' in r.stdout
  assert '| input manifest | `inputs/manifest.csv` | `inputs/large/manifest.csv` | intended |' in r.stdout


# ---- budget rule

def credit(tmp_path, *args, now='2026-11-08T00:00:00Z'):
  for f in ('harness/plans/fu10_parts.csv', 'harness/plans/fu11_12_parts.csv', 'harness/plans/fu7_10_parts.csv',
            'cloud/prices.csv'):
    (tmp_path / f).parent.mkdir(parents=True, exist_ok=True)
    (tmp_path / f).write_text((REPO / f).read_text())
  r = subprocess.run(['bash', str(REPO / 'scripts' / 'monitor' / 'credit_check_fu10.sh'), *args], cwd=tmp_path,
                     capture_output=True, text=True, env=dict(os.environ, NOW=now))
  return r.returncode, r.stdout + r.stderr


def expected(arm, parts='fu10_parts.csv'):
  return sum(float(r['expected_usd_planned']) for r in csv.DictReader(open(PLANS / parts))
             if r['arm'].lower().replace(' ', '_') == arm)


def session(tmp_path, name, events, prov='spot', where='results/af3/.launcher'):
  s = tmp_path / where / name
  s.mkdir(parents=True, exist_ok=True)
  (s / 'launcher_events.txt').write_text(events)
  (s / 'launch_record.txt').write_text(f'PROVISIONING={prov}\n')


RUNNING = '2026-11-07T20:00:00Z  launch: session x\n2026-11-07T20:01:00Z  vm: af3-run-x running in us-central1-a\n'


def test_credit_check_subtracts_running_arms_of_every_follow_up(tmp_path):
  code, out = credit(tmp_path, 'v5l', '800')
  assert code == 0 and f'arm v5l (v5e standard): expected cost {expected("v5l"):.2f} USD' in out
  session(tmp_path, '20261107T200000Z_v6e_prec5_v6h5_01', RUNNING, prov='flex_start')
  session(tmp_path, '20261107T200000Z_l4_large_gl_01', RUNNING)
  code, out = credit(tmp_path, 'v5l', '800')
  left = 800 - expected('v6h5', 'fu11_12_parts.csv') - expected('gl') - expected('v5l')
  assert 'running arm v6h5' in out and 'running arm gl' in out
  assert f'balance minus running arms minus expected {left:.2f} USD' in out and code == 0
  code, out = credit(tmp_path, 'v5l', f'{800 - left + 24:.2f}')
  assert code == 1 and '< 25 USD' in out


def test_credit_check_cap_and_dates(tmp_path):
  session(tmp_path, '20261108T000000Z_l4_large_gl_02', '2026-11-08T00:00:00Z  vm: x\n'
          '2026-11-08T10:00:00Z  delete: done\n2026-11-08T10:00:01Z  launcher: exit code 0\n', where='results/af3')
  spot = next(float(r['usd_per_h_planned']) for r in csv.DictReader(open(PLANS / 'fu10_parts.csv')) if r['platform'] == 'l4')
  code, out = credit(tmp_path, 'v6l', '800')
  assert code == 0 and f'from 1 session record(s): {10 * spot:.2f} USD' in out and '(cap: 221)' in out
  assert credit(tmp_path, 'v6l', '800', now='2026-11-11T23:00:00Z')[0] == 0             # launch on 2026-11-11: allowed
  code, out = credit(tmp_path, 'v6l', '800', now='2026-11-12T00:00:00Z')
  assert code == 1 and 'launch after 2026-11-11' in out
  code, out = credit(tmp_path, 'gl', '800', now='2026-11-13T12:00:00Z')                # 21.6 h: ends 2026-11-14
  assert code == 0
  assert credit(tmp_path, 'gl', '800', now='2026-11-13T23:59:00Z')[0] == 0             # last launch day: ends in time
  code, out = credit(tmp_path, 'gl', '800', now='2026-11-14T10:00:00Z')
  assert code == 1 and 'launch after 2026-11-13' in out and 'cannot finish by 2026-11-14' in out
  assert credit(tmp_path, 'v5x', '1')[0] == 2 and credit(tmp_path, 'gl', 'x')[0] == 2


# ---- analysis

ARGS = types.SimpleNamespace(seed=2026, replicates=40)
LARGE_T, CONTROL_T = ['L1', 'L2', 'L3', 'L4'], ['C1', 'C2', 'C3', 'C4']
SUB = {'L1', 'L2', 'C1', 'C2'}


def world(tmp_path, monkeypatch, tpu_r2_coords='x', r2_targets=None, extra_target=None, others=()):
  info = [{'pdb_id': t, 'group': 'large' if t in LARGE_T else 'control', 'variant_A': 'yes' if t in ('L1', 'L3', 'C1', 'C3') else 'no',
           'method': 'X-ray' if t in ('L1', 'L2', 'C1', 'C2') else 'EM', 'num_tokens': 1200 + 50 * i if t in LARGE_T else 300 + 100 * i,
           'tpu_r2': 'yes' if t in SUB else 'no'} for i, t in enumerate(LARGE_T + CONTROL_T)]
  write(info, tmp_path / 'targets.csv')
  write([{'pdb_id': t} for t in LARGE_T + CONTROL_T], tmp_path / 'manifest.csv')
  write([{'pdb_id': t} for t in others] or [{'pdb_id': 'ZZZZ'}], tmp_path / 'others.csv')
  r2 = sorted(SUB) if r2_targets is None else r2_targets
  ts = LARGE_T + CONTROL_T + ([extra_target] if extra_target else [])
  lt, ct = [t for t in ts if t not in CONTROL_T], CONTROL_T
  new = []
  # V6 flips against V5 on the large targets only; G flips against V6 on the control targets only.
  for rep in ('r1', 'r2'):
    keep = (lambda t: True) if rep == 'r1' else (lambda t: t in r2)
    coords = 'x' if rep == 'r1' else tpu_r2_coords
    new += rows('V5', 'large', rep, [t for t in ts if keep(t)], range(1, 6), coords=coords)
    new += rows('V6', 'large', rep, [t for t in lt if keep(t)], range(1, 6), shift=0.2, coords=coords)
    new += rows('V6', 'large', rep, [t for t in ct if keep(t)], range(1, 6), coords=coords)
    new += rows('G', 'large', rep, lt, range(1, 6), shift=0.2) + rows('G', 'large', rep, ct, range(1, 6), shift=0.2)
  out = tmp_path / 'out'
  monkeypatch.setattr(sys, 'argv', ['large.py', '--samples', str(write(new, tmp_path / 'samples.csv')),
                                    '--manifest', str(tmp_path / 'manifest.csv'), '--targets', str(tmp_path / 'targets.csv'),
                                    '--other_targets', str(tmp_path / 'others.csv'), '--out', str(out),
                                    '--replicates', '40'])
  return out


def test_headline_reading_rule_and_two_sample_rc(tmp_path, monkeypatch):
  out = world(tmp_path, monkeypatch)
  assert large.main() == 0
  rule = pd.read_csv(out / 'reading_rule.csv').set_index('pair')
  assert rule.loc['HW(V5,V6)', 'answer'] == 'the hardware effect grows with input size'     # +100 points
  assert rule.loc['HW(V6,G)', 'answer'] == 'the hardware effect shrinks with input size'    # -100 points
  assert rule.loc['HW(V5,V6)', 'reference'] == 'RC(V5)' and rule.loc['HW(V6,G)', 'reference'] == 'RC(V6)'
  hw = pd.read_csv(out / 'hw_minus_rc.csv')
  x = hw[(hw.targets == 'all') & (hw.hardware == 'HW(V5,V6)') & (hw.recompilation == 'RC(V5)')
         & (hw.statistic == 'flip_top1') & (hw.group == 'large')].iloc[0]
  assert (x.n_targets_hw, x.n_targets_rc, x.resampling) == (4, 2, 'independent')           # RC on the r2 subset
  y = hw[(hw.targets == 'all') & (hw.hardware == 'HW(V6,G)') & (hw.recompilation == 'RC(G)')
         & (hw.statistic == 'flip_top1') & (hw.group == 'control')].iloc[0]
  assert (y.n_targets_rc, y.resampling) == (4, 'paired')                                   # RC(G) on every target
  head = pd.read_csv(out / 'headline.csv')
  h = head[(head.targets == 'all') & (head.hardware == 'HW(V5,V6)') & (head.reference == 'RC(P)')
           & (head.statistic == 'flip_top1')].iloc[0]
  assert h.role == 'headline' and h.estimate == 1.0
  assert {'all', 'variant A', 'X-ray'} == set(head.targets)
  assert set(head[head.targets == 'X-ray'].statistic) == {'flip_success', 'flip_matters'}   # M1 only on X-ray
  assert (head[head.statistic != 'flip_top1'].m1_em_targets.astype(str) != '').all()
  report = (out / 'report.md').read_text()
  assert 'the extension rule of row item c does not apply' in report and 'EXTENSION RULE' not in report
  assert (pd.read_csv(out / 'completeness.csv').status == 'complete').all()
  logi = pd.read_csv(out / 'logistic.csv')
  assert set(logi.model) == {'tokens', 'tokens + method'} and set(logi.pair) == {'HW(V5,V6)', 'HW(V6,G)'}


def test_tpu_r2_not_bit_identical_puts_the_extension_rule_on_top(tmp_path, monkeypatch):
  out = world(tmp_path, monkeypatch, tpu_r2_coords='y')
  assert large.main() == 0
  report = (out / 'report.md').read_text()
  assert report.split('\n')[2].startswith('**EXTENSION RULE TRIGGERED') and 'V5: C1, C2, L1, L2' in report
  bit = pd.read_csv(out / 'bit_identity.csv')
  assert not bit.all_identical.any() and len(bit) == 8


def test_extended_r2_is_paired_and_the_checks_stop_bad_inputs(tmp_path, monkeypatch):
  out = world(tmp_path, monkeypatch, r2_targets=LARGE_T + CONTROL_T)                       # extension applied
  assert large.main() == 0
  hw = pd.read_csv(out / 'hw_minus_rc.csv')
  assert set(hw[hw.recompilation == 'RC(V5)'].resampling) == {'paired'}
  world(tmp_path, monkeypatch, r2_targets=['L1'])                                          # subset not covered
  with pytest.raises(SystemExit, match='a check failed'):
    large.main()
  world(tmp_path, monkeypatch, extra_target='M1', others=['M1'])                           # a main-study target
  with pytest.raises(SystemExit, match='a check failed'):
    large.main()


def test_logistic_fit_recovers_a_known_slope():
  import numpy as np
  rng = np.random.default_rng(0)
  x = rng.uniform(0, 6, 4000)
  y = (rng.random(4000) < 1 / (1 + np.exp(-(-2 + 0.5 * x)))).astype(float)
  b = large.logit_fit(np.column_stack([np.ones_like(x), x]), y)
  assert abs(b[1] - 0.5) < 0.1 and abs(b[0] + 2) < 0.3
  assert large.logit_fit(np.column_stack([np.ones(4), [0, 1, 2, 3]]), np.array([0, 0, 1, 1.])) is None   # separation
