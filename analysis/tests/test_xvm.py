"""Follow-up 21 (label xvm; notes/analysis_plan.md section 10, row "Follow-up experiment 21"): synthetic tests of
analysis/xvm.py (the reading of X(P), the light identity, the VM-name and stack checks, the main output), of the plans,
and an end-to-end run on relabelled earlier rows. No follow-up 21 output exists or is read."""
import json
import pathlib
import re
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'analysis'))
sys.path.insert(0, str(REPO / 'harness'))
import xvm  # noqa: E402
from run_af3 import read_configs  # noqa: E402

PARTS = REPO / 'harness' / 'plans' / 'xvm_parts.csv'
ALL = [f'T{i:02d}' for i in range(60)]


class Args:
  seed, replicates = 2026, 400


def seed_pairs(name, rates, targets, per=5, seed=0):
  rng = np.random.default_rng(seed)
  return pd.DataFrame([{'contrast': name, 'target': t, 'seed_a': s + 1, 'seed_b': s + 1,
                        **{st: bool(rng.random() < r) for st, r in rates.items()}}
                       for t in targets for s in range(per)])


RATES = lambda x: {'flip_top1': x, 'flip_success': x / 4, 'flip_matters': x / 6}  # noqa: E731


# ---------------------------------------------------------------- design and reading

def test_conditions_are_the_specification():
  assert xvm.CONFIG == {'l4_xla': 'G', 'a100_xla': 'A'} and xvm.SEEDS == [1, 2, 3, 4, 5]
  assert xvm.PART == {'G': 'A', 'A': 'B'}
  assert [s[0] for s in xvm.rc_specs('G')] == ['RC_same(G)', 'RC_new(G)', 'RC_new2(G)']
  assert xvm.rc_specs('G')[1][2:4] == (('G', 'r1'), ('G', 'r3')) and xvm.rc_specs('G')[2][2:4] == (('G', 'r2'), ('G', 'r3'))


def test_x_reading_excludes_zero_with_its_sign():
  assert xvm.x_reading(0.01, 0.08) == 'a new VM changes the recompilation rate (higher on a new VM)'
  assert xvm.x_reading(-0.08, -0.01) == 'a new VM changes the recompilation rate (lower on a new VM)'
  assert xvm.x_reading(-0.03, 0.04) == 'not distinguishable'
  assert xvm.x_reading(0.0, 0.04) == 'not distinguishable'                     # an interval touching 0 does not exclude it
  assert xvm.x_reading(float('nan'), 0.1) == 'not computed'


def test_x_table_and_the_table_ii_sentence():
  sp = pd.concat([seed_pairs('RC_same(G)', RATES(0.07), ALL, seed=1), seed_pairs('RC_new(G)', RATES(0.35), ALL, seed=2),
                  seed_pairs('RC_new2(G)', RATES(0.35), ALL, seed=3)], ignore_index=True)
  x = xvm.x_table(sp, ['G'], Args)
  top = x[x.statistic == 'flip_top1'].iloc[0]
  assert top.role == 'primary' and top.reading.endswith('(higher on a new VM)') and top.n_targets == 60
  assert set(x[x.statistic != 'flip_top1'].role) == {'secondary'}
  assert 'Table II' in xvm.table_ii_sentence(x) and 'reports RC_new(G)' in xvm.table_ii_sentence(x)
  sp = pd.concat([seed_pairs('RC_same(G)', RATES(0.07), ALL, seed=1), seed_pairs('RC_new(G)', RATES(0.07), ALL, seed=1)])
  x = xvm.x_table(sp, ['G'], Args)
  assert x[x.statistic == 'flip_top1'].reading.iloc[0] == 'not distinguishable'
  assert x[x.statistic == 'flip_top1'].estimate.iloc[0] == 0
  assert 'nothing is added' in xvm.table_ii_sentence(x)


# ---------------------------------------------------------------- identity, VM names, stack, main output

def write_outputs(root, t, seed, coords='1.000', stamp='2026-12-01'):
  d = root / t
  for k in range(5):
    s = d / f'seed-{seed}_sample-{k}'
    s.mkdir(parents=True, exist_ok=True)
    (s / f'{t}_seed-{seed}_sample-{k}_model.cif').write_text(
        f'data_x\n_ma_model_list.model_group_name "AF3 {stamp}"\nATOM {coords}\n')
    (s / f'{t}_seed-{seed}_sample-{k}_summary_confidences.json').write_text('{"iptm": 0.9}')
  (d / f'{t}_model.cif').write_text(f'data_x\n_ma_model_list.model_group_name "AF3 {stamp}"\nATOM {coords}\n')
  (d / f'{t}_ranking_scores.csv').write_text('seed,sample,ranking_score\n')
  (d / f'{t}_summary_confidences.json').write_text('{"iptm": 0.9}')
  return d


def sample_rows(root, cond, rep, t, seed, coords='1.000', scores=(0.9, 0.8, 0.7, 0.6, 0.5), success=None, stamp='x'):
  d = write_outputs(root / f'{cond}_{rep}', t, seed, coords, stamp)
  succ = success or [True] * 5
  return [{'condition': cond, 'repetition': rep, 'target': t, 'seed': seed, 'sample': k, 'ranking_score': scores[k],
           'm1_success': succ[k],
           'model_path': str(d / f'seed-{seed}_sample-{k}' / f'{t}_seed-{seed}_sample-{k}_model.cif')} for k in range(5)]


def test_light_identity_reads_identical_and_differ(tmp_path):
  rows = sample_rows(tmp_path, 'G', 'r1', 'T1', 1, stamp='2026-10-12') + sample_rows(tmp_path, 'G', 'r3', 'T1', 1) + \
      sample_rows(tmp_path, 'G', 'r1', 'T2', 1) + sample_rows(tmp_path, 'G', 'r3', 'T2', 1, coords='1.001')
  ok = pd.DataFrame(rows)
  s, det = xvm.light_identity(ok, ('G', 'r1'), ('G', 'r3'), 'x')
  assert s['identical'] == 1 and s['units'] == 2 and s['detail'].startswith('not bit-identical (1 of 2')
  ok = pd.DataFrame(rows[:10])
  s, _ = xvm.light_identity(ok, ('G', 'r1'), ('G', 'r3'), 'x')
  assert s['detail'] == 'recompilation bit-identical'
  s, _ = xvm.light_identity(ok, ('A', 'r1'), ('A', 'r3'), 'x')
  assert s['detail'] == 'not run'


def fake_session(af3, name, vm=None, launcher=False, image=None, python='3.12.16'):
  s = (af3 / '.launcher' / name) if launcher else (af3 / name)
  s.mkdir(parents=True)
  (s / 'launch_record.txt').write_text(f'SESSION={name}\nVM_NAME={vm or "af3-run-" + name.lower()}\n')
  if not launcher:
    run = s / 'l4_xla_rep3_seed1_fresh'
    run.mkdir()
    (run / 'session.json').write_text(json.dumps({'host': {'platform': 'Linux-6.8'}, 'probe': {'python': python},
                                                  'config': {'env': {'XLA_FLAGS': '--a'}}, 'xla_env': {'XLA_FLAGS': '--a'}}))
    (s / 'device_check.txt').write_text('Driver Version: 580.178.04 \n')
    (s / 'setup.json').write_text(json.dumps({'packages': {'alphafold3': '3.0.5.dev0+g85c4d2050.d20261201'}}))
    (s / 'pip_freeze.txt').write_text('a==1\n')
  if image:
    log = af3 / '.many' / name[:16] / 'part.log'
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(f'   image {image}\n   session: results/af3/{name}/   VM: x\n')
  return s


def test_vm_names_and_images_from_the_records(tmp_path):
  af3 = tmp_path / 'af3'
  fake_session(af3, '20261012T214519Z_l4_main_g_04', vm='af3-run-l4-20261012t214519z',
               image='ubuntu-os-accelerator-images/ubuntu-accelerator-2204-amd64-with-nvidia-580')
  fake_session(af3, '20261201T100000Z_l4_xvm_g_r3_01', launcher=True)
  assert xvm.vm_name_of('20261012T214519Z_l4_main_g_04', af3) == 'af3-run-l4-20261012t214519z'
  assert xvm.vm_name_of('20261201T100000Z_l4_xvm_g_r3_01', af3) == 'af3-run-20261201t100000z_l4_xvm_g_r3_01'
  assert xvm.vm_name_of('nothing', af3) is None
  assert xvm.image_of('20261012T214519Z_l4_main_g_04', af3).endswith('with-nvidia-580')
  assert xvm.image_of('20261201T100000Z_l4_xvm_g_r3_01', af3) is None


def test_check_vms_finds_a_reused_vm_and_an_unknown_one(tmp_path, monkeypatch):
  af3 = tmp_path / 'af3'
  fake_session(af3, '20261012T214519Z_l4_main_g_04', vm='vm-a')
  fake_session(af3, '20261201T100000Z_l4_xvm_g_r3_01', vm='vm-b')
  fake_session(af3, '20261201T110000Z_l4_xvm_g_r3_02', vm='vm-a')
  monkeypatch.setattr(xvm, 'AF3', af3)
  r12 = pd.DataFrame([{'condition': 'G', 'target': t, 'session': 'results/af3/20261012T214519Z_l4_main_g_04/l4_xla_rep1_seed1_fresh'}
                      for t in ('T1', 'T2', 'T3')])
  d = pd.DataFrame([{'condition': 'G', 'target': 'T1', 'session': 'results/af3/20261201T100000Z_l4_xvm_g_r3_01/r'},
                    {'condition': 'G', 'target': 'T2', 'session': 'results/af3/20261201T110000Z_l4_xvm_g_r3_02/r'},
                    {'condition': 'G', 'target': 'T3', 'session': 'results/af3/20261201T120000Z_l4_xvm_g_r3_03/r'}])
  rows, table = xvm.check_vms(d, r12, af3)
  assert rows[0]['value'] == '1/3 targets'
  assert 'T2 (same VM vm-a)' in rows[0]['detail'] and 'T3 (VM name unknown)' in rows[0]['detail']
  assert table.set_index('target').differs.to_dict() == {'T1': True, 'T2': False, 'T3': False}


def test_check_stack_lists_the_python_and_image_differences(tmp_path):
  af3 = tmp_path / 'af3'
  fake_session(af3, '20261012T214519Z_l4_main_g_04', python='3.12.15', image='ubuntu/fam-580')
  fake_session(af3, '20261201T100000Z_l4_xvm_g_r3_01', image='af2-tpu-benchmark/af3-accel-2204-nvidia-580-gtr1')
  rows, table = xvm.check_stack({'G': ['20261201T100000Z_l4_xvm_g_r3_01']}, {'G': ['20261012T214519Z_l4_main_g_04']},
                                '4b', af3)
  assert rows[0]['detail'].startswith('reported, differs in image:') and 'python: 3.12.16' in rows[0]['detail']
  assert set(table[~table.same].field) == {'image', 'python'}
  rows, _ = xvm.check_stack({'G': ['20261012T214519Z_l4_main_g_04']}, {'G': ['20261012T214519Z_l4_main_g_04']}, '4b', af3)
  assert rows[0]['detail'] == 'pass'


def test_main_output_changes_counts_targets(tmp_path):
  rows = []
  for rep, scores in (('r1', (0.9, 0.8, 0.7, 0.6, 0.5)), ('r2', (0.9, 0.8, 0.7, 0.6, 0.5)), ('r3', (0.5, 0.95, 0.7, 0.6, 0.5))):
    for s in (1, 2):
      rows += sample_rows(tmp_path, 'G', rep, 'T1', s, scores=scores if s == 1 else (0.1,) * 5,
                          success=[True, False, True, True, True])
      rows += sample_rows(tmp_path, 'G', rep, 'T2', s)
  ok = pd.DataFrame(rows)
  mo = xvm.main_output_changes(ok, 'G', [('r1', 'r2', 'same VM'), ('r1', 'r3', 'new VM')]).set_index('vm')
  assert mo.loc['same VM', 'main_output_differs'] == 0 and mo.loc['new VM', 'main_output_differs'] == 1
  assert mo.loc['new VM', 'targets_differ'] == 'T1' and mo.loc['new VM', 'success_differs'] == 1
  assert mo.loc['new VM', 'targets'] == 2


def test_completeness_part_b_optional():
  ok = pd.DataFrame([{'condition': 'G', 'repetition': 'r3', 'seed': s, 'target': t, 'sample': k}
                     for s in xvm.SEEDS for t in ALL for k in range(5)])
  comp = xvm.completeness(ok, ALL)
  assert comp[comp.condition == 'G'].status.eq('complete').all()
  assert comp[comp.condition == 'A'].status.eq('not run').all()


# ---------------------------------------------------------------- plans

def test_plans_are_up_to_date_and_match_the_specification():
  out = subprocess.run([sys.executable, str(REPO / 'harness' / 'plans' / 'make_prep_plans.py'), '--label', 'xvm', '--check'],
                       capture_output=True, text=True)
  assert out.returncode == 0, out.stdout + out.stderr
  parts = pd.read_csv(PARTS)
  assert parts.groupby('arm').processes.sum().to_dict() == {'A_r3': 300, 'G_r3': 300}
  assert parts.groupby('arm').targets.sum().to_dict() == {'A_r3': 60, 'G_r3': 60}
  assert parts.groupby('arm').size().to_dict() == {'A_r3': 9, 'G_r3': 12}
  assert parts.groupby('arm').row_part.first().to_dict() == {'A_r3': 'B', 'G_r3': 'A'}
  assert set(parts.provisioning) == {'spot'}
  configs = read_configs(REPO / 'harness' / 'configs.yaml')
  seen = {'g': set(), 'a': set()}
  for p in sorted((REPO / 'harness' / 'plans' / 'xvm').glob('*.yaml')):
    t = p.read_text()
    assert 'label: xvm\n' in t and 'rep_start: 3\n' in t and 'fresh_reps: 1\n' in t and 'eigen_source: archive\n' in t, p
    cfg = re.search(r'configs:\n  \w+: (\S+)', t).group(1)
    assert cfg in configs and cfg in xvm.CONFIG, p
    seen[p.name.split('_')[1]] |= set(re.search(r'default: \[([^\]]*)\]', t).group(1).split(', '))
  manifest = pd.read_csv(REPO / 'inputs' / 'manifest.csv')
  assert seen['g'] == seen['a'] == set(manifest.pdb_id) and len(seen['g']) == 60


# ---------------------------------------------------------------- end to end, on relabelled earlier rows

PRIMARY = REPO / 'results' / 'analysis' / 'main_20261014' / 'primary_v2'
A100 = REPO / 'results' / 'analysis' / 'a100_20261018'


@pytest.mark.skipif(not all((p / 'samples.csv').is_file() for p in (PRIMARY, A100)), reason='earlier analysis outputs not here')
def test_main_runs_end_to_end_on_relabelled_rows(tmp_path):
  """xvm.main on the r2 rows of G (main study) and A (follow-up 3) relabelled as follow-up 21's r3, four targets: every
  output is written, RC_new equals RC_same (X = 0, "not distinguishable"), RC_new2 is 0 and r3 is light-identical to
  r2, check 6 reproduces the registered RC, and check 5 reports the reused VMs (r3 is r2's own session)."""
  targets = ['7U3J', '7NP6', '7BTT', '8G6P']
  rows = []
  for path, label, cond, config in ((PRIMARY, 'main', 'G', 'l4_xla'), (A100, 'a100', 'A', 'a100_xla')):
    m = pd.read_csv(path / 'samples.csv', low_memory=False)
    g = m[(m.label == label) & (m.platform == cond) & (m.repetition == 'r2') & m.target.isin(targets) & m['sample'].notna()].copy()
    g['label'], g['config'], g['repetition'] = 'xvm', config, 'r3'
    rows.append(g)
  d = pd.concat(rows, ignore_index=True)
  d.to_csv(tmp_path / 'samples.csv', index=False)
  out = tmp_path / 'out'
  r = subprocess.run([sys.executable, str(REPO / 'analysis' / 'xvm.py'), '--samples', str(tmp_path / 'samples.csv'),
                      '--out', str(out), '--replicates', '200'], capture_output=True, text=True, cwd=REPO)
  assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-3000:]
  for name in ('checks', 'completeness', 'contrasts', 'identity', 'x', 'main_output', 'stack', 'vm_names', 'pairs_seed'):
    assert (out / f'{name}.csv').is_file(), name
  chk = pd.read_csv(out / 'checks.csv')
  assert (chk[chk.check.str.startswith(('1 ', '2'))].detail == 'pass').all()
  assert (chk[chk.check.str.startswith('6 ')].detail == 'pass').all()
  assert not (chk[chk.check.str.startswith('5 ')].detail == 'pass').any()       # r3 here is r2's own VM
  x = pd.read_csv(out / 'x.csv')
  top = x[x.statistic == 'flip_top1']
  assert set(top.condition) == {'G', 'A'}
  con = pd.read_csv(out / 'contrasts.csv')
  t = con[con.statistic == 'flip_top1'].set_index('contrast').estimate
  assert t['RC_new2(G)'] == 0 and t['RC_new2(A)'] == 0
  ident = pd.read_csv(out / 'identity.csv')
  assert (ident[ident.condition.str.contains('r3 against') & ident.condition.str.endswith('r2')].detail
          == 'recompilation bit-identical').all()
  mo = pd.read_csv(out / 'main_output.csv')
  assert (mo[mo.sides.str.endswith('r2 against G r3') | mo.sides.str.endswith('r2 against A r3')].main_output_differs == 0).all()
  assert '## X(P) = RC_new(P) minus RC_same(P)' in (out / 'report.md').read_text()
