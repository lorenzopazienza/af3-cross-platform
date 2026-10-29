"""Follow-ups 7 (sw), 8 (seeds) and 9 (h100) (drafts/section10_fu7-10.md): the harness keys (seeds_beyond_input,
stack, platform h100), the plans as harness/plans/make_fu7_10_plans.py writes them, the condition codes, and the
three analyses on synthetic per-sample tables. No real output is read."""
import csv
import json
import pathlib
import subprocess
import sys
import types

import numpy as np
import pandas as pd
import pytest

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'harness'))
import run_plan  # noqa: E402

import collect  # noqa: E402
import contrasts  # noqa: E402
import cost  # noqa: E402
import followup34 as f34  # noqa: E402
import h100  # noqa: E402
import seeds  # noqa: E402
import sw  # noqa: E402

PLANS = REPO / 'harness' / 'plans'
ARGS = types.SimpleNamespace(seed=2026, replicates=40)


def describe(path, platform):
  return run_plan.expand(path, platform, REPO / 'inputs' / 'manifest.csv')


# ---- harness and plans

def test_seed_plans_pass_seeds_beyond_input_and_run_seeds_6_to_20():
  d = describe(PLANS / 'seeds' / 'seeds_v5x_01.yaml', 'v5e')
  assert d['seeds_beyond_input'] is True and d['seeds'] == list(range(6, 21)) and d['label'] == 'seeds'
  assert {r['seed'] for r in d['runs']} == set(range(6, 21)) and {r['rep'] for r in d['runs']} == {1}


def test_stack_plans_pin_the_whole_family(tmp_path):
  tpu = describe(PLANS / 'sw' / 'sw_v6s_01.yaml', 'v6e')['stack']
  gpu = describe(PLANS / 'sw' / 'sw_gs_01.yaml', 'l4')['stack']
  assert tpu == {'jax': '0.10.1', 'jaxlib': '0.10.1', 'libtpu': '0.0.41'}
  assert gpu == {'jax': '0.10.1', 'jaxlib': '0.10.1', 'jax-cuda12-plugin': '0.10.1', 'jax-cuda12-pjrt': '0.10.1'}
  base = (PLANS / 'sw' / 'sw_v6s_01.yaml').read_text()
  for bad, msg in [(base.replace('    libtpu: 0.0.41\n', ''), 'pins exactly'),
                   (base.replace('    jaxlib: 0.10.1', '    jaxlib: 0.10.2'), 'same version'),
                   (base.replace('    libtpu: 0.0.41', '    libtpu: latest'), 'not a version')]:
    p = tmp_path / 'p.yaml'
    p.write_text(bad)
    with pytest.raises(SystemExit, match=msg):
      describe(p, 'v6e')


def test_existing_plans_get_no_new_keys():
  for p in sorted((PLANS / 'main').glob('main_*.yaml')):
    platform = next(iter(run_plan.read_configs(p)['deadline_min']))
    assert not {'stack', 'seeds_beyond_input'} & set(describe(p, platform)), p.name


def test_h100_plans_and_smoke_plans():
  d = describe(PLANS / 'h100' / 'h100_h_01.yaml', 'h100')
  assert d['configs'] == ['h100_xla'] and d['label'] == 'h100' and d['seeds'] == [1, 2, 3, 4, 5]
  r2 = describe(PLANS / 'h100' / 'h100_h_r2_01.yaml', 'h100')
  assert r2['rep_start'] == 2 and len(r2['targets']) == 10
  for f, platform in (('sw_smoke/sw_smoke_v6e.yaml', 'v6e'), ('sw_smoke/sw_smoke_l4.yaml', 'l4'),
                      ('h100_smoke/h100_smoke_h100.yaml', 'h100')):
    text = (PLANS / f).read_text()
    assert 'manifest: inputs/official_data/smoke_manifest.csv' in text and 'default: [2PV7]' in text
    assert ('stack:' in text) == f.startswith('sw_smoke')


def test_every_arm_covers_the_60_targets_once():
  subset = sorted(pd.read_csv(REPO / 'targets' / 'posebusters_subset.csv').pdb_id)
  by = {}
  for label, platform in (('seeds', None), ('sw', None), ('h100', 'h100')):
    for p in sorted((PLANS / label).glob('*.yaml')):
      arm = p.stem.rsplit('_', 1)[0]
      plat = next(iter(run_plan.read_configs(p)['deadline_min']))
      by.setdefault(arm, []).extend(describe(p, plat)['targets'])
  for arm, ts in by.items():
    want = sorted(pd.read_csv(REPO / 'inputs' / 'manifest.csv').query('pilot == "yes"').pdb_id) if arm == 'h100_h_r2' else subset
    assert sorted(ts) == sorted(want), arm


def test_generator_output_is_up_to_date():
  rc = subprocess.run([sys.executable, str(PLANS / 'make_fu7_10_plans.py'), '--check'], capture_output=True, text=True)
  assert rc.returncode == 0, rc.stdout + rc.stderr


def test_h100_condition_code_and_price_key():
  assert collect.platform_of('NVIDIA H100 80GB HBM3', 'xla', None, 'h100_xla') == 'H'
  assert cost.PRICE_KEY['H'] == 'h100'
  p = cost.load_prices()
  assert p[('h100', 'us-central1', 'standard')][0] > p[('h100', 'us-central1', 'spot')][0] > 0
  assert p[('cpu', 'europe-west4', 'spot')] == (0.321312, '2026-10-08')      # the C rows unchanged


# ---- analyses

@pytest.fixture(autouse=True)
def no_files(monkeypatch):
  monkeypatch.setattr(contrasts, 'coords_text', lambda path: path.split('@')[1])
  monkeypatch.setattr(contrasts, 'divergence', lambda t, a, b: {'ligand_rmsd': 1.0, 'ca_rmsd_global': 0.5, 'error': None})


def rows(platform, label, rep, targets, seed_list, shift=0.0, coords='x', success_on_top=True, session=None,
         machine='m', exit_code=0):
  out = []
  for t in targets:
    for s in seed_list:
      for k in range(5):
        top = 1 if shift > 0.1 else 0
        out.append({'session': session or f'results/af3/S_{label}_{platform}_{rep}/run', 'label': label, 'platform': platform,
                    'target': t, 'seed': s, 'repetition': rep, 'sample': k, 'num_recycles': 10, 'num_diffusion_samples': 5,
                    'ranking_score': 0.9 - 0.1 * k + (shift if k == 1 else 0.0),
                    'm1_success': (k == top) if success_on_top else (k == 0), 'mean_plddt': 80.0, 'iptm': 0.8,
                    'model_path': f'm/{label}/{platform}/{rep}/{t}/{s}/{k}@{coords}{k}', 'exit_code': exit_code,
                    'failure': None, 'machine_type': machine})
  return out


def write(df, path):
  pd.DataFrame(df).to_csv(path, index=False)
  return path


TARGETS = ['T1', 'T2', 'T3', 'T4']


def targets_csv(tmp_path):
  return write([{'pdb_id': t, 'rcsb_initial_release': '2022-01-01' if t in ('T3', 'T4') else '2020-01-01',
                 'pilot': 'yes' if t == 'T1' else 'no'} for t in TARGETS], tmp_path / 'targets.csv')


def test_seeds_extends_q3_and_matches_the_main_study(tmp_path, monkeypatch):
  main = tmp_path / 'main'
  (main / 'contrasts').mkdir(parents=True)
  m = pd.DataFrame(rows('V5', 'main', 'r1', TARGETS, range(1, 6)) + rows('G', 'main', 'r1', TARGETS, range(1, 6)))
  m['condition'] = m.platform
  m.to_csv(main / 'samples.csv', index=False)
  q, _ = contrasts.q3(f34.sample_rows(m), ARGS)
  q.to_csv(main / 'contrasts' / 'q3.csv', index=False)
  new = write(rows('V5', 'seeds', 'r1', TARGETS, range(6, 21)) + rows('G', 'seeds', 'r1', TARGETS, range(6, 21)),
              tmp_path / 'samples.csv')
  out = tmp_path / 'out'
  monkeypatch.setattr(sys, 'argv', ['seeds.py', '--samples', str(new), '--main', str(main), '--targets',
                                    str(targets_csv(tmp_path)), '--out', str(out), '--replicates', '40'])
  assert seeds.main() == 0
  chk = pd.read_csv(out / 'checks.csv')
  assert (chk.detail == 'pass').all() and chk.check.str.contains('equals the main study').any()
  q = pd.read_csv(out / 'q3.csv')
  assert q[q.targets == 'all 60'].n_seeds.max() == 20
  ans = json.loads((out / 'answer.json').read_text())
  assert ans['all 60']['answer'] == 1 and ans['all 60']['seeds'] == list(range(1, 21))   # identical platforms
  assert (pd.read_csv(out / 'completeness.csv').status == 'complete').all()


def test_seeds_refuses_a_seed_outside_6_to_20(tmp_path, monkeypatch):
  main = tmp_path / 'main'
  main.mkdir()
  write(rows('V5', 'main', 'r1', TARGETS, [1]), main / 'samples.csv')
  new = write(rows('V5', 'seeds', 'r1', TARGETS, [5]), tmp_path / 'samples.csv')
  monkeypatch.setattr(sys, 'argv', ['seeds.py', '--samples', str(new), '--main', str(main), '--targets',
                                    str(targets_csv(tmp_path)), '--out', str(tmp_path / 'out')])
  with pytest.raises(SystemExit, match='a check failed'):
    seeds.main()


def session_json(root, session, pkgs):
  d = root / session
  d.mkdir(parents=True, exist_ok=True)
  (d / 'session.json').write_text(json.dumps({'probe': {'packages': pkgs}}))


def sw_world(tmp_path, monkeypatch, alt_jax='0.10.1'):
  monkeypatch.setattr(sw, 'REPO', tmp_path)
  main = tmp_path / 'main'
  (main / 'contrasts').mkdir(parents=True)
  m = rows('V6', 'main', 'r1', TARGETS, range(1, 6), session='results/af3/M_v6/run') + \
      rows('G', 'main', 'r1', TARGETS, range(1, 6), session='results/af3/M_g/run')
  write(m, main / 'samples.csv')
  session_json(tmp_path, 'results/af3/M_v6/run', sw.ORIGINAL['V6'])
  session_json(tmp_path, 'results/af3/M_g/run', sw.ORIGINAL['G'])
  write([{'contrast': c, 'target': t, 'seed_a': s, 'seed_b': s, 'flip_top1': c == 'HW(V6,G)', 'flip_success': False,
          'flip_matters': False} for c in ('RC(G)', 'HW(V6,G)') for t in TARGETS for s in range(1, 6)],
        main / 'contrasts' / 'pairs_seed.csv')
  new = rows('V6', 'sw', 'r1', TARGETS, range(1, 6), session='results/af3/S_v6/run') + \
      rows('G', 'sw', 'r1', TARGETS, range(1, 6), shift=0.2, coords='y', session='results/af3/S_g/run')
  session_json(tmp_path, 'results/af3/S_v6/run', dict(sw.ALT['V6s'], jax=alt_jax))
  session_json(tmp_path, 'results/af3/S_g/run', sw.ALT['Gs'])
  out = tmp_path / 'out'
  monkeypatch.setattr(sys, 'argv', ['sw.py', '--samples', str(write(new, tmp_path / 'samples.csv')), '--main', str(main),
                                    '--targets', str(targets_csv(tmp_path)), '--out', str(out), '--replicates', '40'])
  return out


def test_sw_contrasts_comparisons_and_headline(tmp_path, monkeypatch):
  out = sw_world(tmp_path, monkeypatch)
  assert sw.main() == 0
  con = pd.read_csv(out / 'contrasts.csv')
  get = lambda c: float(con[(con.contrast == c) & (con.statistic == 'flip_top1') & (con.targets == 'all 60')].estimate.iloc[0])  # noqa: E731
  assert get('SW(V6)') == 0.0 and get('SW(G)') == 1.0
  comp = pd.read_csv(out / 'comparisons.csv').query('targets == "all 60" and statistic == "flip_top1"').set_index(['software', 'reference'])
  assert comp.loc[('SW(V6)', 'HW(V6,G)'), 'reading'] == 'lower' and comp.loc[('SW(G)', 'RC(G)'), 'reading'] == 'higher'
  rule = pd.read_csv(out / 'reading_rule.csv')
  assert rule.iloc[0].role == 'headline' and rule.iloc[0].statistic == 'flip_top1'
  assert rule.iloc[0].branch == 'the software stack moves the top-1 less than the chip'


def test_sw_stops_if_a_run_did_not_have_the_alternative_stack(tmp_path, monkeypatch):
  sw_world(tmp_path, monkeypatch, alt_jax='0.10.2')
  with pytest.raises(SystemExit, match='a check failed'):
    sw.main()


def test_h100_machine_type_and_rc_not_available(tmp_path, monkeypatch):
  main = tmp_path / 'main'
  main.mkdir()
  write(rows('V5', 'main', 'r1', TARGETS, [1, 2]) + rows('V6', 'main', 'r1', TARGETS, [1, 2]) +
        rows('G', 'main', 'r1', TARGETS, [1, 2]), main / 'samples.csv')
  a = write(rows('A', 'a100', 'r1', TARGETS, [1, 2]) + rows('A', 'a100', 'r2', TARGETS, [1, 2]), tmp_path / 'a.csv')
  h = write(rows('H', 'h100', 'r1', TARGETS, [1, 2], shift=0.2, coords='y', machine='a3-highgpu-1g'), tmp_path / 'h.csv')
  out = tmp_path / 'out'
  argv = ['h100.py', '--samples', str(h), '--a100', str(a), '--main', str(main), '--targets', str(targets_csv(tmp_path)),
          '--out', str(out), '--replicates', '40']
  monkeypatch.setattr(sys, 'argv', argv)
  assert h100.main() == 0
  con = pd.read_csv(out / 'contrasts.csv')
  assert float(con[(con.contrast == 'HW(H,A)') & (con.statistic == 'flip_top1') & (con.targets == 'all 60')].estimate.iloc[0]) == 1.0
  hw = pd.read_csv(out / 'hw_minus_rc.csv')
  assert (hw[hw.recompilation == 'RC(H)'].note.str.startswith('not available')).all()
  assert len(hw[(hw.recompilation == 'RC(A)') & (hw.targets == 'all 60')]) == 3
  comp = pd.read_csv(out / 'completeness.csv').set_index('repetition')
  assert comp.loc['r2', 'status'] == 'not run'
  write(rows('H', 'h100', 'r1', TARGETS, [1, 2], machine='a2-ultragpu-1g'), h)
  with pytest.raises(SystemExit, match='a check failed'):
    h100.main()


# ---- budget rule and settings diff

def credit(tmp_path, *args):
  (tmp_path / 'harness' / 'plans').mkdir(parents=True, exist_ok=True)
  (tmp_path / 'harness' / 'plans' / 'fu7_10_parts.csv').write_text((PLANS / 'fu7_10_parts.csv').read_text())
  r = subprocess.run(['bash', str(REPO / 'scripts' / 'monitor' / 'credit_check_fu7_10.sh'), *args], cwd=tmp_path,
                     capture_output=True, text=True)
  return r.returncode, r.stdout + r.stderr


def test_credit_check_floor_cap_and_spend_counted_once_at_its_provisioning(tmp_path):
  code, out = credit(tmp_path, 'v5x', '200')                       # Spot, as planned
  assert code == 0 and 'arm v5x (v5e spot): expected cost 18.81 USD' in out and out.rstrip().endswith('GO')
  code, out = credit(tmp_path, 'v6s', '200')                       # v6e on demand, as planned
  assert code == 0 and 'arm v6s (v6e standard): expected cost 29.36 USD' in out and '(cap of sw: 55)' in out
  code, out = credit(tmp_path, 'h', '82.71')                       # 82.71 - 58.69 = 24.02 < 25
  assert code == 1 and 'NO GO: balance minus expected 24.02 < 25 USD' in out
  code, out = credit(tmp_path, 'sw_smoke', '100')                  # v6e on demand and L4 Spot
  assert code == 0 and '(l4 spot, v6e standard)' in out
  rows = list(csv.DictReader(open(PLANS / 'fu7_10_parts.csv')))
  spot_l4 = next(float(r['usd_per_h_planned']) for r in rows if r['platform'] == 'l4')
  od_l4 = next(float(r['usd_per_h_on_demand']) for r in rows if r['platform'] == 'l4')
  events = '2026-11-03T00:00:00Z  vm: af3-run-l4-x running in us-central1-a\n2026-11-03T10:00:00Z  delete: done\n'
  for d in ('results/af3', 'results/af3/.launcher'):              # one session in both folders: counted once
    s = tmp_path / d / '20261103T000000Z_l4_seeds_seeds_gx_01'
    s.mkdir(parents=True)
    (s / 'launcher_events.txt').write_text(events)
    (s / 'launch_record.txt').write_text('PROVISIONING=spot\n')
  od = tmp_path / 'results' / 'af3' / '20261103T120000Z_l4_seeds_seeds_gx_02'   # no record: on demand
  od.mkdir(parents=True)
  (od / 'launcher_events.txt').write_text(events)
  smoke = tmp_path / 'results' / 'af3' / '20261102T000000Z_l4_sw_smoke_sw_smoke_l4'   # another follow-up: not counted
  smoke.mkdir(parents=True)
  (smoke / 'launcher_events.txt').write_text(events)
  code, out = credit(tmp_path, 'gx', '500')
  assert f'from 2 session record(s): arms {10 * spot_l4 + 10 * od_l4:.2f} USD' in out
  total = 10 * spot_l4 + 10 * od_l4 + 43.36                          # within the seeds cap of 75
  assert code == 0 and f'arms with this arm {total:.2f} USD (cap of seeds: 75)' in out
  assert credit(tmp_path, 'gx', 'x')[0] == 2 and credit(tmp_path, 'nope', '1')[0] == 2


def test_config_diff_shows_only_the_factor_under_study():
  r = subprocess.run([sys.executable, str(PLANS / 'config_diff_fu7_10.py'), '--out', '/dev/null'], capture_output=True,
                     text=True)
  assert r.returncode == 0, r.stdout[-2000:] + r.stderr
  assert 'Settings that differ beyond the factor under study: 0' in r.stdout
  section = lambda arm: r.stdout.split(f'## {arm} against')[1].split('\n## ')[0]  # noqa: E731
  assert '| software stack (jax, jaxlib, libtpu or jax-cuda12 plugin and pjrt) |' in section('V6s')
  assert '| intended |' in section('V6s').split('software stack')[1].splitlines()[0]
  assert '| seeds of the arm | `1 to 5` | `6 to 20` | intended |' in section('V5x')
  assert '| machine type | `a2-ultragpu-1g` | `a3-highgpu-1g` | intended |' in section('H')
  assert '| provisioning | `standard` | `spot` | intended |' in section('Gx')       # how the VM is obtained
  assert '| provisioning | `standard` | `standard` | same |' in section('V6s')     # v6e on demand
