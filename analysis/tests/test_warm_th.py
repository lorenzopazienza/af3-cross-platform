"""analysis/warm_th.py on synthetic sessions in the layout analysis/collect.py writes (one row per sample) and synthetic
AlphaFold3 output folders. No real session is read."""
import pathlib
import subprocess
import sys

import pandas as pd
import pytest

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'analysis'))
import cost  # noqa: E402
import warm_th  # noqa: E402

TARGETS = warm_th.pilot_targets()
SESSIONS = {'GT': '20261112T080000Z_l4_warm_th_gt_01', 'H': '20261112T080005Z_h100_warm_th_h_01'}
ZONE = {'GT': 'europe-west4-a', 'H': 'us-central1-a'}
RUN = {'GT': 'l4_triton', 'H': 'h100_xla'}
TS = b'_ma_model_list.model_group_name'


def outputs(root, session, run, t, stamp=b'2026-11-12 08:00:00', change=None):
  """One target's AlphaFold3 output folder (contrasts.TARGET_FILES and SAMPLE_FILES); returns the sample-0 model."""
  d = root / session / run / t / 'af3_output' / t
  d.mkdir(parents=True)
  for s in warm_th.contrasts.TARGET_FILES:
    (d / f'{t}{s}').write_bytes(b'data_x\n' + TS + b' "AlphaFold 3 ' + stamp + b'"\nATOM 1\n' if s.endswith('.cif')
                                else b'{"x": 1}\n')
  for k in range(5):
    sd = f'seed-1_sample-{k}'
    (d / sd).mkdir()
    for s in warm_th.contrasts.SAMPLE_FILES:
      body = b'data_x\n' + TS + b' "' + stamp + b'"\nATOM 1 ' + str(k).encode() + b'\n' if s.endswith('.cif') else b'{}\n'
      if change and change == (sd, s):
        body += b'ATOM 2\n'
      (d / sd / f'{t}_{sd}{s}').write_bytes(body)
  return d / 'seed-1_sample-0' / f'{t}_seed-1_sample-0_model.cif'


def synthetic(tmp_path, change=None, bad_key=None, extra_session=False, label='warm_th'):
  """samples.csv of both arms (r1 and w, the 10 pilot targets, seed 1, 5 samples) and their output folders."""
  man = pd.read_csv(REPO / 'inputs' / 'manifest.csv').set_index('pdb_id')
  root, rows = tmp_path / 'af3', []
  for arm in warm_th.ARMS:
    platform, _, config, machine = warm_th.ARM_RUN[arm]
    for rep, run, cache in (('r1', f'{RUN[arm]}_rep1_seed1_fresh', 'fresh'), ('w', f'{RUN[arm]}_seed1_warm', 'warm')):
      for t in TARGETS:
        ch = change if (change and rep == 'w' and arm == change[0] and t == change[1]) else None
        model = outputs(root, SESSIONS[arm], run, t, stamp=b'2026-11-12 0' + (b'8' if rep == 'r1' else b'9') + b':00:00',
                        change=ch[2:] if ch else None)
        key = f'key_{arm}_{t}'
        bucket = int(man.loc[t, 'bucket'])
        wall = (100.0 if rep == 'r1' else 70.0) * bucket / 1024
        for k in range(5):
          rows.append({'session': f'results/af3/{SESSIONS[arm]}/{run}', 'run': run, 'label': label, 'platform': arm, 'config': config,
                       'attention': 'triton' if arm == 'GT' else 'xla', 'matmul_precision': '', 'libtpu': '',
                       'zone': ZONE[arm], 'machine_type': machine, 'target': t, 'num_tokens': int(man.loc[t, 'num_tokens']),
                       'bucket': bucket, 'seed': 1, 'repetition': rep, 'cache': cache, 'sample': k, 'num_recycles': 10,
                       'num_diffusion_samples': 5, 'exit_code': 0, 'failure': '',
                       'model_path': str(model).replace('sample-0', f'sample-{k}'), 'wall_s': wall,
                       'inference_s': wall - 30, 'compile_s': 30.0 if rep == 'r1' else 3.0,
                       'compiled_peak_bytes': 1e9, 'compiled_args_bytes': 1e8, 'alloc_peak_reserved_bytes': 0,
                       'alloc_peak_in_use_bytes': 2e9, 'device_bytes_limit': 2e10, 'max_rss_bytes': 3e9,
                       'cache_files_before': 0 if rep == 'r1' else 3,
                       'model_cache_hit': rep == 'w', 'model_cache_key': (bad_key if bad_key and rep == 'w' and t == TARGETS[0]
                                                                          else key) if rep == 'w' else '',
                       'model_cache_written': key if rep == 'r1' else ''})
  d = pd.DataFrame(rows)
  samples = tmp_path / 'samples.csv'
  d.to_csv(samples, index=False)
  listed = list(SESSIONS.values()) + (['20261112T090000Z_l4_warm_th_gt_01'] if extra_session else [])
  sessions = tmp_path / 'sessions.txt'
  sessions.write_text('# synthetic\n' + '\n'.join(listed) + '\n')
  return samples, sessions


def run(samples, sessions, out):
  return subprocess.run([sys.executable, str(REPO / 'analysis' / 'warm_th.py'), '--samples', str(samples), '--sessions',
                         str(sessions), '--out', str(out)], capture_output=True, text=True, cwd=REPO)


def test_complete_arms_give_the_table_iii_cells(tmp_path):
  samples, sessions = synthetic(tmp_path)
  out = tmp_path / 'warm_th_20261113'
  r = run(samples, sessions, out)
  assert r.returncode == 0, r.stdout + r.stderr
  ck = pd.read_csv(out / 'checks.csv')
  assert (ck.detail == 'pass').all() and len(ck) == 1 + 2 + 2 + 2 * 4          # 0, completeness, 1, then 2a, 2b, bit-identity, overall
  assert set(ck[ck.check.str.startswith('2 bit-identity')].value) == {'10/10 targets'}
  s = pd.read_csv(out / 'summary.csv')
  main = pd.read_csv(REPO / 'results' / 'analysis' / 'main_20261014' / 'primary_v2' / 'cost' / 'cost_summary.csv') \
      if (REPO / 'results' / 'analysis' / 'main_20261014' / 'primary_v2' / 'cost' / 'cost_summary.csv').exists() else None
  if main is not None:
    assert list(s.columns) == list(main.columns)                                    # Table III reads the same columns
  w = s[(s.cache == 'warm') & (s.bucket == 1024)].set_index('platform')
  prices = cost.load_prices()
  for arm, key, region in (('GT', 'l4', 'europe-west4'), ('H', 'h100', 'us-central1')):
    assert w.loc[arm, 'processes'] == 2                                           # 7D5C and 8EYE
    assert w.loc[arm, 'median_usd_per_structure_standard'] == pytest.approx(70 / 3600 * prices[(key, region, 'standard')][0])
    assert w.loc[arm, 'median_usd_per_structure_spot'] == pytest.approx(70 / 3600 * prices[(key, region, 'spot')][0])
  assert 'median_usd_per_structure_amortised_standard' in s.columns and s[s.cache == 'warm'][
      'median_usd_per_structure_amortised_standard'].isna().all()                # warm is not amortised again
  rep = (out / 'report.md').read_text()
  assert 'Spot prices of cloud/prices.csv read on 2026-10-27' in rep
  pu = pd.read_csv(out / 'prices_used.csv')
  assert set(pu.platform) == {'l4', 'h100'} and set(pu.region) == {'europe-west4', 'us-central1'}
  inp = pd.read_csv(out / 'inputs.tsv', sep='\t')
  assert 'cloud/prices.csv' in set(inp.path) and 'inputs/manifest.csv' in set(inp.path)
  r2 = run(samples, sessions, out)
  assert r2.returncode != 0 and 'must be new' in r2.stderr


def test_a_warm_output_that_differs_stops_before_the_cost(tmp_path):
  samples, sessions = synthetic(tmp_path, change=('H', TARGETS[3], 'seed-1_sample-2', '_model.cif'))
  out = tmp_path / 'o'
  r = run(samples, sessions, out)
  assert r.returncode != 0 and 'a check failed' in r.stderr
  ck = pd.read_csv(out / 'checks.csv')
  bit = ck[ck.check.str.startswith('2 bit-identity')].set_index('condition')
  assert bit.loc['GT', 'detail'] == 'pass' and bit.loc['H', 'value'] == '9/10 targets'
  assert TARGETS[3] in bit.loc['H', 'detail'] and 'seed-1_sample-2' in bit.loc['H', 'detail']
  assert not (out / 'summary.csv').exists() and 'not completed' in (out / 'report.md').read_text()


def test_a_warm_process_that_did_not_load_its_source_executable_fails_2b(tmp_path):
  samples, sessions = synthetic(tmp_path, bad_key='another_key')
  out = tmp_path / 'o'
  assert run(samples, sessions, out).returncode != 0
  ck = pd.read_csv(out / 'checks.csv')
  b = ck[ck.check.str.startswith('2b')].set_index('condition')
  assert b.loc['GT', 'value'] == '9/10 processes' and TARGETS[0] in b.loc['GT', 'detail']
  assert not (out / 'summary.csv').exists()


def test_session_list_and_labels_are_checked(tmp_path):
  samples, sessions = synthetic(tmp_path, extra_session=True)
  out = tmp_path / 'o'
  assert run(samples, sessions, out).returncode != 0
  assert 'arm GT: 2 listed sessions' in pd.read_csv(out / 'checks.csv').detail.iloc[0]
  samples, sessions = synthetic(tmp_path / 'b', label='main')
  out = tmp_path / 'o2'
  assert run(samples, sessions, out).returncode != 0
  assert 'labels' in pd.read_csv(out / 'checks.csv').detail.iloc[0]


def test_a_missing_warm_process_fails_completeness(tmp_path):
  samples, sessions = synthetic(tmp_path)
  d = pd.read_csv(samples)
  d = d[~((d.platform == 'GT') & (d.repetition == 'w') & (d.target == '8EYE'))]
  d.to_csv(samples, index=False)
  out = tmp_path / 'o'
  assert run(samples, sessions, out).returncode != 0
  ck = pd.read_csv(out / 'checks.csv')
  c = ck[ck.check.str.startswith('completeness')].set_index('condition')
  assert '8EYE' in c.loc['GT', 'detail'] and c.loc['H', 'detail'] == 'pass'

