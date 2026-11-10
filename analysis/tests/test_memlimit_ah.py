"""Memory limit on A and H (label memlimit_ah; analysis/memlimit_ah.py): synthetic sessions only. No session of label
memlimit_ah is read: they are written here in a temporary folder, in the layout of the 2026-10-14 memory-limit
sessions (<session>/<config>_bucket<S>_rep1_seed1_fresh/session.json and 7D5C/run.json)."""
import csv
import json
import pathlib
import subprocess
import sys

import pandas as pd
import pytest

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'analysis'))
import memlimit  # noqa: E402
import memlimit_ah  # noqa: E402

SIZES = [1280, 1536, 2048, 2560, 3072, 3584, 4096, 4608, 5120]
GIB = 2 ** 30
FROZEN = REPO / 'results' / 'analysis' / 'memlimit_20261014' / 'summary.csv'
COLUMNS = ['arm', 'session', 'sizes_run', 'largest_completed_size', 'stopping_size', 'cause', 'error_required_gib',
           'error_available_gib', 'stopping_compiled_peak_plus_args_gib', 'largest_completed_peak_plus_args_gib',
           'device_limit_gib', 'largest_completed_wall_s', 'largest_completed_usd_per_structure']


def write_session(root, name, platform, config, label, oom_at=None, zone='us-central1-c'):
  for size in SIZES:
    if oom_at and size > oom_at:
      break                                    # stop rule: later sizes not run
    run = root / name / f'{config}_bucket{size}_rep1_seed1_fresh'
    (run / '7D5C').mkdir(parents=True)
    (run / 'session.json').write_text(json.dumps({'label': label, 'config_name': config, 'host': {'gce_zone': zone}}))
    fail = size == oom_at
    scale = (size / 1024) ** 2
    rec = {'exit_code': 1 if fail else 0, 'failure': 'oom' if fail else None, 'af3_buckets': [size],
           'padded_bucket': size, 'num_tokens': 1023, 'bucket': 1024, 'seeds': [1],
           'wall_seconds': 100.0 if fail else 240.0 * scale, 'inference_seconds_by_seed': {} if fail else {'1': 200.0 * scale},
           'compile': {'model_seconds': 60.0},
           'failure_detail': ('W1110 bfc_allocator.cc:514] Allocator (GPU_0_bfc) ran out of memory trying to allocate '
                              '9.50GiB (rounded to 10200547328)requested by op') if fail else None,
           'memory': {'compiled_peak_memory_in_bytes': 6 * GIB * scale, 'compiled_argument_size_in_bytes': 0.3 * GIB,
                      'compiled_temp_size_in_bytes': 5 * GIB * scale, 'peak_bytes_in_use': 5.5 * GIB * scale,
                      'peak_bytes_reserved': 6 * GIB * scale, 'bytes_limit': 75.3 * GIB}}
    (run / '7D5C' / 'run.json').write_text(json.dumps(rec))


def write_samples(path, arm, zone):
  cols = ['target', 'platform', 'repetition', 'seed', 'cache', 'session', 'run', 'bucket', 'num_tokens', 'exit_code',
          'compiled_peak_bytes', 'compiled_args_bytes', 'compiled_temp_bytes', 'alloc_peak_in_use_bytes',
          'alloc_peak_reserved_bytes', 'device_bytes_limit', 'compile_s', 'inference_s', 'wall_s', 'zone', 'sample']
  with open(path, 'w', newline='') as f:
    w = csv.DictWriter(f, cols)
    w.writeheader()
    for s in range(5):     # five sample rows of one process; main_study_rows keeps one
      w.writerow({'target': '7D5C', 'platform': arm, 'repetition': 'r1', 'seed': 1, 'cache': 'fresh', 'session': 'x',
                  'run': 'r', 'bucket': 1024, 'num_tokens': 1023, 'exit_code': 0, 'compiled_peak_bytes': 6 * GIB,
                  'compiled_args_bytes': 0.3 * GIB, 'compiled_temp_bytes': 5 * GIB, 'alloc_peak_in_use_bytes': 5 * GIB,
                  'alloc_peak_reserved_bytes': 6 * GIB, 'device_bytes_limit': 75.3 * GIB, 'compile_s': 55.0,
                  'inference_s': 184.0, 'wall_s': 241.0, 'zone': zone, 'sample': s})


A_S, H_S = '20261110T120000Z_a100_memlimit_ah_a_01', '20261110T120005Z_h100_memlimit_ah_h_01'


@pytest.fixture
def setup(tmp_path):
  root = tmp_path / 'af3'
  write_session(root, A_S, 'a100', 'a100_xla', 'memlimit_ah', oom_at=5120)
  write_session(root, H_S, 'h100', 'h100_xla', 'memlimit_ah', zone='us-central1-a')
  samples = {'A': tmp_path / 'a.csv', 'H': tmp_path / 'h.csv'}
  write_samples(samples['A'], 'A', 'us-central1-c')
  write_samples(samples['H'], 'H', 'us-central1-a')
  return root, samples


def test_summary_stopping_size_and_every_size_completed(setup):
  root, samples = setup
  read = []
  t = memlimit_ah.process_table(root, [A_S, H_S], samples, read)
  with memlimit_ah.arms_a_h():
    s = memlimit.summary(t)
  assert memlimit.ARMS == ['V5', 'V6', 'G', 'GT']          # restored
  a, h = s[s.arm == 'A'].iloc[0], s[s.arm == 'H'].iloc[0]
  assert a.sizes_run == 9 and a.largest_completed_size == 4608 and a.stopping_size == 5120 and a.cause == 'oom'
  assert a.error_required_gib == pytest.approx(9.5) and a.device_limit_gib == pytest.approx(75.3)
  assert h.sizes_run == 9 and h.largest_completed_size == 5120 and pd.isna(h.stopping_size) and h.cause == ''
  assert list(s.columns) == COLUMNS
  if FROZEN.is_file():
    assert list(pd.read_csv(FROZEN, nrows=0).columns) == COLUMNS
  ref = t[t.source != 'memlimit']
  assert sorted(ref.source) == ['follow-up 3 (7D5C, r1, seed 1)', 'follow-up 9 (7D5C, r1, seed 1)']
  assert (ref['size'] == 1024).all() and len(ref) == 2
  assert t[(t.arm == 'A') & (t.source == 'memlimit')].usd_per_structure_standard.notna().sum() == 8   # failed: none
  assert len([p for p in read if p.name == 'run.json']) == 18


def test_checks_refuse_wrong_label_two_sessions_and_unknown_config(tmp_path, setup):
  root, samples = setup
  write_session(root, '20261110T130000Z_a100_memlimit_ah_a_01', 'a100', 'a100_xla', 'memlimit_ah')
  with pytest.raises(SystemExit, match='more than one session'):
    memlimit_ah.process_table(root, [A_S, H_S, '20261110T130000Z_a100_memlimit_ah_a_01'], samples, [])
  write_session(root, '20261110T140000Z_h100_memlimit_ah_h_01', 'h100', 'h100_xla', 'memlimit')
  with pytest.raises(SystemExit, match='label'):
    memlimit_ah.process_table(root, ['20261110T140000Z_h100_memlimit_ah_h_01'], samples, [])
  write_session(root, '20261110T150000Z_a100_memlimit_ah_a_01', 'a100', 'a100_triton', 'memlimit_ah')
  with pytest.raises(SystemExit, match='not arm A or H'):
    memlimit_ah.process_table(root, ['20261110T150000Z_a100_memlimit_ah_a_01'], samples, [])
  with pytest.raises(SystemExit, match='not a memlimit_ah session'):
    memlimit_ah.process_table(root, ['20261014T130116Z_l4_memlimit_g'], samples, [])


def test_main_writes_the_outputs_and_refuses_an_existing_folder(tmp_path, setup):
  root, _ = setup
  sessions = tmp_path / 'sessions.txt'
  sessions.write_text(f'# header\n{A_S}\n{H_S}  # complete\n')
  out = tmp_path / 'out'
  cmd = [sys.executable, str(REPO / 'analysis' / 'memlimit_ah.py'), '--sessions', str(sessions), '--out', str(out),
         '--root', str(root)]
  r = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO)
  assert r.returncode == 0, r.stderr[-2000:]
  for name in ('processes.csv', 'summary.csv', 'report.md', 'inputs.tsv'):
    assert (out / name).is_file(), name
  s = pd.read_csv(out / 'summary.csv')
  assert list(s.columns) == COLUMNS and list(s.arm) == ['A', 'H']
  rep = (out / 'report.md').read_text()
  assert 'label memlimit_ah' in rep and 'main study' not in rep and '| A | 5120 | 1 oom |' in rep
  inp = pd.read_csv(out / 'inputs.tsv', sep='\t')
  assert list(inp.columns) == ['path', 'sha256'] and (inp.sha256.str.len() == 64).all()
  assert inp.path.str.endswith('run.json').sum() == 18 and 'cloud/prices.csv' in set(inp.path)
  r = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO)
  assert r.returncode != 0 and 'must be new' in r.stderr
