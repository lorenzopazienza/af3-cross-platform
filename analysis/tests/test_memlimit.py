"""analysis/memlimit.py: the memory each error message reports, cost per structure, and the recorded outcome of the
memory-limit measurement (section 10, 2026-10-14)."""

import math
import pathlib

import pytest

import memlimit

REPO = pathlib.Path(__file__).resolve().parents[2]
needs_sessions = pytest.mark.skipif(not list((REPO / 'results' / 'af3').glob('*_memlimit_*')),
                                    reason='memory-limit sessions not present (results/ is gitignored)')


def test_error_memory():
  tpu = ('jax.errors.JaxRuntimeError: RESOURCE_EXHAUSTED: Ran out of memory on HBM, the total memory required for '
         'HLO temporaries (31.82G) exceeds available HBM (31.24G).')
  assert memlimit.error_memory(tpu)[:2] == (31.82, 31.24)
  gpu = ('W1014 16:54:55.546229 15333 bfc_allocator.cc:514] Allocator (GPU_0_bfc) ran out of memory trying to '
         'allocate 4.50GiB (rounded to 4831838208)requested by op')
  required, available, kind = memlimit.error_memory(gpu)
  assert required == 4.5 and math.isnan(available) and kind.startswith('GPU')
  assert math.isnan(memlimit.error_memory(None)[0]) and math.isnan(memlimit.error_memory('Killed')[0])


def test_cost_per_structure_is_wall_per_seed_times_the_on_demand_price():
  prices = memlimit.cost.load_prices()
  rec = {'af3_buckets': [1280], 'padded_bucket': 1280, 'num_tokens': 1023, 'seeds': [1], 'exit_code': 0, 'failure': None,
         'wall_seconds': 3600.0, 'inference_seconds_by_seed': {'1': 3500.0}, 'compile': {'model_seconds': 50.0},
         'memory': {'compiled_peak_memory_in_bytes': 2 ** 31, 'compiled_argument_size_in_bytes': 2 ** 30,
                    'bytes_limit': 2 ** 34}}
  row = memlimit.process_rows([('s', 'V6', 'europe-west4-a', rec)], prices)[0]
  assert row['usd_per_structure_standard'] == pytest.approx(2.97)
  assert row['compiled_peak_plus_args_gib'] == 3.0 and row['device_limit_gib'] == 16.0
  failed = dict(rec, exit_code=1, failure='oom', memory={})
  assert math.isnan(memlimit.process_rows([('s', 'V6', 'europe-west4-a', failed)], prices)[0]['usd_per_structure_standard'])


@needs_sessions
def test_recorded_outcome_matches_the_section_10_row():
  s = memlimit.summary(memlimit.process_table()).set_index('arm')
  assert {a: (int(s.loc[a].largest_completed_size), int(s.loc[a].stopping_size)) for a in memlimit.ARMS} == \
      {'V5': (1536, 2048), 'V6': (2048, 2560), 'G': (2560, 3072), 'GT': (2560, 3072)}
  assert (s.cause == 'oom').all()


@needs_sessions
def test_main_study_point_per_platform():
  t = memlimit.process_table()
  main = t[t.source != 'memlimit']
  assert sorted(main.arm.astype(str)) == sorted(memlimit.ARMS) and (main['size'] == 1024).all()
  assert (main.exit_code == 0).all()
