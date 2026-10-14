"""Memory limit per chip (notes/analysis_plan.md section 10, rows "Memory limit per chip" and "Memory-limit
measurement, launches" of 2026-10-14): one row per AlphaFold3 process, from run records only.

    analysis/.venv/bin/python analysis/memlimit.py [--out results/analysis/memlimit_20261014]

Processes: every process of the memory-limit sessions in results/af3/ (label memlimit; 7D5C padded with
AlphaFold3's --buckets to one size per process), and per platform the main study's 7D5C process at its own
1,024-token bucket (fresh cache, r1, seed 1; results/analysis/main_20261014/primary_v2/samples.csv). Per process:
exit status and failure; XLA's memory analysis of the compiled model (peak, arguments, temporaries); the
allocator's peak in use and reserved; the device limit; compile, inference and wall time; the on-demand list price
of the zone's region and the cost per structure (wall time per seed x price, as analysis/cost.py); for a failed
process, the memory the error message reports (TPU: required for HLO temporaries and available HBM; GPU: the
single allocation that failed). The predicted structures are never read.
Outputs in --out: processes.csv, summary.csv, report.md.
"""
import argparse
import json
import pathlib
import re
import sys

import numpy as np
import pandas as pd

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / 'analysis'))
import cost  # noqa: E402

AF3 = REPO / 'results' / 'af3'
SAMPLES = REPO / 'results' / 'analysis' / 'main_20261014' / 'primary_v2' / 'samples.csv'
ARMS = ['V5', 'V6', 'G', 'GT']
CODE = {('v5e', 'tpu_xla'): 'V5', ('v6e', 'tpu_xla'): 'V6', ('l4', 'l4_xla'): 'G', ('l4', 'l4_triton'): 'GT'}
GIB = 2 ** 30
TPU_OOM = re.compile(r'required for HLO temporaries \(([\d.]+)G\) exceeds available HBM \(([\d.]+)G\)')
GPU_OOM = re.compile(r'ran out of memory trying to allocate ([\d.]+)GiB')


def gib(x):
  return np.nan if x is None or (isinstance(x, float) and np.isnan(x)) else x / GIB


def error_memory(detail):
  """(required GiB, available GiB, kind) from a failure_detail, or NaNs."""
  if not isinstance(detail, str):
    return np.nan, np.nan, ''
  m = TPU_OOM.search(detail)
  if m:
    return float(m.group(1)), float(m.group(2)), 'TPU: HLO temporaries required, available HBM'
  m = GPU_OOM.search(detail)
  if m:
    return float(m.group(1)), np.nan, 'GPU: single allocation that failed'
  return np.nan, np.nan, ''


def session_records(root=AF3):
  """[(session, arm, zone, run.json record)] of every memory-limit process."""
  out = []
  for sj in sorted(root.glob('*/*/session.json')):
    try:
      s = json.loads(sj.read_text())
    except ValueError:
      continue
    if s.get('label') != 'memlimit':
      continue
    platform = next((p for p in ('v5e', 'v6e', 'l4') if f'_{p}_' in sj.parent.parent.name), None)
    arm = CODE.get((platform, s.get('config_name')))
    for rj in sorted(sj.parent.glob('*/run.json')):
      out.append((sj.parent.parent.name, arm, (s.get('host') or {}).get('gce_zone'), json.loads(rj.read_text())))
  return out


def process_rows(records, prices):
  rows = []
  for session, arm, zone, r in records:
    m, c = r.get('memory') or {}, r.get('compile') or {}
    price = cost.price_for(prices, arm, zone, 'standard')[0]
    required, available, kind = error_memory(r.get('failure_detail'))
    ok = r['exit_code'] == 0 and not r.get('failure')
    seeds = len(r.get('seeds') or [1])
    inference = sum((r.get('inference_seconds_by_seed') or {}).values()) or np.nan
    peak, args = gib(m.get('compiled_peak_memory_in_bytes')), gib(m.get('compiled_argument_size_in_bytes'))
    rows.append({'source': 'memlimit', 'session': session, 'arm': arm, 'zone': zone,
                 'size': (r.get('af3_buckets') or [r.get('bucket')])[0], 'padded_bucket': r.get('padded_bucket'),
                 'tokens': r.get('num_tokens'), 'exit_code': r['exit_code'], 'failure': r.get('failure') or '',
                 'compiled_peak_gib': peak, 'compiled_args_gib': args, 'compiled_peak_plus_args_gib': peak + args,
                 'compiled_temp_gib': gib(m.get('compiled_temp_size_in_bytes')),
                 'alloc_peak_in_use_gib': gib(m.get('peak_bytes_in_use')),
                 'alloc_peak_reserved_gib': gib(m.get('peak_bytes_reserved')),
                 'device_limit_gib': gib(m.get('bytes_limit')), 'compile_s': c.get('model_seconds'),
                 'inference_s': inference, 'wall_s': r.get('wall_seconds'), 'usd_per_hour_standard': price,
                 'usd_per_structure_standard': r['wall_seconds'] / seeds / 3600 * price if ok and price else np.nan,
                 'error_required_gib': required, 'error_available_gib': available, 'error_memory_kind': kind,
                 'failure_detail': r.get('failure_detail') or ''})
  return rows


def main_study_rows(prices, samples=SAMPLES):
  """The main study's 7D5C process (fresh, r1, seed 1, bucket 1,024) per platform."""
  d = pd.read_csv(samples, low_memory=False)
  d = d[(d.target == '7D5C') & d.platform.isin(ARMS) & (d.repetition == 'r1') & (d.seed == 1)
        & (d.cache == 'fresh')].drop_duplicates(['session', 'run', 'target'])
  rows = []
  for r in d.itertuples():
    price = cost.price_for(prices, r.platform, r.zone, 'standard')[0]
    peak, args = gib(r.compiled_peak_bytes), gib(r.compiled_args_bytes)
    rows.append({'source': 'main study (7D5C, r1, seed 1)', 'session': r.session, 'arm': r.platform, 'zone': r.zone,
                 'size': int(r.bucket), 'padded_bucket': int(r.bucket), 'tokens': int(r.num_tokens),
                 'exit_code': int(r.exit_code), 'failure': '', 'compiled_peak_gib': peak, 'compiled_args_gib': args,
                 'compiled_peak_plus_args_gib': peak + args, 'compiled_temp_gib': gib(r.compiled_temp_bytes),
                 'alloc_peak_in_use_gib': gib(r.alloc_peak_in_use_bytes),
                 'alloc_peak_reserved_gib': gib(r.alloc_peak_reserved_bytes), 'device_limit_gib': gib(r.device_bytes_limit),
                 'compile_s': r.compile_s, 'inference_s': r.inference_s, 'wall_s': r.wall_s, 'usd_per_hour_standard': price,
                 'usd_per_structure_standard': r.wall_s / 3600 * price if price else np.nan,
                 'error_required_gib': np.nan, 'error_available_gib': np.nan, 'error_memory_kind': '',
                 'failure_detail': ''})
  return rows


def process_table(root=AF3, samples=SAMPLES):
  prices = cost.load_prices()
  rows = process_rows(session_records(root), prices)
  if pathlib.Path(samples).exists():
    rows += main_study_rows(prices, samples)
  t = pd.DataFrame(rows)
  if t.empty:
    return t
  t['arm'] = pd.Categorical(t.arm, ARMS, ordered=True)
  return t.sort_values(['arm', 'size']).reset_index(drop=True)


def summary(t):
  rows = []
  for arm in ARMS:
    g = t[(t.arm == arm) & (t.source == 'memlimit')]
    if g.empty:
      continue
    ok = g[(g.exit_code == 0) & (g.failure == '')]
    bad = g[(g.exit_code != 0) | (g.failure != '')].sort_values('size')
    b = bad.iloc[0] if len(bad) else None
    rows.append({'arm': arm, 'session': g.session.iloc[0], 'sizes_run': len(g),
                 'largest_completed_size': int(ok['size'].max()) if len(ok) else None,
                 'stopping_size': int(b['size']) if b is not None else None,
                 'cause': b.failure if b is not None else '',
                 'error_required_gib': b.error_required_gib if b is not None else np.nan,
                 'error_available_gib': b.error_available_gib if b is not None else np.nan,
                 'stopping_compiled_peak_plus_args_gib': b.compiled_peak_plus_args_gib if b is not None else np.nan,
                 'largest_completed_peak_plus_args_gib': ok.sort_values('size').compiled_peak_plus_args_gib.iloc[-1]
                 if len(ok) else np.nan,
                 'device_limit_gib': g.device_limit_gib.dropna().iloc[0] if g.device_limit_gib.notna().any() else np.nan,
                 'largest_completed_wall_s': ok.sort_values('size').wall_s.iloc[-1] if len(ok) else np.nan,
                 'largest_completed_usd_per_structure': ok.sort_values('size').usd_per_structure_standard.iloc[-1]
                 if len(ok) else np.nan})
  return pd.DataFrame(rows)


def report(t, s, out):
  f = lambda x, nd=2: '-' if pd.isna(x) else f'{x:.{nd}f}'  # noqa: E731
  lines = ['# Memory limit per chip (M7)', '', 'Section 10 rows of 2026-10-14. One row per AlphaFold3 process (7D5C, '
           'seed 1, padded with --buckets); the 1,024 rows are the main study\'s 7D5C process (fresh, r1, seed 1). '
           'GiB = 2^30 bytes; the TPU error messages\' "G" equals GiB (their "available HBM" equals the recorded '
           'device limit). Code: analysis/memlimit.py.', '',
           '| arm | size | exit | compiled peak + args | temporaries | allocator peak in use | device limit | '
           'compile (s) | inference (s) | wall (s) | USD/structure | memory in the error |',
           '|---|---|---|---|---|---|---|---|---|---|---|---|']
  for r in t.itertuples():
    err = '' if pd.isna(r.error_required_gib) else (
        f'{r.error_required_gib:.2f} required, {r.error_available_gib:.2f} available' if not pd.isna(r.error_available_gib)
        else f'allocation of {r.error_required_gib:.2f} failed')
    lines.append(f'| {r.arm} | {r.size}{"*" if r.source != "memlimit" else ""} | {r.exit_code}{" " + r.failure if r.failure else ""} | '
                 f'{f(r.compiled_peak_plus_args_gib)} | {f(r.compiled_temp_gib)} | {f(r.alloc_peak_in_use_gib)} | '
                 f'{f(r.device_limit_gib)} | {f(r.compile_s, 0)} | {f(r.inference_s, 0)} | {f(r.wall_s, 0)} | '
                 f'{f(r.usd_per_structure_standard, 3)} | {err} |')
  lines += ['', '\\* main study, bucket 1,024.', '', '| arm | largest completed | stopping size | cause | memory in the error | '
            'peak + args, largest completed | device limit |', '|---|---|---|---|---|---|---|']
  for r in s.itertuples():
    lines.append(f'| {r.arm} | {r.largest_completed_size} | {r.stopping_size} | {r.cause} | '
                 f'{f(r.error_required_gib)} | {f(r.largest_completed_peak_plus_args_gib)} | {f(r.device_limit_gib)} |')
  (out / 'report.md').write_text('\n'.join(lines) + '\n')


def main():
  ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
  ap.add_argument('--out', default=str(REPO / 'results' / 'analysis' / 'memlimit_20261014'))
  args = ap.parse_args()
  out = pathlib.Path(args.out)
  out.mkdir(parents=True, exist_ok=True)
  t = process_table()
  s = summary(t)
  t.to_csv(out / 'processes.csv', index=False)
  s.to_csv(out / 'summary.csv', index=False)
  report(t, s, out)
  print((out / 'report.md').read_text())
  return 0


if __name__ == '__main__':
  sys.exit(main())
