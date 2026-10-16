"""Follow-up 3, the A100 arms (notes/analysis_plan.md section 10, rows of 2026-10-16; label a100; reported as a
follow-up, outside the pre-registered study).

    analysis/.venv/bin/python analysis/collect.py --labels a100 --out results/analysis/a100_<date> --jobs 8
    analysis/.venv/bin/python analysis/a100.py --samples results/analysis/a100_<date>/samples.csv \\
        --out results/analysis/a100_<date> [--cpu60 results/analysis/cpu60_<date>/samples.csv]
        [--main results/analysis/main_20261014/primary_v2] [--replicates 10000] [--seed 2026]

Conditions: A (a100_xla), Ah (a100_xla_highest) and AT (a100_triton) from the a100 sessions (analysis/collect.py
codes them from the device and the run config); the main study's G, GT, V5, V6, V6h and C; with --cpu60, C also on
the follow-up 4 targets (no (target, seed) in both sources).
Checks, before anything else, and the analysis stops if one fails: one machine type (a2-ultragpu-1g) across every
A, Ah and AT session (sessions of two machine types are refused); check 1, exit codes; check 2 for A, as in the
main study (contrasts.cache_checks and contrasts.check2: 2a every fresh process compiled from an empty cache, 2b
every warm process loaded its source executable, then the warm rerun bit-identical to r1 in every output file,
mmCIF timestamp line ignored; FETCH=full on the warm parts brings the files).
Contrasts, paired and computed as in the main study, with the statistics of section 5 and the target-clustered
bootstrap: RC(A), W(A), SD(A); HW(G,A), HW(V6,A), HW(V5,A); HW(A,C) on the CPU targets (6, or 60 with --cpu60);
DU(A) (A against AT, as DU is G against GT); PR(A) (A against Ah), HWh(V6,A) (V6h against Ah) and HWh(A,C) (Ah
against C), read with the rule of section 11; HW minus RC for HW(G,A) and HW(V6,A) against the RC of each of the
two conditions (RC(G) and RC(V6) from the main study's seed pairs, RC(A) from the follow-up); Q3 with A and AT added
to the main study's conditions (V5, V6, G, GT, C; Ah stays out, as V6h and Gh do).
M6: cost per structure at the a2-ultragpu-1g list prices of both regions (us-central1 and europe-west4; on demand,
Spot and Flex-start; cloud/prices.csv), whatever the zone a run used; M7: memory per bucket (analysis/cost.py).
Outputs in --out: checks.csv, counts.csv, contrasts.csv, hw_minus_rc.csv, q3.csv, pairs_seed.csv, pairs_sample.csv,
cost_per_process.csv, cost_both_regions.csv, memory.csv, report.md.
"""
import argparse
import csv
import pathlib
import sys
import types

import numpy as np
import pandas as pd

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / 'analysis'))
import contrasts  # noqa: E402
import cost  # noqa: E402
import followup34 as f34  # noqa: E402

MACHINE_TYPE = 'a2-ultragpu-1g'
A_ARMS = ['A', 'Ah', 'AT']
REGIONS = ['us-central1', 'europe-west4']
CHECK2_SPECS = [('W(A)', 'W', ('A', 'r1'), ('A', 'w'), True)]
SPECS = [('RC(A)', 'RC', ('A', 'r1'), ('A', 'r2'), True), ('SD(A)', 'SD', ('A', 'r1'), ('A', 'r1'), True),
         ('HW(G,A)', 'HW', ('G', 'r1'), ('A', 'r1'), False), ('HW(V6,A)', 'HW', ('V6', 'r1'), ('A', 'r1'), False),
         ('HW(V5,A)', 'HW', ('V5', 'r1'), ('A', 'r1'), False), ('HW(A,C)', 'HW', ('A', 'r1'), ('C', 'r1'), False),
         ('DU(A)', 'DU', ('A', 'r1'), ('AT', 'r1'), True), ('PR(A)', 'PR', ('A', 'r1'), ('Ah', 'r1'), False),
         ('HWh(V6,A)', 'HWh', ('V6h', 'r1'), ('Ah', 'r1'), False), ('HWh(A,C)', 'HWh', ('Ah', 'r1'), ('C', 'r1'), False)]
HW_RC = ['HW(G,A)', 'HW(V6,A)']
MAIN_CONDITIONS = ['G', 'GT', 'V5', 'V6', 'V6h', 'C']
Q3_CONDITIONS = ['V5', 'V6', 'G', 'GT', 'C', 'A', 'AT']


def checks_first(a100, ok):
  """[check rows], and False if the analysis must stop. Raises if the A100 sessions mix machine types."""
  mt = f34.one_machine_type(a100, A_ARMS)
  rows = [{'check': 'machine type of every A, Ah and AT session', 'condition': ', '.join(A_ARMS), 'value': mt,
           'detail': 'pass' if mt == MACHINE_TYPE else f'FAIL: expected {MACHINE_TYPE}'}]
  rows += f34.exit_code_check(a100, A_ARMS, 'a100')
  _, sm, _, _ = f34.build(ok, CHECK2_SPECS)
  rows += contrasts.check2(a100, sm, contrasts.cache_checks(a100), 'A')
  return rows, all(r['detail'] == 'pass' for r in rows)


def q3(ok, args):
  return contrasts.q3(ok[ok.condition.isin(Q3_CONDITIONS)], args)


def both_regions(c, a100_prices):
  """Per process: cost per structure at each region's a2-ultragpu-1g price, and the median per arm and bucket."""
  rows = []
  for r in c.itertuples():
    row = {'session': r.session, 'platform': r.platform, 'target': r.target, 'bucket': r.bucket,
           'repetition': r.repetition, 'zone': r.zone, 'wall_per_structure_s': r.wall_per_structure_s}
    for region in REGIONS:
      for prov in cost.PROVISIONING:
        price = a100_prices.get((region, prov))
        row[f'usd_per_structure_{prov}_{region}'] = r.wall_per_structure_s / 3600 * price if price else np.nan
    rows.append(row)
  per = pd.DataFrame(rows)
  cols = [f'usd_per_structure_{p}_{g}' for g in REGIONS for p in cost.PROVISIONING]
  med = per.groupby(['platform', 'bucket', per.repetition == 'w'])[['wall_per_structure_s'] + cols].median().reset_index()
  return per, med.rename(columns={'repetition': 'warm'})


def a100_price_table():
  out = {}
  for r in csv.DictReader(open(REPO / 'cloud' / 'prices.csv')):
    if r['platform'] == 'a100' and r['machine_type'] == MACHINE_TYPE and r['usd_per_hour']:
      out[(r['region'], r['provisioning'])] = float(r['usd_per_hour'])
  return out


def analyse(ok, args, primary_dir):
  sp, sm, cons, counts = f34.build(ok, CHECK2_SPECS + SPECS)
  con = contrasts.summarize(sp, sm, cons, args)
  rc_rows, rc_cons = f34.main_rc_seed_pairs(primary_dir, ['RC(G)', 'RC(V6)'])
  hwrc = contrasts.hw_minus_rc(pd.concat([sp, rc_rows], ignore_index=True), {**cons, **rc_cons}, args)
  hwrc = hwrc[hwrc.hardware.isin(HW_RC)].reset_index(drop=True)
  q3rows, q3answer = q3(ok, args)
  return counts, con, hwrc, q3rows, q3answer, sp, sm


def report(out, checks, con, hwrc, q3rows, q3answer, med, m, args):
  L = ['# Follow-up 3: the A100 arms', '',
       'notes/analysis_plan.md section 10, rows of 2026-10-16; follow-up outside the pre-registered study. Machine '
       f'type {MACHINE_TYPE} (one A100 80 GB). Code: analysis/a100.py (bootstrap {args.replicates} replicates, seed '
       f'{args.seed}).', '', '## Checks (before any result)', '', '| check | condition | value | detail |', '|---|---|---|---|']
  L += [f'| {r["check"]} | {r["condition"]} | {r["value"]} | {r["detail"]} |' for r in checks]
  if con is None:
    L += ['', 'A check failed: no contrast was computed.']
    (out / 'report.md').write_text('\n'.join(L) + '\n')
    return
  L += ['', '## Contrasts', ''] + f34.contrast_lines(con)
  L += ['', '## HW minus RC (against the RC of each condition)', '', '| hardware | recompilation | reference | statistic | '
        'targets | estimate | 95% CI |', '|---|---|---|---|---|---|---|']
  L += [f'| {r.hardware} | {r.recompilation} | {r.reference} | {r.statistic} | {r.n_targets} | {f34.pct(r.estimate)} | '
        f'{f34.pct(r.ci_low)} to {f34.pct(r.ci_high)} |' for r in hwrc.itertuples()]
  L += ['', f'## Q3 with A and AT (answer: {q3answer or "not reached"})', '', '| n seeds | condition | statistic | targets | '
        'estimate | 95% CI |', '|---|---|---|---|---|---|']
  L += [f'| {r.n_seeds} | {r.condition} | {r.statistic} | {r.n_targets} | {f34.pct(r.estimate)} | {f34.pct(r.ci_low)} to '
        f'{f34.pct(r.ci_high)} |' for r in q3rows.itertuples()]
  L += ['', '## M6: median USD per structure at the list prices of both regions', '',
        '| arm | bucket | warm | wall s | ' + ' | '.join(f'{p} {g}' for g in REGIONS for p in cost.PROVISIONING) + ' |',
        '|---|---|---|---|' + '---|' * 6]
  for r in med.to_dict('records'):   # records: the region columns hold a "-", which itertuples would rename
    L.append(f'| {r["platform"]} | {r["bucket"]} | {r["warm"]} | {f34.num(r["wall_per_structure_s"])} | ' + ' | '.join(
        f34.num(r[f'usd_per_structure_{p}_{g}']) for g in REGIONS for p in cost.PROVISIONING) + ' |')
  L += ['', '## M7: memory per bucket (GiB)', '', '| arm | bucket | ok/runs | compiled peak | allocator peak in use | '
        'device limit |', '|---|---|---|---|---|---|']
  L += [f'| {r.platform} | {r.bucket} | {r.succeeded}/{r.processes} | {f34.num(r.compiled_peak_gib)} | '
        f'{f34.num(r.alloc_peak_in_use_gib)} | {f34.num(r.device_limit_gib)} |' for r in m.itertuples()]
  (out / 'report.md').write_text('\n'.join(L) + '\n')


def main():
  ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
  ap.add_argument('--samples', required=True, help='collect.py table of the a100 sessions')
  ap.add_argument('--cpu60', help='collect.py table of the cpu60 sessions (C on the other 54 targets)')
  ap.add_argument('--main', default=str(REPO / 'results' / 'analysis' / 'main_20261014' / 'primary_v2'))
  ap.add_argument('--out', required=True)
  ap.add_argument('--replicates', type=int, default=10000)
  ap.add_argument('--seed', type=int, default=2026)
  args = ap.parse_args()
  out = pathlib.Path(args.out)
  out.mkdir(parents=True, exist_ok=True)
  main_d = f34.load_samples(pathlib.Path(args.main) / 'samples.csv', ['main'])
  a100 = f34.load_samples(args.samples, ['a100'])
  a100 = a100[a100.condition.isin(A_ARMS)]
  frames = [main_d[main_d.condition.isin(MAIN_CONDITIONS)], a100]
  if args.cpu60:
    c60 = f34.load_samples(args.cpu60, ['cpu60'])
    f34.disjoint_units(main_d, c60, 'C')
    frames.append(c60[c60.condition == 'C'])
  ok = f34.union(*frames)
  checks, passed = checks_first(a100, ok)
  pd.DataFrame(checks).to_csv(out / 'checks.csv', index=False)
  if not passed:
    report(out, checks, None, None, None, None, None, None, args)
    raise SystemExit('a check failed (checks.csv); no contrast is computed')
  counts, con, hwrc, q3rows, q3answer, sp, sm = analyse(ok, args, args.main)
  path = out / 'samples_a100_used.csv'
  a100.to_csv(path, index=False)
  p = cost.processes(types.SimpleNamespace(samples=path, all_settings=False, labels=None, include=None, exclude=None))
  c = cost.cost_table(p, cost.load_prices(), cost.batch_sizes('subset'))
  per, med = both_regions(c, a100_price_table())
  m = cost.memory_table(p)
  for name, df in (('counts', counts), ('contrasts', con), ('hw_minus_rc', hwrc), ('q3', q3rows), ('pairs_seed', sp),
                   ('pairs_sample', sm), ('cost_per_process', per), ('cost_both_regions', med), ('memory', m)):
    df.to_csv(out / f'{name}.csv', index=False)
  report(out, checks, con, hwrc, q3rows, q3answer, med, m, args)
  print((out / 'report.md').read_text())
  return 0


if __name__ == '__main__':
  sys.exit(main())
