"""Follow-up 4, the CPU arm on every target (notes/analysis_plan.md section 10, row of 2026-10-16; label cpu60;
reported as a follow-up, outside the pre-registered study).

    analysis/.venv/bin/python analysis/collect.py --labels cpu60 --out results/analysis/cpu60_<date> --jobs 8
    analysis/.venv/bin/python analysis/cpu60.py --samples results/analysis/cpu60_<date>/samples.csv \\
        --out results/analysis/cpu60_<date> [--main results/analysis/main_20261014/primary_v2] [--replicates 10000]

C on 60 targets: the main study's C runs (6 targets, seeds 1 and 2) and the cpu60 runs (the other 54, seeds 1 and
2); no (target, seed) may appear in both (an error). The other conditions are the main study's.
Checks: 1, exit codes of the cpu60 processes; the machine type of every C session (one type).
Contrasts on the 60 targets, paired and computed as in the main study (analysis/followup34.build, the statistics of
section 5 with the target-clustered bootstrap): HW(G,C) (a primary pair of section 5), HW(V5,C), HW(V6,C),
HW(GT,C), HW(Gd,C), HWh(G,C) (Gh against C, section 11) and SD(C); HW minus RC against the main study's RC seed
pairs (RC(G), RC(V5), RC(V6), RC(Gd); C runs r1 only). Next to them, the 6-target values of primary_v2, unchanged.
M6 and M7 for C at every token bucket (analysis/cost.py on the C rows of both sources; prices: cloud/prices.csv).
Outputs in --out: checks.csv, counts.csv, contrasts.csv, six_vs_sixty.csv, hw_minus_rc.csv, pairs_seed.csv,
pairs_sample.csv, cost_per_process.csv, cost_summary.csv, memory.csv, report.md.
"""
import argparse
import pathlib
import sys
import types

import pandas as pd

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / 'analysis'))
import contrasts  # noqa: E402
import cost  # noqa: E402
import followup34 as f34  # noqa: E402

SPECS = [('HW(G,C)', 'HW', ('G', 'r1'), ('C', 'r1'), True), ('HW(V5,C)', 'HW', ('V5', 'r1'), ('C', 'r1'), False),
         ('HW(V6,C)', 'HW', ('V6', 'r1'), ('C', 'r1'), False), ('HW(GT,C)', 'HW', ('GT', 'r1'), ('C', 'r1'), False),
         ('HW(Gd,C)', 'HW', ('Gd', 'r1'), ('C', 'r1'), False), ('HWh(G,C)', 'HWh', ('Gh', 'r1'), ('C', 'r1'), False),
         ('SD(C)', 'SD', ('C', 'r1'), ('C', 'r1'), True)]
NAMES = [s[0] for s in SPECS]
RC_MAIN = ['RC(G)', 'RC(V5)', 'RC(V6)', 'RC(Gd)']
CONDITIONS = ['G', 'V5', 'V6', 'GT', 'Gd', 'Gh', 'C']
STATS = ['flip_top1', 'flip_success', 'flip_matters', 'm4_ligand_rmsd_median', 'm4_ca_rmsd_global_median',
         'm5_ranking_score_median', 'bit_identical_fraction']


def checks_first(main, c60):
  """Checks, before anything else: exit codes, one machine type, the C units of the two sources disjoint (the last
  two raise)."""
  checks = f34.exit_code_check(c60, ['C'], 'cpu60')
  n_main, n_new = f34.disjoint_units(main, c60, 'C')
  mt = f34.one_machine_type(pd.concat([main[main.condition == 'C'], c60[c60.condition == 'C']]), ['C'])
  checks.append({'check': 'C machine type (main study and cpu60 sessions)', 'condition': 'C', 'value': mt,
                 'detail': 'pass'})
  checks.append({'check': 'C (target, seed) units: main study + cpu60, none in both', 'condition': 'C',
                 'value': f'{n_main} + {n_new}', 'detail': 'pass'})
  return checks


def analyse(main, c60, args, primary_dir):
  """main, c60: per-sample tables (followup34.load_samples), after checks_first passed. Returns the output tables."""
  ok = f34.union(main[main.condition.isin(CONDITIONS)], c60[c60.condition == 'C'])
  sp, sm, cons, counts = f34.build(ok, SPECS)
  con = contrasts.summarize(sp, sm, cons, args)
  rc_rows, rc_cons = f34.main_rc_seed_pairs(primary_dir, RC_MAIN)
  hwrc = contrasts.hw_minus_rc(pd.concat([sp, rc_rows], ignore_index=True), {**cons, **rc_cons}, args)
  old = pd.read_csv(pathlib.Path(primary_dir) / 'contrasts' / 'contrasts.csv')
  six = f34.side_by_side(con, old, NAMES, STATS)
  cpu = pd.concat([main[main.condition == 'C'], c60[c60.condition == 'C']], ignore_index=True)
  return counts, con, six, hwrc, sp, sm, cpu


def m6_m7(cpu, out):
  """M6 and M7 for C (analysis/cost.py, on the C rows of both sources)."""
  path = out / 'samples_c.csv'
  cpu.to_csv(path, index=False)
  p = cost.processes(types.SimpleNamespace(samples=path, all_settings=False, labels=None, include=None, exclude=None))
  c = cost.cost_table(p, cost.load_prices(), cost.batch_sizes('subset'))
  return c, cost.summary_table(c), cost.memory_table(p)


def report(out, checks, con, six, hwrc, s, m, args):
  L = ['# Follow-up 4: the CPU arm on every target', '',
       'notes/analysis_plan.md section 10, row of 2026-10-16; follow-up outside the pre-registered study. Code: '
       f'analysis/cpu60.py (bootstrap {args.replicates} replicates, seed {args.seed}).', '', '## Checks', '',
       '| check | condition | value | detail |', '|---|---|---|---|']
  L += [f'| {r["check"]} | {r["condition"]} | {r["value"]} | {r["detail"]} |' for r in checks]
  L += ['', '## Contrasts on 60 targets', ''] + f34.contrast_lines(con)
  L += ['', '## Next to the 6-target values of the main study (primary_v2, unchanged)', '',
        '| contrast | statistic | 60 targets | main study |', '|---|---|---|---|']
  for r in six.itertuples():
    f = f34.pct if r.statistic.startswith(('flip', 'bit_identical')) else f34.num
    L.append(f'| {r.contrast} | {r.statistic} | {f(r.new_estimate)} ({f(r.new_ci_low)} to {f(r.new_ci_high)}; '
             f'{r.new_n_targets} targets) | {f(r.main_estimate)} ({f(r.main_ci_low)} to {f(r.main_ci_high)}; '
             f'{r.main_n_targets} targets) |')
  L += ['', '## HW minus RC (main-study RC seed pairs)', '', '| hardware | recompilation | reference | statistic | '
        'targets | estimate | 95% CI |', '|---|---|---|---|---|---|---|']
  L += [f'| {r.hardware} | {r.recompilation} | {r.reference} | {r.statistic} | {r.n_targets} | {f34.pct(r.estimate)} | '
        f'{f34.pct(r.ci_low)} to {f34.pct(r.ci_high)} |' for r in hwrc.itertuples()]
  L += ['', '## M6: C per bucket (median USD per structure)', '', '| bucket | cache | processes | wall s | on demand | Spot |',
        '|---|---|---|---|---|---|']
  L += [f'| {r.bucket} | {r.cache} | {r.processes} | {f34.num(r.median_wall_s)} | {f34.num(r.median_usd_per_structure_standard)} | '
        f'{f34.num(r.median_usd_per_structure_spot)} |' for r in s.itertuples()]
  L += ['', '## M7: C per bucket (max RSS, GiB)', '', '| bucket | ok/runs | max RSS |', '|---|---|---|']
  L += [f'| {r.bucket} | {r.succeeded}/{r.processes} | {f34.num(r.max_rss_gib)} |' for r in m.itertuples()]
  (out / 'report.md').write_text('\n'.join(L) + '\n')


def main():
  ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
  ap.add_argument('--samples', required=True, help='collect.py table of the cpu60 sessions')
  ap.add_argument('--main', default=str(REPO / 'results' / 'analysis' / 'main_20261014' / 'primary_v2'))
  ap.add_argument('--out', required=True)
  ap.add_argument('--replicates', type=int, default=10000)
  ap.add_argument('--seed', type=int, default=2026)
  args = ap.parse_args()
  out = pathlib.Path(args.out)
  out.mkdir(parents=True, exist_ok=True)
  main_d = f34.load_samples(pathlib.Path(args.main) / 'samples.csv', ['main'])
  c60 = f34.load_samples(args.samples, ['cpu60'])
  checks = checks_first(main_d, c60)
  pd.DataFrame(checks).to_csv(out / 'checks.csv', index=False)
  if any(r['detail'] != 'pass' for r in checks):
    raise SystemExit('a check failed (checks.csv); no contrast is computed')
  counts, con, six, hwrc, sp, sm, cpu = analyse(main_d, c60, args, args.main)
  c, s, m = m6_m7(cpu, out)
  for name, df in (('checks', pd.DataFrame(checks)), ('counts', counts), ('contrasts', con), ('six_vs_sixty', six),
                   ('hw_minus_rc', hwrc), ('pairs_seed', sp), ('pairs_sample', sm), ('cost_per_process', c),
                   ('cost_summary', s), ('memory', m)):
    df.to_csv(out / f'{name}.csv', index=False)
  report(out, checks, con, six, hwrc, s, m, args)
  print((out / 'report.md').read_text())
  return 0


if __name__ == '__main__':
  sys.exit(main())
