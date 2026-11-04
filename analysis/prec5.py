"""Follow-up 12, the precision-matched arms to five seeds (notes/analysis_plan.md section 10, row of 2026-11-03;
label prec5; reported as a follow-up, outside the pre-registered study, whatever it shows).

    analysis/.venv/bin/python analysis/prec5.py --samples results/analysis/prec5_<date>/samples.csv \\
        --out results/analysis/prec5_<date> [--main results/analysis/main_20261014/primary_v2]

Conditions: V6h and Gh (section 11, JAX_DEFAULT_MATMUL_PRECISION=highest), r1: seeds 1 and 2 from the main-study
sessions (primary_v2/samples.csv, unchanged), seeds 3 to 5 from the prec5 sessions; V6 and G r1, seeds 1 to 5, from the
main study.
Checks, before anything else (the analysis stops if one fails): exit codes of the prec5 runs; their seeds are 3 to 5
only; no (condition, target, seed) in both sources; on seeds 1 and 2 the section 11 contrasts computed here equal the
main study's (its contrasts.csv), so the two-seed results stay as they are. A condition short of a (target, seed) is
reported as incomplete; nothing is imputed.
Contrasts, as section 11 with the statistics of section 5 and the target-clustered bootstrap (10,000 replicates, seed
2026), on seeds 1 to 5: PR(V6) (V6 against V6h), PR(G) (G against Gh), HWh(V6,G) (V6h against Gh), next to HW(V6,G)
(V6 against G) on the same seeds; HWh(G,C) is not computed (C has two seeds). HW minus RC (contrasts.hw_minus_rc)
for HW(V6,G) and HWh(V6,G), against RC(V6) (RC(P)) and RC(G) (RC(Q), secondary), the main study's RC seed pairs.
Reading rule: section 11's, unchanged: a hardware difference that shrinks to the recompilation level under matched
precision is attributed to the default matmul precision, what remains to the platform. Read here as the main study's
report reads it: HWh(V6,G) minus RC(V6) with an interval above 0 does not shrink to the RC level (attributed to the
platform); an interval that contains 0 or lies below 0 does (attributed to the default precision). The two-seed
results (main study) are reported next to the five-seed ones, unchanged. Every M1 result also on the 39 targets
released after 2021-09-30.
Outputs in --out: checks.csv, completeness.csv, contrasts.csv, hw_minus_rc.csv, reading_rule.csv, pairs_seed.csv,
pairs_sample.csv, counts.csv, report.md.
"""
import argparse
import pathlib
import sys

import pandas as pd

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / 'analysis'))
import contrasts  # noqa: E402
import followup34 as f34  # noqa: E402

NEW_SEEDS = [3, 4, 5]
SPECS = [('PR(V6)', 'PR', ('V6', 'r1'), ('V6h', 'r1'), False), ('PR(G)', 'PR', ('G', 'r1'), ('Gh', 'r1'), False),
         ('HWh(V6,G)', 'HWh', ('V6h', 'r1'), ('Gh', 'r1'), False), ('HW(V6,G)', 'HW', ('V6', 'r1'), ('G', 'r1'), True)]
SECTION11 = ['PR(V6)', 'PR(G)', 'HWh(V6,G)']
M1_STATS = ['flip_success', 'flip_matters']
CUTOFF = '2021-09-30'


def completeness(ok, targets):
  rows = []
  for c in ('V6h', 'Gh'):
    units = ok[(ok.condition == c) & (ok.repetition == 'r1')].groupby(['target', 'seed']).size()
    have = int((units == 5).sum())
    rows.append({'condition': c, 'planned_units': len(targets) * 5, 'units_with_5_samples': have,
                 'status': 'complete' if have == len(targets) * 5 else 'incomplete'})
  return pd.DataFrame(rows)


def two_seed_check(ok, main_con, args):
  """The section 11 contrasts on seeds 1 and 2, computed from the union, must equal the main study's."""
  sp, _, cons, _ = f34.build(ok[ok.seed.isin([1, 2])], SPECS[:3])
  rows = []
  for n in SECTION11:
    for st in contrasts.FLIP_STATS:
      a = sp[sp.contrast == n][st].astype(float).mean() if n in cons else float('nan')
      b = main_con[(main_con.contrast == n) & (main_con.statistic == st)].estimate
      same = len(b) == 1 and abs(a - float(b.iloc[0])) < 1e-12
      rows.append({'check': 'seeds 1 and 2 equal the main study', 'condition': f'{n} {st}',
                   'value': f'{a:.4f} against {float(b.iloc[0]) if len(b) else float("nan"):.4f}',
                   'detail': 'pass' if same else 'FAIL'})
  return rows


def reading(hw, label, seeds):
  rows = []
  for st in hw.statistic.unique():
    r = hw[(hw.hardware == 'HWh(V6,G)') & (hw.recompilation == 'RC(V6)') & (hw.statistic == st)]
    if r.empty or r.ci_low.iloc[0] != r.ci_low.iloc[0]:
      read = 'not computed'
    elif r.ci_low.iloc[0] > 0:
      read = 'does not shrink to the RC level: attributed to the platform'
    else:
      read = 'shrinks to the RC level: attributed to the default matmul precision'
    rows.append({'targets': label, 'seeds': seeds, 'statistic': st, 'quantity': 'HWh(V6,G) minus RC(V6)', 'reading': read})
  return pd.DataFrame(rows)


def main():
  ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
  ap.add_argument('--samples', required=True)
  ap.add_argument('--main', default=str(REPO / 'results' / 'analysis' / 'main_20261014' / 'primary_v2'))
  ap.add_argument('--targets', default=str(REPO / 'targets' / 'posebusters_subset.csv'))
  ap.add_argument('--out', required=True)
  ap.add_argument('--replicates', type=int, default=10000)
  ap.add_argument('--seed', type=int, default=2026)
  args = ap.parse_args()
  out = pathlib.Path(args.out)
  out.mkdir(parents=True, exist_ok=True)
  tt = pd.read_csv(args.targets, dtype=str)
  targets, post = sorted(tt.pdb_id), sorted(tt[tt.rcsb_initial_release.str[:10] > CUTOFF].pdb_id)
  new = f34.load_samples(args.samples, ['prec5'])
  new = new[new.condition.isin(['V6h', 'Gh'])]
  main = f34.load_samples(pathlib.Path(args.main) / 'samples.csv', ['main'])
  main = main[main.condition.isin(['V6', 'G', 'V6h', 'Gh']) & (main.repetition == 'r1')]
  chk = f34.exit_code_check(new, ['V6h', 'Gh'], 'prec5')
  seeds = sorted(new.seed.unique())
  chk.append({'check': 'seeds of the prec5 sessions are 3 to 5', 'condition': 'V6h, Gh', 'value': str(seeds),
              'detail': 'pass' if set(seeds) <= set(NEW_SEEDS) and seeds else 'FAIL'})
  for c in ('V6h', 'Gh'):
    try:
      a, b = f34.disjoint_units(main, new, c)
      chk.append({'check': 'no (condition, target, seed) in both sources', 'condition': c,
                  'value': f'main {a}, prec5 {b}', 'detail': 'pass'})
    except SystemExit as e:
      chk.append({'check': 'no (condition, target, seed) in both sources', 'condition': c, 'value': str(e)[:200],
                  'detail': 'FAIL'})
  for stage in ('sources', 'two seeds'):
    if stage == 'two seeds':
      ok = f34.union(main, new)
      main_con = pd.read_csv(pathlib.Path(args.main) / 'contrasts' / 'contrasts.csv')
      chk += two_seed_check(ok, main_con, args)
    pd.DataFrame(chk).to_csv(out / 'checks.csv', index=False)
    if any(r['detail'] != 'pass' for r in chk):
      raise SystemExit('a check failed (checks.csv); nothing is compared')

  comp = completeness(ok, targets)
  bad = set(comp[comp.status != 'complete'].condition)
  sp, sm, cons, counts = f34.build(ok, SPECS)
  rc, rc_cons = f34.main_rc_seed_pairs(args.main, ['RC(V6)', 'RC(G)'])
  main_hw = pd.read_csv(pathlib.Path(args.main) / 'contrasts' / 'hw_minus_rc.csv')
  res = {'contrasts': [], 'hw_minus_rc': [], 'reading_rule': []}
  n_all = f'all {len(targets)}'
  for label, ts, stats in ((n_all, targets, contrasts.FLIP_STATS), (f'post-cutoff {len(post)}', post, M1_STATS)):
    con = contrasts.summarize(sp[sp.target.isin(ts)], sm[sm.target.isin(ts)], cons, args)
    if label != n_all:
      con = con[con.statistic.isin(stats)]
    allc = dict(cons, **rc_cons)
    hw = contrasts.hw_minus_rc(pd.concat([sp, rc])[lambda x: x.target.isin(ts)], allc, args)
    hw = hw[hw.statistic.isin(stats)]
    res['contrasts'].append(con.assign(targets=label, seeds='1 to 5'))
    res['hw_minus_rc'].append(hw.assign(targets=label, seeds='1 to 5'))
    res['reading_rule'].append(reading(hw, label, '1 to 5'))
    if label == n_all:   # the two-seed results of the main study, unchanged, next to them
      two = main_con[main_con.contrast.isin(SECTION11) & main_con.statistic.isin(stats)]
      res['contrasts'].append(two.assign(targets=label, seeds='1 and 2 (main study)'))
      h2 = main_hw[(main_hw.hardware == 'HWh(V6,G)') & main_hw.statistic.isin(stats)]
      res['hw_minus_rc'].append(h2.assign(targets=label, seeds='1 and 2 (main study)'))
      res['reading_rule'].append(reading(h2, label, '1 and 2 (main study)'))
  res = {k: pd.concat(v, ignore_index=True) for k, v in res.items()}
  res['contrasts']['incomplete_arm'] = res['contrasts'].apply(
      lambda r: ', '.join(sorted(bad & {'V6h', 'Gh'})) if r.seeds == '1 to 5' and r.contrast in SECTION11 else '', axis=1)
  for name, df in (('completeness', comp), ('counts', counts), ('pairs_seed', sp), ('pairs_sample', sm), *res.items()):
    df.to_csv(out / f'{name}.csv', index=False)
  pct = f34.pct
  L = ['# Follow-up 12: the precision-matched arms to five seeds', '', '## Reading rule (section 11, unchanged)', '']
  L += [f'- {r.targets}, seeds {r.seeds}, {r.statistic}: {r.quantity}: {r.reading}'
        for r in res['reading_rule'].itertuples()]
  L += ['', '## Completeness', '', '| condition | planned | with 5 samples | status |', '|---|---|---|---|']
  L += [f'| {r.condition} | {r.planned_units} | {r.units_with_5_samples} | {r.status} |' for r in comp.itertuples()]
  L += ['', '## HW and HWh minus RC', '', '| targets | seeds | hardware | minus | statistic | estimate | 95% CI |',
        '|---|---|---|---|---|---|---|']
  L += [f'| {r.targets} | {r.seeds} | {r.hardware} | {r.recompilation} | {r.statistic} | {pct(r.estimate)} | '
        f'{pct(r.ci_low)} to {pct(r.ci_high)} |' for r in res['hw_minus_rc'].itertuples()]
  L += ['', '## Contrasts (flip rates)', '', '| targets | seeds | contrast | statistic | estimate | 95% CI |',
        '|---|---|---|---|---|---|']
  c = res['contrasts']
  L += [f'| {r.targets} | {r.seeds} | {r.contrast} | {r.statistic} | {pct(r.estimate)} | {pct(r.ci_low)} to '
        f'{pct(r.ci_high)} |' for r in c[c.statistic.isin(contrasts.FLIP_STATS)].itertuples()]
  (out / 'report.md').write_text('\n'.join(L) + '\n')
  print((out / 'report.md').read_text())
  return 0


if __name__ == '__main__':
  sys.exit(main())
