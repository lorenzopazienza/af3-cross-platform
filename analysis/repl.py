"""Follow-up 11, replication on new targets (notes/analysis_plan.md section 10, row of 2026-11-03; label repl;
reported as a follow-up, outside the pre-registered study, whatever it shows).

    analysis/.venv/bin/python analysis/repl.py --samples results/analysis/repl_<date>/samples.csv \\
        --out results/analysis/repl_<date> [--main results/analysis/main_20261014/primary_v2]
        [--manifest inputs/repl/manifest.csv] [--targets targets/repl/repl_subset.csv]

Targets: the rows of the follow-up 11 input manifest (every eligible PoseBusters V2 complex not among the 60, less any
whose input could not be built, section 10); release dates from targets/repl/repl_subset.csv (RCSB, as the main
study). Conditions: V5, V6 and G of label repl (the main-study settings), r1 and r2, seeds 1 to 5.
Checks, before anything else (the analysis stops if one fails): exit codes; one machine type per condition; every
target is a manifest target and none is among the 60 of the main study. An arm short of its (target, seed) units is
reported as incomplete; nothing is imputed.
Contrasts, as section 5 (analysis/followup34.build and contrasts.summarize, target-clustered bootstrap, 10,000
replicates, seed 2026): the primary pairs HW(V5,V6) and HW(V6,G); RC(V5), RC(V6), RC(G); SD on each platform. HW
minus RC (contrasts.hw_minus_rc): against RC of the pair's first platform (RC(P), section 5's reference) and of its
second (RC(Q), secondary).
Reading rule (the row, literally): the main result replicates for a pair if the 95% interval of HW minus RC for the
top-1 flip rate lies above 0 (the headline); the same rule for the success flip rate and for flips that matter, as
secondary. RC is section 5's reference RC(P); the reading against RC(Q) is reported next to it. Every M1 result (the
success flip rate, flips that matter, the success rates) is also given on the targets released after 2021-09-30.
Next to the main study's (exploratory): the replication's flip rates and HW minus RC beside the main study's,
overall (the main study's own outputs, primary_v2) and in bucket 512 (computed here from its seed pairs, on its
targets of that bucket). Pooled, secondary and descriptive: the replication's and the main study's seed pairs together
(132 targets if every input was built), flip rates and HW minus RC.
Outputs in --out: checks.csv, completeness.csv, contrasts.csv, hw_minus_rc.csv, reading_rule.csv, success_rates.csv,
next_to_main.csv, pooled.csv, pairs_seed.csv, pairs_sample.csv, counts.csv, report.md.
"""
import argparse
import pathlib
import sys

import numpy as np
import pandas as pd

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / 'analysis'))
import contrasts  # noqa: E402
import followup34 as f34  # noqa: E402

CONDS = ['V5', 'V6', 'G']
SPECS = [('RC(V5)', 'RC', ('V5', 'r1'), ('V5', 'r2'), True), ('RC(V6)', 'RC', ('V6', 'r1'), ('V6', 'r2'), True),
         ('RC(G)', 'RC', ('G', 'r1'), ('G', 'r2'), True),
         ('HW(V5,V6)', 'HW', ('V5', 'r1'), ('V6', 'r1'), True), ('HW(V6,G)', 'HW', ('V6', 'r1'), ('G', 'r1'), True),
         ('SD(V5)', 'SD', ('V5', 'r1'), ('V5', 'r1'), True), ('SD(V6)', 'SD', ('V6', 'r1'), ('V6', 'r1'), True),
         ('SD(G)', 'SD', ('G', 'r1'), ('G', 'r1'), True)]
PAIRS = ['HW(V5,V6)', 'HW(V6,G)']
NAMES = [s[0] for s in SPECS]
HEADLINE = 'flip_top1'
M1_STATS = ['flip_success', 'flip_matters']
CUTOFF = '2021-09-30'
BUCKET = 512


def flip_summary(sp, cons, args):
  """The flip-rate rows of contrasts.summarize (same estimate and bootstrap), without the per-sample statistics."""
  rows = []
  for name, (kind, primary) in cons.items():
    s = sp[sp.contrast == name]
    if s.empty:
      continue
    ts = sorted(s.target.unique())
    for st in contrasts.FLIP_STATS:
      groups = [s[s.target == t][st].astype(float).values for t in ts]
      lo, hi = contrasts.boot_ci(groups, contrasts.mean, np.random.default_rng(args.seed), args.replicates)
      rows.append({'contrast': name, 'kind': kind, 'role': contrasts.role(kind, primary), 'statistic': st,
                   'n': len(s), 'n_targets': len(ts), 'estimate': contrasts.mean(s[st].astype(float).values),
                   'ci_low': lo, 'ci_high': hi})
  return pd.DataFrame(rows)


def completeness(ok, targets):
  rows = []
  for c in CONDS:
    for rep in ('r1', 'r2'):
      units = ok[(ok.condition == c) & (ok.repetition == rep)].groupby(['target', 'seed']).size()
      have = int((units == 5).sum())
      rows.append({'condition': c, 'repetition': rep, 'planned_units': len(targets) * 5, 'units_with_5_samples': have,
                   'status': 'complete' if have == len(targets) * 5 else 'incomplete'})
  return pd.DataFrame(rows)


def success_rates(ok, targets, args):
  """Success rate (M1) of the per-seed top-1, per condition and repetition, with the target bootstrap."""
  rows = []
  for (c, rep), g in ok[ok.target.isin(targets)].groupby(['condition', 'repetition']):
    per = {}
    for (t, s), u in g.groupby(['target', 'seed']):
      per.setdefault(t, []).append(float(contrasts.top(u.set_index('sample').sort_index())[1]))
    groups = [np.array(v) for _, v in sorted(per.items())]
    lo, hi = contrasts.boot_ci(groups, contrasts.mean, np.random.default_rng(args.seed), args.replicates)
    rows.append({'condition': c, 'repetition': rep, 'n_targets': len(groups),
                 'success_rate': contrasts.mean(np.concatenate(groups)) if groups else np.nan, 'ci_low': lo, 'ci_high': hi})
  return pd.DataFrame(rows)


def rule(hw, stats, label, bad):
  rows = []
  for pair in PAIRS:
    for st in stats:
      r = hw[(hw.hardware == pair) & (hw.statistic == st)].set_index('reference')
      if 'RC(P)' not in r.index or r.loc['RC(P)', 'ci_low'] != r.loc['RC(P)', 'ci_low']:
        read, q = 'not computed', ''
      else:
        read = 'replicates' if r.loc['RC(P)', 'ci_low'] > 0 else 'does not replicate (interval not above 0)'
        q = ('above 0' if r.loc['RC(Q), secondary', 'ci_low'] > 0 else 'not above 0') \
            if 'RC(Q), secondary' in r.index else ''
      first, second = pair[3:-1].split(',')
      rows.append({'targets': label, 'pair': pair, 'statistic': st,
                   'role': 'headline' if st == HEADLINE else 'secondary', 'reference': f'RC({first})',
                   'answer': read, 'against_rc_of_second': f'against RC({second}): {q}' if q else '',
                   'incomplete_arms': ', '.join(sorted(bad & {first, second}))})
  return pd.DataFrame(rows)


def main():
  ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
  ap.add_argument('--samples', required=True)
  ap.add_argument('--main', default=str(REPO / 'results' / 'analysis' / 'main_20261014' / 'primary_v2'))
  ap.add_argument('--manifest', default=str(REPO / 'inputs' / 'repl' / 'manifest.csv'))
  ap.add_argument('--targets', default=str(REPO / 'targets' / 'repl' / 'repl_subset.csv'))
  ap.add_argument('--main_targets', default=str(REPO / 'targets' / 'posebusters_subset.csv'))
  ap.add_argument('--main_manifest', default=str(REPO / 'inputs' / 'manifest.csv'))
  ap.add_argument('--out', required=True)
  ap.add_argument('--replicates', type=int, default=10000)
  ap.add_argument('--seed', type=int, default=2026)
  args = ap.parse_args()
  out = pathlib.Path(args.out)
  out.mkdir(parents=True, exist_ok=True)
  man = pd.read_csv(args.manifest, dtype=str)
  tt = pd.read_csv(args.targets, dtype=str).set_index('pdb_id')
  targets = sorted(man.pdb_id)
  post = sorted(t for t in targets if tt.loc[t, 'rcsb_initial_release'][:10] > CUTOFF)
  b512 = sorted(man[man.bucket.astype(int) == BUCKET].pdb_id)
  mt = pd.read_csv(args.main_targets, dtype=str)
  main_targets, main_post = sorted(mt.pdb_id), sorted(mt[mt.rcsb_initial_release.str[:10] > CUTOFF].pdb_id)
  mm = pd.read_csv(args.main_manifest, dtype=str)
  main_b512 = sorted(mm[mm.bucket.astype(int) == BUCKET].pdb_id)

  d = f34.load_samples(args.samples, ['repl'])
  d = d[d.condition.isin(CONDS)]
  chk = f34.exit_code_check(d, CONDS, 'repl')
  for c in CONDS:
    types = sorted(d[d.condition == c].machine_type.dropna().astype(str).unique())
    chk.append({'check': 'one machine type per condition', 'condition': c, 'value': ', '.join(types),
                'detail': 'pass' if len(types) == 1 else f'FAIL: {len(types)} machine types'})
  extra, main60 = sorted(set(d.target) - set(targets)), sorted(set(d.target) & set(main_targets))
  chk.append({'check': 'every target is in the follow-up 11 manifest', 'condition': ', '.join(CONDS),
              'value': f'{len(set(d.target))} targets', 'detail': 'pass' if not extra else f'FAIL: {extra[:5]}'})
  chk.append({'check': 'no target among the 60 of the main study', 'condition': ', '.join(CONDS),
              'value': f'{len(main60)}', 'detail': 'pass' if not main60 else f'FAIL: {main60[:5]}'})
  pd.DataFrame(chk).to_csv(out / 'checks.csv', index=False)
  if any(r['detail'] != 'pass' for r in chk):
    raise SystemExit('a check failed (checks.csv); nothing is compared')

  ok = f34.union(d)
  comp = completeness(ok, targets)
  bad = set(comp[comp.status != 'complete'].condition)
  sp, sm, cons, counts = f34.build(ok, SPECS)
  n_all = f'all {len(targets)}'
  sets = ((n_all, targets, contrasts.FLIP_STATS), (f'post-cutoff {len(post)}', post, M1_STATS))
  res = {'contrasts': [], 'hw_minus_rc': [], 'reading_rule': [], 'success_rates': []}
  for label, ts, stats in sets:
    con = contrasts.summarize(sp[sp.target.isin(ts)], sm[sm.target.isin(ts)], cons, args)
    if label != n_all:
      con = con[con.statistic.isin(stats)]
    hw = contrasts.hw_minus_rc(sp[sp.target.isin(ts)], cons, args)
    hw = hw[hw.statistic.isin(stats)]
    res['contrasts'].append(con.assign(targets=label))
    res['hw_minus_rc'].append(hw.assign(targets=label))
    res['reading_rule'].append(rule(hw, stats, label, bad))
    res['success_rates'].append(success_rates(ok, ts, args).assign(targets=label))
  res = {k: pd.concat(v, ignore_index=True) for k, v in res.items()}
  res['contrasts']['incomplete_arm'] = res['contrasts'].contrast.map(
      lambda n: ', '.join(sorted(bad & set(n[n.index('(') + 1:-1].split(',')))))

  # Next to the main study (exploratory): overall from its own outputs, bucket 512 from its seed pairs.
  mp = pd.read_csv(pathlib.Path(args.main) / 'contrasts' / 'pairs_seed.csv')
  mp = mp[mp.contrast.isin(NAMES)]
  mcons = {n: cons[n] for n in NAMES if n in cons}
  mcon = pd.read_csv(pathlib.Path(args.main) / 'contrasts' / 'contrasts.csv')
  mhw = pd.read_csv(pathlib.Path(args.main) / 'contrasts' / 'hw_minus_rc.csv')
  side = []

  def add(source, label, con, hw):
    for r in con[con.contrast.isin(NAMES) & con.statistic.isin(contrasts.FLIP_STATS)].itertuples():
      side.append({'source': source, 'targets': label, 'quantity': r.contrast, 'statistic': r.statistic,
                   'n_targets': r.n_targets, 'estimate': r.estimate, 'ci_low': r.ci_low, 'ci_high': r.ci_high})
    for r in hw[hw.hardware.isin(PAIRS)].itertuples():
      side.append({'source': source, 'targets': label, 'quantity': f'{r.hardware} minus {r.recompilation}',
                   'statistic': r.statistic, 'n_targets': r.n_targets, 'estimate': r.estimate, 'ci_low': r.ci_low,
                   'ci_high': r.ci_high})
  add('replication', n_all, flip_summary(sp[sp.target.isin(targets)], cons, args),
      contrasts.hw_minus_rc(sp[sp.target.isin(targets)], cons, args))
  add('main study', f'all {len(main_targets)}', mcon, mhw)
  add('replication', f'bucket {BUCKET}: {len(b512)}', flip_summary(sp[sp.target.isin(b512)], cons, args),
      contrasts.hw_minus_rc(sp[sp.target.isin(b512)], cons, args))
  add('main study', f'bucket {BUCKET}: {len(main_b512)}', flip_summary(mp[mp.target.isin(main_b512)], mcons, args),
      contrasts.hw_minus_rc(mp[mp.target.isin(main_b512)], mcons, args))
  next_to_main = pd.DataFrame(side)

  # Pooled, secondary and descriptive: both studies' seed pairs together.
  pool = pd.concat([sp[sp.contrast.isin(NAMES)], mp], ignore_index=True)
  pooled = []
  for label, ts, stats in ((f'pooled {len(targets) + len(main_targets)}', targets + main_targets, contrasts.FLIP_STATS),
                           (f'pooled post-cutoff {len(post) + len(main_post)}', post + main_post, M1_STATS)):
    p = pool[pool.target.isin(ts)]
    fs = flip_summary(p, cons, args)
    hw = contrasts.hw_minus_rc(p, cons, args)
    pooled.append(fs[fs.statistic.isin(stats)].assign(quantity=fs.contrast, targets=label))
    pooled.append(hw[hw.statistic.isin(stats)].assign(quantity=hw.hardware + ' minus ' + hw.recompilation, targets=label))
  pooled = pd.concat(pooled, ignore_index=True)[['targets', 'quantity', 'statistic', 'n_targets', 'estimate', 'ci_low',
                                                 'ci_high']]

  for name, df in (('completeness', comp), ('counts', counts), ('pairs_seed', sp), ('pairs_sample', sm),
                   ('next_to_main', next_to_main), ('pooled', pooled), *res.items()):
    df.to_csv(out / f'{name}.csv', index=False)
  pct = f34.pct
  L = ['# Follow-up 11: replication on new targets', '',
       f'Targets: {len(targets)} ({len(post)} released after {CUTOFF}; {len(b512)} in bucket {BUCKET}).', '',
       '## Reading rule (HW minus RC(P) above 0: replicates; headline: top-1 flip rate)', '']
  for r in res['reading_rule'].itertuples():
    hw = res['hw_minus_rc']
    x = hw[(hw.targets == r.targets) & (hw.hardware == r.pair) & (hw.statistic == r.statistic) & (hw.reference == 'RC(P)')]
    num = f' ({pct(x.estimate.iloc[0])}, {pct(x.ci_low.iloc[0])} to {pct(x.ci_high.iloc[0])})' if len(x) else ''
    L.append(f'- {r.targets}, {r.pair}, {r.statistic} ({r.role}): {r.answer}{num}; '
             f'{r.against_rc_of_second}' + (f' (incomplete arm: {r.incomplete_arms})' if r.incomplete_arms else ''))
  L += ['', '## Completeness', '', '| condition | repetition | planned | with 5 samples | status |', '|---|---|---|---|---|']
  L += [f'| {r.condition} | {r.repetition} | {r.planned_units} | {r.units_with_5_samples} | {r.status} |'
        for r in comp.itertuples()]
  L += ['', '## Next to the main study (bucket rows exploratory)', '',
        '| source | targets | quantity | statistic | estimate | 95% CI |', '|---|---|---|---|---|---|']
  L += [f'| {r.source} | {r.targets} | {r.quantity} | {r.statistic} | {pct(r.estimate)} | {pct(r.ci_low)} to '
        f'{pct(r.ci_high)} |' for r in next_to_main.itertuples()]
  L += ['', '## Pooled (secondary, descriptive)', '', '| targets | quantity | statistic | estimate | 95% CI |',
        '|---|---|---|---|---|']
  L += [f'| {r.targets} | {r.quantity} | {r.statistic} | {pct(r.estimate)} | {pct(r.ci_low)} to {pct(r.ci_high)} |'
        for r in pooled.itertuples()]
  L += ['', '## Success rate of the per-seed top-1 (M1)', '', '| targets | condition | repetition | rate | 95% CI |',
        '|---|---|---|---|---|']
  L += [f'| {r.targets} | {r.condition} | {r.repetition} | {pct(r.success_rate)} | {pct(r.ci_low)} to {pct(r.ci_high)} |'
        for r in res['success_rates'].itertuples()]
  L += ['', '## Contrasts', ''] + f34.contrast_lines(res['contrasts'][res['contrasts'].targets == n_all])
  (out / 'report.md').write_text('\n'.join(L) + '\n')
  print((out / 'report.md').read_text())
  return 0


if __name__ == '__main__':
  sys.exit(main())
