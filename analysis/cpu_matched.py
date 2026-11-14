"""Matched recompilation reference for the CPU pair (section 10, row of drafts/section10_cpu_matched_spec_20261114.md,
label cpu_matched): a post hoc sensitivity analysis of a registered reading, asked by the reviewers. The registered
reading stays; this analysis only says whether it would change. No run and no cloud resource.

    analysis/.venv/bin/python analysis/cpu_matched.py --main . --out results/analysis/cpu_matched_<DATE> \\
        [--replicates 10000] [--seed 2026]

The registered HW(G,C) minus RC(G) (main study, 6 targets; follow-up 4, 60 targets) subtracts recompilation on the L4
over seeds 1 to 5 from the L4-CPU difference over seeds 1 and 2, the only seeds the CPU ran. Here RC(G) is restricted
to the same seeds as HW(G,C) (and, by contrasts.hw_minus_rc, to the same targets), so both terms count the same
(target, seed) predictions.
Inputs, relative to --main: results/analysis/main_20261014/primary_v2/contrasts/pairs_seed.csv (RC(G), and HW(G,C) of
the main study) and hw_minus_rc.csv; results/analysis/cpu60_20261018/pairs_seed.csv (HW(G,C), 60 targets) and
hw_minus_rc.csv.
Statistics: contrasts.hw_minus_rc unchanged (paired bootstrap over targets, 10,000 replicates, seed 2026), for the top-1
flip, the success flip and flips that matter. Check (the analysis stops if it fails): with RC(G) on all its seeds, the
computation reproduces the registered rows within 1e-9.
Reading: the registered rule, "beyond recompilation" if the 95% interval lies above 0; "changed" when the matched
reading differs from the registered one.
Outputs in --out: matched.csv, inputs.tsv, report.md.
"""
import argparse
import hashlib
import pathlib
import sys
import types

import pandas as pd

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / 'analysis'))
import contrasts  # noqa: E402

AN = 'results/analysis'
MAIN = f'{AN}/main_20261014/primary_v2/contrasts'
CPU60 = f'{AN}/cpu60_20261018'
SOURCES = {'main study, 6 targets': MAIN, 'follow-up 4, 60 targets': CPU60}
TOL = 1e-9


def read(rule_low):
  if rule_low != rule_low:
    return 'not computed'
  return 'beyond' if rule_low > 0 else 'not beyond'


def compute(hw, rc, args):
  cons = {'HW(G,C)': ('HW', True), 'RC(G)': ('RC', True)}
  r = contrasts.hw_minus_rc(pd.concat([hw, rc], ignore_index=True), cons, args)
  return r[r.reference == 'RC(P)'].set_index('statistic')


def run(tables, out, args, inputs=(), root=None):
  out = pathlib.Path(out)
  out.mkdir(parents=True, exist_ok=True)
  rc_all = tables['main_sp'][tables['main_sp'].contrast == 'RC(G)']
  rows = []
  for label, key in (('main study, 6 targets', 'main'), ('follow-up 4, 60 targets', 'cpu60')):
    hw = tables[f'{key}_sp'][tables[f'{key}_sp'].contrast == 'HW(G,C)']
    if hw.empty or rc_all.empty:
      raise SystemExit(f'{label}: HW(G,C) or RC(G) has no seed pairs')
    seeds = sorted(set(hw.seed_a))
    reg = tables[f'{key}_hw']
    reg = reg[(reg.hardware == 'HW(G,C)') & (reg.recompilation == 'RC(G)')].set_index('statistic')
    full = compute(hw, rc_all, args)
    for st in contrasts.FLIP_STATS:
      for a, b in (('estimate', 'estimate'), ('ci_low', 'ci_low'), ('ci_high', 'ci_high')):
        if abs(float(full.loc[st, a]) - float(reg.loc[st, b])) > TOL:
          raise SystemExit(f'check: {label}, {st}: the registered row is not reproduced ({a})')
    matched = compute(hw, rc_all[rc_all.seed_a.isin(seeds) & rc_all.seed_b.isin(seeds)], args)
    for st in contrasts.FLIP_STATS:
      m, g = matched.loc[st], reg.loc[st]
      rows.append({'pair': label, 'statistic': st, 'hw_seeds': ' '.join(map(str, seeds)),
                   'n_targets': int(m.n_targets), 'estimate': float(m.estimate), 'ci_low': float(m.ci_low),
                   'ci_high': float(m.ci_high), 'registered_estimate': float(g.estimate),
                   'registered_ci_low': float(g.ci_low), 'registered_ci_high': float(g.ci_high),
                   'registered_reading': read(float(g.ci_low)), 'matched_reading': read(float(m.ci_low))})
  df = pd.DataFrame(rows)
  df['changed'] = (df.registered_reading != df.matched_reading).map({True: 'yes', False: 'no'})
  df.to_csv(out / 'matched.csv', index=False)
  if root is not None:
    (out / 'inputs.tsv').write_text(''.join(f'{p}\t{sha256(pathlib.Path(root) / p)}\n' for p in inputs))
  (out / 'report.md').write_text(report(df, args))
  return df


def pts(x):
  return f'{100 * x:+.1f}'


def report(df, args):
  L = ['# Matched recompilation reference for the CPU pair (label cpu_matched)', '',
       'Post hoc sensitivity analysis (drafts/section10_cpu_matched_spec_20261114.md); the registered readings stay. '
       f'HW(G,C) minus RC(G), percentage points, 95% intervals over targets ({args.replicates} replicates, seed '
       f'{args.seed}).', '',
       '| pair | statistic | targets | RC(G) on the same seeds | registered (RC(G) seeds 1 to 5) | registered reading | '
       'matched reading | changed |', '|---|---|---|---|---|---|---|---|']
  for r in df.itertuples():
    L.append(f'| {r.pair} | {r.statistic} | {r.n_targets} | {pts(r.estimate)} [{pts(r.ci_low)}, {pts(r.ci_high)}] | '
             f'{pts(r.registered_estimate)} [{pts(r.registered_ci_low)}, {pts(r.registered_ci_high)}] | '
             f'{r.registered_reading} | {r.matched_reading} | {r.changed} |')
  return '\n'.join(L) + '\n'


def sha256(path):
  h = hashlib.sha256()
  with open(path, 'rb') as f:
    for chunk in iter(lambda: f.read(1 << 20), b''):
      h.update(chunk)
  return h.hexdigest()


def load(root):
  root = pathlib.Path(root)
  files = {'main_sp': f'{MAIN}/pairs_seed.csv', 'main_hw': f'{MAIN}/hw_minus_rc.csv',
           'cpu60_sp': f'{CPU60}/pairs_seed.csv', 'cpu60_hw': f'{CPU60}/hw_minus_rc.csv'}
  return {k: pd.read_csv(root / v) for k, v in files.items()}, list(files.values())


def main():
  ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
  ap.add_argument('--main', default='.', help='the checkout that holds results/')
  ap.add_argument('--out', required=True)
  ap.add_argument('--replicates', type=int, default=10000)
  ap.add_argument('--seed', type=int, default=2026)
  args = ap.parse_args()
  out = pathlib.Path(args.out)
  if (out / 'matched.csv').exists():
    raise SystemExit(f'{out}/matched.csv exists; the output folder must be new')
  tables, inputs = load(args.main)
  run(tables, out, types.SimpleNamespace(replicates=args.replicates, seed=args.seed), inputs, args.main)
  print(f'wrote {out}')


if __name__ == '__main__':
  main()
