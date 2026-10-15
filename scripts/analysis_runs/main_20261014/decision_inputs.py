"""Decision-input tables (M2, M3, flip rates, HW minus RC) for the main study, from the outputs of analysis/contrasts.py. No decision.

    analysis/.venv/bin/python scripts/analysis_runs/pilot2_20261012/decision_inputs.py STEP2_DIR

Reads STEP2_DIR/samples.csv and STEP2_DIR/contrasts/*.csv. Writes in STEP2_DIR/decision/:
  m2_m3.csv            per (condition, repetition, target, seed): top-1 sample (M2), its success (M1),
                       top-1 minus top-2 ranking_score (M3), as analysis/contrasts.top computes them;
  m3_summary.csv       per condition, r1: number of (target, seed), median, 10th percentile and minimum M3;
  m3_at_flips.csv      per contrast: the smaller of the two M3 margins of each pair, for pairs whose top-1
                       flipped and for those whose top-1 did not (median, minimum, maximum);
  flip_rates.csv       every contrast's flip rates with 95% bootstrap intervals (from contrasts.csv);
  hw_vs_rc.csv         every HW - RC row (hw_minus_rc.csv) with two flags: estimate above 0, and interval
                       entirely above 0;
  decision_inputs.md   the same, as tables.
"""
import argparse
import pathlib
import sys

import numpy as np
import pandas as pd

REPO = pathlib.Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO / 'analysis'), str(REPO / 'harness')]
import contrasts  # noqa: E402

step = pathlib.Path(sys.argv[1])
out = step / 'decision'
out.mkdir(exist_ok=True)
args = argparse.Namespace(samples=str(step / 'samples.csv'), labels='main', include=None, exclude=None,
                          condition='platform')
_, ok = contrasts.load(args)
rows = []
for (c, rep, t, s), g in contrasts.seed_units(ok).items():
  top1, success, margin = contrasts.top(g)
  rows.append({'condition': c, 'repetition': rep, 'target': t, 'seed': s, 'top1_sample': top1,
               'top1_success': success, 'm3_margin': margin})
m23 = pd.DataFrame(rows).sort_values(['condition', 'repetition', 'target', 'seed'], key=lambda x: x.map(
    lambda v: contrasts.rank(v) if isinstance(v, str) and x.name == 'condition' else v))
m23.to_csv(out / 'm2_m3.csv', index=False)
r1 = m23[m23.repetition == 'r1']
m3 = r1.groupby('condition').m3_margin.agg(
    n='size', median='median', p10=lambda x: float(np.percentile(x, 10)), minimum='min').reset_index()
m3['order'] = m3.condition.map(contrasts.rank)
m3 = m3.sort_values('order').drop(columns='order')
m3.to_csv(out / 'm3_summary.csv', index=False)

sp = pd.read_csv(step / 'contrasts' / 'pairs_seed.csv')
sp['min_margin'] = sp[['margin_a', 'margin_b']].min(axis=1)
flips = []
for name, g in sp.groupby('contrast', sort=False):
  for flipped, h in g.groupby('flip_top1'):
    flips.append({'contrast': name, 'top1_flipped': bool(flipped), 'pairs': len(h), 'median': h.min_margin.median(),
                  'minimum': h.min_margin.min(), 'maximum': h.min_margin.max()})
pd.DataFrame(flips).to_csv(out / 'm3_at_flips.csv', index=False)

cs = pd.read_csv(step / 'contrasts' / 'contrasts.csv')
fr = cs[cs.statistic.isin(contrasts.FLIP_STATS)].copy()
fr.to_csv(out / 'flip_rates.csv', index=False)
hw = pd.read_csv(step / 'contrasts' / 'hw_minus_rc.csv')
hw['estimate_above_0'] = hw.estimate > 0
hw['interval_above_0'] = hw.ci_low > 0
hw.to_csv(out / 'hw_vs_rc.csv', index=False)


def pct(x):
  return '-' if pd.isna(x) else f'{100 * x:.1f}%'


def num(x):
  return '-' if pd.isna(x) else f'{x:.3g}'


lines = ['# Main study: M2, M3, flip rates and HW minus RC (no interpretation)', '',
         'From analysis/contrasts.py (10,000 bootstrap replicates over targets, seed 2026). Flip rates and HW - RC '
         'as computed there; M3 = top-1 minus top-2 ranking_score within a seed.', '',
         '## M3 margins, r1, per condition', '', '| condition | (target, seed) | median | 10th percentile | minimum |',
         '|---|---|---|---|---|']
lines += [f'| {r.condition} | {r.n} | {num(r.median)} | {num(r.p10)} | {num(r.minimum)} |' for r in m3.itertuples()]
lines += ['', '## M3 at flips (smaller margin of the two sides of each pair)', '',
          '| contrast | top-1 flipped | pairs | median | minimum | maximum |', '|---|---|---|---|---|---|']
lines += [f'| {r["contrast"]} | {"yes" if r["top1_flipped"] else "no"} | {r["pairs"]} | {num(r["median"])} | '
          f'{num(r["minimum"])} | {num(r["maximum"])} |' for r in flips]
lines += ['', '## Hardware minus recompilation (flip rates)', '',
          '| hardware | recompilation | reference | role | statistic | estimate | 95% CI | estimate > 0 | CI > 0 |',
          '|---|---|---|---|---|---|---|---|---|']
lines += [f'| {r.hardware} | {r.recompilation} | {r.reference} | {r.role} | {r.statistic} | '
          f'{pct(r.estimate)} | {pct(r.ci_low)} to {pct(r.ci_high)} | {"yes" if r.estimate_above_0 else "no"} | '
          f'{"yes" if r.interval_above_0 else "no"} |' for r in hw.itertuples()]
(out / 'decision_inputs.md').write_text('\n'.join(lines) + '\n')
print(f'>> wrote {out}')
