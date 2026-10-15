"""Diagnostic (2026-10-12): the analysis ranks samples by ranking_score from *_summary_confidences.json, which
AlphaFold3 rounds to 2 decimals (alphafold3/model/confidence_types.py); AlphaFold3's own ranking uses the full
value written in <target>_ranking_scores.csv. Per (condition, repetition, target, seed): whether the rounded
top score is tied, and whether the analysis's top-1 (rounded score, ties to the lowest sample index) equals
AlphaFold3's top-1 (full-precision argmax). Also check 3 recomputed with AlphaFold3's ranking.

    analysis/.venv/bin/python scripts/analysis_runs/pilot2_20261012/ranking_precision_diagnostic.py STEP2_DIR
"""
import argparse
import csv
import pathlib
import sys

import pandas as pd

REPO = pathlib.Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO / 'analysis'), str(REPO / 'harness')]
import contrasts  # noqa: E402

step = pathlib.Path(sys.argv[1])
args = argparse.Namespace(samples=str(step / 'samples.csv'), labels='pilot2', include=None, exclude=None,
                          condition='platform')
_, ok = contrasts.load(args)
rows = []
full_scores = {}
for (c, rep, t, s), g in contrasts.seed_units(ok).items():
  top1, _, _ = contrasts.top(g)
  outdir = (REPO / g.model_path.dropna().iloc[0]).parent.parent
  scores = {int(r['sample']): float(r['ranking_score']) for r in csv.DictReader(open(outdir / f'{t}_ranking_scores.csv'))
            if int(r['seed']) == int(s)}
  af3_top = max(scores, key=scores.get)
  rounded = g.ranking_score
  for k, v in scores.items():
    full_scores[(c, rep, t, s, k)] = v
  rows.append({'condition': c, 'repetition': rep, 'target': t, 'seed': s, 'analysis_top1': top1, 'af3_top1': af3_top,
               'rounded_top_tied': int((rounded == rounded.max()).sum()) > 1,
               'rounded_matches_full': all(round(scores[k], 2) == rounded.loc[k] for k in scores),
               'full_margin': sorted(scores.values())[-1] - sorted(scores.values())[-2]})
d = pd.DataFrame(rows)
d['same_top1'] = d.analysis_top1 == d.af3_top1
d.to_csv(step / 'ranking_precision_units.csv', index=False)
summ = d.groupby('condition').agg(units=('same_top1', 'size'), rounded_top_tied=('rounded_top_tied', 'sum'),
                                  analysis_top1_differs_from_af3=('same_top1', lambda x: int((~x).sum())),
                                  rounded_values_consistent=('rounded_matches_full', 'all'),
                                  full_margin_median=('full_margin', 'median'), full_margin_min=('full_margin', 'min'))
summ = summ.loc[sorted(summ.index, key=contrasts.rank)]
print(summ.to_string())
# Check 3 with AlphaFold3's ranking: overall top-1 across seeds of V6 r1 = full-precision argmax over both seeds.
v6 = ok[(ok.condition == 'V6') & (ok.repetition == 'r1')].copy()
v6['full'] = [full_scores[('V6', 'r1', r.target, r.seed, int(r.sample))] for r in v6.itertuples()]
best = v6.sort_values('full', ascending=False, kind='mergesort').groupby('target').head(1)
rate = float((best.m1_success == True).mean())  # noqa: E712
print(f'check 3 with AlphaFold3 full-precision ranking: V6 overall top-1 success {rate:.0%} of {len(best)} targets '
      f'({"pass" if rate >= 0.5 else "BELOW 50%"})')
per_seed = v6.sort_values('full', ascending=False, kind='mergesort').groupby(['target', 'seed']).head(1)
print(f'  secondary, top-1 per seed: {float((per_seed.m1_success == True).mean()):.0%} of {len(per_seed)}')  # noqa: E712
