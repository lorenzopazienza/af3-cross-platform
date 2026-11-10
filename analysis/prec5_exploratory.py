"""Exploratory analysis (c) on follow-up 12's five seeds (notes/analysis_plan.md section 10, row of 2026-11-10 "Two
computations on frozen outputs for the paper", part (1)).

    analysis/.venv/bin/python analysis/prec5_exploratory.py --main . --out results/analysis/prec5_exploratory_<DATE> \\
        [--replicates 10000] [--seed 2026]

Analysis (c) of the exploratory row of 2026-10-14 (perturbation size against margin, and the two-sample model's
predicted per-seed top-1 flip rate), unchanged in every definition: the function `perturbation` of
analysis/exploratory.py is called as it is, on follow-up 12's contrasts PR(V6), PR(G) and HWh(V6,G), 60 targets,
seeds 1 to 5. That function loops over the module's CONTRASTS; this runner sets that list to the three contrasts for
the call and restores it afterwards, and changes nothing else in the module.

Inputs, read only: <main>/results/analysis/prec5_20261107/pairs_seed.csv and pairs_sample.csv (follow-up 12's own
pairs, built by analysis/prec5.py with analysis/contrasts.py; in every contrast A is the first-named condition, so
margin_a is the margin of V6, G and V6h respectively). The folder prec5_20261107 is never written.
Checks before the analysis: each contrast has the expected targets and seeds 1 to 5, one seed pair per (target, seed)
and five sample pairs per seed pair; no margin_a or m5_ranking_score is missing; flip_top1 equals top1_a != top1_b.
Outputs in --out (a new folder): exploratory_c.csv (the columns of main_20261014/exploratory/c_perturbation.csv) and
inputs.tsv (path relative to --main and SHA-256 of every input); run.log is written by
scripts/analysis_runs/prec5_exploratory/reproduce.sh.
"""
import argparse
import hashlib
import pathlib
import sys

import pandas as pd

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / 'analysis'))
import exploratory as ex  # noqa: E402

SOURCE = pathlib.Path('results/analysis/prec5_20261107')
CONTRASTS = ['PR(V6)', 'PR(G)', 'HWh(V6,G)']
SEEDS = [1, 2, 3, 4, 5]
TARGETS = 60
SAMPLES_PER_SEED = 5
COLUMNS = ['contrast', 'n_seed_pairs', 'n_sample_pairs', 'median_abs_score_difference', 'delta_ci_low',
           'delta_ci_high', 'median_margin', 'mu_ci_low', 'mu_ci_high', 'delta_over_mu', 'ratio_ci_low',
           'ratio_ci_high', 'predicted_flip_rate', 'predicted_ci_low', 'predicted_ci_high', 'observed_flip_rate',
           'observed_ci_low', 'observed_ci_high']


def sha256(path):
  h = hashlib.sha256()
  with open(path, 'rb') as f:
    for block in iter(lambda: f.read(1 << 20), b''):
      h.update(block)
  return h.hexdigest()


def check(seed, sample, targets=TARGETS):
  """Problems with the pairs, as a list of strings (empty if none)."""
  bad = []
  for c in CONTRASTS:
    g, h = seed[seed.contrast == c], sample[sample.contrast == c]
    if g.empty or h.empty:
      bad.append(f'{c}: no rows')
      continue
    if g.target.nunique() != targets:
      bad.append(f'{c}: {g.target.nunique()} targets, expected {targets}')
    if sorted(g.seed_a.unique()) != SEEDS or not (g.seed_a == g.seed_b).all():
      bad.append(f'{c}: seeds {sorted(g.seed_a.unique())}, expected {SEEDS} on both sides')
    if g.duplicated(['target', 'seed_a']).any() or len(g) != targets * len(SEEDS):
      bad.append(f'{c}: {len(g)} seed pairs, expected one per (target, seed), {targets * len(SEEDS)}')
    per = h.groupby(['target', 'seed_a']).size()
    if len(h) != len(g) * SAMPLES_PER_SEED or (per != SAMPLES_PER_SEED).any():
      bad.append(f'{c}: {len(h)} sample pairs, expected {SAMPLES_PER_SEED} per seed pair')
    if g.margin_a.isna().any() or h.m5_ranking_score.isna().any():
      bad.append(f'{c}: missing margin_a or m5_ranking_score')
    if not (g.flip_top1.astype(bool) == (g.top1_a != g.top1_b)).all():
      bad.append(f'{c}: flip_top1 differs from top1_a != top1_b')
  return bad


def run(seed, sample, args):
  """exploratory.perturbation, unchanged, on CONTRASTS only."""
  saved = ex.CONTRASTS
  ex.CONTRASTS = list(CONTRASTS)
  try:
    cp = ex.perturbation(seed, sample, args)
  finally:
    ex.CONTRASTS = saved
  return cp[COLUMNS]


def main(argv=None):
  ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
  ap.add_argument('--main', default=str(REPO), help='checkout holding results/ (default: this repository root)')
  ap.add_argument('--out', required=True)
  ap.add_argument('--replicates', type=int, default=10000)
  ap.add_argument('--seed', type=int, default=2026)
  ap.add_argument('--targets', type=int, default=TARGETS, help=argparse.SUPPRESS)  # synthetic tests only
  args = ap.parse_args(argv)
  root = pathlib.Path(args.main).resolve()
  src, out = root / SOURCE, pathlib.Path(args.out).resolve()
  if out == src.resolve() or src.resolve() in out.parents:
    raise SystemExit(f'--out {out} is inside {src}, which this analysis must not change')
  if (out / 'exploratory_c.csv').exists():
    raise SystemExit(f'{out / "exploratory_c.csv"} exists; the output folder must be new')
  paths = [src / 'pairs_seed.csv', src / 'pairs_sample.csv']
  hashes = {str(p.relative_to(root)): sha256(p) for p in paths}
  seed, sample = (pd.read_csv(p) for p in paths)
  seed = seed[seed.contrast.isin(CONTRASTS)].reset_index(drop=True)
  sample = sample[sample.contrast.isin(CONTRASTS)].reset_index(drop=True)
  bad = check(seed, sample, args.targets)
  if bad:
    raise SystemExit('pairs check failed: ' + '; '.join(bad))
  if any(hashes[k] != sha256(root / k) for k in hashes):
    raise SystemExit('an input changed while it was read')
  cp = run(seed, sample, args)
  out.mkdir(parents=True, exist_ok=True)
  cp.to_csv(out / 'exploratory_c.csv', index=False)
  pd.DataFrame(list(hashes.items()), columns=['path', 'sha256']).to_csv(out / 'inputs.tsv', sep='\t', index=False)
  for r in cp.itertuples():
    print(f'{r.contrast}: {r.n_seed_pairs} seed pairs, delta/mu {r.delta_over_mu:.3f} '
          f'[{r.ratio_ci_low:.3f}, {r.ratio_ci_high:.3f}], predicted {r.predicted_flip_rate:.4f}, '
          f'observed {r.observed_flip_rate:.4f} [{r.observed_ci_low:.4f}, {r.observed_ci_high:.4f}]', flush=True)
  print(f'>> prec5_exploratory: {len(cp)} contrasts, {args.replicates} replicates, seed {args.seed}; out {out}',
        flush=True)
  return 0


if __name__ == '__main__':
  sys.exit(main())
