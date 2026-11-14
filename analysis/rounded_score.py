"""Rounded ranking score against the full-precision score in the main study (section 10, row of
drafts/section10_rounded_score_20261114.md, label rounded): a post hoc, descriptive count of frozen outputs. No run and
no cloud resource.

    analysis/.venv/bin/python analysis/rounded_score.py --main . --out results/analysis/rounded_<DATE>

Input, relative to --main: results/analysis/main_20261014/primary_v2/samples.csv (analysis/collect.py's table).
Rows: label main, cache fresh (the first, compiling run of each process; the warm reruns w are left out), exit code 0.
A prediction is one process's (run, platform, target, seed, repetition) with exactly five samples, each with both
scores; any other group is counted as dropped and listed.
Scores (analysis/collect.py): ranking_score is AlphaFold 3's full-precision score, from <target>_ranking_scores.csv;
ranking_score_summary is the same score as *_summary_confidences.json writes it, rounded to two decimals.
Rules: the full-precision top is the sample with the highest ranking_score, and the rounded top is the sample with the
highest ranking_score_summary; in both, an exact tie goes to the lowest sample index (as AlphaFold 3 and
contrasts.top break ties). A prediction "differs" when the two tops are different samples; it "has a tie" when two
or more samples share the highest rounded score.
Outputs in --out: rounded.csv (per platform and repetition, and a total row: predictions, differ, fraction, ties,
tie fraction), dropped.csv (groups left out and why), report.md.
"""
import argparse
import pathlib

import pandas as pd

SAMPLES = 'results/analysis/main_20261014/primary_v2/samples.csv'
COLUMNS = ['run', 'label', 'platform', 'target', 'seed', 'repetition', 'cache', 'sample', 'exit_code', 'ranking_score',
           'ranking_score_summary']
KEY = ['run', 'platform', 'target', 'seed', 'repetition']


def top(g, col):
  """The lowest sample index among the samples with the highest value of col."""
  return int(g[g[col] == g[col].max()]['sample'].min())


def units(s):
  """(per-prediction table, dropped groups) from a samples table."""
  s = s[(s.label == 'main') & (s.cache == 'fresh') & (s.exit_code == 0)]
  rows, dropped = [], []
  for k, g in s.groupby(KEY):
    why = None
    if len(g) != 5 or sorted(g['sample'].astype(int)) != [0, 1, 2, 3, 4]:
      why = f'{len(g)} samples'
    elif g.ranking_score.isna().any() or g.ranking_score_summary.isna().any():
      why = 'a score is missing'
    if why:
      dropped.append(dict(zip(KEY, k), reason=why))
      continue
    full, rnd = top(g, 'ranking_score'), top(g, 'ranking_score_summary')
    rows.append({**dict(zip(KEY, k)), 'top_full': full, 'top_rounded': rnd, 'differ': full != rnd,
                 'tie': int((g.ranking_score_summary == g.ranking_score_summary.max()).sum()) > 1})
  return pd.DataFrame(rows, columns=KEY + ['top_full', 'top_rounded', 'differ', 'tie']), \
      pd.DataFrame(dropped, columns=KEY + ['reason'])


def summary(d):
  def row(g, platform, repetition):
    return {'platform': platform, 'repetition': repetition, 'predictions': len(g), 'differ': int(g.differ.sum()),
            'fraction': g.differ.mean() if len(g) else float('nan'), 'ties': int(g.tie.sum()),
            'tie_fraction': g.tie.mean() if len(g) else float('nan')}
  out = [row(g, p, r) for (p, r), g in d.groupby(['platform', 'repetition'])]
  out.append(row(d, 'all', 'all'))
  return pd.DataFrame(out)


def report(t, dropped):
  L = ['# Rounded ranking score against the full-precision score (label rounded)', '',
       'Post hoc, descriptive (drafts/section10_rounded_score_20261114.md). Main study, fresh processes, exit code 0.', '',
       '| platform | repetition | predictions | differ | fraction | ties | tie fraction |', '|---|---|---|---|---|---|---|']
  L += [f'| {r.platform} | {r.repetition} | {r.predictions} | {r.differ} | {r.fraction:.3f} | {r.ties} | '
        f'{r.tie_fraction:.3f} |' for r in t.itertuples()]
  L += ['', f'Groups left out: {len(dropped)}.']
  return '\n'.join(L) + '\n'


def run(s, out):
  out = pathlib.Path(out)
  out.mkdir(parents=True, exist_ok=True)
  d, dropped = units(s)
  t = summary(d)
  t.to_csv(out / 'rounded.csv', index=False)
  dropped.to_csv(out / 'dropped.csv', index=False)
  (out / 'report.md').write_text(report(t, dropped))
  return d, t, dropped


def main():
  ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
  ap.add_argument('--main', default='.', help='the checkout that holds results/')
  ap.add_argument('--out', required=True)
  args = ap.parse_args()
  out = pathlib.Path(args.out)
  if (out / 'rounded.csv').exists():
    raise SystemExit(f'{out}/rounded.csv exists; the output folder must be new')
  s = pd.read_csv(pathlib.Path(args.main) / SAMPLES, usecols=COLUMNS, low_memory=False)
  run(s, out)
  print(f'wrote {out}')


if __name__ == '__main__':
  main()
