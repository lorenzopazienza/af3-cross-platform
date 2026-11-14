"""Rounded ranking score (drafts/section10_rounded_score_20261114.md, label rounded): the rules, the filters and the
outputs of analysis/rounded_score.py on synthetic per-sample tables. No real AlphaFold3 output is read."""
import pathlib
import sys

import numpy as np
import pandas as pd

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'analysis'))
import rounded_score as rs  # noqa: E402


def unit(scores, platform='V5', target='T1', seed=1, repetition='r1', run='run1', cache='fresh', exit_code=0,
         label='main', summary=None):
  scores = np.asarray(scores, float)
  summ = np.round(scores, 2) if summary is None else np.asarray(summary, float)
  return pd.DataFrame({'run': run, 'label': label, 'platform': platform, 'target': target, 'seed': seed,
                       'repetition': repetition, 'cache': cache, 'sample': range(len(scores)), 'exit_code': exit_code,
                       'ranking_score': scores, 'ranking_score_summary': summ})


def test_rounding_creates_a_tie_that_goes_to_the_lowest_index():
  d, _ = rs.units(unit([0.951, 0.954, 0.90, 0.80, 0.70]))   # rounded 0.95, 0.95: tie, to sample 0
  r = d.iloc[0]
  assert (r.top_full, r.top_rounded, r.differ, r.tie) == (1, 0, True, True)


def test_no_difference_without_a_tie():
  d, _ = rs.units(unit([0.97, 0.954, 0.90, 0.80, 0.70]))
  r = d.iloc[0]
  assert (r.top_full, r.top_rounded, r.differ, r.tie) == (0, 0, False, False)


def test_tie_without_difference():
  d, _ = rs.units(unit([0.954, 0.951, 0.90, 0.80, 0.70]))
  r = d.iloc[0]
  assert (r.differ, r.tie) == (False, True)


def test_full_precision_tie_goes_to_the_lowest_index():
  d, _ = rs.units(unit([0.5, 0.9, 0.9, 0.1, 0.2]))
  assert d.iloc[0].top_full == 1


def test_filters_and_dropped_groups():
  s = pd.concat([unit([0.951, 0.954, 0.9, 0.8, 0.7], run='a'),
                 unit([0.951, 0.954, 0.9, 0.8, 0.7], run='b', cache='warm', repetition='w'),
                 unit([0.951, 0.954, 0.9, 0.8, 0.7], run='c', exit_code=1),
                 unit([0.951, 0.954, 0.9, 0.8, 0.7], run='d', label='pilot'),
                 unit([0.951, 0.954, 0.9, 0.8], run='e'),
                 unit([0.951, 0.954, 0.9, 0.8, 0.7], run='f', summary=[0.95, np.nan, 0.9, 0.8, 0.7])])
  d, dropped = rs.units(s)
  assert list(d.run) == ['a']
  assert sorted(dropped.run) == ['e', 'f']
  assert set(dropped.reason) == {'4 samples', 'a score is missing'}


def test_summary_and_outputs(tmp_path):
  s = pd.concat([unit([0.951, 0.954, 0.9, 0.8, 0.7], platform='V5', run='a'),
                 unit([0.97, 0.954, 0.9, 0.8, 0.7], platform='V5', run='b', target='T2'),
                 unit([0.951, 0.954, 0.9, 0.8, 0.7], platform='G', repetition='r2', run='c')])
  d, t, dropped = rs.run(s, tmp_path)
  tot = t[t.platform == 'all'].iloc[0]
  assert (tot.predictions, tot.differ, tot.ties) == (3, 2, 2)
  v5 = t[(t.platform == 'V5') & (t.repetition == 'r1')].iloc[0]
  assert (v5.predictions, v5.differ, v5.fraction) == (2, 1, 0.5)
  assert (tmp_path / 'rounded.csv').is_file() and (tmp_path / 'dropped.csv').is_file()
  assert (tmp_path / 'report.md').read_text().startswith('# Rounded ranking score')
