"""Pilot check 3 (section 6) on the V6 session only: the overall top-1 success on V6 r1 against 50%.

    analysis/.venv/bin/python scripts/analysis_runs/pilot2_20261012/check3.py STEP_DIR

STEP_DIR holds samples.csv of analysis/collect.py with M1 for the V6 session. The rows are those of
analysis/contrasts.checks (check 3 and its per-seed secondary, and whether M1 was computed); no contrast.
"""
import argparse
import pathlib
import sys

import pandas as pd

REPO = pathlib.Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO / 'analysis'), str(REPO / 'harness')]
import contrasts  # noqa: E402

step = pathlib.Path(sys.argv[1])
args = argparse.Namespace(samples=str(step / 'samples.csv'), labels='pilot2', include=None, exclude=None,
                          condition='platform', replicates=0, seed=2026)
d, ok = contrasts.load(args)
assert set(d.condition) == {'V6'}, set(d.condition)
c = contrasts.checks(d, ok, pd.DataFrame(), args)
c = c[c.check.str.startswith(('3 ', 'M1 ', 'M2 '))]
c.to_csv(step / 'check3.csv', index=False)
for r in c.itertuples():
  print(f'{r.condition} | {r.check} | {r.value} | {r.detail}')
