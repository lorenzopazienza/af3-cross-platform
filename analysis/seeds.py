"""Follow-up 8, how many seeds are needed (drafts/section10_fu7-10.md; label seeds; reported as a follow-up, outside
the pre-registered study, whatever it shows): Q3 of section 5 extended from 5 to 20 seeds on V5 and G.

    analysis/.venv/bin/python analysis/seeds.py --samples results/analysis/seeds_<date>/samples.csv \\
        --out results/analysis/seeds_<date> [--main results/analysis/main_20261014/primary_v2]

Data: seeds 1 to 5 of V5 and G (r1) from the main study's per-sample table, seeds 6 to 20 (r1) from the seeds sessions
(analysis/collect.py --labels seeds); a (condition, target, seed) in both is an error. Checks, before anything else
(the analysis stops if one fails): exit codes; every seed 6 to 20 of every target present once per condition
(an incomplete condition is reported and Q3 is computed only on the seeds every target has, never imputed).
Q3, exactly as section 5 and analysis/contrasts.q3 define it (that function, unchanged): for n = 1..N seeds, the
overall top-1 over the first n seeds (highest ranking_score), its success rate per condition, and the V5-G
difference on the common targets with the target bootstrap (10,000 replicates, seed 2026); the answer is the
smallest n at which every difference interval lies within +-2 percentage points, or "not reached". For n <= 5 the
values must equal the V5-G rows of the main study's Q3 restricted to V5 and G (checked, as a consistency check).
Every result uses M1, so it is also given on the 39 targets released after 2021-09-30.
Outputs in --out: checks.csv, completeness.csv, q3.csv (column targets), answer.json, report.md.
"""
import argparse
import json
import pathlib
import sys

import pandas as pd

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / 'analysis'))
import contrasts  # noqa: E402
import followup34 as f34  # noqa: E402

CONDITIONS = ['V5', 'G']
NEW_SEEDS = list(range(6, 21))
CUTOFF = '2021-09-30'


def load(samples, main_dir):
  new = f34.load_samples(samples, ['seeds'])
  new = new[new.condition.isin(CONDITIONS)]
  main = f34.load_samples(pathlib.Path(main_dir) / 'samples.csv', ['main'])
  main = main[main.condition.isin(CONDITIONS) & (main.repetition == 'r1')]
  return main, new


def checks(main, new, targets):
  rows = f34.exit_code_check(new, CONDITIONS, 'seeds')
  stray = sorted(set(new.seed) - set(NEW_SEEDS))
  rows.append({'check': 'seeds of the seeds sessions are 6 to 20', 'condition': ', '.join(CONDITIONS),
               'value': f'{sorted(set(new.seed))}', 'detail': 'pass' if not stray else f'FAIL: seeds {stray}'})
  both = set(map(tuple, main[['condition', 'target', 'seed']].drop_duplicates().values)) & \
      set(map(tuple, new[['condition', 'target', 'seed']].drop_duplicates().values))
  rows.append({'check': 'no (condition, target, seed) in both sources', 'condition': ', '.join(CONDITIONS),
               'value': len(both), 'detail': 'pass' if not both else f'FAIL: {sorted(both)[:3]}'})
  return rows


def completeness(new, targets):
  rows = []
  for c in CONDITIONS:
    g = f34.sample_rows(new[(new.condition == c) & (new.repetition == 'r1')])
    units = g.groupby(['target', 'seed']).size()
    have = int((units == 5).sum())
    rows.append({'condition': c, 'planned_units': len(targets) * len(NEW_SEEDS), 'units_with_5_samples': have,
                 'status': 'complete' if have == len(targets) * len(NEW_SEEDS) else 'incomplete'})
  return pd.DataFrame(rows)


def common_seeds(ok, targets):
  """The seeds every target has in every condition (Q3 uses the first n of them; a missing one is never imputed)."""
  have = None
  for c in CONDITIONS:
    for t in targets:
      s = set(ok[(ok.condition == c) & (ok.target == t)].seed)
      have = s if have is None else have & s
  return sorted(have or [])


def q3_on(ok, targets, args):
  sub = ok[ok.target.isin(targets)]
  seeds = common_seeds(sub, targets)
  sub = sub[sub.seed.isin(seeds)]
  rows, answer = contrasts.q3(sub, args)
  return rows, answer, seeds


def consistency(q, main_dir):
  """For n <= 5 the V5, G and V5-G rows must equal the main study's Q3 (primary_v2/contrasts/q3.csv)."""
  m = pd.read_csv(pathlib.Path(main_dir) / 'contrasts' / 'q3.csv')
  m = m[(m.n_seeds <= 5) & m.condition.isin(['V5', 'G', 'V5-G'])].set_index(['n_seeds', 'condition', 'statistic'])
  mine = q[(q.targets == 'all 60') & (q.n_seeds <= 5) & q.condition.isin(['V5', 'G', 'V5-G'])]
  mine = mine.set_index(['n_seeds', 'condition', 'statistic'])
  common = sorted(set(m.index) & set(mine.index))
  bad = [k for k in common for col in ('estimate', 'ci_low', 'ci_high')
         if not (pd.isna(m.loc[k, col]) and pd.isna(mine.loc[k, col])) and abs(m.loc[k, col] - mine.loc[k, col]) > 1e-12]
  return {'check': 'Q3 for n <= 5 equals the main study (V5, G, V5-G rows)', 'condition': 'V5, G',
          'value': f'{len(common) - len(set(b for b in bad))}/{len(common)} rows equal',
          'detail': 'pass' if common and not bad else ('FAIL: ' + str(bad[:3]) if bad else 'FAIL: no common rows')}


def report(out, chk, comp, q, answers):
  L = ['# Follow-up 8: how many seeds are needed (Q3 to 20 seeds, V5 and G)', '', '## Checks', '',
       '| check | condition | value | detail |', '|---|---|---|---|']
  L += [f'| {r["check"]} | {r["condition"]} | {r["value"]} | {r["detail"]} |' for r in chk]
  if comp is None:
    (out / 'report.md').write_text('\n'.join(L + ['', 'A check failed: nothing was computed.']) + '\n')
    return
  L += ['', '## Completeness', '', '| condition | planned | with 5 samples | status |', '|---|---|---|---|']
  L += [f'| {r.condition} | {r.planned_units} | {r.units_with_5_samples} | {r.status} |' for r in comp.itertuples()]
  L += ['', '## Answer', '']
  L += [f'- {k}: {v["answer"] if v["answer"] is not None else "not reached"} (seeds used: {v["seeds"][0] if v["seeds"] else "-"} '
        f'to {v["seeds"][-1] if v["seeds"] else "-"})' for k, v in answers.items()]
  L += ['', '## V5-G success difference by number of seeds', '', '| targets | n seeds | estimate | 95% CI |', '|---|---|---|---|']
  d = q[q.statistic == 'success_difference']
  L += [f'| {r.targets} | {r.n_seeds} | {f34.pct(r.estimate)} | {f34.pct(r.ci_low)} to {f34.pct(r.ci_high)} |'
        for r in d.itertuples()]
  (out / 'report.md').write_text('\n'.join(L) + '\n')


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
  targets = sorted(tt.pdb_id)
  post = sorted(tt[tt.rcsb_initial_release.str[:10] > CUTOFF].pdb_id)
  main_d, new = load(args.samples, args.main)
  chk = checks(main_d, new, targets)
  pd.DataFrame(chk).to_csv(out / 'checks.csv', index=False)
  if any(r['detail'] != 'pass' for r in chk):
    report(out, chk, None, None, None)
    raise SystemExit('a check failed (checks.csv); Q3 is not computed')
  comp = completeness(new, targets)
  ok = f34.union(main_d, new)
  parts, answers = [], {}
  for label, ts in (('all 60', targets), (f'post-cutoff {len(post)}', post)):
    q, answer, seeds = q3_on(ok, ts, args)
    parts.append(q.assign(targets=label))
    answers[label] = {'answer': answer, 'seeds': seeds}
  q = pd.concat(parts, ignore_index=True)
  chk.append(consistency(q, args.main))
  pd.DataFrame(chk).to_csv(out / 'checks.csv', index=False)
  for name, df in (('completeness', comp), ('q3', q)):
    df.to_csv(out / f'{name}.csv', index=False)
  (out / 'answer.json').write_text(json.dumps(answers, indent=2) + '\n')
  report(out, chk, comp, q, answers)
  print((out / 'report.md').read_text())
  return 0


if __name__ == '__main__':
  sys.exit(main())
