"""Follow-up 9, the H100 (drafts/section10_fu7-10.md; label h100; reported as a follow-up, outside the pre-registered
study, whatever it shows).

    analysis/.venv/bin/python analysis/h100.py --samples results/analysis/h100_<date>/samples.csv \\
        --out results/analysis/h100_<date> [--a100 results/analysis/a100_20261018/samples.csv]
        [--main results/analysis/main_20261014/primary_v2]

Conditions: H (h100_xla on a3-highgpu-1g, the settings of A), r1 on the 60 targets and, if it ran, r2 on the 10 pilot
targets; A (follow-up 3, r1 and r2); V5, V6 and G (main study, r1).
Checks, before anything else (the analysis stops if one fails): one machine type, a3-highgpu-1g, in every H session;
exit codes. An arm short of its (target, seed) is reported as incomplete; nothing is imputed.
Contrasts, paired as in section 5, with its statistics and the target-clustered bootstrap (10,000 replicates, seed
2026): HW(H,A) (two GPU generations, the counterpart of HW(V5,V6)), HW(H,G), HW(H,V6), HW(H,V5), SD(H), and RC(H) if
H r2 ran. HW minus RC for HW(H,A): against RC(A) (follow-up 3, 60 targets) and, if H r2 ran, against RC(H) on its 10
targets. Headline: the top-1 flip rate; the success flip rate and flips that matter are secondary, and every result
that uses M1 is also given on the 39 targets released after 2021-09-30.
Outputs in --out: checks.csv, completeness.csv, contrasts.csv, hw_minus_rc.csv, pairs_seed.csv, pairs_sample.csv,
report.md.
"""
import argparse
import pathlib
import sys

import pandas as pd

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / 'analysis'))
import contrasts  # noqa: E402
import followup34 as f34  # noqa: E402

MACHINE_TYPE = 'a3-highgpu-1g'
SPECS = [('HW(H,A)', 'HW', ('H', 'r1'), ('A', 'r1'), True), ('HW(H,G)', 'HW', ('H', 'r1'), ('G', 'r1'), False),
         ('HW(H,V6)', 'HW', ('H', 'r1'), ('V6', 'r1'), False), ('HW(H,V5)', 'HW', ('H', 'r1'), ('V5', 'r1'), False),
         ('SD(H)', 'SD', ('H', 'r1'), ('H', 'r1'), True), ('RC(H)', 'RC', ('H', 'r1'), ('H', 'r2'), True),
         ('RC(A)', 'RC', ('A', 'r1'), ('A', 'r2'), True)]
M1_STATS = ('flip_success', 'flip_matters')
CUTOFF = '2021-09-30'


def completeness(ok, targets, pilot):
  rows = []
  for rep, ts in (('r1', targets), ('r2', pilot)):
    units = ok[(ok.condition == 'H') & (ok.repetition == rep)].groupby(['target', 'seed']).size()
    have = int((units == 5).sum())
    rows.append({'condition': 'H', 'repetition': rep, 'planned_units': len(ts) * 5, 'units_with_5_samples': have,
                 'status': 'complete' if have == len(ts) * 5 else ('not run' if have == 0 and rep == 'r2' else 'incomplete')})
  return pd.DataFrame(rows)


def main():
  ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
  ap.add_argument('--samples', required=True)
  ap.add_argument('--a100', default=str(REPO / 'results' / 'analysis' / 'a100_20261018' / 'samples.csv'))
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
  pilot = sorted(tt[tt.pilot == 'yes'].pdb_id)
  h = f34.load_samples(args.samples, ['h100'])
  h = h[h.condition == 'H']
  a = f34.load_samples(args.a100, ['a100'])
  a = a[a.condition == 'A']
  main = f34.load_samples(pathlib.Path(args.main) / 'samples.csv', ['main'])
  main = main[main.condition.isin(['V5', 'V6', 'G']) & (main.repetition == 'r1')]
  mt = f34.one_machine_type(h, ['H'])
  chk = [{'check': 'machine type of every H session', 'condition': 'H', 'value': mt,
          'detail': 'pass' if mt == MACHINE_TYPE else f'FAIL: expected {MACHINE_TYPE}'}]
  chk += f34.exit_code_check(h, ['H'], 'h100')
  pd.DataFrame(chk).to_csv(out / 'checks.csv', index=False)
  if any(r['detail'] != 'pass' for r in chk):
    raise SystemExit('a check failed (checks.csv); nothing is compared')
  ok = f34.union(main, a, h)
  comp = completeness(ok, targets, pilot)
  sp, sm, cons, counts = f34.build(ok, SPECS)
  parts_c, parts_h = [], []
  for label, ts, stats in (('all 60', targets, contrasts.FLIP_STATS), (f'post-cutoff {len(post)}', post, M1_STATS)):
    s1, s2 = sp[sp.target.isin(ts)], sm[sm.target.isin(ts)]
    con = contrasts.summarize(s1, s2, cons, args)
    if label != 'all 60':
      con = con[con.statistic.isin(stats)]
    rc_names = {n: cons[n] for n in ('HW(H,A)', 'RC(H)', 'RC(A)') if n in cons}
    hw = contrasts.hw_minus_rc(s1[s1.contrast.isin(list(rc_names))], rc_names, args)
    hw = hw[hw.statistic.isin(stats)] if len(hw) else hw
    parts_c.append(con.assign(targets=label))
    parts_h.append(hw.assign(targets=label))
  con = pd.concat(parts_c, ignore_index=True)
  hwrc = pd.concat(parts_h, ignore_index=True)
  if 'RC(H)' not in cons:
    hwrc = pd.concat([hwrc, pd.DataFrame([{'hardware': 'HW(H,A)', 'recompilation': 'RC(H)', 'statistic': st,
                                           'estimate': float('nan'), 'targets': 'all 60',
                                           'note': 'not available: H r2 did not run'} for st in contrasts.FLIP_STATS])],
                     ignore_index=True)
  for name, df in (('completeness', comp), ('counts', counts), ('contrasts', con), ('hw_minus_rc', hwrc),
                   ('pairs_seed', sp), ('pairs_sample', sm)):
    df.to_csv(out / f'{name}.csv', index=False)
  L = ['# Follow-up 9: the H100', '', f'Machine type {mt}.', '', '## Completeness', '',
       '| repetition | planned | with 5 samples | status |', '|---|---|---|---|']
  L += [f'| {r.repetition} | {r.planned_units} | {r.units_with_5_samples} | {r.status} |' for r in comp.itertuples()]
  L += ['', '## Contrasts (headline: top-1 flip rate)', ''] + f34.contrast_lines(con[con.targets == 'all 60'])
  (out / 'report.md').write_text('\n'.join(L) + '\n')
  print((out / 'report.md').read_text())
  return 0


if __name__ == '__main__':
  sys.exit(main())
