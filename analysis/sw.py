"""Follow-up 7, the software stack on the same chip (drafts/section10_fu7-10.md; label sw; reported as a follow-up,
outside the pre-registered study, whatever it shows).

    analysis/.venv/bin/python analysis/sw.py --samples results/analysis/sw_<date>/samples.csv \\
        --out results/analysis/sw_<date> [--main results/analysis/main_20261014/primary_v2]

Conditions: V6s and Gs, the sw sessions (V6 and G with the alternative stack: jax and jaxlib 0.10.1 with libtpu 0.0.41,
or with jax-cuda12-plugin and -pjrt 0.10.1), against V6 and G r1 of the main study (jax and jaxlib 0.10.2, libtpu
0.0.43.2 or jax-cuda12 0.10.2).
Checks, before anything else (the analysis stops if one fails): exit codes; every sw run's packages
(session.json probe) are the alternative stack and every main-study run's the original one; no (condition, target,
seed) twice. An arm short of its 300 (target, seed) is reported as incomplete and its results say so; nothing is
imputed.
Contrasts, paired on (target, seed) and (target, seed, sample) as in section 5 (analysis/followup34.build), with its
statistics and the target-clustered bootstrap (10,000 replicates, seed 2026): SW(V6) = V6 r1 against V6s r1, SW(G) =
G r1 against Gs r1. Comparison, on the same (target, seed) and resampled targets: each flip rate of SW(P) minus that of
RC(G) and minus that of HW(V6,G) (main-study seed pairs, primary_v2/contrasts/pairs_seed.csv), read as lower, similar
or higher if the interval lies below 0, contains 0 or lies above 0. Reading rule (drafts row): the headline is the
top-1 flip rate; the software stack moves the top-1 "as much as the chip" if SW(P) minus HW(V6,G) has an interval
containing or above 0, and "less than the chip" if it lies below 0; the success flip rate and flips that matter are
reported alongside as secondary, and every result that uses M1 also on the 39 targets released after 2021-09-30.
Outputs in --out: checks.csv, completeness.csv, contrasts.csv, comparisons.csv, reading_rule.csv, pairs_seed.csv,
pairs_sample.csv, report.md.
"""
import argparse
import json
import pathlib
import sys

import numpy as np
import pandas as pd

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / 'analysis'))
import contrasts  # noqa: E402
import followup34 as f34  # noqa: E402

CODE = {'V6': 'V6s', 'G': 'Gs'}
ALT = {'V6s': {'jax': '0.10.1', 'jaxlib': '0.10.1', 'libtpu': '0.0.41'},
       'Gs': {'jax': '0.10.1', 'jaxlib': '0.10.1', 'jax-cuda12-plugin': '0.10.1', 'jax-cuda12-pjrt': '0.10.1'}}
ORIGINAL = {'V6': {'jax': '0.10.2', 'jaxlib': '0.10.2', 'libtpu': '0.0.43.2'},
            'G': {'jax': '0.10.2', 'jaxlib': '0.10.2', 'jax-cuda12-plugin': '0.10.2', 'jax-cuda12-pjrt': '0.10.2'}}
SPECS = [('SW(V6)', 'SW', ('V6', 'r1'), ('V6s', 'r1'), True), ('SW(G)', 'SW', ('G', 'r1'), ('Gs', 'r1'), True)]
COMPARE = [('RC(G)', 'recompilation on G'), ('HW(V6,G)', 'the chip (TPU v6e against L4)')]
HEADLINE = 'flip_top1'
M1_STATS = ('flip_success', 'flip_matters')
CUTOFF = '2021-09-30'
contrasts.ROLE.setdefault('SW', 'software stack (follow-up 7)')


def packages(session_path):
  """The probe packages of a harness run (results/af3/<session>/<run>/session.json)."""
  p = REPO / session_path / 'session.json'
  return ((json.loads(p.read_text()).get('probe') or {}).get('packages') or {}) if p.is_file() else {}


def stack_check(d, want, label):
  rows = []
  for c, pins in want.items():
    runs = sorted(d[d.condition == c].session.unique())
    bad = [r for r in runs if any(packages(r).get(k) != v for k, v in pins.items())]
    rows.append({'check': f'packages of every {label} run are {", ".join(f"{k} {v}" for k, v in pins.items())}',
                 'condition': c, 'value': f'{len(runs) - len(bad)}/{len(runs)} runs',
                 'detail': 'pass' if runs and not bad else ('FAIL: no run' if not runs else f'FAIL: {bad[:2]}')})
  return rows


def completeness(ok, targets):
  rows = []
  for c in CODE.values():
    units = ok[(ok.condition == c) & (ok.repetition == 'r1')].groupby(['target', 'seed']).size()
    have = int((units == 5).sum())
    rows.append({'condition': c, 'planned_units': len(targets) * 5, 'units_with_5_samples': have,
                 'status': 'complete' if have == len(targets) * 5 else 'incomplete'})
  return pd.DataFrame(rows)


def diff_boot(a_groups, b_groups, args):
  if not a_groups:
    return np.nan, np.nan, np.nan
  est = np.mean(np.concatenate(a_groups)) - np.mean(np.concatenate(b_groups))
  rng = np.random.default_rng(args.seed)
  vals = [np.mean(np.concatenate([a_groups[i] for i in row])) - np.mean(np.concatenate([b_groups[i] for i in row]))
          for row in rng.integers(0, len(a_groups), size=(args.replicates, len(a_groups)))]
  return float(est), float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def reading(lo, hi):
  if lo != lo:
    return 'not computed'
  return 'lower' if hi < 0 else 'higher' if lo > 0 else 'similar'


def comparisons(sp, main_pairs, targets, args, stats):
  """Each flip rate of SW(P) minus that of a main-study contrast, on the (target, seed) both have."""
  rows = []
  for sw in ('SW(V6)', 'SW(G)'):
    a = sp[(sp.contrast == sw) & sp.target.isin(targets)].groupby(['target', 'seed_a']).first()
    for ref, what in COMPARE:
      b = main_pairs[(main_pairs.contrast == ref) & main_pairs.target.isin(targets)].groupby(['target', 'seed_a']).first()
      common = sorted(set(a.index) & set(b.index))
      ts = sorted({t for t, _ in common})
      for st in stats:
        ag = [np.array([float(a.loc[(t, s), st]) for t2, s in common if t2 == t]) for t in ts]
        bg = [np.array([float(b.loc[(t, s), st]) for t2, s in common if t2 == t]) for t in ts]
        est, lo, hi = diff_boot(ag, bg, args)
        rows.append({'software': sw, 'reference': ref, 'reference_is': what, 'statistic': st, 'n': len(common),
                     'n_targets': len(ts), 'estimate': est, 'ci_low': lo, 'ci_high': hi, 'reading': reading(lo, hi)})
  return pd.DataFrame(rows)


def reading_rule(comp, stats, bad):
  rows = []
  for st in sorted(stats, key=lambda x: x != HEADLINE):
    for sw in ('SW(V6)', 'SW(G)'):
      r = comp[(comp.software == sw) & (comp.reference == 'HW(V6,G)') & (comp.statistic == st)]
      read = r.reading.iloc[0] if len(r) else 'not computed'
      branch = {'lower': 'the software stack moves the top-1 less than the chip',
                'similar': 'the software stack moves the top-1 as much as the chip',
                'higher': 'the software stack moves the top-1 as much as the chip (more, by this interval)'}.get(read, read)
      rows.append({'statistic': st, 'role': 'headline' if st == HEADLINE else 'secondary', 'software': sw,
                   'SW minus HW(V6,G)': read, 'branch': branch,
                   'incomplete_arms': ', '.join(sorted(bad & {CODE[sw[3:-1]]}))})
  return pd.DataFrame(rows)


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
  targets, post = sorted(tt.pdb_id), sorted(tt[tt.rcsb_initial_release.str[:10] > CUTOFF].pdb_id)
  sw = f34.load_samples(args.samples, ['sw'])
  sw = sw[sw.platform.isin(list(CODE))].copy()
  sw['condition'] = sw.platform.map(CODE)
  main = f34.load_samples(pathlib.Path(args.main) / 'samples.csv', ['main'])
  main = main[main.condition.isin(list(CODE)) & (main.repetition == 'r1')]
  chk = f34.exit_code_check(sw, list(CODE.values()), 'sw') + stack_check(sw, ALT, 'sw') + stack_check(main, ORIGINAL, 'main-study')
  pd.DataFrame(chk).to_csv(out / 'checks.csv', index=False)
  if any(r['detail'] != 'pass' for r in chk):
    raise SystemExit('a check failed (checks.csv); nothing is compared')
  ok = f34.union(main, sw)
  comp = completeness(ok, targets)
  bad = set(comp[comp.status != 'complete'].condition)
  sp, sm, cons, counts = f34.build(ok, SPECS)
  main_pairs = pd.read_csv(pathlib.Path(args.main) / 'contrasts' / 'pairs_seed.csv')
  outs = {'contrasts': [], 'comparisons': [], 'reading_rule': []}
  for label, ts, stats in (('all 60', targets, contrasts.FLIP_STATS), (f'post-cutoff {len(post)}', post, M1_STATS)):
    con = contrasts.summarize(sp[sp.target.isin(ts)], sm[sm.target.isin(ts)], cons, args)
    if label != 'all 60':
      con = con[con.statistic.isin(stats)]
    c = comparisons(sp, main_pairs, ts, args, stats)
    for k, df in (('contrasts', con), ('comparisons', c), ('reading_rule', reading_rule(c, stats, bad))):
      outs[k].append(df.assign(targets=label))
  res = {k: pd.concat(v, ignore_index=True) for k, v in outs.items()}
  res['contrasts']['incomplete_arm'] = res['contrasts'].contrast.map(
      lambda n: ', '.join(sorted(bad & {CODE[n[3:-1]]})))
  for name, df in (('completeness', comp), ('counts', counts), ('pairs_seed', sp), ('pairs_sample', sm), *res.items()):
    df.to_csv(out / f'{name}.csv', index=False)
  L = ['# Follow-up 7: the software stack on the same chip', '', '## Reading rule (headline: top-1 flip rate)', '']
  L += [f'- {r.targets}, {r.statistic} ({r.role}), {r.software}: {r.branch}'
        + (f' (incomplete arm: {r.incomplete_arms})' if r.incomplete_arms else '') for r in res['reading_rule'].itertuples()]
  L += ['', '## SW minus the main-study contrasts', '', '| targets | SW | minus | statistic | estimate | 95% CI | reading |',
        '|---|---|---|---|---|---|---|']
  L += [f'| {r.targets} | {r.software} | {r.reference} | {r.statistic} | {f34.pct(r.estimate)} | {f34.pct(r.ci_low)} to '
        f'{f34.pct(r.ci_high)} | {r.reading} |' for r in res['comparisons'].itertuples()]
  L += ['', '## Contrasts', ''] + f34.contrast_lines(res['contrasts'][res['contrasts'].targets == 'all 60'])
  (out / 'report.md').write_text('\n'.join(L) + '\n')
  print((out / 'report.md').read_text())
  return 0


if __name__ == '__main__':
  sys.exit(main())
