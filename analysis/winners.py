"""Returned poses compared directly across platforms (section 10, row of drafts/section10_winners_spec_20261114.md, label
winners): a post hoc analysis of frozen outputs, asked by the reviewers, descriptive whatever it shows. No run and no
cloud resource.

    analysis/.venv/bin/python analysis/winners.py --main . --out results/analysis/winners_<DATE> \\
        [--replicates 10000] [--seed 2026] [--jobs 8]

Question: when the platform changes which sample AlphaFold 3 returns, how different are the two returned poses? The
registered contrasts compare samples with the same index (M4); this analysis compares the two returned samples.
Inputs, relative to --main (the checkout that holds results/ and the AlphaFold 3 output folders under results/af3/):
  main: results/analysis/main_20261014/primary_v2/samples.csv and contrasts/pairs_seed.csv: HW(V5,V6), HW(V6,G), RC(G)
  h100: results/analysis/h100_20261104/samples.csv (H) and results/analysis/a100_20261018/samples.csv (A), with
        results/analysis/h100_20261104/pairs_seed.csv: HW(H,A), RC(H), RC(A) (as analysis/h100.py builds them)
Sides as registered: HW(P,Q) is P r1 against Q r1 on the same seed; RC(P) is r1 against r2.
Levels:
  seed: each registered (target, seed) unit; the returned sample of each side is the registered top-1 (pairs_seed.csv).
  target: each target; the returned sample of each side is AlphaFold 3's main output, the sample with the highest
          full-precision ranking_score across the five seeds and their samples (exact ties to the lowest seed, then
          sample), over the seeds both sides ran.
Measures per unit: pose_rmsd, the ligand RMSD between the two returned samples after superposing their pockets
(score_ligand.pair_divergence, the function of M4; 0 when the coordinates are identical as written); the ligand RMSD of
each from the experimental pose (m1_ligand_rmsd) and its PoseBusters validity (m1_pb_valid). Counts:
  distinct: pose_rmsd > 2 A;
  validity change: both validities known and different;
  binding-mode change: distinct, one returned pose within 2 A of the experimental pose and the other beyond 4 A
  (BINDING_IN, BINDING_OUT).
A unit whose pose_rmsd cannot be computed (the three glycan targets: AlphaFold 3 removes atom O1) leaves pose_rmsd and
the counts that need it, and is listed.
Statistics per contrast, level and stratum (all units; units whose returned sample differs, "changed"): the median and
the 90th percentile of pose_rmsd and the shares of the three counts, each with a 95% percentile interval from the
bootstrap over targets (contrasts.boot_ci, --replicates, a fresh generator from --seed for every statistic); the
maximum of pose_rmsd with its target. Descriptive; no reading rule.
Checks (the analysis stops if 1 or 2 fails): 1, every registered top-1 names exactly one row of its per-sample table;
2, that row is the argmax of ranking_score of its (condition, repetition, target, seed) under contrasts.top and has
the registered success; 3 (reported), per contrast the number of targets whose main output changes (for HW(V5,V6) and
HW(V6,G), the paper's Results IV-B gives 16 and 25 from an earlier count).
Outputs in --out: units_seed.csv, units_target.csv, summary.csv, checks.csv, errors.csv, inputs.tsv, report.md.
"""
import argparse
import hashlib
import multiprocessing
import pathlib
import sys

import numpy as np
import pandas as pd

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / 'analysis'))
sys.path.insert(0, str(REPO / 'harness'))
import contrasts  # noqa: E402
import followup34 as f34  # noqa: E402
import returned_structures as rs  # noqa: E402

AN = 'results/analysis'
MAIN = f'{AN}/main_20261014/primary_v2'
A100 = f'{AN}/a100_20261018'
H100 = f'{AN}/h100_20261104'
DISTINCT_A = 2.0
BINDING_IN, BINDING_OUT = 2.0, 4.0
CONTRASTS = {'main': ['HW(V5,V6)', 'HW(V6,G)', 'RC(G)'], 'h100': ['HW(H,A)', 'RC(H)', 'RC(A)']}
EARLIER_TARGET_COUNTS = {'HW(V5,V6)': 16, 'HW(V6,G)': 25}


def pose_rmsd_real(task):
  """(target, path_a, path_b) -> (ligand RMSD between the two poses or None, error or None)."""
  target, a, b = task
  if a == b:
    return 0.0, None
  try:
    if contrasts.coords_text(a) == contrasts.coords_text(b):
      return 0.0, None
    d = contrasts.divergence(target, a, b)
    return d['ligand_rmsd'], d['error']
  except Exception as e:  # noqa: BLE001 - recorded, never silent
    return None, f'{type(e).__name__}: {e}'[:300]


POSE_RMSD = pose_rmsd_real     # tests replace it with a synthetic function


def load_sources(root):
  root = pathlib.Path(root)
  main = f34.load_samples(root / MAIN / 'samples.csv', ['main'])
  a = f34.load_samples(root / A100 / 'samples.csv', ['a100'])
  h = f34.load_samples(root / H100 / 'samples.csv', ['h100'])
  out = {'main': (f34.union(main[main.condition.isin(['V5', 'V6', 'G'])]),
                  pd.read_csv(root / MAIN / 'contrasts' / 'pairs_seed.csv')),
         'h100': (f34.union(a[a.condition == 'A'], h[h.condition == 'H']), pd.read_csv(root / H100 / 'pairs_seed.csv'))}
  inputs = [MAIN + '/samples.csv', MAIN + '/contrasts/pairs_seed.csv', A100 + '/samples.csv', H100 + '/samples.csv',
            H100 + '/pairs_seed.csv']
  return out, inputs


def side_row(units, c, rep, target, seed, sample, succ=None, where=''):
  g = units.get((c, rep, target, int(seed)))
  if g is None or int(sample) not in g.index:
    raise SystemExit(f'check 1: {where}: no sample {sample} of {c} {rep} {target} seed {seed}')
  if succ is not None:
    t, s, _ = contrasts.top(g)
    if int(t) != int(sample) or bool(s) != bool(succ):
      raise SystemExit(f'check 2: {where}: registered top-1 {sample} (success {succ}), samples give {t} (success {s})')
  return g.loc[int(sample)]


def seed_units(samples, pairs, contrast, source):
  sp = pairs[pairs.contrast == contrast]
  if sp.empty:
    raise SystemExit(f'{source}: {contrast} has no rows in pairs_seed.csv')
  (ca, ra), (cb, rb) = rs.sides(contrast)
  units = contrasts.seed_units(samples[samples.condition.isin([ca, cb])])
  rows = []
  for r in sp.itertuples():
    where = f'{source} {contrast} {r.target} seed {r.seed_a}'
    xa = side_row(units, ca, ra, r.target, r.seed_a, r.top1_a, r.success_a, where)
    xb = side_row(units, cb, rb, r.target, r.seed_b, r.top1_b, r.success_b, where)
    rows.append(unit_row(source, contrast, 'seed', r.target, int(r.seed_a), xa, xb, int(r.top1_a), int(r.top1_b)))
  return rows


def main_output(g):
  """AlphaFold 3's main output among a target's samples of several seeds: highest ranking_score, exact ties to the
  lowest (seed, sample)."""
  order = g.sort_values(['ranking_score', 'seed', 'sample'], ascending=[False, True, True], kind='mergesort')
  return order.iloc[0]


def target_units(samples, contrast, source):
  (ca, ra), (cb, rb) = rs.sides(contrast)
  a = samples[(samples.condition == ca) & (samples.repetition == ra)]
  b = samples[(samples.condition == cb) & (samples.repetition == rb)]
  rows = []
  for t in sorted(set(a.target) & set(b.target)):
    ga, gb = a[a.target == t], b[b.target == t]
    seeds = sorted(set(ga.seed) & set(gb.seed))
    xa, xb = main_output(ga[ga.seed.isin(seeds)]), main_output(gb[gb.seed.isin(seeds)])
    rows.append(unit_row(source, contrast, 'target', t, len(seeds), xa, xb, (int(xa.seed), int(xa['sample'])),
                         (int(xb.seed), int(xb['sample']))))
  return rows


def unit_row(source, contrast, level, target, seed_or_n, xa, xb, ta, tb):
  def pb(x):
    v = x.get('m1_pb_valid')
    return None if v is None or (isinstance(v, float) and np.isnan(v)) else str(v) == 'True'
  return {'source': source, 'contrast': contrast, 'level': level, 'target': target,
          ('seed' if level == 'seed' else 'seeds'): seed_or_n, 'returned_a': str(ta), 'returned_b': str(tb),
          'changed': ta != tb, 'path_a': xa['model_path'], 'path_b': xb['model_path'],
          'rmsd_exp_a': float(xa['m1_ligand_rmsd']) if pd.notna(xa['m1_ligand_rmsd']) else np.nan,
          'rmsd_exp_b': float(xb['m1_ligand_rmsd']) if pd.notna(xb['m1_ligand_rmsd']) else np.nan,
          'pb_valid_a': pb(xa), 'pb_valid_b': pb(xb)}


def add_pose_rmsd(df, jobs):
  tasks = sorted({(r.target, r.path_a, r.path_b) for r in df.itertuples()})
  if jobs > 1 and POSE_RMSD is pose_rmsd_real:
    with multiprocessing.Pool(jobs) as pool:
      res = dict(zip(tasks, pool.map(POSE_RMSD, tasks, chunksize=8)))
  else:
    res = {t: POSE_RMSD(t) for t in tasks}
  df['pose_rmsd'] = [res[(r.target, r.path_a, r.path_b)][0] for r in df.itertuples()]
  df['pose_error'] = [res[(r.target, r.path_a, r.path_b)][1] for r in df.itertuples()]
  df['pose_rmsd'] = pd.to_numeric(df['pose_rmsd'], errors='coerce')
  ok = df.pose_rmsd.notna()
  df['distinct'] = np.where(ok, df.pose_rmsd > DISTINCT_A, np.nan)
  known = df.pb_valid_a.notna() & df.pb_valid_b.notna()
  df['validity_change'] = np.where(known, df.pb_valid_a != df.pb_valid_b, np.nan)
  lo = df[['rmsd_exp_a', 'rmsd_exp_b']].min(axis=1)
  hi = df[['rmsd_exp_a', 'rmsd_exp_b']].max(axis=1)
  both = ok & df.rmsd_exp_a.notna() & df.rmsd_exp_b.notna()
  df['binding_mode_change'] = np.where(both, (df.pose_rmsd > DISTINCT_A) & (lo <= BINDING_IN) & (hi > BINDING_OUT),
                                       np.nan)
  return df


def p90(x):
  return float(np.percentile(x, 90)) if len(x) else np.nan


def summarize(df, reps, seed):
  rows = []
  for (source, contrast, level), u in df.groupby(['source', 'contrast', 'level'], sort=False):
    for stratum, g in (('all', u), ('changed', u[u.changed])):
      targets = sorted(g.target.unique())
      out = {'source': source, 'contrast': contrast, 'level': level, 'stratum': stratum, 'units': len(g),
             'n_targets': len(targets)}
      v = g.dropna(subset=['pose_rmsd'])
      groups = [v[v.target == t].pose_rmsd.values for t in sorted(v.target.unique())]
      out['units_with_pose_rmsd'] = len(v)
      for name, fn in (('median', contrasts.med), ('p90', p90)):
        lo, hi = contrasts.boot_ci(groups, fn, np.random.default_rng(seed), reps)
        out.update({f'pose_rmsd_{name}': fn(v.pose_rmsd.values), f'pose_rmsd_{name}_ci_low': lo,
                    f'pose_rmsd_{name}_ci_high': hi})
      if len(v):
        m = v.loc[v.pose_rmsd.idxmax()]
        out.update({'pose_rmsd_max': float(m.pose_rmsd), 'pose_rmsd_max_target': m.target})
      for count in ('distinct', 'validity_change', 'binding_mode_change'):
        w = g.dropna(subset=[count])
        grp = [w[w.target == t][count].astype(float).values for t in sorted(w.target.unique())]
        lo, hi = contrasts.boot_ci(grp, contrasts.mean, np.random.default_rng(seed), reps)
        out.update({f'{count}_n': int(w[count].astype(float).sum()), f'{count}_of': len(w),
                    f'{count}_share': contrasts.mean(w[count].astype(float).values), f'{count}_ci_low': lo,
                    f'{count}_ci_high': hi})
      rows.append(out)
  return pd.DataFrame(rows)


def checks_table(df):
  rows = []
  t = df[df.level == 'target']
  for contrast, g in t.groupby('contrast', sort=False):
    earlier = EARLIER_TARGET_COUNTS.get(contrast)
    n = int(g.changed.sum())
    rows.append({'check': '3 main output changes', 'contrast': contrast, 'value': f'{n} of {len(g)} targets',
                 'detail': '' if earlier is None else ('equals' if n == earlier else 'DIFFERS from')
                 + f' the earlier count {earlier} (Results IV-B)'})
  return pd.DataFrame(rows)


def fmt(x, nd=2):
  return 'n/a' if x != x else f'{x:.{nd}f}'


def report(s, chk, args):
  L = ['# Returned poses compared directly (label winners)', '',
       'Post hoc, descriptive (drafts/section10_winners_spec_20261114.md). pose RMSD in angstrom between the two '
       f'returned samples; 95% percentile intervals over targets ({args.replicates} replicates, seed {args.seed}).', '',
       '| contrast | level | stratum | units | median [CI] | p90 [CI] | max (target) | distinct > 2 A | validity change | '
       'binding-mode change |', '|---|---|---|---|---|---|---|---|---|---|']
  for r in s.to_dict('records'):
    def share(c):
      return f'{r[c + "_n"]} of {r[c + "_of"]}, {100 * r[c + "_share"]:.1f}% [{100 * r[c + "_ci_low"]:.1f}, ' \
             f'{100 * r[c + "_ci_high"]:.1f}]' if r[c + '_of'] else 'n/a'
    L.append(f'| {r["contrast"]} | {r["level"]} | {r["stratum"]} | {r["units"]} | {fmt(r["pose_rmsd_median"])} '
             f'[{fmt(r["pose_rmsd_median_ci_low"])}, {fmt(r["pose_rmsd_median_ci_high"])}] | {fmt(r["pose_rmsd_p90"])} '
             f'[{fmt(r["pose_rmsd_p90_ci_low"])}, {fmt(r["pose_rmsd_p90_ci_high"])}] | {fmt(r.get("pose_rmsd_max", np.nan))} '
             f'({r.get("pose_rmsd_max_target", "")}) | {share("distinct")} | {share("validity_change")} | '
             f'{share("binding_mode_change")} |')
  L += ['', '## Check 3', ''] + [f'- {r.contrast}: {r.value} {r.detail}' for r in chk.itertuples()]
  return '\n'.join(L) + '\n'


def sha256(path):
  h = hashlib.sha256()
  with open(path, 'rb') as f:
    for chunk in iter(lambda: f.read(1 << 20), b''):
      h.update(chunk)
  return h.hexdigest()


def run(sources, out, args, inputs=(), root=None, items=None):
  out = pathlib.Path(out)
  out.mkdir(parents=True, exist_ok=True)
  rows = []
  for source, names in (items or CONTRASTS).items():
    samples, pairs = sources[source]
    for c in names:
      rows += seed_units(samples, pairs, c, source)
      rows += target_units(samples, c, source)
  df = add_pose_rmsd(pd.DataFrame(rows), getattr(args, 'jobs', 1))
  s = summarize(df, args.replicates, args.seed)
  chk = checks_table(df)
  df[df.level == 'seed'].drop(columns=['seeds'], errors='ignore').to_csv(out / 'units_seed.csv', index=False)
  df[df.level == 'target'].drop(columns=['seed'], errors='ignore').to_csv(out / 'units_target.csv', index=False)
  df[df.pose_error.notna()][['source', 'contrast', 'level', 'target', 'pose_error']].to_csv(out / 'errors.csv',
                                                                                              index=False)
  s.to_csv(out / 'summary.csv', index=False)
  chk.to_csv(out / 'checks.csv', index=False)
  if root is not None:
    (out / 'inputs.tsv').write_text(''.join(f'{p}\t{sha256(pathlib.Path(root) / p)}\n' for p in inputs))
  (out / 'report.md').write_text(report(s, chk, args))
  return df, s, chk


def main():
  ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
  ap.add_argument('--main', default='.', help='the checkout that holds results/')
  ap.add_argument('--out', required=True)
  ap.add_argument('--replicates', type=int, default=10000)
  ap.add_argument('--seed', type=int, default=2026)
  ap.add_argument('--jobs', type=int, default=8)
  args = ap.parse_args()
  out = pathlib.Path(args.out)
  if (out / 'summary.csv').exists():
    raise SystemExit(f'{out}/summary.csv exists; the output folder must be new')
  sources, inputs = load_sources(args.main)
  run(sources, out, args, inputs, args.main)
  print(f'wrote {out}')


if __name__ == '__main__':
  main()
