"""Which success criterion changes in a success flip, and by how much (section 10, row of
drafts/section10_pb_flips_spec_20261116.md, label pb_flips): a post hoc analysis of frozen outputs, asked by the
reviewers, descriptive whatever it shows. No run and no cloud resource.

    analysis/.venv/bin/python analysis/pb_flips.py --main . --out results/analysis/pb_flips_<DATE> \\
        [--replicates 10000] [--seed 2026] [--jobs 8]

Question: are success flips changes of pose, or crossings of a threshold near its limit?
Units: every success flip (flip_success, the registered top-1 of one side succeeds and the other's does not) of
  main: HW(V5,V6), HW(V6,G), RC(G) (results/analysis/main_20261014/primary_v2: samples.csv, contrasts/pairs_seed.csv)
  h100: HW(H,A), RC(H), RC(A) (results/analysis/h100_20261104/pairs_seed.csv; H from h100_20261104/samples.csv, A from
        a100_20261018/samples.csv), loaded as analysis/winners.py loads them (winners.load_sources).
Groups: "platform" = HW(V5,V6), HW(V6,G), HW(H,A); "recompilation" = RC(G), RC(H), RC(A).
Per flip, from the saved M1 results (m1_ligand_rmsd, m1_pb_valid, m1_pb_failed, m1_error of the two top-1 rows):
  the failing side (success False) and the criterion that fails on it: "RMSD" (ligand RMSD >= 2 A, PoseBusters-valid),
  "PoseBusters" (RMSD < 2 A, not valid), "RMSD and PoseBusters", or "not scored" (m1_error); the PoseBusters checks
  that fail on it. Success needs RMSD < 2 A and every check, so on the succeeding side nothing fails.
Check 0 (stops): posebusters is version 0.6.5, the study's.
Rerun (only the top-1 sample of each side of each flip; unique files): score_ligand.score, the registered M1 function
  (posebusters 0.6.5, configuration "dock", the predicted protein as mol_cond), must reproduce the saved M1 exactly:
  the ligand RMSD as written (4 decimals), the validity, the set of failed checks, the success and the error
  (check 3; any difference stops the analysis before any statistic). Then the same pipeline once more with
  PoseBusters' full report (bust(full_report=True)), whose 22 checks must give the same failed set (check 3b), for the
  continuous value behind each check (CHECKS below; the thresholds are those of posebusters' dock.yml) and its signed
  excess over the tolerance: (value - threshold) / threshold for an upper limit, (threshold - value) / threshold for a
  lower limit; > 0 fails (check 4: the sign must agree with the check's outcome on every file, or the analysis stops).
  The RMSD criterion is read the same way: excess (RMSD - 2) / 2, >= 0 fails. Six checks are binary (loading,
  sanitization, InChI, connectivity, radicals) and have no excess.
Per flip: the failing side's excess on each failing criterion and its worst (largest) excess; the succeeding side's
  excess on the same criteria (how far inside); the pose RMSD between the two top-1 samples (winners.pose_rmsd_real,
  the function of M4 and of winners).
Classification (chosen before any value): "pose change" if the two top-1 poses are more than 2 A apart (winners'
  "distinct"); otherwise "near-threshold crossing" if every criterion failing on the failing side exceeds its
  tolerance by at most NEAR (10% of the threshold; for the RMSD, below 2.2 A); otherwise "other". "not classified" if
  a needed value is missing (a failing binary check, a pose RMSD that cannot be computed, or a check whose value is
  missing).
Statistics per contrast and group, stratum all flips and, beside it, flips whose top-1 sample changed: counts per
  criterion, per failing check and per class; the shares of "near-threshold crossing" and "pose change" over the
  classified flips, and the median worst excess of the failing side, each with a 95% percentile interval from the
  bootstrap over targets (contrasts.boot_ci, --replicates, a fresh generator from --seed for every statistic; in a
  group the flips of a target in several contrasts are pooled under that target). No correction for multiple
  comparisons; descriptive, no reading rule beyond the classification.
Outputs in --out: flips.csv (one row per flip), rescored.csv (one row per rerun file, every check's value and excess),
  summary.csv, checks.csv, inputs.tsv, report.md.
"""
import argparse
import importlib.metadata
import multiprocessing
import pathlib
import sys
import tempfile

import numpy as np
import pandas as pd

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / 'analysis'))
sys.path.insert(0, str(REPO / 'harness'))
import contrasts  # noqa: E402
import returned_structures as rs  # noqa: E402
import winners as w  # noqa: E402

SUCCESS_RMSD = 2.0
NEAR = 0.10
PB_VERSION = '0.6.5'
CONTRASTS = {'main': ['HW(V5,V6)', 'HW(V6,G)', 'RC(G)'], 'h100': ['HW(H,A)', 'RC(H)', 'RC(A)']}
GROUPS = {'platform': ['HW(V5,V6)', 'HW(V6,G)', 'HW(H,A)'], 'recompilation': ['RC(G)', 'RC(H)', 'RC(A)']}
# check -> (value column or function of the full-report row, 'upper' | 'lower' limit, threshold, what the value is).
# Thresholds of posebusters 0.6.5 configuration "dock" (posebusters/config/dock.yml; tests read the file).
CHECKS = {
    'bond_lengths': (lambda r: max(r['longest_bond_relative_length'] - 1, 1 - r['shortest_bond_relative_length']),
                     'upper', 0.25, 'largest relative deviation of a bond length from its distance-geometry bound'),
    'bond_angles': (lambda r: r['most_extreme_relative_angle'] - 1, 'upper', 0.25,
                    'largest relative deviation of a 1-3 distance from its bound'),
    'internal_steric_clash': (lambda r: 1 - r['shortest_noncovalent_relative_distance'], 'upper', 0.3,
                              'largest relative shortfall of a non-bonded distance below its lower bound'),
    'aromatic_ring_flatness': ('aromatic_ring_maximum_distance_from_plane', 'upper', 0.25,
                               'largest atom distance from the ring plane, A'),
    'non-aromatic_ring_non-flatness': ('non-aromatic_ring_maximum_distance_from_plane', 'lower', 0.05,
                                       'smallest out-of-plane distance of a non-aromatic ring, A'),
    'double_bond_flatness': ('double_bond_maximum_distance_from_plane', 'upper', 0.25,
                             'largest atom distance from the double-bond plane, A'),
    'internal_energy': ('energy_ratio', 'upper', 100.0, 'energy over the mean energy of 50 relaxed conformers'),
    'protein-ligand_maximum_distance': ('smallest_distance_protein', 'upper', 5.0,
                                        'smallest ligand-protein distance, A'),
    'minimum_distance_to_protein': ('most_extreme_relative_distance_protein', 'lower', 0.75,
                                    'smallest ligand-protein distance over the sum of van der Waals radii'),
    'minimum_distance_to_organic_cofactors': ('most_extreme_relative_distance_organic_cofactors', 'lower', 0.75,
                                              'as above, organic cofactors'),
    'minimum_distance_to_inorganic_cofactors': ('most_extreme_relative_distance_inorganic_cofactors', 'lower', 0.75,
                                                'as above, inorganic cofactors (covalent radii)'),
    'minimum_distance_to_waters': ('most_extreme_relative_distance_waters', 'lower', 0.75, 'as above, waters'),
    'volume_overlap_with_protein': ('volume_overlap_protein', 'upper', 0.075, 'share of ligand volume in the protein'),
    'volume_overlap_with_organic_cofactors': ('volume_overlap_organic_cofactors', 'upper', 0.075, 'as above'),
    'volume_overlap_with_inorganic_cofactors': ('volume_overlap_inorganic_cofactors', 'upper', 0.075, 'as above'),
    'volume_overlap_with_waters': ('volume_overlap_waters', 'upper', 0.075, 'as above'),
}
BINARY = ['mol_pred_loaded', 'mol_cond_loaded', 'sanitization', 'inchi_convertible', 'all_atoms_connected',
          'no_radicals']


# ---------------------------------------------------------------- values and excess

def excess(value, kind, threshold):
  """Signed excess over the tolerance as a fraction of the threshold; > 0 fails."""
  if value is None or value != value or value is pd.NA:
    return np.nan
  v = float(value)
  return (v - threshold) / threshold if kind == 'upper' else (threshold - v) / threshold


def check_value(row, check):
  col, _, _, _ = CHECKS[check]
  try:
    v = col(row) if callable(col) else row.get(col)
  except (KeyError, TypeError):
    return np.nan
  return np.nan if v is None or v is pd.NA else float(v)


def rmsd_excess(rmsd):
  return np.nan if rmsd is None or rmsd != rmsd else (float(rmsd) - SUCCESS_RMSD) / SUCCESS_RMSD


# ---------------------------------------------------------------- rerun of M1

def split_failed(x):
  return sorted(s for s in str(x).split(';') if s) if isinstance(x, str) else []


def bool_or_none(x):
  if x is None or (isinstance(x, float) and np.isnan(x)):
    return None
  return str(x) == 'True'


def full_report(target, path):
  """The steps of score_ligand.score (reference, pocket superposition, the predicted protein written in the same
  frame, configuration PB_CONFIG) with PoseBusters' full report, and the standard table beside it."""
  import score_ligand as sl
  from posebusters import PoseBusters
  ref = sl.reference(target)
  pred = sl.Prediction(path, ref.ccd)
  r, t, _ = sl._pocket_superposition(ref.ca, pred.ca, ref.pocket)
  mol = ref.ligand_with(sl._transform(pred.ligand, r, t))
  with tempfile.TemporaryDirectory() as tmp:
    protein = pathlib.Path(tmp) / 'protein.pdb'
    sl._write_protein_pdb(pred, r, t, protein)
    std = PoseBusters(config=sl.PB_CONFIG).bust(mol_pred=mol, mol_cond=str(protein)).iloc[0]
    full = PoseBusters(config=sl.PB_CONFIG).bust(mol_pred=mol, mol_cond=str(protein), full_report=True).iloc[0]
  return std, full


def rescore_real(task):
  """(target, path) -> dict: the registered score() result and the full-report values. Never raises."""
  target, path = task
  import score_ligand as sl
  out = {'target': target, 'path': path}
  res = sl.score(target, path)
  out.update({'re_ligand_rmsd': res['ligand_rmsd'], 're_pb_valid': res['pb_valid'],
              're_pb_failed': ';'.join(res['pb_failed'] or []), 're_success': bool(res['success']),
              're_error': res['error']})
  if res['error'] is not None:
    return out
  try:
    std, full = full_report(target, path)
  except Exception as e:  # noqa: BLE001 - recorded; check 3b reports it
    out['full_error'] = f'{type(e).__name__}: {e}'[:300]
    return out
  std_checks = {str(k): bool(v) for k, v in std.items() if isinstance(v, (bool, np.bool_))}
  out['std_failed'] = ';'.join(sorted(k for k, v in std_checks.items() if not v))
  out['full_agrees'] = all(k in full.index and bool(full[k]) == v for k, v in std_checks.items())
  for c in CHECKS:
    _, kind, thr, _ = CHECKS[c]
    v = check_value(full, c)
    out[f'value:{c}'] = v
    out[f'excess:{c}'] = excess(v, kind, thr)
    out[f'pass:{c}'] = std_checks.get(c)
  return out


RESCORE = rescore_real      # tests replace it with a synthetic function


def rescore_all(tasks, jobs):
  if jobs > 1 and RESCORE is rescore_real:
    with multiprocessing.Pool(jobs) as pool:
      return pool.map(RESCORE, tasks, chunksize=2)
  return [RESCORE(t) for t in tasks]


def reproduction_checks(rescored, recorded):
  """Check 3 (exact reproduction of the saved M1), 3b (full report gives the same failed checks) and 4 (the sign of
  every excess agrees with the check's outcome). Returns (rows, failures)."""
  rows, bad = [], []
  for r in rescored:
    x = recorded[r['path']]
    rec_rmsd = x['m1_ligand_rmsd']
    rec_err = x['m1_error'] if isinstance(x['m1_error'], str) else None
    same = {
        'ligand_rmsd': (rec_rmsd != rec_rmsd and r['re_ligand_rmsd'] is None)
        or (r['re_ligand_rmsd'] is not None and rec_rmsd == rec_rmsd
            and round(float(r['re_ligand_rmsd']), 4) == round(float(rec_rmsd), 4)),
        'pb_valid': bool_or_none(x['m1_pb_valid']) == (None if r['re_pb_valid'] is None else bool(r['re_pb_valid'])),
        'pb_failed': split_failed(x['m1_pb_failed']) == split_failed(r['re_pb_failed']),
        'success': bool(x['m1_success'] == True) == bool(r['re_success']),  # noqa: E712
        'error': rec_err == r['re_error'],
    }
    row = {'path': r['path'], 'target': r['target'], **{f'3:{k}': v for k, v in same.items()}}
    if r['re_error'] is None:
      row['3b:full_report'] = bool(r.get('full_agrees')) and split_failed(r.get('std_failed')) == \
          split_failed(r['re_pb_failed'])
      signs = []
      for c in CHECKS:
        e, p = r.get(f'excess:{c}'), r.get(f'pass:{c}')
        if p is None or e is None or e != e:
          continue
        signs.append((e > 0) == (not p))
      row['4:sign'] = all(signs)
    rows.append(row)
    for k, v in row.items():
      if k[:2] in ('3:', '3b', '4:') and v is False:
        bad.append(f'{k} {r["target"]} {r["path"]}')
  return pd.DataFrame(rows), bad


# ---------------------------------------------------------------- flips

def flip_units(sources, items=None):
  """One row per success flip, with both top-1 rows (checks 1 and 2 through winners.side_row)."""
  rows = []
  for source, names in (items or CONTRASTS).items():
    samples, pairs = sources[source]
    for contrast in names:
      sp = pairs[pairs.contrast == contrast]
      if sp.empty:
        raise SystemExit(f'{source}: {contrast} has no rows in pairs_seed.csv')
      (ca, ra), (cb, rb) = rs.sides(contrast)
      units = contrasts.seed_units(samples[samples.condition.isin([ca, cb])])
      for r in sp.itertuples():
        where = f'{source} {contrast} {r.target} seed {r.seed_a}'
        xa = w.side_row(units, ca, ra, r.target, r.seed_a, r.top1_a, r.success_a, where)
        xb = w.side_row(units, cb, rb, r.target, r.seed_b, r.top1_b, r.success_b, where)
        if not r.flip_success:
          continue
        fail, ok = ('b', 'a') if bool(r.success_a) else ('a', 'b')
        x = {'a': xa, 'b': xb}
        rows.append({'source': source, 'contrast': contrast, 'target': r.target, 'seed': int(r.seed_a),
                     'top1_a': int(r.top1_a), 'top1_b': int(r.top1_b), 'top1_changed': bool(r.flip_top1),
                     'failing_side': fail, 'path_fail': x[fail]['model_path'], 'path_ok': x[ok]['model_path'],
                     'rmsd_fail': float(x[fail]['m1_ligand_rmsd']) if pd.notna(x[fail]['m1_ligand_rmsd']) else np.nan,
                     'rmsd_ok': float(x[ok]['m1_ligand_rmsd']) if pd.notna(x[ok]['m1_ligand_rmsd']) else np.nan,
                     'pb_valid_fail': bool_or_none(x[fail]['m1_pb_valid']),
                     'pb_failed_fail': ';'.join(split_failed(x[fail]['m1_pb_failed'])),
                     'error_fail': x[fail]['m1_error'] if isinstance(x[fail]['m1_error'], str) else '',
                     '_rec_fail': x[fail], '_rec_ok': x[ok]})
  return pd.DataFrame(rows)


def criterion(r):
  if r.error_fail or r.rmsd_fail != r.rmsd_fail or r.pb_valid_fail is None:
    return 'not scored'
  rm, pb = r.rmsd_fail >= SUCCESS_RMSD, not r.pb_valid_fail
  return 'RMSD and PoseBusters' if rm and pb else 'RMSD' if rm else 'PoseBusters' if pb else 'none (inconsistent)'


def add_margins(df, rescored, pose):
  by_path = {r['path']: r for r in rescored}
  out = []
  for r in df.itertuples():
    rf, ro = by_path.get(r.path_fail, {}), by_path.get(r.path_ok, {})
    crit = criterion(r)
    failing = (['RMSD'] if crit in ('RMSD', 'RMSD and PoseBusters') else []) + split_failed(r.pb_failed_fail)
    ex_fail, ex_ok, missing = {}, {}, False
    for c in failing:
      if c == 'RMSD':
        ex_fail[c], ex_ok[c] = rmsd_excess(r.rmsd_fail), rmsd_excess(r.rmsd_ok)
      elif c in CHECKS:
        ex_fail[c], ex_ok[c] = rf.get(f'excess:{c}', np.nan), ro.get(f'excess:{c}', np.nan)
      else:
        ex_fail[c], ex_ok[c] = np.nan, np.nan            # a binary check: no distance
      missing |= ex_fail[c] != ex_fail[c]
    p, perr = pose.get((r.target, r.path_fail, r.path_ok), (None, 'not computed'))
    worst = max(ex_fail.values()) if ex_fail and not missing else np.nan
    if crit in ('not scored', 'none (inconsistent)') or p is None or missing:
      cls = 'not classified'
    elif p > w.DISTINCT_A:
      cls = 'pose change'
    elif worst <= NEAR:
      cls = 'near-threshold crossing'
    else:
      cls = 'other'
    out.append({'criterion': crit, 'failing_criteria': ';'.join(failing),
                'excess_fail': ';'.join(f'{k}={v:.4f}' for k, v in ex_fail.items()),
                'excess_ok': ';'.join(f'{k}={v:.4f}' for k, v in ex_ok.items()),
                'worst_excess_fail': worst, 'pose_rmsd': p if p is not None else np.nan,
                'pose_error': perr, 'class': cls})
  return pd.concat([df.reset_index(drop=True), pd.DataFrame(out)], axis=1)


# ---------------------------------------------------------------- statistics

def summarize(flips, reps, seed):
  rows = []
  sets = [(c, [c]) for names in CONTRASTS.values() for c in names] + [(f'group: {g}', cs) for g, cs in GROUPS.items()]
  for name, cs in sets:
    u = flips[flips.contrast.isin(cs)]
    for stratum, g in (('all flips', u), ('top-1 changed', u[u.top1_changed])):
      out = {'contrast': name, 'stratum': stratum, 'flips': len(g), 'n_targets': g.target.nunique()}
      for k in ('RMSD', 'PoseBusters', 'RMSD and PoseBusters', 'not scored'):
        out[f'criterion: {k}'] = int((g.criterion == k).sum())
      for c in sorted({c for s in g.failing_criteria for c in s.split(';') if c}):
        out[f'fails: {c}'] = int(g.failing_criteria.str.split(';').apply(lambda x, c=c: c in x).sum())
      for k in ('pose change', 'near-threshold crossing', 'other', 'not classified'):
        out[f'class: {k}'] = int((g['class'] == k).sum())
      cl = g[g['class'] != 'not classified']
      targets = sorted(cl.target.unique())
      for k in ('near-threshold crossing', 'pose change'):
        grp = [(cl[cl.target == t]['class'] == k).astype(float).values for t in targets]
        lo, hi = contrasts.boot_ci(grp, contrasts.mean, np.random.default_rng(seed), reps)
        out.update({f'share {k}': contrasts.mean((cl['class'] == k).astype(float).values),
                    f'share {k} ci_low': lo, f'share {k} ci_high': hi})
      v = g.dropna(subset=['worst_excess_fail'])
      grp = [v[v.target == t].worst_excess_fail.values for t in sorted(v.target.unique())]
      lo, hi = contrasts.boot_ci(grp, contrasts.med, np.random.default_rng(seed), reps)
      out.update({'median worst excess': contrasts.med(v.worst_excess_fail.values), 'median worst excess ci_low': lo,
                  'median worst excess ci_high': hi, 'with worst excess': len(v)})
      rows.append(out)
  return pd.DataFrame(rows)


def report(flips, s, chk, args):
  def pct(r, k):
    e, lo, hi = r[f'share {k}'], r[f'share {k} ci_low'], r[f'share {k} ci_high']
    return 'n/a' if e != e else f'{100 * e:.0f}% [{100 * lo:.0f}, {100 * hi:.0f}]'
  L = ['# Which criterion changes in a success flip (label pb_flips)', '',
       'Post hoc, descriptive (drafts/section10_pb_flips_spec_20261116.md). Excess: (value - threshold) / threshold '
       f'beyond the tolerance (> 0 fails); near = at most {NEAR:.0%}. Pose change: the two top-1 poses more than 2 A '
       f'apart. 95% percentile intervals over targets ({args.replicates} replicates, seed {args.seed}).', '',
       '| contrast | stratum | flips | RMSD | PoseBusters | both | near-threshold [CI] | pose change [CI] | other | '
       'not classified | median worst excess [CI] |', '|---|---|---|---|---|---|---|---|---|---|---|']
  for r in s.to_dict('records'):
    med = r['median worst excess']
    L.append(f'| {r["contrast"]} | {r["stratum"]} | {r["flips"]} | {r["criterion: RMSD"]} | '
             f'{r["criterion: PoseBusters"]} | {r["criterion: RMSD and PoseBusters"]} | '
             f'{r["class: near-threshold crossing"]}, {pct(r, "near-threshold crossing")} | '
             f'{r["class: pose change"]}, {pct(r, "pose change")} | {r["class: other"]} | {r["class: not classified"]} | '
             + ('n/a' if med != med else f'{med:.3f} [{r["median worst excess ci_low"]:.3f}, '
                f'{r["median worst excess ci_high"]:.3f}]') + ' |')
  L += ['', '## Flips', '', '| contrast | target | seed | top-1 changed | failing side | criteria failing | '
        'excess (failing side) | excess (other side) | pose RMSD | class |', '|---|---|---|---|---|---|---|---|---|---|']
  for r in flips.to_dict('records'):
    L.append(f'| {r["contrast"]} | {r["target"]} | {r["seed"]} | {r["top1_changed"]} | {r["failing_side"]} | '
             f'{r["failing_criteria"]} | {r["excess_fail"]} | {r["excess_ok"]} | {w.fmt(r["pose_rmsd"])} | {r["class"]} |')
  L += ['', '## Checks', ''] + [f'- {c}' for c in chk]
  return '\n'.join(L) + '\n'


def run(sources, out, args, inputs=(), root=None, items=None):
  out = pathlib.Path(out)
  out.mkdir(parents=True, exist_ok=True)
  flips = flip_units(sources, items)
  chk = [f'1, 2: pass ({len(flips)} success flips; every registered top-1 found, its argmax, its success)']
  recorded = {p: x for p, x in zip(list(flips.path_fail) + list(flips.path_ok),
                                   list(flips._rec_fail) + list(flips._rec_ok))}
  tasks = sorted({(r.target, r.path_fail) for r in flips.itertuples()} | {(r.target, r.path_ok) for r in flips.itertuples()})
  rescored = rescore_all(tasks, getattr(args, 'jobs', 1))
  rep, bad = reproduction_checks(rescored, recorded)
  rep.to_csv(out / 'reproduction.csv', index=False)
  if bad:
    (out / 'checks.csv').write_text('check\n' + '\n'.join(bad) + '\n')
    raise SystemExit('check 3, 3b or 4 failed (the rerun does not reproduce the saved M1); see reproduction.csv:\n'
                     + '\n'.join(bad[:20]))
  chk.append(f'3, 3b, 4: pass ({len(rescored)} files rerun; saved M1 reproduced exactly; full report agrees; '
             'every excess has the sign of its check)')
  pose_tasks = sorted({(r.target, r.path_fail, r.path_ok) for r in flips.itertuples()})
  pose = {t: w.POSE_RMSD(t) for t in pose_tasks}
  flips = add_margins(flips.drop(columns=['_rec_fail', '_rec_ok']), rescored, pose)
  s = summarize(flips, args.replicates, args.seed)
  flips.to_csv(out / 'flips.csv', index=False)
  pd.DataFrame(rescored).to_csv(out / 'rescored.csv', index=False)
  s.to_csv(out / 'summary.csv', index=False)
  pd.DataFrame({'check': chk}).to_csv(out / 'checks.csv', index=False)
  if root is not None:
    (out / 'inputs.tsv').write_text(''.join(f'{p}\t{w.sha256(pathlib.Path(root) / p)}\n' for p in inputs))
  (out / 'report.md').write_text(report(flips, s, chk, args))
  return flips, s, chk


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
  version = importlib.metadata.version('posebusters')
  if version != PB_VERSION:
    raise SystemExit(f'check 0: posebusters {version}, the study used {PB_VERSION}')
  sources, inputs = w.load_sources(args.main)
  run(sources, out, args, inputs, args.main)
  print(f'wrote {out}')


if __name__ == '__main__':
  main()
