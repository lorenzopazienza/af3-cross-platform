"""Returned poses compared directly, further contrasts (section 10, row of
drafts/section10_winners_ext_spec_20261116.md, label winners_ext): a post hoc analysis of frozen outputs, an extension
of analysis/winners.py (label winners) to contrasts it does not cover, descriptive whatever it shows. No run and no
cloud resource.

    analysis/.venv/bin/python analysis/winners_ext.py --main . --out results/analysis/winners_ext_<DATE> \\
        [--replicates 10000] [--seed 2026] [--jobs 8]

Question: as in winners, when the platform changes which sample AlphaFold 3 returns, how different are the two returned
poses? Here for the contrasts winners does not cover:
  main:  HW(G,C) of the main study (L4 against CPU, the 6 targets and seeds 1 and 2 the CPU ran), and DU (default
         attention on the L4: G against GT, as registered, side A G r1 and side B GT r1; 60 targets, seeds 1 to 5);
  cpu60: HW(G,C) of follow-up 4 (60 targets, seeds 1 and 2; C of the 6 main-study targets from the main study, C of
         the other 54 from follow-up 4, as analysis/returned_structures.py and the registered cpu60 analysis pair them);
  rcx:   RC(GT), recompilation of the L4 Triton arm (GT r1 of the main study against GT r2 of follow-up 15; 60 targets,
         seeds 1 to 5), the second recompilation reference of DU beside RC(G) of winners.
Inputs, relative to --main (the checkout that holds results/ and the AlphaFold 3 output folders under results/af3/):
  results/analysis/main_20261014/primary_v2/samples.csv and contrasts/pairs_seed.csv (HW(G,C), DU)
  results/analysis/cpu60_20261018/samples.csv and pairs_seed.csv (HW(G,C), follow-up 4)
  results/analysis/rcx_20261115/samples.csv and pairs_seed.csv (RC(GT), follow-up 15)
Everything else is winners.py unchanged: the same levels (seed: the registered top-1 of each side; target: AlphaFold 3's
main output over the seeds both sides ran, here seeds 1 and 2 for HW(G,C), 1 to 5 for DU and RC(GT)), the same pose
function (winners.pose_rmsd_real: score_ligand.pair_divergence through contrasts.divergence, 0 when the coordinates are
identical as written), the same counts (distinct poses > 2 A; validity changes; binding-mode changes, one returned pose
within 2 A of the experimental pose and the other beyond 4 A), the same statistics (median and 90th percentile of the
pose RMSD and the shares of the counts, 95% percentile intervals from the bootstrap over targets, contrasts.boot_ci,
--replicates, a fresh generator from --seed for every statistic; maximum with its target), the same strata (all units;
units whose returned sample changed) and the same checks (1 and 2 stop the analysis; 3 reports the targets whose main
output changes; no earlier count exists for these contrasts).
Outputs in --out: units_seed.csv, units_target.csv, summary.csv, checks.csv, errors.csv, inputs.tsv, report.md.
"""
import argparse
import pathlib
import sys

import pandas as pd

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / 'analysis'))
sys.path.insert(0, str(REPO / 'harness'))
import contrasts  # noqa: E402
import followup34 as f34  # noqa: E402
import returned_structures as rs  # noqa: E402
import winners as w  # noqa: E402

AN = 'results/analysis'
MAIN = f'{AN}/main_20261014/primary_v2'
CPU60 = f'{AN}/cpu60_20261018'
RCX = f'{AN}/rcx_20261115'
CONTRASTS = {'main': ['HW(G,C)', 'DU'], 'cpu60': ['HW(G,C)'], 'rcx': ['RC(GT)']}
SOURCE_LABEL = {'main': 'main study', 'cpu60': 'follow-up 4', 'rcx': 'follow-up 15'}


def sides(name):
  """(condition, repetition) of side A and side B; DU is G against GT as contrasts.build_pairs registers it."""
  if name == 'DU':
    return ('G', 'r1'), ('GT', 'r1')
  return rs.sides(name)


def load_sources(root):
  root = pathlib.Path(root)
  main = f34.load_samples(root / MAIN / 'samples.csv', ['main'])
  c60 = f34.load_samples(root / CPU60 / 'samples.csv', ['cpu60'])
  rcx = f34.load_samples(root / RCX / 'samples.csv', ['rcx'])
  out = {'main': (f34.union(main[main.condition.isin(['G', 'GT', 'C'])]),
                  pd.read_csv(root / MAIN / 'contrasts' / 'pairs_seed.csv')),
         'cpu60': (f34.union(main[main.condition == 'G'],
                             pd.concat([main[main.condition == 'C'], c60[c60.condition == 'C']])),
                   pd.read_csv(root / CPU60 / 'pairs_seed.csv')),
         'rcx': (f34.union(main[(main.condition == 'GT') & (main.repetition == 'r1')],
                           rcx[(rcx.condition == 'GT') & (rcx.repetition == 'r2')]),
                 pd.read_csv(root / RCX / 'pairs_seed.csv'))}
  inputs = [MAIN + '/samples.csv', MAIN + '/contrasts/pairs_seed.csv', CPU60 + '/samples.csv', CPU60 + '/pairs_seed.csv',
            RCX + '/samples.csv', RCX + '/pairs_seed.csv']
  return out, inputs


def seed_units(samples, pairs, contrast, source):
  """winners.seed_units with sides() above (checks 1 and 2 in winners.side_row)."""
  sp = pairs[pairs.contrast == contrast]
  if sp.empty:
    raise SystemExit(f'{source}: {contrast} has no rows in pairs_seed.csv')
  (ca, ra), (cb, rb) = sides(contrast)
  units = contrasts.seed_units(samples[samples.condition.isin([ca, cb])])
  rows = []
  for r in sp.itertuples():
    where = f'{source} {contrast} {r.target} seed {r.seed_a}'
    xa = w.side_row(units, ca, ra, r.target, r.seed_a, r.top1_a, r.success_a, where)
    xb = w.side_row(units, cb, rb, r.target, r.seed_b, r.top1_b, r.success_b, where)
    rows.append(w.unit_row(source, contrast, 'seed', r.target, int(r.seed_a), xa, xb, int(r.top1_a), int(r.top1_b)))
  return rows


def target_units(samples, contrast, source):
  """winners.target_units with sides() above: the main output over the seeds both sides ran."""
  (ca, ra), (cb, rb) = sides(contrast)
  a = samples[(samples.condition == ca) & (samples.repetition == ra)]
  b = samples[(samples.condition == cb) & (samples.repetition == rb)]
  rows = []
  for t in sorted(set(a.target) & set(b.target)):
    ga, gb = a[a.target == t], b[b.target == t]
    seeds = sorted(set(ga.seed) & set(gb.seed))
    xa, xb = w.main_output(ga[ga.seed.isin(seeds)]), w.main_output(gb[gb.seed.isin(seeds)])
    rows.append(w.unit_row(source, contrast, 'target', t, len(seeds), xa, xb, (int(xa.seed), int(xa['sample'])),
                           (int(xb.seed), int(xb['sample']))))
  return rows


def checks_table(df):
  """Check 3 per source and contrast (winners.checks_table groups by contrast only, and HW(G,C) has two sources)."""
  rows = []
  t = df[df.level == 'target']
  for (source, contrast), g in t.groupby(['source', 'contrast'], sort=False):
    rows.append({'check': '3 main output changes', 'source': source, 'contrast': contrast,
                 'value': f'{int(g.changed.sum())} of {len(g)} targets', 'detail': 'no earlier count'})
  return pd.DataFrame(rows)


def report(s, chk, args):
  L = ['# Returned poses compared directly, further contrasts (label winners_ext)', '',
       'Post hoc, descriptive (drafts/section10_winners_ext_spec_20261116.md); the measures and statistics of '
       'analysis/winners.py. Sources: ' + ', '.join(f'{k} = {v}' for k, v in SOURCE_LABEL.items()) + '. DU: side A G '
       '(L4, default attention), side B GT (L4, Triton attention). pose RMSD in angstrom between the two returned '
       f'samples; 95% percentile intervals over targets ({args.replicates} replicates, seed {args.seed}).', '',
       '| source | contrast | level | stratum | units | with pose RMSD | median [CI] | p90 [CI] | max (target) | '
       'distinct > 2 A | validity change | binding-mode change |', '|---|---|---|---|---|---|---|---|---|---|---|---|']
  for r in s.to_dict('records'):
    def share(c):
      return f'{r[c + "_n"]} of {r[c + "_of"]}, {100 * r[c + "_share"]:.1f}% [{100 * r[c + "_ci_low"]:.1f}, ' \
             f'{100 * r[c + "_ci_high"]:.1f}]' if r[c + '_of'] else 'n/a'
    mx = r.get('pose_rmsd_max', float('nan'))
    L.append(f'| {r["source"]} | {r["contrast"]} | {r["level"]} | {r["stratum"]} | {r["units"]} | '
             f'{r["units_with_pose_rmsd"]} | {w.fmt(r["pose_rmsd_median"])} [{w.fmt(r["pose_rmsd_median_ci_low"])}, '
             f'{w.fmt(r["pose_rmsd_median_ci_high"])}] | {w.fmt(r["pose_rmsd_p90"])} [{w.fmt(r["pose_rmsd_p90_ci_low"])}, '
             f'{w.fmt(r["pose_rmsd_p90_ci_high"])}] | {w.fmt(mx)} ({r.get("pose_rmsd_max_target", "") if mx == mx else ""}) | '
             f'{share("distinct")} | {share("validity_change")} | {share("binding_mode_change")} |')
  L += ['', '## Check 3', ''] + [f'- {r.source} {r.contrast}: {r.value} ({r.detail})' for r in chk.itertuples()]
  return '\n'.join(L) + '\n'


def run(sources, out, args, inputs=(), root=None, items=None):
  out = pathlib.Path(out)
  out.mkdir(parents=True, exist_ok=True)
  rows = []
  for source, names in (items or CONTRASTS).items():
    samples, pairs = sources[source]
    for c in names:
      rows += seed_units(samples, pairs, c, source)
      rows += target_units(samples, c, source)
  df = w.add_pose_rmsd(pd.DataFrame(rows), getattr(args, 'jobs', 1))
  s = w.summarize(df, args.replicates, args.seed)
  chk = checks_table(df)
  df[df.level == 'seed'].drop(columns=['seeds'], errors='ignore').to_csv(out / 'units_seed.csv', index=False)
  df[df.level == 'target'].drop(columns=['seed'], errors='ignore').to_csv(out / 'units_target.csv', index=False)
  df[df.pose_error.notna()][['source', 'contrast', 'level', 'target', 'pose_error']].to_csv(out / 'errors.csv',
                                                                                              index=False)
  s.to_csv(out / 'summary.csv', index=False)
  chk.to_csv(out / 'checks.csv', index=False)
  if root is not None:
    (out / 'inputs.tsv').write_text(''.join(f'{p}\t{w.sha256(pathlib.Path(root) / p)}\n' for p in inputs))
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
