"""Follow-up 19, recompilation of the float32 A100 and H100 arms on the other 50 targets (label rcf;
notes/analysis_plan.md section 10, row "Follow-up experiment 19"). An extension of analysis/rcx.py (follow-up 15,
part A4) to the 60 targets.

    analysis/.venv/bin/python analysis/rcf.py --samples <collect.py samples.csv of the rcf sessions> --out DIR

Conditions. Follow-up 19 (label rcf, repetition r2), from each process's configuration (collect.py unchanged):
a100_xla_fu14_f32 -> Af, h100_xla_fu14_f32 -> Hf, the 50 non-pilot targets. r2 of the 10 pilot targets: follow-up 15
(--rcx, its samples.csv, Af and Hf only). r1: Af and Hf of follow-up 14 (--fu14, its samples.csv). A (condition,
target, seed, repetition) in two sources is an error (followup34.union).
Checks (item c), before anything is compared; the analysis stops if 1 or 2 fails:
  1 every process of the rcf sessions exits 0 or its failure is recorded;
  2 one machine type per condition, the r1's (a2-ultragpu-1g, a3-highgpu-1g);
  3 completeness: Af r2 and Hf r2 of follow-up 19 on the 50 non-pilot targets, seeds 1 to 5, and the 10 pilot targets
    of follow-up 15 (a missing unit leaves the contrast, section 8);
  4 settings and stack: global_config_bfloat16 none (af3_entry_stats.json), JAX_DEFAULT_MATMUL_PRECISION=highest
    (xla_env) and the expected device (NVIDIA A100-SXM4-80GB, NVIDIA H100 80GB HBM3) in every process (4a,
    rcx.check_settings); every rcf session's NVIDIA driver, kernel, Python version, packages, pip_freeze.txt,
    configuration entry, environment and image family against those of the Af r1 and Hf r1 sessions (4b,
    xvm.check_stack). Reported, not a stop (the differences of image and Python that follow-up 15's row lists are
    expected);
  5 (reproduction, reported) RC(Af) and RC(Hf) recomputed here on the 10 pilot targets equal follow-up 15's seed pairs
    (rcx_20261115/pairs_seed.csv) unit by unit.
Statistics (item d): section 5's target-clustered bootstrap (--replicates 10000, --seed 2026), paired on (target,
seed), per-seed top-1, seeds 1 to 5. RC(Af) and RC(Hf) on the 60 targets (r1 of follow-up 14 against r2 of follow-ups
15 and 19), and on the 50 new targets alone ("RC(Af) new 50", "RC(Hf) new 50"): the three flip statistics, the M4 and
M5 medians, the bit-identical fraction of samples (contrasts.summarize); the bit-identical (target, seed): the
light-fetch files on the 50 (xvm.light_identity), every output file on the 10 pilot targets (fu13.identity_rows, the
strict check of follow-up 15), and over the 60 (rcx.identity_reading). D_F(Hf,Af) = [HW(Hf,Af) - RC(Hf)] - [HW(H,A) -
RC(H)] with RC(Hf) measured on the 60 targets, every flip statistic (fu13.d_f; the same draws as the registered D_F
when the four seed-pair sets cover the same targets, which the report states); the registered D_F (RC(Hf) set to 0) is
recomputed first and must equal follow-up 14's d_f.csv: if it does not, every output is written, the report says so,
and the analysis exits with status 1.
Reading rules (item e): none beyond follow-up 15's. The rates with their intervals fill Table IV on 60 targets; the
registered D_F reading stands whatever this shows; the report states whether D_F with RC(Hf) measured on the 60
targets reads as the registered one under rule e1 of follow-up 14 ("reduced" if its 95% interval lies below 0).
Outputs in --out: checks.csv, checks_detail.csv, stack.csv, completeness.csv, contrasts.csv, identity.csv,
d_f.csv, pairs_seed.csv, pairs_sample.csv, counts.csv, report.md.
"""
import argparse
import pathlib
import sys

import pandas as pd

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / 'analysis'))
import contrasts  # noqa: E402
import followup34 as f34  # noqa: E402
import fu13  # noqa: E402
import fu14  # noqa: E402
import rcx  # noqa: E402
import xvm  # noqa: E402

SEEDS = [1, 2, 3, 4, 5]
CONFIG = {'a100_xla_fu14_f32': 'Af', 'h100_xla_fu14_f32': 'Hf'}
CONDS = ('Af', 'Hf')


# ---------------------------------------------------------------- loading

def load_rcf(path):
  d = pd.read_csv(path, low_memory=False)
  d = d[(d.label == 'rcf') & (d.num_recycles == 10) & (d.num_diffusion_samples == 5)].copy()
  d['condition'] = d.config.map(CONFIG)
  if d.condition.isna().any():
    raise SystemExit(f'sessions of label rcf with configurations not of follow-up 19: {sorted(set(d[d.condition.isna()].config))}')
  if not (d.repetition == 'r2').all():
    raise SystemExit(f'rcf processes with repetitions other than r2: {sorted(set(d.repetition) - {"r2"})}')
  return d


def load_pilot_r2(args):
  """Af and Hf r2 of follow-up 15 (the 10 pilot targets)."""
  r = rcx.load_rcx(pathlib.Path(args.rcx) / 'samples.csv')
  return r[r.condition.isin(CONDS) & (r.repetition == 'r2')]


def load_r1(args):
  f = fu14.load_f32(pathlib.Path(args.fu14) / 'samples.csv')
  return f[f.condition.isin(CONDS) & (f.repetition == 'r1')]


# ---------------------------------------------------------------- checks

def completeness(ok_rcf, ok_pilot, pilot, new):
  rows = []
  for source, ok, targets in (('follow-up 19', ok_rcf, new), ('follow-up 15', ok_pilot, pilot)):
    for c in CONDS:
      for s in SEEDS:
        g = ok[(ok.condition == c) & (ok.repetition == 'r2') & (ok.seed == s) & ok.target.isin(targets)]
        have = int((g.groupby('target').size() == 5).sum())
        rows.append({'source': source, 'condition': c, 'repetition': 'r2', 'seed': s, 'planned_targets': len(targets),
                     'targets_with_5_samples': have,
                     'status': 'complete' if have == len(targets) else ('not run' if have == 0 else 'incomplete')})
  return pd.DataFrame(rows)


def check_targets(d, pilot):
  bad = sorted(set(d.target) & set(pilot))
  return [{'check': '3b follow-up 19 runs only non-pilot targets', 'condition': 'Af, Hf',
           'value': f'{d.target.nunique()} targets', 'detail': 'pass' if not bad else 'reported, left out (their r2 is follow-up 15\'s): pilot targets ' + ', '.join(bad)}]


def check_pilot_reproduced(sp, rcx_sp, pilot):
  rows = []
  for c in CONDS:
    name = f'RC({c})'
    reg = rcx_sp[rcx_sp.contrast == name].set_index(['target', 'seed_a'])[contrasts.FLIP_STATS].astype(bool).sort_index()
    new = sp[(sp.contrast == name) & sp.target.isin(pilot)].set_index(['target', 'seed_a'])[contrasts.FLIP_STATS]
    new = new.astype(bool).sort_index()
    same = len(reg) == len(new) and reg.index.equals(new.index) and (reg.values == new.values).all()
    rows.append({'check': '5 RC on the pilot targets equals follow-up 15\'s', 'condition': c,
                 'value': f'{len(new)} recomputed, {len(reg)} registered (seed) units',
                 'detail': 'pass' if same else 'reported: differs from follow-up 15\'s seed pairs'})
  return rows


# ---------------------------------------------------------------- identity

def identity(okall, pilot, new):
  """Bit-identity of every (target, seed) of r2 against r1: light-fetch files on the new targets, every output file on
  the pilot targets; per condition the three summaries (new, pilot, all 60) and the detail rows."""
  rows, det_all = [], []
  for c in CONDS:
    s_new, det_new = xvm.light_identity(okall[okall.target.isin(new)], (c, 'r1'), (c, 'r2'),
                                        f'{c} r2 against r1, 50 new targets (light-fetch files)')
    okp = okall[okall.target.isin(pilot)]
    s_pil, det_pil = fu13.identity_rows(okp, okp, c, c, 'r2', 'r1', SEEDS,
                                        f'{c} r2 against r1, 10 pilot targets (every output file)')
    s_pil['detail'] = rcx.identity_reading(det_pil)
    det = det_new + det_pil
    n_id = sum(x['status'] == 'identical' for x in det)
    rows += [{**s_new, 'condition': c, 'targets': 'new 50'},
             {**s_pil, 'condition': c, 'targets': 'pilot 10'},
             {'check': f'{c} r2 against r1, 60 targets', 'condition': c, 'targets': 'all 60',
              'value': f'{n_id}/{len(det)} identical', 'detail': rcx.identity_reading(det)}]
    det_all += det
  return rows, det_all


# ---------------------------------------------------------------- D_F

def d_f_table(sp, f14_sp, h_sp, args):
  hwf = f14_sp[f14_sp.contrast == 'HW(Hf,Af)']
  hw, rc = h_sp[h_sp.contrast == 'HW(H,A)'], h_sp[h_sp.contrast == 'RC(H)']
  rcf = sp[sp.contrast == 'RC(Hf)']
  rows = []
  reg_targets = sorted(set(hwf.target) & set(hw.target) & set(rc.target))
  for stat in contrasts.FLIP_STATS:
    base = {'pair': 'HW(Hf,Af)', 'reference': 'HW(H,A) minus RC(H)', 'statistic': stat}
    reg = fu13.d_f(hwf, fu14.zero_like(hwf), hw, rc, stat, args.seed, args.replicates)
    rows.append({**base, 'version': 'registered: 60 targets, RC(Hf) set to 0', **reg, 'same_draws': True,
                 'reading': fu13.f_reading(reg['ci_high'])})
    if len(rcf):
      r = fu13.d_f(hwf, rcf, hw, rc, stat, args.seed, args.replicates)
      same = sorted(set(reg_targets) & set(rcf.target)) == reg_targets
      rows.append({**base, 'version': 'RC(Hf) measured on the 60 targets', **r, 'same_draws': same,
                   'reading': fu13.f_reading(r['ci_high'])})
  return pd.DataFrame(rows)


def d_f_sentence(dft):
  top = dft[dft.statistic == 'flip_top1'].set_index('version')
  reg, meas = 'registered: 60 targets, RC(Hf) set to 0', 'RC(Hf) measured on the 60 targets'
  if meas not in top.index:
    return 'RC(Hf) not measured on the 60 targets: D_F with it is not computed.'
  same = top.loc[meas, 'reading'] == top.loc[reg, 'reading']
  return (f'Top-1: D_F with RC(Hf) measured on the 60 targets reads "{top.loc[meas, "reading"]}", the registered D_F '
          f'"{top.loc[reg, "reading"]}": {"the same reading" if same else "a different reading"} under rule e1 of '
          f'follow-up 14; the registered reading stands either way. Same draws as the registered D_F: '
          f'{"yes" if top.loc[meas, "same_draws"] else "no (the target lists differ; see n_targets)"}.')


# ---------------------------------------------------------------- main

def main():
  ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
  ap.add_argument('--samples', required=True)
  ap.add_argument('--fu14', default=str(REPO / 'results' / 'analysis' / 'fu14_20261113b'))
  ap.add_argument('--rcx', default=str(REPO / 'results' / 'analysis' / 'rcx_20261115'))
  ap.add_argument('--h100', default=str(REPO / 'results' / 'analysis' / 'h100_20261104'))
  ap.add_argument('--targets', default=str(REPO / 'targets' / 'posebusters_subset.csv'))
  ap.add_argument('--out', required=True)
  ap.add_argument('--replicates', type=int, default=10000)
  ap.add_argument('--seed', type=int, default=2026)
  args = ap.parse_args()
  out = pathlib.Path(args.out)
  out.mkdir(parents=True, exist_ok=True)
  tt = pd.read_csv(args.targets, dtype=str)
  pilot, all_targets = sorted(tt[tt.pilot == 'yes'].pdb_id), sorted(tt.pdb_id)
  new = sorted(set(all_targets) - set(pilot))

  d = load_rcf(args.samples)
  chk = fu13.check_exit(d) + fu13.check_machine(d) + rcx.check_machine_r1(d)
  pd.DataFrame(chk).to_csv(out / 'checks.csv', index=False)
  if any(r['detail'] != 'pass' for r in chk):
    raise SystemExit('check 1 or 2 failed (checks.csv); nothing is compared')
  pil, r1 = load_pilot_r2(args), load_r1(args)
  chk += check_targets(d, pilot)
  okall = f34.union(d[d.target.isin(new)], pil, r1)
  comp = completeness(f34.sample_rows(d), f34.sample_rows(pil), pilot, new)
  s4, det4 = rcx.check_settings(d)
  s4b, stack = xvm.check_stack({c: [xvm.session_name(s) for s in d[d.condition == c].session] for c in CONDS},
                               {c: [xvm.session_name(s) for s in r1[r1.condition == c].session] for c in CONDS},
                               '4b stack of every rcf session against the r1 sessions')
  chk += s4 + s4b
  ident, det5 = identity(okall, pilot, new)

  specs = [(f'RC({c})', 'RC', (c, 'r1'), (c, 'r2'), False) for c in CONDS]
  sp60, sm60, cons60, counts60 = f34.build(okall, specs)
  sp50, sm50, cons50, counts50 = f34.build(okall[okall.target.isin(new)],
                                           [(f'RC({c}) new 50', 'RC', (c, 'r1'), (c, 'r2'), False) for c in CONDS])
  sp = pd.concat([sp60, sp50], ignore_index=True)
  sm = pd.concat([sm60, sm50], ignore_index=True)
  counts = pd.concat([counts60, counts50], ignore_index=True)
  cons = {**cons60, **cons50}
  chk += check_pilot_reproduced(sp60, pd.read_csv(pathlib.Path(args.rcx) / 'pairs_seed.csv'), pilot)
  pd.DataFrame(chk).to_csv(out / 'checks.csv', index=False)
  pd.DataFrame(det4 + det5).to_csv(out / 'checks_detail.csv', index=False)

  con = contrasts.summarize(sp, sm, cons, args) if len(sp) else pd.DataFrame()
  dft = d_f_table(sp60, pd.read_csv(pathlib.Path(args.fu14) / 'pairs_seed.csv'),
                  pd.read_csv(pathlib.Path(args.h100) / 'pairs_seed.csv'), args)
  same = rcx.registered_d_f_equal(dft, args.fu14)
  outputs = {'completeness': comp, 'contrasts': con, 'identity': pd.DataFrame(ident), 'd_f': dft, 'stack': stack,
             'pairs_seed': sp, 'pairs_sample': sm, 'counts': counts}
  for name, df in outputs.items():
    df.to_csv(out / f'{name}.csv', index=False)
  (out / 'report.md').write_text(report(chk, comp, con, ident, dft, same))
  print((out / 'report.md').read_text())
  if same is False:
    print('!! the registered D_F was not reproduced (follow-up 14\'s d_f.csv): see report.md', file=sys.stderr)
    return 1
  return 0


def report(chk, comp, con, ident, dft, same):
  pct, num = f34.pct, f34.num
  L = ['# Follow-up 19: recompilation of Af and Hf on the 60 targets', '',
       'notes/analysis_plan.md section 10, row "Follow-up experiment 19" (label rcf); reported whatever it shows.', '',
       '## Checks', '', '| check | condition | value | result |', '|---|---|---|---|']
  L += [f'| {r["check"]} | {r["condition"]} | {r["value"]} | {r["detail"]} |' for r in chk]
  inc = comp[comp.status != 'complete']
  L += ['', f'Completeness: {len(comp) - len(inc)} of {len(comp)} planned (source, condition, seed) cells complete'
        + ('' if inc.empty else '; not complete: ' + ', '.join(
            f'{r.source} {r.condition} seed {r.seed} ({r.targets_with_5_samples}/{r.planned_targets}, {r.status})'
            for r in inc.itertuples())) + '.']
  L += ['', '## Bit-identity', '']
  L += [f'- {r["condition"]}, {r["targets"]}: {r["value"]}; {r["detail"]}' for r in ident]
  L += ['', '## Recompilation (Table IV on 60 targets)', '', '| contrast | statistic | estimate | 95% CI | n | targets |',
        '|---|---|---|---|---|---|']
  if len(con):
    keep = con[con.statistic.isin(contrasts.FLIP_STATS + ['m4_ligand_rmsd_median', 'm4_ca_rmsd_global_median',
                                                          'm5_ranking_score_median', 'bit_identical_fraction'])]
    for r in keep.itertuples():
      fmt = pct if r.statistic in contrasts.FLIP_STATS + ['bit_identical_fraction'] else num
      L.append(f'| {r.contrast} | {r.statistic} | {fmt(r.estimate)} | {fmt(r.ci_low)} to {fmt(r.ci_high)} | {r.n} | '
               f'{r.n_targets} |')
  L += ['', '## D_F(Hf,Af) with RC(Hf) measured on 60 targets (the registered reading stands)', '',
        '| version | statistic | D_F | 95% CI | targets | RC(Hf) term | same draws | reading under rule e1 |',
        '|---|---|---|---|---|---|---|---|']
  L += [f'| {r.version} | {r.statistic} | {pct(r.estimate)} | {pct(r.ci_low)} to {pct(r.ci_high)} | {r.n_targets} | '
        f'{pct(r.rc_f)} | {"yes" if r.same_draws else "no"} | {r.reading} |' for r in dft.itertuples()]
  L += ['', ('The registered D_F recomputed here ' + {True: 'equals', False: 'DIFFERS from (the analysis exits 1)',
                                                     None: 'was not compared with (no file)'}[same]
             + " follow-up 14's d_f.csv."), '', d_f_sentence(dft)]
  return '\n'.join(L) + '\n'


if __name__ == '__main__':
  sys.exit(main())
