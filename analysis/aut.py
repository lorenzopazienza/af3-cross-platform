"""Follow-up 20, the L4 without autotuning (label aut; notes/analysis_plan.md section 10, row "Follow-up experiment 20").

    analysis/.venv/bin/python analysis/aut.py --samples <collect.py samples.csv of the aut sessions> --out DIR

Conditions. Follow-up 20 (label aut), from each process's configuration (collect.py unchanged): l4_xla_noautotune ->
Gna, r1 and r2 in the same part (one VM, as G and Gd of the main study). References of the main study (--main,
primary_v2): G r1 and r2 and V6 r1 (samples.csv), RC(G) seed pairs (contrasts/pairs_seed.csv), HW(V6,G), HW(V6,Gd) and
RC(G) as registered (contrasts/contrasts.csv, copied, not recomputed).
Checks (item c), before anything is compared; the analysis stops if 1 or 2 fails:
  1 every process of the aut sessions exits 0 or its failure is recorded;
  2 one machine type, g2-standard-8 (G's);
  3 completeness: Gna r1 and r2, the 60 targets, seeds 1 to 5 (a missing unit leaves the contrast, section 8);
  4 settings and stack: --xla_gpu_autotune_level=0 in XLA_FLAGS of session.json's xla_env of every process, and the
    device NVIDIA L4 (4a); every session's stack against G's sessions (4b: configuration entry and environment, NVIDIA
    driver, kernel, Python version, packages, pip_freeze.txt, image family; xvm.check_stack). Reported, not a stop:
    the difference of configuration (the flag) and of Python (3.12.16 against 3.12.15) are expected.
Statistics (item d): section 5's target-clustered bootstrap (--replicates 10000, --seed 2026), paired on (target,
seed), per-seed top-1, seeds 1 to 5. RC(Gna) = Gna r1 against Gna r2: the three flip statistics, M4 and M5 medians,
the bit-identical fraction of samples (contrasts.summarize) and the bit-identical (target, seed) (the light-fetch
files, mmCIF timestamp masked; xvm.light_identity); RC(Gna) minus RC(G) (fu13.diff_two, the same resampled targets);
NA(G) = G r1 against Gna r1 (as DET in the main study); HW(V6,Gna) = V6 r1 against Gna r1, descriptive, beside the
registered HW(V6,G) and HW(V6,Gd); cost: the ratio of the median process wall time and of the median compile time,
overall and per bucket, Gna to G (r1 processes), with the compiled device memory ratio.
Reading rules (item e): (1) "recompilation bit-identical without autotuning" if every (target, seed) of Gna r2 is
identical to Gna r1 in the compared files; otherwise the number that differ (a (target, seed) whose files are missing
on a side is not identical: rule 1 then does not hold). (2) The top-1 flip rate of recompilation is "reduced without
autotuning" if the 95% interval of RC(Gna) minus RC(G) lies below 0, "not reduced" otherwise (success flip and flips
that matter secondary, same rule). (3) If (2) fails: "autotuning alone does not account for them"; if (1) and (2)
hold: "turning autotuning off removed the recompilation differences on the L4, consistent with kernel selection by
timing as their source"; if (1) fails and (2) holds: "turning autotuning off reduced them". No direction is predicted.
Outputs in --out: checks.csv, checks_detail.csv, stack.csv, completeness.csv, contrasts.csv, identity.csv,
rc_vs_g.csv, registered.csv, cost.csv, pairs_seed.csv, pairs_sample.csv, counts.csv, report.md.
"""
import argparse
import pathlib
import sys

import numpy as np
import pandas as pd

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / 'analysis'))
import contrasts  # noqa: E402
import followup34 as f34  # noqa: E402
import fu13  # noqa: E402
import fu14  # noqa: E402
import xvm  # noqa: E402

SEEDS = [1, 2, 3, 4, 5]
CONFIG = {'l4_xla_noautotune': 'Gna'}
DEVICE = 'NVIDIA L4'
MACHINE = 'g2-standard-8'
FLAG = '--xla_gpu_autotune_level=0'
SPECS = [('RC(Gna)', 'RC', ('Gna', 'r1'), ('Gna', 'r2'), False), ('NA(G)', 'NA', ('G', 'r1'), ('Gna', 'r1'), False),
         ('HW(V6,Gna)', 'HW', ('V6', 'r1'), ('Gna', 'r1'), False)]
REGISTERED = ['HW(V6,G)', 'HW(V6,Gd)', 'RC(G)']
BIT_IDENTICAL = 'recompilation bit-identical without autotuning'


# ---------------------------------------------------------------- loading

def load_aut(path):
  d = pd.read_csv(path, low_memory=False)
  d = d[(d.label == 'aut') & (d.num_recycles == 10) & (d.num_diffusion_samples == 5)].copy()
  d['condition'] = d.config.map(CONFIG)
  if d.condition.isna().any():
    raise SystemExit(f'sessions of label aut with configurations not of follow-up 20: {sorted(set(d[d.condition.isna()].config))}')
  return d


def load_reference(args):
  m = f34.load_samples(pathlib.Path(args.main) / 'samples.csv', ['main'])
  return m[(m.condition.isin(['G']) & m.repetition.isin(['r1', 'r2'])) | ((m.condition == 'V6') & (m.repetition == 'r1'))]


# ---------------------------------------------------------------- checks

def check_machine_g(d):
  types = sorted(d.machine_type.dropna().astype(str).unique())
  return [{'check': '2b the machine type is G\'s', 'condition': 'Gna', 'value': ', '.join(types),
           'detail': 'pass' if types == [MACHINE] else f'FAIL: expected {MACHINE}'}]


def completeness(ok, all_targets):
  rows = []
  for rep in ('r1', 'r2'):
    for s in SEEDS:
      units = ok[(ok.condition == 'Gna') & (ok.repetition == rep) & (ok.seed == s)].groupby('target').size()
      have = int((units == 5).sum())
      rows.append({'condition': 'Gna', 'repetition': rep, 'seed': s, 'planned_targets': len(all_targets),
                   'targets_with_5_samples': have,
                   'status': 'complete' if have == len(all_targets) else ('not run' if have == 0 else 'incomplete')})
  return pd.DataFrame(rows)


def check_settings(d, root=REPO):
  bad = []
  g = fu14.processes(d)
  for r in g.to_dict('records'):
    _, env = fu14.settings_of(r, root)
    if FLAG not in str(env.get('XLA_FLAGS', '')).split() or r.get('device_kind') != DEVICE:
      bad.append(f'{r["target"]} {r["repetition"]} seed {r["seed"]} (XLA_FLAGS {env.get("XLA_FLAGS")}, device '
                 f'{r.get("device_kind")})')
  detail = [{'check': '4a the flag and the device in every process', 'condition': 'Gna', 'process': b} for b in bad]
  return [{'check': '4a the flag and the device in every process', 'condition': 'Gna',
           'value': f'{len(g) - len(bad)}/{len(g)} processes', 'detail': 'pass' if not bad else
           'reported: ' + '; '.join(bad)[:250]}], detail


# ---------------------------------------------------------------- readings

def reduced_reading(ci_high):
  if ci_high != ci_high:
    return 'not computed'
  return 'reduced without autotuning' if ci_high < 0 else 'not reduced'


def identity_reading(ident):
  """Rule e1 from the light identity summary row: bit-identical only if every (target, seed) is identical."""
  if ident['units'] and ident['identical'] == ident['units']:
    return BIT_IDENTICAL
  if not ident['units']:
    return 'not run'
  return (f'not bit-identical without autotuning: {ident["units"] - ident["identical"]} of {ident["units"]} '
          f'(target, seed) differ or lack files ({ident["value"]})')


def statement(e1, e2):
  """Rule e3, from the readings of rules 1 and 2 (top-1)."""
  if e2 != 'reduced without autotuning':
    return 'autotuning alone does not account for the recompilation differences on the L4'
  if e1 == BIT_IDENTICAL:
    return ('turning autotuning off removed the recompilation differences on the L4, consistent with kernel selection '
            'by timing as their source')
  return 'turning autotuning off reduced the recompilation differences on the L4'


def compile_cost(proc, new='Gna', ref='G'):
  """Ratio of the median compile time, overall and per bucket (fu13.cost gives wall time and memory)."""
  rows = []
  for bucket in ['all'] + sorted(proc.bucket.dropna().astype(int).unique()):
    g = proc if bucket == 'all' else proc[proc.bucket == bucket]
    a, b = g[(g.condition == new) & (g.exit_code == 0)], g[(g.condition == ref) & (g.exit_code == 0)]
    ma, mb = (a.compile_s.median() if len(a) else np.nan), (b.compile_s.median() if len(b) else np.nan)
    rows.append({'new': new, 'reference': ref, 'bucket': bucket, 'compile_median_new': ma, 'compile_median_reference': mb,
                 'compile_ratio': ma / mb if mb and mb == mb and ma == ma else np.nan})
  return pd.DataFrame(rows)


# ---------------------------------------------------------------- main

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
  all_targets = sorted(pd.read_csv(args.targets, dtype=str).pdb_id)

  d = load_aut(args.samples)
  chk = fu13.check_exit(d) + fu13.check_machine(d) + check_machine_g(d)
  pd.DataFrame(chk).to_csv(out / 'checks.csv', index=False)
  if any(r['detail'] != 'pass' for r in chk):
    raise SystemExit('check 1 or 2 failed (checks.csv); nothing is compared')
  ref = load_reference(args)
  okall = f34.sample_rows(pd.concat([d, ref], ignore_index=True))
  comp = completeness(okall, all_targets)
  s4, det4 = check_settings(d)
  g_sessions = [xvm.session_name(s) for s in ref[ref.condition == 'G'].session]
  s4b, stack = xvm.check_stack({'Gna': [xvm.session_name(s) for s in d.session]}, {'Gna': g_sessions},
                               '4b stack of every session against G\'s')
  chk += s4 + s4b
  ident, det5 = xvm.light_identity(okall, ('Gna', 'r1'), ('Gna', 'r2'), 'rule e1: Gna r2 against Gna r1 (light-fetch files)')
  e1 = identity_reading(ident)
  ident = {**ident, 'reading': e1}
  pd.DataFrame(chk).to_csv(out / 'checks.csv', index=False)
  pd.DataFrame(det4 + det5).to_csv(out / 'checks_detail.csv', index=False)

  sp, sm, cons, counts = f34.build(okall, SPECS)
  con = contrasts.summarize(sp, sm, cons, args) if len(sp) else pd.DataFrame()
  main_sp = pd.read_csv(pathlib.Path(args.main) / 'contrasts' / 'pairs_seed.csv')
  vs_g = pd.DataFrame(xvm.diff_rows(sp, 'RC(Gna)', 'RC(G)', 'RC(Gna) minus RC(G)', args, other=main_sp))
  vs_g['role'] = ['primary' if s == 'flip_top1' else 'secondary' for s in vs_g.statistic]
  vs_g['reading'] = [reduced_reading(h) if n else 'not computed' for h, n in zip(vs_g.ci_high, vs_g.n_targets)]
  e2 = vs_g[vs_g.statistic == 'flip_top1'].reading.iloc[0]
  reg = pd.read_csv(pathlib.Path(args.main) / 'contrasts' / 'contrasts.csv')
  reg = reg[reg.contrast.isin(REGISTERED) & reg.statistic.isin(contrasts.FLIP_STATS)].assign(source='main study (registered)')
  proc = fu14.processes(pd.concat([d[d.repetition == 'r1'], ref[(ref.condition == 'G') & (ref.repetition == 'r1')]],
                                  ignore_index=True))
  cst = fu13.cost(proc, [('Gna', 'G')]).merge(compile_cost(proc), on=['new', 'reference', 'bucket'], how='left')
  outputs = {'completeness': comp, 'contrasts': con, 'identity': pd.DataFrame([ident]), 'rc_vs_g': vs_g,
             'registered': reg, 'cost': cst, 'stack': stack, 'pairs_seed': sp, 'pairs_sample': sm, 'counts': counts}
  for name, df in outputs.items():
    df.to_csv(out / f'{name}.csv', index=False)
  (out / 'report.md').write_text(report(chk, comp, con, ident, vs_g, reg, cst, e1, e2))
  print((out / 'report.md').read_text())
  return 0


def report(chk, comp, con, ident, vs_g, reg, cst, e1, e2):
  pct, num = f34.pct, f34.num
  L = ['# Follow-up 20: the L4 without autotuning', '',
       'notes/analysis_plan.md section 10, row "Follow-up experiment 20" (label aut); reported whatever it shows.', '',
       '## Checks', '', '| check | condition | value | result |', '|---|---|---|---|']
  L += [f'| {r["check"]} | {r["condition"]} | {r["value"]} | {r["detail"]} |' for r in chk]
  inc = comp[comp.status != 'complete']
  L += ['', f'Completeness: {len(comp) - len(inc)} of {len(comp)} planned (Gna, repetition, seed) cells complete'
        + ('' if inc.empty else '; not complete: ' + ', '.join(
            f'{r.repetition} seed {r.seed} ({r.targets_with_5_samples}/{r.planned_targets}, {r.status})'
            for r in inc.itertuples())) + '.']
  L += ['', '## Readings (item e)', '', f'- Rule 1: {e1} ({ident["value"]}).', f'- Rule 2 (top-1): {e2}.',
        f'- Rule 3: {statement(e1, e2)}.', '', '## RC(Gna) minus RC(G) (rule 2)', '',
        '| comparison | statistic | role | estimate | 95% CI | targets | reading |', '|---|---|---|---|---|---|---|']
  L += [f'| {r.comparison} | {r.statistic} | {r.role} | {pct(r.estimate)} | {pct(r.ci_low)} to {pct(r.ci_high)} | '
        f'{r.n_targets} | {r.reading} |' for r in vs_g.itertuples()]
  L += ['', '## Contrasts', '', '| contrast | statistic | estimate | 95% CI | n | targets |', '|---|---|---|---|---|---|']
  if len(con):
    keep = con[con.statistic.isin(contrasts.FLIP_STATS + ['m4_ligand_rmsd_median', 'm4_ca_rmsd_global_median',
                                                          'm5_ranking_score_median', 'bit_identical_fraction'])]
    for r in keep.itertuples():
      fmt = pct if r.statistic in contrasts.FLIP_STATS + ['bit_identical_fraction'] else num
      L.append(f'| {r.contrast} | {r.statistic} | {fmt(r.estimate)} | {fmt(r.ci_low)} to {fmt(r.ci_high)} | {r.n} | '
               f'{r.n_targets} |')
  L += ['', 'Registered, beside HW(V6,Gna) (main study, copied):', '', '| contrast | statistic | estimate | 95% CI | n |',
        '|---|---|---|---|---|']
  L += [f'| {r.contrast} | {r.statistic} | {pct(r.estimate)} | {pct(r.ci_low)} to {pct(r.ci_high)} | {r.n} |'
        for r in reg.itertuples()]
  L += ['', '## Cost (descriptive, r1 processes)', '',
        '| new / reference | bucket | wall ratio | compile ratio | compiled memory ratio | n |', '|---|---|---|---|---|---|']
  L += [f'| {r.new} / {r.reference} | {r.bucket} | {num(r.wall_ratio)} | {num(r.compile_ratio)} | '
        f'{num(r.compiled_memory_ratio)} | {r.n_new}, {r.n_reference} |' for r in cst.itertuples()]
  return '\n'.join(L) + '\n'


if __name__ == '__main__':
  sys.exit(main())
