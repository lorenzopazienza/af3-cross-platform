"""Follow-up 14, the float32 trunk on every accelerator (label f32; notes/analysis_plan.md section 10, row of 2026-11-12).

    analysis/.venv/bin/python analysis/fu14.py --samples <collect.py samples.csv of the f32 sessions> --out DIR

Conditions. Follow-up 14, from each process's configuration (collect.py's config column; collect.py unchanged):
tpu_xla_fu13_f32 on v5e -> V5f, a100_xla_fu14_f32 -> Af, h100_xla_fu14_f32 -> Hf, a100_xla_det -> Ad,
h100_xla_det -> Hd. Follow-up 13 (--fu13, its samples.csv): tpu_xla_fu13_f32 on v6e -> V6f, l4_xla_fu13_f32 -> Gf
(r1 and r2). bfloat16 references, from their analysis folders: V5, V6, G (r1, r2) and Gh (seeds 1 and 2) of the main
study (--main, primary_v2), A (r1, r2) and Ah of follow-up 3 (--a100), H (r1, r2) of follow-up 9 (--h100); the
earlier rows' seed pairs of HW(P,Q) and RC(R) from their pairs_seed.csv, HWh(V6,G) on five seeds from follow-up 12
(--prec5).
Checks (item c), before anything is compared; the analysis stops if 1 or 2 fails:
  1 every process of the f32 sessions exits 0 or its failure is recorded (collect.py's failure column);
  2 one machine type per condition;
  3 completeness per condition, repetition and seed (missing units leave the contrasts they belong to, section 8);
  4 the settings reached every process: af3_entry_stats.json "fu13" records global_config_bfloat16 none in every
    V5f, Af and Hf process and in no Ad or Hd process; session.json's xla_env holds JAX_DEFAULT_MATMUL_PRECISION=highest
    for the float32 arms and --xla_gpu_deterministic_ops=true in XLA_FLAGS for Ad and Hd; the device of every process
    is the expected one (TPU v5 lite, NVIDIA A100-SXM4-80GB, NVIDIA H100 80GB HBM3);
  5 V5f r2 against V5f r1, the 10 pilot targets, seed 1, every output file byte for byte (contrasts.outputs_identical,
    the mmCIF timestamp line masked); reported; a difference does not extend V5f r2;
  6 memory: per condition and bucket, the largest compiled device memory (peak plus arguments) against the device
    limit; every out-of-memory stop listed.
Statistics (item d): section 5's target-clustered bootstrap (--replicates 10000, --seed 2026), per-seed top-1, seeds 1
to 5 unless stated. Pairs: all ten pairs of {V5f, V6f, Gf, Af, Hf}, r1 against r1; primary (V5f,V6f), (V6f,Gf),
(Hf,Af), (Gf,Af). For each primary pair D_F = [HW(Pf,Qf) - RC(Rf)] - [HW(P,Q) - RC(R)] (fu13.d_f), R as the earlier
row: V5 for (V5,V6), G for (V6,G) and (G,A), H for (H,A); RC(Gf) measured (follow-up 13); RC(V5f) and RC(Hf) not
measured on the 60 targets, their term set to 0 (an upper bound of D_F in every replicate). Descriptive: HW(Pf,Qf)
minus HW(P,Q) for all ten pairs; HW(Pf,Qf) minus HWh(P,Q) where it exists (HWh(V6,G) five seeds; HWh(G,A) on seeds 1
and 2, Gh against Ah, computed here; HWh(V6,A) seeds 1 and 2); F(V5), F(A), F(H) (bfloat16 against float32 on one
chip); RC(V5f) on its pilot subset; per-seed top-1 success rates on the 60 and the 39 post-cutoff targets, and the
success flip and flips that matter of the ten pairs on the 39 post-cutoff targets.
Reading rules (item e): reduced if the 95% interval of D_F lies below 0 (top-1 primary; success flip and flips that
matter secondary); agree if the upper limit of the 95% interval of the float32 top-1 flip rate is below 5.0 points,
"agreement not shown" otherwise, for each of the ten pairs; "agree on every accelerator of the study" only if all ten
pairs agree (an intersection-union test, no multiplicity correction). Item b2: per GPU, "recompilation bit-identical
with the flag" if all 50 (target, seed) of r2 equal r1 in every output file; RC(Ad), RC(Hd) beside RC(A), RC(H) on the
same 10 targets and seeds; DET(A) = A r1 against Ad r1, DET(H) = H r1 against Hd r1. Cost (descriptive): ratios of
median process wall time and of compiled device memory (and, on TPU, the allocator's peak reservation) per bucket,
V5f/V5, Af/A, Hf/H, Ad/A and Hd/H (A and H on the pilot targets for the last two).
Outputs in --out: checks.csv, checks_detail.csv, completeness.csv, contrasts.csv, agree.csv, d_f.csv, secondary.csv,
post_cutoff.csv, success.csv, det_identity.csv, cost.csv, memory.csv, failures.csv, pairs_seed.csv, pairs_sample.csv,
counts.csv, report.md. Items b1 and b2 are read independently: the analysis runs with or without the item b2 sessions.
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
import fu13  # noqa: E402

THRESHOLD = 0.05                          # item e (2): agree if the upper 95% limit of the top-1 flip rate is below it
SEEDS = [1, 2, 3, 4, 5]
FLOAT32 = ['V5f', 'V6f', 'Gf', 'Af', 'Hf']
PRIMARY = [('V5f', 'V6f'), ('V6f', 'Gf'), ('Hf', 'Af'), ('Gf', 'Af')]
SECONDARY = [('V5f', 'Gf'), ('V5f', 'Af'), ('Hf', 'V5f'), ('V6f', 'Af'), ('Hf', 'V6f'), ('Hf', 'Gf')]
PAIRS = PRIMARY + SECONDARY
# float32 pair -> (bfloat16 pair, the analysis folder whose pairs_seed.csv holds it)
BF16 = {('V5f', 'V6f'): ('HW(V5,V6)', 'main'), ('V6f', 'Gf'): ('HW(V6,G)', 'main'), ('Hf', 'Af'): ('HW(H,A)', 'h100'),
        ('Gf', 'Af'): ('HW(G,A)', 'a100'), ('V5f', 'Gf'): ('HW(V5,G)', 'main'), ('V5f', 'Af'): ('HW(V5,A)', 'a100'),
        ('Hf', 'V5f'): ('HW(H,V5)', 'h100'), ('V6f', 'Af'): ('HW(V6,A)', 'a100'), ('Hf', 'V6f'): ('HW(H,V6)', 'h100'),
        ('Hf', 'Gf'): ('HW(H,G)', 'h100')}
# primary pair -> (RC of the bfloat16 reference and its folder, RC of the float32 condition or None: term set to 0)
RC = {('V5f', 'V6f'): (('RC(V5)', 'main'), None), ('V6f', 'Gf'): (('RC(G)', 'main'), 'RC(Gf)'),
      ('Hf', 'Af'): (('RC(H)', 'h100'), None), ('Gf', 'Af'): (('RC(G)', 'main'), 'RC(Gf)')}
HWH = {('V6f', 'Gf'): ('HWh(V6,G)', 'prec5'), ('Gf', 'Af'): ('HWh(G,A)', 'here'), ('V6f', 'Af'): ('HWh(V6,A)', 'a100')}
DEVICE = {'V5f': 'TPU v5 lite', 'Af': 'NVIDIA A100-SXM4-80GB', 'Hf': 'NVIDIA H100 80GB HBM3',
          'Ad': 'NVIDIA A100-SXM4-80GB', 'Hd': 'NVIDIA H100 80GB HBM3'}
DET_FLAG = '--xla_gpu_deterministic_ops=true'
PLANNED = {'V5f': {**{('r1', s): 60 for s in SEEDS}, ('r2', 1): 10},
           'Af': {('r1', s): 60 for s in SEEDS}, 'Hf': {('r1', s): 60 for s in SEEDS},
           'Ad': {(r, s): 10 for r in ('r1', 'r2') for s in SEEDS}, 'Hd': {(r, s): 10 for r in ('r1', 'r2') for s in SEEDS}}


def hw(p, q):
  return f'HW({p},{q})'


# ---------------------------------------------------------------- conditions and loading

def condition_of(config, device_platform):
  """Follow-up 14 condition from the configuration (and the device's platform code for the TPU float32 trunk)."""
  if config == 'tpu_xla_fu13_f32':
    return {'V5': 'V5f', 'V6': 'V6f'}.get(str(device_platform)[:2])
  return {'a100_xla_fu14_f32': 'Af', 'h100_xla_fu14_f32': 'Hf', 'a100_xla_det': 'Ad', 'h100_xla_det': 'Hd',
          'l4_xla_fu13_f32': 'Gf'}.get(config)


def load_f32(path):
  d = pd.read_csv(path, low_memory=False)
  d = d[(d.label == 'f32') & (d.num_recycles == 10) & (d.num_diffusion_samples == 5)].copy()
  d['condition'] = [condition_of(c, p) for c, p in zip(d.config, d.platform)]
  bad = d.condition.isna() | ~d.condition.isin(PLANNED)
  if bad.any():
    raise SystemExit(f'sessions of label f32 with configurations not of follow-up 14: {sorted(set(d[bad].config))}')
  return d


def load_fu13(path):
  d = pd.read_csv(path, low_memory=False)
  d = d[(d.label == 'why') & d.config.isin(['tpu_xla_fu13_f32', 'l4_xla_fu13_f32'])].copy()
  d['condition'] = [condition_of(c, p) for c, p in zip(d.config, d.platform)]
  return d


def load_reference(args):
  """bfloat16 rows: V5, V6, G (r1, r2) and Gh of the main study, A (r1, r2) and Ah of follow-up 3, H (r1, r2) of 9."""
  m = f34.load_samples(pathlib.Path(args.main) / 'samples.csv', ['main'])
  m = m[m.condition.isin(['V5', 'V6', 'G', 'Gh']) & m.repetition.isin(['r1', 'r2'])]
  a = f34.load_samples(pathlib.Path(args.a100) / 'samples.csv', ['a100'])
  a = a[a.condition.isin(['A', 'Ah']) & a.repetition.isin(['r1', 'r2'])]
  h = f34.load_samples(pathlib.Path(args.h100) / 'samples.csv', ['h100'])
  h = h[(h.condition == 'H') & h.repetition.isin(['r1', 'r2'])]
  return pd.concat([m, a, h], ignore_index=True)


def processes(d):
  return d.drop_duplicates(['session', 'target', 'seed'])


# ---------------------------------------------------------------- checks

def completeness(ok):
  rows = []
  for c, plan in PLANNED.items():
    for (rep, seed), n in sorted(plan.items()):
      units = ok[(ok.condition == c) & (ok.repetition == rep) & (ok.seed == seed)].groupby('target').size()
      have = int((units == 5).sum())
      rows.append({'condition': c, 'repetition': rep, 'seed': seed, 'planned_targets': n, 'targets_with_5_samples': have,
                   'status': 'complete' if have == n else ('not run' if have == 0 else 'incomplete')})
  return pd.DataFrame(rows)


def settings_of(row, root=REPO):
  """(global_config_bfloat16 or None, xla_env dict) of one process, from its run folder."""
  run = root / str(row['session'])
  stats = run / str(row['target']) / 'af3_entry_stats.json'
  bf16 = None
  if stats.is_file():
    bf16 = (json.loads(stats.read_text()).get('fu13') or {}).get('global_config_bfloat16')
  env = {}
  sj = run / 'session.json'
  if sj.is_file():
    env = json.loads(sj.read_text()).get('xla_env') or {}
  return bf16, env


def check_settings(d, root=REPO):
  rows, detail = [], []
  for c, g in processes(d).groupby('condition'):
    bad = []
    for r in g.to_dict('records'):
      bf16, env = settings_of(r, root)
      f32 = c in ('V5f', 'Af', 'Hf')
      ok = (bf16 == 'none') if f32 else (bf16 is None)
      ok = ok and ((env.get('JAX_DEFAULT_MATMUL_PRECISION') == 'highest') if f32 else
                   (DET_FLAG in str(env.get('XLA_FLAGS', '')) and 'JAX_DEFAULT_MATMUL_PRECISION' not in env))
      ok = ok and r.get('device_kind') == DEVICE[c]
      if not ok:
        bad.append(f'{r["target"]} {r["repetition"]} seed {r["seed"]} (bfloat16 {bf16}, device {r.get("device_kind")})')
    detail += [{'check': '4 settings reached every process', 'condition': c, 'process': b} for b in bad]
    rows.append({'check': '4 settings reached every process', 'condition': c,
                 'value': f'{len(g) - len(bad)}/{len(g)} processes', 'detail': 'pass' if not bad else
                 'reported: ' + '; '.join(bad)[:250]})
  return rows, detail


def memory(proc):
  rows = []
  for (c, b), g in proc.groupby(['condition', 'bucket']):
    need = (g.compiled_peak_bytes.astype(float) + g.compiled_args_bytes.astype(float)) / 2 ** 30
    lim = g.device_bytes_limit.astype(float) / 2 ** 30
    rows.append({'condition': c, 'bucket': int(b), 'processes': len(g), 'max_compiled_gib': float(need.max()),
                 'device_limit_gib': float(lim.max()), 'fraction_of_limit': float((need / lim).max()),
                 'failed': int((g.exit_code != 0).sum())})
  return pd.DataFrame(rows)


# ---------------------------------------------------------------- contrasts

def specs():
  s = [(hw(p, q), 'HW', (p, 'r1'), (q, 'r1'), (p, q) in PRIMARY) for p, q in PAIRS]
  s += [('RC(Gf)', 'RC', ('Gf', 'r1'), ('Gf', 'r2'), False), ('RC(V5f)', 'RC', ('V5f', 'r1'), ('V5f', 'r2'), False),
        ('F(V5)', 'F', ('V5', 'r1'), ('V5f', 'r1'), False), ('F(A)', 'F', ('A', 'r1'), ('Af', 'r1'), False),
        ('F(H)', 'F', ('H', 'r1'), ('Hf', 'r1'), False), ('HWh(G,A)', 'HW', ('Gh', 'r1'), ('Ah', 'r1'), False),
        ('RC(Ad)', 'RC', ('Ad', 'r1'), ('Ad', 'r2'), False), ('RC(Hd)', 'RC', ('Hd', 'r1'), ('Hd', 'r2'), False),
        ('DET(A)', 'DET', ('A', 'r1'), ('Ad', 'r1'), False), ('DET(H)', 'DET', ('H', 'r1'), ('Hd', 'r1'), False),
        ('RC(A) pilot', 'RC', ('A_pilot', 'r1'), ('A_pilot', 'r2'), False),
        ('RC(H) pilot', 'RC', ('H_pilot', 'r1'), ('H_pilot', 'r2'), False)]
  return s


def zero_like(sp):
  z = sp.copy()
  for st in contrasts.FLIP_STATS:
    z[st] = 0
  return z


def agree_reading(ci_high):
  if ci_high != ci_high:
    return 'not computed'
  return 'agree' if ci_high < THRESHOLD else 'agreement not shown'


def agree_table(con):
  rows = []
  for p, q in PAIRS:
    r = con[(con.contrast == hw(p, q)) & (con.statistic == 'flip_top1')] if len(con) else con
    est, lo, hi, n = (r.estimate.iloc[0], r.ci_low.iloc[0], r.ci_high.iloc[0], int(r.n.iloc[0])) if len(r) else \
        (np.nan, np.nan, np.nan, 0)
    rows.append({'pair': hw(p, q), 'role': 'primary' if (p, q) in PRIMARY else 'secondary', 'n': n,
                 'flip_top1': est, 'ci_low': lo, 'ci_high': hi, 'threshold': THRESHOLD,
                 'reading': agree_reading(hi) if n else 'not computed',
                 'note': 'known at specification (follow-up 13)' if (p, q) == ('V6f', 'Gf') else ''})
  return pd.DataFrame(rows)


def generality(agree):
  if (agree.reading == 'agree').all() and len(agree) == len(PAIRS):
    return 'agree on every accelerator of the study (all ten pairs agree)'
  return ('not shown for every accelerator: agree ' + (', '.join(agree[agree.reading == 'agree'].pair) or 'none')
          + '; not ' + ', '.join(agree[agree.reading != 'agree'].pair))


def d_f_table(sp, ref_sp, args):
  rows = []
  for pair in PRIMARY:
    (rc_name, rc_src), rcf_name = RC[pair]
    hw_name, hw_src = BF16[pair]
    hwf = sp[sp.contrast == hw(*pair)]
    rcf = sp[sp.contrast == rcf_name] if rcf_name else zero_like(hwf)
    for i, stat in enumerate(contrasts.FLIP_STATS):
      r = fu13.d_f(hwf, rcf, ref_sp[hw_src][ref_sp[hw_src].contrast == hw_name],
                   ref_sp[rc_src][ref_sp[rc_src].contrast == rc_name], stat, args.seed, args.replicates)
      rows.append({'pair': hw(*pair), 'reference': f'{hw_name} minus {rc_name}',
                   'float32_recompilation': rcf_name or 'set to 0 (not measured; D_F an upper bound)',
                   'statistic': stat, 'role': 'primary' if i == 0 else 'secondary', **r,
                   'reading': fu13.f_reading(r['ci_high'])})
  return pd.DataFrame(rows)


def secondary_table(sp, ref_sp, args):
  rows = []
  for pair in PAIRS:
    hw_name, src = BF16[pair]
    for stat in contrasts.FLIP_STATS:
      rows.append({'comparison': f'{hw(*pair)} minus {hw_name}', 'statistic': stat,
                   **fu13.diff_two(sp[sp.contrast == hw(*pair)], ref_sp[src][ref_sp[src].contrast == hw_name], stat,
                                   args.seed, args.replicates)})
    if pair in HWH:
      name, src = HWH[pair]
      ref = sp[sp.contrast == name] if src == 'here' else ref_sp[src][ref_sp[src].contrast == name]
      for stat in contrasts.FLIP_STATS:
        rows.append({'comparison': f'{hw(*pair)} minus {name}', 'statistic': stat,
                     **fu13.diff_two(sp[sp.contrast == hw(*pair)], ref, stat, args.seed, args.replicates)})
  return pd.DataFrame(rows)


def det_identity(ok):
  rows, detail = [], []
  for c in ('Ad', 'Hd'):
    if not ((ok.condition == c) & (ok.repetition == 'r2')).any():
      rows.append({'check': f'item b2: {c} r2 against r1', 'condition': c, 'value': 'not run', 'detail': 'not run'})
      continue
    s, det = fu13.identity_rows(ok, ok, c, c, 'r2', 'r1', SEEDS, f'item b2: {c} r2 against r1')
    s['detail'] = ('recompilation bit-identical with the flag' if s['detail'] == 'pass'
                   else s['detail'].replace('reported', 'not bit-identical'))
    rows.append(s)
    detail += det
  return rows, detail


# ---------------------------------------------------------------- main

def main():
  ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
  ap.add_argument('--samples', required=True)
  ap.add_argument('--fu13', default=str(REPO / 'results' / 'analysis' / 'fu13_20261112'))
  ap.add_argument('--main', default=str(REPO / 'results' / 'analysis' / 'main_20261014' / 'primary_v2'))
  ap.add_argument('--a100', default=str(REPO / 'results' / 'analysis' / 'a100_20261018'))
  ap.add_argument('--h100', default=str(REPO / 'results' / 'analysis' / 'h100_20261104'))
  ap.add_argument('--prec5', default=str(REPO / 'results' / 'analysis' / 'prec5_20261107'))
  ap.add_argument('--targets', default=str(REPO / 'targets' / 'posebusters_subset.csv'))
  ap.add_argument('--out', required=True)
  ap.add_argument('--replicates', type=int, default=10000)
  ap.add_argument('--seed', type=int, default=2026)
  args = ap.parse_args()
  out = pathlib.Path(args.out)
  out.mkdir(parents=True, exist_ok=True)
  tt = pd.read_csv(args.targets, dtype=str)
  pilot = set(tt[tt.pilot == 'yes'].pdb_id)
  post = sorted(tt[tt.rcsb_initial_release.str[:10] > '2021-09-30'].pdb_id)

  d = load_f32(args.samples)
  chk = fu13.check_exit(d) + fu13.check_machine(d)
  pd.DataFrame(chk).to_csv(out / 'checks.csv', index=False)
  if any(r['detail'] != 'pass' for r in chk):
    raise SystemExit('check 1 or 2 failed (checks.csv); nothing is compared')
  ok = f34.sample_rows(d)
  comp = completeness(ok)
  s4, det4 = check_settings(d)
  chk += s4
  s5, det5 = fu13.identity_rows(ok, ok, 'V5f', 'V5f', 'r2', 'r1', [1], '5 V5f r2 against V5f r1')
  chk.append(s5)
  di, det_di = det_identity(ok)
  ref = load_reference(args)
  f13 = load_fu13(pathlib.Path(args.fu13) / 'samples.csv')
  pil = ref[ref.condition.isin(['A', 'H']) & ref.target.isin(pilot)].copy()
  pil['condition'] = pil.condition + '_pilot'
  allrows = pd.concat([d, f13, ref, pil], ignore_index=True)
  proc = processes(pd.concat([d, ref[ref.condition.isin(['V5', 'A', 'H']) & (ref.repetition == 'r1')],
                              pil[pil.repetition == 'r1']], ignore_index=True))      # references: r1, as item e 4
  mem = memory(processes(d))
  chk.append({'check': '6 memory', 'condition': 'V5f, Af, Hf, Ad, Hd',
              'value': f'largest {mem.fraction_of_limit.max():.0%} of the device limit' if len(mem) else 'no process',
              'detail': 'pass' if len(mem) and (mem.failed == 0).all() else 'reported (see memory.csv, failures.csv)'})
  pd.DataFrame(chk).to_csv(out / 'checks.csv', index=False)
  pd.DataFrame(det4 + det5 + det_di).to_csv(out / 'checks_detail.csv', index=False)

  okall = f34.sample_rows(allrows)
  sp, sm, cons, counts = f34.build(okall, specs())
  con = contrasts.summarize(sp, sm, cons, args) if len(sp) else pd.DataFrame()
  ref_sp = {'main': pd.read_csv(pathlib.Path(args.main) / 'contrasts' / 'pairs_seed.csv'),
            'a100': pd.read_csv(pathlib.Path(args.a100) / 'pairs_seed.csv'),
            'h100': pd.read_csv(pathlib.Path(args.h100) / 'pairs_seed.csv'),
            'prec5': pd.read_csv(pathlib.Path(args.prec5) / 'pairs_seed.csv')}
  agree = agree_table(con)
  dft = d_f_table(sp, ref_sp, args)
  sec = secondary_table(sp, ref_sp, args)
  post_sp = sp[sp.target.isin(post) & sp.contrast.isin([hw(*p) for p in PAIRS])]
  post_con = contrasts.summarize(post_sp, sm[sm.target.isin(post)], {k: v for k, v in cons.items()
                                                                     if k in set(post_sp.contrast)}, args) \
      if len(post_sp) else pd.DataFrame()
  if len(post_con):
    post_con = post_con[post_con.statistic.isin(['flip_success', 'flip_matters'])]
  conds = [(c, 'r1') for c in ('V5f', 'V6f', 'Gf', 'Af', 'Hf', 'V5', 'V6', 'G', 'A', 'H')]
  succ = pd.concat([fu13.success_rates(okall, conds, args.seed, args.replicates).assign(targets='all 60'),
                    fu13.success_rates(okall[okall.target.isin(post)], conds, args.seed, args.replicates)
                    .assign(targets=f'post-cutoff {len(post)}')], ignore_index=True)
  cst = fu13.cost(proc, [('V5f', 'V5'), ('Af', 'A'), ('Hf', 'H'), ('Ad', 'A_pilot'), ('Hd', 'H_pilot')])
  fails = processes(d)[processes(d).exit_code != 0][['condition', 'repetition', 'target', 'seed', 'session',
                                                     'exit_code', 'failure']]
  fu13_df = pathlib.Path(args.fu13) / 'f_reading.csv'
  same13 = None
  if fu13_df.is_file():
    old = pd.read_csv(fu13_df)
    new = dft[dft.pair == 'HW(V6f,Gf)'].set_index('statistic')
    same13 = all(abs(new.loc[r.statistic, 'estimate'] - r.estimate) < 1e-12 and
                 abs(new.loc[r.statistic, 'ci_high'] - r.ci_high) < 1e-12 for r in old.itertuples())
  outputs = {'completeness': comp, 'contrasts': con, 'agree': agree, 'd_f': dft, 'secondary': sec,
             'post_cutoff': post_con, 'success': succ, 'det_identity': pd.DataFrame(di), 'cost': cst, 'memory': mem,
             'failures': fails, 'pairs_seed': sp, 'pairs_sample': sm, 'counts': counts}
  for name, df in outputs.items():
    df.to_csv(out / f'{name}.csv', index=False)
  (out / 'report.md').write_text(report(chk, comp, agree, dft, sec, con, succ, di, cst, mem, fails, same13))
  print((out / 'report.md').read_text())
  return 0


def report(chk, comp, agree, dft, sec, con, succ, di, cst, mem, fails, same13):
  pct, num = f34.pct, f34.num
  L = ['# Follow-up 14: the float32 trunk on every accelerator', '',
       'notes/analysis_plan.md section 10, row of 2026-11-12; reported whatever it shows.', '',
       '## Checks', '', '| check | condition | value | result |', '|---|---|---|---|']
  L += [f'| {r["check"]} | {r["condition"]} | {r["value"]} | {r["detail"]} |' for r in chk]
  inc = comp[comp.status != 'complete']
  L += ['', f'Completeness: {len(comp) - len(inc)} of {len(comp)} planned (condition, repetition, seed) cells complete'
        + ('' if inc.empty else '; not complete: ' + ', '.join(f'{r.condition} {r.repetition} seed {r.seed} '
                                                               f'({r.targets_with_5_samples}/{r.planned_targets})'
                                                               for r in inc.itertuples())) + '.']
  if len(fails):
    L += ['', 'Failed processes (recorded, not rerun): ' + '; '.join(
        f'{r.condition} {r.repetition} {r.target} seed {r.seed} ({r.failure})' for r in fails.itertuples()) + '.']
  L += ['', '## Agree (item e 2): upper 95% limit of the float32 top-1 flip rate below 5.0 points', '',
        '| pair | role | top-1 flip | 95% CI | n | reading | note |', '|---|---|---|---|---|---|---|']
  L += [f'| {r.pair} | {r.role} | {pct(r.flip_top1)} | {pct(r.ci_low)} to {pct(r.ci_high)} | {r.n} | {r.reading} | {r.note} |'
        for r in agree.itertuples()]
  L += ['', f'Generality (item e 3): {generality(agree)}.', '',
        '## D_F against bfloat16 at the default precision (item e 1)', '',
        '| pair | reference | float32 recompilation | statistic | role | D_F | 95% CI | targets | reading |',
        '|---|---|---|---|---|---|---|---|---|']
  L += [f'| {r.pair} | {r.reference} | {r.float32_recompilation} | {r.statistic} | {r.role} | {pct(r.estimate)} | '
        f'{pct(r.ci_low)} to {pct(r.ci_high)} | {r.n_targets} | {r.reading} |' for r in dft.itertuples()]
  if same13 is not None:
    L += ['', f'D_F of (V6f,Gf) recomputed here {"equals" if same13 else "DIFFERS from"} follow-up 13\'s f_reading.csv.']
  L += ['', '## Descriptive comparisons', '', '| comparison | statistic | estimate | 95% CI | targets |', '|---|---|---|---|---|']
  L += [f'| {r.comparison} | {r.statistic} | {pct(r.estimate)} | {pct(r.ci_low)} to {pct(r.ci_high)} | {r.n_targets} |'
        for r in sec.itertuples()]
  if len(con):
    keep = con[con.statistic.isin(contrasts.FLIP_STATS) & ~con.contrast.str.startswith('HW(')]
    L += ['', '| contrast | statistic | estimate | 95% CI | n |', '|---|---|---|---|---|']
    L += [f'| {r.contrast} | {r.statistic} | {pct(r.estimate)} | {pct(r.ci_low)} to {pct(r.ci_high)} | {r.n} |'
          for r in keep.itertuples()]
  L += ['', '| condition | targets | top-1 success (per seed) | 95% CI | n |', '|---|---|---|---|---|']
  L += [f'| {r.condition} {r.repetition} | {r.targets} | {pct(r.top1_success_rate)} | {pct(r.ci_low)} to {pct(r.ci_high)} | {r.n} |'
        for r in succ.itertuples()]
  L += ['', '## Item b2: deterministic flag on the A100 and the H100', '']
  L += [f'- {r["condition"]}: {r["value"]}; {r["detail"]}' for r in di]
  L += ['', '## Cost and memory (descriptive)', '',
        '| new / reference | bucket | wall ratio | compiled memory ratio | peak reserved ratio | n |', '|---|---|---|---|---|---|']
  L += [f'| {r.new} / {r.reference} | {r.bucket} | {num(r.wall_ratio)} | {num(r.compiled_memory_ratio)} | '
        f'{num(r.peak_reserved_ratio)} | {r.n_new}, {r.n_reference} |' for r in cst.itertuples()]
  L += ['', '| condition | bucket | processes | largest compiled GiB | device limit GiB | failed |', '|---|---|---|---|---|---|']
  L += [f'| {r.condition} | {r.bucket} | {r.processes} | {r.max_compiled_gib:.2f} | {r.device_limit_gib:.2f} | {r.failed} |'
        for r in mem.itertuples()]
  return '\n'.join(L) + '\n'


if __name__ == '__main__':
  sys.exit(main())
