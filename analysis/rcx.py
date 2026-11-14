"""Follow-up 15, recompilation of the float32 trunk on the A100 and the H100 and of the L4 Triton arm (label rcx;
notes/analysis_plan.md section 10, row of 2026-11-14 "Follow-up experiment 15").

    analysis/.venv/bin/python analysis/rcx.py --samples <collect.py samples.csv of the rcx sessions> --out DIR

Conditions. Follow-up 15 (label rcx, repetition r2), from each process's configuration (collect.py unchanged):
a100_xla_fu14_f32 -> Af, h100_xla_fu14_f32 -> Hf, l4_triton -> GT. Their r1: Af and Hf of follow-up 14 (--fu14, its
samples.csv), GT of the main study (--main, primary_v2/samples.csv). Part A3 (GT r2) is optional: without GT sessions
its rows say "not run".
Checks (item c), before anything is compared; the analysis stops if 1 or 2 fails:
  1 every process of the rcx sessions exits 0 or its failure is recorded;
  2 one machine type per condition, the r1's (a2-ultragpu-1g, a3-highgpu-1g, g2-standard-8);
  3 completeness: Af r2 and Hf r2 the 10 pilot targets, GT r2 the 60 targets, seeds 1 to 5 (a missing unit leaves
    the contrast, section 8);
  4 settings and stack: Af and Hf, global_config_bfloat16 none (af3_entry_stats.json) and
    JAX_DEFAULT_MATMUL_PRECISION=highest (xla_env); GT, neither; every process on the expected device (NVIDIA
    A100-SXM4-80GB, NVIDIA H100 80GB HBM3, NVIDIA L4); for every r2 session against the r1 sessions of its arm: the
    NVIDIA driver (device_check.txt), the kernel (session.json host), the Python version, the Python packages
    (pip_freeze.txt byte for byte; setup.json packages with AlphaFold3's build-date suffix removed). Reported, not a
    stop: a difference is named in the report beside the arm's result (GT r2's Python, 3.12.16 against 3.12.15, is
    declared in the row).
Contrasts (item d): RC(Af) = Af r1 against Af r2 and RC(Hf) = Hf r1 against Hf r2, the 10 pilot targets, seeds 1 to
5 (50 (target, seed) each); RC(GT) = GT r1 against GT r2, the 60 targets, seeds 1 to 5 (300): per-seed top-1 flip,
success flip and flips that matter, M4 and M5 medians and 90th percentiles, the fraction of bit-identical samples
(contrasts.summarize; section 5's target bootstrap, --replicates 10000, --seed 2026).
Bit-identity (item e 1): Af and Hf, every output file of every (target, seed) of r2 against r1
(contrasts.outputs_identical, the mmCIF timestamp line masked: the strict check of follow-up 14 item b2); GT, the
files of the light fetch (the top model and the five sample mmCIFs, the ranking scores and the summary confidences;
light_identical), because GT r1 was fetched light and the full files of 600 processes do not fit on the Mac.
"recompilation bit-identical" if every (target, seed) is identical, otherwise the number that differ.
Beside them, registered (follow-up 14's contrasts.csv, copied, not recomputed): RC(A) and RC(H) on the pilot targets,
RC(Ad), RC(Hd), RC(V5f), RC(Gf). RC(GT) against RC(G) (descriptive, item e 3): RC(GT) minus RC(G) on the 60 targets
and seeds 1 to 5 (fu13.diff_two, the same resampled targets), RC(G) from the main study's pairs_seed.csv.
D_F sensitivity (item e 2): D_F(Hf,Af) = [HW(Hf,Af) - RC(Hf)] - [HW(H,A) - RC(H)] with RC(Hf) measured instead of
the registered 0, (i) on the 60 targets of the registered D_F, RC(Hf)'s rate taken over the pilot targets among each
replicate's draws (0 in a replicate that draws none, counted), the same draws as the registered D_F; (ii) on the
10 pilot targets only, every term on them, beside the same with RC(Hf) set to 0. The registered D_F (60 targets,
RC(Hf) = 0) is recomputed first and must equal follow-up 14's d_f.csv; the registered reading stands whatever (i) and
(ii) show; the report states whether (i) reads as the registered one under rule e1.
Outputs in --out: checks.csv, checks_detail.csv, stack.csv, completeness.csv, contrasts.csv, identity.csv,
registered.csv, rc_gt_vs_g.csv, d_f_sensitivity.csv, pairs_seed.csv, pairs_sample.csv, counts.csv, report.md.
"""
import argparse
import hashlib
import json
import pathlib
import re
import sys

import numpy as np
import pandas as pd

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / 'analysis'))
import contrasts  # noqa: E402
import followup34 as f34  # noqa: E402
import fu13  # noqa: E402
import fu14  # noqa: E402

SEEDS = [1, 2, 3, 4, 5]
CONFIG = {'a100_xla_fu14_f32': 'Af', 'h100_xla_fu14_f32': 'Hf', 'l4_triton': 'GT'}
DEVICE = {'Af': 'NVIDIA A100-SXM4-80GB', 'Hf': 'NVIDIA H100 80GB HBM3', 'GT': 'NVIDIA L4'}
MACHINE = {'Af': 'a2-ultragpu-1g', 'Hf': 'a3-highgpu-1g', 'GT': 'g2-standard-8'}
TARGETS = {'Af': 'pilot', 'Hf': 'pilot', 'GT': 'all'}
RCS = [('RC(Af)', 'Af'), ('RC(Hf)', 'Hf'), ('RC(GT)', 'GT')]
REGISTERED = ['RC(A) pilot', 'RC(H) pilot', 'RC(Ad)', 'RC(Hd)', 'RC(V5f)', 'RC(Gf)']
LIGHT_TARGET = ('_model.cif', '_ranking_scores.csv', '_summary_confidences.json')
LIGHT_SAMPLE = ('_model.cif', '_summary_confidences.json')
R1_SESSIONS = {'Af': ('fu14', r'_a100_f32_af_\d+$'), 'Hf': ('fu14', r'_h100_f32_hf_\d+$'),
               'GT': ('main', r'_l4_main_gt_\d+$')}


# ---------------------------------------------------------------- loading

def load_rcx(path):
  d = pd.read_csv(path, low_memory=False)
  d = d[(d.label == 'rcx') & (d.num_recycles == 10) & (d.num_diffusion_samples == 5)].copy()
  d['condition'] = d.config.map(CONFIG)
  if d.condition.isna().any():
    raise SystemExit(f'sessions of label rcx with configurations not of follow-up 15: {sorted(set(d[d.condition.isna()].config))}')
  return d


def load_r1(args):
  """Af and Hf r1 (follow-up 14) and GT r1 (main study), as their analyses collected them."""
  f = fu14.load_f32(pathlib.Path(args.fu14) / 'samples.csv')
  f = f[f.condition.isin(['Af', 'Hf']) & (f.repetition == 'r1')]
  m = f34.load_samples(pathlib.Path(args.main) / 'samples.csv', ['main'])
  m = m[(m.condition == 'GT') & (m.repetition == 'r1')]
  return pd.concat([f, m], ignore_index=True)


# ---------------------------------------------------------------- checks

def completeness(ok, pilot, all_targets):
  rows = []
  for c in ('Af', 'Hf', 'GT'):
    n = len(pilot) if TARGETS[c] == 'pilot' else len(all_targets)
    for s in SEEDS:
      units = ok[(ok.condition == c) & (ok.repetition == 'r2') & (ok.seed == s)].groupby('target').size()
      have = int((units == 5).sum())
      rows.append({'condition': c, 'repetition': 'r2', 'seed': s, 'planned_targets': n, 'targets_with_5_samples': have,
                   'status': 'complete' if have == n else ('not run' if have == 0 else 'incomplete')})
  return pd.DataFrame(rows)


def check_machine_r1(d):
  rows = []
  for c, g in d.groupby('condition'):
    types = sorted(g.machine_type.dropna().astype(str).unique())
    ok = types == [MACHINE[c]]
    rows.append({'check': '2b the machine type of r1', 'condition': c, 'value': ', '.join(types),
                 'detail': 'pass' if ok else f'FAIL: expected {MACHINE[c]}'})
  return rows


def check_settings(d, root=REPO):
  rows, detail = [], []
  for c, g in fu14.processes(d).groupby('condition'):
    bad = []
    for r in g.to_dict('records'):
      bf16, env = fu14.settings_of(r, root)
      if c in ('Af', 'Hf'):
        ok = bf16 == 'none' and env.get('JAX_DEFAULT_MATMUL_PRECISION') == 'highest'
      else:
        ok = bf16 is None and 'JAX_DEFAULT_MATMUL_PRECISION' not in env
      ok = ok and r.get('device_kind') == DEVICE[c]
      if not ok:
        bad.append(f'{r["target"]} seed {r["seed"]} (bfloat16 {bf16}, device {r.get("device_kind")})')
    detail += [{'check': '4a settings reached every process', 'condition': c, 'process': b} for b in bad]
    rows.append({'check': '4a settings reached every process', 'condition': c, 'value': f'{len(g) - len(bad)}/{len(g)} processes',
                 'detail': 'pass' if not bad else 'reported: ' + '; '.join(bad)[:250]})
  return rows, detail


def stack_of(session_dir):
  """The recorded software stack of one session: NVIDIA driver, kernel, Python, packages, pip freeze."""
  d = pathlib.Path(session_dir)
  out = {'driver': None, 'kernel': None, 'python': None, 'packages': None, 'pip_freeze_sha256': None}
  dc = d / 'device_check.txt'
  if dc.is_file():
    m = re.search(r'Driver Version: (\S+)', dc.read_text(errors='replace'))
    out['driver'] = m.group(1) if m else None
  sj = next(iter(sorted(d.glob('*_fresh/session.json'))), None)
  if sj is not None:
    j = json.loads(sj.read_text())
    out['kernel'] = (j.get('host') or {}).get('platform')
    out['python'] = (j.get('probe') or {}).get('python')
  st = d / 'setup.json'
  if st.is_file():
    pk = dict(json.loads(st.read_text()).get('packages') or {})
    if pk.get('alphafold3'):
      pk['alphafold3'] = re.sub(r'\.d\d{8}$', '', pk['alphafold3'])
    out['packages'] = json.dumps(pk, sort_keys=True)
  pf = d / 'pip_freeze.txt'
  if pf.is_file():
    out['pip_freeze_sha256'] = hashlib.sha256(pf.read_bytes()).hexdigest()
  return out


def r1_sessions(c, args):
  src, pat = R1_SESSIONS[c]
  if src == 'fu14':
    text = (pathlib.Path(args.fu14_sessions)).read_text()
    names = [l.split('#')[0].strip() for l in text.splitlines()]
  else:
    names = (pathlib.Path(args.main).parent / 'sessions_main.txt').read_text().split()
  return [n for n in names if n and re.search(pat, n)]


def check_stack(d, args, root=REPO / 'results' / 'af3'):
  """Check 4b: every r2 session's stack against the set of its arm's r1 stacks, field by field."""
  rows, table = [], []
  for c, g in d.groupby('condition'):
    r1 = [stack_of(root / s) for s in r1_sessions(c, args)]
    sessions = sorted({re.search(r'(\d{8}T\d{6}Z_[^/]+)', str(s)).group(1) for s in g.session})
    diffs = {}
    for s in sessions:
      st = stack_of(root / s)
      for k, v in st.items():
        ref = sorted({str(x[k]) for x in r1})
        same = str(v) in ref
        table.append({'condition': c, 'session': s, 'field': k, 'r2': v, 'r1': ' | '.join(ref), 'same': same})
        if not same:
          diffs.setdefault(k, set()).add(f'{v} (r1 {" | ".join(ref)})')
    rows.append({'check': '4b stack of every r2 session against r1', 'condition': c,
                 'value': f'{len(sessions)} r2 sessions against {len(r1)} r1 sessions',
                 'detail': 'pass' if not diffs else 'reported, differs in ' + '; '.join(
                     f'{k}: {", ".join(sorted(v))}' for k, v in sorted(diffs.items()))[:400]})
  return rows, pd.DataFrame(table)


# ---------------------------------------------------------------- identity

def light_identical(dir_a, dir_b, seed):
  """Byte comparison of the files the light fetch brings (LIGHT_TARGET, LIGHT_SAMPLE), the mmCIF timestamp masked."""
  a, b = pathlib.Path(dir_a), pathlib.Path(dir_b)
  t = a.name
  names = [f'{t}{s}' for s in LIGHT_TARGET] + [f'seed-{seed}_sample-{k}/{t}_seed-{seed}_sample-{k}{s}'
                                                for k in range(5) for s in LIGHT_SAMPLE]
  missing = [n for n in names if not (a / n).is_file() or not (b / n).is_file()]
  differ = [n for n in names if n not in missing and contrasts.masked(a / n) != contrasts.masked(b / n)]
  return {'status': 'differ' if differ else 'missing' if missing else 'identical', 'differ': differ, 'missing': missing}


def identity(ok, c, strict):
  check = f'5 {c} r2 against {c} r1 ({"every output file" if strict else "the files of the light fetch"})'
  if not ((ok.condition == c) & (ok.repetition == 'r2')).any():
    return {'check': check, 'condition': c, 'value': 'not run', 'detail': 'not run'}, []
  if strict:
    s, det = fu13.identity_rows(ok, ok, c, c, 'r2', 'r1', SEEDS, check)
  else:
    det = []
    for (t, sd), g in ok[(ok.condition == c) & (ok.repetition == 'r2')].groupby(['target', 'seed']):
      r = ok[(ok.condition == c) & (ok.repetition == 'r1') & (ok.target == t) & (ok.seed == sd)]
      res = light_identical(contrasts.output_dir(r), contrasts.output_dir(g), sd) if len(r) else \
          {'status': 'missing', 'differ': [], 'missing': ['no reference run']}
      det.append({'check': check, 'target': t, 'seed': sd, 'status': res['status'],
                  'files': '; '.join(res['differ'] + res['missing'])[:300]})
    s = {'check': check, 'condition': f'{c} r2 against {c} r1'}
  s['detail'] = identity_reading(det)
  s['value'] = (f'{sum(x["status"] == "identical" for x in det)}/{len(det)} identical, '
                f'{sum(x["status"] == "differ" for x in det)} differ, {sum(x["status"] == "missing" for x in det)} lack files')
  return s, det


def identity_reading(det):
  """Item e 1: bit-identical if every (target, seed) is identical; not bit-identical if any differs; otherwise (files
  missing on a side, none differing) not determined."""
  n = len(det)
  n_diff = sum(x['status'] == 'differ' for x in det)
  n_miss = sum(x['status'] == 'missing' for x in det)
  if n and n_diff == 0 and n_miss == 0:
    return 'recompilation bit-identical'
  if n_diff:
    return f'not bit-identical ({n_diff} of {n} (target, seed) differ)'
  return f'not determined ({n_miss} of {n} (target, seed) lack files)' if n else 'no unit'


# ---------------------------------------------------------------- D_F sensitivity

def d_f_partial(hwf, rcf, hw, rc, stat, rng_seed, reps):
  """D_F on the targets common to HW(Hf,Af), HW(H,A) and RC(H), with RC(Hf)'s rate over the drawn targets that have
  it (0 in a replicate that draws none of them); the same draws as fu13.d_f on that target list."""
  targets = sorted(set(hwf.target) & set(hw.target) & set(rc.target))
  if not targets:
    return {'n_targets': 0, 'estimate': np.nan, 'ci_low': np.nan, 'ci_high': np.nan, 'replicates_without_rc_f': 0}
  sums = [fu13.unit_sums(x.assign(contrast='x'), 'x', stat, targets) for x in (hwf, rcf, hw, rc)]
  empty = [0]

  def value(idx):
    m = []
    for i, (s, n) in enumerate(sums):
      if n[idx].sum() == 0:
        if i != 1:
          raise ValueError('a term without units')
        empty[0] += 1
        m.append(0.0)
      else:
        m.append(s[idx].sum() / n[idx].sum())
    return (m[0] - m[1]) - (m[2] - m[3]), m
  est, parts = value(np.arange(len(targets)))
  empty[0] = 0
  rng = np.random.default_rng(rng_seed)
  vals = [value(idx)[0] for idx in rng.integers(0, len(targets), size=(reps, len(targets)))]
  lo, hi = np.percentile(vals, [2.5, 97.5])
  return {'n_targets': len(targets), 'estimate': float(est), 'ci_low': float(lo), 'ci_high': float(hi),
          'hw_f': parts[0], 'rc_f': parts[1], 'hw': parts[2], 'rc': parts[3], 'replicates_without_rc_f': empty[0]}


def d_f_sensitivity(sp, f14_sp, h_sp, pilot, args):
  hwf = f14_sp[f14_sp.contrast == 'HW(Hf,Af)']
  hw, rc = h_sp[h_sp.contrast == 'HW(H,A)'], h_sp[h_sp.contrast == 'RC(H)']
  rcf = sp[sp.contrast == 'RC(Hf)']
  rows = []
  if rcf.empty:
    return pd.DataFrame(rows)
  pil = lambda x: x[x.target.isin(pilot)]  # noqa: E731
  for stat in contrasts.FLIP_STATS:
    base = {'pair': 'HW(Hf,Af)', 'reference': 'HW(H,A) minus RC(H)', 'statistic': stat}
    reg = fu13.d_f(hwf, fu14.zero_like(hwf), hw, rc, stat, args.seed, args.replicates)
    rows.append({**base, 'version': 'registered: 60 targets, RC(Hf) set to 0', **reg, 'reading': fu13.f_reading(reg['ci_high'])})
    r = d_f_partial(hwf, rcf, hw, rc, stat, args.seed, args.replicates)
    rows.append({**base, 'version': '(i) 60 targets, RC(Hf) measured on the pilot targets', **r,
                 'reading': fu13.f_reading(r['ci_high'])})
    r = fu13.d_f(pil(hwf), rcf, pil(hw), pil(rc), stat, args.seed, args.replicates)
    rows.append({**base, 'version': '(ii) 10 pilot targets, RC(Hf) measured', **r, 'reading': fu13.f_reading(r['ci_high'])})
    r = fu13.d_f(pil(hwf), fu14.zero_like(rcf), pil(hw), pil(rc), stat, args.seed, args.replicates)  # same targets
    rows.append({**base, 'version': '(ii) 10 pilot targets, RC(Hf) set to 0', **r, 'reading': fu13.f_reading(r['ci_high'])})
  return pd.DataFrame(rows)


def registered_d_f_equal(dfs, f14_dir):
  f = pathlib.Path(f14_dir) / 'd_f.csv'
  if not f.is_file() or dfs.empty:
    return None
  old = pd.read_csv(f)
  old = old[old.pair == 'HW(Hf,Af)'].set_index('statistic')
  new = dfs[dfs.version.str.startswith('registered')].set_index('statistic')
  return all(abs(new.loc[s, k] - old.loc[s, k]) < 1e-12 for s in old.index for k in ('estimate', 'ci_low', 'ci_high'))


# ---------------------------------------------------------------- main

def main():
  ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
  ap.add_argument('--samples', required=True)
  ap.add_argument('--fu14', default=str(REPO / 'results' / 'analysis' / 'fu14_20261113b'))
  ap.add_argument('--fu14_sessions', default=str(REPO / 'scripts' / 'analysis_runs' / 'fu14' / 'sessions.txt'))
  ap.add_argument('--main', default=str(REPO / 'results' / 'analysis' / 'main_20261014' / 'primary_v2'))
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

  d = load_rcx(args.samples)
  chk = fu13.check_exit(d) + fu13.check_machine(d) + check_machine_r1(d)
  pd.DataFrame(chk).to_csv(out / 'checks.csv', index=False)
  if any(r['detail'] != 'pass' for r in chk):
    raise SystemExit('check 1 or 2 failed (checks.csv); nothing is compared')
  r1 = load_r1(args)
  okall = f34.sample_rows(pd.concat([d, r1], ignore_index=True))
  comp = completeness(okall, pilot, all_targets)
  s4, det4 = check_settings(d)
  s4b, stack = check_stack(d, args)
  chk += s4 + s4b
  ident, det5 = [], []
  for c in ('Af', 'Hf', 'GT'):
    s, det = identity(okall, c, strict=c != 'GT')
    ident.append(s)
    det5 += det
  chk += ident
  pd.DataFrame(chk).to_csv(out / 'checks.csv', index=False)
  pd.DataFrame(det4 + det5).to_csv(out / 'checks_detail.csv', index=False)

  specs = [(name, 'RC', (c, 'r1'), (c, 'r2'), False) for name, c in RCS]
  sp, sm, cons, counts = f34.build(okall, specs)
  con = contrasts.summarize(sp, sm, cons, args) if len(sp) else pd.DataFrame()
  reg = pd.read_csv(pathlib.Path(args.fu14) / 'contrasts.csv')
  reg = reg[reg.contrast.isin(REGISTERED)].assign(source='follow-up 14 (registered)')
  main_sp = pd.read_csv(pathlib.Path(args.main) / 'contrasts' / 'pairs_seed.csv')
  vs_g = pd.DataFrame([{'comparison': 'RC(GT) minus RC(G)', 'statistic': st,
                        **fu13.diff_two(sp[sp.contrast == 'RC(GT)'], main_sp[main_sp.contrast == 'RC(G)'], st,
                                        args.seed, args.replicates)}
                       for st in contrasts.FLIP_STATS]) if (sp.contrast == 'RC(GT)').any() else pd.DataFrame()
  f14_sp = pd.read_csv(pathlib.Path(args.fu14) / 'pairs_seed.csv')
  h_sp = pd.read_csv(pathlib.Path(args.h100) / 'pairs_seed.csv')
  dfs = d_f_sensitivity(sp, f14_sp, h_sp, pilot, args)
  same = registered_d_f_equal(dfs, args.fu14)
  outputs = {'completeness': comp, 'contrasts': con, 'identity': pd.DataFrame(ident), 'registered': reg,
             'rc_gt_vs_g': vs_g, 'd_f_sensitivity': dfs, 'stack': stack, 'pairs_seed': sp, 'pairs_sample': sm,
             'counts': counts}
  for name, df in outputs.items():
    df.to_csv(out / f'{name}.csv', index=False)
  (out / 'report.md').write_text(report(chk, comp, con, ident, reg, vs_g, dfs, same))
  print((out / 'report.md').read_text())
  return 0


def report(chk, comp, con, ident, reg, vs_g, dfs, same):
  pct, num = f34.pct, f34.num
  L = ['# Follow-up 15: recompilation of Af, Hf (part A4) and GT (part A3)', '',
       'notes/analysis_plan.md section 10, row of 2026-11-14; reported whatever it shows.', '',
       '## Checks', '', '| check | condition | value | result |', '|---|---|---|---|']
  L += [f'| {r["check"]} | {r["condition"]} | {r["value"]} | {r["detail"]} |' for r in chk]
  inc = comp[comp.status != 'complete']
  L += ['', f'Completeness: {len(comp) - len(inc)} of {len(comp)} planned (condition, r2, seed) cells complete'
        + ('' if inc.empty else '; not complete: ' + ', '.join(
            f'{r.condition} seed {r.seed} ({r.targets_with_5_samples}/{r.planned_targets}, {r.status})'
            for r in inc.itertuples())) + '.']
  L += ['', '## Bit-identity (item e 1)', '']
  L += [f'- {r["condition"]}: {r["value"]}; {r["detail"]}' for r in ident]
  L += ['', '## Recompilation (item e 1)', '', '| contrast | statistic | estimate | 95% CI | n | targets |', '|---|---|---|---|---|---|']
  if len(con):
    keep = con[con.statistic.isin(contrasts.FLIP_STATS + ['m4_ligand_rmsd_median', 'm4_ca_rmsd_global_median',
                                                          'bit_identical_fraction'])]
    for r in keep.itertuples():
      fmt = pct if r.statistic in contrasts.FLIP_STATS + ['bit_identical_fraction'] else num
      L.append(f'| {r.contrast} | {r.statistic} | {fmt(r.estimate)} | {fmt(r.ci_low)} to {fmt(r.ci_high)} | {r.n} | '
               f'{r.n_targets} |')
  L += ['', 'Registered, for comparison (follow-up 14, copied):', '', '| contrast | statistic | estimate | 95% CI | n |',
        '|---|---|---|---|---|']
  L += [f'| {r.contrast} | {r.statistic} | {pct(r.estimate)} | {pct(r.ci_low)} to {pct(r.ci_high)} | {r.n} |'
        for r in reg[reg.statistic.isin(contrasts.FLIP_STATS)].itertuples()]
  L += ['', '## RC(GT) against RC(G) (item e 3, descriptive)', '']
  if len(vs_g):
    L += ['| comparison | statistic | estimate | 95% CI | targets |', '|---|---|---|---|---|']
    L += [f'| {r.comparison} | {r.statistic} | {pct(r.estimate)} | {pct(r.ci_low)} to {pct(r.ci_high)} | {r.n_targets} |'
          for r in vs_g.itertuples()]
  else:
    L.append('Part A3 not run.')
  L += ['', '## D_F(Hf,Af) with RC(Hf) measured (item e 2, sensitivity; the registered reading stands)', '']
  if len(dfs):
    L += ['| version | statistic | D_F | 95% CI | targets | RC(Hf) term | reading under rule e1 |',
          '|---|---|---|---|---|---|---|']
    L += [f'| {r.version} | {r.statistic} | {pct(r.estimate)} | {pct(r.ci_low)} to {pct(r.ci_high)} | {r.n_targets} | '
          f'{pct(r.rc_f)} | {r.reading} |' for r in dfs.itertuples()]
    if same is not None:
      L += ['', f'The registered D_F recomputed here {"equals" if same else "DIFFERS from"} follow-up 14\'s d_f.csv.']
    top = dfs[(dfs.statistic == 'flip_top1')].set_index('version')
    v1 = '(i) 60 targets, RC(Hf) measured on the pilot targets'
    rg = 'registered: 60 targets, RC(Hf) set to 0'
    if v1 in top.index and rg in top.index:
      L += ['', f'Top-1, (i) reads "{top.loc[v1, "reading"]}", the registered D_F "{top.loc[rg, "reading"]}": '
            f'{"the same" if top.loc[v1, "reading"] == top.loc[rg, "reading"] else "different"}; replicates of (i) '
            f'without a pilot target: {int(top.loc[v1, "replicates_without_rc_f"])}.']
  else:
    L.append('RC(Hf) not measured.')
  return '\n'.join(L) + '\n'


if __name__ == '__main__':
  sys.exit(main())
