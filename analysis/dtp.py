"""Follow-up 17, the float32 trunk at the default matmul precision (label dtp; notes/analysis_plan.md section 10, row
"Follow-up experiment 17").

    analysis/.venv/bin/python analysis/dtp.py --samples <collect.py samples.csv of the dtp sessions> --out DIR

Conditions. Follow-up 17, from each process's configuration (collect.py's config column; collect.py unchanged):
tpu_xla_dtp_f32def -> V6fd (r1; r2 on the 10 pilot targets, seed 1), l4_xla_dtp_f32def -> Gfd (r1 and r2),
a100_xla_dtp_f32def -> Afd, h100_xla_dtp_f32def -> Hfd (part B, r1; optional: without its sessions its rows say "not
run"). References, from their analysis folders: V6, G (r1, r2), V6h and Gh (seeds 1 and 2) of the main study (--main,
primary_v2); V6h and Gh on seeds 3 to 5 of follow-up 12 (--prec5); V6f, Gf (r1, r2) of follow-up 13 (--fu13); A of
follow-up 3 (--a100); H of follow-up 9 (--h100); Af, Hf of follow-up 14 (--fu14). The registered seed pairs of the
three earlier cells are read from the earlier pairs_seed.csv files, not recomputed: c_bd = HW(V6,G) (main study),
c_bh = HWh(V6,G) on seeds 1 to 5 (follow-up 12), c_fh = HW(V6f,Gf) (follow-up 13).
Checks (item c), before anything is compared; the analysis stops if 1 or 2 fails:
  1 every process of the dtp sessions exits 0 or its failure is recorded (collect.py's failure column);
  2 one machine type per condition;
  3 completeness per condition, repetition and seed (missing units leave the contrasts they belong to, section 8);
  4 the settings reached every process: af3_entry_stats.json "fu13" records global_config_bfloat16 none in every
    V6fd, Gfd, Afd and Hfd process; session.json's xla_env holds NO JAX_DEFAULT_MATMUL_PRECISION; the device is the
    expected one (TPU v6 lite, NVIDIA L4, NVIDIA A100-SXM4-80GB, NVIDIA H100 80GB HBM3); 4b, every session's driver,
    kernel, Python version and Python packages (rcx.stack_of) against those of the source arm's sessions (V6f and Gf of
    follow-up 13, Af and Hf of follow-up 14), any difference listed (reported, not a stop);
  5 V6fd r2 against V6fd r1, the 10 pilot targets, seed 1, every output file byte for byte (contrasts.outputs_identical,
    the mmCIF timestamp line masked); reported; a difference is reported and RC(V6fd) is not assumed to be 0 anywhere;
  6 memory: per condition and bucket, the largest compiled device memory against the device limit; every
    out-of-memory stop listed.
Statistics (item d): section 5's target-clustered bootstrap (--replicates 10000, --seed 2026), per-seed top-1, seeds 1
to 5, every term of a difference on the same resampled targets.
  Primary: c_fd = HW(V6fd,Gfd) and D_T = [HW(V6fd,Gfd) - RC(Gfd)] - [HW(V6,G) - RC(G)] (fu13.d_f, RC(Gfd) measured).
  Secondary, each with its interval: T_d = c_fd - c_bd, P_f = c_fh - c_fd (fu13.diff_two); I = (c_fh - c_fd) -
  (c_bh - c_bd) (fu13.d_f with the four cells as its four terms: one set of draws); the three flip statistics of the four
  cells; RC(Gfd) beside RC(G) and RC(Gf), RC(Gfd) - RC(G); Fd(V6) = V6 against V6fd, Fd(G) = G against Gfd, Pf(V6) =
  V6fd against V6f, Pf(G) = Gfd against Gf; the per-seed top-1 success rate (M1) of V6fd, Gfd, V6, G, V6f, Gf;
  AlphaFold 3's main output of each target (winners.main_output: the highest ranking score among the 25 samples of
  seeds 1 to 5) in each of the four cells: on how many targets it differs between v6e and the L4 and on how many its
  success differs (descriptive counts).
  Part B: the six pairs of {V6fd, Gfd, Afd, Hfd}; D_T for (Hfd,Afd) against HW(H,A) - RC(H) (follow-up 9) with RC(Hfd)
  set to 0, and for (Gfd,Afd) against HW(G,A) - RC(G) (follow-up 3) with RC(Gfd) measured (the recompilation term of
  the first-named condition, as follow-up 14 used RC(Gf) for (Gf,Af)); c_fd minus c_fh for (Hfd,Afd) against HW(Hf,Af)
  and (Gfd,Afd) against HW(Gf,Af) (follow-up 14).
Reading rules (item e): (1) reduced if the 95% interval of D_T lies below 0 (top-1 primary; success flip and flips
that matter secondary); (2) agree if the upper limit of the 95% interval of c_fd is below 5.0 points, "agreement not
shown" otherwise; (3) attribution, for the top-1 flip rate, in this order: "float32 activations suffice" if c_fd
agrees; otherwise "both are needed" if D_T is not reduced; otherwise "both contribute" (T_d, P_f and I descriptive in
every case); (4) recompilation, descriptive: RC(Gfd) with its interval and bit-identity count (the light-fetch files,
the mmCIF timestamp masked; rcx.identity), and check 5; (5) part B: rule 2 for each of the six pairs; "the float32
trunk at the default matmul precision makes the four accelerators agree" only if all six agree (intersection-union,
no multiplicity correction); rule 1 for (Hfd,Afd) and (Gfd,Afd); (6) cost, descriptive: ratios of the median process
wall time, compiled device memory and (TPU) the allocator's peak reservation, overall and per bucket, r1 processes:
V6fd/V6, V6fd/V6f, Gfd/G, Gfd/Gf, Afd/A, Afd/Af, Hfd/H, Hfd/Hf.
Outputs in --out: checks.csv, checks_detail.csv, stack.csv, completeness.csv, contrasts.csv, cells.csv, d_t.csv,
decomposition.csv, attribution.csv, agree.csv, part_b.csv, main_output.csv, success.csv, cost.csv, memory.csv,
failures.csv, pairs_seed.csv, pairs_sample.csv, counts.csv, report.md.
"""
import argparse
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
import rcx  # noqa: E402
import winners  # noqa: E402

THRESHOLD = 0.05                      # rule e2, follow-up 14's threshold, unchanged
SEEDS = [1, 2, 3, 4, 5]
CONFIG = {'tpu_xla_dtp_f32def': 'V6fd', 'l4_xla_dtp_f32def': 'Gfd', 'a100_xla_dtp_f32def': 'Afd',
          'h100_xla_dtp_f32def': 'Hfd'}
DEVICE = {'V6fd': 'TPU v6 lite', 'Gfd': 'NVIDIA L4', 'Afd': 'NVIDIA A100-SXM4-80GB', 'Hfd': 'NVIDIA H100 80GB HBM3'}
PLANNED = {'V6fd': {**{('r1', s): 60 for s in SEEDS}, ('r2', 1): 10},
           'Gfd': {(r, s): 60 for r in ('r1', 'r2') for s in SEEDS},
           'Afd': {('r1', s): 60 for s in SEEDS}, 'Hfd': {('r1', s): 60 for s in SEEDS}}
PART_B = ('Afd', 'Hfd')
# the four cells of the pair (V6,G): name -> (contrast, source folder of its seed pairs)
CELLS = {'c_bd': ('HW(V6,G)', 'main'), 'c_bh': ('HWh(V6,G)', 'prec5'), 'c_fd': ('HW(V6fd,Gfd)', 'here'),
         'c_fh': ('HW(V6f,Gf)', 'fu13')}
CELL_TEXT = {'c_bd': 'bfloat16, default precision (main study)', 'c_bh': 'bfloat16, highest (follow-up 12)',
             'c_fd': 'float32, default precision (new)', 'c_fh': 'float32, highest (follow-up 13)'}
# sides of each cell for the main output: (condition, repetition) of the v6e side and of the L4 side
CELL_SIDES = {'c_bd': ('V6', 'G'), 'c_bh': ('V6h', 'Gh'), 'c_fd': ('V6fd', 'Gfd'), 'c_fh': ('V6f', 'Gf')}
B_PAIRS = [('V6fd', 'Gfd'), ('Hfd', 'Afd'), ('Gfd', 'Afd'), ('V6fd', 'Afd'), ('Hfd', 'V6fd'), ('Hfd', 'Gfd')]
# part B D_T: float32-default pair -> (bfloat16 pair and folder, its RC and folder, the float32-default RC or None)
B_DT = {('Hfd', 'Afd'): (('HW(H,A)', 'h100'), ('RC(H)', 'h100'), None),
        ('Gfd', 'Afd'): (('HW(G,A)', 'a100'), ('RC(G)', 'main'), 'RC(Gfd)')}
B_FH = {('Hfd', 'Afd'): 'HW(Hf,Af)', ('Gfd', 'Afd'): 'HW(Gf,Af)'}
# source arm of each new condition for check 4b: (sessions file, pattern of the session name)
SOURCE_SESSIONS = {'V6fd': ('scripts/analysis_runs/fu13/sessions.txt', r'_v6e_why_v6f_'),
                   'Gfd': ('scripts/analysis_runs/fu13/sessions.txt', r'_l4_why_gf_'),
                   'Afd': ('scripts/analysis_runs/fu14/sessions.txt', r'_a100_f32_af_'),
                   'Hfd': ('scripts/analysis_runs/fu14/sessions.txt', r'_h100_f32_hf_')}
COST = [('V6fd', 'V6'), ('V6fd', 'V6f'), ('Gfd', 'G'), ('Gfd', 'Gf'), ('Afd', 'A'), ('Afd', 'Af'), ('Hfd', 'H'),
        ('Hfd', 'Hf')]


def hw(p, q):
  return f'HW({p},{q})'


# ---------------------------------------------------------------- loading

def load_dtp(path):
  d = pd.read_csv(path, low_memory=False)
  d = d[(d.label == 'dtp') & (d.num_recycles == 10) & (d.num_diffusion_samples == 5)].copy()
  d['condition'] = d.config.map(CONFIG)
  if d.condition.isna().any():
    raise SystemExit(f'sessions of label dtp with configurations not of follow-up 17: '
                     f'{sorted(set(d[d.condition.isna()].config))}')
  return d


def load_reference(args):
  """V6, G (r1, r2), V6h, Gh (seeds 1 to 5), V6f, Gf (r1, r2), A, H, Af, Hf (r1)."""
  m = f34.load_samples(pathlib.Path(args.main) / 'samples.csv', ['main'])
  m = m[m.condition.isin(['V6', 'G', 'V6h', 'Gh']) & m.repetition.isin(['r1', 'r2'])]
  p = f34.load_samples(pathlib.Path(args.prec5) / 'samples.csv', ['prec5'])
  p = p[p.condition.isin(['V6h', 'Gh']) & (p.repetition == 'r1') & p.seed.isin([3, 4, 5])]
  f13 = fu14.load_fu13(pathlib.Path(args.fu13) / 'samples.csv')
  a = f34.load_samples(pathlib.Path(args.a100) / 'samples.csv', ['a100'])
  a = a[(a.condition == 'A') & (a.repetition == 'r1')]
  h = f34.load_samples(pathlib.Path(args.h100) / 'samples.csv', ['h100'])
  h = h[(h.condition == 'H') & (h.repetition == 'r1')]
  f14 = fu14.load_f32(pathlib.Path(args.fu14) / 'samples.csv')
  f14 = f14[f14.condition.isin(['Af', 'Hf']) & (f14.repetition == 'r1')]
  return pd.concat([m, p, f13, a, h, f14], ignore_index=True)


# ---------------------------------------------------------------- checks

def completeness(ok):
  rows = []
  for c, plan in PLANNED.items():
    for (rep, seed), n in sorted(plan.items()):
      units = ok[(ok.condition == c) & (ok.repetition == rep) & (ok.seed == seed)].groupby('target').size()
      have = int((units == 5).sum())
      rows.append({'condition': c, 'repetition': rep, 'seed': seed, 'planned_targets': n,
                   'targets_with_5_samples': have, 'part': 'B' if c in PART_B else 'A',
                   'status': 'complete' if have == n else ('not run' if have == 0 else 'incomplete')})
  return pd.DataFrame(rows)


def check_settings(d, root=REPO):
  """Check 4: bfloat16 none, NO JAX_DEFAULT_MATMUL_PRECISION in xla_env, the expected device, in every process."""
  rows, detail = [], []
  for c, g in fu14.processes(d).groupby('condition'):
    bad = []
    for r in g.to_dict('records'):
      bf16, env = fu14.settings_of(r, root)
      ok = bf16 == 'none' and 'JAX_DEFAULT_MATMUL_PRECISION' not in env and r.get('device_kind') == DEVICE[c]
      if not ok:
        bad.append(f'{r["target"]} {r["repetition"]} seed {r["seed"]} (bfloat16 {bf16}, JAX_DEFAULT_MATMUL_PRECISION '
                   f'{env.get("JAX_DEFAULT_MATMUL_PRECISION", "absent")}, device {r.get("device_kind")})')
    detail += [{'check': '4 settings reached every process', 'condition': c, 'process': b} for b in bad]
    rows.append({'check': '4 settings reached every process', 'condition': c,
                 'value': f'{len(g) - len(bad)}/{len(g)} processes',
                 'detail': 'pass' if not bad else 'reported: ' + '; '.join(bad)[:250]})
  return rows, detail


def source_sessions(c, root=REPO):
  path, pat = SOURCE_SESSIONS[c]
  f = root / path
  if not f.is_file():
    return []
  names = [line.split('#')[0].strip() for line in f.read_text().splitlines()]
  return [n for n in names if n and re.search(pat, n)]


def check_stack(d, sessions_of=source_sessions, af3=REPO / 'results' / 'af3'):
  """Check 4b: every new session's stack against the set of its source arm's stacks, field by field."""
  rows, table = [], []
  for c, g in d.groupby('condition'):
    ref = [rcx.stack_of(af3 / s) for s in sessions_of(c)]
    sessions = sorted({re.search(r'(\d{8}T\d{6}Z_[^/]+)', str(s)).group(1) for s in g.session})
    diffs = {}
    for s in sessions:
      st = rcx.stack_of(af3 / s)
      for k, v in st.items():
        known = sorted({str(x[k]) for x in ref})
        same = str(v) in known
        table.append({'condition': c, 'session': s, 'field': k, 'new': v, 'source': ' | '.join(known), 'same': same})
        if not same:
          diffs.setdefault(k, set()).add(f'{v} (source {" | ".join(known)})')
    rows.append({'check': '4b stack of every session against its source arm', 'condition': c,
                 'value': f'{len(sessions)} sessions against {len(ref)} source sessions',
                 'detail': 'pass' if not diffs else 'reported, differs in ' + '; '.join(
                     f'{k}: {", ".join(sorted(v))}' for k, v in sorted(diffs.items()))[:400]})
  return rows, pd.DataFrame(table)


# ---------------------------------------------------------------- statistics

def specs():
  s = [(hw(p, q), 'HW', (p, 'r1'), (q, 'r1'), (p, q) == ('V6fd', 'Gfd')) for p, q in B_PAIRS]
  s += [('RC(Gfd)', 'RC', ('Gfd', 'r1'), ('Gfd', 'r2'), False),
        ('RC(V6fd)', 'RC', ('V6fd', 'r1'), ('V6fd', 'r2'), False),
        ('Fd(V6)', 'F', ('V6', 'r1'), ('V6fd', 'r1'), False), ('Fd(G)', 'F', ('G', 'r1'), ('Gfd', 'r1'), False),
        ('Pf(V6)', 'F', ('V6fd', 'r1'), ('V6f', 'r1'), False), ('Pf(G)', 'F', ('Gfd', 'r1'), ('Gf', 'r1'), False)]
  return s


def cell_pairs(sp, ref_sp):
  """{cell: seed-pair frame} of the four cells, seeds 1 to 5."""
  out = {}
  for cell, (name, src) in CELLS.items():
    f = sp if src == 'here' else ref_sp[src]
    out[cell] = f[(f.contrast == name) & f.seed_a.isin(SEEDS)]
  return out


def rate(frame, stat, args):
  """Mean of a seed-pair statistic with its target-clustered 95% interval (contrasts.boot_ci)."""
  targets = sorted(frame.target.unique())
  if not targets:
    return {'n': 0, 'n_targets': 0, 'estimate': np.nan, 'ci_low': np.nan, 'ci_high': np.nan}
  groups = [frame[frame.target == t][stat].astype(float).values for t in targets]
  lo, hi = contrasts.boot_ci(groups, contrasts.mean, np.random.default_rng(args.seed), args.replicates)
  return {'n': len(frame), 'n_targets': len(targets), 'estimate': contrasts.mean(frame[stat].astype(float).values),
          'ci_low': lo, 'ci_high': hi}


def cells_table(cp, args):
  return pd.DataFrame([{'cell': c, 'what': CELL_TEXT[c], 'contrast': CELLS[c][0], 'statistic': st, **rate(cp[c], st, args)}
                       for c in CELLS for st in contrasts.FLIP_STATS])


def agree_reading(ci_high):
  if ci_high != ci_high:
    return 'not computed'
  return 'agree' if ci_high < THRESHOLD else 'agreement not shown'


def d_t_table(sp, ref_sp, args):
  """Primary D_T of (V6fd,Gfd) against HW(V6,G) - RC(G), RC(Gfd) measured."""
  rows = []
  hwf, rcf = sp[sp.contrast == 'HW(V6fd,Gfd)'], sp[sp.contrast == 'RC(Gfd)']
  m = ref_sp['main']
  for i, st in enumerate(contrasts.FLIP_STATS):
    r = fu13.d_f(hwf, rcf, m[m.contrast == 'HW(V6,G)'], m[m.contrast == 'RC(G)'], st, args.seed, args.replicates)
    rows.append({'pair': 'HW(V6fd,Gfd)', 'reference': 'HW(V6,G) minus RC(G)', 'float32_recompilation': 'RC(Gfd)',
                 'statistic': st, 'role': 'primary' if i == 0 else 'secondary', **r,
                 'reading': fu13.f_reading(r['ci_high'])})
  return pd.DataFrame(rows)


def decomposition(cp, args):
  """T_d = c_fd - c_bd, P_f = c_fh - c_fd, I = (c_fh - c_fd) - (c_bh - c_bd); descriptive, with intervals."""
  rows = []
  for st in contrasts.FLIP_STATS:
    rows.append({'term': 'T_d', 'definition': 'c_fd minus c_bd (data type at the default precision)', 'statistic': st,
                 **fu13.diff_two(cp['c_fd'], cp['c_bd'], st, args.seed, args.replicates)})
    rows.append({'term': 'P_f', 'definition': 'c_fh minus c_fd (precision in float32)', 'statistic': st,
                 **fu13.diff_two(cp['c_fh'], cp['c_fd'], st, args.seed, args.replicates)})
    r = fu13.d_f(cp['c_fh'], cp['c_fd'], cp['c_bh'], cp['c_bd'], st, args.seed, args.replicates)
    rows.append({'term': 'I', 'definition': '(c_fh minus c_fd) minus (c_bh minus c_bd) (interaction)', 'statistic': st,
                 **{k: r[k] for k in ('n_targets', 'estimate', 'ci_low', 'ci_high')}})
  return pd.DataFrame(rows)


def attribution(c_fd_reading, d_t_reading):
  """Rule e3, in the order of the row: agree first, then D_T."""
  if c_fd_reading == 'agree':
    s = 'float32 activations suffice'
    if d_t_reading == 'not reduced':
      s += ' (note: D_T reads "not reduced" although c_fd agrees; both readings are reported)'
    return s
  if c_fd_reading == 'not computed' or d_t_reading == 'not computed':
    return 'not computed'
  return 'both are needed' if d_t_reading == 'not reduced' else 'both contribute'


def rc_rows(sp, ref_sp, args):
  rows = []
  m, f13 = ref_sp['main'], ref_sp['fu13']
  for st in contrasts.FLIP_STATS:
    for name, f in (('RC(Gfd)', sp[sp.contrast == 'RC(Gfd)']), ('RC(G)', m[m.contrast == 'RC(G)']),
                    ('RC(Gf)', f13[f13.contrast == 'RC(Gf)'])):
      rows.append({'comparison': name, 'statistic': st, **rate(f, st, args)})
    rows.append({'comparison': 'RC(Gfd) minus RC(G)', 'statistic': st,
                 **fu13.diff_two(sp[sp.contrast == 'RC(Gfd)'], m[m.contrast == 'RC(G)'], st, args.seed, args.replicates)})
  return pd.DataFrame(rows)


def agree_table(sp, args):
  rows = []
  for p, q in B_PAIRS:
    r = rate(sp[sp.contrast == hw(p, q)], 'flip_top1', args)
    rows.append({'pair': hw(p, q), 'role': 'primary (c_fd)' if (p, q) == ('V6fd', 'Gfd') else 'part B',
                 'n': r['n'], 'flip_top1': r['estimate'], 'ci_low': r['ci_low'], 'ci_high': r['ci_high'],
                 'threshold': THRESHOLD, 'reading': agree_reading(r['ci_high']) if r['n'] else 'not computed'})
  return pd.DataFrame(rows)


def generality(agree):
  """Rule e5: the sentence only if all six pairs agree."""
  if (agree.reading == 'not computed').any():
    return 'not computed (part B not run or incomplete: ' + ', '.join(agree[agree.reading == 'not computed'].pair) + ')'
  if (agree.reading == 'agree').all():
    return 'the float32 trunk at the default matmul precision makes the four accelerators agree (all six pairs agree)'
  return ('not shown for the four accelerators: agree ' + (', '.join(agree[agree.reading == 'agree'].pair) or 'none')
          + '; agreement not shown ' + ', '.join(agree[agree.reading != 'agree'].pair))


def part_b_table(sp, ref_sp, args):
  rows = []
  for pair, ((hw_name, hw_src), (rc_name, rc_src), rcf_name) in B_DT.items():
    hwf = sp[sp.contrast == hw(*pair)]
    if hwf.empty:
      continue
    rcf = sp[sp.contrast == rcf_name] if rcf_name else fu14.zero_like(hwf)
    for i, st in enumerate(contrasts.FLIP_STATS):
      r = fu13.d_f(hwf, rcf, ref_sp[hw_src][ref_sp[hw_src].contrast == hw_name],
                   ref_sp[rc_src][ref_sp[rc_src].contrast == rc_name], st, args.seed, args.replicates)
      rows.append({'comparison': f'D_T of {hw(*pair)}', 'reference': f'{hw_name} minus {rc_name}',
                   'float32_recompilation': rcf_name or 'set to 0 (not measured; D_T an upper bound)', 'statistic': st,
                   'role': 'rule e1' if i == 0 else 'secondary', **{k: r[k] for k in ('n_targets', 'estimate', 'ci_low',
                                                                                      'ci_high')},
                   'reading': fu13.f_reading(r['ci_high'])})
    f14 = ref_sp['fu14']
    for st in contrasts.FLIP_STATS:
      rows.append({'comparison': f'{hw(*pair)} minus {B_FH[pair]}', 'reference': 'follow-up 14 (highest)',
                   'float32_recompilation': '', 'statistic': st, 'role': 'descriptive',
                   **fu13.diff_two(hwf, f14[f14.contrast == B_FH[pair]], st, args.seed, args.replicates),
                   'reading': ''})
  return pd.DataFrame(rows)


def main_output_table(okall):
  """Per cell: targets whose main output (winners.main_output over seeds 1 to 5) differs between v6e and the L4, and
  targets whose main output's success differs (descriptive)."""
  rows, detail = [], []
  for cell, (a, b) in CELL_SIDES.items():
    ga = okall[(okall.condition == a) & (okall.repetition == 'r1') & okall.seed.isin(SEEDS)]
    gb = okall[(okall.condition == b) & (okall.repetition == 'r1') & okall.seed.isin(SEEDS)]
    n = n_diff = n_succ = 0
    for t in sorted(set(ga.target) & set(gb.target)):
      xa, xb = ga[ga.target == t], gb[gb.target == t]
      seeds = sorted(set(xa.seed) & set(xb.seed))
      if seeds != SEEDS:
        continue
      ma, mb = winners.main_output(xa), winners.main_output(xb)
      changed = (int(ma.seed), int(ma['sample'])) != (int(mb.seed), int(mb['sample']))
      sa, sb = bool(ma.m1_success == True), bool(mb.m1_success == True)  # noqa: E712 - NaN counts as no success
      n += 1
      n_diff += changed
      n_succ += sa != sb
      detail.append({'cell': cell, 'target': t, 'main_a': f'{int(ma.seed)}/{int(ma["sample"])}',
                     'main_b': f'{int(mb.seed)}/{int(mb["sample"])}', 'changed': changed, 'success_a': sa,
                     'success_b': sb})
    rows.append({'cell': cell, 'what': CELL_TEXT[cell], 'sides': f'{a} against {b}', 'targets': n,
                 'main_output_differs': n_diff, 'main_output_success_differs': n_succ})
  return pd.DataFrame(rows), pd.DataFrame(detail)


# ---------------------------------------------------------------- main

def main():
  ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
  ap.add_argument('--samples', required=True)
  ap.add_argument('--main', default=str(REPO / 'results' / 'analysis' / 'main_20261014' / 'primary_v2'))
  ap.add_argument('--prec5', default=str(REPO / 'results' / 'analysis' / 'prec5_20261107'))
  ap.add_argument('--fu13', default=str(REPO / 'results' / 'analysis' / 'fu13_20261112'))
  ap.add_argument('--fu14', default=str(REPO / 'results' / 'analysis' / 'fu14_20261113b'))
  ap.add_argument('--a100', default=str(REPO / 'results' / 'analysis' / 'a100_20261018'))
  ap.add_argument('--h100', default=str(REPO / 'results' / 'analysis' / 'h100_20261104'))
  ap.add_argument('--out', required=True)
  ap.add_argument('--replicates', type=int, default=10000)
  ap.add_argument('--seed', type=int, default=2026)
  args = ap.parse_args()
  out = pathlib.Path(args.out)
  out.mkdir(parents=True, exist_ok=True)

  d = load_dtp(args.samples)
  chk = fu13.check_exit(d) + fu13.check_machine(d)
  pd.DataFrame(chk).to_csv(out / 'checks.csv', index=False)
  if any(r['detail'] != 'pass' for r in chk):
    raise SystemExit('check 1 or 2 failed (checks.csv); nothing is compared')
  ok = f34.sample_rows(d)
  comp = completeness(ok)
  s4, det4 = check_settings(d)
  s4b, stack = check_stack(d)
  s5, det5 = fu13.identity_rows(ok, ok, 'V6fd', 'V6fd', 'r2', 'r1', [1], '5 V6fd r2 against V6fd r1 (every file)')
  ref = load_reference(args)
  okall = f34.sample_rows(pd.concat([d, ref], ignore_index=True))
  s_id, det_id = rcx.identity(okall, 'Gfd', strict=False)
  s_id['check'] = 'rule e4: Gfd r2 against Gfd r1 (the files of the light fetch)'
  mem = fu14.memory(fu14.processes(d))
  chk += s4 + s4b + [s5, s_id, {
      'check': '6 memory', 'condition': ', '.join(sorted(d.condition.unique())),
      'value': f'largest {mem.fraction_of_limit.max():.0%} of the device limit' if len(mem) else 'no process',
      'detail': 'pass' if len(mem) and (mem.failed == 0).all() else 'reported (see memory.csv, failures.csv)'}]
  pd.DataFrame(chk).to_csv(out / 'checks.csv', index=False)
  pd.DataFrame(det4 + det5 + det_id).to_csv(out / 'checks_detail.csv', index=False)

  sp, sm, cons, counts = f34.build(okall, specs())
  con = contrasts.summarize(sp, sm, cons, args) if len(sp) else pd.DataFrame()
  ref_sp = {'main': pd.read_csv(pathlib.Path(args.main) / 'contrasts' / 'pairs_seed.csv'),
            'prec5': pd.read_csv(pathlib.Path(args.prec5) / 'pairs_seed.csv'),
            'fu13': pd.read_csv(pathlib.Path(args.fu13) / 'pairs_seed.csv'),
            'fu14': pd.read_csv(pathlib.Path(args.fu14) / 'pairs_seed.csv'),
            'a100': pd.read_csv(pathlib.Path(args.a100) / 'pairs_seed.csv'),
            'h100': pd.read_csv(pathlib.Path(args.h100) / 'pairs_seed.csv')}
  cp = cell_pairs(sp, ref_sp)
  cells = cells_table(cp, args)
  dtt = d_t_table(sp, ref_sp, args)
  dec = decomposition(cp, args)
  agree = agree_table(sp, args)
  top = dtt[dtt.statistic == 'flip_top1'].iloc[0]
  c_fd_reading = agree[agree.pair == 'HW(V6fd,Gfd)'].reading.iloc[0]
  attr = pd.DataFrame([{'c_fd_reading': c_fd_reading, 'd_t_reading': top.reading,
                        'attribution': attribution(c_fd_reading, top.reading), 'generality': generality(agree)}])
  rcs = rc_rows(sp, ref_sp, args)
  pb = part_b_table(sp, ref_sp, args)
  mo, mo_detail = main_output_table(okall)
  succ = fu13.success_rates(okall, [(c, 'r1') for c in ('V6fd', 'Gfd', 'Afd', 'Hfd', 'V6', 'G', 'V6f', 'Gf')],
                            args.seed, args.replicates)
  proc = fu14.processes(pd.concat([d[d.repetition == 'r1'], ref[ref.repetition == 'r1']], ignore_index=True))
  cst = fu13.cost(proc, COST)
  fails = fu14.processes(d)[fu14.processes(d).exit_code != 0][['condition', 'repetition', 'target', 'seed', 'session',
                                                               'exit_code', 'failure']]
  outputs = {'completeness': comp, 'contrasts': con, 'cells': cells, 'd_t': dtt, 'decomposition': dec,
             'attribution': attr, 'agree': agree, 'recompilation': rcs, 'part_b': pb, 'main_output': mo,
             'main_output_detail': mo_detail, 'success': succ, 'cost': cst, 'memory': mem, 'failures': fails,
             'stack': stack, 'pairs_seed': sp, 'pairs_sample': sm, 'counts': counts}
  for name, df in outputs.items():
    df.to_csv(out / f'{name}.csv', index=False)
  (out / 'report.md').write_text(report(chk, comp, cells, dtt, dec, attr, agree, rcs, pb, mo, succ, cst, mem, fails))
  print((out / 'report.md').read_text())
  return 0


def report(chk, comp, cells, dtt, dec, attr, agree, rcs, pb, mo, succ, cst, mem, fails):
  pct, num = f34.pct, f34.num
  L = ['# Follow-up 17: the float32 trunk at the default matmul precision', '',
       'notes/analysis_plan.md section 10, row "Follow-up experiment 17"; reported whatever it shows.', '',
       '## Checks', '', '| check | condition | value | result |', '|---|---|---|---|']
  L += [f'| {r["check"]} | {r["condition"]} | {r["value"]} | {r["detail"]} |' for r in chk]
  inc = comp[comp.status != 'complete']
  L += ['', f'Completeness: {len(comp) - len(inc)} of {len(comp)} planned (condition, repetition, seed) cells complete'
        + ('' if inc.empty else '; not complete: ' + ', '.join(
            f'{r.condition} {r.repetition} seed {r.seed} ({r.targets_with_5_samples}/{r.planned_targets}, {r.status})'
            for r in inc.itertuples())) + '.']
  if len(fails):
    L += ['', 'Failed processes (recorded, not rerun): ' + '; '.join(
        f'{r.condition} {r.repetition} {r.target} seed {r.seed} ({r.failure})' for r in fails.itertuples()) + '.']
  a = attr.iloc[0]
  L += ['', '## Readings (item e)', '', f'- Rule e2, c_fd = HW(V6fd,Gfd): {a.c_fd_reading}.',
        f'- Rule e1, D_T (top-1): {a.d_t_reading}.', f'- Rule e3, attribution: **{a.attribution}**.',
        f'- Rule e5, part B: {a.generality}.', '',
        '## The four cells of the pair (V6,G)', '', '| cell | what | contrast | statistic | estimate | 95% CI | n |',
        '|---|---|---|---|---|---|---|']
  L += [f'| {r.cell} | {r.what} | {r.contrast} | {r.statistic} | {pct(r.estimate)} | {pct(r.ci_low)} to '
        f'{pct(r.ci_high)} | {r.n} |' for r in cells.itertuples()]
  L += ['', '## D_T (rule e1)', '', '| pair | reference | statistic | role | D_T | 95% CI | targets | reading |',
        '|---|---|---|---|---|---|---|---|']
  L += [f'| {r.pair} | {r.reference} | {r.statistic} | {r.role} | {pct(r.estimate)} | {pct(r.ci_low)} to '
        f'{pct(r.ci_high)} | {r.n_targets} | {r.reading} |' for r in dtt.itertuples()]
  L += ['', '## Decomposition (descriptive)', '', '| term | definition | statistic | estimate | 95% CI | targets |',
        '|---|---|---|---|---|---|']
  L += [f'| {r.term} | {r.definition} | {r.statistic} | {pct(r.estimate)} | {pct(r.ci_low)} to {pct(r.ci_high)} | '
        f'{r.n_targets} |' for r in dec.itertuples()]
  L += ['', '## Recompilation (rule e4)', '', '| comparison | statistic | estimate | 95% CI | targets |',
        '|---|---|---|---|---|']
  L += [f'| {r.comparison} | {r.statistic} | {pct(r.estimate)} | {pct(r.ci_low)} to {pct(r.ci_high)} | {r.n_targets} |'
        for r in rcs.itertuples()]
  L += ['', '## Agree (rule e2) and part B (rule e5)', '', '| pair | role | top-1 flip | 95% CI | n | reading |',
        '|---|---|---|---|---|---|']
  L += [f'| {r.pair} | {r.role} | {pct(r.flip_top1)} | {pct(r.ci_low)} to {pct(r.ci_high)} | {r.n} | {r.reading} |'
        for r in agree.itertuples()]
  if len(pb):
    L += ['', '| comparison | reference | float32 recompilation | statistic | role | estimate | 95% CI | targets | '
          'reading |', '|---|---|---|---|---|---|---|---|---|']
    L += [f'| {r.comparison} | {r.reference} | {r.float32_recompilation} | {r.statistic} | {r.role} | '
          f'{pct(r.estimate)} | {pct(r.ci_low)} to {pct(r.ci_high)} | {r.n_targets} | {r.reading} |'
          for r in pb.itertuples()]
  else:
    L += ['', 'Part B not run.']
  L += ['', '## Main output per cell (descriptive)', '',
        '| cell | what | sides | targets | main output differs | its success differs |', '|---|---|---|---|---|---|']
  L += [f'| {r.cell} | {r.what} | {r.sides} | {r.targets} | {r.main_output_differs} | {r.main_output_success_differs} |'
        for r in mo.itertuples()]
  L += ['', '| condition | top-1 success (per seed) | 95% CI | n |', '|---|---|---|---|']
  L += [f'| {r.condition} {r.repetition} | {pct(r.top1_success_rate)} | {pct(r.ci_low)} to {pct(r.ci_high)} | {r.n} |'
        for r in succ.itertuples()]
  L += ['', '## Cost and memory (rule e6, descriptive)', '',
        '| new / reference | bucket | wall ratio | compiled memory ratio | peak reserved ratio | n |',
        '|---|---|---|---|---|---|']
  L += [f'| {r.new} / {r.reference} | {r.bucket} | {num(r.wall_ratio)} | {num(r.compiled_memory_ratio)} | '
        f'{num(r.peak_reserved_ratio)} | {r.n_new}, {r.n_reference} |' for r in cst.itertuples()]
  L += ['', '| condition | bucket | processes | largest compiled GiB | device limit GiB | failed |',
        '|---|---|---|---|---|---|']
  L += [f'| {r.condition} | {r.bucket} | {r.processes} | {r.max_compiled_gib:.2f} | {r.device_limit_gib:.2f} | '
        f'{r.failed} |' for r in mem.itertuples()]
  return '\n'.join(L) + '\n'


if __name__ == '__main__':
  sys.exit(main())
