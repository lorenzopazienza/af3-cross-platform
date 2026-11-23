"""The L4 against the A100, both with AlphaFold 3's default Triton attention (section 10, row of
drafts/section10_gtat_spec_20261121.md, label gtat): a post hoc analysis of frozen outputs, specified on 2026-11-21
after every result of the study was read and before this pair was ever computed. No run and no cloud resource.

    analysis/.venv/bin/python analysis/gtat.py --main . --out results/analysis/gtat_<DATE> \\
        [--replicates 10000] [--seed 2026]

Question: with AlphaFold 3's GPU default (Triton attention) on both GPUs, does the L4 return other predictions than
the A100, beyond what recompilation already changes?
Sides: GT r1, the main study's L4 Triton arm, against AT r1, follow-up 3's A100 Triton arm; 60 targets, seeds 1 to 5;
GT is the first-named side (A), as G is in HW(G,A).
Inputs, relative to --main (per-sample tables and registered seed pairs only; no structure file is read):
  results/analysis/main_20261014/primary_v2/samples.csv            GT r1 (label main)
  results/analysis/main_20261014/primary_v2/contrasts/pairs_seed.csv  DU (G against GT; check 3) and RC(G)
  results/analysis/a100_20261018/samples.csv                       AT r1 (label a100)
  results/analysis/a100_20261018/pairs_seed.csv                    DU(A) (A against AT; check 3), HW(G,A), RC(A)
  results/analysis/a100_20261018/contrasts.csv, hw_minus_rc.csv    registered rows (check 4; copied beside)
  results/analysis/rcx_20261115/pairs_seed.csv, contrasts.csv      RC(GT) of follow-up 15 and its registered rows
  scripts/analysis_runs/a100/sessions.txt                          the AT sessions of follow-up 3
  results/freeze/followups1316_20261117/sessions.tsv               the sessions of freeze 4ec185c5
Seed pairs: for every (target, seed) both sides ran, each side's top-1 is the sample with the highest ranking score
among the seed's 5 (contrasts.top, the registered M2; a sample without M1 success, as on the three glycan targets, is
no success); flip_top1, flip_success and flip_matters as contrasts.pair_rows defines them.
Statistics (section 5), each with a 95% percentile interval from the paired bootstrap over targets (10,000 replicates,
a fresh generator from seed 2026 for every statistic), on the 60 targets and, as section 8 and the glycan row of
2026-10-14 ask for results that use M1, on the 57 targets without the three glycan targets:
  HW(GT,AT)              the three flip rates (contrasts.boot_ci, as contrasts.summarize);
  HW(GT,AT) minus RC(GT) contrasts.hw_minus_rc unchanged, RC(GT) of follow-up 15 as RC(P); RC(AT) was never
                         measured, so there is no RC(Q) row;
  HW(GT,AT) minus HW(G,A) descriptive: the same pair of GPUs with XLA attention (follow-up 3), on the same resampled
                         targets (the bootstrap of contrasts.hw_minus_rc).
Readings, fixed in the specification row before any value: RC(GT) "beyond RC(GT)" when the 95% interval of HW(GT,AT)
minus RC(GT) lies above 0, otherwise "not beyond RC(GT)"; against HW(G,A) "larger with Triton" (interval above 0),
"smaller with Triton" (below 0), otherwise "no difference shown".
Checks, in order; 1 to 4 stop the analysis if they fail:
  1 sessions: the AT rows come from exactly the AT sessions of scripts/analysis_runs/a100/sessions.txt; the GT rows
    from the main study's GT sessions; every one of them is in the freeze; one machine type per side (g2-standard-8,
    a2-ultragpu-1g), attention triton on every row, the expected device; every process exit 0;
  2 completeness: both sides hold the same 60 targets, each with seeds 1 to 5 and exactly 5 scored samples;
  3 registered top-1: the recomputed top-1 and its success equal the registered ones for every seed unit: GT against
    side B of DU (main study) and side A of RC(GT) (follow-up 15), AT against side B of DU(A) (follow-up 3);
  4 method: from their registered seed pairs, the same code reproduces the registered flip rows of HW(G,A) and RC(GT)
    and the registered HW(G,A) minus RC(G) and minus RC(A) rows within 1e-9;
  5 glycans (reported): the targets whose every sample, on both sides, has no M1 (atom O1 missing) are exactly 7OZC,
    7WDT and 7WJB.
Outputs in --out: pairs_seed.csv, rates.csv, hw_minus_rc.csv, versus_hw_g_a.csv, registered.csv, checks.csv,
inputs.tsv, report.md.
"""
import argparse
import hashlib
import pathlib
import sys
import types

import numpy as np
import pandas as pd

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / 'analysis'))
import contrasts  # noqa: E402
import followup34 as f34  # noqa: E402

AN = 'results/analysis'
MAIN = f'{AN}/main_20261014/primary_v2'
A100 = f'{AN}/a100_20261018'
RCX = f'{AN}/rcx_20261115'
A100_SESSIONS = 'scripts/analysis_runs/a100/sessions.txt'
FREEZE_SESSIONS = 'results/freeze/followups1316_20261117/sessions.tsv'
NAME, RC_NAME, REF_NAME = 'HW(GT,AT)', 'RC(GT)', 'HW(G,A)'
SIDES = {'GT': {'label': 'main', 'machine_type': 'g2-standard-8', 'device': 'NVIDIA L4', 'session_tag': '_l4_main_gt_'},
         'AT': {'label': 'a100', 'machine_type': 'a2-ultragpu-1g', 'device': 'NVIDIA A100-SXM4-80GB',
                'session_tag': '_a100_at_'}}
SEEDS = (1, 2, 3, 4, 5)
N_TARGETS = 60
GLYCAN_TARGETS = ('7OZC', '7WDT', '7WJB')
TOL = 1e-9


# ---------------------------------------------------------------- inputs

def session_name(path):
  """'results/af3/<session>/<run>' -> '<session>'."""
  parts = str(path).split('/')
  return parts[2] if len(parts) > 2 and parts[0] == 'results' and parts[1] == 'af3' else str(path)


def listed_sessions(text, tag):
  return sorted(line.strip() for line in text.splitlines()
                if line.strip() and not line.startswith('#') and tag in line)


def load_sources(root):
  root = pathlib.Path(root)
  main = f34.load_samples(root / MAIN / 'samples.csv', ['main'])
  a100 = f34.load_samples(root / A100 / 'samples.csv', ['a100'])
  sides = {'GT': main[main.condition == 'GT'], 'AT': a100[a100.condition == 'AT']}
  tables = {'main_sp': pd.read_csv(root / MAIN / 'contrasts' / 'pairs_seed.csv'),
            'a100_sp': pd.read_csv(root / A100 / 'pairs_seed.csv'),
            'a100_con': pd.read_csv(root / A100 / 'contrasts.csv'),
            'a100_hw': pd.read_csv(root / A100 / 'hw_minus_rc.csv'),
            'rcx_sp': pd.read_csv(root / RCX / 'pairs_seed.csv'),
            'rcx_con': pd.read_csv(root / RCX / 'contrasts.csv')}
  freeze = pd.read_csv(root / FREEZE_SESSIONS, sep='\t')
  lists = {'AT': listed_sessions((root / A100_SESSIONS).read_text(), SIDES['AT']['session_tag']),
           'freeze': sorted(freeze.session.astype(str))}
  inputs = [MAIN + '/samples.csv', MAIN + '/contrasts/pairs_seed.csv', A100 + '/samples.csv', A100 + '/pairs_seed.csv',
            A100 + '/contrasts.csv', A100 + '/hw_minus_rc.csv', RCX + '/pairs_seed.csv', RCX + '/contrasts.csv',
            A100_SESSIONS, FREEZE_SESSIONS]
  return sides, tables, lists, inputs


# ---------------------------------------------------------------- checks

def check_sessions(sides, lists):
  """Check 1; raises SystemExit on a failure, returns the check rows."""
  rows = []
  freeze = set(lists['freeze'])
  for c, spec in SIDES.items():
    d = sides[c]
    if d.empty:
      raise SystemExit(f'check 1: no rows of {c}')
    names = sorted(set(d.session.map(session_name)))
    if c == 'AT' and names != lists['AT']:
      raise SystemExit(f'check 1: AT sessions {names} are not the AT sessions of {A100_SESSIONS} {lists["AT"]}')
    if any(spec['session_tag'] not in n for n in names):
      raise SystemExit(f'check 1: {c}: a session without {spec["session_tag"]} in its name: {names}')
    missing = [n for n in names if n not in freeze]
    if missing:
      raise SystemExit(f'check 1: {c}: sessions not in the freeze ({FREEZE_SESSIONS}): {missing}')
    for col, want in (('label', spec['label']), ('repetition', 'r1'), ('attention', 'triton'),
                      ('machine_type', spec['machine_type']), ('device_kind', spec['device'])):
      got = sorted(d[col].astype(str).unique())
      if got != [want]:
        raise SystemExit(f'check 1: {c}: {col} is {got}, expected {want}')
    procs = d.drop_duplicates(['session', 'target', 'seed'])
    bad = procs[procs.exit_code != 0]
    if len(bad):
      raise SystemExit(f'check 1: {c}: {len(bad)} processes with a nonzero exit code')
    rows.append({'check': '1 sessions', 'side': c,
                 'value': f'pass ({len(names)} sessions, all in the freeze; {spec["machine_type"]}, {spec["device"]}, '
                          f'triton; {len(procs)} processes exit 0)'})
  return rows


def check_complete(ok):
  """Check 2; ok holds the sample rows of both sides."""
  rows, targets = [], {}
  for c in SIDES:
    g = ok[ok.condition == c]
    n = g.groupby(['target', 'seed']).agg(k=('sample', 'nunique'), rows=('sample', 'size'),
                                          scored=('ranking_score', 'count'))
    bad = n[(n.k != 5) | (n.rows != 5) | (n.scored != 5)]
    if len(bad):
      raise SystemExit(f'check 2: {c}: (target, seed) without exactly 5 scored samples: {list(bad.index[:5])}')
    seeds_of = g.groupby('target').seed.apply(lambda s: tuple(sorted(int(x) for x in s.unique())))
    if (seeds_of != SEEDS).any():
      raise SystemExit(f'check 2: {c}: targets without seeds 1 to 5: {list(seeds_of[seeds_of != SEEDS].index[:5])}')
    targets[c] = set(seeds_of.index)
  if targets['GT'] != targets['AT']:
    raise SystemExit(f'check 2: the sides hold different targets: {sorted(targets["GT"] ^ targets["AT"])[:10]}')
  if len(targets['GT']) != N_TARGETS:
    raise SystemExit(f'check 2: {len(targets["GT"])} targets, expected {N_TARGETS}')
  rows.append({'check': '2 completeness', 'side': 'GT, AT',
               'value': f'pass ({N_TARGETS} targets, seeds 1 to 5, 5 scored samples per (target, seed), both sides)'})
  return rows


def tops(ok, c):
  """{(target, seed): (top-1 sample, success)} of one side."""
  out = {}
  for (t, s), g in ok[ok.condition == c].groupby(['target', 'seed']):
    first, succ, _ = contrasts.top(g.set_index('sample').sort_index())
    out[(t, int(s))] = (int(first), bool(succ))
  return out


def check_registered_top1(top, tables):
  """Check 3: the recomputed top-1 and success of each side equal the registered seed pairs."""
  refs = [('GT', 'main_sp', 'DU', 'b'), ('GT', 'rcx_sp', RC_NAME, 'a'), ('AT', 'a100_sp', 'DU(A)', 'b')]
  rows = []
  for c, src, name, side in refs:
    p = tables[src]
    p = p[(p.contrast == name) & (p.seed_a == p.seed_b)]
    reg = {(r.target, int(r.seed_a)): (int(getattr(r, f'top1_{side}')), bool(getattr(r, f'success_{side}')))
           for r in p.itertuples()}
    if set(reg) != set(top[c]):
      raise SystemExit(f'check 3: {c}: the seed units differ from the registered pairs of {name} ({src})')
    bad = sorted(k for k in reg if reg[k] != top[c][k])
    if bad:
      raise SystemExit(f'check 3: {c}: top-1 or its success differs from side {side.upper()} of {name} on {bad[:5]}')
    rows.append({'check': '3 registered top-1', 'side': c,
                 'value': f'pass ({len(reg)} seed units equal side {side.upper()} of {name}, {src.split("_")[0]})'})
  return rows


# ---------------------------------------------------------------- statistics

def seed_pairs(top, name=NAME, a='GT', b='AT'):
  """Seed pairs with the columns of the registered pairs_seed.csv (contrasts.pair_rows), margins left out."""
  rows = []
  for (t, s) in sorted(set(top[a]) & set(top[b])):
    ta, sa = top[a][(t, s)]
    tb, sb = top[b][(t, s)]
    rows.append({'contrast': name, 'kind': 'HW', 'role': 'post hoc pair', 'target': t, 'seed_a': s, 'seed_b': s,
                 'top1_a': ta, 'top1_b': tb, 'success_a': sa, 'success_b': sb, 'flip_top1': ta != tb,
                 'flip_success': sa != sb, 'flip_matters': ta != tb and sa != sb})
  return pd.DataFrame(rows)


def flip_rates(sp, name, args):
  """The three flip rates of one contrast's seed pairs, as contrasts.summarize computes them."""
  sp = sp[sp.contrast == name]
  targets = sorted(sp.target.unique())
  rows = []
  for st in contrasts.FLIP_STATS:
    groups = [sp[sp.target == t][st].astype(float).values for t in targets]
    lo, hi = contrasts.boot_ci(groups, contrasts.mean, np.random.default_rng(args.seed), args.replicates)
    rows.append({'contrast': name, 'statistic': st, 'n': len(sp), 'n_targets': len(targets),
                 'events': int(sp[st].astype(bool).sum()),
                 'targets_with_event': int(sp[sp[st].astype(bool)].target.nunique()),
                 'estimate': contrasts.mean(sp[st].astype(float).values), 'ci_low': lo, 'ci_high': hi})
  return rows


def paired_difference(sp_x, sp_y, args):
  """Mean of x minus mean of y per flip statistic, on the targets both hold, with the bootstrap of
  contrasts.hw_minus_rc (one generator from args.seed per statistic, the same resampled targets for both terms)."""
  targets = sorted(set(sp_x.target) & set(sp_y.target))
  rows = []
  for st in contrasts.FLIP_STATS:
    xg = [sp_x[sp_x.target == t][st].astype(float).values for t in targets]
    yg = [sp_y[sp_y.target == t][st].astype(float).values for t in targets]
    est = contrasts.mean(np.concatenate(xg)) - contrasts.mean(np.concatenate(yg)) if targets else np.nan
    rng = np.random.default_rng(args.seed)
    diffs = np.array([contrasts.mean(np.concatenate([xg[i] for i in row])) -
                      contrasts.mean(np.concatenate([yg[i] for i in row]))
                      for row in rng.integers(0, len(targets), size=(args.replicates, len(targets)))]) \
        if targets else np.array([])
    rows.append({'statistic': st, 'n_targets': len(targets), 'estimate': est,
                 'ci_low': float(np.percentile(diffs, 2.5)) if len(diffs) else np.nan,
                 'ci_high': float(np.percentile(diffs, 97.5)) if len(diffs) else np.nan})
  return pd.DataFrame(rows)


def read_rc(lo):
  if lo != lo:
    return 'not computed'
  return 'beyond RC(GT)' if lo > 0 else 'not beyond RC(GT)'


def read_vs(lo, hi):
  if lo != lo or hi != hi:
    return 'not computed'
  if lo > 0:
    return 'larger with Triton'
  if hi < 0:
    return 'smaller with Triton'
  return 'no difference shown'


def check_method(tables, args):
  """Check 4: the code reproduces the registered flip rows of HW(G,A) and RC(GT), and the registered HW(G,A) minus
  RC(G) and minus RC(A) rows, from their registered seed pairs."""
  rows = []
  for name, sp, con in ((REF_NAME, tables['a100_sp'], tables['a100_con']),
                        (RC_NAME, tables['rcx_sp'], tables['rcx_con'])):
    mine = pd.DataFrame(flip_rates(sp, name, args)).set_index('statistic')
    reg = con[con.contrast == name].set_index('statistic')
    for st in contrasts.FLIP_STATS:
      for col in ('estimate', 'ci_low', 'ci_high'):
        if st not in reg.index or abs(float(mine.loc[st, col]) - float(reg.loc[st, col])) > TOL:
          raise SystemExit(f'check 4: the registered {name} {st} row is not reproduced ({col})')
  rc_g = tables['main_sp'][tables['main_sp'].contrast == 'RC(G)']
  rc_a = tables['a100_sp'][tables['a100_sp'].contrast == 'RC(A)']
  hw = tables['a100_sp'][tables['a100_sp'].contrast == REF_NAME]
  got = contrasts.hw_minus_rc(pd.concat([hw, rc_g, rc_a], ignore_index=True),
                              {REF_NAME: ('HW', False), 'RC(G)': ('RC', True), 'RC(A)': ('RC', True)}, args)
  reg = tables['a100_hw'][tables['a100_hw'].hardware == REF_NAME]
  m = got.merge(reg, on=['hardware', 'recompilation', 'reference', 'statistic'], suffixes=('', '_reg'), how='outer',
                indicator=True)
  if len(m) != 6 or (m._merge != 'both').any():
    raise SystemExit(f'check 4: the registered {REF_NAME} minus RC rows do not match in number or names')
  for col in ('estimate', 'ci_low', 'ci_high'):
    if (m[col] - m[f'{col}_reg']).abs().max() > TOL:
      raise SystemExit(f'check 4: the registered {REF_NAME} minus RC rows are not reproduced ({col})')
  rows.append({'check': '4 method', 'side': '-', 'value': f'pass (flip rows of {REF_NAME} and {RC_NAME}, 6 rows of '
               f'{REF_NAME} minus RC(G) and RC(A), within {TOL:g})'})
  return rows


def glycan_check(ok):
  """Check 5 (reported): the targets whose every sample, on both sides, has no M1 success and an M1 error naming the
  missing atom O1 (the glycan row of 2026-10-14)."""
  err = ok.m1_error.fillna('').astype(str)
  o1 = ok.assign(o1=err.str.contains('O1') & (ok.m1_success != True))  # noqa: E712
  found = sorted(t for t, g in o1.groupby('target') if g.o1.all())
  partial = sorted(set(o1[o1.o1].target) - set(found))
  value = 'as expected' if tuple(found) == GLYCAN_TARGETS and not partial else 'DIFFERS'
  return [{'check': '5 glycan targets (reported)', 'side': 'GT, AT',
           'value': f'{value}: every sample of both sides without M1 (O1 missing) for {", ".join(found) or "none"}'
                    + (f'; only some samples for {", ".join(partial)}' if partial else '')
                    + '; they count as no success (60 targets) and are left out on 57'}]


def analyse(sp, tables, args):
  rc = tables['rcx_sp'][tables['rcx_sp'].contrast == RC_NAME]
  ref = tables['a100_sp'][tables['a100_sp'].contrast == REF_NAME]
  rates, hwrc, vs = [], [], []
  for subset, drop in (('60 targets', ()), ('57 targets (no glycan)', GLYCAN_TARGETS)):
    x, r, y = (d[~d.target.isin(drop)] for d in (sp, rc, ref))
    for name, d in ((NAME, x), (RC_NAME, r), (REF_NAME, y)):
      rates += [{'targets': subset, **row} for row in flip_rates(d, name, args)]
    h = contrasts.hw_minus_rc(pd.concat([x, r], ignore_index=True), {NAME: ('HW', False), RC_NAME: ('RC', True)}, args)
    h = h[h.reference == 'RC(P)']
    for row in h.to_dict('records'):
      hwrc.append({'targets': subset, 'hardware': NAME, 'recompilation': RC_NAME, 'statistic': row['statistic'],
                   'n_targets': row['n_targets'], 'estimate': row['estimate'], 'ci_low': row['ci_low'],
                   'ci_high': row['ci_high'], 'reading': read_rc(row['ci_low'])})
    for row in paired_difference(x, y, args).to_dict('records'):
      vs.append({'targets': subset, 'difference': f'{NAME} minus {REF_NAME}', **row,
                 'reading': read_vs(row['ci_low'], row['ci_high'])})
  return pd.DataFrame(rates), pd.DataFrame(hwrc), pd.DataFrame(vs)


def registered(tables):
  """Registered rows copied beside the results (60 targets): HW(G,A), RC(A), RC(GT), and HW(G,A) minus RC."""
  rows = []
  for src, con in (('a100', tables['a100_con']), ('rcx', tables['rcx_con'])):
    for name in ((REF_NAME, 'RC(A)') if src == 'a100' else (RC_NAME,)):
      for r in con[(con.contrast == name) & con.statistic.isin(contrasts.FLIP_STATS)].itertuples():
        rows.append({'source': f'{src} (registered)', 'row': name, 'statistic': r.statistic, 'n_targets': r.n_targets,
                     'estimate': r.estimate, 'ci_low': r.ci_low, 'ci_high': r.ci_high})
  hw = tables['a100_hw']
  for r in hw[hw.hardware == REF_NAME].itertuples():
    rows.append({'source': 'a100 (registered)', 'row': f'{r.hardware} minus {r.recompilation} ({r.reference})',
                 'statistic': r.statistic, 'n_targets': r.n_targets, 'estimate': r.estimate, 'ci_low': r.ci_low,
                 'ci_high': r.ci_high})
  return pd.DataFrame(rows)


# ---------------------------------------------------------------- report

def pct(x):
  return 'n/a' if x != x else f'{100 * x:.1f}'


def pts(x):
  return 'n/a' if x != x else f'{100 * x:+.1f}'


def report(rates, hwrc, vs, reg, chk, args):
  L = ['# The L4 against the A100, both with Triton attention (label gtat)', '',
       'Post hoc analysis of frozen outputs (drafts/section10_gtat_spec_20261121.md), specified on 2026-11-21. GT r1 '
       '(main study) against AT r1 (follow-up 3), 60 targets, seeds 1 to 5. Rates in %, differences in points, 95% '
       f'percentile intervals from the paired bootstrap over targets ({args.replicates} replicates, seed {args.seed}). '
       'RC(AT) was not measured: the reference is RC(GT) (follow-up 15) only.', '',
       '## Checks', ''] + [f'- {c["check"]} ({c["side"]}): {c["value"]}' for c in chk]
  L += ['', '## Flip rates', '', '| targets | contrast | statistic | pairs | targets | events | targets with one | '
        'rate % [CI] |', '|---|---|---|---|---|---|---|---|']
  for r in rates.to_dict('records'):
    L.append(f'| {r["targets"]} | {r["contrast"]} | {r["statistic"]} | {r["n"]} | {r["n_targets"]} | {r["events"]} | '
             f'{r["targets_with_event"]} | {pct(r["estimate"])} [{pct(r["ci_low"])}, {pct(r["ci_high"])}] |')
  L += ['', f'## {NAME} minus {RC_NAME} (reading rule of the specification row)', '',
        '| targets | statistic | targets | points [CI] | reading |', '|---|---|---|---|---|']
  for r in hwrc.to_dict('records'):
    L.append(f'| {r["targets"]} | {r["statistic"]} | {r["n_targets"]} | {pts(r["estimate"])} [{pts(r["ci_low"])}, '
             f'{pts(r["ci_high"])}] | {r["reading"]} |')
  L += ['', f'## {NAME} minus {REF_NAME} (descriptive: the same GPUs with XLA attention)', '',
        '| targets | statistic | targets | points [CI] | wording |', '|---|---|---|---|---|']
  for r in vs.to_dict('records'):
    L.append(f'| {r["targets"]} | {r["statistic"]} | {r["n_targets"]} | {pts(r["estimate"])} [{pts(r["ci_low"])}, '
             f'{pts(r["ci_high"])}] | {r["reading"]} |')
  L += ['', '## Registered rows, copied (60 targets)', '', '| source | row | statistic | targets | value [CI] |',
        '|---|---|---|---|---|']
  for r in reg.to_dict('records'):
    f = pts if 'minus' in r['row'] else pct
    L.append(f'| {r["source"]} | {r["row"]} | {r["statistic"]} | {r["n_targets"]} | {f(r["estimate"])} '
             f'[{f(r["ci_low"])}, {f(r["ci_high"])}] |')
  return '\n'.join(L) + '\n'


def sha256(path):
  h = hashlib.sha256()
  with open(path, 'rb') as f:
    for chunk in iter(lambda: f.read(1 << 20), b''):
      h.update(chunk)
  return h.hexdigest()


def run(sides, tables, lists, out, args, inputs=(), root=None):
  out = pathlib.Path(out)
  out.mkdir(parents=True, exist_ok=True)
  chk = check_sessions(sides, lists)
  ok = f34.union(sides['GT'], sides['AT'])
  chk += check_complete(ok)
  top = {c: tops(ok, c) for c in SIDES}
  chk += check_registered_top1(top, tables)
  chk += check_method(tables, args)
  chk += glycan_check(ok)
  sp = seed_pairs(top)
  rates, hwrc, vs = analyse(sp, tables, args)
  reg = registered(tables)
  sp.to_csv(out / 'pairs_seed.csv', index=False)
  rates.to_csv(out / 'rates.csv', index=False)
  hwrc.to_csv(out / 'hw_minus_rc.csv', index=False)
  vs.to_csv(out / 'versus_hw_g_a.csv', index=False)
  reg.to_csv(out / 'registered.csv', index=False)
  pd.DataFrame(chk).to_csv(out / 'checks.csv', index=False)
  if root is not None:
    (out / 'inputs.tsv').write_text(''.join(f'{p}\t{sha256(pathlib.Path(root) / p)}\n' for p in inputs))
  (out / 'report.md').write_text(report(rates, hwrc, vs, reg, chk, args))
  return sp, rates, hwrc, vs, chk


def main():
  ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
  ap.add_argument('--main', default='.', help='the checkout that holds results/')
  ap.add_argument('--out', required=True)
  ap.add_argument('--replicates', type=int, default=10000)
  ap.add_argument('--seed', type=int, default=2026)
  args = ap.parse_args()
  out = pathlib.Path(args.out)
  if (out / 'hw_minus_rc.csv').exists():
    raise SystemExit(f'{out}/hw_minus_rc.csv exists; the output folder must be new')
  sides, tables, lists, inputs = load_sources(args.main)
  run(sides, tables, lists, out, types.SimpleNamespace(replicates=args.replicates, seed=args.seed), inputs, args.main)
  print(f'wrote {out}')


if __name__ == '__main__':
  main()
