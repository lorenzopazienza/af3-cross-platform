"""Follow-up 3, the reading of section 11 for its three precision contrasts (notes/analysis_plan.md, row of 2026-10-16:
"PR(A) (A against Ah), HWh(V6,A) (V6h against Ah) and HWh(A,C) (Ah against C), read with the rule of section 11"),
computed on 2026-11-27 from frozen outputs, after every result of the study was read. No run and no cloud resource.

    analysis/.venv/bin/python analysis/fu03_a6.py --main . --out results/analysis/fu03_a6_<DATE> \\
        [--replicates 10000] [--seed 2026]

The rule (section 11, "Reading rule, fixed now"): HW(P,Q) is reported next to HWh(P,Q) and PR; a difference that
shrinks to the RC level under matched precision is attributed to the default matmul precision, and what remains to
the platform. The reference (row of 2026-10-10, item a): the h arms have r1 only, so HWh(P,Q) is compared with RC(P)
of the default arm on the same platform. The test of "shrinks to the RC level" is the one the main study's report
used and the row of 2026-11-04 (follow-up 12) records: the 95% interval of HWh(P,Q) minus RC(P) above 0 does not
shrink (attributed to the platform); an interval that contains 0 or lies below 0 does (attributed to the default
matmul precision).

Inputs, relative to --main (per-sample tables, registered seed pairs and registered rows; no structure file is read):
  results/analysis/a100_20261018/pairs_seed.csv     PR(A), HWh(V6,A), HWh(A,C), HW(V6,A), HW(A,C), RC(A)
  results/analysis/a100_20261018/contrasts.csv      their registered flip rows (check 3)
  results/analysis/a100_20261018/hw_minus_rc.csv    HW(V6,A) minus RC (check 5)
  results/analysis/a100_20261018/samples.csv        A, Ah (check 2)
  results/analysis/main_20261014/primary_v2/        samples.csv (V6, V6h, C on 6 targets; check 2); contrasts/
                                                    pairs_seed.csv (RC(V6); HWh(V6,G) and RC(G) for check 5),
                                                    contrasts/contrasts.csv and hw_minus_rc.csv (checks 3 and 5)
  results/analysis/cpu60_20261018/samples.csv       C on the other 54 targets (check 2)
  paper/assets/tables/T22b_a100_contrasts.csv       the supplement's table tab:s-fu03-contrasts (check 4)
Seed pairs are the registered ones (contrasts.pair_rows through followup34.build, run on 2026-10-18); check 2 rebuilds
every top-1 from the per-sample tables (contrasts.top) and requires equality.
Statistics: flip rates as contrasts.summarize (gtat.flip_rates); differences x minus y of a flip rate on the targets
both hold, with the bootstrap of contrasts.hw_minus_rc (gtat.paired_difference: resample targets with replacement,
10,000 replicates, a fresh generator from seed 2026 per statistic, both terms on the same resampled targets).
Rows of results.csv, each labelled:
  plan      HWh(V6,A) minus RC(V6) and HWh(A,C) minus RC(A), RC(P) on every seed of the 60 targets (seeds 1 to 5),
            as contrasts.hw_minus_rc computes RC(P) for HWh(V6,G) and HWh(G,C) in the main study and follow-up 4;
  post hoc  (a) the same with RC(P) on seeds 1 and 2 only; (b) against RC(Q): HWh(V6,A) minus RC(A); RC(C) was not
            measured (C ran r1 only), so HWh(A,C) has no RC(Q) row; (c) PR(A) minus RC(A), seeds 1 to 5 and 1 and 2
            (the rule gives PR no reference of its own); (d) the default-precision side of each pair, HW minus RC(P),
            on seeds 1 and 2 (HW(V6,A) minus RC(V6) on seeds 1 to 5 is registered and copied); (e) HW minus HWh on
            seeds 1 and 2, the shrink itself, on the same resampled targets.
Position of an interval: "above the RC level" (lower bound above 0), "below the RC level" (upper bound below 0),
otherwise "at the RC level"; for HW minus HWh the same against 0.
Checks, in order; each stops the analysis if it fails:
  1 design: every contrast has 60 targets and its seeds (1 and 2, or 1 to 5) on every target, one pair each;
  2 top-1: the top-1 and its success rebuilt from the per-sample tables equal both sides of every registered pair;
  3 registered rates: the flip rows of contrasts.csv (follow-up 3 and main study) are reproduced within 1e-9;
  4 supplement: the rates of tab:s-fu03-contrasts (T22b) equal the recomputed ones at one decimal; a mismatch names
    the contrast;
  5 method: the registered HW(V6,A) minus RC rows (follow-up 3) and HWh(V6,G) minus RC rows (main study) are
    reproduced within 1e-9, and the plan rows equal contrasts.hw_minus_rc.
Outputs in --out: rates.csv, results.csv, checks.csv, meta.json (versions, commit, seed, inputs with SHA-256),
report.md.
"""
import argparse
import json
import pathlib
import platform
import subprocess
import sys
import types

import numpy as np
import pandas as pd

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / 'analysis'))
import contrasts  # noqa: E402
import followup34 as f34  # noqa: E402
import gtat  # noqa: E402

AN = 'results/analysis'
MAIN = f'{AN}/main_20261014/primary_v2'
A100 = f'{AN}/a100_20261018'
CPU60 = f'{AN}/cpu60_20261018'
T22B = 'paper/assets/tables/T22b_a100_contrasts.csv'
S12, S15 = (1, 2), (1, 2, 3, 4, 5)
N_TARGETS = 60
TOL = 1e-9
# name: (side A (condition, repetition), side B, seeds, source of the registered pairs)
PAIRS = {'PR(A)': (('A', 'r1'), ('Ah', 'r1'), S12, 'a100'),
         'HWh(V6,A)': (('V6h', 'r1'), ('Ah', 'r1'), S12, 'a100'),
         'HWh(A,C)': (('Ah', 'r1'), ('C', 'r1'), S12, 'a100'),
         'HW(V6,A)': (('V6', 'r1'), ('A', 'r1'), S15, 'a100'),
         'HW(A,C)': (('A', 'r1'), ('C', 'r1'), S12, 'a100'),
         'RC(A)': (('A', 'r1'), ('A', 'r2'), S15, 'a100'),
         'RC(V6)': (('V6', 'r1'), ('V6', 'r2'), S15, 'main')}
T22B_ROWS = ['PR(A)', 'HWh(V6,A)', 'HWh(A,C)', 'HW(V6,A)', 'HW(A,C)', 'RC(A)']
T22B_COLS = {'flip_top1': 'top-1 flip (%)', 'flip_success': 'success flip (%)',
             'flip_matters': 'flips that matter (%)'}
SEEDS_TEXT = {S12: '1, 2', S15: '1 to 5'}
CODE = ['analysis/fu03_a6.py', 'analysis/contrasts.py', 'analysis/followup34.py', 'analysis/gtat.py',
        'analysis/tests/test_fu03_a6.py', 'scripts/analysis_runs/fu03_a6/reproduce.sh']


# ---------------------------------------------------------------- inputs

def load_sources(root):
  root = pathlib.Path(root)
  main = f34.load_samples(root / MAIN / 'samples.csv', ['main'])
  a100 = f34.load_samples(root / A100 / 'samples.csv', ['a100'])
  c60 = f34.load_samples(root / CPU60 / 'samples.csv', ['cpu60'])
  f34.disjoint_units(main, c60, 'C')
  ok = f34.union(main[main.condition.isin(['V6', 'V6h', 'C'])], a100[a100.condition.isin(['A', 'Ah'])],
                 c60[c60.condition == 'C'])
  tables = {'a100_sp': pd.read_csv(root / A100 / 'pairs_seed.csv'),
            'a100_con': pd.read_csv(root / A100 / 'contrasts.csv'),
            'a100_hw': pd.read_csv(root / A100 / 'hw_minus_rc.csv'),
            'main_sp': pd.read_csv(root / MAIN / 'contrasts' / 'pairs_seed.csv'),
            'main_con': pd.read_csv(root / MAIN / 'contrasts' / 'contrasts.csv'),
            'main_hw': pd.read_csv(root / MAIN / 'contrasts' / 'hw_minus_rc.csv'),
            't22b': pd.read_csv(root / T22B, dtype=str)}
  inputs = [A100 + '/pairs_seed.csv', A100 + '/contrasts.csv', A100 + '/hw_minus_rc.csv', A100 + '/samples.csv',
            MAIN + '/samples.csv', MAIN + '/contrasts/pairs_seed.csv', MAIN + '/contrasts/contrasts.csv',
            MAIN + '/contrasts/hw_minus_rc.csv', CPU60 + '/samples.csv', T22B]
  return ok, tables, inputs


def registered_pairs(tables):
  """The registered seed pairs of PAIRS, each from its source."""
  parts = [tables[f'{src}_sp'][tables[f'{src}_sp'].contrast == name] for name, (_, _, _, src) in PAIRS.items()]
  return pd.concat(parts, ignore_index=True)


def seeds(sp, which):
  return sp[sp.seed_a.isin(which)]


# ---------------------------------------------------------------- checks

def check_design(sp):
  rows = []
  for name, (_, _, want, _) in PAIRS.items():
    p = sp[sp.contrast == name]
    if p.empty:
      raise SystemExit(f'check 1: no registered pairs of {name}')
    if (p.seed_a != p.seed_b).any() or p.duplicated(['target', 'seed_a']).any():
      raise SystemExit(f'check 1: {name}: pairs across seeds or a (target, seed) twice')
    per = p.groupby('target').seed_a.apply(lambda s: tuple(sorted(int(x) for x in s)))
    if len(per) != N_TARGETS or (per != want).any():
      raise SystemExit(f'check 1: {name}: {len(per)} targets, or a target without seeds {SEEDS_TEXT[want]}')
    rows.append({'check': '1 design', 'contrast': name,
                 'value': f'pass ({N_TARGETS} targets, seeds {SEEDS_TEXT[want]}, {len(p)} pairs)'})
  return rows


def tops(ok, cond, rep):
  """{(target, seed): (top-1 sample, success)} of one (condition, repetition); every unit must hold 5 samples."""
  g = ok[(ok.condition == cond) & (ok.repetition == rep)]
  out = {}
  for (t, s), u in g.groupby(['target', 'seed']):
    if u['sample'].nunique() != 5 or len(u) != 5 or u.ranking_score.count() != 5:
      raise SystemExit(f'check 2: {cond} {rep} {t} seed {s}: not exactly 5 scored samples')
    first, succ, _ = contrasts.top(u.set_index('sample').sort_index())
    out[(t, int(s))] = (int(first), bool(succ))
  return out


def check_top1(ok, sp):
  rows = []
  for name, (a, b, _, _) in PAIRS.items():
    p = sp[sp.contrast == name]
    for side, key in (('a', a), ('b', b)):
      mine = tops(ok, *key)
      reg = {(r.target, int(r.seed_a)): (int(getattr(r, f'top1_{side}')), bool(getattr(r, f'success_{side}')))
             for r in p.itertuples()}
      missing = sorted(set(reg) - set(mine))
      bad = sorted(k for k in reg if k in mine and mine[k] != reg[k])
      if missing or bad:
        raise SystemExit(f'check 2: {name} side {side.upper()} ({key[0]} {key[1]}): units missing {missing[:3]}, '
                         f'top-1 or success differs on {bad[:3]}')
    rows.append({'check': '2 top-1 from the per-sample tables', 'contrast': name,
                 'value': f'pass ({len(p)} pairs, both sides: {a[0]} {a[1]} and {b[0]} {b[1]})'})
  return rows


def check_rates(sp, tables, args):
  rows = []
  for name, (_, _, _, src) in PAIRS.items():
    mine = pd.DataFrame(gtat.flip_rates(sp, name, args)).set_index('statistic')
    con = tables[f'{src}_con']
    reg = con[con.contrast == name].set_index('statistic')
    for st in contrasts.FLIP_STATS:
      for col in ('estimate', 'ci_low', 'ci_high'):
        if st not in reg.index or abs(float(mine.loc[st, col]) - float(reg.loc[st, col])) > TOL:
          raise SystemExit(f'check 3: the registered {name} {st} row ({src}) is not reproduced ({col})')
    rows.append({'check': '3 registered rates', 'contrast': name,
                 'value': f'pass (three flip rows of {src} contrasts.csv within {TOL:g})'})
  return rows


def cell(r):
  return f'{100 * r["estimate"]:.1f} [{100 * r["ci_low"]:.1f}, {100 * r["ci_high"]:.1f}]'


def check_t22b(sp, tables, args):
  t = tables['t22b'].set_index('contrast')
  rows = []
  for name in T22B_ROWS:
    if name not in t.index:
      raise SystemExit(f'check 4: {name} is not a row of {T22B}')
    mine = {r['statistic']: cell(r) for r in gtat.flip_rates(sp, name, args)}
    diff = [f'{col}: table {t.loc[name, col]!r}, recomputed {mine[st]!r}' for st, col in T22B_COLS.items()
            if t.loc[name, col] != mine[st]]
    if diff or str(t.loc[name, 'targets']) != str(N_TARGETS):
      raise SystemExit(f'check 4: {name} differs from the supplement table (stop on this contrast): ' + '; '.join(diff))
    rows.append({'check': '4 supplement table tab:s-fu03-contrasts', 'contrast': name,
                 'value': 'pass (' + '; '.join(mine[st] for st in T22B_COLS) + ')'})
  return rows


def check_method(sp, tables, args):
  """The difference function reproduces registered HW and HWh minus RC rows; the plan rows equal hw_minus_rc."""
  cases = [('HW(V6,A)', tables['a100_hw'], {'RC(P)': ('RC(V6)', 'main_sp'), 'RC(Q), secondary': ('RC(A)', 'a100_sp')},
            'a100_sp'),
           ('HWh(V6,G)', tables['main_hw'], {'RC(P)': ('RC(V6)', 'main_sp'), 'RC(Q), secondary': ('RC(G)', 'main_sp')},
            'main_sp')]
  n = 0
  for name, reg, refs, src in cases:
    x = tables[src][tables[src].contrast == name]
    for ref, (rc, rsrc) in refs.items():
      y = tables[rsrc][tables[rsrc].contrast == rc]
      got = gtat.paired_difference(x, y, args).set_index('statistic')
      r = reg[(reg.hardware == name) & (reg.recompilation == rc) & (reg.reference == ref)].set_index('statistic')
      for st in contrasts.FLIP_STATS:
        for col in ('estimate', 'ci_low', 'ci_high'):
          if st not in r.index or abs(float(got.loc[st, col]) - float(r.loc[st, col])) > TOL:
            raise SystemExit(f'check 5: the registered {name} minus {rc} {st} row is not reproduced ({col})')
        n += 1
  plan = plan_rows_hw_minus_rc(sp, args)
  for name, rc in (('HWh(V6,A)', 'RC(V6)'), ('HWh(A,C)', 'RC(A)')):
    got = gtat.paired_difference(sp[sp.contrast == name], sp[sp.contrast == rc], args).set_index('statistic')
    h = plan[(plan.hardware == name) & (plan.recompilation == rc)].set_index('statistic')
    for st in contrasts.FLIP_STATS:
      for col in ('estimate', 'ci_low', 'ci_high'):
        if abs(float(got.loc[st, col]) - float(h.loc[st, col])) > TOL:
          raise SystemExit(f'check 5: the plan row {name} minus {rc} differs from contrasts.hw_minus_rc ({col})')
  return [{'check': '5 method', 'contrast': 'HW(V6,A), HWh(V6,G); HWh(V6,A), HWh(A,C)',
           'value': f'pass ({n} registered minus-RC rows within {TOL:g}; plan rows equal contrasts.hw_minus_rc)'}]


def plan_rows_hw_minus_rc(sp, args):
  """contrasts.hw_minus_rc on the two HWh contrasts, unchanged (RC(P) and, where measured, RC(Q))."""
  use = sp[sp.contrast.isin(['HWh(V6,A)', 'HWh(A,C)', 'RC(V6)', 'RC(A)'])]
  cons = {'HWh(V6,A)': ('HWh', False), 'HWh(A,C)': ('HWh', False), 'RC(V6)': ('RC', True), 'RC(A)': ('RC', True)}
  return contrasts.hw_minus_rc(use, cons, args)


# ---------------------------------------------------------------- statistics

def position(lo, hi, level='the RC level'):
  if lo != lo or hi != hi:
    return 'not computed'
  if lo > 0:
    return f'above {level}'
  if hi < 0:
    return f'below {level}'
  return f'at {level}'


def reading(kind, lo, hi):
  pos = position(lo, hi)
  if pos == 'not computed':
    return pos
  if kind == 'rule':
    return 'attributed to the platform' if pos == 'above the RC level' else 'attributed to the default matmul precision'
  if kind == 'pr':
    return {'above the RC level': 'the precision change moves more than recompilation',
            'at the RC level': 'not distinguished from recompilation',
            'below the RC level': 'the precision change moves less than recompilation'}[pos]
  if kind == 'hw':
    return 'a difference beyond recompilation' if pos == 'above the RC level' else \
        'no difference beyond recompilation shown'
  if kind == 'shrink':
    return {'above the RC level': 'larger at default precision', 'below the RC level': 'smaller at default precision',
            'at the RC level': 'no difference shown'}[pos]
  raise ValueError(kind)


def rates(sp, args):
  rows = []
  for name, (_, _, own, _) in PAIRS.items():
    for which, label in [(own, 'registered')] + ([(S12, 'post hoc subset')] if own != S12 else []):
      sub = seeds(sp, which)
      for r in gtat.flip_rates(sub, name, args):
        rows.append({'label': label, 'seeds': SEEDS_TEXT[which], **r})
  return pd.DataFrame(rows)


# (id, label, x, seeds of x, y, seeds of y, kind of reading, note)
ROWS = [('P1', 'plan', 'HWh(V6,A)', S12, 'RC(V6)', S15, 'rule', 'reading of row 2026-10-16 with RC(P), row 2026-10-10 a'),
        ('P2', 'plan', 'HWh(A,C)', S12, 'RC(A)', S15, 'rule', 'reading of row 2026-10-16 with RC(P), row 2026-10-10 a'),
        ('Ha1', 'post hoc', 'HWh(V6,A)', S12, 'RC(V6)', S12, 'rule', '(a) RC(P) on the same seeds 1 and 2'),
        ('Ha2', 'post hoc', 'HWh(A,C)', S12, 'RC(A)', S12, 'rule', '(a) RC(P) on the same seeds 1 and 2'),
        ('Hb1', 'post hoc', 'HWh(V6,A)', S12, 'RC(A)', S15, 'rule', '(b) RC(Q); secondary row of contrasts.hw_minus_rc'),
        ('Hb2', 'post hoc', 'HWh(V6,A)', S12, 'RC(A)', S12, 'rule', '(b) RC(Q) on seeds 1 and 2'),
        ('Hb3', 'post hoc', 'HWh(A,C)', S12, 'RC(C)', S12, 'rule', '(b) RC(C) not measured: C ran r1 only'),
        ('Hc1', 'post hoc', 'PR(A)', S12, 'RC(A)', S15, 'pr', '(c) the rule gives PR no reference of its own'),
        ('Hc2', 'post hoc', 'PR(A)', S12, 'RC(A)', S12, 'pr', '(c) on seeds 1 and 2'),
        ('R1', 'registered', 'HW(V6,A)', S15, 'RC(V6)', S15, 'hw', 'row 2026-10-16, HW minus RC; a100 hw_minus_rc.csv'),
        ('Hd1', 'post hoc', 'HW(V6,A)', S12, 'RC(V6)', S15, 'hw', '(d) the default-precision side on seeds 1 and 2'),
        ('Hd2', 'post hoc', 'HW(V6,A)', S12, 'RC(V6)', S12, 'hw', '(d) on seeds 1 and 2'),
        ('Hd3', 'post hoc', 'HW(A,C)', S12, 'RC(A)', S15, 'hw', '(d) the default-precision side'),
        ('Hd4', 'post hoc', 'HW(A,C)', S12, 'RC(A)', S12, 'hw', '(d) on seeds 1 and 2'),
        ('He1', 'post hoc', 'HW(V6,A)', S12, 'HWh(V6,A)', S12, 'shrink', '(e) HW minus HWh, seeds 1 and 2'),
        ('He2', 'post hoc', 'HW(A,C)', S12, 'HWh(A,C)', S12, 'shrink', '(e) HW minus HWh, seeds 1 and 2')]


def results(sp, args):
  out = []
  for rid, label, x, sx, y, sy, kind, note in ROWS:
    xs, ys = seeds(sp[sp.contrast == x], sx), seeds(sp[sp.contrast == y], sy)
    base = {'id': rid, 'label': label, 'quantity': x, 'minus': y, 'seeds_x': SEEDS_TEXT[sx], 'seeds_y': SEEDS_TEXT[sy],
            'pairs_x': len(xs), 'pairs_y': len(ys), 'note': note}
    if ys.empty:
      for st in contrasts.FLIP_STATS:
        out.append({**base, 'statistic': st, 'n_targets': 0, 'estimate': np.nan, 'ci_low': np.nan, 'ci_high': np.nan,
                    'position': 'not computed', 'reading': 'not computed'})
      continue
    for r in gtat.paired_difference(xs, ys, args).to_dict('records'):
      level = '0 (no shrink shown)' if kind == 'shrink' else 'the RC level'
      out.append({**base, **r, 'position': position(r['ci_low'], r['ci_high'], level),
                  'reading': reading(kind, r['ci_low'], r['ci_high'])})
  return pd.DataFrame(out)


# ---------------------------------------------------------------- outputs

def report(rt, res, chk, args):
  pct, pts = gtat.pct, gtat.pts
  L = ['# Follow-up 3: the section 11 reading of PR(A), HWh(V6,A) and HWh(A,C) (point A6)', '',
       'Computed from frozen outputs, after every result was read. Rates in %, differences in points, 95% percentile '
       f'intervals from the paired bootstrap over targets ({args.replicates} replicates, seed {args.seed}). Labels: '
       'plan (reference and test fixed by the plan rows), registered (an output of 2026-10-18), post hoc.', '',
       '## Checks', ''] + [f'- {c["check"]} ({c["contrast"]}): {c["value"]}' for c in chk]
  L += ['', '## Flip rates', '', '| label | contrast | seeds | statistic | pairs | targets | events | rate % [CI] |',
        '|---|---|---|---|---|---|---|---|']
  for r in rt.to_dict('records'):
    L.append(f'| {r["label"]} | {r["contrast"]} | {r["seeds"]} | {r["statistic"]} | {r["n"]} | {r["n_targets"]} | '
             f'{r["events"]} | {pct(r["estimate"])} [{pct(r["ci_low"])}, {pct(r["ci_high"])}] |')
  L += ['', '## Differences and readings', '', '| id | label | quantity (seeds) | minus (seeds) | statistic | targets | '
        'points [CI] | position | reading |', '|---|---|---|---|---|---|---|---|---|']
  for r in res.to_dict('records'):
    L.append(f'| {r["id"]} | {r["label"]} | {r["quantity"]} ({r["seeds_x"]}) | {r["minus"]} ({r["seeds_y"]}) | '
             f'{r["statistic"]} | {r["n_targets"]} | {pts(r["estimate"])} [{pts(r["ci_low"])}, {pts(r["ci_high"])}] | '
             f'{r["position"]} | {r["reading"]} |')
  return '\n'.join(L) + '\n'


def git(*cmd):
  try:
    return subprocess.run(['git', *cmd], cwd=REPO, capture_output=True, text=True, check=True).stdout.strip()
  except (OSError, subprocess.CalledProcessError):
    return None


def meta(args, inputs, root):
  return {'module': 'analysis/fu03_a6.py', 'commit': git('rev-parse', 'HEAD'),
          'uncommitted': (git('status', '--porcelain', '--', 'analysis', 'scripts') or '').splitlines(),
          'code_sha256': {p: gtat.sha256(REPO / p) for p in CODE},
          'python': platform.python_version(), 'numpy': np.__version__, 'pandas': pd.__version__,
          'replicates': args.replicates, 'seed': args.seed,
          'inputs': {p: gtat.sha256(pathlib.Path(root) / p) for p in inputs}}


def run(ok, tables, out, args, inputs=(), root=None):
  out = pathlib.Path(out)
  out.mkdir(parents=True, exist_ok=True)
  sp = registered_pairs(tables)
  chk = check_design(sp)
  chk += check_top1(ok, sp)
  chk += check_rates(sp, tables, args)
  chk += check_t22b(sp, tables, args)
  chk += check_method(sp, tables, args)
  rt, res = rates(sp, args), results(sp, args)
  rt.to_csv(out / 'rates.csv', index=False)
  res.to_csv(out / 'results.csv', index=False)
  pd.DataFrame(chk).to_csv(out / 'checks.csv', index=False)
  if root is not None:
    (out / 'meta.json').write_text(json.dumps(meta(args, inputs, root), indent=1) + '\n')
  (out / 'report.md').write_text(report(rt, res, chk, args))
  return rt, res, chk


def main():
  ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
  ap.add_argument('--main', default='.', help='the checkout that holds results/ and paper/assets/')
  ap.add_argument('--out', required=True)
  ap.add_argument('--replicates', type=int, default=10000)
  ap.add_argument('--seed', type=int, default=2026)
  args = ap.parse_args()
  out = pathlib.Path(args.out)
  if (out / 'results.csv').exists():
    raise SystemExit(f'{out}/results.csv exists; the output folder must be new')
  ok, tables, inputs = load_sources(args.main)
  run(ok, tables, out, types.SimpleNamespace(replicates=args.replicates, seed=args.seed), inputs, args.main)
  print((out / 'report.md').read_text())


if __name__ == '__main__':
  main()
