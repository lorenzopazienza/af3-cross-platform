"""Follow-up 10, larger inputs on the whole PDB (notes/analysis_plan.md section 10, row of 2026-11-04 "Follow-up
experiment 10, larger inputs on the whole PDB", item d; OSF addendum https://osf.io/nsbgp; label large; reported as a
follow-up, outside the pre-registered study, whatever it shows).

    analysis/.venv/bin/python analysis/large.py --samples results/analysis/large_<date>/samples.csv \\
        --out results/analysis/large_<date> [--manifest inputs/large/manifest.csv]
        [--targets targets/large/large_subset.csv]

Targets: the rows of the follow-up 10 input manifest (34 large, 1,025 to 1,536 tokens; 34 control, at most 1,024;
less any excluded and not replaced, section 10); group, variant A, method, token count and the r2 subset from
targets/large/large_subset.csv. Conditions: V5, V6 and G of label large (the main-study settings), seeds 1 to 5; r1 on
every target; r2 on G for every target and on V5 and V6 for the row's subset (tpu_r2), or for every target of an arm
if the row's extension was applied (RC then uses the targets that have r2, as the data say).
Checks, before anything else (the analysis stops if one fails): exit codes; one machine type per condition; every
target is a manifest target; none is among the 60 of the main study or the 72 of follow-up 11; r2 on V5 and V6 covers
at least the row's subset.
Bit identity of TPU recompilation (row item c), reported first: for every subset target, r2 of V5 and of V6 against
r1, sample by sample (mmCIF coordinates identical as written, as analysis/contrasts.py and interpretation 8 of the
plan read "bit-identical"). If any target is not bit-identical, the report says so at its top: the row's extension
rule (r2 on that arm for all 68 targets, recorded in a row before that run) applies.
Contrasts, as section 5 (analysis/followup34.build and contrasts.summarize; target-clustered bootstrap, 10,000
replicates, seed 2026), in each group: the primary pairs HW(V5,V6) and HW(V6,G); HW(V5,G) (secondary); RC(V5),
RC(V6), RC(G); SD on each platform.
HW minus RC (row item d), in each group, against RC of the pair's first platform (RC(P)) and of its second (RC(Q),
secondary): RC(V5) and RC(V6) on the subset's targets of the group, so HW minus RC is HW over all the group's targets
minus RC over its subset targets, each resampled within its own targets; RC(G) on all targets, paired (the same
resampled targets, as contrasts.hw_minus_rc). Headline, per primary pair: the large minus control difference in HW
minus RC(P) for the top-1 flip rate, with its 95% interval from resampling targets within each group independently.
Reading rule (the row, as written): interval above 0, "the hardware effect grows with input size"; below 0, "it
shrinks"; otherwise "no size difference detected at this sample size", with the interval's width. The same
difference against RC(Q) is reported next to it; the success flip rate and flips that matter are secondary (not read).
Variant A: the same statistics on the variant A targets, a sensitivity analysis, not read with the rule.
Exploratory, no reading rule: the per-seed top-1 flip of each primary pair (r1 seed pairs) against token count over
all targets, logistic regression (maximum likelihood, Newton-Raphson; slope per 256 tokens), 95% interval from the
target-clustered bootstrap (same replicates and seed; a replicate whose fit does not converge, as under complete
separation, is left out and counted), with and without method (EM 1, X-ray 0) as a covariate.
M1 (row item d): every M1 result (the success flip rate, flips that matter, the success rate of the per-seed top-1)
is flagged with the number of EM targets in its set and is also given on the X-ray targets (12 large, 12 control).
Every target is post-cutoff, so there is no post-cutoff set.
Outputs in --out: checks.csv, completeness.csv, bit_identity.csv, contrasts.csv, hw_minus_rc.csv, headline.csv,
reading_rule.csv, logistic.csv, success_rates.csv, pairs_seed.csv, pairs_sample.csv, counts.csv, report.md.
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

LABEL = 'large'
CONDS = ['V5', 'V6', 'G']
TPU = ['V5', 'V6']
SPECS = [('RC(V5)', 'RC', ('V5', 'r1'), ('V5', 'r2'), True), ('RC(V6)', 'RC', ('V6', 'r1'), ('V6', 'r2'), True),
         ('RC(G)', 'RC', ('G', 'r1'), ('G', 'r2'), True),
         ('HW(V5,V6)', 'HW', ('V5', 'r1'), ('V6', 'r1'), True), ('HW(V6,G)', 'HW', ('V6', 'r1'), ('G', 'r1'), True),
         ('HW(V5,G)', 'HW', ('V5', 'r1'), ('G', 'r1'), False),
         ('SD(V5)', 'SD', ('V5', 'r1'), ('V5', 'r1'), True), ('SD(V6)', 'SD', ('V6', 'r1'), ('V6', 'r1'), True),
         ('SD(G)', 'SD', ('G', 'r1'), ('G', 'r1'), True)]
PAIRS = {'HW(V5,V6)': ('V5', 'V6'), 'HW(V6,G)': ('V6', 'G'), 'HW(V5,G)': ('V5', 'G')}
PRIMARY = ['HW(V5,V6)', 'HW(V6,G)']
GROUPS = ['large', 'control']
HEADLINE = 'flip_top1'
M1_STATS = ['flip_success', 'flip_matters']
TOKENS_PER_UNIT = 256


def target_sums(sp, contrast, targets, stat):
  """Per target (in the given order): the sum and the count of the statistic over the contrast's seed pairs."""
  s = sp[(sp.contrast == contrast) & sp.target.isin(targets)]
  g = s.groupby('target')[stat].agg(['sum', 'count']).astype(float)
  ts = [t for t in targets if t in g.index]
  return ts, g.loc[ts, 'sum'].values, g.loc[ts, 'count'].values


def hw_rc(sp, hw, rc, targets, stat, rng, reps):
  """HW minus RC on one group's targets: (estimate, bootstrap replicates, HW targets, RC targets, paired). Paired
  (the same resampled targets) when HW and RC have the same targets, as contrasts.hw_minus_rc; else HW and RC are
  each resampled within their own targets (RC(V5), RC(V6) on the r2 subset: row item d)."""
  th, hs, hc = target_sums(sp, hw, targets, stat)
  tr, rs, rcn = target_sums(sp, rc, targets, stat)
  if not th or not tr:
    return np.nan, np.full(reps, np.nan), th, tr, False
  est = hs.sum() / hc.sum() - rs.sum() / rcn.sum()
  paired = th == tr
  ih = rng.integers(0, len(th), size=(reps, len(th)))
  ir = ih if paired else rng.integers(0, len(tr), size=(reps, len(tr)))
  boot = hs[ih].sum(1) / hc[ih].sum(1) - rs[ir].sum(1) / rcn[ir].sum(1)
  return est, boot, th, tr, paired


def ci(x):
  x = x[~np.isnan(x)]
  return (float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))) if len(x) else (np.nan, np.nan)


def differences(sp, sets, args):
  """Per target set, pair, RC reference and statistic: HW minus RC in each group and the large minus control
  difference, each group resampled independently."""
  hw_rows, head = [], []
  for set_name, by_group, stats in sets:
    for pair in PAIRS:
      p, q = PAIRS[pair]
      for which, rc_c in (('RC(P)', p), ('RC(Q), secondary', q)):
        rc = f'RC({rc_c})'
        for st in stats:
          rng = np.random.default_rng(args.seed)
          res = {}
          for g in GROUPS:
            est, boot, th, tr, paired = hw_rc(sp, pair, rc, by_group[g], st, rng, args.replicates)
            lo, hi = ci(boot)
            res[g] = (est, boot)
            hw_rows.append({'targets': set_name, 'group': g, 'hardware': pair, 'recompilation': rc, 'reference': which,
                            'statistic': st, 'n_targets_hw': len(th), 'n_targets_rc': len(tr),
                            'resampling': 'paired' if paired else 'independent', 'estimate': est, 'ci_low': lo,
                            'ci_high': hi})
          d = res['large'][1] - res['control'][1]
          lo, hi = ci(d)
          head.append({'targets': set_name, 'hardware': pair, 'recompilation': rc, 'reference': which, 'statistic': st,
                       'role': ('headline' if pair in PRIMARY and which == 'RC(P)' and st == HEADLINE else
                                'secondary pair' if pair not in PRIMARY else 'secondary'),
                       'estimate': res['large'][0] - res['control'][0], 'ci_low': lo, 'ci_high': hi,
                       'width': hi - lo if lo == lo else np.nan})
  return pd.DataFrame(hw_rows), pd.DataFrame(head)


def reading(head, bad):
  rows = []
  for pair in PRIMARY:
    r = head[(head.targets == 'all') & (head.hardware == pair) & (head.reference == 'RC(P)') & (head.statistic == HEADLINE)]
    q = head[(head.targets == 'all') & (head.hardware == pair) & (head.reference == 'RC(Q), secondary')
             & (head.statistic == HEADLINE)]
    if r.empty or r.ci_low.iloc[0] != r.ci_low.iloc[0]:
      answer = 'not computed'
    elif r.ci_low.iloc[0] > 0:
      answer = 'the hardware effect grows with input size'
    elif r.ci_high.iloc[0] < 0:
      answer = 'the hardware effect shrinks with input size'
    else:
      answer = 'no size difference detected at this sample size'
    p, q_c = PAIRS[pair]
    rows.append({'pair': pair, 'statistic': HEADLINE, 'reference': f'RC({p})', 'answer': answer,
                 'estimate': r.estimate.iloc[0] if len(r) else np.nan, 'ci_low': r.ci_low.iloc[0] if len(r) else np.nan,
                 'ci_high': r.ci_high.iloc[0] if len(r) else np.nan, 'interval_width': r.width.iloc[0] if len(r) else np.nan,
                 'against_rc_of_second': (f'against RC({q_c}): {q.estimate.iloc[0]:+.3f} [{q.ci_low.iloc[0]:+.3f}, '
                                          f'{q.ci_high.iloc[0]:+.3f}]') if len(q) and q.ci_low.iloc[0] == q.ci_low.iloc[0] else '',
                 'incomplete_arms': ', '.join(sorted(bad & {p, q_c}))})
  return pd.DataFrame(rows)


def logit_fit(X, y, iters=50):
  """Maximum-likelihood logistic regression by Newton-Raphson; None if it does not converge (e.g. separation)."""
  b = np.zeros(X.shape[1])
  for _ in range(iters):
    p = 1 / (1 + np.exp(-np.clip(X @ b, -30, 30)))
    w = p * (1 - p)
    h = X.T @ (X * w[:, None])
    try:
      step = np.linalg.solve(h, X.T @ (y - p))
    except np.linalg.LinAlgError:
      return None
    b = b + step
    if np.abs(b).max() > 50:
      return None
    if np.abs(step).max() < 1e-8:
      return b
  return None


def logistic(sp, info, targets, args):
  rows = []
  for pair in PRIMARY:
    s = sp[(sp.contrast == pair) & sp.target.isin(targets)]
    ts = sorted(s.target.unique())
    for model, cols in (('tokens', ['tokens']), ('tokens + method', ['tokens', 'em'])):
      per = {}
      for t in ts:
        y = s[s.target == t].flip_top1.astype(float).values
        x = [info.loc[t, 'num_tokens'] / TOKENS_PER_UNIT] + ([float(info.loc[t, 'method'] == 'EM')] if 'em' in cols else [])
        per[t] = (np.column_stack([np.ones(len(y))] + [np.full(len(y), v) for v in x]), y)
      fit = lambda idx: logit_fit(np.vstack([per[ts[i]][0] for i in idx]),  # noqa: E731
                                  np.concatenate([per[ts[i]][1] for i in idx]))
      b = fit(range(len(ts))) if ts else None
      rng = np.random.default_rng(args.seed)
      slopes, failed = [], 0
      for idx in rng.integers(0, len(ts), size=(args.replicates, len(ts))) if ts else []:
        bb = fit(idx)
        if bb is None:
          failed += 1
        else:
          slopes.append(bb[1])
      lo, hi = ci(np.array(slopes)) if slopes else (np.nan, np.nan)
      rows.append({'pair': pair, 'model': model, 'n_targets': len(ts), 'n_seed_pairs': len(s),
                   'slope_per_256_tokens': b[1] if b is not None else np.nan, 'ci_low': lo, 'ci_high': hi,
                   'method_coefficient': b[2] if b is not None and len(b) > 2 else np.nan,
                   'replicates_not_converged': failed})
  return pd.DataFrame(rows)


def success_rates(ok, sets, args, em):
  rows = []
  for set_name, by_group, _ in sets:
    for g in GROUPS:
      ts = by_group[g]
      for (c, rep), d in ok[ok.target.isin(ts)].groupby(['condition', 'repetition']):
        per = {}
        for (t, s), u in d.groupby(['target', 'seed']):
          per.setdefault(t, []).append(float(contrasts.top(u.set_index('sample').sort_index())[1]))
        groups = [np.array(v) for _, v in sorted(per.items())]
        lo, hi = contrasts.boot_ci(groups, contrasts.mean, np.random.default_rng(args.seed), args.replicates)
        rows.append({'targets': set_name, 'group': g, 'condition': c, 'repetition': rep, 'n_targets': len(groups),
                     'em_targets': sum(t in em for t in per), 'success_rate':
                     contrasts.mean(np.concatenate(groups)) if groups else np.nan, 'ci_low': lo, 'ci_high': hi})
  return pd.DataFrame(rows)


def main():
  ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
  ap.add_argument('--samples', required=True)
  ap.add_argument('--manifest', default=str(REPO / 'inputs' / 'large' / 'manifest.csv'))
  ap.add_argument('--targets', default=str(REPO / 'targets' / 'large' / 'large_subset.csv'))
  ap.add_argument('--other_targets', nargs='*', default=[str(REPO / 'targets' / 'posebusters_subset.csv'),
                                                         str(REPO / 'targets' / 'repl' / 'repl_subset.csv')])
  ap.add_argument('--out', required=True)
  ap.add_argument('--replicates', type=int, default=10000)
  ap.add_argument('--seed', type=int, default=2026)
  args = ap.parse_args()
  out = pathlib.Path(args.out)
  out.mkdir(parents=True, exist_ok=True)
  man = pd.read_csv(args.manifest, dtype=str)
  info = pd.read_csv(args.targets, dtype=str).set_index('pdb_id')
  info['num_tokens'] = info.num_tokens.astype(int)
  targets = sorted(man.pdb_id)
  subset = sorted(t for t in targets if info.loc[t, 'tpu_r2'] == 'yes')
  em = {t for t in targets if info.loc[t, 'method'] == 'EM'}
  others = set()
  for f in args.other_targets:
    if pathlib.Path(f).is_file():
      others |= set(pd.read_csv(f, dtype=str).pdb_id)

  d = f34.load_samples(args.samples, [LABEL])
  d = d[d.condition.isin(CONDS)]
  chk = f34.exit_code_check(d, CONDS, LABEL)
  for c in CONDS:
    types = sorted(d[d.condition == c].machine_type.dropna().astype(str).unique())
    chk.append({'check': 'one machine type per condition', 'condition': c, 'value': ', '.join(types),
                'detail': 'pass' if len(types) == 1 else f'FAIL: {len(types)} machine types'})
  extra, clash = sorted(set(d.target) - set(targets)), sorted(set(d.target) & others)
  chk.append({'check': 'every target is in the follow-up 10 manifest', 'condition': ', '.join(CONDS),
              'value': f'{len(set(d.target))} targets', 'detail': 'pass' if not extra else f'FAIL: {extra[:5]}'})
  chk.append({'check': 'no target among the 60 of the main study or the 72 of follow-up 11', 'condition': ', '.join(CONDS),
              'value': f'{len(clash)}', 'detail': 'pass' if not clash else f'FAIL: {clash[:5]}'})
  r2_targets = {c: sorted(d[(d.condition == c) & (d.repetition == 'r2')].target.unique()) for c in CONDS}
  for c in TPU:
    missing = sorted(set(subset) - set(r2_targets[c]))
    chk.append({'check': 'r2 covers the row\'s subset', 'condition': c, 'value': f'{len(r2_targets[c])} targets with r2',
                'detail': 'pass' if not missing else f'FAIL: no r2 for {missing[:5]}'})
  pd.DataFrame(chk).to_csv(out / 'checks.csv', index=False)
  if any(r['detail'] != 'pass' for r in chk):
    raise SystemExit('a check failed (checks.csv); nothing is compared')

  ok = f34.union(d)
  comp = []
  for c in CONDS:
    for rep in ('r1', 'r2'):
      planned = targets if rep == 'r1' or c == 'G' or set(r2_targets[c]) - set(subset) else subset
      units = ok[(ok.condition == c) & (ok.repetition == rep) & ok.target.isin(planned)].groupby(['target', 'seed']).size()
      have = int((units == 5).sum())
      comp.append({'condition': c, 'repetition': rep, 'planned_targets': len(planned), 'planned_units': len(planned) * 5,
                   'units_with_5_samples': have, 'status': 'complete' if have == len(planned) * 5 else 'incomplete'})
  comp = pd.DataFrame(comp)
  bad = set(comp[comp.status != 'complete'].condition)
  sp, sm, cons, counts = f34.build(ok, SPECS)

  bit = []
  for c in TPU:
    s = sm[sm.contrast == f'RC({c})']
    for t in subset:
      x = s[s.target == t]
      bit.append({'condition': c, 'target': t, 'group': info.loc[t, 'group'], 'sample_pairs': len(x),
                  'bit_identical': int(x.bit_identical.sum()), 'all_identical': bool(len(x) and x.bit_identical.all())})
  bit = pd.DataFrame(bit)
  extension = {c: sorted(bit[(bit.condition == c) & ~bit.all_identical].target) for c in TPU}

  group_of = {g: [t for t in targets if info.loc[t, 'group'] == g] for g in GROUPS}
  va = {g: [t for t in ts if info.loc[t, 'variant_A'] == 'yes'] for g, ts in group_of.items()}
  xr = {g: [t for t in ts if t not in em] for g, ts in group_of.items()}
  sets = [('all', group_of, contrasts.FLIP_STATS), ('variant A', va, contrasts.FLIP_STATS), ('X-ray', xr, M1_STATS)]
  hw, head = differences(sp, sets, args)
  rule = reading(head, bad)
  con = []
  for set_name, by_group, stats in sets:
    for g in GROUPS:
      ts = by_group[g]
      c = contrasts.summarize(sp[sp.target.isin(ts)], sm[sm.target.isin(ts)], cons, args)
      if set_name == 'X-ray':
        c = c[c.statistic.isin(stats)]
      con.append(c.assign(targets=set_name, group=g, em_targets=sum(t in em for t in ts)))
  con = pd.concat(con, ignore_index=True)
  con['incomplete_arm'] = con.contrast.map(lambda n: ', '.join(sorted(bad & set(n[n.index('(') + 1:-1].split(',')))))
  sets_map = {s: b for s, b, _ in sets}

  def em_count(r):   # the M1 flag: EM targets behind an M1 statistic (its group, or both groups for a difference)
    gs = [r.group] if hasattr(r, 'group') else GROUPS
    return sum(t in em for g in gs for t in sets_map[r.targets][g]) if r.statistic in M1_STATS else ''
  for df in (hw, head):
    df['m1_em_targets'] = [em_count(r) for r in df.itertuples()]
  logi = logistic(sp, info, targets, args)
  succ = success_rates(ok, sets, args, em)

  for name, df in (('completeness', comp), ('bit_identity', bit), ('contrasts', con), ('hw_minus_rc', hw),
                   ('headline', head), ('reading_rule', rule), ('logistic', logi), ('success_rates', succ),
                   ('pairs_seed', sp), ('pairs_sample', sm), ('counts', counts)):
    df.to_csv(out / f'{name}.csv', index=False)
  pct = f34.pct
  pp = lambda x: '-' if x != x else f'{100 * x:+.1f}'  # noqa: E731
  L = ['# Follow-up 10: larger inputs on the whole PDB', '']
  if any(extension.values()):
    L += ['**EXTENSION RULE TRIGGERED (row item c): TPU recompilation is not bit-identical on '
          + '; '.join(f'{c}: {", ".join(ts)}' for c, ts in extension.items() if ts)
          + '. r2 on that arm extends to all 68 targets, recorded in a section 10 row with its own budget check before '
          'that run; the results below use the r2 subset only.**', '']
  else:
    L += [f'TPU recompilation bit-identical on every subset target ({len(subset)} targets, r2 against r1, V5 and V6): '
          'the extension rule of row item c does not apply.', '']
  L += [f'Targets: {len(group_of["large"])} large, {len(group_of["control"])} control; variant A '
        f'{len(va["large"])} and {len(va["control"])}; X-ray {len(xr["large"])} and {len(xr["control"])}; EM {len(em)} '
        f'(M1 results are flagged: their reference is the deposited model). RC targets: '
        + ', '.join(f'{c} {len(r2_targets[c])}' for c in CONDS) + '.', '',
        '## Reading rule (headline: large minus control, HW minus RC(P), top-1 flip rate, percentage points)', '']
  for r in rule.itertuples():
    L.append(f'- {r.pair} minus {r.reference}: {pp(r.estimate)} [{pp(r.ci_low)}, {pp(r.ci_high)}], interval width '
             f'{100 * r.interval_width:.1f} points: {r.answer}; {r.against_rc_of_second}'
             + (f' (incomplete arm: {r.incomplete_arms})' if r.incomplete_arms else ''))
  L += ['', '## Large minus control (all statistics; variant A: sensitivity; X-ray: M1 only)', '',
        '| targets | quantity | reference | statistic | role | difference (points) | 95% CI |', '|---|---|---|---|---|---|---|']
  L += [f'| {r.targets} | {r.hardware} minus {r.recompilation} | {r.reference} | {r.statistic} | {r.role} | '
        f'{pp(r.estimate)} | {pp(r.ci_low)} to {pp(r.ci_high)} |' for r in head.itertuples()]
  L += ['', '## HW minus RC in each group', '', '| targets | group | quantity | statistic | HW targets | RC targets | '
        'resampling | estimate | 95% CI |', '|---|---|---|---|---|---|---|---|---|']
  L += [f'| {r.targets} | {r.group} | {r.hardware} minus {r.recompilation} | {r.statistic} | {r.n_targets_hw} | '
        f'{r.n_targets_rc} | {r.resampling} | {pct(r.estimate)} | {pct(r.ci_low)} to {pct(r.ci_high)} |'
        for r in hw.itertuples()]
  L += ['', '## Bit identity of TPU recompilation (r2 subset)', '', '| condition | targets | all samples identical |',
        '|---|---|---|']
  for c in TPU:
    b = bit[bit.condition == c]
    L.append(f'| {c} | {len(b)} | {int(b.all_identical.sum())} |')
  L += ['', '## Token count (exploratory, no reading rule)', '', '| pair | model | targets | slope per 256 tokens '
        '(log odds) | 95% CI | method (EM) | replicates not converged |', '|---|---|---|---|---|---|---|']
  L += [f'| {r.pair} | {r.model} | {r.n_targets} | {f34.num(r.slope_per_256_tokens)} | {f34.num(r.ci_low)} to '
        f'{f34.num(r.ci_high)} | {f34.num(r.method_coefficient)} | {r.replicates_not_converged} |' for r in logi.itertuples()]
  L += ['', '## Completeness', '', '| condition | repetition | planned targets | with 5 samples (units) | status |',
        '|---|---|---|---|---|']
  L += [f'| {r.condition} | {r.repetition} | {r.planned_targets} | {r.units_with_5_samples} of {r.planned_units} | '
        f'{r.status} |' for r in comp.itertuples()]
  L += ['', '## Success rate of the per-seed top-1 (M1; EM targets flagged)', '',
        '| targets | group | condition | repetition | EM targets | rate | 95% CI |', '|---|---|---|---|---|---|---|']
  L += [f'| {r.targets} | {r.group} | {r.condition} | {r.repetition} | {r.em_targets} of {r.n_targets} | '
        f'{pct(r.success_rate)} | {pct(r.ci_low)} to {pct(r.ci_high)} |' for r in succ.itertuples()]
  L += ['', '## Contrasts, all targets', '']
  for g in GROUPS:
    L += [f'### {g}', ''] + f34.contrast_lines(con[(con.targets == 'all') & (con.group == g)]) + ['']
  (out / 'report.md').write_text('\n'.join(L) + '\n')
  print((out / 'report.md').read_text())
  return 0


if __name__ == '__main__':
  sys.exit(main())
