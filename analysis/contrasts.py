"""M2 to M5 contrasts and statistics of notes/analysis_plan.md, sections 4 to 6 and 11.

    analysis/.venv/bin/python analysis/contrasts.py --samples results/analysis/<name>/samples.csv \\
        [--out results/analysis/<name>/contrasts] [--labels pilot] [--include REGEX] [--exclude REGEX] \\
        [--condition platform|platform_libtpu] [--replicates 10000] [--seed 2026]

Input: the table of analysis/collect.py. Only runs at AlphaFold3's defaults
(10 recycles, 5 diffusion samples; plan section 2) are used. A condition is
the platform code (V5, V6, G, GT, C), or platform and libtpu version with
--condition platform_libtpu. Two rows for the same (condition, target, seed,
repetition, sample) from different sessions are an error, never resolved
silently: select sessions with --labels, --include or --exclude. The
precision-matched arms of plan section 11 (V6h, Gh: analysis/collect.py) are
conditions of their own; they enter only the section 11 contrasts below,
never RC, W, SD, HW, DU or Q3.

Per (condition, repetition, target, seed):
  M2 top-1: the sample with the highest ranking_score (ties: lowest sample
     index); overall top-1 across seeds is the (seed, sample) with the highest
     ranking_score, as AlphaFold3 ranks them;
  M3 margin: top-1 minus top-2 ranking_score within the seed.
Contrasts (plan section 5), paired on (target, seed) and (target, seed, sample):
  RC(P)    r1 against r2 on platform P;
  HW(P,Q)  r1 on P against r1 on Q; primary pairs V6-G, V5-V6, G-C, others secondary;
  SD(P)    seed s against seed s' (r1) on P, as a yardstick; its top-1 flip
           compares sample indices of different seeds, which are unrelated
           draws, so only its success flip and M4/M5 are informative;
  DU       GT against G (r1);
  ST(P,P') the same platform on two software stacks (libtpu versions; only
           with --condition platform_libtpu; not a plan contrast, used for the
           stack validation);
  W(P)     r1 against the warm rerun w (pilot check 2: expected bit-identical).
Section 11 contrasts (secondary), paired and computed in the same way:
  PR(P)    r1 on P against r1 on Ph, P in {V6, G}: the default matmul precision;
  HWh(V6,G) r1 on V6h against r1 on Gh, and HWh(G,C) r1 on Gh against r1 on C:
           the platform difference with float32 matmuls matched.
Contrasts of the 2026-10-11 amendment (plan section 10; secondary), paired
and computed in the same way. Gd is the L4 with --xla_gpu_deterministic_ops
(config l4_xla_det); it enters only these, W(Gd) and pilot checks 1 and 2,
never SD, the other HW pairs, DU or Q3:
  RC(Gd)   r1 against r2 on Gd;
  DET      r1 on G against r1 on Gd: AlphaFold3's GPU defaults against
           deterministic ops on the same hardware;
  HW(V6,Gd), HW(Gd,C)  as HW, with HW - RC like the other hardware pairs.
Statistics per contrast: flip rate of top-1, flip rate of top-1 success,
flips that matter (top-1 changes and the two chosen structures differ in
success), median and 90th percentile of M4 (ligand RMSD after pocket
superposition, protein C-alpha RMSD after global superposition) and M5
(absolute differences of ranking_score, mean pLDDT, iptm), and the fraction
of sample pairs with bit-identical coordinates.
Uncertainty: 95% percentile intervals from a paired bootstrap over targets
(resample targets with replacement, --replicates 10000, --seed 2026), and
for each hardware pair HW(P,Q) - RC(P) of each flip rate on the same
resampled targets (also HW - RC(Q), labelled secondary). For HWh(P,Q) the
same, against RC(P) and RC(Q) of the default-precision arms (the arms with
"h" run r1 only), for section 11's reading rule.
Q3: for n = 1..N seeds, the overall top-1 over the first n seeds, its success
rate per condition (on its own targets) and the pairwise differences on each
pair's common targets, with bootstrap intervals; the answer is the smallest n
where every interval lies within +-2 percentage points, or "not reached".
Pilot checks (section 6 and the 2026-10-11 rows of section 10): 1, exit
codes; 2, per arm with a warm rerun (V5, V6, G, Gd), read in order: 2a every
fresh process started from an empty cache and compiled the model, 2b every
warm process loaded a key its source fresh process wrote, then bit-identity
of the warm rerun with its r1 run: every file of the AlphaFold3 output
folder, byte for byte, except the mmCIF timestamp line
(_ma_model_list.model_group_name); a missing file (FETCH=light leaves the
full confidences and data.json in the bucket) fails it; 3, the top-1 success
on V6 against the 50% sanity threshold. Also reported: whether M1 was
computed for every sample, and the M2 cross-check (section 10, 2026-10-12): every
run's overall top-1 by full-precision ranking_score is the sample AlphaFold3 wrote as its top-level model, with
no tie at the top.

Outputs in --out: contrasts.csv, hw_minus_rc.csv, pairs_seed.csv,
pairs_sample.csv, q3.csv, checks.csv, excluded.csv (section 8 counts) and report.md.
"""

import argparse
import functools
import itertools
import json
import pathlib
import re
import sys

import numpy as np
import pandas as pd

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / 'analysis'))
sys.path.insert(0, str(REPO / 'harness'))

PRIMARY = [('V6', 'G'), ('V5', 'V6'), ('G', 'C')]
ORDER = ['V5', 'V6', 'G', 'GT', 'C']
# Section 11: (kind, condition A, condition B), A the first-named side (M4 reference).
PRECISION_CONTRASTS = [('PR', 'V6', 'V6h'), ('PR', 'G', 'Gh'), ('HWh', 'V6h', 'Gh'), ('HWh', 'Gh', 'C')]
# 2026-10-11 amendment (section 10): the deterministic-ops arm, kept out of the section 5 loops, and its
# secondary contrasts besides RC(Gd) and W(Gd): (kind, condition A, condition B), A the first-named side.
DET_ARMS = ('Gd',)
DET_CONTRASTS = [('DET', 'G', 'Gd'), ('HW', 'V6', 'Gd'), ('HW', 'Gd', 'C')]
WARM_ARMS = ('V5', 'V6', 'G', 'Gd')   # pilot check 2 covers every arm with a warm rerun
TIMESTAMP_TAG = b'_ma_model_list.model_group_name'   # AlphaFold3 version and time of writing
FLIP_STATS = ['flip_top1', 'flip_success', 'flip_matters']
SAMPLE_STATS = ['m4_ligand_rmsd', 'm4_ca_rmsd_global', 'm5_ranking_score', 'm5_mean_plddt', 'm5_iptm']
SANITY_THRESHOLD = 0.5
Q3_BAND = 0.02


def rank(condition):
  base = condition.split('_')[0]
  return (ORDER.index(base) if base in ORDER else len(ORDER), condition)


def arm(condition):
  """The condition's code without the libtpu suffix of --condition platform_libtpu: V6h_0.0.43.2 -> V6h."""
  return condition.split('_')[0]


def precision_base(condition):
  """For a condition with a non-default matmul precision (V6h, Gh, V6[float32]), its section 3 code; else None."""
  a = arm(condition)
  if '[' in a:
    return a.split('[')[0]
  if a.endswith('h') and a[:-1] in ORDER:
    return a[:-1]
  return None


def default_precision(condition):
  return precision_base(condition) is None


def section5_arm(condition):
  """The arms of the section 5 loops (RC, W, SD, HW, DU, Q3): default precision, not an added arm such as Gd."""
  return default_precision(condition) and arm(condition) not in DET_ARMS


def stack(condition):
  """The libtpu suffix of --condition platform_libtpu ('' with --condition platform)."""
  return condition[len(arm(condition)):]


def unh(condition):
  """V6h -> V6, Gh_none -> G_none; other conditions unchanged."""
  base = precision_base(condition)
  return condition if base is None else base + condition[len(arm(condition)):]


@functools.lru_cache(maxsize=None)
def coords_text(path):
  from compare_samples import read_coords
  return tuple(read_coords(REPO / path)[2])


@functools.lru_cache(maxsize=None)
def divergence(target, path_a, path_b):
  import score_ligand
  return score_ligand.pair_divergence(target, REPO / path_a, REPO / path_b)


def load(args):
  d = pd.read_csv(args.samples)
  d = d[(d.num_recycles == 10) & (d.num_diffusion_samples == 5)]
  if args.labels:
    d = d[d.label.isin(args.labels.split(','))]
  if args.include:
    d = d[d.session.str.contains(args.include, regex=True)]
  if args.exclude:
    d = d[~d.session.str.contains(args.exclude, regex=True)]
  d = d.copy()
  d['condition'] = d.platform if args.condition == 'platform' else d.platform + '_' + d.libtpu.fillna('none').astype(str)
  ok = d[d['sample'].notna()].copy()
  ok['sample'] = ok['sample'].astype(int)
  dup = ok.duplicated(['condition', 'target', 'seed', 'repetition', 'sample'], keep=False)
  if dup.any():
    clash = ok[dup].groupby(['condition', 'target', 'seed', 'repetition']).session.unique()
    raise SystemExit('Same (condition, target, seed, repetition) in several sessions; select with '
                     '--labels/--include/--exclude:\n' + clash.head(10).to_string())
  return d, ok


def seed_units(ok):
  """{(condition, repetition, target, seed): DataFrame of its samples}."""
  return {k: g.set_index('sample').sort_index() for k, g in ok.groupby(['condition', 'repetition', 'target', 'seed'])}


def top(g):
  order = g.sort_values(['ranking_score'], ascending=False, kind='mergesort')
  first = order.index[0]
  margin = float(order.ranking_score.iloc[0] - order.ranking_score.iloc[1]) if len(order) > 1 else np.nan
  return first, bool(order.m1_success.iloc[0] == True), margin  # noqa: E712 - NaN counts as not a success


ROLE = {'RC': 'recompilation', 'W': 'warm-cache check', 'SD': 'seed yardstick', 'DU': 'default use',
        'ST': 'software stack', 'PR': 'precision (section 11)', 'HWh': 'matched hardware (section 11)',
        'DET': 'deterministic ops (2026-10-11)'}


def role(kind, primary):
  """The contrast's role in the outputs (column role). Only a section 5 primary hardware pair is "primary pair";
  RC, W and SD of an added arm (Gd) are marked secondary."""
  if kind == 'HW':
    return 'primary pair' if primary else 'secondary pair'
  r = ROLE.get(kind, kind)
  return r if primary or kind not in ('RC', 'W', 'SD') else r + ', secondary arm'


def pair_rows(name, kind, a_key, b_key, units, primary, contrast_pairs, seed_rows, sample_rows):
  """Adds one (target, seed) pair and its sample pairs to the output lists."""
  ga, gb = units[a_key], units[b_key]
  ta, sa, ma = top(ga)
  tb, sb, mb = top(gb)
  target = a_key[2]
  seed_rows.append({'contrast': name, 'kind': kind, 'role': role(kind, primary), 'target': target, 'seed_a': a_key[3],
                    'seed_b': b_key[3], 'top1_a': ta, 'top1_b': tb, 'success_a': sa, 'success_b': sb,
                    'margin_a': ma, 'margin_b': mb, 'flip_top1': ta != tb, 'flip_success': sa != sb,
                    'flip_matters': ta != tb and sa != sb})
  for k in sorted(set(ga.index) & set(gb.index)):
    ra, rb = ga.loc[k], gb.loc[k]
    ident = isinstance(ra.model_path, str) and isinstance(rb.model_path, str) and \
        coords_text(ra.model_path) == coords_text(rb.model_path)
    if ident:
      m4 = {'ligand_rmsd': 0.0, 'ca_rmsd_global': 0.0, 'error': None}
    elif isinstance(ra.model_path, str) and isinstance(rb.model_path, str):
      m4 = divergence(target, ra.model_path, rb.model_path)
    else:
      m4 = {'ligand_rmsd': None, 'ca_rmsd_global': None, 'error': 'model missing'}
    sample_rows.append({'contrast': name, 'kind': kind, 'role': role(kind, primary), 'target': target, 'seed_a': a_key[3],
                        'seed_b': b_key[3], 'sample': k, 'bit_identical': ident,
                        'm4_ligand_rmsd': m4['ligand_rmsd'], 'm4_ca_rmsd_global': m4['ca_rmsd_global'],
                        'm4_error': m4['error'],
                        'm5_ranking_score': abs(ra.ranking_score - rb.ranking_score),
                        'm5_mean_plddt': abs(ra.mean_plddt - rb.mean_plddt),
                        'm5_iptm': abs(ra.iptm - rb.iptm)})
  contrast_pairs.setdefault(name, (kind, primary))


def build_pairs(ok):
  units = seed_units(ok)
  all_conds = sorted(ok.condition.unique())
  conds = [c for c in all_conds if section5_arm(c)]
  reps = {c: sorted(ok[ok.condition == c].repetition.unique()) for c in all_conds}
  seed_rows, sample_rows, contrasts, excluded = [], [], {}, []

  def keys(c, rep):
    return {(k[2], k[3]): k for k in units if k[0] == c and k[1] == rep}

  def add(name, kind, ka, kb, primary):
    common = sorted(set(ka) & set(kb))
    excluded.append({'contrast': name, 'pairs_used': len(common),
                     'excluded_only_a': len(set(ka) - set(kb)), 'excluded_only_b': len(set(kb) - set(ka))})
    for ts in common:
      pair_rows(name, kind, ka[ts], kb[ts], units, primary, contrasts, seed_rows, sample_rows)

  for c in conds:
    if 'r1' in reps[c] and 'r2' in reps[c]:
      add(f'RC({c})', 'RC', keys(c, 'r1'), keys(c, 'r2'), True)
    if 'r1' in reps[c] and 'w' in reps[c]:
      add(f'W({c})', 'W', keys(c, 'r1'), keys(c, 'w'), True)
    if 'r1' in reps[c]:
      r1 = keys(c, 'r1')
      by_target = {}
      for (t, s), k in r1.items():
        by_target.setdefault(t, []).append(s)
      ka, kb = {}, {}
      for t, seeds in by_target.items():
        for s1, s2 in itertools.combinations(sorted(seeds), 2):
          ka[(t, (s1, s2))], kb[(t, (s1, s2))] = r1[(t, s1)], r1[(t, s2)]
      if ka:
        add(f'SD({c})', 'SD', ka, kb, True)
  for p, q in itertools.combinations(sorted(conds, key=rank), 2):
    if 'r1' in reps[p] and 'r1' in reps[q]:
      base = lambda c: c.split('_')[0]  # noqa: E731
      primary = (base(p), base(q)) in PRIMARY or (base(q), base(p)) in PRIMARY
      a, b = (p, q) if (base(p), base(q)) in PRIMARY or not primary else (q, p)
      if base(a) == base(b):
        name, kind = f'ST({a},{b})', 'ST'
      elif {base(a), base(b)} == {'G', 'GT'}:
        name, kind = 'DU', 'DU'
      else:
        name, kind = f'HW({a},{b})', 'HW'
      add(name, kind, keys(a, 'r1'), keys(b, 'r1'), primary or kind == 'DU')
  # Section 11, secondary: the precision-matched arms enter only these.
  for kind, code_a, code_b in PRECISION_CONTRASTS:
    for a in all_conds:
      for b in all_conds:
        if arm(a) != code_a or arm(b) != code_b or 'r1' not in reps[a] or 'r1' not in reps[b]:
          continue
        if kind == 'PR' and unh(b) != a:   # same platform (and libtpu, with --condition platform_libtpu)
          continue
        name = f'PR({a})' if kind == 'PR' else f'HWh({unh(a)},{unh(b)})'
        add(name, kind, keys(a, 'r1'), keys(b, 'r1'), False)
  # 2026-10-11 amendment, secondary: Gd enters only these (after every earlier contrast, which stay unchanged).
  for c in all_conds:
    if arm(c) in DET_ARMS and 'r1' in reps[c]:
      if 'r2' in reps[c]:
        add(f'RC({c})', 'RC', keys(c, 'r1'), keys(c, 'r2'), False)
      if 'w' in reps[c]:
        add(f'W({c})', 'W', keys(c, 'r1'), keys(c, 'w'), False)
  for kind, code_a, code_b in DET_CONTRASTS:
    for a, b in itertools.product(all_conds, all_conds):
      if arm(a) != code_a or arm(b) != code_b or 'r1' not in reps[a] or 'r1' not in reps[b]:
        continue
      if kind == 'DET':
        if stack(a) != stack(b):   # same platform and software stack, only the XLA flag differs
          continue
        name = 'DET' if not stack(a) else f'DET({a},{b})'
      else:
        name = f'HW({a},{b})'
      add(name, kind, keys(a, 'r1'), keys(b, 'r1'), False)
  return pd.DataFrame(seed_rows), pd.DataFrame(sample_rows), contrasts, pd.DataFrame(excluded)


def boot_ci(groups, stat, rng, reps):
  """95% percentile interval of stat over targets resampled with replacement; groups: list of arrays."""
  groups = [np.asarray(g, dtype=float) for g in groups if len(g)]
  if not groups:
    return np.nan, np.nan
  idx = rng.integers(0, len(groups), size=(reps, len(groups)))
  vals = np.array([stat(np.concatenate([groups[i] for i in row])) for row in idx])
  vals = vals[~np.isnan(vals)]
  return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))) if len(vals) else (np.nan, np.nan)


def p90(x):
  return float(np.percentile(x, 90)) if len(x) else np.nan


def med(x):
  return float(np.median(x)) if len(x) else np.nan


def mean(x):
  return float(np.mean(x)) if len(x) else np.nan


def summarize(seed_pairs, sample_pairs, contrasts, args):
  rows = []
  for name, (kind, primary) in contrasts.items():
    sp = seed_pairs[seed_pairs.contrast == name]
    sm = sample_pairs[sample_pairs.contrast == name]
    targets = sorted(sp.target.unique())
    for st in FLIP_STATS:
      if kind == 'SD' and st != 'flip_success':
        note = 'sample indices of different seeds are unrelated draws'
      else:
        note = ''
      groups = [sp[sp.target == t][st].astype(float).values for t in targets]
      lo, hi = boot_ci(groups, mean, np.random.default_rng(args.seed), args.replicates)
      rows.append({'contrast': name, 'kind': kind, 'role': role(kind, primary), 'statistic': st, 'n': len(sp),
                   'n_targets': len(targets), 'estimate': mean(sp[st].astype(float).values),
                   'ci_low': lo, 'ci_high': hi, 'note': note})
    for st in SAMPLE_STATS:
      vals = sm[st].dropna().astype(float)
      for fn, fname in ((med, 'median'), (p90, 'p90')):
        groups = [sm[(sm.target == t)][st].dropna().astype(float).values for t in targets]
        lo, hi = boot_ci(groups, fn, np.random.default_rng(args.seed), args.replicates)
        rows.append({'contrast': name, 'kind': kind, 'role': role(kind, primary), 'statistic': f'{st}_{fname}',
                     'n': len(vals), 'n_targets': len(targets), 'estimate': fn(vals.values),
                     'ci_low': lo, 'ci_high': hi, 'note': ''})
    groups = [sm[sm.target == t].bit_identical.astype(float).values for t in targets]
    lo, hi = boot_ci(groups, mean, np.random.default_rng(args.seed), args.replicates)
    rows.append({'contrast': name, 'kind': kind, 'role': role(kind, primary), 'statistic': 'bit_identical_fraction',
                 'n': len(sm), 'n_targets': len(targets), 'estimate': mean(sm.bit_identical.astype(float).values),
                 'ci_low': lo, 'ci_high': hi, 'note': ''})
    m4err = sm.m4_error.dropna()
    if len(m4err):
      rows.append({'contrast': name, 'kind': kind, 'role': role(kind, primary), 'statistic': 'm4_errors', 'n': len(m4err),
                   'n_targets': len(targets), 'estimate': np.nan, 'ci_low': np.nan, 'ci_high': np.nan,
                   'note': '; '.join(sorted(set(m4err)))[:300]})
  return pd.DataFrame(rows)


def hw_minus_rc(seed_pairs, contrasts, args):
  rows = []
  for name, (kind, primary) in contrasts.items():
    if kind not in ('HW', 'HWh'):
      continue
    # HWh(P,Q) names the default-precision codes, so RC(P) and RC(Q) are those of the default arms.
    p, q = re.match(r'HWh?\((.+),(.+)\)', name).groups()
    for which, rc_platform in (('RC(P)', p), ('RC(Q), secondary', q)):
      rc = f'RC({rc_platform})'
      if rc not in contrasts:
        continue
      hw_sp, rc_sp = seed_pairs[seed_pairs.contrast == name], seed_pairs[seed_pairs.contrast == rc]
      targets = sorted(set(hw_sp.target) & set(rc_sp.target))
      for st in FLIP_STATS:
        hw_g = [hw_sp[hw_sp.target == t][st].astype(float).values for t in targets]
        rc_g = [rc_sp[rc_sp.target == t][st].astype(float).values for t in targets]
        est = mean(np.concatenate(hw_g)) - mean(np.concatenate(rc_g)) if targets else np.nan
        rng = np.random.default_rng(args.seed)
        diffs = []
        if targets:
          for row in rng.integers(0, len(targets), size=(args.replicates, len(targets))):
            diffs.append(mean(np.concatenate([hw_g[i] for i in row])) - mean(np.concatenate([rc_g[i] for i in row])))
        diffs = np.array(diffs)
        rows.append({'hardware': name, 'recompilation': rc, 'reference': which, 'role': role(kind, primary),
                     'statistic': st, 'n_targets': len(targets), 'estimate': est,
                     'ci_low': float(np.percentile(diffs, 2.5)) if len(diffs) else np.nan,
                     'ci_high': float(np.percentile(diffs, 97.5)) if len(diffs) else np.nan})
  return pd.DataFrame(rows)


def q3(ok, args):
  r1 = ok[(ok.repetition == 'r1') & ok.condition.map(section5_arm)]
  conds = sorted(r1.condition.unique(), key=rank)
  seeds = sorted(r1.seed.unique())
  rows, answer = [], None
  for n in range(1, len(seeds) + 1):
    use = seeds[:n]
    succ = {}
    for c in conds:
      for t, g in r1[(r1.condition == c) & (r1.seed.isin(use))].groupby('target'):
        if sorted(g.seed.unique()) != use:
          continue
        best = g.sort_values(['ranking_score'], ascending=False, kind='mergesort').iloc[0]
        succ[(c, t)] = float(best.m1_success == True)  # noqa: E712
    if not succ:
      continue
    for c in conds:
      own = [v for (c2, t), v in succ.items() if c2 == c]
      if own:
        rows.append({'n_seeds': n, 'condition': c, 'statistic': 'success_rate', 'n_targets': len(own),
                     'estimate': mean(own), 'ci_low': np.nan, 'ci_high': np.nan})
    within, compared = True, False
    for a, b in itertools.combinations(conds, 2):
      targets = sorted(t for c, t in succ if c == a and (b, t) in succ)  # each pair on its common targets
      if not targets:
        continue
      compared = True
      diff = np.array([succ[(a, t)] - succ[(b, t)] for t in targets])
      lo, hi = boot_ci([[x] for x in diff], mean, np.random.default_rng(args.seed), args.replicates)
      within &= (lo >= -Q3_BAND) and (hi <= Q3_BAND)
      rows.append({'n_seeds': n, 'condition': f'{a}-{b}', 'statistic': 'success_difference',
                   'n_targets': len(targets), 'estimate': float(diff.mean()), 'ci_low': lo, 'ci_high': hi})
    if compared:
      rows.append({'n_seeds': n, 'condition': 'all conditions', 'statistic': 'spread_max_minus_min',
                   'n_targets': np.nan,
                   'estimate': max(r['estimate'] for r in rows if r['n_seeds'] == n and r['statistic'] == 'success_rate')
                   - min(r['estimate'] for r in rows if r['n_seeds'] == n and r['statistic'] == 'success_rate'),
                   'ci_low': np.nan, 'ci_high': np.nan})
      if within and answer is None:
        answer = n
  return pd.DataFrame(rows), answer


def cache_checks(d):
  """Check 2's preconditions, from the cache evidence harness/run_af3.py records (collect.py columns).

  2a: every fresh process started from an empty persistent cache and compiled jit_apply_fn (section 2).
  2b: every warm process loaded jit_apply_fn from the cache, with a key its source process (r1, same
      condition, target and seed) wrote; only then does its bit-identity test the cache.
  """
  rows = []
  if 'model_cache_hit' not in d.columns:
    return rows
  procs = d.drop_duplicates(['session', 'target', 'seed'])
  for c, g in procs.groupby('condition'):
    fresh = g[g.repetition.astype(str).str.fullmatch(r'r\d+')]   # not w2, w3 (diagnostic warm loads)
    bad = fresh[(fresh.cache_files_before != 0) | (fresh.model_cache_hit == True)]  # noqa: E712
    rows.append({'check': '2a fresh runs started from an empty cache', 'condition': c,
                 'value': f'{len(fresh) - len(bad)}/{len(fresh)} processes',
                 'detail': ('pass' if bad.empty else 'FAIL: ' + '; '.join(
                     f'{r.target} {r.repetition} seed {r.seed}' for r in bad.itertuples()))[:300]})
    warm = g[g.repetition == 'w']
    if warm.empty:
      if arm(c) in WARM_ARMS:
        rows.append({'check': '2b warm rerun loaded its source executable', 'condition': c,
                     'value': '0/0 processes', 'detail': 'FAIL: no warm rerun'})
      continue
    source = {(r.target, r.seed): set(str(r.model_cache_written).split(';'))
              for r in fresh[fresh.repetition == 'r1'].itertuples() if isinstance(r.model_cache_written, str)}
    loaded = {r.Index for r in warm.itertuples()
              if r.model_cache_hit == True and r.model_cache_key in source.get((r.target, r.seed), set())}  # noqa: E712
    # Every warm process must pass, not one per target (two warm loads of a target count twice).
    missing = sorted(r.target for r in warm.itertuples() if r.Index not in loaded)
    rows.append({'check': '2b warm rerun loaded its source executable', 'condition': c,
                 'value': f'{len(loaded)}/{len(warm)} processes',
                 'detail': ('pass' if not missing else 'FAIL: ' + ', '.join(missing))[:300]})
  return rows


# What AlphaFold3 writes for a target and for each sample; FETCH=light leaves the full confidences and data.json in
# the results bucket, so check 2 needs them copied (read-only) for the r1 and w runs it compares.
TARGET_FILES = ('_model.cif', '_summary_confidences.json', '_confidences.json', '_ranking_scores.csv', '_data.json')
SAMPLE_FILES = ('_model.cif', '_summary_confidences.json', '_confidences.json')


def masked(path):
  """The file's bytes; in an mmCIF the timestamp line (TIMESTAMP_TAG and its value) is cut to the tag alone."""
  data = pathlib.Path(path).read_bytes()
  if not str(path).endswith('.cif'):
    return data
  return b'\n'.join(TIMESTAMP_TAG if line.split(None, 1)[:1] == [TIMESTAMP_TAG] else line
                    for line in data.split(b'\n'))


def outputs_identical(dir_a, dir_b, seed):
  """Bit-for-bit comparison of two AlphaFold3 output folders of one target (pilot check 2).

  Compares the target's files and its seed-<seed>_sample-* folders, every file byte for byte except the mmCIF
  timestamp line. A file expected (TARGET_FILES, SAMPLE_FILES) or present on one side only is missing.
  Returns {'status': 'identical' | 'differ' | 'missing', 'differ': [...], 'missing': [...]}.
  """
  def files(d):
    d = pathlib.Path(d)
    if not d.is_dir():
      return {}
    return {str(p.relative_to(d)): p for p in d.rglob('*') if p.is_file() and
            (p.parent == d or (p.parent.parent == d and p.parent.name.startswith(f'seed-{seed}_sample-')))}
  fa, fb = files(dir_a), files(dir_b)
  t = pathlib.Path(dir_a).name
  samples = {pathlib.Path(n).parts[0] for n in fa.keys() | fb.keys() if len(pathlib.Path(n).parts) == 2}
  expected = {f'{t}{s}' for s in TARGET_FILES} | {f'{sd}/{t}_{sd}{s}' for sd in samples for s in SAMPLE_FILES}
  missing = sorted((expected | fa.keys() | fb.keys()) - (fa.keys() & fb.keys()))
  if not samples:
    missing.insert(0, 'no sample folder')
  differ = sorted(n for n in fa.keys() & fb.keys() if masked(fa[n]) != masked(fb[n]))
  return {'status': 'differ' if differ else 'missing' if missing else 'identical', 'differ': differ, 'missing': missing}


def output_dir(rows):
  """The AlphaFold3 output folder of one target in one harness run, from its collect.py rows."""
  paths = rows.model_path.dropna() if 'model_path' in rows.columns else []
  if len(paths):
    return (REPO / paths.iloc[0]).parent.parent
  r = rows.iloc[0]
  return REPO / r.session / r.target / 'af3_output' / r.target


def warm_identity(d, c):
  """[(target, seed, outputs_identical result)]: each warm rerun of condition c against its r1 run."""
  g = d[d.condition == c]
  out = []
  for (t, s), wg in g[g.repetition == 'w'].groupby(['target', 'seed']):
    src = g[(g.repetition == 'r1') & (g.target == t) & (g.seed == s)]
    res = outputs_identical(output_dir(src), output_dir(wg), s) if len(src) else \
        {'status': 'missing', 'differ': [], 'missing': ['no r1 run']}
    out.append((t, s, res))
  return out


def check2(d, sample_pairs, cache_rows, c):
  """Check 2 for condition c, in order: 2a and 2b (cache_checks), bit-identity of all outputs, the overall verdict."""
  rows = [r for r in cache_rows if r['condition'] == c]
  ident = warm_identity(d, c)
  if not ident and arm(c) not in WARM_ARMS:
    return rows
  pre = {r['check'][:2]: r['detail'] == 'pass' for r in rows}
  gate = pre.get('2a', False) and pre.get('2b', False)
  n_ok = sum(res['status'] == 'identical' for _, _, res in ident)
  bad = [f'{t} seed {s} {res["status"]}: ' + ', '.join(res['differ'] or res['missing'])
         for t, s, res in ident if res['status'] != 'identical']
  bit_ok = bool(ident) and not bad
  detail = 'pass' if bit_ok else 'FAIL: ' + ('; '.join(bad) if bad else 'no warm rerun')
  if not gate:
    detail = 'not read (2a or 2b failed); ' + detail
  rows.append({'check': '2 bit-identity: warm rerun against r1, all outputs (mmCIF timestamp line ignored)',
               'condition': c, 'value': f'{n_ok}/{len(ident)} targets', 'detail': detail[:600]})
  w = sample_pairs[sample_pairs.contrast == f'W({c})'] if len(sample_pairs) else sample_pairs
  if len(w):
    rows.append({'check': '2 warm rerun coordinates identical to r1 (as written; M4)', 'condition': c,
                 'value': f'{int(w.bit_identical.sum())}/{len(w)} samples',
                 'detail': 'pass' if w.bit_identical.all() else 'FAIL'})
  failed = [k for k, v in (('2a', pre.get('2a', False)), ('2b', pre.get('2b', False)), ('bit-identity', bit_ok))
            if not v]
  rows.append({'check': '2 overall (2a, 2b, then bit-identity)', 'condition': c, 'value': '',
               'detail': 'pass' if not failed else 'FAIL: ' + ', '.join(failed)})
  return rows


def top_model_check(ok):
  """M2 cross-check (plan section 10, 2026-10-12), per condition, over every (run, target) with sample models.

  The overall top-1 of the run (highest full-precision ranking_score over its seeds and samples) must be the
  sample AlphaFold3 wrote as its top-level <target>_model.cif (equal apart from the mmCIF timestamp line), and
  no full-precision score may tie at the top, overall or within a seed. A missing full-precision score fails.
  """
  rows = []
  if 'model_path' not in ok.columns:
    return rows
  for c in sorted(ok.condition.unique(), key=rank):
    g = ok[(ok.condition == c) & ok.model_path.notna()]
    units, same, bad, ties, seed_ties, n_seed = 0, 0, [], [], 0, 0
    for (session, t), h in g.groupby(['session', 'target']):
      units += 1
      label = f'{t} {h.repetition.iloc[0]} {pathlib.Path(session).name}'
      if h.ranking_score.isna().any():
        bad.append(f'{label}: no full-precision ranking_score')
        continue
      best = h.ranking_score.max()
      if (h.ranking_score == best).sum() > 1:
        ties.append(label)
      for _, u in h.groupby('seed'):
        n_seed += 1
        seed_ties += int((u.ranking_score == u.ranking_score.max()).sum() > 1)
      top = h.sort_values(['ranking_score'], ascending=False, kind='mergesort').iloc[0]
      cif = REPO / top.model_path
      ref = cif.parent.parent / f'{t}_model.cif'
      if not ref.is_file():
        bad.append(f'{label}: no top-level model')
      elif masked(cif) == masked(ref):
        same += 1
      else:
        bad.append(f'{label}: top-1 seed {top.seed} sample {top["sample"]} is not the top-level model')
    rows.append({'check': "M2 overall top-1 is AlphaFold3's top-level model", 'condition': c,
                 'value': f'{same}/{units} (run, target)',
                 'detail': ('pass' if not bad else 'FAIL: ' + '; '.join(bad))[:600]})
    rows.append({'check': 'M2 no full-precision tie at the top', 'condition': c,
                 'value': f'{len(ties)} tie(s) in {units} (run, target); {seed_ties} in {n_seed} (run, target, seed)',
                 'detail': 'pass' if not ties and not seed_ties else 'FAIL: ' + '; '.join(ties)[:600]})
  return rows


def checks(d, ok, sample_pairs, args):
  rows = []
  for c, g in d.groupby('condition'):
    runs = g.drop_duplicates(['session', 'target'])
    failed = runs[runs.exit_code != 0]
    rows.append({'check': '1 exit codes', 'condition': c, 'value': f'{len(runs) - len(failed)}/{len(runs)} processes exit 0',
                 'detail': '; '.join(f'{r.target} {r.repetition} seed {r.seed}: {r.failure}' for r in failed.itertuples())[:300]})
  cache_rows = cache_checks(d)
  for c in sorted(d.condition.unique(), key=rank):
    rows += check2(d, sample_pairs, cache_rows, c)
  present = {arm(c) for c in d.condition.unique()}
  absent = [a for a in WARM_ARMS if a not in present]
  rows.append({'check': '2 every arm with a warm rerun is present', 'condition': ', '.join(WARM_ARMS),
               'value': f'{len(WARM_ARMS) - len(absent)}/{len(WARM_ARMS)} arms',
               'detail': 'pass' if not absent else 'FAIL: absent ' + ', '.join(absent)})
  rows += top_model_check(ok)
  if 'm1_error' in ok.columns:
    for c in sorted(ok.condition.unique(), key=rank):
      g = ok[ok.condition == c]
      err = g[g.m1_error.notna() | g.m1_success.isna()]   # m1_success is empty when collect.py ran --no_m1
      rows.append({'check': 'M1 computed for every sample', 'condition': c,
                   'value': f'{len(g) - len(err)}/{len(g)} samples',
                   'detail': 'pass' if err.empty else 'FAIL, targets: ' + ', '.join(sorted(set(err.target)))})
  v6 = [c for c in ok.condition.unique() if arm(c) == 'V6']
  for c in v6:
    r1 = ok[(ok.condition == c) & (ok.repetition == 'r1')]
    best = r1.sort_values('ranking_score', ascending=False, kind='mergesort').groupby('target').head(1)
    rate = float((best.m1_success == True).mean()) if len(best) else np.nan  # noqa: E712
    rows.append({'check': '3 success-rate sanity (overall top-1, r1)', 'condition': c,
                 'value': f'{rate:.0%} of {len(best)} targets',
                 'detail': 'pass' if rate >= SANITY_THRESHOLD else 'BELOW 50%: stop and investigate the inputs'})
    per_seed = r1.sort_values('ranking_score', ascending=False, kind='mergesort').groupby(['target', 'seed']).head(1)
    rate = float((per_seed.m1_success == True).mean()) if len(per_seed) else np.nan  # noqa: E712
    rows.append({'check': '3 success-rate sanity (top-1 per seed, r1; secondary)', 'condition': c,
                 'value': f'{rate:.0%} of {len(per_seed)} (target, seed)',
                 'detail': 'pass' if rate >= SANITY_THRESHOLD else 'BELOW 50%'})
  return pd.DataFrame(rows)


def fmt(x, pct=False):
  if x is None or (isinstance(x, float) and np.isnan(x)):
    return '-'
  return f'{100 * x:.1f}%' if pct else f'{x:.3g}'


def report(summary, hwrc, q3rows, q3answer, chk, excluded, args, out):
  lines = ['# Contrasts report', '',
           f'Source: `{pathlib.Path(args.samples).name}`; filters: labels={args.labels or "all"}, '
           f'include={args.include or "-"}, exclude={args.exclude or "-"}; condition={args.condition}; '
           f'bootstrap: {args.replicates} replicates over targets, seed {args.seed}. '
           'Definitions: analysis/contrasts.py and notes/analysis_plan.md sections 4, 5 and 11 '
           '(PR and HWh: section 11, secondary; RC(Gd), W(Gd), DET, HW(V6,Gd) and HW(Gd,C): the 2026-10-11 '
           'amendment in section 10, secondary).', '',
           '## Pilot checks (section 6)', '', '| check | condition | value | detail |', '|---|---|---|---|']
  lines += [f'| {r.check} | {r.condition} | {r.value} | {r.detail} |' for r in chk.itertuples()]
  lines += ['', '## Flip rates', '', '| contrast | role | statistic | n pairs | targets | estimate | 95% CI |',
            '|---|---|---|---|---|---|---|']
  for r in summary[summary.statistic.isin(FLIP_STATS + ['bit_identical_fraction'])].itertuples():
    lines.append(f'| {r.contrast} | {r.role} | {r.statistic} | {r.n} | {r.n_targets} | '
                 f'{fmt(r.estimate, True)} | {fmt(r.ci_low, True)} to {fmt(r.ci_high, True)} |'
                 + (f' ({r.note})' if r.note else ''))
  lines += ['', '## M4 and M5 (median and 90th percentile)', '',
            '| contrast | statistic | n | estimate | 95% CI |', '|---|---|---|---|---|']
  for r in summary[summary.statistic.str.startswith(('m4_', 'm5_'))].itertuples():
    lines.append(f'| {r.contrast} | {r.statistic} | {r.n} | {fmt(r.estimate)} | '
                 f'{fmt(r.ci_low)} to {fmt(r.ci_high)} |' + (f' {r.note}' if r.note else ''))
  lines += ['', '## Hardware minus recompilation', '']
  if len(hwrc):
    lines += ['| hardware | recompilation | statistic | estimate | 95% CI |', '|---|---|---|---|---|']
    lines += [f'| {r.hardware} | {r.recompilation} ({r.reference}) | {r.statistic} | {fmt(r.estimate, True)} | '
              f'{fmt(r.ci_low, True)} to {fmt(r.ci_high, True)} |' for r in hwrc.itertuples()]
  else:
    lines.append('No hardware contrast has a recompilation contrast (r2) to subtract.')
  lines += ['', '## Q3: seeds needed', '']
  if len(q3rows):
    lines += ['| seeds | condition | statistic | estimate | 95% CI |', '|---|---|---|---|---|']
    lines += [f'| {r.n_seeds} | {r.condition} | {r.statistic} | {fmt(r.estimate, True)} | '
              f'{fmt(r.ci_low, True)} to {fmt(r.ci_high, True)} |' for r in q3rows.itertuples()]
    lines += ['', f'Answer: {f"{q3answer} seed(s)" if q3answer else "not reached"} '
                  f'(every pairwise difference within +-{100 * Q3_BAND:.0f} percentage points).']
  else:
    lines.append('No condition has r1 runs for the same targets and seeds.')
  lines += ['', '## Pairs excluded (section 8)', '', '| contrast | used | only in A | only in B |', '|---|---|---|---|']
  lines += [f'| {r.contrast} | {r.pairs_used} | {r.excluded_only_a} | {r.excluded_only_b} |' for r in excluded.itertuples()]
  (out / 'report.md').write_text('\n'.join(lines) + '\n')


def main():
  ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
  ap.add_argument('--samples', required=True)
  ap.add_argument('--out')
  ap.add_argument('--labels')
  ap.add_argument('--include')
  ap.add_argument('--exclude')
  ap.add_argument('--condition', choices=('platform', 'platform_libtpu'), default='platform')
  ap.add_argument('--replicates', type=int, default=10000)
  ap.add_argument('--seed', type=int, default=2026)
  args = ap.parse_args()
  out = pathlib.Path(args.out or pathlib.Path(args.samples).parent / 'contrasts')
  out.mkdir(parents=True, exist_ok=True)
  d, ok = load(args)
  seed_pairs, sample_pairs, contrasts, excluded = build_pairs(ok)
  if seed_pairs.empty:
    raise SystemExit('No contrast has pairs with these filters.')
  summary = summarize(seed_pairs, sample_pairs, contrasts, args)
  hwrc = hw_minus_rc(seed_pairs, contrasts, args)
  q3rows, q3answer = q3(ok, args)
  chk = checks(d, ok, sample_pairs, args)
  summary.to_csv(out / 'contrasts.csv', index=False)
  hwrc.to_csv(out / 'hw_minus_rc.csv', index=False)
  seed_pairs.to_csv(out / 'pairs_seed.csv', index=False)
  sample_pairs.to_csv(out / 'pairs_sample.csv', index=False)
  q3rows.to_csv(out / 'q3.csv', index=False)
  chk.to_csv(out / 'checks.csv', index=False)
  excluded.to_csv(out / 'excluded.csv', index=False)
  report(summary, hwrc, q3rows, q3answer, chk, excluded, args, out)
  print(f'>> {len(contrasts)} contrasts, {len(seed_pairs)} (target, seed) pairs, {len(sample_pairs)} sample pairs; '
        f'report: {out / "report.md"}', flush=True)
  return 0


if __name__ == '__main__':
  sys.exit(main())
