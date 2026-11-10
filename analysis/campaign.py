"""Campaign simulation (notes/analysis_plan.md section 10, row A2 of drafts/section10_campaign_ranking_20261110.md;
label campaign): a follow-up analysis, outside the pre-registered study, descriptive, whatever it shows. No run and no
cloud resource: frozen outputs only.

    analysis/.venv/bin/python analysis/campaign.py --out results/analysis/campaign_<date> [--main <checkout>]
        [--replicates 10000] [--seed 2026]

Question: when a scheduler mixes device types within one campaign, what fraction of the campaign's reported
top-ranked samples changes, and by how many points does its benchmark success rate change, against the same campaign
on one device type?
Units: (a) a job is one (target, seed); its reported result is the seed's top-ranked sample (M2) and its M1 success,
from the pair's pairs_seed.csv rows; (b) a job is one target with every seed the pair shares; its reported result is
the sample with the highest full-precision ranking_score across those seeds and its M1 success, from samples.csv.
Policies (home device P, other device Q, all r1; POLICIES): (1) L4 and A100 mixed, P = A, Q = G, HW(G,A) of a100;
(2) A100 and H100 mixed, P = A, Q = H, HW(H,A) of h100; (3) CPU fallback, P = G, Q = C, HW(G,C) of cpu60, seeds 1
and 2; (4) cloud TPU and GPU, P = V6, Q = G, HW(V6,G) of the main study and of repl; (5) cloud TPU generations,
P = V5, Q = V6, HW(V5,V6) of the same two folders. Shares s of jobs on Q: SHARES (assumptions, not measurements).
Statistics per policy, unit and share: (i) changed, the fraction of jobs whose reported sample differs from the all-P
campaign (the pair's top-1 flip for a moved job, none for a job on P); (ii) success_changed, the fraction of jobs whose
reported success differs; (iii) success_points, the mixed campaign's success rate minus the all-P campaign's, in
percentage points. Point estimate: the expectation over assignments on the observed targets (s times the pair's rate,
or s times its success difference). Interval: 95% percentile interval over --replicates replicates, each resampling
targets with replacement and moving each job of the resampled campaign to Q with probability s, independently; one
numpy.random.default_rng(--seed) stream for targets and assignments.
Reference beside each policy: the same campaign rerun entirely on P in a second, separately compiled process (RC(P)
r1 against r2, same unit, same targets and seeds): RC(A) for 1 and 2, RC(G) on seeds 1 and 2 for 3, RC(V6) for 4 (and
RC(G), the rerun of Q, printed beside it), RC(V5) for 5. C ran r1 only: the CPU has no rerun reference. The same
three statistics, estimate the observed rate (every job reruns), interval from the target bootstrap of that stream.
Checks: the script stops with a message if an input file, an expected contrast, an expected (platform, repetition) in
samples.csv, or the samples of a (target, seed) a pair needs is missing, or if a (target, seed) appears twice.

Choices where the row is silent (simplest reading):
  1. Policies 4 and 5 are computed separately on each source (main study, 60 targets; repl, 72), source column;
     no pooling.
  2. The seeds a pair shares (unit b) are the seeds of the pair's pairs_seed.csv rows for that target (the named seeds
     only), so units a and b cover the same (target, seed) set; the reference uses the (target, seed) rows of RC(P)
     that are also rows of the HW pair, on both units.
  3. Unit (b) samples: each platform from the folder that holds it: A from a100 (also for policy 2: the h100
     samples.csv has H only), G, V5, V6 from the main study (repl's own for repl), H from h100, C from the main study
     (6 targets) and cpu60 (54) together, as HW(G,C) of cpu60 was built. Rows at AlphaFold3's defaults
     (num_recycles 10, num_diffusion_samples 5) of the folder's collect.py label, as every pair was built.
  4. ranking_score is parsed with round-trip precision; argmax ties (none expected) go to the lowest seed, then the
     lowest sample. A sample without a ranking_score ranks last.
  5. Statistic (ii) uses the pair's flip_success column (unit a); (iii) uses success of Q minus success of P (for RC,
     r2 minus r1). Success is true only if m1_success is true (missing counts as no success, section 4).
  6. RNG order: for each policy (1 to 5), each source, unit a then b, each share in SHARES: the target draw
     (replicates x targets) then one uniform per job of the resampled campaigns (also at s = 0 and 1); then, for each
     reference contrast of that source, unit a then b: the target draw only.
  7. Rates are job-weighted (total moved changes over total jobs of the resampled campaign).
  8. references.csv also carries source, role, n_targets and n_jobs, since policies 4 and 5 have two sources.
Outputs in --out: campaign.csv, references.csv, report.md, inputs.tsv (run.log: reproduce.sh).
"""
import argparse
import hashlib
import pathlib
import sys

import numpy as np
import pandas as pd

REPO = pathlib.Path(__file__).resolve().parent.parent

SHARES = [0.1, 0.25, 0.5, 1.0]   # fraction s of the campaign's jobs on Q: assumptions, not measurements
UNITS = ['a', 'b']
UNIT_TEXT = {'a': "one (target, seed); the seed's top-ranked sample",
             'b': 'one target with every seed the pair shares; the top-ranked sample across them'}
STATS = ['changed', 'success_changed', 'success_points']
SCALE = np.array([1.0, 1.0, 100.0])   # success_points in percentage points
SEEDS_1_5 = (1, 2, 3, 4, 5)

MAIN = 'results/analysis/main_20261014/primary_v2'
FOLDERS = {'main': MAIN, 'a100': 'results/analysis/a100_20261018', 'h100': 'results/analysis/h100_20261104',
           'cpu60': 'results/analysis/cpu60_20261018', 'repl': 'results/analysis/repl_20261106'}
PAIRS = {k: (f'{v}/contrasts/pairs_seed.csv' if k == 'main' else f'{v}/pairs_seed.csv') for k, v in FOLDERS.items()}
SAMPLES = {k: f'{v}/samples.csv' for k, v in FOLDERS.items()}
LABELS = {'main': 'main', 'a100': 'a100', 'h100': 'h100', 'cpu60': 'cpu60', 'repl': 'repl'}   # collect.py labels
SAMPLE_COLS = ['label', 'platform', 'repetition', 'target', 'seed', 'sample', 'ranking_score', 'm1_success',
               'num_recycles', 'num_diffusion_samples']
PAIR_COLS = ['contrast', 'target', 'seed_a', 'seed_b', 'success_a', 'success_b', 'flip_top1', 'flip_success']


def source(name, hw, seeds, n_targets, refs, samples):
  """One campaign of a policy: the pairs folder (name), the HW contrast, the seeds, the targets the row names, the
  references [(contrast, pairs folder, role)] and, per platform, the folders whose samples.csv hold it."""
  return {'name': name, 'hw': hw, 'seeds': seeds, 'n_targets_row': n_targets, 'refs': refs, 'samples': samples}


POLICIES = [
    {'id': 1, 'name': 'L4 and A100 mixed', 'P': 'A', 'Q': 'G',
     'sources': [source('a100', 'HW(G,A)', SEEDS_1_5, 60, [('RC(A)', 'a100', 'RC(P)')],
                        {'A': ['a100'], 'G': ['main']})]},
    {'id': 2, 'name': 'A100 and H100 mixed', 'P': 'A', 'Q': 'H',
     'sources': [source('h100', 'HW(H,A)', SEEDS_1_5, 60, [('RC(A)', 'h100', 'RC(P)')],
                        {'A': ['a100'], 'H': ['h100']})]},
    {'id': 3, 'name': 'CPU fallback', 'P': 'G', 'Q': 'C',
     'sources': [source('cpu60', 'HW(G,C)', (1, 2), 60, [('RC(G)', 'main', 'RC(P)')],
                        {'G': ['main'], 'C': ['main', 'cpu60']})]},
    {'id': 4, 'name': 'cloud TPU and GPU', 'P': 'V6', 'Q': 'G',
     'sources': [source(s, 'HW(V6,G)', SEEDS_1_5, n, [('RC(V6)', s, 'RC(P)'), ('RC(G)', s, 'RC(Q), printed beside')],
                        {'V6': [s], 'G': [s]}) for s, n in (('main', 60), ('repl', 72))]},
    {'id': 5, 'name': 'cloud TPU generations', 'P': 'V5', 'Q': 'V6',
     'sources': [source(s, 'HW(V5,V6)', SEEDS_1_5, n, [('RC(V5)', s, 'RC(P)')], {'V5': [s], 'V6': [s]})
                 for s, n in (('main', 60), ('repl', 72))]},
]


def policy_label(pol):
  return f'({pol["id"]}) {pol["name"]}: P = {pol["P"]}, Q = {pol["Q"]}'


def truth(x):
  """True only where the value is true (bool True, 'True', 1); missing or anything else is False."""
  return pd.Series(x).astype(str).str.strip().str.lower().isin(['true', '1', '1.0']).values


def sha256(path):
  h = hashlib.sha256()
  with open(path, 'rb') as f:
    for block in iter(lambda: f.read(1 << 20), b''):
      h.update(block)
  return h.hexdigest()


class Inputs:
  """Reads the inputs under --main once each and records the SHA-256 of every file read."""

  def __init__(self, root):
    self.root = pathlib.Path(root)
    self.hashes = {}
    self.cache = {}

  def path(self, rel):
    p = self.root / rel
    if not p.is_file():
      raise SystemExit(f'missing input: {rel} (under --main {self.root})')
    if rel not in self.hashes:
      self.hashes[rel] = sha256(p)
    return p

  def pairs(self, name):
    rel = PAIRS[name]
    if rel not in self.cache:
      d = pd.read_csv(self.path(rel), dtype={'contrast': str, 'target': str}, low_memory=False)
      miss = [c for c in PAIR_COLS if c not in d.columns]
      if miss:
        raise SystemExit(f'{rel}: columns missing: {miss}')
      self.cache[rel] = d
    return self.cache[rel]

  def samples(self, name):
    rel = SAMPLES[name]
    if rel not in self.cache:
      p = self.path(rel)
      head = pd.read_csv(p, nrows=0).columns
      miss = [c for c in SAMPLE_COLS if c not in head]
      if miss:
        raise SystemExit(f'{rel}: columns missing: {miss}')
      d = pd.read_csv(p, usecols=SAMPLE_COLS, dtype={'label': str, 'platform': str, 'repetition': str,
                                                     'target': str, 'm1_success': str},
                      float_precision='round_trip', low_memory=False)
      d = d[(d.label == LABELS[name]) & (d.num_recycles == 10) & (d.num_diffusion_samples == 5)
            & d['sample'].notna()].copy()
      d['seed'] = d.seed.astype(int)
      d['sample'] = d['sample'].astype(int)
      d['success'] = truth(d.m1_success)
      self.cache[rel] = d[['platform', 'repetition', 'target', 'seed', 'sample', 'ranking_score', 'success']]
    return self.cache[rel]


def sides(contrast, P):
  """((platform, repetition) of the all-P campaign, of the other campaign, pairs column side of the first). In
  HW(X,Y) side a is X r1 and side b is Y r1; in RC(X) side a is X r1 and side b is X r2."""
  kind, inner = contrast.split('(')[0], contrast[contrast.index('(') + 1:-1].split(',')
  if kind == 'RC' and len(inner) == 1:
    return (inner[0], 'r1'), (inner[0], 'r2'), 'a'
  if kind == 'HW' and len(inner) == 2 and P in inner:
    x, y = inner
    return ((x, 'r1'), (y, 'r1'), 'a') if P == x else ((y, 'r1'), (x, 'r1'), 'b')
  raise SystemExit(f'{contrast}: not an HW pair holding {P} or an RC contrast')


def pair_rows(pairs, contrast, seeds, where):
  r = pairs[pairs.contrast == contrast]
  if r.empty:
    raise SystemExit(f'{where}: expected contrast {contrast} is missing')
  if (r.seed_a != r.seed_b).any():
    raise SystemExit(f'{where}: {contrast} pairs different seeds')
  r = r[r.seed_a.astype(int).isin(seeds)]
  if r.empty:
    raise SystemExit(f'{where}: {contrast} has no row on seeds {list(seeds)}')
  if r.duplicated(['target', 'seed_a']).any():
    raise SystemExit(f'{where}: {contrast} has a (target, seed) twice')
  return r


def unit_a(rows, first):
  """One job per (target, seed): changed (the pair's top-1 flip), success_changed (the pair's success flip) and the
  success difference (other side minus the all-P side)."""
  other = 'b' if first == 'a' else 'a'
  return pd.DataFrame({'target': rows.target.astype(str).values, 'seed': rows.seed_a.astype(int).values,
                       'changed': truth(rows.flip_top1).astype(float),
                       'success_changed': truth(rows.flip_success).astype(float),
                       'success_points': truth(rows['success_' + other]).astype(float)
                       - truth(rows['success_' + first]).astype(float)})


def top_across_seeds(samples, keys, where):
  """Per target, the sample with the highest full-precision ranking_score among every sample of the target's seeds in
  keys (target, seed); ties to the lowest seed, then sample. Every (target, seed) of keys must have samples."""
  m = samples.merge(keys[['target', 'seed']], on=['target', 'seed'])
  have = set(map(tuple, m[['target', 'seed']].drop_duplicates().values))
  lack = [k for k in map(tuple, keys[['target', 'seed']].values) if k not in have]
  if lack:
    raise SystemExit(f'{where}: no samples for {len(lack)} (target, seed), e.g. {lack[:5]}')
  o = m.sort_values(['target', 'ranking_score', 'seed', 'sample'], ascending=[True, False, True, True],
                    na_position='last', kind='mergesort').drop_duplicates('target')
  return o.set_index('target')[['seed', 'sample', 'success']]


def unit_b(get, first, second, keys, where):
  """One job per target: the top-1 across its seeds on the all-P side (first) and on the other side (second)."""
  f = top_across_seeds(get(first), keys, f'{where}, {first[0]} {first[1]}')
  s = top_across_seeds(get(second), keys, f'{where}, {second[0]} {second[1]}').loc[f.index]
  return pd.DataFrame({'target': f.index.values,
                       'changed': ((f.seed != s.seed) | (f['sample'] != s['sample'])).astype(float).values,
                       'success_changed': (f.success != s.success).astype(float).values,
                       'success_points': s.success.astype(float).values - f.success.astype(float).values})


def stack(jobs):
  """(values (jobs, 3) grouped by target in sorted order, jobs per target)."""
  jobs = jobs.sort_values('target', kind='mergesort')
  counts = jobs.groupby('target', sort=True).size().values
  return jobs[STATS].to_numpy(dtype=float), counts.astype(np.int64)


def draw(values, counts, share, rng, reps):
  """The three statistics per replicate (reps, 3). Each replicate resamples the targets with replacement (cluster);
  with share given, each job of the resampled campaign moves to Q with probability share, independently (one uniform
  per job, drawn after the targets); with share None every job counts (the rerun reference)."""
  T = len(counts)
  starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
  idx = rng.integers(0, T, size=(reps, T)).ravel()
  n = counts[idx]
  total = int(n.sum())
  rep = np.repeat(np.repeat(np.arange(reps), T), n)
  jobs = np.arange(total) + np.repeat(starts[idx] - (np.cumsum(n) - n), n)
  moved = (rng.random(total) < share) if share is not None else np.ones(total, dtype=bool)
  w = values[jobs] * moved[:, None]
  sums = np.stack([np.bincount(rep, weights=w[:, k], minlength=reps) for k in range(3)], axis=1)
  return sums / np.bincount(rep, minlength=reps)[:, None] * SCALE


def summarize(values, counts, share, rng, reps):
  """[(statistic, estimate, ci_low, ci_high)]: estimate the expectation over assignments on the observed targets."""
  est = (1.0 if share is None else share) * values.mean(axis=0) * SCALE
  lo, hi = np.percentile(draw(values, counts, share, rng, reps), [2.5, 97.5], axis=0)
  return [(st, float(est[k]), float(lo[k]), float(hi[k])) for k, st in enumerate(STATS)]


def getter(inputs, src):
  """(platform, repetition) -> the samples of that condition from the folders the policy names for the platform."""
  def get(cond):
    platform, rep = cond
    names = src['samples'].get(platform)
    if not names:
      raise SystemExit(f'{src["hw"]}: no samples.csv named for platform {platform}')
    d = pd.concat([inputs.samples(n) for n in names], ignore_index=True)
    d = d[(d.platform == platform) & (d.repetition == rep)]
    if d.empty:
      raise SystemExit(f'expected platform {platform} repetition {rep} is missing in '
                       + ', '.join(SAMPLES[n] for n in names))
    if d.duplicated(['target', 'seed', 'sample']).any():
      raise SystemExit(f'platform {platform} repetition {rep}: a (target, seed, sample) twice in '
                       + ', '.join(SAMPLES[n] for n in names))
    return d
  return get


def check(inputs):
  """Before any computation: every input file, contrast and (platform, repetition) the row names is present."""
  for pol in POLICIES:
    for src in pol['sources']:
      get = getter(inputs, src)
      for contrast, folder in [(src['hw'], src['name'])] + [(c, f) for c, f, _ in src['refs']]:
        pair_rows(inputs.pairs(folder), contrast, src['seeds'], PAIRS[folder])
        first, second, _ = sides(contrast, pol['P'])
        get(first)
        get(second)


def analyse(root, reps, seed):
  """(campaign rows, reference rows, Inputs). One rng stream in the order of the module docstring, choice 6."""
  inputs = Inputs(root)
  check(inputs)
  rng = np.random.default_rng(seed)
  camp, refs = [], []
  for pol in POLICIES:
    for src in pol['sources']:
      where = f'{policy_label(pol)}, {src["name"]}'
      get = getter(inputs, src)
      hw = pair_rows(inputs.pairs(src['name']), src['hw'], src['seeds'], PAIRS[src['name']])
      first, second, side = sides(src['hw'], pol['P'])
      ua = unit_a(hw, side)
      keys = ua[['target', 'seed']]
      ub = unit_b(get, first, second, keys, where)
      for unit, jobs in (('a', ua), ('b', ub)):
        values, counts = stack(jobs)
        for share in SHARES:
          for st, est, lo, hi in summarize(values, counts, share, rng, reps):
            camp.append({'policy': policy_label(pol), 'unit': unit, 'share': share, 'statistic': st, 'estimate': est,
                         'ci_low': lo, 'ci_high': hi, 'n_targets': len(counts), 'n_jobs': int(counts.sum()),
                         'source': FOLDERS[src['name']], 'contrast': src['hw'],
                         'n_targets_row': src['n_targets_row']})
      for contrast, folder, role in src['refs']:
        rr = pair_rows(inputs.pairs(folder), contrast, src['seeds'], PAIRS[folder])
        rr = rr.merge(keys.rename(columns={'seed': 'seed_a'}), on=['target', 'seed_a'])   # the HW pair's seeds
        if rr.empty:
          raise SystemExit(f'{where}: {contrast} has no (target, seed) of {src["hw"]}')
        rfirst, rsecond, rside = sides(contrast, pol['P'])
        ra = unit_a(rr, rside)
        rb = unit_b(get, rfirst, rsecond, ra[['target', 'seed']], f'{where}, {contrast}')
        for unit, jobs in (('a', ra), ('b', rb)):
          values, counts = stack(jobs)
          for st, est, lo, hi in summarize(values, counts, None, rng, reps):
            refs.append({'policy': policy_label(pol), 'unit': unit, 'contrast': contrast, 'statistic': st,
                         'estimate': est, 'ci_low': lo, 'ci_high': hi, 'source': FOLDERS[folder], 'role': role,
                         'n_targets': len(counts), 'n_jobs': int(counts.sum())})
  return pd.DataFrame(camp), pd.DataFrame(refs), inputs


def fmt(st, x):
  if x != x:
    return 'n/a'
  return f'{x:+.2f}' if st == 'success_points' else f'{x:.3f}'


def report(out, camp, refs, args):
  L = ['# Campaign simulation', '',
       'Follow-up analysis, outside the pre-registered study, descriptive, whatever it shows (section 10, row A2; '
       'label campaign). No run and no cloud resource: frozen outputs only. No reading rule; no decision depends on '
       'it.', '',
       f'Code: analysis/campaign.py; {args.replicates} replicates, seed {args.seed} (one '
       '`numpy.random.default_rng` stream); 95% percentile intervals over replicates that resample targets with '
       'replacement and move each job to Q with probability s. Point estimate: the expectation over assignments on '
       'the observed targets.', '',
       f'Shares s of jobs on Q: {", ".join(str(s) for s in SHARES)}. The shares are assumptions, not measurements.', '',
       'Units: ' + '; '.join(f'({u}) {t}' for u, t in UNIT_TEXT.items()) + '.', '',
       'Statistics: changed, the fraction of jobs whose reported sample differs from the all-P campaign; success '
       'changed, the fraction whose reported success (M1) differs; success points, the mixed campaign\'s success '
       'rate minus the all-P campaign\'s, in percentage points.', '']
  for pol in POLICIES:
    lab = policy_label(pol)
    L += [f'## Policy {lab}', '']
    for src in pol['sources']:
      c = camp[(camp.policy == lab) & (camp.source == FOLDERS[src['name']])]
      r = refs[(refs.policy == lab) & (refs.source.isin([FOLDERS[f] for _, f, _ in src['refs']]))
               & refs.contrast.isin([x for x, _, _ in src['refs']])]
      for unit in UNITS:
        cu = c[c.unit == unit]
        n = cu.iloc[0]
        L += [f'{src["hw"]} from `{FOLDERS[src["name"]]}`, unit ({unit}): {n.n_targets} targets (row: '
              f'{n.n_targets_row}), {n.n_jobs} jobs, seeds {", ".join(map(str, src["seeds"]))}.', '',
              '| s | changed | 95% CI | success changed | 95% CI | success points | 95% CI |',
              '|---|---|---|---|---|---|---|']
        for share in SHARES:
          row = cu[cu.share == share].set_index('statistic')
          L.append(f'| {share} | ' + ' | '.join(
              f'{fmt(st, row.loc[st, "estimate"])} | {fmt(st, row.loc[st, "ci_low"])} to '
              f'{fmt(st, row.loc[st, "ci_high"])}' for st in STATS) + ' |')
        for (contrast, role), ru in r[r.unit == unit].groupby(['contrast', 'role'], sort=False):
          row = ru.set_index('statistic')
          L.append(f'| {contrast} ({role}, rerun on the same targets and seeds; {int(row.n_jobs.iloc[0])} jobs) | '
                   + ' | '.join(f'{fmt(st, row.loc[st, "estimate"])} | {fmt(st, row.loc[st, "ci_low"])} to '
                                f'{fmt(st, row.loc[st, "ci_high"])}' for st in STATS) + ' |')
        L.append('')
    if pol['Q'] == 'C':
      L += ['C ran r1 only: the CPU has no rerun reference.', '']
  L += ['## Inputs', '', '| path (relative to --main) | SHA-256 |', '|---|---|']
  L += [f'| {p} | {h} |' for p, h in args.hashes.items()]
  (out / 'report.md').write_text('\n'.join(L) + '\n')


def main(argv=None):
  ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
  ap.add_argument('--main', default=str(REPO), help='checkout holding results/ (default: this repository root)')
  ap.add_argument('--out', required=True)
  ap.add_argument('--replicates', type=int, default=10000)
  ap.add_argument('--seed', type=int, default=2026)
  args = ap.parse_args(argv)
  out = pathlib.Path(args.out)
  camp, refs, inputs = analyse(args.main, args.replicates, args.seed)
  out.mkdir(parents=True, exist_ok=True)
  camp[['policy', 'unit', 'share', 'statistic', 'estimate', 'ci_low', 'ci_high', 'n_targets', 'n_jobs',
        'source']].to_csv(out / 'campaign.csv', index=False)
  refs[['policy', 'unit', 'contrast', 'statistic', 'estimate', 'ci_low', 'ci_high', 'source', 'role', 'n_targets',
        'n_jobs']].to_csv(out / 'references.csv', index=False)
  pd.DataFrame(list(inputs.hashes.items()), columns=['path', 'sha256']).to_csv(out / 'inputs.tsv', sep='\t',
                                                                                index=False)
  args.hashes = inputs.hashes
  report(out, camp, refs, args)
  print((out / 'report.md').read_text())
  return 0


if __name__ == '__main__':
  sys.exit(main())
