"""Follow-up experiment 2, where the divergence arises (notes/analysis_plan.md section 10, row of 2026-10-14;
descriptive, label embed).

    analysis/.venv/bin/python analysis/embed.py [--out results/analysis/embed_<date>]

Sessions: every session in results/af3/ whose runs have label embed (harness/plans/embed/): the 10 pilot targets,
seed 1, AlphaFold3's --save_embeddings, on V5, V6, G (r1 and r2) and Gd. Per target and for the pairs
V5-V6, V6-G, G r1-G r2 and G-Gd (A the first-named):
  single and pair embeddings (af3_output/<T>/seed-1_embeddings/<T>_seed-1_embeddings.npz, AlphaFold3's name, float16,
  read as float64):
    rel_l2       ||A - B|| / ||A|| over the whole array (Frobenius norm);
    token_max    the largest per-token relative difference: single, max_i ||a_i - b_i|| / ||a_i|| over tokens i
                 (384 channels); pair, max_ij ||a_ij - b_ij|| / ||a_ij|| over token pairs (i, j) (128 channels);
                 a token whose ||a|| is 0 counts 0 if b equals a there, else infinity;
    identical    the two arrays are bit-identical;
  from the same processes, for the 5 samples k (sample k of A against sample k of B): M4 C-alpha RMSD after
  global superposition and ligand RMSD after pocket superposition (analysis/score_ligand.pair_divergence) and the
  M5 absolute ranking_score difference (full precision, <T>_ranking_scores.csv): median and maximum over k, the
  number of bit-identical sample coordinates, and whether the top-1 (highest ranking_score) is the same sample.
Bit-identity with the main study: for each process, its outputs against the main study's run of the same
condition, repetition, target and seed (results/analysis/main_20261014/primary_v2/samples.csv): every sample's
mmCIF (timestamp line masked, analysis/contrasts.masked), summary confidences and the ranking scores, byte for
byte (saving embeddings may change compilation).
Summary per pair (not specified by the row; descriptive): over the targets, the median and maximum of each
measure, the number of targets whose single and pair embeddings are bit-identical, of bit-identical samples, and
of targets with the same top-1 sample.
Outputs in --out: embeddings.csv, structures.csv, summary.csv, versus_main.csv, report.md. --reuse reads
embeddings.csv and structures.csv from --out instead of recomputing them.
"""
import argparse
import json
import pathlib
import sys

import numpy as np
import pandas as pd

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / 'analysis'))
sys.path.insert(0, str(REPO / 'harness'))

AF3 = REPO / 'results' / 'af3'
MAIN_SAMPLES = REPO / 'results' / 'analysis' / 'main_20261014' / 'primary_v2' / 'samples.csv'
SEED = 1
SAMPLES = range(5)
# (condition, repetition) pairs, A first.
PAIRS = [(('V5', 'r1'), ('V6', 'r1')), (('V6', 'r1'), ('G', 'r1')), (('G', 'r1'), ('G', 'r2')),
         (('G', 'r1'), ('Gd', 'r1'))]
CODE = {('v5e', 'tpu_xla'): 'V5', ('v6e', 'tpu_xla'): 'V6', ('l4', 'l4_xla'): 'G', ('l4', 'l4_xla_det'): 'Gd'}


# ---------------------------------------------------------------- measures

def rel_l2(a, b):
  """||a - b|| / ||a|| over all elements, in float64."""
  a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
  if a.shape != b.shape:
    raise ValueError(f'shapes differ: {a.shape} and {b.shape}')
  na = np.linalg.norm(a)
  d = np.linalg.norm(a - b)
  return 0.0 if d == 0 else (d / na if na else np.inf)


def token_max(a, b):
  """Largest relative difference over tokens (single: axis 0) or token pairs (pair: axes 0, 1); the last axis is
  the channel axis."""
  a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
  if a.shape != b.shape:
    raise ValueError(f'shapes differ: {a.shape} and {b.shape}')
  num = np.linalg.norm(a - b, axis=-1)
  den = np.linalg.norm(a, axis=-1)
  with np.errstate(divide='ignore', invalid='ignore'):
    r = np.where(num == 0, 0.0, np.where(den > 0, num / np.where(den > 0, den, 1), np.inf))
  return float(r.max()) if r.size else 0.0


def compare_embeddings(ea, eb):
  out = {}
  for kind in ('single', 'pair'):
    a, b = ea[f'{kind}_embeddings'], eb[f'{kind}_embeddings']
    out[f'{kind}_rel_l2'] = rel_l2(a, b)
    out[f'{kind}_token_max'] = token_max(a, b)
    out[f'{kind}_identical'] = bool(a.shape == b.shape and np.array_equal(a.view(np.uint16), b.view(np.uint16))) \
        if a.dtype == np.float16 and b.dtype == np.float16 else bool(np.array_equal(a, b))
  return out


def embeddings_path(folder, target):
  """AlphaFold3's embeddings file of one target and seed (run_alphafold.py: seed-<S>_embeddings/<job>_seed-<S>_)."""
  return folder / f'seed-{SEED}_embeddings' / f'{target}_seed-{SEED}_embeddings.npz'


def load_embeddings(path):
  with np.load(path) as z:
    return {k: z[k] for k in ('single_embeddings', 'pair_embeddings')}


# ---------------------------------------------------------------- sessions

def processes(root=AF3):
  """{(condition, repetition, target): AlphaFold3 output folder of the target} for every embed process."""
  out = {}
  for sj in sorted(root.glob('*/*/session.json')):
    try:
      s = json.loads(sj.read_text())
    except ValueError:
      continue
    if s.get('label') != 'embed':
      continue
    platform = next((p for p in ('v5e', 'v6e', 'l4') if f'_{p}_' in sj.parent.parent.name), None)
    cond = CODE.get((platform, s.get('config_name')))
    rep = 'r' + sj.parent.name.split('_rep')[1].split('_')[0] if '_rep' in sj.parent.name else None
    for rj in sorted(sj.parent.glob('*/run.json')):
      r = json.loads(rj.read_text())
      if r.get('exit_code') == 0:
        out[(cond, rep, r['pdb_id'])] = rj.parent / 'af3_output' / r['pdb_id']
  return out


def ranking(folder, target):
  d = pd.read_csv(folder / f'{target}_ranking_scores.csv')
  d = d[d.seed == SEED]
  return dict(zip(d['sample'].astype(int), d.ranking_score.astype(float)))


def sample_model(folder, target, k):
  return folder / f'seed-{SEED}_sample-{k}' / f'{target}_seed-{SEED}_sample-{k}_model.cif'


def structures(folder_a, folder_b, target):
  import score_ligand
  from contrasts import coords_text
  ra, rb = ranking(folder_a, target), ranking(folder_b, target)
  ca, lig, m5, ident = [], [], [], 0
  for k in SAMPLES:
    pa, pb = sample_model(folder_a, target, k), sample_model(folder_b, target, k)
    same = coords_text(str(pa.relative_to(REPO))) == coords_text(str(pb.relative_to(REPO)))
    ident += same
    d = {'ca_rmsd_global': 0.0, 'ligand_rmsd': 0.0} if same else score_ligand.pair_divergence(target, pa, pb)
    ca.append(d['ca_rmsd_global'])
    lig.append(d['ligand_rmsd'])
    m5.append(abs(ra[k] - rb[k]))
  f = lambda v, fn: float(fn([x for x in v if x is not None])) if any(x is not None for x in v) else np.nan  # noqa: E731
  return {'m4_ca_rmsd_median': f(ca, np.median), 'm4_ca_rmsd_max': f(ca, np.max),
          'm4_ligand_rmsd_median': f(lig, np.median), 'm4_ligand_rmsd_max': f(lig, np.max),
          'm5_ranking_score_median': f(m5, np.median), 'm5_ranking_score_max': f(m5, np.max),
          'samples_bit_identical': int(ident), 'same_top1': max(ra, key=ra.get) == max(rb, key=rb.get)}


def versus_main(procs, samples_csv=MAIN_SAMPLES):
  """Each embed process against the main study's run of the same condition, repetition, target and seed."""
  from contrasts import masked
  d = pd.read_csv(samples_csv, low_memory=False)
  rows = []
  for (cond, rep, target), folder in sorted(procs.items()):
    m = d[(d.platform == cond) & (d.repetition == rep) & (d.target == target) & (d.seed == SEED) & d.model_path.notna()]
    if m.empty:
      rows.append({'condition': cond, 'repetition': rep, 'target': target, 'main_session': '', 'compared_files': 0,
                   'differ': '', 'identical': None})
      continue
    main_dir = (REPO / m.model_path.iloc[0]).parent.parent
    names = [f'{target}_ranking_scores.csv'] + [f'seed-{SEED}_sample-{k}/{target}_seed-{SEED}_sample-{k}_{s}'
                                                for k in SAMPLES for s in ('model.cif', 'summary_confidences.json')]
    differ = [n for n in names if not (folder / n).is_file() or not (main_dir / n).is_file()
              or masked(folder / n) != masked(main_dir / n)]
    rows.append({'condition': cond, 'repetition': rep, 'target': target, 'main_session': m.session.iloc[0],
                 'compared_files': len(names), 'differ': ';'.join(differ), 'identical': not differ})
  return pd.DataFrame(rows)


def analyse(procs):
  emb, st = [], []
  targets = sorted({t for _, _, t in procs})
  for a, b in PAIRS:
    name = f'{a[0]} {a[1]} - {b[0]} {b[1]}'
    for t in targets:
      fa, fb = procs.get((*a, t)), procs.get((*b, t))
      if fa is None or fb is None:
        continue
      ea, eb = load_embeddings(embeddings_path(fa, t)), load_embeddings(embeddings_path(fb, t))
      emb.append({'pair': name, 'target': t, 'tokens': ea['single_embeddings'].shape[0], **compare_embeddings(ea, eb)})
      st.append({'pair': name, 'target': t, **structures(fa, fb, t)})
  return pd.DataFrame(emb), pd.DataFrame(st)


MEASURES = ['single_rel_l2', 'single_token_max', 'pair_rel_l2', 'pair_token_max', 'm4_ca_rmsd_median',
            'm4_ligand_rmsd_median', 'm5_ranking_score_median']


def summarise(emb, st):
  """One row per pair, in PAIRS order: median and maximum over targets of each measure, and counts."""
  j = emb.merge(st, on=['pair', 'target'])
  rows = []
  for a, b in PAIRS:
    g = j[j.pair == f'{a[0]} {a[1]} - {b[0]} {b[1]}']
    if g.empty:
      continue
    row = {'pair': g.pair.iloc[0], 'targets': len(g)}
    for m in MEASURES:
      row[f'{m}_median'] = float(g[m].median())
      row[f'{m}_max'] = float(g[m].max())
    row['single_identical_targets'] = int(g.single_identical.astype(bool).sum())
    row['pair_identical_targets'] = int(g.pair_identical.astype(bool).sum())
    row['samples_bit_identical'] = int(g.samples_bit_identical.sum())
    row['samples'] = len(g) * len(SAMPLES)
    row['same_top1_targets'] = int(g.same_top1.astype(bool).sum())
    rows.append(row)
  return pd.DataFrame(rows)


def report(emb, st, vm, out, summary=None):
  lines = ['# Follow-up 2: where the divergence arises', '', 'Section 10 row of 2026-10-14; descriptive. Code: '
           'analysis/embed.py.', '']
  if len(emb):
    j = emb.merge(st, on=['pair', 'target'])
    lines += ['| pair | target | tokens | single rel L2 | single token max | pair rel L2 | pair token max | '
              'M4 C-alpha median | M4 ligand median | M5 score median | identical samples | same top-1 |',
              '|---|---|---|---|---|---|---|---|---|---|---|---|']
    lines += [f'| {r.pair} | {r.target} | {r.tokens} | {r.single_rel_l2:.3g} | {r.single_token_max:.3g} | '
              f'{r.pair_rel_l2:.3g} | {r.pair_token_max:.3g} | {r.m4_ca_rmsd_median:.3g} | {r.m4_ligand_rmsd_median:.3g} | '
              f'{r.m5_ranking_score_median:.3g} | {r.samples_bit_identical} | {r.same_top1} |' for r in j.itertuples()]
  if summary is not None and len(summary):
    lines += ['', '## Summary per pair (median over targets, maximum in brackets)', '',
              '| pair | targets | single rel L2 | single token max | pair rel L2 | pair token max | M4 C-alpha | '
              'M4 ligand | M5 score | identical single / pair | identical samples | same top-1 |',
              '|---|---|---|---|---|---|---|---|---|---|---|---|']
    f = lambda r, m: f'{getattr(r, m + "_median"):.3g} ({getattr(r, m + "_max"):.3g})'  # noqa: E731
    lines += [f'| {r.pair} | {r.targets} | ' + ' | '.join(f(r, m) for m in MEASURES) +
              f' | {r.single_identical_targets} / {r.pair_identical_targets} | {r.samples_bit_identical}/{r.samples} | '
              f'{r.same_top1_targets}/{r.targets} |' for r in summary.itertuples()]
  if len(vm):
    lines += ['', '## Against the main study (same condition, repetition, target, seed)', '',
              f'{int(vm.identical.fillna(False).sum())}/{len(vm)} processes bit-identical to the main study.', '',
              '| condition | rep | target | identical | files that differ |', '|---|---|---|---|---|']
    lines += [f'| {r.condition} | {r.repetition} | {r.target} | {r.identical} | {r.differ or "-"} |' for r in vm.itertuples()]
  (out / 'report.md').write_text('\n'.join(lines) + '\n')


def main():
  ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
  ap.add_argument('--out', default=str(REPO / 'results' / 'analysis' / 'embed'))
  ap.add_argument('--reuse', action='store_true', help='read embeddings.csv and structures.csv from --out')
  args = ap.parse_args()
  out = pathlib.Path(args.out)
  out.mkdir(parents=True, exist_ok=True)
  procs = processes()
  if not procs:
    raise SystemExit('no embed session in results/af3/')
  if args.reuse:
    # round_trip: the floats read back exactly, so --reuse rewrites the same bytes as a full run
    emb = pd.read_csv(out / 'embeddings.csv', float_precision='round_trip')
    st = pd.read_csv(out / 'structures.csv', float_precision='round_trip')
  else:
    emb, st = analyse(procs)
  summary = summarise(emb, st) if len(emb) else pd.DataFrame()
  vm = versus_main(procs)
  emb.to_csv(out / 'embeddings.csv', index=False)
  st.to_csv(out / 'structures.csv', index=False)
  vm.to_csv(out / 'versus_main.csv', index=False)
  summary.to_csv(out / 'summary.csv', index=False)
  report(emb, st, vm, out, summary)
  print((out / 'report.md').read_text())
  return 0


if __name__ == '__main__':
  sys.exit(main())
