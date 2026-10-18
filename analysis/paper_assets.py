"""Every table and figure the paper needs, regenerated from the frozen analysis outputs.

    analysis/figures/.venv/bin/python analysis/paper_assets.py [--out results/paper_assets] [--tables_only]

Environment: analysis/requirements-figures.txt (the analysis environment plus matplotlib); --tables_only needs only
the analysis environment. Inputs (read only; their SHA-256 go to sources.tsv):
  results/analysis/main_20261014/primary_v2/   samples.csv, contrasts/*.csv, cost/*.csv (primary analysis after the
                                               section 10 M4 correction)
  results/analysis/main_20261014/glycans/      side_by_side.csv, audit_posebusters_glycans.csv
  results/analysis/main_20261014/exploratory/  a_* to e_*.csv
  results/analysis/main_monitor_20261013/      cross_session.tsv (pilot2 r1 against main r1, strict)
  results/af3/<session>/                       memory-limit sessions (label memlimit) and the follow-up 2 sessions
                                               (label embed, process counts), when they exist
  results/analysis/ppi_20261015/               follow-up 1 (analysis/ppi.py and score_dockq.py outputs)
  results/analysis/embed_20261015/             follow-up 2 (analysis/embed.py outputs)
  targets/posebusters_subset.csv, inputs/manifest.csv, inputs/ppi/manifest.csv
Every source under results/af3/ must match the data freeze (results/freeze/manifest_sha256.tsv, identifier
FREEZE_ID below); a mismatch or a file outside the freeze stops the script.
Every asset carries its role as section 10 of notes/analysis_plan.md labels it (ROLES): design, primary and
secondary (role per row), pre-specified (Q2, Q3), secondary, exploratory, post hoc, descriptive measurement (memory
limit), follow-up.
Outputs in --out: tables/T*.csv and T*.tex (booktabs tabular bodies with a caption and role comment),
figures/F*.pdf and F*.png, sources.tsv, README.md (one line per asset, with its role). Deterministic: the same
inputs give the same CSV and LaTeX bytes; PDF and PNG carry no date. The memory-limit table and figure are
placeholders until memory-limit sessions exist in results/af3/; the follow-up assets are skipped while their
analysis outputs do not exist.
"""
import argparse
import hashlib
import json
import math
import pathlib
import re
import sys

import numpy as np
import pandas as pd

REPO = pathlib.Path(__file__).resolve().parent.parent
MAIN = REPO / 'results' / 'analysis' / 'main_20261014'
PRIMARY = MAIN / 'primary_v2'
GLYCANS = MAIN / 'glycans'
EXPLORATORY = MAIN / 'exploratory'
CROSS = REPO / 'results' / 'analysis' / 'main_monitor_20261013' / 'cross_session.tsv'
PPI = REPO / 'results' / 'analysis' / 'ppi_20261015'
EMBED = REPO / 'results' / 'analysis' / 'embed_20261015'
FREEZE = REPO / 'results' / 'freeze' / 'manifest_sha256.tsv'
FREEZE_ID = '14e1fc4c12b6573de9389d9aba079d5fb5f30b290c9303f75c5de4f478b9d8f9'
# The freeze extended to follow-ups 3 and 4 (2026-10-18; freeze 14e1fc4c's rows unchanged plus the 66 cpu60 and a100
# sessions), named in README.md only with --followups34.
FREEZE34 = REPO / 'results' / 'freeze' / 'followups34_20261018' / 'manifest_sha256.tsv'
FREEZE34_ID = 'fc297328ae079155da0f9e2ac0781b5d207f75e1fafd8d5ccfd7f5f7a8edd9c8'
PPI_CONTRASTS = ['HW(V5,V6)', 'HW(V6,G)', 'RC(V6)', 'RC(G)']
PPI_COLUMNS = ['V5r1', 'V6r1', 'V6r2', 'Gr1', 'Gr2']
EMBED_PAIRS = ['V5 r1 - V6 r1', 'V6 r1 - G r1', 'G r1 - G r2', 'G r1 - Gd r1']
AF3 = REPO / 'results' / 'af3'
ARMS = ['V5', 'V6', 'G', 'GT', 'Gd', 'V6h', 'Gh', 'C']
PRIMARY_PAIRS = ['HW(V5,V6)', 'HW(V6,G)', 'HW(G,C)']
EXPLORATORY_CONTRASTS = ['HW(V5,V6)', 'HW(V6,G)', 'HW(V6,GT)', 'RC(G)', 'DU', 'DET', 'PR(V6)', 'PR(G)', 'HWh(V6,G)']
MEMLIMIT_SIZES = [1280, 1536, 2048, 2560, 3072, 3584, 4096, 4608, 5120]
GIB = 2 ** 30
ASSETS = {}   # name: (role, one line on what it shows) (README.md)
SOURCES = set()
# The role of every asset, as section 10 of notes/analysis_plan.md labels it. Q1 is the primary question (section 1);
# Q2 and Q3 are pre-specified without that label. Tables whose rows have different roles carry a role column.
PRIMARY_SECONDARY = 'primary and secondary (role column per row)'
FOLLOW_UP_1 = 'follow-up 1 (outside the pre-registered study, section 10, 2026-10-14)'
FOLLOW_UP_2 = 'follow-up 2 (outside the pre-registered study, descriptive, section 10, 2026-10-14)'
MEMLIMIT_ROLE = 'descriptive measurement, outside every contrast (memory limit, section 10, 2026-10-14)'
ROLES = {
    'T01_platforms': 'design (main study, memory limit and follow-ups)',
    'T02_targets': 'design (main study and follow-up 1)',
    'T03_checks': 'primary (pre-specified checks)',
    'T04_flip_rates': PRIMARY_SECONDARY, 'T05_hw_minus_rc': PRIMARY_SECONDARY, 'T06_m4_m5': PRIMARY_SECONDARY,
    'T07a_q3_success': 'pre-specified (Q3)', 'T07b_q3_differences': 'pre-specified (Q3)',
    'T08_glycans': 'as specified (1), with post hoc sensitivity analyses (2, 3) (section 10, 2026-10-14)',
    'T09_cost': 'pre-specified (Q2, M6)', 'T10_memory': 'pre-specified (Q2, M7)',
    'T11_cross_session': 'pre-specified check (section 10, 2026-10-13)',
    'T12_cutoff_sensitivity': 'secondary (section 10, 2026-10-13)',
    'T13_exploratory_margin': 'exploratory (section 10, 2026-10-14)',
    'T14_exploratory_concentration': 'exploratory (section 10, 2026-10-14)',
    'T15_exploratory_perturbation': 'exploratory (section 10, 2026-10-14)',
    'T16a_exploratory_buckets': 'exploratory (section 10, 2026-10-14)',
    'T16b_exploratory_flips_that_matter': 'exploratory (section 10, 2026-10-14)',
    'T16c_exploratory_flips_list': 'exploratory (section 10, 2026-10-14)',
    'T17_glycan_audit': 'post hoc (section 10, 2026-10-14)',
    'T18_memlimit': MEMLIMIT_ROLE, 'T18b_memlimit_processes': MEMLIMIT_ROLE,
    'T19a_ppi_success': FOLLOW_UP_1 + '; DockQ 0.49 rows secondary',
    'T19b_ppi_contrasts': FOLLOW_UP_1 + '; main-study column as in T04 and T06',
    'T19c_ppi_hw_minus_rc': FOLLOW_UP_1 + '; role column per row, DockQ 0.49 secondary',
    'T19d_ppi_per_target': FOLLOW_UP_1,
    'T19e_ppi_m3': FOLLOW_UP_1,
    'T19f_ppi_exploratory_margin': FOLLOW_UP_1 + ', exploratory',
    'T19g_ppi_exploratory_perturbation': FOLLOW_UP_1 + ', exploratory',
    'T20a_embed_pairs': FOLLOW_UP_2, 'T20b_embed_per_target': FOLLOW_UP_2, 'T20c_embed_versus_main': FOLLOW_UP_2,
    'F1_flip_rates': PRIMARY_SECONDARY.replace('role column per row', 'see T04'),
    'F2_hw_minus_rc': PRIMARY_SECONDARY.replace('role column per row', 'see T05'),
    'F3_m4_ligand_rmsd': 'primary and secondary (contrasts as in T04)',
    'F4_q3': 'pre-specified (Q3)', 'F5_flip_vs_margin': 'exploratory (section 10, 2026-10-14)',
    'F6_predicted_vs_observed': 'exploratory (section 10, 2026-10-14)', 'F7_cost': 'pre-specified (Q2, M6)',
    'F8_memory': 'pre-specified (Q2, M7) (left); ' + MEMLIMIT_ROLE + ' (right)',
    'F9_glycans': 'as specified, with post hoc sensitivity analyses (section 10, 2026-10-14)',
    'F10_cross_session': 'pre-specified check (section 10, 2026-10-13)',
    'F11_ppi_flip_rates': FOLLOW_UP_1 + '; main-study values as in T04',
    'F12_ppi_dockq_per_target': FOLLOW_UP_1,
    'F13_embed_trunk_vs_structure': FOLLOW_UP_2,
    # Follow-ups 3 and 4 (section 10, 2026-10-16), written only with --followups34 once their analyses exist.
    'T21a_cpu60_contrasts': 'follow-up 4 (outside the pre-registered study, section 10, 2026-10-16); role column per row',
    'T21b_cpu60_six_vs_sixty': 'follow-up 4 (outside the pre-registered study, section 10, 2026-10-16); main-study column as in T04',
    'T21c_cpu60_cost_memory': 'follow-up 4 (outside the pre-registered study, section 10, 2026-10-16)',
    'T22a_a100_checks': 'follow-up 3 (outside the pre-registered study, section 10, 2026-10-16); checks before any result',
    'T22b_a100_contrasts': 'follow-up 3 (outside the pre-registered study, section 10, 2026-10-16); role column per row',
    'T22c_a100_hw_minus_rc': 'follow-up 3 (outside the pre-registered study, section 10, 2026-10-16)',
    'T22d_a100_cost': 'follow-up 3 (outside the pre-registered study, section 10, 2026-10-16)',
    'F14_a100_flip_rates': 'follow-up 3 (outside the pre-registered study, section 10, 2026-10-16)',
}


def read(path, **kw):
  SOURCES.add(pathlib.Path(path))
  return pd.read_csv(path, sep='\t' if str(path).endswith('.tsv') else ',', **kw)


def pct(x, digits=1):
  return '' if x is None or (isinstance(x, float) and math.isnan(x)) else f'{100 * x:.{digits}f}'


def pct_ci(e, lo, hi):
  if e is None or (isinstance(e, float) and math.isnan(e)):
    return ''
  return f'{pct(e)} [{pct(lo)}, {pct(hi)}]' if not (isinstance(lo, float) and math.isnan(lo)) else pct(e)


def num_ci(e, lo, hi, fmt='{:.3g}'):
  if e is None or (isinstance(e, float) and math.isnan(e)):
    return ''
  return f'{fmt.format(e)} [{fmt.format(lo)}, {fmt.format(hi)}]' if not (isinstance(lo, float) and math.isnan(lo)) \
      else fmt.format(e)


LATEX_SPECIAL = {'\\': r'\textbackslash{}', '&': r'\&', '%': r'\%', '$': r'\$', '#': r'\#', '_': r'\_', '{': r'\{',
                 '}': r'\}', '~': r'\textasciitilde{}', '^': r'\textasciicircum{}'}


def tex_escape(text):
  return ''.join(LATEX_SPECIAL.get(c, c) for c in str(text))


def write_table(out, name, df, caption):
  """tables/<name>.csv and tables/<name>.tex (a booktabs tabular, cells as in the CSV)."""
  d = out / 'tables'
  d.mkdir(parents=True, exist_ok=True)
  df = df.reset_index(drop=True)
  df.to_csv(d / f'{name}.csv', index=False, lineterminator='\n')
  cols = list(df.columns)
  align = ''.join('r' if i and pd.api.types.is_numeric_dtype(df[c]) else 'l' for i, c in enumerate(cols))
  role = ROLES[name]
  lines = [f'% {name}: {tex_escape(caption)}', f'% Role: {tex_escape(role)}',
           f'% Generated by analysis/paper_assets.py; do not edit.',
           f'\\begin{{tabular}}{{{align}}}', r'\toprule', ' & '.join(tex_escape(c) for c in cols) + r' \\', r'\midrule']
  for row in df.itertuples(index=False):
    lines.append(' & '.join('' if (isinstance(v, float) and math.isnan(v)) else tex_escape(v) for v in row) + r' \\')
  lines += [r'\bottomrule', r'\end{tabular}', '']
  (d / f'{name}.tex').write_text('\n'.join(lines))
  ASSETS[f'tables/{name}'] = (role, caption)


# ---------------------------------------------------------------- tables

def followup_process_counts():
  """{experiment: {arm: AlphaFold3 processes}} of the memory-limit, ppi and embed runs (empty where absent)."""
  import memlimit
  out = {'memory-limit': {}, 'ppi': {}, 'embed': {}}
  t = memlimit.process_table(AF3, PRIMARY / 'samples.csv')
  if len(t):
    out['memory-limit'] = t[t.source == 'memlimit'].groupby('arm').size().to_dict()
  if (PPI / 'samples_dockq.csv').exists():
    d = read(PPI / 'samples_dockq.csv', low_memory=False)
    out['ppi'] = d.drop_duplicates(['session', 'target']).groupby('platform').size().to_dict()
  import embed
  procs = embed.processes(AF3)
  for (cond, _, target), folder in procs.items():
    out['embed'][cond] = out['embed'].get(cond, 0) + 1
    SOURCES.update([folder.parent.parent / 'run.json', folder.parent.parent.parent / 'session.json'])
  return out


def t01_platforms(out, samples, prices):
  code_platform = {'V5': 'v5e', 'V6': 'v6e', 'V6h': 'v6e', 'G': 'l4', 'GT': 'l4', 'Gd': 'l4', 'Gh': 'l4', 'C': 'cpu'}
  rows, counts = [], followup_process_counts()
  for arm in ARMS:
    g = samples[samples.platform == arm]
    if g.empty:
      continue
    first = lambda c: ';'.join(sorted({str(v) for v in g[c].dropna().unique()}))  # noqa: E731
    p = prices[(prices.platform == code_platform[arm]) & (prices.provisioning == 'standard')]
    rows.append({'arm': arm, 'device': first('device_kind'), 'machine type': first('machine_type'),
                 'config': first('config'), 'attention': first('attention'), 'matmul precision': first('matmul_precision'),
                 'jax': first('jax'), 'libtpu': first('libtpu') or '-',
                 'main processes': int(g.drop_duplicates(['session', 'run', 'target']).shape[0]),
                 **{f'{k} processes': int(v.get(arm, 0)) for k, v in counts.items()},
                 'on-demand USD/h': ';'.join(f'{v:.3f}' for v in sorted(p.usd_per_hour.unique()))})
  write_table(out, 'T01_platforms', pd.DataFrame(rows), 'Arms of the main study, the memory-limit measurement and the '
              'two follow-ups: hardware, configuration, software, AlphaFold3 processes per experiment and on-demand list '
              'price (every run on demand, official weights, the same software stack)')


def t02_targets(out):
  subset = read(REPO / 'targets' / 'posebusters_subset.csv', dtype=str)
  manifest = read(REPO / 'inputs' / 'manifest.csv', dtype=str).set_index('pdb_id')
  subset['bucket'] = subset.pdb_id.map(manifest.bucket).astype(int)
  subset['released'] = np.where(subset.rcsb_initial_release > '2021-09-30', 'after 2021-09-30', 'on or before 2021-09-30')
  t = subset.pivot_table(index='bucket', columns='released', values='pdb_id', aggfunc='count', fill_value=0)
  t['targets'] = t.sum(axis=1)
  t['pilot'] = subset.groupby('bucket').pilot.apply(lambda s: int((s == 'yes').sum()))
  ppi_path = REPO / 'inputs' / 'ppi' / 'manifest.csv'
  if ppi_path.exists():
    ppi = read(ppi_path)
    t['ppi targets'] = ppi.groupby('bucket').size().reindex(t.index, fill_value=0)
    t['ppi, shallow alignment'] = ppi[ppi.paired_rows <= 9].groupby('bucket').size().reindex(t.index, fill_value=0)
  t = t.reset_index().rename(columns={'bucket': 'token bucket'})
  total = {c: int(t[c].sum()) if c != 'token bucket' else 'all' for c in t.columns}
  t = pd.concat([t, pd.DataFrame([total])], ignore_index=True)
  write_table(out, 'T02_targets', t, 'The 60 PoseBusters targets by AlphaFold3 token bucket and by release relative '
              'to the training cutoff (2021-09-30); pilot targets; follow-up 1: the 30 protein-protein targets (all '
              'released 2021-10-01 or later) by bucket, and those with a shallow alignment (at most 9 paired MSA rows)')


def t03_checks(out, checks):
  write_table(out, 'T03_checks', checks.fillna(''), 'Pre-specified checks on the main study: exit codes, cache '
              '(2a, 2b), bit-identity of warm reruns, M2 cross-check, M1 coverage, check 3')


ROLE_ORDER = ['primary pair', 'recompilation', 'default use', 'deterministic ops', 'precision', 'matched hardware',
              'secondary pair', 'warm-cache check', 'seed yardstick']


def role_rank(role):
  """Primary pairs first, then by role as in ROLE_ORDER; a role marked "secondary arm" after its kind."""
  base = next((i for i, r in enumerate(ROLE_ORDER) if str(role).startswith(r)), len(ROLE_ORDER))
  return (base, 'secondary arm' in str(role))


def by_role(df):
  """df's rows in role order, keeping the analysis order within a role."""
  first = {c: i for i, c in enumerate(df.contrast.drop_duplicates())}
  key = df.apply(lambda r: (*role_rank(r.role), first[r.contrast]), axis=1)
  return df.iloc[sorted(range(len(df)), key=lambda i: key.iloc[i])]


def wide_stats(df, stats, fmt):
  rows = []
  for (c, role), g in by_role(df).groupby(['contrast', 'role'], sort=False):
    row = {'contrast': c, 'role': role, 'targets': int(g.n_targets.max())}
    for st in stats:
      r = g[g.statistic == st]
      row[st] = fmt(*r.iloc[0][['estimate', 'ci_low', 'ci_high']]) if len(r) else ''
    rows.append(row)
  return pd.DataFrame(rows)


def t04_flips(out, contrasts):
  t = wide_stats(contrasts, ['flip_top1', 'flip_success', 'flip_matters', 'bit_identical_fraction'], pct_ci)
  t.columns = ['contrast', 'role', 'targets', 'top-1 flip (%)', 'success flip (%)', 'flips that matter (%)',
               'bit-identical samples (%)']
  write_table(out, 'T04_flip_rates', t, 'Flip rates per contrast (per (target, seed) pair) with 95% target-clustered '
              'bootstrap intervals; bit-identical fraction of sample pairs')


def t05_hw_rc(out, hwrc):
  rows = []
  for (h, rc, ref), g in hwrc.groupby(['hardware', 'recompilation', 'reference'], sort=False):
    row = {'hardware pair': h, 'recompilation': rc, 'reference': ref, 'role': g.role.iloc[0]}
    for st in ['flip_top1', 'flip_success', 'flip_matters']:
      r = g[g.statistic == st].iloc[0]
      row[st.replace('_', ' ') + ' (pp)'] = pct_ci(r.estimate, r.ci_low, r.ci_high)
    rows.append(row)
  write_table(out, 'T05_hw_minus_rc', pd.DataFrame(rows), 'Hardware minus recompilation: difference of flip rates '
              '(percentage points) on the same resampled targets, 95% intervals')


def t06_m4_m5(out, contrasts):
  stats = ['m4_ligand_rmsd_median', 'm4_ligand_rmsd_p90', 'm4_ca_rmsd_global_median', 'm4_ca_rmsd_global_p90',
           'm5_ranking_score_median', 'm5_mean_plddt_median', 'm5_iptm_median']
  t = wide_stats(contrasts, stats, lambda e, lo, hi: num_ci(e, lo, hi))
  write_table(out, 'T06_m4_m5', t, 'M4 (ligand RMSD after pocket superposition, protein C-alpha RMSD, angstrom) and '
              'M5 (absolute differences of ranking score, mean pLDDT, ipTM): median and 90th percentile, 95% intervals')


def t07_q3(out, q3, sens):
  rates = q3[q3.statistic == 'success_rate'].pivot(index='condition', columns='n_seeds', values='estimate')
  rates = rates.reindex([a for a in ARMS if a in rates.index])
  t = rates.apply(lambda col: col.map(pct)).reset_index()
  t.columns = ['condition'] + [f'n = {int(c)} (%)' for c in rates.columns]
  write_table(out, 'T07a_q3_success', t, 'Q3: M1 success rate of the overall top-1 over the first n seeds, per '
              'condition (r1)')
  d = q3[q3.statistic == 'success_difference']
  t = pd.DataFrame({'n seeds': d.n_seeds.astype(int), 'pair': d.condition, 'targets': d.n_targets.astype(int),
                    'difference (pp)': [pct_ci(*x) for x in d[['estimate', 'ci_low', 'ci_high']].values]})
  ans = sens[(sens.set == 'all targets') & (sens.quantity == 'Q3') & (sens.name == 'answer')].estimate
  answer = 'not reached' if ans.empty or ans.isna().all() else f'{int(ans.iloc[0])} seed(s)'
  write_table(out, 'T07b_q3_differences', t, f'Q3: pairwise success differences with 95% intervals; answer (every '
              f'interval within +-2 points): {answer}')


def t08_glycans(out, sbs):
  keep = sbs[(sbs.section == 'success rate, r1') | ((sbs.section == 'flips') & sbs.name.isin(PRIMARY_PAIRS))
             | ((sbs.section == 'checks') & sbs.statistic.str.startswith('3 ')) | ((sbs.section == 'Q3') & (sbs.name == 'answer'))]
  rows = []
  for r in keep.to_dict('records'):
    row = {'section': r['section'], 'name': r['name'], 'statistic': r['statistic']}
    for v, label in (('primary', '(1) primary'), ('without_three', '(2) 57 targets'), ('rescored', '(3) rescored')):
      if r['section'] == 'checks':
        row[label] = str(r[f'{v}_text']).split(':')[0]
      elif r['statistic'] == 'seeds needed':
        row[label] = 'not reached' if pd.isna(r[f'{v}_estimate']) else str(int(r[f'{v}_estimate']))
      else:
        row[label] = pct_ci(r[f'{v}_estimate'], r[f'{v}_ci_low'], r[f'{v}_ci_high'])
    rows.append(row)
  write_table(out, 'T08_glycans', pd.DataFrame(rows), 'M1-based results with the three glycan targets as specified '
              '(1), without them (2), and rescored without O1 (3, post hoc); section 10, 2026-10-14')


def t09_cost(out, cost):
  c = cost[cost.cache == 'fresh'] if 'cache' in cost.columns else cost
  t = pd.DataFrame({'platform': c.platform, 'bucket': c.bucket.astype(int), 'processes': c.processes.astype(int),
                    'median wall (s)': c.median_wall_s.round(1), 'median compile (s)': c.median_compile_s.round(1),
                    'USD per structure, on demand': c.median_usd_per_structure_standard.round(3),
                    'USD per structure, Spot': c.median_usd_per_structure_spot.round(3)})
  write_table(out, 'T09_cost', t, 'M6: median wall and compile time per AlphaFold3 process (one seed, 5 samples) and '
              'cost per structure at list prices, by platform and token bucket (fresh cache)')


def t10_memory(out, mem):
  t = pd.DataFrame({'platform': mem.platform, 'bucket': mem.bucket.astype(int), 'max tokens': mem.max_tokens_run,
                    'compiled peak + args (GiB)': mem.compiled_peak_plus_args_gib.round(2),
                    'allocator peak in use (GiB)': mem.alloc_peak_in_use_gib.round(2),
                    'device limit (GiB)': mem.device_limit_gib.round(2),
                    'fraction of limit': mem.peak_plus_args_fraction_of_limit.round(3)})
  write_table(out, 'T10_memory', t, 'M7: device memory by platform and token bucket (compiled peak plus arguments, '
              'allocator peak) against the device limit')


def t11_cross(out, cross):
  rows = []
  for c in [a for a in ARMS if a in set(cross.condition)]:
    g = cross[cross.condition == c]
    rows.append({'arm': c, '(target, seed) units': len(g), 'bit-identical': int((g.strict == 'identical').sum()),
                 'expected': 'identical' if c not in ('G', 'Gh', 'GT') else 'descriptive'})
  write_table(out, 'T11_cross_session', pd.DataFrame(rows), 'Cross-session check: pilot2 r1 against main-study r1 '
              '(fresh VMs, days apart), every output file byte for byte (mmCIF timestamp line ignored)')


def t12_cutoff(out, sens):
  full = sens[sens.set == 'all targets'].set_index(['quantity', 'name', 'statistic', 'reference'])
  post = sens[sens.set != 'all targets'].set_index(['quantity', 'name', 'statistic', 'reference'])
  rows = []
  for key in full.index:
    a = full.loc[key]
    b = post.loc[key] if key in post.index else None
    f = (lambda r: '' if r is None else ('not reached' if pd.isna(r.estimate) else str(int(r.estimate)))) \
        if key[2] == 'seeds needed' else (lambda r: '' if r is None else pct_ci(r.estimate, r.ci_low, r.ci_high))
    rows.append({'quantity': key[0], 'name': key[1], 'statistic': key[2],
                 'reference': key[3] if isinstance(key[3], str) else '', 'all 60 targets': f(a),
                 'released after cutoff': f(b)})
  write_table(out, 'T12_cutoff_sensitivity', pd.DataFrame(rows), 'Training-cutoff sensitivity: M1-based results and '
              'primary pair flips on the targets released after 2021-09-30 (secondary)')


def t13_t16_exploratory(out):
  qa, la = read(EXPLORATORY / 'a_margin_quintiles.csv'), read(EXPLORATORY / 'a_logistic.csv')
  rows = []
  for c in EXPLORATORY_CONTRASTS:
    q = qa[qa.contrast == c].sort_values('quintile')
    f = la[la.contrast == c].iloc[0]
    row = {'contrast': c}
    row.update({f'Q{int(r.quintile)} (%)': pct_ci(r.flip_rate, r.ci_low, r.ci_high) for r in q.itertuples()})
    row['odds ratio per tenfold margin'] = num_ci(f.odds_ratio_per_tenfold_margin, f.odds_ratio_ci_low,
                                                  f.odds_ratio_ci_high, '{:.2f}')
    rows.append(row)
  write_table(out, 'T13_exploratory_margin', pd.DataFrame(rows), 'Exploratory (a): per-seed top-1 flip rate by '
              'quintile of the first condition\'s M3 margin, and logistic regression on log10 margin')
  bd, bs = read(EXPLORATORY / 'b_seeds_with_flip.csv'), read(EXPLORATORY / 'b_top_share.csv')
  t = pd.DataFrame({'contrast': bs.contrast, 'flips': bs.flips, 'targets with a flip': bs.targets_with_a_flip,
                    'targets by seeds with a flip (0, 1, ...)': [', '.join(str(x) for x in bd[bd.contrast == c]
                                                                           .sort_values('seeds_with_flip').targets)
                                                                 for c in bs.contrast],
                    'top 10% share (%)': [pct_ci(*x) for x in bs[['top_share', 'ci_low', 'ci_high']].values],
                    'binomial null share (%)': bs.binomial_null_share.map(pct)})
  write_table(out, 'T14_exploratory_concentration', t, 'Exploratory (b): concentration of flips over targets, '
              'against independent flips at the same rate')
  cp = read(EXPLORATORY / 'c_perturbation.csv')
  t = pd.DataFrame({'contrast': cp.contrast,
                    'median abs score difference': [num_ci(*x, '{:.2g}') for x in cp[['median_abs_score_difference', 'delta_ci_low', 'delta_ci_high']].values],
                    'median margin': [num_ci(*x, '{:.2g}') for x in cp[['median_margin', 'mu_ci_low', 'mu_ci_high']].values],
                    'ratio': [num_ci(*x, '{:.2f}') for x in cp[['delta_over_mu', 'ratio_ci_low', 'ratio_ci_high']].values],
                    'predicted flip rate (%)': [pct_ci(*x) for x in cp[['predicted_flip_rate', 'predicted_ci_low', 'predicted_ci_high']].values],
                    'observed flip rate (%)': [pct_ci(*x) for x in cp[['observed_flip_rate', 'observed_ci_low', 'observed_ci_high']].values]})
  write_table(out, 'T15_exploratory_perturbation', t, 'Exploratory (c): perturbation size against margin, and the '
              'flip rate a two-sample model predicts from them (no free parameter)')
  db = read(EXPLORATORY / 'd_buckets.csv')
  t = db.pivot(index='contrast', columns='bucket', values='flip_rate').reindex(EXPLORATORY_CONTRASTS).map(pct)
  t.columns = [f'{int(c)} tokens (%)' for c in t.columns]
  write_table(out, 'T16a_exploratory_buckets', t.reset_index(), 'Exploratory (d): per-seed top-1 flip rate by token '
              'bucket')
  es, ef = read(EXPLORATORY / 'e_summary.csv'), read(EXPLORATORY / 'e_flips_that_matter.csv')
  write_table(out, 'T16b_exploratory_flips_that_matter', es.fillna(''), 'Exploratory (e): flips that matter, by the '
              'criterion that separates the two top-1 structures')
  t = ef[['contrast', 'target', 'seed_a', 'top1_a', 'ligand_rmsd_a', 'pb_valid_a', 'top1_b', 'ligand_rmsd_b',
          'pb_valid_b', 'separated_by']].copy()
  t['ligand_rmsd_a'] = t.ligand_rmsd_a.round(2)
  t['ligand_rmsd_b'] = t.ligand_rmsd_b.round(2)
  write_table(out, 'T16c_exploratory_flips_list', t, 'Exploratory (e): every flip that matters, with the ligand RMSD '
              '(angstrom) and PoseBusters validity of the two top-1 samples')


def t17_audit(out):
  a = read(GLYCANS / 'audit_posebusters_glycans.csv', dtype=str, keep_default_na=False)
  a = a[a.rule_applies == 'True']
  t = pd.DataFrame({'CCD': a.ccd, 'AF3 set': a.af3_glycan_set, 'O1 leaving flag': a.O1_leaving_flag,
                    'O1 heavy neighbours': a.O1_heavy_neighbours.str.replace(';', ', '),
                    'fragments without O1': a.fragments_without_O1, 'V1': np.where(a.v1_entries != '', 'yes', ''),
                    'V2': np.where(a.v2_entries != '', 'yes', ''), 'study target': a.study_target})
  write_table(out, 'T17_glycan_audit', t, 'PoseBusters V1 and V2 ligands from which AlphaFold3 removes O1 at its '
              'default (glycan sets, no bondedAtomPairs); CCD of 2026-10-08')


def t18_memlimit(out):
  """T18 (per arm) and T18b (per process) from analysis/memlimit.py; placeholders while no memory-limit session
  exists. Returns the per-process table (empty without sessions)."""
  import memlimit
  t = memlimit.process_table(AF3, PRIMARY / 'samples.csv')
  ml = t[t.source == 'memlimit'] if len(t) else t
  for sess in sorted(set(ml.session)) if len(ml) else []:
    SOURCES.update(p for p in (AF3 / sess).glob('*/*/run.json'))
  if ml.empty:
    s = pd.DataFrame({'arm': ['V5', 'V6', 'G', 'GT'], 'largest completed size': ['pending'] * 4,
                      'first failing size': ['pending'] * 4, 'cause': ['memory-limit runs not yet run'] * 4})
  else:
    sm = memlimit.summary(t).set_index('arm')
    rows = []
    for arm in memlimit.ARMS:
      if arm not in sm.index:
        rows.append({'arm': arm, 'largest completed size': 'pending', 'first failing size': 'pending', 'cause': ''})
        continue
      r = sm.loc[arm]
      stop = 'none (all sizes ran)' if pd.isna(r.stopping_size) else str(int(r.stopping_size))
      err = '' if pd.isna(r.error_required_gib) else (
          f'{r.error_required_gib:.2f} GiB required, {r.error_available_gib:.2f} available' if not pd.isna(r.error_available_gib)
          else f'allocation of {r.error_required_gib:.2f} GiB failed; compiled peak + args '
               f'{r.stopping_compiled_peak_plus_args_gib:.2f} GiB')
      rows.append({'arm': arm,
                   'largest completed size': 'none' if pd.isna(r.largest_completed_size) else str(int(r.largest_completed_size)),
                   'first failing size': stop, 'cause': r.cause + (f': {err}' if err else ''),
                   'peak + args at largest completed (GiB)': f'{r.largest_completed_peak_plus_args_gib:.2f}',
                   'device limit (GiB)': f'{r.device_limit_gib:.2f}',
                   'wall at largest completed (s)': f'{r.largest_completed_wall_s:.0f}',
                   'USD per structure at largest completed': f'{r.largest_completed_usd_per_structure:.3f}'})
    s = pd.DataFrame(rows)
  write_table(out, 'T18_memlimit', s, 'Memory limit per chip (section 10, 2026-10-14): 7D5C padded with AlphaFold3\'s '
              '--buckets; the largest size that completed and the first size that failed (the stop rule), with its cause, '
              'memory against the device limit, and time and cost at the largest completed size (every size: T18b)'
              + ('' if len(ml) else ' (placeholder: runs pending)'))
  if len(ml):
    f2 = lambda x, nd=2: '' if pd.isna(x) else f'{x:.{nd}f}'  # noqa: E731
    d = pd.DataFrame({'arm': t.arm.astype(str), 'size': t['size'].astype(int).astype(str) + np.where(t.source == 'memlimit', '', '*'),
                      'exit': t.exit_code.astype(str) + np.where(t.failure != '', ' ' + t.failure, ''),
                      'compiled peak + args (GiB)': t.compiled_peak_plus_args_gib.map(f2),
                      'allocator peak in use (GiB)': t.alloc_peak_in_use_gib.map(f2),
                      'device limit (GiB)': t.device_limit_gib.map(f2), 'compile (s)': t.compile_s.map(lambda x: f2(x, 0)),
                      'inference (s)': t.inference_s.map(lambda x: f2(x, 0)), 'wall (s)': t.wall_s.map(lambda x: f2(x, 0)),
                      'USD per structure': t.usd_per_structure_standard.map(lambda x: f2(x, 3)),
                      'memory in the error (GiB)': [('' if pd.isna(a) else (f'{a:.2f} required of {b:.2f}' if not pd.isna(b)
                                                                            else f'{a:.2f} allocation'))
                                                    for a, b in zip(t.error_required_gib, t.error_available_gib)]})
    write_table(out, 'T18b_memlimit_processes', d, 'Memory limit per chip, every process: compiled peak plus arguments '
                '(XLA memory analysis), allocator peak in use, device limit, compile, inference and wall time, on-demand '
                'cost per structure, and for the failed size the memory the error reports; * the main study\'s 7D5C '
                'process at its 1,024-token bucket (fresh, r1, seed 1). On TPU the allocator counts arguments and '
                'outputs, not the compiled program\'s temporaries')
  return ml


# ---------------------------------------------------------------- follow-ups

PCT_STATS = ('flip', 'overall_flip', 'bit_identical')


def fmt_stat(statistic, e, lo, hi):
  return pct_ci(e, lo, hi) if statistic.startswith(PCT_STATS) else num_ci(e, lo, hi)


def t19_ppi(out, main_contrasts, main_hwrc):
  """Follow-up 1 tables from results/analysis/ppi_20261015 (analysis/ppi.py). False when the outputs are absent."""
  if not (PPI / 'contrasts.csv').exists():
    return False
  sr = read(PPI / 'success_rates.csv')
  rows = []
  for (thr, stat), g in sr.groupby(['threshold', 'statistic'], sort=False):
    row = {'DockQ threshold': f'{thr:.2f}' + ('' if thr < 0.3 else ' (secondary)'),
           'top-1': 'overall (5 seeds)' if stat.endswith('overall_top1') else 'per seed'}
    for c, rep in (('V5', 'r1'), ('V6', 'r1'), ('V6', 'r2'), ('G', 'r1'), ('G', 'r2')):
      r = g[(g.condition == c) & (g.repetition == rep)]
      row[f'{c} {rep} (%)'] = pct_ci(*r.iloc[0][['estimate', 'ci_low', 'ci_high']]) if len(r) else ''
    rows.append(row)
  write_table(out, 'T19a_ppi_success', pd.DataFrame(rows), 'Follow-up 1 (30 protein-protein complexes): success rate '
              '(DockQ 2.1.3 against the deposited first assembly) of the overall top-1 and of the top-1 per seed, per '
              'condition and repetition, 95% target bootstrap intervals')
  con, con49, ov = read(PPI / 'contrasts.csv'), read(PPI / 'contrasts_049.csv'), read(PPI / 'overall_top1.csv')
  stats = [('flip_top1', con, 'top-1 flip per seed (%)'), ('overall_flip_top1', ov, 'top-1 flip, overall top-1 (%)'),
           ('flip_success', con, 'success flip per seed, DockQ 0.23 (%)'),
           ('flip_success', con49, 'success flip per seed, DockQ 0.49, secondary (%)'),
           ('overall_flip_success', ov, 'success flip, overall top-1 (%)'),
           ('flip_matters', con, 'flips that matter, DockQ 0.23 (%)'),
           ('m4_ca_rmsd_global_median', con, 'M4 C-alpha RMSD median (angstrom)'),
           ('m4_ca_rmsd_global_p90', con, 'M4 C-alpha RMSD 90th percentile (angstrom)'),
           ('m5_ranking_score_median', con, 'M5 ranking score median'), ('m5_iptm_median', con, 'M5 ipTM median'),
           ('bit_identical_fraction', con, 'bit-identical sample pairs (%)')]
  rows = []
  for c in PPI_CONTRASTS:
    role = con[con.contrast == c].role.iloc[0]
    for st, src, label in stats:
      r = src[(src.contrast == c) & (src.statistic == st)]
      # The main study has no overall top-1 flips and no second success threshold: no main-study value there.
      m = main_contrasts[(main_contrasts.contrast == c) & (main_contrasts.statistic == st)] if src is con else r.iloc[:0]
      rows.append({'contrast': c, 'role in the main study': role, 'statistic': label,
                   'ppi, 30 targets': fmt_stat(st, *r.iloc[0][['estimate', 'ci_low', 'ci_high']]) if len(r) else '',
                   'main study, 60 targets': fmt_stat(st, *m.iloc[0][['estimate', 'ci_low', 'ci_high']]) if len(m) else ''})
  write_table(out, 'T19b_ppi_contrasts', pd.DataFrame(rows), 'Follow-up 1: contrasts with 95% target bootstrap '
              'intervals next to the main study\'s values (success there is M1 on protein-ligand targets; M4 there is '
              'the protein C-alpha RMSD, here both chains of the complex; overall top-1 flips were not computed in the '
              'main study)')
  rows = []
  for thr, h in (('0.23', read(PPI / 'hw_minus_rc.csv')), ('0.49 (secondary)', read(PPI / 'hw_minus_rc_049.csv'))):
    for (hw, rc, ref), g in h.groupby(['hardware', 'recompilation', 'reference'], sort=False):
      row = {'DockQ threshold': thr, 'hardware pair': hw, 'recompilation': rc, 'reference': ref}
      for st in ['flip_top1', 'flip_success', 'flip_matters']:
        r = g[g.statistic == st].iloc[0]
        row[st.replace('_', ' ') + ' (pp)'] = pct_ci(r.estimate, r.ci_low, r.ci_high)
      m = main_hwrc[(main_hwrc.hardware == hw) & (main_hwrc.recompilation == rc) & (main_hwrc.statistic == 'flip_top1')]
      row['main study, top-1 flip (pp)'] = pct_ci(*m.iloc[0][['estimate', 'ci_low', 'ci_high']]) if len(m) else ''
      rows.append(row)
  write_table(out, 'T19c_ppi_hw_minus_rc', pd.DataFrame(rows), 'Follow-up 1: hardware minus recompilation '
              '(percentage points, same resampled targets, 95% intervals). V5 runs r1 only, so HW(V5,V6) is given '
              'against RC(V6) only, labelled secondary as RC(Q) in the main study (section 10, 2026-10-15)')
  pt = read(PPI / 'per_target.csv')
  t = pd.DataFrame({'target': pt.target, 'tokens': pt.tokens, 'paired MSA rows': pt.paired_rows,
                    'shallow alignment': np.where(pt.shallow.astype(str) == 'True', 'yes', '')})
  for c in PPI_COLUMNS:
    t[f'{c[:-2]} {c[-2:]}'] = pt[f'{c}_top1_dockq'].map(lambda x: f'{x:.3f}')
  for c in PPI_CONTRASTS:
    t[f'{c} flips'] = pt[f'{c}_seed_flips'].astype(str) + np.where(pt[f'{c}_success_flips'] > 0,
                                                                    ' (' + pt[f'{c}_success_flips'].astype(str) + ' success)', '')
  write_table(out, 'T19d_ppi_per_target', t, 'Follow-up 1, per target: DockQ of the overall top-1 per condition and '
              'repetition, and the number of seeds (of 5) whose top-1 flips in each contrast, with success flips in '
              'brackets; in draw order')
  m3 = read(PPI / 'm3.csv')
  t = pd.DataFrame({'condition': m3.condition, 'repetition': m3.repetition, 'seed units': m3.seed_units,
                    'median': m3['median'].map(lambda x: f'{x:.2g}'), 'Q1': m3.q1.map(lambda x: f'{x:.2g}'),
                    'Q3': m3.q3.map(lambda x: f'{x:.2g}'), 'minimum': m3['min'].map(lambda x: f'{x:.2g}'),
                    'zero margins': m3.zero_margins})
  order = {'V5': 0, 'V6': 1, 'G': 2}
  t = t.sort_values(['condition', 'repetition'], key=lambda s: s.map(order) if s.name == 'condition' else s)
  write_table(out, 'T19e_ppi_m3', t, 'Follow-up 1: M3 margin (top-1 minus top-2 ranking score within a seed) per '
              'condition and repetition')
  qa, la = read(PPI / 'exploratory_a_quintiles.csv'), read(PPI / 'exploratory_a_logistic.csv')
  main_la = read(EXPLORATORY / 'a_logistic.csv')
  rows = []
  for c in PPI_CONTRASTS:
    q = qa[qa.contrast == c].sort_values('quintile')
    f = la[la.contrast == c].iloc[0]
    row = {'contrast': c}
    row.update({f'Q{int(r.quintile)} (%)': pct_ci(r.flip_rate, r.ci_low, r.ci_high) for r in q.itertuples()})
    row['odds ratio per tenfold margin'] = num_ci(f.odds_ratio_per_tenfold_margin, f.odds_ratio_ci_low,
                                                  f.odds_ratio_ci_high, '{:.2f}')
    m = main_la[main_la.contrast == c]
    row['main study odds ratio'] = num_ci(*m.iloc[0][['odds_ratio_per_tenfold_margin', 'odds_ratio_ci_low',
                                                       'odds_ratio_ci_high']], '{:.2f}') if len(m) else ''
    rows.append(row)
  write_table(out, 'T19f_ppi_exploratory_margin', pd.DataFrame(rows), 'Follow-up 1, exploratory (a): per-seed top-1 '
              'flip rate by quintile of the first condition\'s M3 margin, and the logistic regression on log10 margin, '
              'next to the main study\'s odds ratio (RC(V6) has no flip, so no fit)')
  cp, main_cp = read(PPI / 'exploratory_c.csv'), read(EXPLORATORY / 'c_perturbation.csv')
  rows = []
  for r in cp.itertuples():
    m = main_cp[main_cp.contrast == r.contrast]
    rows.append({'contrast': r.contrast,
                 'median abs score difference': num_ci(r.median_abs_score_difference, r.delta_ci_low, r.delta_ci_high, '{:.2g}'),
                 'median margin': num_ci(r.median_margin, r.mu_ci_low, r.mu_ci_high, '{:.2g}'),
                 'predicted flip rate (%)': pct_ci(r.predicted_flip_rate, r.predicted_ci_low, r.predicted_ci_high),
                 'observed flip rate (%)': pct_ci(r.observed_flip_rate, r.observed_ci_low, r.observed_ci_high),
                 'main study predicted / observed (%)': f'{pct(m.predicted_flip_rate.iloc[0])} / {pct(m.observed_flip_rate.iloc[0])}'
                 if len(m) else ''})
  write_table(out, 'T19g_ppi_exploratory_perturbation', pd.DataFrame(rows), 'Follow-up 1, exploratory (c): '
              'perturbation size against margin and the flip rate the two-sample model predicts (no free parameter), '
              'next to the main study\'s predicted and observed rates')
  return True


def t20_embed(out, main_pairs_sample):
  """Follow-up 2 tables from results/analysis/embed_20261015 (analysis/embed.py). False when absent."""
  if not (EMBED / 'summary.csv').exists():
    return False
  sm = read(EMBED / 'summary.csv')
  g3 = lambda x: f'{x:.3g}'  # noqa: E731
  mm = lambda r, m: f'{g3(getattr(r, m + "_median"))} ({g3(getattr(r, m + "_max"))})'  # noqa: E731
  t = pd.DataFrame([{'pair': r.pair, 'targets': r.targets, 'single rel L2': mm(r, 'single_rel_l2'),
                     'pair rel L2': mm(r, 'pair_rel_l2'), 'M4 C-alpha (angstrom)': mm(r, 'm4_ca_rmsd_median'),
                     'M4 ligand (angstrom)': mm(r, 'm4_ligand_rmsd_median'), 'M5 ranking score': mm(r, 'm5_ranking_score_median'),
                     'targets with identical single / pair embeddings': f'{r.single_identical_targets} / {r.pair_identical_targets}',
                     'bit-identical samples': f'{r.samples_bit_identical}/{r.samples}',
                     'same top-1': f'{r.same_top1_targets}/{r.targets}'} for r in sm.itertuples()])
  write_table(out, 'T20a_embed_pairs', t, 'Follow-up 2 (10 pilot targets, seed 1): divergence of the final trunk '
              'embeddings (relative L2, float16 read as float64) against divergence after diffusion (M4, M5 of the '
              'same samples), per pair: median over targets (maximum in brackets). The per-token-pair maximum is in '
              'embeddings.csv and not read (it divides by near-zero pair vectors)')
  e, st = read(EMBED / 'embeddings.csv'), read(EMBED / 'structures.csv')
  j = e.merge(st, on=['pair', 'target'])
  t = pd.DataFrame({'pair': j.pair, 'target': j.target, 'tokens': j.tokens, 'single rel L2': j.single_rel_l2.map(g3),
                    'pair rel L2': j.pair_rel_l2.map(g3), 'M4 C-alpha median (angstrom)': j.m4_ca_rmsd_median.map(g3),
                    'M4 ligand median (angstrom)': j.m4_ligand_rmsd_median.map(g3),
                    'M5 ranking score median': j.m5_ranking_score_median.map(g3),
                    'bit-identical samples': j.samples_bit_identical.astype(str) + '/5',
                    'same top-1': np.where(j.same_top1.astype(str) == 'True', 'yes', 'no')})
  write_table(out, 'T20b_embed_per_target', t, 'Follow-up 2, per pair and target: trunk embedding divergence and '
              'the divergence of the same samples after diffusion')
  vm = read(EMBED / 'versus_main.csv')
  pilot = sorted(vm.target.unique())
  rc = main_pairs_sample[main_pairs_sample.target.isin(pilot) & (main_pairs_sample.seed_a.astype(str) == '1')]
  rows = []
  for (c, rep), g in vm.groupby(['condition', 'repetition'], sort=False):
    h = rc[rc.contrast == f'RC({c})']
    same = h.groupby('target').bit_identical.apply(lambda s: bool(s.astype(str).eq('True').all())) if len(h) else pd.Series(dtype=bool)
    rows.append({'condition': c, 'repetition': rep, 'processes': len(g),
                 'bit-identical to the main study': int(g.identical.astype(str).eq('True').sum()),
                 'main study r1 against r2, targets with identical coordinates': f'{int(same.sum())}/{len(same)}' if len(same) else ''})
  order = {'V5': 0, 'V6': 1, 'G': 2, 'Gd': 3}
  t = pd.DataFrame(rows).sort_values(['condition', 'repetition'], key=lambda s: s.map(order) if s.name == 'condition' else s)
  write_table(out, 'T20c_embed_versus_main', t, 'Follow-up 2: processes whose outputs (masked mmCIF, summary '
              'confidences, ranking scores) are bit-identical to the main study\'s run of the same condition, '
              'repetition, target and seed, next to the main study\'s own r1 against r2 reproducibility on the same '
              'targets and seed (RC sample pairs, coordinates). G is not reproducible run to run, so its mismatches '
              'do not show an effect of --save_embeddings')
  return True


def t21_cpu60(out, d):
  """Follow-up 4 tables from an analysis/cpu60.py output folder."""
  con = read(d / 'contrasts.csv')
  t = wide_stats(con, ['flip_top1', 'flip_success', 'flip_matters', 'bit_identical_fraction'], pct_ci)
  t.columns = ['contrast', 'role', 'targets', 'top-1 flip (%)', 'success flip (%)', 'flips that matter (%)',
               'bit-identical samples (%)']
  write_table(out, 'T21a_cpu60_contrasts', t, 'Follow-up 4 (C on every target): flip rates of the contrasts with C '
              'on the 60 targets, 95% target bootstrap intervals')
  six = read(d / 'six_vs_sixty.csv')
  t = pd.DataFrame({'contrast': six.contrast, 'statistic': six.statistic,
                    '60 targets': [fmt_stat(r.statistic, r.new_estimate, r.new_ci_low, r.new_ci_high) for r in six.itertuples()],
                    'main study, 6 targets': [fmt_stat(r.statistic, r.main_estimate, r.main_ci_low, r.main_ci_high)
                                              for r in six.itertuples()]})
  write_table(out, 'T21b_cpu60_six_vs_sixty', t, 'Follow-up 4: each contrast on the 60 targets next to its value on the '
              'main study\'s 6 C targets, which stays as specified')
  s, m = read(d / 'cost_summary.csv'), read(d / 'memory.csv').set_index('bucket')
  t = pd.DataFrame({'bucket': s.bucket.astype(int), 'processes': s.processes, 'median wall (s)': s.median_wall_s.round(0),
                    'USD per structure, on demand': s.median_usd_per_structure_standard.round(3),
                    'USD per structure, Spot': s.median_usd_per_structure_spot.round(3),
                    'max RSS (GiB)': s.bucket.map(m.max_rss_gib).round(1)})
  write_table(out, 'T21c_cpu60_cost_memory', t, 'Follow-up 4: M6 and M7 for C at every token bucket (n2-highmem-16)')
  return con


def t22_a100(out, d):
  """Follow-up 3 tables from an analysis/a100.py output folder."""
  write_table(out, 'T22a_a100_checks', read(d / 'checks.csv').fillna(''), 'Follow-up 3: checks before any result '
              '(machine type, exit codes, check 2 of A)')
  con = read(d / 'contrasts.csv')
  t = wide_stats(con, ['flip_top1', 'flip_success', 'flip_matters', 'bit_identical_fraction'], pct_ci)
  t.columns = ['contrast', 'role', 'targets', 'top-1 flip (%)', 'success flip (%)', 'flips that matter (%)',
               'bit-identical samples (%)']
  write_table(out, 'T22b_a100_contrasts', t, 'Follow-up 3 (A100 80 GB, a2-ultragpu-1g): flip rates per contrast, 95% '
              'target bootstrap intervals')
  rows = []
  for (h, rc, ref), g in read(d / 'hw_minus_rc.csv').groupby(['hardware', 'recompilation', 'reference'], sort=False):
    row = {'hardware pair': h, 'recompilation': rc, 'reference': ref}
    for st in ['flip_top1', 'flip_success', 'flip_matters']:
      r = g[g.statistic == st].iloc[0]
      row[st.replace('_', ' ') + ' (pp)'] = pct_ci(r.estimate, r.ci_low, r.ci_high)
    rows.append(row)
  write_table(out, 'T22c_a100_hw_minus_rc', pd.DataFrame(rows), 'Follow-up 3: HW minus RC for HW(G,A) and HW(V6,A), '
              'against the RC of each condition (percentage points, 95% intervals)')
  c = read(d / 'cost_both_regions.csv')
  t = pd.DataFrame({'arm': c.platform, 'bucket': c.bucket.astype(int), 'warm': c.warm,
                    'median wall per structure (s)': c.wall_per_structure_s.round(0),
                    **{f'USD, {p.replace("_", "-")}, {g}': c[f'usd_per_structure_{p}_{g}'].round(3)
                       for g in [x.rsplit('_', 1)[1] for x in c.columns if x.startswith('usd_per_structure_standard_')]
                       for p in ('standard', 'spot', 'flex_start')}})
  write_table(out, 'T22d_a100_cost', t, 'Follow-up 3: M6, median cost per structure at the a2-ultragpu-1g list prices '
              'of every region with A100 80 GB quota')
  return con


# ---------------------------------------------------------------- figures

def figure_saver(out):
  """(pyplot, save): deterministic PDF and PNG (no dates), each figure registered with its role."""
  import matplotlib
  matplotlib.use('Agg')
  import matplotlib.pyplot as plt
  plt.rcParams.update({'font.size': 8, 'axes.spines.top': False, 'axes.spines.right': False, 'svg.hashsalt': 'af3',
                       'pdf.compression': 6})
  d = out / 'figures'
  d.mkdir(parents=True, exist_ok=True)

  def save(fig, name, what):
    fig.savefig(d / f'{name}.pdf', metadata={'CreationDate': None, 'ModDate': None, 'Producer': None})
    fig.savefig(d / f'{name}.png', dpi=200, metadata={'Software': None})
    plt.close(fig)
    ASSETS[f'figures/{name}'] = (ROLES[name], what)
  return plt, save


def figures(out, contrasts, hwrc, q3, sample_pairs, cost, mem, sbs, cross, recs):
  plt, save = figure_saver(out)

  def forest(ax, df, label_col, title):
    y = np.arange(len(df))[::-1]
    ax.errorbar(100 * df.estimate, y, xerr=[100 * (df.estimate - df.ci_low), 100 * (df.ci_high - df.estimate)],
                fmt='o', ms=3, color='k', elinewidth=0.8, capsize=2)
    ax.set_yticks(y, df[label_col])
    ax.axvline(0, color='0.6', lw=0.6)
    ax.set_title(title)

  order = [c for c in by_role(contrasts).contrast.unique() if not c.startswith(('W(', 'SD('))]
  fig, axes = plt.subplots(1, 3, figsize=(7.2, 4.6), sharey=True)
  for ax, st, title in zip(axes, ['flip_top1', 'flip_success', 'flip_matters'],
                           ['top-1 flip (%)', 'success flip (%)', 'flips that matter (%)']):
    g = contrasts[contrasts.statistic == st].set_index('contrast').loc[order].reset_index()
    forest(ax, g, 'contrast', title)
  fig.tight_layout()
  save(fig, 'F1_flip_rates', 'Flip rates per contrast with 95% target-clustered bootstrap intervals (warm-cache '
       'checks and seed yardsticks omitted)')

  g = hwrc[(hwrc.reference == 'RC(P)')]
  fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.8), sharey=True)
  for ax, st, title in zip(axes, ['flip_top1', 'flip_success', 'flip_matters'],
                           ['top-1 flip (pp)', 'success flip (pp)', 'flips that matter (pp)']):
    h = g[g.statistic == st].assign(label=lambda x: x.hardware + ' - ' + x.recompilation)
    forest(ax, h, 'label', title)
  fig.tight_layout()
  save(fig, 'F2_hw_minus_rc', 'Hardware minus recompilation: flip-rate differences with 95% intervals')

  fig, ax = plt.subplots(figsize=(3.6, 2.8))
  for c in ['RC(G)', 'DET', 'DU', 'HW(V5,V6)', 'HW(V6,G)', 'HW(G,C)']:
    v = np.sort(sample_pairs[sample_pairs.contrast == c].m4_ligand_rmsd.dropna().values)
    if len(v):
      ax.step(np.maximum(v, 1e-4), np.arange(1, len(v) + 1) / len(v), where='post', label=c, lw=1)
  ax.set_xscale('log')
  ax.set_xlabel('M4 ligand RMSD (angstrom; 0 drawn at 1e-4)')
  ax.set_ylabel('fraction of sample pairs')
  ax.legend(frameon=False, fontsize=6)
  fig.tight_layout()
  save(fig, 'F3_m4_ligand_rmsd', 'Distribution of M4 ligand RMSD between paired samples, selected contrasts')

  fig, ax = plt.subplots(figsize=(3.6, 2.8))
  r = q3[q3.statistic == 'success_rate']
  for c in [a for a in ARMS if a in set(r.condition)]:
    h = r[r.condition == c]
    ax.plot(h.n_seeds, 100 * h.estimate, marker='o', ms=3, lw=1, label=f'{c} ({int(h.n_targets.iloc[0])} targets)')
  ax.set_xlabel('seeds')
  ax.set_ylabel('success of overall top-1 (%)')
  ax.legend(frameon=False, fontsize=6)
  fig.tight_layout()
  save(fig, 'F4_q3', 'Q3: success rate of the overall top-1 against the number of seeds, per condition')

  qa = read(EXPLORATORY / 'a_margin_quintiles.csv')
  fig, ax = plt.subplots(figsize=(3.6, 2.8))
  for c in ['HW(V5,V6)', 'HW(V6,G)', 'RC(G)', 'DET']:
    h = qa[qa.contrast == c]
    ax.errorbar(h.margin_median, 100 * h.flip_rate, yerr=[100 * (h.flip_rate - h.ci_low), 100 * (h.ci_high - h.flip_rate)],
                marker='o', ms=3, lw=1, capsize=2, label=c)
  ax.set_xscale('log')
  ax.set_xlabel('M3 margin of the first condition (quintile median)')
  ax.set_ylabel('top-1 flip rate (%)')
  ax.legend(frameon=False, fontsize=6)
  fig.tight_layout()
  save(fig, 'F5_flip_vs_margin', 'Exploratory (a): flip rate by margin quintile, 95% intervals')

  cp = read(EXPLORATORY / 'c_perturbation.csv')
  fig, ax = plt.subplots(figsize=(3.2, 3.0))
  ax.errorbar(100 * cp.predicted_flip_rate, 100 * cp.observed_flip_rate,
              xerr=[100 * (cp.predicted_flip_rate - cp.predicted_ci_low), 100 * (cp.predicted_ci_high - cp.predicted_flip_rate)],
              yerr=[100 * (cp.observed_flip_rate - cp.observed_ci_low), 100 * (cp.observed_ci_high - cp.observed_flip_rate)],
              fmt='o', ms=3, color='k', elinewidth=0.7, capsize=2)
  offsets = {'HW(V5,V6)': (-34, 4), 'HWh(V6,G)': (4, -9), 'RC(G)': (4, -9), 'HW(V6,G)': (4, -9)}
  for row in cp.itertuples():
    ax.annotate(row.contrast, (100 * row.predicted_flip_rate, 100 * row.observed_flip_rate), fontsize=5,
                xytext=offsets.get(row.contrast, (4, 3)), textcoords='offset points')
  lim = 5 + 100 * max(cp.observed_ci_high.max(), cp.predicted_ci_high.max())
  ax.plot([0, lim], [0, lim], color='0.6', lw=0.6)
  ax.set_xlim(0, lim)
  ax.set_ylim(0, lim)
  ax.set_xlabel('predicted flip rate (%)')
  ax.set_ylabel('observed flip rate (%)')
  fig.tight_layout()
  save(fig, 'F6_predicted_vs_observed', 'Exploratory (c): flip rate predicted from perturbation size and margin '
       'against the observed rate')

  c = cost[cost.cache == 'fresh'] if 'cache' in cost.columns else cost
  fig, ax = plt.subplots(figsize=(3.6, 2.8))
  for p in [a for a in ARMS if a in set(c.platform)]:
    h = c[c.platform == p].sort_values('bucket')
    ax.plot(h.bucket, h.median_usd_per_structure_standard, marker='o', ms=3, lw=1, label=p)
  ax.set_yscale('log')
  ax.set_xlabel('token bucket')
  ax.set_ylabel('USD per structure (on demand, median)')
  ax.legend(frameon=False, fontsize=6, ncol=2)
  fig.tight_layout()
  save(fig, 'F7_cost', 'M6: cost per structure by token bucket and platform, on-demand list prices')

  fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8))
  ax = axes[0]
  chip = {'V5': 'v5e', 'V6': 'v6e', 'V6h': 'v6e', 'G': 'L4', 'GT': 'L4', 'Gd': 'L4', 'Gh': 'L4'}
  limits = {}
  for p in [a for a in ARMS if a in set(mem.platform) and a != 'C']:
    h = mem[mem.platform == p].sort_values('bucket')
    ax.plot(h.bucket, h.compiled_peak_plus_args_gib, marker='o', ms=3, lw=1, label=p)
    if h.device_limit_gib.notna().any():
      limits[chip[p]] = float(h.device_limit_gib.dropna().iloc[0])
  for name, gib in sorted(limits.items()):
    ax.axhline(gib, color='k', lw=0.6, ls=':')
    ax.text(ax.get_xlim()[0], gib, f' {name} limit {gib:.1f} GiB', fontsize=6, va='bottom')
  ax.set_xlabel('token bucket')
  ax.set_ylabel('compiled peak + arguments (GiB)')
  ax.set_title('main study')
  ax.legend(frameon=False, fontsize=6, ncol=2)
  ax = axes[1]
  if len(recs):
    import memlimit
    t = memlimit.process_table(AF3, PRIMARY / 'samples.csv')
    limits = {}
    for arm in memlimit.ARMS:
      g = t[t.arm == arm].sort_values('size')
      ok = g[(g.exit_code == 0) & (g.failure == '')]
      line, = ax.plot(ok['size'], ok.compiled_peak_gib, marker='o', ms=3, lw=1, label=arm)
      main = ok[ok.source != 'memlimit']
      ax.plot(main['size'], main.compiled_peak_gib, marker='o', ms=5, mfc='white', ls='', color=line.get_color())
      for r in g[(g.exit_code != 0) | (g.failure != '')].itertuples():
        y = r.error_required_gib if not pd.isna(r.error_available_gib) else r.compiled_peak_gib
        ax.plot([ok['size'].iloc[-1], r.size], [ok.compiled_peak_gib.iloc[-1], y], ls=':', lw=1, color=line.get_color())
        ax.plot([r.size], [y], marker='x', ms=6, ls='', color=line.get_color())
      if g.device_limit_gib.notna().any():
        limits[{'V5': 'v5e', 'V6': 'v6e'}.get(arm, 'L4')] = float(g.device_limit_gib.dropna().iloc[0])
    for name, lim in sorted(limits.items()):
      ax.axhline(lim, color='k', lw=0.6, ls='--')
      ax.text(1000, lim, f' {name} {lim:.1f} GiB', fontsize=6, va='bottom')
    ax.set_xticks([1024] + MEMLIMIT_SIZES[:5])
    ax.tick_params(axis='x', labelsize=6)
    ax.legend(frameon=False, fontsize=6)
    ax.set_title('memory limit, 7D5C padded (x: failed size)')
  else:
    ax.text(0.5, 0.5, 'memory-limit runs pending\n(sizes 1280 to 5120, 7D5C)', ha='center', va='center',
            transform=ax.transAxes)
    ax.set_xlim(1024, 5376)
    ax.set_xticks(MEMLIMIT_SIZES[::2])
    ax.set_yticks([])
    ax.set_title('memory limit (placeholder)')
  ax.set_xlabel('bucket size (tokens)')
  ax.set_ylabel('compiled peak (GiB)')
  fig.tight_layout()
  save(fig, 'F8_memory', 'M7: device memory against token bucket in the main study (left), and the memory-limit '
       'measurement (right: compiled peak per size, open circle the main study\'s 1,024 point, x the failed size at the '
       'memory its error or its compiled program reports, dashed the device limits)' + ('' if len(recs) else ' (placeholder panel until those runs exist)'))

  s = sbs[sbs.section == 'success rate, r1']
  s = s[s.statistic == 'success_rate_overall_top1']
  fig, ax = plt.subplots(figsize=(3.6, 2.6))
  x = np.arange(len(s))
  for i, (v, label) in enumerate((('primary', '(1) primary'), ('without_three', '(2) 57 targets'),
                                  ('rescored', '(3) rescored'))):
    ax.bar(x + (i - 1) * 0.27, 100 * s[f'{v}_estimate'], width=0.27, label=label)
  ax.set_xticks(x, s.name)
  ax.set_ylabel('success of overall top-1 (%)')
  ax.set_ylim(50, 90)
  ax.legend(frameon=False, fontsize=6)
  fig.tight_layout()
  save(fig, 'F9_glycans', 'M1 success rate per condition with the three glycan targets as specified, without them, '
       'and rescored (post hoc)')

  fig, ax = plt.subplots(figsize=(3.2, 2.4))
  arms = [a for a in ARMS if a in set(cross.condition)]
  frac = [100 * (cross[cross.condition == a].strict == 'identical').mean() for a in arms]
  ax.bar(arms, frac, color='0.3')
  ax.set_ylabel('(target, seed) bit-identical (%)')
  ax.set_ylim(0, 105)
  fig.tight_layout()
  save(fig, 'F10_cross_session', 'Cross-session check: share of (target, seed) units whose outputs are bit-identical '
       'between pilot2 and the main study')


def figure_a100(out, con, main_contrasts):
  plt, save = figure_saver(out)
  names = [c for c in ['HW(G,A)', 'HW(V6,A)', 'HW(V5,A)', 'HW(A,C)', 'RC(A)', 'DU(A)', 'PR(A)'] if c in set(con.contrast)]
  fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=True)
  y = np.arange(len(names))[::-1]
  for ax, st, title in zip(axes, ['flip_top1', 'flip_success'], ['top-1 flip per seed (%)', 'success flip per seed (%)']):
    g = con[con.statistic == st].set_index('contrast').reindex(names)
    ax.errorbar(100 * g.estimate, y, xerr=[100 * (g.estimate - g.ci_low), 100 * (g.ci_high - g.estimate)], fmt='o',
                ms=3.5, elinewidth=0.8, capsize=2, color='k')
    ax.set_yticks(y, names)
    ax.axvline(0, color='0.6', lw=0.6)
    ax.set_title(title)
  fig.tight_layout()
  save(fig, 'F14_a100_flip_rates', 'Follow-up 3: per-seed top-1 and success flip rates of the A100 contrasts, 95% '
       'target bootstrap intervals')


def figures_followups(out, main_contrasts, have_ppi, have_embed):
  plt, save = figure_saver(out)
  if have_ppi:
    con, ov = read(PPI / 'contrasts.csv'), read(PPI / 'overall_top1.csv')
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9), sharey=True)
    y = np.arange(len(PPI_CONTRASTS))[::-1]
    for ax, st, title in zip(axes, ['flip_top1', 'flip_success'], ['top-1 flip per seed (%)', 'success flip per seed (%)']):
      for src, dy, marker, label in ((main_contrasts, 0.15, 'o', 'main study (60 protein-ligand targets)'),
                                     (con, -0.15, 's', 'follow-up 1 (30 protein-protein targets)')):
        g = src[src.statistic == st].set_index('contrast').reindex(PPI_CONTRASTS)
        ax.errorbar(100 * g.estimate, y + dy, xerr=[100 * (g.estimate - g.ci_low), 100 * (g.ci_high - g.estimate)],
                    fmt=marker, ms=3.5, elinewidth=0.8, capsize=2, label=label,
                    mfc='white' if marker == 's' else None)
      ax.set_yticks(y, PPI_CONTRASTS)
      ax.axvline(0, color='0.6', lw=0.6)
      ax.set_title(title)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=6, loc='upper center', ncol=2)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    save(fig, 'F11_ppi_flip_rates', 'Follow-up 1 against the main study: per-seed top-1 and success flip rates per '
         'contrast, 95% target bootstrap intervals (success: DockQ 0.23 in the follow-up, M1 in the main study)')
    pt = read(PPI / 'per_target.csv')
    d = read(PPI / 'samples_dockq.csv', low_memory=False)
    fig, ax = plt.subplots(figsize=(7.2, 2.8))
    x = np.arange(len(pt))
    rng = d.groupby('target').DockQ.agg(['min', 'max']).reindex(pt.target)
    ax.vlines(x, rng['min'], rng['max'], color='0.85', lw=3, label='range over every sample of every condition')
    for i, (c, marker) in enumerate(zip(PPI_COLUMNS, ['o', 's', 'D', '^', 'v'])):
      ax.plot(x + (i - 2) * 0.12, pt[f'{c}_top1_dockq'], marker=marker, ls='', ms=3, label=f'{c[:-2]} {c[-2:]} overall top-1',
              mfc='white' if c.endswith('r2') else None)
    for thr, label in ((0.23, 'success 0.23'), (0.49, 'secondary 0.49')):
      ax.axhline(thr, color='k', lw=0.6, ls=':' if thr < 0.3 else '--')
      ax.text(len(pt) - 0.5, thr, label, fontsize=6, va='bottom', ha='right')
    ax.set_xticks(x, [t + ('*' if str(s) == 'True' else '') for t, s in zip(pt.target, pt.shallow)], rotation=90, fontsize=6)
    ax.set_xlim(-0.6, len(pt) - 0.4)
    ax.set_ylim(0, 1)
    ax.set_ylabel('DockQ')
    ax.legend(frameon=False, fontsize=5.5, ncol=3, loc='lower left')
    fig.tight_layout()
    save(fig, 'F12_ppi_dockq_per_target', 'Follow-up 1, per target (draw order; * shallow alignment): DockQ of the '
         'overall top-1 per condition and repetition, over the range of all samples, with the 0.23 and 0.49 thresholds')
  if have_embed:
    e, st = read(EMBED / 'embeddings.csv'), read(EMBED / 'structures.csv')
    j = e.merge(st, on=['pair', 'target'])
    floor = 3e-3   # bit-identical trunk outputs (relative L2 0) are drawn here, open
    fig, ax = plt.subplots(figsize=(3.8, 3.2))
    for pair, marker in zip(EMBED_PAIRS, ['o', 's', '^', 'D']):
      g = j[j.pair == pair]
      zero = g.pair_rel_l2 == 0
      line, = ax.plot(g.pair_rel_l2[~zero], np.maximum(g.m4_ca_rmsd_median[~zero], 1e-4), marker=marker,
                      ls='', ms=3.5, label=pair)
      if zero.any():
        ax.plot(np.full(int(zero.sum()), floor), np.maximum(g.m4_ca_rmsd_median[zero], 1e-4), marker=marker, ls='', ms=3.5,
                mfc='white', color=line.get_color())
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.axvline(4.5e-3, color='0.7', lw=0.6, ls=':')
    import matplotlib.ticker as mticker
    ax.xaxis.set_minor_formatter(mticker.NullFormatter())
    ax.set_xticks([floor, 0.01, 0.02, 0.05], ['identical', '0.01', '0.02', '0.05'])
    ax.set_xlabel('trunk pair embedding, relative L2')
    ax.set_ylabel('M4 C-alpha RMSD, sample median (angstrom)')
    ax.legend(frameon=False, fontsize=6)
    fig.tight_layout()
    save(fig, 'F13_embed_trunk_vs_structure', 'Follow-up 2: divergence of the final trunk pair embedding against '
         'the C-alpha divergence of the same samples after diffusion, per target and pair (bit-identical trunk '
         'outputs drawn open at the left, labelled identical)')


def asset_key(name):
  """tables before figures, then by number (T2 before T10), then by suffix."""
  kind, base = name.split('/')
  m = re.match(r'[TF](\d+)(.*)', base)
  return (kind != 'tables', int(m.group(1)), m.group(2))


def check_freeze(rows):
  """The sources under results/af3/ must be in the data freeze with the same SHA-256; returns their paths."""
  if hashlib.sha256(FREEZE.read_bytes()).hexdigest() != FREEZE_ID:
    raise SystemExit(f'{FREEZE} is not freeze {FREEZE_ID}')
  frozen = {}
  for line in FREEZE.read_text().splitlines()[1:]:
    session, path, _, sha, _ = line.split('\t')
    frozen[f'results/af3/{session}/{path}'] = sha
  bad = [p for p, h in rows if p.startswith('results/af3/') and frozen.get(p) != h]
  if bad:
    raise SystemExit(f'{len(bad)} source(s) under results/af3/ differ from freeze {FREEZE_ID[:12]} or are not in it: '
                     + ', '.join(bad[:5]))
  return {p for p, _ in rows if p.startswith('results/af3/')}


def main():
  ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
  ap.add_argument('--out', default=str(REPO / 'results' / 'paper_assets'))
  ap.add_argument('--tables_only', action='store_true')
  ap.add_argument('--followups34', action='store_true', help='add the follow-up 3 and 4 assets (off until their '
                  'analyses exist; with it off, the other assets are unchanged)')
  ap.add_argument('--cpu60_dir', help='analysis/cpu60.py output folder (with --followups34)')
  ap.add_argument('--a100_dir', help='analysis/a100.py output folder (with --followups34)')
  args = ap.parse_args()
  out = pathlib.Path(args.out)
  out.mkdir(parents=True, exist_ok=True)
  c = PRIMARY / 'contrasts'
  samples = read(PRIMARY / 'samples.csv', low_memory=False)
  contrasts, hwrc, q3 = read(c / 'contrasts.csv'), read(c / 'hw_minus_rc.csv'), read(c / 'q3.csv')
  checks, sens = read(c / 'checks.csv'), read(c / 'sensitivity_post_cutoff.csv')
  cost, mem, prices = read(PRIMARY / 'cost' / 'cost_summary.csv'), read(PRIMARY / 'cost' / 'memory.csv'), \
      read(PRIMARY / 'cost' / 'prices_used.csv')
  sbs, cross = read(GLYCANS / 'side_by_side.csv'), read(CROSS)
  t01_platforms(out, samples, prices)
  t02_targets(out)
  t03_checks(out, checks)
  t04_flips(out, contrasts)
  t05_hw_rc(out, hwrc)
  t06_m4_m5(out, contrasts)
  t07_q3(out, q3, sens)
  t08_glycans(out, sbs)
  t09_cost(out, cost)
  t10_memory(out, mem)
  t11_cross(out, cross)
  t12_cutoff(out, sens)
  t13_t16_exploratory(out)
  t17_audit(out)
  recs = t18_memlimit(out)
  pairs_sample = read(c / 'pairs_sample.csv')
  have_ppi = t19_ppi(out, contrasts, hwrc)
  have_embed = t20_embed(out, pairs_sample)
  if not args.tables_only:
    figures(out, contrasts, hwrc, q3, pairs_sample, cost, mem, sbs, cross, recs)
    figures_followups(out, contrasts, have_ppi, have_embed)
  if args.followups34:
    if not (args.cpu60_dir or args.a100_dir):
      raise SystemExit('--followups34 needs --cpu60_dir and/or --a100_dir')
    if args.cpu60_dir:
      t21_cpu60(out, REPO / args.cpu60_dir)
    if args.a100_dir:
      a_con = t22_a100(out, REPO / args.a100_dir)
      if not args.tables_only:
        figure_a100(out, a_con, contrasts)
  rows = sorted((str(p.relative_to(REPO)), hashlib.sha256(p.read_bytes()).hexdigest()) for p in SOURCES)
  frozen = check_freeze(rows)
  (out / 'sources.tsv').write_text('path\tsha256\tin_freeze\n' + ''.join(
      f'{p}\t{h}\t{"yes" if p in frozen else "no (derived analysis output or repository file)"}\n' for p, h in rows))
  lines = ['# Paper assets', '', 'Generated by analysis/paper_assets.py from the frozen analysis outputs (sources.tsv '
           'lists every input with its SHA-256). Regenerate: '
           '`analysis/figures/.venv/bin/python analysis/paper_assets.py`.', '',
           f'Data freeze: `{FREEZE_ID}` (SHA-256 of results/freeze/manifest_sha256.tsv). Every source under '
           f'results/af3/ ({len(frozen)} files) matches it byte for byte; the other sources are analysis outputs '
           'derived from those sessions, and repository files.', '',
           'Each line: asset, [role as section 10 of notes/analysis_plan.md labels it], what it shows.', '']
  if args.followups34:
    if hashlib.sha256(FREEZE34.read_bytes()).hexdigest() != FREEZE34_ID:
      raise SystemExit(f'{FREEZE34} is not freeze {FREEZE34_ID}')
    lines[-2:-2] = [f'Follow-ups 3 and 4 (T21, T22, F14): their analyses read the sessions of the extended data freeze '
                    f'`{FREEZE34_ID}` (SHA-256 of {FREEZE34.relative_to(REPO)}), which holds every file of freeze '
                    f'`{FREEZE_ID[:8]}` unchanged and the 66 cpu60 and a100 sessions.', '']
  lines += [f'- `{k}` [{role}]: {v}' for k, (role, v) in sorted(ASSETS.items(), key=lambda kv: asset_key(kv[0]))]
  (out / 'README.md').write_text('\n'.join(lines) + '\n')
  print(f'>> {sum(k.startswith("tables/") for k in ASSETS)} tables, {sum(k.startswith("figures/") for k in ASSETS)} '
        f'figures, {len(rows)} sources; {out}')
  return 0


if __name__ == '__main__':
  sys.exit(main())
