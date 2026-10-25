"""Follow-up 6, AlphaFold3's own data pipeline (notes/analysis_plan.md section 10, row of 2026-10-19; label msa;
reported as a follow-up, outside the pre-registered study, whatever it shows). Computes what the row lists, and
nothing else. Written before any inference run; the inference part runs only after the arms finish.

    analysis/.venv/bin/python analysis/msa.py --out results/analysis/msa_<date> [--samples .../samples.csv]
        [--descriptive_only] [--fetch_rcsb] [--main results/analysis/main_20261014/primary_v2]
        [--replicates 10000] [--seed 2026]

Descriptive, without inference (inputs/official_data/manifest.csv and data/nas/, as scripts/nas/ingest_official.py
froze them):
  msa_depth.csv    per target, official (the naspaz8-af3 run1 _data.json) against ColabFold (the frozen input of
                   inputs/manifest.csv): unpaired and paired MSA rows, and Neff of each, as fixed here: the a3m rows
                   with insertions (lower-case letters) removed, so every row has the query's length L; 22 symbols
                   (20 amino acids, any other letter, the gap); the identity of two rows is the fraction of the L
                   columns where they hold the same symbol, a gap matching a gap (the convention of Morcos et al. 2011,
                   PNAS 108:E1293); each row's weight is 1 over the number of rows, itself included, at identity 0.8 or
                   more; Neff is the sum of the weights. The query row is counted. An empty MSA has Neff 0.
  templates.csv    the template audit: per official template, its PDB ID and chain, release date, sequence identity to
                   the query over the aligned pairs (queryIndices to templateIndices, the template sequence from its
                   mmCIF _entity_poly_seq), aligned length and query coverage, and whether that PDB entry contains the
                   target's ligand (CCD code; the template mmCIF holds the hit's chain only, so entry-level ligands and
                   any release date the mmCIF lacks come from RCSB, cached in rcsb_templates.json; --fetch_rcsb fills
                   the cache, otherwise unknown); flagged: the target's own entry, identity 0.95 or more.
  pipeline.csv     per target, wall seconds and VM memory (baseline, peak, peak minus baseline) of naspaz8-af3 run1 only.
  determinism.csv  the 10 pilot targets: _data.json SHA-256 of naspaz8-af3 run1, run2 and naspaz6-af3 run1, and when
                   they differ, the difference by field (unpaired and paired MSA rows: counts and rows in one only;
                   templates: hits in one only).
Inference (--samples: analysis/collect.py on the msa sessions; conditions V5o, V6o, Go, the platform code plus o):
  completeness.csv  per arm and repetition, the (target, seed) with five samples, against the plan; an arm short of
                    them is reported as incomplete, never imputed, and every result that uses it says so.
  contrasts.csv     with the statistics of section 5 and the target-clustered bootstrap: HW(V5o,V6o), HW(V6o,Go),
                    HW(V5o,Go), RC(Go), W(V5o), W(V6o), SD(V5o), SD(V6o), SD(Go); and IN(P), P against Po on the same
                    chip (ColabFold against official inputs) for P in V5, V6, G, as a yardstick.
  hw_minus_rc.csv   HW(V6o,Go) minus RC(Go), each flip rate, on the same resampled targets.
  official_minus_colabfold.csv  on the same (target, seed): each flip rate of HW(V5o,V6o), HW(V6o,Go), HW(V5o,Go)
                    minus that of HW(V5,V6), HW(V6,G), HW(V5,G) (main study, primary_v2 seed pairs), paired bootstrap
                    over targets; read as lower, similar or higher if the interval lies below 0, contains 0, or lies
                    above 0.
  success_rates.csv M1 success rate of the top-1 per (target, seed), per condition (V5, V6, G main study r1; V5o,
                    V6o, Go r1, and Go r2), with a target bootstrap interval.
  reading_rule.csv  the row's reading rule: the main conclusion holds with official inputs if HW(V6o,Go) minus
                    RC(Go) and HW(V5o,V6o) both have a 95% interval above 0; with the pair readings above. Headline
                    statistic: the top-1 flip rate (flip_top1), the one the main study's primary result is about
                    (section 10 implementation row of follow-up 6); written first. The success flip rate and flips that
                    matter are reported alongside as secondary. If Go r2 did not run, RC(Go) is not available: HW(V6o,Go)
                    minus RC(Go) is reported as not available, the rule is not applied to that pair, and the main
                    study's RC(G) is not substituted.
Every result that uses M1 (success flips, flips that matter, success rates, and the rule and comparisons on them) is
also given on the 39 targets released after 2021-09-30 (column targets: post-cutoff 39).
Outputs in --out also: counts.csv, pairs_seed.csv, pairs_sample.csv, report.md.
"""
import argparse
import csv
import hashlib
import json
import pathlib
import re
import sys
import urllib.request

import numpy as np
import pandas as pd

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / 'analysis'))
import contrasts  # noqa: E402
import followup34 as f34  # noqa: E402

OFFICIAL = REPO / 'inputs' / 'official_data' / 'manifest.csv'
COLABFOLD = REPO / 'inputs' / 'manifest.csv'
NAS = REPO / 'data' / 'nas'
RUN1, RUN2, NAS6 = 'naspaz8-af3_run1', 'naspaz8-af3_run2', 'naspaz6-af3_run1'
CODE = {'V5': 'V5o', 'V6': 'V6o', 'G': 'Go'}
SPECS = [('HW(V5o,V6o)', 'HW', ('V5o', 'r1'), ('V6o', 'r1'), True),
         ('HW(V6o,Go)', 'HW', ('V6o', 'r1'), ('Go', 'r1'), True),
         ('HW(V5o,Go)', 'HW', ('V5o', 'r1'), ('Go', 'r1'), False),
         ('RC(Go)', 'RC', ('Go', 'r1'), ('Go', 'r2'), True),
         ('W(V5o)', 'W', ('V5o', 'r1'), ('V5o', 'w'), True), ('W(V6o)', 'W', ('V6o', 'r1'), ('V6o', 'w'), True),
         ('SD(V5o)', 'SD', ('V5o', 'r1'), ('V5o', 'r1'), True), ('SD(V6o)', 'SD', ('V6o', 'r1'), ('V6o', 'r1'), True),
         ('SD(Go)', 'SD', ('Go', 'r1'), ('Go', 'r1'), True)]
IN_SPECS = [(f'IN({p})', 'IN', (p, 'r1'), (CODE[p], 'r1'), False) for p in ('V5', 'V6', 'G')]
COMPARE = [('HW(V5,V6)', 'HW(V5o,V6o)'), ('HW(V6,G)', 'HW(V6o,Go)'), ('HW(V5,G)', 'HW(V5o,Go)')]
RULE_HW, RULE_RC = 'HW(V6o,Go)', 'RC(Go)'
PLAN = {('Go', 'r1'): 300, ('Go', 'r2'): 300, ('V6o', 'r1'): 300, ('V5o', 'r1'): 300, ('V6o', 'w'): 10, ('V5o', 'w'): 10}
M1_STATS = ('flip_success', 'flip_matters')
HEADLINE = 'flip_top1'   # the main study's primary result is about the top-1 flip rate (implementation row)
contrasts.ROLE.setdefault('IN', 'input contrast (yardstick)')

# ---------------------------------------------------------------- MSA depth and Neff

AA = 'ACDEFGHIKLMNPQRSTVWY'
SYMBOL = {c: i for i, c in enumerate(AA)}
GAP, OTHER, NSYM = 21, 20, 22


def a3m_rows(text):
  """The aligned rows of an a3m: insertions (lower case) and '.' removed; [] for an empty MSA."""
  seqs, cur = [], None
  for line in (text or '').splitlines():
    if line.startswith('>'):
      if cur is not None:
        seqs.append(''.join(cur))
      cur = []
    elif cur is not None:
      cur.append(re.sub(r'[a-z.]', '', line.strip()))
  if cur is not None:
    seqs.append(''.join(cur))
  return seqs


def msa_matrix(rows):
  if not rows:
    return np.zeros((0, 0), dtype=np.int8)
  L = len(rows[0])
  bad = [i for i, r in enumerate(rows) if len(r) != L]
  if bad:
    raise ValueError(f'a3m rows {bad[:3]} do not have the query length {L} after removing insertions')
  m = np.full((len(rows), L), OTHER, dtype=np.int8)
  for i, r in enumerate(rows):
    m[i] = [GAP if c == '-' else SYMBOL.get(c, OTHER) for c in r]
  return m


def neff(matrix, threshold=0.8, block=1024):
  """Sum over rows of 1 / (rows at identity >= threshold, itself included); identity over all L columns, gap = gap."""
  n, L = matrix.shape
  if n == 0:
    return 0.0
  onehot = [(matrix == s).astype(np.float32) for s in range(NSYM)]
  need = threshold * L - 1e-6
  total = 0.0
  for i in range(0, n, block):
    counts = np.zeros((min(block, n - i), n), dtype=np.float32)
    for a in onehot:
      counts += a[i:i + block] @ a.T
    total += float((1.0 / (counts >= need).sum(axis=1)).sum())
  return total


def protein(d):
  return next(e['protein'] for e in d['sequences'] if 'protein' in e)


def load_checked(path, sha):
  raw = (REPO / path).read_bytes()
  if hashlib.sha256(raw).hexdigest() != sha:
    raise SystemExit(f'{path}: SHA-256 differs from its manifest')
  return json.loads(raw)


def msa_depth(official, colabfold):
  rows = []
  for t in sorted(official):
    o = protein(load_checked(official[t]['input_path'], official[t]['input_sha256']))
    c = protein(load_checked(colabfold[t]['input_path'], colabfold[t]['input_sha256']))
    row = {'pdb_id': t, 'post_cutoff': official[t]['released_on_or_before_cutoff'] == 'no'}
    for src, p in (('colabfold', c), ('official', o)):
      for kind in ('unpaired', 'paired'):
        r = a3m_rows(p.get(f'{kind}Msa'))
        row[f'{src}_{kind}_rows'] = len(r)
        row[f'{src}_{kind}_neff80'] = round(neff(msa_matrix(r)), 3)
      row[f'{src}_templates'] = len(p.get('templates') or [])
    rows.append(row)
  return pd.DataFrame(rows)

# ---------------------------------------------------------------- template audit

THREE = {'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C', 'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H',
         'ILE': 'I', 'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P', 'SER': 'S', 'THR': 'T', 'TRP': 'W',
         'TYR': 'Y', 'VAL': 'V', 'MSE': 'M'}


def cif_items(text):
  """{tag: [values]} of an mmCIF data block (single items and loops; values without spaces or quoted)."""
  out, lines, i = {}, text.splitlines(), 0
  tok = lambda s: [a or b or c for a, b, c in re.findall(r"'([^']*)'|\"([^\"]*)\"|(\S+)", s)]  # noqa: E731
  while i < len(lines):
    line = lines[i].strip()
    if line == 'loop_':
      tags, i = [], i + 1
      while i < len(lines) and lines[i].strip().startswith('_'):
        tags.append(lines[i].strip())
        i += 1
      vals = []
      while i < len(lines) and lines[i].strip() and not lines[i].strip().startswith(('_', 'loop_', '#', 'data_')):
        vals += tok(lines[i])
        i += 1
      for k, tag in enumerate(tags):
        out[tag] = vals[k::len(tags)]
      continue
    if line.startswith('_') and len(line.split(None, 1)) == 2:
      tag, value = line.split(None, 1)
      out[tag] = tok(value)[:1]
    elif line.startswith('data_'):
      out['data_'] = [line[5:]]
    i += 1
  return out


def template_info(mmcif):
  it = cif_items(mmcif)
  pdb = (it.get('_entry.id') or it.get('data_') or [''])[0].upper()
  seq = ''.join(THREE.get(m, 'X') for m in it.get('_entity_poly_seq.mon_id', []))
  dates = sorted(it.get('_pdbx_audit_revision_history.revision_date', []))
  chain = (it.get('_entity_poly.pdbx_strand_id') or [''])[0].split(',')[0]
  return {'pdb_id': pdb, 'chain': chain, 'sequence': seq, 'release_date': dates[0] if dates else None}


def rcsb_meta(pdb_ids, cache_path, fetch):
  cache = json.loads(cache_path.read_text()) if cache_path.is_file() else {}
  todo = sorted(set(pdb_ids) - set(cache))
  if todo and fetch:
    for i in range(0, len(todo), 100):
      ids = todo[i:i + 100]
      q = ('{entries(entry_ids: %s) {rcsb_id rcsb_accession_info {initial_release_date} '
           'nonpolymer_entities {nonpolymer_comp {chem_comp {id}}}}}' % json.dumps(ids))
      req = urllib.request.Request('https://data.rcsb.org/graphql', data=json.dumps({'query': q}).encode(),
                                   headers={'Content-Type': 'application/json'})
      for e in json.load(urllib.request.urlopen(req, timeout=60))['data']['entries'] or []:
        cache[e['rcsb_id']] = {'release': (e['rcsb_accession_info'] or {}).get('initial_release_date', '')[:10],
                               'ligands': sorted({n['nonpolymer_comp']['chem_comp']['id'] for n in e['nonpolymer_entities'] or []})}
    cache_path.write_text(json.dumps(dict(sorted(cache.items())), indent=1) + '\n')
  return cache


def template_audit(official, cache_path, fetch):
  per = []
  for t in sorted(official):
    p = protein(load_checked(official[t]['input_path'], official[t]['input_sha256']))
    for k, tpl in enumerate(p.get('templates') or []):
      info = template_info(tpl['mmcif'])
      pairs = list(zip(tpl.get('queryIndices', []), tpl.get('templateIndices', [])))
      matches = sum(1 for q, s in pairs if s < len(info['sequence']) and p['sequence'][q] == info['sequence'][s])
      per.append({'pdb_id': t, 'template': k, 'template_pdb_id': info['pdb_id'], 'template_chain': info['chain'],
                  'release_date_mmcif': info['release_date'], 'aligned': len(pairs),
                  'identity': round(matches / len(pairs), 4) if pairs else np.nan,
                  'query_coverage': round(len(pairs) / len(p['sequence']), 4), 'ligand_ccd': official[t]['ligand_ccd'],
                  'target_post_cutoff': official[t]['released_on_or_before_cutoff'] == 'no'})
  meta = rcsb_meta([r['template_pdb_id'] for r in per], cache_path, fetch)
  for r in per:
    m = meta.get(r['template_pdb_id'])
    r['release_date'] = r['release_date_mmcif'] or (m or {}).get('release') or 'unknown'
    r['contains_target_ligand'] = 'unknown' if m is None else ('yes' if r['ligand_ccd'] in m['ligands'] else 'no')
    r['own_entry'] = r['template_pdb_id'] == r['pdb_id']
    r['identity_ge_95'] = bool(r['identity'] >= 0.95) if r['identity'] == r['identity'] else False
  return pd.DataFrame(per, columns=['pdb_id', 'template', 'template_pdb_id', 'template_chain', 'release_date',
                                    'release_date_mmcif', 'aligned', 'identity', 'query_coverage', 'ligand_ccd',
                                    'contains_target_ligand', 'own_entry', 'identity_ge_95', 'target_post_cutoff'])

# ---------------------------------------------------------------- pipeline and determinism


def last_rows(run):
  rows = {}
  for r in csv.DictReader(open(NAS / run / 'results.csv')):
    rows[r['pdb_id']] = r
  return rows


def pipeline(official):
  rows = last_rows(RUN1)
  return pd.DataFrame([{'pdb_id': t, 'seconds': int(rows[t]['seconds']), 'vm_mem_baseline_mib': int(rows[t]['vm_mem_baseline_mib']),
                        'vm_mem_peak_mib': int(rows[t]['vm_mem_peak_mib']),
                        'vm_mem_increase_mib': int(rows[t]['vm_mem_peak_mib']) - int(rows[t]['vm_mem_baseline_mib'])}
                       for t in sorted(official)])


def local(run, vm_path):
  return NAS / run / vm_path[vm_path.index('/output/') + 1:]


def field_diff(a, b):
  pa, pb = protein(a), protein(b)
  out = {}
  for kind in ('unpairedMsa', 'pairedMsa'):
    ra, rb = a3m_rows(pa.get(kind)), a3m_rows(pb.get(kind))
    sa, sb = set(ra), set(rb)
    out[kind] = {'rows': (len(ra), len(rb)), 'only_first': len(sa - sb), 'only_second': len(sb - sa),
                 'same_rows_same_order': ra == rb}
  ta = [(template_info(x['mmcif'])['pdb_id'], template_info(x['mmcif'])['chain']) for x in pa.get('templates') or []]
  tb = [(template_info(x['mmcif'])['pdb_id'], template_info(x['mmcif'])['chain']) for x in pb.get('templates') or []]
  out['templates'] = {'hits': (len(ta), len(tb)), 'only_first': sorted(set(ta) - set(tb)), 'only_second': sorted(set(tb) - set(ta)),
                      'same_hits_same_order': ta == tb, 'same_content': [x for x in pa.get('templates') or []] ==
                      [x for x in pb.get('templates') or []]}
  return out


def determinism(official):
  runs = {r: last_rows(r) for r in (RUN1, RUN2, NAS6)}
  pilot = sorted(t for t, r in official.items() if r['pilot'] == 'yes')
  rows = []
  for t in pilot:
    sha = {r: runs[r][t]['data_json_sha256'] for r in runs}
    row = {'pdb_id': t, **{f'sha256_{r}': sha[r] for r in runs},
           'run2_identical': sha[RUN2] == sha[RUN1], 'naspaz6_identical': sha[NAS6] == sha[RUN1]}
    base = json.loads(local(RUN1, runs[RUN1][t]['data_json']).read_text())
    for r, col in ((RUN2, 'run2'), (NAS6, 'naspaz6')):
      if sha[r] != sha[RUN1]:
        row[f'{col}_difference'] = json.dumps(field_diff(base, json.loads(local(r, runs[r][t]['data_json']).read_text())))
      else:
        row[f'{col}_difference'] = ''
    rows.append(row)
  return pd.DataFrame(rows)

# ---------------------------------------------------------------- inference


def msa_samples(path, official):
  d = f34.load_samples(path, ['msa'])
  d = d[d.platform.isin(list(CODE))].copy()
  d['condition'] = d.platform.map(CODE)
  bad = d[d.apply(lambda r: r.input_sha256 != official.get(r.target, {}).get('input_sha256'), axis=1)]
  if len(bad):
    raise SystemExit(f'{len(bad)} msa rows ran an input that is not the official _data.json of their target: '
                     f'{sorted(set(bad.target))[:5]}')
  return d


def completeness(ok):
  rows = []
  for (c, rep), want in PLAN.items():
    g = ok[(ok.condition == c) & (ok.repetition == rep)]
    units = g.groupby(['target', 'seed']).size()
    have = int((units == 5).sum())
    rows.append({'condition': c, 'repetition': rep, 'planned_units': want, 'units_with_5_samples': have,
                 'status': 'complete' if have == want else 'incomplete'})
  return pd.DataFrame(rows)


def incomplete_conditions(comp):
  return set(comp[comp.status != 'complete'].condition)


def flag(df, col, names_of, bad):
  df = df.copy()
  df['incomplete_arm'] = df[col].map(lambda n: ', '.join(sorted(c for c in names_of(n) if c in bad)))
  return df


def conditions_in(name):
  return set(re.findall(r'(V5o|V6o|Go)', name))


def diff_boot(a_groups, b_groups, args):
  """Estimate and 95% interval of mean(a) - mean(b), targets resampled together (paired)."""
  if not a_groups:
    return np.nan, np.nan, np.nan
  est = np.mean(np.concatenate(a_groups)) - np.mean(np.concatenate(b_groups))
  rng = np.random.default_rng(args.seed)
  vals = [np.mean(np.concatenate([a_groups[i] for i in row])) - np.mean(np.concatenate([b_groups[i] for i in row]))
          for row in rng.integers(0, len(a_groups), size=(args.replicates, len(a_groups)))]
  return float(est), float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def reading(lo, hi):
  if lo != lo or hi != hi:
    return 'not computed'
  return 'lower' if hi < 0 else 'higher' if lo > 0 else 'similar'


def official_minus_colabfold(sp, main_pairs, targets, args, stats):
  rows = []
  for cf, off in COMPARE:
    a = sp[(sp.contrast == off) & sp.target.isin(targets)].set_index(['target', 'seed_a'])
    b = main_pairs[(main_pairs.contrast == cf) & main_pairs.target.isin(targets)].set_index(['target', 'seed_a'])
    common = sorted(set(a.index) & set(b.index))
    ts = sorted({t for t, _ in common})
    for st in stats:
      ag = [np.array([float(a.loc[(t, s), st]) for t2, s in common if t2 == t]) for t in ts]
      bg = [np.array([float(b.loc[(t, s), st]) for t2, s in common if t2 == t]) for t in ts]
      est, lo, hi = diff_boot(ag, bg, args)
      rows.append({'official': off, 'colabfold': cf, 'statistic': st, 'n': len(common), 'n_targets': len(ts),
                   'official_rate': float(np.mean(np.concatenate(ag))) if ag else np.nan,
                   'colabfold_rate': float(np.mean(np.concatenate(bg))) if bg else np.nan,
                   'estimate': est, 'ci_low': lo, 'ci_high': hi, 'reading': reading(lo, hi)})
  return pd.DataFrame(rows)


def success_rates(units, conditions, targets, args):
  rows = []
  for c, rep in conditions:
    per = {}
    for k, g in units.items():
      if k[0] == c and k[1] == rep and k[2] in targets:
        per.setdefault(k[2], []).append(float(contrasts.top(g)[1]))
    groups = [np.array(v) for _, v in sorted(per.items())]
    lo, hi = contrasts.boot_ci(groups, contrasts.mean, np.random.default_rng(args.seed), args.replicates)
    rows.append({'condition': c, 'repetition': rep, 'n': sum(len(g) for g in groups), 'n_targets': len(groups),
                 'success_rate': float(np.mean(np.concatenate(groups))) if groups else np.nan, 'ci_low': lo, 'ci_high': hi})
  return pd.DataFrame(rows)


def reading_rule(con, hwrc, omc, stats, bad):
  """One row per flip rate, the headline (top-1) first. Without RC(Go) (Go r2 not run) the rule is not applied to
  HW(V6o,Go) and HW(V6o,Go) minus RC(Go) is not available; RC(G) of the main study is never substituted."""
  rows = []
  rc_available = RULE_RC in set(con.contrast)
  for st in sorted(stats, key=lambda x: x != HEADLINE):
    r = hwrc[(hwrc.hardware == RULE_HW) & (hwrc.statistic == st)]
    h = con[(con.contrast == 'HW(V5o,V6o)') & (con.statistic == st)]
    lo_r = float(r.ci_low.iloc[0]) if len(r) and rc_available else np.nan
    lo_h = float(h.ci_low.iloc[0]) if len(h) else np.nan
    v5v6 = 'not computed (no pairs)' if lo_h != lo_h else ('interval above 0' if lo_h > 0 else 'interval not above 0')
    if not rc_available:
      branch = ('not applied: RC(Go) is not available (Go r2 did not run), so HW(V6o,Go) minus RC(Go) is not available '
                f'and the main study\'s RC(G) is not substituted; HW(V5o,V6o): {v5v6}')
    elif lo_r != lo_r or lo_h != lo_h:
      branch = 'not computed (a contrast has no pairs)'
    elif lo_r > 0 and lo_h > 0:
      branch = 'the main conclusion holds with official inputs'
    else:
      branch = 'the main conclusion does not hold with official inputs (an interval not above 0)'
    pairs = {f'{x.official} minus {x.colabfold}': x.reading for x in omc[omc.statistic == st].itertuples()}
    rows.append({'statistic': st, 'role': 'headline' if st == HEADLINE else 'secondary',
                 'HW(V6o,Go) minus RC(Go), ci_low': lo_r if rc_available else 'not available',
                 'HW(V5o,V6o), ci_low': lo_h, 'branch': branch,
                 'incomplete_arms': ', '.join(sorted(bad & {'V5o', 'V6o', 'Go'})), **pairs})
  return pd.DataFrame(rows)


def rc_not_available(stats):
  """The rows of hw_minus_rc.csv when RC(Go) is not available: stated, never substituted."""
  return pd.DataFrame([{'hardware': RULE_HW, 'recompilation': RULE_RC, 'reference': 'RC(Go), the row', 'statistic': st,
                        'n_targets': 0, 'estimate': np.nan, 'ci_low': np.nan, 'ci_high': np.nan,
                        'note': 'not available: Go r2 did not run; the main study\'s RC(G) is not substituted'}
                       for st in stats])


def inference(args, official, out):
  comp_all = []
  msa = msa_samples(args.samples, official)
  main = f34.load_samples(pathlib.Path(args.main) / 'samples.csv', ['main'])
  main = main[main.condition.isin(list(CODE))]
  ok_msa = f34.union(msa)
  ok_all = f34.union(main, msa)
  comp = completeness(ok_msa)
  bad = incomplete_conditions(comp)
  sp, sm, cons, counts = f34.build(ok_msa, SPECS)
  isp, ism, icons, icounts = f34.build(ok_all, IN_SPECS)
  sp_all = pd.concat([sp, isp], ignore_index=True)
  sm_all = pd.concat([sm, ism], ignore_index=True)
  cons_all = {**cons, **icons}
  main_pairs = pd.read_csv(pathlib.Path(args.main) / 'contrasts' / 'pairs_seed.csv')
  post = sorted(t for t, r in official.items() if r['released_on_or_before_cutoff'] == 'no')
  every = sorted(official)
  outs = {'contrasts': [], 'hw_minus_rc': [], 'official_minus_colabfold': [], 'success_rates': [], 'reading_rule': []}
  for label, targets, stats in (('all 60', every, contrasts.FLIP_STATS), ('post-cutoff 39', post, M1_STATS)):
    s1, s2 = sp_all[sp_all.target.isin(targets)], sm_all[sm_all.target.isin(targets)]
    con = contrasts.summarize(s1, s2, cons_all, args)
    if label != 'all 60':
      con = con[con.statistic.isin(stats)]
    hwrc = contrasts.hw_minus_rc(s1[s1.contrast.isin([RULE_HW, RULE_RC])],
                                 {k: cons[k] for k in (RULE_HW, RULE_RC) if k in cons}, args)   # an arm may be missing
    hwrc = hwrc[(hwrc.recompilation == RULE_RC) & hwrc.statistic.isin(stats)].assign(reference='RC(Go), the row', note='') \
        if len(hwrc) and RULE_RC in cons else rc_not_available(stats)
    omc = official_minus_colabfold(sp, main_pairs, targets, args, stats)
    units = contrasts.seed_units(ok_all)
    sr = success_rates(units, [('V5', 'r1'), ('V6', 'r1'), ('G', 'r1'), ('V5o', 'r1'), ('V6o', 'r1'), ('Go', 'r1'),
                               ('Go', 'r2')], set(targets), args)
    rule = reading_rule(con, hwrc, omc, stats, bad)
    for k, df in (('contrasts', con), ('hw_minus_rc', hwrc), ('official_minus_colabfold', omc), ('success_rates', sr),
                  ('reading_rule', rule)):
      outs[k].append(df.assign(targets=label))
  res = {k: pd.concat(v, ignore_index=True) for k, v in outs.items()}
  res['contrasts'] = flag(res['contrasts'], 'contrast', conditions_in, bad)
  res['hw_minus_rc'] = flag(res['hw_minus_rc'], 'hardware', lambda n: conditions_in(n) | {'Go'}, bad)
  res['official_minus_colabfold'] = flag(res['official_minus_colabfold'], 'official', conditions_in, bad)
  res['success_rates'] = flag(res['success_rates'], 'condition', lambda n: {n}, bad)
  res.update(completeness=comp, counts=pd.concat([counts, icounts], ignore_index=True), pairs_seed=sp_all,
             pairs_sample=sm_all)
  return res


def report(out, desc, res, args):
  L = ['# Follow-up 6: AlphaFold3\'s own data pipeline (label msa)', '',
       'notes/analysis_plan.md section 10, row of 2026-10-19; follow-up outside the pre-registered study. Code: '
       f'analysis/msa.py (bootstrap {args.replicates} replicates, seed {args.seed}).', '']
  d = desc['msa_depth']
  L += ['## MSA depth (official against ColabFold)', '', '| | ColabFold | official |', '|---|---|---|']
  for k in ('unpaired_rows', 'unpaired_neff80', 'paired_rows', 'paired_neff80', 'templates'):
    L.append(f'| median {k} | {d[f"colabfold_{k}"].median():g} | {d[f"official_{k}"].median():g} |')
  t = desc['templates']
  L += ['', f'Templates: {len(t)} in all; own entry {int(t.own_entry.sum())}; identity >= 0.95 {int(t.identity_ge_95.sum())}; '
        f'entry contains the target ligand: {int((t.contains_target_ligand == "yes").sum())} '
        f'(unknown {int((t.contains_target_ligand == "unknown").sum())}).', '']
  p = desc['pipeline']
  L += [f'Pipeline (naspaz8-af3 run1): median {p.seconds.median():.0f} s, max {p.seconds.max()} s per target; peak VM '
        f'memory median {p.vm_mem_peak_mib.median():.0f} MiB, max {p.vm_mem_peak_mib.max()} MiB.', '']
  dt = desc['determinism']
  L += [f'Determinism (10 pilot targets): run2 identical to run1 {int(dt.run2_identical.sum())}/{len(dt)}, naspaz6 identical '
        f'{int(dt.naspaz6_identical.sum())}/{len(dt)}.', '']
  if res is None:
    L += ['Inference: not run (descriptive only).']
    (out / 'report.md').write_text('\n'.join(L) + '\n')
    return
  L += ['## Completeness', '', '| condition | repetition | planned | with 5 samples | status |', '|---|---|---|---|---|']
  L += [f'| {r.condition} | {r.repetition} | {r.planned_units} | {r.units_with_5_samples} | {r.status} |'
        for r in res['completeness'].itertuples()]
  L += ['', '## Reading rule (headline: top-1 flip rate; the other flip rates secondary)', '']
  for r in res['reading_rule'].itertuples():
    L.append(f'- {r.targets}, {r.statistic} ({r.role}): {r.branch}'
             + (f' (incomplete arms: {r.incomplete_arms})' if r.incomplete_arms else ''))
  L += ['', '## Official minus ColabFold', '', '| targets | pair | statistic | estimate | 95% CI | reading |', '|---|---|---|---|---|---|']
  L += [f'| {r.targets} | {r.official} - {r.colabfold} | {r.statistic} | {f34.pct(r.estimate)} | {f34.pct(r.ci_low)} to '
        f'{f34.pct(r.ci_high)} | {r.reading} |' for r in res['official_minus_colabfold'].itertuples()]
  L += ['', '## Contrasts', ''] + f34.contrast_lines(res['contrasts'][res['contrasts'].targets == 'all 60'])
  (out / 'report.md').write_text('\n'.join(L) + '\n')


def main():
  ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
  ap.add_argument('--out', required=True)
  ap.add_argument('--samples', help='analysis/collect.py table of the msa sessions')
  ap.add_argument('--descriptive_only', action='store_true')
  ap.add_argument('--fetch_rcsb', action='store_true', help='fill rcsb_templates.json from RCSB (read-only)')
  ap.add_argument('--main', default=str(REPO / 'results' / 'analysis' / 'main_20261014' / 'primary_v2'))
  ap.add_argument('--replicates', type=int, default=10000)
  ap.add_argument('--seed', type=int, default=2026)
  args = ap.parse_args()
  out = pathlib.Path(args.out)
  out.mkdir(parents=True, exist_ok=True)
  official = {r['pdb_id']: r for r in csv.DictReader(open(OFFICIAL))}
  colabfold = {r['pdb_id']: r for r in csv.DictReader(open(COLABFOLD))}
  desc = {'msa_depth': msa_depth(official, colabfold), 'templates': template_audit(official, out / 'rcsb_templates.json',
                                                                                   args.fetch_rcsb),
          'pipeline': pipeline(official), 'determinism': determinism(official)}
  for k, df in desc.items():
    df.to_csv(out / f'{k}.csv', index=False)
  res = None
  if not args.descriptive_only:
    if not args.samples:
      raise SystemExit('--samples is required for the inference part (or --descriptive_only)')
    res = inference(args, official, out)
    for k, df in res.items():
      df.to_csv(out / f'{k}.csv', index=False)
  report(out, desc, res, args)
  print((out / 'report.md').read_text())
  return 0


if __name__ == '__main__':
  sys.exit(main())
