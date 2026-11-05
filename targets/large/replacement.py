"""Follow-up 10 (label large): the replacement for an excluded target, by the rule of the specification row
(notes/analysis_plan.md section 10, row of 2026-11-04, item b): "A target is replaced only for a permanent cause
(AlphaFold 3 rejects its input, or its token count differs from the screen's): by the next member of its cluster that
meets rules 1 to 4 (large) or the next cluster of its stratum's permutation (control), recorded in a row before any
run." It names the replacement and applies nothing: no file of targets/ or inputs/ is written.

Run inside the pinned AlphaFold3 environment, like the screen:

    cd third_party/alphafold3
    .venv/bin/python <repository>/targets/large/replacement.py PDB [PDB ...]
    .venv/bin/python <repository>/targets/large/replacement.py --exclusions inputs/large/excluded.csv

How: it rebuilds the screen's state from its cache (targets/large/pdb_screen.py with the amended rule: the RCSB
search, entries and clusters in data/pdb_large_screen/, and the outcome of every entry the screen checked, from
targets/large/pdb_screen_entries_amended.csv), replays the selection and the control draw (the same generator,
strata and order), and checks that the replay gives exactly the 34 large and 34 control targets of
targets/large/pdb_targets_amended.csv before naming anything.
  - large target: its cluster's other members that meet rules 1 to 4 with 1,025 to 1,536 tokens (every large-band
    entry was checked by the screen), in the order of rule 5 (resolution, release, PDB ID); the first one not
    excluded. None: the target is dropped without replacement (the cluster has no other member).
  - control target: the draw of its stratum continues where it stopped (the next clusters of the stratum's
    permutation, skipping clusters already taken); each cluster gives its first member of the stratum that meets
    rules 1 to 4 with at most 1,024 tokens. A member the screen never checked is checked now with the screen's own
    function (pdb_screen.check: its mmCIF from RCSB if not cached, AlphaFold3's token count); only the screen's caches
    grow.
  - several exclusions: in the order given; a replacement is never used twice.
The replacement still needs its MSA and input (inputs/build_msas.py, inputs/build_inputs.py) and a section 10 row
before any run.
"""
import argparse
import collections
import csv
import json
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pdb_screen as ps  # noqa: E402

REPO = ps.REPO
ENTRIES = HERE / 'pdb_screen_entries_amended.csv'
TARGETS = HERE / 'pdb_targets_amended.csv'


def read(path):
  return list(csv.DictReader(l for l in open(path) if not l.startswith('#')))


def state():
  """The screen's rows after its run (static rules from the cache, checks from the entries CSV), the static
  candidates, and the order key."""
  ps.LIGAND_RULE = 'amended'
  exclude = {r['pdb_id'] for f in ('targets/posebusters_subset.csv', 'targets/repl/repl_subset.csv')
             for r in csv.DictReader(open(REPO / f))}
  s_large, s_ctrl = ps.search('large', *ps.SEARCH_LARGE), ps.search('control', *ps.SEARCH_CONTROL)
  ids = s_large['ids'] + s_ctrl['ids']
  ent, cl30 = ps.entries(ids), ps.clusters(ids)
  checked = {r['pdb_id']: r for r in read(ENTRIES) if r['checked'] == 'yes'}
  rows, static_candidate = {}, set()
  for pid in sorted(set(ids)):
    if pid not in ent:
      continue
    row, why = ps.static(ent[pid], exclude, cl30)
    row.update(status='excluded' if why else 'candidate', reason=why or '', checked='no')
    if not why:
      static_candidate.add(pid)
    c = checked.get(pid)
    if c:
      row.update(checked='yes', status=c['status'], reason=c['reason'], ligand_ccd=c['ligand_ccd'],
                 ligand_asym=c['ligand_asym'], pocket_chains=c['pocket_chains'])
      if c['num_tokens']:
        row.update(num_tokens=int(c['num_tokens']), bucket=int(c['bucket']))
    rows[pid] = row
  return rows, static_candidate


def replay(rows, static_candidate):
  order = lambda r: (float(r['resolution']), r['release'], r['pdb_id'])  # noqa: E731
  by = collections.defaultdict(list)
  for r in rows.values():
    if r['status'] == 'eligible' and ps.LARGE[0] <= r['num_tokens'] <= ps.LARGE[1]:
      by[r['cluster30']].append(r)
  large = sorted((min(m, key=order) for m in by.values()), key=lambda r: r['pdb_id'])
  quota = collections.Counter((r['method'], ps.band_of(float(r['resolution']))) for r in large)
  large_clusters = {r['cluster30'] for r in large}
  rng = np.random.default_rng(ps.SEED)
  strata = [(m, b) for m in ('X-ray', 'EM') for _, _, b in ps.BANDS]
  pool = [rows[p] for p in sorted(static_candidate) if rows[p]['residues'] < ps.SEARCH_LARGE[0]
          and rows[p]['residues'] <= ps.SEARCH_CONTROL[1] and rows[p]['cluster30'] not in large_clusters]
  draw, control, used = {}, [], set()
  for st in strata:
    members = collections.defaultdict(list)
    for r in pool:
      if (r['method'], ps.band_of(float(r['resolution']))) == st:
        members[r['cluster30']].append(r)
    cids = sorted(members)
    perm = [cids[i] for i in rng.permutation(len(cids))]
    need, pos = quota.get(st, 0), 0
    while need and pos < len(perm):
      cid = perm[pos]
      pos += 1
      if cid in used:
        continue
      for r in sorted(members[cid], key=order):
        if r['checked'] == 'no':
          raise SystemExit(f'{r["pdb_id"]}: drawn but not in {ENTRIES.name}; the replay differs from the screen')
        if r['status'] == 'eligible' and r['num_tokens'] <= 1024:
          control.append(dict(r, stratum=f'{st[0]} {st[1]}'))
          used.add(cid)
          need -= 1
          break
    draw[st] = {'perm': perm, 'pos': pos, 'members': members}
  want = {g: sorted(r['pdb_id'] for r in read(TARGETS) if r['group'] == g) for g in ('large', 'control')}
  got = {'large': sorted(r['pdb_id'] for r in large), 'control': sorted(r['pdb_id'] for r in control)}
  if got != want:
    raise SystemExit(f'the replay does not give the targets of {TARGETS.name}; nothing is named')
  return large, control, draw, used, by, order


def replace(excluded):
  rows, static_candidate = state()
  large, control, draw, used, by, order = replay(rows, static_candidate)
  groups = {r['pdb_id']: 'large' for r in large}
  groups.update({r['pdb_id']: 'control' for r in control})
  stratum = {r['pdb_id']: r['stratum'] for r in control}
  tpath = ps.CACHE / 'token_counts.json'
  tokcache = json.loads(tpath.read_text()) if tpath.exists() else {}
  out, taken = [], set(excluded)
  for pid in excluded:
    g = groups.get(pid)
    if g is None:
      out.append({'excluded': pid, 'group': '', 'replacement': '', 'note': 'not a follow-up 10 target'})
      continue
    r = rows[pid]
    if g == 'large':
      cands = [m for m in sorted(by[r['cluster30']], key=order) if m['pdb_id'] not in taken]
      if cands:
        taken.add(cands[0]['pdb_id'])
        out.append({'excluded': pid, 'group': g, 'replacement': cands[0]['pdb_id'],
                    'note': f'next member of cluster {r["cluster30"]} ({cands[0]["num_tokens"]} tokens, '
                            f'{cands[0]["method"]} {cands[0]["resolution"]} A, ligand {cands[0]["ligand_ccd"]})'})
      else:
        out.append({'excluded': pid, 'group': g, 'replacement': '',
                    'note': f'no other member of cluster {r["cluster30"]} meets rules 1 to 4: dropped, not replaced'})
      continue
    st = tuple(stratum[pid].split(' ', 1))
    d = draw[st]
    found = None
    while found is None and d['pos'] < len(d['perm']):
      cid = d['perm'][d['pos']]
      d['pos'] += 1
      if cid in used:
        continue
      for m in sorted(d['members'][cid], key=order):
        if m['pdb_id'] in taken:
          continue
        if m['checked'] == 'no':
          why = ps.check(m, tokcache)
          m.update(checked='yes', status='excluded' if why else 'eligible', reason=why or '')
        if m['status'] == 'eligible' and m['num_tokens'] <= 1024:
          found = m
          used.add(cid)
          taken.add(m['pdb_id'])
          break
    out.append({'excluded': pid, 'group': g, 'replacement': found['pdb_id'] if found else '',
                'note': (f'next cluster of stratum {stratum[pid]}: {found["cluster30"]} ({found["num_tokens"]} tokens, '
                         f'{found["resolution"]} A, ligand {found["ligand_ccd"]})') if found else
                        f'stratum {stratum[pid]} has no further cluster with an eligible member'})
  return out


def main():
  ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
  ap.add_argument('pdb_ids', nargs='*')
  ap.add_argument('--exclusions', help='the CSV inputs/build_inputs.py writes (pdb_id, cause)')
  args = ap.parse_args()
  excluded = list(args.pdb_ids)
  if args.exclusions:
    excluded += [r['pdb_id'] for r in csv.DictReader(open(REPO / args.exclusions))]
  if not excluded:
    ap.error('give PDB IDs or --exclusions')
  print('excluded,group,replacement,note')
  for r in replace(excluded):
    print(f'{r["excluded"]},{r["group"]},{r["replacement"]},"{r["note"]}"')


if __name__ == '__main__':
  main()
