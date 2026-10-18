"""Follow-up 5, host-side featurisation (notes/analysis_plan.md section 10, row of 2026-10-18; label feat; reported as
a follow-up whatever it shows): AlphaFold3's features of every (target, seed) compared across the four host families.

    analysis/.venv/bin/python analysis/feat.py --sessions scripts/analysis_runs/feat/sessions.txt \\
        --out results/analysis/feat_<date>

Sessions: one per host family, named <stamp>_cpu_feat_<family> (milan, genoa, cascadelake, icelake), each with
featurise/host.json and featurise/<target>/features.jsonl from harness/featurise_only.py.
Checks, before any comparison (the analysis stops if one fails): exactly one session per family; every session
has every (target, seed) of the plan (90 targets, seeds 1 to 5) once, without error; the per-seed input is the
same on every host (run_input_sha256); the CCD pickles build_data wrote, the packages that featurise (numpy,
rdkit, scipy) and Python are the same on every host; the AlphaFold3 version is the same apart from its local build
label (+g<commit>.d<date>).
Comparison: for every (target, seed, feature), the SHA-256 on each host; identical when all four agree. Features
are every entry of the featurised batch: the arrays the model receives (model_input true) and the host-side ones
(the empty output structure and the atom layouts). When a feature differs and the sessions hold their arrays
(featurise/<target>/arrays/seed-<S>.npz), the maximum absolute difference and the number of differing elements
against the first host in FAMILIES order are added.
Decision rule (the row, fixed before any run; "every array" read as every model input in the implementation row of
2026-10-18): if every model-input array (the 59 the model receives) is identical on every host for every (target,
seed), the host is excluded as a source of the HW differences; if any differs, the differing (target, seed,
array) are reported and follow-up 5b is run. The 5 host-side entries are compared and reported apart, and a
difference in them alone does not trigger 5b.
Outputs in --out: checks.csv, hosts.csv, counts.csv (per feature), differences.csv, decision.json, report.md.
"""
import argparse
import collections
import json
import pathlib
import re
import sys

import numpy as np
import pandas as pd

REPO = pathlib.Path(__file__).resolve().parent.parent
FAMILIES = ['milan', 'genoa', 'cascadelake', 'icelake']
ARMS = {'milan': 'V5', 'genoa': 'V6, V6h', 'cascadelake': 'G, GT, Gd, Gh, A, Ah, AT', 'icelake': 'C'}
SESSION = re.compile(r'^\d{8}T\d{6}Z_cpu_feat_([a-z]+)$')
SAME_PACKAGES = ('numpy', 'rdkit', 'scipy')
BRANCH_EXCLUDED = ('host excluded: every model-input array is identical on every host for every (target, seed), so '
                   'the host is excluded as a source of the HW differences')
BRANCH_5B = ('follow-up 5b: some model-input arrays differ across hosts; the differing (target, seed, array) are '
             'reported and follow-up 5b is run')


def read_sessions(path):
  return [l.split('#')[0].strip() for l in pathlib.Path(path).read_text().splitlines() if l.split('#')[0].strip()]


def load(root, sessions):
  """{family: (session, host dict, {(target, seed): record})}; raises on a session name that is not a feat one."""
  out = {}
  for s in sessions:
    m = SESSION.match(s)
    if not m or m.group(1) not in FAMILIES:
      raise SystemExit(f'{s}: not a follow-up 5 session (<stamp>_cpu_feat_<{"|".join(FAMILIES)}>)')
    d = root / s / 'featurise'
    host = json.loads((d / 'host.json').read_text()) if (d / 'host.json').is_file() else {}
    recs = collections.defaultdict(list)
    for f in sorted(d.glob('*/features.jsonl')):
      for line in f.read_text().splitlines():
        r = json.loads(line)
        recs[(r['target'], r['seed'])].append(r)
    out.setdefault(m.group(1), []).append((s, host, dict(recs)))
  return out


def base_version(v):
  return str(v).split('+')[0] if v else v


def checks(data, expected_units):
  rows = []

  def add(check, value, ok, detail=''):
    rows.append({'check': check, 'value': value, 'detail': 'pass' if ok else 'FAIL' + (f': {detail}' if detail else '')})
  have = {f: len(v) for f, v in data.items()}
  add('one session per host family', ', '.join(f'{f} {have.get(f, 0)}' for f in FAMILIES),
      all(have.get(f) == 1 for f in FAMILIES) and set(have) <= set(FAMILIES))
  if not all(have.get(f) == 1 for f in FAMILIES):
    return rows
  one = {f: data[f][0] for f in FAMILIES}
  for f, (s, host, recs) in one.items():
    dup = sorted(k for k, v in recs.items() if len(v) != 1)
    err = sorted(k for k, v in recs.items() if any(r.get('error') for r in v))
    missing = sorted(set(expected_units) - set(recs))
    extra = sorted(set(recs) - set(expected_units))
    ok = not (dup or err or missing or extra) and bool(host)
    add(f'{f}: every (target, seed) once, without error', f'{len(recs)}/{len(expected_units)}', ok,
        '; '.join(x for x in (f'no host.json' if not host else '', f'missing {missing[:3]}' if missing else '',
                               f'extra {extra[:3]}' if extra else '', f'twice {dup[:3]}' if dup else '',
                               f'errors {err[:3]}' if err else '') if x))
  units = sorted(set.intersection(*(set(r) for _, _, r in one.values())))
  diff_in = [k for k in units if len({one[f][2][k][0].get('run_input_sha256') for f in FAMILIES}) != 1]
  add('the same one-seed input on every host (run_input_sha256)', f'{len(units) - len(diff_in)}/{len(units)}',
      not diff_in, f'{diff_in[:3]}')
  for label, get in [('CCD pickles build_data wrote (SHA-256)', lambda h: json.dumps(h.get('ccd_pickles_sha256'),
                                                                                       sort_keys=True)),
                     ('Python', lambda h: h.get('python')),
                     *[(p, (lambda p: lambda h: (h.get('packages') or {}).get(p))(p)) for p in SAME_PACKAGES],
                     ('AlphaFold3 (without the local build label)',
                      lambda h: base_version((h.get('packages') or {}).get('alphafold3')))]:
    vals = {f: get(one[f][1]) for f in FAMILIES}
    add(f'same on every host: {label}', '; '.join(f'{f} {str(v)[:40]}' for f, v in vals.items()),
        len(set(vals.values())) == 1 and None not in vals.values())
  return rows


def host_table(data):
  rows = []
  for f in FAMILIES:
    for s, h, _ in data.get(f, []):
      ci = h.get('cpuinfo') or {}
      rows.append({'family': f, 'arms': ARMS[f], 'session': s, 'machine_type': h.get('gce_machine_type'),
                   'zone': h.get('gce_zone'), 'cpu_platform': h.get('gce_min_cpu_platform'), 'cpu': h.get('cpu'),
                   'cpu_family_model_stepping': '/'.join(str(ci.get(k, '')) for k in ('cpu family', 'model', 'stepping')),
                   'avx512f': ci.get('avx512f'), 'vcpus': h.get('cpu_count'),
                   'os': (h.get('os_release') or {}).get('PRETTY_NAME'), 'kernel': h.get('kernel'), 'glibc': h.get('glibc'),
                   'python': h.get('python'), **{p: (h.get('packages') or {}).get(p) for p in ('numpy', 'rdkit', 'alphafold3')},
                   'numpy_blas': (h.get('numpy_blas') or {}).get('name') if isinstance(h.get('numpy_blas'), dict)
                   else h.get('numpy_blas')})
  return pd.DataFrame(rows)


def arrays_of(root, session, target, seed):
  p = root / session / 'featurise' / target / 'arrays' / f'seed-{seed}.npz'
  return np.load(p) if p.is_file() else None


def compare(root, data):
  one = {f: data[f][0] for f in FAMILIES}
  units = sorted(set.intersection(*(set(r) for _, _, r in one.values())))
  counts = collections.defaultdict(lambda: {'identical': 0, 'different': 0, 'model_input': None})
  diffs = []
  for t, seed in units:
    feats = {f: {a['name']: a for a in one[f][2][(t, seed)][0]['arrays']} for f in FAMILIES}
    names = sorted(set().union(*(set(v) for v in feats.values())))
    loaded = {}
    for name in names:
      recs = {f: feats[f].get(name) for f in FAMILIES}
      hashes = {f: (r['sha256'] if r else None) for f, r in recs.items()}
      c = counts[name]
      c['model_input'] = next((r['model_input'] for r in recs.values() if r), None)
      if len(set(hashes.values())) == 1 and None not in hashes.values():
        c['identical'] += 1
        continue
      c['different'] += 1
      groups = collections.defaultdict(list)
      for f, h in hashes.items():
        groups[h].append(f)
      row = {'target': t, 'seed': seed, 'feature': name, 'model_input': c['model_input'],
             'host_groups': ' | '.join('+'.join(v) for v in sorted(groups.values(), key=lambda v: FAMILIES.index(v[0]))),
             'shapes': '; '.join(f'{f} {recs[f]["shape"] if recs[f] else "missing"}' for f in FAMILIES),
             'max_abs_difference': None, 'differing_elements': None, 'arrays_compared': False}
      if c['model_input']:
        for f in FAMILIES:
          if f not in loaded:
            loaded[f] = arrays_of(root, one[f][0], t, seed)
        if all(loaded[f] is not None and name in loaded[f] for f in FAMILIES):
          ref = np.asarray(loaded[FAMILIES[0]][name])
          worst, n = 0.0, 0
          for f in FAMILIES[1:]:
            a = np.asarray(loaded[f][name])
            if a.shape != ref.shape:
              worst, n = float('inf'), max(n, a.size)
              continue
            d = np.abs(a.astype(np.float64) - ref.astype(np.float64))
            worst, n = max(worst, float(np.nanmax(d)) if d.size else 0.0), max(n, int((a != ref).sum()))
          row.update(max_abs_difference=worst, differing_elements=n, arrays_compared=True)
      diffs.append(row)
  c = pd.DataFrame([{'feature': k, **v} for k, v in sorted(counts.items())])
  d = pd.DataFrame(diffs, columns=['target', 'seed', 'feature', 'model_input', 'host_groups', 'shapes',
                                   'max_abs_difference', 'differing_elements', 'arrays_compared'])
  return len(units), c, d


def decide(counts, diffs, n_units):
  """The row's rule on the model-input arrays only; the host-side entries are counted apart and never decide."""
  def part(model_input):
    c = counts[counts.model_input.fillna(False).astype(bool) == model_input] if len(counts) else counts
    d = diffs[diffs.model_input.fillna(False).astype(bool) == model_input] if len(diffs) else diffs
    return {'compared': int(c.identical.sum() + c.different.sum()) if len(c) else 0,
            'identical': int(c.identical.sum()) if len(c) else 0, 'different': int(c.different.sum()) if len(c) else 0,
            'units_with_a_difference': int(d[['target', 'seed']].drop_duplicates().shape[0]) if len(d) else 0}
  model, host_side = part(True), part(False)
  excluded = model['compared'] > 0 and model['different'] == 0
  return {'branch': 'host excluded' if excluded else '5b', 'statement': BRANCH_EXCLUDED if excluded else BRANCH_5B,
          'units': n_units, 'model_inputs': model, 'host_side': host_side}


def report(out, chk, hosts, counts, diffs, dec):
  L = ['# Follow-up 5: host-side featurisation', '', 'notes/analysis_plan.md section 10, row of 2026-10-18; follow-up '
       'outside the pre-registered study. Code: analysis/feat.py.', '', '## Checks', '', '| check | value | detail |',
       '|---|---|---|'] + [f'| {r["check"]} | {r["value"]} | {r["detail"]} |' for r in chk]
  if dec is None:
    L += ['', 'A check failed: nothing was compared.']
    (out / 'report.md').write_text('\n'.join(L) + '\n')
    return
  L += ['', '## Hosts', '', '| family | arms | machine type | CPU | AVX-512 | vCPUs | OS | kernel | glibc |',
        '|---|---|---|---|---|---|---|---|---|']
  L += [f'| {r.family} | {r.arms} | {r.machine_type} | {r.cpu} ({r.cpu_family_model_stepping}) | {r.avx512f} | {r.vcpus} | '
        f'{r.os} | {r.kernel} | {r.glibc} |' for r in hosts.itertuples()]
  m, h = dec['model_inputs'], dec['host_side']
  L += ['', f'## Decision: {dec["branch"]}', '', dec['statement'] + '.', '',
        f'{dec["units"]} (target, seed). Model inputs (the rule): {m["compared"]} (target, seed, array) compared, '
        f'{m["identical"]} identical on all four hosts, {m["different"]} different, in {m["units_with_a_difference"]} '
        f'(target, seed). Host-side entries (reported apart, never deciding): {h["compared"]} compared, {h["identical"]} '
        f'identical, {h["different"]} different, in {h["units_with_a_difference"]} (target, seed).', '',
        '## Per feature', '', '| feature | model input | identical | different |', '|---|---|---|---|']
  L += [f'| {r.feature} | {r.model_input} | {r.identical} | {r.different} |' for r in counts.itertuples()]
  for model_input, title in ((True, 'Differing model-input arrays (target, seed, array)'),
                             (False, 'Differing host-side entries (reported apart; they do not decide)')):
    d = diffs[diffs.model_input.fillna(False).astype(bool) == model_input] if len(diffs) else diffs
    if len(d):
      L += ['', f'## {title}', '', '| target | seed | feature | hosts that agree | max abs difference | '
            'differing elements |', '|---|---|---|---|---|---|']
      L += [f'| {r.target} | {r.seed} | {r.feature} | {r.host_groups} | '
            f'{"-" if not r.arrays_compared else f"{r.max_abs_difference:.3g}"} | '
            f'{"-" if not r.arrays_compared else r.differing_elements} |' for r in d.itertuples()]
  (out / 'report.md').write_text('\n'.join(L) + '\n')


def expected_units(plan_seeds, manifest):
  import csv
  targets = [r['pdb_id'] for r in csv.DictReader(open(REPO / manifest))]
  return [(t, s) for t in targets for s in plan_seeds]


def main():
  ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
  ap.add_argument('--sessions', required=True, help='file with the four feat session folder names')
  ap.add_argument('--root', default=str(REPO / 'results' / 'af3'))
  ap.add_argument('--manifest', default='inputs/feat/manifest.csv')
  ap.add_argument('--seeds', default='1,2,3,4,5')
  ap.add_argument('--out', required=True)
  args = ap.parse_args()
  out = pathlib.Path(args.out)
  out.mkdir(parents=True, exist_ok=True)
  root = pathlib.Path(args.root)
  data = load(root, read_sessions(args.sessions))
  chk = checks(data, expected_units([int(s) for s in args.seeds.split(',')], args.manifest))
  pd.DataFrame(chk).to_csv(out / 'checks.csv', index=False)
  hosts = host_table(data)
  hosts.to_csv(out / 'hosts.csv', index=False)
  if any(r['detail'] != 'pass' for r in chk):
    report(out, chk, hosts, None, None, None)
    raise SystemExit('a check failed (checks.csv); nothing is compared')
  n_units, counts, diffs = compare(root, data)
  dec = decide(counts, diffs, n_units)
  counts.to_csv(out / 'counts.csv', index=False)
  diffs.to_csv(out / 'differences.csv', index=False)
  (out / 'decision.json').write_text(json.dumps(dec, indent=2) + '\n')
  report(out, chk, hosts, counts, diffs, dec)
  print((out / 'report.md').read_text())
  return 0


if __name__ == '__main__':
  sys.exit(main())
