"""Follow-up 21, recompilation on a new VM of the same GPU type (label xvm; notes/analysis_plan.md section 10, row
"Follow-up experiment 21").

    analysis/.venv/bin/python analysis/xvm.py --samples <collect.py samples.csv of the xvm sessions> --out DIR

Conditions. Follow-up 21 (label xvm, repetition r3), from each process's configuration (collect.py unchanged):
l4_xla -> G (part A), a100_xla -> A (part B, optional: without A sessions its rows say "not run"). r1 and r2: G of the
main study (--main, primary_v2/samples.csv), A of follow-up 3 (--a100, samples.csv), one VM per target for both (same
VM).
Checks (item c), before anything is compared; the analysis stops if 1 or 2 fails:
  1 every process of the xvm sessions exits 0 or its failure is recorded;
  2 one machine type per condition, the r1's (g2-standard-8, a2-ultragpu-1g);
  3 completeness per condition and seed: the 60 targets, seeds 1 to 5, r3 (a missing unit leaves the contrast,
    section 8);
  4 settings and stack of every r3 session against the r1 and r2 sessions of its arm: the configuration entry and
    environment (session.json config and xla_env), the NVIDIA driver, the kernel, the Python version, the Python
    packages (setup.json, AlphaFold3's build-date suffix removed), pip_freeze.txt, the image family (the wrapper log);
    the device of every process (NVIDIA L4, NVIDIA A100-SXM4-80GB). Reported, not a stop;
  5 every r3 session's VM name differs from the VM names of the r1 and r2 sessions of the same target (VM_NAME of
    launch_record.txt, results/af3/<session>/ or results/af3/.launcher/<session>/). Reported, not a stop;
  6 (reproduction, reported) RC_same(P) recomputed here equals the registered RC(P) seed pairs (main study, follow-up
    3) unit by unit.
Statistics (item d): section 5's target-clustered bootstrap (--replicates 10000, --seed 2026), paired on (target,
seed), per-seed top-1, seeds 1 to 5, all terms on the same resampled targets. For P in {G, A}: RC_same(P) = r1 against
r2 (same VM), RC_new(P) = r1 against r3 and RC_new2(P) = r2 against r3 (new VM): the three flip statistics (top-1
primary; success flip and flips that matter secondary), the M4 and M5 medians, the bit-identical fraction of samples
(contrasts.summarize) and the bit-identical (target, seed) (the light-fetch files, mmCIF timestamp masked,
rcx.light_identical); X(P) = RC_new(P) minus RC_same(P) (fu13.diff_two: the same resampled targets for both terms).
Main output (descriptive): per target, AlphaFold 3's main output (winners.main_output over the seeds the sides share)
under r1, r2 and r3; the number of targets whose main output differs, r1 against r2 (same VM), r1 against r3 and r2
against r3 (new VM).
Reading rules (item e): (1) for each P, "a new VM changes the recompilation rate" if the 95% interval of X(P) excludes
0, with its sign (higher or lower), "not distinguishable" otherwise (not a test of equivalence); top-1 primary, the
same rule for the success flip and flips that matter as secondary. (2) The registered recompilation references stand
whatever X shows; if rule 1 reads "changes" for G (top-1), the report states that RC_new(G) is reported beside RC(G) in
Table II. (3) No direction is predicted.
Outputs in --out: checks.csv, checks_detail.csv, stack.csv, vm_names.csv, completeness.csv, contrasts.csv,
identity.csv, x.csv, main_output.csv, pairs_seed.csv, pairs_sample.csv, counts.csv, report.md.

The helpers stack_of, image_of, vm_name_of, check_stack, light_identity and main_output_changes are also used by
analysis/aut.py and analysis/rcf.py (follow-ups 20 and 19).
"""
import argparse
import json
import pathlib
import re
import sys

import numpy as np
import pandas as pd

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / 'analysis'))
import contrasts  # noqa: E402
import followup34 as f34  # noqa: E402
import fu13  # noqa: E402
import rcx  # noqa: E402
import winners  # noqa: E402

AF3 = REPO / 'results' / 'af3'
SEEDS = [1, 2, 3, 4, 5]
CONFIG = {'l4_xla': 'G', 'a100_xla': 'A'}
DEVICE = {'G': 'NVIDIA L4', 'A': 'NVIDIA A100-SXM4-80GB'}
MACHINE = {'G': 'g2-standard-8', 'A': 'a2-ultragpu-1g'}
PART = {'G': 'A', 'A': 'B'}
REGISTERED_RC = {'G': ('main', 'RC(G)'), 'A': ('a100', 'RC(A)')}
SESSION_RE = re.compile(r'(\d{8}T\d{6}Z_[^/]+)')


def rc_specs(p):
  return [(f'RC_same({p})', 'RC', (p, 'r1'), (p, 'r2'), False), (f'RC_new({p})', 'RC', (p, 'r1'), (p, 'r3'), False),
          (f'RC_new2({p})', 'RC', (p, 'r2'), (p, 'r3'), False)]


# ---------------------------------------------------------------- loading

def load_xvm(path):
  d = pd.read_csv(path, low_memory=False)
  d = d[(d.label == 'xvm') & (d.num_recycles == 10) & (d.num_diffusion_samples == 5)].copy()
  d['condition'] = d.config.map(CONFIG)
  if d.condition.isna().any():
    raise SystemExit(f'sessions of label xvm with configurations not of follow-up 21: {sorted(set(d[d.condition.isna()].config))}')
  if not (d.repetition == 'r3').all():
    raise SystemExit(f'xvm processes with repetitions other than r3: {sorted(set(d.repetition) - {"r3"})}')
  return d


def load_r12(args):
  """G r1 and r2 (main study) and A r1 and r2 (follow-up 3), as their analyses collected them."""
  m = f34.load_samples(pathlib.Path(args.main) / 'samples.csv', ['main'])
  m = m[(m.condition == 'G') & m.repetition.isin(['r1', 'r2'])]
  a = f34.load_samples(pathlib.Path(args.a100) / 'samples.csv', ['a100'])
  a = a[(a.condition == 'A') & a.repetition.isin(['r1', 'r2'])]
  return pd.concat([m, a], ignore_index=True)


def session_name(s):
  m = SESSION_RE.search(str(s))
  return m.group(1) if m else None


# ---------------------------------------------------------------- shared helpers (also aut.py, rcf.py)

def image_of(session, af3=AF3):
  """The image project/family the wrapper printed before the session's VM was created (results/af3/.many logs), or
  None. A log may hold several attempts: the image line nearest above the session's line."""
  for log in sorted((pathlib.Path(af3) / '.many').glob('*/*.log')):
    t = log.read_text(errors='replace')
    i = t.find(f'session: results/af3/{session}/')
    if i < 0:
      continue
    found = re.findall(r'^\s+image (\S+)', t[:i], flags=re.M)
    if found:
      return found[-1]
  return None


def vm_name_of(session, af3=AF3):
  """VM_NAME of the session's launch record (the session folder or .launcher/), or None."""
  for d in (pathlib.Path(af3) / session, pathlib.Path(af3) / '.launcher' / session):
    rec = d / 'launch_record.txt'
    if rec.is_file():
      for line in rec.read_text(errors='replace').splitlines():
        if line.startswith('VM_NAME='):
          return line.split('=', 1)[1].strip()
  return None


def stack_of(session, af3=AF3):
  """rcx.stack_of (driver, kernel, Python, packages, pip freeze) plus the configuration entry and environment of the
  session's runs (session.json config and xla_env) and the image family."""
  d = pathlib.Path(af3) / session
  out = rcx.stack_of(d)
  out['config'], out['xla_env'] = None, None
  sj = next(iter(sorted(d.glob('*_fresh/session.json'))), None)
  if sj is not None:
    j = json.loads(sj.read_text())
    out['config'] = json.dumps(j.get('config'), sort_keys=True) if j.get('config') is not None else None
    out['xla_env'] = json.dumps(j.get('xla_env'), sort_keys=True) if j.get('xla_env') is not None else None
  out['image'] = image_of(session, af3)
  return out


def check_stack(new_sessions, ref_sessions, check, af3=AF3):
  """new_sessions, ref_sessions: {condition: [session names]}. Every new session's stack, field by field, against the
  set of its condition's reference stacks. Returns (summary rows, table)."""
  rows, table = [], []
  for c in sorted(new_sessions):
    refs = [stack_of(s, af3) for s in sorted(set(ref_sessions.get(c, [])))]
    diffs = {}
    for s in sorted(set(new_sessions[c])):
      st = stack_of(s, af3)
      for k, v in st.items():
        ref = sorted({str(x[k]) for x in refs})
        same = str(v) in ref
        table.append({'condition': c, 'session': s, 'field': k, 'new': v, 'reference': ' | '.join(ref), 'same': same})
        if not same:
          diffs.setdefault(k, set()).add(f'{str(v)[:80]} (reference {" | ".join(r[:80] for r in ref)})')
    rows.append({'check': check, 'condition': c,
                 'value': f'{len(set(new_sessions[c]))} sessions against {len(refs)} reference sessions',
                 'detail': 'pass' if not diffs else 'reported, differs in ' + '; '.join(
                     f'{k}: {", ".join(sorted(v))}' for k, v in sorted(diffs.items()))[:500]})
  return rows, pd.DataFrame(table)


def light_identity(ok, a, b, check, seeds=SEEDS):
  """rcx.light_identical of every (target, seed) of side b against side a, a and b (condition, repetition).
  Returns (summary row with the rcx reading, detail rows)."""
  det = []
  ga = ok[(ok.condition == a[0]) & (ok.repetition == a[1]) & ok.seed.isin(seeds)]
  gb = ok[(ok.condition == b[0]) & (ok.repetition == b[1]) & ok.seed.isin(seeds)]
  for (t, sd), g in gb.groupby(['target', 'seed']):
    r = ga[(ga.target == t) & (ga.seed == sd)]
    res = rcx.light_identical(contrasts.output_dir(r), contrasts.output_dir(g), sd) if len(r) else \
        {'status': 'missing', 'differ': [], 'missing': ['no reference run']}
    det.append({'check': check, 'target': t, 'seed': int(sd), 'status': res['status'],
                'files': '; '.join(res['differ'] + res['missing'])[:300]})
  if gb.empty:
    return {'check': check, 'condition': f'{b[0]} {b[1]} against {a[0]} {a[1]}', 'value': 'not run',
            'detail': 'not run', 'identical': 0, 'units': 0}, []
  n_id = sum(x['status'] == 'identical' for x in det)
  return {'check': check, 'condition': f'{b[0]} {b[1]} against {a[0]} {a[1]}',
          'value': (f'{n_id}/{len(det)} identical, {sum(x["status"] == "differ" for x in det)} differ, '
                    f'{sum(x["status"] == "missing" for x in det)} lack files'),
          'detail': rcx.identity_reading(det), 'identical': n_id, 'units': len(det)}, det


def main_outputs(ok, c, rep):
  """{target: (seed, sample, success, seeds)} of AlphaFold 3's main output of (c, rep), over all its seeds."""
  g = ok[(ok.condition == c) & (ok.repetition == rep)]
  return {t: gg for t, gg in g.groupby('target')}


def main_output_changes(ok, c, pairs):
  """Per pair of repetitions of condition c, the targets whose main output (winners.main_output over the seeds both
  sides ran) is a different (seed, sample), and those whose main output's success differs. Descriptive."""
  rows = []
  for ra, rb, what in pairs:
    a, b = main_outputs(ok, c, ra), main_outputs(ok, c, rb)
    targets = sorted(set(a) & set(b))
    changed, succ = [], []
    for t in targets:
      seeds = sorted(set(a[t].seed) & set(b[t].seed))
      xa = winners.main_output(a[t][a[t].seed.isin(seeds)])
      xb = winners.main_output(b[t][b[t].seed.isin(seeds)])
      if (int(xa.seed), int(xa['sample'])) != (int(xb.seed), int(xb['sample'])):
        changed.append(t)
      if bool(xa.m1_success == True) != bool(xb.m1_success == True):  # noqa: E712
        succ.append(t)
    rows.append({'condition': c, 'sides': f'{c} {ra} against {c} {rb}', 'vm': what, 'targets': len(targets),
                 'main_output_differs': len(changed), 'success_differs': len(succ),
                 'targets_differ': ' '.join(changed)})
  return pd.DataFrame(rows)


def diff_rows(sp, name_a, name_b, label, args, other=None):
  """fu13.diff_two of every flip statistic: name_a minus name_b (both in sp unless other holds name_b)."""
  b_frame = (other if other is not None else sp)
  return [{'comparison': label, 'statistic': st,
           **fu13.diff_two(sp[sp.contrast == name_a], b_frame[b_frame.contrast == name_b], st, args.seed, args.replicates)}
          for st in contrasts.FLIP_STATS]


# ---------------------------------------------------------------- checks

def check_machine_ref(r12, conds):
  rows = []
  for c in conds:
    g = r12[r12.condition == c]
    types = sorted(g.machine_type.dropna().astype(str).unique())
    rows.append({'check': '2b the machine type of r1 and r2', 'condition': c, 'value': ', '.join(types),
                 'detail': 'pass' if types == [MACHINE[c]] else f'FAIL: expected {MACHINE[c]}'})
  return rows


def check_r3_machine(d):
  rows = []
  for c, g in d.groupby('condition'):
    types = sorted(g.machine_type.dropna().astype(str).unique())
    rows.append({'check': '2c the machine type of r3 is the r1\'s', 'condition': c, 'value': ', '.join(types),
                 'detail': 'pass' if types == [MACHINE[c]] else f'FAIL: expected {MACHINE[c]}'})
  return rows


def completeness(ok, all_targets):
  rows = []
  for c in ('G', 'A'):
    for s in SEEDS:
      units = ok[(ok.condition == c) & (ok.repetition == 'r3') & (ok.seed == s)].groupby('target').size()
      have = int((units == 5).sum())
      rows.append({'condition': c, 'part': PART[c], 'repetition': 'r3', 'seed': s, 'planned_targets': len(all_targets),
                   'targets_with_5_samples': have,
                   'status': 'complete' if have == len(all_targets) else ('not run' if have == 0 else 'incomplete')})
  return pd.DataFrame(rows)


def check_device(d):
  rows = []
  for c, g in d.drop_duplicates(['session', 'target', 'seed']).groupby('condition'):
    bad = g[g.device_kind != DEVICE[c]]
    rows.append({'check': '4a the expected device in every process', 'condition': c,
                 'value': f'{len(g) - len(bad)}/{len(g)} processes on {DEVICE[c]}',
                 'detail': 'pass' if bad.empty else 'reported: ' + '; '.join(
                     f'{r.target} seed {r.seed} ({r.device_kind})' for r in bad.itertuples())[:250]})
  return rows


def check_vms(d, r12, af3=AF3):
  """Check 5: per (condition, target), the r3 VM names against the r1 and r2 VM names."""
  rows, table = [], []
  for c, g in d.groupby('condition'):
    bad = []
    for t, gt in g.groupby('target'):
      new = sorted({session_name(s) for s in gt.session})
      ref = sorted({session_name(s) for s in r12[(r12.condition == c) & (r12.target == t)].session})
      vm_new = {s: vm_name_of(s, af3) for s in new}
      vm_ref = {s: vm_name_of(s, af3) for s in ref}
      clash = sorted(v for v in vm_new.values() if v is not None and v in set(vm_ref.values()))
      unknown = [s for s, v in list(vm_new.items()) + list(vm_ref.items()) if v is None]
      ok = not clash and not unknown and bool(ref)
      table.append({'condition': c, 'target': t, 'r3_sessions': ' '.join(new),
                    'r3_vms': ' '.join(str(v) for v in vm_new.values()), 'r12_sessions': ' '.join(ref),
                    'r12_vms': ' '.join(str(v) for v in vm_ref.values()), 'differs': ok})
      if not ok:
        bad.append(f'{t} ({"same VM " + ", ".join(clash) if clash else "VM name unknown" if unknown else "no r1/r2"})')
    rows.append({'check': '5 every r3 VM differs from the r1 and r2 VMs of its target', 'condition': c,
                 'value': f'{g.target.nunique() - len(bad)}/{g.target.nunique()} targets',
                 'detail': 'pass' if not bad else 'reported: ' + '; '.join(bad)[:300]})
  return rows, pd.DataFrame(table)


def check_registered(sp, ref_sp, conds):
  """Check 6: RC_same(P) recomputed equals the registered RC(P) seed pairs."""
  rows = []
  for c in conds:
    src, name = REGISTERED_RC[c]
    reg = ref_sp[src]
    reg = reg[reg.contrast == name].set_index(['target', 'seed_a'])[contrasts.FLIP_STATS].astype(bool).sort_index()
    new = sp[sp.contrast == f'RC_same({c})'].set_index(['target', 'seed_a'])[contrasts.FLIP_STATS].astype(bool).sort_index()
    same = len(reg) == len(new) and reg.index.equals(new.index) and (reg.values == new.values).all()
    rows.append({'check': '6 RC_same recomputed equals the registered RC', 'condition': c,
                 'value': f'{len(new)} recomputed, {len(reg)} registered (seed) units',
                 'detail': 'pass' if same else 'reported: differs from the registered seed pairs'})
  return rows


# ---------------------------------------------------------------- readings

def x_reading(lo, hi):
  if lo != lo or hi != hi:
    return 'not computed'
  if lo > 0:
    return 'a new VM changes the recompilation rate (higher on a new VM)'
  if hi < 0:
    return 'a new VM changes the recompilation rate (lower on a new VM)'
  return 'not distinguishable'


def x_table(sp, conds, args):
  rows = []
  for c in conds:
    for i, r in enumerate(diff_rows(sp, f'RC_new({c})', f'RC_same({c})', f'X({c}) = RC_new({c}) minus RC_same({c})', args)):
      rows.append({'condition': c, **r, 'role': 'primary' if r['statistic'] == 'flip_top1' else 'secondary',
                   'reading': x_reading(r['ci_low'], r['ci_high']) if r['n_targets'] else 'not computed'})
  return pd.DataFrame(rows)


def table_ii_sentence(x):
  top = x[(x.condition == 'G') & (x.statistic == 'flip_top1')]
  if top.empty:
    return 'G r3 not analysed.'
  if top.reading.iloc[0].startswith('a new VM changes'):
    return ('Rule e2: X(G) reads "changes": the paper reports RC_new(G) beside RC(G) in Table II and states which '
            'reference each excess uses; the registered references stand.')
  return 'Rule e2: X(G) reads "not distinguishable": the registered references stand, nothing is added to Table II.'


# ---------------------------------------------------------------- main

def main():
  ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
  ap.add_argument('--samples', required=True)
  ap.add_argument('--main', default=str(REPO / 'results' / 'analysis' / 'main_20261014' / 'primary_v2'))
  ap.add_argument('--a100', default=str(REPO / 'results' / 'analysis' / 'a100_20261018'))
  ap.add_argument('--targets', default=str(REPO / 'targets' / 'posebusters_subset.csv'))
  ap.add_argument('--out', required=True)
  ap.add_argument('--replicates', type=int, default=10000)
  ap.add_argument('--seed', type=int, default=2026)
  args = ap.parse_args()
  out = pathlib.Path(args.out)
  out.mkdir(parents=True, exist_ok=True)
  all_targets = sorted(pd.read_csv(args.targets, dtype=str).pdb_id)

  d = load_xvm(args.samples)
  conds = [c for c in ('G', 'A') if (d.condition == c).any()]
  r12 = load_r12(args)
  chk = fu13.check_exit(d) + fu13.check_machine(d) + check_machine_ref(r12, conds) + check_r3_machine(d)
  pd.DataFrame(chk).to_csv(out / 'checks.csv', index=False)
  if any(r['detail'] != 'pass' for r in chk):
    raise SystemExit('check 1 or 2 failed (checks.csv); nothing is compared')
  okall = f34.sample_rows(pd.concat([d, r12], ignore_index=True))
  comp = completeness(okall, all_targets)
  chk += check_device(d)
  new_s = {c: [session_name(s) for s in d[d.condition == c].session] for c in conds}
  ref_s = {c: [session_name(s) for s in r12[r12.condition == c].session] for c in conds}
  s4, stack = check_stack(new_s, ref_s, '4b settings and stack of every r3 session against r1 and r2')
  s5, vms = check_vms(d, r12)
  chk += s4 + s5

  specs = [s for c in conds for s in rc_specs(c)]
  sp, sm, cons, counts = f34.build(okall, specs)
  ref_sp = {'main': pd.read_csv(pathlib.Path(args.main) / 'contrasts' / 'pairs_seed.csv'),
            'a100': pd.read_csv(pathlib.Path(args.a100) / 'pairs_seed.csv')}
  chk += check_registered(sp, ref_sp, conds)
  ident, det = [], []
  for c in conds:
    for a, b, what in (('r1', 'r2', 'same VM'), ('r1', 'r3', 'new VM'), ('r2', 'r3', 'new VM')):
      s, dd = light_identity(okall, (c, a), (c, b), f'bit-identity {c} {b} against {a} ({what}, light-fetch files)')
      ident.append({**s, 'vm': what})
      det += dd
  pd.DataFrame(chk).to_csv(out / 'checks.csv', index=False)
  pd.DataFrame(det).to_csv(out / 'checks_detail.csv', index=False)

  con = contrasts.summarize(sp, sm, cons, args) if len(sp) else pd.DataFrame()
  x = x_table(sp, conds, args)
  mo = pd.concat([main_output_changes(okall, c, [('r1', 'r2', 'same VM'), ('r1', 'r3', 'new VM'), ('r2', 'r3', 'new VM')])
                  for c in conds], ignore_index=True) if conds else pd.DataFrame()
  outputs = {'completeness': comp, 'contrasts': con, 'identity': pd.DataFrame(ident), 'x': x, 'main_output': mo,
             'stack': stack, 'vm_names': vms, 'pairs_seed': sp, 'pairs_sample': sm, 'counts': counts}
  for name, df in outputs.items():
    df.to_csv(out / f'{name}.csv', index=False)
  (out / 'report.md').write_text(report(chk, comp, con, ident, x, mo))
  print((out / 'report.md').read_text())
  return 0


def report(chk, comp, con, ident, x, mo):
  pct, num = f34.pct, f34.num
  L = ['# Follow-up 21: recompilation on a new VM of the same GPU type', '',
       'notes/analysis_plan.md section 10, row "Follow-up experiment 21" (label xvm); reported whatever it shows.', '',
       '## Checks', '', '| check | condition | value | result |', '|---|---|---|---|']
  L += [f'| {r["check"]} | {r["condition"]} | {r["value"]} | {r["detail"]} |' for r in chk]
  inc = comp[comp.status != 'complete']
  L += ['', f'Completeness: {len(comp) - len(inc)} of {len(comp)} planned (condition, r3, seed) cells complete'
        + ('' if inc.empty else '; not complete: ' + ', '.join(
            f'{r.condition} seed {r.seed} ({r.targets_with_5_samples}/{r.planned_targets}, {r.status})'
            for r in inc.itertuples())) + '.']
  L += ['', '## X(P) = RC_new(P) minus RC_same(P) (item e 1)', '',
        '| comparison | statistic | role | estimate | 95% CI | targets | reading |', '|---|---|---|---|---|---|---|']
  L += [f'| {r.comparison} | {r.statistic} | {r.role} | {pct(r.estimate)} | {pct(r.ci_low)} to {pct(r.ci_high)} | '
        f'{r.n_targets} | {r.reading} |' for r in x.itertuples()]
  L += ['', table_ii_sentence(x) if len(x) else 'Nothing analysed.', '',
        '## Recompilation rates', '', '| contrast | statistic | estimate | 95% CI | n | targets |', '|---|---|---|---|---|---|']
  if len(con):
    keep = con[con.statistic.isin(contrasts.FLIP_STATS + ['m4_ligand_rmsd_median', 'm4_ca_rmsd_global_median',
                                                          'm5_ranking_score_median', 'bit_identical_fraction'])]
    for r in keep.itertuples():
      fmt = pct if r.statistic in contrasts.FLIP_STATS + ['bit_identical_fraction'] else num
      L.append(f'| {r.contrast} | {r.statistic} | {fmt(r.estimate)} | {fmt(r.ci_low)} to {fmt(r.ci_high)} | {r.n} | '
               f'{r.n_targets} |')
  L += ['', '## Bit-identical (target, seed), the light-fetch files', '']
  L += [f'- {r["condition"]} ({r["vm"]}): {r["value"]}; {r["detail"]}' for r in ident]
  L += ['', '## Main output (descriptive)', '', '| sides | VM | targets | main output differs | its success differs |',
        '|---|---|---|---|---|']
  L += [f'| {r.sides} | {r.vm} | {r.targets} | {r.main_output_differs} | {r.success_differs} |' for r in mo.itertuples()]
  return '\n'.join(L) + '\n'


if __name__ == '__main__':
  sys.exit(main())
