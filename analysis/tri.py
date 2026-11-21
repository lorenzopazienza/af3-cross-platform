"""Follow-up 18, AlphaFold 3's default Triton attention on the H100 and the A100 (label tri; notes/analysis_plan.md
section 10, row "Follow-up experiment 18").

    analysis/.venv/bin/python analysis/tri.py --samples <collect.py samples.csv of the tri sessions> --out DIR \\
        [--gtat results/analysis/gtat_<DATE>] [--pb_full] [--replicates 10000] [--seed 2026]

Conditions. Follow-up 18 (label tri), from each process's configuration (collect.py unchanged): h100_triton -> HT
(r1 and r2, part A), a100_triton -> AT (r2, part B, optional: without AT r2 sessions its rows say "not run").
Used as they are, from their analysis folders: AT r1, A r1 and r2 (follow-up 3, --a100 samples.csv and
pairs_seed.csv); H r1 and r2 (follow-up 9, --h100); G r1 and r2, GT r1 (main study, --main); GT r2 (follow-up 15,
--rcx). Seed pairs of the earlier rows (HW(H,A), RC(H), RC(A), HW(H,G), HW(G,A), DU(A), DU, RC(G), RC(GT)) are read
from their pairs_seed.csv, never recomputed.
Checks (item c), before any result is read; the analysis stops if 1 or 2 fails:
  1 every process of the tri sessions exits 0 or its failure is recorded (collect.py's failure column);
  2 one machine type per condition, and AT r2's is AT r1's (a2-ultragpu-1g); HT a3-highgpu-1g;
  3 completeness: HT r1, HT r2 and AT r2, the 60 targets, seeds 1 to 5 (a missing unit leaves the contrasts it
    belongs to, section 8; reported);
  4 settings and stack: run_alphafold called with --flash_attention_implementation=triton in every HT and AT r2 process
    (the command of runs.jsonl, the harness's command record); no AF3_FU13_* variable and no
    JAX_DEFAULT_MATMUL_PRECISION in the environment of any (session.json xla_env, env_overrides and the configuration's
    env) and no float32 trunk recorded (af3_entry_stats.json "fu13"); every process on the expected device (NVIDIA
    H100 80GB HBM3, NVIDIA A100-SXM4-80GB); for every session, the NVIDIA driver, kernel, Python version, Python
    packages, pip_freeze.txt (rcx.stack_of) and the image family (as the wrapper log printed it; "not recorded"
    otherwise) against those of H (for HT) and of AT r1 (for AT r2); reported, any difference listed;
  5 every HT r2 session ran on a VM whose name (VM_NAME of launch_record.txt) differs from the VM of the HT r1
    session of the same target; reported.
Statistics (item d): section 5's target-clustered bootstrap (--replicates 10000, --seed 2026), every contrast paired on
(target, seed) with the per-seed top-1 (M2), seeds 1 to 5, all terms of a difference on the same resampled targets
(fu13.diff_two).
  Primary (T1): HW(HT,AT) = HT r1 against AT r1 and E_T = HW(HT,AT) minus RC(HT), RC(HT) = HT r1 against HT r2 (new
    VMs); top-1 flip primary, success flip and flips that matter secondary.
  Co-primary (T2): per target, AlphaFold 3's main output (winners.main_output: the highest full-precision
    ranking_score among the 25 samples of seeds 1 to 5, exact ties to the lowest (seed, sample), over the seeds both
    sides ran) under HT r1, HT r2 and AT r1. MO(HT,AT), the fraction of targets whose main output is a different
    (seed, sample) under HT and AT; MO(HT), the same between HT r1 and HT r2; E_MO = MO(HT,AT) minus MO(HT) on the
    targets common to both, the same resampled targets (one unit per target); the same three for the main output's
    success (M1). For every target whose main output differs (HT against AT, and HT r1 against HT r2): the ligand RMSD
    between the two returned ligands after pocket superposition (winners.POSE_RMSD, the function of M4), each one's
    ligand RMSD from the crystal (m1_ligand_rmsd), both, one or neither within 2 A (RMSD below 2 A, the M1
    threshold); where success differs, the criterion that fails on the failing side ("RMSD", "PoseBusters", "RMSD and
    PoseBusters", "not scored"; pb_flips.criterion on the saved M1 results), the failing PoseBusters checks, and the
    excess over each criterion's tolerance (pb_flips.add_margins: the RMSD excess always; the PoseBusters checks'
    excess only with --pb_full, which reruns M1 with PoseBusters' full report as the post hoc analysis pb_flips does,
    after its exact-reproduction checks; without it those excesses are missing and the class "not classified").
  Secondary: E_T for success flips and flips that matter; E_T against the larger of RC(HT) and RC(AT), the larger
    chosen per statistic by the point estimates on the full sample (as Table III), the difference with its interval
    on the common targets (with part B only); RC(HT) and RC(AT) with their bit-identity counts (the files of the light
    fetch, the mmCIF timestamp masked: rcx.light_identical) and the bit-identical sample fraction; RC(AT) minus RC(A)
    (same VM), RC(HT) minus RC(H) (new VMs); HW(HT,AT) minus HW(H,A) (the attention's effect on the pair); DU(H) = H r1
    against HT r1, beside DU(A) and DU (L4).
  Descriptive (T3): HW(GT,AT) (a declared replication of the post hoc analysis gtat: recomputed from the same seed
    pairs with the same bootstrap and compared with its outputs, --gtat; "equals", "DIFFERS" or "gtat output not
    found"; the gtat values stand), HW(GT,HT) and HW(HT,AT), each with HW minus the larger of the two RC rates (RC(GT)
    of follow-up 15; RC(AT) only with part B, otherwise the available one, named), beside the XLA pairs HW(G,A),
    HW(H,G) and HW(H,A) with theirs. Per-seed top-1 success rate (M1) of HT beside H, A and AT. Cost: ratios of the
    median process wall time and of the compiled device memory, overall and per bucket, HT to H and AT r2 to AT r1.
Reading rules (item e): (1) T1, "beyond recompilation" if the 95% interval of E_T lies above 0, "not distinguishable
from recompilation" otherwise (not a test of equivalence); the same for the success flip rate and flips that matter,
secondary. (2) T2, "the main output differs beyond same-type rescheduling" if the 95% interval of E_MO lies above 0,
"not distinguishable from same-type rescheduling" otherwise; the same for the main output's success, secondary; the
categories are counted without a threshold. (3) With part B, E_T against the larger RC read with rule 1 beside it;
if the two readings differ, both are stated. (4) RC(HT), RC(AT): descriptive, nothing attributed to the VM change.
(5) T3 descriptive; HW(GT,AT) keeps only the reading of gtat.
Outputs in --out: checks.csv, checks_detail.csv, stack.csv, completeness.csv, contrasts.csv, e_t.csv, comparisons.csv,
identity.csv, main_output_units.csv, main_output.csv, main_output_changes.csv, default_pairs.csv,
gtat_replication.csv, success.csv, cost.csv, failures.csv, pairs_seed.csv, pairs_sample.csv, counts.csv, report.md.
"""
import argparse
import importlib.metadata
import json
import pathlib
import re
import sys
import types

import numpy as np
import pandas as pd

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / 'analysis'))
sys.path.insert(0, str(REPO / 'harness'))
import contrasts  # noqa: E402
import followup34 as f34  # noqa: E402
import fu13  # noqa: E402
import fu14  # noqa: E402
import pb_flips  # noqa: E402
import rcx  # noqa: E402
import winners  # noqa: E402

SEEDS = [1, 2, 3, 4, 5]
CONFIG = {'h100_triton': 'HT', 'a100_triton': 'AT'}
DEVICE = {'HT': 'NVIDIA H100 80GB HBM3', 'AT': 'NVIDIA A100-SXM4-80GB'}
MACHINE = {'HT': 'a3-highgpu-1g', 'AT': 'a2-ultragpu-1g'}
TRITON = '--flash_attention_implementation=triton'
# condition, repetition of the planned arms (60 targets, seeds 1 to 5)
PLANNED = [('HT', 'r1'), ('HT', 'r2'), ('AT', 'r2')]
# reference sessions of check 4: (sessions file, pattern)
REF_SESSIONS = {'HT': ('scripts/analysis_runs/h100/sessions.txt', r'_h100_h100_h_\d+$'),
                'AT': ('scripts/analysis_runs/a100/sessions.txt', r'_a100_a100_at_\d+$')}
SPECS = [('HW(HT,AT)', 'HW', ('HT', 'r1'), ('AT', 'r1'), True), ('RC(HT)', 'RC', ('HT', 'r1'), ('HT', 'r2'), False),
         ('RC(AT)', 'RC', ('AT', 'r1'), ('AT', 'r2'), False), ('DU(H)', 'DU', ('H', 'r1'), ('HT', 'r1'), False),
         ('HW(GT,AT)', 'HW', ('GT', 'r1'), ('AT', 'r1'), False), ('HW(GT,HT)', 'HW', ('GT', 'r1'), ('HT', 'r1'), False)]
# registered seed pairs: name -> analysis folder key
REGISTERED = {'HW(H,A)': 'h100', 'RC(H)': 'h100', 'HW(H,G)': 'h100', 'RC(A)': 'a100', 'HW(G,A)': 'a100',
              'DU(A)': 'a100', 'DU': 'main', 'RC(G)': 'main', 'RC(GT)': 'rcx'}
# T3: pair -> (the two RC rates whose larger is subtracted)
DEFAULT_PAIRS = {'HW(GT,AT)': ('RC(GT)', 'RC(AT)'), 'HW(GT,HT)': ('RC(GT)', 'RC(HT)'), 'HW(HT,AT)': ('RC(HT)', 'RC(AT)')}
XLA_PAIRS = {'HW(G,A)': ('RC(G)', 'RC(A)'), 'HW(H,G)': ('RC(H)', 'RC(G)'), 'HW(H,A)': ('RC(H)', 'RC(A)')}
WITHIN_A = 2.0                              # within 2 A of the crystal pose: ligand RMSD below 2 A (the M1 threshold)
PB_VERSION = pb_flips.PB_VERSION


# ---------------------------------------------------------------- loading

def load_tri(path):
  d = pd.read_csv(path, low_memory=False)
  d = d[(d.label == 'tri') & (d.num_recycles == 10) & (d.num_diffusion_samples == 5)].copy()
  d['condition'] = d.config.map(CONFIG)
  if d.condition.isna().any():
    raise SystemExit(f'sessions of label tri with configurations not of follow-up 18: {sorted(set(d[d.condition.isna()].config))}')
  bad = d[(d.condition == 'AT') & (d.repetition != 'r2')]
  if len(bad):
    raise SystemExit('AT rows of label tri that are not r2 (follow-up 18 runs only AT r2)')
  return d


def load_references(args):
  """AT r1, A r1 and r2 (follow-up 3), H r1 and r2 (follow-up 9), G r1 and r2 and GT r1 (main study), GT r2 (rcx)."""
  a = f34.load_samples(pathlib.Path(args.a100) / 'samples.csv', ['a100'])
  a = a[((a.condition == 'AT') & (a.repetition == 'r1')) | ((a.condition == 'A') & a.repetition.isin(['r1', 'r2']))]
  h = f34.load_samples(pathlib.Path(args.h100) / 'samples.csv', ['h100'])
  h = h[(h.condition == 'H') & h.repetition.isin(['r1', 'r2'])]
  m = f34.load_samples(pathlib.Path(args.main) / 'samples.csv', ['main'])
  m = m[((m.condition == 'G') & m.repetition.isin(['r1', 'r2'])) | ((m.condition == 'GT') & (m.repetition == 'r1'))]
  frames = [a, h, m]
  rp = pathlib.Path(args.rcx) / 'samples.csv'
  if rp.is_file():
    r = pd.read_csv(rp, low_memory=False)
    r = r[(r.label == 'rcx') & (r.config == 'l4_triton') & (r.repetition == 'r2')].copy()
    r['condition'] = 'GT'
    frames.append(r)
  return pd.concat(frames, ignore_index=True)


def registered_pairs(args):
  src = {'h100': pathlib.Path(args.h100) / 'pairs_seed.csv', 'a100': pathlib.Path(args.a100) / 'pairs_seed.csv',
         'main': pathlib.Path(args.main) / 'contrasts' / 'pairs_seed.csv', 'rcx': pathlib.Path(args.rcx) / 'pairs_seed.csv'}
  out = []
  for name, key in REGISTERED.items():
    if src[key].is_file():
      p = pd.read_csv(src[key])
      out.append(p[p.contrast == name])
  return pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=['contrast', 'target'])


# ---------------------------------------------------------------- checks

def completeness(ok, all_targets):
  rows = []
  for c, rep in PLANNED:
    for s in SEEDS:
      units = ok[(ok.condition == c) & (ok.repetition == rep) & (ok.seed == s)].groupby('target').size()
      have = int((units == 5).sum())
      n = len(all_targets)
      rows.append({'condition': c, 'repetition': rep, 'seed': s, 'planned_targets': n, 'targets_with_5_samples': have,
                   'status': 'complete' if have == n else ('not run' if have == 0 else 'incomplete')})
  return pd.DataFrame(rows)


def check_machine_ref(d):
  rows = []
  for c, g in d.groupby('condition'):
    types_ = sorted(g.machine_type.dropna().astype(str).unique())
    rows.append({'check': '2b the machine type of the source arm', 'condition': c, 'value': ', '.join(types_),
                 'detail': 'pass' if types_ == [MACHINE[c]] else f'FAIL: expected {MACHINE[c]}'})
  return rows


def command_of(row, root=REPO):
  """The run_alphafold command the harness recorded for one process (runs.jsonl of its run), or None."""
  f = root / str(row['session']) / 'runs.jsonl'
  if not f.is_file():
    return None
  for line in f.read_text().splitlines():
    if line.strip():
      j = json.loads(line)
      if j.get('pdb_id') == row['target']:
        return j.get('command')
  return None


def env_of(row, root=REPO):
  """Every environment variable recorded for the process's run: xla_env, env_overrides and the configuration's env."""
  sj = root / str(row['session']) / 'session.json'
  if not sj.is_file():
    return {}
  j = json.loads(sj.read_text())
  env = {}
  for part in (j.get('xla_env'), j.get('env_overrides'), (j.get('config') or {}).get('env')):
    if isinstance(part, dict):
      env.update({str(k): v for k, v in part.items()})
  return env


def check_settings(d, root=REPO):
  rows, detail = [], []
  for c, g in fu14.processes(d).groupby('condition'):
    bad = []
    for r in g.to_dict('records'):
      cmd = command_of(r, root)
      env = env_of(r, root)
      bf16, _ = fu14.settings_of(r, root)
      why = []
      if not cmd or TRITON not in cmd:
        why.append('no ' + TRITON if cmd else 'no command recorded')
      extra = sorted(k for k in env if k.startswith('AF3_FU13_') or k == 'JAX_DEFAULT_MATMUL_PRECISION')
      if extra:
        why.append('environment holds ' + ', '.join(extra))
      if bf16 is not None:
        why.append(f'float32 trunk recorded ({bf16})')
      if r.get('device_kind') != DEVICE[c]:
        why.append(f'device {r.get("device_kind")}')
      if why:
        bad.append(f'{r["target"]} {r["repetition"]} seed {r["seed"]} ({"; ".join(why)})')
    detail += [{'check': '4a settings reached every process', 'condition': c, 'process': b} for b in bad]
    rows.append({'check': '4a settings reached every process', 'condition': c,
                 'value': f'{len(g) - len(bad)}/{len(g)} processes',
                 'detail': 'pass' if not bad else 'reported: ' + '; '.join(bad)[:250]})
  return rows, detail


def session_name(s):
  m = re.search(r'(\d{8}T\d{6}Z_[^/]+)', str(s))
  return m.group(1) if m else str(s)


def image_of(session, many=None):
  """The image the session's VM was created from, as its wrapper log printed it ("image PROJECT/FAMILY"), or None
  (harness/plans/config_diff_rcx.image_of)."""
  many = many or REPO / 'results' / 'af3' / '.many'
  for log in sorted(many.glob('*/*.log')):
    text = log.read_text(errors='replace')
    if f'results/af3/{session}/' in text:
      m = re.findall(r'^\s+image (\S+?)(?:,|$)', text, re.M)
      if m:
        return m[-1]
  return None


def ref_sessions(c, root=REPO):
  path, pat = REF_SESSIONS[c]
  f = root / path
  if not f.is_file():
    return []
  return [l.split('#')[0].strip() for l in f.read_text().splitlines()
          if l.split('#')[0].strip() and re.search(pat, l.split('#')[0].strip())]


def stack_row(session, af3, many):
  st = rcx.stack_of(af3 / session)
  st['image'] = image_of(session, many) or 'not recorded'
  return st


def check_stack(d, af3=REPO / 'results' / 'af3', many=None, refs=None):
  """Check 4b: every session of HT (against H) and of AT r2 (against AT r1), field by field; reported."""
  refs = refs or ref_sessions
  rows, table = [], []
  for c, g in d.groupby('condition'):
    ref = [stack_row(s, af3, many) for s in refs(c)]
    diffs = {}
    sessions = sorted({session_name(s) for s in g.session})
    for s in sessions:
      st = stack_row(s, af3, many)
      for k, v in st.items():
        known = sorted({str(x[k]) for x in ref})
        same = str(v) in known
        table.append({'condition': c, 'session': s, 'field': k, 'value': v, 'reference': ' | '.join(known), 'same': same})
        if not same:
          diffs.setdefault(k, set()).add(f'{v} (reference {" | ".join(known)})')
    rows.append({'check': '4b stack and image of every session against the source arm', 'condition': c,
                 'value': f'{len(sessions)} sessions against {len(ref)} reference sessions',
                 'detail': 'pass' if not diffs else 'reported, differs in ' + '; '.join(
                     f'{k}: {", ".join(sorted(v))}' for k, v in sorted(diffs.items()))[:400]})
  return rows, pd.DataFrame(table)


def vm_name(session, af3=REPO / 'results' / 'af3'):
  for d in (af3 / session, af3 / '.launcher' / session):
    f = d / 'launch_record.txt'
    if f.is_file():
      for line in f.read_text().splitlines():
        if line.startswith('VM_NAME='):
          return line.split('=', 1)[1].strip()
  return None


def check_vms(d, af3=REPO / 'results' / 'af3'):
  """Check 5: the VM of every HT r2 session differs from the VM of the HT r1 session of each of its targets."""
  ht = d[d.condition == 'HT']
  r1 = {t: {session_name(s) for s in g.session} for t, g in ht[ht.repetition == 'r1'].groupby('target')}
  bad, n = [], 0
  for (s, t), _ in ht[ht.repetition == 'r2'].groupby([ht.session.map(session_name), 'target']):
    n += 1
    v2 = vm_name(s, af3)
    v1 = {vm_name(x, af3) for x in r1.get(t, set())}
    if v2 is None or None in v1 or not v1:
      bad.append(f'{t}: VM not recorded or no r1 session ({s})')
    elif v2 in v1:
      bad.append(f'{t}: same VM {v2}')
  return {'check': '5 HT r2 on a VM other than HT r1 (same target)', 'condition': 'HT r2 against HT r1',
          'value': f'{n - len(bad)}/{n} (session, target) on another VM' if n else 'not run',
          'detail': ('pass' if not bad else 'reported: ' + '; '.join(bad)[:300]) if n else 'not run'}


def identity(ok, c):
  """Bit-identity of r2 against r1 of condition c, the files of the light fetch (rcx.light_identical)."""
  check = f'{c} r2 against {c} r1 (the files of the light fetch)'
  if not ((ok.condition == c) & (ok.repetition == 'r2')).any():
    return {'check': check, 'condition': c, 'value': 'not run', 'detail': 'not run'}, []
  det = []
  for (t, sd), g in ok[(ok.condition == c) & (ok.repetition == 'r2')].groupby(['target', 'seed']):
    r = ok[(ok.condition == c) & (ok.repetition == 'r1') & (ok.target == t) & (ok.seed == sd)]
    res = rcx.light_identical(contrasts.output_dir(r), contrasts.output_dir(g), sd) if len(r) else \
        {'status': 'missing', 'differ': [], 'missing': ['no reference run']}
    det.append({'check': check, 'target': t, 'seed': sd, 'status': res['status'],
                'files': '; '.join(res['differ'] + res['missing'])[:300]})
  s = {'check': check, 'condition': c, 'detail': rcx.identity_reading(det),
       'value': (f'{sum(x["status"] == "identical" for x in det)}/{len(det)} identical, '
                 f'{sum(x["status"] == "differ" for x in det)} differ, {sum(x["status"] == "missing" for x in det)} lack files')}
  return s, det


# ---------------------------------------------------------------- readings

def beyond_reading(ci_low):
  """Rule e1 (and e3): beyond recompilation if the 95% interval of the excess lies above 0."""
  if ci_low != ci_low:
    return 'not computed'
  return 'beyond recompilation' if ci_low > 0 else 'not distinguishable from recompilation'


def mo_reading(ci_low):
  """Rule e2: the main output differs beyond same-type rescheduling if the 95% interval of E_MO lies above 0."""
  if ci_low != ci_low:
    return 'not computed'
  return ('the main output differs beyond same-type rescheduling' if ci_low > 0
          else 'not distinguishable from same-type rescheduling')


def rate(frame, stat, args):
  """Mean of a per-unit statistic with the target-clustered interval."""
  if frame is None or not len(frame):
    return {'n': 0, 'n_targets': 0, 'estimate': np.nan, 'ci_low': np.nan, 'ci_high': np.nan}
  targets = sorted(frame.target.unique())
  groups = [frame[frame.target == t][stat].astype(float).values for t in targets]
  lo, hi = contrasts.boot_ci(groups, contrasts.mean, np.random.default_rng(args.seed), args.replicates)
  return {'n': len(frame), 'n_targets': len(targets), 'estimate': contrasts.mean(frame[stat].astype(float).values),
          'ci_low': lo, 'ci_high': hi}


def pairs_of(sp, reg, name):
  if name in set(sp.contrast):
    return sp[sp.contrast == name]
  return reg[reg.contrast == name] if len(reg) else reg


def e_t_table(sp, reg, args):
  """E_T = HW(HT,AT) minus RC(HT) (rule e1), and, with part B, against the larger of RC(HT) and RC(AT) (rule e3)."""
  rows = []
  hw = pairs_of(sp, reg, 'HW(HT,AT)')
  for i, stat in enumerate(contrasts.FLIP_STATS):
    rc = pairs_of(sp, reg, 'RC(HT)')
    r = fu13.diff_two(hw, rc, stat, args.seed, args.replicates)
    rows.append({'comparison': 'E_T = HW(HT,AT) minus RC(HT)', 'reference': 'RC(HT)', 'statistic': stat,
                 'role': 'primary' if i == 0 else 'secondary', **r, 'reading': beyond_reading(r['ci_low'])})
    rat = pairs_of(sp, reg, 'RC(AT)')
    if len(rat):
      est = {n: (x[stat].astype(float).mean() if len(x) else np.nan) for n, x in (('RC(HT)', rc), ('RC(AT)', rat))}
      big = 'RC(AT)' if est['RC(AT)'] > est['RC(HT)'] else 'RC(HT)'
      r = fu13.diff_two(hw, rat if big == 'RC(AT)' else rc, stat, args.seed, args.replicates)
      rows.append({'comparison': 'HW(HT,AT) minus the larger of RC(HT) and RC(AT)', 'reference': big, 'statistic': stat,
                   'role': 'secondary (rule e3)', **r, 'reading': beyond_reading(r['ci_low'])})
    else:
      rows.append({'comparison': 'HW(HT,AT) minus the larger of RC(HT) and RC(AT)', 'reference': 'RC(AT) not run',
                   'statistic': stat, 'role': 'secondary (rule e3)', 'n_targets': 0, 'estimate': np.nan,
                   'ci_low': np.nan, 'ci_high': np.nan, 'reading': 'not run (part B)'})
  return pd.DataFrame(rows)


def comparisons(sp, reg, args):
  rows = []
  for a, b in (('RC(AT)', 'RC(A)'), ('RC(HT)', 'RC(H)'), ('HW(HT,AT)', 'HW(H,A)'), ('DU(H)', 'DU(A)'), ('DU(H)', 'DU')):
    fa, fb = pairs_of(sp, reg, a), pairs_of(sp, reg, b)
    for stat in contrasts.FLIP_STATS:
      r = fu13.diff_two(fa, fb, stat, args.seed, args.replicates) if len(fa) and len(fb) else \
          {'n_targets': 0, 'estimate': np.nan, 'ci_low': np.nan, 'ci_high': np.nan}
      rows.append({'comparison': f'{a} minus {b}', 'statistic': stat, **r})
  return pd.DataFrame(rows)


def default_pairs(sp, reg, args):
  """T3: each default-attention pair (and the XLA pairs beside them) with its rate and HW minus the larger RC rate."""
  rows = []
  for group, pairs in (('default attention (Triton)', DEFAULT_PAIRS), ('XLA attention (earlier rows)', XLA_PAIRS)):
    for name, (r1, r2) in pairs.items():
      hw = pairs_of(sp, reg, name)
      for stat in contrasts.FLIP_STATS:
        base = {'group': group, 'pair': name, 'statistic': stat, **{f'hw_{k}': v for k, v in rate(hw, stat, args).items()}}
        rcs = {n: pairs_of(sp, reg, n) for n in (r1, r2)}
        have = {n: f for n, f in rcs.items() if len(f)}
        if not len(hw) or not have:
          rows.append({**base, 'reference': 'not computed', 'n_targets': 0, 'estimate': np.nan, 'ci_low': np.nan,
                       'ci_high': np.nan})
          continue
        big = max(have, key=lambda n: have[n][stat].astype(float).mean())
        note = '' if len(have) == 2 else f' ({", ".join(sorted(set(rcs) - set(have)))} not measured)'
        r = fu13.diff_two(hw, have[big], stat, args.seed, args.replicates)
        rows.append({**base, 'reference': big + note, **r})
  return pd.DataFrame(rows)


GTAT_TARGETS = '60 targets'          # the rows of gtat's rates.csv on the 60 targets (the other set leaves out glycans)
GTAT_RATES = ['targets', 'contrast', 'statistic', 'n', 'n_targets', 'events', 'targets_with_event', 'estimate', 'ci_low',
              'ci_high']
GTAT_PAIRS = ['contrast', 'target', 'seed_a', 'seed_b', 'top1_a', 'top1_b'] + contrasts.FLIP_STATS


def gtat_replication(sp, con, gtat_dir):
  """HW(GT,AT) recomputed here against the outputs of the post hoc analysis gtat (results/analysis/gtat_20261121):
  its pairs_seed.csv (rows "HW(GT,AT)": every seed pair, unit by unit) and its rates.csv (rows "60 targets",
  "HW(GT,AT)": n, n_targets, estimate and 95% interval per flip statistic, within 1e-12). A file without the expected
  columns is reported, not read. The gtat values stand whatever this shows."""
  rows = []
  d = pathlib.Path(gtat_dir) if gtat_dir else None
  mine = sp[sp.contrast == 'HW(GT,AT)'].sort_values(['target', 'seed_a']) if len(sp) else sp
  ps = d / 'pairs_seed.csv' if d else None
  if ps is not None and ps.is_file():
    g = pd.read_csv(ps)
    if not set(GTAT_PAIRS) <= set(g.columns):
      rows.append({'what': 'seed pairs (pairs_seed.csv)', 'statistic': 'all', 'here': len(mine), 'gtat': None,
                   'result': 'gtat pairs_seed.csv lacks ' + ', '.join(sorted(set(GTAT_PAIRS) - set(g.columns)))})
    else:
      g = g[g.contrast == 'HW(GT,AT)'].sort_values(['target', 'seed_a'])
      if len(g):
        cols = GTAT_PAIRS[1:]
        a = mine[cols].reset_index(drop=True).astype(str) if len(mine) else pd.DataFrame(columns=cols)
        b = g[cols].reset_index(drop=True).astype(str)
        same = len(a) == len(b) and a.equals(b)
        rows.append({'what': 'seed pairs (pairs_seed.csv)', 'statistic': 'all', 'here': len(mine), 'gtat': len(g),
                     'result': 'equals' if same else 'DIFFERS'})
  rs = d / 'rates.csv' if d else None
  if rs is not None and rs.is_file():
    g = pd.read_csv(rs)
    if not set(GTAT_RATES) <= set(g.columns):
      rows.append({'what': 'estimate and 95% interval (rates.csv)', 'statistic': 'all', 'here': None, 'gtat': None,
                   'result': 'gtat rates.csv lacks ' + ', '.join(sorted(set(GTAT_RATES) - set(g.columns)))})
    else:
      g = g[(g.targets.astype(str) == GTAT_TARGETS) & (g.contrast == 'HW(GT,AT)')]
      for stat in contrasts.FLIP_STATS:
        x = con[(con.contrast == 'HW(GT,AT)') & (con.statistic == stat)] if len(con) else con
        y = g[g.statistic == stat]
        if len(x) and len(y):
          same = (int(x.n.iloc[0]) == int(y.n.iloc[0]) and int(x.n_targets.iloc[0]) == int(y.n_targets.iloc[0])
                  and all(abs(float(x[k].iloc[0]) - float(y[k].iloc[0])) < 1e-12 for k in ('estimate', 'ci_low', 'ci_high')))
          rows.append({'what': f'n, estimate and 95% interval (rates.csv, {GTAT_TARGETS})', 'statistic': stat,
                       'here': f'n {int(x.n.iloc[0])}, {x.estimate.iloc[0]:.6f} [{x.ci_low.iloc[0]:.6f}, {x.ci_high.iloc[0]:.6f}]',
                       'gtat': f'n {int(y.n.iloc[0])}, {y.estimate.iloc[0]:.6f} [{y.ci_low.iloc[0]:.6f}, {y.ci_high.iloc[0]:.6f}]',
                       'result': 'equals' if same else 'DIFFERS'})
  if not rows:
    rows.append({'what': 'HW(GT,AT)', 'statistic': 'all', 'here': len(mine), 'gtat': None,
                 'result': 'gtat output not found'})
  return pd.DataFrame(rows)


# ---------------------------------------------------------------- main output (T2)

MO_CONTRASTS = [('MO(HT,AT)', ('HT', 'r1'), ('AT', 'r1')), ('MO(HT)', ('HT', 'r1'), ('HT', 'r2'))]


def mo_units(ok):
  """One row per (contrast, target): the main output of each side (winners.main_output over the seeds both ran)."""
  rows = []
  for name, (ca, ra), (cb, rb) in MO_CONTRASTS:
    a = ok[(ok.condition == ca) & (ok.repetition == ra) & ok.seed.isin(SEEDS)]
    b = ok[(ok.condition == cb) & (ok.repetition == rb) & ok.seed.isin(SEEDS)]
    for t in sorted(set(a.target) & set(b.target)):
      ga, gb = a[a.target == t], b[b.target == t]
      seeds = sorted(set(ga.seed) & set(gb.seed))
      xa, xb = winners.main_output(ga[ga.seed.isin(seeds)]), winners.main_output(gb[gb.seed.isin(seeds)])
      sa, sb = bool(xa['m1_success'] == True), bool(xb['m1_success'] == True)  # noqa: E712
      ka, kb = (int(xa.seed), int(xa['sample'])), (int(xb.seed), int(xb['sample']))
      rows.append({'contrast': name, 'target': t, 'seeds': len(seeds), 'main_a': f'{ka[0]}:{ka[1]}',
                   'main_b': f'{kb[0]}:{kb[1]}', 'mo_changed': ka != kb, 'mo_success_flip': sa != sb,
                   'success_a': sa, 'success_b': sb, '_a': xa, '_b': xb})
  return pd.DataFrame(rows)


def mo_summary(units, args):
  rows = []
  for stat, role in (('mo_changed', 'co-primary'), ('mo_success_flip', 'co-primary, secondary statistic')):
    for name, _, _ in MO_CONTRASTS:
      g = units[units.contrast == name] if len(units) else units
      rows.append({'quantity': name, 'statistic': stat, 'role': role, 'count': int(g[stat].sum()) if len(g) else 0,
                   **rate(g, stat, args), 'reading': ''})
    a = units[units.contrast == 'MO(HT,AT)'] if len(units) else units
    b = units[units.contrast == 'MO(HT)'] if len(units) else units
    r = fu13.diff_two(a, b, stat, args.seed, args.replicates) if len(a) and len(b) else \
        {'n_targets': 0, 'estimate': np.nan, 'ci_low': np.nan, 'ci_high': np.nan}
    rows.append({'quantity': 'E_MO = MO(HT,AT) minus MO(HT)', 'statistic': stat, 'role': role, 'count': np.nan, 'n': np.nan,
                 **r, 'reading': mo_reading(r['ci_low'])})
  return pd.DataFrame(rows)


def within(x):
  return x == x and x < WITHIN_A


def mo_changes(units, pb_full=False, jobs=1):
  """For each target whose main output differs: pose RMSD between the two returned ligands, their RMSD from the
  crystal, both / one / neither within 2 A; where success differs, the failing criterion and its excess
  (pb_flips.criterion and add_margins; the PoseBusters checks' excess only with pb_full)."""
  if not len(units):
    return pd.DataFrame()
  ch = units[units.mo_changed].copy()
  if not len(ch):
    return pd.DataFrame()
  rows = []
  for r in (types.SimpleNamespace(**x) for x in ch.to_dict('records')):
    xa, xb = r.__dict__['_a'], r.__dict__['_b']
    ra = float(xa['m1_ligand_rmsd']) if pd.notna(xa['m1_ligand_rmsd']) else np.nan
    rb = float(xb['m1_ligand_rmsd']) if pd.notna(xb['m1_ligand_rmsd']) else np.nan
    pose, perr = winners.POSE_RMSD((r.target, xa['model_path'], xb['model_path']))
    n_in = int(within(ra)) + int(within(rb))
    cat = 'not determined' if ra != ra or rb != rb else {2: 'both', 1: 'one', 0: 'neither'}[n_in]
    row = {'contrast': r.contrast, 'target': r.target, 'main_a': r.main_a, 'main_b': r.main_b,
           'pose_rmsd': pose if pose is not None else np.nan, 'pose_error': perr, 'rmsd_exp_a': ra, 'rmsd_exp_b': rb,
           'within_2A': cat, 'success_a': r.success_a, 'success_b': r.success_b, 'success_differs': r.mo_success_flip}
    if r.mo_success_flip:
      fail, okx = (xb, xa) if r.success_a else (xa, xb)
      row.update({'failing_side': 'b' if r.success_a else 'a', 'path_fail': fail['model_path'], 'path_ok': okx['model_path'],
                  'rmsd_fail': float(fail['m1_ligand_rmsd']) if pd.notna(fail['m1_ligand_rmsd']) else np.nan,
                  'rmsd_ok': float(okx['m1_ligand_rmsd']) if pd.notna(okx['m1_ligand_rmsd']) else np.nan,
                  'pb_valid_fail': pb_flips.bool_or_none(fail['m1_pb_valid']),
                  'pb_failed_fail': ';'.join(pb_flips.split_failed(fail['m1_pb_failed'])),
                  'error_fail': fail['m1_error'] if isinstance(fail['m1_error'], str) else '',
                  '_rec_fail': fail, '_rec_ok': okx})
    rows.append(row)
  df = pd.DataFrame(rows)
  flips = df[df.success_differs].copy()
  if len(flips):
    rescored = []
    if pb_full:
      recorded = {p: x for p, x in zip(list(flips.path_fail) + list(flips.path_ok),
                                       list(flips._rec_fail) + list(flips._rec_ok))}
      tasks = sorted({(r.target, r.path_fail) for r in flips.itertuples()} |
                     {(r.target, r.path_ok) for r in flips.itertuples()})
      rescored = pb_flips.rescore_all(tasks, jobs)
      _, bad = pb_flips.reproduction_checks(rescored, recorded)
      if bad:
        raise SystemExit('the M1 rerun does not reproduce the saved M1 (pb_flips checks 3, 3b, 4): ' + '; '.join(bad[:10]))
    pose = {(r.target, r.path_fail, r.path_ok): (r.pose_rmsd if r.pose_rmsd == r.pose_rmsd else None, r.pose_error)
            for r in flips.itertuples()}
    m = pb_flips.add_margins(flips.drop(columns=['_rec_fail', '_rec_ok', 'pose_rmsd', 'pose_error']), rescored, pose)
    keep = ['criterion', 'failing_criteria', 'excess_fail', 'excess_ok', 'worst_excess_fail', 'class']
    df = df.merge(m[['contrast', 'target'] + keep], on=['contrast', 'target'], how='left')
  return df.drop(columns=[c for c in ('_rec_fail', '_rec_ok') if c in df.columns])


def change_counts(ch):
  """Counts per contrast: targets whose main output differs, distinct poses (> 2 A), within-2-A categories, and the
  failing criterion where success differs."""
  rows = []
  if not len(ch):
    return pd.DataFrame(rows)
  for name, g in ch.groupby('contrast'):
    rows.append({'contrast': name, 'category': 'main output differs', 'count': len(g)})
    rows.append({'contrast': name, 'category': 'poses more than 2 A apart', 'count': int((g.pose_rmsd > 2.0).sum())})
    for k in ('both', 'one', 'neither', 'not determined'):
      rows.append({'contrast': name, 'category': f'{k} within 2 A of the crystal', 'count': int((g.within_2A == k).sum())})
    rows.append({'contrast': name, 'category': 'success differs', 'count': int(g.success_differs.sum())})
    if 'criterion' in g.columns:
      for crit, n in g[g.success_differs].criterion.value_counts().items():
        rows.append({'contrast': name, 'category': f'failing criterion: {crit}', 'count': int(n)})
  return pd.DataFrame(rows)


def cost_table(rows):
  """Item d, cost: HT r1 against H r1, AT r2 against AT r1 (fu13.cost: wall time and compiled memory ratios)."""
  p = fu14.processes(rows)
  keep = (((p.condition == 'HT') | (p.condition == 'H')) & (p.repetition == 'r1')) | (p.condition == 'AT')
  p = p[keep].copy()
  p['condition'] = np.where(p.condition == 'AT', 'AT ' + p.repetition.astype(str), p.condition)
  return fu13.cost(p, [('HT', 'H'), ('AT r2', 'AT r1')])


# ---------------------------------------------------------------- main

def main(argv=None):
  ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
  ap.add_argument('--samples', required=True)
  ap.add_argument('--a100', default=str(REPO / 'results' / 'analysis' / 'a100_20261018'))
  ap.add_argument('--h100', default=str(REPO / 'results' / 'analysis' / 'h100_20261104'))
  ap.add_argument('--main', default=str(REPO / 'results' / 'analysis' / 'main_20261014' / 'primary_v2'))
  ap.add_argument('--rcx', default=str(REPO / 'results' / 'analysis' / 'rcx_20261115'))
  ap.add_argument('--gtat', default=None, help='the output folder of the post hoc analysis gtat (default: the newest '
                  'results/analysis/gtat_*)')
  ap.add_argument('--targets', default=str(REPO / 'targets' / 'posebusters_subset.csv'))
  ap.add_argument('--out', required=True)
  ap.add_argument('--pb_full', action='store_true', help='rerun M1 with PoseBusters\' full report (pb_flips) for the '
                  'main outputs whose success differs')
  ap.add_argument('--jobs', type=int, default=1)
  ap.add_argument('--replicates', type=int, default=10000)
  ap.add_argument('--seed', type=int, default=2026)
  args = ap.parse_args(argv)
  out = pathlib.Path(args.out)
  out.mkdir(parents=True, exist_ok=True)
  if args.pb_full and importlib.metadata.version('posebusters') != PB_VERSION:
    raise SystemExit(f'--pb_full needs posebusters {PB_VERSION}, the study\'s')
  if args.gtat is None:
    found = sorted((REPO / 'results' / 'analysis').glob('gtat_*'))
    args.gtat = str(found[-1]) if found else None
  tt = pd.read_csv(args.targets, dtype=str)
  all_targets = sorted(tt.pdb_id)

  d = load_tri(args.samples)
  chk = fu13.check_exit(d) + fu13.check_machine(d) + check_machine_ref(d)
  pd.DataFrame(chk).to_csv(out / 'checks.csv', index=False)
  if any(r['detail'] != 'pass' for r in chk):
    raise SystemExit('check 1 or 2 failed (checks.csv); nothing is compared')
  ref = load_references(args)
  okall = f34.sample_rows(pd.concat([d, ref], ignore_index=True))
  comp = completeness(okall, all_targets)
  s4, det4 = check_settings(d)
  s4b, stack = check_stack(d)
  chk += s4 + s4b + [check_vms(d)]
  ident, det_id = [], []
  for c in ('HT', 'AT'):
    s, det = identity(okall, c)
    ident.append(s)
    det_id += det
  pd.DataFrame(chk).to_csv(out / 'checks.csv', index=False)
  pd.DataFrame(det4 + det_id).to_csv(out / 'checks_detail.csv', index=False)

  sp, sm, cons, counts = f34.build(okall, SPECS)
  con = contrasts.summarize(sp, sm, cons, args) if len(sp) else pd.DataFrame()
  reg = registered_pairs(args)
  et = e_t_table(sp, reg, args)
  cmp_ = comparisons(sp, reg, args)
  dp = default_pairs(sp, reg, args)
  gt = gtat_replication(sp, con, args.gtat)
  units = mo_units(okall)
  mo = mo_summary(units, args)
  ch = mo_changes(units, args.pb_full, args.jobs)
  cc = change_counts(ch)
  succ = fu13.success_rates(okall, [('HT', 'r1'), ('H', 'r1'), ('A', 'r1'), ('AT', 'r1')], args.seed, args.replicates)
  cst = cost_table(pd.concat([d, ref], ignore_index=True))
  fails = fu14.processes(d)[fu14.processes(d).exit_code != 0][['condition', 'repetition', 'target', 'seed', 'session',
                                                               'exit_code', 'failure']]
  outputs = {'completeness': comp, 'contrasts': con, 'e_t': et, 'comparisons': cmp_, 'identity': pd.DataFrame(ident),
             'main_output_units': units.drop(columns=['_a', '_b']) if len(units) else units, 'main_output': mo,
             'main_output_changes': ch, 'main_output_counts': cc, 'default_pairs': dp, 'gtat_replication': gt,
             'success': succ, 'cost': cst, 'failures': fails, 'stack': stack, 'pairs_seed': sp, 'pairs_sample': sm,
             'counts': counts}
  for name, df in outputs.items():
    df.to_csv(out / f'{name}.csv', index=False)
  (out / 'report.md').write_text(report(chk, comp, con, et, cmp_, ident, mo, cc, dp, gt, succ, cst, fails, args))
  print((out / 'report.md').read_text())
  return 0


def report(chk, comp, con, et, cmp_, ident, mo, cc, dp, gt, succ, cst, fails, args):
  pct, num = f34.pct, f34.num
  L = ['# Follow-up 18: AlphaFold 3\'s default Triton attention on the H100 and the A100', '',
       'notes/analysis_plan.md section 10, row "Follow-up experiment 18" (label tri); reported whatever it shows.', '',
       '## Checks', '', '| check | condition | value | result |', '|---|---|---|---|']
  L += [f'| {r["check"]} | {r["condition"]} | {r["value"]} | {r["detail"]} |' for r in chk]
  inc = comp[comp.status != 'complete']
  L += ['', f'Completeness: {len(comp) - len(inc)} of {len(comp)} planned (condition, repetition, seed) cells complete'
        + ('' if inc.empty else '; not complete: ' + ', '.join(
            f'{r.condition} {r.repetition} seed {r.seed} ({r.targets_with_5_samples}/{r.planned_targets}, {r.status})'
            for r in inc.itertuples())) + '.']
  if len(fails):
    L += ['', 'Failed processes (recorded, not rerun): ' + '; '.join(
        f'{r.condition} {r.repetition} {r.target} seed {r.seed} ({r.failure})' for r in fails.itertuples()) + '.']
  L += ['', '## T1: E_T = HW(HT,AT) minus RC(HT) (rule e1; with part B, rule e3)', '',
        '| comparison | reference | statistic | role | estimate | 95% CI | targets | reading |',
        '|---|---|---|---|---|---|---|---|']
  L += [f'| {r.comparison} | {r.reference} | {r.statistic} | {r.role} | {pct(r.estimate)} | {pct(r.ci_low)} to '
        f'{pct(r.ci_high)} | {r.n_targets} | {r.reading} |' for r in et.itertuples()]
  top = et[et.statistic == 'flip_top1'].set_index('comparison')
  if len(top) == 2 and top.reading.iloc[1] not in ('not run (part B)',):
    a, b = top.reading.iloc[0], top.reading.iloc[1]
    L += ['', f'Top-1 against RC(HT): "{a}"; against the larger RC ({top.reference.iloc[1]}): "{b}"'
          + ('' if a == b else ': the two readings differ, both are stated') + '.']
  L += ['', '## T2: the main output (rule e2)', '',
        '| quantity | statistic | role | count | estimate | 95% CI | targets | reading |', '|---|---|---|---|---|---|---|---|']
  L += [f'| {r.quantity} | {r.statistic} | {r.role} | {"" if r.count != r.count else int(r.count)} | {pct(r.estimate)} | '
        f'{pct(r.ci_low)} to {pct(r.ci_high)} | {r.n_targets} | {r.reading} |' for r in mo.itertuples()]
  if len(cc):
    L += ['', 'Targets whose main output differs (counts, no threshold):', '', '| contrast | category | count |',
          '|---|---|---|']
    L += [f'| {r.contrast} | {r.category} | {r.count} |' for r in cc.itertuples()]
  if not args.pb_full:
    L += ['', 'PoseBusters checks\' excess not computed (run with --pb_full); the RMSD excess and the failing checks are '
          'in main_output_changes.csv.']
  L += ['', '## Recompilation and bit-identity (rule e4, descriptive)', '']
  L += [f'- {r["condition"]}: {r["value"]}; {r["detail"]}' for r in ident]
  if len(con):
    keep = con[con.statistic.isin(contrasts.FLIP_STATS + ['m4_ligand_rmsd_median', 'm5_ranking_score_median',
                                                          'bit_identical_fraction'])]
    L += ['', '| contrast | statistic | estimate | 95% CI | n | targets |', '|---|---|---|---|---|---|']
    for r in keep.itertuples():
      fmt = pct if r.statistic in contrasts.FLIP_STATS + ['bit_identical_fraction'] else num
      L.append(f'| {r.contrast} | {r.statistic} | {fmt(r.estimate)} | {fmt(r.ci_low)} to {fmt(r.ci_high)} | {r.n} | '
               f'{r.n_targets} |')
  L += ['', '| comparison (descriptive) | statistic | estimate | 95% CI | targets |', '|---|---|---|---|---|']
  L += [f'| {r.comparison} | {r.statistic} | {pct(r.estimate)} | {pct(r.ci_low)} to {pct(r.ci_high)} | {r.n_targets} |'
        for r in cmp_.itertuples()]
  L += ['', '## T3: default-configuration pairs among the GPUs (rule e5, descriptive)', '',
        '| group | pair | statistic | rate | 95% CI | minus | excess | 95% CI |', '|---|---|---|---|---|---|---|---|']
  L += [f'| {r.group} | {r.pair} | {r.statistic} | {pct(r.hw_estimate)} | {pct(r.hw_ci_low)} to {pct(r.hw_ci_high)} | '
        f'{r.reference} | {pct(r.estimate)} | {pct(r.ci_low)} to {pct(r.ci_high)} |' for r in dp.itertuples()]
  L += ['', f'HW(GT,AT), replication of the post hoc analysis gtat ({args.gtat or "no gtat folder"}): '
        + '; '.join(f'{r.what} {r.statistic}: {r.result}' for r in gt.itertuples()) + '. The gtat values stand.']
  L += ['', '| condition | top-1 success (per seed) | 95% CI | n |', '|---|---|---|---|']
  L += [f'| {r.condition} {r.repetition} | {pct(r.top1_success_rate)} | {pct(r.ci_low)} to {pct(r.ci_high)} | {r.n} |'
        for r in succ.itertuples()]
  L += ['', '## Cost (descriptive)', '',
        '| new / reference | bucket | wall ratio | compiled memory ratio | n |', '|---|---|---|---|---|']
  L += [f'| {r.new} / {r.reference} | {r.bucket} | {num(r.wall_ratio)} | {num(r.compiled_memory_ratio)} | '
        f'{r.n_new}, {r.n_reference} |' for r in cst.itertuples()]
  return '\n'.join(L) + '\n'


if __name__ == '__main__':
  sys.exit(main())
