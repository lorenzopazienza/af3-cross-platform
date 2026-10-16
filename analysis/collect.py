"""Builds the tidy per-sample table of all AlphaFold3 runs (notes/analysis_plan.md, sections 4 and 6).

    analysis/.venv/bin/python analysis/collect.py [--root results/af3] [--root COPY_OF_BUCKET/af3] \\
        [--out results/analysis/<name>] [--labels pilot2] [--jobs 4] [--no_m1]

Walks every harness run folder (one with session.json and runs.jsonl) under
the roots: results/af3/ by default, or a local copy of the results bucket
(gcloud storage rsync gs://BUCKET/af3 DIR; this script reads local files
only); with --labels, only the harness runs whose session.json label is one
of them (the repeated pilot: --labels pilot2). One row per (platform,
attention, libtpu, target, seed, repetition, sample), with:
  - condition: platform code (V5, V6, G, GT, C; plan section 3), attention,
    libtpu version and build label, device kind, zone and machine type, and
    the default matmul precision (JAX_DEFAULT_MATMUL_PRECISION as recorded
    in session.json xla_env, empty if unset). The precision-matched arms of
    plan section 11 (configs tpu_xla_highest and l4_xla_highest) set it to
    highest and get the code with an "h": V6h, Gh. Any other value is kept
    apart as CODE[value], never merged with the default arm. The run config
    (session.json config_name) is kept too; config l4_xla_det, the L4 with
    --xla_gpu_deterministic_ops=true, gets the code Gd (the secondary arm of
    the 2026-10-11 amendment, plan section 10);
  - repetition: r1, r2, ... (fresh-cache repetitions) or w (warm rerun), from
    the run name harness/run_plan.py gives; num_recycles and
    num_diffusion_samples (AlphaFold3 defaults 10 and 5 when not set);
  - per sample: ranking_score at full precision, as AlphaFold3 ranks the
    samples (<target>_ranking_scores.csv in the same output folder; empty
    if that file is missing), ranking_score_summary (the same score as
    *_summary_confidences.json writes it, rounded to 2 decimals; plan
    section 10, 2026-10-12), ptm, iptm, has_clash, fraction_disordered
    (summary confidences; AlphaFold3 writes them at 2 decimals), mean pLDDT (mean of the per-atom pLDDT AlphaFold3
    writes in the mmCIF B-factor column), the model path, and M1 (ligand RMSD,
    PoseBusters validity, success, error; analysis/score_ligand.py);
  - per run: the persistent cache evidence of harness/run_af3.py (files in the
    cache before the process, whether jit_apply_fn was loaded from it and
    with which key, the jit_apply_fn entries the process wrote; keys are
    recorded from 2026-10-11 on);
  - per run: exit code, failure, wall, inference and compile seconds, compiled
    memory (peak, temp, arguments, needed), allocator peak reservation and
    peak in use, the device memory limit, max RSS;
  - weights file and SHA-256, input SHA-256 (frozen and derived), session id.
A run that failed (no model files) still gets one row per seed, with the
failure and empty sample columns (plan section 8: never dropped silently).

Duplicate sessions (notes/analysis_plan.md section 10, row of 2026-10-13): sessions named
<UTC stamp>_<platform>_<plan> belong to one part (platform, plan). If a part has more than one fully exit-0
session (its job finished with exit code 0 in final_status.json, and every process in its runs.jsonl exited
0), only the one that completed first (final_status.json finished_utc) is kept, and every other session of that
part is excluded and listed in excluded_sessions.csv; the choice reads no output. A part with at most one fully
exit-0 session is left as it is.

Outputs in --out (default results/analysis/collect, gitignored like all of
results/, so nothing produced with official weights reaches git):
samples.csv, excluded_sessions.csv and m1_cache.jsonl (M1 by model-file SHA-256, reused on reruns).
"""

import argparse
import csv
import hashlib
import json
import pathlib
import re
import sys
from concurrent.futures import ProcessPoolExecutor

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / 'analysis'))

PLATFORM_BY_KIND = {'TPU v5 lite': 'V5', 'TPU v6 lite': 'V6', 'cpu': 'C'}
# Arms set by the run config rather than the device: Gd, the L4 with deterministic ops (plan section 10, 2026-10-11).
CODE_BY_CONFIG = {'l4_xla_det': 'Gd'}
COLUMNS = [
    'session', 'run', 'label', 'platform', 'config', 'attention', 'matmul_precision', 'device_kind', 'libtpu',
    'libtpu_build', 'jax',
    'zone', 'machine_type', 'target', 'num_tokens', 'bucket', 'seed', 'repetition', 'cache', 'sample',
    'num_recycles', 'num_diffusion_samples', 'exit_code', 'failure', 'ranking_score', 'ranking_score_summary', 'ptm', 'iptm',
    'mean_plddt', 'has_clash', 'fraction_disordered', 'm1_ligand_rmsd', 'm1_pb_valid', 'm1_success',
    'm1_pb_failed', 'm1_error', 'model_path', 'model_sha256', 'wall_s', 'inference_s', 'compile_s',
    'compiled_peak_bytes', 'compiled_temp_bytes', 'compiled_args_bytes', 'compiled_needed_bytes',
    'alloc_peak_reserved_bytes', 'alloc_peak_in_use_bytes', 'device_bytes_limit', 'max_rss_bytes', 'weights_file',
    'weights_sha256', 'input_sha256', 'run_input_sha256', 'cache_files_before', 'model_cache_hit',
    'model_cache_key', 'model_cache_written',
]


def sha256(path):
  h = hashlib.sha256()
  with open(path, 'rb') as f:
    for block in iter(lambda: f.read(1 << 20), b''):
      h.update(block)
  return h.hexdigest()


def platform_of(kind, attention, matmul_precision=None, config_name=None):
  """Condition code: section 3's platform code, Gd for config l4_xla_det, "h" for the section 11 arms (highest);
  follow-up 3 (section 10, 2026-10-16): A and AT for the A100 (xla and triton attention), Ah with highest."""
  if config_name in CODE_BY_CONFIG:
    code = CODE_BY_CONFIG[config_name]
  elif 'L4' in kind:
    code = 'GT' if attention == 'triton' else 'G'
  elif 'A100' in kind:
    code = 'AT' if attention == 'triton' else 'A'
  else:
    code = PLATFORM_BY_KIND.get(kind, kind)
  if not matmul_precision:
    return code
  # No "_" in the code: contrasts.py --condition platform_libtpu splits the condition at the first "_".
  return f'{code}h' if matmul_precision == 'highest' else f'{code}[{matmul_precision.replace("_", "-")}]'


def repetition_of(run_name, cache_mode):
  m = re.search(r'_rep(\d+)_seed\d+_fresh$', run_name)
  if m:
    return f'r{m.group(1)}'
  m = re.search(r'_seed\d+_warm(\d*)$', run_name)   # _warm2, _warm3: further warm reruns (diagnostics)
  if m:
    return f'w{m.group(1)}'
  if cache_mode == 'warm':
    return 'w'
  return 'r1'


def ranking_scores(outdir, target):
  """{(seed, sample): full-precision ranking_score} from AlphaFold3's <target>_ranking_scores.csv ({} if absent)."""
  f = pathlib.Path(outdir) / f'{target}_ranking_scores.csv'
  if not f.is_file():
    return {}
  return {(int(r['seed']), int(r['sample'])): float(r['ranking_score']) for r in csv.DictReader(open(f))}


def mean_plddt(cif):
  """Mean of the B-factor column (AlphaFold3 writes per-atom pLDDT there) of the mmCIF atom_site loop."""
  lines = pathlib.Path(cif).read_text().splitlines()
  i = next(n for n, l in enumerate(lines) if l.startswith('_atom_site.'))
  fields = []
  while lines[i].startswith('_atom_site.'):
    fields.append(lines[i].split('.', 1)[1].strip())
    i += 1
  col = fields.index('B_iso_or_equiv')
  vals = []
  while i < len(lines) and lines[i].strip() and not lines[i].startswith(('#', 'loop_', '_')):
    vals.append(float(lines[i].split()[col]))
    i += 1
  return round(sum(vals) / len(vals), 4) if vals else None


def run_folders(roots, labels=None):
  for root in roots:
    for ses in sorted(pathlib.Path(root).rglob('session.json')):
      if (ses.parent / 'runs.jsonl').exists():
        if labels is None or json.loads(ses.read_text()).get('label') in labels:
          yield ses.parent


SESSION_NAME = re.compile(r'^\d{8}T\d{6}Z_(cpu|l4|v5e|v6e)_(.+)$')


def session_status(session_dir):
  """(fully exit-0, completion time) of a session folder, from final_status.json and its runs' runs.jsonl."""
  status = session_dir / 'final_status.json'
  final = json.loads(status.read_text()) if status.is_file() else {}
  codes = [json.loads(l).get('exit_code') for r in session_dir.glob('*/runs.jsonl') for l in r.read_text().splitlines()]
  ok = final.get('exit_code') == 0 and bool(codes) and all(c == 0 for c in codes)
  return ok, final.get('finished_utc') or ''


def resolve_duplicate_sessions(folders):
  """(kept run folders, excluded rows): the section 10 rule of 2026-10-13 for parts with several exit-0 sessions."""
  folders = list(folders)
  parts = {}
  for f in folders:
    m = SESSION_NAME.match(f.parent.name)
    if m:
      parts.setdefault((m.group(1), m.group(2)), set()).add(f.parent)
  excluded, drop = [], set()
  for (platform, plan), sessions in sorted(parts.items()):
    status = {s: session_status(s) for s in sessions}
    full = sorted((done, s.name, s) for s, (ok, done) in status.items() if ok)
    if len(full) < 2:
      continue
    keep = full[0][2]
    for s in sorted(sessions, key=lambda x: x.name):
      if s != keep:
        drop.add(s)
        excluded.append({'part': f'{platform} {plan}', 'session': s.name, 'kept_session': keep.name,
                         'fully_exit_0': status[s][0], 'completed_utc': status[s][1],
                         'kept_completed_utc': status[keep][1],
                         'reason': 'part has more than one fully exit-0 session; the first completed is kept'})
  return [f for f in folders if f.parent not in drop], excluded


def _m1(args):
  target, path = args
  import score_ligand
  return score_ligand.score(target, path)


def rows_of_run(folder, manifest):
  session = json.loads((folder / 'session.json').read_text())
  probe = session.get('probe') or {}
  cfg = session.get('config') or {}
  sargs = session.get('args') or {}
  attention = cfg.get('flash_attention_implementation')
  kind = probe.get('device_kind', '')
  # What the run's environment held (run_af3.py records XLA_, JAX_ and TF_ variables).
  precision = (session.get('xla_env') or {}).get('JAX_DEFAULT_MATMUL_PRECISION') or None
  tpu = session.get('tpu_runtime') or {}
  host = session.get('host') or {}
  base = {
      'session': str(folder.relative_to(REPO)) if folder.resolve().is_relative_to(REPO) else folder.name,
      'run': folder.name, 'label': session.get('label'),
      'platform': platform_of(kind, attention, precision, session.get('config_name')),
      'config': session.get('config_name'), 'attention': attention, 'matmul_precision': precision, 'device_kind': kind,
      'libtpu': (probe.get('packages') or {}).get('libtpu'),
      'libtpu_build': tpu.get('build_label'), 'jax': (probe.get('packages') or {}).get('jax'),
      'zone': host.get('gce_zone'), 'machine_type': host.get('gce_machine_type') or host.get('gce_accelerator_type'),
      'num_recycles': sargs.get('num_recycles') or 10,
      'num_diffusion_samples': sargs.get('num_diffusion_samples') or 5,
      'cache': (session.get('compile_cache') or {}).get('mode'),
      'weights_file': (session.get('weights') or {}).get('file'),
      'weights_sha256': (session.get('weights') or {}).get('sha256'),
  }
  base['repetition'] = repetition_of(folder.name, base['cache'])
  out, todo = [], []
  for line in (folder / 'runs.jsonl').read_text().splitlines():
    r = json.loads(line)
    t = r['pdb_id']
    m = r.get('memory') or {}
    stats = r.get('entry_stats') or {}
    compiled = [c for c in (stats.get('compiled_memory') or []) if 'error' not in c]
    c = compiled[-1] if compiled else {}
    dev = stats.get('device_memory_stats') or {}
    run_cols = dict(base, target=t, num_tokens=r.get('num_tokens'), bucket=r.get('bucket'),
                    exit_code=r.get('exit_code'), failure=r.get('failure'), wall_s=r.get('wall_seconds'),
                    compile_s=(r.get('compile') or {}).get('model_seconds'),
                    cache_files_before=((r.get('compile_cache') or {}).get('before') or {}).get('files'),
                    model_cache_hit=(r.get('compile') or {}).get('model_cache_hit'),
                    model_cache_key=(r.get('compile') or {}).get('model_cache_key'),
                    model_cache_written=';'.join((r.get('compile_cache') or {}).get('model_entries_written') or [])
                    or None,
                    compiled_peak_bytes=c.get('peak_memory_in_bytes'), compiled_temp_bytes=c.get('temp_size_in_bytes'),
                    compiled_args_bytes=c.get('argument_size_in_bytes'),
                    compiled_needed_bytes=c.get('device_bytes_needed'),
                    alloc_peak_reserved_bytes=dev.get('peak_bytes_reserved', m.get('peak_bytes_reserved')),
                    alloc_peak_in_use_bytes=dev.get('peak_bytes_in_use', m.get('peak_bytes_in_use')),
                    device_bytes_limit=dev.get('bytes_limit', m.get('bytes_limit')),
                    max_rss_bytes=stats.get('max_rss_bytes', m.get('max_rss_bytes')),
                    input_sha256=r.get('input_sha256'), run_input_sha256=r.get('run_input_sha256'))
    outdir = REPO / r['output_dir'] / t if not pathlib.Path(r['output_dir']).is_absolute() else pathlib.Path(r['output_dir']) / t
    if not outdir.exists():
      outdir = folder / t / 'af3_output' / t
    full = ranking_scores(outdir, t) if outdir.exists() else {}
    for seed in r.get('seeds') or []:
      inf = (r.get('inference_seconds_by_seed') or {}).get(str(seed))
      samples = sorted(outdir.glob(f'seed-{seed}_sample-*')) if outdir.exists() else []
      if not samples:
        out.append(dict(run_cols, seed=seed, inference_s=inf, sample=None))
        continue
      for d in samples:
        k = int(d.name.rsplit('-', 1)[1])
        row = dict(run_cols, seed=seed, inference_s=inf, sample=k)
        summ = next((p for p in d.glob('*_summary_confidences.json')), None)
        if summ:
          s = json.loads(summ.read_text())
          row.update(ranking_score=full.get((int(seed), k)), ranking_score_summary=s.get('ranking_score'),
                     ptm=s.get('ptm'), iptm=s.get('iptm'),
                     has_clash=s.get('has_clash'), fraction_disordered=s.get('fraction_disordered'))
        cif = next((p for p in d.glob('*_model.cif')), None)
        if cif:
          row['model_path'] = str(cif.resolve().relative_to(REPO)) if cif.resolve().is_relative_to(REPO) else str(cif)
          row['model_sha256'] = sha256(cif)
          row['mean_plddt'] = mean_plddt(cif)
          todo.append((len(out), t, cif))
        out.append(row)
  return out, todo


def main():
  ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
  ap.add_argument('--root', action='append', help='folder holding sessions (default results/af3)')
  ap.add_argument('--out', default='results/analysis/collect')
  ap.add_argument('--labels', help='comma-separated session labels to keep (default: all)')
  ap.add_argument('--jobs', type=int, default=4)
  ap.add_argument('--no_m1', action='store_true', help='skip M1 (quick look)')
  args = ap.parse_args()
  roots = [REPO / r if not pathlib.Path(r).is_absolute() else pathlib.Path(r) for r in (args.root or ['results/af3'])]
  out_dir = REPO / args.out if not pathlib.Path(args.out).is_absolute() else pathlib.Path(args.out)
  out_dir.mkdir(parents=True, exist_ok=True)
  manifest = {r['pdb_id']: r for r in csv.DictReader(open(REPO / 'inputs' / 'manifest.csv'))}

  rows, todo = [], []
  folders, excluded = resolve_duplicate_sessions(run_folders(roots, args.labels.split(',') if args.labels else None))
  with open(out_dir / 'excluded_sessions.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=['part', 'session', 'kept_session', 'fully_exit_0', 'completed_utc',
                                      'kept_completed_utc', 'reason'])
    w.writeheader()
    w.writerows(excluded)
  for e in excluded:
    print(f'>> excluded session {e["session"]} ({e["part"]}): kept {e["kept_session"]}', flush=True)
  for folder in folders:
    r, t = rows_of_run(folder, manifest)
    todo += [(len(rows) + i, target, cif) for i, target, cif in t]
    rows += r
  print(f'>> {len(rows)} rows from {len({r["session"] for r in rows})} harness runs; '
        f'{len(todo)} models to score for M1', flush=True)

  if not args.no_m1 and todo:
    cache_path = out_dir / 'm1_cache.jsonl'
    cache = {}
    if cache_path.exists():
      for line in cache_path.read_text().splitlines():
        c = json.loads(line)
        cache[(c['model_sha256'], c['target'])] = c
    need = [(i, t, cif) for i, t, cif in todo if (rows[i]['model_sha256'], t) not in cache]
    if need:
      with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        results = list(pool.map(_m1, [(t, str(cif)) for _, t, cif in need], chunksize=4))
      with open(cache_path, 'a') as f:
        for (i, t, _), res in zip(need, results):
          res['model_sha256'] = rows[i]['model_sha256']
          cache[(res['model_sha256'], t)] = res
          f.write(json.dumps(res) + '\n')
    for i, t, _ in todo:
      res = cache[(rows[i]['model_sha256'], t)]
      rows[i].update(m1_ligand_rmsd=res['ligand_rmsd'], m1_pb_valid=res['pb_valid'], m1_success=res['success'],
                     m1_pb_failed=';'.join(res['pb_failed'] or []), m1_error=res['error'])
    print(f'>> M1: {len(todo) - len(need)} from the cache, {len(need)} scored now', flush=True)

  with open(out_dir / 'samples.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction='ignore')
    w.writeheader()
    w.writerows(rows)
  print(f'>> wrote {(out_dir / "samples.csv").relative_to(REPO) if out_dir.is_relative_to(REPO) else out_dir}', flush=True)
  return 0


if __name__ == '__main__':
  sys.exit(main())
