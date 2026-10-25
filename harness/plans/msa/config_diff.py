"""Follow-up 6 (label msa): every setting of each new arm against its main-study arm (notes/analysis_plan.md section 10,
row of 2026-10-19: "the settings of V5, V6 and G and only the input changed").

    python3 harness/plans/msa/config_diff.py [--out harness/plans/msa/config_diff.md]

Main-study side, as recorded: every session of the arm in results/analysis/main_20261014/sessions_main.txt
(launch_record.txt; session.json of every harness run: config, environment, AlphaFold3 commit, patch and
run_alphafold.py SHA-256, libtpu build, buckets, recycles, samples; runs.jsonl: the run_alphafold command with paths
masked), and the image the launcher used at the session's commit (cloud/af3_run.sh at that commit). A setting that
differs between the arm's own sessions is shown with every value.
Follow-up 6 side, as it will run: the arm's plans (harness/run_plan.py's description), harness/configs.yaml, the
environment and command harness/run_af3.py builds from them (the same rules: the config's env, JAX_LOG_COMPILES=1
with --log_compiles, the cache variable with a fresh cache), cloud/af3_run.sh's defaults for the platform (machine
type, image, libtpu), af3_tpu/pin.sh and af3_tpu/af3_tpu.patch, with the launch settings of the commands in the
section 10 implementation row (on demand: SPOT=0, v6e PROVISIONING=standard).
The only intended difference is the input (manifest and label). Any other difference is printed as DIFFERS and the
script exits 1.
"""
import argparse
import hashlib
import json
import pathlib
import re
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / 'harness'))
from run_af3 import CACHE_ENV, read_configs  # noqa: E402

PAIRS = [('V5', 'V5o', 'v5e', 'v5'), ('V6', 'V6o', 'v6e', 'v6'), ('G', 'Go', 'l4', 'g')]
# The input and the label; and the repetitions, which the row sets (RC(V5o) and RC(V6o) are not run: r1 and w only).
INTENDED = {'input manifest', 'label', 'repetitions of the arm (r = fresh, w = warm rerun)'}


def rep_tag(run_name):
  m = re.search(r'_rep(\d+)_seed\d+_fresh$', run_name)
  return f'r{m.group(1)}' if m else 'w'
PATHS = re.compile(r'(\.\./\.\./results/af3/)[^ ]+?/([A-Za-z0-9]{4})/')


def sha256(path):
  return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def launcher_defaults(text, platform):
  """Machine type, image (or TPU runtime) and libtpu default of cloud/af3_run.sh text for a platform."""
  block = re.search(rf'\n  {platform}\)\n(.*?);;', text, re.S).group(1)
  get = lambda k: (re.search(rf'{k}="\$\{{{k}:-([^}}]*)\}}"', block) or [None, None])[1]  # noqa: E731
  libtpu = re.search(r'LIBTPU_VERSION="\$\{LIBTPU_VERSION:-([0-9.]+)\}"', text)
  return {'machine type': get('MACHINE_TYPE'),
          'image': f'{get("IMAGE_PROJECT")}/{get("IMAGE_FAMILY")}' if get('IMAGE_FAMILY') else f'TPU runtime {get("RUNTIME")}',
          'libtpu default': libtpu.group(1) if libtpu and platform in ('v5e', 'v6e') else None}


def masked_command(cmd):
  """The run_alphafold command with the session, target and paths replaced by placeholders."""
  c = PATHS.sub(r'\1<session>/<target>/', cmd)
  c = re.sub(r'AF3_ENTRY_STATS=\S+', 'AF3_ENTRY_STATS=<target>/af3_entry_stats.json', c)
  c = re.sub(r'--json_path=\S+', '--json_path=<target>/work/input_seeds.json', c)
  c = re.sub(r'--output_dir=\S+', '--output_dir=<target>/af3_output', c)
  c = re.sub(r'--jax_compilation_cache_dir=\S+', '--jax_compilation_cache_dir=<cache>', c)
  return c


def main_side(arm, platform, code):
  names = (REPO / 'results' / 'analysis' / 'main_20261014' / 'sessions_main.txt').read_text().split()
  sessions = [s for s in names if re.search(rf'_{platform}_main_{code}_\d+$', s)]
  vals = {}

  def add(k, v):
    vals.setdefault(k, set()).add(json.dumps(v, sort_keys=True) if not isinstance(v, str) else v)
  tags = set()
  for s in sessions:
    root = REPO / 'results' / 'af3' / s
    tags |= {rep_tag(r.name) for r in root.iterdir() if (r / 'session.json').is_file()}
    rec = {}
    for line in (root / 'launch_record.txt').read_text().splitlines():
      k, _, v = line.partition('=')
      rec[k] = v
    add('platform', rec['PLATFORM'])
    add('provisioning', rec['PROVISIONING'])
    add('weights', rec['WEIGHTS'])
    text = subprocess.run(['git', 'show', f'{rec["COMMIT"]}:cloud/af3_run.sh'], capture_output=True, text=True, cwd=REPO).stdout
    d = launcher_defaults(text, platform)
    add('machine type', rec['MACHINE_TYPE'])
    add('image', d['image'])
    if platform in ('v5e', 'v6e'):
      add('libtpu', rec['LIBTPU_VERSION'])
    for sj in sorted(root.glob('*/session.json')):
      j = json.loads(sj.read_text())
      if '_warm' in sj.parent.name:
        continue
      add('label', j['label'])
      add('config name', j['config_name'])
      add('config (harness/configs.yaml entry)', j['config'])
      add('environment (xla_env)', j['xla_env'])
      add('AlphaFold3 commit', j['alphafold3']['commit'])
      add('af3_tpu.patch SHA-256', j['alphafold3']['patch_sha256'])
      add('run_alphafold.py SHA-256', j['alphafold3']['run_alphafold_sha256'])
      add('buckets', j.get('af3_buckets'))
      add('save_embeddings', bool(j.get('save_embeddings')))
      a = j['args']
      add('recycles, diffusion samples', [a['num_recycles'], a['num_diffusion_samples']])
      add('per-process timeout (min), entry, log_compiles, order', [a['timeout_min'], a['entry'], a['log_compiles'],
                                                                   a['order']])
      add('input manifest', a['manifest'])
      add('seeds per process', len(a['seeds'].split(',')))
      if j.get('tpu_runtime'):
        add('libtpu build', j['tpu_runtime']['build_label'])
      r = json.loads((sj.parent / 'runs.jsonl').read_text().splitlines()[0])
      add('run_alphafold command', masked_command(r['command']))
  vals['repetitions of the arm (r = fresh, w = warm rerun)'] = {', '.join(sorted(tags))}
  return {k: sorted(v) for k, v in vals.items()}, len(sessions)


def msa_side(arm, platform):
  import run_plan
  plans = sorted((REPO / 'harness' / 'plans' / 'msa').glob(f'msa_{arm.lower()}*.yaml'))
  head = (REPO / 'cloud' / 'af3_run.sh').read_text()
  d = launcher_defaults(head, platform)
  configs = read_configs(REPO / 'harness' / 'configs.yaml')
  vals = {}

  def add(k, v):
    vals.setdefault(k, set()).add(json.dumps(v, sort_keys=True) if not isinstance(v, str) else v)
  pin = dict(re.findall(r'export (\w+)="([^"]+)"', (REPO / 'af3_tpu' / 'pin.sh').read_text()))
  tags = set()
  for p in plans:
    plan = run_plan.read_configs(p)
    if (REPO / 'inputs' / 'official_data' / 'manifest.csv').is_file():
      tags |= {rep_tag(r['run']) for r in run_plan.expand(p, platform, REPO / 'inputs' / 'manifest.csv')['runs']}
    else:   # before the ingest (the plans' manifest does not exist yet): the plan's own keys
      first = int(plan.get('rep_start', 1))
      tags |= {f'r{r}' for r in range(first, first + int(plan['fresh_reps']))}
      tags |= {'w'} if plan.get('warm_rerun_seed', 'none') != 'none' else set()
    config = plan['configs'][platform]
    cfg = dict(configs[config])
    add('platform', platform)
    add('provisioning', 'standard')
    add('weights', plan['weights'])
    add('machine type', d['machine type'])
    add('image', d['image'])
    if d['libtpu default']:
      add('libtpu', d['libtpu default'])
    add('label', plan['label'])
    add('config name', config)
    add('config (harness/configs.yaml entry)', cfg)
    env = dict(cfg.get('env', {}), JAX_LOG_COMPILES='1', **CACHE_ENV)
    add('environment (xla_env)', {k: str(v) for k, v in env.items() if k.startswith(('XLA_', 'JAX_', 'TF_'))})
    add('AlphaFold3 commit', pin['AF3_COMMIT'])
    add('af3_tpu.patch SHA-256', sha256(REPO / 'af3_tpu' / 'af3_tpu.patch'))
    add('run_alphafold.py SHA-256', sha256(REPO / 'third_party' / 'alphafold3' / 'run_alphafold.py'))
    add('buckets', plan.get('buckets'))
    add('save_embeddings', str(plan.get('save_embeddings', 'false')).lower() == 'true')
    add('recycles, diffusion samples', [None if plan['num_recycles'] == 'default' else int(plan['num_recycles']),
                                        None if plan['num_diffusion_samples'] == 'default' else int(plan['num_diffusion_samples'])])
    add('per-process timeout (min), entry, log_compiles, order', [float(plan['run_timeout_min']), True, True, 'tokens'])
    add('input manifest', plan['manifest'])
    add('seeds per process', 1)
    if platform in ('v5e', 'v6e'):
      add('libtpu build', 'libtpu_lts_20260630_b_RC06' if d['libtpu default'] == '0.0.43.2' else f'libtpu {d["libtpu default"]}')
    shown = ' '.join(f'{k}={v}' for k, v in dict(cfg.get('env', {}), JAX_LOG_COMPILES='1', **CACHE_ENV).items())
    add('run_alphafold command', masked_command(
        f'cd third_party/alphafold3 && {shown} AF3_ENTRY_STATS=x .venv/bin/python ../../harness/af3_entry.py '
        f'--json_path=x --output_dir=x \'--model_dir=<model_dir>\' --jax_backend={cfg["jax_backend"]} '
        f'--flash_attention_implementation={cfg["flash_attention_implementation"]} --run_data_pipeline=false '
        f'--jax_compilation_cache_dir=x'))
  vals['repetitions of the arm (r = fresh, w = warm rerun)'] = {', '.join(sorted(tags))}
  return {k: sorted(v) for k, v in vals.items()}, len(plans)


def main():
  ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
  ap.add_argument('--out', default=str(REPO / 'harness' / 'plans' / 'msa' / 'config_diff.md'))
  args = ap.parse_args()
  lines = ['# Follow-up 6: settings of each arm against its main-study arm (harness/plans/msa/config_diff.py)', '',
           'Main study: as recorded in its sessions. Follow-up 6: as its plans, harness/configs.yaml, harness/run_af3.py '
           'and cloud/af3_run.sh resolve them, launched on demand. The input manifest and the label are the intended '
           'differences.', '']
  bad = 0
  for main_arm, new_arm, platform, code in PAIRS:
    m, n_main = main_side(main_arm, platform, code)
    o, n_plans = msa_side('go' if new_arm == 'Go' else new_arm, platform)
    lines += [f'## {new_arm} against {main_arm} ({platform}; {n_main} main-study sessions, {n_plans} follow-up 6 plans)', '',
              '| setting | main study (recorded) | follow-up 6 (planned) | |', '|---|---|---|---|']
    for k in m:
      same = m[k] == o.get(k)
      mark = 'same' if same else ('intended' if k in INTENDED else 'DIFFERS')
      bad += mark == 'DIFFERS'
      fmt = lambda v: '<br>'.join(f'`{x}`' for x in v) if v else '(none)'  # noqa: E731
      lines.append(f'| {k} | {fmt(m[k])} | {fmt(o.get(k))} | {mark} |')
    lines.append('')
  lines.append(f'Settings that differ beyond the input: {bad}')
  pathlib.Path(args.out).write_text('\n'.join(lines) + '\n')
  print('\n'.join(lines))
  return 1 if bad else 0


if __name__ == '__main__':
  sys.exit(main())
