"""Follow-ups 7 to 9: every setting of each new arm against the arm it derives from, as harness/plans/msa/config_diff.py
does for follow-up 6 ("the same settings as the source arm; only the factor under study changes").

    python3 harness/plans/config_diff_fu7_10.py [--out harness/plans/config_diff_fu7_10.md]

Source side, as recorded in its sessions: V5, V6 and G in the main study (results/analysis/main_20261014/
sessions_main.txt), A in follow-up 3 (scripts/analysis_runs/a100/sessions.txt, the A parts): launch_record.txt, the
image of cloud/af3_run.sh at the session's commit, and every fresh harness run's session.json (config, environment,
packages, AlphaFold3 commit, patch and run_alphafold.py SHA-256, libtpu build, buckets, recycles, samples) and
run_alphafold command (paths masked). New side, as it will run: the arm's plans (harness/run_plan.py), harness/
configs.yaml, the environment and command harness/run_af3.py builds, cloud/af3_run.sh's defaults, af3_tpu/pin.sh,
af3_tpu/af3_tpu.patch, with the provisioning decided on 2026-10-27 (make_fu7_10_plans.PLANNED: Spot on v5e, L4 and
H100, on demand on v6e; the source arms ran on demand). The factor under study, the label, the repetitions and, where
it is Spot, the provisioning (how the VM is obtained, not the computation) are the intended differences of each arm;
any other difference is DIFFERS and the script exits 1.
"""
import argparse
import json
import pathlib
import re
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'harness'))
sys.path.insert(0, str(REPO / 'harness' / 'plans' / 'msa'))
import run_plan  # noqa: E402
from config_diff import masked_command, rep_tag, sha256  # noqa: E402
from run_af3 import CACHE_ENV, read_configs  # noqa: E402

sys.path.insert(0, str(REPO / 'harness' / 'plans'))
from make_fu7_10_plans import PLANNED  # noqa: E402

REPS = 'repetitions of the arm (r = fresh, w = warm rerun)'
STACK = 'software stack (jax, jaxlib, libtpu or jax-cuda12 plugin and pjrt)'
SEEDS = 'seeds of the arm'
PKGS = ('jax', 'jaxlib', 'libtpu', 'jax-cuda12-plugin', 'jax-cuda12-pjrt')
# new arm: (plans glob, platform, source arm, intended differences)
ARMS = {
    'V5x': ('seeds/seeds_v5x_*.yaml', 'v5e', 'V5', {'label', SEEDS, REPS, 'provisioning'}),
    'Gx': ('seeds/seeds_gx_*.yaml', 'l4', 'G', {'label', SEEDS, REPS, 'provisioning'}),
    'V6s': ('sw/sw_v6s_*.yaml', 'v6e', 'V6', {'label', STACK, 'libtpu', 'libtpu build', REPS}),
    'Gs': ('sw/sw_gs_*.yaml', 'l4', 'G', {'label', STACK, REPS, 'provisioning'}),
    'H': ('h100/h100_*.yaml', 'h100', 'A', {'label', 'platform', 'machine type', 'config name',
                                           'config description', REPS, 'provisioning'}),
}


def launcher_defaults(text, platform):
  """As msa/config_diff.launcher_defaults, with the platform's block taken up to the next platform (the a100 and h100
  blocks hold a zone case whose ';;' ends the block there)."""
  block = re.search(rf'\n  {platform}\)\n(.*?)\n  (?:[a-z0-9]+|\*)\)[ \n]', text, re.S).group(1)
  get = lambda k: (re.search(rf'{k}="\$\{{{k}:-([^}}]*)\}}"', block) or [None, None])[1]  # noqa: E731
  libtpu = re.search(r'LIBTPU_VERSION="\$\{LIBTPU_VERSION:-([0-9.]+)\}"', text)
  return {'machine type': get('MACHINE_TYPE'),
          'image': f'{get("IMAGE_PROJECT")}/{get("IMAGE_FAMILY")}' if get('IMAGE_FAMILY') else f'TPU runtime {get("RUNTIME")}',
          'libtpu default': libtpu.group(1) if libtpu and platform in ('v5e', 'v6e') else None}


def source_sessions(arm):
  if arm == 'A':
    return [l.split('#')[0].strip() for l in (REPO / 'scripts' / 'analysis_runs' / 'a100' / 'sessions.txt').read_text().splitlines()
            if re.search(r'_a100_a100_a_\d', l.split('#')[0])]
  code = {'V5': ('v5e', 'v5'), 'V6': ('v6e', 'v6'), 'G': ('l4', 'g')}[arm]
  return [s for s in (REPO / 'results' / 'analysis' / 'main_20261014' / 'sessions_main.txt').read_text().split()
          if re.search(rf'_{code[0]}_main_{code[1]}_\d+$', s)]


def recorded(arm, platform):
  vals, tags, seeds = {}, set(), set()

  def add(k, v):
    vals.setdefault(k, set()).add(json.dumps(v, sort_keys=True) if not isinstance(v, str) else v)
  sessions = source_sessions(arm)
  for s in sessions:
    root = REPO / 'results' / 'af3' / s
    tags |= {rep_tag(r.name) for r in root.iterdir() if (r / 'session.json').is_file()}
    rec = dict(l.partition('=')[::2] for l in (root / 'launch_record.txt').read_text().splitlines())
    text = subprocess.run(['git', 'show', f'{rec["COMMIT"]}:cloud/af3_run.sh'], capture_output=True, text=True, cwd=REPO).stdout
    add('platform', rec['PLATFORM'])
    add('provisioning', rec['PROVISIONING'])
    add('weights', rec['WEIGHTS'])
    add('machine type', rec['MACHINE_TYPE'])
    add('image', launcher_defaults(text, rec['PLATFORM'])['image'])
    if platform in ('v5e', 'v6e'):
      add('libtpu', rec['LIBTPU_VERSION'])
    for sj in sorted(root.glob('*/session.json')):
      if '_warm' in sj.parent.name:
        continue
      j = json.loads(sj.read_text())
      cfg = dict(j['config'])
      add('label', j['label'])
      add('config name', j['config_name'])
      add('config description', cfg.pop('description', None))
      add('config (harness/configs.yaml entry, without its description)', cfg)
      add('environment (xla_env)', j['xla_env'])
      pk = (j.get('probe') or {}).get('packages') or {}
      add(STACK, {k: pk.get(k) for k in PKGS if pk.get(k)})
      add('AlphaFold3 commit', j['alphafold3']['commit'])
      add('af3_tpu.patch SHA-256', j['alphafold3']['patch_sha256'])
      add('run_alphafold.py SHA-256', j['alphafold3']['run_alphafold_sha256'])
      add('buckets', j.get('af3_buckets'))
      add('save_embeddings', bool(j.get('save_embeddings')))
      a = j['args']
      add('recycles, diffusion samples', [a['num_recycles'], a['num_diffusion_samples']])
      add('per-process timeout (min), entry, log_compiles, order', [a['timeout_min'], a['entry'], a['log_compiles'], a['order']])
      add('input manifest', a['manifest'])
      add('seeds per process', len(a['seeds'].split(',')))
      seeds |= {int(x) for x in a['seeds'].split(',')}
      if j.get('tpu_runtime'):
        add('libtpu build', j['tpu_runtime']['build_label'])
      r = json.loads((sj.parent / 'runs.jsonl').read_text().splitlines()[0])
      add('run_alphafold command', masked_command(r['command']))
  vals[REPS] = {', '.join(sorted(tags))}
  vals[SEEDS] = {f'{min(seeds)} to {max(seeds)}'}
  return {k: sorted(v) for k, v in vals.items()}, len(sessions)


def planned(glob, platform, source_stack):
  head = (REPO / 'cloud' / 'af3_run.sh').read_text()
  d = launcher_defaults(head, platform)
  configs = read_configs(REPO / 'harness' / 'configs.yaml')
  pin = dict(re.findall(r'export (\w+)="([^"]+)"', (REPO / 'af3_tpu' / 'pin.sh').read_text()))
  vals, tags, seeds = {}, set(), set()

  def add(k, v):
    vals.setdefault(k, set()).add(json.dumps(v, sort_keys=True) if not isinstance(v, str) else v)
  plans = sorted((REPO / 'harness' / 'plans').glob(glob))
  for p in plans:
    desc = run_plan.expand(p, platform, REPO / 'inputs' / 'manifest.csv')
    tags |= {rep_tag(r['run']) for r in desc['runs']}
    seeds |= set(desc['seeds'])
    config = desc['configs'][0]
    cfg = dict(configs[config])
    stack = desc.get('stack')
    add('platform', platform)
    add('provisioning', PLANNED[platform])
    add('weights', desc['weights'])
    add('machine type', d['machine type'])
    add('image', d['image'])
    if platform in ('v5e', 'v6e'):
      add('libtpu', (stack or {}).get('libtpu') or d['libtpu default'])
      add('libtpu build', 'libtpu_lts_20260630_b_RC06' if not stack else f'the build of libtpu {stack["libtpu"]}')
    add('label', desc['label'])
    add('config name', config)
    add('config description', cfg.pop('description', None))
    add('config (harness/configs.yaml entry, without its description)', cfg)
    env = dict(cfg.get('env', {}), JAX_LOG_COMPILES='1', **CACHE_ENV)
    add('environment (xla_env)', {k: str(v) for k, v in env.items() if k.startswith(('XLA_', 'JAX_', 'TF_'))})
    add(STACK, stack or source_stack)
    add('AlphaFold3 commit', pin['AF3_COMMIT'])
    add('af3_tpu.patch SHA-256', sha256(REPO / 'af3_tpu' / 'af3_tpu.patch'))
    add('run_alphafold.py SHA-256', sha256(REPO / 'third_party' / 'alphafold3' / 'run_alphafold.py'))
    add('buckets', desc.get('buckets'))
    add('save_embeddings', bool(desc.get('save_embeddings')))
    add('recycles, diffusion samples', [desc['num_recycles'], desc['num_diffusion_samples']])
    add('per-process timeout (min), entry, log_compiles, order', [desc['run_timeout_min'], True, True, 'tokens'])
    add('input manifest', desc.get('manifest', 'inputs/manifest.csv'))
    add('seeds per process', 1)
    shown = ' '.join(f'{k}={v}' for k, v in env.items())
    add('run_alphafold command', masked_command(
        f'cd third_party/alphafold3 && {shown} AF3_ENTRY_STATS=x .venv/bin/python ../../harness/af3_entry.py '
        f'--json_path=x --output_dir=x \'--model_dir=<model_dir>\' --jax_backend={cfg["jax_backend"]} '
        f'--flash_attention_implementation={cfg["flash_attention_implementation"]} --run_data_pipeline=false '
        f'--jax_compilation_cache_dir=x'))
  vals[REPS] = {', '.join(sorted(tags))}
  vals[SEEDS] = {f'{min(seeds)} to {max(seeds)}'}
  return {k: sorted(v) for k, v in vals.items()}, len(plans)


def main():
  ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
  ap.add_argument('--out', default=str(REPO / 'harness' / 'plans' / 'config_diff_fu7_10.md'))
  args = ap.parse_args()
  L = ['# Follow-ups 7 to 9: settings of each new arm against its source arm (harness/plans/config_diff_fu7_10.py)', '',
       'Source arm: as recorded in its sessions. New arm: as its plans, harness/configs.yaml, harness/run_af3.py and '
       'cloud/af3_run.sh resolve them, with the planned provisioning (Spot on v5e, L4 and H100; on demand on v6e). '
       'Intended differences: the factor under study, the label, the repetitions and, where it is Spot, the '
       'provisioning (how the VM is obtained, not the computation).', '']
  bad = 0
  for arm, (glob, platform, source, intended) in ARMS.items():
    src, n_src = recorded(source, platform if source != 'A' else 'a100')
    one = json.loads(src[STACK][0])
    new, n_new = planned(glob, platform, one)
    L += [f'## {arm} against {source} ({n_src} source sessions, {n_new} plans)', '',
          '| setting | source arm (recorded) | new arm (planned) | |', '|---|---|---|---|']
    for k in src:
      same = src[k] == new.get(k)
      mark = 'same' if same else ('intended' if k in intended else 'DIFFERS')
      bad += mark == 'DIFFERS'
      fmt = lambda v: '<br>'.join(f'`{x}`' for x in v) if v else '(none)'  # noqa: E731
      L.append(f'| {k} | {fmt(src[k])} | {fmt(new.get(k))} | {mark} |')
    L.append('')
  L.append(f'Settings that differ beyond the factor under study: {bad}')
  pathlib.Path(args.out).write_text('\n'.join(L) + '\n')
  print('\n'.join(L))
  return 1 if bad else 0


if __name__ == '__main__':
  sys.exit(main())
