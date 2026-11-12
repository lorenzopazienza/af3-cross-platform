"""Follow-up 14: every setting of each new arm against the bfloat16 arm it derives from (V5 of the main study, A of
follow-up 3, H of follow-up 9), as harness/plans/config_diff_fu13.py does for follow-up 13 (config_diff_fu7_10's
functions, imported).

    python3 harness/plans/config_diff_fu14.py [--out harness/plans/config_diff_fu14.md]

Source side, as recorded in the source arm's sessions (V5: results/analysis/main_20261014/sessions_main.txt; A:
scripts/analysis_runs/a100/sessions.txt, the A parts; H: scripts/analysis_runs/h100/sessions.txt, the H parts of r1).
New side, as the arm will run: its plans (harness/plans/f32), harness/configs.yaml, harness/run_af3.py, cloud/
af3_run.sh's defaults (the A100's machine type, which the launcher takes from A100_MACHINE_TYPE at launch, is
a2-ultragpu-1g), af3_tpu/, with the provisioning of item f (make_fu14_plans.PROVISIONING: on demand on v5e, Spot on
the A100 and the H100). Allowed differences (row of 2026-11-12, items a and f): the label; the follow-up 14 settings,
that is the configuration's name and description and, for the float32 arms, AF3_FU13_BFLOAT16=none and
JAX_DEFAULT_MATMUL_PRECISION=highest, for the deterministic arms --xla_gpu_deterministic_ops=true in XLA_FLAGS; the
seeds and repetitions; the provisioning. The rows that carry those settings are then compared again with them removed
("... without the follow-up 14 settings"), and those must be the same. Any other difference is DIFFERS and the script
exits 1.
"""
import argparse
import json
import pathlib
import re
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'harness' / 'plans'))
import config_diff_fu7_10 as cd  # noqa: E402
from make_fu14_plans import PROVISIONING  # noqa: E402

CFG = 'config (harness/configs.yaml entry, without its description)'
ENV = 'environment (xla_env)'
CMD = 'run_alphafold command'
FU14 = {'label', 'config name', 'config description', CFG, ENV, CMD, cd.SEEDS, cd.REPS, 'provisioning'}
DET_FLAG = ' --xla_gpu_deterministic_ops=true'
# new arm: (plans glob, platform, source arm, kind)
ARMS = {
    'V5f': ('f32/f32_v5f_*.yaml', 'v5e', 'V5', 'float32 trunk'),
    'Af': ('f32/f32_af_*.yaml', 'a100', 'A', 'float32 trunk'),
    'Hf': ('f32/f32_hf_*.yaml', 'h100', 'H', 'float32 trunk'),
    'Ad': ('f32/f32_ad_*.yaml', 'a100', 'A', 'deterministic ops'),
    'Hd': ('f32/f32_hd_*.yaml', 'h100', 'H', 'deterministic ops'),
}
A100_MACHINE_TYPE = 'a2-ultragpu-1g'
_source_sessions = cd.source_sessions


def source_sessions(arm):
  if arm == 'H':
    return [l.split('#')[0].strip() for l in (REPO / 'scripts' / 'analysis_runs' / 'h100' / 'sessions.txt').read_text().splitlines()
            if re.search(r'_h100_h100_h_\d+$', l.split('#')[0].strip())]
  return _source_sessions(arm)


cd.source_sessions = source_sessions     # cd.recorded looks the name up at call time


def strip_env_dict(env, kind):
  out = {}
  for k, v in env.items():
    if kind == 'float32 trunk' and (k.startswith('AF3_FU13_') or k == 'JAX_DEFAULT_MATMUL_PRECISION'):
      continue
    if kind == 'deterministic ops' and k == 'XLA_FLAGS':
      v = str(v).replace(DET_FLAG, '')
    out[k] = v
  return out


def strip_cfg(text, kind):
  c = json.loads(text)
  env = strip_env_dict(c.get('env') or {}, kind)
  if env:
    c['env'] = env
  else:
    c.pop('env', None)
  return json.dumps(c, sort_keys=True)


def strip_env(text, kind):
  return json.dumps(strip_env_dict(json.loads(text), kind), sort_keys=True)


def strip_cmd(text, kind):
  if kind == 'float32 trunk':
    return re.sub(r' JAX_DEFAULT_MATMUL_PRECISION=\S+', '', re.sub(r' AF3_FU13_\w+=\S+', '', text))
  return text.replace(DET_FLAG, '')


def ours(new_cfgs, kind):
  out = set()
  for t in new_cfgs:
    for k, v in (json.loads(t).get('env') or {}).items():
      if kind == 'float32 trunk' and (k.startswith('AF3_FU13_') or k == 'JAX_DEFAULT_MATMUL_PRECISION'):
        out.add(f'{k}={v}')
      if kind == 'deterministic ops' and k == 'XLA_FLAGS' and DET_FLAG.strip() in str(v):
        out.add(DET_FLAG.strip())
  return sorted(out)


def main():
  ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
  ap.add_argument('--out', default=str(REPO / 'harness' / 'plans' / 'config_diff_fu14.md'))
  args = ap.parse_args()
  cd.PLANNED = PROVISIONING          # cd.planned reads the provisioning of each platform from this name
  L = ['# Follow-up 14: settings of each new arm against its source arm (harness/plans/config_diff_fu14.py)', '',
       'Source arm: as recorded in its sessions (V5: main study; A: follow-up 3; H: follow-up 9, r1). New arm: as its '
       'plans, harness/configs.yaml, harness/run_af3.py and cloud/af3_run.sh resolve them (A100 machine type '
       f'{A100_MACHINE_TYPE}, set at launch by A100_MACHINE_TYPE), with the provisioning of item f (on demand on v5e, '
       'Spot on the A100 and the H100). Allowed differences: the label, the follow-up 14 settings (configuration name '
       'and description; AF3_FU13_BFLOAT16=none and JAX_DEFAULT_MATMUL_PRECISION=highest for the float32 arms; '
       '--xla_gpu_deterministic_ops=true in XLA_FLAGS for the deterministic arms), the seeds, the repetitions and the '
       'provisioning. Rows marked "without the follow-up 14 settings" compare the configuration, the environment and '
       'the command again with those settings removed, and must be the same.', '']
  bad = 0
  cache = {}
  for arm, (glob, platform, source, kind) in ARMS.items():
    if source not in cache:
      cache[source] = cd.recorded(source, platform)
    src, n_src = cache[source]
    one = json.loads(src[cd.STACK][0])
    new, n_new = cd.planned(glob, platform, one)
    if platform == 'a100':
      new['machine type'] = [A100_MACHINE_TYPE]
    rows = [(k, src[k], new.get(k)) for k in src]
    rows += [(f'{CFG}, without the follow-up 14 settings', sorted({strip_cfg(t, kind) for t in src[CFG]}),
              sorted({strip_cfg(t, kind) for t in new.get(CFG, [])})),
             (f'{ENV}, without the follow-up 14 settings', sorted({strip_env(t, kind) for t in src[ENV]}),
              sorted({strip_env(t, kind) for t in new.get(ENV, [])})),
             (f'{CMD}, without the follow-up 14 settings', sorted({strip_cmd(t, kind) for t in src[CMD]}),
              sorted({strip_cmd(t, kind) for t in new.get(CMD, [])}))]
    L += [f'## {arm} against {source} ({n_src} source sessions, {n_new} plans): {kind}', '',
          f'Follow-up 14 settings of {arm}: ' + ', '.join(f'`{x}`' for x in ours(new.get(CFG, []), kind)), '',
          '| setting | source arm (recorded) | new arm (planned) | |', '|---|---|---|---|']
    for k, a, b in rows:
      mark = 'same' if a == b else ('intended' if k in FU14 else 'DIFFERS')
      bad += mark == 'DIFFERS'
      fmt = lambda v: '<br>'.join(f'`{x}`' for x in v) if v else '(none)'  # noqa: E731
      L.append(f'| {k} | {fmt(a)} | {fmt(b)} | {mark} |')
    L.append('')
  L.append(f'Settings that differ beyond the allowed ones: {bad}')
  pathlib.Path(args.out).write_text('\n'.join(L) + '\n')
  print('\n'.join(L))
  return 1 if bad else 0


if __name__ == '__main__':
  sys.exit(main())
