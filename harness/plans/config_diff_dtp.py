"""Follow-up 17 (label dtp): every setting of each new arm against the float32-trunk arm it derives from (V6f and Gf of
follow-up 13, Af and Hf of follow-up 14), as harness/plans/config_diff_fu14.py does for follow-up 14 (config_diff_fu7_10's
functions, imported).

    python3 harness/plans/config_diff_dtp.py [--out harness/plans/config_diff_dtp.md]

Source side, as recorded in the source arm's sessions (V6f and Gf: scripts/analysis_runs/fu13/sessions.txt; Af and Hf:
scripts/analysis_runs/fu14/sessions.txt). New side, as the arm will run: its plans (harness/plans/dtp),
harness/configs.yaml, harness/run_af3.py, cloud/af3_run.sh's defaults (the A100's machine type, which the launcher
takes from A100_MACHINE_TYPE at launch, is a2-ultragpu-1g), af3_tpu/, with the provisioning of item f (Flex-start on
the v6e, Spot on the GPUs). Allowed differences (row "Follow-up experiment 17", item a): the label; the configuration's
name and description; the absence of JAX_DEFAULT_MATMUL_PRECISION=highest (the one setting this follow-up changes); the
seeds and repetitions; the provisioning. The configuration, the environment and the command are then compared again
with JAX_DEFAULT_MATMUL_PRECISION removed from both sides ("... without the follow-up 17 setting"), and those must be
the same: in particular AF3_FU13_BFLOAT16=none must be on both. Any other difference is DIFFERS and the script exits 1.
"""
import argparse
import json
import pathlib
import re
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'harness' / 'plans'))
import config_diff_fu7_10 as cd  # noqa: E402

CFG = 'config (harness/configs.yaml entry, without its description)'
ENV = 'environment (xla_env)'
CMD = 'run_alphafold command'
ALLOWED = {'label', 'config name', 'config description', CFG, ENV, CMD, cd.SEEDS, cd.REPS, 'provisioning'}
PRECISION = 'JAX_DEFAULT_MATMUL_PRECISION'
PROVISIONING = {'v6e': 'flex_start', 'l4': 'spot', 'a100': 'spot', 'h100': 'spot'}
# new arm: (plans glob, platform, source arm)
ARMS = {
    'V6fd': ('dtp/dtp_v6fd_*.yaml', 'v6e', 'V6f'),
    'Gfd': ('dtp/dtp_gfd_*.yaml', 'l4', 'Gf'),
    'Afd': ('dtp/dtp_afd_*.yaml', 'a100', 'Af'),
    'Hfd': ('dtp/dtp_hfd_*.yaml', 'h100', 'Hf'),
}
SOURCES = {'V6f': ('fu13', r'_v6e_why_v6f_\d+$'), 'Gf': ('fu13', r'_l4_why_gf_\d+$'),
           'Af': ('fu14', r'_a100_f32_af_\d+$'), 'Hf': ('fu14', r'_h100_f32_hf_\d+$')}
A100_MACHINE_TYPE = 'a2-ultragpu-1g'


def source_sessions(arm):
  folder, pat = SOURCES[arm]
  text = (REPO / 'scripts' / 'analysis_runs' / folder / 'sessions.txt').read_text()
  return [line.split('#')[0].strip() for line in text.splitlines() if re.search(pat, line.split('#')[0].strip())]


cd.source_sessions = source_sessions     # cd.recorded looks the name up at call time


def strip_env_dict(env):
  return {k: v for k, v in env.items() if k != PRECISION}


def strip_cfg(text):
  c = json.loads(text)
  env = strip_env_dict(c.get('env') or {})
  if env:
    c['env'] = env
  else:
    c.pop('env', None)
  return json.dumps(c, sort_keys=True)


def strip_env(text):
  return json.dumps(strip_env_dict(json.loads(text)), sort_keys=True)


def strip_cmd(text):
  return re.sub(rf' {PRECISION}=\S+', '', text)


def main():
  ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
  ap.add_argument('--out', default=str(REPO / 'harness' / 'plans' / 'config_diff_dtp.md'))
  args = ap.parse_args()
  cd.PLANNED = PROVISIONING          # cd.planned reads the provisioning of each platform from this name
  L = ['# Follow-up 17: settings of each new arm against its source arm (harness/plans/config_diff_dtp.py)', '',
       'Source arm: as recorded in its sessions (V6f and Gf: follow-up 13; Af and Hf: follow-up 14). New arm: as its '
       'plans, harness/configs.yaml, harness/run_af3.py and cloud/af3_run.sh resolve them (A100 machine type '
       f'{A100_MACHINE_TYPE}, set at launch by A100_MACHINE_TYPE), with the provisioning of item f. Allowed '
       'differences: the label, the configuration name and description, the absence of '
       'JAX_DEFAULT_MATMUL_PRECISION=highest, the seeds, the repetitions and the provisioning. Rows marked "without the '
       'follow-up 17 setting" compare the configuration, the environment and the command again with '
       'JAX_DEFAULT_MATMUL_PRECISION removed from both sides, and must be the same.', '']
  bad = 0
  for arm, (glob, platform, source) in ARMS.items():
    src, n_src = cd.recorded(source, platform)
    one = json.loads(src[cd.STACK][0])
    new, n_new = cd.planned(glob, platform, one)
    if platform == 'a100':
      new['machine type'] = [A100_MACHINE_TYPE]
    leftover = sorted(t for t in new.get(ENV, []) if PRECISION in json.loads(t))
    rows = [(k, src[k], new.get(k)) for k in src]
    rows += [(f'{CFG}, without the follow-up 17 setting', sorted({strip_cfg(t) for t in src[CFG]}),
              sorted({strip_cfg(t) for t in new.get(CFG, [])})),
             (f'{ENV}, without the follow-up 17 setting', sorted({strip_env(t) for t in src[ENV]}),
              sorted({strip_env(t) for t in new.get(ENV, [])})),
             (f'{CMD}, without the follow-up 17 setting', sorted({strip_cmd(t) for t in src[CMD]}),
              sorted({strip_cmd(t) for t in new.get(CMD, [])}))]
    L += [f'## {arm} against {source} ({n_src} source sessions, {n_new} plans)', '',
          f'{PRECISION} in the new arm\'s environment: ' + ('none (as intended)' if not leftover else 'PRESENT'), '',
          '| setting | source arm (recorded) | new arm (planned) | |', '|---|---|---|---|']
    bad += bool(leftover)
    for k, a, b in rows:
      mark = 'same' if a == b else ('intended' if k in ALLOWED else 'DIFFERS')
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
