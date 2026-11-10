"""Follow-up 13: every setting of each new arm against the main-study arm it derives from (V5, V6 or G), as
harness/plans/config_diff_fu11_12.py does for follow-ups 11 and 12 (config_diff_fu7_10's functions, imported).

    python3 harness/plans/config_diff_fu13.py [--out harness/plans/config_diff_fu13.md]

Source side, as recorded in the main-study sessions of V5, V6 and G (results/analysis/main_20261014/sessions_main.txt).
New side, as the arm will run: its plans (harness/plans/why), harness/configs.yaml, harness/run_af3.py, cloud/
af3_run.sh's defaults, af3_tpu/, with the provisioning of the specification row (make_fu13_plans.PROPOSED: on demand on
v5e and L4, Flex-start on v6e). Allowed differences (row of 2026-11-10): the label; the follow-up 13 settings, that is
the configuration's name and description, its AF3_FU13_* variables and, for the float32 arms, JAX_DEFAULT_MATMUL_
PRECISION=highest; the seeds and repetitions; the provisioning. The rows that carry those settings are then compared
again with them removed ("... without the follow-up 13 settings"), and those must be the same: so nothing else in
the configuration, the environment or the run_alphafold command may differ. Any other difference is DIFFERS and the
script exits 1.
"""
import argparse
import json
import pathlib
import re
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'harness' / 'plans'))
import config_diff_fu7_10 as cd  # noqa: E402
from make_fu13_plans import PROPOSED  # noqa: E402

CFG = 'config (harness/configs.yaml entry, without its description)'
ENV = 'environment (xla_env)'
CMD = 'run_alphafold command'
FU13 = {'label', 'config name', 'config description', CFG, ENV, CMD, cd.SEEDS, cd.REPS, 'provisioning'}
# new arm: (plans glob, platform, source arm, its follow-up 13 settings)
ARMS = {
    'L5': ('why/why_l5_*.yaml', 'v5e', 'V5', 'capture'),
    'L6': ('why/why_l6_*.yaml', 'v6e', 'V6', 'capture'),
    'LG': ('why/why_lg_*.yaml', 'l4', 'G', 'capture'),
    'V6f': ('why/why_v6f_*.yaml', 'v6e', 'V6', 'float32 trunk'),
    'P': ('why/why_p_*.yaml', 'v6e', 'V6', 'perturbation'),
    'Gf': ('why/why_gf_*.yaml', 'l4', 'G', 'float32 trunk'),
}
OURS = re.compile(r'^(AF3_FU13_\w+)$')


def strip_cfg(text, float32):
  """A config entry (JSON) without its AF3_FU13_* variables and, for the float32 arms, JAX_DEFAULT_MATMUL_PRECISION."""
  c = json.loads(text)
  env = {k: v for k, v in (c.get('env') or {}).items()
         if not OURS.match(k) and not (float32 and k == 'JAX_DEFAULT_MATMUL_PRECISION')}
  if env:
    c['env'] = env
  else:
    c.pop('env', None)
  return json.dumps(c, sort_keys=True)


def strip_env(text, float32):
  e = json.loads(text)
  return json.dumps({k: v for k, v in e.items() if not (float32 and k == 'JAX_DEFAULT_MATMUL_PRECISION')},
                    sort_keys=True)


def strip_cmd(text, float32):
  c = re.sub(r' AF3_FU13_\w+=\S+', '', text)
  return re.sub(r' JAX_DEFAULT_MATMUL_PRECISION=\S+', '', c) if float32 else c


def ours(new_cfgs):
  """The follow-up 13 variables of the new arm's configuration(s), as 'NAME=value'."""
  out = set()
  for t in new_cfgs:
    for k, v in (json.loads(t).get('env') or {}).items():
      if OURS.match(k) or k == 'JAX_DEFAULT_MATMUL_PRECISION':
        out.add(f'{k}={v}')
  return sorted(out)


def main():
  ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
  ap.add_argument('--out', default=str(REPO / 'harness' / 'plans' / 'config_diff_fu13.md'))
  args = ap.parse_args()
  cd.PLANNED = PROPOSED          # cd.planned reads the provisioning of each platform from this name
  L = ['# Follow-up 13: settings of each new arm against its source arm (harness/plans/config_diff_fu13.py)', '',
       'Source arm: as recorded in its main-study sessions. New arm: as its plans, harness/configs.yaml, '
       'harness/run_af3.py and cloud/af3_run.sh resolve them, with the provisioning of the row (on demand on v5e and '
       'L4, Flex-start on v6e). Allowed differences: the label, the follow-up 13 settings (configuration name and '
       'description, AF3_FU13_* variables, JAX_DEFAULT_MATMUL_PRECISION=highest for the float32 arms), the seeds, the '
       'repetitions and the provisioning. Rows marked "without the follow-up 13 settings" compare the configuration, '
       'the environment and the command again with those settings removed, and must be the same.', '']
  bad = 0
  cache = {}
  for arm, (glob, platform, source, what) in ARMS.items():
    if source not in cache:
      cache[source] = cd.recorded(source, platform)
    src, n_src = cache[source]
    one = json.loads(src[cd.STACK][0])
    new, n_new = cd.planned(glob, platform, one)
    f32 = what == 'float32 trunk'
    rows = [(k, src[k], new.get(k)) for k in src]
    rows += [(f'{CFG}, without the follow-up 13 settings', sorted({strip_cfg(t, f32) for t in src[CFG]}),
              sorted({strip_cfg(t, f32) for t in new.get(CFG, [])})),
             (f'{ENV}, without the follow-up 13 settings', sorted({strip_env(t, f32) for t in src[ENV]}),
              sorted({strip_env(t, f32) for t in new.get(ENV, [])})),
             (f'{CMD}, without the follow-up 13 settings', sorted({strip_cmd(t, f32) for t in src[CMD]}),
              sorted({strip_cmd(t, f32) for t in new.get(CMD, [])}))]
    L += [f'## {arm} against {source} ({n_src} source sessions, {n_new} plans): {what}', '',
          f'Follow-up 13 settings of {arm}: ' + ', '.join(f'`{x}`' for x in ours(new.get(CFG, []))), '',
          '| setting | source arm (recorded) | new arm (planned) | |', '|---|---|---|---|']
    for k, a, b in rows:
      same = a == b
      mark = 'same' if same else ('intended' if k in FU13 else 'DIFFERS')
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
