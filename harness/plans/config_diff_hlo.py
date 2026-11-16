"""Follow-up 16: every setting of the two dump arms against the main-study arm each repeats (V5dump against V5, V6dump
against V6), as harness/plans/config_diff_rcx.py does for follow-up 15 (config_diff_fu7_10's functions, imported).

    python3 harness/plans/config_diff_hlo.py [--out harness/plans/config_diff_hlo.md]

Source side, as recorded in the main study's V5 and V6 sessions (results/analysis/main_20261014/sessions_main.txt):
launch_record.txt, every fresh harness run's session.json and command, the image of cloud/af3_run.sh at the session's
commit, and the Eigen source of the build (no eigen.json: the GitLab clone of tag 3.4.0, before follow-up 7).
New side, as the arm will run: its plan (harness/plans/hlo), harness/configs.yaml, harness/run_af3.py, cloud/
af3_run.sh's defaults, af3_tpu/, on demand on both chips (make_hlo_plans.PROVISIONING).
Allowed differences (row of 2026-11-16, items a and b): the label; the configuration (tpu_xla_hlo is tpu_xla plus
AF3_HLO_DUMP_FLAGS, so its name, description, entry and the run_alphafold command, which shows the variable, differ;
the environment recorded as xla_env, the XLA_, JAX_ and TF_ variables the harness sets, does not); the repetitions and
seeds (seed 1, r1 and r2, no warm rerun); the Eigen source (the archive of the same commit, 3147391d, as every plan
since follow-up 7). The configuration entry may differ from tpu_xla only by that variable: anything else is DIFFERS.
Any other difference is DIFFERS and the script exits 1.
"""
import argparse
import json
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'harness' / 'plans'))
import config_diff_fu7_10 as cd  # noqa: E402
import config_diff_rcx as cr  # noqa: E402
import make_hlo_plans as mk  # noqa: E402

CFG = 'config (harness/configs.yaml entry, without its description)'
CMD = 'run_alphafold command'
DUMP = {'AF3_HLO_DUMP_FLAGS': '--xla_dump_hlo_as_text --xla_dump_hlo_module_re=jit_apply_fn'}
ALLOWED = {'label', 'config name', 'config description', CFG, CMD, cd.REPS, cd.SEEDS, cr.EIGEN}
# new arm: (plans glob, platform, source arm)
ARMS = {'V5dump': ('hlo/hlo_v5_*.yaml', 'v5e', 'V5'), 'V6dump': ('hlo/hlo_v6_*.yaml', 'v6e', 'V6')}


def only_the_dump(src_cfg, new_cfg):
  """True if the new configuration entry is the recorded one plus AF3_HLO_DUMP_FLAGS, nothing else."""
  a, b = json.loads(src_cfg), json.loads(new_cfg)
  env = dict(a.get('env', {}), **DUMP)
  return dict(a, env=env) == b


def only_the_dump_command(src_cmd, new_cmd):
  """True if the new command is the recorded one with AF3_HLO_DUMP_FLAGS=<flags> added before JAX_LOG_COMPILES (quoted
  as harness/run_af3.py writes it, or unquoted as config_diff_fu7_10.planned shows it)."""
  flags = DUMP['AF3_HLO_DUMP_FLAGS']
  for form in (f"AF3_HLO_DUMP_FLAGS='{flags}' ", f'AF3_HLO_DUMP_FLAGS={flags} '):
    if new_cmd.replace(form, '', 1) == src_cmd:
      return True
  return False


def main():
  ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
  ap.add_argument('--out', default=str(REPO / 'harness' / 'plans' / 'config_diff_hlo.md'))
  args = ap.parse_args()
  cd.PLANNED = mk.PROVISIONING          # cd.planned reads the provisioning of each platform from this name
  L = ['# Follow-up 16: settings of each dump arm against the main-study arm it repeats (harness/plans/config_diff_hlo.py)',
       '', 'Source: as recorded in the main study\'s V5 and V6 sessions, with the Eigen source of their build. New: as '
       'the plans of harness/plans/hlo, harness/configs.yaml, harness/run_af3.py and cloud/af3_run.sh resolve them, on '
       'demand. Allowed differences: the label, the configuration (tpu_xla plus AF3_HLO_DUMP_FLAGS only, checked '
       'below), the repetitions and seeds, and the Eigen source (the archive of the same commit). Every other row '
       'must be the same.', '']
  bad = 0
  for arm, (glob, platform, source) in ARMS.items():
    src, n_src = cd.recorded(source, platform)
    sessions = cd.source_sessions(source)
    src[cr.EIGEN] = sorted({cr.eigen_of(s) for s in sessions})
    one = json.loads(src[cd.STACK][0])
    new, n_new = cd.planned(glob, platform, one)
    new[cr.EIGEN] = ['archive 3.4.0 commit 3147391d946bb4b6c68edd901f2add6ac1f31f8c (plan key eigen_source: archive)']
    L += [f'## {arm} against {source} ({n_src} main-study sessions, {n_new} plan)', '',
          '| setting | source arm (recorded) | dump arm (planned) | |', '|---|---|---|---|']
    for k in src:
      a, b = src[k], new.get(k)
      if a == b:
        mark = 'same'
      elif k not in ALLOWED:
        mark = 'DIFFERS'
      elif k == CFG:
        mark = 'intended (tpu_xla plus AF3_HLO_DUMP_FLAGS)' if len(a) == 1 and len(b or []) == 1 and only_the_dump(
            a[0], b[0]) else 'DIFFERS'
      elif k == CMD:
        mark = 'intended (AF3_HLO_DUMP_FLAGS added)' if len(a) == 1 and len(b or []) == 1 and only_the_dump_command(
            a[0], b[0]) else 'DIFFERS'
      else:
        mark = 'intended'
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
