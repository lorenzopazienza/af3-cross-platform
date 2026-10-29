#!/bin/bash
# Suite 12 (2026-10-27): follow-ups 7 to 9 launcher support (drafts/section10_fu7-10.md; labels sw, seeds, h100), on
# fake VMs. The plans are the follow-up plans cut to one scratch-repository target (7U3J) on the frozen inputs, since
# the smoke target's _data.json is not in the scratch repository:
#   h100_smoke          the H100 smoke plan on PLATFORM=h100, Spot: a3-highgpu-1g in us-central1-a, an H100 80 GB in
#                       setup.json, config h100_xla.
#   sw_l4               the L4 sw smoke plan: STACK goes to the VM, the four GPU packages are installed at 0.10.1 and
#                       checked, the launch record keeps the stack, the harness probe reports jax 0.10.1.
#   sw_v6e              the v6e sw smoke plan: jax/jaxlib 0.10.1 with libtpu 0.0.41 in the TPU swap.
#   seeds_beyond        a Gx plan cut to seed 6 (not in the frozen input): the run l4_xla_rep1_seed6_fresh, exit 0.
#   refuse_stack_shell  STACK set in the shell: refused before any VM.
#   refuse_stack_libtpu a stack plan with a LIBTPU_VERSION that disagrees: refused before any VM.
#   refuse_h100_machine a2-ultragpu-1g on PLATFORM=h100: refused before any VM.
#   refuse_h100_zone    europe-west4-b (not an H100 zone of the launcher): refused before any VM.
#   eigen_archive       an eigen_check plan (eigen_source: archive): the VM installs the pinned Eigen archive from the
#                       bucket (real cloud/vm_eigen_source.sh; a 3-file test archive and its pin in the scratch
#                       repository), records eigen.json, and the build gets FETCHCONTENT_SOURCE_DIR_MY-EIGEN3.
#   eigen_bad           the bucket holds other bytes under the archive's name: setup refuses it (exit 3), no run.
#   refuse_eigen_shell  EIGEN_SOURCE set in the shell: refused before any VM.
#   eigen_default       the same plan without the key: no Eigen step, no eigen.json, no EIGEN_SOURCE in the record.
# Every follow-up 7 to 9 plan takes the archive, so their scenarios run with FAKE_DEPS_DIR (the fake bucket's deps/).
# Each checked scenario prints "followup check: N problem(s)".
. "$(dirname "$0")/common.sh"
R="$KIT/run.sh"
bash "$KIT/mkrepo.sh"
# A 3-file Eigen stand-in archived as cloud/make_eigen_archive.sh does, and its pin in the scratch repository.
DEPS="$WORK/deps" DEPS_BAD="$WORK/deps_bad" EG="$WORK/eigen-git"
rm -rf "$DEPS" "$DEPS_BAD" "$EG"; mkdir -p "$DEPS" "$DEPS_BAD" "$EG"
for i in 0 1 2; do echo "// header $i" > "$EG/h$i.h"; done
git -C "$EG" init -q && git -C "$EG" add . && git -C "$EG" -c user.name=test -c user.email=test@invalid commit -qm eigen
EC=$(git -C "$EG" rev-parse HEAD)
git -C "$EG" archive --format=tar --prefix=eigen/ "$EC" | gzip -n > "$DEPS/eigen-test.tar.gz"
ESHA=$(shasum -a 256 "$DEPS/eigen-test.tar.gz" | cut -d" " -f1)
echo "not the archive" > "$DEPS_BAD/eigen-test.tar.gz"
cat > "$WORK/repo/cloud/eigen_pin.sh" <<PIN
export EIGEN_TAG="test"
export EIGEN_COMMIT="$EC"
export EIGEN_ARCHIVE_NAME="eigen-test.tar.gz"
export EIGEN_FILES="3"
export EIGEN_ARCHIVE_URI="gs://fake-deps/deps/eigen-test.tar.gz"
export EIGEN_ARCHIVE_SHA256="$ESHA"
PIN
W=gs://private-weights/af3.bin.zst
B=gs://followup-results/study
cut() {   # SOURCE DEST [sed expressions]: a follow-up plan on 7U3J, the frozen inputs, launch dates far ahead
  local src=$1 dest=$2; shift 2
  sed -e 's/^  default: \[.*\]$/  default: [7U3J]/' -e '/^manifest: /d' -e 's/^launch_by: .*/launch_by: 2099-12-31/' \
    -e 's/^finish_by: .*/finish_by: 2099-12-31/' "$@" "$WORK/repo/$src" > "$WORK/repo/$dest"
}
cut harness/plans/h100_smoke/h100_smoke_h100.yaml harness/plans/h100_smoke/h100_smoke_test.yaml
cut harness/plans/sw_smoke/sw_smoke_l4.yaml harness/plans/sw_smoke/sw_smoke_test_l4.yaml
cut harness/plans/sw_smoke/sw_smoke_v6e.yaml harness/plans/sw_smoke/sw_smoke_test_v6e.yaml
cut harness/plans/seeds/seeds_gx_01.yaml harness/plans/seeds/seeds_test_gx.yaml -e 's/^seeds: .*/seeds: [6]/'
cut harness/plans/eigen_check/eigen_check_v6e.yaml harness/plans/eigen_check/eigen_test_v6e.yaml
cut harness/plans/eigen_check/eigen_check_v6e.yaml harness/plans/eigen_check/eigen_test_default.yaml -e '/^eigen_source: /d'
(cd "$WORK/repo" && git add -A && git -c user.name=test -c user.email=test@invalid commit -qm fu7-10-test)
check() {   # NAME KIND: the create call, the fetched session, the stack on the VM
  python3 - "$WORK/state_$1" "$2" <<'PYEOF'
import json, pathlib, sys
d, kind = pathlib.Path(sys.argv[1]), sys.argv[2]
problems = []
args = (d / 'create_args').read_text().split() if (d / 'create_args').is_file() else []
log = (d / 'log').read_text() if (d / 'log').is_file() else ''
out = (d / 'out').read_text()
runs_dirs = [p for p in (d / 'fetched' / 'results' / 'af3').glob('*/*') if (p / 'session.json').is_file()]
sessions = [json.loads((p / 'session.json').read_text()) for p in runs_dirs]
runs = [json.loads(l) for p in runs_dirs for l in (p / 'runs.jsonl').read_text().splitlines()]
if not runs or any(r['exit_code'] != 0 for r in runs):
  problems.append(f'processes {[(r["pdb_id"], r["exit_code"]) for r in runs]}')
records = [f.read_text() for f in d.rglob('launch_record.txt')]
if kind == 'h100':
  problems += [f'create without {w}' for w in ('--machine-type=a3-highgpu-1g', '--provisioning-model=SPOT') if w not in args]
  zone = (d / 'created_zone').read_text().strip() if (d / 'created_zone').is_file() else ''
  if zone != 'us-central1-a':
    problems.append(f'created in {zone or "no zone"}')
  setups = [json.loads(f.read_text()) for f in (d / 'fetched' / 'results' / 'af3').glob('*/setup.json')]
  if not setups or 'H100' not in str(setups[0].get('device_kind')) or (setups[0].get('gpu') or {}).get('memory_total_mib') != 81559:
    problems.append(f'setup.json {setups[:1]}')
  if any(s.get('config_name') != 'h100_xla' or s.get('label') != 'h100_smoke' for s in sessions):
    problems.append('config or label')
if kind == 'sw_l4':
  want = 'jax==0.10.1 jaxlib==0.10.1 jax-cuda12-plugin==0.10.1 jax-cuda12-pjrt==0.10.1'
  if want not in log:
    problems.append('GPU stack not installed')
  if not any('STACK=jax==0.10.1,jaxlib==0.10.1,jax-cuda12-plugin==0.10.1,jax-cuda12-pjrt==0.10.1' in r for r in records):
    problems.append('stack not in the launch record')
  if any((s.get('probe') or {}).get('packages', {}).get('jax') != '0.10.1' for s in sessions) or not sessions:
    problems.append(f'probe jax {[(s.get("probe") or {}).get("packages") for s in sessions][:1]}')
  if 'GPU stack: the plan\'s alternative stack' not in out:
    problems.append('stack not printed')
if kind == 'sw_v6e':
  if 'jax==0.10.1 jaxlib==0.10.1 libtpu==0.0.41' not in log:
    problems.append('TPU stack not installed')
  pk = [(s.get('probe') or {}).get('packages', {}) for s in sessions]
  if not pk or any((p.get('jaxlib'), p.get('libtpu')) != ('0.10.1', '0.0.41') for p in pk):
    problems.append(f'probe packages {pk[:1]}')
  if not any('STACK=jax==0.10.1,jaxlib==0.10.1,libtpu==0.0.41' in r for r in records):
    problems.append('stack not in the launch record')
if kind == 'seeds':
  if sorted(p.name for p in runs_dirs) != ['l4_xla_rep1_seed6_fresh']:
    problems.append(f'runs {sorted(p.name for p in runs_dirs)}')
  if any(not (s.get('args') or {}).get('seeds_beyond_input') for s in sessions):
    problems.append('seeds_beyond_input not recorded')
if 'private-weights' in out:
  problems.append('weights URI printed')
print(f'   followup check: {len(problems)} problem(s)' + (': ' + '; '.join(problems[:4]) if problems else ''))
PYEOF
}
refused() {   # NAME TEXT: refused before any VM, with TEXT in the message
  python3 - "$WORK/state_$1" "$2" <<'PYEOF'
import pathlib, sys
d, text = pathlib.Path(sys.argv[1]), sys.argv[2]
problems = [] if text in (d / 'out').read_text() else [f'no "{text}"']
problems += ['a VM was created'] if (d / 'create_args').exists() else []
print(f'   followup check: {len(problems)} problem(s)' + (': ' + '; '.join(problems) if problems else ''))
PYEOF
}
echo "======== follow-up 9: H100 (a3-highgpu-1g, Spot)"
bash "$R" h100_smoke ok FAKE_DEPS_DIR=$DEPS PLATFORM=h100 FAKE_PLATFORM=h100 SPOT=1 PLAN=harness/plans/h100_smoke/h100_smoke_test.yaml \
  WEIGHTS_GCS_URI=$W RESULTS_GCS_URI=$B
grep -E "Platform h100|zones tried in order|worst-case cost" "$WORK/state_h100_smoke/out" | sed 's/^/   | /'
check h100_smoke h100
bash "$R" refuse_h100_machine ok PLATFORM=h100 MACHINE_TYPE=a2-ultragpu-1g PLAN=harness/plans/h100_smoke/h100_smoke_test.yaml \
  WEIGHTS_GCS_URI=$W
refused refuse_h100_machine 'takes a3-highgpu-1g only'
bash "$R" refuse_h100_zone ok PLATFORM=h100 ZONES=europe-west4-b PLAN=harness/plans/h100_smoke/h100_smoke_test.yaml \
  WEIGHTS_GCS_URI=$W
refused refuse_h100_zone 'is not one of the zones with a3-highgpu-1g'
echo "======== follow-up 7: alternative stack from the plan"
bash "$R" sw_l4 ok FAKE_DEPS_DIR=$DEPS PLATFORM=l4 FAKE_PLATFORM=l4 SPOT=1 ZONES=us-central1-a PLAN=harness/plans/sw_smoke/sw_smoke_test_l4.yaml \
  WEIGHTS_GCS_URI=$W RESULTS_GCS_URI=$B
grep -E "stack" "$WORK/state_sw_l4/out" | sed 's/^/   | /'
check sw_l4 sw_l4
bash "$R" sw_v6e ok FAKE_DEPS_DIR=$DEPS PLATFORM=v6e FAKE_PLATFORM=v6e PLAN=harness/plans/sw_smoke/sw_smoke_test_v6e.yaml \
  WEIGHTS_GCS_URI=$W RESULTS_GCS_URI=$B
grep -E "TPU stack" "$WORK/state_sw_v6e/out" | sed 's/^/   | /'
check sw_v6e sw_v6e
bash "$R" refuse_stack_shell ok PLATFORM=l4 STACK=jax==0.10.1 PLAN=harness/plans/sw_smoke/sw_smoke_test_l4.yaml \
  WEIGHTS_GCS_URI=$W
refused refuse_stack_shell 'STACK is set in this shell'
bash "$R" refuse_stack_libtpu ok PLATFORM=v6e LIBTPU_VERSION=0.0.43.2 PLAN=harness/plans/sw_smoke/sw_smoke_test_v6e.yaml \
  WEIGHTS_GCS_URI=$W
refused refuse_stack_libtpu "disagrees with the plan's stack"
echo "======== follow-up 8: seeds beyond the frozen input"
bash "$R" seeds_beyond ok FAKE_DEPS_DIR=$DEPS PLATFORM=l4 FAKE_PLATFORM=l4 SPOT=1 ZONES=us-central1-a PLAN=harness/plans/seeds/seeds_test_gx.yaml \
  WEIGHTS_GCS_URI=$W RESULTS_GCS_URI=$B
check seeds_beyond seeds
echo "======== Eigen from the bucket (eigen_source: archive)"
eigen_case() {   # NAME KIND (archive, bad or default)
  python3 - "$WORK/state_$1" "$2" "$ESHA" "$EC" <<'PYEOF'
import json, pathlib, sys
d, kind, sha, commit = pathlib.Path(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4]
problems = []
fetched = d / 'fetched' / 'results' / 'af3'
sessions = [s for s in fetched.glob('*_v6e_*') if s.is_dir()] if fetched.is_dir() else []
records = ''.join(f.read_text() for f in d.rglob('launch_record.txt'))
eig = [json.loads(f.read_text()) for s in sessions for f in s.glob('eigen.json')]
cma = [f.read_text() for s in sessions for f in s.glob('fake_cmake_args.txt')]
runs = [json.loads(l) for s in sessions for f in s.glob('*/runs.jsonl') for l in f.read_text().splitlines()]
out = (d / 'out').read_text()
if kind == 'archive':
  if not eig or (eig[0]['archive_sha256'], eig[0]['commit'], eig[0]['files']) != (sha, commit, 3):
    problems.append(f'eigen.json {eig[:1]}')
  if not cma or '-DFETCHCONTENT_SOURCE_DIR_MY-EIGEN3=' not in cma[0] or not cma[0].strip().endswith('/eigen'):
    problems.append(f'CMAKE_ARGS {cma[:1]}')
  if 'EIGEN_SOURCE=archive' not in records:
    problems.append('EIGEN_SOURCE not in the launch record')
  if 'Eigen (libcifpp build): the plan' not in out:
    problems.append('Eigen source not printed')
  if not runs or any(r['exit_code'] != 0 for r in runs):
    problems.append(f'processes {[(r["pdb_id"], r["exit_code"]) for r in runs]}')
elif kind == 'bad':
  if runs:
    problems.append('AlphaFold3 ran with a refused archive')
  if eig:
    problems.append('eigen.json written for a refused archive')
  if 'Eigen archive refused' not in ''.join(f.read_text(errors='replace') for f in d.rglob('job.log')) + out:
    problems.append('refusal not in the job log')
elif kind == 'default':
  if eig or cma or 'EIGEN_SOURCE' in records:
    problems.append('Eigen step ran without the key')
  if not runs or any(r['exit_code'] != 0 for r in runs):
    problems.append(f'processes {[(r["pdb_id"], r["exit_code"]) for r in runs]}')
print(f'   followup check: {len(problems)} problem(s)' + (': ' + '; '.join(problems[:4]) if problems else ''))
PYEOF
}
bash "$R" eigen_archive ok PLATFORM=v6e FAKE_PLATFORM=v6e FAKE_DEPS_DIR=$DEPS PLAN=harness/plans/eigen_check/eigen_test_v6e.yaml \
  WEIGHTS_GCS_URI=$W RESULTS_GCS_URI=$B
grep -E "Eigen" "$WORK/state_eigen_archive/out" | sed 's/^/   | /'
eigen_case eigen_archive archive
bash "$R" eigen_bad ok PLATFORM=v6e FAKE_PLATFORM=v6e FAKE_DEPS_DIR=$DEPS_BAD PLAN=harness/plans/eigen_check/eigen_test_v6e.yaml \
  WEIGHTS_GCS_URI=$W RESULTS_GCS_URI=$B
eigen_case eigen_bad bad
bash "$R" refuse_eigen_shell ok PLATFORM=v6e EIGEN_SOURCE=archive PLAN=harness/plans/eigen_check/eigen_test_v6e.yaml \
  WEIGHTS_GCS_URI=$W
refused refuse_eigen_shell 'EIGEN_SOURCE is set in this shell'
bash "$R" eigen_default ok PLATFORM=v6e FAKE_PLATFORM=v6e PLAN=harness/plans/eigen_check/eigen_test_default.yaml \
  WEIGHTS_GCS_URI=$W RESULTS_GCS_URI=$B
eigen_case eigen_default default
