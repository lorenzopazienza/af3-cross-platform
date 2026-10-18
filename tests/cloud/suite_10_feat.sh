#!/bin/bash
# Suite 10 (2026-10-18): follow-up 5, host-side featurisation (notes/analysis_plan.md section 10, row of
# 2026-10-18; label feat), on fake VMs. The plan is harness/plans/feat/feat_milan.yaml with two of the scratch
# repository's targets and the SHA-256 of a small fake components.cif (the real plans need all 90 inputs and the
# 519 MB CCD); everything else as launched: Spot, the plan's host, WEIGHTS=none with a weights URI in the shell
# (af3_run_many.sh passes it to every launcher), a results bucket.
#   feat_part        the VM is created with the plan's machine type, min CPU platform and image; no weights URI is
#                    uploaded and weights.json says none; the CCD reaches the VM and matches (ccd.json); one
#                    plan_runs.jsonl line per target, exit 0; features.jsonl for 2 targets x 5 seeds fetched.
#   refuse_feat_*    a MACHINE_TYPE or IMAGE_FAMILY that disagrees with the plan, WEIGHTS=gcs, no CCD_FILE, a
#                    CCD_FILE with another SHA-256, PLATFORM=l4, WEIGHTS=none on a model plan: refused, no VM.
# The checked scenario prints "followup check: N problem(s)".
. "$(dirname "$0")/common.sh"
R="$KIT/run.sh"
bash "$KIT/mkrepo.sh"
W=gs://private-weights/af3.bin.zst
B=gs://followup-results/study
P=harness/plans/feat/feat_fake.yaml
CCD="$WORK/components_fake.cif"
printf 'data_000\n_chem_comp.id 000\n' > "$CCD"
SHA=$( (sha256sum "$CCD" 2> /dev/null || shasum -a 256 "$CCD") | cut -d' ' -f1)
sed -e 's/^  default: all$/  default: [7U3J, 8EYE]/' -e "s/^ccd_sha256: .*/ccd_sha256: $SHA/" \
  "$WORK/repo/harness/plans/feat/feat_milan.yaml" > "$WORK/repo/$P"
(cd "$WORK/repo" && git add -A && git -c user.name=test -c user.email=test@invalid commit -qm feat-test)
echo "======== follow-up 5: featurisation only (Spot, plan host, no weights)"
bash "$R" feat_part ok PLATFORM=cpu FAKE_PLATFORM=cpu SPOT=1 PLAN=$P CCD_FILE="$CCD" WEIGHTS_GCS_URI=$W RESULTS_GCS_URI=$B
grep -E "Platform cpu|image |weights:|CCD:|worst-case cost" "$WORK/state_feat_part/out" | sed 's/^/   | /'
python3 - "$WORK/state_feat_part" "$SHA" "$(grep -c 'private-weights' "$WORK/state_feat_part/out")" <<'PYEOF'
import json, pathlib, sys
d, sha, leaks = pathlib.Path(sys.argv[1]), sys.argv[2], int(sys.argv[3])
problems = []
args = (d / 'create_args').read_text() if (d / 'create_args').is_file() else ''
for want in ('--machine-type=n2d-standard-32', '--provisioning-model=SPOT', '--image-family=ubuntu-2204-lts',
             '--image-project=ubuntu-os-cloud', '--min-cpu-platform=AMD Milan'):
  if want not in args:
    problems.append(f'create without {want}')
home = d / 'home'
if (home / '.af3_weights_uri').exists():
  problems.append('a weights URI reached the VM')
sessions = sorted(s for s in (d / 'fetched' / 'results' / 'af3').glob('*_cpu_feat_fake') if s.is_dir())
if not sessions:
  problems.append('nothing fetched')
else:
  s = sessions[-1]
  w = json.loads((s / 'weights.json').read_text()) if (s / 'weights.json').is_file() else {}
  if w.get('mode') != 'none':
    problems.append(f'weights.json {w}')
  c = json.loads((s / 'ccd.json').read_text()) if (s / 'ccd.json').is_file() else {}
  if c.get('used_sha256') != sha:
    problems.append(f'ccd.json {c}')
  runs = [json.loads(l) for l in (s / 'plan_runs.jsonl').read_text().splitlines()] if (s / 'plan_runs.jsonl').is_file() else []
  if sorted(r['target'] for r in runs) != ['7U3J', '8EYE'] or any(r['exit_code'] != 0 for r in runs):
    problems.append(f'plan_runs {[(r.get("target"), r.get("exit_code")) for r in runs]}')
  for t in ('7U3J', '8EYE'):
    f = s / 'featurise' / t / 'features.jsonl'
    seeds = sorted(json.loads(l)['seed'] for l in f.read_text().splitlines()) if f.is_file() else []
    if seeds != [1, 2, 3, 4, 5]:
      problems.append(f'{t}: seeds {seeds}')
  host = json.loads((s / 'featurise' / 'host.json').read_text()) if (s / 'featurise' / 'host.json').is_file() else {}
  if host.get('weights_seen'):
    problems.append(f'weights on the VM: {host["weights_seen"]}')
if leaks:
  problems.append('weights URI printed')
print(f'   followup check: {len(problems)} problem(s)' + (': ' + '; '.join(problems[:4]) if problems else ''))
PYEOF
bash "$R" refuse_feat_machine_type ok PLATFORM=cpu PLAN=$P CCD_FILE="$CCD" MACHINE_TYPE=n2-standard-8
bash "$R" refuse_feat_image ok PLATFORM=cpu PLAN=$P CCD_FILE="$CCD" IMAGE_FAMILY=ubuntu-2404-lts-amd64
bash "$R" refuse_feat_weights ok PLATFORM=cpu PLAN=$P CCD_FILE="$CCD" WEIGHTS=gcs WEIGHTS_GCS_URI=$W
bash "$R" refuse_feat_no_ccd ok PLATFORM=cpu PLAN=$P
printf 'another file\n' > "$WORK/components_other.cif"
bash "$R" refuse_feat_ccd_sha ok PLATFORM=cpu PLAN=$P CCD_FILE="$WORK/components_other.cif"
bash "$R" refuse_feat_platform ok PLATFORM=l4 PLAN=$P CCD_FILE="$CCD"
bash "$R" refuse_feat_none_model_plan ok PLATFORM=cpu PLAN=harness/plans/probe.yaml WEIGHTS=none
for n in refuse_feat_machine_type refuse_feat_image refuse_feat_weights refuse_feat_no_ccd refuse_feat_ccd_sha \
    refuse_feat_platform refuse_feat_none_model_plan; do
  grep -m1 '^!!' "$WORK/state_$n/out" | sed 's/^/   | /'
done
