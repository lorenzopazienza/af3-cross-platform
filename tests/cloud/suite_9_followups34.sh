#!/bin/bash
# Suite 9 (2026-10-16): follow-ups 3 (A100 arms) and 4 (CPU on every target), notes/analysis_plan.md section 10,
# rows of 2026-10-16, on fake VMs, as they will be launched (official weights from a bucket, results bucket).
#   a100_smoke     harness/plans/a100_smoke.yaml on PLATFORM=a100 with MACHINE_TYPE=a2-ultragpu-1g, on demand: the VM
#                  is created with that machine type, --provisioning-model=STANDARD and --maintenance-policy=TERMINATE
#                  in us-central1-a (the first default zone); setup.json records an A100 with 81920 MiB.
#   a100_zones     the first default zone has no capacity: the launcher moves on to us-central1-c.
#   a100_40gb      the card reports 40 GB: the device check refuses it (exit 5), the VM is deleted.
#   a100_east5     a100_smoke_t.yaml in us-east5 (quota approved later on 2026-10-16): us-east5-a has no capacity, the
#                  VM is created in us-east5-b; the worst case uses the us-east5 price.
#   refuse_*       a2-highgpu-1g, no MACHINE_TYPE, or a zone without a2-ultragpu-1g (us-east4-a): refused before any VM.
#   cpu60_part     a cpu60 part (harness/plans/cpu60/cpu60_7d5c.yaml, one target, seeds 1 and 2): Spot, n2-highmem-16.
# Each checked scenario prints "followup check: N problem(s)".
. "$(dirname "$0")/common.sh"
R="$KIT/run.sh"
bash "$KIT/mkrepo.sh"
W=gs://private-weights/af3.bin.zst
B=gs://followup-results/study
check() {   # NAME KIND (a100 or cpu60) ZONE [CONFIG]: create flags, the fetched session and no weights URI in the output
  python3 - "$WORK/state_$1" "$2" "$3" "$(grep -c 'private-weights' "$WORK/state_$1/out")" "${4:-}" <<'PYEOF'
import json, pathlib, sys
d, kind, zone, leaks, config = pathlib.Path(sys.argv[1]), sys.argv[2], sys.argv[3], int(sys.argv[4]), sys.argv[5]
problems = []
args = (d / 'create_args').read_text().split() if (d / 'create_args').is_file() else []
want = {'a100': ['--machine-type=a2-ultragpu-1g', '--provisioning-model=STANDARD', '--maintenance-policy=TERMINATE'],
        'cpu60': ['--machine-type=n2-highmem-16', '--provisioning-model=SPOT']}[kind]
problems += [f'create without {w}' for w in want if w not in args]
created = (d / 'created_zone').read_text().strip() if (d / 'created_zone').is_file() else ''
if created != zone:
  problems.append(f'created in {created or "no zone"}, expected {zone}')
sessions = sorted(s for s in (d / 'fetched' / 'results' / 'af3').glob(f'*_{"a100" if kind == "a100" else "cpu"}_*')
                  if s.is_dir())
if not sessions:
  problems.append('nothing fetched')
else:
  s = sessions[-1]
  if kind == 'a100':
    setup = json.loads((s / 'setup.json').read_text()) if (s / 'setup.json').is_file() else {}
    gpu = setup.get('gpu') or {}
    if 'A100' not in str(setup.get('device_kind')) or gpu.get('memory_total_mib') != 81920:
      problems.append(f'setup.json device {setup.get("device_kind")}, gpu {gpu}')
  for sj in s.glob('*/session.json'):
    j = json.loads(sj.read_text())
    want_label, want_config = ('smoke', config or 'a100_xla') if kind == 'a100' else ('cpu60', 'cpu_xla')
    if j.get('label') != want_label or j.get('config_name') != want_config:
      problems.append(f'{sj.parent.name}: label {j.get("label")} config {j.get("config_name")}')
  runs = [json.loads(l) for f in s.glob('*/runs.jsonl') for l in f.read_text().splitlines()]
  if not runs or any(r['exit_code'] != 0 for r in runs):
    problems.append(f'processes {[(r["pdb_id"], r["exit_code"]) for r in runs]}')
if leaks:
  problems.append('weights URI printed')
print(f'   followup check: {len(problems)} problem(s)' + (': ' + '; '.join(problems[:4]) if problems else ''))
PYEOF
}
echo "======== follow-up 3: A100 (a2-ultragpu-1g, on demand)"
bash "$R" a100_smoke ok PLATFORM=a100 FAKE_PLATFORM=a100 MACHINE_TYPE=a2-ultragpu-1g SPOT=0 \
  PLAN=harness/plans/a100_smoke.yaml WEIGHTS_GCS_URI=$W RESULTS_GCS_URI=$B
grep -E "Platform a100|zones tried in order|worst-case cost" "$WORK/state_a100_smoke/out" | sed 's/^/   | /'
check a100_smoke a100 us-central1-a
bash "$R" a100_zones ok PLATFORM=a100 FAKE_PLATFORM=a100 MACHINE_TYPE=a2-ultragpu-1g SPOT=0 \
  PLAN=harness/plans/a100_smoke.yaml WEIGHTS_GCS_URI=$W RESULTS_GCS_URI=$B FAKE_ZONES="us-central1-a:sync"
check a100_zones a100 us-central1-c
bash "$R" a100_east5 ok PLATFORM=a100 FAKE_PLATFORM=a100 MACHINE_TYPE=a2-ultragpu-1g SPOT=0 ZONES="us-east5-a us-east5-b" \
  PLAN=harness/plans/a100_smoke_t.yaml WEIGHTS_GCS_URI=$W RESULTS_GCS_URI=$B FAKE_ZONES="us-east5-a:sync"
grep -E "worst-case cost" "$WORK/state_a100_east5/out" | sed 's/^/   | /'
check a100_east5 a100 us-east5-b a100_triton
bash "$R" a100_40gb ok PLATFORM=a100 FAKE_PLATFORM=a100 FAKE_A100_GB=40 MACHINE_TYPE=a2-ultragpu-1g SPOT=0 \
  PLAN=harness/plans/a100_smoke.yaml WEIGHTS_GCS_URI=$W RESULTS_GCS_URI=$B
grep -rh "expected an A100 80 GB" "$WORK/state_a100_40gb" 2> /dev/null | head -1 | sed 's/^/   | /'
bash "$R" refuse_a100_a2-highgpu-1g ok PLATFORM=a100 MACHINE_TYPE=a2-highgpu-1g PLAN=harness/plans/a100_smoke.yaml
bash "$R" refuse_a100_no_machine_type ok PLATFORM=a100 PLAN=harness/plans/a100_smoke.yaml
bash "$R" refuse_a100_zone ok PLATFORM=a100 MACHINE_TYPE=a2-ultragpu-1g ZONES=us-east4-a PLAN=harness/plans/a100_smoke.yaml
for n in refuse_a100_a2-highgpu-1g refuse_a100_no_machine_type refuse_a100_zone; do
  grep -m1 '^!!' "$WORK/state_$n/out" | sed 's/^/   | /'
done
echo "======== follow-up 4: cpu60 (Spot)"
bash "$R" cpu60_part ok PLATFORM=cpu FAKE_PLATFORM=cpu SPOT=1 PLAN=harness/plans/cpu60/cpu60_7d5c.yaml \
  WEIGHTS_GCS_URI=$W RESULTS_GCS_URI=$B
check cpu60_part cpu60 europe-west4-a
