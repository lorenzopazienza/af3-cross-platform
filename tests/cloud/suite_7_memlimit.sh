#!/bin/bash
# Suite 7 (2026-10-14): the memory-limit plans (harness/plans/memlimit/, notes/analysis_plan.md section 10 row of
# 2026-10-14) launched as they will be run (official weights from a bucket, results bucket, on demand; v6e with
# PROVISIONING=standard and the main study's zones), on fake VMs whose run_alphafold.py runs out of memory
# (FAKE_OOM_BUCKET) or stalls (FAKE_SLOW_BUCKET) from a given bucket size. Checked per scenario ("memlimit check"):
# one fresh process per size in increasing order, each with AlphaFold3's --buckets=<size> in its command, recorded
# in session.json (af3_buckets) and run.json (af3_buckets, padded_bucket); the stop rule: the first size that fails
# or exceeds run_timeout_min ends the plan, its cause recorded (failure, failure_detail), every later size logged as
# not run, stop_rule.json correct; the launcher exits 0; the weights URI never printed; deadline and worst case.
. "$(dirname "$0")/common.sh"
R="$KIT/run.sh"
bash "$KIT/mkrepo.sh"
W=gs://private-weights/af3.bin.zst
B=gs://memlimit-results/study
V6_ZONES="europe-west4-a us-east5-b us-east5-a us-central1-b us-east1-d"
SIZES=1280,1536,2048,2560,3072,3584,4096,4608,5120
# A copy of the V6 plan whose processes time out after 6 s (the real plans allow 240 min), for the timeout case.
( cd "$WORK/repo" && sed -e 's/^run_timeout_min: 240$/run_timeout_min: 0.1/' -e 's/^description: .*/description: memlimit timeout test/' \
    harness/plans/memlimit/memlimit_v6.yaml > harness/plans/memlimit/test_timeout_v6.yaml \
  && git add -A && git -c user.name=test -c user.email=test@invalid commit -qm 'timeout test plan' )
memlimit() {   # NAME STOP_SIZE(or none) CAUSE(oom, timeout or none): checks the fetched session
  local s
  grep -E "follow deadline|VM lifetime limit|worst-case cost|Stop rule|Plan done" "$WORK/state_$1/out" | sed 's/^/   | /'
  echo "   weights URI printed: $(grep -c 'private-weights' "$WORK/state_$1/out") time(s)"
  s=$(ls -d "$WORK/state_$1/fetched/results/af3/"*/ 2> /dev/null | grep -v 20261008T165218Z_v6e_probe | tail -1)
  [ -n "$s" ] || { echo "   memlimit check: 1 problem(s): nothing fetched"; return; }
  python3 - "$s" "$2" "$3" "$SIZES" "$(grep -c 'private-weights' "$WORK/state_$1/out")" <<'PYEOF'
import json, pathlib, re, sys
s, stop, cause, sizes, leaks = pathlib.Path(sys.argv[1]), sys.argv[2], sys.argv[3], [int(x) for x in sys.argv[4].split(',')], int(sys.argv[5])
problems = []
log = [json.loads(l) for l in (s / 'plan_runs.jsonl').read_text().splitlines()]
ran = [r for r in log if r.get('harness_exit_code') is not None]
skipped = [r for r in log if r.get('not_run')]
if [r['buckets'] for r in log] != sizes:
  problems.append(f'plan_runs.jsonl sizes {[r.get("buckets") for r in log]}')
stop_i = sizes.index(int(stop)) if stop != 'none' else len(sizes)
if [r['buckets'] for r in ran] != sizes[:stop_i + 1][:len(sizes)]:
  problems.append(f'ran {[r["buckets"] for r in ran]}')
if [r['buckets'] for r in skipped] != sizes[stop_i + 1:]:
  problems.append(f'not run {[r["buckets"] for r in skipped]}')
for r in ran:
  d = s / r['run']
  sj = json.loads((d / 'session.json').read_text())
  rec = json.loads((d / '7D5C' / 'run.json').read_text())
  cmd = (d / '7D5C' / 'command.txt').read_text()
  if sj.get('af3_buckets') != [r['buckets']] or sj['args'].get('buckets') != str(r['buckets']) or sj['label'] != 'memlimit':
    problems.append(f'{r["run"]}: session.json af3_buckets {sj.get("af3_buckets")}, label {sj["label"]}')
  if f'--buckets={r["buckets"]}' not in cmd or rec.get('af3_buckets') != [r['buckets']]:
    problems.append(f'{r["run"]}: --buckets not in the command or run.json')
  if rec['seeds'] != [1] or rec['compile_cache']['mode'] != 'fresh' or not rec['compile_cache']['started_empty']:
    problems.append(f'{r["run"]}: seeds {rec["seeds"]}, cache {rec["compile_cache"]["mode"]}')
  last = stop != 'none' and r['buckets'] == int(stop)
  if not last:
    if r['harness_exit_code'] != 0 or rec['exit_code'] != 0 or rec['padded_bucket'] != r['buckets'] or rec['failure']:
      problems.append(f'{r["run"]}: exit {rec["exit_code"]}, padded {rec["padded_bucket"]}, failure {rec["failure"]}')
    for k in ('wall_seconds', 'compile', 'memory', 'inference_seconds_by_seed'):
      if not rec.get(k):
        problems.append(f'{r["run"]}: no {k} (M6, M7)')
  else:
    want = {'oom': 'RESOURCE_EXHAUSTED', 'timeout': 'killed by the harness'}[cause]
    if r.get('stop_rule') != 'stopped here' or rec['failure'] != cause or want not in (rec.get('failure_detail') or ''):
      problems.append(f'{r["run"]}: stop {r.get("stop_rule")}, failure {rec["failure"]}, detail {rec.get("failure_detail")}')
    if cause == 'timeout' and not rec['timed_out']:
      problems.append(f'{r["run"]}: not recorded as timed out')
sr = s / 'stop_rule.json'
if stop == 'none':
  if sr.exists():
    problems.append('stop_rule.json written although no size failed')
else:
  j = json.loads(sr.read_text())
  prev = sizes[stop_i - 1] if stop_i else None
  if (j['stopped_at_bucket'] != int(stop) or j['last_completed_bucket'] != prev or len(j['not_run']) != len(sizes) - stop_i - 1
      or j['processes'][0]['failure'] != cause):
    problems.append(f'stop_rule.json {j}')
if leaks:
  problems.append('weights URI printed')
print(f'   runs: ' + ', '.join(f'{r["buckets"]}:{"exit " + str(r["harness_exit_code"]) if r.get("harness_exit_code") is not None else "not run"}' for r in log))
print(f'   memlimit check: {len(problems)} problem(s)' + (': ' + '; '.join(problems[:4]) if problems else ''))
PYEOF
}
echo "======== memory-limit launches (official weights from a bucket, results bucket, on demand)"
bash "$R" memlimit_v5e_oom ok PLATFORM=v5e FAKE_PLATFORM=v5e SPOT=0 PLAN=harness/plans/memlimit/memlimit_v5.yaml \
  WEIGHTS_GCS_URI=$W RESULTS_GCS_URI=$B FAKE_OOM_BUCKET=2048
memlimit memlimit_v5e_oom 2048 oom
bash "$R" memlimit_v6e_oom ok PLATFORM=v6e FAKE_PLATFORM=v6e PROVISIONING=standard ZONES="$V6_ZONES" \
  PLAN=harness/plans/memlimit/memlimit_v6.yaml WEIGHTS_GCS_URI=$W RESULTS_GCS_URI=$B FAKE_OOM_BUCKET=3072
memlimit memlimit_v6e_oom 3072 oom
bash "$R" memlimit_g_all ok PLATFORM=l4 FAKE_PLATFORM=l4 SPOT=0 PLAN=harness/plans/memlimit/memlimit_g.yaml \
  WEIGHTS_GCS_URI=$W RESULTS_GCS_URI=$B
memlimit memlimit_g_all none none
bash "$R" memlimit_gt_first ok PLATFORM=l4 FAKE_PLATFORM=l4 SPOT=0 PLAN=harness/plans/memlimit/memlimit_gt.yaml \
  WEIGHTS_GCS_URI=$W RESULTS_GCS_URI=$B FAKE_OOM_BUCKET=1280
memlimit memlimit_gt_first 1280 oom
bash "$R" memlimit_v6e_timeout ok PLATFORM=v6e FAKE_PLATFORM=v6e PROVISIONING=standard ZONES="$V6_ZONES" \
  PLAN=harness/plans/memlimit/test_timeout_v6.yaml WEIGHTS_GCS_URI=$W RESULTS_GCS_URI=$B FAKE_SLOW_BUCKET=2560 FAKE_SLOW_S=30
memlimit memlimit_v6e_timeout 2560 timeout
