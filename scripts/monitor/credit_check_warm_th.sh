#!/bin/bash
# Warm cost on GT and H (label warm_th; drafts/section10_warm_th_20261111.md, the specification row): launch gate and
# budget rule, run immediately before the launch of each arm (the two arms are launched in parallel, one wrapper each).
#   bash scripts/monitor/credit_check_warm_th.sh ARM BALANCE_USD        (ARM: gt or h)
# GO only if all of these hold:
#   1 GPU quota: the project's GPUS_ALL_REGIONS quota, read now (read-only) with
#       gcloud beta quotas info describe GPUS-ALL-REGIONS-per-project --service=compute.googleapis.com
#         --project=af2-tpu-benchmark
#     (its global value), is at least the GPU sessions of other launchers still running (l4, a100 or h100 sessions
#     whose launcher_events.txt lacks "launcher: exit" and whose name stamp is less than 7 days old: two pilot
#     launchers of 2026-10-12 never wrote that line) plus 2 (one GPU per warm_th arm); if the quota cannot be read,
#     only when no GPU session of another launcher is running (the 8 GPUs read on 2026-10-27 then hold both arms);
#   2 the Google TPU Builders credit balance (BALANCE_USD, read from the console), minus the expected cost of every arm
#     of another follow-up still running (follow-ups 7 to 13, memory limit on A and H) and of every follow-up 13 arm
#     the chain ledger does not yet record as ok or stopped, minus the expected cost of every warm_th arm not yet
#     finished (this one and, if it has no finished session, the other), is at least 25 USD;
#   3 this measurement's spend so far plus the expected cost of its unfinished arms is within its cap of 30 USD (about
#     twice the worst case of harness/plans/warm_th/warm_th_parts.csv, so one unchanged relaunch of an interrupted
#     part fits);
#   4 the launch is on or before 2026-11-12 (UTC) and the arm's expected end by 2026-11-13 24:00 UTC.
# Expected and worst cost: harness/plans/warm_th/warm_th_parts.csv (GT on demand, H Spot); of a running arm of another
# follow-up: harness/plans/fu13_parts.csv, fu10_parts.csv, fu11_12_parts.csv, fu7_10_parts.csv and, when present,
# harness/plans/memlimit_ah/memlimit_ah_parts.csv. A warm_th arm is finished when a session of it has
# "launcher: exit 0".
# Record: every run appends one line to GATE_LOG (default results/warm_th_gate.tsv, outside git): UTC time, the arm,
# the GPUS_ALL_REGIONS value read (or "unread"), the GPU sessions of other launchers running, the balance given and the
# answer; the launch row of section 10 copies the line of each launch.
# Tests: LEDGER (default ~/Desktop/af3_drafts/chain_ledger.tsv), NOW=YYYY-MM-DDTHH:MM:SSZ, GCLOUD (default gcloud),
# RESULTS_AF3 (default results/af3) and GATE_LOG replace the ledger, the clock, the gcloud command, the session folder
# and the record. Prints GO or NO GO with the reasons; exit 0 for GO, 1 for NO GO, 2 for a usage error. Reads local
# files and the one quota above (read-only); creates nothing in the cloud.
set -u
ARM=${1:-}
BAL=${2:-}
case "$ARM" in gt | h) ;; *) echo "usage: credit_check_warm_th.sh gt|h BALANCE_USD (read from the console)" >&2; exit 2 ;; esac
echo "$BAL" | grep -Eq '^[0-9]+(\.[0-9]+)?$' || { echo "usage: credit_check_warm_th.sh gt|h BALANCE_USD (read from the console)" >&2; exit 2; }
cd "$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
QUOTA=$(${GCLOUD:-gcloud} beta quotas info describe GPUS-ALL-REGIONS-per-project --service=compute.googleapis.com \
  --project=af2-tpu-benchmark --format=json 2> /dev/null | python3 -c '
import json, sys
try:
  d = json.load(sys.stdin)
  v = [x["details"]["value"] for x in d.get("dimensionsInfos", []) if "global" in x.get("applicableLocations", [])]
  print(int(v[0]) if v else "unread")
except Exception:
  print("unread")')
python3 - "$ARM" "$BAL" "${NOW:-}" "${LEDGER:-$HOME/Desktop/af3_drafts/chain_ledger.tsv}" "$QUOTA" \
  "${RESULTS_AF3:-results/af3}" "${GATE_LOG:-results/warm_th_gate.tsv}" <<'PY'
import csv, datetime as dt, pathlib, re, sys
arm, bal, now_arg, ledger, quota, af3, gate_log = sys.argv[1], float(sys.argv[2]), sys.argv[3], pathlib.Path(sys.argv[4]), \
    sys.argv[5], pathlib.Path(sys.argv[6]), pathlib.Path(sys.argv[7])
FLOOR, CAP, LABEL, OWN_GPUS, STALE_DAYS = 25.0, 30.0, 'warm_th', 2, 7
LAUNCH_BY, FINISH_BY = dt.datetime(2026, 11, 13), dt.datetime(2026, 11, 14)
now = dt.datetime.strptime(now_arg, '%Y-%m-%dT%H:%M:%SZ') if now_arg else \
    dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)
parts = {p['arm'].lower(): p for p in csv.DictReader(open('harness/plans/warm_th/warm_th_parts.csv'))}
rate = {p['platform']: float(p['usd_per_h']) for p in parts.values()}
expected = {}
for f, col in (('harness/plans/fu7_10_parts.csv', 'expected_usd_planned'), ('harness/plans/fu11_12_parts.csv', 'expected_usd_planned'),
               ('harness/plans/fu10_parts.csv', 'expected_usd_planned'), ('harness/plans/fu13_parts.csv', 'expected_usd_planned'),
               ('harness/plans/memlimit_ah/memlimit_ah_parts.csv', 'expected_usd')):
  if pathlib.Path(f).is_file():
    for p in csv.DictReader(open(f)):
      k = ('memlimit_ah_' if 'memlimit_ah' in f else '') + p['arm'].lower().replace(' ', '_')
      expected[k] = expected.get(k, 0.0) + float(p[col])
FU13 = ('l5', 'l6', 'lg', 'v6f', 'p', 'gf')
status = {}
for l in (ledger.read_text().splitlines() if ledger.is_file() else []):
  f = l.split('\t')
  if len(f) > 3 and f[2] in FU13:
    status.setdefault(f[2], []).append(f[3])
fu13_open = [a for a in FU13 if not (status.get(a) and status[a][-1] in ('ok', 'stopped'))]


def events(f):
  ev = {}
  for line in f.read_text().splitlines():
    m = re.match(r'(\S+)\s+(\w+): (\w+)', line)
    if m:
      ev.setdefault(f'{m.group(2)}:{m.group(3)}', dt.datetime.strptime(m.group(1), '%Y-%m-%dT%H:%M:%SZ'))
  return ev


spent, n, running, gpu_other, finished, own_live, seen = 0.0, 0, {}, [], set(), set(), set()
for s in sorted(af3.glob('*')) + sorted((af3 / '.launcher').glob('*')):
  f = s / 'launcher_events.txt'
  if not f.is_file() or s.name in seen:
    continue
  seen.add(s.name)
  text = f.read_text()
  ev = events(f)
  started = re.match(r'(\d{8}T\d{6}Z)_', s.name)
  recent = bool(started) and now - dt.datetime.strptime(started.group(1), '%Y%m%dT%H%M%SZ') < dt.timedelta(days=STALE_DAYS)
  live = 'launcher:exit' not in ev and recent
  own = re.match(r'\d{8}T\d{6}Z_(l4|h100)_warm_th_(gt|h)_01$', s.name)
  m = re.match(r'\d{8}T\d{6}Z_(cpu|v5e|v6e|l4|h100|a100)_(.+)$', s.name)
  if m and live and not own:
    if m.group(1) in ('l4', 'a100', 'h100'):
      gpu_other.append(s.name)
    k = re.match(r'(why|large|repl|prec5|h100|seeds|sw)_(.+)_\d\d$', m.group(2))
    k = k.group(2) if k else re.sub(r'_\d\d$', '', m.group(2))
    running.setdefault(k, []).append(s.name)
  if not own:
    continue
  if re.search(r'launcher: exit 0\b', text):
    finished.add(own.group(2))
  if live:
    own_live.add(own.group(2))
  start = next((v for k, v in ev.items() if k.startswith('vm:')), None)
  if start is None:
    continue
  end = ev.get('delete:done') or ev.get('delete:the') or ev.get('launcher:exit') or now
  spent += (end - start).total_seconds() / 3600 * rate[own.group(1)]
  n += 1
others = {a: expected.get(a, 0.0) for a in set(running) | set(fu13_open)}
todo = [a for a in parts if a not in finished and (a == arm or a not in own_live)]
todo_usd = sum(float(parts[a]['expected_usd']) for a in todo)
left = bal - sum(others.values()) - todo_usd
p = parts[arm]
end = now + dt.timedelta(hours=float(p['expected_vm_h']))
print(f'warm_th arm {arm} ({p["platform"]}, {p["provisioning"]}): expected cost {float(p["expected_usd"]):.2f} USD (worst case '
      f'{float(p["worst_usd"]):.2f}); TPU Builders balance {bal:.2f} USD')
quota_ok = (len(gpu_other) + OWN_GPUS <= int(quota)) if quota != 'unread' else not gpu_other
print(f'GPUS_ALL_REGIONS read at {now:%Y-%m-%dT%H:%M:%SZ}: {quota}; GPU sessions of other launchers running: '
      f'{len(gpu_other)}{" (" + ", ".join(gpu_other[:3]) + ")" if gpu_other else ""}; needed {len(gpu_other) + OWN_GPUS}')
for a, usd in sorted(others.items()):
  print(f'{"running" if a in running else "not yet finished (follow-up 13)"} arm {a}'
        f'{f" ({len(running[a])} session(s))" if a in running else ""}: expected {usd:.2f} USD subtracted')
print(f'warm_th arms not yet finished, counted: {", ".join(todo) or "none"} ({todo_usd:.2f} USD); finished: '
      f'{", ".join(sorted(finished)) or "none"}; running: {", ".join(sorted(own_live)) or "none"}')
print(f'balance minus running arms minus expected {left:.2f} USD (floor {FLOOR:.0f}); spend so far from {n} session(s) '
      f'{spent:.2f} USD, with the arms not yet finished {spent + todo_usd:.2f} (cap {CAP:.0f})')
print(f'expected duration {float(p["expected_vm_h"]):.1f} h: to about {end:%Y-%m-%d %H:%M} UTC (launch by 2026-11-12, '
      'finish by 2026-11-13)')
reasons = []
if arm in own_live:
  reasons.append(f'a session of arm {arm} is running')
if arm in finished:
  reasons.append(f'arm {arm} has a finished session (launcher: exit 0)')
if not quota_ok:
  reasons.append(f'GPU quota: GPUS_ALL_REGIONS {quota} < {len(gpu_other) + OWN_GPUS} needed' if quota != 'unread'
                 else 'GPUS_ALL_REGIONS unread and a GPU session of another launcher is running')
if left < FLOOR:
  reasons.append(f'balance minus running arms minus expected {left:.2f} < {FLOOR:.0f} USD')
if spent + todo_usd > CAP:
  reasons.append(f'spend with the arms not yet finished {spent + todo_usd:.2f} > {CAP:.0f} USD')
if now >= LAUNCH_BY:
  reasons.append('launch after 2026-11-12')
if end > FINISH_BY:
  reasons.append('its runs cannot finish by 2026-11-13')
answer = 'GO' if not reasons else 'NO GO: ' + '; '.join(reasons)
print(answer)
gate_log.parent.mkdir(parents=True, exist_ok=True)
with open(gate_log, 'a') as f:
  f.write(f'{now:%Y-%m-%dT%H:%M:%SZ}\t{arm}\tGPUS_ALL_REGIONS={quota}\tgpu_other_live={len(gpu_other)}\t'
          f'balance={bal:.2f}\t{answer}\n')
sys.exit(0 if not reasons else 1)
PY
