#!/bin/bash
# Memory limit on A and H (label memlimit_ah; drafts/section10_memlimit_ah.md, the specification row): launch gate and
# budget rule, run once before the launch of the two plans (one wrapper run, both parts).
#   bash scripts/monitor/credit_check_memlimit_ah.sh BALANCE_USD
# GO only if all of these hold:
#   1 start condition, either of:
#     a the project's GPUS_ALL_REGIONS quota is at least 10, read now (read-only) with
#       gcloud beta quotas info describe GPUS-ALL-REGIONS-per-project --service=compute.googleapis.com
#         --project=af2-tpu-benchmark
#       (its global value; `gcloud compute project-info describe` does not list it), so A and H (one GPU each) run
#       beside follow-up 13's eight L4 launchers without taking any of their quota; or
#     b follow-up 13's Gf arm has finished (the chain ledger records gf as "ok" or "stopped") and no L4 arm of
#       follow-up 13 runs (no session <stamp>_l4_why_<lg|gf>_NN whose launcher_events.txt lacks "launcher: exit");
#   2 the Google TPU Builders credit balance (BALANCE_USD, read from the console), minus the expected cost of every arm
#     still running (follow-ups 7 to 13) and of every follow-up 13 arm the ledger does not yet record as ok or stopped
#     (started or not: with start condition a this measurement can start before them), minus the expected cost of this
#     measurement, is at least 25 USD;
#   3 this measurement's spend so far plus its expected cost is within its cap of 200 USD (the worst case of
#     harness/plans/memlimit_ah/memlimit_ah_parts.csv, 199.84 USD);
#   4 the launch is on or before 2026-12-02 (UTC) and the expected end by 2026-12-04 24:00 UTC.
# Expected and worst cost: harness/plans/memlimit_ah/memlimit_ah_parts.csv (A on demand, H Spot); of a running arm of
# another follow-up: harness/plans/fu13_parts.csv, fu10_parts.csv, fu11_12_parts.csv, fu7_10_parts.csv.
# Record: every run appends one line to GATE_LOG (default results/memlimit_ah_gate.tsv, outside git): UTC time, the
# GPUS_ALL_REGIONS value read (or "unread"), Gf's ledger status, live follow-up 13 L4 sessions, the balance given and
# the answer; the launch row of section 10 copies the line of the launch.
# Tests: LEDGER (default ~/Desktop/af3_drafts/chain_ledger.tsv), NOW=YYYY-MM-DDTHH:MM:SSZ, GCLOUD (default gcloud),
# RESULTS_AF3 (default results/af3) and GATE_LOG replace the ledger, the clock, the gcloud command, the session folder
# and the record. Prints GO or NO GO with the reasons; exit 0 for GO, 1 for NO GO, 2 for a usage error. Reads local
# files and the one quota above (read-only); creates nothing in the cloud.
set -u
BAL=${1:-}
echo "$BAL" | grep -Eq '^[0-9]+(\.[0-9]+)?$' || { echo "usage: credit_check_memlimit_ah.sh BALANCE_USD (read from the console)" >&2; exit 2; }
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
python3 - "$BAL" "${NOW:-}" "${LEDGER:-$HOME/Desktop/af3_drafts/chain_ledger.tsv}" "$QUOTA" "${RESULTS_AF3:-results/af3}" \
  "${GATE_LOG:-results/memlimit_ah_gate.tsv}" <<'PY'
import csv, datetime as dt, pathlib, re, sys
bal, now_arg, ledger, quota, af3, gate_log = float(sys.argv[1]), sys.argv[2], pathlib.Path(sys.argv[3]), sys.argv[4], \
    pathlib.Path(sys.argv[5]), pathlib.Path(sys.argv[6])
QUOTA_MIN = 10
FLOOR, CAP, LABEL = 25.0, 200.0, 'memlimit_ah'
LAUNCH_BY, FINISH_BY = dt.datetime(2026, 12, 3), dt.datetime(2026, 12, 5)
now = dt.datetime.strptime(now_arg, '%Y-%m-%dT%H:%M:%SZ') if now_arg else \
    dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)
parts = list(csv.DictReader(open('harness/plans/memlimit_ah/memlimit_ah_parts.csv')))
est = sum(float(p['expected_usd']) for p in parts)
worst = sum(float(p['worst_usd']) for p in parts)
rate = {p['platform']: float(p['usd_per_h']) for p in parts}
expected = {}
for f in ('harness/plans/fu7_10_parts.csv', 'harness/plans/fu11_12_parts.csv', 'harness/plans/fu10_parts.csv',
          'harness/plans/fu13_parts.csv'):
  if pathlib.Path(f).is_file():
    for p in csv.DictReader(open(f)):
      k = p['arm'].lower().replace(' ', '_')
      expected[k] = expected.get(k, 0.0) + float(p['expected_usd_planned'])
FU13 = ('l5', 'l6', 'lg', 'v6f', 'p', 'gf')
status = {}
for l in (ledger.read_text().splitlines() if ledger.is_file() else []):
  f = l.split('\t')
  if len(f) > 3 and f[2] in FU13:
    status.setdefault(f[2], []).append(f[3])
gf = status.get('gf', [])
gf_done = bool(gf) and gf[-1] in ('ok', 'stopped')
fu13_open = [a for a in FU13 if not (status.get(a) and status[a][-1] in ('ok', 'stopped'))]


def events(f):
  ev = {}
  for line in f.read_text().splitlines():
    m = re.match(r'(\S+)\s+(\w+): (\w+)', line)
    if m:
      ev.setdefault(f'{m.group(2)}:{m.group(3)}', dt.datetime.strptime(m.group(1), '%Y-%m-%dT%H:%M:%SZ'))
  return ev


spent, n, running, l4_why, seen = 0.0, 0, {}, [], set()
for s in sorted(af3.glob('*')) + sorted((af3 / '.launcher').glob('*')):
  f = s / 'launcher_events.txt'
  if not f.is_file() or s.name in seen:
    continue
  seen.add(s.name)
  ev = events(f)
  live = 'launcher:exit' not in ev
  m = re.match(r'\d{8}T\d{6}Z_(v5e|v6e|l4|h100|a100)_(why|large|repl|prec5|h100|seeds|sw)_(.+)_\d\d$', s.name)
  if m and live:
    running.setdefault(m.group(3), []).append(s.name)
    if m.group(1) == 'l4' and m.group(2) == 'why':
      l4_why.append(s.name)
  m = re.match(r'\d{8}T\d{6}Z_(a100|h100)_memlimit_ah_[ah]_01$', s.name)
  if not m:
    continue
  start = next((v for k, v in ev.items() if k.startswith('vm:')), None)
  if start is None:
    continue
  end = ev.get('delete:done') or ev.get('delete:the') or ev.get('launcher:exit') or now
  spent += (end - start).total_seconds() / 3600 * rate[m.group(1)]
  n += 1
others = {a: expected.get(a, 0.0) for a in set(running) | set(fu13_open)}
left = bal - sum(others.values()) - est
hours = max(float(p['expected_vm_h']) for p in parts)        # the two plans run at once, one VM each
end = now + dt.timedelta(hours=hours)
print(f'memlimit_ah (A on demand, H Spot): expected cost {est:.2f} USD (worst case {worst:.2f}); TPU Builders balance {bal:.2f} USD')
quota_ok = quota != 'unread' and int(quota) >= QUOTA_MIN
print(f'GPUS_ALL_REGIONS read at {now:%Y-%m-%dT%H:%M:%SZ}: {quota} (start condition a: at least {QUOTA_MIN})')
print(f'follow-up 13 Gf in the ledger {ledger}: {gf[-1] if gf else "absent"}; L4 sessions of follow-up 13 still running: '
      f'{len(l4_why)}{" (" + ", ".join(l4_why[:3]) + ")" if l4_why else ""} (start condition b: Gf finished, none running)')
for a, usd in sorted(others.items()):
  print(f'{"running" if a in running else "not yet finished (follow-up 13)"} arm {a}'
        f'{f" ({len(running[a])} session(s))" if a in running else ""}: expected {usd:.2f} USD subtracted')
print(f'balance minus running arms minus expected {left:.2f} USD (floor {FLOOR:.0f}); spend so far from {n} session(s) '
      f'{spent:.2f} USD, with this measurement {spent + est:.2f} (cap {CAP:.0f})')
print(f'expected duration {hours:.1f} h: to about {end:%Y-%m-%d %H:%M} UTC (launch by 2026-12-02, finish by 2026-12-04)')
reasons = []
start_b = gf_done and not l4_why
if not quota_ok and not start_b:
  reasons.append(f'start condition: GPUS_ALL_REGIONS {quota} < {QUOTA_MIN}, and '
                 + ('follow-up 13 Gf has not finished (ledger)' if not gf_done else 'an L4 arm of follow-up 13 is running'))
if left < FLOOR:
  reasons.append(f'balance minus running arms minus expected {left:.2f} < {FLOOR:.0f} USD')
if spent + est > CAP:
  reasons.append(f'spend with this measurement {spent + est:.2f} > {CAP:.0f} USD')
if now >= LAUNCH_BY:
  reasons.append('launch after 2026-12-02')
if end > FINISH_BY:
  reasons.append('its runs cannot finish by 2026-12-04')
answer = 'GO' if not reasons else 'NO GO: ' + '; '.join(reasons)
print(answer)
gate_log.parent.mkdir(parents=True, exist_ok=True)
with open(gate_log, 'a') as f:
  f.write(f'{now:%Y-%m-%dT%H:%M:%SZ}\tGPUS_ALL_REGIONS={quota}\tgf={gf[-1] if gf else "absent"}\t'
          f'live_l4_why={len(l4_why)}\tbalance={bal:.2f}\t{answer}\n')
sys.exit(0 if not reasons else 1)
PY
