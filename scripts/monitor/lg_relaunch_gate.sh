#!/bin/bash
# Follow-up 13, relaunch of LG after the fix of the L4 capture (drafts/section10_fu13_lg_fix.md): launch gate.
#   bash scripts/monitor/lg_relaunch_gate.sh BALANCE_USD
# LG needs one L4. With GPUS_ALL_REGIONS at 10, Gf's eight L4 launchers and the memory-limit measurement's A and H
# (one GPU each) fill the quota, so LG starts only when a GPU is free, that is when either:
#   a the memory-limit measurement has ended: each of its two parts (memlimit_ah_a_01 on a100, memlimit_ah_h_01 on
#     h100) has a session, and no session of label memlimit_ah still runs (launcher_events.txt without
#     "launcher: exit"); Gf's at most eight L4 VMs plus LG then use at most 9 GPUs; or
#   b follow-up 13's Gf has finished: the chain ledger records gf as "ok" or "stopped" and no Gf session still runs.
# Then follow-up 13's budget rule for the arm: scripts/monitor/credit_check_fu13.sh lg BALANCE_USD.
# Prints GO or NO GO with the reasons, appends one line to GATE_LOG (default results/lg_relaunch_gate.tsv, outside
# git); exit 0 for GO, 1 for NO GO, 2 for a usage error. Reads only local files. Tests: LEDGER, RESULTS_AF3, NOW,
# CREDIT_CHECK (default scripts/monitor/credit_check_fu13.sh) and GATE_LOG replace the ledger, the session folder,
# the clock, the budget script and the record.
set -u
BAL=${1:-}
echo "$BAL" | grep -Eq '^[0-9]+(\.[0-9]+)?$' || { echo "usage: lg_relaunch_gate.sh BALANCE_USD (read from the console)" >&2; exit 2; }
cd "$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
GPU=$(python3 - "${LEDGER:-$HOME/Desktop/af3_drafts/chain_ledger.tsv}" "${RESULTS_AF3:-results/af3}" <<'PY'
import pathlib, re, sys
ledger, af3 = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
live, have = {'memlimit_ah': 0, 'gf': 0}, set()
for s in sorted(af3.glob('*')) + sorted((af3 / '.launcher').glob('*')):
  f = s / 'launcher_events.txt'
  if not f.is_file():
    continue
  running = 'launcher: exit' not in f.read_text()
  m = re.match(r'\d{8}T\d{6}Z_(a100|h100)_memlimit_ah_([ah])_01$', s.name)
  if m:
    have.add(m.group(2))
    live['memlimit_ah'] += running
  if re.match(r'\d{8}T\d{6}Z_l4_why_gf_\d\d$', s.name):
    live['gf'] += running
gf = [l.split('\t')[3] for l in (ledger.read_text().splitlines() if ledger.is_file() else [])
      if len(l.split('\t')) > 3 and l.split('\t')[2] == 'gf']
a = have == {'a', 'h'} and live['memlimit_ah'] == 0
b = bool(gf) and gf[-1] in ('ok', 'stopped') and live['gf'] == 0
why = (f'memory limit A and H: sessions of {sorted(have) or "none"}, {live["memlimit_ah"]} running; '
       f'Gf: ledger {gf[-1] if gf else "absent"}, {live["gf"]} session(s) running')
print(('free ' if a or b else 'busy ') + why)
PY
)
echo ">> GPU for LG: $GPU"
ANSWER="NO GO: no GPU free for LG under GPUS_ALL_REGIONS 10 (wait for A and H to end, or for Gf)"
RC=1
if [ "${GPU%% *}" = free ]; then
  if bash "${CREDIT_CHECK:-scripts/monitor/credit_check_fu13.sh}" lg "$BAL"; then ANSWER=GO; RC=0
  else ANSWER="NO GO: budget rule of follow-up 13 (above)"; fi
fi
echo "$ANSWER"
LOG=${GATE_LOG:-results/lg_relaunch_gate.tsv}
mkdir -p "$(dirname "$LOG")"
printf '%s\t%s\tbalance=%s\t%s\n' "${NOW:-$(date -u +%Y-%m-%dT%H:%M:%SZ)}" "$GPU" "$BAL" "$ANSWER" >> "$LOG"
exit $RC
