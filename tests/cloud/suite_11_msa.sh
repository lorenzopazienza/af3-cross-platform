#!/bin/bash
# Suite 11 (2026-10-25): follow-up 6 launcher support (notes/analysis_plan.md section 10, row of 2026-10-19; label
# msa), on fake VMs. The plans are follow-up 6 plans cut to one scratch-repository target (7U3J, seed 1) on the
# frozen inputs, since the official _data.json files are not in the scratch repository:
#   msa_dates          a V5o plan: the launch dates are printed (launch by, finish by), the part runs as usual.
#   msa_rep2           a Go r2 plan (rep_start: 2): the only harness run is l4_xla_rep2_seed1_fresh.
#   refuse_msa_late    launch_by in the past: refused before any VM, with the date in the message.
. "$(dirname "$0")/common.sh"
R="$KIT/run.sh"
bash "$KIT/mkrepo.sh"
W=gs://private-weights/af3.bin.zst
B=gs://followup-results/study
cut() {   # SOURCE DEST [sed expressions]: a follow-up 6 plan on 7U3J, seed 1, the frozen inputs
  local src=$1 dest=$2; shift 2
  sed -e 's/^  default: \[.*\]$/  default: [7U3J]/' -e 's/^seeds: .*/seeds: [1]/' -e '/^manifest: /d' \
    -e 's/^warm_rerun_seed: .*/warm_rerun_seed: none/' "$@" "$WORK/repo/$src" > "$WORK/repo/$dest"
}
# Positive scenarios with launch_by far ahead, so they keep passing after 2026-11-02.
cut harness/plans/msa/msa_v5o_01.yaml harness/plans/msa/msa_test_v5o.yaml -e 's/^launch_by: .*/launch_by: 2099-12-31/'
cut harness/plans/msa/msa_go_r2_01.yaml harness/plans/msa/msa_test_go_r2.yaml -e 's/^launch_by: .*/launch_by: 2099-12-31/'
cut harness/plans/msa/msa_v5o_01.yaml harness/plans/msa/msa_test_late.yaml -e 's/^launch_by: .*/launch_by: 2020-01-01/'
(cd "$WORK/repo" && git add -A && git -c user.name=test -c user.email=test@invalid commit -qm msa-test)
echo "======== follow-up 6: plan dates and rep_start"
bash "$R" msa_dates ok PLATFORM=v5e FAKE_PLATFORM=v5e SPOT=0 PLAN=harness/plans/msa/msa_test_v5o.yaml \
  WEIGHTS_GCS_URI=$W RESULTS_GCS_URI=$B
grep -E "dates:|worst-case cost" "$WORK/state_msa_dates/out" | sed 's/^/   | /'
bash "$R" msa_rep2 ok PLATFORM=l4 FAKE_PLATFORM=l4 SPOT=0 ZONES=europe-west4-a PLAN=harness/plans/msa/msa_test_go_r2.yaml \
  WEIGHTS_GCS_URI=$W RESULTS_GCS_URI=$B
bash "$R" refuse_msa_late ok PLATFORM=v5e PLAN=harness/plans/msa/msa_test_late.yaml WEIGHTS_GCS_URI=$W
grep -m1 '^!!' "$WORK/state_refuse_msa_late/out" | sed 's/^/   | /'
python3 - "$WORK" <<'PYEOF'
import pathlib, sys
w = pathlib.Path(sys.argv[1])
problems = []
out = (w / 'state_msa_dates' / 'out').read_text()
if 'dates: launch by 2099-12-31; a part not finished by 2026-11-08 is reported as incomplete' not in out:
  problems.append('dates not printed')
runs = sorted(p.name for p in (w / 'state_msa_rep2' / 'fetched' / 'results' / 'af3').glob('*_l4_msa_test_go_r2/*')
              if (p / 'session.json').is_file())
if runs != ['l4_xla_rep2_seed1_fresh']:
  problems.append(f'rep2 runs {runs}')
late = (w / 'state_refuse_msa_late' / 'out').read_text()
if 'may be launched only by 2020-01-01' not in late or (w / 'state_refuse_msa_late' / 'create_args').exists():
  problems.append('late launch not refused before any VM')
print(f'   followup check: {len(problems)} problem(s)' + (': ' + '; '.join(problems) if problems else ''))
PYEOF
