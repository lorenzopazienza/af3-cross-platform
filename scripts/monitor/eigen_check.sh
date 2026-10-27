#!/bin/bash
# Validation of the Eigen archive (cloud/eigen_pin.sh), run by Lorenzo on the Mac from the repository root after the
# eigen_check sessions (harness/plans/eigen_check/, launched with FETCH=full):
#
#   bash scripts/monitor/eigen_check.sh v5e|v6e NEW_SESSION
#
# The eigen_check plan is the follow-up 6 smoke plan of the platform with only eigen_source: archive added, so its 2PV7
# output must be bit-identical to that smoke session's, whose build cloned Eigen from gitlab.com (TPU compiles
# deterministically: RC(V5) and RC(V6) are 0 and the cross-session check found V5 and V6 identical across VMs).
# Reference sessions: v5e 20261026T134939Z_v5e_msa_smoke_v5e, v6e 20261026T142437Z_v6e_msa_smoke_v6e (follow-up 6
# smoke rows of section 10). They were fetched light, so their full outputs are copied once, read-only, from the
# results bucket into results/eigen_check_ref/ (never into results/af3/).
# Checks, in order (any failure: FAIL, exit 1): the new session ran with EIGEN_SOURCE=archive and its eigen.json
# holds the pinned SHA-256 and commit; the reference ran without it (Eigen cloned by the build); the new session holds
# the full outputs (FETCH=full); harness/compare_runs.py --strict on 2PV7 (every output file byte for byte, only the
# mmCIF timestamp line masked) reports identical, with all 5 samples' coordinates identical. Prints PASS or FAIL.
set -u
PLATFORM=${1:-} NEW=${2:-}
case "$PLATFORM" in
  v5e) REF=20261026T134939Z_v5e_msa_smoke_v5e ;;
  v6e) REF=20261026T142437Z_v6e_msa_smoke_v6e ;;
  *) echo "usage: eigen_check.sh v5e|v6e NEW_SESSION" >&2; exit 2 ;;
esac
[ -n "$NEW" ] || { echo "usage: eigen_check.sh v5e|v6e NEW_SESSION" >&2; exit 2; }
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
source "$REPO/cloud/eigen_pin.sh"
BUCKET=${EIGEN_CHECK_BUCKET:-gs://af2-tpu-benchmark-af3-results}
RUN=tpu_xla_rep1_seed1_fresh
S=results/af3/$NEW
REFRUN=results/eigen_check_ref/$REF/$RUN
fail() { echo "FAIL: $*"; exit 1; }
rec() { sed -n "s/^$1=//p" "$2" | tail -1; }

[ -d "$S/$RUN/2PV7/af3_output/2PV7" ] || fail "no 2PV7 output in $S/$RUN"
[ "$(rec EIGEN_SOURCE "$S/launch_record.txt")" = "archive" ] || fail "$NEW did not run with EIGEN_SOURCE=archive"
[ -f "$S/eigen.json" ] || fail "no eigen.json in $NEW"
python3 - "$S/eigen.json" "$EIGEN_ARCHIVE_SHA256" "$EIGEN_COMMIT" <<'PY' || fail "eigen.json does not hold the pinned archive"
import json, sys
j = json.load(open(sys.argv[1]))
sys.exit(0 if (j.get('source'), j.get('archive_sha256'), j.get('commit')) == ('archive', sys.argv[2], sys.argv[3]) else 1)
PY
[ -z "$(rec EIGEN_SOURCE "results/af3/$REF/launch_record.txt" 2> /dev/null)" ] || fail "the reference $REF did not clone Eigen"
[ "$(rec FETCH "$S/launch_record.txt")" = "full" ] || fail "$NEW was fetched light: relaunch with FETCH=full (the comparison needs every file)"
if [ ! -d "$REFRUN/2PV7/af3_output/2PV7" ]; then
  echo ">> copying the reference's full 2PV7 output from the bucket (read-only) into $REFRUN"
  mkdir -p "$REFRUN/2PV7"
  gcloud storage cp -r "$BUCKET/af3/$REF/$RUN/2PV7/af3_output" "$REFRUN/2PV7/" > /dev/null || fail "could not copy the reference outputs"
fi
python3 "$REPO/harness/compare_runs.py" --session "$S" --reference "$REFRUN" --strict --out eigen_check
python3 - "$S/eigen_check.json" <<'PY' || fail "outputs differ from the reference (see $S/eigen_check.txt)"
import json, sys
r = json.load(open(sys.argv[1]))
c = [x for x in r['comparisons'] if x['label'].endswith(' 2PV7') and x['reference'].find('eigen_check_ref') >= 0]
ok = len(c) == 1 and c[0]['strict']['status'] == 'identical' and c[0]['summary']['coordinates_identical'] == 5 \
    and c[0]['summary']['samples_compared_bitwise'] == 5
sys.exit(0 if ok else 1)
PY
echo "PASS: $PLATFORM 2PV7 with the Eigen archive is bit-identical to $REF (every output file; $S/eigen_check.txt)"
