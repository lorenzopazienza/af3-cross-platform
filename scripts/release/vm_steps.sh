#!/bin/bash
# The VM part of the release build (RUNBOOK.md, steps 4 to 6): copy the released sessions from the bucket, check every
# *_data.json against its frozen input, build the archives made from the bucket, verify. Reads the bucket; writes only
# under $HOME (the VM has a read-only storage scope). Stops at the first failure. Run detached, so that a lost SSH
# connection does not stop it:
#   nohup bash ~/repo/scripts/release/vm_steps.sh all > ~/vm_steps.log 2>&1 &
# One step at a time: vm_steps.sh copy | check | build
set -eu
set -o pipefail
BUCKET=gs://af2-tpu-benchmark-af3-results/af3
REPO=$HOME/repo
ROOT=$HOME/bucket/af3
OUT=$HOME/release
JOBS=${JOBS:-6}
# Sessions outside the freeze: records only, so their AlphaFold3 outputs are not copied.
OUTPUTS='.*(_model\.cif|_confidences\.json|_data\.json|_ranking_scores\.csv|TERMS_OF_USE\.md|run_alphafold\.log|\.npz)$'
export CLOUDSDK_STORAGE_PROCESS_COUNT=2 CLOUDSDK_STORAGE_THREAD_COUNT=8

copy_list() {  # copy_list FILE [gcloud storage rsync options]
  list=$1
  shift
  # shellcheck disable=SC2016
  xargs -P "$JOBS" -I{} sh -c 'b=$1; r=$2; shift 2; mkdir -p "$r/{}"; gcloud storage rsync --recursive "$@" "$b/{}" "$r/{}" > /dev/null 2>&1 || echo "FAILED {}"' \
    _ "$BUCKET" "$ROOT" "$@" < "$list"
}

copy() {
  date -u +'%FT%TZ copy: start'
  mkdir -p "$ROOT"
  { copy_list "$HOME/sessions_release.txt"; copy_list "$HOME/sessions_outside.txt" "--exclude=$OUTPUTS"; } | tee "$HOME/copy_failed.txt"
  if [ -s "$HOME/copy_failed.txt" ]; then
    echo "copy: $(wc -l < "$HOME/copy_failed.txt") session(s) failed; rerun this step (rsync copies only what is missing)"
    exit 1
  fi
  du -sh "$ROOT"
  echo "folders: $(find "$ROOT" -mindepth 1 -maxdepth 1 -type d | wc -l); expected: $(cat "$HOME/sessions_release.txt" "$HOME/sessions_outside.txt" | wc -l)"
  date -u +'%FT%TZ copy: done'
}

check() {
  date -u +'%FT%TZ check-inputs: start'
  python3 "$REPO/scripts/release/release.py" check-inputs --root "$ROOT" --repo "$REPO" --out "$OUT"
  date -u +'%FT%TZ check-inputs: done'
}

build() {
  date -u +'%FT%TZ build: start'
  archives=$(python3 "$REPO/scripts/release/release.py" archives --built_on vm)
  python3 "$REPO/scripts/release/release.py" build --root "$ROOT" --repo "$REPO" --out "$OUT" --archives "$archives"
  python3 "$REPO/scripts/release/release.py" verify --out "$OUT"
  du -sh "$OUT/R1" "$OUT/R2"
  date -u +'%FT%TZ build: done'
}

case "${1:-all}" in
  copy) copy ;;
  check) check ;;
  build) build ;;
  all) copy; check; build ;;
  *) echo "usage: vm_steps.sh [copy|check|build|all]"; exit 2 ;;
esac
