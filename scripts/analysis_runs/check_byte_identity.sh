#!/bin/bash
# Reruns analysis reproduce scripts and checks that they regenerate their outputs byte for byte. For each NAME:
# snapshot the output folder, run scripts/analysis_runs/<script>, compare every file except logs (*.log, whose
# timings differ), report, and restore the snapshot if anything differs (the frozen outputs are never left changed).
# Run from the repository root:
#   bash scripts/analysis_runs/check_byte_identity.sh [NAME ...]   (default: primary_v2 glycans exploratory ppi embed)
set -u
SNAP="${TMPDIR:-/tmp}/af3_byte_identity_$$"
mkdir -p "$SNAP"
run_one() {   # NAME OUTPUT_DIR SCRIPT
  local name=$1 out=$2 script=$3 t0 t1 nfiles diffs
  rm -rf "$SNAP/$name"; cp -a "$out" "$SNAP/$name"
  t0=$(date +%s)
  if ! bash "$script" > "$SNAP/$name.stdout" 2>&1; then
    echo "$name: FAILED to run $script (see $SNAP/$name.stdout); outputs restored"
    rm -rf "$out"; cp -a "$SNAP/$name" "$out"; return 1
  fi
  t1=$(date +%s)
  nfiles=$(cd "$SNAP/$name" && find . -type f ! -name '*.log' | wc -l | tr -d ' ')
  diffs=$(diff -rq -x '*.log' "$SNAP/$name" "$out")
  if [ -z "$diffs" ]; then
    echo "$name: byte-identical, $nfiles files (logs excluded), $((t1 - t0)) s, $script"
  else
    echo "$name: DIFFERS ($script); outputs restored from the snapshot:"; echo "$diffs" | head -20 | sed 's/^/   /'
    rm -rf "$out"; cp -a "$SNAP/$name" "$out"
  fi
}
M=results/analysis/main_20261014
S=scripts/analysis_runs
for n in ${@:-primary_v2 glycans exploratory ppi embed}; do
  case $n in
    primary_v2) run_one primary_v2 "$M/primary_v2" "$S/main_20261014/primary_v2/reproduce.sh" ;;
    glycans) run_one glycans "$M/glycans" "$S/main_20261014/glycans/reproduce.sh" ;;
    exploratory) run_one exploratory "$M/exploratory" "$S/main_20261014/exploratory/reproduce.sh" ;;
    ppi) run_one ppi results/analysis/ppi_20261015 "$S/ppi_20261015/reproduce.sh" ;;
    embed) run_one embed results/analysis/embed_20261015 "$S/embed_20261015/reproduce.sh" ;;
    paper_assets) run_one paper_assets results/paper_assets "$S/paper_assets/reproduce.sh" ;;
    *) echo "unknown: $n" ;;
  esac
done
