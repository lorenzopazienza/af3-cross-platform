#!/bin/bash
# Independent cross-check of check 2 bit-identity with harness/compare_runs.py --strict. For each session with a
# warm rerun, a scratch session holds only the r1 seed 1 run and the w run (target folders as symlinks), so that
# compare_runs compares exactly w against r1 and nothing else (no r1 against r2). Run from the repository root.
# Usage: bash crosscheck_compare_runs.sh SCRATCH_DIR
set -eu
W=$1
O=results/analysis/pilot2_20261012
mkdir -p "$O/crosscheck"
for pair in \
  20261010T215616Z_v5e_pilot:tpu_xla 20261012T115650Z_v6e_pilot:tpu_xla \
  20261011T190832Z_l4_pilot_l4_1:l4_xla 20261011T214316Z_l4_pilot_l4_2:l4_xla \
  20261012T025406Z_l4_pilot_l4d_1:l4_xla_det 20261012T065347Z_l4_pilot_l4d_2:l4_xla_det; do
  s=${pair%%:*}; c=${pair#*:}
  rm -rf "$W/$s"
  for run in ${c}_rep1_seed1_fresh ${c}_seed1_warm; do
    src="$PWD/results/af3/$s/$run"
    mkdir -p "$W/$s/$run"
    cp "$src/session.json" "$W/$s/$run/"
    for t in "$src"/*/; do
      [ -d "$t/af3_output" ] && ln -s "${t%/}" "$W/$s/$run/$(basename "$t")"
    done
  done
  python3 harness/compare_runs.py --session "$W/$s" --strict > /dev/null
  cp "$W/$s/compare_runs.txt" "$O/crosscheck/$s.txt"
  cp "$W/$s/compare_runs.json" "$O/crosscheck/$s.json"
done
