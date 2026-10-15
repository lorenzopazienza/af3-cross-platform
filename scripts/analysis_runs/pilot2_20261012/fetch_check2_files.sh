#!/bin/bash
# Read-only copies from the results bucket of the files check 2 bit-identity needs and FETCH=light left there:
# the full *_confidences.json (not *_summary_confidences.json) and *_data.json of the r1 seed 1 and w runs of
# V5, G and Gd. Nothing else is copied; nothing is written to the bucket. Run from the repository root.
set -eu
B=gs://af2-tpu-benchmark-af3-results/af3
X='^(?!.*(_data\.json|(?<!_summary)_confidences\.json)$)'
for pair in \
  20261010T215616Z_v5e_pilot:tpu_xla 20261012T115650Z_v6e_pilot:tpu_xla \
  20261011T190832Z_l4_pilot_l4_1:l4_xla 20261011T214316Z_l4_pilot_l4_2:l4_xla \
  20261012T025406Z_l4_pilot_l4d_1:l4_xla_det 20261012T065347Z_l4_pilot_l4d_2:l4_xla_det; do
  s=${pair%%:*}; c=${pair#*:}
  for run in ${c}_rep1_seed1_fresh ${c}_seed1_warm; do
    echo ">> $s/$run"
    gcloud storage rsync --recursive --exclude="$X" "$B/$s/$run" "results/af3/$s/$run"
  done
done
