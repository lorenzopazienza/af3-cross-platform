#!/bin/bash
# Check 2 bit-identity on the finished main parts with a warm run, with harness/compare_runs.py --strict (only the
# mmCIF timestamp line masked). First, read-only copies from the results bucket of the files FETCH=light left
# there (full *_confidences.json and *_data.json) for the r1 seed 1 and w runs only. Then a scratch session per
# part holding only those two runs (target folders as symlinks), so compare_runs compares exactly w against r1.
# Only the strict verdicts are kept (strict_summary.txt); compare_runs' own text and JSON stay in SCRATCH.
# Usage, from the repository root: bash fetch_and_crosscheck_warm.sh SCRATCH SESSION:CONFIG [SESSION:CONFIG ...]
set -eu
W=$1; shift
O=results/analysis/main_monitor_20261013
B=gs://af2-tpu-benchmark-af3-results/af3
X='^(?!.*(_data\.json|(?<!_summary)_confidences\.json)$)'
OUT="${STRICT_OUT:-$O/strict_summary.txt}"; : > "$OUT"
for pair in "$@"; do
  s=${pair%%:*}; c=${pair#*:}
  for run in ${c}_rep1_seed1_fresh ${c}_seed1_warm; do
    gcloud storage rsync --recursive --exclude="$X" "$B/$s/$run" "results/af3/$s/$run" > /dev/null 2>&1
  done
  rm -rf "$W/$s"
  for run in ${c}_rep1_seed1_fresh ${c}_seed1_warm; do
    mkdir -p "$W/$s/$run"; cp "results/af3/$s/$run/session.json" "$W/$s/$run/"
    for t in "results/af3/$s/$run"/*/; do
      [ -d "$t/af3_output" ] && ln -s "$PWD/${t%/}" "$W/$s/$run/$(basename "$t")"
    done
  done
  python3 harness/compare_runs.py --session "$W/$s" --strict > /dev/null
  python3 - "$W/$s/compare_runs.json" "$s" >> "$OUT" <<'PY'
import json, sys
r = json.load(open(sys.argv[1]))
for c in r['comparisons']:
  s = c['strict']
  print(f"{sys.argv[2]} {c['label']}: {s['status']}, {len(s['identical'])}/{s['files']} files identical, "
        f"target files compared {s['target_files_compared']}, missing {len(s['missing'])}, differ {len(s['differ'])}")
PY
done
cat "$OUT"
