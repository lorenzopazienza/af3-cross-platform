#!/bin/bash
# Pilot Step 1 (notes/analysis_plan.md section 6, checks 1 to 3), from the repository root:
#   bash scripts/analysis_runs/pilot_20261010/reproduce_step1.sh
# Needs analysis/.venv (analysis/requirements.txt) and, for check 2 on G, the per-sample
# *_confidences.json of l4_xla_rep1_seed1_fresh and l4_xla_seed1_warm of both L4 pilot sessions
# (copied from the results bucket into their sample folders; see the summary). Reads local files only.
set -e
cd "$(dirname "$0")/../../.."
O=results/analysis/pilot_20261010
PY=analysis/.venv/bin/python

# Check 1: exit codes of every process of every arm (run records).
$PY - <<'EOF'
import json, pathlib
import pandas as pd
O = pathlib.Path('results/analysis/pilot_20261010')
S = {'V5': ['20261009T154308Z_v5e_pilot', '20261010T134201Z_v5e_v5e_warm_completion'], 'V6': ['20261009T154318Z_v6e_pilot'],
     'G': ['20261009T230055Z_l4_pilot_l4_1', '20261010T013052Z_l4_pilot_l4_2'],
     'C': ['20261009T154331Z_cpu_pilot_cpu_1', '20261009T172653Z_cpu_pilot_cpu_2', '20261009T172743Z_cpu_pilot_cpu_3',
           '20261009T172752Z_cpu_pilot_cpu_4'],
     'V6h': ['20261010T134219Z_v6e_pilot_v6h'], 'Gh': ['20261010T134240Z_l4_pilot_l4h_1', '20261010T150023Z_l4_pilot_l4h_2']}
rows = []
for arm, ss in S.items():
  for s in ss:
    root = pathlib.Path('results/af3') / s
    ec = (root / 'EXIT_CODE').read_text().strip() if (root / 'EXIT_CODE').exists() else 'absent'
    for run in sorted(p for p in root.iterdir() if (p / 'session.json').exists()):
      sj = json.loads((run / 'session.json').read_text())
      rr = [json.loads(l) for l in (run / 'runs.jsonl').read_text().splitlines()]
      for r in rr:
        rows.append({'arm': arm, 'session': s, 'session_label': sj['label'], 'run': run.name, 'target': r['pdb_id'],
                     'exit_code': r['exit_code'], 'failure': r.get('failure'), 'session_EXIT_CODE': ec})
      for t in sj['targets']:
        if t not in {r['pdb_id'] for r in rr}:
          rows.append({'arm': arm, 'session': s, 'session_label': sj['label'], 'run': run.name, 'target': t,
                       'exit_code': None, 'session_EXIT_CODE': ec,
                       'failure': 'not run: VM deleted by the launcher during the warm rerun (network incident, '
                                  'section 10 row of 2026-10-10)'})
pd.DataFrame(rows).to_csv(O / 'check1_exit_codes.csv', index=False)
EOF

# Check 2: harness/compare_runs.py on views holding only the source fresh run and its warm rerun.
for spec in V5_pilot:20261009T154308Z_v5e_pilot:tpu_xla V5_completion:20261010T134201Z_v5e_v5e_warm_completion:tpu_xla \
            V6_pilot:20261009T154318Z_v6e_pilot:tpu_xla G_part1:20261009T230055Z_l4_pilot_l4_1:l4_xla \
            G_part2:20261010T013052Z_l4_pilot_l4_2:l4_xla; do
  IFS=: read n s c <<< "$spec"
  mkdir -p $O/check2/$n
  for r in ${c}_rep1_seed1_fresh ${c}_seed1_warm; do ln -sfn ../../../../af3/$s/$r $O/check2/$n/$r; done
  python3 harness/compare_runs.py --session $O/check2/$n --out compare_runs_check2 > /dev/null
done
# ... and the M4 bit-identical flag as implemented (analysis/contrasts.py coords_text), with the compile records.
$PY - <<'EOF'
import json, pathlib, sys
sys.path.insert(0, 'analysis'); sys.path.insert(0, 'harness')
import pandas as pd, contrasts
O = pathlib.Path('results/analysis/pilot_20261010')
rows = []
for v, arm in {'V5_pilot': 'V5', 'V5_completion': 'V5', 'V6_pilot': 'V6', 'G_part1': 'G', 'G_part2': 'G'}.items():
  for c in json.loads((O / 'check2' / v / 'compare_runs_check2.json').read_text())['comparisons']:
    t = c['label'].split(' ')[1]
    s = c['summary']
    run, ref = pathlib.Path(c['run']), pathlib.Path(c['reference'])
    m4 = [contrasts.coords_text(str(next((ref / f'seed-1_sample-{k}').glob('*_model.cif')))) ==
          contrasts.coords_text(str(next((run / f'seed-1_sample-{k}').glob('*_model.cif')))) for k in range(5)]
    rj = json.loads((run.parents[1] / 'run.json').read_text())
    fj = json.loads((ref.parents[1] / 'run.json').read_text())
    rows.append({'arm': arm, 'session': run.resolve().parents[3].name, 'target': t,
                 'samples': s['samples_compared_bitwise'], 'coordinates_identical': s['coordinates_identical'],
                 'confidences_identical': s['confidences_identical'], 'confidences_missing': s['confidences_missing'],
                 'summary_identical': s['summary_identical'], 'm4_bit_identical': sum(m4),
                 'same_index_rmsd_min_A': s['same_index_rmsd_min'], 'same_index_rmsd_max_A': s['same_index_rmsd_max'],
                 'pass': s['coordinates_identical'] == s['confidences_identical'] == s['summary_identical']
                 == sum(m4) == s['samples_compared_bitwise'] == 5,
                 'warm_model_cache_hit': rj['compile']['model_cache_hit'],
                 'warm_model_compile_s': rj['compile']['model_seconds'],
                 'fresh_model_compile_s': fj['compile']['model_seconds'],
                 'cache_files_before_warm_process': rj['compile_cache']['before']['files'],
                 'cache_files_after_warm_process': rj['compile_cache']['after']['files']})
pd.DataFrame(rows).to_csv(O / 'check2_warm_reruns.csv', index=False)
EOF

# Check 3: M1 on V6 r1 (collect.py, both r1 runs as roots), then the implemented check (contrasts.checks).
S=results/af3/20261009T154318Z_v6e_pilot
$PY analysis/collect.py --root $S/tpu_xla_rep1_seed1_fresh --root $S/tpu_xla_rep1_seed2_fresh --out $O/check3 --jobs 6
$PY - <<'EOF'
import argparse, sys
sys.path.insert(0, 'analysis'); sys.path.insert(0, 'harness')
import pandas as pd, contrasts
args = argparse.Namespace(samples='results/analysis/pilot_20261010/check3/samples.csv', labels='pilot', include=None,
                          exclude=None, condition='platform', replicates=10000, seed=2026)
d, ok = contrasts.load(args)
assert d.m1_error.isna().all(), 'M1 could not be computed for some model'
chk = contrasts.checks(d, ok, pd.DataFrame(columns=['contrast', 'bit_identical']), args)
chk[chk.check.str.startswith('3')].to_csv('results/analysis/pilot_20261010/check3_success_sanity.csv', index=False)
EOF
echo ">> Step 1 tables written to $O"
