#!/bin/bash
# Suite 8 (2026-10-14): the follow-up launches of notes/analysis_plan.md section 10 (rows of 2026-10-14), on fake
# VMs, as they will be run (official weights from a bucket, results bucket, on demand; v6e with
# PROVISIONING=standard and the main study's zones).
#   Follow-up 2 (harness/plans/embed/): every process passes --save_embeddings=true, session.json records it,
#   run.json lists the embedding file, and FETCH=light brings the .npz files to the laptop ("followup check").
#   Follow-up 1 (plan key manifest): inputs listed in a manifest other than inputs/manifest.csv are checked
#   against it on the laptop and on the VM, uploaded, and passed to the harness (session.json args.manifest); a
#   tampered input is refused before any VM is created. The first manifest and inputs are synthetic (two small
#   protein-protein inputs); the last scenario launches the real part ppi_g_06 (2 targets, 20 processes) with
#   the real inputs/ppi/manifest.csv and its two inputs from data/inputs/ppi/.
. "$(dirname "$0")/common.sh"
R="$KIT/run.sh"
bash "$KIT/mkrepo.sh"
W=gs://private-weights/af3.bin.zst
B=gs://followup-results/study
V6_ZONES="europe-west4-a us-east5-b us-east5-a us-central1-b us-east1-d"
# Synthetic protein-protein inputs, their manifest and a plan that uses it, committed in the scratch repository.
( cd "$WORK/repo" && mkdir -p data/inputs/ppi inputs/ppi harness/plans/ppi && python3 - <<'PYEOF'
import csv, hashlib, json, pathlib
rows = []
for pid, a, b in (('9AAA', 'MKTAYIAKQRQISFVKSHFSRQ', 'GSHMLEDPVDAFQ'), ('9AAB', 'MSEQNNTEMTFQIQRIYTKDI', 'MADQLTEEQIAEFKEAF')):
  text = json.dumps({'name': pid, 'modelSeeds': [1, 2, 3, 4, 5], 'sequences': [
      {'protein': {'id': c, 'sequence': s, 'unpairedMsa': f'>101\n{s}\n', 'pairedMsa': f'>101\n{s}\n', 'templates': []}}
      for c, s in (('A', a), ('B', b))], 'dialect': 'alphafold3', 'version': 1}, indent=2) + '\n'
  path = pathlib.Path(f'data/inputs/ppi/{pid}.json')
  path.write_text(text)
  rows.append({'pdb_id': pid, 'ligand_ccd': '', 'pilot': 'no', 'input_path': str(path),
               'input_sha256': hashlib.sha256(text.encode()).hexdigest(), 'num_tokens': len(a) + len(b), 'bucket': 256})
with open('inputs/ppi/manifest.csv', 'w', newline='') as f:
  w = csv.DictWriter(f, list(rows[0]), lineterminator='\n')
  w.writeheader()
  w.writerows(rows)
pathlib.Path('harness/plans/ppi/test_ppi.yaml').write_text('''description: ppi manifest test
label: ppi
weights: gcs
manifest: inputs/ppi/manifest.csv

configs:
  v5e: tpu_xla

targets:
  default: [9AAA, 9AAB]

seeds: [1, 2]
fresh_reps: 1
warm_rerun_seed: none
num_recycles: default
num_diffusion_samples: default
run_timeout_min: 180

deadline_min:
  v5e: 60
''')
PYEOF
  git add -A && git -c user.name=test -c user.email=test@invalid commit -qm 'ppi test inputs' )
check() {   # NAME KIND (embed or ppi) [RUNS, repetitions x seeds]: checks the fetched session
  local s
  grep -E "follow deadline|VM lifetime limit|worst-case cost|Checking the plan" "$WORK/state_$1/out" | sed 's/^/   | /'
  s=$(ls -d "$WORK/state_$1/fetched/results/af3/"*/ 2> /dev/null | grep -v 20261008T165218Z_v6e_probe | tail -1)
  [ -n "$s" ] || { echo "   followup check: 1 problem(s): nothing fetched"; return; }
  python3 - "$s" "$2" "${3:-1}" "$(grep -c 'private-weights' "$WORK/state_$1/out")" <<'PYEOF'
import json, pathlib, sys
s, kind, reps, leaks = pathlib.Path(sys.argv[1]), sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
problems = []
runs = [json.loads(l) for l in (s / 'plan_runs.jsonl').read_text().splitlines()]
if len(runs) != reps or any(r['harness_exit_code'] != 0 for r in runs):
  problems.append(f'runs {[(r["run"], r["harness_exit_code"]) for r in runs]}')
n = 0
for r in runs:
  d = s / r['run']
  sj = json.loads((d / 'session.json').read_text())
  for rj in sorted(d.glob('*/run.json')):
    rec, n = json.loads(rj.read_text()), n + 1
    cmd = (rj.parent / 'command.txt').read_text()
    if kind == 'embed':
      npz = list((rj.parent / 'af3_output').rglob('*_embeddings.npz'))
      if not (sj.get('save_embeddings') and '--save_embeddings=true' in cmd and rec.get('embedding_files') and npz):
        problems.append(f'{r["run"]} {rec["pdb_id"]}: save_embeddings {sj.get("save_embeddings")}, files {rec.get("embedding_files")}, fetched npz {len(npz)}')
    else:
      if sj['args'].get('manifest') != 'inputs/ppi/manifest.csv' or '--save_embeddings' in cmd or sj.get('save_embeddings'):
        problems.append(f'{r["run"]}: manifest {sj["args"].get("manifest")}, save_embeddings {sj.get("save_embeddings")}')
    if rec['exit_code'] != 0:
      problems.append(f'{r["run"]} {rec["pdb_id"]}: exit {rec["exit_code"]}')
if leaks:
  problems.append('weights URI printed')
print(f'   processes: {n}; followup check: {len(problems)} problem(s)' + (': ' + '; '.join(problems[:3]) if problems else ''))
PYEOF
}
echo "======== follow-up 2: trunk embeddings"
bash "$R" embed_v6e ok PLATFORM=v6e FAKE_PLATFORM=v6e PROVISIONING=standard ZONES="$V6_ZONES" \
  PLAN=harness/plans/embed/embed_v6.yaml WEIGHTS_GCS_URI=$W RESULTS_GCS_URI=$B
check embed_v6e embed 1
bash "$R" embed_g ok PLATFORM=l4 FAKE_PLATFORM=l4 SPOT=0 PLAN=harness/plans/embed/embed_g.yaml \
  WEIGHTS_GCS_URI=$W RESULTS_GCS_URI=$B
check embed_g embed 2
echo "======== follow-up 1: inputs from inputs/ppi/manifest.csv"
bash "$R" ppi_manifest ok PLATFORM=v5e FAKE_PLATFORM=v5e SPOT=0 PLAN=harness/plans/ppi/test_ppi.yaml \
  WEIGHTS_GCS_URI=$W RESULTS_GCS_URI=$B
check ppi_manifest ppi 2
( cd "$WORK/repo" && echo ' ' >> data/inputs/ppi/9AAB.json )
bash "$R" ppi_tampered ok PLATFORM=v5e FAKE_PLATFORM=v5e SPOT=0 PLAN=harness/plans/ppi/test_ppi.yaml \
  WEIGHTS_GCS_URI=$W RESULTS_GCS_URI=$B
grep -E "do not match|9AAB" "$WORK/state_ppi_tampered/out" | sed 's/^/   | /' | head -3
echo "======== follow-up 1: the real part ppi_g_06 with the real manifest"
( cd "$WORK/repo" && cp "$REPO/inputs/ppi/manifest.csv" inputs/ppi/ \
  && cp "$REPO/data/inputs/ppi/8FJ3.json" "$REPO/data/inputs/ppi/8HFP.json" data/inputs/ppi/ \
  && git add -A && git -c user.name=test -c user.email=test@invalid commit -qm 'real ppi part' )
bash "$R" ppi_real_g ok PLATFORM=l4 FAKE_PLATFORM=l4 SPOT=0 PLAN=harness/plans/ppi/ppi_g_06.yaml \
  WEIGHTS_GCS_URI=$W RESULTS_GCS_URI=$B
check ppi_real_g ppi 10
