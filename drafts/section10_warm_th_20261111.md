# Section 10 draft: warm cost per structure on the L4 Triton and the H100 (label `warm_th`)

Drafted 2026-11-11 in branch `warm-th` (worktree `~/Desktop/af3-warm-th`, from `main` at `cb6ea79`), while follow-up
13's P arm runs on the v6e from the main checkout, and before any run of this measurement. The row below is also
inserted, unchanged, at the end of the section 10 table of `notes/analysis_plan.md` in the main checkout (`notes/`
only; nothing under `harness/` there changes while follow-up 13 runs). Lorenzo commits the row on `main` and the
branch in the worktree before the launch. A descriptive completion measurement like the memory limit on A and H (rows
of 2026-11-10): not a numbered follow-up, in no comparison.

## Row: specification

| 2026-11-11 | Warm cost per structure on the L4 Triton and the H100 (label `warm_th`), specified after the main-study and follow-up results were read (Table III's warm column is empty for GT and H: neither arm had a warm rerun) and before any run of it; a descriptive completion measurement, like the memory limit on A and H (rows of 2026-11-10), outside the pre-registered study and every contrast, reported whatever it shows. Question: what does one structure (one target, one seed, five samples) cost at the 1,024-token bucket on GT and on H when the process loads the model compiled by an earlier run from a saved cache instead of compiling it (M6, warm)? Arms, nothing changed from their source: GT, configuration `l4_triton` (the main study's GT arm) on `g2-standard-8`, on demand, the launcher's default L4 zones; H, configuration `h100_xla` (follow-up 9's H) on `a3-highgpu-1g`, Spot, because the project has no on-demand H100 quota (rows of 2026-11-10), the launcher's H100 zones (us-central1, us-east4, us-east5). Both build AlphaFold3 with the Eigen source of our bucket (`eigen_source: archive`, as every plan since follow-up 7; the same commit, so the same files). Design: the 10 pilot targets (`inputs/manifest.csv`), official weights, seed 1, AlphaFold3 defaults; per arm, on one VM, one fresh run whose seed-1 executables are kept (`--save_fresh_cache`), then one warm rerun on the same VM that loads them (`--compile_cache warm:...`), the mechanism of the main study's warm reruns (design (a), row of 2026-10-12) and of follow-up 3's; 20 processes per arm. The fresh runs serve only as the warm reruns' source and for the run's cost; they enter no contrast and no Table III cell (the fresh, amortized, compile and inference cells of GT and H stay those of the main study and of follow-up 9). Plans: `harness/plans/warm_th/warm_th_gt_01.yaml` and `warm_th_h_01.yaml`, written by `harness/plans/warm_th/make_warm_th_plans.py --check` (job from the source arms' records: setup, the seed-1 fresh walls of the pilot targets, warm walls estimated as fresh minus (1 - 0.110) of the compile time, 0.110 being the main study's G warm over fresh compile time; deadline ceil(job min x 1.25 + 20): GT 123 min, H 76 min); `launch_by: 2026-11-12`, `finish_by: 2026-11-13`; parts files `results/parts_warm_th_gt.txt` and `results/parts_warm_th_h.txt` with `FETCH=full`, so the bit-identity check reads every output file. Checks, before anything is computed (`analysis/warm_th.py`; if one fails the analysis writes `checks.csv` and `report.md`, no `summary.csv`, and the arm is reported here as not completed): the listed sessions are the two `warm_th` parts, one per arm, with the arm's configuration and machine type, every row at AlphaFold3's defaults; completeness (per arm, the 10 pilot targets, seed 1, one fresh and one warm process each); 1, every process exit 0; 2a, every fresh process compiled from an empty cache; 2b, every warm process loaded `jit_apply_fn` from the cache with a key its own fresh process wrote; bit-identity, every warm process's outputs equal to its fresh process's, every file byte for byte except the mmCIF timestamp line (`contrasts.cache_checks` and `contrasts.check2` of the main study, unchanged). Measure, as M6 (plan section 4; interpretations 2 and 3 of the entry of 2026-10-09; `analysis/cost.py`'s `processes`, `cost_table` and `summary_table`, unchanged): cost per structure = process wall / seeds x the hourly list price of `cloud/prices.csv` in the region of the run's zone (europe-west4 when the region has no row, noted); warm processes as measured, not amortised. Table III's warm cell of each arm is the median over its warm processes at bucket 1,024 (7D5C and 8EYE, 2 processes, as the existing warm cells) at the on-demand list price, as every cost cell of Table III, for H too although it ran on Spot; H's warm cost at the Spot price of `cloud/prices.csv` (read on 2026-10-27; Spot prices are dynamic, and the date is stated) is reported next to it and declared in the table's note. Analysis: `DATE=YYYYMMDD bash scripts/analysis_runs/warm_th/reproduce.sh` (the synthetic tests, then `analysis/collect.py --labels warm_th --sessions ... --no_m1`: no prediction is scored or read for any finding, then `analysis/warm_th.py`) into a new folder `results/analysis/warm_th_<DATE>/` (`checks.csv`, `summary.csv` with the columns of the main study's `cost_summary.csv`, `cost_per_process.csv`, `prices_used.csv`, `report.md`, `inputs.tsv`, `samples.csv`, `run.log`); the sessions in `scripts/analysis_runs/warm_th/sessions.txt`, written from the launch rows once both parts have finished and committed before the analysis. The paper's generator (`analysis/paper_assets_isc.py`, branch `paper`) fills the two cells only when that folder exists and every check in it passed. Files (SHA-256): `analysis/warm_th.py` f5d083fc204eda9580068c3a37d731090dca6b0f4e13d2d41c4aa9d4b2782997, `analysis/tests/test_warm_th.py` e8a6542a9bba391656ad25ad4d2624cad68d4a54f9315287804218eb0387491c, `analysis/tests/test_warm_th_plans.py` 44be63382b3d5f6c58ed62863ca3b8fc3fc14e6188b57c3d2219260cabccc6ef, `scripts/analysis_runs/warm_th/reproduce.sh` edd158ba3b987e09449fbc5d31b712bdd3b8092ec767766a60bad00dac2f75e7, `scripts/monitor/credit_check_warm_th.sh` 38e55a11601114059e5366c6d9c7c78337d686ee764baeb945988692e00f4bff, `harness/plans/warm_th/make_warm_th_plans.py` b2bafebc66337eddcb289006bc363d3afa72914bfa28a23d386ee644e735a507, `warm_th_gt_01.yaml` e673eb8be3518df68ba2c8b0a8aef91ff723cc2702de21840b42ad78955ba737, `warm_th_h_01.yaml` ad9197ca58a5ad6a041e97800199d9b8bd3bea8e71f4bfcfdc0294e45dc7265d; `analysis/.venv/bin/python -m pytest -q analysis/tests scripts/release -p no:cacheprovider`: 889 passed, 2 skipped. Budget, on the Google TPU Builders credit only (`cloud/prices.csv`, the highest price among each launcher's regions: L4 on demand 0.8972 USD/h, H100 Spot 6.5874 USD/h): expected, provisioning, setup and teardown (source-session medians) plus the job: GT 1.50 VM h, 1.35 USD; H 0.78 VM h, 5.11 USD; total 6.46 USD. Worst, the VM lifetimes the deadlines set: GT 2.29, H 11.64, total 13.93 USD. Cap 30 USD (one unchanged relaunch of each part fits). Gate, `scripts/monitor/credit_check_warm_th.sh ARM BALANCE` immediately before each launch: GO only if the project's GPUS_ALL_REGIONS quota (read then, read-only; it read 10 on 2026-11-10) is at least the GPU sessions of other launchers still running plus 2 (follow-up 13's remaining arm, P, runs on the v6e and holds no GPU); the balance minus the expected cost of every running arm of follow-ups 7 to 13 and of every follow-up 13 arm the chain ledger does not record as ok or stopped, minus the expected cost of the `warm_th` arms not yet finished, is at least 25 USD; the spend so far plus that expected cost is within 30 USD; the launch is on or before 2026-11-12 and the expected end by 2026-11-13 (UTC). Each run is appended to `results/warm_th_gate.tsv` and copied into the launch row. Stop rule: launch by 2026-11-12 (UTC; the launcher refuses a plan after its `launch_by`), runs finished by 2026-11-13 (`finish_by`); otherwise the arm is reported here as not completed and its Table III cell stays empty. No run is discarded or chosen on its results; an interrupted part (a Spot preemption of H, a capacity or connection failure) is relaunched unchanged as a new session after the gate, the interrupted session recorded here and not listed; a relaunch that would take the spend above the cap needs a cap amendment in a row first. Launch: one `cloud/af3_run_many.sh` wrapper per arm, in parallel, from the worktree `~/Desktop/af3-warm-th` at the committed branch `warm-th` (the main checkout, from which follow-up 13's launchers upload `harness/`, does not change), with `ALLOW_OTHER_LAUNCHERS=1`, `MAX_L4=1` or `MAX_H100=1`; the branch is merged into `main` after follow-up 13's last launcher has exited. Every launch, relaunch and stop is recorded in section 10. | Table III's warm column is empty for the L4 Triton and the H100 |

## Commit (Lorenzo)

Worktree (the untracked symlinks `data`, `results`, `third_party`, `analysis/.venv` stay out):

```
cd ~/Desktop/af3-warm-th && git add harness/plans/warm_th scripts/monitor/credit_check_warm_th.sh analysis/warm_th.py analysis/tests/test_warm_th.py analysis/tests/test_warm_th_plans.py analysis/tests/test_eigen.py scripts/analysis_runs/warm_th drafts/section10_warm_th_20261111.md drafts/apply_warm_th_generator.py drafts/paper_assets_isc_warm_th.patch && git status --short && git commit -m "Warm cost on GT and H (warm_th): plans, gate, analysis and tests"
```

Main (`notes/` only):

```
cd ~/Desktop/alphafold-on-tpu && git diff --stat && git add notes/analysis_plan.md && git commit -m "Section 10: warm cost on the L4 Triton and the H100, specification"
```

When `warm-th` and `memlimit-ah` are both merged into `main`, `analysis/tests/test_eigen.py` conflicts on one line (the
set of plan folders with `eigen_source`): keep both, `'memlimit_ah'` and `'warm_th'`.

## Launch commands (one per arm, two terminals, in parallel)

After both commits. BALANCE is the TPU Builders credit read from the console, typed as a number. Start the second
command a few seconds after the first (each wrapper names its state folder by the UTC second). `SPOT=0` is the wrapper
default; the H parts file sets `SPOT=1` on its line. Capacity refusals (no VM created) are retried by the wrapper
(`MAX_ATTEMPTS=100`, every 5 minutes) until the plan's `launch_by` makes the launcher refuse.

GT (L4 Triton, on demand):

```
cd ~/Desktop/af3-warm-th && python3 harness/plans/warm_th/make_warm_th_plans.py --check && bash scripts/monitor/credit_check_warm_th.sh gt BALANCE && caffeinate -dims env PARTS=results/parts_warm_th_gt.txt MAX_L4=1 MAX_ATTEMPTS=100 RETRY_WAIT_S=300 ALLOW_OTHER_LAUNCHERS=1 SPOT=0 YES=1 RESULTS_GCS_URI=gs://af2-tpu-benchmark-af3-results WEIGHTS_GCS_URI=gs://af2-tpu-benchmark-af3-weights/af3.bin.zst bash cloud/af3_run_many.sh
```

H (H100, Spot):

```
cd ~/Desktop/af3-warm-th && python3 harness/plans/warm_th/make_warm_th_plans.py --check && bash scripts/monitor/credit_check_warm_th.sh h BALANCE && caffeinate -dims env PARTS=results/parts_warm_th_h.txt MAX_H100=1 MAX_ATTEMPTS=100 RETRY_WAIT_S=300 ALLOW_OTHER_LAUNCHERS=1 SPOT=0 YES=1 RESULTS_GCS_URI=gs://af2-tpu-benchmark-af3-results WEIGHTS_GCS_URI=gs://af2-tpu-benchmark-af3-weights/af3.bin.zst bash cloud/af3_run_many.sh
```

Relaunch of an interrupted part (for example H preempted): the gate again, then the same command with
`ALLOW_RELAUNCH=warm_th_h_01` (or `warm_th_gt_01`) added after `env`.

## Estimates

| arm | VM | provisioning | processes | fresh run | warm rerun | job (with setup) | deadline | expected VM h | expected USD | worst USD |
|---|---|---|---|---|---|---|---|---|---|---|
| GT | g2-standard-8 | on demand | 20 | 39 min | 31 min | 82 min | 123 min | 1.50 | 1.35 | 2.29 |
| H | a3-highgpu-1g | Spot | 20 | 15 min | 10 min | 44 min | 76 min | 0.78 | 5.11 | 11.64 |
| both | | | 40 | | | | | 2.28 | 6.46 | 13.93 |

In parallel the measurement takes about 1.5 hours of wall time once both VMs are up (GT is the long pole).

## After the runs

1. Launch and completion row (sessions, gate lines, any interruption).
2. `scripts/analysis_runs/warm_th/sessions.txt`: the complete session of each part; commit.
3. `DATE=YYYYMMDD bash scripts/analysis_runs/warm_th/reproduce.sh` (from the worktree or from `main` after the merge).
4. Paper generator: the change is not applied to `~/Desktop/af3-paper` (another session was editing
   `analysis/paper_assets_isc.py` on 2026-11-11 and left a `TODO warm_th` hook there). When no session edits it:
   `cd ~/Desktop/af3-paper && python3 ~/Desktop/af3-warm-th/drafts/apply_warm_th_generator.py .` (exact anchors; it
   writes nothing if one is missing), then
   `~/Desktop/alphafold-on-tpu/analysis/figures/.venv/bin/python -m pytest -q -p no:cacheprovider analysis/tests/test_paper_assets_pending.py analysis/tests/test_paper_assets_isc.py`
   (on a copy of the version of 2026-11-11 20:30: 27 passed before, 30 after). `drafts/paper_assets_isc_warm_th.patch`
   is the same change as a diff, for reading. Until `results/analysis/warm_th_<DATE>/` exists the generator's output
   is unchanged (the two cells empty, the `% TODO warm_th` line kept).
