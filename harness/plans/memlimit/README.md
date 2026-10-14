# Memory limit per chip (M7)

Plans for the measurement of `notes/analysis_plan.md` section 10, row of 2026-10-14 ("Memory limit per chip"):
7D5C (1,023 tokens), seed 1, official weights, AlphaFold3 defaults otherwise, one fresh process per bucket size
(1,280 to 5,120 tokens, AlphaFold3's own `--buckets` with one size), in increasing order, stop rule
`first_failure` (the first size that fails, out of memory or otherwise, or whose process exceeds 4 hours ends the
plan; later sizes are logged as not run), label `memlimit`. `make_memlimit_plans.py` writes them and sizes each
deadline (see its docstring); `harness/run_plan.py` documents the `buckets` and `stop_rule` keys.

| plan | platform | config | deadline (min) | VM lifetime | worst-case cost |
|---|---|---|---|---|---|
| memlimit_v5.yaml | v5e, on demand | tpu_xla | 878 | 16 h (watchdog) | $24.96 = $1.56/h x 16 h |
| memlimit_v6.yaml | v6e, on demand | tpu_xla | 1040 | 17.83 h | $52.96 = $2.97/h x 17.83 h |
| memlimit_g.yaml | L4, on demand | l4_xla | 751 | 13.02 h | $11.68 = $0.897/h x 13.02 h |
| memlimit_gt.yaml | L4, on demand | l4_triton | 792 | 13.7 h | $12.29 = $0.897/h x 13.7 h |

Launch (from the repository root, after committing; one command per terminal; the weights URI is a placeholder):

```
PLATFORM=v5e PLAN=harness/plans/memlimit/memlimit_v5.yaml SPOT=0 RESULTS_GCS_URI=gs://af2-tpu-benchmark-af3-results WEIGHTS_GCS_URI=gs://BUCKET/af3.bin.zst bash cloud/af3_run.sh
PLATFORM=v6e PLAN=harness/plans/memlimit/memlimit_v6.yaml PROVISIONING=standard ZONES="europe-west4-a us-east5-b us-east5-a us-central1-b us-east1-d" RESULTS_GCS_URI=gs://af2-tpu-benchmark-af3-results WEIGHTS_GCS_URI=gs://BUCKET/af3.bin.zst bash cloud/af3_run.sh
PLATFORM=l4 PLAN=harness/plans/memlimit/memlimit_g.yaml SPOT=0 RESULTS_GCS_URI=gs://af2-tpu-benchmark-af3-results WEIGHTS_GCS_URI=gs://BUCKET/af3.bin.zst bash cloud/af3_run.sh
PLATFORM=l4 PLAN=harness/plans/memlimit/memlimit_gt.yaml SPOT=0 RESULTS_GCS_URI=gs://af2-tpu-benchmark-af3-results WEIGHTS_GCS_URI=gs://BUCKET/af3.bin.zst bash cloud/af3_run.sh
```

Outcome per session: `stop_rule.json` (the size that stopped the plan and why, the last completed size, the sizes
not run), `plan_runs.jsonl`, and per process `run.json` (exit code, failure, failure_detail, padded_bucket,
compile and inference times, wall time, compiled and allocator memory). Fake-gcloud tests:
`tests/cloud/suite_7_memlimit.sh`.
