# tests/many: fake-gcloud tests for cloud/af3_run_many.sh

```bash
bash tests/many/test_run_many.sh      # about 5 minutes; creates no cloud resources
```

Uses the kit of `tests/cloud` (its guard against the real gcloud, `mkrepo.sh`, the fake gcloud) and writes only
under `$CLOUD_TEST_WORK` (default `${TMPDIR:-/tmp}/af3-cloud-tests`).

| File | Purpose |
|---|---|
| `test_run_many.sh` | Part 1: the wrapper's own logic with a stub launcher (caps per platform, staggering, one launcher per part, retries only without a VM, kept VMs, the relaunch list, v6e on demand, the STOP file, Ctrl-C, the prompt, refusals). Part 2: the real `cloud/af3_run.sh` against the fake gcloud, one fake state per launch, including a v6e part created with `--provisioning-model=STANDARD`. Part 3: `harness/plans/main/parts_all.txt` with the stub launcher |
| `stub_launcher.sh` | Stand-in for `cloud/af3_run.sh` (no gcloud): session line, launch record, exit codes |
| `e2e_launcher.sh` | Runs the real launcher in the fake environment, one state folder per launch |
| `many_ctrlc.py` | Ctrl-C (SIGINT to the process group, as a terminal sends it) during a run |
