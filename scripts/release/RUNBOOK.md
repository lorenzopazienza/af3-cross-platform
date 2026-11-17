# Building the Zenodo release (runbook)

The first build (Part B) ran on 2026-11-09 from tag `zenodo-v1` (freeze `689230120499`, 295 sessions; R1 9 archives and 35 files,
R2 36 archives and 76 files; row of section 10 of 2026-11-09) and is uploaded to the two drafts, which are not published. This
release (Part A) brings both drafts to the final freeze `4ec185c5`, 439 sessions, before the arXiv preprint: incrementally, so that
only the new and changed files are built and uploaded. Part C says how a later version is made the same way.

Commands are written out in full for zsh on the laptop, from the repository root. Steps marked VM run on the VM
through `gcloud compute ssh --command`. Lorenzo runs every step that creates or deletes a cloud resource, and every step on
Zenodo. Nothing here publishes: Lorenzo publishes by hand after checking each draft.

The two DOIs are reserved (drafts, not published): R1 `10.5281/zenodo.24057470`, R2 `10.5281/zenodo.24057544`; they are already in
`scripts/release/records/{R1,R2}/README.md` and `zenodo.json`.

# Part A. This release: freeze 4ec185c5 (follow-ups 13 to 16, `memlimit_ah`, `warm_th`), incremental

What changes, against the drafts as uploaded on 2026-11-09 (`results/release/vm_build_20261109/`, the `--previous` of A7):

- **New archives** (one table entry per label, `LATER_LABELS` in `release.py`): R1 `session_records_followups1316.tar.gz`
  (laptop) and `outside_freeze_records_followups1316.tar.gz` (VM, 247 sessions outside the freeze, records only); R2
  `af3_fu13.tar.gz`, `af3_fu14.tar.gz`, `af3_rcx.tar.gz`, `af3_hlo.tar.gz`, `hlo_xla_dumps.tar.gz` and
  `af3_memlimit_ah_warm_th.tar.gz` (VM). One output archive per follow-up (structures, full confidences and logs together), the
  two later measurements in one, so that R2 stays at 88 files. The layer captures of follow-up 13 are left out (decision of
  2026-11-17: they compress only to 0.80 under gzip, 5.15 GiB, and R2 would pass 50 GB with them; R2's README says so).
- **Rebuilt, so replaced:** `freeze.tar.gz` (the new freeze folder), `session_lists_launch_records.tar.gz` (the launches since
  2026-11-09) and `analysis_outputs.tar.gz` (the analyses since 8 November and the paper's `paper/assets/` from the tagged paper
  commit, `--paper_ref`; the ISC draft's `results/paper_assets_isc/` is left out), with the plain files (READMEs, licences,
  `REDACTIONS.tsv`, the `release_*` lists, `SHA256SUMS`).
- **Kept byte for byte, not rebuilt on the VM and not uploaded again:** every other archive of the first build. The laptop rebuilds
  its own archives anyway (minutes); A7 compares each with the draft's SHA-256 and marks it keep or replace.
- **Before the VM:** the hlo dumps were checked for the model weights (none: `records/LICENCES.md`, last section, which also
  holds the open licence point of the dumps); the systemd unit names of the new job logs no longer stop the build; 110 new wrapper
  logs with the account address are on `records/redact_email_paths.txt`.

Before A1: commit, then create and push the tag: `git tag -a zenodo-4ec185c5 -m "Zenodo release build"` and `git push origin zenodo-4ec185c5`;
build from that tag. Tag the paper commit whose `paper/assets/` goes into R2 as well, for example
`git tag -a paper-arxiv-v1 -m "arXiv version" COMMIT` and `git push origin paper-arxiv-v1`, where COMMIT is that commit of the
paper branch (the two tags may name the same commit if the paper branch is merged first). The READMEs name `zenodo-4ec185c5`.

## A1. Laptop: keep the first build's lists, list the bucket, inventory, layout, scan

```
mkdir -p results/release/zenodo-v1
cp -Rp results/release/inventory results/release/layout.md results/release/layout.csv results/release/zenodo-v1/
gcloud storage ls -l -r 'gs://af2-tpu-benchmark-af3-results/af3/**' > results/release/inventory/bucket_listing.txt
python3 scripts/release/release.py inventory
python3 scripts/release/release.py layout
python3 scripts/release/release.py scan --root results/af3 --out $HOME/af3_release_1316_scan --archives freeze.tar.gz,session_lists_launch_records.tar.gz,analysis_outputs.tar.gz,session_records_followups1316.tar.gz,outside_freeze_records_followups1316.tar.gz,af3_fu13.tar.gz,af3_fu14.tar.gz,af3_rcx.tar.gz,af3_hlo.tar.gz,hlo_xla_dumps.tar.gz,af3_memlimit_ah_warm_th.tar.gz
```

The listing only reads the bucket (a few minutes; about 485,000 objects). Expected: `inventory` exits 0, with 247 sessions of kind
`outside_followups1316` at most (the local estimate of 2026-11-17 found 247; launches that wrote nothing to the bucket have no
line) and no unmapped session; if it exits 2, `unmapped_sessions.csv` lists them: stop, map each to its row in its
`LATER_LABELS` entry or decide that it stays out, do not guess. `layout` puts R1 at 11 archives and 41 files and R2 at 42 archives and
88 files, about 45.8 GB by its ratios (A7 measures about 44.6 GB: the first build's R2 archives are smaller than the estimate). `scan` takes about 4 minutes and exits 0 with the counts of the row (home paths, the account in 312 wrapper
logs, bare NAS host names in 4 analysis files; no other address, no private IPv4 address, no secret).

## A2. Laptop: build the archives made locally

```
python3 scripts/release/release.py build --root results/af3 --out $HOME/af3_release_1316 --archives $(python3 scripts/release/release.py archives --built_on laptop) --paper_ref paper-arxiv-v1
python3 scripts/release/release.py verify --out $HOME/af3_release_1316
```

Expected: 8 R1 archives and 5 R2 archives, "verify: OK", about 15 minutes, about 2 GB free needed. Of these, the four input
archives, `session_records.tar.gz`, `session_records_followups.tar.gz` and the four pipeline archives of R2 should have the
SHA-256 of the first build (A7 checks it and marks them keep).

## A3. Laptop: the bundle the VM needs

```
mkdir -p $HOME/af3_release_bundle_1316
python3 scripts/release/release.py vm-lists --batch followups1316 --out $HOME/af3_release_bundle_1316
COPYFILE_DISABLE=1 tar czf $HOME/af3_release_bundle_1316/repo.tgz scripts/release LICENSES results/freeze/followups1316_20261117/sessions.tsv results/freeze/followups1316_20261117/comparison.tsv data/inputs inputs/manifest.csv
ls -lh $HOME/af3_release_bundle_1316/repo.tgz
```

Expected: 144 lines in `sessions_release.txt` (the frozen sessions of the six labels) and about 241 in `sessions_outside.txt`
(the freeze's `summary.md` counts 241 bucket folders of these labels outside the lists); a `repo.tgz` of about 0.2 GB. Every
process of these labels ran on `data/inputs/` (seeds 1 to 5), so no other input folder is needed.

## A4. Laptop: create the VM (cloud resource; Lorenzo)

```
gcloud compute instances create af3-release-build --project=af2-tpu-benchmark --zone=europe-west4-a --machine-type=e2-standard-8 --image-family=debian-12 --image-project=debian-cloud --boot-disk-size=150GB --boot-disk-type=pd-balanced --scopes=storage-ro --max-run-duration=8h --instance-termination-action=DELETE
gcloud compute scp --project=af2-tpu-benchmark --zone=europe-west4-a $HOME/af3_release_bundle_1316/repo.tgz $HOME/af3_release_bundle_1316/sessions_release.txt $HOME/af3_release_bundle_1316/sessions_outside.txt af3-release-build:~/
gcloud compute ssh af3-release-build --project=af2-tpu-benchmark --zone=europe-west4-a --command='mkdir -p ~/repo && tar xzf ~/repo.tgz -C ~/repo && python3 --version && ls ~/repo'
```

Disk: the 144 sessions take 75.3 GB in the bucket (`*_data.json` 17.9 GB, full confidences 41.5 GB, the captures 6.8 GB,
copied but not packaged, dumps 1.3 GB), the outside sessions without outputs well under 1 GB, the archives about 7 GB, the bundle
0.2 GB: about 84 GB, so 150 GB leaves room. `--max-run-duration=8h` against about 2 h expected.

## A5. VM: copy, check, build and verify, detached

```
gcloud compute ssh af3-release-build --project=af2-tpu-benchmark --zone=europe-west4-a --command='BATCH=followups1316 nohup bash ~/repo/scripts/release/vm_steps.sh all > ~/vm_steps.log 2>&1 < /dev/null & disown; echo started'
gcloud compute ssh af3-release-build --project=af2-tpu-benchmark --zone=europe-west4-a --command='grep -E "start|done|FAILED|!!|verify|status|archives" ~/vm_steps.log | cut -c1-200; tail -2 ~/vm_steps.log | cut -c1-200; df -h ~ | tail -1'
```

| step | what | expected |
|---|---|---|
| copy | about 110,000 objects, 75 GB, within the region (the first build copied 324 GiB in 23 min) | about 10 min |
| check-inputs | 144 sessions, 3,374 processes (`fu13` 1,790, `fu14` 1,110, `rcx` 400, `hlo` 16, `memlimit_ah` 18, `warm_th` 40), about 18 GB of `*_data.json`; status pass | about 3 min |
| build | the 7 VM archives of the batch (`release.py archives --built_on vm --batch followups1316`), about 48 GB raw | about 1.5 h |
| verify | every member | about 15 min |
| | **total, without surprises** | **about 2 h** |

A refusal stops the build and removes what it wrote; the causes are those of step 6 of Part B, plus an XLA dump whose printed
constants exceed 65,536 characters or that holds an MLIR resource blob (look at it for the model weights before anything else).
The full confidences and the outside sessions' records exist only in the bucket, so they are scanned here for the first time.

## A6. Laptop: copy the archives back (internet egress: about 6.5 GiB)

```
mkdir -p $HOME/af3_release_1316/from_vm/R1 $HOME/af3_release_1316/from_vm/R2
gcloud compute scp --project=af2-tpu-benchmark --zone=europe-west4-a 'af3-release-build:~/release/R2/*.tar.gz' 'af3-release-build:~/release/R2/*.sha256' $HOME/af3_release_1316/R2/
gcloud compute scp --project=af2-tpu-benchmark --zone=europe-west4-a 'af3-release-build:~/release/R1/outside_freeze_records_followups1316.tar.gz*' $HOME/af3_release_1316/R1/
gcloud compute scp --project=af2-tpu-benchmark --zone=europe-west4-a 'af3-release-build:~/release/inputs_check.*' $HOME/af3_release_1316/from_vm/
gcloud compute scp --project=af2-tpu-benchmark --zone=europe-west4-a 'af3-release-build:~/release/R1/REDACTIONS.tsv' $HOME/af3_release_1316/from_vm/R1/
gcloud compute scp --project=af2-tpu-benchmark --zone=europe-west4-a 'af3-release-build:~/release/R2/REDACTIONS.tsv' $HOME/af3_release_1316/from_vm/R2/
gcloud compute ssh af3-release-build --project=af2-tpu-benchmark --zone=europe-west4-a --command='cd ~/release && sha256sum R2/*.tar.gz R2/*.sha256 R1/outside_freeze_records_followups1316.tar.gz*' > $HOME/af3_release_1316/vm_sha256.txt
gcloud compute instances delete af3-release-build --project=af2-tpu-benchmark --zone=europe-west4-a --quiet
gcloud compute instances list --project=af2-tpu-benchmark --filter='name=af3-release-build'
```

The VM's R2 folder holds only the 6 new archives (the laptop's R2 archives are not there), so the wildcard copies those. The
last command prints nothing once the VM is deleted (its disk goes with it). The upload goes through the web form from the laptop
(A9): with about 7 GB to send, the API form of Part B is not needed.

## A7. Laptop: finish and verify the two records against the drafts

```
(cd $HOME/af3_release_1316 && shasum -a 256 -c vm_sha256.txt)
python3 scripts/release/release.py merge-redactions --out $HOME/af3_release_1316 --record R1 --from $HOME/af3_release_1316/from_vm/R1/REDACTIONS.tsv
python3 scripts/release/release.py merge-redactions --out $HOME/af3_release_1316 --record R2 --from $HOME/af3_release_1316/from_vm/R2/REDACTIONS.tsv
cp $HOME/af3_release_1316/from_vm/inputs_check.csv $HOME/af3_release_1316/R1/release_inputs_check_followups1316.csv
cp $HOME/af3_release_1316/from_vm/inputs_check.json $HOME/af3_release_1316/R1/release_inputs_check_followups1316.json
python3 scripts/release/release.py sums --out $HOME/af3_release_1316 --previous results/release/vm_build_20261109
python3 scripts/release/release.py verify --out $HOME/af3_release_1316 --previous results/release/vm_build_20261109
```

`sums --previous` keeps, in each record's `SHA256SUMS` and `REDACTIONS.tsv`, the files of the drafts that are not rebuilt here,
with their recorded SHA-256 and redaction rows; it stops if a record would pass 100 files or 50 GB, kept files included. It
writes `zenodo_changes_R1.csv` and `zenodo_changes_R2.csv` in `$HOME/af3_release_1316`: every file of each draft with its
action, keep, replace, add or delete, and its size. Expected: R1 41 files, R2 88 files, R2 about 44.6 GB; "verify: OK", with the kept
files counted as kept. In R1, `release_inputs_check.csv` and `.json` of the first build are kept (sessions of freeze `68923012`).
If an archive the laptop rebuilt is marked replace although nothing of it should have changed (the input and pipeline archives,
`session_records.tar.gz`, `session_records_followups.tar.gz`), stop and compare its `.sha256` manifest with the first build's
(`$HOME/af3_release/R1` and `R2` hold them) before uploading.

## A8. Laptop: keep the build records (the `--previous` of the next version)

```
mkdir -p results/release/build_20261117/R1 results/release/build_20261117/R2
cp $HOME/af3_release_1316/R1/SHA256SUMS $HOME/af3_release_1316/R1/REDACTIONS.tsv results/release/build_20261117/R1/
cp $HOME/af3_release_1316/R2/SHA256SUMS $HOME/af3_release_1316/R2/REDACTIONS.tsv results/release/build_20261117/R2/
cp $HOME/af3_release_1316/zenodo_files_*.json $HOME/af3_release_1316/zenodo_changes_*.csv $HOME/af3_release_1316/vm_sha256.txt results/release/build_20261117/
```

Use the date of the actual build in the folder name. `results/` is not committed; the row of section 10 names the folder.

## A9. Zenodo: update the two drafts in the web form (Lorenzo; nothing publishes)

For each record, open its draft at https://zenodo.org/uploads (the one with the reserved DOI) and follow its
`zenodo_changes_<record>.csv`:

1. **Files.** Delete every file marked `replace` or `delete` from the draft (the bin icon of the file row); then drag in every
   file marked `replace` or `add` from `$HOME/af3_release_1316/<record>/`, and wait until each shows as finished. Files marked
   `keep` stay as they are. Expected, if A7 marks the unchanged archives keep:
   - R1: replace `README.md`, `LICENCES.md`, `REDACTIONS.tsv`, `SHA256SUMS`, `freeze.tar.gz` and its `.sha256`,
     `session_lists_launch_records.tar.gz` and its `.sha256`, `release_release.py`, `release_RUNBOOK.md`, and the lists A1
     remade (`release_inventory.md`, `release_layout.md`, `release_layout.csv`, `release_summary.csv`, `release_sessions.csv`,
     `release_excluded_sessions.csv`, `release_excluded_files.csv`, `release_OUTSIDE_SESSIONS.csv`);
     add `session_records_followups1316.tar.gz`, `outside_freeze_records_followups1316.tar.gz` (each with its `.sha256`),
     `release_inputs_check_followups1316.csv` and `.json`; nothing to delete. 35 files become 41.
   - R2: replace `README.md`, `LICENCES.md`, `REDACTIONS.tsv`, `SHA256SUMS`, `analysis_outputs.tar.gz` and its `.sha256`; add
     the 6 new archives with their `.sha256` (12 files, about 6.9 GB); nothing to delete. 76 files become 88.
   The csv files are what counts: if they differ from this list, they say why (a rebuilt archive that differs).
2. **Description.** Replace it with the `description` of the record's `zenodo.json` (now 439 sessions and freeze `4ec185c5...`,
   follow-ups 1 to 16, and the OSF links).
3. **Related works.** Add the eight OSF links of the record's `zenodo.json`, each with the relation "Is documented by" and the
   resource type "Other": https://osf.io/qtz3v, https://osf.io/fegsy, https://osf.io/nsbgp, https://osf.io/64bpy,
   https://osf.io/c4kh7, https://osf.io/eafkx, https://osf.io/h9yw8, https://osf.io/hcxm7. Add the keyword `pre-registration`.
4. **Check.** Compare the MD5 Zenodo shows for the new archives with `md5 -q FILE`, the large ones at least; the number of files
   (41 and 88) and the total size against A7's figures. For R2, check that the licence is still the custom "AlphaFold 3 Output
   Terms of Use" and no CC licence.
5. **Save** each draft. Publish by hand only when the preprint goes out (row of 2026-11-09: publication with the arXiv preprint,
   after the repository is made public).

The browser must stay open and the laptop awake (`caffeinate -dims` in a terminal) while the files upload.

## Expected cost of Part A (indicative list prices, as in Part B; check the current price pages before running)

| item | quantity | price | cost |
|---|---|---|---|
| e2-standard-8, europe-west4, on demand | about 2.5 h (8 h at most) | about 0.30 USD/h | about 0.75 USD (2.40 at most) |
| pd-balanced boot disk, 150 GB | about 2.5 h (8 h at most) | about 0.11 USD per GB-month | about 0.06 USD (0.18 at most) |
| bucket to VM, same region | about 76 GB | no charge | 0 |
| Cloud Storage reads (class B operations) and listings | about 111,000 objects, plus the laptop's listing of about 485,000 | 0.0004 USD per 1,000 | about 0.25 USD |
| internet egress, VM to laptop (A6) | about 6.5 GiB | about 0.12 USD/GiB | about 0.80 USD |
| **total** | | | **about 2 USD (4 at most)** |

# Part B. The full build (the first build, 2026-11-09; for a rebuild from scratch)

These are the steps of the first build, kept as they ran (row of 2026-11-09). A rebuild of everything from scratch follows them
with today's script: `build` then needs `--paper_ref` for `analysis_outputs.tar.gz`, the bundle needs `LICENSES` and the new
freeze folder, and the counts and sizes are those of `layout` (R1 41 files, R2 88 files, about 45 GB) rather than those below.
The first build was made from tag `zenodo-v1`; a rebuild would tag its own commit, as Part A does.

## 0. Laptop: list the bucket and build the archives made locally

The listing only reads the bucket and creates nothing (a few minutes; about 400,000 objects).

```
gcloud storage ls -l -r 'gs://af2-tpu-benchmark-af3-results/af3/**' > results/release/inventory/bucket_listing.txt
python3 scripts/release/release.py inventory
python3 scripts/release/release.py layout
python3 scripts/release/release.py scan --root results/af3 --out $HOME/af3_release_scan
python3 scripts/release/release.py build --root results/af3 --out $HOME/af3_release --archives $(python3 scripts/release/release.py archives --built_on laptop)
python3 scripts/release/release.py verify --out $HOME/af3_release
```

Expected: `inventory` exits 0. If it exits 2, `unmapped_sessions.csv` lists sessions named like follow-ups 6 to 12 that no row of section
10 maps: stop, and either add the row's mapping to `OUTSIDE_RULES` in release.py or decide that the session stays out; do not
guess. `scan` takes about 10 minutes and exits 0, with the counts of the row of 2026-11-08 (no secret, no private address). `build` writes
7 R1 archives and 5 R2 archives (`inputs_official_pipeline.tar.gz`, the three `nas_*` and `analysis_outputs.tar.gz`), "verify: OK". These R2 archives need no check-inputs: they are made from the
repository, not from session folders. It took about 14 minutes on 2026-11-08 (with the two local partial archives) and needs about 2 GB free. A refusal stops the build and removes what it wrote: see step 6.

## 1. Laptop: the bundle the VM needs

The VM needs the release script, the freeze's session list and comparison (to classify sessions), the frozen inputs of every
follow-up (for check-inputs), their manifests, and the lists of sessions to copy. No weights, no outputs.

```
mkdir -p $HOME/af3_release_bundle
awk -F, 'NR > 1 && $2 != "outside" { print $1 }' results/release/inventory/sessions.csv > $HOME/af3_release_bundle/sessions_release.txt
awk -F, 'NR > 1 && $2 == "outside" { print $1 }' results/release/inventory/sessions.csv > $HOME/af3_release_bundle/sessions_outside.txt
COPYFILE_DISABLE=1 tar czf $HOME/af3_release_bundle/repo.tgz scripts/release results/freeze/followups612_20261108/sessions.tsv results/freeze/followups612_20261108/comparison.tsv data/inputs data/inputs_repl data/inputs_large data/inputs_official_data inputs/manifest.csv inputs/ppi/manifest.csv inputs/repl/manifest.csv inputs/large/manifest.csv inputs/official_data/manifest.csv
wc -l $HOME/af3_release_bundle/sessions_release.txt $HOME/af3_release_bundle/sessions_outside.txt
ls -lh $HOME/af3_release_bundle/repo.tgz
```

Expected: at most 832 lines in `sessions_release.txt` (288 frozen sessions, 15 superseded, up to 529 attempts) and at most 384 in
`sessions_outside.txt` (the counts of the estimate of 2026-11-08; launches that wrote nothing to the bucket, 43 of them, have no line);
a `repo.tgz` of about 1 GB (3.0 GB of frozen inputs with their MSAs and alignments, compressed).

## 2. Laptop: create the VM (cloud resource; Lorenzo)

```
gcloud compute instances create af3-release-build --project=af2-tpu-benchmark --zone=europe-west4-a --machine-type=e2-standard-8 --image-family=debian-12 --image-project=debian-cloud --boot-disk-size=500GB --boot-disk-type=pd-balanced --scopes=storage-ro --max-run-duration=16h --instance-termination-action=DELETE
```

`--scopes=storage-ro`: the VM can read the results bucket and write nowhere in Cloud Storage. With
`--instance-termination-action=DELETE`, Google deletes the VM after 16 hours even if the last step is forgotten.

Disk, from the bucket sizes in the final freeze: the released sessions take about 325 GiB (of which `*_data.json`, needed only by
the check, 101 GiB, and full confidences 186 GiB), the archives about 36 GiB and the bundle 3 GiB: about 365 GiB, or 392 GB. A 400 GB
disk would hold them with almost nothing to spare, so the disk is 500 GB; the difference costs about 0.15 USD.

Duration, an estimate from the rates measured on the laptop (gzip 27 MB/s, the privacy checks 34 MB/s on full confidences), taken
lower on the VM's slower vCPUs:

| step | what | expected |
|---|---|---|
| 4 | copy about 370,000 objects, 325 GiB, within the region | about 2 h |
| 5 | check-inputs: about 100 GiB of `*_data.json` | about 30 min |
| 6 | build: about 250 GB raw through the guard (about 2.5 h) and gzip (about 3.5 h), one archive after another | about 6 h |
| 6 | verify: every member of every archive | about 1 h |
|   | **total, without surprises** | **about 10 h** |

Hence `--max-run-duration=16h`. If the build is the bottleneck, the duration can be cut by running archives in parallel, but they would
have to be merged by hand (`REDACTIONS.tsv`, `SHA256SUMS`): not worth it for one run.

## 3. Laptop: copy the bundle to the VM

```
gcloud compute scp --project=af2-tpu-benchmark --zone=europe-west4-a $HOME/af3_release_bundle/repo.tgz $HOME/af3_release_bundle/sessions_release.txt $HOME/af3_release_bundle/sessions_outside.txt af3-release-build:~/
gcloud compute ssh af3-release-build --project=af2-tpu-benchmark --zone=europe-west4-a --command='mkdir -p ~/repo && tar xzf ~/repo.tgz -C ~/repo && python3 --version && ls ~/repo'
```

## 4 to 6. VM: copy, check, build and verify, detached

`scripts/release/vm_steps.sh` runs the three steps in order and stops at the first failure. It runs detached, so that a lost SSH
connection or a sleeping laptop does not stop it:

```
gcloud compute ssh af3-release-build --project=af2-tpu-benchmark --zone=europe-west4-a --command='nohup bash ~/repo/scripts/release/vm_steps.sh all > ~/vm_steps.log 2>&1 < /dev/null & disown; echo started'
gcloud compute ssh af3-release-build --project=af2-tpu-benchmark --zone=europe-west4-a --command='grep -E "start|done|FAILED|!!|verify|status" ~/vm_steps.log | cut -c1-200; tail -2 ~/vm_steps.log | cut -c1-200; df -h ~ | tail -1'
```

The second command shows the progress, and can be repeated. A failed step can be run again alone: `vm_steps.sh copy`, `check` or `build`.

**Step 4, copy** (same region: no egress). Sessions outside the freeze are copied without their AlphaFold3 outputs. Expected: no FAILED
line and as many folders as sessions in the two lists. A rerun copies only what is missing.

**Step 5, every `*_data.json` against its frozen input.** Expected: `"status": "pass"`, every process `same` except failed processes
(memory-limit sizes that ran out of memory) without a data JSON. Any `differs` stops the release: the build refuses to start, and
`~/release/inputs_check.csv` names the process and the field. What is compared: name and dialect; modelSeeds equal
to the process's seeds and among the input's, except that follow-up 8 (kind `seeds`) may use seeds 6 to 20, which its frozen inputs do
not list (`SEEDS_BEYOND_INPUT`; no other kind and no other seed); bondedAtomPairs and userCCD absent or empty in both; the same sequence
entries in the same order, each with every field equal (id, sequence, unpairedMsa, pairedMsa, templates, ccdCodes,
...), a field missing from the frozen input being allowed only as AlphaFold3's default (`modifications: []`); the
frozen input's SHA-256 equal to the one the run recorded. Not compared: AlphaFold3's dialect `version`. If the status
is `incomplete` (processes that exited 0 without a data JSON), stop and look before building.

**Step 6, build and verify.** Expected: 31 R2 archives (about 37 GB, the largest `af3_large_full_confidences.tar.gz` at about 9 GB), and
`attempt_superseded_records.tar.gz` and `outside_freeze_records.tar.gz` in R1, "verify: OK". With the 5 R2 archives from step 0, R2 has
36.

A refusal stops the build and removes what it wrote. It is one of: weights by name, size or SHA-256; a file over 1 GiB; a bucket other
than the results bucket named in a text file; an email address in a text file that is not on the redaction list; a WireGuard or ssh
key or an API token in a text file; a redaction that would change an AlphaFold3 output file; a superseded session without the official
weights; a session no row maps; or no passing check-inputs. The scan of 2026-11-08 covered every releasable file on the laptop (278,833
members). Files that exist only in the bucket (the full confidences, and the records of sessions outside the freeze with no local
folder) are scanned here for the first time, so a refusal is possible here and only here. If one is refused for an email address, stop,
look at it, and add it to `records/redact_email_paths.txt` only if the address is to be redacted, with a reason in `REDACT_REASONS`. If
one is refused for a secret, stop: revoke the secret, and only then decide what to do with the file.

## 7. The two ways on from here

Choose by how the records will be uploaded (last section), since each sends the R2 archives over the internet once:

- **Web interface, from the laptop:** do steps 7a and 8, then delete the VM (step 9), then upload.
- **Zenodo API, from the VM:** do step 7b and upload from the VM, then delete the VM. No copy of R2 comes back to the laptop.

### 7a. Laptop: copy the archives back (internet egress: about 35 GiB)

```
mkdir -p $HOME/af3_release/R1 $HOME/af3_release/R2 $HOME/af3_release/from_vm/R1 $HOME/af3_release/from_vm/R2
gcloud compute scp --project=af2-tpu-benchmark --zone=europe-west4-a 'af3-release-build:~/release/R2/*.tar*' 'af3-release-build:~/release/R2/*.sha256' $HOME/af3_release/R2/
gcloud compute scp --project=af2-tpu-benchmark --zone=europe-west4-a 'af3-release-build:~/release/R1/attempt_superseded_records.tar.gz*' 'af3-release-build:~/release/R1/outside_freeze_records.tar.gz*' $HOME/af3_release/R1/
gcloud compute scp --project=af2-tpu-benchmark --zone=europe-west4-a 'af3-release-build:~/release/inputs_check.*' $HOME/af3_release/from_vm/
gcloud compute scp --project=af2-tpu-benchmark --zone=europe-west4-a 'af3-release-build:~/release/R1/REDACTIONS.tsv' $HOME/af3_release/from_vm/R1/
gcloud compute scp --project=af2-tpu-benchmark --zone=europe-west4-a 'af3-release-build:~/release/R2/REDACTIONS.tsv' $HOME/af3_release/from_vm/R2/
gcloud compute ssh af3-release-build --project=af2-tpu-benchmark --zone=europe-west4-a --command='cd ~/release && sha256sum R2/*.tar* R2/*.sha256 R1/attempt_superseded_records.tar.gz* R1/outside_freeze_records.tar.gz*' > $HOME/af3_release/vm_sha256.txt
```

The last line records the SHA-256 the VM computed, to compare with the laptop's copies in step 8. The patterns are quoted so that zsh
does not expand them on the laptop. Both REDACTIONS.tsv files exist: every AlphaFold3 log on the VM has a home path replaced.

### 7b. VM: bring the laptop's archives to it (no egress: ingress is free)

```
gcloud compute ssh af3-release-build --project=af2-tpu-benchmark --zone=europe-west4-a --command='mkdir -p ~/repo_lists'
gcloud compute scp --project=af2-tpu-benchmark --zone=europe-west4-a $HOME/af3_release/R1/*.tar.gz $HOME/af3_release/R1/*.sha256 af3-release-build:~/release/R1/
gcloud compute scp --project=af2-tpu-benchmark --zone=europe-west4-a $HOME/af3_release/R2/inputs_official_pipeline.tar.gz* $HOME/af3_release/R2/nas_*.tar.gz* $HOME/af3_release/R2/analysis_outputs.tar.gz* af3-release-build:~/release/R2/
gcloud compute scp --project=af2-tpu-benchmark --zone=europe-west4-a $HOME/af3_release/R1/REDACTIONS.tsv af3-release-build:~/laptop_R1_REDACTIONS.tsv
gcloud compute scp --project=af2-tpu-benchmark --zone=europe-west4-a $HOME/af3_release/R2/REDACTIONS.tsv af3-release-build:~/laptop_R2_REDACTIONS.tsv
gcloud compute scp --project=af2-tpu-benchmark --zone=europe-west4-a results/release/layout.md results/release/layout.csv results/release/inventory/inventory.md results/release/inventory/summary.csv results/release/inventory/sessions.csv results/release/inventory/excluded_sessions.csv results/release/inventory/excluded_files.csv results/release/inventory/OUTSIDE_SESSIONS.csv af3-release-build:~/repo_lists/
gcloud compute ssh af3-release-build --project=af2-tpu-benchmark --zone=europe-west4-a --command='mkdir -p ~/repo/results/release/inventory && cd ~/repo_lists && mv layout.* ~/repo/results/release/ && mv * ~/repo/results/release/inventory/ && cd ~ && python3 repo/scripts/release/release.py merge-redactions --out release --record R1 --from laptop_R1_REDACTIONS.tsv && python3 repo/scripts/release/release.py merge-redactions --out release --record R2 --from laptop_R2_REDACTIONS.tsv && cp release/inputs_check.csv release/R1/release_inputs_check.csv && cp release/inputs_check.json release/R1/release_inputs_check.json && python3 repo/scripts/release/release.py sums --out release && python3 repo/scripts/release/release.py verify --out release && ls release/R1 | wc -l && ls release/R2 | wc -l'
```

Expected: "verify: OK", 35 files in R1 and 76 in R2 (at most 100 per record; `sums` stops if either
is over Zenodo's limits). Then go to the last section, API form.

## 8. Laptop: finish and verify the two records (after 7a)

```
(cd $HOME/af3_release && shasum -a 256 -c vm_sha256.txt)
python3 scripts/release/release.py merge-redactions --out $HOME/af3_release --record R1 --from $HOME/af3_release/from_vm/R1/REDACTIONS.tsv
python3 scripts/release/release.py merge-redactions --out $HOME/af3_release --record R2 --from $HOME/af3_release/from_vm/R2/REDACTIONS.tsv
cp $HOME/af3_release/from_vm/inputs_check.csv $HOME/af3_release/R1/release_inputs_check.csv
cp $HOME/af3_release/from_vm/inputs_check.json $HOME/af3_release/R1/release_inputs_check.json
python3 scripts/release/release.py sums --out $HOME/af3_release
python3 scripts/release/release.py verify --out $HOME/af3_release
ls $HOME/af3_release/R1 | wc -l
ls $HOME/af3_release/R2 | wc -l
```

Expected: every copied file "OK" against the VM's SHA-256; "verify: OK"; R1: 9 archives, 35 files; R2: 36 archives, 76 files (Zenodo:
at most 100 per record and 50 GB; `sums` stops otherwise). `sums` adds each record's README.md and LICENCES.md (from
`scripts/release/records/`) and, for R1, the release tooling and lists as `release_*` files (a Zenodo record has no folders), then writes
SHA256SUMS. `verify` reads all of R2 and takes about 30 minutes on the laptop.

## 9. Laptop: delete the VM (Lorenzo)

```
gcloud compute instances delete af3-release-build --project=af2-tpu-benchmark --zone=europe-west4-a --quiet
gcloud compute instances list --project=af2-tpu-benchmark --filter='name=af3-release-build'
```

Expected: the list prints nothing (the boot disk is deleted with the VM).

## Expected cost (indicative list prices; check the current price pages before running)

| item | quantity | price | cost |
|---|---|---|---|
| e2-standard-8, europe-west4, on demand | about 10 h (16 h at most: `--max-run-duration`) | about 0.30 USD/h | about 3.00 USD (4.80 at most) |
| pd-balanced boot disk, 500 GB | about 10 h (16 h at most) | about 0.11 USD per GB-month | about 0.75 USD (1.20 at most) |
| bucket to VM, same region | about 325 GiB | no charge | 0 |
| Cloud Storage reads (class B operations) and listings | about 370,000 objects | 0.0004 USD per 1,000 | about 0.15 USD |
| internet egress, once: VM to laptop (7a) or VM to Zenodo (API form) | about 35 GiB | about 0.12 USD/GiB | about 4.20 USD |
| **total** | | | **about 8 USD (10.50 at most)** |

For comparison, building everything on the laptop would download about 325 GiB, of which about 270 GiB are not already on the laptop (about 32
USD of egress). Sending the archives both to the laptop and from the VM to Zenodo would double the egress line, so do only one.

## Upload to Zenodo (Lorenzo; last, and nothing here publishes)

Both forms leave two drafts, with the DOIs reserved before step 0. Lorenzo checks each draft (files, sizes, metadata, licence, DOI, a
README that opens) and presses Publish by hand; a published record cannot be deleted. Zenodo's limits apply per record: 50 GB and 100
files, with no folders.

The arXiv preprint and the conference version are added later as related works ("is supplement to") in the Zenodo metadata of each
record, with no new version needed. The READMEs and `zenodo.json` already point there.

### Form 1: the web interface, from the laptop (after steps 7a, 8 and 9)

1. Open the draft of R1 at https://zenodo.org/uploads, the one with the reserved DOI.
2. Drag in every file of `$HOME/af3_release/R1` (35 files, about 0.5 GB), and wait until each shows as finished.
3. Fill in the form from `scripts/release/records/R1/zenodo.json` (title, creators and ORCID, description, keywords, licence, related
   identifiers, notes for the contact).
4. Save the draft. Do the same for R2 with `$HOME/af3_release/R2` (76 files, about 38 GB). **R2's licence is a custom one:** in the
   Licenses field remove the default licence the draft was saved with, choose "Add custom", title `AlphaFold 3 Output Terms of Use`,
   link `https://github.com/google-deepmind/alphafold3/blob/main/OUTPUT_TERMS_OF_USE.md` (the `rights` entry of R2's `zenodo.json`).
5. Compare the MD5 Zenodo shows for a few files with `md5 -q FILE`, the large ones at least.
6. Preview, check, and publish by hand, R1 first or R2 first as you prefer: each DOI already exists. **Before publishing R2, check that
   the preview shows the custom licence and no CC licence.** R2's outputs are not licensed CC BY or CC0.

The browser must stay open and the laptop awake (`caffeinate -dims` in a terminal). With a slow or metered link, R2 takes many hours and
the largest archive is the one most likely to fail: use the API form instead.

### Form 2: the Zenodo API, on the VM (after step 7b, before step 9)

Create the token on https://zenodo.org/account/settings/applications/ with the `deposit:write` scope only, so that it cannot publish
(publishing needs `deposit:actions`). Open an interactive shell on the VM, so that the token is not on any command line of the laptop:

```
gcloud compute ssh af3-release-build --project=af2-tpu-benchmark --zone=europe-west4-a
```

The rest runs in that shell. The token is typed at the prompt, not echoed, not in the shell's history and never written to a file; it
lives in this shell's memory until the shell exits.

```
read -rs ZENODO_TOKEN; echo "token read: ${#ZENODO_TOKEN} characters"
API=https://zenodo.org/api/deposit/depositions
R1_ID=24057470   # the end of the R1 draft's URL, the same number as its DOI: check
R2_ID=24057544   # the same for R2
upload() {  # upload ID FOLDER: every file of FOLDER into the draft ID
  bucket=$(curl -sS --fail -H "Authorization: Bearer $ZENODO_TOKEN" "$API/$1" | python3 -c 'import json,sys; print(json.load(sys.stdin)["links"]["bucket"])')
  for f in "$2"/*; do
    echo "== $f"
    curl -sS --fail --retry 3 --retry-delay 30 --retry-all-errors -o /dev/null -A 'alphafold-on-tpu-release' -H "Authorization: Bearer $ZENODO_TOKEN" --upload-file "$f" "$bucket/$(basename "$f")" || echo "FAILED $f"
  done
}
upload "$R1_ID" ~/release/R1 2>&1 | tee ~/upload_R1.log
upload "$R2_ID" ~/release/R2 2>&1 | tee ~/upload_R2.log
grep -c FAILED ~/upload_R1.log ~/upload_R2.log
```

Expected: 0 and 0. Uploading a file again replaces it, so a failed one can simply be sent again. If the connection may drop, run this inside `tmux` (`sudo apt-get install -y tmux`): typed into the `tmux` shell, the token stays in that shell only.

Check what arrived against the local files, by name, size and MD5:

```
check() {  # check ID FOLDER
  curl -sS --fail -H "Authorization: Bearer $ZENODO_TOKEN" "$API/$1/files" > ~/zenodo_files_$1.json
  python3 - "$2" ~/zenodo_files_$1.json <<'PY'
import hashlib, json, pathlib, sys
folder, listing = pathlib.Path(sys.argv[1]), json.load(open(sys.argv[2]))
remote = {f['filename']: f for f in listing}
local = {p.name: p for p in folder.iterdir() if p.is_file()}
print('files: local', len(local), 'remote', len(remote), 'missing', sorted(set(local) - set(remote)), 'extra', sorted(set(remote) - set(local)))
bad = 0
for name in sorted(set(local) & set(remote)):
  h = hashlib.md5()
  with open(local[name], 'rb') as f:
    while block := f.read(1 << 22):
      h.update(block)
  if h.hexdigest() != remote[name]['checksum'].replace('md5:', ''):
    print('DIFFERS', name)
    bad += 1
print('differing', bad)
PY
}
check "$R1_ID" ~/release/R1
check "$R2_ID" ~/release/R2
```

Expected: no file missing or extra, `differing 0`.

The metadata: **only R1 goes through this endpoint.** The deposit API takes only licence ids and, by its reference, defaults to `cc-zero`
for a dataset, with no field for a custom licence (developers.zenodo.org, read on 2026-11-08), so sending R2's metadata this way could
publish it under a licence the Terms do not allow. R2's metadata, licence included, is filled in by hand in the form, as in form 1, steps
3, 4 and 6. A `PUT` also replaces the whole metadata of the draft: `zenodo.json` has no affiliation or funding, so add them in the form
afterwards if the draft should carry them. R1, from its `zenodo.json`:

```
curl -sS --fail -o ~/put_R1.json -X PUT -H 'Content-Type: application/json' -H "Authorization: Bearer $ZENODO_TOKEN" --data @$HOME/repo/scripts/release/records/R1/zenodo.json "$API/$R1_ID"
python3 -c "import json,os; m=json.load(open(os.path.expanduser('~/put_R1.json')))['metadata']; print(m.get('license'), m.get('doi') or m.get('prereserve_doi'))"
unset ZENODO_TOKEN
exit
```

Expected: `curl` exits 0 (a 400
prints an error naming the field); the last line prints `cc-by-4.0` and `10.5281/zenodo.24057470`. If the DOI differs, stop. Then delete the
VM (step 9), revoke the token on the same Zenodo page, fill in R2's form (custom licence included) and check both drafts in the browser, and
publish by hand. The API form leaves no copy of R2 on the
laptop: keep a copy of `SHA256SUMS` of each record, which `results/release/` does not hold, by copying it from the draft.

# Part C. Later versions (new follow-ups after the arXiv preprint)

A later version adds the archives of its new labels and the updated READMEs, lists, manifests and freeze files, and keeps every
other archive of the previous version byte for byte: those are neither rebuilt nor uploaded again. The steps:

1. **Freeze extension.** Extend `scripts/freeze/make_freeze_manifest.py` with a new flag that implies the previous one (as
   `--followups1316` implies `--followups612`), run it as in the row of 2026-11-17, and commit the new freeze with its row of
   section 10. `results/freeze/INDEX.md` then names it as final.
2. **One entry per label in `release.py`.** Point `FREEZE` and `FREEZE_ID` at the new freeze; add a batch to `BATCHES` (for example
   `'followups17': {'freeze': ..., 'freeze_id': FREEZE_ID, 'sessions': N, 'what': ...}`) and set `CURRENT_BATCH` to it; add one
   entry to `LATER_LABELS` for each new label: its freeze kind, the plan prefixes of its session names, the batch, its R2 archive
   (one per follow-up, structures, full confidences and logs; a further archive only for data of another kind, as the XLA dumps
   of follow-up 16) and the rules that map each of its sessions outside the freeze to its row of section 10. Nothing else in the
   script changes: the archives, the R1 records of the batch (`session_records_<batch>.tar.gz`,
   `outside_freeze_records_<batch>.tar.gz`), the classification and the VM lists follow from the entry. Run the tests: the archives
   of the earlier batches keep their members (`test_first_build_archives_keep_their_members_without_the_new_kinds`); add the
   new labels to the parametrised tests. Add a row for each new archive to the README of its record, and update the session count
   and freeze identifier in both READMEs and both `zenodo.json`.
3. **Laptop part**, as steps A1 to A3 of Part A, with the new tag, `--paper_ref` naming the paper commit of that version, and
   `--batch` the new batch.
4. **VM part**, as steps A4 to A6 with `BATCH=` the new batch: only the new sessions are copied and only the new archives built.
5. **Finish**, as step A7, with `--previous` the folder where the build records of the version before were kept (A8 says which:
   for the version after this one, `results/release/build_20261117/`, or the date of the actual build).
6. **Zenodo.** If the records are still drafts, change the draft's files as `zenodo_changes_R1.csv` and `zenodo_changes_R2.csv`
   say (Part A, A10). If they are published, press "New version" on each record: Zenodo starts the new version with every file of
   the previous one; then delete, replace and add the files as the two lists say, edit the description, and publish by hand. A new
   version gets its own DOI; the concept DOI of each record stays and always resolves to the latest version.

Limits: R2 holds about 44.6 GB and 88 files after this release (step A7 prints the exact figures), so a later version can add
at most about 5 GB and 12 files to it. A follow-up larger than that needs a third record: add `R3` to `RECORDS`, `PLAIN_FILES` and
`ZENODO_IDS`, set the record of its archives to `'R3'` in its `LATER_LABELS` entry, and reserve a DOI for it first.
