# Building the Zenodo release (runbook; not yet run)

Plan of 2026-10-15: the archives made from the repository and the local freeze copy on the laptop (R1's
inputs, freeze, session lists and session records, and R2's `analysis_outputs.tar.gz`); every archive that needs
the bucket (the rest of R2, and R1's `attempt_superseded_records.tar.gz`) on a VM in the bucket's region
(europe-west4), from a copy of the bucket made there. Only the finished archives come back to the laptop. The
VM is deleted at the end. Nothing is uploaded to Zenodo by these steps.

Commands are written out in full for zsh on the laptop, from the repository root. Steps marked VM run on the VM
through `gcloud compute ssh --command`. Lorenzo runs every step that creates or deletes a cloud resource.

## 0. Laptop: the archives made locally (no cloud)

```
python3 scripts/release/release.py inventory
python3 scripts/release/release.py layout
python3 scripts/release/release.py build --root results/af3 --out $HOME/af3_release --archives inputs_targets_manifests.tar.gz,freeze.tar.gz,session_lists_launch_records.tar.gz,session_records.tar.gz,analysis_outputs.tar.gz
python3 scripts/release/release.py verify --out $HOME/af3_release
```

Expected: 4 R1 archives (about 0.2 GB) and R2's `analysis_outputs.tar.gz` (about 14 MB, with the AlphaFold3 terms
files), "verify: OK". `analysis_outputs.tar.gz` needs no check-inputs: it is made from the repository, not from
session folders. Needs about 1 GB free.

## 1. Laptop: the bundle the VM needs

The VM needs the release script, the freeze's session list and comparison (to classify sessions), the frozen inputs
(for check-inputs) and the list of sessions to copy. No weights, no outputs.

```
mkdir -p $HOME/af3_release_bundle
awk -F, 'NR > 1 { print $1 }' results/release/inventory/sessions.csv > $HOME/af3_release_bundle/sessions_release.txt
tar czf $HOME/af3_release_bundle/repo.tgz scripts/release results/freeze/sessions.tsv results/freeze/comparison.tsv data/inputs inputs/manifest.csv inputs/ppi/manifest.csv
wc -l $HOME/af3_release_bundle/sessions_release.txt
```

Expected (inventory of 2026-10-15): 144 sessions, namely 83 of the freeze (main 53, pilot2 13, memlimit 4, ppi 9,
embed 4), 15 superseded or engineering, and 46 attempts that never ran (launcher records only); and a `repo.tgz` of
about 160 MB (the frozen inputs carry their MSAs).

## 2. Laptop: create the VM (cloud resource; Lorenzo)

```
gcloud compute instances create af3-release-build --project=af2-tpu-benchmark --zone=europe-west4-a --machine-type=e2-standard-8 --image-family=debian-12 --image-project=debian-cloud --boot-disk-size=200GB --boot-disk-type=pd-balanced --scopes=storage-ro --max-run-duration=6h --instance-termination-action=DELETE
```

`--scopes=storage-ro`: the VM can read the results bucket and write nowhere in Cloud Storage. `--max-run-duration=6h`
with `--instance-termination-action=DELETE`: Google deletes the VM after 6 hours even if step 9 is forgotten. 200 GB
holds the bucket copy of the released sessions (about 80 GiB with the `*_data.json` files the check needs) and the
archives (about 12 GiB).

## 3. Laptop: copy the bundle to the VM

```
gcloud compute scp --project=af2-tpu-benchmark --zone=europe-west4-a $HOME/af3_release_bundle/repo.tgz $HOME/af3_release_bundle/sessions_release.txt af3-release-build:~/
gcloud compute ssh af3-release-build --project=af2-tpu-benchmark --zone=europe-west4-a --command='mkdir -p ~/repo && tar xzf ~/repo.tgz -C ~/repo && python3 --version && ls ~/repo'
```

## 4. VM: copy the released sessions from the bucket (same region: no egress)

```
gcloud compute ssh af3-release-build --project=af2-tpu-benchmark --zone=europe-west4-a --command='mkdir -p ~/bucket/af3 && while read -r s; do gcloud storage rsync --recursive "gs://af2-tpu-benchmark-af3-results/af3/$s" "$HOME/bucket/af3/$s" > /dev/null || echo "FAILED $s"; done < ~/sessions_release.txt; du -sh ~/bucket/af3; ls ~/bucket/af3 | wc -l'
```

Expected: no FAILED line, about 80 GiB, 144 folders, one per session of the list (sessions whose only bucket files are
launcher records included).

## 5. VM: every *_data.json against its frozen input

```
gcloud compute ssh af3-release-build --project=af2-tpu-benchmark --zone=europe-west4-a --command='python3 ~/repo/scripts/release/release.py check-inputs --root ~/bucket/af3 --repo ~/repo --out ~/release'
```

Expected: `"status": "pass"`, every process `same` except failed processes (memory-limit sizes that ran out of
memory) without a data JSON. Any `differs` stops the release: the build in step 6 refuses to start, and
`~/release/inputs_check.csv` names the process and the field. What is compared: name and dialect; modelSeeds equal
to the process's seeds and among the input's; bondedAtomPairs and userCCD absent or empty in both; the same sequence
entries in the same order, each with every field equal (id, sequence, unpairedMsa, pairedMsa, templates, ccdCodes,
...), a field missing from the frozen input being allowed only as AlphaFold3's default (`modifications: []`); the
frozen input's SHA-256 equal to the one the run recorded. Not compared: AlphaFold3's dialect `version`. If the status
is `incomplete` (processes that exited 0 without a data JSON), stop and look before building.

## 6. VM: build and verify

```
gcloud compute ssh af3-release-build --project=af2-tpu-benchmark --zone=europe-west4-a --command='python3 ~/repo/scripts/release/release.py build --root ~/bucket/af3 --repo ~/repo --out ~/release --archives af3_main_structures.tar.gz,af3_main_full_confidences.tar.gz,af3_pilot2_structures.tar.gz,af3_pilot2_full_confidences.tar.gz,af3_memlimit.tar.gz,af3_ppi_structures.tar.gz,af3_ppi_full_confidences.tar.gz,af3_embed_structures.tar.gz,af3_embed_embeddings.tar,af3_logs.tar.gz,af3_superseded_engineering.tar.gz,attempt_superseded_records.tar.gz && python3 ~/repo/scripts/release/release.py verify --out ~/release && du -sh ~/release/R1 ~/release/R2'
```

Expected: 11 R2 archives (about 11.4 GB, the largest `af3_main_full_confidences.tar.gz` at about 4 GB) and
`attempt_superseded_records.tar.gz` in R1, "verify: OK". With `analysis_outputs.tar.gz` from step 0, R2 has 12. A refusal (weights by name, size or SHA-256, a file over
1 GiB, a bucket other than the results bucket named in a text file, an email address in a text file that is not on
the redaction list, a superseded session without the official weights, or no passing check-inputs) stops the build
and removes what it wrote. The email scan of 2026-10-15 covered every releasable file on the laptop (86,580 text
files: only two wrapper logs, both on the redaction list); files that exist only in the bucket (the full confidences,
and bucket copies of logs) are scanned here for the first time. If one is refused for an email address, stop, look
at it, and add it to `REDACT` in release.py only if the address is to be redacted.

## 7. Laptop: copy only the archives back (internet egress: about 11.5 GiB)

```
mkdir -p $HOME/af3_release/R1 $HOME/af3_release/R2
gcloud compute scp --project=af2-tpu-benchmark --zone=europe-west4-a 'af3-release-build:~/release/R2/*.tar*' 'af3-release-build:~/release/R2/*.sha256' $HOME/af3_release/R2/
gcloud compute scp --project=af2-tpu-benchmark --zone=europe-west4-a 'af3-release-build:~/release/R1/attempt_superseded_records.tar.gz*' 'af3-release-build:~/release/inputs_check.*' $HOME/af3_release/R1/
gcloud compute ssh af3-release-build --project=af2-tpu-benchmark --zone=europe-west4-a --command='cd ~/release && sha256sum R2/*.tar* R1/attempt_superseded_records.tar.gz*' > $HOME/af3_release/vm_sha256.txt
```

The last line records the SHA-256 the VM computed, to compare with the laptop's copies in step 8. The VM writes no
REDACTIONS.tsv with the current redaction list (both redacted files are built on the laptop in step 0); if
`ls ~/release/R1 ~/release/R2` on the VM shows one, copy it too and add its rows to the laptop's file of the same record. The patterns
are quoted so that zsh does not expand them on the laptop. `inputs_check.csv` and `inputs_check.json` go with R1
(they document step 5).

## 8. Laptop: finish and verify the two records

```
(cd $HOME/af3_release && shasum -a 256 -c vm_sha256.txt)
mv $HOME/af3_release/R1/inputs_check.* $HOME/af3_release/R1/release/ 2>/dev/null || (mkdir -p $HOME/af3_release/R1/release && mv $HOME/af3_release/R1/inputs_check.* $HOME/af3_release/R1/release/)
python3 scripts/release/release.py sums --out $HOME/af3_release
python3 scripts/release/release.py verify --out $HOME/af3_release
ls $HOME/af3_release/R1 $HOME/af3_release/R2 | wc -l
```

Expected: every copied file "OK" against the VM's SHA-256; "verify: OK"; R1: 5 archives, 25 files with REDACTIONS.tsv and
`release/` (inputs_check included); R2: 12 archives, 26 files (Zenodo: at most 100 per record and 50 GB). `sums` adds each record's README.md (from `scripts/release/records/`) and, for
R1, the release tooling and lists, then writes SHA256SUMS.

## 9. Laptop: delete the VM (Lorenzo)

```
gcloud compute instances delete af3-release-build --project=af2-tpu-benchmark --zone=europe-west4-a --quiet
gcloud compute instances list --project=af2-tpu-benchmark --filter='name=af3-release-build'
```

Expected: the list prints nothing (the boot disk is deleted with the VM).

## Expected cost (indicative list prices; check the current price pages before running)

| item | quantity | price | cost |
|---|---|---|---|
| e2-standard-8, europe-west4, on demand | about 3 h (6 h at most: `--max-run-duration`) | about 0.30 USD/h | about 0.90 USD (1.80 at most) |
| pd-balanced boot disk, 200 GB | about 3 h (6 h at most) | about 0.11 USD per GB-month | about 0.10 USD (0.20 at most) |
| bucket to VM, same region | about 80 GiB | no charge | 0 |
| Cloud Storage reads (class B operations) | about 105,000 objects | 0.0004 USD per 1,000 | about 0.05 USD |
| internet egress, VM to laptop (step 7) | about 11.5 GiB | about 0.12 USD/GiB | about 1.40 USD |
| **total** | | | **about 2.50 USD (3.50 at most)** |

For comparison, building everything on the laptop would download about 53.6 GiB (about 6.40 USD of egress), or
43.4 GiB (about 5.20 USD) reusing the outputs already on the laptop, and would still lack the `*_data.json` files
the input check needs (another 22.8 GiB, about 2.70 USD).
