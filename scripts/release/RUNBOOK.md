# Building the Zenodo release (runbook; the VM part not yet run)

Plan of 2026-10-15, extended to the whole study on 2026-11-08 (final freeze `689230120499`, 295 sessions; record R1 9 archives,
record R2 36 archives): the archives made from the repository and the local freeze copy on the laptop; every archive that needs
the bucket on a VM in the bucket's region (europe-west4), from a copy of the bucket made there. The VM is deleted at the end.
Nothing is uploaded to Zenodo before the last section, and nothing there publishes: Lorenzo publishes by hand after checking the
draft.

Commands are written out in full for zsh on the laptop, from the repository root. Steps marked VM run on the VM
through `gcloud compute ssh --command`. Lorenzo runs every step that creates or deletes a cloud resource.

The two DOIs are reserved (drafts, not published): R1 `10.5281/zenodo.24057470`, R2 `10.5281/zenodo.24057544`; they are already in
`scripts/release/records/{R1,R2}/README.md` and `zenodo.json`.

Before step 0: commit, then create and push the tag: `git tag -a zenodo-v1 -m "Zenodo release build"` and `git push origin zenodo-v1`;
build from that tag.

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
