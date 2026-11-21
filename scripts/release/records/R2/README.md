# AlphaFold3 on TPU and GPU: predicted structures, confidences and derived analysis outputs

Author: Lorenzo Pazienza, ORCID 0009-0006-1687-0125. Contact: lorenzo.pazienza@gmail.com.
Paper: see the related identifiers of this Zenodo record (arXiv preprint and conference version, added when available). The paper's supplementary material is the PDF `Pazienza_AlphaFold3_cross_platform_reproducibility_supplement.pdf` in this record. Companion record (inputs with ColabFold MSAs, targets, manifests, data freeze, run
records): 10.5281/zenodo.24057470. This record: 10.5281/zenodo.24057544. Code: https://github.com/lorenzopazienza/alphafold-on-tpu, the repository commit tagged zenodo-v1.
A copy of that commit is in record R1.

Pre-registration (OSF, public since 2026-11-17): project https://osf.io/qtz3v; the main study's registration
https://osf.io/fegsy; the addenda of follow-up 10 (https://osf.io/nsbgp), 11 (https://osf.io/64bpy), 13
(https://osf.io/c4kh7), 14 (https://osf.io/eafkx), 15 (https://osf.io/h9yw8) and 16 (https://osf.io/hcxm7).

## Terms of use of this record

**This record holds AlphaFold 3 Output, provided under and subject to the AlphaFold 3 Output Terms of Use.**

**By using this information, you agree to AlphaFold 3 Output Terms of Use found at
https://github.com/google-deepmind/alphafold3/blob/main/OUTPUT_TERMS_OF_USE.md.**

**To request access to the AlphaFold 3 model parameters, follow the process set out at
https://github.com/google-deepmind/alphafold3. You may only use these if received directly from Google. Use is
subject to terms of use available at https://github.com/google-deepmind/alphafold3/blob/main/WEIGHTS_TERMS_OF_USE.md.**

Every archive of this record holds the "Legally Binding Terms of Use" text file, `LEGALLY_BINDING_TERMS_OF_USE.txt`, with this
notice, and a copy of the Terms, `OUTPUT_TERMS_OF_USE.md`. Every AlphaFold3 output folder also keeps the `TERMS_OF_USE.md`
AlphaFold3 wrote.

In short (the Terms govern; this summary does not change them): the outputs are for non-commercial use only; they
must not be used to train or create machine learning models or related technology for biomolecular structure
prediction similar to AlphaFold 3, including via distillation; any distribution must keep this notice; and any use
must cite:

> Abramson, J. et al. Accurate structure prediction of biomolecular interactions with AlphaFold 3. *Nature* 630,
> 493-500 (2024). https://doi.org/10.1038/s41586-024-07487-w

**Licence of this record: the custom licence "AlphaFold 3 Output Terms of Use", at the address above. No Creative Commons or other
licence applies to this record.**

What counts as Output here: the Terms define it as "the structure predictions and all related information provided by AlphaFold 3,
together with any visual representations, computational predictions, descriptions, modifications, copies, or adaptations that are
substantially derived from Output", and "AlphaFold 3" as the AlphaFold 3 Code and Model Parameters. So the results of AlphaFold3's
data pipeline (follow-up 6), its feature arrays (follow-up 5) and the XLA dumps of follow-up 16 are treated as Output (the dumps are also compiled from AlphaFold 3's code, Apache License 2.0: see the table
below), although the pipeline and the feature arrays involve no model inference.

Modifications: we have not modified any model, confidence file, ranking score, embedding, feature array or pipeline `_data.json`;
each is as AlphaFold3 wrote it (SHA-256 in the per-archive manifests and in the data freeze), and the build stops rather than change
one. The only changes are the redactions in logs, records and XLA dumps listed in `REDACTIONS.tsv` (below).

Please also cite the paper when using these data. Paper: see the related identifiers of this Zenodo record (arXiv preprint and conference version, added when available).

## What this record contains

AlphaFold3 v3.0.4 outputs (official model parameters, AlphaFold3's defaults: 10 recycles, 5 diffusion samples per
seed) from the study, run on TPU v5e, TPU v6e, NVIDIA L4, A100 and H100, and CPU, covering the main study,
follow-ups 1 to 16 and the two later measurements `memlimit_ah` and `warm_th`.
**No model parameters (weights) are included.** They are available only from Google DeepMind
(https://github.com/google-deepmind/alphafold3).

The outputs of the study come from the 439 sessions of the data freeze
`4ec185c5a929738439de3b916a9c8475c4c3abd218fa140bfd460acf917d4486` (the SHA-256 of
`results/freeze/followups1316_20261117/manifest_sha256.tsv` in record R1, which lists every frozen file with its SHA-256). Other
archives hold other material, under the same terms: the results of AlphaFold3's data pipeline with the three deliveries of follow-up
6, which have their own freezes (below), the superseded or engineering sessions (not used in any reported result) and the analysis
outputs derived from the study's outputs.

| archive | sessions | contents |
|---|---|---|
| `af3_main_structures.tar.gz` | main study (53 sessions) | per-sample and top-level mmCIF models, `*_summary_confidences.json`, `*_ranking_scores.csv`, `TERMS_OF_USE.md` |
| `af3_main_full_confidences.tar.gz` | main study | per-sample and top-level `*_confidences.json` (per-atom pLDDT, PAE, contact probabilities) |
| `af3_pilot2_structures.tar.gz`, `af3_pilot2_full_confidences.tar.gz` | repeated pilot (13 sessions) | as the main study |
| `af3_memlimit.tar.gz` | memory-limit measurement (4 sessions; 7D5C padded to larger buckets, not real inputs of those sizes) | structures and full confidences |
| `af3_ppi_structures.tar.gz`, `af3_ppi_full_confidences.tar.gz` | follow-up 1, 30 protein-protein complexes (9 sessions) | as the main study |
| `af3_embed_structures.tar.gz` | follow-up 2, 10 pilot targets, seed 1 (4 sessions) | structures and full confidences |
| `af3_embed_embeddings.tar` | follow-up 2 | final trunk single and pair embeddings (`--save_embeddings`, float16 `.npz`) |
| `af3_a100_structures.tar.gz`, `af3_a100_full_confidences.tar.gz` | follow-up 3, A100 arms (12 sessions) | as the main study |
| `af3_cpu60_structures.tar.gz`, `af3_cpu60_full_confidences.tar.gz` | follow-up 4, the CPU arm on every target (54 sessions) | as the main study |
| `af3_feat_features.tar.gz` | follow-up 5, host-side featurisation (4 sessions, no model inference, no weights) | the feature arrays of each target and seed (`arrays/seed-<s>.npz`); their hashes are in `features.jsonl`, in record R1 |
| `af3_msa_structures.tar.gz`, `af3_msa_full_confidences.tar.gz` | follow-up 6, inference on AlphaFold3's own data pipeline (20 sessions) | as the main study |
| `af3_sw_structures.tar.gz`, `af3_sw_full_confidences.tar.gz` | follow-up 7, the software stack (9 sessions) | as the main study |
| `af3_seeds_structures.tar.gz`, `af3_seeds_full_confidences.tar.gz` | follow-up 8, seeds 6 to 20 (27 sessions) | as the main study |
| `af3_h100_structures.tar.gz`, `af3_h100_full_confidences.tar.gz` | follow-up 9, the H100 (6 sessions) | as the main study |
| `af3_large_structures.tar.gz`, `af3_large_full_confidences.tar.gz` | follow-up 10, larger inputs on the whole PDB (49 sessions) | as the main study |
| `af3_repl_structures.tar.gz`, `af3_repl_full_confidences.tar.gz` | follow-up 11, replication on new targets (18 sessions) | as the main study |
| `af3_prec5_structures.tar.gz`, `af3_prec5_full_confidences.tar.gz` | follow-up 12, the precision-matched arms to five seeds (6 sessions) | as the main study |
| `inputs_official_pipeline.tar.gz` | follow-up 6, AlphaFold3's own data pipeline | the sequence-only inputs given to the pipeline (`data/inputs_official/`, and the tarball of the same files), the inference inputs it produced (`data/inputs_official_data/`, 60 targets, plus the 2PV7 smoke input), with alignments and templates, and their manifest (`inputs/official_data/`: SHA-256, MSA rows, templates, source) |
| `nas_naspaz8-af3_run1.tar.gz` | follow-up 6, the delivery used for inference | the pipeline on all 60 targets, with its logs, results table and provenance (`data/nas/naspaz8-af3_run1/`), and its freeze (`results/freeze/msa_naspaz8-af3_run1/`) |
| `nas_naspaz6-af3_run1.tar.gz`, `nas_naspaz8-af3_run2.tar.gz` | follow-up 6, the two determinism checks | the pipeline on the 10 pilot targets, each with its freeze; never inference inputs. `failed/7D5C_20261025T223505Z` in `naspaz6-af3_run1` is an empty folder left by a first start stopped by hand after 102 s |
| `af3_logs.tar.gz` | main study and follow-ups 1 and 2 | `run_alphafold.log` of each AlphaFold3 process |
| `af3_logs_followups.tar.gz` | follow-ups 3 to 12 | the same |
| `af3_fu13.tar.gz` | follow-up 13, why the top-ranked sample flips (79 sessions) | structures, full confidences and `run_alphafold.log`, in one archive per follow-up from here on |
| `af3_fu14.tar.gz` | follow-up 14, the float32 trunk (44 sessions) | structures, full confidences and logs |
| `af3_rcx.tar.gz` | follow-up 15, recompilation of the float32 trunk and of the L4 Triton arm (15 sessions) | structures, full confidences and logs |
| `af3_hlo.tar.gz` | follow-up 16, the compiled model on the v5e and the v6e (2 sessions) | structures, full confidences and logs |
| `hlo_xla_dumps.tar.gz` | follow-up 16 | the XLA dumps of every process (`<run>/<target>/xla_dump/`, 21,696 files: HLO and MLIR text, buffer assignment and memory reports, compiler options), with the Apache License 2.0 of AlphaFold 3's code they are compiled from (`LICENSE_ALPHAFOLD3_CODE_APACHE-2.0.txt`) and `NOTICE_XLA_DUMPS.txt`. They hold no model parameters: the parameters are arguments of the compiled model, not constants, and the build refuses any dump with more than 65,536 characters of printed constants. Home paths are redacted |
| `af3_memlimit_ah_warm_th.tar.gz` | the memory limit on the A100 and H100 (`memlimit_ah`, 2 sessions; 7D5C padded to larger buckets) and the warm cost on the L4 Triton arm and the H100 (`warm_th`, 2 sessions) | structures, full confidences and logs |
| `analysis_outputs.tar.gz` | every analysis of the study and its follow-ups | the derived analysis outputs (`results/analysis/`: per-sample tables such as `samples.csv` with ranking scores, pLDDT, ipTM, M1 and DockQ, contrasts, reports and run logs, including every analysis after 8 November: `campaign`, `ranking`, `prec5_exploratory`, `memlimit_ah`, `warm_th`, `fu13`, `fu14`, `returned`, `rare`, `rounded`, `winners`, `cpu_matched`, `rcx`, `winners_ext`, `pb_flips`, `best25` and `hlo`), the tables and figures of the earlier analyses (`results/paper_assets/`, `results/paper_assets_*/`, with `sources.tsv`, the SHA-256 of every input they were made from), and the paper's own tables and figures (`paper/assets/`, from the repository commit tagged `zenodo-v1`, named in `PAPER_ASSETS_COMMIT.txt`); they carry values computed from AlphaFold3 outputs, so they are released here under the same terms. The code that made them is in the repository (`analysis/`, `scripts/analysis_runs/`) and in its copy in record R1, and the session lists they were run on are in record R1 |
| `af3_superseded_engineering.tar.gz` | **superseded or engineering, not used in any reported result** (15 sessions with the official weights: the first pilot, its V6h and Gh arms and V5 warm completion, the L4 determinism diagnostic, a GT smoke test, and one interrupted pilot session) | structures, full confidences and logs; `SUPERSEDED_SESSIONS.csv` maps each session to the section 10 row of the analysis plan that records it |
| `REDACTIONS.tsv` (plain file) | every file packaged with a redaction (below) |
| `LICENCES.md` (plain file) | every third-party source, its licence, the official page, and the licence line of each archive |
| `Pazienza_AlphaFold3_cross_platform_reproducibility_supplement.pdf` (plain file) | the paper's supplementary material; its tables and figures are computed from the AlphaFold 3 outputs of this record, so it is released with them under the Output Terms |

### Where the follow-up 6 data pipeline ran

| delivery | label and what it is | CPU and memory of the VM | OS release | role | recorded in |
|---|---|---|---|---|---|
| `naspaz8-af3_run1` | `naspaz8-af3`: an Ubuntu VM on a UGREEN DXP8800 Plus NAS running TrueNAS | 12 vCPU, 12th Gen Intel(R) Core(TM) i5-1235U, 104,616 MiB | Ubuntu 24.04.5 LTS | run1, the pipeline whose results were used for inference (60 targets) | `data/nas/naspaz8-af3_run1/provenance.txt` |
| `naspaz6-af3_run1` | `naspaz6-af3`: a clone of that VM on a UGREEN iDX6011 Pro NAS running TrueNAS | 16 vCPU, Intel(R) Core(TM) Ultra 7 255H, 48,166 MiB | Ubuntu 24.04.5 LTS | run1, determinism check on the other machine (10 pilot targets) | `data/nas/naspaz6-af3_run1/provenance.txt` |
| `naspaz8-af3_run2` | `naspaz8-af3`, as above | 12 vCPU, 12th Gen Intel(R) Core(TM) i5-1235U, 104,616 MiB | Ubuntu 24.04.5 LTS | run2, determinism check on the same machine (10 pilot targets) | `data/nas/naspaz8-af3_run2/provenance.txt` |

CPU, memory and OS are as the runner recorded them inside the VM (`vcpu`, `cpu`, `mem_total_mib`, `os`). The analysis plan gives the
memory assigned to the VMs as 104 GiB and 48 GiB (row of 2026-10-20). The labels stay because they are part of the frozen file names and
freeze identifiers; the machines' private addresses are not in this record.

The rows of section 10 of the analysis plan, on the role of each:

- Row of 2026-10-26: "Inference inputs, from `naspaz8-af3` run1 only: `data/inputs_official_data/<target>.json`, each the run1
  `<target>_data.json` byte-unchanged (SHA-256 checked after the copy)".
- Row of 2026-10-27: "Determinism: the `_data.json` SHA-256 of run2 and of `naspaz6-af3` equal run1's on 10 of 10 pilot targets, so no
  comparison by field was needed." And of the two determinism deliveries: "their runner output manifests are identical".

The deliveries were never in the results bucket and keep their own freezes (identifier: SHA-256 of `files_sha256.tsv`):

| delivery | freeze identifier | files, bytes |
|---|---|---|
| `naspaz8-af3_run1` | `3c21b4afbdfbcf310e7e289ea7ff326c2e1579ed18cea425a1616dd8f1c7b9da` | 184, 1,594,914,931 |
| `naspaz6-af3_run1` | `781f4ee3221df5f2e8f09b9dec91ad919a87c80480eba838eedb946cf0c131b2` | 34, 294,912,013 |
| `naspaz8-af3_run2` | `ed7f634e3e0832462ab52ea918f0d47954fee7a8ca8fda98c18add3c9ff3d0c4` | 34, 294,916,131 |

### Paths

Members of the session archives are `af3/<session>/<run>/<target>/af3_output/<target>/...`, as on the machines that ran them:

- `<session>`: `<UTC launch time>_<platform>_<plan>`, for example `20261013T014901Z_l4_main_g_01`;
- `<run>`: `<config>_rep<r>_seed<s>_fresh` (fresh compilation cache, repetition r), or `<config>_seed<s>_warm`
  (warm cache rerun); the config names the arm (`tpu_xla`: V5 or V6 by platform, `l4_xla`: G, `l4_triton`: GT,
  `l4_xla_det`: Gd, `tpu_xla_highest`: V6h, `l4_xla_highest`: Gh, `cpu_xla`: C, `a100_xla`: A, `a100_xla_highest`: Ah,
  `a100_triton`: AT; the configs of follow-ups 13 to 16 and of the two later measurements are named in their plans,
  `harness/plans/` in the repository, and in each `session.json`);
- `seed-<s>_sample-<k>/`: one diffusion sample; the top-level `<target>_model.cif` and `<target>_confidences.json`
  are AlphaFold3's copies of its top-ranked sample.

The run records of each session (`session.json`, `runs.jsonl`, `run.json`, launch records) are in record R1
(`session_records.tar.gz`, `session_records_followups.tar.gz`, `session_records_followups1316.tar.gz`), under the same paths.

### Left out

- `*_data.json`: AlphaFold3's copy of each run's input (about 100 GiB). Before these archives were built, every one of
  them was compared with its frozen input, in record R1 or, for follow-up 6, in `inputs_official_pipeline.tar.gz` here (`release.py check-inputs`: name, dialect, the run's seed, and
  every field of every sequence entry, including the MSAs and templates) and all matched; AlphaFold3 adds only an
  empty `modifications` field, the run's seed and its own dialect version. Follow-up 8 ran seeds 6 to 20 on frozen inputs that list
  seeds 1 to 5, as its specification says; the check allows that for this follow-up and these seeds only.
- Engineering sessions that used random weights, and 31 files of the interrupted pilot session that
  the data freeze could not reconcile between the run machine and the results bucket; both lists are in record R1
  (`release_excluded_sessions.csv`, `release_excluded_files.csv`). Launch attempts that never ran have no outputs;
  their launcher records are in R1.
- The AlphaFold3 outputs of the sessions of follow-ups 3 to 16 outside the freeze (smoke tests, Spot preemptions, interruptions,
  relaunched-over sessions, `eigen_check`, the first LG session of follow-up 13). They enter no analysis. Their records are in
  R1, each mapped to its row of section 10 (`outside_freeze_records.tar.gz`, `outside_freeze_records_followups1316.tar.gz`).
- The layer-wise captures of follow-up 13 (arm L: `fu13_capture_seed-<s>.npz`, 40 files, 6.3 GiB): they are bfloat16 values in
  uncompressed `.npz` files, which gzip shrinks only to 0.80 of their size, so with them this record would pass Zenodo's 50 GB.
  They are listed with their SHA-256 in the data freeze (record R1), the follow-up 13 analysis outputs computed from them are in
  `analysis_outputs.tar.gz`, and the files are available from the author on request.

## Third-party data in the pipeline results

The alignments and templates of follow-up 6 (`inputs_official_pipeline.tar.gz` and the three `nas_*` archives) come from AlphaFold3's
data pipeline searching the databases below, whose own licences still apply to them. The attribution they ask for is given here;
`LICENCES.md` has the official page of each.

The AlphaFold 3 Output Terms apply to these files as AlphaFold 3 output. They do not restrict the third-party content the files contain: sequences and entries from UniProt, UniRef90 and BFD (CC BY 4.0) and from MGnify and the PDB (CC0 1.0) remain available under their own licences from their sources, and are attributed below.

- **UniProt**, release 2021_04 (`uniprot_all_2021_04.fa`), CC BY 4.0, https://www.uniprot.org/help/license. Cite: UniProt Consortium, UniProt: the Universal Protein Knowledgebase.
- **UniRef90**, release 2022_05 (`uniref90_2022_05.fa`), CC BY 4.0, https://www.uniprot.org/help/license. Cite: UniProt Consortium, UniProt: the Universal Protein Knowledgebase.
- **BFD**, small BFD as modified by Google DeepMind for AlphaFold 3 (`bfd-first_non_consensus_sequences.fasta`, AlphaFold 3 database set `v3.0`; no BFD release number is recorded), CC BY 4.0, https://bfd.mmseqs.com/. Cite: Steinegger, M. and Söding, J. *Nature Communications* (2018), https://doi.org/10.1038/s41467-018-04964-5; Steinegger, M., Mirdita, M. and Söding, J. *Nature Methods* (2019), https://doi.org/10.1038/s41592-019-0437-4.
- **MGnify**, release 2022_05 (`mgy_clusters_2022_05.fa`), CC0 1.0, https://ftp.ebi.ac.uk/pub/databases/metagenomics/peptide_database/2022_05/README.txt. Citation requested: Mitchell, A. L. et al. *Nucleic Acids Research* (2019), https://doi.org/10.1093/nar/gkz1035.
- **PDB**, snapshot of 2022-09-28 (`pdb_seqres_2022_09_28.fasta` and the mmCIF template files), CC0 1.0, https://www.wwpdb.org/about/usage-policies. Citation encouraged, not required: wwPDB consortium, *Nature Structural Biology* 10, 980 (2003); RCSB PDB: Berman, H. M. et al. *Nucleic Acids Research* 28, 235-242 (2000).

No RNA database was used: every target is proteins and ligands, and the pipeline logs show no `nhmmer` run.

## Redactions

Logs, analysis run logs and the XLA dumps of follow-up 16 can record a home path of the machine that ran them, or a bare host name of the follow-up 6 machines. Those
are replaced by `/home/<user>` and `<nas-host>`, and any other private network data by a placeholder, and `REDACTIONS.tsv` has one row
per file: the archive, the path, the SHA-256 of the original
and of the released copy, and what was replaced, never the value. So for a redacted log, the SHA-256 in the freeze manifest is
the original's, the first hash in that row. No email address is in this record except the contact above.

## Verify

```
shasum -a 256 -c SHA256SUMS                     # the archives and their manifests (sha256sum -c on Linux)
tar xzf af3_ppi_structures.tar.gz               # .tar for the embeddings: tar xf
shasum -a 256 -c af3_ppi_structures.tar.gz.sha256   # every extracted file
```

The archives are deterministic (sorted members, fixed time and owner, gzip without a timestamp): rebuilding them
with `release.py` (record R1) from the same files gives the same SHA-256.
