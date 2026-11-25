# AlphaFold3 on TPU and GPU: inputs, targets, manifests, data freeze and run records

Author: Lorenzo Pazienza, ORCID 0009-0006-1687-0125. Contact: lorenzo.pazienza@gmail.com.
Paper: see the related identifiers of this Zenodo record (arXiv preprint and conference version, added when available). Companion record with the AlphaFold3 outputs, the results of its data pipeline and the
follow-up 6 deliveries: 10.5281/zenodo.24057544. This record: 10.5281/zenodo.24057470. Code: https://github.com/lorenzopazienza/af3-cross-platform, the repository commit tagged zenodo-v1.
A copy of that commit is in this record (`code_alphafold-on-tpu_zenodo-v1.tar.gz`).

Pre-registration (OSF): project https://osf.io/qtz3v; the main study's registration https://osf.io/fegsy; the addenda
of follow-up 10 (https://osf.io/nsbgp), 11 (https://osf.io/64bpy), 13 (https://osf.io/c4kh7), 14 (https://osf.io/eafkx),
15 (https://osf.io/h9yw8), 16 (https://osf.io/hcxm7), 17 (https://osf.io/x4hry), 18 (https://osf.io/mq8fh), 19
(https://osf.io/4cbgf), 20 (https://osf.io/hjf8z) and 21 (https://osf.io/ctv2d). Every change
after registration is in section 10 of the analysis plan (`notes/analysis_plan.md` in the repository).

The record covers the whole study: the main study, follow-ups 1 to 21, and the two later measurements of the memory limit
on the A100 and H100 (`memlimit_ah`) and of the warm cost on the L4 Triton arm and the H100 (`warm_th`).

## What this record contains

| archive | contents |
|---|---|
| `inputs_targets_manifests.tar.gz` | the frozen AlphaFold3 input JSONs with their MSAs (`data/inputs/<ID>.json`, 60 PoseBusters protein-ligand targets; `data/inputs/ppi/<ID>.json`, 30 protein-protein complexes), the manifests that fix their SHA-256 (`inputs/`, with MSA provenance; the manifest of follow-up 6's pipeline results, `inputs/official_data/`, is in record R2), the scripts that built them, and the target lists with the rules and records of their selection (`targets/`, including `targets/repl/` and `targets/large/`) |
| `inputs_repl_large.tar.gz` | the frozen inputs of follow-up 11 (`data/inputs_repl/`, 72 targets) and follow-up 10 (`data/inputs_large/`, 68 targets), with ColabFold MSAs |
| `msa_provenance.tar.gz` | one JSON per MSA query (`data/msa/`, `data/msa/pair/`): server, mode, times, ticket and hashes of what the ColabFold server returned. The `user_agent` field of 99 of them is redacted (below). The MSAs themselves are in the inputs |
| `freeze.tar.gz` | every data freeze (`results/freeze/`): the session freezes, each with `manifest_sha256.tsv` (every frozen file of every session, with size, SHA-256 and CRC32C), the session list, the comparison with the results bucket and its summary; the file lists of the three follow-up 6 deliveries (the deliveries themselves are in record R2); `INDEX.md`, which lists them all and names the final one; and the scripts that made them (`scripts/freeze/`) |
| `session_lists_launch_records.tar.gz` | every list and record the scripts read besides the session folders: the session lists of the analyses (`results/analysis/**/sessions_*.txt`, `scripts/analysis_runs/*/sessions.txt`), the CCD codes of the build check (`ccd_codes.json`), the wrapper parts files (`results/parts_*.txt`), the wrapper state and logs of every launch (`results/af3/.many/`) and the per-session launch records (`results/af3/.launcher/`) |
| `session_records.tar.gz` | the run records of every frozen session of the main study and follow-ups 1 and 2, from the frozen local copy: `session.json` (software, configuration, device, weights SHA-256 and file name), `runs.jsonl` and `run.json` (per AlphaFold3 process: exit code, timings, compiled memory, cache evidence, input and output SHA-256), launch and job records, harness and job logs |
| `session_records_followups.tar.gz` | the same for the frozen sessions of follow-ups 3 to 12 (`cpu60`, `a100`, `feat`, `msa`, `sw`, `seeds`, `h100`, `repl`, `prec5`, `large`) |
| `session_records_followups1316.tar.gz` | the same for the frozen sessions of follow-ups 13 to 16 and the two later measurements (`fu13`, `fu14`, `rcx`, `hlo`, `memlimit_ah`, `warm_th`; 144 sessions). The XLA dumps of follow-up 16 are in record R2; the layer captures of follow-up 13 are listed in the freeze but not released (R2's README says why) |
| `session_records_followups1721.tar.gz` | the same for the frozen sessions of follow-ups 17 to 21 (`dtp`, `tri`, `rcf`, `aut`, `xvm`; 168 sessions) |
| `attempt_superseded_records.tar.gz` | from the results bucket: the launcher records of launch attempts that never ran (no VM run; for example no TPU capacity), and the run records of the superseded or engineering sessions whose outputs are in R2 (`af3_superseded_engineering.tar.gz`, not used in any reported result) |
| `outside_freeze_records.tar.gz` | from the results bucket: the launcher and session records of the sessions of follow-ups 3 to 12 outside the freeze (smoke tests, Spot preemptions, interruptions, relaunched-over sessions, `eigen_check`, and attempts of follow-ups 6 to 12 that created no VM). Their AlphaFold3 outputs are not released and enter no analysis. `OUTSIDE_SESSIONS.csv` maps each session to the row of section 10 of the analysis plan that explains it |
| `outside_freeze_records_followups1316.tar.gz` | the same for follow-ups 13 to 16 (247 sessions: Spot preemptions, setup failures, a first LG session whose processes all failed, and attempts that created no VM), each mapped by `OUTSIDE_SESSIONS.csv` to its row |
| `outside_freeze_records_followups1721.tar.gz` | the same for follow-ups 17 to 21 (882 sessions: the smoke test of follow-up 18, Spot VMs lost during the run, a run stopped at its deadline after a slow setup, and attempts that created no VM), each mapped by `OUTSIDE_SESSIONS.csv` to its row |
| `code_alphafold-on-tpu_zenodo-v1.tar.gz` | a copy of the code: the files of the repository commit tagged `zenodo-v1`, under `alphafold-on-tpu-zenodo-v1/`, read with `git archive` and packaged byte for byte, never redacted (the harness, the analyses, the pre-registered plan `notes/analysis_plan.md`, the freeze and release scripts, the generators of the paper's tables and figures; the text of the paper is on arXiv). `paper/assets/` is left out: its tables and figures are computed from AlphaFold 3 outputs, so they are in record R2 (`analysis_outputs.tar.gz`, from the same commit). `CODE_COMMIT.txt` gives the full commit hash. MIT licence (`LICENSE` inside); the files `NOTICE` lists come from github.com/lorenzopazienza/alphafold-tpu-benchmark under its MIT licence (`LICENSES/MIT-alphafold-tpu-benchmark` inside), and the AlphaFold 3 lines of `af3_tpu/af3_tpu.patch` stay under the Apache License 2.0 (`LICENSES/Apache-2.0` inside) |
| `REDACTIONS.tsv` (plain file) | every file packaged with a redaction: for each, the archive, the path, the SHA-256 of the original and of the released copy, and what was replaced, never the value. Every other file is packaged byte for byte as recorded |
| `LICENCES.md` (plain file) | every third-party source, its licence, the official page, and the licence line of each archive |
| `release_*` (plain files; a Zenodo record has no folders) | `release_release.py` (inventory, privacy scan, layout, input check, archive building and verification), `release_RUNBOOK.md` (how the archives were built), the bucket inventory, the layout, the lists of excluded sessions and files, `release_SUPERSEDED_SESSIONS.csv` and `release_OUTSIDE_SESSIONS.csv` (each session with its row of section 10), `release_inputs_check.csv` and `release_inputs_check.json` (the check of every `_data.json` against its frozen input, sessions of freeze `68923012`), `release_inputs_check_followups1316.csv` and `.json` (the same check for the 144 sessions added by freeze `4ec185c5`), and `release_inputs_check_followups1721.csv` and `.json` (the same for the 168 sessions added by freeze `d706dffc`) |

Data freeze identifier: `d706dffc07fec6d86471e28f062178e4f309193d3bfd6a66de47959eb3bfdd8f`, the SHA-256 of
`results/freeze/followups1721_20261125/manifest_sha256.tsv`: 607 sessions, cumulative (the 439 sessions of freeze
`4ec185c5...` of 2026-11-17 and the 168 of follow-ups 17 to 21). It holds every row of the earlier
session freezes, in order and byte for byte; the freeze `689230120499...` (295 sessions) is not a byte prefix of the later ones
only because 700 full outputs of two main-study sessions, which that freeze lists as bucket only, were later copied to
the local freeze copy (`results/freeze/INDEX.md`: "yes, plus 700 files added later to its sessions"). The AlphaFold3
outputs of the same sessions are in record R2, under the same `af3/<session>/...` paths.

The follow-up 6 deliveries were never in the results bucket and keep their own freezes. The deliveries are in record R2, whose README
says what each is and where it ran; their file lists are in `freeze.tar.gz`:

| delivery | freeze identifier (SHA-256 of `files_sha256.tsv`) | files, bytes |
|---|---|---|
| `naspaz8-af3_run1` | `3c21b4afbdfbcf310e7e289ea7ff326c2e1579ed18cea425a1616dd8f1c7b9da` | 184, 1,594,914,931 |
| `naspaz6-af3_run1` | `781f4ee3221df5f2e8f09b9dec91ad919a87c80480eba838eedb946cf0c131b2` | 34, 294,912,013 |
| `naspaz8-af3_run2` | `ed7f634e3e0832462ab52ea918f0d47954fee7a8ca8fda98c18add3c9ff3d0c4` | 34, 294,916,131 |

**No model parameters (weights) are included.** The records name the weights file (`af3.bin.zst`) and its SHA-256
so that a run can be matched to the parameters Google DeepMind distributes; the parameters themselves are available
only from Google DeepMind (https://github.com/google-deepmind/alphafold3).

## Redactions

The contact address above is the only email address in this record. It was redacted from the files where it appeared, each replaced
by `<account>`: the `user_agent` field of 99 MSA provenance files, where it appeared only as the contact the ColabFold server asks
for, and 202 launch wrapper logs, where `gcloud` printed the account in an "authenticated as" line. Attribution and contact do not
come from those files. Home paths on the VMs are replaced by `/home/<user>` and any other private network data by a placeholder:
`REDACTIONS.tsv` has one row per file. So for a redacted file, the SHA-256 in the freeze manifest is the original's, which is the
first hash in that row; the released copy has the second. The job logs of the VMs also name systemd units of the form
NAME, at sign, INSTANCE, dot, service (the getty service of a terminal, for example); these are not addresses and are
packaged unchanged.

## Licence

This record is released under the **Creative Commons Attribution 4.0 International licence (CC BY 4.0)**,
https://creativecommons.org/licenses/by/4.0/. The scripts are also available under the MIT licence of the code
repository. `LICENCES.md` gives the licence line of each archive and the official page of each source.

Attribution for third-party data it contains or derives from:

- **ColabFold databases** (CC BY 4.0, https://colabfold.mmseqs.com/): the MSAs of the main, follow-up 1, 10 and 11 inputs come from
  the ColabFold MSA server, searched against UniRef30 and ColabFoldDB. Mirdita, M. et al. ColabFold: making protein folding
  accessible to all. *Nature Methods* 19, 679-682 (2022).
- **UniProt** (CC BY 4.0, https://www.uniprot.org/help/license): UniRef30 in the MSAs above. UniProt Consortium, UniProt: the
  Universal Protein Knowledgebase.
- **PDB data** (CC0 1.0, https://www.wwpdb.org/about/usage-policies): target identities, sequences and metadata (wwPDB; RCSB PDB:
  Berman, H. M. et al. *Nucleic Acids Research* 28, 235-242 (2000)). No PDB coordinates are redistributed here.
- **PoseBusters benchmark** (CC BY 4.0, https://zenodo.org/records/8278563): the protein-ligand targets. Buttenschoen, M. et al.
  *Chemical Science* 15, 3130-3139 (2024).
- **Runs N' Poses** (CC BY 4.0, https://zenodo.org/records/14794785): the follow-up 10 feasibility screen,
  `targets/large/rnp_screen.csv`; https://doi.org/10.1101/2025.02.03.636309.

No RNA database was used. No AlphaFold 3 Output is in this record: the run records and freeze lists give file names, sizes and
SHA-256 of the outputs, but no predicted coordinates, scores, confidences, alignments or templates. These are in record R2,
under the AlphaFold 3 Output Terms of Use (https://github.com/google-deepmind/alphafold3/blob/main/OUTPUT_TERMS_OF_USE.md):
the outputs, AlphaFold3's logs, the feature arrays of follow-up 5, the results of
AlphaFold3's data pipeline with the follow-up 6 deliveries, the XLA dumps of follow-up 16, and the analysis outputs derived
from them. The sequence-only inputs of follow-up 6 are there too, with the pipeline
results they led to; their SHA-256 are in `inputs/official/manifest.csv`, here.

## Verify

```
shasum -a 256 -c SHA256SUMS                          # the archives and their manifests (sha256sum -c on Linux)
tar xzf freeze.tar.gz
shasum -a 256 results/freeze/followups1721_20261125/manifest_sha256.tsv    # must print the data freeze identifier above
tar xzf inputs_targets_manifests.tar.gz
shasum -a 256 -c inputs_targets_manifests.tar.gz.sha256
tar xzf code_alphafold-on-tpu_zenodo-v1.tar.gz
head -1 CODE_COMMIT.txt                              # the full hash of the commit tagged zenodo-v1
```

In a clone of the repository, `git archive --prefix=alphafold-on-tpu-zenodo-v1/` of that commit gives the same files, plus
`paper/assets/`.

Each input's SHA-256 is also in its manifest (`input_sha256`), and every run re-checked it before running.
