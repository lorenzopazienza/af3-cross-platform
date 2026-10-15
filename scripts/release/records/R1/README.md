# AlphaFold3 on TPU and GPU: inputs, targets, manifests, data freeze and run records (DRAFT)

**Draft, not yet published.** Fields in angle brackets are to be filled in at publication.

Authors: <authors>. Paper: <paper citation, DOI>. Companion record with the AlphaFold3 outputs: <DOI of record R2>.
Code: <repository URL, commit>.

## What this record contains

| archive | contents |
|---|---|
| `inputs_targets_manifests.tar.gz` | the frozen AlphaFold3 input JSONs with their MSAs (`data/inputs/<ID>.json`, 60 PoseBusters protein-ligand targets; `data/inputs/ppi/<ID>.json`, 30 protein-protein complexes), the manifests that fix their SHA-256 (`inputs/manifest.csv`, `inputs/ppi/manifest.csv`, with MSA provenance), the scripts that built them, and the target lists with the rules and records of their selection (`targets/`, `targets/ppi/`) |
| `freeze.tar.gz` | the data freeze: `manifest_sha256.tsv` (every frozen file of every session, with size, SHA-256 and CRC32C), the session list, the comparison with the results bucket, its summary and the script that made it |
| `session_lists_launch_records.tar.gz` | every list and record the scripts read besides the session folders: the session lists of the analyses (`results/analysis/**/sessions_*.txt`), the CCD codes of the build check (`ccd_codes.json`), the wrapper parts files (`results/parts_*.txt`), the wrapper state and logs of every launch (`results/af3/.many/`) and the per-session launch records (`results/af3/.launcher/`) |
| `session_records.tar.gz` | the run records of every frozen session, from the frozen local copy: `session.json` (software, configuration, device, weights SHA-256 and file name), `runs.jsonl` and `run.json` (per AlphaFold3 process: exit code, timings, compiled memory, cache evidence, input and output SHA-256), launch and job records, harness and job logs |
| `attempt_superseded_records.tar.gz` | from the results bucket: the launcher records of launch attempts that never ran (no VM run; for example no TPU capacity), and the run records of the superseded or engineering sessions whose outputs are in R2 (`af3_superseded_engineering.tar.gz`, not used in any reported result) |
| `REDACTIONS.tsv` (plain file) | the files packaged with a redaction: two wrapper logs in `session_lists_launch_records.tar.gz` whose gcloud account email, in "authenticated as" lines, is replaced by `<account>`; for each, the archive, the path, the SHA-256 of the original and of the released copy, and what was replaced. Every other file is packaged byte for byte as recorded |
| `release/` (plain files) | `release.py` (inventory, layout, input check, archive building and verification), `RUNBOOK.md` (how the archives were built), the bucket inventory, the layout, the lists of excluded sessions and files, and `SUPERSEDED_SESSIONS.csv` (each superseded or engineering session with the section 10 row of the analysis plan that records it) |

Data freeze identifier: `14e1fc4c12b6573de9389d9aba079d5fb5f30b290c9303f75c5de4f478b9d8f9`, the SHA-256 of
`manifest_sha256.tsv`. The AlphaFold3 outputs of the same sessions are in record R2, under the same
`af3/<session>/...` paths.

**No model parameters (weights) are included.** The records name the weights file (`af3.bin.zst`) and its SHA-256
so that a run can be matched to the parameters Google DeepMind distributes; the parameters themselves are available
only from Google DeepMind (https://github.com/google-deepmind/alphafold3).

## Licence

This record is released under the **Creative Commons Attribution 4.0 International licence (CC BY 4.0)**,
https://creativecommons.org/licenses/by/4.0/. The scripts are also available under the MIT licence of the code
repository.

Attribution for third-party data it contains or derives from:

- **ColabFold databases** (CC BY 4.0, https://colabfold.mmseqs.com/): the MSAs in the input JSONs come from the
  ColabFold MSA server, searched against the ColabFold databases (ColabFoldDB environmental sequences). Mirdita, M.
  et al. ColabFold: making protein folding accessible to all. *Nature Methods* 19, 679-682 (2022).
- **UniRef30 via UniProt** (CC BY 4.0, https://www.uniprot.org/help/license), searched by the same server: UniProt
  Consortium, UniProt: the Universal Protein Knowledgebase; UniRef30 as distributed for HH-suite and ColabFold.
- **PDB data** (CC0, https://www.wwpdb.org/about/usage-policies): target identities, sequences and metadata come
  from the Protein Data Bank (wwPDB; RCSB PDB: Berman, H. M. et al. *Nucleic Acids Research* 28, 235-242 (2000)),
  and the protein-ligand targets from the PoseBusters benchmark (Buttenschoen, M. et al. *Chemical Science* 15,
  3130-3139 (2024)). No PDB coordinate files are redistributed here.
- No AlphaFold3 output is in this record: the run records list output file names but no predicted coordinates,
  scores or confidences. The outputs, AlphaFold3's logs and the analysis outputs derived from them (tables, reports
  and the paper's figures, which carry per-sample values computed from the outputs) are in record R2, under the
  AlphaFold 3 Output Terms of Use (https://github.com/google-deepmind/alphafold3/blob/main/OUTPUT_TERMS_OF_USE.md).

## Verify

```
shasum -a 256 -c SHA256SUMS                          # the archives and their manifests (sha256sum -c on Linux)
tar xzf freeze.tar.gz
shasum -a 256 results/freeze/manifest_sha256.tsv     # must print the data freeze identifier above
tar xzf inputs_targets_manifests.tar.gz
shasum -a 256 -c inputs_targets_manifests.tar.gz.sha256
```

Each input's SHA-256 is also in `inputs/manifest.csv` (`input_sha256`), and every run re-checked it before running.
