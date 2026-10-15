# AlphaFold3 on TPU and GPU: predicted structures, confidences and derived analysis outputs (DRAFT)

**Draft, not yet published.** Fields in angle brackets are to be filled in at publication.

Authors: <authors>. Paper: <paper citation, DOI>. Companion record (inputs, targets, manifests, data freeze, run
records): <DOI of record R1>. Code: <repository URL, commit>.

## What this record contains

AlphaFold3 v3.0.4 outputs (official model parameters, AlphaFold3's defaults: 10 recycles, 5 diffusion samples per
seed) from the study "<paper title>", run on TPU v5e (V5), TPU v6e (V6, V6h), NVIDIA L4 (G, GT, Gd, Gh) and CPU (C).
**No model parameters (weights) are included.** They are available only from Google DeepMind
(https://github.com/google-deepmind/alphafold3).

The outputs of the study come from the sessions of the data freeze
`14e1fc4c12b6573de9389d9aba079d5fb5f30b290c9303f75c5de4f478b9d8f9` (the SHA-256 of `manifest_sha256.tsv` in record
R1, which lists every frozen file with its SHA-256). Two archives hold other material, under the same terms: the
superseded or engineering sessions (not used in any reported result) and the analysis outputs derived from the
study's outputs.

| archive | sessions | contents |
|---|---|---|
| `af3_main_structures.tar.gz` | main study (53 sessions) | per-sample and top-level mmCIF models, `*_summary_confidences.json`, `*_ranking_scores.csv`, `TERMS_OF_USE.md` |
| `af3_main_full_confidences.tar.gz` | main study | per-sample and top-level `*_confidences.json` (per-atom pLDDT, PAE, contact probabilities) |
| `af3_pilot2_structures.tar.gz` | repeated pilot (13 sessions) | as the main-study structures |
| `af3_pilot2_full_confidences.tar.gz` | pilot | as the main-study full confidences |
| `af3_memlimit.tar.gz` | memory-limit measurement (4 sessions; 7D5C padded to larger buckets, not real inputs of those sizes) | structures and full confidences |
| `af3_ppi_structures.tar.gz` | follow-up 1, 30 protein-protein complexes (9 sessions) | structures |
| `af3_ppi_full_confidences.tar.gz` | follow-up 1 | full confidences |
| `af3_embed_structures.tar.gz` | follow-up 2, 10 pilot targets, seed 1 (4 sessions) | structures and full confidences |
| `af3_embed_embeddings.tar` | follow-up 2 | final trunk single and pair embeddings (`--save_embeddings`, float16 `.npz`) |
| `af3_logs.tar.gz` | every session above | `run_alphafold.log` of each AlphaFold3 process |
| `analysis_outputs.tar.gz` | every analysis of the study and its follow-ups | the derived analysis outputs (`results/analysis/`: per-sample tables such as `samples.csv` with ranking scores, pLDDT, ipTM, M1 and DockQ, contrasts, reports and run logs) and the paper's tables and figures (`results/paper_assets/`, with `sources.tsv`, the SHA-256 of every input they were made from); they carry values computed from AlphaFold3 outputs, so they are released here under the same terms. The code that made them is in the repository (`analysis/`, `scripts/analysis_runs/`), and the session lists they were run on are in record R1 |
| `af3_superseded_engineering.tar.gz` | **superseded or engineering, not used in any reported result** (15 sessions with the official weights: the first pilot, its V6h and Gh arms and V5 warm completion, the L4 determinism diagnostic, a GT smoke test, and one interrupted pilot session) | structures, full confidences and logs; `SUPERSEDED_SESSIONS.csv` maps each session to the section 10 row of the analysis plan that records it |

Each archive also holds `LEGALLY_BINDING_TERMS_OF_USE.txt` and `OUTPUT_TERMS_OF_USE.md` (below), and every
AlphaFold3 output folder keeps the `TERMS_OF_USE.md` AlphaFold3 wrote.

### Paths

Members are `af3/<session>/<run>/<target>/af3_output/<target>/...`, as on the machines that ran them:

- `<session>`: `<UTC launch time>_<platform>_<plan>`, for example `20261013T014901Z_l4_main_g_01`;
- `<run>`: `<config>_rep<r>_seed<s>_fresh` (fresh compilation cache, repetition r), or `<config>_seed<s>_warm`
  (warm cache rerun); the config names the arm (`tpu_xla`: V5 or V6 by platform, `l4_xla`: G, `l4_triton`: GT,
  `l4_xla_det`: Gd, `tpu_xla_highest`: V6h, `l4_xla_highest`: Gh, `cpu_xla`: C);
- `seed-<s>_sample-<k>/`: one diffusion sample; the top-level `<target>_model.cif` and `<target>_confidences.json`
  are AlphaFold3's copies of its top-ranked sample.

The run records of each session (`session.json`, `runs.jsonl`, `run.json`, launch records) are in record R1
(`session_records.tar.gz`), under the same paths.

### Left out

- `*_data.json`: AlphaFold3's copy of each run's input (22.8 GiB). Before these archives were built, every one of
  them was compared with its frozen input in record R1 (`release.py check-inputs`: name, dialect, the run's seed, and
  every field of every sequence entry, including the MSAs and templates) and all matched; AlphaFold3 adds only an
  empty `modifications` field, the run's seed and its own dialect version.
- Engineering sessions that used random weights (6 sessions), and 31 files of the interrupted pilot session that
  the data freeze could not reconcile between the run machine and the results bucket; both lists are in record R1
  (`release/excluded_sessions.csv`, `release/excluded_files.csv`). Launch attempts that never ran have no outputs;
  their launcher records are in R1.

## Terms of use of AlphaFold3 outputs

**By using this information, you agree to AlphaFold 3 Output Terms of Use found at
https://github.com/google-deepmind/alphafold3/blob/main/OUTPUT_TERMS_OF_USE.md.**

**To request access to the AlphaFold 3 model parameters, follow the process set out at
https://github.com/google-deepmind/alphafold3. You may only use these if received directly from Google. Use is
subject to terms of use available at https://github.com/google-deepmind/alphafold3/blob/main/WEIGHTS_TERMS_OF_USE.md.**

In short (the Terms govern; this summary does not change them): the outputs are for non-commercial use only; they
must not be used to train or create machine learning models or related technology for biomolecular structure
prediction similar to AlphaFold 3, including via distillation; any distribution must keep this notice; and any use
must cite:

> Abramson, J. et al. Accurate structure prediction of biomolecular interactions with AlphaFold 3. *Nature* 630,
> 493-500 (2024). https://doi.org/10.1038/s41586-024-07487-w

**Licence of this record: "Other", namely the AlphaFold 3 Output Terms of Use above. No Creative Commons or other
licence applies to the AlphaFold3 outputs in this record.** We have not modified them: every file is as
AlphaFold3 wrote it (SHA-256 in the per-archive manifests and in the data freeze).

Please also cite <paper citation> when using these data.

## Verify

```
shasum -a 256 -c SHA256SUMS                     # the archives and their manifests (sha256sum -c on Linux)
tar xzf af3_ppi_structures.tar.gz               # .tar for the embeddings: tar xf
shasum -a 256 -c af3_ppi_structures.tar.gz.sha256   # every extracted file
```

The archives are deterministic (sorted members, fixed time and owner, gzip without a timestamp): rebuilding them
with `release.py` (record R1) from the same files gives the same SHA-256.
