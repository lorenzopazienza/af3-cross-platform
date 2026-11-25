# Same Seed, Different Top Prediction: Cross-Platform Reproducibility of AlphaFold 3 Inference on TPUs, GPUs, and CPUs

This repository holds the code, the pre-registered analysis plan and the release scripts of a cross-platform
reproducibility study of AlphaFold 3 inference on TPUs, GPUs and CPUs.

## Status

The study is complete. Its data are frozen in the final data freeze
`d706dffc07fec6d86471e28f062178e4f309193d3bfd6a66de47959eb3bfdd8f`, made on 25 November 2026: 607 sessions.

## Where things are

| Path | What |
|---|---|
| `notes/analysis_plan.md` | the pre-registered analysis plan; section 10 is its change log, one dated row for every change after registration |
| `harness/` | the harness that runs AlphaFold 3 on the frozen inputs, one fresh process per run, and records each run |
| `cloud/` | the launchers: `cloud/af3_run.sh` creates a VM, runs a plan of `harness/plans/` on it, fetches the session's results and deletes the VM |
| `af3_tpu/` | the patch that adds a TPU backend to AlphaFold 3 v3.0.4 (see `af3_tpu/README.md`) |
| `inputs/`, `targets/` | the manifests of the frozen inputs and the target lists with their selection records (the inputs themselves are in record R1) |
| `analysis/` | the analyses (see `analysis/README.md`) |
| `scripts/analysis_runs/` | one folder per analysis run, with its session list and the `reproduce.sh` that regenerates its outputs |
| `scripts/freeze/` | the data freeze and its index |
| `scripts/release/` | the Zenodo release: `release.py`, `RUNBOOK.md`, and the README, metadata and licences of each record in `records/` |
| `paper/` | the paper's tables and figures (`paper/assets/`) and `paper/NUMBERS.md`, every number in the paper with its source; the text of the paper is on arXiv |
| `analysis/paper_assets_isc.py` | the generator of the paper's tables and figures (with `paper_assets_fu678.py`, `paper_assets_fu9.py` and `paper_assets_fu10_11_12.py`) |
| `tests/` | tests of the cloud scripts against a fake `gcloud` |
| `src/` | the AlphaFold 2 benchmark scripts of the earlier study |
| `drafts/` | drafts of plan rows and launch notes, kept as written |

The plan and its addenda are registered on OSF: the main study at https://osf.io/fegsy, and the addenda of follow-ups
10 (https://osf.io/nsbgp), 11 (https://osf.io/64bpy), 13 (https://osf.io/c4kh7), 14 (https://osf.io/eafkx),
15 (https://osf.io/h9yw8), 16 (https://osf.io/hcxm7), 17 (https://osf.io/x4hry), 18 (https://osf.io/mq8fh),
19 (https://osf.io/4cbgf), 20 (https://osf.io/hjf8z) and 21 (https://osf.io/ctv2d).

## Reproducing an analysis

1. Create the environment:
   `uv venv --python 3.12 analysis/.venv && uv pip install --python analysis/.venv/bin/python -r analysis/requirements.txt`.
   The figures use a second one, `analysis/figures/.venv`, from `analysis/requirements-figures.txt`.
2. Download records R1 and R2 (below). Extract the archives of session folders (members `af3/<session>/...`) into
   `results/`, and the other archives (members `results/...`, `data/...`, `inputs/...`, `targets/...`) at the repository
   root.
3. From the repository root, run `bash scripts/analysis_runs/<name>/reproduce.sh`. Some scripts take the analysis date,
   as `DATE=YYYYMMDD bash ...`; the header of each script says so. `scripts/analysis_runs/check_byte_identity.sh` reruns
   analyses and checks that their outputs come back byte for byte.

To rebuild the paper's tables and figures, run `scripts/analysis_runs/paper_assets/reproduce_followups34.sh` and the four
generators `analysis/paper_assets_*.py` with `analysis/figures/.venv/bin/python`; they read `results/analysis/` (rerun, or
from R2's `analysis_outputs.tar.gz`) and write `results/paper_assets_*/`. Then `bash paper/build_assets.sh` copies them
into `paper/assets/`, checks every input's SHA-256 and writes the derived and relabeled tables with `paper/tools/`
(see `paper/README.md`).

## Data

- Record R1, DOI [10.5281/zenodo.24057470](https://doi.org/10.5281/zenodo.24057470), CC BY 4.0: inputs with their
  alignments, target lists, manifests, the data freeze, session lists and run records, and a copy of this repository at
  the commit tagged `zenodo-v1`, without `paper/assets/` (in R2, under the AlphaFold 3 Output Terms).
- Record R2, DOI [10.5281/zenodo.24057544](https://doi.org/10.5281/zenodo.24057544), under the AlphaFold 3 Output Terms
  of Use: AlphaFold 3 outputs, the results of its data pipeline, and the analysis outputs derived from them.
- The layer captures of follow-up 13, and the full confidences and `*_data.json` of follow-ups 17 to 21 (except the 20
  processes of follow-up 17's check 5, in R2), are in neither record. The freeze manifest lists them with their SHA-256;
  they are available on request.

## Paper

[arXiv ID]. To cite this work, see `CITATION.cff`.

It follows the preprint *Accelerator Choice Is Not Enough: AlphaFold2 Inference on Cloud TPUs*
([arXiv:2609.34818](https://arxiv.org/abs/2609.34818)). The code and data of that preprint are archived at
[github.com/lorenzopazienza/alphafold-tpu-benchmark](https://github.com/lorenzopazienza/alphafold-tpu-benchmark).

## Author and contact

Lorenzo Pazienza, LUISS Guido Carli University. lorenzo.pazienza@gmail.com. ORCID
[0009-0006-1687-0125](https://orcid.org/0009-0006-1687-0125).

## Licences

- Code of this repository: MIT (`LICENSE`).
- Files listed in `NOTICE`, copied from github.com/lorenzopazienza/alphafold-tpu-benchmark: MIT, Copyright (c) 2026
  Lorenzo Pazienza and Ihab El Bani (`LICENSES/MIT-alphafold-tpu-benchmark`).
- `af3_tpu/af3_tpu.patch`: the lines of AlphaFold 3's `run_alphafold.py` it carries stay under the Apache License 2.0
  (`LICENSES/Apache-2.0`).
- Tables and figures in `paper/assets/`: AlphaFold 3 Output Terms of Use,
  https://github.com/google-deepmind/alphafold3/blob/main/OUTPUT_TERMS_OF_USE.md.
- Third-party data in `targets/` and `inputs/` and in the Zenodo records: see
  [`scripts/release/records/LICENCES.md`](scripts/release/records/LICENCES.md).
- AlphaFold 3 model parameters: never included in this repository or in the Zenodo records; they are available only
  from Google DeepMind, under its own terms of use.

## Use of AI

The author used Claude Code (Anthropic) to write the code and the analyses and to assist with the writing of the paper.
He reviewed the code and the analyses at every step and is responsible for the content.
